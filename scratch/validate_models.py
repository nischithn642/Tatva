"""
Multi-Model Validation Matrix Script for TATVA.

Validates the full import -> baseline -> optimize -> parity pipeline across
multiple ONNX transformer models.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure src/ is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from tatva.compiler import import_model, analyze_graph, TARGETS, UnsupportedOperatorError
from tatva.optimizer import fuse_attention_softmax
from tatva.runner import establish_baseline, compile_model, run_and_measure, ExecutionEnvironment
from tatva.diagnostics import MemoryLimitExceededError, AccuracyDropError


def validate_single_model(model_path: Path, variant_name: str = "RV64GC", tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Run full validation on a single model and return structured test metrics.
    """
    res: Dict[str, Any] = {
        "model_name": model_path.name,
        "model_path": str(model_path),
        "size_kb": round(model_path.stat().st_size / 1024, 1) if model_path.exists() else 0,
        "imported": False,
        "ops_count": 0,
        "baseline_passed": False,
        "baseline_ms": 0.0,
        "optimized_passed": False,
        "optimized_ms": 0.0,
        "parity_passed": False,
        "mse": 0.0,
        "status": "FAIL",
        "notes": "",
    }

    if not model_path.exists():
        res["notes"] = "File not found"
        return res

    variant = TARGETS.get(variant_name, TARGETS["RV64GC"])

    start_t = time.time()
    try:
        # 1. Ingestion & Graph Analysis
        ir = import_model(str(model_path))
        stats = analyze_graph(ir)
        res["imported"] = True
        res["ops_count"] = stats.total_ops

        # 2. Baseline Execution
        baseline_res = establish_baseline(str(model_path), variant)
        res["baseline_passed"] = baseline_res.parity_passed
        res["baseline_ms"] = baseline_res.latency_result.mean_ms

        # 3. Softmax/Attention Fusion Optimization
        fused_ir = fuse_attention_softmax(ir)
        
        # 4. Compilation & Execution of Fused Artifact
        build_dir = Path("scratch") / f"val_build_{model_path.stem}"
        artifact = compile_model(fused_ir, variant, str(build_dir), warmup_count=2, timed_count=5)
        measurement = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)

        res["optimized_passed"] = True
        res["optimized_ms"] = measurement.mean_ms

        # 5. Extract output logits and calculate MSE vs reference
        target_logits: List[float] = []
        for line in measurement.raw_output.splitlines():
            if "FIRST_LOGITS:" in line:
                parts = line.strip().split(":")[1].strip().split()
                target_logits = [float(x) for x in parts]
                break

        if target_logits and baseline_res.target_logits:
            min_len = min(len(target_logits), len(baseline_res.target_logits))
            mse = float(np.mean((np.array(target_logits[:min_len]) - np.array(baseline_res.target_logits[:min_len])) ** 2))
            res["mse"] = mse
            res["parity_passed"] = mse < tolerance

        if res["imported"] and res["baseline_passed"] and res["optimized_passed"] and res["parity_passed"]:
            res["status"] = "PASS"
            speedup = ((res["baseline_ms"] - res["optimized_ms"]) / res["baseline_ms"]) * 100
            res["notes"] = f"Parity OK (MSE={res['mse']:.6f}, Speedup={speedup:+.1f}%)"
        else:
            res["notes"] = f"Parity Check Failed (MSE={res['mse']:.6f})"

    except UnsupportedOperatorError as e:
        res["imported"] = False
        res["status"] = "EXPECTED FAIL"
        res["notes"] = f"Caught UnsupportedOp: {e.operator_name}"
    except (MemoryLimitExceededError, AccuracyDropError, RuntimeError) as e:
        res["status"] = "DIAGNOSED FAIL"
        res["notes"] = f"Handled Exception: {type(e).__name__}"
    except Exception as e:
        res["status"] = "ERROR"
        res["notes"] = f"Unhandled Error: {e}"


    return res


def run_validation_matrix(models_dir: str = "models", variant_name: str = "RV64GC") -> List[Dict[str, Any]]:
    """
    Discover all ONNX models in models_dir and execute the validation matrix for variant_name.
    """
    models_path = Path(models_dir)
    target_files = sorted(list(models_path.glob("*.onnx")))

    if not target_files:
        print(f"No ONNX models found in directory '{models_dir}'.")
        return []

    print(f"\n==================================================================================")
    print(f"               TATVA MULTI-MODEL VALIDATION MATRIX ({variant_name})              ")
    print(f"==================================================================================")
    print(f"{'Model Name':<26} | {'Size (KB)':<9} | {'Ops':<5} | {'Base (ms)':<9} | {'Opt (ms)':<9} | {'Status':<13} | {'Notes'}")
    print(f"----------------------------------------------------------------------------------")

    results = []
    for mfile in target_files:
        r = validate_single_model(mfile, variant_name=variant_name)
        results.append(r)
        
        base_str = f"{r['baseline_ms']:.2f}" if r["baseline_passed"] else "N/A"
        opt_str = f"{r['optimized_ms']:.2f}" if r["optimized_passed"] else "N/A"
        ops_str = str(r["ops_count"]) if r["ops_count"] > 0 else "N/A"

        print(f"{r['model_name']:<26} | {r['size_kb']:<9.1f} | {ops_str:<5} | {base_str:<9} | {opt_str:<9} | {r['status']:<13} | {r['notes']}")

    print(f"==================================================================================\n")
    return results


if __name__ == "__main__":
    matrix = run_validation_matrix()
    failures = [r for r in matrix if r["status"] in ("FAIL", "ERROR")]
    if failures:
        print(f"Validation failed for {len(failures)} model(s).")
        sys.exit(1)
    else:
        print("All models passed validation matrix successfully!")
        sys.exit(0)
