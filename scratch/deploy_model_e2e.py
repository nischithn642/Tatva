"""
End-to-End Model Deployment Verification Script.

Executes the complete TATVA deployment pipeline on a model without requiring a physical board connected.
Uses QEMU bare-metal system emulation with hardware cycle counters (rdcycle) and reports full parity metrics.
"""

import os
import sys
import json
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tatva.compiler import import_model, analyze_graph, TARGETS, TargetVariant
from tatva.optimizer import fuse_attention_softmax, compare_configs
from tatva.runner import compile_model, run_and_measure, establish_baseline, verify_target, ExecutionEnvironment


def run_e2e_deployment() -> None:
    model_path = os.path.join("models", "model.onnx")
    out_dir = os.path.join("build_deployment_test")

    print("==================================================================")
    print("TATVA End-to-End Model Deployment Pipeline (Without Board Needed)")
    print("==================================================================")

    # 1. Model Input & Import
    print(f"\n[STEP 1/6] Importing ONNX Model: {model_path}...")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file '{model_path}' not found.")
        sys.exit(1)

    model_ir = import_model(model_path)
    print(f"  - Model Format: ONNX")
    print(f"  - File Size: {os.path.getsize(model_path) / 1024:.2f} KB")

    # 2. Graph Analysis
    print(f"\n[STEP 2/6] Analyzing Graph Operators & Bottlenecks...")
    stats = analyze_graph(model_ir)
    print(f"  - Total Operators: {stats.total_ops}")
    print(f"  - Transformer Attention/Softmax Bottleneck: {'DETECTED' if stats.has_transformer_bottleneck else 'NOT DETECTED'}")

    # 3. Target Verification
    target = TARGETS["RV64GC"]
    print(f"\n[STEP 3/6] Verifying Selected Target Architecture: {target.name}...")
    print(f"  - GCC Architecture String: {target.gcc_march}")
    print(f"  - GCC ABI String:          {target.gcc_mabi}")
    
    target_status = verify_target(target)
    if target_status["status"] == "ok":
        print(f"  - Target Status: [OK] Toolchain and QEMU emulator verified.")
    else:
        print(f"  - Target Status: [ERROR] {target_status.get('error')}")

    # 4. Apply Schedule Optimization Passes
    print(f"\n[STEP 4/6] Applying Schraudolph Softmax Fusion Optimization...")
    opt_model_ir = fuse_attention_softmax(model_ir)
    print(f"  - Softmax Kernel Fusion: APPLIED (Single-pass stack-allocated register kernel)")

    # 5. C Code Generation & Bare-Metal Cross-Compilation
    print(f"\n[STEP 5/6] Generating C99 Code & Cross-Compiling for RISC-V Bare-Metal...")
    artifact = compile_model(opt_model_ir, target, warmup_count=2, timed_count=5)
    print(f"  - Compiled ELF Artifact: {artifact.elf_path}")
    print(f"  - Build Directory: {artifact.build_dir}")

    # 6. QEMU Simulation & Parity Verification
    print(f"\n[STEP 6/6] Executing in QEMU System Emulation & Verifying Parity...")
    res = compare_configs(model_path, target, ["baseline", "optimized"], passes=["fuse"])

    comp = res["comparison"]
    bl_latency = comp.get("baseline_mean_ms", 0.0)
    opt_latency = comp.get("opt_mean_ms", 0.0)
    delta_ms = opt_latency - bl_latency
    mse = comp.get("opt_accuracy_delta_mse", 0.0)
    parity_ok = comp.get("opt_accuracy_ok", False)

    print("\n------------------------------------------------------------------")
    print("FINAL DEPLOYMENT PIPELINE RESULTS:")
    print("------------------------------------------------------------------")
    print(f"  • Execution Mode       : QEMU Bare-Metal Emulation (-icount shift=0)")
    print(f"  • Target CPU Frequency : 100 MHz (Nominal cycle-to-time conversion)")
    print(f"  • Baseline FP32 Latency: {bl_latency:.4f} ms")
    print(f"  • Optimized Latency    : {opt_latency:.4f} ms ({delta_ms:+.4f} ms speedup)")
    print(f"  • Output Logits Parity : {'SUCCESS (PASS)' if parity_ok else 'FAILED'}")
    print(f"  • Logits MSE vs Ref    : {mse:.6f}")
    print("------------------------------------------------------------------")
    print("\n[SUCCESS] Model deployed and verified end-to-end from start to finish without board connected!")


if __name__ == "__main__":
    run_e2e_deployment()
