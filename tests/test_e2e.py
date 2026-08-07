import shutil
from pathlib import Path
import pytest
import numpy as np
from tatva.compiler import import_model, analyze_graph, TARGETS
from tatva.optimizer import fuse_attention_softmax
from tatva.runner import establish_baseline, compile_model, run_and_measure, ExecutionEnvironment

@pytest.mark.integration
def test_e2e_pipeline_verification(pretrained_model_path, skip_if_no_toolchain, tolerance):
    # 1. Ingestion & Analysis
    ir = import_model(str(pretrained_model_path), input_shapes={
        "input_ids": (1, 32),
        "attention_mask": (1, 32),
        "token_type_ids": (1, 32)
    })
    stats = analyze_graph(ir)
    assert stats.total_ops > 0

    # 2. Baseline Verification
    variant = TARGETS["RV64GC"]
    baseline_res = establish_baseline(str(pretrained_model_path), variant)
    assert baseline_res.parity_passed is True

    # 3. Softmax/Attention Fusion Optimization
    fused_ir = fuse_attention_softmax(ir)
    assert fused_ir is not None
    assert fused_ir.metadata.get("softmax_optimized") is True

    # 4. Compilation & Execution of Optimized Artifact
    build_dir = Path("build_e2e")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    artifact = compile_model(fused_ir, variant, str(build_dir), warmup_count=2, timed_count=5)
    try:
        measurement = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)
        assert measurement.simulated is True
        assert measurement.mean_ms > 0.0
        
        # Assert parity of fused run vs baseline reference
        target_logits = []
        for line in measurement.raw_output.splitlines():
            if "FIRST_LOGITS:" in line:
                parts = line.strip().split(":")[1].strip().split()
                target_logits = [float(x) for x in parts]
                break
        
        # Verify both outputs have the same number of logits and match numerically
        assert len(target_logits) == len(baseline_res.target_logits)
        
        # Calculate MSE to check parity (Schraudolph's approximation has slightly higher variance but low MSE)
        mse = float(np.mean((np.array(target_logits) - np.array(baseline_res.target_logits)) ** 2))
        assert mse < tolerance
        
        # Verify latency speedup (fused is faster than baseline)
        assert measurement.mean_ms < baseline_res.latency_result.mean_ms
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)
