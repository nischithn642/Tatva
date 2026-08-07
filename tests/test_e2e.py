import pytest
import numpy as np
from tatva.compiler import import_model, analyze_graph, TARGETS
from tatva.optimizer import select_fast_softmax_kernel
from tatva.runner import establish_baseline, compile_model, run_and_measure, ExecutionEnvironment

@pytest.mark.integration
def test_e2e_pipeline_verification(pretrained_model_path, skip_if_no_toolchain, tolerance, tmp_path):
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
    fused_ir = select_fast_softmax_kernel(ir)
    assert fused_ir is not None
    assert fused_ir.metadata.get("softmax_optimized") is True

    # 4. Compilation & Execution of Optimized Artifact.
    # Build under tmp_path: the old fixed "build_e2e" directory in the repo root was
    # shared with every other run and cleaned up by hand in a finally block.
    build_dir = tmp_path / "e2e_build"

    artifact = compile_model(fused_ir, variant, str(build_dir), warmup_count=2, timed_count=5)
    measurement = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)
    assert measurement.simulated is True
    assert measurement.mean_ms > 0.0

    # Assert parity of the fused run against the HOST reference, not the QEMU
    # baseline. Scoring one RISC-V build against another only proves the two
    # agree; it says nothing about either being right.
    target_logits = []
    for line in measurement.raw_output.splitlines():
        if "FIRST_LOGITS:" in line:
            parts = line.strip().split(":")[1].strip().split()
            target_logits = [float(x) for x in parts]
            break

    assert len(target_logits) == len(baseline_res.ref_logits)

    # Schraudolph's approximation has slightly higher variance but low MSE
    mse = float(np.mean((np.array(target_logits) - np.array(baseline_res.ref_logits)) ** 2))
    assert mse < tolerance
