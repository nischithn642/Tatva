"""
Tests for RISC-V Vector Extension (RV64GCV) compilation, code generation, and QEMU execution.
"""

import numpy as np
import pytest

from tatva.compiler import TARGETS, import_model, analyze_graph
from tatva.optimizer import fuse_attention_softmax
from tatva.runner import (
    ExecutionEnvironment,
    compile_model,
    establish_baseline,
    run_and_measure,
    verify_target,
)


@pytest.mark.unit
def test_rv64gcv_target_definition() -> None:
    """
    Assert that RV64GCV target is present in TARGETS registry, configured for rv64gcv,
    and marked as non-experimental production target.
    """
    assert "RV64GCV" in TARGETS
    variant = TARGETS["RV64GCV"]
    assert variant.name == "RV64GCV"
    assert variant.gcc_march == "rv64gcv"
    assert variant.gcc_mabi == "lp64d"
    assert variant.bitness == 64
    assert variant.experimental is False


@pytest.mark.integration
def test_rv64gcv_target_verification(skip_if_no_toolchain) -> None:
    """
    Assert that verify_target succeeds for RV64GCV target with GCC -march=rv64gcv and QEMU -cpu rv64,v=true.
    """
    variant = TARGETS["RV64GCV"]
    res = verify_target(variant)
    assert res["status"] == "ok"
    assert "Hello from Target: RV64GCV" in res["output"]


@pytest.mark.integration
def test_rv64gcv_model_compilation_and_parity(pretrained_model_path, skip_if_no_toolchain, tolerance, tmp_path) -> None:
    """
    Assert that an ONNX model compiles for RV64GCV, runs in QEMU with vector extension enabled,
    and maintains numerical parity against host reference logits.
    """
    ir = import_model(
        str(pretrained_model_path),
        input_shapes={"input_ids": (1, 32), "attention_mask": (1, 32), "token_type_ids": (1, 32)},
    )
    stats = analyze_graph(ir)
    assert stats.total_ops > 0

    variant = TARGETS["RV64GCV"]
    fused_ir = fuse_attention_softmax(ir)
    assert fused_ir.metadata.get("softmax_optimized") is True

    build_dir = tmp_path / "rvv_build"
    artifact = compile_model(fused_ir, variant, str(build_dir), warmup_count=2, timed_count=5)
    assert artifact.variant.name == "RV64GCV"

    measurement = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)
    assert measurement.simulated is True
    assert measurement.mean_ms > 0.0

    target_logits = []
    for line in measurement.raw_output.splitlines():
        if "FIRST_LOGITS:" in line:
            parts = line.strip().split(":")[1].strip().split()
            target_logits = [float(x) for x in parts]
            break

    assert len(target_logits) > 0

    baseline_res = establish_baseline(str(pretrained_model_path), variant)
    assert baseline_res.parity_passed is True

    mse = float(np.mean((np.array(target_logits) - np.array(baseline_res.target_logits[:len(target_logits)])) ** 2))
    assert mse < tolerance
