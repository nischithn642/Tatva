"""
Tests for tatva compilation and latency measurement runner.
"""

import pytest

from tatva.compiler import TARGETS, import_model
from tatva.runner import (
    ExecutionEnvironment,
    compile_model,
    find_qemu,
    find_riscv_gcc,
    run_and_measure,
)


@pytest.mark.integration
def test_compile_and_measure_e2e(skip_if_no_toolchain) -> None:
    """
    Assert that we can compile and run a model under QEMU system mode
    and get deterministic simulated latency measurements.
    """
    _gcc_name, gcc_path = find_riscv_gcc()
    _qemu_name, qemu_path = find_qemu(64)

    if not gcc_path or not qemu_path:
        pytest.skip("Skipping runner e2e test because GCC or QEMU-64 is missing.")

    # Load small ONNX model
    model_path = "models/model.onnx"
    model_ir = import_model(model_path)

    # Compile for default target RV64GC
    variant = TARGETS["RV64GC"]
    artifact = compile_model(model_ir, variant, warmup_count=1, timed_count=3)
    assert artifact is not None
    assert artifact.elf_path.endswith(".elf")

    # Run in simulator
    result = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)
    assert result is not None
    assert result.environment == "QEMU_SIM"
    assert result.simulated is True
    assert len(result.raw_samples_ms) == 3
    assert result.mean_ms > 0.0
    assert result.median_ms > 0.0
    assert result.p95_ms > 0.0
