"""
Tests for tatva target registry and verification.
"""

import pytest

from tatva.compiler import DEFAULT_TARGET, TARGETS
from tatva.runner import find_qemu, find_riscv_gcc, verify_target


@pytest.mark.unit
def test_registry_contents() -> None:
    """
    Verify that all expected targets exist and have the correct attributes.
    """
    assert len(TARGETS) == 6
    expected_targets = {"RV32IMC", "RV32IMAC", "RV64GC", "RV64IMAFDC", "RV64GCV", "RV32EMC"}
    assert set(TARGETS.keys()) == expected_targets

    # Bitness checks
    assert TARGETS["RV32IMC"].bitness == 32
    assert TARGETS["RV32IMAC"].bitness == 32
    assert TARGETS["RV64GC"].bitness == 64
    assert TARGETS["RV64IMAFDC"].bitness == 64
    assert TARGETS["RV64GCV"].bitness == 64
    assert TARGETS["RV32EMC"].bitness == 32

    # Default Target
    assert DEFAULT_TARGET == "RV64GC"
    assert TARGETS[DEFAULT_TARGET].experimental is False

    # Production vs Experimental Target checks
    assert TARGETS["RV64GCV"].experimental is False
    assert TARGETS["RV32EMC"].experimental is True


@pytest.mark.integration
def test_verify_default_target_e2e(skip_if_no_toolchain) -> None:
    """
    Run verify_target end-to-end for the default target (RV64GC) if tools are present.
    """
    gcc_name, gcc_path = find_riscv_gcc()
    qemu_name, qemu_path = find_qemu(64)

    if not gcc_path or not qemu_path:
        pytest.skip("Skipping end-to-end target verification because GCC or QEMU-64 is missing.")

    variant = TARGETS["RV64GC"]
    result = verify_target(variant)

    assert result["status"] == "ok"
    assert result["error"] == ""
    assert f"Hello from Target: {variant.name}" in result["output"]


@pytest.mark.integration
def test_verify_32bit_target_e2e(skip_if_no_toolchain) -> None:
    """
    Run verify_target end-to-end for a 32-bit target (RV32IMC) if tools are present.
    """
    gcc_name, gcc_path = find_riscv_gcc()
    qemu_name, qemu_path = find_qemu(32)

    if not gcc_path or not qemu_path:
        pytest.skip("Skipping end-to-end target verification because GCC or QEMU-32 is missing.")

    variant = TARGETS["RV32IMC"]
    result = verify_target(variant)

    assert result["status"] == "ok"
    assert result["error"] == ""
    assert f"Hello from Target: {variant.name}" in result["output"]
