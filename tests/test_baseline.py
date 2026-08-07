"""
Tests for tatva baseline establishment and numerical parity checks.
"""

import sys
from unittest.mock import patch

import pytest

from tatva.compiler import TARGETS
from tatva.runner import establish_baseline


@pytest.mark.integration
def test_baseline_parity_success(skip_if_no_toolchain) -> None:
    """
    Assert that establish_baseline runs successfully and passes numerical parity checks.
    """
    res = establish_baseline("models/model.onnx", TARGETS["RV64GC"])
    assert res is not None
    assert res.parity_passed is True
    assert len(res.ref_logits) == 5
    assert len(res.target_logits) == 5
    assert res.latency_result.simulated is True


@pytest.mark.integration
def test_baseline_parity_failure_on_corrupt(skip_if_no_toolchain) -> None:
    """
    Assert that a mismatch between host reference and simulated output fails parity assertion.
    """
    original_run_and_measure = sys.modules["tatva.runner"].run_and_measure

    def corrupt_run_and_measure(*args, **kwargs):
        measurement = original_run_and_measure(*args, **kwargs)
        # Corrupt the printed logits in raw output
        corrupted_lines = []
        for line in measurement.raw_output.splitlines():
            if "FIRST_LOGITS:" in line:
                corrupted_lines.append("FIRST_LOGITS: 9.99 9.99 9.99 9.99 9.99")
            else:
                corrupted_lines.append(line)
        measurement.raw_output = "\n".join(corrupted_lines)
        return measurement

    with patch("tatva.runner.run_and_measure", side_effect=corrupt_run_and_measure):
        with pytest.raises(AssertionError) as exc_info:
            establish_baseline("models/model.onnx", TARGETS["RV64GC"])
        assert "Numerical parity check failed" in str(exc_info.value)
