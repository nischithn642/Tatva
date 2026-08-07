"""
Tests for tatva CLI command surface, target validation, and stub commands.
"""

import pytest
from click.testing import CliRunner

from tatva.cli import cli


@pytest.mark.unit
def test_cli_help() -> None:
    """
    Assert that help menus exist and display successfully for all commands.
    """
    runner = CliRunner()
    commands = [None, "doctor", "targets", "analyze", "baseline-test", "optimize", "diagnose"]
    for cmd in commands:
        args = ["--help"] if cmd is None else [cmd, "--help"]
        res = runner.invoke(cli, args)
        assert res.exit_code == 0
        assert "Options:" in res.output or "Usage:" in res.output


# No stubs remaining to test as of Milestone M3


@pytest.mark.unit
def test_cli_experimental_rejection() -> None:
    """
    Assert that selecting an experimental target is gated by --allow-experimental.
    """
    runner = CliRunner()

    # Rejects experimental variant without the allowance option
    res = runner.invoke(cli, ["baseline-test", "models/model.onnx", "--target", "RV32EMC"])
    assert res.exit_code != 0
    assert "is experimental" in res.output

    # Also check the optimize command target validation
    res = runner.invoke(cli, ["optimize", "models/model.onnx", "--target", "RV32EMC"])
    assert res.exit_code != 0
    assert "is experimental" in res.output

    # Accepts experimental variant if allow-experimental is set (mock compile/run)
    from unittest.mock import MagicMock, patch
    with patch("tatva.cli.establish_baseline") as mock_baseline:
        mock_res = MagicMock()
        mock_res.parity_passed = True
        mock_res.latency_result.mean_ms = 10.0
        mock_res.latency_result.median_ms = 10.0
        mock_res.latency_result.p95_ms = 10.0
        mock_res.latency_result.environment = "QEMU_SIM"
        mock_res.latency_result.simulated = True
        mock_res.latency_result.raw_samples_ms = [10.0]
        mock_res.ref_logits = [1.0]
        mock_res.target_logits = [1.0]
        mock_res.tolerance = 1e-4
        mock_baseline.return_value = mock_res

        res2 = runner.invoke(cli, ["baseline-test", "models/model.onnx", "--allow-experimental", "--target", "RV32EMC"])
        assert res2.exit_code == 0


@pytest.mark.integration
def test_cli_baseline_test(skip_if_no_toolchain) -> None:
    """
    Assert that the baseline-test CLI flow executes successfully on a valid model.
    """
    runner = CliRunner()
    res = runner.invoke(cli, ["baseline-test", "models/model.onnx", "--target", "RV64GC"])
    assert res.exit_code == 0
    assert "Numerical Parity Verification: SUCCESS" in res.output
