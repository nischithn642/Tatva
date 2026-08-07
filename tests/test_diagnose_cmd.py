"""
Tests for tatva diagnose CLI command and diagnostics pipeline integrations.
"""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tatva.cli import cli
from tatva.diagnostics import AccuracyDropError, MemoryLimitExceededError, UnsupportedOperatorError


@pytest.mark.unit
def test_diagnose_unsupported_operator() -> None:
    """
    Assert that UnsupportedOperatorError during pipeline runs displays the friendly
    explanation, hides tracebacks by default, displays them with --debug, and
    emits structured JSON with --json option.
    """
    runner = CliRunner()

    # 1. Test baseline-test fails with clean explanation and no traceback
    with patch("tatva.cli.establish_baseline") as mock_baseline:
        mock_baseline.side_effect = UnsupportedOperatorError(
            operator_name="CustomUnsupportedOp"
        )
        res = runner.invoke(cli, ["baseline-test", "models/model.onnx"])
        assert res.exit_code != 0
        assert "Diagnostics Explanation:" in res.output
        assert "Unsupported operator" in res.output
        assert "CustomUnsupportedOp" in res.output
        assert "Traceback" not in res.output

        # Test baseline-test with --debug prints traceback
        res_debug = runner.invoke(cli, ["baseline-test", "models/model.onnx", "--debug"])
        assert res_debug.exit_code != 0
        assert "Traceback" in res_debug.output

    # 2. Test diagnose command error handling
    with patch("tatva.cli.import_model") as mock_import:
        mock_import.side_effect = UnsupportedOperatorError(
            operator_name="CustomUnsupportedOp"
        )

        # Test diagnose --json prints structured JSON error payload
        res_json = runner.invoke(cli, ["diagnose", "models/model.onnx", "--json"])
        assert res_json.exit_code != 0
        assert "unsupported_operator" in res_json.output

        # Test diagnose prints plain-text explanation
        res_txt = runner.invoke(cli, ["diagnose", "models/model.onnx"])
        assert res_txt.exit_code != 0
        assert "Unsupported operator" in res_txt.output


@pytest.mark.unit
def test_diagnose_memory_limit_exceeded() -> None:
    """
    Assert that MemoryLimitExceededError triggers clean diagnostics explanations
    in optimize pipeline runs without raw tracebacks.
    """
    runner = CliRunner()

    with patch("tatva.optimizer.compare_configs") as mock_compare:
        mock_compare.side_effect = MemoryLimitExceededError(
            limit_bytes=100, required_bytes=200
        )

        res = runner.invoke(
            cli,
            ["optimize", "models/model.onnx", "--out", "build_nonexistent_temp"],
        )
        assert res.exit_code != 0
        assert "Diagnostics Explanation:" in res.output
        assert "Memory limit exceeded" in res.output
        assert "100" in res.output
        assert "200" in res.output
        assert "Traceback" not in res.output


@pytest.mark.unit
def test_diagnose_accuracy_drop() -> None:
    """
    Assert that AccuracyDropError displays friendly diagnostics explanations.
    """
    runner = CliRunner()

    with patch("tatva.cli.import_model") as mock_import:
        mock_import.side_effect = AccuracyDropError(mse=0.123, tolerance=0.05)

        res = runner.invoke(cli, ["diagnose", "models/model.onnx"])
        assert res.exit_code != 0
        assert "Diagnostics Explanation:" in res.output
        assert "Accuracy degradation check failed" in res.output
        assert "0.123" in res.output
        assert "0.05" in res.output
        assert "Traceback" not in res.output


@pytest.mark.unit
def test_diagnose_json_report(tmp_path) -> None:
    """
    Assert that the diagnose command accepts a saved JSON failure report and re-explains it.
    """
    runner = CliRunner()
    report_file = tmp_path / "failure_report.json"
    report_data = {
        "error_type": "memory_limit_exceeded",
        "metadata": {"limit_bytes": 500, "required_bytes": 1000},
    }

    with open(report_file, "w") as f:
        json.dump(report_data, f)

    res = runner.invoke(cli, ["diagnose", str(report_file)])
    assert res.exit_code == 0
    assert "Diagnostics Explanation:" in res.output
    assert "Memory limit exceeded" in res.output
    assert "500" in res.output
    assert "1000" in res.output
