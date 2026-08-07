"""
Tests for tatva doctor CLI command.
"""

import json
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tatva.cli import cli


@pytest.mark.integration
def test_doctor_json_all_present(skip_if_no_toolchain) -> None:
    """
    Assert that doctor --json command works when all dependencies are present.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)

    assert "python" in data
    assert "tvm" in data
    assert "onnxruntime" in data
    assert "riscv_gcc" in data
    assert "qemu" in data

    assert data["python"]["status"] == "ok"
    assert data["tvm"]["status"] == "ok"
    assert data["onnxruntime"]["status"] == "ok"
    assert data["riscv_gcc"]["status"] == "ok"
    assert data["qemu"]["status"] == "ok"


@pytest.mark.integration
def test_doctor_text_all_present(skip_if_no_toolchain) -> None:
    """
    Assert that doctor command prints OK statuses in human-readable mode.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "[OK] python" in result.output
    assert "[OK] tvm" in result.output
    assert "[OK] onnxruntime" in result.output
    assert "[OK] riscv_gcc" in result.output
    assert "[OK] qemu" in result.output


@pytest.mark.unit
def test_doctor_missing_gcc() -> None:
    """
    Assert friendly error report when RISC-V GCC is missing.
    """
    with patch("tatva.cli.find_riscv_gcc", return_value=(None, None)):
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["riscv_gcc"]["status"] == "error"
        assert "not found" in data["riscv_gcc"]["error"]


@pytest.mark.unit
def test_doctor_missing_qemu() -> None:
    """
    Assert friendly error report when QEMU is missing.
    """
    with patch("tatva.cli.find_qemu", return_value=(None, None)):
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["qemu"]["status"] == "error"
        assert "not found" in data["qemu"]["error"]


@pytest.mark.unit
def test_doctor_says_where_it_looked_for_a_missing_tool(tmp_path, monkeypatch) -> None:
    """
    "not found" invites the question "where did you look?". Answer it up front.
    """
    monkeypatch.setenv("TATVA_TOOLS_DIR", str(tmp_path / "tools"))

    with patch("tatva.cli.find_riscv_gcc", return_value=(None, None)):
        result = CliRunner().invoke(cli, ["doctor"])

    assert result.exit_code == 1
    assert "Looked in: PATH" in result.output
    assert str(tmp_path / "tools") in result.output
    assert "tatva setup" in result.output


@pytest.mark.unit
def test_doctor_json_carries_the_search_paths(tmp_path, monkeypatch) -> None:
    """The same information has to reach anything parsing --json, not just humans."""
    monkeypatch.setenv("TATVA_TOOLS_DIR", str(tmp_path / "tools"))

    with patch("tatva.cli.find_qemu", return_value=(None, None)):
        result = CliRunner().invoke(cli, ["doctor", "--json"])

    searched = json.loads(result.output)["qemu"]["searched"]
    assert searched[0] == "PATH"
    assert any("qemu-system-riscv64" in entry for entry in searched[1:])


@pytest.mark.unit
def test_doctor_missing_tvm() -> None:
    """
    Assert friendly error report when tvm is missing.
    """
    # Remove tvm from sys.modules and map it to None to simulate ImportError
    with patch.dict(sys.modules, {"tvm": None}):
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["tvm"]["status"] == "error"
        assert "not installed" in data["tvm"]["error"]


@pytest.mark.unit
def test_doctor_missing_onnxruntime() -> None:
    """
    Assert friendly error report when onnxruntime is missing.
    """
    with patch.dict(sys.modules, {"onnxruntime": None}):
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["onnxruntime"]["status"] == "error"
        assert "not installed" in data["onnxruntime"]["error"]
