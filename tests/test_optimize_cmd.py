"""
Tests for tatva optimize CLI command.
"""

import os

from click.testing import CliRunner

from tatva.cli import cli


def test_optimize_fuse_success(tmp_path) -> None:
    """
    Assert that running tatva optimize with fuse pass succeeds, writes binary
    and report.json, and prints comparison table.
    """
    runner = CliRunner()
    model_path = "models/model.onnx"
    out_dir = str(tmp_path / "build_fused")

    result = runner.invoke(
        cli,
        [
            "optimize",
            model_path,
            "--target",
            "RV64GC",
            "--passes",
            "fuse",
            "--out",
            out_dir,
        ],
    )

    assert result.exit_code == 0
    assert os.path.exists(os.path.join(out_dir, "model.elf"))
    assert os.path.exists(os.path.join(out_dir, "report.json"))
    assert "Benchmark Results:" in result.output
    assert "Mean Latency" in result.output
    assert "Success: Optimized artifact written" in result.output


def test_optimize_quantize_warning(tmp_path) -> None:
    """
    Assert that running tatva optimize with quantize pass prints a warning
    notifying the user of the known dynamic quantization regression.
    """
    runner = CliRunner()
    model_path = "models/model.onnx"
    out_dir = str(tmp_path / "build_quantized")

    result = runner.invoke(
        cli,
        [
            "optimize",
            model_path,
            "--target",
            "RV64GC",
            "--passes",
            "quantize",
            "--out",
            out_dir,
        ],
    )

    assert (
        "Warning: 'quantize' is an experimental pass with a known latency regression"
        in result.output
    )
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(out_dir, "model.elf"))
    assert os.path.exists(os.path.join(out_dir, "report.json"))


def test_optimize_already_exists(tmp_path) -> None:
    """
    Assert that running tatva optimize on an existing directory aborts
    safely to prevent silently overwriting artifacts.
    """
    runner = CliRunner()
    model_path = "models/model.onnx"
    out_dir = str(tmp_path / "already_exists")

    # Create the directory beforehand to trigger safety block
    os.makedirs(out_dir)

    result = runner.invoke(
        cli,
        [
            "optimize",
            model_path,
            "--target",
            "RV64GC",
            "--passes",
            "fuse",
            "--out",
            out_dir,
        ],
    )

    assert result.exit_code != 0
    assert "Error: Output directory" in result.output
    assert "Re-run with --force to replace it" in result.output


def test_optimize_force_replaces_existing_output(tmp_path) -> None:
    """
    Assert --force replaces an existing --out directory instead of aborting.

    The refusal above is correct, but before --force existed the only way past it was
    to delete the directory by hand, so re-running a build was a two-step chore.
    """
    runner = CliRunner()
    out_dir = tmp_path / "force_me"
    out_dir.mkdir()
    stale = out_dir / "stale.txt"
    stale.write_text("output from a previous run")

    result = runner.invoke(
        cli,
        ["optimize", "models/model.onnx", "--target", "RV64GC", "--passes", "fuse", "--out", str(out_dir), "--force"],
    )

    assert result.exit_code == 0, result.output
    assert os.path.exists(os.path.join(str(out_dir), "model.elf"))
    # The directory was replaced, not merged into.
    assert not stale.exists()
