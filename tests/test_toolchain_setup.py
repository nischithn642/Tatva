"""
Tests for `tatva setup` and the cross-platform toolchain installer behind it.

Nothing here touches the network. What is worth testing is the part that used to be
wrong: which asset gets picked for which host, where it lands, and whether the rest of
TATVA can then find it. The old setup_env.py hardcoded `win32-x64` in the URL, so on
Linux and macOS it downloaded Windows binaries and then failed at the first compile.
"""

import os
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tatva.cli import cli
from tatva.runner import find_qemu, find_riscv_gcc
from tatva.toolchain import (
    COMPONENTS,
    ToolchainUnavailableError,
    current_platform_slug,
    install_component,
    installed_components,
    plan_install,
    tools_dir,
)

KNOWN_SLUGS = {"win32-x64", "linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64"}


@pytest.fixture
def tools_root(tmp_path, monkeypatch):
    """Point the installer at a throwaway tools directory."""
    root = tmp_path / "tools"
    monkeypatch.setenv("TATVA_TOOLS_DIR", str(root))
    return root


def _plant_fake_exe(tools_root, install_name: str, exe: str) -> str:
    """Create a file where a real installed binary would be."""
    bin_dir = tools_root / install_name / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / (exe + ".exe" if sys.platform == "win32" else exe)
    path.write_text("#!/bin/sh\nexit 0\n")
    return str(path)


@pytest.mark.unit
def test_platform_slug_is_one_xpack_actually_publishes() -> None:
    assert current_platform_slug() in KNOWN_SLUGS


@pytest.mark.unit
def test_unknown_cpu_architecture_says_so_instead_of_guessing_a_url() -> None:
    """
    A 404 halfway through a 430 MB download is a bad way to learn your CPU is not
    supported. Fail before the request, with something actionable.
    """
    with (
        patch("platform.machine", return_value="riscv64"),
        pytest.raises(ToolchainUnavailableError) as excinfo,
    ):
        current_platform_slug()

    message = str(excinfo.value)
    assert "riscv64" in message
    assert "PATH" in message


@pytest.mark.unit
def test_tools_dir_honours_the_override(tools_root) -> None:
    assert tools_dir() == str(tools_root)


@pytest.mark.unit
def test_tools_dir_defaults_outside_the_source_tree(monkeypatch) -> None:
    """
    setup_env.py unpacked into `<repo>/riscv-toolchain`, so a pip-installed TATVA could
    never find the toolchain and a second checkout re-downloaded the whole thing.
    """
    monkeypatch.delenv("TATVA_TOOLS_DIR", raising=False)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    resolved = tools_dir()

    assert os.path.commonpath([resolved, repo_root]) != repo_root
    assert resolved.endswith(os.path.join("tatva", "toolchains"))


@pytest.mark.unit
@pytest.mark.parametrize("key", ["gcc", "qemu"])
def test_plan_targets_this_platforms_asset(key, tools_root) -> None:
    plan = plan_install(key)

    slug = current_platform_slug()
    expected_ext = "zip" if slug.startswith("win32") else "tar.gz"

    assert plan.url.startswith("https://github.com/xpack-dev-tools/")
    assert plan.url.endswith(f"{slug}.{expected_ext}")
    assert plan.component.version in plan.url
    assert plan.dest == os.path.join(str(tools_root), COMPONENTS[key].install_name)
    assert plan.already_installed is False


@pytest.mark.unit
def test_plan_rejects_an_unknown_component() -> None:
    with pytest.raises(ToolchainUnavailableError) as excinfo:
        plan_install("llvm")

    assert "llvm" in str(excinfo.value)
    assert "gcc" in str(excinfo.value)


@pytest.mark.unit
def test_plan_notices_an_existing_install(tools_root) -> None:
    _plant_fake_exe(tools_root, "qemu-riscv", "qemu-system-riscv64")

    assert plan_install("qemu").already_installed is True
    assert installed_components()["qemu"] is not None
    assert installed_components()["gcc"] is None


@pytest.mark.unit
def test_install_does_not_redownload_what_is_already_there(tools_root) -> None:
    """
    Re-running `tatva setup` should be free, not another 430 MB.
    """
    planted = _plant_fake_exe(tools_root, "riscv-none-elf-gcc", "riscv-none-elf-gcc")

    with patch("tatva.toolchain._download", side_effect=AssertionError("should not download")) as mock_dl:
        result = install_component("gcc")

    mock_dl.assert_not_called()
    assert result == planted


@pytest.mark.unit
def test_a_failed_download_is_reported_with_the_url(tools_root) -> None:
    with (
        patch("tatva.toolchain._download", side_effect=OSError("HTTP 404")),
        pytest.raises(ToolchainUnavailableError) as excinfo,
    ):
        install_component("qemu")

    message = str(excinfo.value)
    assert "404" in message
    assert "xpack-dev-tools" in message
    assert "PATH" in message


@pytest.mark.unit
def test_runner_finds_a_toolchain_installed_by_setup(tools_root) -> None:
    """
    The installer and the discovery code have to agree on a directory layout. They
    previously did not: setup wrote to `<repo>/riscv-toolchain`, discovery searched it,
    and neither worked for a wheel install.
    """
    gcc = _plant_fake_exe(tools_root, "riscv-none-elf-gcc", "riscv-none-elf-gcc")
    qemu = _plant_fake_exe(tools_root, "qemu-riscv", "qemu-system-riscv64")

    with patch("shutil.which", return_value=None):
        gcc_name, gcc_path = find_riscv_gcc()
        qemu_name, qemu_path = find_qemu(64)

    assert gcc_name == "riscv-none-elf-gcc"
    assert os.path.samefile(gcc_path, gcc)
    assert qemu_name == "qemu-system-riscv64"
    assert os.path.samefile(qemu_path, qemu)


@pytest.mark.unit
def test_path_still_wins_over_the_tools_directory(tools_root) -> None:
    """A toolchain the user put on PATH deliberately is the one they meant."""
    _plant_fake_exe(tools_root, "riscv-none-elf-gcc", "riscv-none-elf-gcc")

    with patch("shutil.which", side_effect=lambda name: "/usr/bin/riscv-none-elf-gcc" if "gcc" in name else None):
        _name, path = find_riscv_gcc()

    assert path == "/usr/bin/riscv-none-elf-gcc"


@pytest.mark.unit
def test_legacy_repo_toolchain_directory_is_still_searched(tmp_path, monkeypatch) -> None:
    """
    Anyone who already ran the old setup_env.py has a working toolchain in
    `<repo>/riscv-toolchain`. Dropping that search path would make them re-download it
    for no reason.
    """
    monkeypatch.setenv("TATVA_TOOLS_DIR", str(tmp_path / "empty-tools"))
    monkeypatch.setattr("tatva.runner.PROJECT_DIR", str(tmp_path / "repo"))

    legacy_bin = tmp_path / "repo" / "riscv-toolchain" / "bin"
    legacy_bin.mkdir(parents=True)
    exe = legacy_bin / ("riscv-none-elf-gcc.exe" if sys.platform == "win32" else "riscv-none-elf-gcc")
    exe.write_text("")

    with patch("shutil.which", return_value=None):
        _name, path = find_riscv_gcc()

    assert os.path.samefile(path, str(exe))


@pytest.mark.unit
def test_setup_dry_run_downloads_nothing(tools_root) -> None:
    runner = CliRunner()

    with patch("tatva.toolchain.install_component", side_effect=AssertionError("should not install")):
        res = runner.invoke(cli, ["setup", "--dry-run"])

    assert res.exit_code == 0, res.output
    assert "Dry run: nothing downloaded" in res.output
    assert "xpack-dev-tools" in res.output
    assert str(tools_root) in res.output


@pytest.mark.unit
def test_setup_asks_before_downloading_half_a_gigabyte(tools_root) -> None:
    runner = CliRunner()

    with patch("tatva.toolchain.install_component", side_effect=AssertionError("should not install")):
        res = runner.invoke(cli, ["setup"], input="n\n")

    assert res.exit_code == 1
    assert "Aborted." in res.output


@pytest.mark.unit
def test_setup_reports_a_component_that_failed(tools_root) -> None:
    def _fail(key, force=False, progress=True):
        raise ToolchainUnavailableError(f"no build for {key}")

    runner = CliRunner()
    with patch("tatva.toolchain.install_component", side_effect=_fail):
        res = runner.invoke(cli, ["setup", "--component", "qemu", "--yes"])

    assert res.exit_code == 1
    assert "no build for qemu" in res.output
    assert "failed to install" in res.output


@pytest.mark.unit
def test_setup_is_a_no_op_when_everything_is_present(tools_root) -> None:
    _plant_fake_exe(tools_root, "riscv-none-elf-gcc", "riscv-none-elf-gcc")
    _plant_fake_exe(tools_root, "qemu-riscv", "qemu-system-riscv64")

    runner = CliRunner()
    with patch("tatva.toolchain.install_component", side_effect=AssertionError("should not install")):
        res = runner.invoke(cli, ["setup"])

    assert res.exit_code == 0, res.output
    assert "already installed" in res.output


@pytest.mark.unit
def test_setup_rejects_an_unknown_component_name() -> None:
    res = CliRunner().invoke(cli, ["setup", "--component", "llvm"])

    assert res.exit_code == 2
    assert "llvm" in res.output
