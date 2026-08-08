"""
Cross-platform installer for the RISC-V toolchain TATVA compiles and emulates with.

This replaces `setup_env.py` and `setup_simulators.py`, which between them:
  - hardcoded `win32-x64` asset URLs, so they only worked on Windows;
  - installed `apache-tvm` unpinned, contradicting the pins in pyproject.toml;
  - lived at the repo root, so `pip install tatva-compiler` never got them at all.

Everything here is pinned, verified after install, and reports what it is about to do
before it does it (`--dry-run`). Nothing is downloaded implicitly: `tatva setup` asks
first, because the two archives together are roughly half a gigabyte.

Binaries land in a per-user tools directory rather than the source tree, so an installed
wheel and a git checkout find the same toolchain. TATVA_TOOLS_DIR overrides.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass

# Called with (bytes_read, bytes_total) as a download runs; total is 0 when the server
# sends no Content-Length. The desktop app passes one of these so it can show a real
# progress bar instead of a spinner that sits there for 430 MB.
ProgressFn = Callable[[int, int], None] | None

__all__ = [
    "COMPONENTS",
    "Component",
    "InstallPlan",
    "ToolchainUnavailableError",
    "current_platform_slug",
    "install_component",
    "installed_components",
    "plan_install",
    "tools_dir",
]

# Pinned. These are the exact versions this project is tested against; `tatva doctor`
# reports what it actually found, which is the number that matters if you use your own.
GCC_VERSION = "15.2.0-1"
QEMU_VERSION = "9.2.4-1"


class ToolchainUnavailableError(RuntimeError):
    """Raised when there is no published build for the host platform."""


@dataclass(frozen=True)
class Component:
    """One downloadable toolchain component."""

    key: str
    label: str
    version: str
    # xPack release asset stem, e.g. "xpack-riscv-none-elf-gcc-15.2.0-1".
    asset_stem: str
    release_url: str
    # Directory name under tools_dir(), and the executable that proves it worked.
    install_name: str
    probe_exe: str
    approx_size_mb: int


COMPONENTS: dict[str, Component] = {
    "gcc": Component(
        key="gcc",
        label="RISC-V GCC cross-compiler (xPack riscv-none-elf-gcc)",
        version=GCC_VERSION,
        asset_stem=f"xpack-riscv-none-elf-gcc-{GCC_VERSION}",
        release_url=(
            "https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/"
            f"v{GCC_VERSION}"
        ),
        install_name="riscv-none-elf-gcc",
        probe_exe="riscv-none-elf-gcc",
        approx_size_mb=430,
    ),
    "qemu": Component(
        key="qemu",
        label="QEMU RISC-V system emulator (xPack qemu-riscv)",
        version=QEMU_VERSION,
        asset_stem=f"xpack-qemu-riscv-{QEMU_VERSION}",
        release_url=f"https://github.com/xpack-dev-tools/qemu-riscv-xpack/releases/download/v{QEMU_VERSION}",
        install_name="qemu-riscv",
        probe_exe="qemu-system-riscv64",
        approx_size_mb=90,
    ),
}


def tools_dir() -> str:
    """
    Where downloaded toolchains live.

    Not the source tree. `setup_env.py` unpacked into `<repo>/riscv-toolchain`, which
    means a pip-installed TATVA could never find it and a second checkout downloaded
    another 430 MB copy. TATVA_TOOLS_DIR overrides.
    """
    override = os.environ.get("TATVA_TOOLS_DIR")
    if override:
        return os.path.abspath(override)

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")

    return os.path.join(base, "tatva", "toolchains")


def current_platform_slug() -> str:
    """
    Return the xPack platform suffix for this host, e.g. "win32-x64", "linux-arm64".

    Raises ToolchainUnavailableError on a host xPack does not publish for, rather than
    guessing a URL that will 404 halfway through a download.
    """
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        arch = "x64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        raise ToolchainUnavailableError(
            f"No prebuilt RISC-V toolchain is published for CPU architecture '{platform.machine()}'. "
            "Install riscv-none-elf-gcc and qemu-system-riscv64 with your system package manager "
            "and put them on PATH; `tatva doctor` will pick them up."
        )

    if sys.platform == "win32":
        if arch != "x64":
            raise ToolchainUnavailableError(
                "xPack publishes Windows builds for x64 only. On Windows-on-ARM, use the x64 build "
                "under emulation or install the toolchain via WSL."
            )
        return "win32-x64"
    if sys.platform == "darwin":
        return f"darwin-{arch}"
    return f"linux-{arch}"


@dataclass(frozen=True)
class InstallPlan:
    """What install_component() would download, and where it would put it."""

    component: Component
    platform_slug: str
    url: str
    dest: str
    already_installed: bool

    def describe(self) -> str:
        state = "already installed" if self.already_installed else f"~{self.component.approx_size_mb} MB download"
        return (
            f"{self.component.label}\n"
            f"  version : {self.component.version} ({self.platform_slug})\n"
            f"  source  : {self.url}\n"
            f"  install : {self.dest}\n"
            f"  status  : {state}"
        )


def plan_install(component_key: str) -> InstallPlan:
    """
    Work out the exact URL and destination without touching the network.

    Kept separate from install_component so `tatva setup --dry-run` can show a user
    precisely what is about to be fetched from where before they agree to it.
    """
    try:
        component = COMPONENTS[component_key]
    except KeyError:
        raise ToolchainUnavailableError(
            f"Unknown component '{component_key}'. Known components: {', '.join(sorted(COMPONENTS))}."
        ) from None

    slug = current_platform_slug()
    ext = "zip" if slug.startswith("win32") else "tar.gz"
    url = f"{component.release_url}/{component.asset_stem}-{slug}.{ext}"
    dest = os.path.join(tools_dir(), component.install_name)

    return InstallPlan(
        component=component,
        platform_slug=slug,
        url=url,
        dest=dest,
        already_installed=find_installed_exe(component) is not None,
    )


def find_installed_exe(component: Component) -> str | None:
    """Return the path to this component's probe executable under tools_dir(), if present."""
    bin_dir = os.path.join(tools_dir(), component.install_name, "bin")
    for name in (component.probe_exe + ".exe", component.probe_exe):
        candidate = os.path.join(bin_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def installed_components() -> dict[str, str | None]:
    """Map component key -> installed executable path (or None)."""
    return {key: find_installed_exe(comp) for key, comp in COMPONENTS.items()}


def _download(url: str, dest_path: str, progress: bool = True, on_progress: ProgressFn = None) -> None:
    # GitHub release redirects reject the default urllib User-Agent with a 403.
    request = urllib.request.Request(url, headers={"User-Agent": "tatva-setup"})
    with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if progress and total:
                pct = read * 100 // total
                print(f"\r  downloading... {pct:3d}%  ({read // 1048576} / {total // 1048576} MB)", end="", flush=True)
            if on_progress is not None:
                on_progress(read, total)
    if progress:
        print()


def _extract(archive_path: str, into: str) -> str:
    """
    Unpack the archive and return the single top-level directory it contained.

    xPack archives wrap everything in one versioned folder; we strip it so the install
    path stays stable across version bumps.
    """
    os.makedirs(into, exist_ok=True)
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(into)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            # filter="data" refuses absolute paths, parent traversal and device nodes.
            # An archive fetched over the network gets no more trust than that.
            tf.extractall(into, filter="data")

    entries = [e for e in os.listdir(into) if os.path.isdir(os.path.join(into, e))]
    if len(entries) != 1:
        raise ToolchainUnavailableError(
            f"Expected exactly one top-level directory in {os.path.basename(archive_path)}, found {len(entries)}."
        )
    return os.path.join(into, entries[0])


def _restore_exec_bits(root: str) -> None:
    """
    Put the executable bit back on everything under bin/ and libexec/.

    tarfile's `data` filter clamps permissions, and a cross-compiler that cannot be
    executed fails later with a confusing PermissionError instead of at install time.
    """
    if sys.platform == "win32":
        return
    for sub in ("bin", "libexec"):
        base = os.path.join(root, sub)
        for dirpath, _dirnames, filenames in os.walk(base):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if os.path.islink(path):
                    continue
                mode = os.stat(path).st_mode
                os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_component(
    component_key: str,
    force: bool = False,
    progress: bool = True,
    on_progress: ProgressFn = None,
) -> str:
    """
    Download, unpack and verify one component. Returns the path to its probe executable.

    Refuses to clobber an existing install unless `force` is set -- re-downloading
    430 MB because a command was run twice is not a good default.
    """
    plan = plan_install(component_key)

    existing = find_installed_exe(plan.component)
    if existing and not force:
        return existing

    with tempfile.TemporaryDirectory(prefix="tatva-setup-") as tmp:
        archive = os.path.join(tmp, os.path.basename(plan.url))
        try:
            _download(plan.url, archive, progress=progress, on_progress=on_progress)
        except Exception as e:
            raise ToolchainUnavailableError(
                f"Could not download {plan.component.label} from {plan.url} ({e}).\n"
                "Check the release page for an asset matching this platform, or install the tool "
                "with your system package manager and put it on PATH."
            ) from e

        unpacked = _extract(archive, os.path.join(tmp, "unpacked"))
        _restore_exec_bits(unpacked)

        os.makedirs(tools_dir(), exist_ok=True)
        if os.path.exists(plan.dest):
            shutil.rmtree(plan.dest, ignore_errors=True)
        shutil.move(unpacked, plan.dest)

    installed = find_installed_exe(plan.component)
    if not installed:
        raise ToolchainUnavailableError(
            f"{plan.component.label} unpacked to {plan.dest} but '{plan.component.probe_exe}' is not in its bin/. "
            "The release layout may have changed; install manually and put it on PATH."
        )

    # Prove it runs on this host before claiming success. A wrong-architecture archive
    # extracts perfectly happily and only fails at the first real compile.
    try:
        subprocess.run([installed, "--version"], capture_output=True, timeout=60, check=True)
    except Exception as e:
        raise ToolchainUnavailableError(
            f"Installed {plan.component.label} to {plan.dest}, but '{plan.component.probe_exe} --version' "
            f"did not run ({e}). This usually means the archive does not match this platform."
        ) from e

    return installed
