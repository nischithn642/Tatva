"""
Build TATVA.exe and wrap it in the ZIP that gets shared.

    python build_exe.py            # build, then zip
    python build_exe.py --no-zip   # build only
    python build_exe.py --zip-only # zip an existing dist/TATVA

Produces:
    dist/TATVA/TATVA.exe              double-click to launch
    dist/TATVA-<version>-windows.zip  unzip anywhere, run TATVA.exe

Why a folder in a zip rather than a single self-extracting .exe: a one-file build
unpacks Apache TVM's native libraries to a temp directory on every launch, which is
most of a gigabyte of copying before the window appears. The folder build starts in
seconds. The zip is what makes it one thing to send someone.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, "dist")
APP_DIR = os.path.join(DIST, "TATVA")
EXE_PATH = os.path.join(APP_DIR, "TATVA.exe")


def version() -> str:
    """The PEP 440 version, e.g. "2.0.0b1". What packaging cares about."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        from tatva import __version__

        return __version__
    except Exception:
        return "0.0.0"


def display_version() -> str:
    """
    The name the release goes by, e.g. "Beta 2.0".

    The zip filename uses this rather than the PEP 440 string. Someone receiving
    "TATVA-2.0.0b1-windows.zip" has to work out that b1 means beta 2.0; the file is
    the first thing they see and it should say what it is.
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        from tatva import DISPLAY_VERSION

        return DISPLAY_VERSION
    except Exception:
        return version()


def zip_slug() -> str:
    """display_version() as a filename fragment: "Beta 2.0" -> "beta-2.0"."""
    return display_version().strip().lower().replace(" ", "-")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def tree_size(path: str) -> int:
    total = 0
    for dirpath, _, names in os.walk(path):
        for name in names:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(dirpath, name))
    return total


def build() -> None:
    if shutil.which("pyinstaller") is None:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            sys.exit(
                "PyInstaller is not installed in this environment.\n"
                "  pip install pyinstaller      (or: uv pip install pyinstaller)"
            )

    print("Building TATVA.exe — this takes a few minutes.\n")
    cmd = [sys.executable, "-m", "PyInstaller", "tatva.spec", "--noconfirm", "--clean"]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"PyInstaller exited with {result.returncode}. The build did not complete.")

    if not os.path.exists(EXE_PATH):
        sys.exit(f"Build reported success but {EXE_PATH} is not there. Nothing to ship.")


def write_readme() -> None:
    """
    A plain-text note next to the exe, for whoever receives the zip.

    They will not have the repo, the docs site, or this conversation -- so anything
    they need in the first five minutes has to be in the folder itself.
    """
    text = f"""TATVA {display_version()} — Bare-Metal RISC-V Optimization Studio
==================================================================
build {version()}


TO START
--------
Double-click TATVA.exe.

Windows SmartScreen may warn that the publisher is unknown, because this build is
not code-signed. Click "More info" then "Run anyway".

The first launch takes a few seconds while the compiler backend loads.


THE FIVE STAGES
---------------
  01 INPUT      load an ONNX model (samples are bundled under models/)
  02 ANALYZE    read the graph, count operators, detect the attention pattern
  03 MAP        check every operator against the chosen RISC-V target
  04 OPTIMIZE   choose the passes: softmax fusion, INT8 quantization
  05 GENERATE   emit C99, cross-compile, run under QEMU, measure both builds

Stages 01-04 work the moment you unzip this. Stage 05 shells out to a RISC-V
cross-compiler and QEMU, so it needs them present -- see below.


THE RISC-V TOOLCHAIN
--------------------
Not bundled: the two archives together are about half a gigabyte and licensed
separately.

Get them from inside the app:

  Diagnostics  ->  Install toolchain

That downloads the pinned xPack builds (riscv-none-elf-gcc and
qemu-system-riscv64), unpacks them into your own user folder, and re-checks.
No admin rights. Nothing is added to PATH. Nothing else on the machine changes.

If you already have your own build of either tool on PATH, TATVA uses that and
the install step is unnecessary.


EVERYTHING RUNS LOCALLY
-----------------------
No account, no upload, no telemetry. Models never leave this machine.

Two exceptions, both opt-in and both obvious when they happen: pressing "Install
toolchain" downloads from the xPack project's GitHub releases, and the optional
Assistant talks to whichever LLM provider you configure yourself.


MEASUREMENT
-----------
Latency figures come from QEMU system-mode emulation with the cycle counter read
on the target, converted at a nominal 100 MHz. They are emulator cycles, not
silicon, and are meant for comparing two builds of the same model against each
other -- not for quoting absolute performance on real hardware.

A run that reports 0.00% change is a real result, not a failure: it means the
passes you selected had nothing to change in that particular graph. Softmax
fusion needs an attention pattern to fuse -- model_mlp.onnx has none, model.onnx
does.
"""
    with open(os.path.join(APP_DIR, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)


def make_zip() -> str:
    if not os.path.isdir(APP_DIR):
        sys.exit(f"{APP_DIR} does not exist. Build first (drop --zip-only).")

    write_readme()
    zip_path = os.path.join(DIST, f"TATVA-{zip_slug()}-windows.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    files = []
    for dirpath, _, names in os.walk(APP_DIR):
        for name in names:
            files.append(os.path.join(dirpath, name))

    print(f"\nZipping {len(files):,} files…")
    # ZIP_DEFLATED rather than the newer codecs: the receiving machine has to be able
    # to open this with the Explorer right-click "Extract All" and nothing else.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, path in enumerate(files, 1):
            # Everything sits under a single TATVA/ folder, so extracting never sprays
            # a thousand files across whatever directory they happened to be in.
            arc = os.path.join("TATVA", os.path.relpath(path, APP_DIR))
            z.write(path, arc)
            if i % 500 == 0:
                print(f"  {i:,}/{len(files):,}")

    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and package TATVA.exe")
    ap.add_argument("--no-zip", action="store_true", help="build the exe but do not zip it")
    ap.add_argument("--zip-only", action="store_true", help="zip an existing dist/TATVA")
    args = ap.parse_args()

    if not args.zip_only:
        build()

    print("\n" + "=" * 62)
    print(f"  exe     {EXE_PATH}")
    print(f"  folder  {human(tree_size(APP_DIR))}")

    if not args.no_zip:
        zip_path = make_zip()
        print(f"  zip     {zip_path}")
        print(f"          {human(os.path.getsize(zip_path))}")
    print("=" * 62)
    print("\nSend the zip. The person who receives it unzips it and runs TATVA.exe.")


if __name__ == "__main__":
    main()
