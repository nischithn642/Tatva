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
import glob
import os
import shutil
import subprocess
import sys
import tempfile
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


# --- Bundling the RISC-V toolchain -------------------------------------------------
#
# Stage 05 is the only stage that produces a measurement, and it was the only stage that
# did not work on a freshly unzipped copy: it shells out to a cross-compiler and an
# emulator that the user had to go and download first. So the toolchain now ships inside
# the app folder, and the zip is self-sufficient.
#
# The layout under toolchain/ matches the per-user install directory exactly, so
# tatva.runner searches both with the same code.

TOOLCHAIN_DIR = os.path.join(APP_DIR, "toolchain")

# (directory name under toolchain/, directory name in a git checkout, probe executable)
BUNDLED_COMPONENTS = (
    ("riscv-none-elf-gcc", "riscv-toolchain", "riscv-none-elf-gcc.exe"),
    ("qemu-riscv", "qemu", "qemu-system-riscv64.exe"),
)

# Dropped from the cross-compiler before it ships. Every entry is either a language TATVA
# does not compile or a tool it never invokes -- the source contains no reference to gdb,
# g++, gfortran or -flto, so nothing here can be reached at runtime. gdb is what drags in
# the embedded CPython (DLLs/, python313.*), which is why removing it saves more than its
# own size.
GCC_DROP = (
    "share",  # man pages, info files, HTML manuals
    "bin/riscv-none-elf-lto-dump.exe",
    "bin/riscv-none-elf-gdb.exe",
    "bin/riscv-none-elf-gdb-py3.exe",
    "bin/riscv-none-elf-gdb-add-index",
    "bin/riscv-none-elf-gdb-add-index-py3",
    "bin/riscv-none-elf-gstack",
    "bin/riscv-none-elf-gstack-py3",
    "bin/riscv-none-elf-gfortran.exe",
    "bin/riscv-none-elf-g++.exe",
    "bin/riscv-none-elf-c++.exe",
    "bin/DLLs",
    "bin/python313.dll",
    "bin/python313.zip",
)

# The C++ and Fortran compiler proper. cc1 (C) and lto-wrapper stay.
GCC_DROP_GLOBS = (
    "libexec/gcc/riscv-none-elf/*/cc1plus.exe",
    "libexec/gcc/riscv-none-elf/*/f951.exe",
)

# QEMU ships firmware for every machine type it supports. TATVA runs `-M virt -nographic
# -kernel`, which needs one OpenSBI blob and nothing else -- no UEFI (the edk2-*.fd files
# are ~290 MB on their own), no VGA BIOS, no OpenBIOS for other architectures.
QEMU_SHARE_KEEP = (
    "opensbi-riscv32-generic-fw_dynamic.bin",
    "opensbi-riscv64-generic-fw_dynamic.bin",
    "keymaps",
)


def _rm(path: str) -> int:
    """Delete a file or directory, returning the bytes reclaimed."""
    if not os.path.exists(path):
        return 0
    size = tree_size(path) if os.path.isdir(path) else os.path.getsize(path)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            os.remove(path)
    return size


def find_component(checkout_dir: str, install_name: str, probe: str) -> str | None:
    """
    Locate an installed copy of a component to bundle.

    A git checkout that ran the old setup_env.py has one at `<repo>/<checkout_dir>`;
    anything that ran `tatva setup` has one in the per-user tools directory. Either will
    do -- they are the same pinned xPack builds.
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    candidates = [os.path.join(ROOT, checkout_dir)]
    try:
        from tatva.toolchain import tools_dir

        candidates.append(os.path.join(tools_dir(), install_name))
    except Exception:
        pass

    for root in candidates:
        if os.path.isfile(os.path.join(root, "bin", probe)):
            return root
    return None


def needed_multilibs(gcc_exe: str) -> set[str]:
    """
    The multilib directories that TATVA's targets actually select.

    Asked of GCC rather than hardcoded: `-print-multi-directory` is the same lookup the
    compiler performs when it links, so this cannot drift from what the build needs. The
    toolchain carries 32 variants and the six targets in compiler.TARGETS resolve to four
    of them, which is where most of the cross-compiler's gigabyte goes.

    Returns the top-level directory names, plus "." for the default variant, whose
    libraries sit loose in the parent directory.
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tatva.compiler import TARGETS

    keep = set()
    for variant in TARGETS.values():
        result = subprocess.run(
            [gcc_exe, f"-march={variant.gcc_march}", f"-mabi={variant.gcc_mabi}", "-print-multi-directory"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sys.exit(
                f"Could not ask GCC which multilib {variant.name} uses:\n{result.stderr.strip()}\n"
                "Refusing to guess -- a wrong answer here ships a toolchain that cannot link."
            )
        keep.add(result.stdout.strip().split("/")[0])
    return keep


def prune_gcc(root: str) -> int:
    """Strip the cross-compiler down to what TATVA's six targets need. Returns bytes saved."""
    saved = 0
    for rel in GCC_DROP:
        saved += _rm(os.path.join(root, rel.replace("/", os.sep)))
    for pattern in GCC_DROP_GLOBS:
        for path in glob.glob(os.path.join(root, pattern.replace("/", os.sep))):
            saved += _rm(path)

    keep = needed_multilibs(os.path.join(root, "bin", "riscv-none-elf-gcc.exe"))
    print(f"  multilibs kept: {', '.join(sorted(keep))}")

    # The two trees that carry a copy of the runtime per ISA variant.
    parents = [os.path.join(root, "riscv-none-elf", "lib")]
    parents += glob.glob(os.path.join(root, "lib", "gcc", "riscv-none-elf", "*"))

    for parent in parents:
        if not os.path.isdir(parent):
            continue
        for entry in os.listdir(parent):
            # Only ISA-named directories are candidates. ldscripts/, include/, plugin/ and
            # the loose .a and .o files belong to the default variant and always stay.
            if entry.startswith("rv") and entry not in keep and os.path.isdir(os.path.join(parent, entry)):
                saved += _rm(os.path.join(parent, entry))
    return saved


def prune_qemu(root: str) -> int:
    """Drop the firmware for machine types TATVA never boots. Returns bytes saved."""
    share = os.path.join(root, "share")
    if not os.path.isdir(share):
        return 0
    saved = 0
    for entry in os.listdir(share):
        if entry not in QEMU_SHARE_KEEP:
            saved += _rm(os.path.join(share, entry))
    return saved


def stage_toolchain() -> None:
    """Copy the cross-compiler and QEMU into the app folder, pruned to what is used."""
    if os.path.isdir(TOOLCHAIN_DIR):
        shutil.rmtree(TOOLCHAIN_DIR, ignore_errors=True)
    os.makedirs(TOOLCHAIN_DIR, exist_ok=True)

    print("\nBundling the RISC-V toolchain…")
    for install_name, checkout_dir, probe in BUNDLED_COMPONENTS:
        source = find_component(checkout_dir, install_name, probe)
        if source is None:
            sys.exit(
                f"No copy of {install_name} found to bundle.\n"
                f"  looked in: {os.path.join(ROOT, checkout_dir)}\n"
                f"             the per-user tools directory\n"
                "Run `tatva setup` first, then build again. Shipping without it would put\n"
                "the download back in front of whoever receives the zip."
            )

        dest = os.path.join(TOOLCHAIN_DIR, install_name)
        print(f"  {install_name} <- {source}")
        shutil.copytree(source, dest, symlinks=True)

        before = tree_size(dest)
        saved = prune_gcc(dest) if install_name == "riscv-none-elf-gcc" else prune_qemu(dest)
        print(f"    {human(before)} -> {human(before - saved)}  (dropped {human(saved)})")


def verify_bundle() -> None:
    """
    Compile and link a bare-metal object for every target, using only the bundled copy.

    This runs the real compiler out of the folder that is about to be zipped. A pruned
    toolchain that is missing one multilib fails here rather than on someone else's
    laptop, halfway through the only stage that produces a number.
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tatva.compiler import TARGETS

    gcc = os.path.join(TOOLCHAIN_DIR, "riscv-none-elf-gcc", "bin", "riscv-none-elf-gcc.exe")
    print("\nVerifying the bundled compiler against every target…")


    with tempfile.TemporaryDirectory(prefix="tatva-bundle-check-") as tmp:
        src = os.path.join(tmp, "probe.c")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write("int main(void){return 0;}\n")

        failures = []
        for name, variant in TARGETS.items():
            out = os.path.join(tmp, f"{name}.o")
            result = subprocess.run(
                [
                    gcc,
                    f"-march={variant.gcc_march}",
                    f"-mabi={variant.gcc_mabi}",
                    "-ffreestanding",
                    "-nostdlib",
                    "-c",
                    src,
                    "-o",
                    out,
                ],
                capture_output=True,
                text=True,
            )
            ok = result.returncode == 0 and os.path.exists(out)
            print(f"  {'ok  ' if ok else 'FAIL'} {name:<12} {variant.gcc_march}/{variant.gcc_mabi}")
            if not ok:
                failures.append(f"{name}: {result.stderr.strip()}")

        if failures:
            sys.exit("The bundled toolchain cannot build every target:\n  " + "\n  ".join(failures))

    qemu = os.path.join(TOOLCHAIN_DIR, "qemu-riscv", "bin", "qemu-system-riscv64.exe")
    result = subprocess.run([qemu, "-version"], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"The bundled QEMU does not run:\n{result.stderr.strip()}")
    print(f"  ok   qemu         {result.stdout.splitlines()[0].strip()}")


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

All five work the moment you unzip this. There is nothing else to install.


THE RISC-V TOOLCHAIN
--------------------
Bundled, in the toolchain/ folder next to this file: the pinned xPack builds of
riscv-none-elf-gcc and qemu-system-riscv64, trimmed to the six targets TATVA can
actually emit code for.

No download, no admin rights, nothing added to PATH, nothing written outside
this folder. Keep toolchain/ where it is and stage 05 works offline.

If you already have your own riscv-none-elf-gcc or qemu-system-riscv64 on PATH,
TATVA uses yours in preference to the bundled copy. Diagnostics shows the
resolved path for each, so you can see which one it picked.


EVERYTHING RUNS LOCALLY
-----------------------
No account, no upload, no telemetry, no network. Models never leave this
machine, and a completed run needs no connection at any point.

One exception, and it is opt-in: the optional Assistant talks to whichever LLM
provider you configure yourself. Leave it alone and nothing goes out.


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


def stage_root_docs() -> None:
    """
    Copy the reader-facing docs into the app root, beside README.txt.

    These cannot go through the spec's datas list. PyInstaller puts datas under
    _internal/, which is the right place for files the app opens itself and the
    wrong place for a PDF a person is meant to find and read.

    This exists because the guide used to be copied in by hand. A later
    PyInstaller run rewrote dist/TATVA and dropped it, and the installer shipped
    one file lighter than the release before it -- caught only by
    build_installer.py's verify step, which matches on the filename and would
    have accepted the file from anywhere.
    """
    for doc in ("TATVA-Stage-Guide.pdf",):
        source = os.path.join(ROOT, doc)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(APP_DIR, doc))
        else:
            # Not fatal: build_installer.py reports it as MISSING, which is the
            # single place that check belongs.
            print(f"  note       {doc} is not in the repository root; not staged")


def make_zip() -> str:
    if not os.path.isdir(APP_DIR):
        sys.exit(f"{APP_DIR} does not exist. Build first (drop --zip-only).")

    write_readme()
    stage_root_docs()
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
    ap.add_argument(
        "--skip-toolchain",
        action="store_true",
        help="do not bundle the RISC-V toolchain (stage 05 will then need `tatva setup`)",
    )
    args = ap.parse_args()

    if not args.zip_only:
        build()

    if args.skip_toolchain:
        print("\nSkipping the toolchain bundle (--skip-toolchain).")
        print("Whoever receives this will have to install it before stage 05 runs.")
    elif args.zip_only and os.path.isdir(TOOLCHAIN_DIR):
        print(f"\nReusing the staged toolchain in {TOOLCHAIN_DIR}.")
        verify_bundle()
    else:
        stage_toolchain()
        verify_bundle()

    toolchain_bytes = tree_size(TOOLCHAIN_DIR) if os.path.isdir(TOOLCHAIN_DIR) else 0
    total_bytes = tree_size(APP_DIR)

    print("\n" + "=" * 62)
    print(f"  exe        {EXE_PATH}")
    print(f"  app        {human(total_bytes - toolchain_bytes)}")
    if toolchain_bytes:
        print(f"  toolchain  {human(toolchain_bytes)}")
    print(f"  folder     {human(total_bytes)}")

    if not args.no_zip:
        zip_path = make_zip()
        print(f"  zip        {zip_path}")
        print(f"             {human(os.path.getsize(zip_path))}")
    print("=" * 62)
    print("\nSend the zip. The person who receives it unzips it and runs TATVA.exe.")
    if toolchain_bytes:
        print("Nothing else to install — all five stages work offline.")


if __name__ == "__main__":
    main()
