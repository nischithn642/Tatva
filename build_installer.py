"""Build TATVA-Setup-beta-2.0.exe — a single-file Windows installer.

    python build_installer.py

Two steps. First, PyInstaller freezes installer/tatva_setup.py into a small
one-file stub — just Python and Tkinter, about 15 MB. Second, the distribution
zip is appended to that stub, followed by a fixed-size footer recording where it
starts and how long it is.

Appending rather than bundling is what keeps the installer responsive: a
--add-data payload would be unpacked to %TEMP% before the first window appeared,
which for a 250 MB archive means a long silent wait and 250 MB of temp space. The
appended copy is read in place.

Run build_exe.py first — this script installs whatever that produced.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "installer" / "tatva_setup.py"
ICON = ROOT / "assets" / "tatva.ico"
DIST = ROOT / "dist"
BUILD = ROOT / "build" / "installer"
STUB_DIR = BUILD / "stub"
STUB_NAME = "TATVA-Setup-stub"
OUTPUT = DIST / "TATVA-Setup-beta-2.0.exe"

FOOTER_MAGIC = b"TATVAPKG1"
FOOTER_FORMAT = "<QQ"

# Nothing in the wizard needs the scientific stack, but PyInstaller will happily
# vacuum it up out of the environment if a transitive import mentions it.
EXCLUDES = (
    "numpy",
    "onnx",
    "onnxruntime",
    "tvm",
    "scipy",
    "PIL",
    "matplotlib",
    "pandas",
    "webview",
    "pytest",
)

RULE = "=" * 62


def find_payload() -> Path:
    candidates = sorted(DIST.glob("TATVA-*-windows.zip"))
    if not candidates:
        raise SystemExit("No distribution zip in dist/. Run: python build_exe.py")
    payload = candidates[-1]
    with zipfile.ZipFile(payload) as archive:  # refuse to ship a truncated archive
        names = archive.namelist()
        if not names:
            raise SystemExit(f"{payload.name} is empty.")
        prefixes = {name.split("/")[0] for name in names}
        if len(prefixes) != 1:
            raise SystemExit(f"{payload.name} has multiple top-level folders: {sorted(prefixes)}")
    return payload


def python_with_pyinstaller() -> str:
    """PyInstaller lives in the project venv, which is not always the interpreter
    this script was started with."""
    candidates = [sys.executable, str(ROOT / ".venv" / "Scripts" / "python.exe")]
    for candidate in candidates:
        if not Path(candidate).exists():
            continue
        probe = subprocess.run([candidate, "-c", "import PyInstaller"], capture_output=True, text=True)
        if probe.returncode == 0:
            return candidate
    raise SystemExit("PyInstaller is not installed. Run: uv pip install pyinstaller")


def build_stub() -> Path:
    if STUB_DIR.exists():
        shutil.rmtree(STUB_DIR, ignore_errors=True)

    cmd = [
        python_with_pyinstaller(),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        STUB_NAME,
        "--distpath",
        str(STUB_DIR),
        "--workpath",
        str(BUILD / "work"),
        "--specpath",
        str(BUILD),
    ]
    if ICON.exists():
        cmd += ["--icon", str(ICON)]
    for module in EXCLUDES:
        cmd += ["--exclude-module", module]
    cmd.append(str(SCRIPT))

    print("Freezing the wizard…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    stub = STUB_DIR / f"{STUB_NAME}.exe"
    if result.returncode != 0 or not stub.exists():
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit("PyInstaller failed to build the installer stub.")
    print(f"  stub       {stub.stat().st_size / 1024**2:6.1f} MB")
    return stub


def append_payload(stub: Path, payload: Path) -> None:
    if OUTPUT.exists():
        OUTPUT.unlink()

    print("Appending the payload…")
    with OUTPUT.open("wb") as out:
        with stub.open("rb") as handle:
            shutil.copyfileobj(handle, out, 1024 * 1024)
        offset = out.tell()
        with payload.open("rb") as handle:
            shutil.copyfileobj(handle, out, 1024 * 1024)
        length = out.tell() - offset
        out.write(FOOTER_MAGIC + struct.pack(FOOTER_FORMAT, offset, length))

    print(f"  payload    {length / 1024**2:6.1f} MB  at offset {offset:,}")


def verify() -> None:
    """Read the finished exe back the same way the installer will."""
    size = OUTPUT.stat().st_size
    footer_size = len(FOOTER_MAGIC) + struct.calcsize(FOOTER_FORMAT)
    with OUTPUT.open("rb") as handle:
        handle.seek(size - footer_size)
        footer = handle.read(footer_size)
        if not footer.startswith(FOOTER_MAGIC):
            raise SystemExit("The footer is missing from the finished installer.")
        offset, length = struct.unpack(FOOTER_FORMAT, footer[len(FOOTER_MAGIC) :])

        sys.path.insert(0, str(ROOT / "installer"))
        from tatva_setup import PayloadSlice

        with zipfile.ZipFile(PayloadSlice(handle, offset, length)) as archive:
            names = archive.namelist()

    has_exe = any(name.endswith("/TATVA.exe") for name in names)
    has_gcc = any("riscv-none-elf-gcc.exe" in name for name in names)
    has_qemu = any("qemu-system-riscv64.exe" in name for name in names)
    has_guide = any(name.endswith("TATVA-Stage-Guide.pdf") for name in names)

    print("Verifying…")
    print(f"  entries    {len(names):,}")
    for label, ok in (
        ("TATVA.exe", has_exe),
        ("RISC-V gcc", has_gcc),
        ("QEMU", has_qemu),
        ("stage guide", has_guide),
    ):
        print(f"  {label:<12} {'found' if ok else 'MISSING'}")
    if not (has_exe and has_gcc and has_qemu):
        raise SystemExit("The payload is incomplete.")


def main() -> int:
    print(RULE)
    print("TATVA installer build")
    print(RULE)

    if not SCRIPT.exists():
        raise SystemExit(f"Missing {SCRIPT}")

    payload = find_payload()
    print(f"Payload    {payload.name}  ({payload.stat().st_size / 1024**2:.1f} MB)")

    stub = build_stub()
    append_payload(stub, payload)
    verify()

    print(RULE)
    print(f"Setup file  {OUTPUT}")
    print(f"            {OUTPUT.stat().st_size / 1024**2:.1f} MB")
    print(RULE)
    print()
    print("Send this one file. Double-click installs to the user's own profile —")
    print("no administrator password, no downloads, nothing else to set up.")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
