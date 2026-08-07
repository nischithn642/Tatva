"""
Build Automation Script for Packaging Tatva Desktop GUI into a Standalone Executable.

Usage:
    python scratch/build_app.py
"""

import os
import shutil
import subprocess
import sys
import time

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_dir_size_mb(path: str) -> float:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)


def main() -> None:
    print("=" * 80)
    print("              TATVA DESKTOP GUI STANDALONE PACKAGING BUILD              ")
    print("=" * 80)

    spec_file = os.path.join(PROJECT_DIR, "tatva.spec")
    if not os.path.exists(spec_file):
        print(f"Error: Spec file not found at {spec_file}", file=sys.stderr)
        sys.exit(1)

    pyinstaller_bin = shutil.which("pyinstaller") or os.path.join(PROJECT_DIR, ".venv", "Scripts", "pyinstaller.exe")
    if not os.path.exists(pyinstaller_bin) and not shutil.which("pyinstaller"):
        print("Error: PyInstaller executable not found. Install via 'pip install pyinstaller'.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Project Directory:     {PROJECT_DIR}")
    print(f"[*] PyInstaller Binary:    {pyinstaller_bin}")
    print(f"[*] Specification File:    {spec_file}")
    print("[*] Launching PyInstaller build process...")

    start_time = time.time()
    cmd = [pyinstaller_bin, spec_file, "--noconfirm", "--clean"]

    try:
        res = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=300)
        elapsed = round(time.time() - start_time, 2)

        if res.returncode != 0:
            print("\n[ERROR] PyInstaller compilation failed:")
            print(res.stderr)
            sys.exit(1)

        dist_dir = os.path.join(PROJECT_DIR, "dist", "tatva")
        exe_path = os.path.join(dist_dir, "tatva.exe") if sys.platform == "win32" else os.path.join(dist_dir, "tatva")

        if not os.path.exists(dist_dir):
            print(f"\n[ERROR] Output directory not found at {dist_dir}")
            sys.exit(1)

        size_mb = get_dir_size_mb(dist_dir)

        print("\n" + "=" * 80)
        print("                       BUILD SUMMARY & ARTIFACT METRICS                  ")
        print("=" * 80)
        print(f"Status:            SUCCESS")
        print(f"Build Time:        {elapsed} seconds")
        print(f"Artifact Path:     {exe_path}")
        print(f"Bundle Directory:  {dist_dir}")
        print(f"Bundle Total Size: {size_mb} MB")
        print("=" * 80)
        print("\nStandalone app is ready for deployment!")
    except Exception as e:
        print(f"\n[ERROR] Build execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
