"""Render docs/hardware-gap.html to TATVA-Hardware-Gap.pdf.

One A4 page explaining why TATVA cannot yet flash a physical RISC-V board.

Same approach as tools/make_stage_guide.py: headless Microsoft Edge, which is present
on every Windows 11 machine, so the build needs no Python PDF library and no download.
Chrome is accepted as a fallback.

    python tools/make_hardware_gap.py

Writes the PDF next to the repository root.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "hardware-gap.html"
OUTPUT = ROOT / "TATVA-Hardware-Gap.pdf"

BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def find_browser() -> Path:
    for candidate in BROWSERS:
        path = Path(candidate)
        if path.exists():
            return path
    found = shutil.which("msedge") or shutil.which("chrome")
    if found:
        return Path(found)
    raise SystemExit("No Edge or Chrome found; cannot render the PDF.")


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source document: {SOURCE}")

    browser = find_browser()
    print(f"Renderer : {browser}")
    print(f"Source   : {SOURCE}")

    # Headless Chromium refuses to reuse a live profile, so give it a throwaway one.
    with tempfile.TemporaryDirectory(prefix="tatva-pdf-") as profile:
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=6000",
            f"--print-to-pdf={OUTPUT}",
            SOURCE.as_uri(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit("The renderer produced no PDF.")

    print(f"Wrote    : {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
