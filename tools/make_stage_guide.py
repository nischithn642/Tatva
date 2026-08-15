"""Render docs/stage-guide.html to TATVA-Stage-Guide.pdf.

Uses headless Microsoft Edge, which is present on every Windows 11 machine, so the
build needs no Python PDF library and no download. Chrome is accepted as a fallback
for anyone building on a machine where Edge has been removed.

    python tools/make_stage_guide.py

Writes the PDF next to the repository root and, when a packaged build exists, into
dist/TATVA/ as well so it ships with the app.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "stage-guide.html"
OUTPUT = ROOT / "TATVA-Stage-Guide.pdf"

# The cover line carrying the release name. It is the only version string in the
# document, and it is stamped from the package before rendering -- the 2.1 zip
# otherwise shipped a guide whose cover read "Beta 2.0 · build 2.0.0b1", because a
# static HTML file has nothing to keep it honest.
META_LINE = re.compile(r'(<div class="meta">)TATVA · [^<]*(</div>)')

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


def stamp_version() -> str:
    """Write the current release onto the cover, in the source and so in the PDF."""
    sys.path.insert(0, str(ROOT / "src"))
    from tatva import DISPLAY_VERSION, __version__

    caption = f"TATVA · {DISPLAY_VERSION} · build {__version__}"
    source = SOURCE.read_text(encoding="utf-8")
    stamped, count = META_LINE.subn(rf"\g<1>{caption}\g<2>", source)
    if count != 1:
        raise SystemExit(f'{SOURCE.name}: expected one <div class="meta"> cover line, found {count}.')
    if stamped != source:
        SOURCE.write_text(stamped, encoding="utf-8")
    return caption


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source document: {SOURCE}")

    print(f"Cover    : {stamp_version()}")
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

    shipped = ROOT / "dist" / "TATVA"
    if shipped.is_dir():
        shutil.copy2(OUTPUT, shipped / OUTPUT.name)
        print(f"Copied   : {shipped / OUTPUT.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
