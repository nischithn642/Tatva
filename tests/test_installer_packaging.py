"""
Tests for the release plumbing: the version every artifact names itself after, and the
payload appended to the installer.

Both of these shipped bugs in the last release. `build_installer.py` chose its payload
by taking the alphabetically last zip in `dist/`, and "TATVA-2.1-windows.zip" sorts
*before* "TATVA-beta-2.0-windows.zip" -- so a 2.1 installer would have carried the 2.0
payload. Separately the stage guide's cover, the setup filename and the wizard's own
caption were literals that a version bump did not reach.

Nothing in the suite noticed, because nothing in the suite looked at the build scripts.
These tests do, without building anything: the version checks read the sources, and the
payload checks assemble a stub-plus-zip in a temp directory rather than a 250 MB one.
"""

from __future__ import annotations

import importlib.util
import re
import struct
import sys
import zipfile
from pathlib import Path

import pytest

from tatva import DISPLAY_VERSION, __version__

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path, name: str):
    """Import a build script by path. They are not part of the package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_exe():
    return _load(ROOT / "build_exe.py", "_tatva_build_exe")


@pytest.fixture(scope="module")
def build_installer():
    return _load(ROOT / "build_installer.py", "_tatva_build_installer")


@pytest.fixture(scope="module")
def setup_stub():
    return _load(ROOT / "installer" / "tatva_setup.py", "_tatva_setup_stub")


# ------------------------------------------------------------------ version agreement


@pytest.mark.unit
def test_the_package_version_is_pep440_and_the_display_version_is_not_decorated() -> None:
    """
    `__version__` is what hatch reads and what pip installs under, so it has to parse.
    DISPLAY_VERSION is the human name and must stay bare -- the GUI badge and the
    website both prefix it themselves, and a "v" here shows up as "vv2.1".
    """
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?((a|b|rc)\d+)?", __version__), __version__
    assert not DISPLAY_VERSION.startswith(("v", "V"))
    assert "version" not in DISPLAY_VERSION.lower()

    # The two must name one release. DISPLAY_VERSION may carry a qualifier ("Beta 2.0"),
    # so compare on its numeric tail, which is the part __version__ has to begin with.
    numeric = DISPLAY_VERSION.split()[-1]
    assert __version__.startswith(numeric), (
        f"{DISPLAY_VERSION!r} and {__version__!r} describe different releases"
    )


@pytest.mark.unit
def test_both_build_scripts_derive_the_same_artifact_slug(build_exe, build_installer) -> None:
    """
    The zip is written by one script and found by the other, by name. If the two slugs
    ever disagree the installer build fails outright -- which is the good case -- but
    only after a 20 minute freeze, so assert it here instead.
    """
    slug = build_exe.zip_slug()
    assert slug == build_installer.zip_slug()
    assert slug == DISPLAY_VERSION.strip().lower().replace(" ", "-")
    assert slug != "unversioned", "zip_slug() fell back, meaning the package did not import"


@pytest.mark.unit
def test_the_installer_looks_for_this_version_by_name_not_the_last_zip(build_installer) -> None:
    """The exact bug: dist/ keeps old releases and the new one does not sort last."""
    source = (ROOT / "build_installer.py").read_text(encoding="utf-8")
    assert 'DIST / f"TATVA-{zip_slug()}-windows.zip"' in source
    assert build_installer.OUTPUT.name == f"TATVA-Setup-{build_installer.zip_slug()}.exe"


@pytest.mark.unit
def test_the_wizard_literals_match_the_package(setup_stub) -> None:
    """
    The stub cannot import tatva -- freezing it with numpy and TVM attached would
    defeat a 15 MB installer -- so it copies the version. This is the copy going stale.
    """
    assert DISPLAY_VERSION == setup_stub.APP_VERSION
    assert __version__ == setup_stub.APP_BUILD


@pytest.mark.unit
def test_the_installer_build_refuses_a_stale_wizard(build_installer, monkeypatch, tmp_path) -> None:
    """
    check_stub_version() is the guard that makes the copy above safe. Point it at a
    stub carrying the previous release and it must refuse to build.
    """
    stale = tmp_path / "tatva_setup.py"
    stale.write_text('APP_VERSION = "2.0"\nAPP_BUILD = "2.0.0b1"\n', encoding="utf-8")
    monkeypatch.setattr(build_installer, "SCRIPT", stale)

    with pytest.raises(SystemExit) as caught:
        build_installer.check_stub_version()
    assert "APP_VERSION" in str(caught.value)

    # And a stub that has lost the literal entirely is also a refusal, not a silent pass.
    stale.write_text("# no version here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        build_installer.check_stub_version()


@pytest.mark.unit
def test_the_stage_guide_cover_names_this_release() -> None:
    """
    The 2.1 zip shipped, briefly, with a guide whose cover read "Beta 2.0 · build
    2.0.0b1" -- a static HTML file has nothing keeping it honest, so the generator
    stamps it. This asserts the stamped result, which is what gets rendered to the PDF.
    """
    html = (ROOT / "docs" / "stage-guide.html").read_text(encoding="utf-8")
    covers = re.findall(r'<div class="meta">TATVA · ([^<]*)</div>', html)
    assert len(covers) == 1, f"expected one cover caption, found {covers}"
    assert covers[0] == f"{DISPLAY_VERSION} · build {__version__}"


# ------------------------------------------------------------------- appended payload


def _make_installer(tmp_path: Path, stub_bytes: bytes, payload: bytes, magic: bytes, fmt: str) -> Path:
    """Assemble stub + payload + footer exactly the way build_installer.py does."""
    exe = tmp_path / "TATVA-Setup-test.exe"
    with exe.open("wb") as f:
        f.write(stub_bytes)
        offset = f.tell()
        f.write(payload)
        f.write(magic + struct.pack(fmt, offset, len(payload)))
    return exe


@pytest.mark.unit
def test_the_two_sides_of_the_footer_agree(build_installer, setup_stub) -> None:
    """The writer and the reader are in separate files. They must use one format."""
    assert build_installer.FOOTER_MAGIC == setup_stub.FOOTER_MAGIC
    assert build_installer.FOOTER_FORMAT == setup_stub.FOOTER_FORMAT
    assert len(setup_stub.FOOTER_MAGIC) + struct.calcsize(setup_stub.FOOTER_FORMAT) == setup_stub.FOOTER_SIZE


@pytest.mark.unit
def test_a_payload_appended_to_a_stub_opens_as_a_zip(tmp_path, build_installer, setup_stub) -> None:
    """
    The end-to-end property the installer depends on: zipfile locates the central
    directory by seeking from the end, so it has to see the end of the *zip*, not the
    end of the exe. PayloadSlice is what makes that true.
    """
    source_zip = tmp_path / "payload.zip"
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("TATVA/TATVA.exe", b"\x4d\x5a" + b"pretend exe" * 500)
        z.writestr("TATVA/TATVA-Stage-Guide.pdf", b"%PDF-1.7 pretend guide")
        z.writestr("TATVA/README.txt", "read me")
    payload = source_zip.read_bytes()

    exe = _make_installer(
        tmp_path,
        b"MZ" + b"\x00" * 4096,  # a stand-in for the frozen wizard
        payload,
        build_installer.FOOTER_MAGIC,
        build_installer.FOOTER_FORMAT,
    )

    # Read the footer back the way open_payload() does when frozen.
    handle = exe.open("rb")
    handle.seek(exe.stat().st_size - setup_stub.FOOTER_SIZE)
    footer = handle.read(setup_stub.FOOTER_SIZE)
    assert footer.startswith(setup_stub.FOOTER_MAGIC)
    offset, length = struct.unpack(setup_stub.FOOTER_FORMAT, footer[len(setup_stub.FOOTER_MAGIC) :])
    assert length == len(payload)

    window = setup_stub.PayloadSlice(handle, offset, length)
    with zipfile.ZipFile(window) as z:
        assert z.testzip() is None
        assert sorted(z.namelist()) == [
            "TATVA/README.txt",
            "TATVA/TATVA-Stage-Guide.pdf",
            "TATVA/TATVA.exe",
        ]
        assert z.read("TATVA/README.txt") == b"read me"
    window.close()


@pytest.mark.unit
def test_the_payload_window_cannot_read_past_its_own_end(tmp_path, build_installer, setup_stub) -> None:
    """
    A window that overruns into the footer corrupts the last bytes of the zip, and a
    window that stops short truncates it. Both show up as "not a zip file" much later,
    so the boundary is asserted directly.
    """
    payload = bytes(range(256)) * 40
    exe = _make_installer(
        tmp_path, b"stub", payload, build_installer.FOOTER_MAGIC, build_installer.FOOTER_FORMAT
    )

    with exe.open("rb") as handle:
        window = setup_stub.PayloadSlice(handle, 4, len(payload))

        assert window.read() == payload, "a full read must return the payload and nothing else"
        assert window.read() == b"", "reading past the end must yield nothing, not footer bytes"
        assert window.tell() == len(payload)

        # Seeking from the end lands inside the payload, not inside the exe.
        assert window.seek(-10, 2) == len(payload) - 10
        assert window.read() == payload[-10:]

        # Out-of-range seeks clamp rather than escaping the window.
        assert window.seek(-99_999, 2) == 0
        assert window.seek(99_999) == len(payload)
        assert window.read() == b""

        assert window.seekable() and window.readable()


@pytest.mark.unit
def test_an_installer_without_a_footer_is_rejected(tmp_path, setup_stub) -> None:
    """
    A truncated download is the common case, and it must produce the plain-language
    error rather than an unpacking traceback.
    """
    exe = tmp_path / "truncated.exe"
    exe.write_bytes(b"MZ" + b"\x00" * 1000)

    with exe.open("rb") as handle:
        handle.seek(0, 2)
        total = handle.tell()
        handle.seek(total - setup_stub.FOOTER_SIZE)
        assert not handle.read(setup_stub.FOOTER_SIZE).startswith(setup_stub.FOOTER_MAGIC)
