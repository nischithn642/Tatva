"""
Smoke test for tatva package.
"""

import re

import pytest

import tatva


@pytest.mark.unit
def test_smoke() -> None:
    """
    Assert tatva can be imported and has version metadata.
    """
    assert hasattr(tatva, "__version__")
    assert isinstance(tatva.__version__, str)
    # PEP 440, which is what hatchling and pip accept: a release like 2.0.0 or a
    # pre-release like 2.0.0b1. The old pattern allowed only three plain numbers and
    # rejected the beta the project actually ships as.
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[ab]|rc)?\d*", tatva.__version__), tatva.__version__

    assert isinstance(tatva.DISPLAY_VERSION, str)
    assert tatva.DISPLAY_VERSION.strip()


@pytest.mark.unit
def test_version_has_exactly_one_source() -> None:
    """
    The GUI badge and the CLI --version must both come from tatva/__init__.py.

    Pinning the literal here would just recreate the drift this replaced: pyproject,
    __init__, gui.py and five spots in index.html each carried their own version
    string, and no two agreed.

    Two names, one source: __version__ is PEP 440 for packaging, DISPLAY_VERSION is
    what the badge shows. The badge must be the display name, and the full label must
    still contain the exact packaging version so a bug report identifies the build.
    """
    from tatva.gui import BUILD_LABEL, BUILD_VERSION

    assert BUILD_VERSION == tatva.DISPLAY_VERSION
    assert tatva.__version__ in BUILD_LABEL
    assert tatva.DISPLAY_VERSION in BUILD_LABEL
