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
    assert re.fullmatch(r"\d+\.\d+\.\d+", tatva.__version__), tatva.__version__


@pytest.mark.unit
def test_version_has_exactly_one_source() -> None:
    """
    The GUI badge and the CLI --version must both come from tatva.__version__.

    Pinning the literal here would just recreate the drift this replaced: pyproject,
    __init__, gui.py and five spots in index.html each carried their own version
    string, and no two agreed.
    """
    from tatva.gui import BUILD_VERSION

    assert f"v{tatva.__version__}" == BUILD_VERSION
