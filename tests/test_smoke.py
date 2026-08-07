"""
Smoke test for tatva package.
"""

import pytest
import tatva


@pytest.mark.unit
def test_smoke() -> None:
    """
    Assert tatva can be imported and has version metadata.
    """
    assert hasattr(tatva, "__version__")
    assert isinstance(tatva.__version__, str)
    assert tatva.__version__ == "1.2.0-gemini"
