"""
Security & Data Egress Test Suite for Tatva (Security Hardening Phase).

Verifies strict whitelist payload sanitization, zero weight/path leakage to the Claude API,
and safe centralized secret management.
"""

import os
import numpy as np
import pytest

from tatva.config import (
    SecretMissingError,
    get_anthropic_api_key,
    get_resend_api_key,
    get_secret,
    get_supabase_key,
    mask_secret,
)
from tatva.diagnostics import whitelist_payload, _sanitize_string


@pytest.mark.unit
def test_diagnostics_payload_strict_whitelist() -> None:
    """
    EGRESS WHITELIST TEST:
    Assert that the diagnostics payload contains ONLY whitelisted metadata keys.
    Non-whitelisted keys (e.g. 'raw_weights', 'model_binary', 'api_secret') are stripped.
    """
    raw_metadata = {
        "limit_bytes": 524288,
        "required_bytes": 1048576,
        "details": "Compilation workspace allocation failed.",
        "raw_weights": [0.1, 0.2, 0.3],  # Non-whitelisted!
        "user_secret": "sk-ant-secret123",  # Non-whitelisted!
        "absolute_path": "C:\\Users\\Secret\\model.onnx",  # Non-whitelisted!
    }

    payload = whitelist_payload("memory_limit_exceeded", raw_metadata)

    # Assert ONLY whitelisted keys exist
    assert set(payload.keys()) == {"limit_bytes", "required_bytes", "details"}
    assert "raw_weights" not in payload
    assert "user_secret" not in payload
    assert "absolute_path" not in payload


@pytest.mark.unit
def test_diagnostics_payload_no_weight_leakage() -> None:
    """
    WEIGHT LEAKAGE TEST:
    Assert that numpy weight tensors or large float arrays are rejected from the egress payload.
    """
    fake_weights = np.random.rand(100, 100)
    large_list = list(range(500))

    raw_metadata = {
        "operator_name": "UnsupportedOpXYZ",
        "details": "Model layer convolution failed.",
        "weight_tensor": fake_weights,
        "bias_array": large_list,
    }

    payload = whitelist_payload("unsupported_operator", raw_metadata)

    assert payload["operator_name"] == "UnsupportedOpXYZ"
    assert "weight_tensor" not in payload
    assert "bias_array" not in payload


@pytest.mark.unit
def test_diagnostics_payload_path_sanitization() -> None:
    """
    PATH SANITIZATION TEST:
    Assert that absolute Windows and Unix host file paths are sanitized to prevent leaking host directory structures.
    """
    win_path = "C:\\Users\\Rahul\\AppData\\Local\\Temp\\model.onnx"
    unix_path = "/home/developer/secret_projects/tatva/models/model.onnx"

    sanitized_win = _sanitize_string(f"Failed loading {win_path}")
    sanitized_unix = _sanitize_string(f"Failed loading {unix_path}")

    assert "C:\\Users\\Rahul" not in sanitized_win
    assert "model.onnx" in sanitized_win

    assert "/home/developer/secret_projects" not in sanitized_unix
    assert "model.onnx" in sanitized_unix


@pytest.mark.unit
def test_centralized_secret_loader_missing_key() -> None:
    """
    SECRET LOADER MISSING KEY TEST:
    Assert that get_secret raises SecretMissingError when a required environment key is absent.
    """
    # Ensure key is unset for test
    os.environ.pop("MISSING_TEST_KEY_XYZ", None)

    with pytest.raises(SecretMissingError) as exc_info:
        get_secret("MISSING_TEST_KEY_XYZ", required=True, usage_context="Test validation")

    assert "MISSING_TEST_KEY_XYZ" in str(exc_info.value)
    assert "Test validation" in str(exc_info.value)


@pytest.mark.unit
def test_secret_loader_never_logs_plain_secret() -> None:
    """
    SECRET MASKING TEST:
    Assert that secret values are properly masked for debugging.
    """
    raw_key = "sk-ant-1234567890abcdef"
    masked = mask_secret(raw_key)

    assert masked == "sk-a***cdef"
    assert raw_key not in masked or len(masked) < len(raw_key)

    # Test short secret masking
    assert mask_secret("short") == "*****"
    assert mask_secret(None) == "<NOT_SET>"


@pytest.mark.unit
def test_secret_loader_environment_resolution(monkeypatch) -> None:
    """
    SECRET RESOLUTION TEST:
    Assert that get_anthropic_api_key, get_resend_api_key, and get_supabase_key load correctly.
    """
    monkeypatch.delenv("TATVA_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_anthropic_api_key() is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key_anthropic_std")
    assert get_anthropic_api_key() == "key_anthropic_std"

    monkeypatch.setenv("TATVA_ANTHROPIC_KEY", "key_tatva_override")
    assert get_anthropic_api_key() == "key_tatva_override"

    monkeypatch.setenv("RESEND_API_KEY", "re_123456789")
    assert get_resend_api_key() == "re_123456789"

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_123456789")
    assert get_supabase_key() == "sb_123456789"


@pytest.mark.unit
def test_fake_secret_pattern_detection() -> None:
    """
    SECRET SCANNER PATTERN TEST:
    Assert that regex pattern detectors for secrets (such as Anthropic, Resend, or Supabase JWT tokens)
    flag planted example fake keys.
    """
    import re

    # Standard secret patterns
    anthropic_pattern = re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{32,}")
    resend_pattern = re.compile(r"re_[A-Za-z0-9]{24,}")
    supabase_pattern = re.compile(r"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\.[A-Za-z0-9_-]+")

    fake_anthropic_secret = "sk-ant-api03-1234567890abcdefghijklmnopqrstuvwxyz123456"
    fake_resend_secret = "re_1234567890abcdefghijklmnopqrstuv"
    fake_supabase_secret = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSJ9"

    assert anthropic_pattern.search(fake_anthropic_secret) is not None
    assert resend_pattern.search(fake_resend_secret) is not None
    assert supabase_pattern.search(fake_supabase_secret) is not None

