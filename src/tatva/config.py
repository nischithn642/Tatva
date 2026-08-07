"""
Centralized Configuration & Secret Management Module for Tatva.

Enforces environment variable loading for API keys (Claude Anthropic, Resend, Supabase).
Ensures secrets are never hardcoded, logged, or leaked into error payloads.
"""

import os
from typing import Optional


class SecretMissingError(Exception):
    """
    Raised when a required secret or API key environment variable is not set.
    """

    def __init__(self, key_name: str, usage_context: str = "") -> None:
        msg = (
            f"Missing required environment variable '{key_name}'."
            + (f" Context: {usage_context}." if usage_context else "")
            + f" Please set '{key_name}' in your environment or .env file."
        )
        super().__init__(msg)
        self.key_name = key_name


def mask_secret(secret_val: Optional[str]) -> str:
    """
    Safely mask a secret string for debugging without revealing its full contents.
    Example: 'sk-ant-1234567890abcdef' -> 'sk-a***cdef'
    """
    if not secret_val:
        return "<NOT_SET>"
    val = secret_val.strip()
    if len(val) <= 8:
        return "*" * len(val)
    return f"{val[:4]}***{val[-4:]}"


def get_secret(
    key_name: str,
    default: Optional[str] = None,
    required: bool = False,
    usage_context: str = "",
) -> Optional[str]:
    """
    Retrieve a secret environment variable cleanly.

    Args:
        key_name: The name of the environment variable.
        default: Fallback value if not set and not required.
        required: If True, raises SecretMissingError if the variable is missing or empty.
        usage_context: Description of why the key is required.

    Returns:
        The stripped secret string, or default.
    """
    val = os.environ.get(key_name)
    if val is not None:
        val = val.strip()

    if not val:
        if required:
            raise SecretMissingError(key_name, usage_context)
        return default

    return val


def get_anthropic_api_key(required: bool = False) -> Optional[str]:
    """
    Retrieve Anthropic Claude API Key from TATVA_ANTHROPIC_KEY or ANTHROPIC_API_KEY.
    """
    key = get_secret("TATVA_ANTHROPIC_KEY") or get_secret("ANTHROPIC_API_KEY")
    if not key and required:
        raise SecretMissingError(
            "TATVA_ANTHROPIC_KEY",
            "Required for Claude automated failure diagnostics API calls",
        )
    return key


def get_resend_api_key(required: bool = False) -> Optional[str]:
    """
    Retrieve Resend API Key for website contact notifications.
    """
    return get_secret(
        "RESEND_API_KEY",
        required=required,
        usage_context="Required for web contact form notifications",
    )


def get_supabase_key(required: bool = False) -> Optional[str]:
    """
    Retrieve Supabase Service Role Key for backend storage operations.
    """
    return get_secret(
        "SUPABASE_SERVICE_ROLE_KEY",
        required=required,
        usage_context="Required for Supabase backend operations",
    )


def get_nvidia_api_key(required: bool = False) -> Optional[str]:
    """
    Retrieve NVIDIA API Key from TATVA_NVIDIA_KEY or NVIDIA_API_KEY.
    """
    key = get_secret("TATVA_NVIDIA_KEY") or get_secret("NVIDIA_API_KEY")
    if not key and required:
        raise SecretMissingError(
            "NVIDIA_API_KEY",
            "Required for live NVIDIA build.nvidia.com model catalog integration",
        )
    return key

