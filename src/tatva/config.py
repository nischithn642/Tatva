"""
Centralized Configuration & Secret Management Module for Tatva.

Enforces environment variable loading for API keys (Claude Anthropic, Resend, Supabase).
Ensures secrets are never hardcoded, logged, or leaked into error payloads.
"""

import os
from pathlib import Path

# One place for the Anthropic model id. It was previously written out three times --
# "claude-sonnet-4-6" in diagnostics and "claude-3-5-sonnet-20241022" twice in
# scaffolding -- and the first of those is not a model that exists, so every
# diagnostics call that actually had an API key silently 404'd into the offline path.
ANTHROPIC_MODEL = "claude-sonnet-5"

# Human-facing label for the same model, shown in the GUI's provider dropdown.
ANTHROPIC_MODEL_LABEL = "Claude Sonnet 5 (Anthropic)"


def load_dotenv_file(start: str | None = None) -> str | None:
    """
    Load a `.env` from the working directory or any parent, if one exists.

    SecretMissingError has always told people to "set it in your environment or .env
    file", and python-dotenv has always been a declared dependency -- but nothing ever
    read the file, so the second half of that sentence was a lie. Existing environment
    variables win; a .env never overrides what the shell already set.

    Returns the path loaded, or None.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    here = Path(start or os.getcwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return str(candidate)
    return None


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


def mask_secret(secret_val: str | None) -> str:
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
    default: str | None = None,
    required: bool = False,
    usage_context: str = "",
) -> str | None:
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


def get_anthropic_api_key(required: bool = False) -> str | None:
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


def get_resend_api_key(required: bool = False) -> str | None:
    """
    Retrieve Resend API Key for website contact notifications.
    """
    return get_secret(
        "RESEND_API_KEY",
        required=required,
        usage_context="Required for web contact form notifications",
    )


def get_supabase_key(required: bool = False) -> str | None:
    """
    Retrieve Supabase Service Role Key for backend storage operations.
    """
    return get_secret(
        "SUPABASE_SERVICE_ROLE_KEY",
        required=required,
        usage_context="Required for Supabase backend operations",
    )


def get_nvidia_api_key(required: bool = False) -> str | None:
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

