"""
Structured Logging & Secret-Masking Setup Module for TATVA.

Provides:
- Verbosity mapping (-v, -vv, --debug) to standard Python logging levels.
- SecretMaskingFormatter to guarantee API keys and sensitive tokens are never written to logs.
- JsonFormatter for structured JSON log output.
- Log file redirection support via --log-file.
"""

import json
import logging
import re
import sys
from typing import Optional

from tatva.config import mask_secret

# Common regex patterns for secrets (Anthropic, Resend, Supabase JWT)
SECRET_PATTERNS = [
    re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),
    re.compile(r"re_[A-Za-z0-9]{20,}"),
    re.compile(r"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\.[A-Za-z0-9_-]+"),
]


class SecretMaskingFormatter(logging.Formatter):
    """
    Logging Formatter that intercepts and masks API keys/secrets in log output.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return self.mask_secrets(formatted)

    @staticmethod
    def mask_secrets(text: str) -> str:
        """
        Scan text for known secret patterns and replace them with masked representations.
        """
        if not text:
            return ""

        masked_text = text
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                raw_secret = match.group(0)
                masked_val = mask_secret(raw_secret)
                masked_text = masked_text.replace(raw_secret, masked_val)

        return masked_text


class JsonFormatter(logging.Formatter):
    """
    Logging Formatter that outputs structured JSON lines for automated log parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "message": SecretMaskingFormatter.mask_secrets(record.getMessage()),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a named logger under the 'tatva' hierarchy.
    """
    logger_name = f"tatva.{name}" if not name.startswith("tatva") else name
    return logging.getLogger(logger_name)


def configure_logging(
    verbosity: int = 0,
    debug: bool = False,
    log_file: Optional[str] = None,
    json_log: bool = False,
) -> logging.Logger:
    """
    Configure global Tatva root logger based on CLI/GUI options.

    Args:
        verbosity: Verbosity level (0: WARNING, 1: INFO, >=2: DEBUG)
        debug: If True, forces DEBUG level
        log_file: Optional path to write log output
        json_log: If True, uses JsonFormatter instead of standard text formatting
    """
    root_logger = logging.getLogger("tatva")
    root_logger.setLevel(logging.DEBUG)  # Capture all logs at root level

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Determine console log level
    if debug or verbosity >= 2:
        console_level = logging.DEBUG
    elif verbosity == 1:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    # Select Formatter
    fmt_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if json_log:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = SecretMaskingFormatter(fmt_str)

    # Console Handler (Stderr to separate logs from stdout CLI tabular outputs)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File handler captures full DEBUG output
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
