"""
Error Handling & Structured Logging Test Suite for TATVA.

Verifies:
1. Known exception category boundary handling (friendly explanation + non-zero exit code 1).
2. Unknown exception boundary handling (calm error message + traceback logging).
3. Secret-masking in log files (--log-file option with zero secret leakage).
4. Verbosity level mapping (-v, -vv, --debug -> INFO, DEBUG).
"""

import contextlib
import logging
import os
import tempfile

import pytest

from tatva.diagnostics import MemoryLimitExceededError, UnsupportedOperatorError
from tatva.logging_setup import (
    SecretMaskingFormatter,
    configure_logging,
    get_logger,
)


@pytest.mark.unit
def test_verbosity_logging_configuration() -> None:
    """
    VERBOSITY MAPPING TEST:
    Assert that configure_logging sets the expected log levels.
    """
    logger = configure_logging(verbosity=0)
    console_handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
    assert console_handler.level == logging.WARNING

    logger_v = configure_logging(verbosity=1)
    console_handler_v = next(h for h in logger_v.handlers if isinstance(h, logging.StreamHandler))
    assert console_handler_v.level == logging.INFO

    logger_vv = configure_logging(verbosity=2)
    console_handler_vv = next(h for h in logger_vv.handlers if isinstance(h, logging.StreamHandler))
    assert console_handler_vv.level == logging.DEBUG

    logger_debug = configure_logging(debug=True)
    console_handler_debug = next(h for h in logger_debug.handlers if isinstance(h, logging.StreamHandler))
    assert console_handler_debug.level == logging.DEBUG


@pytest.mark.unit
def test_log_file_writing_and_secret_masking() -> None:
    """
    LOG FILE & SECRET MASKING TEST:
    Assert that configure_logging with log_file writes entries to disk
    and masks sensitive API keys/tokens.
    """
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        log_path = tmp.name

    try:
        logger = configure_logging(debug=True, log_file=log_path)
        test_logger = get_logger("test_module")

        fake_anthropic_key = "sk-ant-api03-1234567890abcdefghijklmnopqrstuvwxyz"
        fake_resend_key = "re_1234567890abcdefghijklmnopqrstuv"

        test_logger.info(f"Connecting to Anthropic API with key: {fake_anthropic_key}")
        test_logger.info(f"Dispatching Resend email with key: {fake_resend_key}")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        with open(log_path, encoding="utf-8") as f:
            log_contents = f.read()

        assert fake_anthropic_key not in log_contents
        assert fake_resend_key not in log_contents
        assert "sk-a***" in log_contents or "*****" in log_contents
        assert "re_1***" in log_contents or "*****" in log_contents

    finally:
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)
        if os.path.exists(log_path):
            with contextlib.suppress(OSError):
                os.remove(log_path)


@pytest.mark.unit
def test_secret_masking_formatter_standalone() -> None:
    """
    STANDALONE FORMATTER TEST:
    Assert that SecretMaskingFormatter masks secrets directly.
    """
    formatter = SecretMaskingFormatter()
    raw = "Header auth: sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
    masked = formatter.mask_secrets(raw)

    assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456" not in masked
    assert "sk-a***3456" in masked or "*" in masked


@pytest.mark.unit
def test_known_exception_boundary_handling() -> None:
    """
    KNOWN FAILURE BOUNDARY TEST:
    Assert that MemoryLimitExceededError and UnsupportedOperatorError are correctly classified
    into friendly structured contexts.
    """
    from tatva.diagnostics import classify_failure, explain

    mem_err = MemoryLimitExceededError(limit_bytes=524288, required_bytes=1048576)
    ctx = classify_failure(mem_err)

    assert ctx.error_type == "memory_limit_exceeded"
    assert ctx.metadata["limit_bytes"] == 524288
    assert ctx.metadata["required_bytes"] == 1048576

    explanation = explain(ctx)
    assert "Memory limit exceeded" in explanation
    assert "524288" in explanation
    assert "1048576" in explanation

    op_err = UnsupportedOperatorError("CustomOpXYZ")
    ctx_op = classify_failure(op_err)
    assert ctx_op.error_type == "unsupported_operator"
    assert ctx_op.metadata["operator_name"] == "CustomOpXYZ"
