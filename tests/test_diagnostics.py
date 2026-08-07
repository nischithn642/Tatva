"""
Tests for tatva diagnostics taxonomy, classification, security, and explanation layers.
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tatva.diagnostics import (
    AccuracyDropError,
    CompilationError,
    MemoryLimitExceededError,
    UnsupportedOperatorError,
    classify_failure,
    explain,
    whitelist_payload,
)


@pytest.mark.unit
def test_classify_failure_scenarios() -> None:
    """
    Assert that caught exceptions map cleanly to structured context categories and metadata.
    """
    # 1. Memory Limit Exceeded
    exc_mem = MemoryLimitExceededError(limit_bytes=512, required_bytes=1024, details="Low RAM")
    ctx_mem = classify_failure(exc_mem)
    assert ctx_mem.error_type == "memory_limit_exceeded"
    assert ctx_mem.metadata["limit_bytes"] == 512
    assert ctx_mem.metadata["required_bytes"] == 1024

    # 2. Accuracy Drop
    exc_acc = AccuracyDropError(mse=0.15, tolerance=0.05, details="Regressed")
    ctx_acc = classify_failure(exc_acc)
    assert ctx_acc.error_type == "accuracy_drop"
    assert ctx_acc.metadata["mse"] == 0.15
    assert ctx_acc.metadata["tolerance"] == 0.05

    # 3. Unsupported Operator
    exc_op = UnsupportedOperatorError(operator_name="CustomOp", details="Unknown node")
    ctx_op = classify_failure(exc_op)
    assert ctx_op.error_type == "unsupported_operator"
    assert ctx_op.metadata["operator_name"] == "CustomOp"

    # 4. Compilation Error
    exc_comp = CompilationError(stage="linking", command="gcc", stderr="undefined reference")
    ctx_comp = classify_failure(exc_comp)
    assert ctx_comp.error_type == "compilation_error"
    assert ctx_comp.metadata["stage"] == "linking"


@pytest.mark.unit
def test_offline_explanations() -> None:
    """
    Assert that the offline diagnostics rules engine maps categorized errors
    to clear, deterministic plain-English mitigation descriptions.
    """
    # Memory limit offline message
    ctx_mem = classify_failure(MemoryLimitExceededError(limit_bytes=512, required_bytes=1024))
    exp_mem = explain(ctx_mem)
    assert "Memory limit exceeded" in exp_mem
    assert "1024" in exp_mem
    assert "512" in exp_mem

    # Accuracy drop offline message
    ctx_acc = classify_failure(AccuracyDropError(mse=0.15, tolerance=0.05))
    exp_acc = explain(ctx_acc)
    assert "Accuracy degradation check failed" in exp_acc
    assert "0.15" in exp_acc

    # Unsupported operator offline message
    ctx_op = classify_failure(UnsupportedOperatorError(operator_name="CustomOp"))
    exp_op = explain(ctx_op)
    assert "Unsupported operator" in exp_op
    assert "CustomOp" in exp_op


@pytest.mark.unit
def test_security_whitelist_and_no_weights() -> None:
    """
    Assert that sensitive/large payload elements (like secrets or numpy weight arrays)
    are filtered out, ensuring only safe, whitelisted metadata leaves the machine.
    """
    bad_metadata = {
        "operator_name": "CustomOp",
        "details": "Details",
        "weights": np.random.rand(10, 10),
        "secret_key": "mysecret",
    }

    clean = whitelist_payload("unsupported_operator", bad_metadata)

    assert "operator_name" in clean
    assert "details" in clean
    assert "weights" not in clean
    assert "secret_key" not in clean

    json_payload = json.dumps(clean)
    assert "weights" not in json_payload
    assert "secret" not in json_payload


@pytest.mark.unit
@patch("urllib.request.urlopen")
def test_claude_api_online_and_fallback(mock_urlopen: MagicMock) -> None:
    """
    Assert that when an API key is configured, explain queries the Claude API
    with whitelisted metadata, using the configured model, and falls back
    to offline explanations gracefully upon timeouts or connectivity failures.
    """
    from tatva.config import ANTHROPIC_MODEL

    # 1. API Success Path
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "content": [{"text": "API Explanation: Memory limit was exceeded."}]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-xyz"}):
        ctx_mem = classify_failure(MemoryLimitExceededError(limit_bytes=512, required_bytes=1024))
        exp = explain(ctx_mem)
        assert exp == "API Explanation: Memory limit was exceeded."

        assert mock_urlopen.call_count == 1
        args, _ = mock_urlopen.call_args
        req = args[0]

        # Verify exact headers required
        assert req.headers["X-api-key"] == "test-key-xyz"
        assert req.headers["Anthropic-version"] == "2023-06-01"

        # Verify payload and requested model identifier
        req_data = json.loads(req.data.decode("utf-8"))
        assert req_data["model"] == ANTHROPIC_MODEL
        assert "limit_bytes" in req_data["messages"][0]["content"]

    # 2. API Failure Path (graceful fallback offline)
    mock_urlopen.side_effect = Exception("Connection Timeout")
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-xyz"}):
        ctx_acc = classify_failure(AccuracyDropError(mse=0.15, tolerance=0.05))
        exp_fallback = explain(ctx_acc)
        assert "Accuracy degradation check failed" in exp_fallback
        assert "0.15" in exp_fallback
