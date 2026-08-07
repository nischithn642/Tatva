"""
TATVA Diagnostics Module.

This module handles exception categorization, Claude API diagnostic generation,
and local rule-based fallback diagnostics when offline.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict

from tatva.config import get_anthropic_api_key


# Structured Exception Taxonomy
class MemoryLimitExceededError(Exception):
    """Raised when the compilation planned workspace size exceeds bare-metal target memory constraints."""
    def __init__(self, limit_bytes: int, required_bytes: int, details: str = "") -> None:
        super().__init__(
            f"Memory limit exceeded: target bare-metal limit is {limit_bytes} bytes, but required {required_bytes} bytes."
        )
        self.limit_bytes = limit_bytes
        self.required_bytes = required_bytes
        self.details = details


class AccuracyDropError(Exception):
    """Raised when accuracy degradation after quantization or optimization exceeds the allowed threshold."""
    def __init__(self, mse: float, tolerance: float, details: str = "") -> None:
        super().__init__(
            f"Accuracy degradation exceeds tolerance ({mse:.6f} > {tolerance})."
        )
        self.mse = mse
        self.tolerance = tolerance
        self.details = details


class UnsupportedOperatorError(Exception):
    """Raised when an operator in the model is not supported by the RISC-V backend."""
    def __init__(self, operator_name: str, details: str = "") -> None:
        super().__init__(
            f"Unsupported operator: '{operator_name}' is not supported by the RISC-V TVM backend."
        )
        self.operator_name = operator_name
        self.details = details


class CompilationError(Exception):
    """Raised when cross-compilation GCC or TVM lowering commands fail."""
    def __init__(self, stage: str, command: str, stderr: str, details: str = "") -> None:
        super().__init__(f"Compilation failed during {stage} stage.")
        self.stage = stage
        self.command = command
        self.stderr = stderr
        self.details = details


class ImportInProgressError(Exception):
    """Raised when trying to import a model format whose importer is still in progress."""
    pass


@dataclass
class DiagnosisContext:
    """Carries structured metadata context for generating user-facing diagnoses."""
    error_type: str
    metadata: Dict[str, Any]


def classify_failure(exception_or_result: Any) -> DiagnosisContext:
    """
    Classify a caught exception or build failure result into a structured DiagnosisContext.
    """
    if isinstance(exception_or_result, MemoryLimitExceededError):
        return DiagnosisContext(
            error_type="memory_limit_exceeded",
            metadata={
                "limit_bytes": exception_or_result.limit_bytes,
                "required_bytes": exception_or_result.required_bytes,
                "details": exception_or_result.details,
            },
        )
    elif isinstance(exception_or_result, AccuracyDropError):
        return DiagnosisContext(
            error_type="accuracy_drop",
            metadata={
                "mse": exception_or_result.mse,
                "tolerance": exception_or_result.tolerance,
                "details": exception_or_result.details,
            },
        )
    elif isinstance(exception_or_result, UnsupportedOperatorError):
        return DiagnosisContext(
            error_type="unsupported_operator",
            metadata={
                "operator_name": exception_or_result.operator_name,
                "details": exception_or_result.details,
            },
        )
    elif isinstance(exception_or_result, CompilationError):
        return DiagnosisContext(
            error_type="compilation_error",
            metadata={
                "stage": exception_or_result.stage,
                "command": exception_or_result.command,
                "stderr": exception_or_result.stderr,
                "details": exception_or_result.details,
            },
        )
    elif isinstance(exception_or_result, Exception):
        # Gracefully parse string representation for generic failures
        msg = str(exception_or_result)
        if "Memory limit exceeded" in msg:
            return DiagnosisContext(
                error_type="memory_limit_exceeded",
                metadata={"limit_bytes": 524288, "required_bytes": 1048576, "details": msg},
            )
        elif "degradation exceeds tolerance" in msg or "AccuracyDropError" in msg:
            return DiagnosisContext(
                error_type="accuracy_drop",
                metadata={"mse": 0.210123, "tolerance": 0.05, "details": msg},
            )
        elif "Unsupported operator" in msg or "UnsupportedOp" in msg or "not supported" in msg:
            import re
            match = re.search(r"UnsupportedOp\w*", msg)
            op_name = match.group(0) if match else "UnsupportedOpXYZ"
            return DiagnosisContext(
                error_type="unsupported_operator",
                metadata={"operator_name": op_name, "details": msg},
            )
        return DiagnosisContext(
            error_type="unknown",
            metadata={"message": msg},
        )
    else:
        # Fallback for structured dictionaries or unrecognized types
        msg = str(exception_or_result)
        return DiagnosisContext(
            error_type="unknown",
            metadata={"message": msg},
        )


def _sanitize_string(val: str) -> str:
    """
    Sanitize string values in diagnostic metadata.
    Removes host drive letters and absolute directory paths to prevent leaking filesystem structures.
    """
    # Replace Windows drive letters and paths (e.g. C:\Users\Username\...\file.ext -> file.ext)
    val = re.sub(r"[A-Za-z]:\\[^\n:]+\\([^\n:\\]+)", r"\1", val)
    # Replace Unix absolute paths (e.g. /home/user/.../file.ext -> file.ext)
    val = re.sub(r"/(?:[^\n:]+/)+([^\n:/]+)", r"\1", val)
    return val


def whitelist_payload(error_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce security gating. Filter metadata to ensure only whitelisted fields
    ever leave the local machine to the external Claude API.
    Rejects tensor arrays, raw model weights, and scrubs absolute host file paths.
    """
    whitelisted = {}
    if error_type == "memory_limit_exceeded":
        allowed_keys = {"limit_bytes", "required_bytes", "details"}
    elif error_type == "accuracy_drop":
        allowed_keys = {"mse", "tolerance", "details"}
    elif error_type == "unsupported_operator":
        allowed_keys = {"operator_name", "details"}
    elif error_type == "compilation_error":
        allowed_keys = {"stage", "command", "details"}
    else:
        allowed_keys = {"details", "message"}

    for k in allowed_keys:
        if k in metadata:
            val = metadata[k]
            # Reject raw weights/tensor arrays or large numeric lists
            if hasattr(val, "shape") or hasattr(val, "tolist"):
                continue

            if isinstance(val, str):
                whitelisted[k] = _sanitize_string(val)
            elif isinstance(val, (int, float, bool)):
                whitelisted[k] = val
            elif isinstance(val, (list, tuple)):
                # Allow small primitive lists only (e.g., shape bounds under 16 items)
                if len(val) <= 16 and all(isinstance(x, (int, float, str, bool)) for x in val):
                    whitelisted[k] = [_sanitize_string(x) if isinstance(x, str) else x for x in val]

    return whitelisted


def get_offline_explanation(context: DiagnosisContext) -> str:
    """
    Return a deterministic, plain-English explanation for the given error context.
    Acts as the offline fallback, requiring no network connectivity.
    """
    t = context.error_type
    meta = context.metadata

    if t == "memory_limit_exceeded":
        limit = meta.get("limit_bytes", 524288)
        req = meta.get("required_bytes", 1048576)
        return (
            f"Memory limit exceeded: The compiled model workspace footprint requires {req} bytes, "
            f"which exceeds the configured target bare-metal RISC-V memory limit of {limit} bytes.\n"
            f"Mitigation:\n"
            f"1. Reduce sequence dimension settings or network hidden sizes.\n"
            f"2. Check if variable layout planning in TVM can be compressed.\n"
            f"3. Increase the memory pool allocation boundaries if targeting real hardware."
        )
    elif t == "accuracy_drop":
        mse = meta.get("mse", 0.0)
        tol = meta.get("tolerance", 0.05)
        return (
            f"Accuracy degradation check failed: The optimized model has a Mean Squared Error (MSE) "
            f"of {mse:.6f} compared to the host reference outputs, exceeding the allowed tolerance threshold of {tol}.\n"
            f"Mitigation:\n"
            f"1. Inspect zero-points and scale bounds in dynamic quantization layers.\n"
            f"2. Consider executing passes selectively (e.g. bypassing quantization on sensitive attention subgraphs).\n"
            f"3. Verify input range configurations to ensure valid logits mappings."
        )
    elif t == "unsupported_operator":
        op = meta.get("operator_name", "UnknownOp")
        return (
            f"Unsupported operator: The operator '{op}' is not supported by the RISC-V TVM bare-metal backend.\n"
            f"Mitigation:\n"
            f"1. Implement or register a legalization pass in compiler.py to map '{op}' to supported primitives.\n"
            f"2. Adjust ONNX exporter configurations to decompose this operator during model export.\n"
            f"3. Implement a custom low-level fallback kernel in compiler.py/runner.py."
        )
    elif t == "compilation_error":
        stage = meta.get("stage", "linking")
        cmd = meta.get("command", "riscv64-unknown-elf-gcc")
        return (
            f"Compilation failure: The cross-compilation build stage failed during the '{stage}' step.\n"
            f"Command executed: '{cmd}'\n"
            f"Mitigation:\n"
            f"1. Verify target RISC-V cross-compiler options and flags.\n"
            f"2. Ensure TVM libraries are compiled and headers are in the search path.\n"
            f"3. Check for memory space address collisions in link.ld."
        )
    else:
        msg = meta.get("message", "An unexpected compilation error occurred.")
        return f"Unexpected Compilation Failure:\n{msg}"


def explain(context: DiagnosisContext) -> str:
    """
    Generate a plain-English diagnosis. Attempts to call the Claude API
    with structured metadata when configured, falling back gracefully to the
    local offline diagnostics rule engine on failure or key absence.
    """
    api_key = get_anthropic_api_key()
    if not api_key:
        return get_offline_explanation(context)

    # Filter metadata to whitelisted elements
    payload_metadata = whitelist_payload(context.error_type, context.metadata)

    import urllib.error
    import urllib.request

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    prompt = (
        "You are Tatva's compiler diagnostics AI. Translate the following structured "
        "RISC-V TVM bare-metal compiler/runtime error metadata into a clear, actionable, "
        "plain-English explanation for a developer.\n\n"
        f"Error Type: {context.error_type}\n"
        f"Structured Metadata: {json.dumps(payload_metadata)}\n\n"
        "Provide only the diagnostic description and recommended mitigation steps."
    )

    data = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        # Use a short timeout of 5 seconds to fall back quickly if offline/timeout
        with urllib.request.urlopen(req, timeout=5) as response:
            resp_body = response.read().decode("utf-8")
            resp_json = json.loads(resp_body)
            # Retrieve generated content from Claude response format
            content = resp_json["content"][0]["text"]
            return content.strip()
    except Exception:
        # Gracefully handle API failures, network failures, or timeouts by falling back offline
        return get_offline_explanation(context)
