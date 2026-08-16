"""
TATVA Diagnostics Module.

This module handles exception categorization, Claude API diagnostic generation,
and local rule-based fallback diagnostics when offline.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

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


class EmulatorImageLimitError(MemoryLimitExceededError):
    """
    Raised when the linked image is too large for the bundled QEMU `virt` board to load.

    This is a limit of the emulator, not of RISC-V and not of the model: the ELF is
    valid and would deploy to real hardware unchanged. QEMU pins the flattened device
    tree at a fixed address, and an image that reaches it is refused at load with
    "Some ROM regions are overlapping" -- a message that names the FDT and never
    mentions size. Subclasses MemoryLimitExceededError so existing handlers still
    catch it, while carrying enough context to say which limit was actually hit.
    """
    def __init__(self, limit_bytes: int, required_bytes: int, fdt_address: int, details: str = "") -> None:
        super().__init__(limit_bytes=limit_bytes, required_bytes=required_bytes, details=details)
        self.fdt_address = fdt_address


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
    """
    Raised when cross-compilation, linking, or TVM lowering fails.

    This is the only CompilationError in TATVA. The runner used to define a second,
    unrelated one, so every compiler failure it raised fell through classify_failure()
    into the 'unknown' bucket and lost its stage and command.
    """
    def __init__(self, stage: str = "compilation", command: str = "", stderr: str = "", details: str = "") -> None:
        msg = f"Compilation failed during {stage} stage."
        if details:
            msg += f" {details}"
        if stderr:
            tail = stderr.strip().splitlines()[-15:]
            if tail:
                msg += "\n" + "\n".join(tail)
        super().__init__(msg)
        self.stage = stage
        self.command = command
        self.stderr = stderr
        self.details = details


class SimulationTimeoutError(Exception):
    """
    Raised when the QEMU run outlives its wall-clock ceiling.

    A timeout is not a compilation failure and must not be reported as one. Left as a
    bare RuntimeError it reached the user as "Unexpected Compilation Failure" followed
    by the whole QEMU command line, which says nothing about the fact that the model
    was still computing when the clock ran out. The usual cause is a soft-float target:
    an FP32 transformer on a core with no F/D extension runs every multiply through a
    library call.
    """
    def __init__(
        self,
        timeout_seconds: int,
        target: str,
        run_count: int,
        soft_float: bool = False,
        details: str = "",
    ) -> None:
        super().__init__(
            f"Simulation timed out: {target} did not finish {run_count} inference(s) "
            f"within {timeout_seconds} seconds."
        )
        self.timeout_seconds = timeout_seconds
        self.target = target
        self.run_count = run_count
        self.soft_float = soft_float
        self.details = details


class ImportInProgressError(Exception):
    """Raised when trying to import a model format whose importer is still in progress."""
    pass


@dataclass
class DiagnosisContext:
    """Carries structured metadata context for generating user-facing diagnoses."""
    error_type: str
    metadata: dict[str, Any]


def classify_failure(exception_or_result: Any) -> DiagnosisContext:
    """
    Classify a caught exception or build failure result into a structured DiagnosisContext.
    """
    # Checked before MemoryLimitExceededError, which it subclasses -- the base class
    # would otherwise swallow it and report the emulator's limit as a target one.
    if isinstance(exception_or_result, EmulatorImageLimitError):
        return DiagnosisContext(
            error_type="emulator_image_limit",
            metadata={
                "limit_bytes": exception_or_result.limit_bytes,
                "required_bytes": exception_or_result.required_bytes,
                "fdt_address": exception_or_result.fdt_address,
                "details": exception_or_result.details,
            },
        )
    elif isinstance(exception_or_result, MemoryLimitExceededError):
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
    elif isinstance(exception_or_result, SimulationTimeoutError):
        return DiagnosisContext(
            error_type="simulation_timeout",
            metadata={
                "timeout_seconds": exception_or_result.timeout_seconds,
                "target": exception_or_result.target,
                "run_count": exception_or_result.run_count,
                "soft_float": exception_or_result.soft_float,
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
        # Generic failures reach us as text only. Recover whatever numbers the message
        # actually contains and leave the rest absent -- this branch used to invent
        # plausible-looking values (a 512 KB limit, an MSE of 0.210123), which then
        # appeared in the user-facing diagnosis as if they had been measured.
        msg = str(exception_or_result)
        if "Memory limit exceeded" in msg:
            nums = [int(n) for n in re.findall(r"(\d+) bytes", msg)]
            metadata: dict[str, Any] = {"details": msg}
            if len(nums) >= 2:
                metadata["limit_bytes"], metadata["required_bytes"] = nums[0], nums[1]
            return DiagnosisContext(error_type="memory_limit_exceeded", metadata=metadata)
        elif "degradation exceeds tolerance" in msg or "AccuracyDropError" in msg:
            match = re.search(r"\(([0-9.eE+-]+)\s*>\s*([0-9.eE+-]+)\)", msg)
            metadata = {"details": msg}
            if match:
                metadata["mse"] = float(match.group(1))
                metadata["tolerance"] = float(match.group(2))
            return DiagnosisContext(error_type="accuracy_drop", metadata=metadata)
        elif "Unsupported operator" in msg or "UnsupportedOp" in msg or "not supported" in msg:
            match = re.search(r"[Uu]nsupported operator:?\s*'([^']+)'", msg) or re.search(r"UnsupportedOp\w*", msg)
            metadata = {"details": msg}
            if match:
                metadata["operator_name"] = match.group(1) if match.groups() else match.group(0)
            return DiagnosisContext(error_type="unsupported_operator", metadata=metadata)
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


def _mib_str(n: int) -> str:
    """Format a byte count as MiB for a user-facing diagnosis."""
    return f"{n / (1024 * 1024):.1f} MiB"


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


def whitelist_payload(error_type: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Enforce security gating. Filter metadata to ensure only whitelisted fields
    ever leave the local machine to the external Claude API.
    Rejects tensor arrays, raw model weights, and scrubs absolute host file paths.
    """
    whitelisted = {}
    if error_type == "memory_limit_exceeded":
        allowed_keys = {"limit_bytes", "required_bytes", "details"}
    elif error_type == "emulator_image_limit":
        allowed_keys = {"limit_bytes", "required_bytes", "fdt_address", "details"}
    elif error_type == "accuracy_drop":
        allowed_keys = {"mse", "tolerance", "details"}
    elif error_type == "unsupported_operator":
        allowed_keys = {"operator_name", "details"}
    elif error_type == "compilation_error":
        allowed_keys = {"stage", "command", "details"}
    elif error_type == "simulation_timeout":
        allowed_keys = {"timeout_seconds", "target", "run_count", "soft_float", "details"}
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
            # Allow small primitive lists only (e.g., shape bounds under 16 items)
            elif (
                isinstance(val, (list, tuple))
                and len(val) <= 16
                and all(isinstance(x, (int, float, str, bool)) for x in val)
            ):
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
        limit = meta.get("limit_bytes")
        req = meta.get("required_bytes")
        if limit is None or req is None:
            headline = (
                "Memory limit exceeded: the compiled model's workspace footprint does not fit the "
                "target's bare-metal memory budget. (The exact sizes were not reported by the "
                "failing stage.)"
            )
        else:
            headline = (
                f"Memory limit exceeded: The compiled model workspace footprint requires {req} bytes, "
                f"which exceeds the configured target bare-metal RISC-V memory limit of {limit} bytes."
            )
        detail = meta.get("details") or ""
        return (
            f"{headline}\n"
            + (f"{detail}\n" if detail else "")
            + "Mitigation:\n"
            "1. Shorten the sequence axis. TATVA binds a symbolic sequence dimension to 32; "
            "the activation pool scales with it, though the weights do not.\n"
            "2. Quantize the weights, which is the dominant term for a transformer of this size "
            "(`--passes quantize`). Note that this build's INT8 pass is fake-quantization: it "
            "shrinks nothing on its own and is measured slower than FP32.\n"
            "3. Fit the model to the target. A model whose weights alone exceed the board's RAM "
            "cannot be linked flat, and TATVA has no weight-streaming backend."
        )
    elif t == "emulator_image_limit":
        limit = meta.get("limit_bytes")
        req = meta.get("required_bytes")
        fdt = meta.get("fdt_address")
        if limit is None or req is None:
            headline = (
                "Model too large to simulate: the linked image does not fit the address range "
                "the bundled QEMU board leaves free. (The exact sizes were not reported by the "
                "failing stage.)"
            )
        else:
            headline = (
                f"Model too large to simulate: the linked image spans {_mib_str(req)}, but the bundled "
                f"QEMU `virt` board leaves only {_mib_str(limit)} free below its device tree"
                + (f" at {fdt:#x}" if isinstance(fdt, int) else "")
                + "."
            )
        detail = meta.get("details") or ""
        return (
            f"{headline}\n"
            + (f"{detail}\n" if detail else "")
            + "This is a limit of the emulator, not of RISC-V and not of your model. The ELF "
            "that was just built is valid and would deploy to real hardware unchanged; only "
            "the simulated run is refused.\n"
            "Mitigation:\n"
            "1. Use a smaller model. The weight blob is the dominant term, and it is linked "
            "flat -- TATVA has no weight-streaming backend.\n"
            "2. Deploy the built ELF to a physical RISC-V board with enough RAM. Nothing about "
            "the image is wrong.\n"
            "3. Do not raise QEMU's `-m`. The device tree address is fixed; it was measured at "
            "the same address for every memory size from 1100M to 8192M, so more memory moves "
            "nothing."
        )
    elif t == "simulation_timeout":
        secs = meta.get("timeout_seconds")
        target = meta.get("target") or "the selected target"
        runs = meta.get("run_count")
        soft = meta.get("soft_float")
        if secs is None:
            headline = f"Simulation timed out on {target} before the model finished."
        else:
            headline = (
                f"Simulation timed out: {target} did not finish "
                + (f"{runs} inference(s) " if runs else "")
                + f"within {secs} seconds. The model was still computing; nothing crashed."
            )
        detail = meta.get("details") or ""
        soft_note = (
            f"{target} has no hardware floating-point unit (no F/D extension), so every FP32 "
            "multiply in the model runs through a soft-float library call. On a transformer "
            "that is the difference between finishing and not.\n"
            if soft
            else ""
        )
        return (
            f"{headline}\n"
            + (f"{detail}\n" if detail else "")
            + soft_note
            + "Mitigation:\n"
            + (
                "1. Select a target with hardware floating point -- RV64GC or RV64IMAFDC. "
                "This is usually the whole fix for an FP32 model.\n"
                if soft
                else "1. Re-run on an idle machine; the ceiling is wall clock, so a busy host counts against it.\n"
            )
            + "2. Reduce the inference count (fewer warm-up or timed runs); the ceiling scales with it.\n"
            "3. Use a smaller model. The ceiling also scales with the linked image size."
        )
    elif t == "accuracy_drop":
        mse = meta.get("mse")
        tol = meta.get("tolerance", 0.05)
        if mse is None:
            headline = (
                "Accuracy degradation check failed: the optimized model's outputs diverge from the "
                f"host reference by more than the allowed tolerance of {tol}. (The measured MSE was "
                "not reported by the failing stage.)"
            )
        else:
            headline = (
                f"Accuracy degradation check failed: The optimized model has a Mean Squared Error (MSE) "
                f"of {mse:.6f} compared to the host reference outputs, exceeding the allowed tolerance "
                f"threshold of {tol}."
            )
        return (
            f"{headline}\n"
            f"Mitigation:\n"
            f"1. Re-check the calibrated activation scale (metadata['activation_scale_source']) "
            f"and the per-tensor weight scales; a clipped scale shows up only in the output.\n"
            f"2. Consider executing passes selectively (e.g. bypassing quantization on sensitive attention subgraphs).\n"
            f"3. Verify input range configurations to ensure valid logits mappings."
        )
    elif t == "unsupported_operator":
        op = meta.get("operator_name") or "the reported operator"
        return (
            f"Unsupported operator: The operator '{op}' is not supported by the RISC-V TVM bare-metal backend.\n"
            f"Mitigation:\n"
            f"1. Implement or register a legalization pass in compiler.py to map '{op}' to supported primitives.\n"
            f"2. Adjust ONNX exporter configurations to decompose this operator during model export.\n"
            f"3. Implement a custom low-level fallback kernel in compiler.py/runner.py."
        )
    elif t == "compilation_error":
        stage = meta.get("stage") or "compilation"
        cmd = meta.get("command") or "(command not recorded)"
        detail = meta.get("details") or ""
        if stage == "import":
            # An import-stage failure never reached a compiler, so the cross-compilation
            # advice below is not merely unhelpful, it is misleading: it sends the user
            # to check gcc flags and link.ld for a model that TVM refused to read.
            return (
                "Import failure: TVM's ONNX frontend could not convert this model, so nothing "
                "was compiled.\n"
                + (f"{detail}\n" if detail else "")
                + f"Command executed: '{cmd}'\n"
                "Mitigation:\n"
                "1. Check which operator the error names. The frontend converts operator by "
                "operator, so the failure is almost always one node rather than the whole graph.\n"
                "2. Re-export the model with a different opset version, or with the exporter's "
                "operator-decomposition option enabled, so that node is expressed differently.\n"
                "3. If the operator is one TATVA can rewrite, run `tatva repair` -- but note that "
                "repair works on the imported graph, so an operator that fails here never gets "
                "that far and has to be changed at export time."
            )
        return (
            f"Compilation failure: The cross-compilation build stage failed during the '{stage}' step.\n"
            + (f"{detail}\n" if detail else "")
            + f"Command executed: '{cmd}'\n"
            f"Mitigation:\n"
            f"1. Verify target RISC-V cross-compiler options and flags.\n"
            f"2. Ensure TVM libraries are compiled and headers are in the search path.\n"
            f"3. Read the linker output above. TATVA sizes link.ld's RAM region to the build, "
            f"so a region overflow is reported separately as a memory-limit failure."
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

    from tatva.config import ANTHROPIC_MODEL

    data = {
        "model": ANTHROPIC_MODEL,
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
