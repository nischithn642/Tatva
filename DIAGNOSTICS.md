# TATVA Diagnostics Engine Documentation (Milestone M3–M6)

This document details the structured exception taxonomy, security whitelist filters, online Claude API prompt design, and offline local rule-based fallback engines.

---

## 1. Security-First Structured Payload Design

To prevent sensitive intellectual property, proprietary model weights, or private local filesystem paths from leaking, the diagnostics engine enforces **strict metadata whitelist gating** in `tatva.diagnostics.whitelist_payload`. Only primitive, non-sensitive data types (integers, floats, flat primitive lists &le; 16 items, and sanitized basenames) are serialized into the outbound request body.

Outbound API Request Format (`https://api.anthropic.com/v1/messages`):
```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "User-facing prompt containing whitelisted metadata"
    }
  ]
}
```

---

## 2. Exception Taxonomy & Failure Scenarios

The diagnostics engine manages four primary compiler failure categories:

### Scenario 1: Memory Limit Exceeded (`MemoryLimitExceededError`)
Raised when the compiler's planned memory allocation offsets or dynamic tensor workspace requirements exceed target bare-metal memory limits.

#### Whitelisted Metadata Payload
```json
{
  "limit_bytes": 524288,
  "required_bytes": 1048576,
  "details": "Model workspace allocation check failed."
}
```

#### Local Offline Fallback Explanation
```text
Memory limit exceeded: The compiled model workspace footprint requires 1048576 bytes, which exceeds the configured target bare-metal RISC-V memory limit of 524288 bytes.
Mitigation:
1. Reduce sequence dimension settings or network hidden sizes.
2. Check if variable layout planning in TVM can be compressed.
3. Increase the memory pool allocation boundaries if targeting real hardware.
```

---

### Scenario 2: Accuracy Degradation Check (`AccuracyDropError`)
Raised when validation benchmarks indicate that calibrated INT8 quantization (or custom scheduling passes) degrades output numerical parity past the allowed accuracy tolerance.

#### Whitelisted Metadata Payload
```json
{
  "mse": 0.210123,
  "tolerance": 0.05,
  "details": "Quantized logits check failed."
}
```

#### Local Offline Fallback Explanation
```text
Accuracy degradation check failed: The optimized model has a Mean Squared Error (MSE) of 0.210123 compared to the host reference outputs, exceeding the allowed tolerance threshold of 0.05.
Mitigation:
1. Re-check the calibrated activation scale (metadata['activation_scale_source']) and the per-tensor weight scales; a clipped scale shows up only in the output.
2. Consider executing passes selectively (e.g. bypassing quantization on sensitive attention subgraphs).
3. Verify input range configurations to ensure valid logits mappings.
```

---

### Scenario 3: Unsupported Operator (`UnsupportedOperatorError`)
Raised during ONNX import or lowering when the model contains operators that cannot be legalized to standard RISC-V bare-metal compiler primitives.

#### Whitelisted Metadata Payload
```json
{
  "operator_name": "UnsupportedOpXYZ",
  "details": "No registered legalizer found."
}
```

#### Local Offline Fallback Explanation
```text
Unsupported operator: The operator 'UnsupportedOpXYZ' is not supported by the RISC-V TVM bare-metal backend.
Mitigation:
1. Implement or register a legalization pass in compiler.py to map 'UnsupportedOpXYZ' to supported primitives.
2. Adjust ONNX exporter configurations to decompose this operator during model export.
3. Implement a custom low-level fallback kernel in compiler.py/runner.py.
```

---

### Scenario 4: Compilation & Linking Failure (`CompilationError`)
Raised when TVM code emission or RISC-V GCC cross-compilation linking fails.

#### Whitelisted Metadata Payload
```json
{
  "stage": "linking",
  "command": "riscv64-unknown-elf-gcc",
  "details": "Undefined reference to '__tvm_ffi_softmax'"
}
```

#### Local Offline Fallback Explanation
```text
Compilation failure: The cross-compilation build stage failed during the 'linking' step.
Command executed: 'riscv64-unknown-elf-gcc'
Mitigation:
1. Verify target RISC-V cross-compiler options and flags.
2. Ensure TVM libraries are compiled and headers are in the search path.
3. Check for memory space address collisions in link.ld.
```
