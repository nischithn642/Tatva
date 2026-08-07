# TATVA Security Policy & Threat Model Documentation

This document outlines Tatva's security posture, threat model, data egress policy, and secret management architecture.

---

## 1. Executive Summary & Core Security Posture

Tatva is designed with a **privacy-first, local-only processing architecture**:

1. **100% Local Model Processing:** All ONNX model files, weight tensors, graph structures, TVM Relay/Relax legalizations, and compiled ELF binaries are processed **strictly on the user's local machine**. No model weights or raw binary models leave the device.
2. **Strict Whitelist Data Egress:** When the user explicitly triggers AI diagnostics for build failures, only non-sensitive, whitelisted error metadata fields are sent to the Claude API.
3. **Centralized Secret Loader:** All secrets (Claude API keys, website Resend keys, Supabase credentials) are loaded from environment variables via `tatva.config`. Secrets are **never logged or printed**, even in verbose debugging modes.
4. **CI Secret Scanning:** Continuous Integration automatically scans every commit for leaked API keys, tokens, or private credentials using `gitleaks` and `detect-secrets`.

---

## 2. Threat Model

### Asset Classification & Boundaries

| Asset Class | Storage Location | Egress Allowed? | Mitigation Strategy |
|---|---|---|---|
| **ONNX Model Weights & Tensors** | Local Filesystem | **NEVER** | Processed 100% locally in TVM & QEMU. |
| **Compiled RISC-V ELF Binaries** | Local `build/` Directory | **NEVER** | Generated freestanding without network sockets. |
| **API Credentials (`TATVA_ANTHROPIC_KEY`)** | Environment Vars | **NEVER** | Loaded via `tatva.config`; masked in logs (`sk-a***cdef`). |
| **Compiler Error Summary Metadata** | In-Memory Object | **WHITELISTED ONLY** | Filtered by `whitelist_payload(...)` before API calls. |

---

## 3. Data Egress Section (What Data Leaves the Device and When)

Data leaves the local device **ONLY** when the Claude AI diagnostics engine is invoked to analyze a build failure.

### Egress Trigger Condition
- Invoked automatically or via `tatva diagnose <model.onnx>` when an error occurs AND `TATVA_ANTHROPIC_KEY` or `ANTHROPIC_API_KEY` is present.
- If no API key is configured or network is unavailable, diagnostics run **100% offline** via local rule-based fallback without making network requests.

### Enforced Egress Whitelist Schema
Before network serialization to `https://api.anthropic.com/v1/messages`, error metadata is processed through `tatva.diagnostics.whitelist_payload(...)`.

Only the following whitelisted metadata keys are allowed:

```json
{
  "error_type": "memory_limit_exceeded | accuracy_drop | unsupported_operator | compilation_error | unknown",
  "metadata": {
    "limit_bytes": 524288,
    "required_bytes": 1048576,
    "mse": 0.210123,
    "tolerance": 0.05,
    "operator_name": "UnsupportedOpXYZ",
    "stage": "linking",
    "command": "riscv64-unknown-elf-gcc",
    "details": "Sanitized error message summary without absolute host file paths"
  }
}
```

### Whitelist Enforcement & Automatic Scrubbing Rules:
1. **Raw Weight Tensor Scrubbing:** Any dictionary key containing numpy arrays, raw tensor shapes, or numeric lists > 16 elements is automatically dropped.
2. **Path Sanitization:** Absolute host filesystem paths (e.g. `C:\Users\Username\...\file.ext` or `/home/user/.../file.ext`) are stripped to basenames or sanitized strings to prevent leaking host directory structures.
3. **No File Transmission:** The ONNX file, C code (`operators.c`), header files, and ELF binaries are **NEVER** transmitted over the network.

---

## 4. Secret Management & Environment Variables

All credentials are loaded via the centralized `tatva.config` module:

```python
from tatva.config import get_anthropic_api_key, get_resend_api_key, get_supabase_key

api_key = get_anthropic_api_key(required=False)
```

- **Environment Variable Names:**
  - `TATVA_ANTHROPIC_KEY` or `ANTHROPIC_API_KEY` (Claude Diagnostics & Project Scaffolding)
  - `RESEND_API_KEY` (Web contact form notifications)
  - `SUPABASE_SERVICE_ROLE_KEY` (Web backend storage)
- **Zero Logging Policy:** Functions in `tatva.config` wrap key output with `mask_secret(...)` for debug output (`sk-a***cdef`). Missing required keys raise clean `SecretMissingError` exceptions without exposing system state.

---

## 5. Security Vulnerability Reporting

If you discover a potential security vulnerability in Tatva, please report it via email to security@tatva-compiler.dev rather than opening a public issue.
