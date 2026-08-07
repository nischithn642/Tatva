# TATVA Error Handling, Logging & Exit Code Specification

This document details Tatva's unified error boundary, structured logging system, secret-masking rules, and command exit codes.

---

## 1. Overview & Core Posture

Tatva implements a dual-tier error handling policy:
1. **User Experience Tier (Clean Default Output):** CLI and GUI users receive plain-English, non-colour-dependent failure explanations for known compiler issues (e.g. memory limits, unsupported operators, accuracy degradation). Raw stack trace dumps are suppressed by default.
2. **Developer & Support Tier (Structured Observability):** Developers can enable `-v`, `-vv`, `--debug`, or `--log-file <path>` to capture full debug logs and tracebacks. All log channels pass through a secret-masking formatter to guarantee private credentials (`TATVA_ANTHROPIC_KEY`, `RESEND_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`) are masked.

---

## 2. Command-Line Options & Verbosity Mapping

Global logging options are available across all Tatva commands:

| Option | Verbosity Level | Logging Level | Description |
|---|---|---|---|
| *(Default)* | `0` | `WARNING` | Clean terminal output; warnings and errors only. |
| `-v, --verbose` | `1` | `INFO` | High-level execution step updates. |
| `-vv, --verbose -v` | `2` | `DEBUG` | Full internal execution details. |
| `--debug` | N/A | `DEBUG` | Enables full debug logging and python traceback printing on errors. |
| `--log-file <PATH>` | N/A | `DEBUG` | Writes secret-masked structured logs to the specified file path. |
| `--json-log` | N/A | Current Level | Formats log entries as single-line JSON objects (`JSONL`). |

### Example Usage:
```bash
# Clean default execution
tatva optimize models/model.onnx --passes fuse --out build_opt

# Enable INFO logging
tatva -v optimize models/model.onnx --passes fuse --out build_opt

# Enable full DEBUG logging and output to log file
tatva --debug --log-file compilation.log optimize models/model.onnx --passes fuse --out build_opt
```

---

## 3. Exit Code Standard

Tatva enforces standardized exit codes across all CLI commands:

| Exit Code | Classification | Description |
|---|---|---|
| `0` | **Success** | Command completed successfully with expected outputs. |
| `1` | **Compiler / Diagnostic Error** | Known compilation failure (memory limit, unsupported operator, accuracy drop) or unexpected runtime exception. |
| `2` | **CLI Usage / Syntax Error** | Invalid command flags, missing required arguments, or unknown target variant parameter syntax. |

---

## 4. Error Boundaries & Failure Handling Flow

### Known Failures
When a known compiler exception occurs (e.g. `MemoryLimitExceededError`, `AccuracyDropError`, `UnsupportedOperatorError`, `CompilationError`), the tool:
1. Classifies the error into structured metadata via `classify_failure`.
2. Formats a plain-English explanation via `explain(context)` (offline rule-engine or Claude API).
3. Prints the diagnostic explanation to terminal / displays a GUI error dialog.
4. Logs the full exception traceback at `DEBUG` level to `--log-file` (if enabled).
5. Exits with code `1`.

### Unknown Failures
When an unhandled unexpected Python exception occurs, the tool:
1. Displays a calm user message:
   `An unexpected error occurred while processing your request.`
2. Instructs the user how to re-run with `--debug` or `--log-file`.
3. Logs the full exception traceback to `--log-file` or `DEBUG` stream.
4. Exits with code `1`.

---

## 5. Secret Hygiene in Logs

All logging handlers use `SecretMaskingFormatter` which automatically scans log messages for secret patterns (e.g. `sk-ant-...`, `re_...`, `eyJhbGci...`) and replaces them with masked placeholders (e.g. `sk-a***cdef`). Raw model weights and tensor arrays are excluded from logs.
