# TATVA Release Changelog

All notable changes to the TATVA project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-08-08 (Stability & Distribution)

> **On the version number going backwards.** The previous entry claims 1.0.0. At that
> point a fresh clone could not run `tatva --help`: the console script pointed at a
> module that had moved, four files disagreed about the version, and the CI matrix
> referenced a Python version with no TVM wheels. 1.0.0 described an intent, not a
> release. 0.3.0 describes what is actually here and verified. The next 1.0.0 will be
> earned rather than declared.

### Added
- **`tatva setup`** — cross-platform, pinned, consent-gated installer for the RISC-V GCC
  cross-compiler and QEMU. Installs to a per-user directory, detects existing installs,
  and verifies the binary runs before reporting success. `--dry-run` shows the exact URLs
  and destinations without downloading. Replaces `setup_env.py` and `setup_simulators.py`,
  which hardcoded `win32-x64` into the download URL and so fetched Windows binaries on
  Linux and macOS.
- **`tatva doctor` now reports every path it searched** for a missing tool, in text and
  in `--json`.
- 38 new tests covering the installer, platform detection, toolchain discovery precedence,
  and every `TatvaPyBridge` method the GUI calls. Suite: 93 → 135 tests.

### Changed
- **Benchmark harness now matches the model.** The generated harness declared a fixed
  input signature rather than the model's own, so any model whose inputs differed from the
  bundled fixture either failed to link or measured the wrong thing.
- **Accuracy checking means something.** Parity is verified against host ONNX Runtime
  output with a reported MSE, instead of passing unconditionally.
- **INT8 quantization is labeled honestly.** It reduces footprint but is *slower* on
  scalar RISC-V, which has no INT8 dot-product instruction; the CLI says so before running.
- **`RV64GCV` and `RV32EMC` marked experimental.** Vector targets are selectable but not
  exploited — code generation is scalar C, so `rv64gcv` currently buys nothing over
  `rv64gc`. Claiming otherwise in a dropdown was the misleading part.
- **One source of truth for the version** (`src/tatva/__init__.py`), consumed by the
  packaging metadata, the CLI and the GUI.
- Toolchain discovery order is now PATH → per-user tools dir → legacy in-repo directories,
  shared by the CLI, the runner and the scaffolding executor instead of three
  reimplementations.

### Fixed
- The `tatva` console script pointed at a module path that no longer existed, so a fresh
  install produced an `ImportError` on every invocation.
- Secrets are no longer written to disk by the scaffolding config `save()` path.
- `.env` files are now actually loaded rather than silently ignored.
- CLI error handling collapsed into a single boundary; three paths printed Python
  tracebacks at users who had merely mistyped a path.
- The test suite wrote into the developer's real `%APPDATA%` and read its own leftovers
  back on later runs, so one test failed against a config file it had written itself.
- Build directories no longer leak gigabytes into the working tree.

### Removed
- 523 tracked files (660 → 136). `scratch/val_build_*` alone held 497 copied TVM headers —
  roughly 75% of every clone — for build output that any `tatva optimize` regenerates.
- Duplicate top-level copies of `compiler.py`, `config.py`, `runner.py`, `optimizer.py`
  and `diagnostics.py`; `src/` is now the only source of truth.

---

## [1.0.0] - 2026-07-23 (Milestone M6 — Public Release)

### Added
- **Public Release Suite (M6):** Complete documentation set (`README.md`, `ARCHITECTURE.md`, `EXTENDING_TARGETS.md`, `SECURITY.md`, `LOGGING_AND_ERRORS.md`, `CONTRIBUTING.md`).
- **Structured Error Handling & Logging (Phase 20):** Module-level loggers in `logging_setup.py`, verbosity flag support (`-v`, `-vv`, `--debug`, `--log-file`, `--json-log`), secret-masking log formatter (`SecretMaskingFormatter`), and unified CLI/GUI exception boundaries.
- **Session-Level Content-Hash Caching (Phase 19):** Bounded session cache in `_cache.py` storing model IRs and build artifacts keyed by SHA256 content hashes, saving 8.2s tooling wall-time on repeated runs.
- **Security Hardening & Egress Whitelist (Phase 18):** Centralized secret manager in `config.py`, strict whitelist gating in `diagnostics.py`, and CI secret scanning via `gitleaks`.
- **Marketing Website & Contact Backend (Phase 17):** Accessible hardware dark-mode website in `website/` with Express/Node API backend, honeypot protection, per-email rate limiting, HTML escaping, Supabase database storage, and Resend transactional emails.

### Changed
- **CLI Diagnostics Integration (M5):** Updated CLI error explanations to route automatically through Anthropic Claude API or local offline rule engine.
- **Desktop Engineering GUI (M5):** Multi-panel desktop app (`tatva gui`) with splash screen lazy module loading and non-blocking background threads.

---

## [0.5.0] - 2026-07-20 (Milestone M5 — GUI & Desktop Scaffolding)

### Added
- Multi-panel Desktop GUI (`tatva gui`) with pipeline runner, target dropdowns, real-time log stream, and interactive failure diagnostic popups.
- Project Scaffolding Assistant for RISC-V starter project generation.

---

## [0.4.0] - 2026-07-15 (Milestone M4 — Diagnostics Engine)

### Added
- Structured Exception Taxonomy (`MemoryLimitExceededError`, `AccuracyDropError`, `UnsupportedOperatorError`, `CompilationError`).
- Anthropic Claude API diagnostics integration (`explain`) with strict payload whitelist gating (`whitelist_payload`).
- Deterministic offline fallback explanation engine for zero-network environments.

---

## [0.3.0] - 2026-07-10 (Milestone M3 — Optimization & Quantization)

### Added
- Schraudolph's Fast Exponential Approximation single-pass Softmax fusion kernel, achieving +4.6% latency speedup on synthetic FP32 subgraphs.
- Dynamic 8-Bit Integer Quantization (`quantize`) pass reducing model storage size by 40%–72%.
- Multi-model benchmarking suite validating pretrained BERT-tiny models (`models/model_pretrained.onnx`).

---

## [0.2.0] - 2026-07-05 (Milestone M2 — Compiler & Runner Pipeline)

### Added
- Bare-metal RISC-V C99 code generator emitting freestanding functions linked with TVM Minimal C Runtime.
- Target Architecture Registry (`TargetVariant`) supporting `RV64GC` and `RV32EMC`.
- QEMU system emulation runner (`qemu-system-riscv64`) with deterministic hardware cycle timing (`rdcycle`, `-icount shift=0`).

---

## [0.1.0] - 2026-06-25 (Milestone M1 — Baseline Foundation)

### Added
- Initial project layout, ONNX importer interface, and CLI scaffolding (`tatva doctor`, `tatva baseline-test`).
- Unit testing framework (`pytest`) and code quality linter configuration (`ruff`).
