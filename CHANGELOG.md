# TATVA Release Changelog

All notable changes to the TATVA project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
