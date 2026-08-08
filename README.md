# TATVA: Bare-Metal RISC-V Transformer Optimization Toolchain

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

<!-- Add the CI badge once this repo has a GitHub remote:
[![TATVA CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/<repo>/actions/workflows/ci.yml)
The previous badge pointed at github.com/tatva-compiler/tatva, which does not exist,
so it rendered as a permanently broken image at the top of the README. -->


**TATVA** is an AI model compilation and schedule optimization toolchain designed for deploying Transformer models on bare-metal RISC-V target architectures. It extends Apache TVM to optimize computational schedules (specifically attention, linear projection, and softmax operators), generating standalone, dependency-free C99 binaries for cycle-accurate execution on bare-metal RISC-V systems.

---

> [!IMPORTANT]
> **What TATVA Is:**
> A local-only, command-line and desktop GUI developer toolchain for compiling, optimizing, and simulating ONNX Transformer models on bare-metal RISC-V targets.
>
> **What TATVA Is NOT:**
> TATVA is **NOT** a SaaS platform, web service, cloud dashboard, user account manager, or model marketplace. It operates 100% locally on the developer's machine with zero mandatory cloud connectivity.

---

> [!NOTE]
> **Benchmarking Honesty Statement:**
> All benchmark timing metrics (cycle counts, execution latencies in milliseconds) reported in this repository and tool outputs are measured using **QEMU System-Mode Emulation (`-icount shift=0`) at a nominal 100 MHz target CPU frequency** unless explicitly labeled as physical silicon.

---

## Running It

> **New here?** [GETTING_STARTED.md](GETTING_STARTED.md) walks you from an unzipped
> folder to a measured result, and lists the failures worth recognising. This section is
> the short version.

**The app is `TATVA.exe`.** Unzip `TATVA-beta-2.0-windows.zip` anywhere and double-click
it — no Python, no install, no account.

Stages 01–04 (input, analyze, map, optimize) work the moment it opens. Stage 05
(generate) is the only one that shells out to the RISC-V cross-compiler and QEMU, and as
of Beta 2.0 the app can fetch them itself: **Diagnostics → Install the RISC-V
toolchain**. That downloads pinned xPack
builds (~520 MB) into `%LOCALAPPDATA%\tatva\toolchains`, needs no admin rights and adds
nothing to `PATH`. The card lists every URL and the destination before you press it.

Four sample models ship inside the zip. Stage 01 shows them as cards — pick **Tiny
transformer block** for a run with an attention pattern in it, which is the one the
softmax fusion pass can actually change.

To produce that zip from this repository, see
[Sharing TATVA With Someone Else](#sharing-tatva-with-someone-else).

The `tatva` CLI below is the same compiler with a terminal in front of it, for
scripting and CI. It is an alternative to the app, not a prerequisite for it.

---

## Key Features

- **Bare-Metal RISC-V Code Generation:** Compiles ONNX Transformer models to standalone C99 static functions linked with TVM Minimal C Runtime.
- **Custom Softmax Schedule Optimization:** Replaces generic heap-allocated multi-pass Softmax loops with a stack-allocated single-pass kernel using **Schraudolph's Fast Exponential Approximation**.
- **INT8 Quantization (experimental):** A symmetric QDQ pass implemented directly in TVM Relax — per-tensor weight scales, activation scales calibrated at the 99.9th percentile from real inference runs. It shrinks the SRAM footprint but is *slower* on scalar RISC-V targets, which have no INT8 dot-product instruction to exploit; `tatva optimize --passes quantize` says so before it runs.
- **Deterministic Emulation:** Execution timing using RISC-V hardware cycle counters (`rdcycle`) under system-mode QEMU (`-icount shift=0`).
- **Interactive Security-First Diagnostics:** Plain-English failure explanations and resolution guides via Anthropic Claude API (with strict metadata whitelist egress) or local rule-based fallback engines.
- **Session-Level Content-Hash Cache:** Fast repeated developer compilation runs reusing content-hashed model IRs and build artifacts.
- **Desktop Optimization Studio:** The primary interface, shipped as `TATVA.exe`. Walks the five stages — 01 input, 02 analyze, 03 map, 04 optimize, 05 generate — with a branded splash while the compiler backend loads, and a measured baseline-vs-optimized chart at the end. Its Diagnostics page installs the RISC-V toolchain in place, so a recipient of the zip never needs a terminal. Runs entirely offline apart from that one explicit download: no CDN, no fonts, no telemetry. Also reachable as `tatva gui` from a source checkout.

---

## 5-Minute Quickstart

Requires Python 3.12 or 3.13. (Apache TVM publishes no wheels for 3.14 yet, and TATVA
uses features not present in 3.11 — `pip` will refuse to install on anything else rather
than half-work.)

### 1. Installation

```bash
git clone <this-repo-url>
cd tatva
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. Install the RISC-V toolchain

TATVA needs a RISC-V cross-compiler and a QEMU system emulator. If you already have
`riscv-none-elf-gcc` and `qemu-system-riscv64` on your `PATH`, skip this — `tatva doctor`
will find them.

Otherwise, fetch pinned prebuilt binaries for your platform (Windows, Linux and macOS,
x64 and arm64). They install to a per-user directory, not into the source tree:

```bash
tatva setup
```

Use `tatva setup --dry-run` first to see exactly what it would download and where.

### 3. Verify Environment & Toolchain

Checks host Python, Apache TVM, ONNX Runtime, the RISC-V GCC cross-compiler and the QEMU
emulator, and tells you which is missing:

```bash
tatva doctor
```

Output:
```text
=== TATVA Environment & Toolchain Doctor ===
[OK] python (version: 3.13.13)
[OK] tvm (version: 0.25.0.post1)
[OK] onnxruntime (version: 1.28.0)
[OK] riscv_gcc (version: riscv-none-elf-gcc.exe (xPack GNU RISC-V Embedded GCC x86_64) 15.2.0)
[OK] qemu (version: xPack QEMU emulator version 9.2.4)
```

`tatva doctor` exits non-zero if anything is missing, so it works as a CI gate.

### 4. Analyze Model Graph

Inspect graph operators, parameter counts, and identify bottlenecks:

```bash
tatva analyze models/model.onnx
```

### 5. Run Baseline Test

Establishes baseline cycle latency and floating-point reference outputs on bare-metal RISC-V QEMU:

```bash
tatva baseline-test models/model.onnx --target RV64GC
```

### 6. Apply Schedule Optimizations

Applies Softmax kernel fusion and compiles an optimized binary into `build_opt`:

```bash
tatva optimize models/model.onnx --passes fuse --out build_opt
```

### 7. Interactive Failure Diagnostics

Diagnoses compiler errors or accuracy drops with structured explanations:

```bash
tatva diagnose models/model_unsupported.onnx
```

### 8. Launch Desktop GUI

Launch the multi-panel desktop engineering interface:

```bash
tatva gui
```

---

## Multi-Model Performance Summary

Empirical benchmark results under bare-metal RISC-V `RV64GC` system emulation (`-icount shift=0` @ 100MHz nominal frequency):

| Model | Model Size | Ops | Baseline Cycles | Optimized Cycles | Latency Gain | Parity (MSE) | Environment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **model.onnx** (Synthetic FP32) | 17.3 KB | 9 | 16,122,694 | **15,383,021** | **+4.6%** | 0.000000 | QEMU Simulated |
| **model_quant.onnx** (Dynamic INT8) | 6.0 KB | 14 | 19,856,579 | **18,989,000** | **+4.4%** | 0.000000 | QEMU Simulated |
| **model_pretrained.onnx** (BERT-tiny) | 16.8 MB | 156 | 118,806,393 | **118,540,094** | **+0.2%** | 0.000000 | QEMU Simulated |
| **model_unsupported.onnx** | 17.3 KB | N/A | N/A | N/A | N/A | N/A | EXPECTED FAIL |

<!-- Model Size is measured from models/*.onnx on disk. The column previously read
     439.1 KB / 262.2 KB / 17.4 MB / 439.1 KB, which matched no file in the repository;
     439.1 KB in particular appeared twice for two files that are both 17.3 KB. Cycle
     counts are unchanged -- those are the measured figures. -->

Ops counts are ONNX graph nodes. The GUI's stage 02 reports Relax calls after import,
which is a larger number for the same model (model.onnx: 9 ONNX nodes → 17 Relax calls
across 11 operator kinds) — both are counting honestly, just at different layers.

> [!NOTE]
> **Scalar Quantization Note:** On scalar RISC-V cores (`rv64gc` without vector extensions), INT8 quantization reduces file storage footprint by 40%–72%, but introduces a +19% to +22% cycle count latency overhead due to loop-by-loop software emulation of dequantization scaling and zero-point casts.

---

## Sharing TATVA With Someone Else

### 1. Send them the ZIP — this is the way TATVA ships

The person on the other end needs no Python, no `pip`, no repository and no reading.
They unzip a folder and double-click `TATVA.exe`.

```bash
python build_exe.py
```

That runs PyInstaller against [tatva.spec](tatva.spec) and produces:

| Output | What it is |
| :--- | :--- |
| `dist/TATVA/TATVA.exe` | the app — double-click to launch |
| `dist/TATVA-<version>-windows.zip` | the same folder, zipped; this is what you send |

`build_exe.py --no-zip` builds without zipping; `build_exe.py --zip-only` re-zips a
folder you already built. A `README.txt` is written next to the exe for whoever receives
it, since they will not have this file.

Notes that matter:

- **It is a folder, not a single file, on purpose.** A one-file build unpacks Apache
  TVM's native libraries — most of a gigabyte — into a temp directory on every launch.
  The folder build starts in seconds and zips just as well.
- **PyInstaller does not cross-compile.** Build on the OS you are shipping to. The zip
  above is Windows-only; run the same command on Linux or macOS for those.
- **Windows SmartScreen will warn** that the publisher is unknown, because the build is
  not code-signed. "More info" → "Run anyway". Signing it needs a code-signing
  certificate, which is a purchase, not a build flag.
- **The RISC-V toolchain is not inside the zip** — it is ~520 MB and platform-specific,
  which would quadruple the download for the people who already have it. Stages 01–04
  (input, analyze, map, optimize) work on a bare machine. For stage 05 the recipient presses
  **Diagnostics → Install the RISC-V toolchain** inside the app; `riscv-none-elf-gcc` and
  `qemu-system-riscv64` already on `PATH` are picked up instead, and the Diagnostics page
  shows the resolved path for each.

### 2. Send them a wheel (for someone who already has Python)

```bash
uv build
```

That writes `dist/tatva_compiler-<version>-py3-none-any.whl` (~1.3 MB). Anyone with
Python 3.12 or 3.13 installs it with `pip install tatva_compiler-<version>-py3-none-any.whl`
and gets the `tatva` CLI as well as the GUI — which the zip does not include, since the
exe launches straight into the desktop app.

To publish it so `pip install tatva-compiler` works for everyone, upload to PyPI with
`uv publish` (or `twine upload dist/*`). The package name `tatva-compiler` is not yet
claimed on PyPI — check before you rely on it.

### 3. Send them the repository (for collaborators)

```bash
git clone <repo-url> && cd tatva
pip install -e ".[dev]"
tatva setup && tatva doctor
```

`tatva doctor` is the gate — if it prints five `[OK]` lines, everything else in this
README works. If it doesn't, it names the missing piece and where it looked.

### What not to send

- **`riscv-toolchain/`, `qemu/`, `renode/`, `.venv/`, `build_*/`, `dist/`** — all
  gitignored, all reproducible with `tatva setup`, `pip install` and a rebuild. Together
  they are several gigabytes.
- **`scratch/*_build_*/`** — generated build trees, each one a few hundred copied TVM
  headers. The helper scripts in `scratch/` are tracked; their output is not.
- **Your `.env` or API keys.** TATVA reads `TATVA_ANTHROPIC_KEY` / `ANTHROPIC_API_KEY`
  and `TATVA_NVIDIA_KEY` / `NVIDIA_API_KEY` from the environment or a local `.env`, and
  never writes a key to disk. Both are optional — every AI feature falls back to a local
  rules engine when no key is present.

---

## Architecture & Developer Guides

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Compiler pipeline, TVM extension decision table, module map, and target abstraction layer.
- [docs/EXTENDING_TARGETS.md](docs/EXTENDING_TARGETS.md): Tutorial for adding new RISC-V target variants.
- [docs/SECURITY.md](docs/SECURITY.md): Threat model, metadata whitelist egress policy, and secret management architecture.
- [docs/LOGGING_AND_ERRORS.md](docs/LOGGING_AND_ERRORS.md): Logging levels, `--log-file` options, and exit code standards.
- [BASELINE.md](BASELINE.md): Baseline performance tracking.
- [OPTIMIZATION.md](OPTIMIZATION.md): Schraudolph Softmax mathematical derivations and empirical quantization findings.
- [DIAGNOSTICS.md](DIAGNOSTICS.md): Exception taxonomy and Claude API prompt design.
- [GETTING_STARTED.md](GETTING_STARTED.md): First run, from an unzipped folder to a measured result — app, CLI and source checkout.
- [CONTRIBUTING.md](CONTRIBUTING.md): Guidelines for developer environment setup, testing, and PR expectations.
- [CHANGELOG.md](CHANGELOG.md): Milestone release history.

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
