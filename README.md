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

## Key Features

- **Bare-Metal RISC-V Code Generation:** Compiles ONNX Transformer models to standalone C99 static functions linked with TVM Minimal C Runtime.
- **Custom Softmax Schedule Optimization:** Replaces generic heap-allocated multi-pass Softmax loops with a stack-allocated single-pass kernel using **Schraudolph's Fast Exponential Approximation**.
- **INT8 Quantization (experimental):** A symmetric QDQ pass implemented directly in TVM Relax — per-tensor weight scales, activation scales calibrated at the 99.9th percentile from real inference runs. It shrinks the SRAM footprint but is *slower* on scalar RISC-V targets, which have no INT8 dot-product instruction to exploit; `tatva optimize --passes quantize` says so before it runs.
- **Deterministic Emulation:** Execution timing using RISC-V hardware cycle counters (`rdcycle`) under system-mode QEMU (`-icount shift=0`).
- **Interactive Security-First Diagnostics:** Plain-English failure explanations and resolution guides via Anthropic Claude API (with strict metadata whitelist egress) or local rule-based fallback engines.
- **Session-Level Content-Hash Cache:** Fast repeated developer compilation runs reusing content-hashed model IRs and build artifacts.
- **Desktop Engineering GUI:** Multi-panel desktop UI (`tatva gui`) with splash screen lazy module loading and non-blocking background workers.

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
| **model.onnx** (Synthetic FP32) | 439.1 KB | 9 | 16,122,694 | **15,383,021** | **+4.6%** | 0.000000 | QEMU Simulated |
| **model_quant.onnx** (Dynamic INT8) | 262.2 KB | 14 | 19,856,579 | **18,989,000** | **+4.4%** | 0.000000 | QEMU Simulated |
| **model_pretrained.onnx** (BERT-tiny) | 17.4 MB | 156 | 118,806,393 | **118,540,094** | **+0.2%** | 0.000000 | QEMU Simulated |
| **model_unsupported.onnx** | 439.1 KB | N/A | N/A | N/A | N/A | N/A | EXPECTED FAIL |

> [!NOTE]
> **Scalar Quantization Note:** On scalar RISC-V cores (`rv64gc` without vector extensions), INT8 quantization reduces file storage footprint by 40%–72%, but introduces a +19% to +22% cycle count latency overhead due to loop-by-loop software emulation of dequantization scaling and zero-point casts.

---

## Sharing TATVA With Someone Else

Three ways, in increasing order of how little the other person has to know.

### 1. Send them the repository (best for collaborators)

```bash
git push -u origin master
```

They then need three commands:

```bash
git clone <repo-url> && cd tatva
pip install -e ".[dev]"
tatva setup && tatva doctor
```

`tatva doctor` is the gate — if it prints five `[OK]` lines, everything else in this
README works. If it doesn't, it names the missing piece and where it looked.

### 2. Send them a wheel (best for someone who just wants to use it)

```bash
uv build
```

That writes `dist/tatva_compiler-<version>-py3-none-any.whl` (~1.3 MB). Anyone with
Python 3.12 or 3.13 installs it with `pip install tatva_compiler-<version>-py3-none-any.whl`
and gets the `tatva` command, the GUI, and the bundled web UI. They still need
`tatva setup` for the RISC-V toolchain itself, which is too large to ship in a wheel.

To publish it so `pip install tatva-compiler` works for everyone, upload to PyPI with
`uv publish` (or `twine upload dist/*`). The package name `tatva-compiler` is not yet
claimed on PyPI — check before you rely on it.

### 3. Send them an executable (for someone without Python — expect some work)

```bash
pip install pyinstaller
pyinstaller tatva.spec
```

Produces a self-contained `dist/Tatva` binary. Build it on the platform you are shipping
to; PyInstaller does not cross-compile.

Honest caveat: this route is the least tested of the three. Apache TVM loads its native
libraries through `ctypes`, so no `import` statement points at them and PyInstaller
cannot find them by tracing imports. `tatva.spec` calls `collect_dynamic_libs('tvm')` to
pull them in, but if a compile fails inside the frozen app with a missing-library error,
that is where to look. The wheel is the supported path.

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
- [CONTRIBUTING.md](CONTRIBUTING.md): Guidelines for developer environment setup, testing, and PR expectations.
- [CHANGELOG.md](CHANGELOG.md): Milestone release history.

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
