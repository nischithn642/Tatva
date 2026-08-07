# TATVA: Bare-Metal RISC-V Transformer Optimization Toolchain

[![TATVA CI](https://github.com/tatva-compiler/tatva/actions/workflows/ci.yml/badge.svg)](https://github.com/tatva-compiler/tatva/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

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
- **Dynamic 8-Bit Quantization:** Supports ONNX Runtime INT8 quantization for constrained SRAM footprints.
- **Deterministic Emulation:** Execution timing using RISC-V hardware cycle counters (`rdcycle`) under system-mode QEMU (`-icount shift=0`).
- **Interactive Security-First Diagnostics:** Plain-English failure explanations and resolution guides via Anthropic Claude API (with strict metadata whitelist egress) or local rule-based fallback engines.
- **Session-Level Content-Hash Cache:** Fast repeated developer compilation runs reusing content-hashed model IRs and build artifacts.
- **Desktop Engineering GUI:** Multi-panel desktop UI (`tatva gui`) with splash screen lazy module loading and non-blocking background workers.

---

## 5-Minute Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/tatva-compiler/tatva.git
cd tatva

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install in editable mode with development dependencies
pip install -e .[dev]
```

### 2. Verify Environment & Toolchain

Verify host Python, Apache TVM, ONNX Runtime, RISC-V GCC cross-compiler, and QEMU emulator:

```bash
tatva doctor
```

Output:
```text
=== TATVA Environment & Toolchain Doctor ===
[OK] python (version: 3.13.7)
[OK] tvm (version: 0.19.0)
[OK] onnxruntime (version: 1.20.1)
[OK] riscv_gcc (version: riscv-none-elf-gcc 14.2.0)
[OK] qemu (version: qemu-system-riscv64 9.2.0)
```

### 3. Analyze Model Graph

Inspect graph operators, parameter counts, and identify bottlenecks:

```bash
tatva analyze models/model.onnx
```

### 4. Run Baseline Test

Establishes baseline cycle latency and floating-point reference outputs on bare-metal RISC-V QEMU:

```bash
tatva baseline-test models/model.onnx --target RV64GC
```

### 5. Apply Schedule Optimizations

Applies Softmax kernel fusion and compiles an optimized binary into `build_opt`:

```bash
tatva optimize models/model.onnx --passes fuse --out build_opt
```

### 6. Interactive Failure Diagnostics

Diagnoses compiler errors or accuracy drops with structured explanations:

```bash
tatva diagnose models/model_unsupported.onnx
```

### 7. Launch Desktop GUI

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
