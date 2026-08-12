# TATVA Living Optimization & Quantization Documentation

This document details the mathematical derivation, C99 implementation, and empirical benchmark validation of TATVA's optimization passes.

---

> [!IMPORTANT]
> **Benchmarking Honesty Statement:**
> All cycle counts and millisecond latencies reported in this document are measured under **QEMU System-Mode Emulation (`-icount shift=0`) at a nominal 100 MHz target CPU frequency** running bare-metal RISC-V `rv64gc` under OpenSBI v1.5.1 firmware.

---

## 1. Technical Optimization Details

### Softmax Operation Analysis (Default TVM vs. TATVA Single-Pass Kernel)

In default TVM C codegen, a 2D Softmax tensor `(rows, cols)` is lowered into four separate loop nests requiring temporary heap allocation (`TVMBackendAllocWorkspace`):
1. **Max reduction loop:** Scans row elements to find `max_val`.
2. **Exponential calculation loop:** Allocates heap memory and writes `expf(val - max_val)`.
3. **Sum reduction loop:** Reads exponentials from heap and computes the sum.
4. **Division normalization loop:** Reads exponentials again, divides by row sum, and writes outputs.

**Bottlenecks on Bare-Metal RISC-V:**
- **Heap Overhead:** Dynamic memory allocation in bare-metal supervisor environments causes allocator traps.
- **Bandwidth Friction:** Four sequential sweeps over external memory limit throughput.
- **Transcendental Library Overhead:** Software `expf` functions consume hundreds of clock cycles per element on scalar hardware.

### Our Optimized Softmax Kernel Design

We replace standard Softmax with a custom register-based, single-pass implementation utilizing **Schraudolph's Fast Exponential Approximation**:

1. **Single-Pass Row Execution:**
   - Computes `max_val`, exponentials, and row sums in a single pass using a fast stack-allocated variable-length array (`float local_exp[cols]`).
   - Completely eliminates `TVMBackendAllocWorkspace` heap calls.

2. **Schraudolph Fast Exponent Mathematical Derivation:**
   An IEEE-754 single-precision float represents $x$ as $1.m \times 2^{e}$. By scaling and shifting the exponent bits directly into integer representation, $e^x$ is approximated as:
   $$\text{Bits}(e^x) \approx \text{int}(x \cdot \log_2(e) \cdot 2^{23} + 1065353216)$$

   In optimized C:
   ```c
   union {
       float f;
       int32_t i;
   } u;
   float val = in_row[k] - max_val;
   if (val < -15.0f) {
       local_exp[k] = 0.0f; // Underflow clipping
   } else {
       u.i = (int32_t)(val * 12102203.0f + 1065353216.0f);
       local_exp[k] = u.f;
   }
   ```
   This reduces the software `expf` call to **1 float multiply, 1 float-to-int cast, 1 int add, and 1 bitwise copy** (3–5 instructions total).

---

## 2. Empirical Benchmark Results

> [!NOTE]
> **Every row below was re-measured against the current repository.** An earlier
> version of this table reported a 161.2269 ms baseline for `models/model.onnx`
> from a 439.1 KB file. `models/model.onnx` is 17,719 bytes and has been since
> the only commit that touched it, so those rows described a fixture that is not
> in this repository and could not be reproduced. They have been replaced rather
> than adjusted. Reproduce with `tatva baseline-test <model> --target RV64GC`;
> per-kernel breakdowns come from `tatva profile <model>`.
>
> Parity MSE is measured against the **host ONNX Runtime** result, not against
> TATVA's own FP32 binary — scoring an optimized build against the baseline build
> would only measure how far the two RISC-V binaries drift from each other.

### Synthetic Attention Subgraph (`models/model.onnx`, 17,719 B, 14 kernels)

| Configuration | ELF Size | Simulated Cycles | Simulated Time (@ 100MHz) | Latency Gain | Parity (MSE) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP32 + Standard Softmax)** | 83,576 B | 73,090 | 0.73090 ms | Reference | 0.000000 |
| **FP32 + Fast Softmax Kernel** | 79,288 B | **66,518** | **0.66518 ms** | **+8.99%** | 0.000029 |
| **Simulated INT8 (Standard Softmax)** | 91,072 B | 114,488 | 1.14488 ms | **−56.64%** | 0.000050 |
| **Simulated INT8 + Fast Softmax** | 86,896 B | 107,916 | 1.07916 ms | **−47.65%** | 0.000175 |

### Pretrained BERT-tiny (`models/model_pretrained.onnx`, 17,607,002 B, 57 kernels)

| Configuration | ELF Size | Simulated Cycles | Simulated Time (@ 100MHz) | Latency Gain | Parity (MSE) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP32 + Standard Softmax)** | 17,822,152 B | 118,806,414 | 1188.06414 ms | Reference | 0.000000 |
| **FP32 + Fast Softmax Kernel** | 17,818,616 B | **118,658,484** | **1186.58484 ms** | **+0.12%** | 0.000001 |
| **Simulated INT8 (Standard Softmax)** | 17,886,240 B | 123,291,353 | 1232.91353 ms | **−3.77%** | 0.036593 |
| **Simulated INT8 + Fast Softmax** | 17,882,696 B | 123,144,896 | 1231.44896 ms | **−3.65%** | 0.031808 |

---

## 3. Honest Technical Analysis & Key Insights

1. **Fast-Softmax Impact Shrinks With Model Size:**
   - On the small attention subgraph the fast softmax kernel is worth **+8.99%**.
   - On full BERT-tiny it is worth **+0.12%**.
   - **Reason:** softmax runs on `(seq_len, seq_len)` query-key tensors. As the
     embedding and dense dimensions grow, MatMul dominates: per-kernel profiling
     puts softmax at 109,340 of 729,410 attributed cycles (15.0%) on the small
     model, so even eliminating it entirely could not have paid more than that.

2. **The INT8 Pass Is a Numerical Study, Not a Latency or Footprint Optimization.**

   This is the single most important correction in this document. A previous
   version claimed INT8 "reduces binary and storage size by **40% to 72%**" and
   cost "+19.0% to +23.1%" in latency. Both figures are wrong, and the size claim
   is wrong in the wrong direction:

   - **Footprint: INT8 makes the binary *larger*.** 83,576 → 91,072 B (**+9.0%**)
     on the small model and 17,822,152 → 17,886,240 B (**+0.36%**) on BERT-tiny.
     The pass round-trips values through INT8 and computes in FP32, so the weights
     stay FP32 on device and the extra quantize/dequantize kernels add code. The
     emitted `weights.h` moves 72,079,605 → 71,906,733 B (**−0.24%**), not −72%.
   - **Latency: INT8 is slower, and the profiler says exactly why.** On the small
     model the regression is **+414,160 cycles**, of which `quantize` accounts for
     339,730 (**82.0%**) and `dequantize` for 74,130 (**17.9%**) — **99.9% of the
     total**. Every one of the 14 kernels shared with the FP32 build is *bit-identical*,
     drift of exactly **0 cycles**, including `matmul` at 427,700 cycles in both builds.
   - **The dominant kernel never becomes integer.** That `matmul` is unchanged is
     the proof: this is fake-quantization (QDQ), so the MatMul still executes in
     FP32 and gains nothing. The pass can only add work.
   - **The same result holds at scale.** On BERT-tiny the regression is
     **+44,854,330 cycles** against 46,192,150 cycles spent in the 16
     quantize/dequantize kernels. That is *103%* of the delta — the shared kernels
     got 1,345,360 cycles **faster**, almost all of it `erf`
     (31,763,450 → 30,393,760), because quantized activations change its input
     distribution. The overhead is therefore slightly larger than the net
     regression, not smaller.
   - **Why the quantize kernel is so expensive:** TVM lowers `relax.quantize` to a
     divide plus `roundf()`. On this bare-metal target `roundf` is a non-inlined
     libm call per element, which is why `quantize` costs ~4.6x what `dequantize`
     (a plain multiply) costs on the same tensors.

   **What the pass is legitimately for:** measuring the accuracy cost of INT8
   before committing to it. The MSE column is the deliverable — 0.036593 on
   BERT-tiny against a 0.05 tolerance is a real, useful number.

3. **Constant weights are folded at compile time.** Both halves of the QDQ pair on
   a constant weight tensor are evaluated during compilation and replaced by a
   single FP32 constant, so no device cycles are spent re-quantizing values that
   cannot change. On BERT-tiny this covers 12 of the 18 quantized ops and removed
   about 155.7 ms of the original regression. It reduces activation-workspace RAM
   (global_pool 4,869,696 → 2,903,616 B), **not** flash — anything reporting this
   as a footprint reduction is reporting a number this code does not deliver.

---

## 4. Tracked Optimization Issue

- [ ] **Tracked Optimization Issue: `quantize` Kernel Cost on RISC-V Bare-Metal**
  - **Status:** Open / Tracked
  - **Description:** The `quantize` kernels are 82% of the INT8 latency regression
    on the small model and the largest single group on BERT-tiny. The cost is
    dominated by the per-element `roundf()` libm call TVM emits, not by the divide.
  - **Remediation Plan:** Replace TVM's default lowering of `relax.quantize` with a
    custom legalization that emits a hardware round-and-convert on targets that
    have both RV64 and hardware floating point (`fcvt.l.s` is RV64-only), and
    revisit integer MatMul under RVV, where a true int8×int8→int32 GEMM can pay for
    itself. Measured on scalar `rv64gc`, a real integer GEMM is **12–14% slower**
    than FP32 because the core has a hardware FPU and a single-instruction `fmadd.s`
    — so integer MatMul is deliberately **not** enabled on scalar targets.
