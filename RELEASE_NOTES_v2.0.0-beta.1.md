# TATVA Beta 2.0

A hardware-aware AI compiler for RISC-V. Imports an ONNX model, emits C99, cross-compiles
it for a bare-metal RISC-V target, runs it under QEMU, and measures the result against a
host ONNX Runtime reference.

Windows, self-contained. The RISC-V toolchain ships inside — nothing to download, no admin
rights, no PATH changes.

## Downloads

| File | Size | Use |
| :--- | :--- | :--- |
| `TATVA-Setup-beta-2.0.exe` | 254.5 MB | Installer. Installs to your own user profile — no administrator password. |
| `TATVA-beta-2.0-windows.zip` | 244.5 MB | Portable. Unzip and run `TATVA.exe`. Nothing to install. |

```
SHA256  TATVA-Setup-beta-2.0.exe
        A114E9C475EB9652ED3AA50E04495515BD0EFFC9052535C8CBF6C197E006EC66

SHA256  TATVA-beta-2.0-windows.zip
        C532683B114B77B649F8D7F2D727E96A49275E6E6378386B383262176F8C25E5
```

## What's in this build

**Per-kernel cycle profiling.** The RISC-V code generator can bracket every kernel call
with a hardware cycle counter read, so latency is attributed to individual operators
rather than reported as one opaque total. Surface it with `tatva profile <model>`.

It is off by default and emits byte-identical C when off, so no latency TATVA reports is
ever measured with instrumentation active.

**Corrected benchmark documentation.** Earlier documentation carried performance figures
that were not reproducible against this repository. They have been re-measured end to end
rather than adjusted, and the corrections are stated plainly in `OPTIMIZATION.md`.

**Bundled toolchain**, verified at build time against all six supported targets:
`RV32IMC`, `RV32IMAC`, `RV64GC`, `RV64IMAFDC`, `RV64GCV`, `RV32EMC`, plus QEMU 9.2.4.

## Measured results

`models/model.onnx` — a synthetic attention subgraph, 17,719 B, 14 kernels:

| Configuration | ELF | Cycles | @100 MHz | Change | Parity (MSE) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline FP32 | 83,576 B | 73,090 | 0.73090 ms | reference | 0.000000 |
| FP32 + fast softmax | 79,288 B | 66,518 | 0.66518 ms | **+8.99%** | 0.000029 |
| Simulated INT8 | 91,072 B | 114,488 | 1.14488 ms | **−56.64%** | 0.000050 |

`models/model_pretrained.onnx` — BERT-tiny, 17.6 MB, 57 kernels:

| Configuration | Cycles | @100 MHz | Change | Parity (MSE) |
| :--- | :--- | :--- | :--- | :--- |
| Baseline FP32 | 118,806,414 | 1188.06414 ms | reference | 0.000000 |
| FP32 + fast softmax | 118,658,484 | 1186.58484 ms | **+0.12%** | 0.000001 |
| Simulated INT8 | 123,291,353 | 1232.91353 ms | **−3.77%** | 0.036593 |

Reproduce any row with `tatva baseline-test <model> --target RV64GC`.

> These ELF sizes are the ones tag `v2.0.0-beta.1` actually produced and are left as
> measured. A later change emits weights as a `.incbin` blob instead of a C array, which
> adds 24–64 B of alignment padding to every binary; see `OPTIMIZATION.md` for the current
> figures. No cycle count in either table moved.

## Please read: what these numbers are, and are not

**They are emulator cycles, not silicon.** Measurement is QEMU system-mode under
`-icount shift=0`, converted at a **nominal** 100 MHz. Use them to compare two builds of
the same model against each other. Do not quote them as performance on real hardware.

Because `-icount shift=0` makes execution deterministic, every timed sample is identical.
Mean, median, P95 and P99 are the same number by construction. That is a property of the
measurement setup, not a claim of zero variance.

**INT8 is an accuracy study, not a speed or size optimization — and this build is honest
about that.** The `quantize` pass round-trips values through INT8 while the matmuls stay
FP32. It is fake-quantization (QDQ), so it can only add work:

- It is **slower**. On the small model the cost is +414,160 cycles, of which `quantize`
  is 82.0% and `dequantize` 17.9% — 99.9% of the total.
- It makes the binary **larger**, 83,576 → 91,072 B, not smaller.
- Every kernel shared with the FP32 build is unchanged to the cycle, `matmul` included.
  That is the proof the MatMul never becomes integer.

What the pass is genuinely for is the MSE column: measuring the accuracy cost of INT8
before committing to it. The CLI warns about the expected slowdown when you select it.

A real integer GEMM was measured at **12–14% slower** than FP32 on scalar `rv64gc`,
because the core has a hardware FPU and a single-instruction `fmadd.s`. Integer MatMul is
therefore deliberately not enabled on scalar targets.

## Known limitations

- **`model_medium` fails numerical parity**, MSE ≈ 0.43 against a 0.05 tolerance. Open,
  unresolved, and reported as a failure rather than hidden.
- The `quantize` kernel is dominated by a per-element `roundf()` libm call that TVM emits.
  Tracked in `OPTIMIZATION.md` §4; not yet fixed.
- Fast softmax needs an attention pattern to fuse. On a graph without one, a 0.00% change
  is a real result, not a failure.
- Optimization is currently scalar. RVV-first compilation and auto-tiling are not in this
  build.
- Windows only.

## Verified

445 tests passing, lint clean. The installer payload is checked at build time for the
application, both toolchain executables, and the documentation.

These artifacts are **build-verified**: the payload is complete and the bundled toolchain
compiles all six targets. A full install-and-launch pass on a clean machine has not been
performed.
