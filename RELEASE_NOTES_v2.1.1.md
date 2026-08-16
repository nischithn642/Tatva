# TATVA 2.1.1

A hardware-aware AI compiler for RISC-V. Imports an ONNX model, emits C99, cross-compiles
it for a bare-metal RISC-V target, runs it under QEMU, and measures the result against a
host ONNX Runtime reference.

Windows, self-contained. The RISC-V toolchain ships inside — nothing to download, no admin
rights, no PATH changes.

**This release is about the operator list being true.** It had been written by hand, and it
was wrong in both directions: operators that compile were reported as unsupported, and one
that cannot be compiled produced a measurement anyway. 2.1 is unchanged in what it can
size; 2.1.1 changes what it will claim.

Everything in 2.1 still holds — the 1 GB ceiling, the computed memory budget, the
simulation-limit diagnoses. See `RELEASE_NOTES_v2.1.0.md` for those tables.

## Downloads

| File | Size | Use |
| :--- | :--- | :--- |
| `TATVA-Setup-2.1.1.exe` | 254.5 MB | Installer. Installs to your own user profile — no administrator password. |
| `TATVA-2.1.1-windows.zip` | 244.5 MB | Portable. Unzip and run `TATVA.exe`. Nothing to install. |

```
SHA256  TATVA-Setup-2.1.1.exe
        D1367C3281D9243BB422984887456B1B490F7835ED5BC22F1309F1F2D1AED1B5

SHA256  TATVA-2.1.1-windows.zip
        31B43C09A88B4C9157C49DD3B7AB1613EDFD371AC56BEE863D236D3A821A5C78
```

`v2.1.0` and its two assets are untouched. Their published checksums still describe the
bytes they always described; this is a separate release rather than a swap.

## The bug this release exists for

A `CumSum` model compiled without error, booted under QEMU, printed a cycle count, and
returned a tensor of `0.0`.

The harness emitter writes C only for bindings that legalize into a `call_tir` — a call
into a generated kernel — and it skipped every other binding in silence. TVM has no
lowering rule for `relax.cumsum`, so the operator survived legalization with no kernel
generated for it, its binding was skipped, and the graph's output buffer was never
written. Every stage reported success. The number printed was real. The answer was zeros.

**Nothing that only asks whether the pipeline raised would have found this.** It is the
same class of bug that once made LayerNorm and Gather return zeros, still live in a second
place, and it is the failure mode a compiler can least afford: a wrong answer that looks
exactly like a right one.

The emitter now separates a binding nothing reads — harmless, still skipped — from one
something reads and nothing writes. The second case raises, naming the operator:

```
'cumsum' could not be lowered to a kernel. TVM has no C implementation for it, so
there is nothing for the bare-metal harness to call.
```

Being told an operator cannot be compiled is worth more than a measurement of the wrong
program.

The same check caught a quieter instance. `Shape` becomes the pair `shape_of` /
`call_pure_packed("relax.run.shape_to_tensor")`, neither of which lowers to a kernel — but
the second is a real int64 tensor, and a downstream `take` dereferences it. In
`models/model.onnx` that is `Gather(Shape(x), 0)`, reading a buffer nothing had written,
harmless only because the reshape extents downstream were already static. Those extents are
constants after shape inference, so they are emitted as stores now.

## The operator list is measured now

Eleven operators were reported to users as unsupported by a backend that compiles every one
of them into a real C loop nest: `Conv` at one and three spatial dimensions,
`ConvTranspose`, `AveragePool`, `PRelu`, `LogSoftmax`, `Resize`, `InstanceNormalization`,
`Hardmax`, `Trilu`, and the `variance` and `argmax`/`one_hot` the last two decompose to.
Three of them carried a written, confident explanation of why they could never work. An
exported CNN is made of these.

`Pad`, `Tile` and `Einsum` were reported under the name `call_tir` — TVM's calling
convention, not anything in the user's file, and so a name that could not be looked up,
removed or replaced. The callee's name is reported now.

The list is 66 operators, and **no name enters it without an ONNX model in the test corpus
that compiles for RV64GC, boots under QEMU and matches ONNX Runtime to 1e-4.**

The one operator still refused is `cumsum`, and the reason is now measured rather than
assumed: the relax op is still standing after `LegalizeOps` with no PrimFunc generated,
which is what "no kernel" actually means. That test is what disproved the three
convolutions above — `relax.build` succeeding had been mistaken for evidence a model runs,
and CumSum is the proof it is not.

## How it is tested

`tests/onnx_corpus.py` builds 88 ONNX graphs from `onnx.helper` at collection time — 36
single-operator graphs, several whole models, and deliberately malformed files. 80 of them
run end to end: compiled for RV64GC, executed under QEMU, compared against ONNX Runtime at
1e-4. The rest must be *refused by name*; a refusal that does not name the operator fails
the suite as surely as a wrong number does.

The capability tables are pinned to each other by tests. The lowering table is exactly
`SUPPORTED_OPS`; the repair rules and the "cannot be fixed" reasons are disjoint from it;
every name in it is a real relax operator; and everything the corpus runs is in it. A claim
shown in the UI can no longer drift from what the backend does.

Full suite: 763 unit tests, 109 integration tests.

## Please read: what the numbers are, and are not

Unchanged from 2.1, and still the point rather than filler:

**They are emulator cycles, not silicon.** Measurement is QEMU system-mode under
`-icount shift=0`, converted at a **nominal** 100 MHz. Use them to compare two builds of
the same model against each other. Do not quote them as performance on real hardware.
Because `-icount shift=0` makes execution deterministic, mean, median and P95 are the same
number by construction — a property of the setup, not a claim of zero variance.

**INT8 remains an accuracy study, not a speed or size optimization.** The `quantize` pass
is fake-quantization: values round-trip through INT8 while the matmuls stay FP32, so it is
measured slower than FP32 and makes the binary larger.

## Known limitations

Carried forward from 2.1 unchanged — the emulator's 1020 MiB image ceiling, the soft-float
targets' ~20x cycle cost, and protobuf's 2 GB message limit on the optimizer path. See
`RELEASE_NOTES_v2.1.0.md` for the measurements behind each.

New to this release:

- **`CumSum` is refused, not compiled.** A prefix sum is sequential by definition, and the
  elementwise and reduction operators available here cannot express the carry between
  elements, so there is no rewrite to offer either. A model containing it fails at compile
  time with that explanation.
- **The operator list is 66 names, not everything ONNX defines.** It is now an honest floor
  rather than an optimistic guess: what is listed has been run. An operator absent from it
  may still work — it has simply not been proven, and TATVA will say so rather than
  pretend.
