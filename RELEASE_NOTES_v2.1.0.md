# TATVA 2.1

A hardware-aware AI compiler for RISC-V. Imports an ONNX model, emits C99, cross-compiles
it for a bare-metal RISC-V target, runs it under QEMU, and measures the result against a
host ONNX Runtime reference.

Windows, self-contained. The RISC-V toolchain ships inside — nothing to download, no admin
rights, no PATH changes.

**This release is about model size.** Beta 2.0 compiled the models in `models/` and fell
over somewhere below 100 MB. 2.1 compiles and runs a 1 GB ONNX end to end, including the
external-data form those models actually ship in.

1 GB is close to the ceiling, not a round number chosen for the headline. The linked image
has to fit below QEMU's device tree at `0xBFE00000`, which leaves **1020 MiB** — the 1 GB
model above clears it by 11.9 MiB. See *Known limitations*.

## Downloads

| File | Size | Use |
| :--- | :--- | :--- |
| `TATVA-Setup-2.1.exe` | 254.5 MB | Installer. Installs to your own user profile — no administrator password. |
| `TATVA-2.1-windows.zip` | 244.5 MB | Portable. Unzip and run `TATVA.exe`. Nothing to install. |

```
SHA256  TATVA-Setup-2.1.exe
        E78B4F427F9EB89BDA96AFBA09EDF5A0AECAD52B1C7E11F288B2DDAADDEF14D7

SHA256  TATVA-2.1-windows.zip
        0921142793F51742C655A24297A54BD0CF8F5307A1D8B7C1C3AC1BCAFA9CB332
```

## What's in this build

**Large models compile.** Weights are written to a binary blob and pulled into the image
with `.incbin`, instead of being emitted as a C initializer list. The initializer cost
about 4.1 source bytes per model byte, so a 116 MB model produced a ~480 MB `weights.h`
and drove `cc1` past 3 GB of RSS before it was killed — the wall was the C front end, not
anything about RISC-V. The assembler now copies the bytes; nothing parses them.

**The memory budget is computed, not fixed.** The linker's RAM region is sized from the
weights, the planned activation offsets, the workspace pool, model I/O and the stack, plus
16 MiB of slack rounded to a 16 MiB granule, with a 128 MiB floor. QEMU is then given that
much memory plus the 2 MiB reserved for firmware, and a timeout that scales with it
(0.6 s per MiB per run, 30 s floor). A 1 GB model used to die on a fixed timeout even when
the ELF was correct.

**A region overflow is reported as a memory limit.** `region 'RAM' overflowed by N bytes`
from `ld` now surfaces as a memory-limit failure carrying the byte count, and the
explanation names what will actually help — shorten the sequence axis, which is bound to
32 and drives the activation pool but not the weights — rather than offering a pool size
to raise that no longer exists.

**A console dock in the studio window** — Messages, Log and Reports, with per-tab counts,
collapsible and clearable. The sidebar sections became real collapsible controls.

**The release plumbing no longer hardcodes the version.** The installer's own filename,
its payload selection, and the stage guide's cover all derive from
`src/tatva/__init__.py`, and the wizard's version literals are compared against the
package before it is built. That set of bugs is the reason this release exists in the
shape it does: `build_installer.py` picked the alphabetically last zip in `dist/`, and
`TATVA-2.1-windows.zip` sorts *before* `TATVA-beta-2.0-windows.zip`, so it would have
shipped the previous release's payload inside an installer named 2.1.

## Measured results

Both ends of the supported size range were compiled and run **on RV64GC**. Timings are wall
clock on a 15.2 GB machine; the latency figure is QEMU system mode with `-icount shift=0`
at a nominal 100 MHz. The target matters more than the table suggests — see the soft-float
note under *Known limitations*.

| Model | Import | Compile | ELF | RAM region | QEMU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 116.1 MB | 6.0 s | 3.9 s | 116.0 MB | 150,994,944 B (144 MiB) | `-m 146M`, ran in 25.0 s |
| 1008.5 MB | 11.4 s | 32.8 s | 1008.1 MB | 1,090,519,040 B (1040 MiB) | `-m 1042M`, ran in 232.7 s |

Both booted OpenSBI v1.5.1 and finished with `=== Latency Test Finished ===`. The 116 MB
model printed `RUN_CYCLES: 244065619` → **2440.65619 ms**, against a scaled timeout of
172 s; the 1 GB model printed `RUN_CYCLES: 2381738678` → **23817.38678 ms**, against
1248 s. Both finished with room to spare.

These two are size tests, not accuracy tests. Each is a stack of `MatMul → Add → ReLU`
with weights from N(0, 0.02), so the signal decays with depth and the host reference
itself lands around 1e-11 — the target's logits print as `0.000000` because that is the
value, not because anything mis-compiled. Numerical parity is measured on the fixtures in
`models/`.

**External data works, and produces the same bytes.** Models at or above 1 GB are stored
as `model.onnx` plus a sidecar `model.onnx_data`, because protobuf cannot encode a single
message larger than 2 GB. Compiling the split form and the single-file form of the same
model produced byte-identical `weights.bin`.

GUI paths at 1 GB: `inspect_model` 7.3 s, `validate_model_file` 3.1 s, `analyze_model`
10.3 s — all returned, none raised.

The measurements from Beta 2.0 on `models/model.onnx` and `models/model_pretrained.onnx`
are unchanged; see `RELEASE_NOTES_v2.0.0-beta.1.md` for those tables and for the full note
on what emulator cycles are and are not.

## Please read: what these numbers are, and are not

**They are emulator cycles, not silicon.** Measurement is QEMU system-mode under
`-icount shift=0`, converted at a **nominal** 100 MHz. Use them to compare two builds of
the same model against each other. Do not quote them as performance on real hardware.

Because `-icount shift=0` makes execution deterministic, every timed sample is identical.
Mean, median and P95 are the same number by construction. That is a property of the
measurement setup, not a claim of zero variance.

**INT8 remains an accuracy study, not a speed or size optimization.** The `quantize` pass
is fake-quantization: it round-trips values through INT8 while the matmuls stay FP32, so
it is measured slower than FP32 and makes the binary larger. The memory-limit diagnostics
now say this where they mention quantization, so nobody reads it as a way to fit a model
that does not fit.

## Known limitations

- **The emulator stops accepting images at 1020 MiB**, which is the real size ceiling —
  lower than the linker's ~2 GiB relocation reach and lower than protobuf's 2 GB. QEMU's
  `virt` board pins its device tree at `0xBFE00000` once RAM base plus size passes 3 GiB,
  and the image loads at `0x80200000`, so the gap between them is all an image ever gets.
  Measured: the device tree stayed at that address for every `-m` from 1100M to 8192M, a
  1008.1 MiB image ran, and a 1036 MiB one was refused at load. Raising `-m` does not move
  it. A model past this point now fails with that explanation instead of a QEMU stack dump.
- **Soft-float targets are much slower, and the tables above are RV64GC.** `RV32IMC`,
  `RV32IMAC` and `RV32EMC` have no F/D extension, so every FP32 multiply in a transformer
  becomes a soft-float library call. Measured on `all_minilm_l6_v2.onnx` with the same
  128 MiB region and run count: RV64GC 2,835,467,134 cycles per inference in 220.5 s of
  wall clock, RV32IMC 58,085,441,820 cycles in 799.4 s — **20.5x the cycles, 3.6x the wall
  clock**. The QEMU timeout now scales by target, so such a run completes rather than being
  killed at a ceiling sized for an FPU; the run is still slow, and the diagnosis says which
  targets are not. `optimizer.py` calls
  `ort.InferenceSession(model.SerializeToString())`, which re-serializes the whole graph
  in memory and cannot exceed protobuf's 2 GB message ceiling. A failure there returns a
  fallback scale that is *labelled* as a fallback, so it degrades honestly — but it is the
  one path that did not benefit from the changes above.
- **A model whose weights alone exceed the target's RAM cannot be linked.** There is no
  weight-streaming backend. The diagnostics say so rather than suggesting a knob.
- **`model_medium` fails numerical parity**, MSE ≈ 0.43 against a 0.05 tolerance. Open,
  unresolved, and reported as a failure rather than hidden.
- Optimization is scalar. RVV-first compilation and auto-tiling are not in this build.
- Windows only.

## Verified

487 tests passing, lint clean across `src/`, `tests/`, `tools/`, `installer/` and both
build scripts.

42 of those tests are new, and they cover the machinery this release is named for, which
had none: the RAM-region, QEMU-memory and timeout arithmetic; that weights really are
emitted as a blob with in-range, 16-aligned `.incbin` offsets and no initializer list left
in `weights.h`; that a region overflow provoked from a real `ld` run surfaces as a
memory-limit failure with a byte count; that the external-data form of a model compiles to
byte-identical weights; that calibration hitting protobuf's 2 GB ceiling returns a scale
labelled as a fallback; and that the version, the artifact filenames, the wizard's
literals and the guide's cover all agree with the package.

The last 17 cover the two limits above: that the 1020 MiB window is the gap between the
load address and the device tree and agrees with the images that were actually accepted
and refused; that hardware-float detection is right for all six shipped targets; that a
soft-float target gets a ceiling longer than the run that was measured under it, while a
caller that passes no target keeps the rate it always had; that PT_LOAD spans parse from
both ELF32 and ELF64 and that an unparsable file yields no answer rather than a wrong one;
that an oversized image is refused before QEMU is started and one that exactly fits is
not; that QEMU's real overlap message is still classified if the pre-check misses; and
that both new failures produce a diagnosis naming the actual cause rather than a stack
dump.

The soft-float fix was also checked end to end, unpatched: `all_minilm_l6_v2.onnx` on
RV32IMC — the case that used to be killed at 537 s and reported as an unexpected
compilation failure — now computes a 1971 s ceiling for itself and finishes in 806.8 s,
returning 580854.4182 ms, bit-identical to the same model timed with the ceiling removed.

Every ONNX file on this machine — 14 of them, 0.0 MB to 86.2 MB, including the two
fixtures that exist to fail — was then swept through the whole pipeline on RV64GC. Twelve
compiled and ran. The two that did not were named correctly: `CompilationError` at the
`tvm-lowering` step for `model_repairable.onnx`, `UnsupportedOperatorError` for
`model_unsupported.onnx`. Nothing fell through to the generic "Unexpected Compilation
Failure" text — `ran=12  diagnosed=2  UNEXPLAINED=0`.

The packaged `TATVA.exe` was launched from `dist/TATVA/`: it opens its main window and
exits cleanly with nothing on stderr. The frozen bytecode inside it was also read back out
of its embedded archive and checked symbol by symbol, so the download and the source are
known to agree: both new exception types, the ELF and hardware-float helpers, the image
ceiling, and the measured soft-float constant `2.2` are all in the code the installed app
imports.

The artifacts were checked independently of what the build printed: the zip opens with
5,324 entries under a single `TATVA/` root and `testzip()` clean; the stage guide sits at
the app root rather than under `_internal/`; the payload appended to
`TATVA-Setup-2.1.exe` hashes byte-for-byte identical to `TATVA-2.1-windows.zip`; and the
version compiled into `TATVA.exe`'s frozen archive reads `2.1.0` / `2.1`, which is the
check the previous release's tooling could not make.

These artifacts are **build-verified**: the payload is complete and the bundled toolchain
compiles all six targets — `RV32IMC`, `RV32IMAC`, `RV64GC`, `RV64IMAFDC`, `RV64GCV`,
`RV32EMC` — with QEMU 9.2.4, and the app itself starts from the unpacked folder. What has
*not* been done is running `TATVA-Setup-2.1.exe` through its own wizard on a clean
machine: the appended payload is verified by hash and its zip is verified to open through
the same `PayloadSlice` window the installer uses, but the install, the Start Menu
shortcut and the uninstaller have not been exercised end to end.
