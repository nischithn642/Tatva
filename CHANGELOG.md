# TATVA Release Changelog

All notable changes to the TATVA project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.1] - 2026-08-16 (`2.1.1`)

The operator list was written by hand, and it was wrong in both directions. This release
replaces it with one that is measured: no operator name is claimed as supported without an
ONNX model in the test corpus that compiles for RV64GC, boots under QEMU and matches ONNX
Runtime to 1e-4. Along the way it fixes the failure the project was most exposed to — a
model that compiled, ran, reported a cycle count and returned zeros.

A patch release rather than a rebuild of 2.1: `v2.1.0` is published with two assets built
from commit `4904396`, and their SHA-256 sums are printed in its release notes. Replacing
those bytes in place would break every checksum already handed out.

### Fixed
- **A model whose operator has no kernel is refused instead of measured.** The harness
  emitter wrote C only for bindings that legalized into a `call_tir` and skipped every
  other one in silence. `relax.cumsum` survives `LegalizeOps`, so a `CumSum` model
  compiled cleanly, booted, printed `RUN_CYCLES` and returned a tensor of `0.0` — the
  graph's own output, never written by anything. The emitter now distinguishes a binding
  nothing reads, which stays skippable, from one something reads and nothing writes, which
  raises a `CompilationError` naming the operator. This is the same class of bug that made
  LayerNorm and Gather return zeros, still live in a second place.
- **`Shape` no longer leaves an unwritten buffer behind.** `Shape` becomes the pair
  `shape_of` / `call_pure_packed("relax.run.shape_to_tensor")`, neither of which lowers to
  a kernel — but the second is a real int64 tensor that a downstream `take` dereferences.
  In `models/model.onnx` that is `Gather(Shape(x), 0)` reading zeros, harmless only by
  accident. The extents are constants after shape inference, so they are now emitted as
  stores, which is what the capability table always claimed for `shape_to_tensor`.
- **Eleven operators were reported as unsupported by a backend that compiles them.**
  `Conv` at one and three spatial dimensions, `ConvTranspose`, `AveragePool`, `PRelu`,
  `LogSoftmax`, `Resize`, `InstanceNormalization`, `Hardmax`, `Trilu`, and the `variance`
  and `argmax`/`one_hot` those last two decompose to. Each legalizes into a real C loop
  nest — conv1d six nested loops, conv2d nine, conv3d twelve. Three of them carried a
  written explanation of why they could never work; those explanations described nothing
  that was true. `SUPPORTED_OPS` is now 66 names.
- **`Pad`, `Tile` and `Einsum` were reported under the name `call_tir`.** The ONNX frontend
  lowers them straight to a TIR PrimFunc with no operator node, and the analysis reported
  TVM's calling convention rather than anything in the user's file — a name that cannot be
  looked up, removed or replaced. The callee's name is reported now.
- **One operator is still refused, and it is now refused for a measured reason.** TVM has
  no lowering rule for `cumsum`; the relax op is still standing after legalization with no
  PrimFunc generated, which is what "no kernel" actually means. That test — survives
  legalization — replaced the guesswork that had put three convolutions on the same list.

### Added
- **An ONNX corpus of 88 graphs, 80 of which run end to end.** `tests/onnx_corpus.py`
  builds them from `onnx.helper` at collection time: 36 single-operator graphs, several
  whole models, and deliberately broken files. Each runnable one is compiled for RV64GC,
  run under QEMU and compared against ONNX Runtime at 1e-4. Twenty of the builders are new
  in this release, one for every operator claim above.
- **The capability tables are pinned to each other by tests.** The lowering table is
  exactly `SUPPORTED_OPS`; the repair rules and the "cannot be fixed" reasons are disjoint
  from it; every name in it is a real relax operator; and everything the corpus runs is in
  it. A claim shown in the UI can no longer drift from what the backend does.

## [2.1] - 2026-08-15 (`2.1.0`)

Beta 2.0 compiled the models in `models/` and nothing much larger. This release is
about the size of model TATVA will accept: 2.0 fell over somewhere below 100 MB, and
2.1 compiles and runs a 1 GB ONNX end to end. That is near the ceiling rather than a
round number: the linked image must fit below QEMU's device tree at `0xBFE00000`, leaving
1020 MiB, and the 1 GB model clears it by 11.9 MiB. See *Known limitations*. The version
is `2.1.0` where PEP 440 applies and **2.1** everywhere a person reads it — for 2.0 those
two spellings differed (`2.0.0b1` vs "Beta 2.0"); they agree now.

### Added
- **Weights are emitted as a binary blob and pulled in with `.incbin`, not as a C
  initializer list.** This is the change that makes large models compilable at all. A
  decimal initializer costs about 4.1 source bytes per model byte, so a 116 MB model
  became a ~480 MB `weights.h` and drove `cc1` past 3 GB of RSS before it was killed —
  the failure was in the C front end, not in anything about RISC-V. `runner` now writes
  `weights.bin` verbatim, a `weights.S` of `.incbin "weights.bin", <start>, <len>`
  directives, and a `weights.h` of incomplete array declarations. The assembler copies
  bytes; nothing parses them.
- **The linker's RAM region is sized to the build instead of being a fixed 128 MB.**
  `_ram_region_bytes()` adds up the weights, the planned activation offsets, the
  workspace pool, model I/O and the stack, adds 16 MiB of slack, rounds up to a 16 MiB
  granule, and floors at 128 MiB. `render_link_ld()` takes that number.
- **QEMU is given memory and a timeout that scale with the model.** `-m` is the RAM
  region plus the 2 MiB reserved for firmware; the timeout is 0.6 s per MiB per run
  with a 30 s floor. A 1 GB model previously died on the old fixed timeout even when
  the ELF was correct.
- **A linker region overflow is reported as a memory limit, not a generic build
  failure.** `region 'RAM' overflowed by N bytes` from `ld` is now raised as
  `MemoryLimitExceededError` and carries the byte count into the diagnostics text.
- **The QEMU timeout scales by target, not only by size.** A target without an F/D
  extension runs FP32 through soft-float calls and needs several times the wall clock for
  the same model; it was being judged against a ceiling measured on RV64GC and killed.
  `_has_hardware_float()` reads the ISA string and `_qemu_timeout_seconds()` takes the
  variant, with the soft-float rate derived from a measured 3.625x ratio. The parameter is
  optional, so callers that do not pass it keep the rate they were written against.
- **An image too large for the emulator is refused before QEMU starts, and says so.**
  `_elf_load_span()` reads the ELF's PT_LOAD program headers — `p_memsz`, not the file
  size, since `.bss` occupies addresses without occupying bytes — and a span reaching
  `QEMU_FDT_BASE` raises the new `EmulatorImageLimitError` with the limit, the span and the
  device tree address. If the pre-check cannot tell (unparsable ELF, a QEMU that places the
  FDT elsewhere), the emulator's own "ROM regions are overlapping" output is parsed as a
  backstop and classified the same way.
- **A console dock in the studio window** — Messages, Log and Reports tabs with per-tab
  counts, collapsible, clearable. The stage log previously had nowhere persistent to go.
  The sidebar section headers became real collapsible buttons with `aria-expanded`.

### Changed
- **The memory-limit explanation says what will actually help.** It named three
  mitigations, one of which ("increase the memory pool allocation boundaries") no longer
  describes anything the user can do now that the pool is computed. It now states that
  the sequence axis is bound to 32 and that the activation pool — not the weights —
  scales with it, and it repeats that this build's INT8 pass is fake-quantization that
  shrinks nothing and measured slower than FP32, rather than offering quantization as a
  size fix without that caveat.
- **`validate_model_file()` hashes in 4 MiB chunks.** It read the whole file into memory
  purely to fingerprint it, which on a 1 GB ONNX is a 1 GB spike inside the process that
  is also about to hold the parsed graph and TVM's copy of the weights.
- **`build_exe.py` stages `README.txt` and the stage guide on the folder path, not only
  the zip path.** PyInstaller's COLLECT deletes `dist/TATVA` outright on every run, so a
  `--no-zip` build handed back a folder missing both.

### Fixed
- **The installer no longer hardcodes its own filename or its version.** `OUTPUT` was the
  literal `TATVA-Setup-beta-2.0.exe`, so bumping the version produced an installer named
  after the previous release. It now derives from `DISPLAY_VERSION` in
  `src/tatva/__init__.py`, the same source `build_exe.py` uses for the zip.
- **The installer picked the wrong payload.** `find_payload()` took the alphabetically
  last `TATVA-*-windows.zip` in `dist/`. `dist/` keeps previous releases, and
  `TATVA-2.1-windows.zip` sorts *before* `TATVA-beta-2.0-windows.zip` — so a 2.1 build
  would have wrapped the old 2.0 payload in an installer named 2.1, which is the exact
  failure that prompted this release. It now asks for this version's zip by name and
  says which others it found if that one is missing.
- **The stage guide's cover is stamped from the package instead of typed in.**
  `docs/stage-guide.html` had `TATVA · Beta 2.0 · build 2.0.0b1` written into it, and the
  PDF built from it ships in the zip — so the first 2.1 zip carried a guide whose cover
  named the previous release. `tools/make_stage_guide.py` now rewrites that line from
  `DISPLAY_VERSION`/`__version__` before rendering, and fails if the line it expects is
  not there.
- **A timed-out or oversized simulation is no longer reported as "Unexpected Compilation
  Failure".** Both cases were wrapped by a blanket `except Exception` in
  `run_and_measure()` and re-raised as `RuntimeError`, so a run that was merely slow, and a
  model that was merely too big for the emulator, both reached the console as a QEMU
  command line under a heading saying the compile had failed. Nothing had failed to
  compile in either case. They are now `SimulationTimeoutError` and
  `EmulatorImageLimitError`, each with its own offline explanation naming the cause and a
  mitigation that is true — for the timeout, that an FP32 model wants a target with an FPU;
  for the size, that raising QEMU's `-m` does nothing because the device tree does not
  move.
- **The wizard's version literals are checked against the package before it is built.**
  `installer/tatva_setup.py` cannot import `tatva` — it is a 15 MB Tkinter stub frozen
  without numpy, onnx and TVM — so `APP_VERSION` and `APP_BUILD` have to be copies.
  `check_stub_version()` now compares them and refuses to build on a mismatch, which is
  what keeps the copies from drifting the way they already had.

### Verified

Both ends of the requested range were compiled and run **on RV64GC**, not estimated.
Timings are wall clock on this machine (15.2 GB RAM); the latency figure is QEMU system
mode with `-icount shift=0` at a nominal 100 MHz, which makes every sample bit-identical
by construction. Not silicon. A soft-float target is far slower for the same model — see
*Known limitations*.

| Model | Import | Compile | ELF | RAM region | QEMU |
|---|---|---|---|---|---|
| 116.1 MB | 6.0 s | 3.9 s | 116.0 MB | 150,994,944 B (144 MiB) | `-m 146M`, ran in 25.0 s |
| 1008.5 MB | 11.4 s | 32.8 s | 1008.1 MB | 1,090,519,040 B (1040 MiB) | `-m 1042M`, ran in 232.7 s |

Both booted OpenSBI v1.5.1 and finished with `=== Latency Test Finished ===`. The 116 MB
model printed `RUN_CYCLES: 244065619` → **2440.65619 ms** against a scaled timeout of
172 s; the 1 GB model printed `RUN_CYCLES: 2381738678` → **23817.38678 ms** against 1248 s.
Mean = median = p95 in both cases, which is what `-icount shift=0` guarantees.

Neither figure is a parity check. Both fixtures are stacks of `MatMul → Add → ReLU` with
weights drawn from N(0, 0.02), so the signal decays layer over layer: at 29 layers the
host reference itself comes out around 1e-11, and the target's `FIRST_LOGITS` print as
`0.000000` because that is what the model computes, not because the compile is wrong.
Numerical parity is covered by the fixtures in `models/`, which have output worth
comparing.

**The external-data form works, and produces the same bytes.** Models at or above 1 GB
are normally stored as `model.onnx` plus a sidecar `model.onnx_data`, because protobuf
cannot encode a single message larger than 2 GB. Compiling the split form and the
single-file form of the same model produced byte-identical `weights.bin` (`cmp`, no
differences), so this is real support for the shape these models actually ship in.

GUI paths at 1 GB: `inspect_model` 7.3 s, `validate_model_file` 3.1 s, `analyze_model`
10.3 s — all returned, none raised. `calibrate_activation_scale` on the 116 MB model took
1.3 s and returned a calibrated scale, not the labelled fallback.

487 tests pass and lint is clean. 42 of those are new, and they exist because none of the
above had any regression coverage at all: a change restoring the fixed 128 MiB region or
the decimal weight emission would have left the old suite entirely green. They cover the
region/memory/timeout arithmetic against the byte counts measured here, the `.incbin`
emission (offsets 16-aligned and inside the blob, no initializer list left in
`weights.h`), a region overflow provoked from a real `ld` invocation so the parse is
tested against what the linker prints, the external-data form compiling to identical
weights, calibration's fallback when protobuf's 2 GB ceiling is hit, and agreement
between the package version and every place that copies it. The last 17 cover the
emulator ceiling and the target-scaled timeout: the 1020 MiB window against the images
that were actually accepted and refused, hardware-float detection across all six targets,
PT_LOAD parsing for ELF32 and ELF64 with no answer rather than a wrong one on a bad file,
the pre-check refusing an oversized image before QEMU is started, QEMU's real overlap
message as a backstop, and the diagnosis text for both new failures.

The soft-float path was re-run end to end with nothing patched: `all_minilm_l6_v2.onnx` on
RV32IMC, the case that was being killed at 537 s, now sizes its own ceiling at 1971 s and
finishes in 806.8 s with `mean_ms = 580854.4182` — bit-identical to the figure measured
with the ceiling lifted, which is what `-icount shift=0` should give.

Every ONNX file on this machine was then put through the whole pipeline on RV64GC as a
sweep — 14 models from 0.0 MB to 86.2 MB, including both deliberate failure fixtures.
Twelve compiled and ran; the two that did not are `model_repairable.onnx` and
`model_unsupported.onnx`, which exist to fail, and both were named correctly
(`CompilationError` at the `tvm-lowering` step, `UnsupportedOperatorError` for
`UnsupportedOpXYZ`). Nothing landed in the generic "Unexpected Compilation Failure"
bucket: `ran=12  diagnosed=2  UNEXPLAINED=0`. `all_minilm_l6_v2.onnx` returned
28354.6713 ms, the same value as the earlier standalone run.

The packaged `TATVA.exe` was launched from `dist/TATVA/`: it opens its main window and
exits cleanly with nothing on stderr. Because the fix changes shipped source, the
artifacts were rebuilt and the frozen bytecode inside `TATVA.exe` was read back out of its
embedded archive and checked directly — both new exception types, all five new runner
symbols, and the measured soft-float constant `2.2` are present in the code the installed
app will import. The installer's wizard has not been run end to end on a clean machine;
its payload is verified by hash and opened through the same `PayloadSlice` window the
installer itself uses.

### Known limitations
- **1020 MiB is the largest image QEMU will load**, and it is the first ceiling reached —
  below the linker's ~2 GiB relocation reach and below protobuf's 2 GB. The `virt` board
  pins its device tree at `0xBFE00000` once RAM base plus size passes 3 GiB; the image
  loads at `0x80200000`; the gap is `0x3FC00000` = 1,069,547,520 bytes. Measured from both
  sides — a 1008.1 MiB image ran, a 1036 MiB image was refused — and the device tree stayed
  at that address for `-m` 1100M, 1600M, 2500M, 4096M and 8192M, so raising `-m` does not
  help. `run_and_measure` now checks the ELF's PT_LOAD span before starting QEMU and raises
  `EmulatorImageLimitError`, so this reads as a size limit rather than an unexpected
  compilation failure.
- **A soft-float target is 20.5x the guest cycles for an FP32 model**, and the published
  benchmarks are all RV64GC. `RV32IMC`, `RV32IMAC` and `RV32EMC` have no F/D extension, so
  every float multiply becomes a library call. Measured on `all_minilm_l6_v2.onnx`, same
  128 MiB region, same run count: RV64GC 2,835,467,134 cycles per inference and 220.5 s of
  wall clock; RV32IMC 58,085,441,820 cycles and 799.4 s. The timeout now scales by target
  (`QEMU_SOFT_FLOAT_SECONDS_PER_MIB_PER_RUN`), so these runs finish instead of being killed
  — but if you want a transformer to be fast, the fix is an FPU, not a longer ceiling.
- `optimizer.py` calls `ort.InferenceSession(model.SerializeToString())`, which
  re-serializes the whole graph in memory and cannot exceed protobuf's 2 GB message
  ceiling. It is already wrapped so that a failure returns a fallback scale *labelled as
  a fallback*, so it degrades honestly rather than silently — but calibration is the one
  path that does not benefit from the streaming changes above.
- A model whose weights alone exceed the target board's RAM still cannot be linked.
  There is no weight-streaming backend, and the diagnostics now say so instead of
  suggesting a pool size to raise.

---

## [Beta 2.0] - 2026-08-08 (`2.0.0b1`)

Everything below came out of driving a real five-stage run in the packaged
`TATVA.exe` and fixing what that run actually showed. The version is `2.0.0b1` where
PEP 440 applies — wheel names, `pip`, hatchling — and **Beta 2.0** everywhere a person
reads it: the badge, the zip filename, this file.

### Added
- **The RISC-V toolchain now ships inside the zip, so all five stages work offline.**
  Stage 05 is the only stage that produces a measurement and it was the only one that did
  not work on a freshly unzipped folder — it shells out to a cross-compiler and an
  emulator the recipient had to go and fetch first. `build_exe.py` now copies both into
  `toolchain/` beside `TATVA.exe`, and `runner.bundled_tools_dir()` resolves that path
  from `sys.executable` so the folder works wherever it is unzipped. No download, no
  admin rights, nothing added to `PATH`, nothing written outside the folder. A
  `riscv-none-elf-gcc` or `qemu-system-riscv64` already on the user's `PATH` still wins,
  and a per-user `tatva setup` install still overrides the bundled copy, so nobody has to
  delete files out of the app folder to use their own build.
- **The bundle is pruned by asking the compiler, not by guessing.** The xPack toolchain
  carries 32 multilib variants; the six targets in `compiler.TARGETS` resolve to four of
  them, and the build reads that from `gcc -print-multi-directory` — the same lookup GCC
  performs when it links — rather than from a hardcoded list that could drift. Dropping
  the unused variants, the C++/Fortran compilers, gdb and its embedded CPython, and
  QEMU's firmware for machine types TATVA never boots (the edk2 UEFI images alone are
  ~290 MB) takes 2.1 GB down to 436 MB. `build_exe.py` then compiles all six targets out
  of the pruned copy and runs the bundled QEMU before zipping, so a prune that removed
  something needed fails the build rather than the recipient's first run.
- **Install the RISC-V toolchain from inside the app.** Diagnostics → *Install toolchain*
  downloads the same pinned xPack builds `tatva setup` uses, unpacks them into the
  per-user tools directory, and re-probes — with a real progress bar, byte counts and a
  log. This closes the one hole that made the zip not actually shareable: a recipient has
  an exe, not a checkout, so "run `tatva setup` in a terminal" was not a fix available to
  them. They got through stages 01–04 and dead-ended at *"RISC-V GCC cross-compiler binary
  not found"*. The installer machinery already existed in `tatva.toolchain`; it had no
  button.
- **Stage 05 preflight.** The build now checks for GCC and QEMU *before* starting, and
  says what is missing and where to get it, with a button that goes there. Previously the
  run failed several seconds in with the name of a binary the user had never heard of.
- **Natural-language configuration** (`tatva.nl_config`) — stage 04 now has a box that
  turns a stated priority into a build configuration: *"this has to fit in SRAM on a
  32-bit MCU"* selects RV32IMC and switches INT8 quantization on. It was listed under
  Experimental in the beta scope and existed nowhere in the build; grepping for it
  returned the deck and nothing else. It is a phrase matcher, not a language model, and
  the implementation is deliberately narrow: it runs offline, it is deterministic, it can
  only move the same switches the cards below it move, and it prints the phrase behind
  every decision so the user can overrule it before anything is built. It also says when
  it understood nothing, rather than returning defaults that look like an answer.
  Conflicting priorities — *"smallest and fastest"* — are reported as a conflict, because
  on a scalar RISC-V core those two ask for opposite passes.
- `toolchain.install_component(..., on_progress=)` — an optional `(read, total)` callback,
  so the GUI can show download progress instead of a spinner that sits still for 430 MB.
- `validate_model_file()` now returns `size_bytes` alongside `size_mb`.

### Fixed
- **Latency was rounded to 2 dp on both sides of the bridge**, which collapsed a real
  baseline-vs-optimized gap into `0.06 ms` against `0.06 ms` — two identical numbers and
  two identical-looking bars for a run that did measure a difference. The payload now
  carries 4 dp and the UI picks its precision from the magnitude.
- **The chart's Y axis was unreadable at sub-millisecond scale**, printing
  `0.1 / 0.1 / 0.0 / 0.0 / 0.0 ms` up the side: five labels, three the same, none the
  actual value. Tick precision is now derived from the tick step.
- **The delta bracket collided with the top gridline** and its label could be drawn off
  the top of the canvas — the one number the whole run exists to produce was the least
  legible thing on the chart. `padT` now reserves room for it and the label position is
  clamped.
- **A 0.00% change was painted green**, reading as a win. Zero is now neutral, in the
  terminal, on the report card, in the table and on the chart, and is labelled *no
  measurable change* rather than left to be read as a failure.
- **Model size showed "0 MB"** for every bundled sample, all of which are well under a
  megabyte. Sizes now render in B, KB or MB.
- **Diagnostics counted CMake and Make as missing components**, so a machine that could
  run all five stages reported *"2 component(s) missing"*. They are used by the Project
  Scaffolding page, not the compile pipeline; the table now marks each row Required or
  Optional and only the required ones drive the health pill.
- **"Use a bundled sample" silently loaded the first model in the list** and said nothing
  about the other three. The transformer fixture — the only bundled model with an attention
  pattern, and so the only one where the fusion pass has anything to fuse — was
  unreachable from the UI. Stage 01 now shows all four as cards with a size and a
  one-line note each. This is what made the pipeline look broken: the reachable model was
  an MLP, and it correctly reported no change.
- **Parity MSE printed as a raw double** — `0.000029431902811034336`, twenty-two
  characters of which four carry information, wide enough to push the verdict badge out
  of the table. It now reads `2.94e-5`, and the baseline row shows `—` rather than `0`,
  since the baseline is what parity is measured *against*.
- **The Diagnostics table wrapped every row to four lines** by putting each component's
  purpose in the narrow Required column. The purpose now sits under the component name.
- **The install card's SIZE column showed no size** for anything already installed. Size
  and install state are both shown.

- **Diagnostics could report a toolchain that stage 05 then refused to use.** Two
  separate resolvers had drifted apart: the health check accepted `riscv64-linux-gnu-gcc`
  and the user-mode `qemu-riscv64`, while the build only ever looks for a bare-metal
  `riscv-none-elf-gcc` and `qemu-system-riscv64`. On a machine with the common
  `gcc-riscv64-linux-gnu` package installed — apt, WSL, MSYS2 — the badge went green, the
  stage 05 preflight stayed silent, and the run then died on `RISC-V GCC cross-compiler
  binary not found` with no explanation of what to install. `ToolchainManager` now
  delegates to the build's own resolver, so the two cannot disagree. Neither of the
  dropped binaries could have done the job: the build links `-ffreestanding -nostdlib -T
  link.ld` against its own `start.S`, and the measurement boots that ELF on `-machine
  virt` to read the cycle CSR.
- A bundled `riscv64-unknown-elf-gcc` was reported on the Diagnostics page as
  `riscv-none-elf-gcc`, under a column headed "resolved path".
- **Stage 05's preflight fails open** — deliberately, so a broken health check cannot
  block a machine that can actually build — but nothing caught the case where the run
  then died on a missing binary. A "binary not found" failure now prints the install
  instructions and raises the toolchain banner, so that message can never be the last
  word again regardless of how the preflight was bypassed.

- **Plain-English diagnostics reached the desktop app.** The beta scope calls this the key
  differentiator — *"plain-English cause and recommendation, not a raw compiler error"* —
  and the rule engine in `diagnostics.py` has always backed it for `tatva diagnose` and
  the legacy Tk front end. The web GUI, which is the app people actually launch, returned
  `str(exception)`: a failed run showed `Pipeline failed: Memory limit exceeded: 720896
  bytes` and stopped there. Stages 02, 03 and 05 now route every failure through
  `classify_failure` → `explain`, and stage 05 also diagnoses a run that completes but
  misses the parity tolerance, which is a result the user has to act on and which
  `FAIL [accuracy outside tolerance]` did not explain. Diagnosis is additive: the raw
  message is still printed above it, because that is what gets pasted into a search.

### Changed
- **The beta-scope card claimed two things this build does not do.** It listed *Renode
  validation* under Experimental — Renode appears in the README, this file and
  `.gitignore`, and in no code path — and *Natural-language config*, which did not exist.
  Natural-language config is now implemented (above) and stays Experimental. Renode moved
  to Roadmap. QEMU system-mode validation moved the other way, from Experimental to
  Available: it is the measurement path every number in the app comes from, so calling it
  experimental understated it. The card's subtitle now reads "The status of this build,
  not a plan", matching the beta scope's own "nothing below is aspirational".
- Stage 01 said PyTorch and TF/Keras files "are exported through ONNX first", which reads
  as though TATVA performs the export. It does not — `compiler.py` raises
  `ImportInProgressError` for both. The line now says to export to ONNX yourself first.
- Diagnostics offered "Install the RISC-V toolchain" on a build that already ships one,
  inviting a ~520 MB download for something sitting next to the exe. The card now appears
  only when the health check actually finds something missing, which is the case it was
  written for: a source checkout or a pip install that never ran `tatva setup`.
- Stage 03 said "Stage 05 can compile this model" whenever every operator had a lowering.
  Mapping proves operator coverage, not that a compiler exists, so it now reads "Nothing
  in this graph will block stage 05" and leaves the toolchain question to stage 05's
  preflight.
- The zip is `TATVA-beta-2.0-windows.zip`. `TATVA-2.0.0b1-windows.zip` makes the person
  receiving it decode a PEP 440 pre-release tag to find out what they have.
- `README.txt` in the zip now lists all five stages by name, says the toolchain is
  already in the folder, and states plainly that a 0.00% result is a real measurement —
  softmax fusion needs an attention pattern, which `model_mlp.onnx` does not have and
  `model.onnx` does.
- Diagnostics said a missing toolchain leaves "stages 01–03 working; 04 and 05 need it".
  Only stage 05 shells out, so 04 was sending people after an install they did not need.
  Both that line and stage 05's own guidance now name the `toolchain/` folder first,
  since on a packaged build a missing toolchain almost always means `TATVA.exe` was
  dragged out of its folder rather than that anything needs downloading.
- README's multi-model table listed model sizes that matched no file in the repository —
  `439.1 KB` appeared twice for two files that are both 17.3 KB. Sizes are now measured
  from `models/*.onnx`. Cycle counts are unchanged; those were the measured figures.

### Verified

A full 01 → 05 run in the packaged `TATVA.exe` on the bundled `model.onnx`
(RV64GC, softmax fusion): baseline **0.731 ms**, optimized **0.665 ms**,
**−8.99% latency**, parity MSE **2.94e-5**, status PASS, 5.7 s wall clock.
Measured under QEMU system mode, `-icount shift=0`, at a nominal 100 MHz. Not silicon.

The same packaged build on `model_mlp.onnx`: baseline **0.060 ms**, optimized
**0.060 ms**, **0.00%**, parity MSE **4.05e-13**, status PASS, 4.5 s. Fusion has no
attention pattern to rewrite in that graph, so no change is the correct result.

**The offline claim was tested by taking the toolchain away.** `TATVA.exe` was launched
with `TATVA_TOOLS_DIR` pointed at an empty directory and `PATH` cut to
`C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem` — no cross-compiler and no
emulator reachable by any route except the `toolchain/` folder in the zip. The sidebar
still read **Toolchain ready**, stage 05 showed no preflight warning, and a full run
completed: baseline **0.731 ms**, optimized **0.665 ms**, **−8.99%**, parity MSE
**2.94e-5**, PASS, 5.3 s. Same figures as the run above on a machine with everything
installed, which is the point.

The resolver fix was checked against the machine that provoked it: with a
`riscv64-linux-gnu-gcc` and a `qemu-riscv64` on PATH and no real toolchain installed,
stage 05 now stops at the red "The RISC-V toolchain is not installed" banner with an
**Install it now** button, instead of failing mid-build.

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
- **`TATVA.exe` — the app now launches as an executable**, built by `build_exe.py` from
  [tatva.spec](tatva.spec) and shipped as `dist/TATVA-<version>-windows.zip`. Unzip,
  double-click, done: no Python, no `pip`, no repository. A one-*folder* build, not
  one-file, because a one-file build unpacks TVM's native libraries into a temp directory
  on every launch. `build_exe.py` also writes a `README.txt` next to the exe for whoever
  receives the zip, and a startup-failure handler writes `~/.tatva/startup-error.log` and
  shows a dialog, since a windowed build has no console to print a traceback to.
- **Stage 03 — Map.** The promised pipeline is 01 input → 02 analyze → 03 **map** →
  04 optimize → 05 generate, but the map stage had no UI and no bridge method, which is
  why the numbering visibly skipped. `map_operators()` checks every operator in the graph
  against the selected target's capability and reports what is supported, what is not, and
  what that means for the build.
- **Splash animation using the wordmark**, in both front ends: the web UI fades in
  `logo-dark.png` over a progress bar while the backend loads, and the Tkinter fallback
  now displays the shipped PNG instead of hand-drawing five letterforms in the old navy.
- `list_sample_models()` — the bundled fixtures are offered in the app, so a first run has
  something to compile without hunting for an ONNX file.

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
- **The benchmark chart was never drawn.** `index.html` declared `latencyChartInstance`
  and nothing ever assigned it — there was no `new Chart(...)` call anywhere in the
  repository — so the canvas stayed empty after every run. The chart is now drawn on a 2D
  context from the measured result, theme-aware and DPI-scaled.
- **`studio.js` was silently overwriting the page's own script.** The same function names
  were defined twice, once inline in `index.html` and once in `studio.js`, which loaded
  last and won — including a model handler that never called the Python bridge and wrote
  to element IDs that did not exist. There is now one implementation.
- **The offline app was loading four remote hosts.** Tailwind, Lucide, Chart.js and Google
  Fonts all came from CDNs, so a tool that advertises "works offline" rendered unstyled
  and iconless without a network. All four are gone: hand-written CSS, an inline SVG
  sprite, the hand-drawn chart, and a system font stack.
- **Stage progress was tracked in three places** — sidebar, top rail, overview table —
  each with its own idea of which stages existed. All three now generate from one `STAGES`
  array and update through one function, which is what stopped stage 03 from going
  missing again.
- **The frozen build could not import TVM's backend modules**, failing on the first
  analyze with `No module named 'tvm.backend.cuda.operator.intrinsics'`. TVM imports those
  by name at runtime, so static analysis never saw them; the spec now collects all TVM
  submodules explicitly.
- The version badge read `vv0.3.0` — `BUILD_VERSION` already carries its own `v`.
- Placeholder performance figures no longer appear when the backend is absent. The UI
  displays a measured number or says it is not connected.

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
