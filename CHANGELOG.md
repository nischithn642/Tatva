# TATVA Release Changelog

All notable changes to the TATVA project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
