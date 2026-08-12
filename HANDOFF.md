# TATVA — Handoff

Written 2026-08-12. Everything below is the state of the work as of that date, so a
new session can pick it up without re-deriving anything.

Read sections 1–3 before touching anything. Section 3 describes a broken build
artifact on disk. Section 6 is the list of traps that have already cost time once.

---

## 1. Where things are

| What | Path |
| :--- | :--- |
| **Source repository (work here)** | `C:\Users\nisch\OneDrive\Desktop\tatva` |
| Installed copy (do NOT edit) | `C:\Users\nisch\AppData\Local\Programs\TATVA` |
| Remote | `https://github.com/nischithn642/Tatva.git` |
| Branch | `beta-2.0` |

The installed copy is a build output dated **Aug 8**. It predates all recent work.
It has a flat layout — `TATVA.exe`, `_internal/`, `toolchain/` at the top level, with
**no `dist/` folder**. If a session's working directory is set to the installed copy,
paths like `dist/TATVA/TATVA.exe` resolve to nothing.

Always use the project venv: **`.venv/Scripts/python.exe`**. A bare `python` lacks
`onnx` and `tvm`.

---

## 2. What is done and pushed

Two commits are on `origin/beta-2.0`, working tree clean apart from section 3:

- `3ace98d` — Phase 2: per-kernel profiling, and remove the fabricated INT8 numbers
- `fb36f6b` — Ship TATVA-Stage-Guide.pdf from the spec, not from a hand copy

Verified at that point: **445 tests pass**, `ruff check src/ tests/` clean.

### Phase 2 result, in short

The phase brief asked to fix an INT8 regression stated as "FP32 161.23 ms, INT8
197.00 ms". **Neither number was ever a measurement.**

- 161.23 ms described a `models/model.onnx` of 439.1 KB. The file in this repo is
  17,719 bytes and measures **0.73090 ms**.
- 197.00 ms was a string literal in `cli.py`, not a benchmark.

Both were removed from shipping code. A profiler was added to answer the question
with real data. On `models/model.onnx` the INT8 regression is **+414,160 cycles**:
`quantize` 339,730 (**82.0%**), `dequantize` 74,130 (**17.9%**) — 99.9% of it. All 14
kernels shared with the FP32 build drifted **exactly 0 cycles**, `matmul` included at
427,700 in both. That zero proves the pass is fake-quantization (QDQ): the MatMul
never becomes integer, so the pass can only add work. INT8 also makes the ELF
**larger** (83,576 → 91,072 B), the opposite of the "40–72% smaller" previously
claimed.

Measured tables live in `OPTIMIZATION.md` §2. Do not restate any performance number
that is not in there or reproducible with the commands in §3 of that file.

### The profiler

`compile_model(profile=True)` in `src/tatva/runner.py` emits a `tatva_read_cycles()`
prologue and brackets each kernel call site in `model_run.c` with an rdcycle pair;
`main.c` prints `KERNEL_CYCLES:` lines that `run_and_measure` parses into
`MeasurementResult.kernel_profiles`. It is **off by default and emits byte-identical
sources when off**, so no reported latency is ever measured with instrumentation on.
Surface it with `tatva profile <model>`. Covered by `tests/test_profiling.py`.

---

## 3. BROKEN RIGHT NOW — read this first

### 3a. The distribution zip is corrupt

`dist/TATVA-beta-2.0-windows.zip` is **107,860,900 bytes**; it should be ~244.5 MB.
A `build_exe.py --zip-only` run was interrupted mid-write. Confirmed:

```bash
cd /c/Users/nisch/OneDrive/Desktop/tatva && .venv/Scripts/python.exe -c "import zipfile; zipfile.ZipFile('dist/TATVA-beta-2.0-windows.zip')"
```

fails with `BadZipFile: File is not a zip file`. **Do not ship it.**

`dist/TATVA-Setup-beta-2.0.exe` (266,862,847 B, 09:35) is **valid and installable** —
it was built from the good 244.5 MB zip before the truncation. Its only flaw is that
the stage guide sits in `_internal/` rather than the app root. It is a usable
fallback if a rebuild is not wanted.

`dist/TATVA/` (the frozen folder) is in the **correct final layout**: the guide is at
the root, and not in `_internal/`.

### 3b. Two files are modified and uncommitted

`tatva.spec` and `build_exe.py`. Both parse (`ast.parse` checked). They fix a real
bug, described below. Decide to keep or revert them before rebuilding.

**The bug:** `build_installer.py` verifies four things are in the payload and
reported `stage guide MISSING` where the previous release said `found`. The zip had
gone 5,324 → 5,323 entries. `TATVA-Stage-Guide.pdf` had been copied into `dist/TATVA`
**by hand** and was referenced by nothing, so a PyInstaller run rewrote that folder
and dropped it.

Commit `fb36f6b` fixed reproducibility by adding it to the spec's `datas`. That was
**half right**: PyInstaller 6 puts `datas` under `_internal/`, so a document meant for
a human to read landed among the DLLs. `build_installer.py` did not catch it because
its check matches on `endswith`, so any path passes.

The uncommitted edits correct that: removed from `tatva.spec` `datas`, and a new
`stage_root_docs()` in `build_exe.py` copies it into the app root next to
`README.txt`, called from `make_zip()` right after `write_readme()`.

### 3c. To finish the installer

The frozen folder is already correct, so a full re-freeze is not required:

```bash
cd /c/Users/nisch/OneDrive/Desktop/tatva && .venv/Scripts/python.exe build_exe.py --zip-only
```

then

```bash
cd /c/Users/nisch/OneDrive/Desktop/tatva && .venv/Scripts/python.exe build_installer.py
```

Expect `entries 5,324` and all four verify lines `found`. The zip step takes a few
minutes — **do not interrupt it**, that is what produced the corrupt file. If in any
doubt about `dist/TATVA`, run `build_exe.py` with no flags for a clean freeze (~9 min).

Then commit `tatva.spec` and `build_exe.py` and push.

**Nothing under `dist/` is committed** — it is build output. Only the two source files.

---

## 4. Remaining work

### Task #14 — reduce the `quantize` kernel cost (not started)

TVM lowers `relax.quantize` to a per-element `roundf()`, a non-inlined libm call on
this bare-metal target. That is why `quantize` costs ~4.6× `dequantize` on the same
tensors. Plan: a custom legalization emitting `lrintf()`, which compiles to a single
`fcvt.l.s` at `-O3`.

Two corrections already established — both were wrong in the first draft:

1. The gate must test **RV64 *and* hardware FP**. `fcvt.l.s` is RV64-only.
2. There are **two `LegalizeOps` call sites in `src/`**. Only the first may change;
   the second must stay stock.

**Known risk:** `roundf` is half-away-from-zero, `lrintf` under the default rounding
mode is half-to-even. This changes numerics on every quantized tensor. BERT-tiny is
at MSE 0.036593 against a 0.05 tolerance — not much headroom. It may need a bias
trick to restore exact semantics, which gives back part of the win.

**Ceiling:** even if `quantize` were free, INT8 stays slower than FP32, because the
MatMul remains FP32 under fake-quant. This makes a slow pass less slow. It does not
make INT8 a win — that needs real integer GEMM under RVV.

Estimate: 3–4 h, most of it QEMU validation. A ~45 min scoped version on
`models/model.onnx` alone would prove the approach before committing to the BERT-tiny
cycle.

### Phases 3–6 — gated

The user's protocol: **"Do not proceed to the next phase until I explicitly type
'PROCEED'."** Phases 1 and 2 are complete and approved. Do not start 3–6 unprompted.

| Phase | Scope | Estimate | Confidence |
| :--- | :--- | :--- | :--- |
| 3 | Reproducible benchmark framework; `tatva benchmark model.onnx --target riscv --precision int8`; pre/exec/post separation; record model version, target ISA, RVV availability, precision, passes, compile time, latency mean/median/P95/P99, memory; JSON/CSV out | 4–6 h | good |
| 4 | Operator resolution, "pre-fix → re-map → continue" loop, real softmax/attention fusion, actionable logs (`E001 - Unsupported Operator`, `E005 - Memory Constraint`) | 8–12 h | low |
| 5 | Hardware-aware compilation; capability detection (ISA, RVV, VLEN, cores); extensible auto-tiling; RVV as a primary target | 10–14 h | low |
| 6 | Clean Frontend → IR → Passes → Backends; CLI as primary driver, GUI calls those APIs and never duplicates compiler logic; exportable `tatva_config.json` | 5–7 h | medium |

**Phase 3 has an honesty constraint to settle first.** Under `-icount shift=0` every
timed sample is bit-identical, so mean == median == P95 == P99. Emit all four as the
spec asks, but label them deterministic-emulation figures. Printing four
different-looking numbers derived from one repeated value would be fabrication.

**Phase 5 has a hardware reality to respect.** A real integer GEMM measured **12–14%
slower** than FP32 on scalar `rv64gc`, because the core has a hardware FPU and a
single-instruction `fmadd.s`. Integer MatMul is therefore deliberately not enabled on
scalar targets. RVV is where that inverts, and validating it needs QEMU with
`v=true`, a different measurement setup from everything benchmarked so far. If RVV
auto-tiling cannot be validated properly, document the limit rather than ship an
unmeasured backend.

Before building anything in 3–6, check whether it already exists. RVV softmax does,
and some capability detection likely does. The standing instruction is to improve and
reuse rather than duplicate.

---

## 5. Guardrails — these override convenience

From the user, verbatim and binding:

1. **ZERO HARDCODING.** "Never hardcode an artificial performance improvement,
   latency metric, or time-saved percentage. All numbers must be derived from actual
   execution or strictly labeled as 'Estimated' with an exposed calculation
   methodology."
2. **PRESERVE THE MVP.** "Do not invent functionality the existing codebase does not
   support." / "DO NOT rebuild the project from scratch. DO NOT remove or break
   working features simply to redesign the architecture."
3. **NO FAKE BACKENDS.** "Do not create placeholder hardware backends that do not
   function. If a feature is partially supported, document it as such."
4. **FOCUS ON TRANSFORMERS** on constrained RISC-V hardware.

Also carried forward and still binding:

- Never label a blocked or partial compilation as successful. Tatva must not hide
  failures.
- No fake features: never ship a UI control whose backend does not exist. Show
  "Coming Soon" / "Experimental" / "Not available for this target" instead.
- Do not compute one number in the backend and a different one in the frontend.
- Do not let an emulated timing result look like a silicon measurement.
- Never perform an unsafe rewrite just to make compilation succeed; validate every
  automatic transformation.
- Do not sacrifice the working ONNX → RV64GC pipeline while adding features.
- Backend-first: prefer structured JSON/API responses over parsing terminal text.
- Keep the dark engineering-workstation UI, gold accent, green/amber/red semantics.
  Do not turn it into a generic SaaS dashboard.

`docs/ARCHITECTURE_ASSESSMENT.md` is the Phase 1 deliverable. Lines around 187, 214,
237 and 287 cite 197 / 161.23 / "40% to 72%" **correctly, as documentation of the
problem**. Do not "fix" those.

---

## 6. Traps that have already cost time

**Environment**

- The Bash tool **resets cwd after every call**. Begin every command with
  `cd /c/Users/nisch/OneDrive/Desktop/tatva && ...`. Shell variables must be
  `export`ed to reach Python subprocesses.
- **Piping a command masks its exit code.** `pytest -q 2>&1 | tail -25` reports
  `tail`'s status, not pytest's. Redirect to a file and echo `$?` instead. This
  produced a false "tests passed" once.
- The repo has **mixed line endings**. `optimizer.py` and `runner.py` are 100% CRLF;
  `compiler.py` is pure LF; `core.autocrlf = true`. The Edit tool matches LF anchors
  against CRLF files and preserves CRLF correctly — verified.

**CI and tests**

- `.github/workflows/ci.yml` runs `ruff check src/ tests/`, and the `unit-tests` job
  declares `needs: lint`. **A lint failure means zero tests run.** `scratch/` is not
  in CI's lint scope and has 11 pre-existing errors — leave them.
- ruff line-length 120, E501 ignored (`pyproject.toml:105`).
- `tests/test_cli.py` invokes `baseline-test` without `--out`, and `cli.py:547`
  defaults it to `BASELINE.md` — a **tracked** file. After a test run, check
  `git status --porcelain` and `git checkout -- BASELINE.md` if it appears. Never
  commit it.

**TVM / codegen**

- `tvm.relax.analysis.well_formed` returns `None` and **raises**. Always use
  `relax.analysis.check_well_formed()`.
- **`operators.c` is the only TVM-generated file.** `model_run.c`, `main.c`,
  `weights.h`, `model_info.h`, `start.S` and `link.ld` are all TATVA's own Python
  string emission. That is why `inject_optimized_softmax` regex-patches only
  `operators.c`, and why profiling is a clean codegen change rather than a text patch.
- `optimizer.py:218` still re-emits a **float32** matmul. This is fake-quant by
  design; the MatMul never becomes integer.
- TVM's **C** backend lowers `relax.quantize` to `roundf()` (half away from zero);
  TVM's **LLVM host** backend lowers the same `te.round` to half-to-even. They
  disagree. TVM's C codegen prints float literals with `%.6e`.

**Target / hardware**

- QEMU runs with `-icount shift=0`, so `RUN_CYCLES` is a retired-instruction count
  and every timed sample is identical. Nominal 100 MHz literal at `runner.py:1458`.
- **Nothing zeroes `.bss`** — `start.S` only sets `sp` and tails into `main`. Profile
  accumulators must be explicitly reset by `tatva_profile_reset()` after warm-up.
- `riscv_vector.h` is a 45-line `#pragma riscv intrinsic "vector"` stub. It **cannot
  be grepped**; intrinsic availability must be established by compiling.
- `__riscv_vfmerge_vfm_f32m1` takes **(vector, scalar, mask, vl)** — not mask-first.

**Build**

- Never hand-copy files into `dist/TATVA`. A PyInstaller run rewrites that folder and
  silently drops them. Stage from `build_exe.py` or the spec. This is exactly the bug
  in §3b.
- PyInstaller `datas` land under `_internal/`, not the app root.
- The `scipy` missing / `tvm.topi.testing` warning during the freeze is benign — a
  test-helper module, not on the runtime path. The previous shipping build had it too.

---

## 7. Reproducing the measurements

```bash
cd /c/Users/nisch/OneDrive/Desktop/tatva && .venv/Scripts/python.exe -m tatva.cli baseline-test models/model.onnx --target RV64GC --out /tmp/baseline.md
```

```bash
cd /c/Users/nisch/OneDrive/Desktop/tatva && .venv/Scripts/python.exe -m tatva.cli profile models/model.onnx --target RV64GC --json
```

Add `--passes quantize` to either for the INT8 side. Full suite, unpiped:

```bash
cd /c/Users/nisch/OneDrive/Desktop/tatva && .venv/Scripts/python.exe -m pytest -q > /tmp/pytest.txt 2>&1; echo "EXIT=$?"; tail -20 /tmp/pytest.txt
```

`models/model_pretrained.onnx` is 17.6 MB / 57 kernels and runs ~123M cycles under
QEMU — expect multi-minute waits per configuration.

---

## 8. One known unresolved defect

`model_medium` fails accuracy at **MSE ≈ 0.43 against a 0.05 tolerance**. It was
decided this must surface as a **failure** rather than be tolerated or quietly
widened. That work was not done. Do not raise the tolerance to make it pass.
