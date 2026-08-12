# TATVA — Architecture Assessment Document

**Phase 1 deliverable: System Audit & Architecture Mapping**
Commit audited: `b79a3a5` ("Turn the compiler into a system that shows its work"), branch `beta-2.0`
Repository: `C:\Users\nisch\OneDrive\Desktop\tatva`
Environment: Python 3.13.13, apache-tvm 0.25.0.post1, onnx 1.22.0, xPack riscv-none-elf-gcc 15.2.0, xPack QEMU 9.2.4
Method: read-only. 12,292 lines of `src/` read; 441 tests executed; three empirical probes run against real builds under real QEMU. No repository file was modified during this phase.

---

## 0. Executive summary

TATVA today is **an ONNX-to-bare-metal-RISC-V harness generator and build/emulate driver built on top of TVM**, not a compiler with its own optimization stack. It works end-to-end and is honest about most of what it does. But three claims that the product's own documentation and UI make are not supported by the code:

| Claim | Reality | Evidence |
|---|---|---|
| "INT8 quantization" | Fake-quant (QDQ). Two dequantized **float32** tensors are fed back into the original float32 matmul. Zero integer arithmetic is emitted. | `src/tatva/optimizer.py:218` |
| "Softmax fusion" | A regex + brace-match **textual substitution** of the C source TVM already emitted. The Relax module is returned unchanged. | `src/tatva/runner.py:753-800`, applied at `1291-1292` |
| "Optimization passes" | The entire TVM pipeline is `Sequential([LegalizeOps()])`. No tiling, no scheduling, no fusion, no constant folding, no DCE, no tuning. | `src/tatva/runner.py:927`; `src/tatva/compiler.py:300` is the file's last line: `# TODO: Implement TVM schedule optimization for RISC-V targets` |

The user-reported INT8 regression is **real, reproducible, and fully explained** — see §2.1. It is not a tuning problem; it is a design consequence of QDQ.

The audit also found one **correctness bug** that must be fixed before anything else ships: the RVV softmax kernel omits a numerical clamp the scalar kernel has, and produces an inverted probability distribution on a specific input band (§4.1).

---

## 1. CURRENT ARCHITECTURE

### 1.1 Where the line between TVM and TATVA actually falls

This distinction determines whether each planned phase is "write a TVM pass" or "write more Python string emission", so it is stated first.

**TVM owns:**
- ONNX ingest — `tvm.relax.frontend.onnx.from_onnx(model, shape_dict=...)` (`src/tatva/compiler.py:~216`)
- The single legalization pass — `relax.transform.LegalizeOps()` (`src/tatva/runner.py:927`)
- **All operator kernel C source**, retrieved wholesale via `lib.mod.imports[0].inspect_source()` after `relax.build(mod_legalized, target="c")` (`src/tatva/runner.py:931-932`)

**TATVA owns:**
- Python string emission of `weights.h` (`runner.py:1062`), the activation pool (`runner.py:1081`), DLTensor statics (`runner.py:1100-1130`), the workspace bump allocator (`runner.py:1149+`), `main.c` harness, `START_S` (`runner.py:114-121`), `LINK_LD` (`runner.py:122-164`)
- The gcc invocation (`runner.py:1361`) and the QEMU invocation (`runner.py:1435`)
- One textual post-hoc patch of TVM's emitted C to swap in the fused/RVV softmax (`runner.py:753-800`)
- Relax→Relax repair rewrites (`repair.py`) and the QDQ quantizer (`optimizer.py`)

Consequence: the only place TATVA generates machine-relevant code itself, it does by regex-replacing TVM's output — fragile across any TVM version bump.

### 1.2 Frontend

| Component | File | Notes |
|---|---|---|
| `import_model` | `compiler.py:181-250` | ONNX only. Content-hashed cache. `.pt/.pth/.pb/.h5/.savedmodel` raise `ImportInProgressError` (`compiler.py:188,192`) — honest refusal, not a fake claim. |
| `resolve_input_shapes` | `compiler.py:155-178` | Silently invents shapes. A dim is literal only if `HasField('dim_value') and dim_value > 0`; a declared `dim_value` of 0 falls through to 1. Symbolic dims become 32 **only if the name contains the substring `seq` or `length`**, otherwise 1. No provenance flag records which dims were guessed. |
| `frontends.describe` | `frontends.py:396-546` | Extracts opset, producer, precision, `parameter_bytes`. Reaches the GUI's model_info only — never any benchmark output. |
| `analyze_graph` | `compiler.py:253-297` | Visits `mod["main"]` only (`compiler.py:286`). Op histogram, unsupported list, `has_transformer_bottleneck` = `any softmax` AND `any matmul/dense` (`compiler.py:288-290`). |

### 1.3 IR

Relax IR (`tvm.relax`), single `IRModule`. `ModelIR` (`compiler.py:134-141`) is a three-field wrapper; `params` is always `None`. Graph rewrites use `relax.PyExprMutator` + `relax.BlockBuilder`, validated with `relax.analysis.check_well_formed()`.

**`SUPPORTED_OPS` (`compiler.py:92-129`) is advisory, not a gate.** 36 names. Seven of them are Relay-era spellings that are not registered TVM Relax operators and can therefore never match anything `analyze_graph` produces: `cast`, `concatenate`, `nn.batch_matmul`, `nn.bias_add`, `nn.dense`, `transpose`, `unsqueeze`. 19% of the declared support surface is unreachable. Meanwhile `analyze_graph` reports the real Relax spelling with the `relax.` prefix stripped (`compiler.py:272-283`), so a model using `concat` is reported unsupported while the dead alias `concatenate` sits in the set doing nothing.

### 1.4 Passes

**The complete pass inventory of the entire `tatva` package is one transform.**

```python
seq = tvm.transform.Sequential([relax.transform.LegalizeOps()])   # runner.py:927
mod_legalized = seq(model_ir.mod)
lib = relax.build(mod_legalized, target="c")                      # runner.py:931
```

`grep -rn 'tir\.|Schedule|meta_schedule|autotvm|tune|FuseOps|FuseTIR|transform\.'` across `src/tatva/*.py` (excluding gui.py) returns exactly two lines: `repair.py:551` and `runner.py:927` — both `LegalizeOps()`.

TATVA's own three "passes", as exposed to the user:

| Pass | What it actually is | File |
|---|---|---|
| `quantize` | Inserts Relax `quantize`→`dequantize` pairs around MatMul operands, then **re-emits the original float32 op on the dequantized floats**. | `optimizer.py:185-218` |
| `fuse` | Sets a metadata flag; later triggers a **textual** replacement of TVM's emitted softmax C with a Schraudolph fast-exp kernel (scalar) or an RVV intrinsics kernel (vector targets). | `optimizer.py:291-322`, `runner.py:753-800`, `runner.py:1291-1292` |
| `repair` | Genuine Relax→Relax rewrites: 13 rules, structural check + seeded numerical check, discards its own output on failure. **GUI-only** — no CLI command invokes it. | `repair.py:141-737`, called at `gui.py:1915, 2206, 2211` |

The only performance lever applied to generated code is `gcc -O2` — hardcoded at two independent sites, `runner.py:841` (toolchain hello-world) and `runner.py:1361` (model build).

### 1.5 Backends

There is **one** production backend and **one** undocumented second implementation.

**Primary path** (`runner.compile_model`, `runner.py:904-1401`):
`relax.build(target="c")` → TVM's C source → optional textual softmax swap → TATVA emits `weights.bin` (the raw blob), `weights.S` (a `.incbin` wrapper), `weights.h` (declarations only), `model_run.c`, `model_info.h`, `operators.c`, `start.S`, `link.ld`, `main.c` from module-level string constants → `riscv-none-elf-gcc -O2 -march=<...> -mabi=<...> -mcmodel=medany -ffreestanding -nostdlib` → `model.elf` → `qemu-system-riscv64 -M virt -m <sized to the build> -kernel -nographic -icount shift=0`. The `link.ld` RAM region and the QEMU `-m` value are both derived from the model's actual demand, and the subprocess timeout is scaled from that size rather than fixed — a flat 30 s could not have run anything larger than the small fixture.

**Second, unaudited path**: `src/scaffolding/executor.py:89-239` — `compile_workspace(workspace_dir, target='RV64GCV')` and `emulate_workspace(..., target='RV64GCV')`, reached from `gui.py:1491`. It defaults to the **experimental** RV64GCV target, does not go through `TargetVariant` validation, and has no test. Any backend refactor that only touches `runner.py` will silently leave this fork behind.

**Target registry** (`compiler.py:29-87`): RV32EMC, RV32IMAC, RV32IMC, RV64GC (default), RV64GCV (experimental), RV64IMAFDC. `TargetVariant.tvm_target` carries `-mattr=+v` for RV64GCV — **and is never read anywhere in `src/` or `tests/`**. `compile_model` hardcodes `target="c"`, the architecture-neutral C backend.

**Measurement**: `rdcycle`/`rdcycleh` around `tvmgen_default_run()` only (`runner.py:310-312`). Because QEMU runs with `-icount shift=0`, the counter returns **retired instructions, not cycles** (proved by exact 8× scaling at `shift=3`). The host divides by a hardcoded nominal 100 MHz (`runner.py:1458`) to produce milliseconds. `REAL_HW` is an explicit `NotImplementedError` (`runner.py:1413-1417`).

### 1.6 UI

**CLI** — nine commands (`cli.py`): `analyze` (:442), `baseline-test` (:511), `diagnose` (:881), `doctor` (:168), `gui` (:1028), `optimize` (:669), `setup` (:281), `targets` (:364), `validate` (:997). There is no `benchmark`.

**GUI** — `pywebview` (Edge WebView2) loading `website/index.html` from `file:///`, with `TatvaPyBridge` (28 methods) as the sole transport via `window.pywebview.api`; CSP `connect-src 'self'`. A legacy Tkinter `TatvaApp` (`gui.py:232-1292`) exists as fallback. Frozen entry point is `src/tatva/gui.py` (`tatva.spec:33`).

**The CLI is not the driver; it is a parallel path.** GUI and CLI share exactly three functions: `import_model`, `analyze_graph`, `compare_configs`. Eight subsystems are GUI-exclusive: `repair`, `artifacts.write_manifest` (`gui.py:2320`), `effort.write_effort` (`gui.py:2338`), `audit trail.write` (`gui.py:2348`), `validation.evaluate`, `runs.REGISTRY`, `capabilities` rendering, and the scaffolding agent. A CLI build therefore produces **no provenance record at all**.

### 1.7 Concurrency, memory, and everything else the map usually omits

**Concurrency model, complete**: five unjoined daemon threads in the GUI (`gui.py:141, 717, 1045, 1203, 1563`), three locks total (`gui.py:1351, 1367`; `runs.py:176`). Zero `multiprocessing`/`concurrent.futures`/`asyncio` anywhere. No parallelism in compilation. `_prune_old_builds` keeps only 8 build dirs under one shared root, so **two TATVA processes on the same machine delete each other's build trees mid-run** — this happened during the audit.

**Memory model**: three uncoordinated static allocations with no budget check and no reported footprint — a 128 MB RAM region with a 64 KB stack (`runner.py:122-164`), an activation pool sized from the graph (`runner.py:1081`), and a hardcoded 1 MB scratch pool (`runner.py:1149`) whose exhaustion path prints a message and returns NULL (`runner.py:1155-1157`). Nothing sums them; nothing compares them to a device.

**`src/scaffolding/` is 14% of the shipped Python** (1,727 lines, 7 modules) and is on the live GUI path via 12 import sites. `agent.py:208-436` embeds a complete PyTorch training/ONNX-export script as a fallback template, written to arbitrary user directories via `write_to_disk` (`agent.py:535`, called from `gui.py:1259`) — the only capability in the product that writes generated source to disk.

---

## 2. WHAT ALREADY WORKS

Everything in this section was verified by execution, not by reading.

### 2.1 The end-to-end pipeline is real
ONNX → Relax → TVM C → cross-compile → boot under QEMU → parse logits off the serial console → compare against a host onnxruntime reference. `test_e2e_pipeline_verification` (68.4 s) does exactly this on a BERT-shaped model and asserts MSE < 0.05. It passes.

### 2.2 The test suite is real
**441 tests, 441 passed, 0 skipped, 0 failed** (292.67 s, repeated at 194.87 s). All 21 integration tests genuinely executed — models were really cross-compiled and really emulated. `ruff check src/ tests/` passes clean.

### 2.3 The RVV softmax kernel is genuine
Disassembly of a fresh RV64GCV build shows `__tvm_ffi_softmax` containing `vfmv.v.f, vmv.s.x, vle32.v, vfredmax.vs, vfmv.f.s, vfsub.vf, vfmul.vf, vfadd.vf, vfcvt.x.f.v, vse32.v, vfredusum.vs` — 15 vector instructions matching `SOFTMAX_VECTOR` line for line. It is correctly gated: RV64GCV without `fuse` emits zero TATVA vector code; RV64GC with `fuse` emits the scalar kernel. It is VLEN-agnostic by construction (`__riscv_vsetvl_e32m1` at LMUL=1, `runner.py:712,721,736`).

### 2.4 The repair engine is genuine and self-checking
13 Relax→Relax rules, each validated by a structural check (`repair.py:472-529`) **and** a seeded numerical check that builds both modules on `target="llvm"` and compares at 1e-5 (`repair.py:532-609`). It **discards its own output** when validation fails — `test_repair` injects a deliberately wrong rule and asserts the engine rejects it. Measured live: `model_repairable.onnx` (LeakyRelu, Abs, Neg) → all 3 rewritten, structural passed, numerical passed, `max_abs_diff=0.0`, then built and measured successfully.

### 2.5 Failure is not hidden
`inject_optimized_softmax` raises `CompilationError` rather than silently reporting an "optimized" build identical to the baseline when it patches zero kernels (`runner.py:1293-1302`), and raises on unbalanced braces rather than emitting corrupt C (`runner.py:779-787`). `verify_target` cross-compiles and boots a hello-world before touching the model. `scan_hardware_boards` is a stub **and is tested for being honest about it** (`tests/test_pybridge.py:315-323`).

### 2.6 The packaged app genuinely carries a working toolchain
`build_exe.py` stages and prunes a real xPack gcc/QEMU into `dist/TATVA/toolchain/`; `prune_gcc` asks the real compiler `-print-multi-directory` for every entry in `TARGETS` rather than guessing. `verify_bundle` cross-compiles a probe for **all six targets** with the pruned bundled gcc and fails the build listing every failing target. `bundled_tools_dir()` derives the path from `sys.executable`, so the frozen app finds its own toolchain. Both bundled binaries were executed during the audit and work.

### 2.7 Determinism
`-icount shift=0` makes measurement bit-reproducible: 10 timed iterations returned the identical value (108.65177 ms ×10). Two builds of the same model produce the same number. This is a genuine asset for regression testing — it is only mislabeled (§4.3).

### 2.8 Effort accounting refuses to fabricate
`effort.py` computes engineering-time estimates from a documented, overridable rate table (`DEFAULT_RATES`, every line beginning "Assumption"), carries a `DISCLAIMER` that states "No engineer was timed to produce this figure", and has an explicit refusal gate (`effort.py:255-277`) that declines to emit a number when inputs are insufficient.

---

## 3. WHAT IS MISSING

Ordered by impact on the stated objective ("production-grade, hardware-aware AI compiler").

### 3.1 Integer arithmetic — the INT8 path does no integer math
`optimizer.py:218` returns `relax.Call(call.op, [dq0, dq1], ...)` — the **original float32 matmul** on **dequantized float32** operands. In the generated C every matmul signature is `float32 x float32 -> float32`; `grep -c "int32_t\*)matmul"` on the INT8 `operators.c` returns 0. There is no integer GEMM, no int32 accumulator, no requantization path.

### 3.2 Any scheduling layer
No tiling, no blocking, no loop reordering, no unrolling, no vectorization pass, no MetaSchedule, no dlight, no AutoTVM, no cost model — and not even TVM's own `FuseOps` or `FoldConstant`. Tiling is a **prerequisite** for RVV paying off, not an independent nice-to-have.

### 3.3 RVV for every operator except softmax
`matmul`, `nn.dense`, `nn.batch_matmul`, `nn.layer_norm` — the operators TATVA's own capability DB labels `KIND_HOT` and "usually the largest single share of the cycle count" — compile to scalar RISC-V. Disassembly of `__tvm_ffi_matmul` in an RV64GCV build: the inner loop is `flw / flw / fmadd.s / fsw`, one element at a time.

### 3.4 Any hardware capability detection
No ISA string parsing, no `misa`/`mvendorid` read, no `vlenb` CSR read, no cache probing, no core count, no `/proc/cpuinfo`, no device tree. `scan_hardware_boards` (`gui.py:1778-1780`) returns a hardcoded dict. VLEN appears exactly twice in the codebase, both the literal `vlen=128` in a QEMU argument list — written, never read back.

### 3.5 A `tatva benchmark` command, and the fields a benchmark needs
No such command exists. Against the Phase 3 requirement list, measured on `report.json` (573 bytes — the only machine-readable benchmark artifact any CLI command writes):

| Required field | Status |
|---|---|
| Model version / hash | **Absent** — report.json does not even name the model file. No model hash exists anywhere in the codebase. |
| Target ISA | **Partial** — the label `"RV64GC"` only. march/mabi and the gcc flags are never recorded. |
| RVV availability | **Absent** — decided at run time (`runner.py:1428-1429`) and discarded. |
| Precision | **Absent** — `ModelInfo.precision` is computed (`frontends.py:513-517`) and never reaches any output. |
| Passes applied | **Present** (`cli.py:802`). The one field properly recorded. |
| Compilation time | **Absent** — no timing instrumentation exists except `AuditEvent.elapsed_s`, GUI-only. |
| Latency mean/median/p95 | Present (`runner.py:1461-1463`). **p99 absent.** |
| Memory usage | **Absent** — zero hits for tracemalloc/psutil/getrusage/rss across `src/tatva`. |
| JSON / CSV | JSON partial (no raw samples, no run identity, no timestamp, no version). **CSV: zero hits for "csv" in `src/tatva`.** |

Also missing: preprocessing/execution/postprocessing separation (one number covers the whole `tvmgen_default_run()` call), per-operator attribution, and user-controllable warmup/iteration counts. The two CLI entry points silently disagree — `baseline-test` measures 3 warmup / 10 timed (`runner.py:908-909`), `optimize` measures 2/5 (`optimizer.py:369,382,396,417`) — and neither records the count, so two `report.json` files are not comparable and nothing says so.

### 3.6 A structured error-code taxonomy
`grep -rn "E0[0-9][0-9]\|error_code\|ERROR_CODE" src/` returns **nothing**. What exists is five exception classes collapsed by `classify_failure()` into five free-text strings, plus regex-over-the-message rescue rules. Worse, there are **three uncoordinated classification vocabularies**: `diagnostics.error_type`, `capabilities._UNFIXABLE_REASONS` (`capabilities.py:248-263`), and `repair.RepairResult.status` (`repair.py:112`).

Measured behaviour today:
- **Toolchain absence** (`FileNotFoundError`, `runner.py:1355`) and **QEMU failure** (`RuntimeError`, `runner.py:1443`) — the two most likely first-run failures — both land in the `unknown` bucket.
- **`MemoryLimitExceededError`** (`diagnostics.py:17`) has **no producer anywhere in `src/`**. Its only "coverage" is a test that mocks its raise site. `validation.py:69` already declares memory footprint NOT_IMPLEMENTED.
- The **GUI's BLOCKED path returns `"diagnosis": ""`** — the single most common failure the tool advertises never reaches `diagnostics.py` at all.
- `diagnostics.py:143` classifies any message containing `"not supported"` as an operator problem: `Exception("Feature X is not supported on this platform")` renders as "The operator 'the reported operator' is not supported by the RISC-V TVM bare-metal backend."

### 3.7 An iterating repair loop
The "pre-fix → re-map → continue" loop exists as **straight-line code in `gui.py:2197-2254` only**. No `while`, no fixpoint, no retry. `repair_graph` is called once; `_rewrite_module` is called once; the mutator returns `replacement` **without re-visiting it** (`repair.py:419-463`), so an op introduced by a rewrite is never itself repaired. The "re-map" is `_structural_check`'s `OpScan` inside repair — `record.mapping` is set once at `gui.py:2187` **before** repair and never recomputed, so the mapping table the UI shows is always pre-repair.

### 3.8 A real capability gate
`compile_model` never consults `SUPPORTED_OPS`. Measured: `models/model_blocked.onnx` (containing `exp`, declared unsupported) **compiles to a working RISC-V ELF**, and `tatva diagnose` on it prints "No errors detected. Model compiled successfully." with exit 0. The gate exists only in the pywebview bridge; `tatva optimize` reaches `compile_model` at `cli.py:984-985` with no check at all.

### 3.9 Test coverage where it matters most
Zero tests for the entire packaging layer (`build_exe.py`, `build_installer.py`, `tatva.spec`, `installer/tatva_setup.py`). Zero tests that assert on generated code — no test greps `operators.c` for `__riscv_v`, no test disassembles the ELF, no test calls `inject_optimized_softmax`. The RVV kernel could silently stop being injected and all three vector tests would still pass. `models/model_blocked.onnx` and `model_repairable.onnx` are referenced by zero tests. `anthropic` is imported in two places in `src/` and declared nowhere.

---

## 4. CURRENT LIMITATIONS

### 4.1 BLOCKING — the RVV softmax kernel is numerically wrong on a specific input band
`SOFTMAX_VECTOR` omits the `if (val < -15.0f) local_exp[k] = 0.0f;` clamp that `SOFTMAX_SCALAR` has (`runner.py:680-682` vs the vector loop at `720-730`). When `(x - row_max)` falls below roughly −88, the Schraudolph integer expression goes negative and reinterpreting it as float yields a large *negative* "exponential" that poisons the sum.

Verified by compiling both kernel bodies for `-march=rv64gcv` and running under the repo's own QEMU on the row `[0, -16, -30, -60, -90, -120, -150, -170]`:

| | index 0 | index 4 |
|---|---|---|
| Scalar (correct) | `0x3f800000` (1.0) | `0x00000000` (0.0) |
| Vector | `0x806e9661` (−1.0e−38) | `0x3f800001` (1.0000001) |

**The argmax moved from position 0 to position 4 and the distribution inverted sign.** (Very large negatives such as a −10000 attention mask happen to survive, because `vfcvt.x.f.v` saturates to `INT32_MIN` which reinterprets to −0.0f. The dangerous band is the intermediate one.) `check_softmax_fusable` (`optimizer.py:248-288`) validates dtype and reduction axis but **not dynamic range**, so nothing prevents this kernel from being selected.

Related: `vfcvt.x.f.v` uses round-to-nearest-even while the scalar kernel's C cast truncates toward zero, so RV64GC and RV64GCV builds of the same model are not guaranteed to agree bit-for-bit even in the benign range.

### 4.2 BLOCKING — the INT8 regression, root-caused
The specific numbers 161.23 / 197.00 ms **are not reproducible and correspond to no fixture in this repository.** 161.23 ms is the `models/model.onnx` row in `OPTIMIZATION.md`, which describes a 439.1 KB file; `models/model.onnx` has been 17,719 bytes since the only commit that ever touched it, and measures 0.7309 ms. **197.00 ms is not a measurement at all — it is a hardcoded string at `src/tatva/cli.py:742`.**

The regression itself is real and reproduces on all three fixtures:

| model | FP32 | INT8 | delta |
|---|---|---|---|
| `models/model.onnx` | 0.73090 ms (73,090) | 1.21995 ms (121,995) | **+66.91%** |
| `models/model_medium.onnx` | 2447.54 ms | 2876.44 ms | **+17.52%** |
| `models/model_pretrained.onnx` | 1188.06 ms | 1388.66 ms | **+16.88%** |

Instrumented per-kernel `rdcycle` attribution accounts for the entire delta to the millisecond:

- **medium**: measured delta +428.897 ms; quantize 352.68 + dequantize 76.24 = **428.92 ms**. The matmul bucket is **bit-identical** between FP32 and INT8 (1,672,129,200 cycles in both).
- **pretrained**: measured delta +200.593 ms; quantize 165.85 + dequantize 36.08 − 1.35 = **200.58 ms**.

Ranked mechanisms, all confirmed in the emitted binary:
1. **Constant weights are re-quantised and re-dequantised on every inference.** `optimizer.py:204-215` quantizes `arg1` even when it is a `relax.Constant`, and nothing hoists it. 76.7% (pretrained) / 82.2% (medium) of all elements quantized per inference are compile-time constants. **Hoisting alone recovers ~154.9 ms of pretrained's 200.6 ms regression.**
2. The quantize kernel costs 32.4 cyc/element vs dequantize's 7.0, because its inner loop does `fdiv.s` (divide by scale, not multiply by a precomputed reciprocal) plus a **non-inlined `jal roundf` libm call per element**.
3. Kernel invocations go 96 → 196 per inference; the activation pool grows 1,494,592 → 6,916,672 bytes (**4.63×**).
4. Values go fp32 → int8 → fp32 for every operand of every layer and never stay in int8.

Ruled out: calibration overhead is **not** in the measurement (it runs host-side inside `quantize()`, 0.18–0.24 s wall, while the timed window is only `rdcycle` around `tvmgen_default_run`).

**Side findings**: the INT8 build is **not smaller** — `weights.h` is byte-identical between FP32 and INT8 builds (still `static float constant_data_N[]`) and the ELF grows in all three cases. `OPTIMIZATION.md` §3's claim of "40% to 72%" size reduction does not hold for this code path. INT8 accuracy also fails on the larger fixtures: `model_medium` FP32 logits `[1.762, -0.600, -1.089, 0.461, -0.320]` vs INT8 `[0.599, -0.107, -1.224, -0.026, 0.233]` — MSE ≈ 0.43 against a 0.05 tolerance.

### 4.3 SIGNIFICANT — reported milliseconds are instruction counts divided by a fictional clock
`-icount shift=0` makes `rdcycle` return retired instructions, and the host divides by a bare literal `nominal_clock_hz = 100.0 * 1000.0 * 1000.0` (`runner.py:1458`). Mean, median and p95 are therefore three names for the same number and carry **zero dispersion information**. The UI presents them as three statistics.

### 4.4 SIGNIFICANT — RVV's measured benefit is within noise
Three builds of `model_pretrained.onnx`: RV64GCV+fuse = 118,587,502; RV64GC+fuse = 118,658,484; RV64GCV without fuse = 118,806,702. **RVV softmax buys 0.06%.** Softmax is 0.19 ms of medium's 2447 ms.

Also: the ~1738 `vsetvli` instructions in a *non-fused* RV64GCV build are **not TATVA's** — they are `e8,m1` with `vle8.v`/`vse8.v`, GCC's inline memcpy expansion in TVM FFI error-string paths. Only 3 float-vector instructions exist in that whole binary, both from GCC's auto-vectorizer. **A march string containing 'v' is worthless as evidence.**

### 4.5 SIGNIFICANT — the capability database reports the opposite of the truth for the one vectorized operator
`capability_for('nn.softmax', TARGETS['RV64GCV'])` returns the constraint *"Emitted as scalar C on this target; the vector unit is available but not targeted by codegen."* — because `nn.softmax` is `KIND_FUSED` and `_vector_note` is appended to `KIND_FUSED` unconditionally (`capabilities.py:196-198`). On RV64GCV the UI tells the user that softmax — the one operator that **is** vectorized — is scalar C. The function's own docstring at `capabilities.py:171-173` states the correct fact and the code below it does the opposite.

### 4.6 SIGNIFICANT — numerical parity is a five-element spot check transported as decimal text
`establish_baseline` compares `ref_output[:5]` against five floats parsed out of a `FIRST_LOGITS:` line in QEMU stdout, at rtol=atol=1e-4 (`runner.py:1534-1558`). It cannot detect a wrong answer in element 6, and text round-tripping caps achievable precision independently of the tolerance.

### 4.7 SIGNIFICANT — `-O2` vs the documented `-O3`, and the free vectorization it leaves on the table
`BASELINE.md:12`, `cli.py:629` and `docs/ARCHITECTURE.md:27` all say `-O3`; the code passes `-O2` at two sites. Recompiling the identical TVM-generated `operators.c` at `-O3 -march=rv64gcv`: **3 vector-float instructions at -O2, 109 at -O3.** GCC vectorizes softmax, mean and most elementwise kernels at -O3 — but **not matmul**, at any flag level tried.

### 4.8 SIGNIFICANT — 49 of 441 tests are invisible to the PR gate
`pytest -m unit` collects 371, `-m integration` 21, and **49 carry no marker** — all 45 of `test_nl_config.py` and all 4 of `test_optimize_cmd.py`. The only job that runs on `pull_request` is `pytest -m unit`. Additionally, the 4 tests in `test_optimize_cmd.py` require the full toolchain but are neither marked `integration` nor guarded by `skip_if_no_toolchain`, so they **fail rather than skip** on a machine without it.

### 4.9 BLOCKING for release — every artifact in `dist/` predates the current commit
`dist/tatva_compiler-2.0.0b1-py3-none-any.whl` (Aug 8) is missing all eight modules the HEAD commit added: `artifacts.py`, `audit.py`, `capabilities.py`, `effort.py`, `frontends.py`, `repair.py`, `runs.py`, `validation.py`. `dist/TATVA/TATVA.exe` is Aug 8 10:03; HEAD is Aug 11 23:28. **The HEAD commit touched no packaging file** — `tatva.spec`'s `hiddenimports` still omit all eight, and they are imported *function-locally* in `gui.py`. Whether a fresh freeze picks them up is unverified.

Also: **the frozen build ships no CLI** (`tatva.spec` defines one EXE from `gui.py` with `console=False`), so installer users have the GUI and nothing else.

### 4.10 MINOR but structural
- The "is this a vector target" test is duplicated at **five sites with three different spellings** (`runner.py:876, 1291, 1428`; `capabilities.py:176`; `gui.py:1853`). A march like `rv64gc_zve32f` would be classified inconsistently across them.
- The RVV kernel's scratch buffer is a fixed 8192-float static array; wider rows return −1 at **runtime** rather than failing at compile time (`runner.py:626-631, 657-660`).
- `TatvaPyBridge._repaired` is a process-lifetime IR cache keyed on a **file path with no content hash** (`gui.py:1367-1368`) — edit the `.onnx` between Auto-Fix and Run and the pipeline compiles the stale rewritten graph. Deliberate contrast with `import_model`'s cache, which **is** content-keyed.
- The toolchain health badge probes cmake and make/ninja but computes green/red from gcc and qemu only (`executor.py:72-85`), and emits raw emoji into a data dict.
- `.pre-commit-config.yaml` references a `.secrets.baseline` file that does not exist, so that gate is effectively off.
- `repair.py:510-512` names its comparison variables backwards; the check is symmetric but a failure message would report the swap in reverse.
- The `sign`/`maximum`/`minimum`/`abs` repair rules are marked `exact=True` and are exact on every input the seeded uniform checker can generate; they diverge only on **NaN** (sign/maximum/minimum) and **−0.0** (abs). Neither the "exact" claim nor a "numerically wrong" claim is actually tested, because the checker cannot produce those inputs.

---

## 5. DETAILED IMPLEMENTATION PLAN — PHASES 2 THROUGH 9

Design constraints applied throughout, per the standing guardrails: no hardcoded performance numbers; nothing labelled supported that isn't; no placeholder backends; every claim traceable to an execution. Phases 2–6 follow the directive; 7–9 are proposed to close what the audit found and to make the result shippable.

---

### PHASE 2 — Fix the INT8 regression, and instrument before touching anything

**2.1 Profiling first (no optimization in this step).**
Add per-kernel cycle attribution to the generated harness. The insertion point already exists: `read_cycles()` in the C harness (`runner.py:212-226`) and the per-kernel call sites in `model_run.c`. Emit a `KERNEL_CYCLES: <name> <count>` line per bucket and parse it into a structured `ProfileResult`. This is the same technique the audit used to attribute the regression to the millisecond — measured instrumentation overhead was 1,884 cycles out of 287.6M (0.0007%).

**2.2 Correct the false premise in the codebase.**
Delete the hardcoded `"~197ms vs ~161ms baseline"` string at `cli.py:742` and replace it with the actual measured delta from a real run, or with nothing. Re-measure or delete the stale `OPTIMIZATION.md` §2 rows (documented −23.16% / −19.03%; measured +66.91% / +16.88%) and §3's unsupported "40% to 72%" size claim.

**2.3 Hoist constant-weight quantization out of the inference loop.** *Largest single win: ~154.9 ms of pretrained's 200.6 ms regression.*
In `optimizer.quantize`, when `arg1` is a `relax.Constant`, quantize it **at compile time** in NumPy and emit the int8 tensor as a constant, rather than emitting a runtime `quantize` call. This alone removes 76.7–82.2% of the per-inference quantized element count.

**2.4 Fix the quantize kernel's arithmetic.**
Replace `x / scale` with `x * (1/scale)` precomputed at compile time (removes `fdiv.s` from the inner loop), and replace the `roundf` libm call with an inline round (removes a `jal` per element). Together these attack the 32.4 cyc/element cost directly. If the kernel is TVM-generated and not directly editable, this becomes a legalization/schedule override — see §5, Phase 5.

**2.5 Decide the honest position on real INT8.**
Two options, and the choice must be explicit rather than implied:
- **(a) Real integer GEMM.** Replace the QDQ pattern with a genuine int8×int8→int32 matmul plus a requantize step, so values stay in int8 between layers. This is the only path to an actual speedup and the only thing that makes the "INT8" label true.
- **(b) Keep QDQ, relabel it.** Rename the pass `quantize-simulate`, surface it in the UI as *"Simulated quantization (accuracy study only — not a latency optimization)"*, and stop reporting it in the latency comparison as though it were one.

Recommendation: **(a)**, with **(b)** shipped in the same commit as the interim honest label so no build ever claims a speedup it cannot deliver. Whichever is chosen, the accuracy failure found on `model_medium` (MSE ≈ 0.43 vs a 0.05 tolerance) must surface as a **failure**, not a footnote.

**2.6 Pin it.** Add a regression test that asserts the FP32→INT8 delta on a fixture, using the deterministic `-icount` property. No such test exists today — `test_quantize.py:45-46` only asserts both numbers are > 0.

**Exit criteria:** per-kernel profile emitted from a real run; INT8 delta re-measured and recorded; no hardcoded latency string anywhere in `src/`; a test that fails if the regression returns.

---

### PHASE 3 — Reproducible benchmark framework

**3.1 `tatva benchmark <model> --target <T> --precision <P>`** as a first-class command, with `--warmup N --iterations N --repeat N --format json|csv --out <path>`. It calls the same library functions the GUI calls — no duplicated logic.

**3.2 Stage separation.** The current single number covers all of `tvmgen_default_run()`. Split into:
- **Preprocessing** — input marshalling, currently untimed above the warmup loop (`runner.py:301`)
- **Execution** — the graph invocation (today's number)
- **Postprocessing** — output extraction, currently untimed (`runner.py:318-324`)

**3.3 Statistics.** Add p99 and stdev alongside mean/median/p95 (`runner.py:1461-1463`). **And label them correctly:** under `-icount shift=0` all samples are identical by construction. Either report `cycles + stated clock` and suppress the dispersion statistics as meaningless in simulation mode, or run without `icount` for the statistics path. Reporting three identical numbers as three statistics is false precision.

**3.4 A real run-identity record.** Every field below already exists in-process at the moment `report.json` is written:
`tatva_version, timestamp, model_path, model_sha256, opset, producer, precision, target_name, march, mabi, gcc_flags, gcc_version, qemu_version, qemu_argv, rvv_enabled, vlen, passes_applied, warmup_count, timed_count, nominal_clock_hz, compile_time_s, raw_samples`.

**3.5 Compilation time.** Nothing records it today. Wrap the import / legalize / build / codegen / gcc / link stages with `perf_counter` and record each separately.

**3.6 Memory.** Host-side peak via `tracemalloc`; **target-side** via the numbers TATVA already computes but never reports — `global_pool` size (`runner.py:1081`), `workspace_pool` (`runner.py:1149`), `weights.h` bytes, stack reservation, and `.text`/`.data`/`.bss` from the ELF. Sum them and compare against the linker region. This is the more decision-relevant number than latency for a bare-metal target.

**3.7 CSV writer.** Zero hits for "csv" in `src/tatva` today.

**3.8 Unify iteration counts.** `baseline-test` (3/10) and `optimize` (2/5) currently disagree silently. One default, recorded in every output, overridable by flag.

**Exit criteria:** two `report.json` files from different runs are provably comparable because every field that could differ is recorded; a CSV of N runs loads into a spreadsheet; `tatva benchmark` output for a model with no passes reproduces `baseline-test`'s number exactly.

---

### PHASE 4 — Operator resolution and real Softmax/Attention fusion

**4.1 The error-code taxonomy.** Build it as **one** vocabulary spanning all three current schemes (`diagnostics.error_type`, `capabilities._UNFIXABLE_REASONS`, `repair.RepairResult.status`). Each code carries: *what* (op/stage/file:line), *where* (graph position), *why* (the mechanism), and *how TATVA handled it*.

Proposed initial set, each mapped to a **real** producer:
| Code | Condition | Producer today |
|---|---|---|
| `E001` Unsupported Operator | no lowering and no repair rule | `compiler.py:228`, plus a new real gate |
| `E002` Repaired Operator | rewritten and validated | `repair.py` (exists, GUI-only) |
| `E003` Numerical Validation Failed | repair or quantization exceeds tolerance | `repair.py:532-609`, `cli.py:780` |
| `E004` Toolchain Missing | gcc/QEMU absent | `runner.py:1355, 1443` — **currently `unknown`** |
| `E005` Memory Constraint | footprint exceeds target budget | **no producer today** — created in Phase 7 |
| `E006` Cross-Compilation Failed | gcc non-zero exit | `runner.py:1386` (exists) |
| `E007` Codegen Failed | `relax.build` raised | `runner.py:934` (exists) |
| `E008` Shape Inference Substituted | a dim was guessed, not declared | `compiler.py:167-176` — **currently silent** |

Rule: **no code without a producer.** `E005` must not exist in the enum until Phase 7 implements the check — the existing `MemoryLimitExceededError` is exactly the anti-pattern to avoid (a class whose only test mocks its own raise site).

**4.2 Route the BLOCKED path through diagnostics.** `gui.py:2245-2254` currently returns `"diagnosis": ""` for the single most common failure. Fix, and tighten the overbroad `"not supported"` regex at `diagnostics.py:143`.

**4.3 Make the pre-fix → re-map → continue loop actually iterate.** Wrap `_rewrite_module` in a bounded fixpoint (`max_iterations`, default 4, with a no-progress break) so an op introduced by a rewrite can itself be repaired. Recompute the mapping **after** each pass — `record.mapping` is currently set once, pre-repair, and never recomputed. Today iteration would be a no-op (no rule's replacement is itself repairable), but it is a precondition for adding rules that decompose into other repairable ops.

**4.4 Expose repair on the CLI.** `repair_graph` is GUI-only. `tatva compile --auto-fix` and `tatva inspect --repairs` must reach the same code.

**4.5 Resolve `SUPPORTED_OPS`.** It does not describe what the backend can lower — `model_blocked.onnx` compiles to a working ELF with `exp` in it. Decide: either (a) it is a real gate, and `compile_model` enforces it, or (b) it is advisory, and the GUI stops calling models BLOCKED on its say-so. Either way, **delete the seven dead Relay-era names** and add the ops that demonstrably lower (`concat`, `broadcast_to`). Generate the list from `tvm.ir.Op.list_op_names()` ∩ a lowering probe rather than hand-curating it.

**4.6 Make softmax fusion a real compiler optimization.** The current implementation is a textual patch of TVM's output. Replace it with a Relax pattern-match pass that recognizes the attention softmax subgraph (`matmul → scale → (mask) → softmax → matmul`) and emits a fused kernel with the reduction, the exp, and the normalization in **one pass over memory** — instead of TVM's current three separate passes. That is where the actual win is: the present kernel swap buys 0.06% precisely because it does not change the memory traffic.

**4.7 Fix the RVV softmax clamp (§4.1) before anything else in this phase.** Add the `< -15.0f → 0.0f` guard to `SOFTMAX_VECTOR`, add a dynamic-range check to `check_softmax_fusable`, and add a test that runs both kernels on the divergent row and asserts they agree. Reconcile the rounding-mode difference (`vfcvt.x.f.v` RNE vs C truncation) or document it as an accepted tolerance.

**Exit criteria:** every error path a user can hit emits a code with a real producer; the fixed row from §4.1 produces identical output on RV64GC and RV64GCV; the fused attention kernel shows a **measured** reduction in kernel invocations and bytes touched, reported by the Phase 3 profiler.

---

### PHASE 5 — Hardware-aware compilation and RVV-first optimization

**5.1 Real capability detection.** A `HardwareProfile` dataclass populated from, in order of preference: an explicit `--hardware <json>` override; a target-side probe (a tiny ELF that reads `misa`, `mvendorid`, `marchid`, and `vlenb`, prints them, and exits — this fits the existing compile-and-boot machinery exactly); the target registry's declared ISA as a last resort, **labelled as declared, not detected**. Fields: `xlen, extensions[], has_v, vlen, elen, lmul_max, cache_line_bytes, l1d_bytes, cores`. Replace the hardcoded `scan_hardware_boards` stub and the QEMU `vlen=128` literal with values read from this profile — and read the value **back**, which nothing does today.

**5.2 Consolidate the five duplicated vector-target tests** into one `HardwareProfile.has_vector` derived from a proper ISA-string parse.

**5.3 Introduce a scheduling layer — this is greenfield.** Order matters:
1. **Switch the model build from `target="c"` to the LLVM backend with the real target string.** `TargetVariant.tvm_target` already carries `-mattr=+v` and is dead code. This is the single cheapest route to auto-vectorized kernels. **Keep the C path working** and selectable — it is what makes the bare-metal harness portable, and the guardrails forbid sacrificing the working ONNX→RV64GC pipeline.
2. **Raise `-O2` to `-O3` at both sites** (`runner.py:841`, `runner.py:1361`) — measured: 3 → 109 vector-float instructions on the identical source. Re-baseline every documented number afterwards; this shifts all of `BASELINE.md` and `OPTIMIZATION.md`.
3. **Tiling for matmul**, parameterized on `VLEN`, element width, and L1 size from the `HardwareProfile`, via a TVM schedule or `dlight`. Tiling is a **prerequisite** for RVV paying off, not an independent item — an untiled vectorized matmul stays memory-bound.
4. **A hand-written RVV matmul micro-kernel** as the fallback if the schedule route underdelivers. `matmul`/`nn.dense`/`nn.batch_matmul` are the KIND_HOT operators and are 100% scalar today (`flw/flw/fmadd.s/fsw`).

**5.4 Auto-tiling must be extensible, not a table.** Tile sizes derive from `(tensor dims, VLEN, element bytes, cache line, L1 bytes)` — a function, so an unseen VLEN produces a sensible tile rather than a lookup miss.

**5.5 Fix the inverted capability note** (`capabilities.py:196-198`) so the UI stops calling the one vectorized operator "scalar C", and make the note derive from **what was actually emitted** in that build rather than from an op-kind table.

**Exit criteria:** a probe ELF reports real `vlenb` from QEMU and the number flows into a tile-size decision; disassembly of `__tvm_ffi_matmul` shows float-vector instructions; the Phase 3 benchmark shows a **measured** end-to-end delta, reported with its methodology, not an estimate.

---

### PHASE 6 — Backend architecture and CLI refinement

**6.1 Establish the layering that partially exists:** `Frontend (frontends.py, compiler.import_model)` → `IR (Relax)` → `Passes (a real pass registry)` → `Backend (a `Backend` protocol with `CBackend` and `LLVMBackend` implementations)`. `runner.py` at 1,573 lines currently mixes codegen, harness emission, toolchain discovery, subprocess driving, and measurement parsing — split it along those seams.

**6.2 Fold the second compile pipeline in.** `scaffolding/executor.py:89-239` is an entirely independent compile-and-emulate implementation defaulting to the **experimental** RV64GCV target, bypassing `TargetVariant` validation, with no test. It must either use the same backend or be deleted. A refactor that only touches `runner.py` leaves it behind.

**6.3 Make the CLI the driver.** `tatva compile`, `tatva profile`, `tatva inspect`, `tatva benchmark` become the API surface; the GUI bridge calls them. Concretely, the eight GUI-exclusive subsystems must become reachable from the CLI: `repair`, `artifacts.write_manifest`, `effort.write_effort`, `audit trail`, `validation.evaluate`, `runs.REGISTRY`, capability rendering, scaffolding. Today a CLI build produces no provenance at all.

**6.4 `tatva_config.json` for reproducibility.** Emitted by every compile/benchmark, consumed by `tatva compile --config`. Contents: everything in the Phase 3.4 run-identity record, plus pass list with parameters, tile sizes chosen, hardware profile used and how it was obtained (detected vs declared vs overridden). A second machine running `tatva compile --config x.json` must produce a byte-identical ELF, and that must be a test.

**6.5 Fix the concurrency hazard.** `_prune_old_builds` under a single shared root causes two TATVA processes to delete each other's build trees — this happened during the audit. Per-process build roots or a lock.

**Exit criteria:** the GUI contains no compiler logic that the CLI cannot reach; a config round-trip reproduces a build byte-for-byte; `scaffolding/executor.py` no longer has its own gcc/QEMU invocation.

---

### PHASE 7 — Memory model and footprint (proposed)

The primary product constraint for a bare-metal RISC-V target, and currently unmeasured. Three uncoordinated allocations exist with no reconciliation: a 128 MB region with a 64 KB stack, a graph-sized activation pool, and a hardcoded 1 MB scratch pool whose exhaustion prints a message and returns NULL at runtime.

1. Compute the real static footprint: `.text + .data + .bss + weights + global_pool + workspace_pool + stack`, from the ELF plus the values TATVA already knows.
2. Add a **device budget** to `TargetVariant` (flash/RAM), and check the footprint against it at compile time. This gives `E005` its first real producer and lets `validation.py:69`'s NOT_IMPLEMENTED entry become implemented.
3. Size `workspace_pool` from the graph instead of the 1 MB literal, and fail at **compile** time rather than returning NULL at runtime. Same for the RVV kernel's 8192-float scratch array.
4. Report peak activation memory in the benchmark output — for a 4.63× activation-pool growth like INT8's, this is the number that decides deployability.

---

### PHASE 8 — Numerical validation depth (proposed)

Parity is currently a five-element spot check at 1e-4 transported as decimal text through a serial console.

1. Compare **full output tensors**, transported as hex or base64 rather than decimal text, so the tolerance is the tolerance and not a printf artifact.
2. Report MSE, max-abs-error, and cosine similarity per output — and **fail** rather than footnote when tolerance is exceeded, per the standing "never label a blocked or partial compilation as successful" constraint.
3. Extend the repair engine's numerical checker to cover the inputs it structurally cannot generate today: NaN, ±0.0, ±inf, and denormals. The `exact=True` claims on `sign`/`maximum`/`minimum`/`abs` are untested precisely there.
4. Add golden tests that assert on **generated code**, not just end-to-end MSE: grep `operators.c` for `__riscv_v`, disassemble the ELF and assert vector instructions in `__tvm_ffi_matmul`, unit-test `inject_optimized_softmax`. None of this exists, so the RVV kernel could silently stop being injected with all tests green.
5. Add end-to-end tests for `models/model_blocked.onnx` and `model_repairable.onnx` — purpose-built fixtures referenced by zero tests.

---

### PHASE 9 — Packaging, provenance, and documentation truth-up (proposed)

1. **Rebuild the distributables.** Every artifact in `dist/` predates the current commit and is missing all eight modules it added. Add the eight to `tatva.spec` `hiddenimports` explicitly rather than relying on modulegraph following function-local imports.
2. **Ship the CLI in the frozen build.** A second console EXE from `cli.py` in the same `COLLECT`. Installer users currently get the GUI and nothing else — which is incompatible with Phase 6's "CLI is the driver".
3. **Marker hygiene.** Add `--strict-markers`, mark the 49 unmarked tests, and guard `test_optimize_cmd.py`'s four toolchain-dependent tests with `skip_if_no_toolchain`.
4. **Declare the undeclared:** `anthropic` (imported in `src/`, declared nowhere, absent from the frozen build) and `pyinstaller` (required to build anything shippable, pinned nowhere). Remove `customtkinter`/`pillow` from the `gui` extra — nothing in `src/` imports either.
5. **Test the packaging layer.** Currently zero tests. At minimum, pin the footer format that is duplicated between `build_installer.py:37-38` and `installer/tatva_setup.py:45-47`.
6. **Truth-up the docs.** `-O3` vs `-O2`; `OPTIMIZATION.md` §2's unreproducible latency rows and §3's unsupported size-reduction claim; the missing `.secrets.baseline`; `build_installer.py` documented nowhere; the wheel shipping the Node web backend that `tatva.spec` goes out of its way to exclude.
7. **Push to `https://github.com/nischithn642/Tatva.git`** with the rebuilt installer.

---

## 6. Dependency order

```
Phase 2 (profiling + INT8)  ─┬─> Phase 3 (benchmark) ──> Phase 5 (hardware + scheduling) ──> Phase 7 (memory)
                             │                                      ^
Phase 4.7 (RVV clamp fix) ───┘                                      │
Phase 4 (errors + real fusion) ─────────────────────────────────────┘
Phase 6 (architecture + CLI) ──> Phase 8 (validation) ──> Phase 9 (packaging)
```

Phase 3 gates Phase 5: without a benchmark that records the hardware profile, tile sizes and flags, no scheduling change can be attributed. Phase 4.7 is independent and should land first — it is a correctness bug in shipped code.

---

## 7. Evidence and method

- 9 subsystem readers over `src/`, `tests/`, `website/`, packaging, and docs; 3 empirical probes; 1 adversarial completeness critic that re-verified claims and resolved six inter-report contradictions.
- 441 tests run to completion (twice). `ruff` run. Three models built and emulated in both FP32 and INT8 legs. Both softmax kernel bodies compiled standalone and run under QEMU on a hand-chosen divergent input. Four ELFs disassembled. `operators.c` recompiled at `-O2` and `-O3` and vector instructions counted.
- Every number in this document came from an execution on this machine or from a cited `file:line`. Where the repository's own documentation disagrees with a measurement, both are shown and the measurement is labelled as such.
- Read-only: `git status --porcelain` in the repository was empty throughout. All scratch work lived in the session scratchpad.
