# Getting Started With TATVA

TATVA compiles an ONNX Transformer model into a standalone C99 binary for a bare-metal
RISC-V target, then measures it under emulation — baseline against optimized, in the same
run.

This page takes you to your first measured result. [README.md](README.md) is the
reference; this is the walkthrough.

---

## Pick your path

| You have | Start here | Time to first result |
| :--- | :--- | :--- |
| `TATVA-beta-2.0-windows.zip` | [The app](#the-app) | ~5 minutes, plus a download |
| Python 3.12 or 3.13 and a terminal | [The CLI](#the-cli) | ~10 minutes |
| A clone of this repository | [From source](#from-source) | ~15 minutes |

The app and the CLI are the same compiler with different front ends. Neither is a
prerequisite for the other.

---

## The app

### 1. Launch it

Unzip the folder anywhere and double-click **`TATVA.exe`**.

Keep the folder together — the exe is not standalone, it loads the `_internal` directory
next to it. Moving `TATVA.exe` out on its own will stop it working.

Two things to expect on first launch:

- **Windows SmartScreen** will say the publisher is unknown. The build is not
  code-signed; signing requires a purchased certificate, not a build flag. Click
  **More info → Run anyway**.
- **A splash screen for a few seconds** while Apache TVM loads. This is the slowest part
  of startup and it only happens once per launch.

### 2. Install the RISC-V toolchain

Stages 01–04 work the moment you unzip. Stage 05 shells out to a cross-compiler and an
emulator, so it needs both present.

They are not in the zip — together they are about 520 MB and licensed separately, which
would more than quadruple the download for everyone who already has them.

Get them from inside the app:

> **Diagnostics → Install the RISC-V toolchain**

The card lists every URL and the destination before you press anything. It downloads
pinned xPack builds of `riscv-none-elf-gcc` and `qemu-system-riscv64` into
`%LOCALAPPDATA%\tatva\toolchains`, then re-checks. No admin rights. Nothing is added to
`PATH`. Nothing else on the machine changes.

If you already have either tool on your `PATH`, TATVA uses that one and you can skip
this — the Diagnostics page shows the resolved path for each, so you can confirm which
binary it picked.

The bottom-left of the sidebar reads **Toolchain ready** or **Toolchain not installed**
at all times. Check it before you start rather than after.

### 3. Walk the five stages

The sidebar and the rail across the top show the same five stages. Each unlocks when the
one before it produces a result.

| Stage | What it does | What you do |
| :--- | :--- | :--- |
| **01 INPUT** | Loads an ONNX model and picks a target | Click a sample card, or browse to your own `.onnx` |
| **02 ANALYZE** | Imports the graph, counts operators, finds the attention pattern | Press **Run analysis** |
| **03 MAP** | Checks every operator against what the target can execute | Press **Run mapping** |
| **04 OPTIMIZE** | Chooses the passes and shows the exact build plan | Leave **Softmax fusion** on |
| **05 GENERATE** | Emits C99, cross-compiles, runs under QEMU, measures both builds | Press **Build & benchmark** |

Four sample models ship inside the zip. For a first run:

- **MLP fixture** (`model_mlp.onnx`) — fastest end to end, about 5 seconds of building
  and emulation. Two dense layers, no attention.
- **Tiny transformer block** (`model.onnx`) — **pick this one to see the fusion pass do
  something.** It has an attention pattern, which is what softmax fusion rewrites.
- **Nano** and **Medium** — realistic, and slower. Use them once a small one works.

Stage 05 runs two full builds and two emulated runs, so it takes longer than the others.
The terminal reports once at the end; it does not stream.

### 4. Read the result

A finished run prints a block like this:

```text
baseline    0.731 ms
optimized   0.665 ms
change      -8.99% latency
parity MSE  2.94e-5 vs host ONNX Runtime
status      PASS [OK]
```

Then **View the benchmark report →** gives you the same numbers as a chart plus a
per-configuration table.

Three things worth knowing before you quote any of it:

- **`0.00%` is a real result, not a failure.** It means the passes you selected had
  nothing to change in that graph. Softmax fusion needs an attention pattern to fuse;
  `model_mlp.onnx` has none and `model.onnx` does. Running the MLP fixture with fusion on
  correctly reports no measurable difference.
- **INT8 quantization is slower here, and says so before it runs.** A scalar `rv64gc`
  core has no INT8 dot-product instruction, so the dequantize scaling is emulated in
  software. It shrinks the binary and raises the cycle count. Turn it on when SRAM is the
  constraint, not for speed.
- **These are emulator cycles, not silicon.** Every timing comes from QEMU system-mode
  emulation with `-icount shift=0`, reading the target's own cycle counter, converted at
  a nominal 100 MHz. They are for comparing two builds of the same model against each
  other. They are not a claim about real hardware.

---

## The CLI

Same compiler, terminal in front of it. Useful for scripting and CI.

```bash
pip install tatva_compiler-2.0.0b1-py3-none-any.whl
```

Requires Python 3.12 or 3.13. Apache TVM publishes no wheels for 3.14 yet, and TATVA uses
features absent from 3.11, so `pip` refuses to install on anything else rather than
half-work.

Then, in order:

```bash
tatva setup
```

Downloads the same pinned toolchain the app's Diagnostics page installs, to the same
per-user directory. Use `tatva setup --dry-run` first to see exactly what it would fetch
and where.

```bash
tatva doctor
```

The gate. It checks host Python, Apache TVM, ONNX Runtime, the cross-compiler and the
emulator, and names whatever is missing along with every path it searched. It exits
non-zero when something is absent, so it works as a CI check.

```bash
tatva analyze models/model.onnx
tatva baseline-test models/model.onnx --target RV64GC
tatva optimize models/model.onnx --passes fuse --out build_opt
```

`analyze` reads the graph, `baseline-test` establishes the unoptimized reference, and
`optimize` applies the passes and compiles into `build_opt`.

The full command set is `doctor`, `setup`, `targets`, `analyze`, `baseline-test`,
`optimize`, `diagnose`, `validate`, `gui`. Add `--help` to any of them.

---

## From source

```bash
git clone <repo-url>
cd tatva
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
tatva setup && tatva doctor
```

If `tatva doctor` prints five `[OK]` lines, everything else works. If it doesn't, it
names the missing piece and where it looked.

To run the test suite and launch the desktop app from the checkout:

```bash
pytest -q
tatva gui
```

To build the shareable zip yourself:

```bash
python build_exe.py
```

That writes `dist/TATVA/TATVA.exe` and `dist/TATVA-beta-2.0-windows.zip`. PyInstaller
does not cross-compile — build on the OS you are shipping to.

---

## When it doesn't work

**"RISC-V GCC cross-compiler binary not found" / "emulator binary not found"**
The toolchain is missing. Open **Diagnostics → Install the RISC-V toolchain**, or run
`tatva setup`. From Beta 2.0 onward stage 05 checks for both binaries before it starts
and shows this as a banner with an install button rather than failing mid-build.

Note that a `riscv64-linux-gnu-gcc` — the common apt/WSL/MSYS2 package — does not count.
It targets Linux userspace, and TATVA links bare metal (`-ffreestanding -nostdlib`)
against its own startup code. Same for the user-mode `qemu-riscv64`: the measurement
boots the ELF on `-machine virt` to read the cycle counter, which user-mode emulation
cannot do. You need `riscv-none-elf-gcc` and `qemu-system-riscv64` specifically.

**`tatva: 'compile' is not a command`**
Stage 05's terminal opens with a line like
`$ tatva compile model.onnx --target RV64GC --fuse`. That is a summary of the run's
configuration, not a command you can type — there is no `compile` subcommand. The CLI
equivalent is `tatva optimize models/model.onnx --passes fuse`.

**Stage 03 reports operators with no lowering**
The model uses an operator TATVA cannot generate code for on that target. The stage names
each one. Either remove them from the model or add a lowering — see
[docs/EXTENDING_TARGETS.md](docs/EXTENDING_TARGETS.md). Finding this at stage 03 is the
point of stage 03; it would otherwise be a compile failure at stage 05.

**The benchmark report is empty**
Nothing has been measured yet. The page deliberately stays blank rather than showing a
placeholder number. Complete a stage 05 run first.

**The app won't start, or closes immediately**
Check that `TATVA.exe` still sits beside its `_internal` folder. If it does, run
`tatva doctor` from a source install for a readable diagnosis.

---

## Where to go next

- [README.md](README.md) — full feature reference and how to share TATVA with someone else
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the compiler pipeline and module map
- [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) — what gets installed, from where, and how it is pinned
- [OPTIMIZATION.md](OPTIMIZATION.md) — the softmax derivation and the quantization findings
- [docs/EXTENDING_TARGETS.md](docs/EXTENDING_TARGETS.md) — adding a RISC-V target variant
- [CHANGELOG.md](CHANGELOG.md) — what changed in Beta 2.0
