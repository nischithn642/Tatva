# TATVA: Why It Can't Flash a Real Board Yet

A plain-language summary of where TATVA stands, what's missing, and what it would
take to run on real RISC-V silicon.

---

## Where things stand

TATVA takes an ONNX model and turns it into a RISC-V program. That part works. It
imports the graph, applies a softmax optimization, generates C, cross-compiles it,
and runs it under QEMU to measure speed.

The last step is the catch. **QEMU is the only place the program can run.** The
generated binary is not just "RISC-V code" — it is code built specifically for
QEMU's virtual machine, and it depends on that machine in ways that are easy to
miss.

---

## Why you can't dump it to a board

### 1. The feature was never written

`ExecutionEnvironment.REAL_HW` exists as a name in the code, but the function
behind it just raises an error telling you to deploy the ELF yourself
(`runner.py:1413`). There is no flashing tool anywhere in the project — no
OpenOCD, no serial library, no esptool. The pipeline genuinely stops at a file.

### 2. The memory addresses are QEMU's, not your board's

The linker script hardcodes one block of memory starting at `0x80200000`, 128 MB
long (`runner.py:124`). That address is where QEMU's virtual machine puts programs.
Every real board is different:

| Board | Where its memory actually lives |
| :--- | :--- |
| CH32V307 | flash `0x08000000`, RAM `0x20000000` |
| ESP32-C3 | RAM around `0x3FC80000` |
| HiFive1 Rev B | flash `0x20010000`, RAM `0x80000000` |

There is also no separation between flash and RAM. Real boards store the program
in flash and run it from RAM — TATVA's script doesn't know the difference exists.

### 3. Printing and shutdown rely on firmware that isn't there

Every message the program prints goes through an `ecall` — a request to firmware
called OpenSBI (`runner.py:241`). QEMU boots OpenSBI automatically, so the request
is answered.

A small RISC-V board has no OpenSBI. The request goes nowhere, and the board hangs
on the very first line it tries to print.

### 4. The startup code skips everything real hardware needs

The startup routine is two instructions: set the stack pointer, jump to `main`
(`runner.py:115`). On real hardware you also have to:

- **Zero out the blank memory area.** QEMU hands you memory that is already all
  zeros. Real memory powers up full of garbage.
- **Copy initial data from flash into RAM.** QEMU's loader does this for you. A
  flash programmer does not.
- **Turn on the floating-point unit.** It starts switched off on real chips, and
  the first decimal calculation crashes.
- **Set up a global pointer and a crash handler.**

Worth flagging: **this is a real bug even on QEMU today.** The linker script defines
markers for the blank memory area (`_bss_start`, `_bss_end`) and nothing ever uses
them. It only works because QEMU is generous.

### 5. The model is far too big for a small board

Measured from an actual built BERT-tiny binary:

| Part | Size |
| :--- | :--- |
| Program code | 115 KB |
| **Model weights (in RAM)** | **17.5 MB** |
| **Scratch space (in RAM)** | **3.4 MB** |

That's about **21 MB of RAM**. A typical RISC-V microcontroller has a few hundred
kilobytes.

Most of that is self-inflicted. The weights never change, so they should sit in
flash — but they're written to the file without the `const` keyword
(`runner.py:1062`), so the compiler puts all 17.5 MB in RAM instead. Two more
buffers are fixed sizes regardless of model: 1 MB of scratch and 32 KB for softmax.

### 6. The timing numbers wouldn't carry over

The program reads a cycle counter called `rdcycle`, which many small RISC-V chips
either block or don't have. And the conversion from cycles to milliseconds assumes
a 100 MHz clock, written directly into the code (`runner.py:1458`). Real boards run
at 144, 160, 320 MHz or faster, so every reported time would be wrong.

---

## Other weak spots (unrelated to hardware)

- **Correctness is checked on 5 numbers.** Only the first five outputs are compared
  against the reference. A model that goes wrong from the sixth value onward passes.
  The "MSE 0.000000" in the README is over those five numbers.
- **Only single-output models work.** Most real transformer exports return several
  tensors; those are rejected.
- **Sequence length is guessed.** Unknown dimensions become 32 or 1 based on their
  name. A dimension named `T` or `N` quietly becomes 1, and you benchmark a much
  smaller model than you meant to.
- **The "vector" target isn't vectorized.** RV64GCV compiles and runs, but only the
  hand-written softmax uses vector instructions. Everything else is plain scalar code.
- **There are only two optimizations**, and neither is strong. Softmax fusion only
  helps models with attention (+0.2% on BERT-tiny). INT8 quantization makes things
  *slower* on every currently supported target.
- **The memory allocator can leak.** It only reclaims memory if freed in reverse
  order, which TVM doesn't guarantee.

---

## How to add board support

The work splits sharply depending on what kind of board you have.

### Path A — a Linux board (VisionFive 2, Milk-V, LicheePi, K230)

Much easier, and it suits TATVA's current design. These boards already run OpenSBI
and have gigabytes of RAM.

**Fastest version:** stop building bare-metal. Compile a normal Linux program,
replace the firmware printing with `printf`, drop the linker script and startup code
entirely. Copy it over SSH, run it, and read the same output lines you already parse.
Roughly a day's work for a genuine silicon measurement.

### Path B — a microcontroller board (ESP32-C3, CH32V307, HiFive1)

More work. In order:

1. **Fix the startup code and mark weights `const`.** Both are bugs today and
   everything else depends on them. Marking weights `const` alone moves 17.5 MB out
   of RAM.
2. **Add a `BoardProfile`** describing each board: memory addresses, clock speed,
   how it prints, how it's flashed. This is the abstraction the project is missing.
3. **Make the linker script a template** driven by that profile, with flash and RAM
   properly separated.
4. **Make printing swappable** — firmware calls for QEMU, a direct serial-port write
   for boards, or semihosting over the debug cable.
5. **Size the memory pools from the actual model** instead of fixed constants, and
   refuse to flash when it won't fit — with a clear message, not a crash.
6. **Add a `tatva flash` command**: convert the ELF, run the programmer, open the
   serial port, read the results. The output format needs no changes at all.
7. **Move the clock speed into the profile** instead of assuming 100 MHz.
8. **Bundle OpenOCD** in the zip, the same way GCC and QEMU already are.

---

## Suggested order

1. Fix the startup code and the `const` weights — correctness bugs that matter even
   on QEMU.
2. Do Path A on a Linux board — the shortest route to a real measurement.
3. Build the `BoardProfile` layer for microcontrollers once that's proven.
