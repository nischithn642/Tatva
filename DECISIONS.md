# Architecture & Tooling Decisions - TATVA Step 1

This document records the engineering decisions made during Step 1 of the TATVA optimization project, explaining the reasoning behind target hardware, simulator configurations, and compiler stack choices.

## 1. RISC-V Target Architecture: `RV64GC`
* **Decision:** We chose **`RV64GC`** (64-bit RISC-V with Integer, Multiplication, Atomic, Floating Point, Double-Precision, and Compressed instruction extensions) as our base architecture target.
* **Rationale:**
  * Transformer execution requires floating-point operations. `RV64GC` matches standard modern application processors (such as the SiFive U74).
  * In subsequent steps (Step 2+), we will evaluate vector/matrix extensions. Target `RV64GC` provides the standard baseline for comparison before adding Vector (V) or Matrix (P) extensions.

## 2. Compilation Strategy: Apache TVM Ahead-of-Time (AOT) + C Target
* **Decision:** We chose **TVM AOT Compilation** targeting generic C code, rather than standard Graph/VM Executors or JIT compiled LLVM.
* **Rationale:**
  * Bare-metal systems lack virtual memory and OS process models, which standard TVM executors rely on.
  * TVM AOT compiles model operators and the schedule graph directly into static C functions, generating a single entry point `tvmgen_default_run`.
  * Using the generic `c` target compiles operators directly into standard C code. This code can be linked with the standalone C Runtime (CRT) provided in the Model Library Format (MLF) export.
  * This strategy completely bypasses the need for dynamic shared libraries (`.so`/`.dll`) or dynamic loading on the RISC-V target.

## 3. Simulation Environment: Bare-Metal QEMU Virt
* **Decision:** We chose to run the compiled binaries on **QEMU system-mode emulation (`qemu-system-riscv64 -M virt`)** using a custom bare-metal startup routine, instead of running a Linux OS environment inside QEMU.
* **Rationale:**
  * Booting a full Linux kernel and root filesystem in QEMU is slow, adds hundreds of megabytes of overhead, and complicates file sharing.
  * Bare-metal execution jumps directly to `0x80000000` (RAM base) and runs in less than 5 seconds.
  * Performance measurements are completely deterministic and transparent, avoiding OS-level context switching, interrupts, or page faults.
  * Memory allocation is deterministic, managed by a simple static bump allocator implemented in `main.c`.

## 4. Measurement Methodology: Hardware Cycle Counter (`rdcycle`)
* **Decision:** We read the RISC-V Control and Status Register `cycle` (via the `rdcycle` instruction) directly before and after the inference execution loop.
* **Rationale:**
  * Real-world elapsed time (wall-clock time) inside QEMU is affected by the host CPU performance and QEMU's translation speed. It is not representative of real RISC-V performance.
  * The cycle counter (`rdcycle`) measures the exact number of virtual clock cycles the emulator took to execute the guest instructions.
  * Cycle counts are reproducible, hardware-aligned, and can be converted to simulated execution time by assuming a nominal clock frequency (e.g., 100 MHz).

## 5. Tooling Portability: Local Extract and User Scope
* **Decision:** We chose to download the RISC-V cross-compiler and QEMU/Renode as portable ZIP archives and extract them locally into the project directory rather than relying on system installers.
* **Rationale:**
  * Windows system installers (from `winget`) require Administrator privileges (UAC elevation) to write to `C:\Program Files`. Running them programmatically in sandboxed or background environments fails.
  * Portable ZIP extractions require no administrative rights and are isolated to the project folder, preventing system-wide PATH contamination and ensuring a fresh clone on a clean machine can be set up autonomously.
