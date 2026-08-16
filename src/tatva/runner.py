"""
TATVA Execution and Simulation Runner Module.

This module handles compiling the generated C code with the RISC-V cross-compiler,
executing the resulting binary in the QEMU simulator environment, and running
target architecture verification gates and performance measurements.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from enum import Enum
from typing import Any

import numpy as np

from tatva.compiler import ModelIR, TargetVariant, c_safe_name

# Single shared exception type. There used to be a second, unrelated CompilationError
# defined here, which meant diagnostics.classify_failure() never recognised the errors
# the runner actually raised and always fell through to the generic branch.
from tatva.diagnostics import (
    CompilationError,
    EmulatorImageLimitError,
    MemoryLimitExceededError,
    SimulationTimeoutError,
)

__all__ = [
    "BaselineResult",
    "CompilationError",
    "CompiledArtifact",
    "EmulatorImageLimitError",
    "ExecutionEnvironment",
    "KernelProfile",
    "MeasurementResult",
    "SimulationTimeoutError",
    "bundled_tools_dir",
    "compile_model",
    "default_input_array",
    "default_inputs_for",
    "establish_baseline",
    "find_qemu",
    "find_riscv_gcc",
    "reference_output",
    "run_and_measure",
    "verify_target",
]

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# How many unnamed build directories to keep before pruning the oldest.
_BUILD_DIR_KEEP = 8


def bundled_tools_dir() -> str | None:
    """
    The toolchain shipped inside the app folder, or None if this is not a packaged build.

    The desktop build ships the cross-compiler and QEMU in `toolchain/` beside TATVA.exe,
    so stage 05 works the moment the zip is unpacked -- no download, no network, no admin
    rights. Previously the app could only reach a toolchain the user had gone and fetched,
    which meant the one stage that produces a measurement was the one stage that did not
    work out of the box.

    Layout matches the per-user install directory exactly (`<name>/bin/...`), so the same
    search code handles both and there is only one thing to get right.
    """
    if getattr(sys, "frozen", False):
        # One-folder PyInstaller build: sys.executable is <app>/TATVA.exe.
        return os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "toolchain")

    # A source checkout can stage the same directory to test the packaged layout without
    # running a five-minute PyInstaller build.
    candidate = os.path.join(PROJECT_DIR, "toolchain")
    return candidate if os.path.isdir(candidate) else None


def build_root() -> str:
    """
    Where a build goes when the caller does not name a directory.

    This used to be `<project>/scratch`, which is wrong twice over. PROJECT_DIR is
    derived from this file's location, so for a pip-installed wheel it resolves inside
    site-packages and TATVA writes build trees into the user's virtualenv. And when the
    project *is* checked out normally it frequently lives in a synced folder (OneDrive,
    Dropbox), where every file GCC writes is immediately locked, scanned and uploaded --
    which is how a 120-second compile step starts timing out during a long test run.

    Set TATVA_BUILD_DIR to put builds somewhere specific.
    """
    root = os.environ.get("TATVA_BUILD_DIR") or os.path.join(tempfile.gettempdir(), "tatva-builds")
    os.makedirs(root, exist_ok=True)
    return root


def _prune_old_builds(root: str, keep: int = _BUILD_DIR_KEEP) -> None:
    """
    Drop all but the `keep` newest build directories under `root`.

    Only directories this module created are considered -- the `tatva_build_` prefix is
    the whole guard. Without this the old default grew to 71 directories and 1.3 GB on
    the development machine, because nothing ever deleted one.
    """
    try:
        entries = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if name.startswith("tatva_build_") and os.path.isdir(os.path.join(root, name))
        ]
        entries.sort(key=os.path.getmtime, reverse=True)
        for stale in entries[keep:]:
            shutil.rmtree(stale, ignore_errors=True)
    except OSError:
        # Pruning is housekeeping; never let it fail a compile.
        pass

# Assembly entry point stub for bare-metal targets under OpenSBI
START_S = """
.section .text.init
.global _start
_start:
    la sp, _stack_top
    tail main
"""

# QEMU's `virt` board puts RAM at 0x80000000 and the firmware (OpenSBI) in the first
# 2 MiB of it, which is why the image is linked at 0x80200000. Anything sizing a RAM
# region against a QEMU `-m` value has to account for that gap.
RAM_ORIGIN = 0x80200000
FIRMWARE_RESERVED_BYTES = 0x200000

# The floor for the linked region. 128 MiB is what the region was fixed at before it
# was computed, and it is also QEMU virt's default `-m`, so every model that used to
# build and run keeps exactly the memory map it had.
MIN_RAM_BYTES = 128 * 1024 * 1024

# Headroom over the demand TATVA can account for: .text, the linker's own alignment,
# and anything GCC emits that is not one of the pools counted in _ram_region_bytes.
# .text measured 188 KB on a 6-layer BERT, so this is generous by design -- it costs
# emulated address space, which is free, and getting it wrong costs a failed link.
RAM_SLACK_BYTES = 16 * 1024 * 1024
RAM_GRANULE_BYTES = 16 * 1024 * 1024

# The two fixed pools in the generated sources. Named here so the memory budget and the
# code that emits them cannot drift apart.
WORKSPACE_POOL_BYTES = 1024 * 1024
STACK_BYTES = 0x10000


# Wall-clock allowance per MiB of linked image, per inference, for the QEMU timeout.
#
# Derived from a measurement, not picked: all_minilm_l6_v2.onnx (90.3 MB of weights,
# 128 MiB region) ran 1 warm-up plus 1 timed inference in 100.9 s on this machine, which
# is 0.39 s per MiB per inference. 0.6 leaves roughly 1.5x margin for a slower or busier
# host. It is a timeout, not a budget -- a correct run never waits for it.
QEMU_SECONDS_PER_MIB_PER_RUN = 0.6

# The same allowance for a target without hardware floating point.
#
# Scaled from the constant above by a measured ratio, so the two carry the same margin.
# The same model (all_minilm_l6_v2.onnx, 128 MiB region, run_count 7) was run on both
# kinds of target on an otherwise idle machine:
#
#   RV64GC   220.5 s wall clock,  2,835,467,134 cycles per inference
#   RV32IMC  799.4 s wall clock, 58,085,441,820 cycles per inference
#
# 20.5x the guest cycles, because without an F/D extension every FP32 multiply becomes a
# soft-float library call; 3.625x the wall clock, the emulator absorbing some of it.
# 0.6 x 3.625 = 2.175, rounded up. Cross-checked against a contended run of the same
# build, which came out 1% slower -- host load is not what this is measuring.
#
# Without this, RV32IMC was judged against the hardware-float ceiling of 537 s and killed
# at 553.7 s having done nothing wrong.
QEMU_SOFT_FLOAT_SECONDS_PER_MIB_PER_RUN = 2.2

# The floor, and what the timeout was fixed at before it was computed. Small models keep
# the behaviour they had.
QEMU_MIN_TIMEOUT_SECONDS = 30


# Where QEMU's `virt` board puts the flattened device tree, and therefore the hard
# ceiling on how far the image may extend.
#
# QEMU places the FDT below 3 GiB when RAM base plus size passes 3 GiB. The virt board's
# RAM base is 0x80000000, so every `-m` above 1024M lands the FDT at 0xBFE00000 and
# leaves it there. Measured against the bundled qemu-system-riscv64 at -m 1100M, 1600M,
# 2500M, 4096M and 8192M -- 0xBFE00000 in all five cases. Raising `-m` does not move it.
#
# An image that reaches this address is refused at load with "Some ROM regions are
# overlapping", which names the FDT and never mentions model size. Verified from both
# sides: a 1008.1 MiB image runs, a 1036 MiB image is refused.
QEMU_FDT_BASE = 0xBFE00000

# 0xBFE00000 - 0x80200000 = 1020 MiB. This is the real upper bound on model size under
# the bundled emulator, and it is lower than the linker's and lower than protobuf's.
MAX_LOADABLE_IMAGE_BYTES = QEMU_FDT_BASE - RAM_ORIGIN


def _qemu_memory_mib(ram_bytes: int) -> int:
    """
    RAM to give the QEMU `virt` board, in MiB, for an image linked into `ram_bytes`.

    The board defaults to 128 MiB. A model linked into a larger region needs `-m` raised
    to match or it loads into memory the machine does not have, so a build that got past
    the linker would still die at boot. Returns at least 128, which is the default -- so
    passing it changes nothing for models that already fit.
    """
    total = max(MIN_RAM_BYTES, ram_bytes) + FIRMWARE_RESERVED_BYTES
    return -(-total // (1024 * 1024))


def _has_hardware_float(variant: TargetVariant) -> bool:
    """
    Whether the target's ISA string includes hardware floating point.

    `g` is shorthand for `imafd`, so it counts, as do explicit `f` and `d`. Everything
    else -- rv32imc, rv32imac, rv32emc -- executes FP32 through soft-float library calls,
    which is a difference in kind, not in degree, for a model full of float multiplies.
    """
    base = variant.gcc_march.lower().split("_", 1)[0]
    exts = base[4:] if base.startswith(("rv32", "rv64")) else base
    return any(letter in exts for letter in ("g", "f", "d"))


def _qemu_timeout_seconds(ram_bytes: int, run_count: int, variant: TargetVariant | None = None) -> int:
    """
    Wall-clock ceiling for one QEMU run, scaled by image size, inference count and target.

    The ceiling was fixed at 30 s, which is fine for the small fixtures and nowhere near
    enough for a real transformer: the 6-layer BERT above needed 100.9 s for two
    inferences, so the fixed timeout killed the run and reported it as a QEMU failure
    rather than as the timeout it was.

    `variant` is optional so existing callers keep the hardware-float rate they were
    written against. Passing it is what makes a soft-float target survive: the same
    model on the same region needs several times the wall clock without an FPU.

    Note this is deliberately generous. `run_count` is warm-up plus timed, but the timed
    loop breaks as soon as two runs agree (STABLE_BREAK), which under `-icount shift=0`
    is always after two -- so the real work is usually fewer inferences than the ceiling
    is sized for. A timeout should never be the thing that ends a correct run.
    """
    mib = max(MIN_RAM_BYTES, ram_bytes) / (1024 * 1024)
    rate = QEMU_SECONDS_PER_MIB_PER_RUN
    if variant is not None and not _has_hardware_float(variant):
        rate = QEMU_SOFT_FLOAT_SECONDS_PER_MIB_PER_RUN
    scaled = mib * max(1, run_count) * rate
    return max(QEMU_MIN_TIMEOUT_SECONDS, int(scaled))


def _elf_load_span(elf_path: str) -> tuple[int, int] | None:
    """
    Lowest and highest physical address the ELF's PT_LOAD segments occupy.

    Read from the program headers rather than inferred from the file size, because the
    two differ: .bss occupies address space without occupying file bytes, and it is the
    address span QEMU checks for overlap. On a 1.012 GiB image the two differ by 3.2 MiB,
    which is the wrong side of the margin to guess at.

    Returns None if the file cannot be parsed as an ELF. The caller falls back to reading
    QEMU's own complaint, so a parse failure costs a worse message, never a wrong answer.
    """
    try:
        with open(elf_path, "rb") as fh:
            ident = fh.read(16)
            if len(ident) < 16 or ident[:4] != b"\x7fELF":
                return None
            is_64 = ident[4] == 2
            endian = "<" if ident[5] == 1 else ">"

            # e_phoff, e_phentsize and e_phnum sit at different offsets in the two classes.
            if is_64:
                fh.seek(32)
                (phoff,) = struct.unpack(f"{endian}Q", fh.read(8))
                fh.seek(54)
                phentsize, phnum = struct.unpack(f"{endian}HH", fh.read(4))
            else:
                fh.seek(28)
                (phoff,) = struct.unpack(f"{endian}I", fh.read(4))
                fh.seek(42)
                phentsize, phnum = struct.unpack(f"{endian}HH", fh.read(4))

            if not phnum or phentsize < (56 if is_64 else 32):
                return None

            low: int | None = None
            high = 0
            for i in range(phnum):
                fh.seek(phoff + i * phentsize)
                entry = fh.read(phentsize)
                if len(entry) < phentsize:
                    return None
                if is_64:
                    # p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz
                    p_type, _flags, _off, _vaddr, p_paddr, _filesz, p_memsz = struct.unpack(
                        f"{endian}IIQQQQQ", entry[:48]
                    )
                else:
                    # p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz
                    p_type, _off, _vaddr, p_paddr, _filesz, p_memsz = struct.unpack(
                        f"{endian}IIIIII", entry[:24]
                    )
                if p_type != 1:  # PT_LOAD
                    continue
                low = p_paddr if low is None else min(low, p_paddr)
                high = max(high, p_paddr + p_memsz)

            if low is None:
                return None
            return low, high
    except (OSError, struct.error):
        return None


def _mib(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MiB"


def _memory_budget_note(
    weights_bytes: int, activation_bytes: int, io_bytes: int, ram_bytes: int, target: str
) -> str:
    """
    Spell out where a build's memory went, for a failure the user has to act on.

    Every figure is the size of a pool this build actually emitted, not an estimate.
    """
    return (
        f"Memory budget for {target}: "
        f"weights {_mib(weights_bytes)}, "
        f"activations {_mib(activation_bytes)}, "
        f"workspace {_mib(WORKSPACE_POOL_BYTES)}, "
        f"harness IO {_mib(io_bytes)}, "
        f"stack {_mib(STACK_BYTES)} "
        f"-- linked into a {_mib(ram_bytes)} region."
    )


def render_link_ld(ram_bytes: int = MIN_RAM_BYTES) -> str:
    """
    Fill the linker-script template for a region of `ram_bytes`.

    LINK_LD is a template with markers, not a usable script, so every caller has to go
    through here -- writing the raw constant would put "@TATVA_RAM_LENGTH@" into the file
    and fail the link with a parse error.
    """
    return LINK_LD.replace("@TATVA_RAM_LENGTH@", str(ram_bytes)).replace(
        "@TATVA_STACK_BYTES@", str(STACK_BYTES)
    )


def _ram_region_bytes(accounted_bytes: int) -> int:
    """
    Size the linker's RAM region from what the build actually needs.

    `accounted_bytes` is every pool compile_model can measure before invoking GCC:
    the weight blob, the activation pool, the workspace pool, the harness IO buffers
    and the stack. The result adds RAM_SLACK_BYTES for the sections it cannot measure
    and rounds up to RAM_GRANULE_BYTES, with MIN_RAM_BYTES as a floor.

    The region used to be fixed at 128 MiB. A model whose weights alone exceeded that
    failed at the linker with ".bss will not fit in region ram" -- a message that names
    .bss because .bss is simply what the linker was placing when it ran out, not because
    the activations were the problem.
    """
    needed = max(MIN_RAM_BYTES, accounted_bytes + RAM_SLACK_BYTES)
    return -(-needed // RAM_GRANULE_BYTES) * RAM_GRANULE_BYTES


# Linker script defining a bare-metal memory space. LENGTH is filled in per build by
# _ram_region_bytes; see the note there about why it is no longer a fixed 128M.
LINK_LD = """
OUTPUT_ARCH( "riscv" )
ENTRY( _start )

MEMORY
{
  ram (wxa) : ORIGIN = 0x80200000, LENGTH = @TATVA_RAM_LENGTH@
}

SECTIONS
{
  . = 0x80200000;
  
  .text : {
    *(.text.init)
    *(.text .text.*)
  } > ram

  .rodata : {
    *(.rodata .rodata.*)
  } > ram

  .data : {
    *(.data .data.*)
  } > ram

  .bss : {
    . = ALIGN(8);
    _bss_start = .;
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    _bss_end = .;
  } > ram

  . = ALIGN(16);
  _stack_bottom = .;
  . += @TATVA_STACK_BYTES@;
  _stack_top = .;
}
"""

# C Driver stub for hello-world targets
MAIN_C_HELLO = """
void sbi_putchar(char c) {
    register unsigned long a0 asm("a0") = (unsigned long)c;
    register unsigned long a7 asm("a7") = 0x01; // Console Putchar EID
    asm volatile ("ecall" : : "r"(a0), "r"(a7) : "memory");
}

void sbi_print(const char* str) {
    while (*str) {
        sbi_putchar(*str++);
    }
}

void sbi_shutdown(void) {
    register unsigned long a0 asm("a0") = 0; // Shutdown type
    register unsigned long a1 asm("a1") = 0; // Shutdown reason (0 = success)
    register unsigned long a6 asm("a6") = 0; // Function ID
    register unsigned long a7 asm("a7") = 0x53525354; // System Reset (SRST) EID
    asm volatile ("ecall" : : "r"(a0), "r"(a1), "r"(a6), "r"(a7) : "memory");
}

int main(void) {
    sbi_print("Hello from Target: ");
    sbi_print(TARGET_NAME);
    sbi_print("\\n");
    sbi_shutdown();
    while (1);
    return 0;
}
"""

# C Driver for model inference benchmarking inside QEMU system-mode.
#
# The input buffers, their types and the call signature are NOT fixed: they are
# generated from the model's own graph inputs by `compile_model`, which fills in
# the @TATVA_*@ markers below. Hardcoding a tensor list here would silently limit
# TATVA to models that happen to look like BERT.
MAIN_C_BENCHMARK = """
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "model_info.h"

extern int32_t tvmgen_default_run(@TATVA_RUN_PARAMS@);
@TATVA_PROFILE_DECLS@
static inline uint64_t read_cycles(void) {
    uint64_t cycles;
#if __riscv_xlen == 64
    asm volatile ("rdcycle %0" : "=r" (cycles));
#else
    uint32_t cycle_h, cycle_l, cycle_h_again;
    do {
        asm volatile ("rdcycleh %0" : "=r" (cycle_h));
        asm volatile ("rdcycle %0" : "=r" (cycle_l));
        asm volatile ("rdcycleh %0" : "=r" (cycle_h_again));
    } while (cycle_h != cycle_h_again);
    cycles = (((uint64_t)cycle_h) << 32) | cycle_l;
#endif
    return cycles;
}

static int local_errno;
int* __errno(void) {
    return &local_errno;
}

void* memset(void* dest, int c, size_t n) {
    unsigned char* d = dest;
    while (n--) {
        *d++ = (unsigned char)c;
    }
    return dest;
}

void sbi_putchar(char c) {
    register unsigned long a0 asm("a0") = (unsigned long)c;
    register unsigned long a7 asm("a7") = 0x01;
    asm volatile ("ecall" : : "r"(a0), "r"(a7) : "memory");
}

void sbi_print(const char* str) {
    while (*str) {
        sbi_putchar(*str++);
    }
}

void sbi_print_uint(uint64_t val) {
    char buf[32];
    int i = 0;
    if (val == 0) {
        sbi_print("0");
        return;
    }
    while (val > 0) {
        buf[i++] = '0' + (val % 10);
        val /= 10;
    }
    for (int j = i - 1; j >= 0; j--) {
        sbi_putchar(buf[j]);
    }
}

void sbi_print_float(float val) {
    // Non-finite values first. (uint64_t)val is undefined for NaN and infinity, and on
    // RISC-V the conversion saturates to 2^64-1, so a model that overflowed printed
    // "FIRST_LOGITS: 18446744073709551615.//////" -- which reads as a broken emulator
    // rather than as the broken computation it actually is. Sqrt of a negative input
    // reaches here on any run whose dummy data crosses zero.
    union { float f; uint32_t u; } tatva_bits;
    tatva_bits.f = val;
    if ((tatva_bits.u & 0x7F800000u) == 0x7F800000u) {
        if (tatva_bits.u & 0x007FFFFFu) {
            sbi_print("nan");
        } else {
            sbi_print((tatva_bits.u & 0x80000000u) ? "-inf" : "inf");
        }
        return;
    }
    if (val < 0) {
        sbi_print("-");
        val = -val;
    }
    uint64_t integer_part = (uint64_t)val;
    sbi_print_uint(integer_part);
    sbi_print(".");
    float fraction = val - (float)integer_part;
    for (int i = 0; i < 6; i++) {
        fraction *= 10.0f;
        uint32_t digit = (uint32_t)fraction;
        sbi_putchar('0' + digit);
        fraction -= (float)digit;
    }
}

void sbi_shutdown(void) {
    register unsigned long a0 asm("a0") = 0;
    register unsigned long a1 asm("a1") = 0;
    register unsigned long a6 asm("a6") = 0;
    register unsigned long a7 asm("a7") = 0x53525354;
    asm volatile ("ecall" : : "r"(a0), "r"(a1), "r"(a6), "r"(a7) : "memory");
}

@TATVA_IO_BUFFERS@

int main(void) {
    sbi_print("\\n=== Starting Latency Test inside QEMU ===\\n");

    // Deterministic dummy inputs. These MUST match runner.default_input_array(),
    // which feeds the host ONNX Runtime reference used for the parity check.
@TATVA_IO_INIT@

    // Warm-up runs
    for (int i = 0; i < WARMUP_COUNT; i++) {
        tvmgen_default_run(@TATVA_RUN_ARGS@);
    }
@TATVA_PROFILE_RESET@
    // Timed runs.
    //
    // The loop bound below is a ceiling, not a quota. QEMU runs with -icount shift=0,
    // which counts retired instructions rather than sampling a real clock, so a run
    // that repeats identical work returns a bit-identical cycle count. Once two
    // consecutive runs agree exactly there is no distribution left to sample and
    // every further iteration costs wall-clock time to reprint a number we already
    // have -- on a 90 MB transformer that was 12 redundant inferences, about ten
    // minutes. The loop therefore stops the first time it *observes* stability; it
    // does not assume it. If the counts ever disagree, every iteration is taken.
    //
    // A profiled build emits no break at all and always runs the ceiling. The
    // per-kernel accumulators are reset once, above, and reported once, below, so
    // they cover exactly the iterations the RUN_CYCLES lines cover -- an early exit
    // would leave the kernel totals describing a different number of runs than the
    // samples they are compared against.
    uint64_t prev_cycles = 0;
    for (int i = 0; i < TIMED_COUNT; i++) {
        uint64_t start = read_cycles();
        tvmgen_default_run(@TATVA_RUN_ARGS@);
        uint64_t end = read_cycles();
        uint64_t elapsed = end - start;
        sbi_print("RUN_CYCLES: ");
        sbi_print_uint(elapsed);
        sbi_print("\\n");
@TATVA_STABLE_BREAK@
        prev_cycles = elapsed;
    }
    (void)prev_cycles;
@TATVA_PROFILE_REPORT@
    // Print first 5 output values for correctness verification
    sbi_print("FIRST_LOGITS: ");
    for (int i = 0; i < 5 && i < OUTPUT_SIZE; i++) {
        sbi_print_float((float)tatva_output[i]);
        sbi_print(" ");
    }
    sbi_print("\\n");

    sbi_print("=== Latency Test Finished ===\\n");
    sbi_shutdown();
    while (1);
    return 0;
}
"""

# ---------------------------------------------------------------------------
# Per-kernel cycle attribution.
#
# These three fragments fill the @TATVA_PROFILE_*@ markers in MAIN_C_BENCHMARK.
# When compile_model(profile=False) -- the default -- every marker is replaced
# with the empty string. Each marker sits alone on a line that used to be blank,
# so the unprofiled main.c is byte-for-byte what it was before this feature
# existed. That is the whole point of the marker placement: the measurement you
# ship must not be the measurement you changed.
# ---------------------------------------------------------------------------

# Fills @TATVA_STABLE_BREAK@ in an unprofiled build. Two consecutive identical cycle
# counts mean the simulation is deterministic over this workload, at which point the
# remaining iterations cannot change mean, median or p95.
STABLE_BREAK = """        if (i > 0 && elapsed == prev_cycles) {
            break;
        }"""

PROFILE_MAIN_DECLS = """
// Per-kernel accumulators live in model_run.c, next to the call sites they time.
extern uint64_t tatva_kernel_cycles[];
extern uint64_t tatva_kernel_calls[];
extern const char* const tatva_kernel_names[];
extern const int tatva_kernel_count;
extern void tatva_profile_reset(void);
"""

# Warm-up exists to fault in the workspace bump-allocator and settle the caches;
# attributing those runs to the kernels would inflate whichever kernel happens to
# run first. Reset here so the per-kernel totals cover exactly the same TIMED_COUNT
# iterations the RUN_CYCLES samples cover, which is what makes the two comparable.
PROFILE_MAIN_RESET = """
    tatva_profile_reset();
"""

# One line per distinct kernel, not per call site: output size is bounded by the
# number of PrimFuncs TVM emitted, not by how many times the graph calls them.
PROFILE_MAIN_REPORT = """
    for (int k = 0; k < tatva_kernel_count; k++) {
        sbi_print("KERNEL_CYCLES: ");
        sbi_print_uint((uint64_t)k);
        sbi_print(" ");
        sbi_print(tatva_kernel_names[k]);
        sbi_print(" ");
        sbi_print_uint(tatva_kernel_calls[k]);
        sbi_print(" ");
        sbi_print_uint(tatva_kernel_cycles[k]);
        sbi_print("\\n");
    }
"""


def _profile_prologue(kernel_order: list[str]) -> list[str]:
    """
    The accumulator table and cycle reader that model_run.c's call sites use.

    Accumulators are uint64_t because rdcycle is a 64-bit counter; a 32-bit total
    wraps after ~43 seconds of simulated time at the nominal 100 MHz, which a long
    timed loop on a real model will reach.

    Nothing zeroes .bss on this target -- start.S sets the stack pointer and tails
    straight into main -- so the table is not assumed to start at zero. main.c calls
    tatva_profile_reset() after the warm-up loop, and that call, not the loader, is
    what makes the counts trustworthy.
    """
    count = len(kernel_order)
    names = ",\n".join(f'    "{name}"' for name in kernel_order)
    return [
        "// --- Per-kernel cycle attribution (compile_model(profile=True)) ---",
        "static inline uint64_t tatva_read_cycles(void) {",
        "    uint64_t cycles;",
        "#if __riscv_xlen == 64",
        '    asm volatile ("rdcycle %0" : "=r" (cycles) : : "memory");',
        "#else",
        "    uint32_t cycle_h, cycle_l, cycle_h_again;",
        "    do {",
        '        asm volatile ("rdcycleh %0" : "=r" (cycle_h) : : "memory");',
        '        asm volatile ("rdcycle %0" : "=r" (cycle_l) : : "memory");',
        '        asm volatile ("rdcycleh %0" : "=r" (cycle_h_again) : : "memory");',
        "    } while (cycle_h != cycle_h_again);",
        "    cycles = (((uint64_t)cycle_h) << 32) | cycle_l;",
        "#endif",
        "    return cycles;",
        "}",
        "",
        f"#define TATVA_KERNEL_COUNT {count}",
        "const int tatva_kernel_count = TATVA_KERNEL_COUNT;",
        "uint64_t tatva_kernel_cycles[TATVA_KERNEL_COUNT] = {0};",
        "uint64_t tatva_kernel_calls[TATVA_KERNEL_COUNT] = {0};",
        "const char* const tatva_kernel_names[TATVA_KERNEL_COUNT] = {",
        names,
        "};",
        "",
        "void tatva_profile_reset(void) {",
        "    for (int k = 0; k < TATVA_KERNEL_COUNT; k++) {",
        "        tatva_kernel_cycles[k] = 0;",
        "        tatva_kernel_calls[k] = 0;",
        "    }",
        "}",
        "",
    ]


class KernelProfile:
    """
    Cycles attributed to one generated kernel over the timed runs.

    `cycles` and `calls` are totals across all TIMED_COUNT iterations, not per-run
    averages, because that is what the C side can accumulate without dividing on a
    target that has no libm. Divide on the host, where it is cheap and visible.
    """

    __slots__ = ("calls", "cycles", "index", "name")

    def __init__(self, index: int, name: str, calls: int, cycles: int):
        self.index = index
        self.name = name
        self.calls = calls
        self.cycles = cycles

    @property
    def cycles_per_call(self) -> float:
        return self.cycles / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "calls": self.calls,
            "cycles": self.cycles,
            "cycles_per_call": self.cycles_per_call,
        }

    def __repr__(self) -> str:
        return f"KernelProfile(name={self.name!r}, calls={self.calls}, cycles={self.cycles})"


class ExecutionEnvironment(Enum):
    QEMU_SIM = "QEMU_SIM"
    REAL_HW = "REAL_HW"


class CompiledArtifact:
    """
    Wraps the paths and metadata of a compiled model ELF binary.
    """
    def __init__(
        self,
        elf_path: str,
        build_dir: str,
        variant: TargetVariant,
        ram_bytes: int = 0,
        run_count: int = 1,
    ):
        self.elf_path = elf_path
        self.build_dir = build_dir
        self.variant = variant
        # Size of the linker's RAM region for this build. run_and_measure gives QEMU at
        # least this much, because the board's default is 128 MiB and a model whose
        # weights exceed that would load into memory the machine does not have. Defaults
        # to 0 for artifacts built by something other than compile_model; the runner
        # falls back to the board default in that case.
        self.ram_bytes = ram_bytes
        # Total inferences baked into main.c (warm-up plus timed). The QEMU timeout
        # scales with it, because thirteen inferences take thirteen times as long as one.
        self.run_count = run_count


class MeasurementResult:
    """
    Stores metrics and metadata from a performance measurement run.
    """
    def __init__(
        self,
        environment: str,
        simulated: bool,
        mean_ms: float,
        median_ms: float,
        p95_ms: float,
        raw_samples_ms: list[float],
        raw_output: str = "",
        units: str = "ms",
        kernel_profiles: list["KernelProfile"] | None = None,
        total_cycles: int = 0,
    ):
        self.environment = environment
        self.simulated = simulated
        self.mean_ms = mean_ms
        self.median_ms = median_ms
        self.p95_ms = p95_ms
        self.raw_samples_ms = raw_samples_ms
        self.raw_output = raw_output
        self.units = units
        # Empty unless the artifact was built with compile_model(profile=True).
        self.kernel_profiles = kernel_profiles or []
        # Sum of the RUN_CYCLES samples, i.e. the same window the kernel totals cover.
        self.total_cycles = total_cycles

    @property
    def attributed_cycles(self) -> int:
        """Cycles the per-kernel table accounts for. 0 when profiling was off."""
        return sum(k.cycles for k in self.kernel_profiles)

    @property
    def attribution_coverage(self) -> float:
        """
        Fraction of the measured window the per-kernel table explains.

        Exactly 0.0 when the artifact was built unprofiled -- there is no table, so
        nothing is attributed. Read it together with `kernel_profiles`: an empty list
        means "profiling was off", not "the table explained nothing".

        When profiling was on it lands slightly below 1.0. The residue is
        tvmgen_default_run's own prologue, the DLTensor pointer stores, and the
        read_cycles pair that brackets each call.
        """
        return self.attributed_cycles / self.total_cycles if self.total_cycles else 0.0

    def to_json(self) -> str:
        """
        Serialize results into a machine-readable JSON format.
        """
        return json.dumps(
            {
                "environment": self.environment,
                "simulated": self.simulated,
                "mean_ms": self.mean_ms,
                "median_ms": self.median_ms,
                "p95_ms": self.p95_ms,
                "raw_samples_ms": self.raw_samples_ms,
                "units": self.units,
                "total_cycles": self.total_cycles,
                "attributed_cycles": self.attributed_cycles,
                "attribution_coverage": self.attribution_coverage,
                "kernel_profiles": [k.to_dict() for k in self.kernel_profiles],
            },
            indent=2,
        )


class BaselineResult:
    """
    Wraps reference parity verification and latency results.
    """
    def __init__(
        self,
        latency_result: MeasurementResult,
        parity_passed: bool,
        tolerance: float,
        ref_logits: list[float],
        target_logits: list[float],
    ):
        self.latency_result = latency_result
        self.parity_passed = parity_passed
        self.tolerance = tolerance
        self.ref_logits = ref_logits
        self.target_logits = target_logits


def _search_bin_dirs(install_name: str, legacy_dir: str) -> list[str]:
    """
    Directories to search for a toolchain binary, in priority order.

    `tatva setup` installs into a per-user tools directory so that a pip-installed TATVA
    and a git checkout share one copy. That comes first: someone who deliberately ran
    `tatva setup` wants the copy they installed.

    The bundled directory comes next. It is the one the desktop build ships with, so it
    is what makes stage 05 work on an unpacked zip with no network. It sits below the
    per-user path rather than above it so that installing a newer toolchain by hand still
    takes effect without anyone having to delete files out of the app folder.

    The legacy `<repo>/riscv-toolchain` and `<repo>/qemu` paths are still searched last so
    nobody who ran the old setup_env.py has to re-download half a gigabyte.
    """
    from tatva.toolchain import tools_dir

    dirs = [os.path.join(tools_dir(), install_name, "bin")]

    bundled = bundled_tools_dir()
    if bundled:
        dirs.append(os.path.join(bundled, install_name, "bin"))

    dirs.append(os.path.join(PROJECT_DIR, legacy_dir, "bin"))
    return dirs


def _first_existing_exe(bin_dirs: list[str], exe_names: list[str]) -> str | None:
    for bin_dir in bin_dirs:
        for name in exe_names:
            for candidate in (os.path.join(bin_dir, name + ".exe"), os.path.join(bin_dir, name)):
                if os.path.isfile(candidate):
                    return candidate
    return None


def find_riscv_gcc() -> tuple[str | None, str | None]:
    """
    Find the RISC-V GCC cross-compiler on PATH, in the tools directory, or in the repo.

    This is the single resolver for the compiler. Stage 05 calls it, `tatva doctor` calls
    it, and the Diagnostics page reaches it through ToolchainManager -- because anything
    that reports on the toolchain has to report what the build will actually run.

    The two names below are the only ones accepted. riscv64-linux-gnu-gcc is deliberately
    not among them: compile_model links bare metal (-ffreestanding -nostdlib -T link.ld),
    which a Linux/glibc cross-compiler is not configured for.
    """
    candidates = ["riscv-none-elf-gcc", "riscv64-unknown-elf-gcc"]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return name, path
        path_exe = shutil.which(name + ".exe")
        if path_exe:
            return name, path_exe

    found = _first_existing_exe(_search_bin_dirs("riscv-none-elf-gcc", "riscv-toolchain"), candidates)
    if found:
        # Report the binary that was actually found. Returning candidates[0] regardless
        # labelled a riscv64-unknown-elf-gcc install as riscv-none-elf-gcc on the
        # Diagnostics page, under a column headed "resolved path".
        return os.path.splitext(os.path.basename(found))[0], found

    return None, None


def find_qemu(bitness: int = 64) -> tuple[str | None, str | None]:
    """
    Find the QEMU emulator binary matching the requested target bitness (32 or 64).

    System mode only. The user-mode `qemu-riscv64` is not an accepted substitute: the
    measurement boots an ELF with `-machine virt` and reads rdcycle on the target, which
    a user-mode emulator cannot do.
    """
    name_prefix = f"qemu-system-riscv{bitness}"
    path = shutil.which(name_prefix)
    if path:
        return name_prefix, path
    path_exe = shutil.which(name_prefix + ".exe")
    if path_exe:
        return name_prefix, path_exe

    found = _first_existing_exe(_search_bin_dirs("qemu-riscv", "qemu"), [name_prefix])
    if found:
        return name_prefix, found

    return None, None


def map_dtype(dtype: str) -> tuple[int, int]:
    """
    Maps a type name to DLPack DataType structure (code, bits).
    """
    dtype = str(dtype)
    if dtype == "int64":
        return 0, 64
    elif dtype == "float32":
        return 2, 32
    elif dtype == "int32":
        return 0, 32
    elif dtype == "int8":
        return 0, 8
    elif dtype == "uint8":
        return 1, 8
    elif dtype == "bool":
        return 6, 8
    else:
        if "int" in dtype:
            if "64" in dtype:
                return 0, 64
            elif "32" in dtype:
                return 0, 32
            else:
                return 0, 8
        return 2, 32


def c_type_for_dtype(dtype: str) -> str:
    """
    Map a tensor dtype name onto the C type used for its static buffer.
    """
    dtype = str(dtype)
    return {
        "float32": "float",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int32_t",
        "int16": "int16_t",
        "int8": "int8_t",
        "uint64": "uint64_t",
        "uint32": "uint32_t",
        "uint16": "uint16_t",
        "uint8": "uint8_t",
        "bool": "uint8_t",
    }.get(dtype, "float" if "float" in dtype else "int32_t")


def c_identifier(name: str) -> str:
    """
    Turn a tensor name into a safe C identifier.

    ONNX permits names such as 'input.1' or '/encoder/Add_output_0', which TVM
    carries through into its variable names. Emitting those verbatim produces C
    that does not parse.

    Delegates to compiler.c_safe_name, which is the same rule import_model applies to
    the graph itself. These used to be one function copied into two files; if they ever
    disagreed, main.c and operators.c would spell the same tensor differently and the
    link would fail on a name neither file appears to contain.
    """
    return c_safe_name(name)


def input_fill_kind(name: str, dtype: str) -> str:
    """
    Decide how a given model input is filled with dummy data.

    Returns one of 'ones', 'zeros', 'mod5' or 'ramp'. Both the host reference and
    the on-target harness derive their values from this, so they cannot drift apart.

    The name is normalized through c_identifier() first because TVM rewrites ONNX
    names that are not valid identifiers (`attention.mask` -> `attention_mask`).
    Without this, the host would classify the original name and the target the
    rewritten one, and the two sides could fill the same tensor differently.
    """
    lname = c_identifier(name).lower()
    if "attention_mask" in lname or lname.endswith("_mask") or lname == "mask":
        return "ones"
    if "token_type" in lname or "segment_id" in lname:
        return "zeros"
    if "float" in str(dtype):
        return "ramp"
    return "mod5"


def default_input_array(name: str, shape: Any, dtype: str) -> np.ndarray:
    """
    Build the deterministic dummy tensor for one model input.

    'ramp' deliberately uses quarter steps: every value is exactly representable
    in binary floating point, so the host array and the value the target computes
    from the same formula are bit-identical rather than merely close.
    """
    shape = tuple(int(d) for d in shape)
    n = int(np.prod(shape)) if shape else 1
    kind = input_fill_kind(name, dtype)

    if kind == "ones":
        flat = np.ones(n, dtype=np.float64)
    elif kind == "zeros":
        flat = np.zeros(n, dtype=np.float64)
    elif kind == "ramp":
        flat = ((np.arange(n, dtype=np.float64) % 9.0) - 4.0) * 0.25
    else:
        flat = np.arange(n, dtype=np.float64) % 5.0

    return flat.reshape(shape).astype(np.dtype(dtype))


def _c_fill_expression(kind: str, c_type: str) -> str:
    """
    The C expression, in terms of the loop variable `i`, matching default_input_array().
    """
    if kind == "ones":
        return f"({c_type})1"
    if kind == "zeros":
        return f"({c_type})0"
    if kind == "ramp":
        return f"({c_type})((((float)(i % 9)) - 4.0f) * 0.25f)"
    return f"({c_type})(i % 5)"


# Prelude injected at the top of the TVM-generated operators.c when the optimized
# softmax kernel is swapped in. The scratch buffer is static rather than a VLA: a
# variable-length array of `cols` floats lives on a 64 KB bare-metal stack and
# silently corrupts memory the moment a row is wide enough.
SOFTMAX_PRELUDE = """/* --- TATVA optimized softmax support --- */
extern void sbi_print(const char* str);
#define TATVA_SOFTMAX_MAX_COLS 8192
static float tatva_softmax_exp_buf[TATVA_SOFTMAX_MAX_COLS];
/* --- end TATVA softmax support --- */
"""

# Shared shape-handling preamble for both softmax variants. Softmax is normalized
# over the LAST axis, so any leading dimensions are folded into the row count --
# reading shape[1] as the column count silently ignores most of an N-D tensor.
_SOFTMAX_HEADER = """TVM_DLL int32_t __tvm_ffi_softmax(void* self_handle, void* args, int32_t num_args, void* result) {
  if (num_args != 2) return -1;
  void* var_input = (((TVMFFIAny*)args)[0].type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr);
  void* var_T_softmax_norm = (((TVMFFIAny*)args)[1].type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr);

  DLTensor* t_in = (DLTensor*)var_input;
  DLTensor* t_out = (DLTensor*)var_T_softmax_norm;
  float* input_ptr = (float*)t_in->data;
  float* T_softmax_norm = (float*)t_out->data;

  int32_t ndim = (int32_t)t_in->ndim;
  if (ndim < 1) {
    sbi_print("[TATVA] optimized softmax: rank-0 tensor is not supported\\n");
    return -1;
  }
  int64_t* shape = t_in->shape;
  int32_t cols = (int32_t)shape[ndim - 1];
  int32_t rows = 1;
  for (int32_t d = 0; d < ndim - 1; ++d) {
    rows *= (int32_t)shape[d];
  }
  if (cols <= 0 || cols > TATVA_SOFTMAX_MAX_COLS) {
    sbi_print("[TATVA] optimized softmax: row width exceeds TATVA_SOFTMAX_MAX_COLS\\n");
    return -1;
  }
  float* local_exp = tatva_softmax_exp_buf;
"""

# Scalar Schraudolph fast-exponential softmax.
SOFTMAX_SCALAR = _SOFTMAX_HEADER + """
  for (int32_t i0 = 0; i0 < rows; ++i0) {
    float* in_row = input_ptr + (int64_t)i0 * cols;
    float* out_row = T_softmax_norm + (int64_t)i0 * cols;

    float max_val = in_row[0];
    for (int32_t k = 1; k < cols; ++k) {
      if (in_row[k] > max_val) {
        max_val = in_row[k];
      }
    }

    float sum = 0.0f;
    for (int32_t k = 0; k < cols; ++k) {
      float val = in_row[k] - max_val;
      if (val < -15.0f) {
        local_exp[k] = 0.0f;
      } else {
        union {
          float f;
          int32_t i;
        } u;
        u.i = (int32_t)(val * 12102203.0f + 1065353216.0f);
        local_exp[k] = u.f;
      }
      sum += local_exp[k];
    }

    float inv_sum = 1.0f / sum;
    for (int32_t k = 0; k < cols; ++k) {
      out_row[k] = local_exp[k] * inv_sum;
    }
  }
  return 0;
}"""

# RVV 1.0 vectorized variant of the same kernel.
SOFTMAX_VECTOR = _SOFTMAX_HEADER + """
  for (int32_t i0 = 0; i0 < rows; ++i0) {
    float* in_row = input_ptr + (int64_t)i0 * cols;
    float* out_row = T_softmax_norm + (int64_t)i0 * cols;

    // RVV 1.0 Vector Max Reduction
    size_t vl;
    float max_val = in_row[0];
    vfloat32m1_t v_max = __riscv_vfmv_s_f_f32m1(max_val, 1);
    for (int32_t k = 0; k < cols; k += vl) {
      vl = __riscv_vsetvl_e32m1(cols - k);
      vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_row + k, vl);
      v_max = __riscv_vfredmax_vs_f32m1_f32m1(v_in, v_max, vl);
    }
    max_val = __riscv_vfmv_f_s_f32m1_f32(v_max);

    // RVV 1.0 Vector Exponent & Sum Accumulation.
    //
    // The -15.0f clamp is not an optimization, it is what keeps the Schraudolph
    // trick in range. Once (x - max) drops below about -88 the integer
    // (x-max)*12102203 + 1065353216 goes negative, and reinterpreting a negative
    // int32 as a float yields a NaN or a large NEGATIVE "exponential" that poisons
    // the sum and inverts the sign of the entire row. SOFTMAX_SCALAR guards this
    // with `if (val < -15.0f)`; the vector path must apply the identical predicate
    // to the *unscaled* difference, hence vmflt + a vfmerge of +0.0f. Clamping does
    // cost a little accuracy against a true exponential for (x - max) in (-88, -15),
    // where the unclamped path returned a small but correct value -- that is the
    // price of matching the scalar kernel, and it is bounded by ~3e-7 per lane.
    //
    // vfmacc rather than vfmul + vfadd: GCC contracts the scalar
    // `val * 12102203.0f + 1065353216.0f` into a single fmadd.s, and two separate
    // roundings disagree with one fused rounding by 1 ULP on ~6% of lanes.
    vfloat32m1_t v_sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (int32_t k = 0; k < cols; k += vl) {
      vl = __riscv_vsetvl_e32m1(cols - k);
      vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_row + k, vl);
      vfloat32m1_t v_diff = __riscv_vfsub_vf_f32m1(v_in, max_val, vl);
      vbool32_t v_underflow = __riscv_vmflt_vf_f32m1_b32(v_diff, -15.0f, vl);
      vfloat32m1_t v_base = __riscv_vfmv_v_f_f32m1(1065353216.0f, vl);
      vfloat32m1_t v_offset = __riscv_vfmacc_vf_f32m1(v_base, 12102203.0f, v_diff, vl);
      vint32m1_t v_int = __riscv_vfcvt_rtz_x_f_v_i32m1(v_offset, vl);
      vfloat32m1_t v_exp = __riscv_vreinterpret_v_i32m1_f32m1(v_int);
      v_exp = __riscv_vfmerge_vfm_f32m1(v_exp, 0.0f, v_underflow, vl);
      __riscv_vse32_v_f32m1(local_exp + k, v_exp, vl);
      v_sum = __riscv_vfredusum_vs_f32m1_f32m1(v_exp, v_sum, vl);
    }
    float sum = __riscv_vfmv_f_s_f32m1_f32(v_sum);

    // RVV 1.0 Vector Normalization Division
    float inv_sum = 1.0f / sum;
    for (int32_t k = 0; k < cols; k += vl) {
      vl = __riscv_vsetvl_e32m1(cols - k);
      vfloat32m1_t v_e = __riscv_vle32_v_f32m1(local_exp + k, vl);
      vfloat32m1_t v_out = __riscv_vfmul_vf_f32m1(v_e, inv_sum, vl);
      __riscv_vse32_v_f32m1(out_row + k, v_out, vl);
    }
  }
  return 0;
}"""

# Matches every generated softmax entry point, whatever TVM chose to number them.
# The trailing '{' keeps it from matching the forward declaration.
_SOFTMAX_DEF_RE = re.compile(
    r"TVM_DLL int32_t __tvm_ffi_(softmax\d*)"
    r"\(void\* self_handle, void\* args, int32_t num_args, void\* result\) \{"
)


def inject_optimized_softmax(operators_c: str, is_vector: bool) -> tuple[str, int]:
    """
    Replace every TVM-generated softmax kernel in `operators_c` with TATVA's own.

    Returns the rewritten source and the number of kernels replaced. Replacing zero
    kernels is reported to the caller rather than passed off as a successful
    optimization -- an unpatched build is just the baseline wearing a different name.
    """
    custom = SOFTMAX_VECTOR if is_vector else SOFTMAX_SCALAR

    patched = 0
    search_from = 0
    while True:
        match = _SOFTMAX_DEF_RE.search(operators_c, search_from)
        if match is None:
            break

        start, pos = match.start(), match.end()
        depth = 1
        while depth > 0 and pos < len(operators_c):
            ch = operators_c[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        if depth != 0:
            raise CompilationError(
                stage="softmax-injection",
                command="operators.c rewrite",
                details=(
                    f"Unbalanced braces while scanning the body of __tvm_ffi_{match.group(1)}; "
                    "refusing to emit corrupted C."
                ),
            )

        replacement = custom.replace("__tvm_ffi_softmax", f"__tvm_ffi_{match.group(1)}")
        operators_c = operators_c[:start] + replacement + operators_c[pos:]
        search_from = start + len(replacement)
        patched += 1

    if patched:
        prelude = SOFTMAX_PRELUDE
        if is_vector:
            prelude = "#include <riscv_vector.h>\n" + prelude
        operators_c = prelude + operators_c

    return operators_c, patched


def verify_target(variant: TargetVariant) -> dict[str, Any]:
    """
    Compile a tiny hello-world C program with the variant's march/mabi via the RISC-V GCC,
    execute it under matching QEMU emulator, and confirm expected console output.
    Returns:
        dict: {"status": "ok"/"fail", "output": str, "error": str}
    """
    _gcc_name, gcc_path = find_riscv_gcc()
    _qemu_name, qemu_path = find_qemu(variant.bitness)

    if not gcc_path:
        return {
            "status": "fail",
            "output": "",
            "error": "RISC-V GCC cross-compiler binary not found.",
        }
    if not qemu_path:
        return {
            "status": "fail",
            "output": "",
            "error": f"QEMU RISC-V {variant.bitness}-bit emulator binary not found.",
        }

    with tempfile.TemporaryDirectory(dir=build_root()) as tmpdir:
        main_c_path = os.path.join(tmpdir, "main.c")
        start_s_path = os.path.join(tmpdir, "start.S")
        link_ld_path = os.path.join(tmpdir, "link.ld")
        elf_path = os.path.join(tmpdir, "test.elf")

        with open(start_s_path, "w") as f:
            f.write(START_S)
        with open(link_ld_path, "w") as f:
            # A hello-world needs nothing beyond the floor.
            f.write(render_link_ld())
        with open(main_c_path, "w") as f:
            f.write(MAIN_C_HELLO.replace("TARGET_NAME", f'"{variant.name}"'))

        compile_cmd = [
            gcc_path,
            "-O2",
            f"-march={variant.gcc_march}",
            f"-mabi={variant.gcc_mabi}",
            "-mcmodel=medany",
            "-ffreestanding",
            "-nostdlib",
            "-T",
            link_ld_path,
            start_s_path,
            main_c_path,
            "-o",
            elf_path,
            "-lgcc",
        ]

        try:
            res = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=10)
            if res.returncode != 0:
                return {
                    "status": "fail",
                    "output": "",
                    "error": f"Compilation failed:\nStdout: {res.stdout}\nStderr: {res.stderr}",
                }
        except Exception as e:
            return {
                "status": "fail",
                "output": "",
                "error": f"Compilation process error: {e}",
            }

        qemu_cmd = [
            qemu_path,
            "-M",
            "virt",
        ]
        if variant.gcc_march.endswith("v") or "gcv" in variant.gcc_march.lower():
            qemu_cmd.extend(["-cpu", "rv64,v=true,vlen=128"])
        qemu_cmd.extend(
            [
                "-kernel",
                elf_path,
                "-nographic",
                "-icount",
                "shift=0",
            ]
        )

        try:
            qemu_res = subprocess.run(qemu_cmd, capture_output=True, text=True, timeout=15)
            output = qemu_res.stdout
            expected_pattern = f"Hello from Target: {variant.name}"
            if expected_pattern in output:
                return {"status": "ok", "output": output, "error": ""}
            else:
                return {
                    "status": "fail",
                    "output": output,
                    "error": f"Expected pattern '{expected_pattern}' not found in QEMU output:\n{output}",
                }
        except Exception as e:
            return {"status": "fail", "output": "", "error": f"QEMU execution error: {e}"}


def static_shape_tensor_values(call: Any) -> list[int] | None:
    """The int64 elements of a `shape_to_tensor` call whose shape is known at compile time.

    A `Shape` node in the ONNX graph becomes the pair `lv = R.shape_of(x)` /
    `lv1 = R.call_pure_packed("relax.run.shape_to_tensor", lv)`. Neither survives
    `LegalizeOps` as a `call_tir`, so the harness emitter has no kernel to call for
    either -- but `lv1` is a real int64 tensor with a real buffer, and a downstream
    `take` will happily dereference it. In `models/model.onnx` that is
    `Gather(Shape(x), 0)`, reading a buffer nothing had written.

    The values are not actually unknown. Shape inference has already resolved the
    extents to constants by this point, which is why they can be emitted as stores
    instead of computed -- exactly what the capability table has always claimed for
    `shape_to_tensor`. Returns None if this is not that call, or if any extent is
    still symbolic, in which case the caller refuses rather than guessing.
    """
    from tvm import relax

    if not isinstance(call, relax.Call):
        return None
    if str(getattr(call.op, "name", call.op)) != "relax.call_pure_packed":
        return None
    if len(call.args) < 2:
        return None
    callee = call.args[0]
    if str(getattr(callee, "global_symbol", "")) != "relax.run.shape_to_tensor":
        return None
    sinfo = call.args[1].struct_info
    values = getattr(sinfo, "values", None)
    if values is None:
        return None
    out: list[int] = []
    for value in values:
        # An IntImm carries a Python int; a symbolic tir.Var carries nothing, and a
        # dimension the compiler cannot name is a dimension it must not invent.
        extent = getattr(value, "value", None)
        if not isinstance(extent, int):
            return None
        out.append(extent)
    return out


def compile_model(
    model_ir: ModelIR,
    variant: TargetVariant,
    output_dir: str | None = None,
    # One warm-up, not three. Warm-up exists to settle caches and fault in the
    # workspace allocator, and neither happens here: the target is bare metal with a
    # statically placed workspace and no MMU, and -icount counts instructions rather
    # than modelling a cache. Measured on models/model.onnx, a build with zero warm-up
    # reports 0.7309 ms on its very first timed run -- the same figure a three-warm-up
    # build reports -- so the extra two runs bought nothing and cost a full inference
    # each. The one that remains is cheap insurance, and if it ever turns out to
    # matter the timed loop's stability check will see run 1 disagree with run 2 and
    # take the full sample set.
    warmup_count: int = 1,
    timed_count: int = 10,
    profile: bool = False,
) -> CompiledArtifact:
    """
    Compiles a ModelIR module to C code via TVM Relax, maps operations and weights
    to static variables, writes bare-metal wrappers, and cross-compiles to a RISC-V ELF.

    Set `profile=True` to bracket every generated kernel call with rdcycle and print a
    `KERNEL_CYCLES:` line per kernel, which run_and_measure turns into MeasurementResult
    .kernel_profiles. It is off by default and the generated sources are byte-identical
    to the unprofiled build when it is off, so the number TATVA reports as the model's
    latency is never a number measured with the profiler switched on.
    """
    import tvm
    import tvm_ffi
    from tvm import relax

    if output_dir is None:
        root = build_root()
        _prune_old_builds(root)
        build_dir = tempfile.mkdtemp(prefix="tatva_build_", dir=root)
    else:
        build_dir = output_dir
        os.makedirs(build_dir, exist_ok=True)

    seq = tvm.transform.Sequential([relax.transform.LegalizeOps()])
    mod_legalized = seq(model_ir.mod)

    try:
        lib = relax.build(mod_legalized, target="c")
        operators_c = lib.mod.imports[0].inspect_source()
    except Exception as e:
        raise CompilationError(
            stage="tvm-lowering",
            command="relax.build(mod, target='c')",
            stderr=str(e),
            details="TVM Relax could not lower this model to C.",
        ) from e

    func = mod_legalized["main"]
    constants_list = []

    def find_constants(expr):
        if isinstance(expr, relax.Constant):
            arr = expr.data.numpy()
            for c in constants_list:
                c_arr = c.data.numpy()
                if c_arr.shape == arr.shape and c_arr.dtype == arr.dtype and np.array_equal(c_arr, arr):
                    return
            constants_list.append(expr)
        elif isinstance(expr, relax.Call):
            for arg in expr.args:
                find_constants(arg)
        elif isinstance(expr, relax.Tuple):
            for field in expr.fields:
                find_constants(field)

    for block in func.body.blocks:
        for binding in block.bindings:
            find_constants(binding.value)

    vars_to_process = []
    for param in func.params:
        vars_to_process.append((param.name_hint, param.struct_info, True))
    for block in func.body.blocks:
        for binding in block.bindings:
            vars_to_process.append((binding.var.name_hint, binding.var.struct_info, False))

    tensors_mapped = {}
    curr_offset = 0
    used_cnames: set[str] = set()

    for name, sinfo, is_input in vars_to_process:
        if not hasattr(sinfo, "shape") or sinfo.shape is None:
            continue
        try:
            shape = [int(dim) for dim in sinfo.shape]
        except (TypeError, ValueError):
            # A symbolic dimension. NonZero's output is R.Tensor((2, nonzero_numbers)),
            # whose extent is known only once the data is seen, and int() on that raises
            # a bare TypeError out of the middle of code generation. Leaving the tensor
            # unmapped routes it to the diagnosed error below instead.
            continue
        dtype = str(sinfo.dtype)

        dtype_code, dtype_bits = map_dtype(dtype)
        num_elements = int(np.prod(shape)) if shape else 1
        size_bytes = num_elements * (dtype_bits // 8)

        # ONNX names survive into TVM's var names and are not always valid C.
        cname = c_identifier(name)
        if cname in used_cnames:
            suffix = 2
            while f"{cname}_{suffix}" in used_cnames:
                suffix += 1
            cname = f"{cname}_{suffix}"
        used_cnames.add(cname)

        tensors_mapped[name] = {
            "name": name,
            "cname": cname,
            "shape": shape,
            "ndim": len(shape),
            "dtype": dtype,
            "dtype_code": dtype_code,
            "dtype_bits": dtype_bits,
            "num_elements": num_elements,
            "size_bytes": size_bytes,
            "is_input": is_input,
        }

        size_aligned = (size_bytes + 15) & ~15
        if not is_input:
            tensors_mapped[name]["offset"] = curr_offset
            curr_offset += size_aligned

    # The graph inputs, in declaration order. This list -- not a fixed BERT triple --
    # defines the tvmgen_default_run() signature and the harness buffers.
    input_infos = [
        tensors_mapped[p.name_hint] for p in func.params if p.name_hint in tensors_mapped
    ]
    if not input_infos:
        raise CompilationError(
            stage="harness-generation",
            command="compile_model",
            details="The model exposes no statically shaped graph inputs, so no benchmark harness can be generated.",
        )

    # Constant tensors go into the build as a raw binary blob that the assembler pulls
    # in with .incbin, not as a C initializer list.
    #
    # Spelling the weights out as decimal text cost about 4.1 bytes of source per byte
    # of model -- all_minilm_l6_v2.onnx turned 90.3 MB of weights into a 371.8 MB
    # weights.h -- and then made the C frontend parse all of it, which took cc1 past
    # 3 GB of resident memory and minutes of wall clock before a single kernel was
    # compiled. The assembler copies the blob through verbatim instead, so emission is
    # a file write and the bytes are exact by construction rather than by round-tripping
    # through decimal.
    #
    # The symbol names are unchanged, so model_run.c still writes `.data = constant_data_N`
    # and neither it nor the TVM-generated operators.c can tell the difference.
    weights_bin_path = os.path.join(build_dir, "weights.bin")

    weights_h_lines = [
        "#ifndef WEIGHTS_H",
        "#define WEIGHTS_H",
        "#include <stdint.h>",
        "",
        "// Storage lives in weights.S, which .incbin's weights.bin. The declarations are",
        "// left as incomplete array types on purpose: nothing takes sizeof() of them, and",
        "// an incomplete type keeps the header honest about where the bytes come from.",
        "",
    ]
    weights_s_lines = [
        "/* Generated by TATVA -- storage for the model's constant tensors, taken",
        "   verbatim from weights.bin. */",
        "    .section .rodata",
        "",
    ]

    # GNU as reads escape sequences inside the .incbin string, so a Windows path with
    # backslashes would be mangled. Absolute, forward-slashed, so the directive does not
    # depend on the assembler's working directory either.
    blob_ref = weights_bin_path.replace("\\", "/")

    weights_bytes = 0
    with open(weights_bin_path, "wb") as blob:
        for idx, const in enumerate(constants_list):
            arr = np.ascontiguousarray(const.data.numpy())

            # 16 satisfies the alignment of every type map_dtype knows about, and the
            # padding is written into the blob so the .incbin offsets below stay in step
            # with the .balign directives.
            pad = -weights_bytes % 16
            if pad:
                blob.write(b"\x00" * pad)
                weights_bytes += pad

            payload = arr.tobytes()
            start = weights_bytes
            blob.write(payload)
            weights_bytes += len(payload)

            weights_h_lines.append(f"extern {c_type_for_dtype(str(arr.dtype))} constant_data_{idx}[];")
            weights_s_lines.extend(
                [
                    "    .balign 16",
                    f"    .globl constant_data_{idx}",
                    f"constant_data_{idx}:",
                    f'    .incbin "{blob_ref}", {start}, {len(payload)}',
                    "",
                ]
            )

    weights_h_lines.append("")
    weights_h_lines.append("#endif // WEIGHTS_H")

    with open(os.path.join(build_dir, "weights.h"), "w") as f:
        f.write("\n".join(weights_h_lines))
    with open(os.path.join(build_dir, "weights.S"), "w") as f:
        f.write("\n".join(weights_s_lines))

    model_run_lines = [
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "#include <stddef.h>",
        '#include "tvm/runtime/c_backend_api.h"',
        '#include "tvm/ffi/c_api.h"',
        '#include "weights.h"',
        "",
        "extern void sbi_print(const char* str);",
        "",
        "// Global Memory Pool for activations",
        f"static uint8_t global_pool[{max(16, curr_offset)}] __attribute__((aligned(16)));",
        "",
    ]

    declared_funcs = set()
    # Same predicate as the call-site emission below, so the profile index space and
    # the call sites can never disagree about which kernels exist. Execution order,
    # not sorted order, because a profile is read against the shape of the graph.
    kernel_order: list[str] = []
    for block in func.body.blocks:
        for binding in block.bindings:
            if isinstance(binding.value, relax.Call) and isinstance(binding.value.args[0], tvm.ir.GlobalVar):
                func_name = binding.value.args[0].name_hint
                if func_name not in declared_funcs:
                    kernel_order.append(func_name)
                declared_funcs.add(func_name)

    kernel_index = {name: i for i, name in enumerate(kernel_order)}

    for func_name in sorted(declared_funcs):
        model_run_lines.append(
            f"extern int32_t __tvm_ffi_{func_name}(void* self_handle, void* args, int32_t num_args, void* result);"
        )
    model_run_lines.append("")

    if profile:
        model_run_lines.extend(_profile_prologue(kernel_order))

    for info in tensors_mapped.values():
        shape_str = ", ".join(str(x) for x in info["shape"]) or "1"
        model_run_lines.append(f"static int64_t shape_{info['cname']}[] = {{ {shape_str} }};")

        data_init = "NULL" if info["is_input"] else f"&global_pool[{info['offset']}]"
        model_run_lines.extend(
            [
                f"static DLTensor tensor_{info['cname']} = {{",
                f"    .data = {data_init},",
                "    .device = {1, 0},",
                f"    .ndim = {info['ndim']},",
                f"    .dtype = {{{info['dtype_code']}, {info['dtype_bits']}, 1}},",
                f"    .shape = shape_{info['cname']},",
                "    .strides = NULL,",
                "    .byte_offset = 0",
                "};",
                "",
            ]
        )

    for idx, const in enumerate(constants_list):
        arr = const.data.numpy()
        shape = list(arr.shape)
        shape_str = ", ".join(str(x) for x in shape)
        dtype = str(arr.dtype)
        dtype_code, dtype_bits = map_dtype(dtype)

        model_run_lines.append(f"static int64_t shape_constant_{idx}[] = {{ {shape_str} }};")
        model_run_lines.extend(
            [
                f"static DLTensor tensor_constant_{idx} = {{",
                f"    .data = constant_data_{idx},",
                "    .device = {1, 0},",
                f"    .ndim = {len(shape)},",
                f"    .dtype = {{{dtype_code}, {dtype_bits}, 1}},",
                f"    .shape = shape_constant_{idx},",
                "    .strides = NULL,",
                "    .byte_offset = 0",
                "};",
                "",
            ]
        )

    model_run_lines.extend(
        [
            "// Bump allocator with LIFO release. Each block carries a 16-byte header",
            "// holding the pool offset to restore and the block's own size, so a free",
            "// gives back exactly one block. The previous implementation reset the",
            "// offset to 0 on every free, which handed the entire pool back out while",
            "// earlier allocations from the same kernel were still live.",
            "#define TATVA_WS_HEADER 16u",
            f"static uint8_t workspace_pool[{WORKSPACE_POOL_BYTES}] __attribute__((aligned(16)));",
            "static size_t workspace_offset = 0;",
            "",
            "void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_or_handle, int dtype_bits) {",
            "    (void)device_type; (void)device_id; (void)dtype_code_or_handle; (void)dtype_bits;",
            "    size_t size = ((size_t)nbytes + 15u) & ~(size_t)15u;",
            "    if (workspace_offset + TATVA_WS_HEADER + size > sizeof(workspace_pool)) {",
            '        sbi_print("[TATVA] workspace pool exhausted; rebuild with a larger pool\\n");',
            "        return NULL;",
            "    }",
            "    size_t prev = workspace_offset;",
            "    uint8_t* block = &workspace_pool[workspace_offset + TATVA_WS_HEADER];",
            "    ((size_t*)block)[-2] = prev;",
            "    ((size_t*)block)[-1] = size;",
            "    workspace_offset += TATVA_WS_HEADER + size;",
            "    return block;",
            "}",
            "",
            "int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {",
            "    (void)device_type; (void)device_id;",
            "    if (ptr == NULL) {",
            "        return 0;",
            "    }",
            "    size_t prev = ((size_t*)ptr)[-2];",
            "    size_t size = ((size_t*)ptr)[-1];",
            "    if (prev + TATVA_WS_HEADER + size == workspace_offset) {",
            "        workspace_offset = prev;",
            "    }",
            "    // Out-of-order free: leave the pool untouched rather than reclaiming",
            "    // memory that later allocations are still using.",
            "    return 0;",
            "}",
            "",
            "void TVMFFIErrorSetRaisedFromCStrParts(const char* name, const char** parts, int32_t num_parts) {",
            '    sbi_print("\\n[TVM FFI Exception] ");',
            "    sbi_print(name);",
            '    sbi_print(": ");',
            "    for (int32_t i = 0; i < num_parts; i++) {",
            "        if (parts[i] != NULL) {",
            "            sbi_print(parts[i]);",
            "        }",
            "    }",
            '    sbi_print("\\n");',
            "}",
            "",
        ]
    )

    def no_buffer_error(what: str, var_name: str, sinfo: Any) -> CompilationError:
        """
        Explain why a value cannot be given a buffer, in terms of the model.

        Every tensor in a bare-metal build is allocated before the program runs, so a
        value whose size is not known at compile time has nowhere to live. There is more
        than one way to arrive here and they need different advice, so the reason is
        worked out rather than assumed -- this used to say "tuple of outputs" for every
        case, including NonZero, whose problem is the opposite.
        """
        if isinstance(sinfo, relax.TupleStructInfo):
            why = (
                "it is a tuple of several tensors. Operators returning more than one "
                "output -- BatchNorm, LSTM, Split -- are not supported: the harness "
                "allocates exactly one output buffer. Re-export the model with only the "
                "output you need, or drop the extra outputs from the graph."
            )
        elif getattr(sinfo, "shape", None) is None:
            why = "it has no shape, so no buffer can be sized for it."
        else:
            why = (
                f"its shape {sinfo.shape} contains a symbolic dimension, which means the "
                "size depends on the input data. NonZero, Unique, NonMaxSuppression and "
                "similar operators only learn their output size at run time, and every "
                "buffer in a bare-metal build is allocated before the program starts."
            )
        return CompilationError(
            stage="harness-generation",
            command="compile_model",
            details=f"{what} '{var_name}' cannot be allocated because {why}",
        )

    last_binding = func.body.blocks[-1].bindings[-1]
    out_var_name = last_binding.var.name_hint
    out_info = tensors_mapped.get(out_var_name)
    if out_info is None:
        raise no_buffer_error("The model's output", out_var_name, last_binding.var.struct_info)

    # Relax binds more than calls. `gv = lv` (a rename), `gv = x` (a pass-through graph)
    # and `gv = <constant>` (an operator that constant-folded away) are all ordinary
    # bindings, and the emitter below only writes code for call_tir. Everything else used
    # to be skipped in silence -- so when the graph's output was one of these forms, the
    # output buffer was never written and the model returned whatever was in BSS. Zeros,
    # with no error anywhere: Gather folded to a constant, LayerNorm (whose frontend
    # emits `gv = lv`) and Identity all ran to completion and reported a tensor of 0.0.
    #
    # A wrong answer that looks like a successful run is worse than a failed build, so
    # these bindings are resolved rather than ignored.
    alias_source: dict[str, Any] = {}
    for block in func.body.blocks:
        for binding in block.bindings:
            if isinstance(binding.value, (relax.Var, relax.Constant)):
                alias_source[binding.var.name_hint] = binding.value

    def resolve_alias(name: str) -> Any:
        """
        Follow a chain of renames back to whatever actually produces the value.

        Returns the name of the var a kernel writes (or a graph input), or the
        relax.Constant the chain ends at. A name that is not a rename comes back
        unchanged, so callers can use this unconditionally.
        """
        seen: set[str] = set()
        while name in alias_source and name not in seen:
            seen.add(name)
            src = alias_source[name]
            if isinstance(src, relax.Constant):
                return src
            name = src.name_hint
        return name

    def constant_index(const: relax.Constant) -> int:
        """Position of this constant in constants_list, matched by value as elsewhere."""
        arr = const.data.numpy()
        for i, c in enumerate(constants_list):
            c_arr = c.data.numpy()
            if c_arr.shape == arr.shape and c_arr.dtype == arr.dtype and np.array_equal(c_arr, arr):
                return i
        return -1

    # Every value something downstream actually reads: the operands of each call_tir the
    # loop below will emit, plus the graph's own output. Used to decide whether a binding
    # that produces no kernel can be passed over -- see the refusal in that loop.
    consumed: set[str] = set()

    def note_consumed(expr: Any) -> None:
        if not isinstance(expr, relax.Var):
            return
        src = resolve_alias(expr.name_hint)
        if not isinstance(src, relax.Constant):
            consumed.add(src)

    for block in func.body.blocks:
        for binding in block.bindings:
            val = binding.value
            if not isinstance(val, relax.Call) or not isinstance(val.args[0], tvm.ir.GlobalVar):
                continue
            operands = val.args[1].fields if isinstance(val.args[1], relax.Tuple) else [val.args[1]]
            for operand in operands:
                note_consumed(operand)
    out_src = resolve_alias(out_var_name)
    if not isinstance(out_src, relax.Constant):
        consumed.add(out_src)

    # Signature is derived from the graph's own inputs, in order, plus the output.
    run_params = ", ".join(f"void* {info['cname']}_ptr" for info in input_infos) + ", void* output_ptr"
    model_run_lines.append(f"int32_t tvmgen_default_run({run_params}) {{")
    for info in input_infos:
        model_run_lines.append(f"  tensor_{info['cname']}.data = {info['cname']}_ptr;")
    model_run_lines.extend([f"  tensor_{out_info['cname']}.data = output_ptr;", ""])

    for block in func.body.blocks:
        for binding in block.bindings:
            if not isinstance(binding.value, relax.Call):
                continue
            val = binding.value
            if not isinstance(val.args[0], tvm.ir.GlobalVar):
                # A Call that is not a call_tir into a generated PrimFunc: LegalizeOps had
                # no rule for the operator and left the relax op standing, so there is no
                # kernel to call.
                #
                # The question is not whether it lowered. It is whether anything reads the
                # buffer it was supposed to fill. A binding nothing reads costs nothing to
                # pass over; a binding something reads and nothing writes is how LayerNorm
                # and Gather used to return zeros.
                if binding.var.name_hint not in consumed:
                    continue

                # Read, and resolvable: the `shape_of` / `shape_to_tensor` pair every
                # transformer from the ONNX frontend contains. Its extents are already
                # constants, so the values are emitted as stores. Without this the `take`
                # kernel in models/model.onnx reads an unwritten buffer.
                shape_values = static_shape_tensor_values(val)
                if shape_values is not None and binding.var.name_hint in tensors_mapped:
                    dest = tensors_mapped[binding.var.name_hint]
                    ctype = c_type_for_dtype(dest["dtype"])
                    model_run_lines.append(f"  // {dest['name']} = shape_to_tensor(...), resolved at compile time")
                    for index, extent in enumerate(shape_values):
                        model_run_lines.append(f"  (({ctype}*)tensor_{dest['cname']}.data)[{index}] = {extent};")
                    model_run_lines.append("")
                    continue

                # Read, and not resolvable. relax.cumsum lands here: it survives
                # legalization, so CumSum built cleanly, booted under QEMU, reported a
                # cycle count and returned a tensor of 0.0 -- the graph's own output,
                # never written by anything. Being told the operator cannot be compiled
                # is worth more than a measurement of the wrong program.
                op_name = str(getattr(val.op, "name", val.op)).removeprefix("relax.")
                raise CompilationError(
                    stage="harness-generation",
                    command="compile_model",
                    details=(
                        f"'{op_name}' could not be lowered to a kernel. TVM has no C "
                        f"implementation for it, so there is nothing for the bare-metal "
                        f"harness to call. Replace it in the exported graph with "
                        f"operators that are supported, or remove it."
                    ),
                )

            func_name = val.args[0].name_hint
            ffi_name = f"__tvm_ffi_{func_name}"

            args_fields = val.args[1].fields if isinstance(val.args[1], relax.Tuple) else [val.args[1]]
            c_args = []
            for arg in args_fields:
                if isinstance(arg, relax.Var):
                    # Through any renames: a kernel reading `lv5` when `lv5 = lv4` has to
                    # be handed lv4's buffer, because lv5's own pool slot is never written.
                    src = resolve_alias(arg.name_hint)
                    if isinstance(src, relax.Constant):
                        c_args.append(f"&tensor_constant_{constant_index(src)}")
                    else:
                        c_args.append(f"&tensor_{tensors_mapped[src]['cname']}")
                elif isinstance(arg, relax.Constant):
                    c_args.append(f"&tensor_constant_{constant_index(arg)}")

            dest_info = tensors_mapped.get(binding.var.name_hint)
            if dest_info is None:
                # BatchNorm lands here: it returns R.Tuple(y, mean, var), so the kernel's
                # destination is a tuple var with no buffer of its own. Unhandled, the
                # lookup raised KeyError('lv') from inside code generation -- a Python
                # traceback about a variable that appears nowhere in the user's model.
                raise no_buffer_error(
                    f"The result of '{func_name}',", binding.var.name_hint, binding.var.struct_info
                )
            c_args.append(f"&tensor_{dest_info['cname']}")
            num_args = len(c_args)

            model_run_lines.extend(
                [
                    f"  // {binding.var.name_hint} = call_tir({func_name})",
                    "  {",
                    f"    TVMFFIAny args[{num_args}] = {{0}};",
                ]
            )
            for idx, arg_expr in enumerate(c_args):
                model_run_lines.extend(
                    [
                        f"    args[{idx}].v_ptr = {arg_expr};",
                        f"    args[{idx}].type_index = 0;",
                    ]
                )
            if profile:
                # The timer brackets the call and nothing else. The return-code check
                # moves below the second read so error handling is not inside the
                # measured window, and the accumulate happens before the early return
                # so a failing kernel still reports the cycles it burned.
                k = kernel_index[func_name]
                model_run_lines.extend(
                    [
                        "    uint64_t tatva_t0 = tatva_read_cycles();",
                        f"    int32_t tatva_rc = {ffi_name}(NULL, args, {num_args}, NULL);",
                        f"    tatva_kernel_cycles[{k}] += tatva_read_cycles() - tatva_t0;",
                        f"    tatva_kernel_calls[{k}] += 1;",
                    ]
                )
                tatva_check = "    if (tatva_rc != 0) {"
            else:
                tatva_check = f"    if ({ffi_name}(NULL, args, {num_args}, NULL) != 0) {{"

            model_run_lines.extend(
                [
                    tatva_check,
                    f'        sbi_print("Failed call_tir: {func_name}\\n");',
                    "        return -1;",
                    "    }",
                    "  }",
                    "",
                ]
            )

    # The output binding, when it is not a kernel call. tensor_<out>.data already points
    # at output_ptr, so the copy lands straight in the caller's buffer.
    out_source = resolve_alias(out_var_name)
    if isinstance(out_source, relax.Constant) or out_source != out_var_name:
        out_c_type = c_type_for_dtype(out_info["dtype"])
        count = max(1, out_info["num_elements"])
        if isinstance(out_source, relax.Constant):
            src_expr = f"(({out_c_type}*)constant_data_{constant_index(out_source)})"
            src_desc = "a constant -- the operator producing it folded away at compile time"
        else:
            src_expr = f"(({out_c_type}*)tensor_{tensors_mapped[out_source]['cname']}.data)"
            src_desc = f"'{out_source}'"
        model_run_lines.extend(
            [
                f"  // '{out_var_name}' is bound to {src_desc}, not computed by a kernel.",
                "  // Without this copy the output buffer is never written at all.",
                f"  for (int i = 0; i < {count}; i++) {{",
                f"    (({out_c_type}*)output_ptr)[i] = {src_expr}[i];",
                "  }",
                "",
            ]
        )

    model_run_lines.extend(["  return 0;", "}"])

    with open(os.path.join(build_dir, "model_run.c"), "w") as f:
        f.write("\n".join(model_run_lines))

    # OUTPUT_SIZE is an element count derived from the output tensor's own dtype.
    # It used to be hardcoded as bytes // 4, which is only correct for float32.
    output_c_type = c_type_for_dtype(out_info["dtype"])
    model_info_lines = [
        "#ifndef MODEL_INFO_H",
        "#define MODEL_INFO_H",
        f"#define OUTPUT_SIZE {max(1, out_info['num_elements'])}",
        f"#define OUTPUT_C_TYPE {output_c_type}",
        "#endif // MODEL_INFO_H",
    ]
    with open(os.path.join(build_dir, "model_info.h"), "w") as f:
        f.write("\n".join(model_info_lines))

    # Swap TVM's generated softmax for TATVA's Schraudolph fast-exponential kernel.
    if model_ir.metadata.get("softmax_optimized", False):
        is_vector = variant.gcc_march.endswith("v") or "gcv" in variant.gcc_march.lower()
        operators_c, patched = inject_optimized_softmax(operators_c, is_vector)
        if patched == 0:
            raise CompilationError(
                stage="softmax-injection",
                command="compile_model",
                details=(
                    "The fusion pass was requested but TVM emitted no __tvm_ffi_softmax kernel "
                    "to replace, so this build would be identical to the baseline. Re-run without "
                    "the fuse pass, or check that the model's softmax survived legalization."
                ),
            )

    with open(os.path.join(build_dir, "operators.c"), "w") as f:
        f.write(operators_c)

    # Point GCC at TVM's headers where they already are. Copying both include trees
    # into every build directory meant thousands of files per compile -- the single
    # slowest step in the pipeline, and the one that made builds fight the OS file
    # scanner. They are read-only inputs; there is no reason to duplicate them.
    include_dirs = [
        os.path.join(os.path.dirname(tvm.__file__), "include"),
        os.path.join(os.path.dirname(tvm_ffi.__file__), "include"),
    ]
    include_dirs = [d for d in include_dirs if os.path.isdir(d)]

    with open(os.path.join(build_dir, "start.S"), "w") as f:
        f.write(START_S)

    # Every pool this build is about to ask the linker for, added up before GCC runs so
    # the RAM region can be sized to fit it. Sizing after the fact is not an option --
    # the linker is what fails.
    io_bytes = sum(max(1, i["num_elements"]) * (i["dtype_bits"] // 8) for i in input_infos)
    io_bytes += max(1, out_info["num_elements"]) * (out_info["dtype_bits"] // 8)
    accounted_bytes = weights_bytes + curr_offset + WORKSPACE_POOL_BYTES + io_bytes + STACK_BYTES
    ram_bytes = _ram_region_bytes(accounted_bytes)

    with open(os.path.join(build_dir, "link.ld"), "w") as f:
        f.write(render_link_ld(ram_bytes))

    # Fill the harness template from the model's own inputs. Every buffer, its C
    # type and its dummy-data expression come from input_infos, so a model that is
    # not BERT-shaped gets a harness that matches it instead of one that will not link.
    # `run_params` is the same string used for the tvmgen_default_run() definition
    # above, so the extern declaration in main.c cannot drift from it.
    run_args = ", ".join(f"in_{info['cname']}" for info in input_infos) + ", tatva_output"

    io_buffer_lines: list[str] = []
    io_init_lines: list[str] = []
    for info in input_infos:
        c_type = c_type_for_dtype(info["dtype"])
        count = max(1, info["num_elements"])
        kind = input_fill_kind(info["name"], info["dtype"])
        fill = _c_fill_expression(kind, c_type)
        io_buffer_lines.append(f"static {c_type} in_{info['cname']}[{count}];  // {info['name']} {info['shape']}")
        io_init_lines.append(f"    for (int i = 0; i < {count}; i++) {{")
        io_init_lines.append(f"        in_{info['cname']}[i] = {fill};")
        io_init_lines.append("    }")
    io_buffer_lines.append("static OUTPUT_C_TYPE tatva_output[OUTPUT_SIZE];")

    main_c_content = (
        MAIN_C_BENCHMARK.replace("@TATVA_RUN_PARAMS@", run_params)
        .replace("@TATVA_RUN_ARGS@", run_args)
        .replace("@TATVA_IO_BUFFERS@", "\n".join(io_buffer_lines))
        .replace("@TATVA_IO_INIT@", "\n".join(io_init_lines))
        .replace("WARMUP_COUNT", str(warmup_count))
        .replace("TIMED_COUNT", str(timed_count))
        .replace("@TATVA_STABLE_BREAK@", "" if profile else STABLE_BREAK)
        .replace("@TATVA_PROFILE_DECLS@", PROFILE_MAIN_DECLS if profile else "")
        .replace("@TATVA_PROFILE_RESET@", PROFILE_MAIN_RESET if profile else "")
        .replace("@TATVA_PROFILE_REPORT@", PROFILE_MAIN_REPORT if profile else "")
    )
    with open(os.path.join(build_dir, "main.c"), "w") as f:
        f.write(main_c_content)

    gcc_name, gcc_path = find_riscv_gcc()
    if not gcc_path:
        raise FileNotFoundError("RISC-V GCC cross-compiler binary not found.")

    elf_path = os.path.join(build_dir, "model.elf")

    compile_cmd = [
        gcc_path,
        "-O2",
        f"-march={variant.gcc_march}",
        f"-mabi={variant.gcc_mabi}",
        "-mcmodel=medany",
        "-ffreestanding",
        "-nostdlib",
        "-T",
        os.path.join(build_dir, "link.ld"),
        os.path.join(build_dir, "start.S"),
        os.path.join(build_dir, "weights.S"),
        os.path.join(build_dir, "main.c"),
        os.path.join(build_dir, "model_run.c"),
        os.path.join(build_dir, "operators.c"),
        *[f"-I{d}" for d in include_dirs],
        "-o",
        elf_path,
        "-lm",
        "-lgcc",
    ]

    try:
        # Generous: the largest fixture takes a few seconds on an idle machine, but a
        # laptop that is also running QEMU and a file scanner is a different story, and
        # a timeout here surfaces as an unexplained CompilationError.
        res = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            gcc_stderr = res.stderr or res.stdout

            # A region overflow is a memory-budget failure, not a toolchain-flags failure,
            # and it deserves to be reported as one. Left as a generic CompilationError it
            # came out advising the user to "check for memory space address collisions in
            # link.ld", which describes neither the cause nor a fix. The linker names
            # whichever section it happened to be placing when the region ran out --
            # usually .bss -- so its message points at the activations even when the
            # weights are what did not fit; the breakdown below says which.
            overflow = re.search(r"region `(\w+)' overflowed by (\d+) bytes", gcc_stderr)
            if overflow:
                raise MemoryLimitExceededError(
                    limit_bytes=ram_bytes,
                    required_bytes=ram_bytes + int(overflow.group(2)),
                    details=_memory_budget_note(
                        weights_bytes, curr_offset, io_bytes, ram_bytes, variant.name
                    ),
                )

            raise CompilationError(
                stage="cross-compilation",
                command=" ".join(compile_cmd),
                stderr=gcc_stderr,
                details=f"{gcc_name} exited with code {res.returncode} targeting {variant.name}.",
            )
    except (CompilationError, MemoryLimitExceededError):
        raise
    except Exception as e:
        raise CompilationError(
            stage="cross-compilation",
            command=" ".join(compile_cmd),
            details=f"Could not launch the cross-compiler: {e}",
        ) from e

    return CompiledArtifact(
        elf_path=elf_path,
        build_dir=build_dir,
        variant=variant,
        ram_bytes=ram_bytes,
        run_count=warmup_count + timed_count,
    )


def run_and_measure(
    artifact: CompiledArtifact,
    inputs: Any = None,
    environment: ExecutionEnvironment = ExecutionEnvironment.QEMU_SIM,
) -> MeasurementResult:
    """
    Run a compiled model ELF binary and gather latency measurements.
    Supports QEMU_SIM simulation and hooks for REAL_HW execution.
    """
    if environment == ExecutionEnvironment.REAL_HW:
        raise NotImplementedError(
            "REAL_HW execution is not supported locally. Please deploy the compiled ELF binary "
            f"({artifact.elf_path}) directly to your physical RISC-V hardware board."
        )

    _qemu_name, qemu_path = find_qemu(artifact.variant.bitness)
    if not qemu_path:
        raise FileNotFoundError(f"QEMU {artifact.variant.bitness}-bit emulator binary not found.")

    qemu_cmd = [
        qemu_path,
        "-M",
        "virt",
        "-m",
        f"{_qemu_memory_mib(artifact.ram_bytes)}M",
    ]
    if artifact.variant.gcc_march.endswith("v") or "gcv" in artifact.variant.gcc_march.lower():
        qemu_cmd.extend(["-cpu", "rv64,v=true,vlen=128"])
    qemu_cmd.extend(
        [
            "-kernel",
            artifact.elf_path,
            "-nographic",
            "-icount",
            "shift=0",
        ]
    )

    # Checked before QEMU is spawned, because QEMU's own refusal talks about overlapping
    # ROM regions and names the device tree -- it never mentions the model, so on its own
    # it reads as a toolchain bug rather than as "this model is too big to simulate".
    span = _elf_load_span(artifact.elf_path)
    if span is not None and span[1] > QEMU_FDT_BASE:
        raise EmulatorImageLimitError(
            limit_bytes=MAX_LOADABLE_IMAGE_BYTES,
            required_bytes=span[1] - span[0],
            fdt_address=QEMU_FDT_BASE,
            details=(
                f"The linked image spans {_mib(span[1] - span[0])}, from {span[0]:#x} to "
                f"{span[1]:#x}, which runs past the device tree at {QEMU_FDT_BASE:#x}. The "
                f"bundled QEMU `virt` board leaves {_mib(MAX_LOADABLE_IMAGE_BYTES)} between "
                f"{RAM_ORIGIN:#x} and that address, and the address does not move with `-m`."
            ),
        )

    timeout_seconds = _qemu_timeout_seconds(artifact.ram_bytes, artifact.run_count, artifact.variant)
    soft_float = not _has_hardware_float(artifact.variant)

    try:
        res = subprocess.run(qemu_cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as e:
        raise SimulationTimeoutError(
            timeout_seconds=timeout_seconds,
            target=artifact.variant.name,
            run_count=artifact.run_count,
            soft_float=soft_float,
            details=(
                f"Image linked into {_mib(artifact.ram_bytes)}; the ceiling is "
                f"{QEMU_SOFT_FLOAT_SECONDS_PER_MIB_PER_RUN if soft_float else QEMU_SECONDS_PER_MIB_PER_RUN}"
                f" s per MiB per inference for {artifact.variant.gcc_march}."
            ),
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to execute QEMU process: {e}") from e

    if res.returncode != 0:
        stderr = res.stderr or ""
        # Backstop for the case the pre-check could not see -- an unparsable ELF, or a
        # future QEMU that moves the device tree somewhere the constant does not predict.
        if "ROM regions are overlapping" in stderr:
            segment = re.search(r"segment \d+ \(addresses 0x([0-9a-fA-F]+) - 0x([0-9a-fA-F]+)\)", stderr)
            fdt = re.search(r"fdt \(addresses 0x([0-9a-fA-F]+)", stderr)
            low, high = (int(segment.group(1), 16), int(segment.group(2), 16)) if segment else (RAM_ORIGIN, 0)
            fdt_address = int(fdt.group(1), 16) if fdt else QEMU_FDT_BASE
            raise EmulatorImageLimitError(
                limit_bytes=max(0, fdt_address - RAM_ORIGIN),
                required_bytes=max(0, high - low),
                fdt_address=fdt_address,
                details=(
                    "QEMU refused to load the image because it overlaps the device tree at "
                    f"{fdt_address:#x}. Reported by the emulator, not predicted:\n{stderr.strip()}"
                ),
            )
        raise RuntimeError(f"QEMU simulation execution failed:\nStdout: {res.stdout}\nStderr: {res.stderr}")

    # Parse RUN_CYCLES
    cycle_samples = []
    kernel_profiles: list[KernelProfile] = []
    for line in res.stdout.splitlines():
        if "RUN_CYCLES:" in line:
            parts = line.strip().split(":")
            if len(parts) >= 2:
                cycle_samples.append(int(parts[1].strip()))
        elif "KERNEL_CYCLES:" in line:
            # "KERNEL_CYCLES: <index> <name> <calls> <cycles>". Split on the tag rather
            # than on ':' so a kernel name never has to be free of colons, and require
            # exactly four fields so a torn line from the UART is dropped, not guessed at.
            fields = line.split("KERNEL_CYCLES:", 1)[1].split()
            if len(fields) == 4:
                try:
                    kernel_profiles.append(
                        KernelProfile(
                            index=int(fields[0]),
                            name=fields[1],
                            calls=int(fields[2]),
                            cycles=int(fields[3]),
                        )
                    )
                except ValueError:
                    continue

    # Heaviest first: the reason to open a profile is to find what to optimise.
    kernel_profiles.sort(key=lambda k: k.cycles, reverse=True)

    if not cycle_samples:
        raise RuntimeError(f"No latency metrics found in QEMU output:\n{res.stdout}")

    nominal_clock_hz = 100.0 * 1000.0 * 1000.0
    ms_samples = [(cycles / nominal_clock_hz) * 1000.0 for cycles in cycle_samples]

    mean_ms = float(np.mean(ms_samples))
    median_ms = float(np.median(ms_samples))
    p95_ms = float(np.percentile(ms_samples, 95))

    return MeasurementResult(
        environment="QEMU_SIM",
        simulated=True,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        raw_samples_ms=ms_samples,
        raw_output=res.stdout,
        kernel_profiles=kernel_profiles,
        total_cycles=sum(cycle_samples),
    )


def default_inputs_for(onnx_path: str) -> dict[str, np.ndarray]:
    """
    Build the host-side input set that the generated benchmark harness reproduces.

    Shapes come from compiler.resolve_input_shapes -- the same function that binds
    the symbolic dims for the TVM import -- and values from default_input_array, the
    same function whose C translation main.c emits. The old version hardcoded the
    three BERT tensors, so any other model silently got no inputs at all and the
    parity check compared the target against whatever onnxruntime happened to do.
    """
    import onnx

    from tatva.compiler import resolve_input_shapes

    onnx_model = onnx.load(onnx_path)
    shapes = resolve_input_shapes(onnx_model)

    dtypes: dict[str, str] = {}
    for inp in onnx_model.graph.input:
        elem_type = inp.type.tensor_type.elem_type
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
        dtypes[inp.name] = str(np_dtype)

    return {
        name: default_input_array(name, shape, dtypes.get(name, "float32"))
        for name, shape in shapes.items()
    }


def reference_output(onnx_path: str, inputs: dict[str, Any] | None = None) -> np.ndarray:
    """
    Get the ground-truth output from the host using onnxruntime.
    If inputs are not provided, the same dummy inputs main.c generates are used.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)

    if inputs is None:
        inputs = default_inputs_for(onnx_path)

    # Only feed tensors the session actually declares; ONNX graph inputs can include
    # initializers that onnxruntime folds away.
    expected = {inp.name for inp in sess.get_inputs()}
    feed = {k: v for k, v in inputs.items() if k in expected}

    outputs = sess.run(None, feed)
    return outputs[0].flatten()


def establish_baseline(
    onnx_path: str, variant: TargetVariant, inputs: dict[str, Any] | None = None
) -> BaselineResult:
    """
    Compile the model, run in QEMU, verify numerical parity with ONNX Runtime on host,
    and return the baseline execution results.
    """
    ref_output = reference_output(onnx_path, inputs)
    ref_logits = [float(x) for x in ref_output[:5]]

    from tatva.compiler import import_model

    model_ir = import_model(onnx_path)
    artifact = compile_model(model_ir, variant)

    measurement = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)

    target_logits = []
    for line in measurement.raw_output.splitlines():
        if "FIRST_LOGITS:" in line:
            parts = line.strip().split(":")[1].strip().split()
            target_logits = [float(x) for x in parts[:5]]
            break

    if not target_logits:
        raise RuntimeError(f"Could not find FIRST_LOGITS in simulator output:\n{measurement.raw_output}")

    # Check numerical parity
    tolerance = 1e-4
    ref_arr = np.array(ref_logits)
    tgt_arr = np.array(target_logits)

    parity_passed = np.allclose(ref_arr, tgt_arr, rtol=tolerance, atol=tolerance)
    if not parity_passed:
        raise AssertionError(
            f"Numerical parity check failed.\n"
            f"Host Reference: {ref_logits}\n"
            f"QEMU Simulator: {target_logits}\n"
            f"Difference: {np.abs(ref_arr - tgt_arr)}"
        )

    return BaselineResult(
        latency_result=measurement,
        parity_passed=True,
        tolerance=tolerance,
        ref_logits=ref_logits,
        target_logits=target_logits,
    )
