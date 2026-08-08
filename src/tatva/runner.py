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
import subprocess
import tempfile
from enum import Enum
from typing import Any

import numpy as np

from tatva.compiler import ModelIR, TargetVariant

# Single shared exception type. There used to be a second, unrelated CompilationError
# defined here, which meant diagnostics.classify_failure() never recognised the errors
# the runner actually raised and always fell through to the generic branch.
from tatva.diagnostics import CompilationError

__all__ = [
    "BaselineResult",
    "CompilationError",
    "CompiledArtifact",
    "ExecutionEnvironment",
    "MeasurementResult",
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

# Linker script defining a bare-metal memory space
LINK_LD = """
OUTPUT_ARCH( "riscv" )
ENTRY( _start )

MEMORY
{
  ram (wxa) : ORIGIN = 0x80200000, LENGTH = 128M
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
  . += 0x10000; /* 64KB stack */
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

    // Timed runs
    for (int i = 0; i < TIMED_COUNT; i++) {
        uint64_t start = read_cycles();
        tvmgen_default_run(@TATVA_RUN_ARGS@);
        uint64_t end = read_cycles();
        sbi_print("RUN_CYCLES: ");
        sbi_print_uint(end - start);
        sbi_print("\\n");
    }

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


class ExecutionEnvironment(Enum):
    QEMU_SIM = "QEMU_SIM"
    REAL_HW = "REAL_HW"


class CompiledArtifact:
    """
    Wraps the paths and metadata of a compiled model ELF binary.
    """
    def __init__(self, elf_path: str, build_dir: str, variant: TargetVariant):
        self.elf_path = elf_path
        self.build_dir = build_dir
        self.variant = variant


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
    ):
        self.environment = environment
        self.simulated = simulated
        self.mean_ms = mean_ms
        self.median_ms = median_ms
        self.p95_ms = p95_ms
        self.raw_samples_ms = raw_samples_ms
        self.raw_output = raw_output
        self.units = units

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
    and a git checkout share one copy. The legacy `<repo>/riscv-toolchain` and
    `<repo>/qemu` paths are still searched so nobody who ran the old setup_env.py has to
    re-download half a gigabyte.
    """
    from tatva.toolchain import tools_dir

    return [
        os.path.join(tools_dir(), install_name, "bin"),
        os.path.join(PROJECT_DIR, legacy_dir, "bin"),
    ]


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
    """
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    if not safe or safe[0].isdigit():
        safe = "t_" + safe
    return safe


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

    // RVV 1.0 Vector Exponent & Sum Accumulation
    vfloat32m1_t v_sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (int32_t k = 0; k < cols; k += vl) {
      vl = __riscv_vsetvl_e32m1(cols - k);
      vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_row + k, vl);
      vfloat32m1_t v_diff = __riscv_vfsub_vf_f32m1(v_in, max_val, vl);
      vfloat32m1_t v_scaled = __riscv_vfmul_vf_f32m1(v_diff, 12102203.0f, vl);
      vfloat32m1_t v_offset = __riscv_vfadd_vf_f32m1(v_scaled, 1065353216.0f, vl);
      vint32m1_t v_int = __riscv_vfcvt_x_f_v_i32m1(v_offset, vl);
      vfloat32m1_t v_exp = __riscv_vreinterpret_v_i32m1_f32m1(v_int);
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
            f.write(LINK_LD)
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


def compile_model(
    model_ir: ModelIR,
    variant: TargetVariant,
    output_dir: str | None = None,
    warmup_count: int = 3,
    timed_count: int = 10,
) -> CompiledArtifact:
    """
    Compiles a ModelIR module to C code via TVM Relax, maps operations and weights
    to static variables, writes bare-metal wrappers, and cross-compiles to a RISC-V ELF.
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
        shape = [int(dim) for dim in sinfo.shape]
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

    weights_h_lines = [
        "#ifndef WEIGHTS_H",
        "#define WEIGHTS_H",
        "#include <stdint.h>",
        "",
    ]
    for idx, const in enumerate(constants_list):
        arr = const.data.numpy()
        dtype = str(arr.dtype)

        # `dtype` is bound as a default so the closure captures this iteration's value
        # rather than whatever the loop variable happens to be when it is called. It is
        # only ever called inside this iteration today, but a late-binding closure over
        # a loop variable is the kind of thing that quietly breaks the day someone makes
        # this lazy.
        def fmt(x, dtype=dtype):
            if "int64" in dtype:
                return f"{x}LL"
            elif "int32" in dtype or "int8" in dtype or "uint8" in dtype or "int" in dtype:
                return str(x)
            else:
                s = f"{x:.10g}"
                if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                    return s + ".0f"
                return s + "f"

        if "int64" in dtype:
            c_type = "int64_t"
        elif "int32" in dtype:
            c_type = "int32_t"
        elif "int8" in dtype:
            c_type = "int8_t"
        elif "uint8" in dtype:
            c_type = "uint8_t"
        else:
            c_type = "int32_t" if "int" in dtype else "float"

        flat_data = arr.flatten()
        data_str = ", ".join(fmt(x) for x in flat_data)
        weights_h_lines.append(f"static {c_type} constant_data_{idx}[] = {{ {data_str} }};")

    weights_h_lines.append("")
    weights_h_lines.append("#endif // WEIGHTS_H")

    with open(os.path.join(build_dir, "weights.h"), "w") as f:
        f.write("\n".join(weights_h_lines))

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
    for block in func.body.blocks:
        for binding in block.bindings:
            if isinstance(binding.value, relax.Call) and isinstance(binding.value.args[0], tvm.ir.GlobalVar):
                func_name = binding.value.args[0].name_hint
                declared_funcs.add(func_name)

    for func_name in sorted(declared_funcs):
        model_run_lines.append(
            f"extern int32_t __tvm_ffi_{func_name}(void* self_handle, void* args, int32_t num_args, void* result);"
        )
    model_run_lines.append("")

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
            "static uint8_t workspace_pool[1024 * 1024] __attribute__((aligned(16)));",
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

    last_binding = func.body.blocks[-1].bindings[-1]
    out_var_name = last_binding.var.name_hint
    out_info = tensors_mapped.get(out_var_name)
    if out_info is None:
        raise CompilationError(
            stage="harness-generation",
            command="compile_model",
            details=(
                f"The model's final value '{out_var_name}' has no static tensor shape "
                "(models returning a tuple of outputs are not supported yet)."
            ),
        )

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
                continue

            func_name = val.args[0].name_hint
            ffi_name = f"__tvm_ffi_{func_name}"

            args_fields = val.args[1].fields if isinstance(val.args[1], relax.Tuple) else [val.args[1]]
            c_args = []
            for arg in args_fields:
                if isinstance(arg, relax.Var):
                    c_args.append(f"&tensor_{tensors_mapped[arg.name_hint]['cname']}")
                elif isinstance(arg, relax.Constant):
                    const_idx = -1
                    arr = arg.data.numpy()
                    for i, c in enumerate(constants_list):
                        c_arr = c.data.numpy()
                        if c_arr.shape == arr.shape and c_arr.dtype == arr.dtype and np.array_equal(c_arr, arr):
                            const_idx = i
                            break
                    c_args.append(f"&tensor_constant_{const_idx}")

            c_args.append(f"&tensor_{tensors_mapped[binding.var.name_hint]['cname']}")
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
            model_run_lines.extend(
                [
                    f"    if ({ffi_name}(NULL, args, {num_args}, NULL) != 0) {{",
                    f'        sbi_print("Failed call_tir: {func_name}\\n");',
                    "        return -1;",
                    "    }",
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
    with open(os.path.join(build_dir, "link.ld"), "w") as f:
        f.write(LINK_LD)

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
            raise CompilationError(
                stage="cross-compilation",
                command=" ".join(compile_cmd),
                stderr=res.stderr or res.stdout,
                details=f"{gcc_name} exited with code {res.returncode} targeting {variant.name}.",
            )
    except CompilationError:
        raise
    except Exception as e:
        raise CompilationError(
            stage="cross-compilation",
            command=" ".join(compile_cmd),
            details=f"Could not launch the cross-compiler: {e}",
        ) from e

    return CompiledArtifact(elf_path=elf_path, build_dir=build_dir, variant=variant)


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

    try:
        res = subprocess.run(qemu_cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(f"QEMU simulation execution failed:\nStdout: {res.stdout}\nStderr: {res.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed to execute QEMU process: {e}") from e

    # Parse RUN_CYCLES
    cycle_samples = []
    for line in res.stdout.splitlines():
        if "RUN_CYCLES:" in line:
            parts = line.strip().split(":")
            if len(parts) >= 2:
                cycle_samples.append(int(parts[1].strip()))

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
