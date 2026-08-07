"""
TATVA Execution and Simulation Runner Module.

This module handles compiling the generated C code with the RISC-V cross-compiler,
executing the resulting binary in the QEMU simulator environment, and running
target architecture verification gates and performance measurements.
"""

import json
import os
import shutil
import subprocess
import tempfile
from enum import Enum
from typing import Any

import numpy as np

from tatva.compiler import ModelIR, TargetVariant

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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

# C Driver for model inference benchmarking inside QEMU system-mode
MAIN_C_BENCHMARK = """
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "model_info.h"

extern int32_t tvmgen_default_run(void* input_ids, void* attention_mask, void* token_type_ids, void* output);

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

static int64_t input_ids[32];
static int64_t attention_mask[32];
static int64_t token_type_ids[32];
static float output_logits[OUTPUT_SIZE];

int main(void) {
    sbi_print("\\n=== Starting Latency Test inside QEMU ===\\n");
    
    // Initialize dummy inputs
    for (int i = 0; i < 32; i++) {
        input_ids[i] = i % 5;
        attention_mask[i] = 1;
        token_type_ids[i] = 0;
    }
    
    // Warm-up runs
    for (int i = 0; i < WARMUP_COUNT; i++) {
        tvmgen_default_run(input_ids, attention_mask, token_type_ids, output_logits);
    }
    
    // Timed runs
    for (int i = 0; i < TIMED_COUNT; i++) {
        uint64_t start = read_cycles();
        tvmgen_default_run(input_ids, attention_mask, token_type_ids, output_logits);
        uint64_t end = read_cycles();
        sbi_print("RUN_CYCLES: ");
        sbi_print_uint(end - start);
        sbi_print("\\n");
    }
    
    // Print first 5 logits for correctness verification
    sbi_print("FIRST_LOGITS: ");
    for (int i = 0; i < 5 && i < OUTPUT_SIZE; i++) {
        sbi_print_float(output_logits[i]);
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


class CompilationError(Exception):
    """Raised when compilation or linking fails."""
    pass


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


def find_riscv_gcc() -> tuple[str | None, str | None]:
    """
    Find the RISC-V GCC cross-compiler binary on system PATH or in the local project directory.
    """
    candidates = ["riscv-none-elf-gcc", "riscv64-unknown-elf-gcc"]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return name, path
        path_exe = shutil.which(name + ".exe")
        if path_exe:
            return name, path_exe

    local_bin = os.path.join(PROJECT_DIR, "riscv-toolchain", "bin", "riscv-none-elf-gcc.exe")
    if os.path.exists(local_bin):
        return "riscv-none-elf-gcc", local_bin
    local_bin_no_exe = os.path.join(PROJECT_DIR, "riscv-toolchain", "bin", "riscv-none-elf-gcc")
    if os.path.exists(local_bin_no_exe):
        return "riscv-none-elf-gcc", local_bin_no_exe

    return None, None


def find_qemu(bitness: int = 64) -> tuple[str | None, str | None]:
    """
    Find the QEMU emulator binary matching the requested target bitness (32 or 64).
    """
    name_prefix = f"qemu-system-riscv{bitness}"
    path = shutil.which(name_prefix)
    if path:
        return name_prefix, path
    path_exe = shutil.which(name_prefix + ".exe")
    if path_exe:
        return name_prefix, path_exe

    local_bin = os.path.join(PROJECT_DIR, "qemu", "bin", f"{name_prefix}.exe")
    if os.path.exists(local_bin):
        return name_prefix, local_bin
    local_bin_no_exe = os.path.join(PROJECT_DIR, "qemu", "bin", name_prefix)
    if os.path.exists(local_bin_no_exe):
        return name_prefix, local_bin_no_exe

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


def verify_target(variant: TargetVariant) -> dict[str, Any]:
    """
    Compile a tiny hello-world C program with the variant's march/mabi via the RISC-V GCC,
    execute it under matching QEMU emulator, and confirm expected console output.
    Returns:
        dict: {"status": "ok"/"fail", "output": str, "error": str}
    """
    gcc_name, gcc_path = find_riscv_gcc()
    qemu_name, qemu_path = find_qemu(variant.bitness)

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

    scratch_dir = os.path.join(PROJECT_DIR, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
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
        scratch_dir = os.path.join(PROJECT_DIR, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        build_dir = tempfile.mkdtemp(prefix="tatva_build_", dir=scratch_dir)
    else:
        build_dir = output_dir
        os.makedirs(build_dir, exist_ok=True)

    seq = tvm.transform.Sequential([relax.transform.LegalizeOps()])
    mod_legalized = seq(model_ir.mod)

    try:
        lib = relax.build(mod_legalized, target="c")
        operators_c = lib.mod.imports[0].inspect_source()
    except Exception as e:
        raise CompilationError(f"TVM Relax compilation to C failed: {e}") from e

    func = mod_legalized["main"]
    constants_list = []

    def find_constants(expr):
        if isinstance(expr, relax.Constant):
            arr = expr.data.numpy()
            for c in constants_list:
                c_arr = c.data.numpy()
                if c_arr.shape == arr.shape and c_arr.dtype == arr.dtype:
                    if np.array_equal(c_arr, arr):
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

    for name, sinfo, is_input in vars_to_process:
        if not hasattr(sinfo, "shape") or sinfo.shape is None:
            continue
        shape = [int(dim) for dim in sinfo.shape]
        dtype = sinfo.dtype

        dtype_code, dtype_bits = map_dtype(dtype)
        num_elements = int(np.prod(shape))
        size_bytes = num_elements * (dtype_bits // 8)

        tensors_mapped[name] = {
            "name": name,
            "shape": shape,
            "ndim": len(shape),
            "dtype_code": dtype_code,
            "dtype_bits": dtype_bits,
            "size_bytes": size_bytes,
            "is_input": is_input,
        }

        size_aligned = (size_bytes + 15) & ~15
        if not is_input:
            tensors_mapped[name]["offset"] = curr_offset
            curr_offset += size_aligned

    weights_h_lines = [
        "#ifndef WEIGHTS_H",
        "#define WEIGHTS_H",
        "#include <stdint.h>",
        "",
    ]
    for idx, const in enumerate(constants_list):
        arr = const.data.numpy()
        dtype = str(arr.dtype)

        def fmt(x):
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
            if "int" in dtype:
                c_type = "int32_t"
            else:
                c_type = "float"

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

    for name, info in tensors_mapped.items():
        shape_str = ", ".join(str(x) for x in info["shape"])
        model_run_lines.append(f"static int64_t shape_{name}[] = {{ {shape_str} }};")

        data_init = "NULL" if info["is_input"] else f"&global_pool[{info['offset']}]"
        model_run_lines.extend(
            [
                f"static DLTensor tensor_{name} = {{",
                f"    .data = {data_init},",
                "    .device = {1, 0},",
                f"    .ndim = {info['ndim']},",
                f"    .dtype = {{{info['dtype_code']}, {info['dtype_bits']}, 1}},",
                f"    .shape = shape_{name},",
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
            "static uint8_t workspace_pool[1024 * 1024] __attribute__((aligned(16)));",
            "static size_t workspace_offset = 0;",
            "",
            "void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_or_handle, int dtype_bits) {",
            "    size_t size = (nbytes + 15) & ~15;",
            "    if (workspace_offset + size > sizeof(workspace_pool)) {",
            "        return NULL;",
            "    }",
            "    void* ptr = &workspace_pool[workspace_offset];",
            "    workspace_offset += size;",
            "    return ptr;",
            "}",
            "",
            "int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {",
            "    workspace_offset = 0;",
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

    model_run_lines.append(
        "int32_t tvmgen_default_run(void* input_ids_ptr, void* attention_mask_ptr, void* token_type_ids_ptr, void* output_ptr) {"
    )
    model_run_lines.extend(
        [
            "  tensor_input_ids.data = input_ids_ptr;",
            "  tensor_attention_mask.data = attention_mask_ptr;",
            "  tensor_token_type_ids.data = token_type_ids_ptr;",
            f"  tensor_{out_var_name}.data = output_ptr;",
            "",
        ]
    )

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
                    c_args.append(f"&tensor_{arg.name_hint}")
                elif isinstance(arg, relax.Constant):
                    const_idx = -1
                    arr = arg.data.numpy()
                    for i, c in enumerate(constants_list):
                        c_arr = c.data.numpy()
                        if c_arr.shape == arr.shape and c_arr.dtype == arr.dtype:
                            if np.array_equal(c_arr, arr):
                                const_idx = i
                                break
                    c_args.append(f"&tensor_constant_{const_idx}")

            c_args.append(f"&tensor_{binding.var.name_hint}")
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

    output_size_bytes = tensors_mapped[out_var_name]["size_bytes"]
    model_info_lines = [
        "#ifndef MODEL_INFO_H",
        "#define MODEL_INFO_H",
        f"#define OUTPUT_SIZE {output_size_bytes // 4}",
        "#endif // MODEL_INFO_H",
    ]
    with open(os.path.join(build_dir, "model_info.h"), "w") as f:
        f.write("\n".join(model_info_lines))

    # Apply custom RISC-V optimized Softmax operator replacement if enabled in metadata
    if model_ir.metadata.get("softmax_optimized", False):
        is_vector = variant.gcc_march.endswith("v") or "gcv" in variant.gcc_march.lower()
        if is_vector:
            operators_c = "#include <riscv_vector.h>\n" + operators_c
            custom_softmax = """TVM_DLL int32_t __tvm_ffi_softmax(void* self_handle, void* args, int32_t num_args, void* result) {
  if (num_args != 2) return -1;
  void* var_input = (((TVMFFIAny*)args)[0].type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr);
  void* var_T_softmax_norm = (((TVMFFIAny*)args)[1].type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr);
  
  float* input_ptr = (float*)(((DLTensor*)var_input)[0].data);
  float* T_softmax_norm = (float*)(((DLTensor*)var_T_softmax_norm)[0].data);
  
  int64_t* shape = ((DLTensor*)var_input)[0].shape;
  int32_t rows = (int32_t)shape[0];
  int32_t cols = (int32_t)shape[1];
  
  for (int32_t i0 = 0; i0 < rows; ++i0) {
    float* in_row = input_ptr + i0 * cols;
    float* out_row = T_softmax_norm + i0 * cols;
    
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
    float local_exp[cols];
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
        else:
            custom_softmax = """TVM_DLL int32_t __tvm_ffi_softmax(void* self_handle, void* args, int32_t num_args, void* result) {
  if (num_args != 2) return -1;
  void* var_input = (((TVMFFIAny*)args)[0].type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr);
  void* var_T_softmax_norm = (((TVMFFIAny*)args)[1].type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr);
  
  float* input_ptr = (float*)(((DLTensor*)var_input)[0].data);
  float* T_softmax_norm = (float*)(((DLTensor*)var_T_softmax_norm)[0].data);
  
  int64_t* shape = ((DLTensor*)var_input)[0].shape;
  int32_t rows = (int32_t)shape[0];
  int32_t cols = (int32_t)shape[1];
  
  for (int32_t i0 = 0; i0 < rows; ++i0) {
    float* in_row = input_ptr + i0 * cols;
    float* out_row = T_softmax_norm + i0 * cols;
    
    float max_val = in_row[0];
    for (int32_t k = 1; k < cols; ++k) {
      if (in_row[k] > max_val) {
        max_val = in_row[k];
      }
    }
    
    float sum = 0.0f;
    float local_exp[cols];
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
        for suffix in ["", "1", "2", "3", "4", "5"]:
            start_str = f"TVM_DLL int32_t __tvm_ffi_softmax{suffix}(void* self_handle, void* args, int32_t num_args, void* result) {{"
            idx = operators_c.find(start_str)
            if idx != -1:
                brace_count = 1
                pos = idx + len(start_str)
                while brace_count > 0 and pos < len(operators_c):
                    if operators_c[pos] == "{":
                        brace_count += 1
                    elif operators_c[pos] == "}":
                        brace_count -= 1
                    pos += 1
                replaced_code = custom_softmax.replace("__tvm_ffi_softmax", f"__tvm_ffi_softmax{suffix}")
                operators_c = operators_c[:idx] + replaced_code + operators_c[pos:]

    with open(os.path.join(build_dir, "operators.c"), "w") as f:
        f.write(operators_c)

    tvm_dir = os.path.dirname(tvm.__file__)
    tvm_ffi_dir = os.path.dirname(tvm_ffi.__file__)
    tvm_include = os.path.join(tvm_dir, "include")
    tvm_ffi_include = os.path.join(tvm_ffi_dir, "include")

    dest_include = os.path.join(build_dir, "include")
    os.makedirs(dest_include, exist_ok=True)
    if os.path.exists(tvm_include):
        shutil.copytree(tvm_include, dest_include, dirs_exist_ok=True)
    if os.path.exists(tvm_ffi_include):
        shutil.copytree(tvm_ffi_include, dest_include, dirs_exist_ok=True)

    with open(os.path.join(build_dir, "start.S"), "w") as f:
        f.write(START_S)
    with open(os.path.join(build_dir, "link.ld"), "w") as f:
        f.write(LINK_LD)

    main_c_content = (
        MAIN_C_BENCHMARK.replace("WARMUP_COUNT", str(warmup_count))
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
        f"-I{os.path.join(build_dir, 'include')}",
        "-o",
        elf_path,
        "-lm",
        "-lgcc",
    ]

    try:
        res = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise CompilationError(
                f"GCC cross-compilation failed with exit code {res.returncode}:\nStdout: {res.stdout}\nStderr: {res.stderr}"
            )
    except Exception as e:
        if not isinstance(e, CompilationError):
            raise CompilationError(f"Failed to launch GCC compilation: {e}") from e
        raise e

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

    qemu_name, qemu_path = find_qemu(artifact.variant.bitness)
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


def reference_output(onnx_path: str, inputs: dict[str, Any] | None = None) -> np.ndarray:
    """
    Get the ground-truth output from the host using onnxruntime.
    If inputs are not provided, default inputs matching main.c are used.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)

    if inputs is None:
        input_names = [inp.name for inp in sess.get_inputs()]
        seq_len = 32

        # Check if the model has a specified shape
        for inp in sess.get_inputs():
            if inp.name == "input_ids":
                if len(inp.shape) > 1 and isinstance(inp.shape[1], int):
                    seq_len = inp.shape[1]

        input_ids = np.array([[i % 5 for i in range(seq_len)]], dtype=np.int64)
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        token_type_ids = np.zeros((1, seq_len), dtype=np.int64)

        inputs = {}
        if "input_ids" in input_names:
            inputs["input_ids"] = input_ids
        if "attention_mask" in input_names:
            inputs["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = token_type_ids

    outputs = sess.run(None, inputs)
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
