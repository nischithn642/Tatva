"""
Subprocess & Toolchain Environment Execution Engine for Project Scaffolding.

Provides thread-safe cross-compilation, QEMU emulation, and execution result logging
with strict timeouts and toolchain discovery.
"""

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class ExecutionResult:
    """Standardized diagnostic payload for toolchain and emulation runs."""
    stage: str  # "compilation" | "emulation" | "parity_check"
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time_ms: float


class ToolchainManager:
    """
    Dynamic discoverer for RISC-V cross-compilation toolchains and QEMU emulators.
    """

    @staticmethod
    def discover_gcc() -> Tuple[Optional[str], Optional[str]]:
        """
        Find a RISC-V cross-compiler.

        The bundled toolchain is checked FIRST so a host x86 'gcc' on PATH can never be
        selected -- it would be invoked with -march=rv64gc* and fail in a confusing way.
        Host gcc is never a valid fallback for a RISC-V cross-compile.
        """
        for exe in ("riscv-none-elf-gcc.exe", "riscv-none-elf-gcc"):
            local_bin = os.path.join(PROJECT_DIR, "riscv-toolchain", "bin", exe)
            if os.path.exists(local_bin):
                return "riscv-none-elf-gcc", local_bin

        candidates = [
            "riscv64-unknown-elf-gcc",
            "riscv-none-elf-gcc",
            "riscv64-linux-gnu-gcc",
        ]
        for name in candidates:
            path = shutil.which(name) or shutil.which(name + ".exe")
            if path:
                return name, path

        return None, None

    @staticmethod
    def discover_qemu(bitness: int = 64) -> Tuple[Optional[str], Optional[str]]:
        """Find qemu-system-riscv64 or qemu-riscv64 binary."""
        candidates = [f"qemu-system-riscv{bitness}", f"qemu-riscv{bitness}"]
        for name in candidates:
            path = shutil.which(name)
            if path:
                return name, path
            path_exe = shutil.which(name + ".exe")
            if path_exe:
                return name, path_exe

        # Check local project QEMU directory
        local_bin = os.path.join(PROJECT_DIR, "qemu", "bin", f"qemu-system-riscv{bitness}.exe")
        if os.path.exists(local_bin):
            return f"qemu-system-riscv{bitness}", local_bin

        local_bin_no_exe = os.path.join(PROJECT_DIR, "qemu", "bin", f"qemu-system-riscv{bitness}")
        if os.path.exists(local_bin_no_exe):
            return f"qemu-system-riscv{bitness}", local_bin_no_exe

        return None, None

    @classmethod
    def get_health_status(cls) -> Dict[str, Any]:
        """Return diagnostic health check for toolchain dependencies."""
        gcc_name, gcc_path = cls.discover_gcc()
        qemu_name, qemu_path = cls.discover_qemu()
        cmake_path = shutil.which("cmake") or shutil.which("cmake.exe")
        make_path = shutil.which("make") or shutil.which("make.exe") or shutil.which("ninja")

        return {
            "gcc": bool(gcc_path),
            "gcc_name": gcc_name or "Missing",
            "gcc_path": gcc_path or "",
            "qemu": bool(qemu_path),
            "qemu_name": qemu_name or "Missing",
            "qemu_path": qemu_path or "",
            "cmake": bool(cmake_path),
            "make": bool(make_path),
            "status_badge": "🟢 Toolchain Ready" if (gcc_path and qemu_path) else "🔴 Toolchain Partial/Missing",
        }


class ScaffoldingExecutor:
    """
    Subprocess execution engine for cross-compiling, emulating, and verifying projects.
    """

    def __init__(self, timeout_sec: float = 30.0) -> None:
        self.timeout_sec = timeout_sec
        self.toolchain_health = ToolchainManager.get_health_status()

    def run_subprocess(self, cmd: list[str], cwd: str, stage: str) -> ExecutionResult:
        """Run command with strict timeout and standard stream capture."""
        start_t = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            elapsed_ms = (time.time() - start_t) * 1000.0
            return ExecutionResult(
                stage=stage,
                success=(proc.returncode == 0),
                return_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                execution_time_ms=round(elapsed_ms, 2),
            )
        except subprocess.TimeoutExpired as e:
            elapsed_ms = (time.time() - start_t) * 1000.0
            return ExecutionResult(
                stage=stage,
                success=False,
                return_code=-1,
                stdout=e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=f"TIMEOUT EXPIRED ({self.timeout_sec}s threshold exceeded)",
                execution_time_ms=round(elapsed_ms, 2),
            )
        except FileNotFoundError as e:
            elapsed_ms = (time.time() - start_t) * 1000.0
            return ExecutionResult(
                stage=stage,
                success=False,
                return_code=127,
                stdout="",
                stderr=(
                    f"Executable not found: '{cmd[0] if cmd else '<empty command>'}'. "
                    f"Install the RISC-V toolchain or run 'tatva doctor'. ({e})"
                ),
                execution_time_ms=round(elapsed_ms, 2),
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_t) * 1000.0
            return ExecutionResult(
                stage=stage,
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Subprocess Execution Exception: {e}",
                execution_time_ms=round(elapsed_ms, 2),
            )

    def compile_workspace(self, workspace_dir: str, target: str = "RV64GCV") -> ExecutionResult:
        """Cross-compile workspace files with the RISC-V toolchain."""
        gcc_name, gcc_path = ToolchainManager.discover_gcc()
        if not gcc_path:
            return ExecutionResult(
                stage="compilation",
                success=False,
                return_code=127,
                stdout="",
                stderr=(
                    "No RISC-V cross-compiler found. Expected 'riscv-none-elf-gcc' on PATH or under "
                    "'riscv-toolchain/bin/'. Run 'python setup_env.py' or 'tatva doctor'."
                ),
                execution_time_ms=0.0,
            )

        src_main = os.path.join(workspace_dir, "src", "main.c")
        src_inf = os.path.join(workspace_dir, "src", "model_inference.c")
        inc_dir = os.path.join(workspace_dir, "include")
        out_elf = os.path.join(workspace_dir, "build_app.elf")

        march = "-march=rv64gcv" if "V" in target else "-march=rv64gc"
        if "32" in target:
            march = "-march=rv32imc"

        cmd = [
            gcc_path,
            "-O2",
            march,
            f"-I{inc_dir}",
            src_main,
            src_inf,
            "-o",
            out_elf,
        ]

        if not os.path.exists(src_main):
            c_files = []
            for root, _, files in os.walk(workspace_dir):
                for f in files:
                    if f.endswith(".c"):
                        c_files.append(os.path.join(root, f))
            if c_files:
                cmd = [gcc_path, "-O2", f"-I{inc_dir}"] + c_files + ["-o", out_elf]

        return self.run_subprocess(cmd, cwd=workspace_dir, stage="compilation")

    def emulate_workspace(self, workspace_dir: str, target: str = "RV64GCV") -> ExecutionResult:
        """Emulate the cross-compiled ELF under QEMU system mode."""
        out_elf = os.path.join(workspace_dir, "build_app.elf")
        bitness = 32 if "32" in target else 64
        qemu_name, qemu_path = ToolchainManager.discover_qemu(bitness)

        if not qemu_path:
            return ExecutionResult(
                stage="emulation",
                success=False,
                return_code=127,
                stdout="",
                stderr=(
                    f"qemu-system-riscv{bitness} not found. Expected on PATH or under 'qemu/bin/'. "
                    "Run 'python setup_env.py' or 'tatva doctor'."
                ),
                execution_time_ms=0.0,
            )

        if not os.path.exists(out_elf):
            return ExecutionResult(
                stage="emulation",
                success=False,
                return_code=1,
                stdout="",
                stderr=f"No ELF to emulate: '{out_elf}' does not exist (compilation must run first).",
                execution_time_ms=0.0,
            )

        # System-mode QEMU needs a machine, a kernel image and no graphical console.
        cpu = "rv64,v=true,vext_spec=v1.0" if bitness == 64 and "V" in target.upper() else f"rv{bitness}"
        cmd = [
            qemu_path,
            "-M", "virt",
            "-cpu", cpu,
            "-nographic",
            "-bios", "none",
            "-kernel", out_elf,
        ]
        return self.run_subprocess(cmd, cwd=workspace_dir, stage="emulation")
