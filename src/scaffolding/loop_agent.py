"""
Autonomous Self-Correction Closed-Loop State Machine Engine for Scaffolding ("Tatva Antigravity Engine").

Executes iterative generation -> cross-compilation -> QEMU emulation -> parity check -> self-correction feedback loop.
"""

import os
import shutil
import tempfile
from typing import Any, Callable, Dict, Optional

from scaffolding.executor import ExecutionResult, ScaffoldingExecutor
from scaffolding.llm_provider import LLMProvider


class LoopAgent:
    """
    Closed-loop autonomous state machine engine with max 5 attempts.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        executor: Optional[ScaffoldingExecutor] = None,
        max_attempts: int = 5,
    ) -> None:
        self.llm_provider = llm_provider or LLMProvider()
        self.executor = executor or ScaffoldingExecutor()
        self.max_attempts = max_attempts
        self.cumulative_cost_usd: float = 0.0

    def run_autonomous_loop(
        self,
        prompt_text: str,
        target: str = "RV64GCV",
        model_name: str = "Ollama: qwen2.5-coder (Local / Free)",
        api_key: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute closed-loop iterative build & verification cycle.
        Returns final structured result payload with 9-file workspace.
        """
        def log(msg: str) -> None:
            if log_callback:
                log_callback(msg)

        self.cumulative_cost_usd = 0.0
        log("🚀 Starting Tatva Antigravity Engine — Autonomous Closed-Loop Cycle")
        log(f"  Target: {target} | LLM Backend: {model_name} | Max Attempts: {self.max_attempts}")

        temp_dir = tempfile.mkdtemp(prefix="tatva_loop_")
        try:
            workspace_files, workspace_source = self._generate_initial_workspace(
                prompt_text, target, model_name, api_key, log
            )
            current_attempt = 1
            last_exec_result: Optional[ExecutionResult] = None

            while current_attempt <= self.max_attempts:
                log(f"\n🔄 [Attempt {current_attempt}/{self.max_attempts}] Writing workspace files to sandbox...")
                self._write_files_to_dir(temp_dir, workspace_files)

                # Step A: Cross-Compilation
                log(f"  [1/3 Cross-Compile GCC] Compiling workspace targeting {target}...")
                comp_res = self.executor.compile_workspace(temp_dir, target)

                if not comp_res.success:
                    log(f"  ❌ Compilation Error (Return Code: {comp_res.return_code}):")
                    log(f"     {comp_res.stderr[:300]}")

                    if current_attempt < self.max_attempts:
                        log("  🤖 Triggering Autonomous LLM Self-Correction Feedback Loop...")
                        fix_prompt = self._construct_feedback_prompt(
                            "compilation", comp_res, prompt_text, workspace_files
                        )
                        before = dict(workspace_files)
                        workspace_files = self._apply_self_correction(
                            fix_prompt, model_name, api_key, workspace_files, log
                        )
                        if workspace_files == before:
                            log("  ⛔ Workspace unchanged — retrying would repeat this failure. Stopping.")
                            last_exec_result = comp_res
                            break
                        current_attempt += 1
                        continue
                    else:
                        last_exec_result = comp_res
                        break

                log("  ✅ Cross-Compilation Succeeded! Binary built cleanly.")

                # Step B: QEMU Emulation
                log("  [2/3 QEMU Emulation] Executing binary in simulator...")
                emu_res = self.executor.emulate_workspace(temp_dir, target)

                if not emu_res.success:
                    log(f"  ❌ QEMU Emulation Crash (Return Code: {emu_res.return_code}):")
                    log(f"     {emu_res.stderr[:300]}")

                    if current_attempt < self.max_attempts:
                        log("  🤖 Triggering Autonomous LLM Self-Correction Feedback Loop...")
                        fix_prompt = self._construct_feedback_prompt(
                            "emulation", emu_res, prompt_text, workspace_files
                        )
                        before = dict(workspace_files)
                        workspace_files = self._apply_self_correction(
                            fix_prompt, model_name, api_key, workspace_files, log
                        )
                        if workspace_files == before:
                            log("  ⛔ Workspace unchanged — retrying would repeat this failure. Stopping.")
                            last_exec_result = emu_res
                            break
                        current_attempt += 1
                        continue
                    else:
                        last_exec_result = emu_res
                        break

                log("  ✅ QEMU Emulation Succeeded! 0 core dumps or exceptions.")

                # Step C: Parity & Metrics Verification
                #
                # NOTE: the generated starter workspace has no reference output to compare
                # against, so there is nothing to verify numerically here. We report that
                # honestly rather than claiming a match that was never computed.
                log("  [3/3 Parity & Verification] Skipped — scaffolded starter projects")
                log("      carry no reference outputs. Use 'tatva optimize' on a real")
                log("      .onnx model for verified numerical parity against ONNX Runtime.")

                last_exec_result = emu_res
                log(f"\n🎉 Closed-Loop Cycle Completed Successfully in Attempt {current_attempt}!")
                break

            files_list = [
                {"path": rel_path, "content": content} for rel_path, content in workspace_files.items()
            ]

            return {
                "success": (last_exec_result.success if last_exec_result else False),
                "project_name": "tatva_riscv_antigravity_starter",
                "target": target,
                "attempts_used": min(current_attempt, self.max_attempts),
                "cumulative_cost_usd": round(self.cumulative_cost_usd, 6),
                "files": files_list,
                # "llm" means the prompt actually shaped the code; "template" means no
                # model was reachable and the caller got the built-in starter instead.
                # Without this the two outcomes are indistinguishable to the UI.
                "workspace_source": workspace_source,
                # The sandbox is deleted below; files are returned in-memory above.
                # Callers must write them somewhere durable themselves.
                "workspace_dir": None,
                "last_stage": last_exec_result.stage if last_exec_result else None,
                "last_stderr": (last_exec_result.stderr[:2000] if last_exec_result else ""),
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _construct_feedback_prompt(
        self,
        stage: str,
        exec_res: ExecutionResult,
        prompt_text: str,
        current_files: Dict[str, str],
    ) -> str:
        """Construct targeted diagnostic self-correction feedback prompt."""
        return (
            f"You generated code for a RISC-V target, but execution failed during the [{stage}] stage.\n\n"
            f"ORIGINAL PROMPT: {prompt_text}\n"
            f"RETURN CODE: {exec_res.return_code}\n\n"
            f"STDERR OUTPUT:\n{exec_res.stderr[:1000]}\n\n"
            f"STDOUT OUTPUT:\n{exec_res.stdout[:500]}\n\n"
            "TASK: Fix ONLY the files that caused this error. Maintain compatibility with the rest of the workspace.\n"
            "Return the updated workspace files."
        )

    def _apply_self_correction(
        self,
        fix_prompt: str,
        model_name: str,
        api_key: Optional[str],
        current_files: Dict[str, str],
        log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, str]:
        """
        Query the LLM for corrected files.

        Failures are reported, not swallowed: a silent failure here looks identical to a
        successful correction and makes the loop appear to work when it never ran.
        """
        def _log(msg: str) -> None:
            if log:
                log(msg)

        try:
            res_text, cost = self.llm_provider.query(
                prompt=fix_prompt,
                system_prompt="You are Tatva's Autonomous RISC-V Self-Correction Agent.",
                messages=[],
                model_name=model_name,
                api_key=api_key,
            )
            self.cumulative_cost_usd += cost or 0.0
        except Exception as e:
            _log(f"     ⚠ Self-correction unavailable: {e}")
            _log("     Workspace left unchanged; the next attempt would repeat this failure.")
            return current_files

        # Same parser as the initial generation, so a correction cannot smuggle in a
        # path that escapes the sandbox (`../../..`) that generation would have rejected.
        files = self._parse_workspace_payload(res_text)
        if not files:
            _log("     ⚠ LLM response contained no usable 'files' payload; workspace unchanged.")
            return current_files

        current_files.update(files)
        _log(f"     Applied {len(files)} corrected file(s).")
        return current_files

    def _write_files_to_dir(self, target_dir: str, files_dict: Dict[str, str]) -> None:
        root = os.path.realpath(target_dir)
        for rel_path, content in files_dict.items():
            abs_p = os.path.realpath(os.path.join(root, rel_path))
            # Last line of defence: these paths came from a language model, and the
            # only thing between them and an arbitrary overwrite is this check.
            if abs_p != root and not abs_p.startswith(root + os.sep):
                raise ValueError(f"Refusing to write outside the sandbox: {rel_path!r}")
            os.makedirs(os.path.dirname(abs_p), exist_ok=True)
            with open(abs_p, "w", encoding="utf-8") as f:
                f.write(content)

    # Files the generated workspace must contain for scripts/build.sh and
    # CMakeLists.txt to work. An LLM reply missing any of these is rejected rather
    # than written to disk half-formed.
    REQUIRED_WORKSPACE_FILES = ("src/main.c", "CMakeLists.txt")

    def _generate_initial_workspace(
        self,
        prompt_text: str,
        target: str,
        model_name: str = "",
        api_key: Optional[str] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> tuple[Dict[str, str], str]:
        """
        Generate the starting workspace from the user's prompt.

        Returns (files, source) where source is "llm" or "template". This used to
        ignore prompt_text entirely and return a fixed 9-file template -- the prompt
        appeared only quoted in the generated README -- while the UI described an
        autonomous engine building what you asked for. Now the prompt is really sent
        to the configured model, and when no model is reachable the caller is told
        the template was used instead of being left to assume otherwise.
        """
        def _log(msg: str) -> None:
            if log:
                log(msg)

        template = self._builtin_workspace_template(prompt_text, target)

        if not model_name:
            _log("  ⚠ No LLM backend selected — using the built-in starter template.")
            return template, "template"

        system_prompt = (
            "You are Tatva's RISC-V scaffolding agent. You emit complete, compilable "
            "bare-metal C99 projects for RISC-V targets.\n"
            "Reply with JSON ONLY, no prose and no markdown fences, in exactly this shape:\n"
            '{"files": [{"path": "src/main.c", "content": "..."}]}\n'
            f"The project MUST include these paths: {', '.join(self.REQUIRED_WORKSPACE_FILES)}. "
            "Include a CMakeLists.txt that builds every .c file you emit. "
            "Do not reference headers or sources you did not include."
        )
        gen_prompt = (
            f"Target architecture: {target}\n"
            f"Build the project described here:\n{prompt_text}\n\n"
            "Emit the full contents of every file. The code must compile with a "
            "riscv GCC cross-compiler and run under qemu without an operating system."
        )

        _log(f"  🤖 Generating initial workspace from your prompt via {model_name}...")
        try:
            res_text, cost = self.llm_provider.query(
                prompt=gen_prompt,
                system_prompt=system_prompt,
                messages=[],
                model_name=model_name,
                api_key=api_key,
            )
            self.cumulative_cost_usd += cost or 0.0
        except Exception as e:
            _log(f"  ⚠ LLM unavailable ({e}) — using the built-in starter template.")
            return template, "template"

        files = self._parse_workspace_payload(res_text)
        if files is None:
            _log("  ⚠ LLM reply was not the expected JSON — using the built-in starter template.")
            return template, "template"

        missing = [p for p in self.REQUIRED_WORKSPACE_FILES if p not in files]
        if missing:
            _log(f"  ⚠ LLM workspace omitted {', '.join(missing)} — using the built-in starter template.")
            return template, "template"

        _log(f"  ✅ LLM generated {len(files)} file(s) from your prompt.")
        return files, "llm"

    @staticmethod
    def _parse_workspace_payload(res_text: str) -> Optional[Dict[str, str]]:
        """
        Pull {"files": [{"path", "content"}]} out of an LLM reply.

        Models routinely wrap JSON in prose or a markdown fence, so we take the widest
        brace span rather than requiring the whole reply to parse. Returns None when
        nothing usable is present -- an empty dict would be mistaken for a valid
        (but empty) workspace.
        """
        import json

        if not res_text or "{" not in res_text or "}" not in res_text:
            return None
        blob = res_text[res_text.find("{") : res_text.rfind("}") + 1]
        try:
            data = json.loads(blob)
        except Exception:
            return None

        entries = data.get("files") if isinstance(data, dict) else None
        if not isinstance(entries, list) or not entries:
            return None

        files: Dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path, content = entry.get("path"), entry.get("content")
            if not isinstance(path, str) or not isinstance(content, str) or not path.strip():
                continue
            # Keep the LLM inside the workspace: a path that escapes it would be
            # written outside the sandbox by _write_files_to_dir.
            norm = os.path.normpath(path.replace("\\", "/")).replace(os.sep, "/")
            if norm.startswith("..") or os.path.isabs(norm) or ":" in norm:
                continue
            files[norm] = content

        return files or None

    def _builtin_workspace_template(self, prompt_text: str, target: str) -> Dict[str, str]:
        """The fixed 9-file starter project, used when no LLM is reachable."""
        return {
            "src/main.c": """\
/*
 * Main Driver App for RISC-V Model Execution.
 * Generated by Tatva Antigravity Engine.
 */
#include <stdio.h>
#include <stdint.h>
#include "../include/riscv_config.h"

extern int run_model_inference(void);

int main(void) {
    printf("[TATVA ANTIGRAVITY ENGINE] Running RISC-V Model Execution Harness...\\n");
    int res = run_model_inference();
    printf("[TATVA ANTIGRAVITY ENGINE] Inference Execution Finished with status: %d\\n", res);
    return res;
}
""",
            "src/model_inference.c": """\
/*
 * Model Execution Harness & Tensor Benchmark Loop.
 */
#include <stdio.h>
#include <stdint.h>
#include "../include/riscv_config.h"

int run_model_inference(void) {
    uint8_t input_tensor[32];
    for (int i = 0; i < 32; i++) {
        input_tensor[i] = (uint8_t)(i * 3);
    }
    return 0;
}
""",
            "src/riscv_vector_kernel.s": """\
/*
 * RISC-V Vector Assembly Acceleration Kernel (RV64GCV).
 */
.global riscv_vec_add
.type riscv_vec_add, @function
riscv_vec_add:
    ret
""",
            "include/riscv_config.h": """\
/*
 * Memory Alignment & CSR Definitions.
 */
#ifndef RISCV_CONFIG_H
#define RISCV_CONFIG_H

#define RISCV_TENSOR_ALIGNMENT 16
#define RISCV_MAX_BATCH_SIZE 4

#endif
""",
            "scripts/build.sh": """\
#!/usr/bin/env bash
mkdir -p build && cd build
cmake ..
make -j4
""",
            "scripts/run_qemu.sh": """\
#!/usr/bin/env bash
qemu-system-riscv64 -M virt -cpu rv64,v=true,vext_spec=v1.0 -nographic -kernel build/build_app.elf
""",
            "tests/test_parity.py": '''\
"""
Numerical Parity Verification Test Suite.

These are placeholders. Replace the bodies with real comparisons against a
reference implementation before trusting them.
"""
import pytest


@pytest.mark.skip(reason="Placeholder: no reference output wired up yet.")
def test_numerical_parity():
    raise NotImplementedError("Compare model output against a reference run here.")
''',
            "CMakeLists.txt": """\
cmake_minimum_required(VERSION 3.10)
project(tatva_riscv_antigravity C)

set(CMAKE_C_STANDARD 99)
include_directories(include)

add_executable(build_app.elf src/main.c src/model_inference.c)
""",
            "README.md": f"""\
# TATVA ANTIGRAVITY ENGINE — Autonomous RISC-V Starter Project

> Prompt: *"{prompt_text}"*
> Target Architecture: `{target}`

## Project Structure
- `src/main.c`: Driver app & main loop
- `src/model_inference.c`: Execution harness
- `src/riscv_vector_kernel.s`: Vector assembly optimization
- `include/riscv_config.h`: CSR & alignment config
- `scripts/build.sh`: GCC cross-compilation pipeline script
- `scripts/run_qemu.sh`: QEMU invocation command script
- `tests/test_parity.py`: Parity verification tests
- `CMakeLists.txt`: Build manifest
- `README.md`: Project documentation
""",
        }
