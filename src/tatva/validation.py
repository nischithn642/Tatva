"""
TATVA validation stage.

Benchmarking answers "how fast is it". Validation answers a different and prior
question: "is this build correct, and how do we know". They were the same page in the
studio, which meant a fast build read as a correct one.

The rule this module exists to enforce: a check that TATVA does not perform is never
reported as having passed. Checks TATVA genuinely runs report PASSED or FAILED with the
evidence behind them. Checks that are on the roadmap and not implemented report
NOT_IMPLEMENTED, which the UI renders as "Coming Soon" -- visibly different from a
pass, never green, and never counted in the summary as a check that succeeded.

Every check carries `evidence`: the concrete thing that was observed. A pass with no
evidence is not a pass, it is an assertion.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

PASSED = "PASSED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass
class Check:
    key: str
    name: str
    status: str
    detail: str
    evidence: str = ""
    category: str = "correctness"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    not_implemented: int = 0
    verdict: str = "UNKNOWN"
    summary: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "checks": [c.to_json() for c in self.checks],
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "not_implemented": self.not_implemented,
            "verdict": self.verdict,
            "summary": self.summary,
        }


# Checks TATVA does not perform. They are listed rather than hidden, because an
# engineer evaluating the tool needs to know what it does *not* check just as much as
# what it does -- but they can never be mistaken for a result.
_ROADMAP: list[tuple[str, str, str, str]] = [
    ("memory_footprint", "Static memory footprint against a device budget", "resource",
     "TATVA does not yet model a target's RAM and flash budget, so it cannot tell you whether this "
     "binary fits a specific part. The ELF size is reported under Artifacts; the comparison is not made."),
    ("silicon_timing", "Timing validated on real silicon", "performance",
     "Timings come from QEMU with a nominal clock. Nothing in TATVA has run this binary on a physical "
     "RISC-V device, so no silicon-timing claim is made."),
    ("determinism", "Bit-exact repeatability across runs", "correctness",
     "TATVA measures a single emulated run per configuration and does not yet re-run to confirm the "
     "output is bit-identical."),
    ("quantization_sweep", "Quantization accuracy across a calibration set", "accuracy",
     "The INT8 activation scale is calibrated on generated inputs and parity is checked on one input. "
     "Accuracy across a real calibration dataset is not evaluated."),
    ("power", "Energy and power characterisation", "performance",
     "TATVA has no power model and takes no power measurement."),
]


def _roadmap_checks() -> list[Check]:
    return [
        Check(key=key, name=name, status=NOT_IMPLEMENTED, detail=detail, evidence="", category=category)
        for key, name, category, detail in _ROADMAP
    ]


def evaluate(run: dict[str, Any]) -> ValidationReport:
    """
    Build the validation report for a run.

    `run` is the structured record the pipeline assembles -- graph analysis, repair
    result, build directories, measurement and parity outcome. Nothing is inferred from
    console text; every check reads a field that the stage which produced it set.
    """
    checks: list[Check] = []

    # ---- 1. the model became an IR at all
    analysis = run.get("analysis") or {}
    total_ops = int(analysis.get("total_ops") or 0)
    if total_ops > 0:
        checks.append(Check(
            key="graph_import", name="Model imported into TATVA's graph IR", status=PASSED,
            detail="The ONNX graph was read and lowered to TVM Relax.",
            evidence=f"{total_ops} operator calls across {analysis.get('distinct_ops', 0)} distinct kinds.",
            category="correctness",
        ))
    else:
        checks.append(Check(
            key="graph_import", name="Model imported into TATVA's graph IR", status=FAILED,
            detail=run.get("import_error") or "The model was never imported, so nothing downstream ran.",
            category="correctness",
        ))

    # ---- 2. every operator has a lowering
    unsupported = list(run.get("unsupported_ops") or [])
    target = run.get("target_name") or "the selected target"
    if total_ops <= 0:
        checks.append(Check(
            key="operator_coverage", name="Every operator has a lowering on the target", status=SKIPPED,
            detail="Not reached: the graph was never imported.", category="correctness",
        ))
    elif unsupported:
        checks.append(Check(
            key="operator_coverage", name="Every operator has a lowering on the target", status=FAILED,
            detail=f"{len(unsupported)} operator kind(s) have no lowering on {target}.",
            evidence=", ".join(unsupported), category="correctness",
        ))
    else:
        checks.append(Check(
            key="operator_coverage", name="Every operator has a lowering on the target", status=PASSED,
            detail=f"All {analysis.get('distinct_ops', 0)} operator kinds map to a kernel on {target}.",
            evidence="Checked against the target capability database.", category="correctness",
        ))

    # ---- 3 & 4. the repair engine's own two checks, reported only when it ran
    repair = run.get("repair") or {}
    if repair.get("attempted"):
        struct = repair.get("structural_validation") or "not_run"
        checks.append(Check(
            key="rewrite_structural", name="Graph rewrites preserve the graph's structure",
            status=PASSED if struct == "passed" else FAILED if struct == "failed" else SKIPPED,
            detail=(
                "Every rewritten graph was re-checked for well-formedness, operator coverage and an "
                "unchanged output signature."
            ),
            evidence=repair.get("message", ""), category="correctness",
        ))
        num = repair.get("numerical_validation") or "not_run"
        diff = repair.get("max_abs_diff")
        if num == "passed":
            checks.append(Check(
                key="rewrite_numeric", name="Graph rewrites preserve the model's output", status=PASSED,
                detail="The original and rewritten graphs were both executed on the host and compared.",
                evidence=f"Maximum absolute difference {diff:.3g}." if isinstance(diff, (int, float)) else "",
                category="correctness",
            ))
        elif num == "failed":
            checks.append(Check(
                key="rewrite_numeric", name="Graph rewrites preserve the model's output", status=FAILED,
                detail="A rewrite changed the model's output and was discarded.",
                evidence=f"Maximum absolute difference {diff:.3g}." if isinstance(diff, (int, float)) else "",
                category="correctness",
            ))
        else:
            checks.append(Check(
                key="rewrite_numeric", name="Graph rewrites preserve the model's output", status=SKIPPED,
                detail=(
                    "The host could not execute both graphs, so the rewrites were not compared numerically "
                    "here. The end-to-end parity check below still applies to the final build."
                ),
                category="correctness",
            ))

    # ---- 5. code generation actually wrote files
    build_dirs = {k: v for k, v in (run.get("build_dirs") or {}).items() if v}
    generated = int(run.get("generated_files") or 0)
    if generated > 0:
        checks.append(Check(
            key="codegen", name="Source generated for the target", status=PASSED,
            detail="TVM lowered the graph to C and the bare-metal harness was written alongside it.",
            evidence=f"{generated} file(s) across {len(build_dirs)} build director"
                     f"{'y' if len(build_dirs) == 1 else 'ies'}, {run.get('generated_lines', 0)} lines.",
            category="build",
        ))
    else:
        checks.append(Check(
            key="codegen", name="Source generated for the target",
            status=FAILED if run.get("build_attempted") else SKIPPED,
            detail=run.get("build_error") or "No generated source was found on disk.",
            category="build",
        ))

    # ---- 6. it linked
    elfs = [
        os.path.join(d, "model.elf") for d in build_dirs.values()
        if os.path.isfile(os.path.join(d, "model.elf"))
    ]
    if elfs:
        sizes = ", ".join(f"{os.path.getsize(p):,} bytes" for p in elfs)
        checks.append(Check(
            key="cross_compile", name="Cross-compiled and linked for the target", status=PASSED,
            detail="The generated C was compiled by the RISC-V cross-compiler and linked against the target memory map.",
            evidence=f"{len(elfs)} ELF binar{'y' if len(elfs) == 1 else 'ies'}: {sizes}.",
            category="build",
        ))
    elif run.get("build_attempted"):
        checks.append(Check(
            key="cross_compile", name="Cross-compiled and linked for the target", status=FAILED,
            detail=run.get("build_error") or "No linked binary was produced.", category="build",
        ))
    else:
        checks.append(Check(
            key="cross_compile", name="Cross-compiled and linked for the target", status=SKIPPED,
            detail="Not reached.", category="build",
        ))

    # ---- 7. it ran to completion under emulation
    measured = bool(run.get("measured"))
    if measured:
        checks.append(Check(
            key="target_execution", name="Binary runs to completion under emulation", status=PASSED,
            detail="The binary booted, executed the timed loop and reported cycles over the SBI console.",
            evidence=f"Environment {run.get('environment', 'QEMU_SIM')}; "
                     f"baseline {run.get('base_ms')} ms at a nominal {run.get('nominal_clock_mhz', 100)} MHz.",
            category="build",
        ))
    else:
        checks.append(Check(
            key="target_execution", name="Binary runs to completion under emulation",
            status=FAILED if run.get("build_attempted") else SKIPPED,
            detail=run.get("build_error") or "Nothing was executed, so nothing was measured.",
            category="build",
        ))

    # ---- 8. the numbers coming off the target match the host
    if not measured:
        checks.append(Check(
            key="numerical_parity", name="Target output matches the host reference", status=SKIPPED,
            detail="Not reached: no run completed.", category="accuracy",
        ))
    elif run.get("parity_applicable") is False:
        checks.append(Check(
            key="numerical_parity", name="Target output matches the host reference", status=SKIPPED,
            detail="No optimization pass was selected, so there is no second build to compare against the first.",
            category="accuracy",
        ))
    elif run.get("accuracy_ok"):
        checks.append(Check(
            key="numerical_parity", name="Target output matches the host reference", status=PASSED,
            detail=f"Output compared against {run.get('accuracy_reference', 'host ONNX Runtime')}.",
            evidence=f"MSE {run.get('mse')} against a tolerance of {run.get('tolerance')}.",
            category="accuracy",
        ))
    else:
        checks.append(Check(
            key="numerical_parity", name="Target output matches the host reference", status=FAILED,
            detail=f"Output diverges from {run.get('accuracy_reference', 'host ONNX Runtime')} beyond tolerance.",
            evidence=f"MSE {run.get('mse')} against a tolerance of {run.get('tolerance')}.",
            category="accuracy",
        ))

    checks.extend(_roadmap_checks())

    passed = sum(1 for c in checks if c.status == PASSED)
    failed = sum(1 for c in checks if c.status == FAILED)
    skipped = sum(1 for c in checks if c.status == SKIPPED)
    pending = sum(1 for c in checks if c.status == NOT_IMPLEMENTED)

    if failed:
        verdict = "FAILED"
        summary = f"{failed} check(s) failed. {passed} passed."
    elif passed == 0:
        verdict = "NOT_RUN"
        summary = "No validation check completed for this run."
    elif skipped:
        verdict = "PARTIAL"
        summary = f"{passed} check(s) passed, {skipped} could not run."
    else:
        verdict = "PASSED"
        summary = f"All {passed} implemented checks passed."
    # The roadmap items are named in the summary so a "PASSED" verdict cannot be read
    # as "everything worth checking was checked".
    summary += f" {pending} further check(s) are not implemented and were not evaluated."

    return ValidationReport(
        checks=checks, passed=passed, failed=failed, skipped=skipped,
        not_implemented=pending, verdict=verdict, summary=summary,
    )
