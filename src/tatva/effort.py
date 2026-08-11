"""
TATVA engineering-effort estimation.

What this answers: given what this compiler run actually did, how much hand work would
it have taken to reach the same place without it?

What this is not: a measurement. Nobody timed an engineer doing this by hand, and this
module does not pretend otherwise. Every figure it produces is an estimate built from
two clearly separated halves:

  * Quantities  -- counted from the run. Operator kinds lowered, calls in the graph,
                   lines of C actually written to disk, rewrites actually applied,
                   builds actually linked, benchmark runs actually measured. These are
                   real. If they cannot be counted, no estimate is produced.

  * Rates       -- hours per unit of the above. These are assumptions of the effort
                   model, declared as data, versioned, shown in full in the UI's
                   "How is this calculated?" panel, and overridable from a JSON file
                   so nobody has to take the defaults on trust.

The separation is the whole point. A reader can accept the counts, reject the rates,
substitute their own, and recompute -- which is what makes the number auditable rather
than promotional. The result is labelled an estimate everywhere it appears, and it
carries the full breakdown so the top-line figure can always be traced back to the
counts it came from.

There is exactly one implementation of this arithmetic. The GUI, the report page and
`engineering_effort.json` all render the object this module returns; none of them adds,
scales or re-derives anything. That is deliberate -- an effort figure computed in two
places is an effort figure that will eventually disagree with itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

EFFORT_FILENAME = "engineering_effort.json"

# Bumped whenever a rate or a line item changes, so two results are never silently
# comparable across different methodologies.
EFFORT_MODEL_VERSION = "1.0"

# Where an override may live. A site that disagrees with the default rates points this
# at its own table rather than editing the source.
EFFORT_MODEL_ENV = "TATVA_EFFORT_MODEL"


@dataclass
class RateLine:
    """
    One line of the effort model.

    `basis` is required and is written to be read by a sceptic: it says where the rate
    came from and, where the honest answer is "this is an assumption", it says that.
    """
    key: str
    label: str
    unit: str
    hours_per_unit: float
    basis: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- the model
#
# Defaults. Each rate is an engineering judgement about bare-metal RISC-V work, not a
# measurement, and each says so. They are deliberately conservative: the goal is a
# figure that survives being argued with, not the largest number that can be justified.

DEFAULT_RATES: list[RateLine] = [
    RateLine(
        key="operator_kinds",
        label="Hand-writing a bare-metal kernel per distinct operator",
        unit="operator kind",
        hours_per_unit=4.0,
        basis=(
            "Assumption. Covers writing a scalar C kernel for one operator against fixed shapes, "
            "plus a correctness check against a host reference. Excludes tuning."
        ),
    ),
    RateLine(
        key="operator_calls",
        label="Wiring the call sequence and intermediate buffers",
        unit="operator call",
        hours_per_unit=0.15,
        basis=(
            "Assumption. Per call in the graph: allocating or aliasing its output buffer and "
            "placing it in the right order in the forward pass."
        ),
    ),
    RateLine(
        key="generated_lines",
        label="Writing the equivalent source by hand",
        unit="100 lines of C / asm / linker script",
        hours_per_unit=1.0,
        basis=(
            "Assumption, applied per 100 lines. The line count itself is counted from the files "
            "on disk, not estimated."
        ),
    ),
    RateLine(
        key="repaired_op_kinds",
        label="Diagnosing and rewriting an unsupported operator",
        unit="operator kind rewritten",
        hours_per_unit=3.0,
        basis=(
            "Assumption. Covers identifying that an operator has no lowering, finding an exact "
            "decomposition, applying it and re-verifying numerics."
        ),
    ),
    RateLine(
        key="target_bringup",
        label="Target bring-up: linker script, reset vector, console, timing",
        unit="build configuration",
        hours_per_unit=6.0,
        basis=(
            "Assumption, charged once per build configuration produced. Covers the memory map, "
            "the startup path before main(), the SBI console and cycle-counter plumbing."
        ),
    ),
    RateLine(
        key="benchmark_runs",
        label="Standing up an emulated measurement and reading it back",
        unit="measured run",
        hours_per_unit=1.5,
        basis=(
            "Assumption. Covers the emulator invocation, the timing harness and extracting "
            "cycles and outputs from the console."
        ),
    ),
]


@dataclass
class EffortInputs:
    """
    The counted quantities. Every field is a real count taken from the run; there are
    no defaults that stand in for a measurement that did not happen.
    """
    distinct_operator_kinds: int = 0
    total_operator_calls: int = 0
    generated_files: int = 0
    generated_lines: int = 0
    repaired_op_kinds: int = 0
    build_configs: int = 0
    benchmark_runs: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EffortResult:
    available: bool
    reason: str = ""
    total_hours: float = 0.0
    total_days: float = 0.0
    model_version: str = EFFORT_MODEL_VERSION
    model_source: str = "built-in defaults"
    kind: str = "estimate"
    measured: bool = False
    disclaimer: str = ""
    hours_per_day: float = 8.0
    inputs: dict[str, Any] = field(default_factory=dict)
    breakdown: list[dict[str, Any]] = field(default_factory=list)
    rates: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# A working day, used only to render hours in a second unit. Stated as data so the UI
# does not divide by its own idea of a day.
HOURS_PER_DAY = 8.0

DISCLAIMER = (
    "Estimate, not a measurement. The quantities below were counted from this compiler run. "
    "The hours-per-unit rates are assumptions of TATVA's effort model, listed in full so they "
    "can be checked or replaced. No engineer was timed to produce this figure."
)


def load_rates() -> tuple[list[RateLine], str]:
    """
    Load the rate table, honouring an override file when one is configured.

    Returns the table and a human-readable description of where it came from, which is
    carried into the result so the UI can say which model produced a figure.
    """
    path = os.environ.get(EFFORT_MODEL_ENV, "").strip()
    if not path:
        return list(DEFAULT_RATES), "built-in defaults"
    if not os.path.isfile(path):
        return list(DEFAULT_RATES), f"built-in defaults ({EFFORT_MODEL_ENV} points at a missing file: {path})"
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        rates = [
            RateLine(
                key=str(item["key"]),
                label=str(item["label"]),
                unit=str(item["unit"]),
                hours_per_unit=float(item["hours_per_unit"]),
                basis=str(item.get("basis", "Supplied by an override file; no basis given.")),
            )
            for item in raw["rates"]
        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return list(DEFAULT_RATES), f"built-in defaults (override at {path} could not be read: {exc})"
    if not rates:
        return list(DEFAULT_RATES), f"built-in defaults (override at {path} contained no rates)"
    return rates, f"override file: {path}"


def _quantity_for(key: str, inputs: EffortInputs) -> tuple[float, str]:
    """
    Map a rate key to the counted quantity it applies to, plus where that count came
    from. Returning the provenance alongside the number is what lets the UI show a
    per-line "counted from" without the frontend guessing.
    """
    if key == "operator_kinds":
        return float(inputs.distinct_operator_kinds), "distinct operator kinds in the graph (graph analysis)"
    if key == "operator_calls":
        return float(inputs.total_operator_calls), "operator calls in the graph (graph analysis)"
    if key == "generated_lines":
        return inputs.generated_lines / 100.0, f"{inputs.generated_lines} lines counted in the generated files on disk"
    if key == "repaired_op_kinds":
        return float(inputs.repaired_op_kinds), "operator kinds the repair engine rewrote and validated"
    if key == "target_bringup":
        return float(inputs.build_configs), "build configurations linked in this run"
    if key == "benchmark_runs":
        return float(inputs.benchmark_runs), "runs measured under emulation"
    return 0.0, "no counted quantity is mapped to this rate"


def compute(inputs: EffortInputs) -> EffortResult:
    """
    Produce the effort estimate, or refuse to.

    Refusal is a first-class outcome. A run that never reached code generation has no
    generated lines to count and no build to charge bring-up against; there is no
    defensible figure for it, so none is produced and the reason is stated. Filling the
    gap with a plausible number is the exact failure this module is built to avoid.
    """
    rates, source = load_rates()
    now = datetime.now(UTC).isoformat(timespec="seconds")

    missing: list[str] = []
    if inputs.distinct_operator_kinds <= 0:
        missing.append("the graph was never analysed, so no operator counts exist")
    if inputs.generated_files <= 0 or inputs.generated_lines <= 0:
        missing.append("no source was generated, so there is nothing to count against")
    if inputs.build_configs <= 0:
        missing.append("no build configuration was linked")

    if missing:
        return EffortResult(
            available=False,
            reason=(
                "No engineering-effort estimate for this run: "
                + "; ".join(missing)
                + ". TATVA reports an estimate only when it can count the work that was done."
            ),
            model_version=EFFORT_MODEL_VERSION,
            model_source=source,
            disclaimer=DISCLAIMER,
            inputs=inputs.to_json(),
            rates=[r.to_json() for r in rates],
            generated_at=now,
        )

    breakdown: list[dict[str, Any]] = []
    total = 0.0
    for rate in rates:
        qty, provenance = _quantity_for(rate.key, inputs)
        hours = qty * rate.hours_per_unit
        total += hours
        breakdown.append({
            "key": rate.key,
            "label": rate.label,
            "unit": rate.unit,
            "quantity": round(qty, 4),
            "quantity_source": provenance,
            "hours_per_unit": rate.hours_per_unit,
            "hours": round(hours, 2),
            "basis": rate.basis,
        })

    return EffortResult(
        available=True,
        total_hours=round(total, 2),
        total_days=round(total / HOURS_PER_DAY, 2),
        model_version=EFFORT_MODEL_VERSION,
        model_source=source,
        kind="estimate",
        measured=False,
        disclaimer=DISCLAIMER,
        hours_per_day=HOURS_PER_DAY,
        inputs=inputs.to_json(),
        breakdown=breakdown,
        rates=[r.to_json() for r in rates],
        generated_at=now,
    )


def write_effort(result: EffortResult, build_dir: str, *, run_id: str = "") -> str:
    """Persist the estimate next to the artifacts it describes."""
    if not build_dir or not os.path.isdir(build_dir):
        return ""
    payload = {"schema": "tatva.engineering_effort/1", "run_id": run_id, **result.to_json()}
    path = os.path.join(build_dir, EFFORT_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError:
        return ""
    return path
