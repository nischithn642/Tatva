"""
TATVA run registry.

A compile is no longer one call returning one number. It analyses a graph, maps it
against a target, possibly repairs it, generates source, links, measures, validates and
estimates effort -- and the studio has a separate page for each of those. Those pages
need to be able to ask about a run *after* it finished, without recompiling and without
the frontend having to hold the whole result in a JavaScript variable and hand it back.

This module is that memory. One `RunRecord` per pipeline execution, addressed by
`run_id`, holding the structured output of every stage. The bridge writes it during the
run; `get_artifacts`, `get_validation`, `get_effort` and the rest read from it.

Two rules keep it honest:

  * Every stage writes what it observed, including failures. A stage that did not run
    leaves its field empty, and empty means "did not run" -- never a zero standing in
    for a measurement.

  * The run's overall `status` is derived, in one place, from what the stages recorded.
    A run is SUCCESS only when it compiled, linked, ran and passed the checks that
    applied to it. Everything else is PARTIAL, BLOCKED or FAILED, and carries the reason.

The registry is in-memory and bounded. Build directories persist on disk with their own
manifest, so a closed session loses the index, not the evidence.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from tatva.audit import AuditTrail

# Run outcomes, in the vocabulary the report and the studio both use.
SUCCESS = "SUCCESS"      # compiled, linked, ran, and every applicable check passed
PARTIAL = "PARTIAL"      # produced real output, but something did not hold
BLOCKED = "BLOCKED"      # stopped before code generation for a reason TATVA can name
FAILED = "FAILED"        # attempted and errored
RUNNING = "RUNNING"

# How many runs to keep. Enough to move between pages and compare a couple of attempts;
# small enough that a long session does not accumulate large result objects.
MAX_RUNS = 24


@dataclass
class RunRecord:
    run_id: str
    model_path: str = ""
    model_name: str = ""
    target_name: str = ""
    march: str = ""
    mabi: str = ""
    passes: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    status: str = RUNNING
    status_reason: str = ""
    error: str = ""
    diagnosis: str = ""

    # Per-stage structured output. Empty means the stage did not run.
    model_info: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    mapping: dict[str, Any] = field(default_factory=dict)
    repair: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    effort: dict[str, Any] = field(default_factory=dict)
    optimization_history: list[dict[str, Any]] = field(default_factory=list)

    build_dirs: dict[str, str] = field(default_factory=dict)
    trail: AuditTrail | None = None

    def summary(self) -> dict[str, Any]:
        """The header every page shows, small enough to send on every poll."""
        return {
            "run_id": self.run_id,
            "model": self.model_name,
            "model_path": self.model_path,
            "target": self.target_name,
            "march": self.march,
            "passes": list(self.passes),
            "status": self.status,
            "status_reason": self.status_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    def to_json(self) -> dict[str, Any]:
        out = self.summary()
        out.update({
            "model_info": self.model_info,
            "analysis": self.analysis,
            "mapping": self.mapping,
            "repair": self.repair,
            "benchmark": self.benchmark,
            "manifest": self.manifest,
            "validation": self.validation,
            "effort": self.effort,
            "optimization_history": self.optimization_history,
            "build_dirs": dict(self.build_dirs),
            "audit": self.trail.to_json() if self.trail else {},
        })
        return out


def derive_status(record: RunRecord) -> tuple[str, str]:
    """
    Decide a run's outcome from what its stages recorded.

    Kept in one function on purpose. The status appears on the pipeline header, the
    benchmark report and the artifact manifest, and three independent derivations of it
    would eventually disagree -- which is precisely the situation where a build gets
    labelled successful because one of the three forgot to check something.
    """
    if record.error:
        return FAILED, record.error

    # Nothing was built. Distinguish "we know why" from "it broke".
    if not record.build_dirs:
        # After a repair, the operators that still block are the ones the repair could
        # not express -- not the ones the mapping stage found before it ran. Naming the
        # pre-repair list here would tell the user an operator blocked their build when
        # TATVA had in fact just fixed it.
        unsupported = ((record.repair or {}).get("remaining_unsupported")
                       if record.repair else None)
        if unsupported is None:
            unsupported = (record.mapping or {}).get("unsupported") or []
        if unsupported:
            return BLOCKED, (
                f"{len(unsupported)} operator kind(s) have no lowering on {record.target_name} and no "
                f"validated rewrite: {', '.join(unsupported)}. Code generation did not start."
            )
        return FAILED, "No build was produced and no reason was recorded."

    measured = bool((record.benchmark or {}).get("measured"))
    if not measured:
        return FAILED, "The build was produced but no measurement completed."

    verdict = (record.validation or {}).get("verdict")
    if verdict == "FAILED":
        failed = [c["name"] for c in (record.validation or {}).get("checks", []) if c.get("status") == "FAILED"]
        return PARTIAL, (
            "The build compiled, linked and ran, but "
            + (f"{len(failed)} validation check(s) failed: {'; '.join(failed)}." if failed
               else "validation reported a failure.")
        )

    # A repair that could only fix part of the graph still yields a build here only if
    # the remaining operators never blocked codegen; say so rather than claim a clean run.
    remaining = (record.repair or {}).get("remaining_unsupported") or []
    if remaining:
        return PARTIAL, (
            f"Compiled and measured, but {len(remaining)} operator kind(s) were never mapped: "
            f"{', '.join(remaining)}."
        )

    if verdict == "PARTIAL":
        return PARTIAL, (record.validation or {}).get("summary", "Some validation checks could not run.")

    return SUCCESS, (record.validation or {}).get("summary", "Compiled, linked, ran and validated.")


class _Registry:
    """Bounded, thread-safe, insertion-ordered store of recent runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, RunRecord] = OrderedDict()
        self._counter = 0
        self._latest = ""

    def new_run(self, **kw: Any) -> RunRecord:
        with self._lock:
            self._counter += 1
            stamp = datetime.now(UTC)
            run_id = f"run-{stamp.strftime('%Y%m%d-%H%M%S')}-{self._counter:03d}"
            record = RunRecord(
                run_id=run_id,
                started_at=stamp.isoformat(timespec="seconds"),
                **kw,
            )
            record.trail = AuditTrail(run_id)
            self._runs[run_id] = record
            self._latest = run_id
            while len(self._runs) > MAX_RUNS:
                self._runs.popitem(last=False)
            return record

    def get(self, run_id: str = "") -> RunRecord | None:
        """Fetch by id, or the most recent run when the id is empty."""
        with self._lock:
            if run_id:
                return self._runs.get(run_id)
            return self._runs.get(self._latest)

    def finish(self, record: RunRecord) -> RunRecord:
        record.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        record.status, record.status_reason = derive_status(record)
        return record

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            return [r.summary() for r in list(self._runs.values())[-limit:][::-1]]


REGISTRY = _Registry()
