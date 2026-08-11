"""
TATVA audit trail -- the run's engineering evidence.

The studio shows conclusions: this operator mapped, that rewrite was applied, the build
linked, the speedup was 2.4x. This module records how each of those conclusions was
reached, in order, as the run happens.

Two properties matter and are enforced here rather than by convention:

  * Append-only. `record` adds; nothing edits or removes an entry once written. A run
    that went badly keeps the evidence that it went badly. If a stage is retried, both
    attempts are in the log.

  * Evidence, not narration. Every entry carries a structured `evidence` mapping of the
    values the stage actually observed -- counts, paths, hashes, differences, exit
    codes. An entry that says "validation passed" and carries nothing is worthless, so
    the entries here always carry the numbers they were derived from.

The trail is what turns "TATVA says this build is correct" into something a reviewer can
check without rerunning anything. It is written to the build directory as
`audit_trail.json` and surfaced in the studio under Engineering Evidence.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

AUDIT_FILENAME = "audit_trail.json"

# Outcome vocabulary. Deliberately small, and deliberately includes the unhappy ones --
# a log that can only express success is a marketing document.
OK = "ok"
WARN = "warn"
ERROR = "error"
INFO = "info"
BLOCKED = "blocked"


@dataclass
class AuditEvent:
    seq: int
    timestamp: str
    elapsed_s: float
    stage: str
    event: str
    outcome: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class AuditTrail:
    """
    An ordered log of what a single run did.

    Not thread-safe by design: one trail belongs to one run on one thread, and sharing
    one across concurrent runs would interleave two stories into one file.
    """

    def __init__(self, run_id: str = "") -> None:
        self.run_id = run_id
        self._events: list[AuditEvent] = []
        self._t0 = time.monotonic()
        self._started = datetime.now(UTC).isoformat(timespec="seconds")

    def record(
        self,
        stage: str,
        event: str,
        *,
        outcome: str = OK,
        detail: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Append one entry. The returned event is a copy of what was stored."""
        entry = AuditEvent(
            seq=len(self._events) + 1,
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            elapsed_s=round(time.monotonic() - self._t0, 3),
            stage=stage,
            event=event,
            outcome=outcome,
            detail=detail,
            evidence=_jsonable(evidence or {}),
        )
        self._events.append(entry)
        return entry

    # Convenience wrappers. They exist so call sites read as what happened rather than
    # as a status-code argument, which makes the log easier to keep honest.
    def ok(self, stage: str, event: str, **kw: Any) -> AuditEvent:
        return self.record(stage, event, outcome=OK, **kw)

    def info(self, stage: str, event: str, **kw: Any) -> AuditEvent:
        return self.record(stage, event, outcome=INFO, **kw)

    def warn(self, stage: str, event: str, **kw: Any) -> AuditEvent:
        return self.record(stage, event, outcome=WARN, **kw)

    def error(self, stage: str, event: str, **kw: Any) -> AuditEvent:
        return self.record(stage, event, outcome=ERROR, **kw)

    def blocked(self, stage: str, event: str, **kw: Any) -> AuditEvent:
        return self.record(stage, event, outcome=BLOCKED, **kw)

    @property
    def events(self) -> list[AuditEvent]:
        """A copy. Handing out the live list would make the trail editable."""
        return list(self._events)

    def counts(self) -> dict[str, int]:
        out = {OK: 0, INFO: 0, WARN: 0, ERROR: 0, BLOCKED: 0}
        for e in self._events:
            out[e.outcome] = out.get(e.outcome, 0) + 1
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "tatva.audit_trail/1",
            "run_id": self.run_id,
            "started_at": self._started,
            "event_count": len(self._events),
            "counts": self.counts(),
            "events": [e.to_json() for e in self._events],
        }

    def write(self, build_dir: str) -> str:
        """Persist the trail beside the artifacts. Returns the path, or "" if not written."""
        if not build_dir or not os.path.isdir(build_dir):
            return ""
        path = os.path.join(build_dir, AUDIT_FILENAME)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_json(), fh, indent=2)
        except OSError:
            return ""
        return path


def _jsonable(value: Any) -> Any:
    """
    Coerce evidence into something `json.dump` will accept.

    Evidence comes from all over the pipeline -- numpy scalars from the parity check,
    dataclasses from the repair engine, Path objects from the runner. Rather than make
    every call site remember to convert, anything unrecognised is stringified. Losing
    the exact type of a value is acceptable; losing the value because the log failed to
    serialise is not.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_json"):
        try:
            return _jsonable(value.to_json())
        except Exception:
            return str(value)
    # `continue`, not `break`: numpy's `.item()` raises on any array with more than one
    # element, and breaking there meant `.tolist()` was never reached. A shape or a
    # per-output difference vector -- exactly the evidence this log exists to carry --
    # fell through to `str()` and was recorded as "[2 4]", which no reader can parse
    # back into numbers.
    for attr in ("item", "tolist"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:
                continue
    return str(value)
