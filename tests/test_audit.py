"""
Tests for the audit trail.

Two properties are the whole value of this module: it is append-only, and every entry
carries the values it was derived from. A log that can be edited after the fact proves
nothing, and an entry that says "validation passed" while carrying no numbers is
narration.
"""

import json
import os

import pytest

from tatva.audit import AUDIT_FILENAME, BLOCKED, ERROR, INFO, OK, WARN, AuditTrail


@pytest.mark.unit
def test_events_are_numbered_in_the_order_they_happened() -> None:
    trail = AuditTrail("run-1")
    trail.ok("Import", "Model imported")
    trail.ok("Mapping", "Operators mapped")
    trail.warn("Repair", "One operator rewritten")

    assert [e.seq for e in trail.events] == [1, 2, 3]
    assert [e.stage for e in trail.events] == ["Import", "Mapping", "Repair"]
    assert all(e.elapsed_s >= 0 for e in trail.events)
    assert all(e.timestamp for e in trail.events)


@pytest.mark.unit
def test_the_live_list_is_not_handed_out() -> None:
    """`events` returning the internal list would make an append-only log editable by
    anyone holding a reference to it."""
    trail = AuditTrail("run-1")
    trail.ok("Import", "Model imported")

    snapshot = trail.events
    snapshot.clear()
    snapshot.append("nonsense")

    assert len(trail.events) == 1
    assert trail.events[0].event == "Model imported"


@pytest.mark.unit
def test_a_failure_stays_in_the_log() -> None:
    """A run that went badly keeps the evidence that it went badly; nothing removes an
    entry once written."""
    trail = AuditTrail("run-1")
    trail.error("Link", "Link failed", detail="undefined reference to expf")
    trail.ok("Link", "Retried with libm", detail="second attempt")

    outcomes = [e.outcome for e in trail.events]
    assert outcomes == [ERROR, OK]
    assert trail.counts()[ERROR] == 1


@pytest.mark.unit
def test_every_outcome_in_the_vocabulary_is_reachable() -> None:
    """The vocabulary deliberately includes the unhappy states -- a log that can only
    express success is a marketing document."""
    trail = AuditTrail("run-1")
    trail.ok("s", "a")
    trail.info("s", "b")
    trail.warn("s", "c")
    trail.error("s", "d")
    trail.blocked("s", "e")

    assert trail.counts() == {OK: 1, INFO: 1, WARN: 1, ERROR: 1, BLOCKED: 1}


@pytest.mark.unit
def test_evidence_is_preserved_as_structured_data() -> None:
    trail = AuditTrail("run-1")
    trail.ok(
        "Validation", "Rewrites compared on the host",
        detail="Original and rewritten graphs executed and compared.",
        evidence={"max_abs_diff": 1.19e-07, "tolerance": 1e-05, "outputs": 2},
    )
    event = trail.events[0]
    assert event.evidence["max_abs_diff"] == pytest.approx(1.19e-07)
    assert event.evidence["outputs"] == 2
    assert event.detail


@pytest.mark.unit
def test_numpy_and_dataclass_evidence_survives_serialisation() -> None:
    """
    Evidence arrives from all over the pipeline. Losing the exact type of a value is
    acceptable; losing the value because the log failed to serialise is not.
    """
    np = pytest.importorskip("numpy")

    class WithJson:
        def to_json(self):
            return {"op": "nn.silu", "occurrences": 3}

    trail = AuditTrail("run-1")
    trail.ok("Repair", "Rewrite applied", evidence={
        "diff": np.float32(0.25),
        "shape": np.array([2, 4]),
        "record": WithJson(),
        "paths": ("a.c", "b.h"),
        "nested": {"kinds": {"add", "multiply"}},
        "none": None,
    })

    payload = trail.to_json()
    json.dumps(payload)  # must not raise

    evidence = payload["events"][0]["evidence"]
    assert evidence["diff"] == pytest.approx(0.25)
    assert evidence["shape"] == [2, 4]
    assert evidence["record"] == {"op": "nn.silu", "occurrences": 3}
    assert evidence["paths"] == ["a.c", "b.h"]
    assert sorted(evidence["nested"]["kinds"]) == ["add", "multiply"]
    assert evidence["none"] is None


@pytest.mark.unit
def test_evidence_that_cannot_be_converted_is_stringified_rather_than_dropped() -> None:
    class Opaque:
        def __repr__(self):
            return "<Opaque target>"

    trail = AuditTrail("run-1")
    trail.ok("Codegen", "Target selected", evidence={"target": Opaque()})
    payload = trail.to_json()
    json.dumps(payload)
    assert payload["events"][0]["evidence"]["target"] == "<Opaque target>"


@pytest.mark.unit
def test_an_object_whose_to_json_raises_falls_back_to_its_string_form() -> None:
    class Hostile:
        def to_json(self):
            raise RuntimeError("no")

        def __repr__(self):
            return "<Hostile>"

    trail = AuditTrail("run-1")
    trail.ok("s", "e", evidence={"x": Hostile()})
    assert trail.to_json()["events"][0]["evidence"]["x"] == "<Hostile>"


@pytest.mark.unit
def test_the_trail_serialises_with_its_counts_and_schema() -> None:
    trail = AuditTrail("run-abc")
    trail.ok("Import", "Model imported", evidence={"ops": 42})
    trail.blocked("Mapping", "Operator has no lowering", detail="nn.conv2d")

    payload = trail.to_json()
    assert payload["schema"] == "tatva.audit_trail/1"
    assert payload["run_id"] == "run-abc"
    assert payload["event_count"] == 2
    assert payload["counts"][BLOCKED] == 1
    assert payload["started_at"]
    assert len(payload["events"]) == 2
    assert set(payload["events"][0]) == {
        "seq", "timestamp", "elapsed_s", "stage", "event", "outcome", "detail", "evidence",
    }


@pytest.mark.unit
def test_the_trail_is_written_beside_the_artifacts(tmp_path) -> None:
    build = tmp_path / "baseline"
    build.mkdir()
    trail = AuditTrail("run-w")
    trail.ok("Import", "Model imported", evidence={"ops": 7})

    path = trail.write(str(build))
    assert path == os.path.join(str(build), AUDIT_FILENAME)

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["run_id"] == "run-w"
    assert payload["events"][0]["evidence"] == {"ops": 7}


@pytest.mark.unit
def test_writing_into_a_missing_directory_reports_failure_rather_than_raising(tmp_path) -> None:
    trail = AuditTrail("run-w")
    assert trail.write(str(tmp_path / "nope")) == ""
    assert trail.write("") == ""


@pytest.mark.unit
def test_an_empty_trail_is_written_as_empty_not_as_a_success() -> None:
    trail = AuditTrail("run-empty")
    payload = trail.to_json()
    assert payload["event_count"] == 0
    assert payload["events"] == []
    assert payload["counts"] == {OK: 0, INFO: 0, WARN: 0, ERROR: 0, BLOCKED: 0}
