"""
Tests for the run registry and its status derivation.

`derive_status` is the one place a run is declared SUCCESS. Every page in the studio and
`artifact_manifest.json` read the answer from here, so a bug that lets a partial build
read as a clean one would be reported identically in four places and look corroborated.
These tests walk each branch, and in particular the ones where something went wrong.
"""

import pytest

from tatva.audit import AuditTrail
from tatva.runs import (
    BLOCKED,
    FAILED,
    MAX_RUNS,
    PARTIAL,
    RUNNING,
    SUCCESS,
    RunRecord,
    _Registry,
    derive_status,
)


def _built(**kw) -> RunRecord:
    """A run that compiled, linked, ran and validated cleanly."""
    record = RunRecord(run_id="run-1", target_name="RV64GC")
    record.build_dirs = {"baseline": "/tmp/baseline"}
    record.benchmark = {"measured": True}
    record.validation = {"verdict": "PASSED", "summary": "All 6 implemented checks passed.", "checks": []}
    for key, value in kw.items():
        setattr(record, key, value)
    return record


@pytest.mark.unit
def test_a_clean_run_is_success_and_carries_validation_s_own_words() -> None:
    status, reason = derive_status(_built())
    assert status == SUCCESS
    assert reason == "All 6 implemented checks passed."


@pytest.mark.unit
def test_an_error_outranks_everything_else() -> None:
    record = _built(error="riscv-none-elf-gcc not found on PATH")
    status, reason = derive_status(record)
    assert status == FAILED
    assert reason == "riscv-none-elf-gcc not found on PATH"


@pytest.mark.unit
def test_an_unsupported_operator_blocks_rather_than_fails() -> None:
    """BLOCKED means TATVA can name the reason it stopped. That is a different thing
    from a crash, and the report says which."""
    record = RunRecord(run_id="r", target_name="RV32IMC")
    record.mapping = {"unsupported": ["nn.conv2d", "exp"]}
    status, reason = derive_status(record)

    assert status == BLOCKED
    assert "nn.conv2d" in reason and "exp" in reason
    assert "RV32IMC" in reason
    assert "Code generation did not start" in reason


@pytest.mark.unit
def test_after_a_repair_the_blocked_list_is_what_is_still_unmapped() -> None:
    """
    The regression this guards: naming the pre-repair mapping list here told the user an
    operator blocked their build when TATVA had just fixed it.
    """
    record = RunRecord(run_id="r", target_name="RV64GC")
    record.mapping = {"unsupported": ["nn.silu", "nn.conv2d"]}
    record.repair = {"attempted": True, "applied": True, "remaining_unsupported": ["nn.conv2d"]}

    status, reason = derive_status(record)
    assert status == BLOCKED
    assert "nn.conv2d" in reason
    assert "nn.silu" not in reason
    assert "1 operator kind(s)" in reason


@pytest.mark.unit
def test_a_repair_that_cleared_everything_does_not_report_a_stale_blockage() -> None:
    """An empty `remaining_unsupported` is a real answer -- "nothing is left" -- and must
    not fall through to the pre-repair list."""
    record = RunRecord(run_id="r", target_name="RV64GC")
    record.mapping = {"unsupported": ["nn.silu"]}
    record.repair = {"attempted": True, "applied": True, "remaining_unsupported": []}

    status, reason = derive_status(record)
    assert status == FAILED
    assert "nn.silu" not in reason
    assert "no reason was recorded" in reason


@pytest.mark.unit
def test_nothing_built_and_nothing_to_blame_is_a_failure_that_admits_it() -> None:
    status, reason = derive_status(RunRecord(run_id="r"))
    assert status == FAILED
    assert reason == "No build was produced and no reason was recorded."


@pytest.mark.unit
def test_a_build_with_no_measurement_is_not_a_success() -> None:
    record = _built()
    record.benchmark = {"measured": False}
    status, reason = derive_status(record)
    assert status == FAILED
    assert "no measurement completed" in reason


@pytest.mark.unit
def test_a_failed_validation_check_downgrades_a_measured_run_to_partial() -> None:
    """It compiled, linked and ran -- that is real output, so not FAILED -- but a check
    did not hold, so not SUCCESS either. The failing checks are named."""
    record = _built()
    record.validation = {
        "verdict": "FAILED",
        "checks": [
            {"name": "Target output matches the host reference", "status": "FAILED"},
            {"name": "Model imported into TATVA's graph IR", "status": "PASSED"},
        ],
    }
    status, reason = derive_status(record)
    assert status == PARTIAL
    assert "1 validation check(s) failed" in reason
    assert "Target output matches the host reference" in reason


@pytest.mark.unit
def test_a_validation_failure_with_no_named_check_still_reports_partial() -> None:
    record = _built()
    record.validation = {"verdict": "FAILED", "checks": []}
    status, reason = derive_status(record)
    assert status == PARTIAL
    assert "validation reported a failure" in reason


@pytest.mark.unit
def test_operators_that_were_never_mapped_downgrade_an_otherwise_clean_run() -> None:
    """
    A model can compile with an unmapped operator when that operator never blocked
    codegen. The build is real, so this is PARTIAL -- claiming a clean run would hide
    that part of the graph was never lowered.
    """
    record = _built()
    record.repair = {"attempted": True, "applied": True, "remaining_unsupported": ["nn.attention"]}
    status, reason = derive_status(record)
    assert status == PARTIAL
    assert "nn.attention" in reason
    assert "never mapped" in reason


@pytest.mark.unit
def test_unmapped_operators_outrank_a_passing_validation_verdict() -> None:
    """Validation can pass while an operator is still unmapped; the status must not
    round that up to SUCCESS."""
    record = _built()
    record.validation = {"verdict": "PASSED", "summary": "All checks passed.", "checks": []}
    record.repair = {"remaining_unsupported": ["exp"]}
    assert derive_status(record)[0] == PARTIAL


@pytest.mark.unit
def test_skipped_validation_checks_make_the_run_partial_not_successful() -> None:
    record = _built()
    record.validation = {"verdict": "PARTIAL", "summary": "5 passed, 1 could not run.", "checks": []}
    status, reason = derive_status(record)
    assert status == PARTIAL
    assert reason == "5 passed, 1 could not run."


@pytest.mark.unit
def test_a_new_run_starts_as_running_with_no_result_fields_filled_in() -> None:
    reg = _Registry()
    record = reg.new_run(model_name="mlp.onnx", target_name="RV64GC")
    assert record.status == RUNNING
    assert record.run_id.startswith("run-")
    assert record.started_at
    assert record.finished_at == ""
    # An empty stage field means "did not run", never a zero standing in for a result.
    assert record.analysis == {} and record.benchmark == {} and record.effort == {}
    assert isinstance(record.trail, AuditTrail)
    assert record.trail.run_id == record.run_id


@pytest.mark.unit
def test_finishing_a_run_stamps_it_and_derives_the_status_once() -> None:
    reg = _Registry()
    record = reg.new_run(target_name="RV64GC")
    record.build_dirs = {"baseline": "/tmp/b"}
    record.benchmark = {"measured": True}
    record.validation = {"verdict": "PASSED", "summary": "ok", "checks": []}

    reg.finish(record)
    assert record.finished_at
    assert record.status == SUCCESS
    assert record.status_reason == "ok"


@pytest.mark.unit
def test_runs_are_addressable_by_id_and_the_latest_is_the_default() -> None:
    reg = _Registry()
    first = reg.new_run(model_name="a.onnx")
    second = reg.new_run(model_name="b.onnx")

    assert reg.get(first.run_id) is first
    assert reg.get("") is second
    assert reg.get() is second
    assert reg.get("run-that-never-existed") is None


@pytest.mark.unit
def test_run_ids_are_unique_even_within_the_same_second() -> None:
    reg = _Registry()
    ids = {reg.new_run().run_id for _ in range(20)}
    assert len(ids) == 20


@pytest.mark.unit
def test_the_registry_is_bounded_and_drops_the_oldest_first() -> None:
    reg = _Registry()
    made = [reg.new_run(model_name=f"m{i}.onnx") for i in range(MAX_RUNS + 5)]

    assert reg.get(made[0].run_id) is None
    assert reg.get(made[-1].run_id) is made[-1]
    assert len(reg.recent(limit=1000)) == MAX_RUNS


@pytest.mark.unit
def test_recent_lists_newest_first() -> None:
    reg = _Registry()
    reg.new_run(model_name="old.onnx")
    newest = reg.new_run(model_name="new.onnx")
    rows = reg.recent(limit=5)
    assert rows[0]["run_id"] == newest.run_id
    assert rows[0]["model"] == "new.onnx"


@pytest.mark.unit
def test_the_summary_carries_the_status_and_its_reason_together() -> None:
    """Anywhere the status appears, the reason appears with it -- a bare "PARTIAL" with
    no explanation is what sends someone hunting through logs."""
    record = _built()
    record.status, record.status_reason = derive_status(record)
    summary = record.summary()
    assert summary["status"] == SUCCESS
    assert summary["status_reason"]
    assert set(summary) >= {"run_id", "model", "target", "status", "status_reason", "error"}


@pytest.mark.unit
def test_to_json_exposes_every_stage_and_the_audit_trail() -> None:
    reg = _Registry()
    record = reg.new_run(target_name="RV64GC")
    record.trail.ok("Import", "Model imported", evidence={"ops": 42})

    payload = record.to_json()
    for key in ("model_info", "analysis", "mapping", "repair", "benchmark", "manifest",
                "validation", "effort", "optimization_history", "build_dirs", "audit"):
        assert key in payload
    assert payload["audit"]["event_count"] == 1
    assert payload["audit"]["events"][0]["evidence"] == {"ops": 42}
