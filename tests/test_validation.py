"""
Tests for the validation stage.

The rule this module exists to enforce is a negative one: a check TATVA does not perform
is never reported as having passed. Most of what follows tests that rule from the
direction it would actually fail -- a run where everything went well, where it would be
easiest for an unimplemented check to slip into the green column.
"""

import os

import pytest

from tatva.validation import FAILED, NOT_IMPLEMENTED, PASSED, SKIPPED, evaluate


def _good_run(build_dir: str) -> dict:
    """A run where every stage succeeded."""
    return {
        "analysis": {"total_ops": 42, "distinct_ops": 9},
        "unsupported_ops": [],
        "target_name": "RV64GC",
        "build_dirs": {"baseline": build_dir},
        "generated_files": 7,
        "generated_lines": 900,
        "build_attempted": True,
        "measured": True,
        "environment": "QEMU_SIM",
        "base_ms": 12.5,
        "nominal_clock_mhz": 100,
        "accuracy_ok": True,
        "accuracy_reference": "host ONNX Runtime",
        "mse": 1e-9,
        "tolerance": 1e-4,
    }


@pytest.fixture
def linked_build(tmp_path):
    d = tmp_path / "baseline"
    d.mkdir()
    (d / "model.elf").write_bytes(b"\x7fELF" + b"\x00" * 128)
    return str(d)


def _by_key(report) -> dict:
    return {c.key: c for c in report.checks}


@pytest.mark.unit
def test_a_clean_run_passes_every_implemented_check(linked_build) -> None:
    report = evaluate(_good_run(linked_build))
    checks = _by_key(report)
    for key in ("graph_import", "operator_coverage", "codegen", "cross_compile",
                "target_execution", "numerical_parity"):
        assert checks[key].status == PASSED, f"{key}: {checks[key].detail}"
    assert report.verdict == PASSED
    assert report.failed == 0


@pytest.mark.unit
def test_a_clean_run_still_reports_what_was_not_checked(linked_build) -> None:
    """
    The failure this guards against: a page of green ticks that reads as "everything was
    verified". The roadmap items are present, never green, never counted as passes, and
    named in the summary.
    """
    report = evaluate(_good_run(linked_build))
    pending = [c for c in report.checks if c.status == NOT_IMPLEMENTED]

    assert pending, "a PASSED verdict with no statement of what was not checked"
    assert report.not_implemented == len(pending)
    assert all(c.evidence == "" for c in pending)
    assert "not implemented" in report.summary
    # They are excluded from the pass count, so the count cannot overstate coverage.
    assert report.passed == sum(1 for c in report.checks if c.status == PASSED)
    assert report.passed + report.failed + report.skipped + report.not_implemented == len(report.checks)


@pytest.mark.unit
def test_the_things_tatva_does_not_do_are_named_specifically(linked_build) -> None:
    """Silicon timing and power in particular: those are the two an evaluator is most
    likely to assume happened."""
    checks = _by_key(evaluate(_good_run(linked_build)))
    assert checks["silicon_timing"].status == NOT_IMPLEMENTED
    assert "physical" in checks["silicon_timing"].detail
    assert checks["power"].status == NOT_IMPLEMENTED
    assert checks["memory_footprint"].status == NOT_IMPLEMENTED
    assert checks["determinism"].status == NOT_IMPLEMENTED
    assert checks["quantization_sweep"].status == NOT_IMPLEMENTED


@pytest.mark.unit
def test_every_passing_check_carries_the_thing_that_was_observed(linked_build) -> None:
    """A pass with no evidence is an assertion, not a result."""
    report = evaluate(_good_run(linked_build))
    for check in report.checks:
        if check.status == PASSED:
            assert check.evidence, f"{check.key} passed with nothing behind it"


@pytest.mark.unit
def test_a_run_that_never_imported_the_graph_fails_and_skips_the_rest() -> None:
    report = evaluate({"analysis": {}, "import_error": "opset 21 is not supported"})
    checks = _by_key(report)
    assert checks["graph_import"].status == FAILED
    assert "opset 21" in checks["graph_import"].detail
    assert checks["operator_coverage"].status == SKIPPED
    assert checks["cross_compile"].status == SKIPPED
    assert report.verdict == FAILED


@pytest.mark.unit
def test_unsupported_operators_fail_coverage_and_name_themselves() -> None:
    report = evaluate({
        "analysis": {"total_ops": 12, "distinct_ops": 5},
        "unsupported_ops": ["nn.conv2d", "exp"],
        "target_name": "RV32IMC",
    })
    check = _by_key(report)["operator_coverage"]
    assert check.status == FAILED
    assert "nn.conv2d" in check.evidence and "exp" in check.evidence
    assert "RV32IMC" in check.detail


@pytest.mark.unit
def test_a_failed_measurement_is_not_dressed_up_as_a_skip(linked_build) -> None:
    """The build was attempted and produced nothing, which is a failure. Only a run that
    never got there is a skip."""
    run = _good_run(linked_build)
    run["measured"] = False
    run["build_error"] = "QEMU exited before the console printed cycles"
    checks = _by_key(evaluate(run))
    assert checks["target_execution"].status == FAILED
    assert "QEMU exited" in checks["target_execution"].detail
    assert checks["numerical_parity"].status == SKIPPED


@pytest.mark.unit
def test_diverging_output_fails_parity_with_the_numbers_attached(linked_build) -> None:
    run = _good_run(linked_build)
    run["accuracy_ok"] = False
    run["mse"] = 0.42
    check = _by_key(evaluate(run))["numerical_parity"]
    assert check.status == FAILED
    assert "0.42" in check.evidence
    assert evaluate(run).verdict == FAILED


@pytest.mark.unit
def test_parity_is_skipped_when_there_is_no_second_build_to_compare(linked_build) -> None:
    """With no optimization pass selected there is nothing to compare against, which is
    a different statement from "the outputs matched"."""
    run = _good_run(linked_build)
    run["parity_applicable"] = False
    check = _by_key(evaluate(run))["numerical_parity"]
    assert check.status == SKIPPED
    assert "no optimization pass" in check.detail.lower()
    assert evaluate(run).verdict == "PARTIAL"


@pytest.mark.unit
def test_the_repair_engine_s_checks_appear_only_when_it_ran(linked_build) -> None:
    plain = _by_key(evaluate(_good_run(linked_build)))
    assert "rewrite_structural" not in plain
    assert "rewrite_numeric" not in plain

    run = _good_run(linked_build)
    run["repair"] = {
        "attempted": True, "structural_validation": "passed", "numerical_validation": "passed",
        "max_abs_diff": 1.2e-7, "message": "3 rewrites applied",
    }
    repaired = _by_key(evaluate(run))
    assert repaired["rewrite_structural"].status == PASSED
    assert repaired["rewrite_numeric"].status == PASSED
    assert "1.2e-07" in repaired["rewrite_numeric"].evidence


@pytest.mark.unit
def test_a_discarded_rewrite_shows_as_a_failed_check_not_a_missing_one(linked_build) -> None:
    run = _good_run(linked_build)
    run["repair"] = {
        "attempted": True, "structural_validation": "passed", "numerical_validation": "failed",
        "max_abs_diff": 3.0,
    }
    checks = _by_key(evaluate(run))
    assert checks["rewrite_numeric"].status == FAILED
    assert "discarded" in checks["rewrite_numeric"].detail
    assert evaluate(run).verdict == FAILED


@pytest.mark.unit
def test_an_unrunnable_numerical_comparison_is_a_skip_that_says_what_still_applies(linked_build) -> None:
    run = _good_run(linked_build)
    run["repair"] = {"attempted": True, "structural_validation": "passed", "numerical_validation": "not_run"}
    check = _by_key(evaluate(run))["rewrite_numeric"]
    assert check.status == SKIPPED
    assert "end-to-end parity check below still applies" in check.detail


@pytest.mark.unit
def test_a_build_directory_with_no_elf_fails_the_link_check(tmp_path) -> None:
    d = tmp_path / "baseline"
    d.mkdir()
    (d / "model_run.c").write_text("x\n", encoding="utf-8")
    run = _good_run(str(d))
    run["build_error"] = "riscv-none-elf-gcc: undefined reference to expf"
    check = _by_key(evaluate(run))["cross_compile"]
    assert check.status == FAILED
    assert "undefined reference" in check.detail


@pytest.mark.unit
def test_the_link_check_reports_the_size_of_the_binary_it_found(linked_build) -> None:
    check = _by_key(evaluate(_good_run(linked_build)))["cross_compile"]
    size = os.path.getsize(os.path.join(linked_build, "model.elf"))
    assert f"{size:,} bytes" in check.evidence


@pytest.mark.unit
def test_a_run_where_nothing_completed_reports_a_failure_not_an_empty_pass() -> None:
    """
    An empty run is FAILED, because the import check itself failed. The point being
    tested is the negative: nothing passed, and the verdict is not PASSED by virtue of
    there being no failures to find.
    """
    report = evaluate({})
    assert report.verdict == FAILED
    assert report.passed == 0
    assert report.failed >= 1
    assert _by_key(report)["graph_import"].status == FAILED


@pytest.mark.unit
def test_a_failure_anywhere_outranks_the_passes(linked_build) -> None:
    """Verdict precedence: one failed check makes the run FAILED regardless of how many
    others passed."""
    run = _good_run(linked_build)
    run["accuracy_ok"] = False
    report = evaluate(run)
    assert report.verdict == FAILED
    assert report.passed > 0
    assert "failed" in report.summary


@pytest.mark.unit
def test_the_report_serialises_with_every_field_the_studio_reads(linked_build) -> None:
    payload = evaluate(_good_run(linked_build)).to_json()
    assert set(payload) == {"checks", "passed", "failed", "skipped", "not_implemented", "verdict", "summary"}
    for row in payload["checks"]:
        assert set(row) == {"key", "name", "status", "detail", "evidence", "category"}
        assert row["status"] in (PASSED, FAILED, SKIPPED, NOT_IMPLEMENTED)
