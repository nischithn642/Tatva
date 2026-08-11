"""
Tests for the engineering-effort estimate.

The number this module produces is the one most likely to be quoted out of context, so
the tests are about the properties that keep it defensible: it refuses to produce a
figure it cannot count, it never labels itself a measurement, every line traces back to a
counted quantity, and the arithmetic exists in exactly one place.
"""

import json

import pytest

from tatva.effort import (
    DEFAULT_RATES,
    EFFORT_MODEL_ENV,
    EFFORT_MODEL_VERSION,
    HOURS_PER_DAY,
    EffortInputs,
    compute,
    load_rates,
    write_effort,
)


def _real_run() -> EffortInputs:
    return EffortInputs(
        distinct_operator_kinds=9,
        total_operator_calls=42,
        generated_files=7,
        generated_lines=900,
        repaired_op_kinds=2,
        build_configs=2,
        benchmark_runs=2,
    )


@pytest.mark.unit
def test_a_run_that_produced_nothing_gets_no_estimate_at_all() -> None:
    """
    Refusal is a first-class outcome. A run that never reached code generation has no
    work to count, and filling that gap with a plausible number is the exact failure
    this module exists to avoid.
    """
    result = compute(EffortInputs())
    assert result.available is False
    assert result.total_hours == 0.0
    assert result.total_days == 0.0
    assert result.breakdown == []
    assert "never analysed" in result.reason
    assert "no source was generated" in result.reason
    assert "no build configuration was linked" in result.reason


@pytest.mark.unit
@pytest.mark.parametrize("field", ["distinct_operator_kinds", "generated_lines", "build_configs"])
def test_any_single_missing_count_is_enough_to_withhold_the_figure(field) -> None:
    inputs = _real_run()
    setattr(inputs, field, 0)
    result = compute(inputs)
    assert result.available is False
    assert result.total_hours == 0.0


@pytest.mark.unit
def test_a_refusal_still_publishes_the_model_so_the_absence_is_explicable() -> None:
    result = compute(EffortInputs())
    assert result.rates, "refusing to estimate is not a reason to hide the method"
    assert result.disclaimer
    assert result.model_version == EFFORT_MODEL_VERSION


@pytest.mark.unit
def test_the_estimate_never_calls_itself_a_measurement() -> None:
    result = compute(_real_run())
    assert result.available is True
    assert result.kind == "estimate"
    assert result.measured is False
    assert "not a measurement" in result.disclaimer
    assert "No engineer was timed" in result.disclaimer


@pytest.mark.unit
def test_the_total_is_the_sum_of_its_own_breakdown() -> None:
    """
    There is one implementation of this arithmetic and the UI renders it. A total that
    does not reconcile with the lines beneath it would be a number nobody could check.
    """
    result = compute(_real_run())
    assert result.total_hours == pytest.approx(sum(row["hours"] for row in result.breakdown), abs=0.02)
    assert result.total_days == pytest.approx(result.total_hours / HOURS_PER_DAY, abs=0.01)
    assert result.hours_per_day == HOURS_PER_DAY


@pytest.mark.unit
def test_every_line_multiplies_a_counted_quantity_by_a_declared_rate() -> None:
    result = compute(_real_run())
    for row in result.breakdown:
        assert row["hours"] == pytest.approx(row["quantity"] * row["hours_per_unit"], abs=0.02)
        assert row["quantity_source"], f"{row['key']} charges hours against an unexplained quantity"
        assert row["basis"], f"{row['key']} has a rate with no stated basis"


@pytest.mark.unit
def test_every_rate_is_reachable_from_a_counted_quantity() -> None:
    """A rate with nothing mapped to it would silently contribute zero hours and read as
    though it had been accounted for."""
    result = compute(_real_run())
    assert {row["key"] for row in result.breakdown} == {r.key for r in DEFAULT_RATES}
    unmapped = [r for r in result.breakdown if "no counted quantity" in r["quantity_source"]]
    assert unmapped == []


@pytest.mark.unit
def test_the_counted_quantities_are_the_ones_that_were_passed_in() -> None:
    inputs = _real_run()
    rows = {r["key"]: r for r in compute(inputs).breakdown}
    assert rows["operator_kinds"]["quantity"] == 9
    assert rows["operator_calls"]["quantity"] == 42
    assert rows["generated_lines"]["quantity"] == 9.0          # 900 lines, charged per 100
    assert rows["repaired_op_kinds"]["quantity"] == 2
    assert rows["target_bringup"]["quantity"] == 2
    assert rows["benchmark_runs"]["quantity"] == 2


@pytest.mark.unit
def test_the_inputs_are_republished_alongside_the_result() -> None:
    """The reader is meant to be able to accept the counts and reject the rates. That
    requires seeing the counts."""
    result = compute(_real_run())
    assert result.inputs == _real_run().to_json()


@pytest.mark.unit
def test_a_bigger_run_estimates_more_hours() -> None:
    small = compute(_real_run())
    big = _real_run()
    big.generated_lines *= 3
    big.distinct_operator_kinds *= 2
    assert compute(big).total_hours > small.total_hours


@pytest.mark.unit
def test_rates_can_be_replaced_and_the_source_is_reported(tmp_path, monkeypatch) -> None:
    """
    The point of the override is that nobody has to take the defaults on trust. When one
    is used, the result says so -- two figures from two rate tables must not look alike.
    """
    override = tmp_path / "rates.json"
    override.write_text(json.dumps({"rates": [
        {"key": "operator_kinds", "label": "Kernels", "unit": "op", "hours_per_unit": 1.0,
         "basis": "Our own historical data."},
    ]}), encoding="utf-8")
    monkeypatch.setenv(EFFORT_MODEL_ENV, str(override))

    rates, source = load_rates()
    assert len(rates) == 1
    assert str(override) in source

    result = compute(_real_run())
    assert result.total_hours == 9.0                      # 9 operator kinds x 1.0 h
    assert str(override) in result.model_source


@pytest.mark.unit
@pytest.mark.parametrize("content", ["{ not json", '{"rates": []}', '{"rates": [{"key": "x"}]}'])
def test_an_unusable_override_falls_back_and_says_why(tmp_path, monkeypatch, content) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(content, encoding="utf-8")
    monkeypatch.setenv(EFFORT_MODEL_ENV, str(bad))

    rates, source = load_rates()
    assert [r.key for r in rates] == [r.key for r in DEFAULT_RATES]
    assert "built-in defaults" in source
    assert str(bad) in source


@pytest.mark.unit
def test_an_override_path_that_does_not_exist_says_so(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(EFFORT_MODEL_ENV, str(tmp_path / "missing.json"))
    _, source = load_rates()
    assert "missing file" in source


@pytest.mark.unit
def test_no_override_configured_uses_the_defaults(monkeypatch) -> None:
    monkeypatch.delenv(EFFORT_MODEL_ENV, raising=False)
    rates, source = load_rates()
    assert rates == DEFAULT_RATES
    assert source == "built-in defaults"


@pytest.mark.unit
def test_every_default_rate_declares_that_it_is_an_assumption() -> None:
    """These are engineering judgements, not measurements, and each one says so in the
    panel the UI shows."""
    for rate in DEFAULT_RATES:
        assert rate.basis.startswith("Assumption"), rate.key
        assert rate.hours_per_unit > 0
        assert rate.unit and rate.label


@pytest.mark.unit
def test_the_estimate_is_persisted_next_to_the_artifacts_it_describes(tmp_path) -> None:
    build = tmp_path / "baseline"
    build.mkdir()
    result = compute(_real_run())

    path = write_effort(result, str(build), run_id="run-x")
    assert path

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["schema"] == "tatva.engineering_effort/1"
    assert payload["run_id"] == "run-x"
    # The persisted figure and the one the UI renders are the same object, so they
    # cannot disagree.
    assert payload["total_hours"] == result.total_hours
    assert payload["breakdown"] == result.breakdown
    assert payload["measured"] is False


@pytest.mark.unit
def test_writing_into_a_missing_directory_reports_failure_rather_than_raising(tmp_path) -> None:
    assert write_effort(compute(_real_run()), str(tmp_path / "nope")) == ""
    assert write_effort(compute(_real_run()), "") == ""
