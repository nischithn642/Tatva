"""
Tests for the generated-artifact manifest.

The Artifacts page's claim is that it reports what is on disk, not what a build is
expected to produce. These tests hold it to that: an unexpected file must still be
listed, a missing one must simply be absent, and the totals shown in the studio must be
the totals in the persisted manifest.
"""

import json
import os

import pytest

from tatva.artifacts import (
    MANIFEST_FILENAME,
    STAGE_CODEGEN,
    STAGE_HARNESS,
    STAGE_LINK,
    STAGE_TVM_LOWERING,
    build_manifest,
    describe,
    discover,
    generated_source_stats,
    read_manifest,
    write_manifest,
)


@pytest.fixture
def build_dir(tmp_path):
    """A directory shaped like a real TATVA build."""
    d = tmp_path / "build" / "baseline"
    d.mkdir(parents=True)
    (d / "weights.h").write_text("static const float W[4] = {1,2,3,4};\n", encoding="utf-8")
    (d / "model_run.c").write_text("void run(void) {}\n" * 10, encoding="utf-8")
    (d / "model_info.h").write_text("#define IN 4\n", encoding="utf-8")
    (d / "main.c").write_text("int main(void) { return 0; }\n" * 3, encoding="utf-8")
    (d / "start.S").write_text(".global _start\n_start:\n", encoding="utf-8")
    (d / "link.ld").write_text("SECTIONS { . = 0x80000000; }\n", encoding="utf-8")
    (d / "model.elf").write_bytes(b"\x7fELF" + b"\x00" * 512)
    return str(d)


@pytest.mark.unit
def test_discover_reports_what_is_on_disk(build_dir) -> None:
    names = [a.name for a in discover(build_dir)]
    assert set(names) == {
        "weights.h", "model_run.c", "model_info.h", "main.c", "start.S", "link.ld", "model.elf",
    }


@pytest.mark.unit
def test_discovery_is_ordered_by_pipeline_stage_not_alphabetically(build_dir) -> None:
    """The list reads as the build happened. Alphabetical order would put the ELF first,
    which is the last thing produced."""
    stages = [a.stage for a in discover(build_dir)]
    assert stages[0] == STAGE_TVM_LOWERING
    assert stages[-1] == STAGE_LINK
    assert stages.index(STAGE_CODEGEN) < stages.index(STAGE_HARNESS) < stages.index(STAGE_LINK)


@pytest.mark.unit
def test_a_file_nobody_wrote_a_rule_for_is_still_listed(build_dir) -> None:
    """
    An inventory that silently drops what it does not recognise is not an inventory.
    A stray file gets a generic description and its real size and hash.
    """
    stray = os.path.join(build_dir, "something_unexpected.xyz")
    with open(stray, "w", encoding="utf-8") as fh:
        fh.write("hello")

    found = {a.name: a for a in discover(build_dir)}
    assert "something_unexpected.xyz" in found
    assert found["something_unexpected.xyz"].size_bytes == 5
    assert found["something_unexpected.xyz"].sha256


@pytest.mark.unit
def test_a_file_that_was_never_written_does_not_appear(tmp_path) -> None:
    """No placeholder rows for expected-but-absent files."""
    d = tmp_path / "half"
    d.mkdir()
    (d / "model_run.c").write_text("x\n", encoding="utf-8")
    names = [a.name for a in discover(str(d))]
    assert names == ["model_run.c"]
    assert "model.elf" not in names


@pytest.mark.unit
def test_missing_directory_yields_nothing_rather_than_raising(tmp_path) -> None:
    assert discover(str(tmp_path / "never-created")) == []
    assert discover("") == []


@pytest.mark.unit
def test_text_files_carry_a_line_count_and_binaries_do_not(build_dir) -> None:
    """The effort model counts lines of generated source. An ELF has none, and
    reporting a number for it would put a made-up quantity into that estimate."""
    found = {a.name: a for a in discover(build_dir)}
    assert found["model_run.c"].line_count == 10
    assert found["main.c"].line_count == 3
    assert found["model.elf"].line_count is None


@pytest.mark.unit
def test_hash_and_size_describe_the_actual_bytes(build_dir) -> None:
    import hashlib

    path = os.path.join(build_dir, "model.elf")
    with open(path, "rb") as fh:
        raw = fh.read()
    art = describe(path)
    assert art is not None
    assert art.size_bytes == len(raw)
    assert art.sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.unit
def test_describe_returns_none_for_a_directory_or_a_missing_path(tmp_path) -> None:
    assert describe(str(tmp_path)) is None
    assert describe(str(tmp_path / "nope.c")) is None


@pytest.mark.unit
def test_manifest_totals_match_the_files_it_lists(build_dir) -> None:
    """
    The Artifacts page shows a file count and a byte total. Both are read from here, so
    a manifest whose totals disagree with its own rows would put two different numbers
    on the same screen.
    """
    m = build_manifest(
        run_id="run-test-001", model_path="model.onnx", target_name="RV64GC",
        march="rv64gc", mabi="lp64d", configs={"baseline": build_dir},
    )
    section = m["builds"][0]
    rows = section["artifacts"]

    assert section["file_count"] == len(rows) == m["totals"]["file_count"]
    counted = sum(r["size_bytes"] for r in rows if r["size_bytes"] is not None)
    assert section["total_bytes"] == counted == m["totals"]["total_bytes"]


@pytest.mark.unit
def test_manifest_lists_itself_with_no_size(build_dir) -> None:
    """
    A reader comparing the manifest against the directory finds one extra file --
    the manifest. It is listed so that is not a mystery, with a null size because at the
    moment the inventory is taken it does not exist yet.
    """
    m = build_manifest(
        run_id="r", model_path="m.onnx", target_name="RV64GC", march="rv64gc", mabi="lp64d",
        configs={"baseline": build_dir},
    )
    self_row = next(r for r in m["builds"][0]["artifacts"] if r["name"] == MANIFEST_FILENAME)
    assert self_row["size_bytes"] is None
    assert self_row["sha256"] == ""
    assert "not listed here" in self_row["description"]


@pytest.mark.unit
def test_a_stale_manifest_from_a_previous_run_is_replaced_not_counted_twice(build_dir) -> None:
    """
    Build directories are reused when the model, passes and target are unchanged. The old
    manifest sitting there must not be inventoried at its old size alongside the new
    self-entry, or the file count gains a phantom.
    """
    with open(os.path.join(build_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as fh:
        json.dump({"stale": True}, fh)

    m = build_manifest(
        run_id="r2", model_path="m.onnx", target_name="RV64GC", march="rv64gc", mabi="lp64d",
        configs={"baseline": build_dir},
    )
    rows = [r for r in m["builds"][0]["artifacts"] if r["name"] == MANIFEST_FILENAME]
    assert len(rows) == 1
    assert rows[0]["size_bytes"] is None


@pytest.mark.unit
def test_files_written_after_the_manifest_are_named_rather_than_left_unexplained(build_dir) -> None:
    m = build_manifest(
        run_id="r", model_path="m.onnx", target_name="RV64GC", march="rv64gc", mabi="lp64d",
        configs={"baseline": build_dir},
        written_after=["engineering_effort.json", "audit_trail.json"],
    )
    assert m["written_after_this_manifest"] == ["engineering_effort.json", "audit_trail.json"]


@pytest.mark.unit
def test_two_build_configurations_are_kept_apart(build_dir, tmp_path) -> None:
    """Baseline and optimized are genuinely different generated C. Merging them would
    hide what the optimization pass changed."""
    other = tmp_path / "build" / "optimized"
    other.mkdir(parents=True)
    (other / "model_run.c").write_text("optimized\n", encoding="utf-8")

    m = build_manifest(
        run_id="r", model_path="m.onnx", target_name="RV64GC", march="rv64gc", mabi="lp64d",
        configs={"baseline": build_dir, "optimized": str(other)},
    )
    assert [s["config"] for s in m["builds"]] == ["baseline", "optimized"]
    assert m["totals"]["file_count"] == sum(s["file_count"] for s in m["builds"])


@pytest.mark.unit
def test_a_configuration_whose_directory_is_absent_is_reported_as_absent(tmp_path) -> None:
    m = build_manifest(
        run_id="r", model_path="m.onnx", target_name="RV64GC", march="rv64gc", mabi="lp64d",
        configs={"optimized": str(tmp_path / "gone")},
    )
    section = m["builds"][0]
    assert section["exists"] is False
    assert section["file_count"] == 0
    assert section["artifacts"] == []


@pytest.mark.unit
def test_manifest_round_trips_through_disk(build_dir) -> None:
    m = build_manifest(
        run_id="run-x", model_path="m.onnx", target_name="RV64GC", march="rv64gc", mabi="lp64d",
        configs={"baseline": build_dir}, passes=["quantize"], repaired_ops=["nn.silu"],
    )
    path = write_manifest(m, build_dir)
    assert os.path.isfile(path)

    back = read_manifest(build_dir)
    assert back is not None
    assert back["run_id"] == "run-x"
    assert back["passes"] == ["quantize"]
    assert back["repaired_ops"] == ["nn.silu"]
    assert back["schema"] == "tatva.artifact_manifest/1"


@pytest.mark.unit
def test_reading_a_manifest_that_is_not_there_or_is_corrupt_returns_none(tmp_path) -> None:
    assert read_manifest(str(tmp_path)) is None
    with open(tmp_path / MANIFEST_FILENAME, "w", encoding="utf-8") as fh:
        fh.write("{ not json")
    assert read_manifest(str(tmp_path)) is None


@pytest.mark.unit
def test_write_manifest_into_a_missing_directory_reports_failure_by_returning_empty(tmp_path) -> None:
    assert write_manifest({"a": 1}, str(tmp_path / "nope")) == ""


@pytest.mark.unit
def test_source_stats_count_only_real_files(build_dir) -> None:
    """These four numbers feed the effort estimate directly."""
    stats = generated_source_stats({"baseline": build_dir})
    assert stats["generated_files"] == 7
    assert stats["generated_lines"] == 1 + 10 + 1 + 3 + 2 + 1
    assert stats["generated_header_lines"] == 2          # weights.h + model_info.h
    assert stats["elf_bytes"] == 4 + 512


@pytest.mark.unit
def test_source_stats_exclude_the_manifest_itself(build_dir) -> None:
    """The manifest is not generated source and must not inflate the count the effort
    model charges hours against."""
    before = generated_source_stats({"baseline": build_dir})
    write_manifest(
        build_manifest(run_id="r", model_path="", target_name="", march="", mabi="",
                       configs={"baseline": build_dir}),
        build_dir,
    )
    after = generated_source_stats({"baseline": build_dir})
    assert after == before


@pytest.mark.unit
def test_source_stats_are_zero_when_nothing_was_built(tmp_path) -> None:
    stats = generated_source_stats({"baseline": str(tmp_path / "never")})
    assert stats == {
        "generated_files": 0, "generated_lines": 0, "generated_header_lines": 0, "elf_bytes": 0,
    }
