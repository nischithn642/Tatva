"""
Tests for the memory budget: how big the linked RAM region is, how much memory QEMU
is given, and how long it is allowed to run.

These three numbers were fixed constants until 2.1 -- a 128 MiB region, a 128 MiB
board, a 30 s timeout -- and that is what stopped TATVA compiling anything much over
100 MB. They are computed now, and the computation had no test of any kind: nothing
in the suite called these functions, so a change that quietly restored the fixed
region would leave every test green while the feature this release is named for
stopped working.

The arithmetic is deliberately asserted against hand-computed byte counts rather than
against the constants, so a change to a constant has to be a deliberate edit here too.
"""

import pytest

from tatva.runner import (
    FIRMWARE_RESERVED_BYTES,
    MIN_RAM_BYTES,
    QEMU_MIN_TIMEOUT_SECONDS,
    QEMU_SECONDS_PER_MIB_PER_RUN,
    RAM_GRANULE_BYTES,
    RAM_SLACK_BYTES,
    STACK_BYTES,
    WORKSPACE_POOL_BYTES,
    _memory_budget_note,
    _qemu_memory_mib,
    _qemu_timeout_seconds,
    _ram_region_bytes,
    render_link_ld,
)

MIB = 1024 * 1024


@pytest.mark.unit
def test_small_models_keep_the_region_they_always_had() -> None:
    """
    Everything that fitted in the old fixed 128 MiB region must still be linked into
    exactly 128 MiB. The floor is what makes this change safe for existing models:
    same memory map, same addresses, same behaviour.
    """
    assert _ram_region_bytes(0) == 128 * MIB
    assert _ram_region_bytes(1024) == 128 * MIB
    # Right up to the point where demand plus slack still fits under the floor.
    assert _ram_region_bytes(128 * MIB - RAM_SLACK_BYTES) == 128 * MIB


@pytest.mark.unit
def test_the_region_grows_past_the_floor_once_the_model_needs_it() -> None:
    """A 116 MB model was measured linking into 144 MiB; a 1 GB one into 1040 MiB."""
    # 116.0 MiB of weights and pools -> 144 MiB, the figure the real build produced.
    assert _ram_region_bytes(int(116.0 * MIB)) == 144 * MIB == 150_994_944

    # 1008.1 MiB -> 1040 MiB, likewise measured.
    assert _ram_region_bytes(int(1008.1 * MIB)) == 1040 * MIB == 1_090_519_040


@pytest.mark.unit
def test_the_region_is_always_slack_plus_demand_rounded_to_a_granule() -> None:
    """
    The two properties the linker actually depends on: never less than what was
    accounted for plus the slack, and always a whole number of granules.
    """
    for accounted in (0, 1, 200 * MIB, 200 * MIB + 1, 999 * MIB, 3 * 1024 * MIB):
        region = _ram_region_bytes(accounted)
        assert region % RAM_GRANULE_BYTES == 0, f"{accounted} produced a ragged region"
        assert region >= MIN_RAM_BYTES
        assert region >= accounted + RAM_SLACK_BYTES or region == MIN_RAM_BYTES
        assert region >= accounted, "a region smaller than its own contents cannot link"


@pytest.mark.unit
def test_the_region_is_monotonic_in_demand() -> None:
    """A bigger model can never be handed a smaller region."""
    demands = [0, 64 * MIB, 128 * MIB, 129 * MIB, 512 * MIB, 1024 * MIB, 2048 * MIB]
    regions = [_ram_region_bytes(d) for d in demands]
    assert regions == sorted(regions)


@pytest.mark.unit
def test_qemu_is_given_the_region_plus_the_firmware_gap() -> None:
    """
    The image is linked at 0x80200000 because OpenSBI occupies the first 2 MiB of RAM.
    `-m` has to cover both or the image loads into memory the board does not have --
    a build that passed the linker then dies at boot, which reads as a QEMU failure.
    """
    assert _qemu_memory_mib(128 * MIB) == 130
    assert _qemu_memory_mib(144 * MIB) == 146
    # The 1 GB model measured `-m 1042M`.
    assert _qemu_memory_mib(1040 * MIB) == 1042

    for ram in (0, 1, 128 * MIB, 1040 * MIB):
        mib = _qemu_memory_mib(ram)
        assert mib * MIB >= max(MIN_RAM_BYTES, ram) + FIRMWARE_RESERVED_BYTES


@pytest.mark.unit
def test_qemu_memory_never_drops_below_the_board_default() -> None:
    """
    128 MiB is virt's default. Passing anything smaller would be a regression for every
    model that already ran, so the floor applies before the firmware gap is added.
    """
    assert _qemu_memory_mib(0) == 130
    assert _qemu_memory_mib(1) == 130


@pytest.mark.unit
def test_the_timeout_scales_with_image_size_and_inference_count() -> None:
    """
    The fixed 30 s ceiling killed a 6-layer BERT that needed 100.9 s and reported it as
    a QEMU failure rather than as the timeout it was.
    """
    assert _qemu_timeout_seconds(128 * MIB, 1) == int(128 * 1 * QEMU_SECONDS_PER_MIB_PER_RUN)
    # The 1 GB run measured a 1248 s allowance and finished in 232.7 s.
    assert _qemu_timeout_seconds(1040 * MIB, 2) == 1248

    doubled = _qemu_timeout_seconds(512 * MIB, 2)
    single = _qemu_timeout_seconds(512 * MIB, 1)
    assert doubled == 2 * single, "two inferences must be allowed twice the wall clock"


@pytest.mark.unit
def test_the_timeout_has_a_floor_and_treats_zero_runs_as_one() -> None:
    assert _qemu_timeout_seconds(0, 0) >= QEMU_MIN_TIMEOUT_SECONDS
    assert _qemu_timeout_seconds(0, 0) == _qemu_timeout_seconds(0, 1)
    assert _qemu_timeout_seconds(1, 1) >= QEMU_MIN_TIMEOUT_SECONDS


@pytest.mark.unit
def test_the_linker_script_carries_the_computed_length_and_no_markers() -> None:
    """
    LINK_LD is a template. Writing it out without substitution puts "@TATVA_RAM_LENGTH@"
    into the script and fails the link with a parse error, so the absence of every
    marker matters as much as the presence of the number.
    """
    region = _ram_region_bytes(600 * MIB)
    script = render_link_ld(region)

    assert f"LENGTH = {region}" in script
    assert "@TATVA_RAM_LENGTH@" not in script
    assert "@TATVA_STACK_BYTES@" not in script
    assert "@" not in script, "an unsubstituted marker would reach the linker"
    assert str(STACK_BYTES) in script
    assert "ORIGIN = 0x80200000" in script


@pytest.mark.unit
def test_the_linker_script_defaults_to_the_floor() -> None:
    assert f"LENGTH = {MIN_RAM_BYTES}" in render_link_ld()


@pytest.mark.unit
def test_the_budget_note_accounts_for_every_pool_it_names() -> None:
    """
    This string is what a user reads after a build fails for memory. Every figure in it
    is a pool that was actually emitted, so the note must name all of them and the
    region they were linked into -- a missing term reads as unexplained memory.
    """
    weights, activations, io = 700 * MIB, 40 * MIB, 2 * MIB
    accounted = weights + activations + WORKSPACE_POOL_BYTES + io + STACK_BYTES
    region = _ram_region_bytes(accounted)

    note = _memory_budget_note(weights, activations, io, region, "RV64GC")

    assert "RV64GC" in note
    for label in ("weights", "activations", "workspace", "harness IO", "stack"):
        assert label in note, f"the budget note does not account for {label}"
    assert f"{region / MIB:.1f} MiB" in note
    assert f"{weights / MIB:.1f} MiB" in note
