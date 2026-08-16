"""
Tests for the model-to-target fit check (`tatva.fit`).

The three verdicts are covered from opposite directions: BLOCKED and the 32-bit
soft-float penalty are driven from synthetic `ModelInfo` values, because no bundled
model is a gigabyte and none is FP64; FITS and the "best fit" mark are driven from a
real file on disk, so the arithmetic that reads a model is exercised too.
"""

import pytest

from tatva.compiler import TARGETS
from tatva.fit import (
    VERDICT_BLOCKED,
    VERDICT_FITS,
    VERDICT_SLOW,
    fit_for,
    fit_targets,
    image_floor_bytes,
)
from tatva.frontends import ModelInfo
from tatva.runner import MAX_LOADABLE_IMAGE_BYTES


def _info(**kw) -> ModelInfo:
    """A minimal inspected model. Only the fields `fit` reads need to be real."""
    base = dict(
        ok=True,
        name="synthetic.onnx",
        precision="FP32",
        parameter_bytes=4 * 1024 * 1024,
        inputs=[{"name": "x", "shape": [1, 3, 224, 224], "dtype": "FP32"}],
        outputs=[{"name": "y", "shape": [1, 1000], "dtype": "FP32"}],
    )
    base.update(kw)
    return ModelInfo(**base)


@pytest.mark.unit
def test_image_floor_counts_weights_io_and_the_fixed_pools() -> None:
    """
    The floor has to be above the weights, or a BLOCKED verdict could be wrong.

    It is a floor and not an estimate: activations are unknown until TVM legalizes the
    graph, so the only guarantee this module makes is that the real region is at least
    this large.
    """
    info = _info()
    floor = image_floor_bytes(info)
    assert floor > info.parameter_bytes
    # Rounded up to the runner's granule, so it is never a bare sum.
    assert floor % (16 * 1024 * 1024) == 0


@pytest.mark.unit
def test_a_model_past_the_emulator_ceiling_is_blocked_on_every_target() -> None:
    """
    The load ceiling belongs to the bundled QEMU board, not to any one chip, so no
    target escapes it -- including the 64-bit ones with the most address space.
    """
    huge = _info(parameter_bytes=MAX_LOADABLE_IMAGE_BYTES + (64 * 1024 * 1024))
    for variant in TARGETS.values():
        f = fit_for(huge, variant)
        assert f.verdict == VERDICT_BLOCKED, variant.name
        assert "emulator" in " ".join(f.reasons).lower()


@pytest.mark.unit
def test_float_weights_on_a_target_without_an_fpu_are_slow_not_blocked() -> None:
    """
    A user targeting an RV32IMC part still needs that build. The verdict says what it
    costs; it does not withhold the target.
    """
    info = _info(precision="FP32")
    slow = fit_for(info, TARGETS["RV32IMC"])
    assert slow.verdict == VERDICT_SLOW
    assert "soft-float" in " ".join(slow.reasons)

    fast = fit_for(info, TARGETS["RV64GC"])
    assert fast.verdict == VERDICT_FITS


@pytest.mark.unit
def test_an_integer_only_model_needs_no_fpu_anywhere() -> None:
    """A fully quantized model has nothing for an FPU to do, so no target is slow for it."""
    info = _info(
        precision="INT8",
        inputs=[{"name": "x", "shape": [1, 3, 224, 224], "dtype": "INT8"}],
        outputs=[{"name": "y", "shape": [1, 1000], "dtype": "INT8"}],
    )
    assert fit_for(info, TARGETS["RV32IMC"]).verdict == VERDICT_FITS
    assert fit_for(info, TARGETS["RV64GC"]).verdict == VERDICT_FITS


@pytest.mark.unit
def test_mixed_precision_still_counts_as_needing_an_fpu() -> None:
    """
    `inspect_model` reports mixed models as "Mixed (FP32/INT8)". The FP32 half still
    goes through the FPU, so parsing that string wrong would silently mark a slow
    target as a clean fit.
    """
    info = _info(precision="Mixed (FP32/INT8)")
    assert fit_for(info, TARGETS["RV32IMC"]).verdict == VERDICT_SLOW


@pytest.mark.unit
def test_experimental_targets_are_never_the_best_fit() -> None:
    """
    RV64GCV fits every model RV64GC fits -- it is the same scalar code -- so nothing but
    the experimental flag stops it being recommended. That flag has to be what decides.
    """
    r = fit_targets("models/model.onnx")
    assert r["success"] is True, r.get("error")
    assert r["best"] == "RV64GC"

    by_name = {f["target"]: f for f in r["targets"]}
    assert by_name["RV64GCV"]["best"] is False
    assert by_name["RV64GCV"]["experimental"] is True
    assert sum(1 for f in r["targets"] if f["best"]) == 1


@pytest.mark.unit
def test_fit_targets_reports_every_registered_target() -> None:
    """Stage 01 draws one card per target; a target missing here is a card with no verdict."""
    r = fit_targets("models/model.onnx")
    assert {f["target"] for f in r["targets"]} == set(TARGETS.keys())
    assert r["image_limit_bytes"] == MAX_LOADABLE_IMAGE_BYTES
    assert r["image_floor_bytes"] <= MAX_LOADABLE_IMAGE_BYTES


@pytest.mark.unit
def test_a_missing_model_fails_without_raising() -> None:
    """The bridge calls this on every model load; a bad path is a message, not a traceback."""
    r = fit_targets("models/does_not_exist.onnx")
    assert r["success"] is False
    assert r["targets"] == []
    assert "not found" in r["error"]
