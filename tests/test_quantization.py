"""
Tests that INT8 quantization derives its scales from the model.

The pass was advertised as "dynamic quantization" while hardcoding 0.05 for every
activation and 0.02 for every weight, on every model. Those constants clip anything
outside +-6.35 and +-2.54 respectively, silently. These tests pin down that the
numbers now come from measured data and that the metadata says where from.
"""

import numpy as np
import pytest

from tatva.compiler import import_model
from tatva.optimizer import (
    CALIBRATION_PERCENTILE,
    FALLBACK_ACTIVATION_SCALE,
    INT8_QMAX,
    _weight_scale,
    calibrate_activation_scale,
    quantize,
)


@pytest.mark.unit
def test_weight_scale_spans_the_tensors_own_range() -> None:
    """The largest magnitude in the tensor must map to the top of the INT8 range."""
    arr = np.array([-3.0, 0.5, 2.0], dtype=np.float32)
    scale = _weight_scale(arr)
    assert scale == pytest.approx(3.0 / INT8_QMAX)
    # Round-tripping the peak through the scale must not clip.
    assert round(3.0 / scale) <= INT8_QMAX


@pytest.mark.unit
def test_weight_scale_ignores_non_finite_values() -> None:
    """One inf in a weight tensor must not collapse the scale for the whole tensor."""
    arr = np.array([1.0, np.inf, -np.nan, 2.0], dtype=np.float32)
    assert _weight_scale(arr) == pytest.approx(2.0 / INT8_QMAX)


@pytest.mark.unit
def test_weight_scale_of_all_zeros_is_not_degenerate() -> None:
    """An all-zero tensor has no range; the scale must still be usable, not 0."""
    scale = _weight_scale(np.zeros(8, dtype=np.float32))
    assert scale > 0.0


@pytest.mark.unit
def test_calibration_measures_the_model(mlp_model_path) -> None:
    """The activation scale must come from a real host run, not the fallback constant."""
    scale, source = calibrate_activation_scale(str(mlp_model_path))
    assert scale > 0.0
    assert scale != FALLBACK_ACTIVATION_SCALE
    assert "calibrated on host" in source
    assert f"p{CALIBRATION_PERCENTILE:g}" in source


@pytest.mark.unit
def test_calibration_reports_the_fallback_honestly(tmp_path) -> None:
    """
    When calibration cannot run, the caller must be able to tell. Returning 0.05 with
    no explanation is how a guessed scale gets reported as a measured one.
    """
    missing = tmp_path / "not_a_model.onnx"
    scale, source = calibrate_activation_scale(str(missing))
    assert scale == FALLBACK_ACTIVATION_SCALE
    assert source.startswith("fallback")


@pytest.mark.unit
def test_calibration_percentile_clips_outliers() -> None:
    """
    A lower percentile must yield a smaller scale on data with outliers -- that is the
    whole point of clipping rather than calibrating on the max.
    """
    pytest.importorskip("onnxruntime")
    path = "models/model_pretrained.onnx"
    tight, _ = calibrate_activation_scale(path, percentile=99.0)
    loose, _ = calibrate_activation_scale(path, percentile=100.0)
    assert tight < loose


@pytest.mark.unit
def test_quantize_records_where_its_scales_came_from(mlp_model_path) -> None:
    """Every scale the pass used must be inspectable afterwards."""
    pytest.importorskip("tvm")
    ir = quantize(import_model(str(mlp_model_path)))

    assert ir.metadata["quantized"] is True
    assert ir.metadata["activation_scale"] > 0.0
    assert "calibrated on host" in ir.metadata["activation_scale_source"]
    assert ir.metadata["quantized_ops"] == len(ir.metadata["weight_scales"])
    assert ir.metadata["quantized_ops"] > 0, "the MLP has matmuls; none were quantized"


@pytest.mark.unit
def test_quantize_gives_each_weight_tensor_its_own_scale(mlp_model_path) -> None:
    """
    The MLP's two weight matrices have different magnitudes. One shared constant
    would necessarily be wrong for at least one of them.
    """
    pytest.importorskip("tvm")
    scales = quantize(import_model(str(mlp_model_path))).metadata["weight_scales"]
    assert len(scales) >= 2
    assert len(set(scales)) > 1, f"all weights got the same scale: {scales}"


@pytest.mark.unit
def test_explicit_activation_scale_overrides_calibration(mlp_model_path) -> None:
    """Callers benchmarking a specific scale must get exactly that scale."""
    pytest.importorskip("tvm")
    ir = quantize(import_model(str(mlp_model_path)), activation_scale=0.125)
    assert ir.metadata["activation_scale"] == 0.125
    assert ir.metadata["activation_scale_source"] == "caller-supplied"


@pytest.mark.unit
def test_quantize_without_a_source_path_says_so(mlp_model_path) -> None:
    """
    An IR that has lost its source model cannot be calibrated. That must show up in
    the metadata rather than being passed off as a measurement.
    """
    pytest.importorskip("tvm")
    ir = import_model(str(mlp_model_path))
    ir.metadata.pop("source_path", None)
    quantized = quantize(ir)
    assert quantized.metadata["activation_scale"] == FALLBACK_ACTIVATION_SCALE
    assert quantized.metadata["activation_scale_source"].startswith("fallback")
