"""
Tests for tatva model importing and computation-graph analysis.
"""

import pytest

from tatva.compiler import (
    ImportInProgressError,
    UnsupportedOperatorError,
    analyze_graph,
    import_model,
)


@pytest.mark.integration
def test_import_onnx_success(skip_if_no_toolchain) -> None:
    """
    Assert that a valid ONNX model can be imported and analyzed.
    """
    model_ir = import_model("models/model.onnx")
    assert model_ir is not None
    assert model_ir.metadata["format"] == "ONNX"
    assert model_ir.mod is not None

    report = analyze_graph(model_ir)
    assert report.total_ops > 0
    # Op names are reported unprefixed, matching compiler.SUPPORTED_OPS.
    assert "nn.softmax" in report.op_histogram
    assert not any(name.startswith("relax.") for name in report.op_histogram), report.op_histogram
    assert report.has_transformer_bottleneck is True
    assert len(report.unsupported_ops) == 0


@pytest.mark.integration
def test_import_pytorch_in_progress(skip_if_no_toolchain) -> None:
    """
    Assert that PyTorch (.pt) files raise ImportInProgressError.
    """
    with pytest.raises(ImportInProgressError) as exc_info:
        import_model("models/model_traced.pt")
    assert "PyTorch import is in progress" in str(exc_info.value)


@pytest.mark.integration
def test_import_tensorflow_in_progress(skip_if_no_toolchain) -> None:
    """
    Assert that TensorFlow (.pb) files raise ImportInProgressError.
    """
    with pytest.raises(ImportInProgressError) as exc_info:
        import_model("models/model_tf.pb")
    assert "TensorFlow/Keras import is in progress" in str(exc_info.value)


@pytest.mark.integration
def test_import_unsupported_operator(skip_if_no_toolchain) -> None:
    """
    Assert that importing a model with unsupported operators raises UnsupportedOperatorError.
    """
    with pytest.raises(UnsupportedOperatorError) as exc_info:
        import_model("models/model_unsupported.onnx")
    assert "UnsupportedOpXYZ" in str(exc_info.value)
