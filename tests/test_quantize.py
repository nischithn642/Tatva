"""
Tests for tatva model quantization and config comparison benchmarking.
"""

from unittest.mock import patch

import pytest

from tatva.compiler import TARGETS, import_model
from tatva.optimizer import compare_configs, quantize


@pytest.mark.integration
def test_quantize_returns_valid_module(skip_if_no_toolchain) -> None:
    """
    Assert that quantize produces a valid mutated IRModule with metadata flags.
    """
    model_path = "models/model.onnx"
    model_ir = import_model(model_path)

    quant_ir = quantize(model_ir)
    assert quant_ir is not None
    assert quant_ir.mod is not None
    assert quant_ir.metadata.get("quantized") is True


@pytest.mark.integration
def test_compare_configs_single_import_and_metrics(skip_if_no_toolchain) -> None:
    """
    Assert that compare_configs runs both baseline and quantized benchmarks
    from a single import, and returns accurate comparison latency and accuracy deltas.
    """
    model_path = "models/model.onnx"
    variant = TARGETS["RV64GC"]

    # Wrap import_model to assert that it is called exactly once
    with patch("tatva.compiler.import_model", wraps=import_model) as mock_import:
        res = compare_configs(model_path, variant, ["baseline", "quantized"])
        mock_import.assert_called_once()

    assert "baseline" in res["results"]
    assert "quantized" in res["results"]

    comp = res["comparison"]
    assert comp["baseline_mean_ms"] > 0.0
    assert comp["quantized_mean_ms"] > 0.0
    assert "latency_delta_ms" in comp
    assert "accuracy_delta_mse" in comp
    assert isinstance(comp["accuracy_ok"], bool)
