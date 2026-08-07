"""
Tests for tatva attention/softmax fusion optimization pass.
"""

import pytest
from unittest.mock import MagicMock, patch

from tatva.compiler import TARGETS, import_model
from tatva.optimizer import compare_configs, select_fast_softmax_kernel


@pytest.mark.integration
def test_fusion_applies_only_when_transformer_bottleneck_present(skip_if_no_toolchain) -> None:
    """
    Assert that the fusion pass applies the softmax_optimized flag only when
    a transformer bottleneck (matmul + softmax) is detected in the graph.
    """
    model_path = "models/model.onnx"
    model_ir = import_model(model_path)

    # 1. Bottleneck present (default for this model)
    fused_ir = select_fast_softmax_kernel(model_ir)
    assert fused_ir is not None
    assert fused_ir.metadata.get("softmax_optimized") is True

    # 2. No bottleneck present (mocked)
    with patch("tatva.compiler.analyze_graph") as mock_analyze:
        mock_report = MagicMock()
        mock_report.has_transformer_bottleneck = False
        mock_analyze.return_value = mock_report

        non_fused_ir = select_fast_softmax_kernel(model_ir)
        assert non_fused_ir.metadata.get("softmax_optimized") is not True


@pytest.mark.integration
def test_compare_configs_fused_and_metrics(skip_if_no_toolchain) -> None:
    """
    Assert that compare_configs correctly profiles the fused configuration,
    verifies output parity vs reference, and confirms latency improvements.
    """
    model_path = "models/model.onnx"
    variant = TARGETS["RV64GC"]

    res = compare_configs(model_path, variant, ["baseline", "fused"])

    assert "baseline" in res["results"]
    assert "fused" in res["results"]

    comp = res["comparison"]
    assert comp["fused_mean_ms"] > 0.0
    assert comp["fused_median_ms"] > 0.0
    assert comp["fused_p95_ms"] > 0.0

    # Latency delta must be negative, verifying performance improvement
    assert comp["fused_mean_delta_ms"] < 0.0
    assert comp["fused_median_delta_ms"] < 0.0
    assert comp["fused_p95_delta_ms"] < 0.0

    # Numerical correctness: MSE must be well within standard tolerance
    assert comp["fused_accuracy_ok"] is True
    assert comp["fused_accuracy_delta_mse"] < 1e-4
