"""
Tests for the model frontend registry and family detection.

The format table is the place TATVA is most tempted to overclaim: listing PyTorch and
TensorFlow as supported costs nothing until someone tries. These tests hold the registry
to the rule that SUPPORTED is earned -- an adapter TATVA implements, a Relax frontend
that exists in the installed TVM, and the framework package importable -- and that
everything else states the specific thing that is missing.

Family detection is a classifier, and it is allowed to fail. What it is not allowed to do
is pick the nearest label when the signals do not add up.
"""

import pytest

from tatva.frontends import (
    COMING_SOON,
    FAMILY_UNKNOWN,
    FORMATS,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    ModelFormat,
    detect_family,
    format_for_path,
    format_table,
    inspect_model,
)

# --------------------------------------------------------------------- format registry

@pytest.mark.unit
def test_only_onnx_claims_an_implemented_adapter() -> None:
    """
    `implemented` is the honest flag: True only for the path TATVA actually compiles
    models through today. Setting it for anything else makes every downstream status
    a lie, which is what this table exists to prevent.
    """
    implemented = {f.key for f in FORMATS if f.implemented}
    assert implemented == {"onnx"}


@pytest.mark.unit
def test_onnx_is_supported_in_an_environment_that_can_run_the_suite() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("tvm")
    onnx_fmt = next(f for f in FORMATS if f.key == "onnx")
    status, reason = onnx_fmt.status()
    assert status == SUPPORTED
    assert reason == ""


@pytest.mark.unit
def test_no_unimplemented_format_is_ever_reported_as_supported() -> None:
    for fmt in FORMATS:
        if not fmt.implemented:
            status, reason = fmt.status()
            assert status != SUPPORTED, f"{fmt.key} claims support with no adapter"
            assert reason, f"{fmt.key} is unavailable with no stated reason"


@pytest.mark.unit
def test_an_unavailable_format_names_the_concrete_thing_that_is_missing() -> None:
    """"Coming Soon" on its own tells an engineer nothing. Each reason has to say what
    is absent and what to do instead."""
    for fmt in FORMATS:
        status, reason = fmt.status()
        if status == COMING_SOON:
            assert "Export to ONNX" in reason
            assert fmt.tvm_frontend in reason or "No Relax frontend exists" in reason


@pytest.mark.unit
def test_a_format_with_no_relax_frontend_says_conversion_is_required() -> None:
    keras = next(f for f in FORMATS if f.key == "keras")
    status, reason = keras.status()
    assert status == COMING_SOON
    assert "No Relax frontend exists" in reason


@pytest.mark.unit
def test_an_implemented_format_missing_its_package_is_partial_not_supported() -> None:
    """The distinction matters: PARTIAL means TATVA's side is done and the environment
    is not, which is a fixable problem the message should name."""
    fmt = ModelFormat(
        key="onnx_probe", label="ONNX", framework="ONNX", extensions=[".probe"],
        implemented=True, tvm_frontend="onnx", requires=["a_package_that_is_not_installed"],
    )
    status, reason = fmt.status()
    assert status == PARTIAL
    assert "a_package_that_is_not_installed" in reason
    assert "not installed" in reason


@pytest.mark.unit
def test_an_implemented_format_whose_tvm_frontend_is_absent_is_partial() -> None:
    fmt = ModelFormat(
        key="probe", label="Probe", framework="Probe", extensions=[".probe"],
        implemented=True, tvm_frontend="a_frontend_tvm_does_not_have", requires=[],
    )
    status, reason = fmt.status()
    assert status == PARTIAL
    assert "a_frontend_tvm_does_not_have" in reason


@pytest.mark.unit
def test_the_table_the_ui_renders_carries_a_status_and_reason_for_every_row() -> None:
    """The studio holds no format list of its own; it renders whatever this returns."""
    rows = format_table()
    assert len(rows) == len(FORMATS)
    for row in rows:
        assert row["status"] in (SUPPORTED, PARTIAL, COMING_SOON, UNSUPPORTED)
        assert row["extensions_label"]
        assert row["notes"]
        if row["status"] != SUPPORTED:
            assert row["reason"]


@pytest.mark.unit
@pytest.mark.parametrize("path,key", [
    ("model.onnx", "onnx"), ("MODEL.ONNX", "onnx"), ("m.tflite", "tflite"),
    ("m.pt", "torchscript"), ("m.pt2", "torch_exported"), ("m.h5", "keras"),
])
def test_extensions_resolve_to_the_right_format(path, key) -> None:
    fmt = format_for_path(path)
    assert fmt is not None and fmt.key == key


@pytest.mark.unit
@pytest.mark.parametrize("path", ["model.bin", "model", "", "archive.zip"])
def test_an_unrecognised_extension_resolves_to_nothing(path) -> None:
    assert format_for_path(path) is None


# ------------------------------------------------------------------- family detection

@pytest.mark.unit
def test_a_graph_that_matches_nothing_says_so_instead_of_guessing() -> None:
    verdict = detect_family({"Identity": 3, "Cast": 1})
    assert verdict.family == FAMILY_UNKNOWN
    assert verdict.confidence == "none"
    assert "Identity" in verdict.signals[0]
    assert "descriptive only" in verdict.detail


@pytest.mark.unit
def test_an_empty_graph_is_undetermined_rather_than_an_mlp() -> None:
    assert detect_family({}).family == FAMILY_UNKNOWN


@pytest.mark.unit
def test_classical_ml_is_recognised_from_its_own_operator_domain() -> None:
    verdict = detect_family({"TreeEnsembleClassifier": 1, "Scaler": 1})
    assert verdict.family == "Classical ML"
    assert verdict.confidence == "high"


@pytest.mark.unit
@pytest.mark.parametrize("op", ["LSTM", "GRU", "RNN"])
def test_recurrent_models_are_read_from_the_fused_onnx_node(op) -> None:
    """ONNX keeps these fused. By the time the graph reaches Relax they are decomposed
    into arithmetic and the evidence is gone -- which is why detection works on the
    source graph."""
    verdict = detect_family({op: 2, "MatMul": 4})
    assert verdict.family == op
    assert verdict.confidence == "high"


@pytest.mark.unit
def test_a_fused_attention_node_is_conclusive() -> None:
    verdict = detect_family({"Attention": 6, "MatMul": 12, "Softmax": 6})
    assert verdict.family == "Transformer"
    assert verdict.confidence == "high"
    assert "fused attention node" in verdict.signals


@pytest.mark.unit
def test_decomposed_layernorm_still_reads_as_a_transformer() -> None:
    """
    The regression this guards: requiring a fused LayerNormalization node made the
    classifier call a two-layer BERT an MLP. Plenty of real exports ship it decomposed.
    """
    ops = {"ReduceMean": 8, "Sqrt": 4, "Pow": 4, "Mul": 20, "Softmax": 2, "MatMul": 16, "Add": 24}
    verdict = detect_family(ops)
    assert verdict.family == "Transformer"
    assert any("decomposed form" in s for s in verdict.signals)


@pytest.mark.unit
def test_a_fused_layernorm_is_reported_as_what_it_is() -> None:
    ops = {"LayerNormalization": 4, "Softmax": 2, "MatMul": 16}
    verdict = detect_family(ops)
    assert verdict.family == "Transformer"
    assert any("layer/RMS normalization" in s for s in verdict.signals)


@pytest.mark.unit
def test_attention_behind_a_small_conv_stem_is_a_vision_transformer() -> None:
    ops = {"Conv": 1, "LayerNormalization": 12, "Softmax": 6, "MatMul": 24}
    verdict = detect_family(ops)
    assert verdict.family == "Vision Transformer (ViT)"
    assert any("patch embedding" in s for s in verdict.signals)


@pytest.mark.unit
def test_a_deep_conv_stack_ending_in_attention_is_not_called_a_vit() -> None:
    ops = {"Conv": 30, "LayerNormalization": 2, "Softmax": 1, "MatMul": 6, "MaxPool": 4}
    assert detect_family(ops).family == "Transformer"


@pytest.mark.unit
def test_an_embedding_table_plus_an_lm_head_is_a_language_model() -> None:
    ops = {"Gather": 1, "LayerNormalization": 12, "Softmax": 6, "MatMul": 30}
    verdict = detect_family(ops, param_count=120_000_000, vocab_hint=32_000, output_dims=[32_000])
    assert verdict.family == "Small Language Model (SLM)"
    assert any("projects back to the" in s for s in verdict.signals)
    assert "parameter count alone" in verdict.detail


@pytest.mark.unit
def test_the_same_graph_without_the_lm_head_is_an_encoder_not_an_slm() -> None:
    """
    An embedding table alone does not make a language model -- a text encoder with a
    two-class head has one too. Calling that an SLM would be a guess dressed as a
    classification.
    """
    ops = {"Gather": 1, "LayerNormalization": 12, "Softmax": 6, "MatMul": 30}
    verdict = detect_family(ops, param_count=110_000_000, vocab_hint=30_522, output_dims=[2])
    assert verdict.family == "Transformer (text encoder)"
    assert "encoder with a task head" in verdict.detail


@pytest.mark.unit
def test_a_very_large_parameter_count_is_reported_as_an_llm() -> None:
    ops = {"Gather": 1, "LayerNormalization": 80, "Softmax": 40, "MatMul": 200}
    verdict = detect_family(ops, param_count=7_000_000_000, vocab_hint=32_000, output_dims=[32_000])
    assert verdict.family == "Large Language Model (LLM)"


@pytest.mark.unit
def test_convolutions_dominate_a_cnn() -> None:
    verdict = detect_family({"Conv": 20, "Relu": 20, "MaxPool": 5, "BatchNormalization": 20})
    assert verdict.family == "CNN"
    assert verdict.confidence == "high"
    assert any("pooling" in s for s in verdict.signals)


@pytest.mark.unit
def test_dense_layers_with_nothing_structural_are_an_mlp() -> None:
    verdict = detect_family({"Gemm": 3, "Relu": 2})
    assert verdict.family == "MLP"
    assert verdict.confidence == "medium"


@pytest.mark.unit
def test_an_mlp_with_no_activations_is_reported_with_low_confidence() -> None:
    """The classifier grades its own certainty rather than presenting every answer with
    the same weight."""
    assert detect_family({"MatMul": 2}).confidence == "low"


@pytest.mark.unit
def test_every_verdict_carries_the_signals_it_was_based_on() -> None:
    """A wrong answer should be arguable rather than mysterious."""
    for ops in ({"Conv": 4}, {"LSTM": 1}, {"Gemm": 2, "Relu": 1}, {"Identity": 1}, {}):
        verdict = detect_family(ops)
        assert verdict.signals, ops
        assert verdict.detail, ops
        assert verdict.to_json()["family"] == verdict.family


# ---------------------------------------------------------------------- inspect_model

@pytest.mark.unit
def test_inspecting_a_file_that_is_not_there_fails_without_inventing_fields(tmp_path) -> None:
    info = inspect_model(str(tmp_path / "absent.onnx"))
    assert info.ok is False
    assert info.error == "File not found."
    assert info.parameter_count == 0
    assert info.op_types == {}


@pytest.mark.unit
def test_an_unrecognised_extension_is_reported_as_unsupported(tmp_path) -> None:
    path = tmp_path / "model.weights"
    path.write_bytes(b"\x00" * 16)
    info = inspect_model(str(path))
    assert info.ok is False
    assert info.format_status == UNSUPPORTED
    assert ".weights" in info.format_reason


@pytest.mark.unit
def test_a_format_tatva_cannot_read_reports_the_file_and_the_reason_only(tmp_path) -> None:
    """Nothing dishonest to add: the graph fields stay empty rather than guessed."""
    path = tmp_path / "model.tflite"
    path.write_bytes(b"\x00" * 64)
    info = inspect_model(str(path))

    assert info.ok is False
    assert info.framework == "TensorFlow"
    assert info.file_size_bytes == 64
    assert info.error
    assert info.op_count == 0
    assert info.inputs == [] and info.outputs == []
    assert info.family == {}


@pytest.mark.unit
def test_a_file_that_is_not_really_onnx_says_so(tmp_path) -> None:
    pytest.importorskip("onnx")
    path = tmp_path / "not_really.onnx"
    path.write_bytes(b"this is not a protobuf")
    info = inspect_model(str(path))
    assert info.ok is False
    assert "could not be parsed as ONNX" in info.error


@pytest.mark.unit
def test_a_real_model_is_described_from_what_the_file_contains(baseline_model_path) -> None:
    """Every field is read out of the protobuf; none is defaulted or estimated."""
    pytest.importorskip("onnx")
    info = inspect_model(str(baseline_model_path))

    assert info.ok is True
    assert info.framework == "ONNX"
    assert info.format_status == SUPPORTED
    assert info.file_size_bytes > 0
    assert info.op_count > 0
    assert info.distinct_op_count == len(info.op_types)
    assert sum(info.op_types.values()) == info.op_count
    assert info.parameter_count > 0
    assert info.parameter_bytes >= info.parameter_count      # FP32 weights are 4 bytes each
    assert info.precision
    assert info.inputs and info.outputs
    assert all(i["shape_label"] for i in info.inputs)
    assert info.family["family"]
    assert info.opset


@pytest.mark.unit
def test_initializers_are_not_listed_as_graph_inputs(baseline_model_path) -> None:
    """Older exporters list every weight as a graph input. Showing those on the Input
    page would tell the user the model takes 40 tensors."""
    pytest.importorskip("onnx")
    import onnx

    info = inspect_model(str(baseline_model_path))
    initializers = {i.name for i in onnx.load(str(baseline_model_path)).graph.initializer}
    assert not ({i["name"] for i in info.inputs} & initializers)
