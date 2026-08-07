"""
Tests that TATVA compiles models which are not BERT-shaped.

The benchmark harness used to hardcode three int64 tensors named input_ids /
attention_mask / token_type_ids. Every fixture in models/ had exactly those, so
the limitation was invisible: the harness looked general and was not. These tests
pin the generalization down.
"""

import numpy as np
import pytest

from tatva.compiler import TARGETS, import_model, resolve_input_shapes
from tatva.optimizer import check_softmax_fusable, select_fast_softmax_kernel
from tatva.runner import (
    ExecutionEnvironment,
    compile_model,
    default_input_array,
    default_inputs_for,
    reference_output,
    run_and_measure,
)


@pytest.mark.unit
def test_resolve_input_shapes_binds_symbolic_dims(mlp_model_path) -> None:
    """Every resolved dimension must be a positive concrete int -- codegen cannot emit symbols."""
    import onnx

    shapes = resolve_input_shapes(onnx.load(str(mlp_model_path)))
    assert shapes == {"input.1": (1, 16)}
    for dims in shapes.values():
        assert all(isinstance(d, int) and d > 0 for d in dims)


@pytest.mark.unit
def test_default_input_array_is_deterministic_and_typed() -> None:
    """The dummy data must be reproducible and must honour the tensor's declared dtype."""
    a = default_input_array("x", (2, 3), "float32")
    b = default_input_array("x", (2, 3), "float32")
    assert np.array_equal(a, b)
    assert a.dtype == np.float32
    assert a.shape == (2, 3)

    mask = default_input_array("attention_mask", (1, 4), "int64")
    assert mask.dtype == np.int64
    assert np.array_equal(mask, np.ones((1, 4), dtype=np.int64))

    seg = default_input_array("token_type_ids", (1, 4), "int64")
    assert np.array_equal(seg, np.zeros((1, 4), dtype=np.int64))


@pytest.mark.unit
def test_input_fill_kind_survives_onnx_name_rewriting() -> None:
    """
    TVM renames `attention.mask` to `attention_mask`. Host and target must still
    classify it identically, or they would fill the same tensor with different data.
    """
    from tatva.runner import input_fill_kind

    assert input_fill_kind("attention.mask", "int64") == input_fill_kind("attention_mask", "int64")
    assert input_fill_kind("input.1", "float32") == input_fill_kind("input_1", "float32")


@pytest.mark.unit
def test_check_softmax_fusable_rejects_graph_without_softmax(mlp_model_path) -> None:
    """The optimized softmax kernel has nothing to replace in a plain MLP."""
    pytest.importorskip("tvm")
    ir = import_model(str(mlp_model_path))
    fusable, reason = check_softmax_fusable(ir)
    assert fusable is False
    assert "softmax" in reason.lower()


@pytest.mark.unit
def test_fuse_records_skip_reason_instead_of_silently_passing(mlp_model_path) -> None:
    """
    A declined optimization must say why. Returning the model unchanged with no
    explanation is how a 'fused' build ends up identical to the baseline while
    still being reported as optimized.
    """
    pytest.importorskip("tvm")
    ir = import_model(str(mlp_model_path))
    fused = select_fast_softmax_kernel(ir)
    assert fused.metadata.get("softmax_optimized") is not True
    assert fused.metadata.get("softmax_fusion_skipped")


@pytest.mark.unit
def test_default_inputs_match_the_graph(mlp_model_path) -> None:
    """The host feed must cover exactly the model's declared inputs."""
    feed = default_inputs_for(str(mlp_model_path))
    assert set(feed) == {"input.1"}
    assert feed["input.1"].shape == (1, 16)
    assert feed["input.1"].dtype == np.float32


@pytest.mark.integration
def test_single_input_mlp_compiles_and_matches_host(mlp_model_path, skip_if_no_toolchain, tmp_path) -> None:
    """
    Full pipeline on a model with one float32 input and a C-hostile ONNX name.

    This is the test the old harness could not pass: it emitted a three-argument
    tvmgen_default_run() call for a one-input model and failed to compile.
    """
    ir = import_model(str(mlp_model_path))
    artifact = compile_model(ir, TARGETS["RV64GC"], str(tmp_path / "mlp"), warmup_count=2, timed_count=3)

    measurement = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)
    assert measurement.mean_ms > 0.0

    target_logits: list[float] = []
    for line in measurement.raw_output.splitlines():
        if "FIRST_LOGITS:" in line:
            target_logits = [float(x) for x in line.split(":")[1].strip().split()]
            break
    assert len(target_logits) == 5

    ref = reference_output(str(mlp_model_path))[:5]
    assert np.allclose(np.array(target_logits), np.array(ref), rtol=1e-4, atol=1e-4), (
        f"target={target_logits} host={list(ref)}"
    )


@pytest.mark.integration
def test_generated_harness_declares_only_the_models_own_inputs(mlp_model_path, skip_if_no_toolchain, tmp_path) -> None:
    """The emitted main.c must not reference tensors this model does not have."""
    ir = import_model(str(mlp_model_path))
    compile_model(ir, TARGETS["RV64GC"], str(tmp_path / "mlp_src"), warmup_count=1, timed_count=1)

    main_c = (tmp_path / "mlp_src" / "main.c").read_text()
    assert "@TATVA_" not in main_c, "harness template markers were left unsubstituted"
    assert "in_input_1" in main_c
    for absent in ("input_ids", "attention_mask", "token_type_ids"):
        assert absent not in main_c
