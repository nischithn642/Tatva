"""
Tests for tatva model importing and computation-graph analysis.
"""

import re

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tatva.compiler import (
    ImportInProgressError,
    UnsupportedOperatorError,
    analyze_graph,
    c_safe_name,
    import_model,
    rename_inputs_to_c_identifiers,
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


# --------------------------------------------------------------------------------------
# Input-name sanitising.
#
# ONNX puts no constraints on tensor names and real exporters use that freedom, so this
# is the first thing an arbitrary file off disk exercises. It is also invisible when it
# breaks: a name that is not a C identifier reaches gcc as a syntax error in generated
# code the user never wrote. The tests below are unit tests because none of this needs a
# cross-compiler -- the sanitiser is pure string handling over an in-memory graph.
# --------------------------------------------------------------------------------------

C_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _graph(inputs, nodes, outputs, initializer=()):
    """
    Build an in-memory ONNX model. Deliberately not run through onnx.checker: several
    cases here are shapes the checker rejects but a real file can still reach us in
    (an initializer also listed as a graph input is the old-style convention), and the
    sanitiser has to cope with the graph as given.
    """
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializer=list(initializer))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _tensor(name, shape=(1, 8)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


@pytest.mark.unit
@pytest.mark.parametrize(
    "original, expected",
    [
        # What PyTorch actually exports. This is the name that motivated the sanitiser.
        ("/layers.0/attn/MatMul_output_0", "_layers_0_attn_MatMul_output_0"),
        ("input.1", "input_1"),
        ("conv1-weight", "conv1_weight"),
        ("some name", "some_name"),
        ("scope::tensor", "scope__tensor"),
        # A leading digit is legal in ONNX and not in C.
        ("3d_input", "t_3d_input"),
        ("0", "t_0"),
        # Degenerate but reachable: an empty name must still come back an identifier.
        ("", "t_"),
        # Already valid, including names that happen to be C keywords. Those are left
        # alone on purpose: every identifier the harness emits is prefixed
        # (tensor_int, in_int, int_ptr), so a keyword never appears bare in the C.
        ("input_0", "input_0"),
        ("int", "int"),
        ("register", "register"),
    ],
)
def test_c_safe_name_produces_a_c_identifier(original: str, expected: str) -> None:
    """
    Assert that sanitising yields the expected identifier, and that it is one at all.
    """
    got = c_safe_name(original)
    assert got == expected
    assert C_IDENTIFIER.match(got), f"{original!r} sanitised to {got!r}, not a C identifier"
    # Sanitising an already-sanitised name must not move it again, or the host feed and
    # the on-target harness would normalise to different identifiers.
    assert c_safe_name(got) == got


@pytest.mark.unit
def test_the_runner_sanitises_through_the_same_function() -> None:
    """
    Assert runner.c_identifier delegates to compiler.c_safe_name rather than reimplementing it.

    Two sanitisers that disagreed would let the host reference and the on-target harness
    classify the same tensor differently, which is the bug input_fill_kind() normalises
    to prevent -- and it would only show up as a numerical mismatch on one odd name.
    """
    from tatva import runner

    for name in ["/a.b/c", "3x", "", "plain", "a-b c:d"]:
        assert runner.c_identifier(name) == c_safe_name(name)


@pytest.mark.unit
def test_renaming_is_a_no_op_when_every_input_is_already_valid() -> None:
    """
    Assert a graph needing no renames reports none, and is left untouched.
    """
    model = _graph([_tensor("x")], [helper.make_node("Relu", ["x"], ["y"])], [_tensor("y")])
    assert rename_inputs_to_c_identifiers(model) == {}
    assert [inp.name for inp in model.graph.input] == ["x"]


@pytest.mark.unit
def test_renaming_rewrites_the_input_and_every_reference_to_it() -> None:
    """
    Assert a renamed input is renamed everywhere it is mentioned, not just in graph.input.

    Rewriting the declaration alone would leave the node consuming a name nothing
    produces -- a graph that no longer type-checks, from a file that did.
    """
    model = _graph(
        [_tensor("in/put.0"), _tensor("other")],
        [helper.make_node("Add", ["in/put.0", "other"], ["y"])],
        [_tensor("y")],
    )
    mapping = rename_inputs_to_c_identifiers(model)

    assert mapping == {"in_put_0": "in/put.0"}
    assert [inp.name for inp in model.graph.input] == ["in_put_0", "other"]
    assert list(model.graph.node[0].input) == ["in_put_0", "other"]


@pytest.mark.unit
def test_renaming_follows_a_pass_through_to_the_output() -> None:
    """
    Assert that a graph whose output is its own input renames the output too.

    A zero-node graph is a real export (an identity branch, or a model reduced to one
    by constant folding). Leaving the output on the old name would emit a harness that
    copies from a tensor that no longer exists.
    """
    model = _graph([_tensor("x.0")], [], [_tensor("x.0")])
    mapping = rename_inputs_to_c_identifiers(model)

    assert mapping == {"x_0": "x.0"}
    assert [inp.name for inp in model.graph.input] == ["x_0"]
    assert [out.name for out in model.graph.output] == ["x_0"]


@pytest.mark.unit
def test_renaming_keeps_inputs_apart_that_sanitise_to_the_same_name() -> None:
    """
    Assert two inputs colliding on one identifier are disambiguated, not merged.

    'a.b' and 'a/b' both sanitise to 'a_b'. Silently merging them would wire one tensor
    to two inputs: the model would compile, run, and be wrong, with nothing to look at.
    """
    model = _graph(
        [_tensor("a.b"), _tensor("a/b")],
        [helper.make_node("Add", ["a.b", "a/b"], ["y"])],
        [_tensor("y")],
    )
    mapping = rename_inputs_to_c_identifiers(model)

    assert sorted(mapping.items()) == [("a_b", "a.b"), ("a_b_1", "a/b")]
    names = [inp.name for inp in model.graph.input]
    assert names == ["a_b", "a_b_1"]
    assert len(set(names)) == 2
    assert list(model.graph.node[0].input) == ["a_b", "a_b_1"]


@pytest.mark.unit
def test_renaming_does_not_collide_with_a_name_the_graph_already_uses() -> None:
    """
    Assert sanitising never lands on an identifier some other input already holds.
    """
    model = _graph(
        [_tensor("w.1"), _tensor("w_1")],
        [helper.make_node("Add", ["w.1", "w_1"], ["y"])],
        [_tensor("y")],
    )
    mapping = rename_inputs_to_c_identifiers(model)

    assert mapping == {"w_1_1": "w.1"}
    assert [inp.name for inp in model.graph.input] == ["w_1_1", "w_1"]


@pytest.mark.unit
def test_renaming_leaves_initializers_alone() -> None:
    """
    Assert an initializer listed as a graph input is not renamed.

    Old-style ONNX declares every initializer as an input as well. Those are folded to
    constants and never become parameters of the emitted C function, so renaming them
    buys nothing and would desynchronise graph.initializer from graph.input.
    """
    weight = helper.make_tensor("w.1", TensorProto.FLOAT, [1, 8], np.ones(8, dtype=np.float32))
    model = _graph(
        [_tensor("x.0"), _tensor("w.1")],
        [helper.make_node("Add", ["x.0", "w.1"], ["y"])],
        [_tensor("y")],
        initializer=[weight],
    )
    mapping = rename_inputs_to_c_identifiers(model)

    assert mapping == {"x_0": "x.0"}
    assert [inp.name for inp in model.graph.input] == ["x_0", "w.1"]
    assert [init.name for init in model.graph.initializer] == ["w.1"]
    assert list(model.graph.node[0].input) == ["x_0", "w.1"]


@pytest.mark.unit
def test_import_records_the_original_name_of_every_renamed_input(tmp_path) -> None:
    """
    Assert metadata['renamed_inputs'] maps each new identifier back to the file's name.

    The renamed identifier appears nowhere in the model the user handed us, so anything
    reporting an input back to them has to be able to recover the original.
    """
    model = _graph(
        [_tensor("/layers.0/input")],
        [helper.make_node("Relu", ["/layers.0/input"], ["y"])],
        [_tensor("y")],
    )
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / "renamed.onnx"
    onnx.save(model, str(path))

    model_ir = import_model(str(path))

    assert model_ir.metadata["renamed_inputs"] == {"_layers_0_input": "/layers.0/input"}
    # The shape table is keyed by the name the rest of the pipeline will use.
    assert "_layers_0_input" in model_ir.metadata["input_shapes"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "op, extra_inits, callee",
    [
        ("Pad", [("p", [0, 1, 0, 1])], "pad"),
        ("Tile", [("r", [2, 1])], "tile"),
    ],
)
def test_a_call_tir_operator_is_reported_by_its_callee_not_as_call_tir(
    tmp_path, op, extra_inits, callee
) -> None:
    """
    Assert an operator TVM lowers straight to TIR is neither called unsupported nor
    reported under the name 'call_tir'.

    Pad, Tile and Einsum arrive from the ONNX frontend as a `call_tir` into a generated
    PrimFunc, with no operator node in the graph at all. Diffing that name against
    SUPPORTED_OPS told the user their model used an unsupported operator called
    'call_tir' -- which names TVM's calling convention, not anything in their file, so
    there was nothing for them to look up, remove or replace. It is also always
    lowerable, since TVM wrote the callee itself.
    """
    model = _graph(
        [_tensor("x")],
        [helper.make_node(op, ["x", extra_inits[0][0]], ["y"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=[
            numpy_helper.from_array(np.array(vals, dtype=np.int64), name)
            for name, vals in extra_inits
        ],
    )
    model.ir_version = 9
    path = tmp_path / f"{callee}.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(model), str(path))

    report = analyze_graph(import_model(str(path)))

    assert report.unsupported_ops == []
    assert "call_tir" not in report.op_histogram
    # The callee's name is what the generated C kernel and the profile will call it.
    assert callee in report.op_histogram, report.op_histogram
