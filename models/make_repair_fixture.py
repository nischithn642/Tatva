"""
Build the two fixtures the graph-repair engine is tested against.

  model_repairable.onnx  every unsupported operator in it has an exact rewrite rule,
                         so repair should end with a fully mapped graph.
  model_blocked.onnx     contains Exp, which has no exact decomposition into TATVA's
                         operator set, alongside operators that do. Repair should
                         rewrite what it can and then stop honestly on the rest.

Both are tiny and weightless on purpose: they exist to exercise the rewrite path and
the host-side numerical check, not to measure anything.

Regenerate with:  python models/make_repair_fixture.py
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIM, OUT_DIM = 8, 4


def _weights(seed: int):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((IN_DIM, OUT_DIM)).astype(np.float32) * 0.3
    b = rng.standard_normal((OUT_DIM,)).astype(np.float32) * 0.1
    return numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")


def _save(graph, path: str, opset: int = 17) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def build_repairable() -> None:
    """
    MatMul -> Add -> LeakyRelu -> Abs -> Neg, all of which rewrite exactly.

    The Add against a 1-D bias is what makes the ONNX frontend emit `broadcast_to`,
    the operator the product spec names as the canonical UNMAPPED case -- so this
    fixture covers it without being contrived.

    Deliberately no ONNX `Min` node: the frontend lowers a two-input Min to `stack`
    followed by a `min` *reduction*, and a reduction has no exact decomposition into
    TATVA's operator set. It would have made the fixture look repairable while
    testing a path that cannot be repaired.
    """
    w, b = _weights(11)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, IN_DIM])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, OUT_DIM])
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "B"], ["bias"]),
        helper.make_node("LeakyRelu", ["bias"], ["act"], alpha=0.125),
        helper.make_node("Abs", ["act"], ["mag"]),
        helper.make_node("Neg", ["mag"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "tatva_repairable", [x], [y], initializer=[w, b])
    _save(graph, os.path.join(HERE, "model_repairable.onnx"))


def build_blocked() -> None:
    """Same spine, but Exp sits in the middle and has no exact decomposition."""
    w, b = _weights(29)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, IN_DIM])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, OUT_DIM])
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "B"], ["bias"]),
        helper.make_node("Neg", ["bias"], ["neg"]),      # rewritable
        helper.make_node("Exp", ["neg"], ["y"]),         # not rewritable
    ]
    graph = helper.make_graph(nodes, "tatva_blocked", [x], [y], initializer=[w, b])
    _save(graph, os.path.join(HERE, "model_blocked.onnx"))


if __name__ == "__main__":
    build_repairable()
    build_blocked()
