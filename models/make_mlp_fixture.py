"""
Generate models/model_mlp.onnx -- a deliberately non-BERT fixture.

Every other fixture in this directory is a transformer with the same three int64
inputs named input_ids / attention_mask / token_type_ids. That uniformity hid a
real bug: the benchmark harness hardcoded those three tensors, so it happened to
work on all five fixtures while being incapable of compiling anything else.

This model has ONE float32 input, an awkward ONNX name that is not a valid C
identifier, and no softmax at all. Run it with:

    uv run python models/make_mlp_fixture.py
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

RNG = np.random.default_rng(0)
IN_DIM, HIDDEN, OUT_DIM = 16, 24, 8

# An ONNX name that C would reject verbatim; c_identifier() has to sanitize it.
INPUT_NAME = "input.1"


def main() -> None:
    w1 = numpy_helper.from_array(RNG.standard_normal((IN_DIM, HIDDEN)).astype(np.float32), "w1")
    b1 = numpy_helper.from_array(RNG.standard_normal(HIDDEN).astype(np.float32), "b1")
    w2 = numpy_helper.from_array(RNG.standard_normal((HIDDEN, OUT_DIM)).astype(np.float32), "w2")
    b2 = numpy_helper.from_array(RNG.standard_normal(OUT_DIM).astype(np.float32), "b2")

    nodes = [
        helper.make_node("MatMul", [INPUT_NAME, "w1"], ["h0"]),
        helper.make_node("Add", ["h0", "b1"], ["h1"]),
        helper.make_node("Relu", ["h1"], ["h2"]),
        helper.make_node("MatMul", ["h2", "w2"], ["h3"]),
        helper.make_node("Add", ["h3", "b2"], ["logits"]),
    ]

    graph = helper.make_graph(
        nodes,
        "tatva_mlp_fixture",
        inputs=[helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, IN_DIM])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, OUT_DIM])],
        initializer=[w1, b1, w2, b2],
    )

    model = helper.make_model(
        graph,
        producer_name="tatva-fixtures",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, "models/model_mlp.onnx")
    print("wrote models/model_mlp.onnx")


if __name__ == "__main__":
    main()
