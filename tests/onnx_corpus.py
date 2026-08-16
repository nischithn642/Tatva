"""
A synthetic corpus of small ONNX models, built on demand.

The fixtures in `models/` are four hand-picked graphs, all of them things TATVA is
known to handle. They say nothing about the question a user actually asks, which is
"what happens when I point this at *my* model" -- and the honest answer to that has to
cover operators TATVA does not support and shapes it was never shown.

So this module builds the awkward cases deliberately: every operator family the ONNX
opset offers, models whose only operator is one TVM's frontend refuses, symbolic
dimensions, rank-0 and rank-5 tensors, tensor names that are not C identifiers, several
inputs and several outputs, int64 data, no initializers, an initializer nothing reads,
two opset versions either side of the one the fixtures use, and a chain deep enough to
be worth walking.

Nothing here is committed as a binary. `build_corpus` writes the whole set into a
directory the caller owns -- in tests, a session-scoped tmp dir -- so the corpus is
regenerated from this source every run and can never drift from it.

Most of these models are not expected to compile. That is the point: the contract under
test is that TATVA reaches a *verdict* on each of them and says which operator is
responsible, not that it accepts them.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

F = TensorProto.FLOAT
B = TensorProto.BOOL
U8 = TensorProto.UINT8

# Matches models/make_mlp_fixture.py. ir_version is pinned because onnx writes the
# newest one it knows, and a version ahead of the installed onnxruntime/TVM is rejected
# at load time -- a corpus-wide failure that has nothing to do with the graph.
_DEFAULT_OPSET = 17
_IR_VERSION = 10


def _randn(case: str, shape) -> np.ndarray:
    """
    Deterministic normal weights, seeded from the case name.

    A single shared RNG would make the values depend on how many models were built
    before this one, so a case built alone and the same case built as part of the sweep
    would not be the same model. The repair engine verifies its rewrites numerically --
    that is exactly the kind of difference that makes a check pass in one run and fail
    in the next.
    """
    seed = zlib.crc32(case.encode()) & 0xFFFFFFFF
    return np.random.default_rng(seed).standard_normal(shape)


def _weight(name: str, arr: np.ndarray):
    return numpy_helper.from_array(np.asarray(arr).astype(np.float32), name)


def _i64(name: str, values) -> object:
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def _vi(name: str, dtype: int, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def _build(
    dest: Path, name: str, nodes, inputs, outputs, inits=(), opset: int = _DEFAULT_OPSET,
    check: bool = True,
) -> Path:
    import onnx

    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=list(inits))
    model = helper.make_model(
        graph, producer_name="tatva-corpus", opset_imports=[helper.make_opsetid("", opset)]
    )
    model.ir_version = _IR_VERSION

    # full_check runs shape and type inference, which is the only thing that catches a
    # fixture that is merely plausible: `Less` declared with a float output passes the
    # default check and is rejected by onnxruntime at load. A corpus of invalid models
    # would prove nothing -- "TATVA refused it" is not evidence when the file is broken.
    #
    # check=False is for the deliberately malformed cases, which exist to be refused.
    if check:
        onnx.checker.check_model(model, full_check=True)

    path = dest / f"{name}.onnx"
    onnx.save(model, str(path))
    return path


# --------------------------------------------------------------------------- families
#
# Each entry below is (case name -> builder). The names are grouped by prefix because
# the sweep test parametrizes on them and the prefix is what tells you, from the test id
# alone, which kind of thing broke.

# One input, one operator, one output. The cheapest possible probe of "does this
# operator have a path through the pipeline". Float in and float out unless the
# operator's type constraint says otherwise -- see the explicit entries in _case_table.
_UNARY_OPS = [
    "Relu", "Sigmoid", "Tanh", "Erf", "Sqrt", "Softmax", "Exp", "Log", "Floor", "Ceil",
    "Round", "Neg", "Abs", "Sin", "Cos", "Sign", "Reciprocal", "Softplus",
    "HardSigmoid", "Elu", "Selu", "Identity",
]

# Elementwise pairs, second operand baked in as an initializer so the graph has exactly
# one input regardless of arity.
_BINARY_OPS = ["Add", "Sub", "Mul", "Div", "Min", "Max"]


def _unary(op: str, dtype: int = F, out_dtype: int | None = None,
           opset: int = _DEFAULT_OPSET, **attrs):
    def make(dest: Path, name: str) -> Path:
        return _build(
            dest, name,
            [helper.make_node(op, ["x"], ["y"], **attrs)],
            [_vi("x", dtype, [1, 8])],
            [_vi("y", dtype if out_dtype is None else out_dtype, [1, 8])],
            opset=opset,
        )
    return make


def _operand(case: str, dtype: int, values=None):
    """
    The baked-in second operand of a binary case, typed to match the operator.

    `And` takes booleans and `BitShift` takes unsigned integers; handing either a float
    initializer produces a model that ONNX type inference rejects outright, which would
    make the case a test of the checker rather than of TATVA.

    Weights stay seeded from the case name, per _randn.
    """
    if values is not None:
        return _weight("w", values) if dtype == F else numpy_helper.from_array(values, "w")
    if dtype == B:
        return numpy_helper.from_array((np.arange(8).reshape(1, 8) % 2 == 0), "w")
    if dtype == U8:
        return numpy_helper.from_array((np.arange(8).reshape(1, 8) % 4).astype(np.uint8), "w")
    return _weight("w", _randn(case, (1, 8)))


def _binary(op: str, dtype: int = F, out_dtype: int | None = None, operand=None, **attrs):
    def make(dest: Path, name: str) -> Path:
        return _build(
            dest, name,
            [helper.make_node(op, ["x", "w"], ["y"], **attrs)],
            [_vi("x", dtype, [1, 8])],
            [_vi("y", dtype if out_dtype is None else out_dtype, [1, 8])],
            inits=[_operand(name, dtype, operand)],
        )
    return make


def _op_matmul(dest, name):
    return _build(dest, name, [helper.make_node("MatMul", ["x", "w"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 4])],
                  inits=[_weight("w", _randn(name, (8, 4)))])


def _op_gemm(dest, name):
    return _build(dest, name, [helper.make_node("Gemm", ["x", "w", "b"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 4])],
                  inits=[_weight("w", _randn(name, (8, 4))),
                         _weight("b", _randn(name, 4))])


def _op_conv(dest, name):
    return _build(dest, name,
                  [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
                  [_vi("x", F, [1, 3, 8, 8])], [_vi("y", F, [1, 4, 8, 8])],
                  inits=[_weight("w", _randn(name, (4, 3, 3, 3)))])


def _op_maxpool(dest, name):
    return _build(dest, name,
                  [helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2])],
                  [_vi("x", F, [1, 3, 8, 8])], [_vi("y", F, [1, 3, 4, 4])])


def _op_batchnorm(dest, name):
    return _build(dest, name,
                  [helper.make_node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"])],
                  [_vi("x", F, [1, 3, 8, 8])], [_vi("y", F, [1, 3, 8, 8])],
                  inits=[_weight("s", np.ones(3)), _weight("b", np.zeros(3)),
                         _weight("m", np.zeros(3)), _weight("v", np.ones(3))])


def _op_layernorm(dest, name):
    return _build(dest, name,
                  [helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])],
                  inits=[_weight("s", np.ones(8)), _weight("b", np.zeros(8))])


def _op_reshape(dest, name):
    return _build(dest, name, [helper.make_node("Reshape", ["x", "shp"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 4, 2])],
                  inits=[_i64("shp", [1, 4, 2])])


def _op_transpose(dest, name):
    return _build(dest, name, [helper.make_node("Transpose", ["x"], ["y"], perm=[1, 0])],
                  [_vi("x", F, [2, 8])], [_vi("y", F, [8, 2])])


def _op_concat(dest, name):
    # The regression case for the `concatenate`/`concat` spelling bug: Relax names this
    # operator `concat`, SUPPORTED_OPS spelled it the Relay way, and so every model with
    # a Concat in it was reported unsupported by a backend that compiles it fine.
    return _build(dest, name, [helper.make_node("Concat", ["x", "w"], ["y"], axis=1)],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 16])],
                  inits=[_weight("w", _randn(name, (1, 8)))])


def _op_reducemean(dest, name):
    return _build(dest, name,
                  [helper.make_node("ReduceMean", ["x"], ["y"], axes=[1], keepdims=1)],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 1])])


def _op_gather(dest, name):
    return _build(dest, name, [helper.make_node("Gather", ["w", "idx"], ["y"], axis=0)],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [2, 4])],
                  inits=[_weight("w", _randn(name, (8, 4))), _i64("idx", [0, 2])])


def _op_slice(dest, name):
    return _build(dest, name, [helper.make_node("Slice", ["x", "s", "e", "a"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 4])],
                  inits=[_i64("s", [0]), _i64("e", [4]), _i64("a", [1])])


# The convolutional and shape-plumbing operators a real exported CNN is made of. Every
# one of these was reported unsupported by a backend that lowers it, because
# SUPPORTED_OPS had been written from memory: AveragePool, ConvTranspose, Pad, Resize,
# PRelu, LogSoftmax, Tile, CumSum, Trilu, InstanceNormalization, Hardmax and Conv at 1
# and 3 spatial dims. They are here so that claim is never made from memory again.


def _op_avgpool(dest, name):
    return _build(dest, name,
                  [helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2])],
                  [_vi("x", F, [1, 3, 8, 8])], [_vi("y", F, [1, 3, 4, 4])])


def _op_globalavgpool(dest, name):
    return _build(dest, name, [helper.make_node("GlobalAveragePool", ["x"], ["y"])],
                  [_vi("x", F, [1, 3, 8, 8])], [_vi("y", F, [1, 3, 1, 1])])


def _op_conv1d(dest, name):
    return _build(dest, name,
                  [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3], pads=[1, 1])],
                  [_vi("x", F, [1, 3, 8])], [_vi("y", F, [1, 4, 8])],
                  inits=[_weight("w", _randn(name, (4, 3, 3)))])


def _op_conv3d(dest, name):
    return _build(dest, name,
                  [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3, 3],
                                    pads=[1, 1, 1, 1, 1, 1])],
                  [_vi("x", F, [1, 2, 4, 4, 4])], [_vi("y", F, [1, 2, 4, 4, 4])],
                  inits=[_weight("w", _randn(name, (2, 2, 3, 3, 3)))])


def _op_convtranspose(dest, name):
    return _build(dest, name,
                  [helper.make_node("ConvTranspose", ["x", "w"], ["y"], kernel_shape=[3, 3])],
                  [_vi("x", F, [1, 2, 4, 4])], [_vi("y", F, [1, 2, 6, 6])],
                  inits=[_weight("w", _randn(name, (2, 2, 3, 3)))])


def _op_prelu(dest, name):
    return _build(dest, name, [helper.make_node("PRelu", ["x", "s"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])],
                  inits=[_weight("s", np.full(8, 0.1))])


def _op_logsoftmax(dest, name):
    return _build(dest, name, [helper.make_node("LogSoftmax", ["x"], ["y"], axis=1)],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])])


def _op_instancenorm(dest, name):
    return _build(dest, name,
                  [helper.make_node("InstanceNormalization", ["x", "s", "b"], ["y"])],
                  [_vi("x", F, [1, 3, 4, 4])], [_vi("y", F, [1, 3, 4, 4])],
                  inits=[_weight("s", np.ones(3)), _weight("b", np.zeros(3))])


def _op_hardmax(dest, name):
    return _build(dest, name, [helper.make_node("Hardmax", ["x"], ["y"], axis=1)],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])])


def _op_trilu(dest, name):
    return _build(dest, name, [helper.make_node("Trilu", ["x"], ["y"], upper=1)],
                  [_vi("x", F, [8, 8])], [_vi("y", F, [8, 8])])


def _op_cumsum(dest, name):
    # The axis is a 0-D tensor, not a 1-element one. ONNX accepts the latter from a
    # lenient checker and onnxruntime does not.
    return _build(dest, name, [helper.make_node("CumSum", ["x", "ax"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])],
                  inits=[_i64("ax", 1)])


def _op_resize(dest, name):
    return _build(dest, name,
                  [helper.make_node("Resize", ["x", "", "s"], ["y"], mode="nearest")],
                  [_vi("x", F, [1, 2, 4, 4])], [_vi("y", F, [1, 2, 8, 8])],
                  inits=[_weight("s", np.array([1.0, 1.0, 2.0, 2.0]))])


# The three below arrive as `call_tir` -- TVM's frontend lowers them straight to a TIR
# PrimFunc with no operator node at all. That used to be reported to the user as an
# unsupported operator named 'call_tir', which names TVM's calling convention rather than
# anything in their model, and cannot be looked up, removed or replaced.


def _op_pad(dest, name):
    return _build(dest, name, [helper.make_node("Pad", ["x", "p"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 10])],
                  inits=[_i64("p", [0, 1, 0, 1])])


def _op_tile(dest, name):
    return _build(dest, name, [helper.make_node("Tile", ["x", "r"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [2, 8])],
                  inits=[_i64("r", [2, 1])])


def _op_einsum(dest, name):
    return _build(dest, name, [helper.make_node("Einsum", ["x", "w"], ["y"], equation="ij,jk->ik")],
                  [_vi("x", F, [2, 8])], [_vi("y", F, [2, 4])],
                  inits=[_weight("w", _randn(name, (8, 4)))])


# Shape plumbing. Already covered by name in SUPPORTED_OPS, never covered from the ONNX
# side, and present in almost every exported model.


def _op_flatten(dest, name):
    return _build(dest, name, [helper.make_node("Flatten", ["x"], ["y"], axis=1)],
                  [_vi("x", F, [1, 2, 2, 2])], [_vi("y", F, [1, 8])])


def _op_squeeze(dest, name):
    return _build(dest, name, [helper.make_node("Squeeze", ["x", "ax"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [8])], inits=[_i64("ax", [0])])


def _op_unsqueeze(dest, name):
    return _build(dest, name, [helper.make_node("Unsqueeze", ["x", "ax"], ["y"])],
                  [_vi("x", F, [8])], [_vi("y", F, [1, 8])], inits=[_i64("ax", [0])])


def _op_where(dest, name):
    return _build(dest, name, [helper.make_node("Where", ["c", "x", "b"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])],
                  inits=[numpy_helper.from_array((np.arange(8).reshape(1, 8) % 2 == 0), "c"),
                         _weight("b", _randn(name, (1, 8)))])


def _op_reducesum(dest, name):
    return _build(dest, name,
                  [helper.make_node("ReduceSum", ["x", "ax"], ["y"], keepdims=1)],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 1])], inits=[_i64("ax", [1])])


def _op_dropout(dest, name):
    # TVM's ONNX frontend refuses this one outright, which makes it the case that proves
    # the blocking operator is named. It used to come back as 'unknown'.
    return _build(dest, name, [helper.make_node("Dropout", ["x"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])])


def _op_lstm(dest, name):
    return _build(dest, name, [helper.make_node("LSTM", ["x", "w", "r"], ["y"], hidden_size=4)],
                  [_vi("x", F, [1, 1, 8])], [_vi("y", F, [1, 1, 1, 4])],
                  inits=[_weight("w", _randn(name, (1, 16, 8))),
                         _weight("r", _randn(name, (1, 16, 4)))])


def _op_nonzero(dest, name):
    # Data-dependent output shape: nothing downstream can give it a static extent.
    return _build(dest, name, [helper.make_node("NonZero", ["x"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", TensorProto.INT64, [2, "n"])])


def _op_fabricated(dest, name):
    # An operator that does not exist in any opset. The one case built unchecked on
    # purpose: it is here to be refused, and by a component further down than the checker.
    return _build(dest, name, [helper.make_node("UnsupportedOpXYZ", ["x"], ["y"], domain="")],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])], check=False)


def _struct_dynamic_batch(dest, name):
    return _build(dest, name, [helper.make_node("Relu", ["x"], ["y"])],
                  [_vi("x", F, ["batch", 8])], [_vi("y", F, ["batch", 8])])


def _struct_dynamic_seq(dest, name):
    return _build(dest, name, [helper.make_node("Relu", ["x"], ["y"])],
                  [_vi("x", F, [1, "seq_len"])], [_vi("y", F, [1, "seq_len"])])


def _struct_scalar_input(dest, name):
    return _build(dest, name, [helper.make_node("Relu", ["x"], ["y"])],
                  [_vi("x", F, [])], [_vi("y", F, [])])


def _struct_rank5(dest, name):
    return _build(dest, name, [helper.make_node("Relu", ["x"], ["y"])],
                  [_vi("x", F, [1, 2, 2, 2, 2])], [_vi("y", F, [1, 2, 2, 2, 2])])


def _struct_hostile_names(dest, name):
    # Slash, colon, dot and a space -- none of which can survive into a C identifier.
    # TVM's own sanitizer only replaces '.', so the rest reached the generated C and gcc
    # died on `var_in/put_0`. PyTorch's exporter names tensors '/layers.0/attn/MatMul',
    # so this is the common case rather than an exotic one.
    #
    # Sigmoid rather than Relu on purpose: Relu of the harness's default ramp is zero
    # across the first half of the tensor, and a parity check that passes by comparing
    # zeros to zeros would not notice the input being wired up wrong.
    return _build(dest, name, [helper.make_node("Sigmoid", ["in/put:0"], ["out.put 1"])],
                  [_vi("in/put:0", F, [1, 8])], [_vi("out.put 1", F, [1, 8])])


def _struct_multi_io(dest, name):
    return _build(dest, name,
                  [helper.make_node("Add", ["a", "b"], ["s"]),
                   helper.make_node("Mul", ["a", "b"], ["p"])],
                  [_vi("a", F, [1, 4]), _vi("b", F, [1, 4])],
                  [_vi("s", F, [1, 4]), _vi("p", F, [1, 4])])


def _struct_int64_input(dest, name):
    return _build(dest, name, [helper.make_node("Add", ["x", "x"], ["y"])],
                  [_vi("x", TensorProto.INT64, [1, 8])], [_vi("y", TensorProto.INT64, [1, 8])])


def _struct_no_initializers(dest, name):
    return _build(dest, name, [helper.make_node("Add", ["a", "b"], ["y"])],
                  [_vi("a", F, [1, 8]), _vi("b", F, [1, 8])], [_vi("y", F, [1, 8])])


def _struct_unused_initializer(dest, name):
    return _build(dest, name, [helper.make_node("Relu", ["x"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])],
                  inits=[_weight("dead", _randn(name, (4, 4)))])


def _struct_opset11(dest, name):
    return _build(dest, name, [helper.make_node("Relu", ["x"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])], opset=11)


def _struct_opset21(dest, name):
    return _build(dest, name, [helper.make_node("Relu", ["x"], ["y"])],
                  [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])], opset=21)


def _struct_deep_chain(dest, name):
    nodes, prev = [], "x"
    for i in range(40):
        nodes.append(helper.make_node("Relu", [prev], [f"t{i}"]))
        prev = f"t{i}"
    nodes.append(helper.make_node("Identity", [prev], ["y"]))
    return _build(dest, name, nodes, [_vi("x", F, [1, 8])], [_vi("y", F, [1, 8])])


_OP_BUILDERS = [
    _op_matmul, _op_gemm, _op_conv, _op_maxpool, _op_batchnorm, _op_layernorm,
    _op_reshape, _op_transpose, _op_concat, _op_reducemean, _op_gather, _op_slice,
    _op_dropout, _op_lstm, _op_nonzero, _op_fabricated,
    _op_avgpool, _op_globalavgpool, _op_conv1d, _op_conv3d, _op_convtranspose,
    _op_prelu, _op_logsoftmax, _op_instancenorm, _op_hardmax, _op_trilu, _op_cumsum,
    _op_resize, _op_pad, _op_tile, _op_einsum,
    _op_flatten, _op_squeeze, _op_unsqueeze, _op_where, _op_reducesum,
]

_STRUCT_BUILDERS = [
    _struct_dynamic_batch, _struct_dynamic_seq, _struct_scalar_input, _struct_rank5,
    _struct_hostile_names, _struct_multi_io, _struct_int64_input,
    _struct_no_initializers, _struct_unused_initializer, _struct_opset11,
    _struct_opset21, _struct_deep_chain,
]


def _case_table() -> dict:
    cases: dict = {}
    for op in _UNARY_OPS:
        cases[f"unary_{op}"] = _unary(op)
    # Clip and LeakyRelu need an attribute to be well-formed, so they are spelled out.
    cases["unary_LeakyRelu"] = _unary("LeakyRelu", alpha=0.01)
    cases["unary_Clip"] = _unary("Clip")
    # `Not` is boolean-only. Mish and Gelu do not exist below opset 18 and 20, so a
    # fixture at the default 17 would be rejected before TATVA ever saw the operator.
    cases["unary_Not"] = _unary("Not", dtype=B)
    cases["unary_Mish"] = _unary("Mish", opset=18)
    cases["unary_Gelu"] = _unary("Gelu", opset=20)

    for op in _BINARY_OPS:
        cases[f"binary_{op}"] = _binary(op)
    # Comparisons return bool whatever they compare.
    for op in ("Greater", "Less", "Equal"):
        cases[f"binary_{op}"] = _binary(op, out_dtype=B)
    cases["binary_And"] = _binary("And", dtype=B)
    cases["binary_BitShift"] = _binary("BitShift", dtype=U8, direction="LEFT")
    # Mod on floats is only defined with fmod=1.
    cases["binary_Mod"] = _binary("Mod", fmod=1)
    # A squared ramp rather than random exponents: a negative base raised to a fractional
    # power is NaN everywhere, and a parity check comparing NaN to NaN would pass without
    # the operator having computed anything.
    cases["binary_Pow"] = _binary("Pow", operand=np.full((1, 8), 2.0, dtype=np.float32))
    for fn in _OP_BUILDERS + _STRUCT_BUILDERS:
        cases[fn.__name__.lstrip("_")] = fn
    return cases


#: Every case name in the corpus, sorted. Import this to parametrize without paying for
#: the models -- building them costs a few hundred milliseconds and a test that only
#: needs the names should not.
CASE_NAMES: list[str] = sorted(_case_table())


def build_corpus(dest: Path) -> dict[str, Path]:
    """
    Write the whole corpus into `dest` and return {case name: path}.

    `dest` is created if it does not exist. Existing files are overwritten, so calling
    this twice into the same directory is safe and gives the same bytes both times.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return {name: make(dest, name) for name, make in sorted(_case_table().items())}
