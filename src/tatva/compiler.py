"""
TATVA Optimization Compiler Module.

This module handles ONNX model ingestion, computational-graph analysis,
and optimization schedule generation using Apache TVM. It also defines
the supported RISC-V target variants.
"""

import os
import re
from dataclasses import dataclass
from typing import Any

from tatva.diagnostics import CompilationError, ImportInProgressError, UnsupportedOperatorError


# Central target configuration from Phase 3
@dataclass
class TargetVariant:
    name: str
    gcc_march: str
    gcc_mabi: str
    tvm_target: str
    bitness: int
    experimental: bool = False
    notes: str = ""


# Target registry dict containing all supported target architectures
TARGETS: dict[str, TargetVariant] = {
    "RV32IMC": TargetVariant(
        name="RV32IMC",
        gcc_march="rv32imc",
        gcc_mabi="ilp32",
        tvm_target="llvm -mtriple=riscv32-unknown-elf -mabi=ilp32 -mcpu=generic-rv32",
        bitness=32,
        notes="32-bit integer, multiplication, and compressed instructions.",
    ),
    "RV32IMAC": TargetVariant(
        name="RV32IMAC",
        gcc_march="rv32imac",
        gcc_mabi="ilp32",
        tvm_target="llvm -mtriple=riscv32-unknown-elf -mabi=ilp32 -mcpu=generic-rv32",
        bitness=32,
        notes="32-bit integer, multiplication, atomic, and compressed instructions.",
    ),
    "RV64GC": TargetVariant(
        name="RV64GC",
        gcc_march="rv64gc",
        gcc_mabi="lp64d",
        tvm_target="llvm -mtriple=riscv64-unknown-elf -mabi=lp64d -mcpu=generic-rv64",
        bitness=64,
        notes="64-bit standard G (IMAFD) and C extensions. Default target.",
    ),
    "RV64IMAFDC": TargetVariant(
        name="RV64IMAFDC",
        gcc_march="rv64imafdc",
        gcc_mabi="lp64d",
        tvm_target="llvm -mtriple=riscv64-unknown-elf -mabi=lp64d -mcpu=generic-rv64",
        bitness=64,
        notes="64-bit integer, multiplication, atomic, single/double float, and compressed instructions.",
    ),
    "RV64GCV": TargetVariant(
        name="RV64GCV",
        gcc_march="rv64gcv",
        gcc_mabi="lp64d",
        tvm_target="llvm -mtriple=riscv64-unknown-elf -mabi=lp64d -mcpu=generic-rv64 -mattr=+v",
        bitness=64,
        # Experimental, and it always was. The C backend emits scalar loops; nothing in
        # TATVA generates RVV intrinsics, so the "+v" attribute buys you QEMU's vector
        # unit being *available*, not used. Marking it stable made the CLI hand it out
        # without the --allow-experimental warning that says exactly this.
        experimental=True,
        notes=(
            "64-bit standard extensions plus the Vector extension (RVV 1.0). The target builds and "
            "runs, but codegen is scalar C -- vector units are not yet targeted."
        ),
    ),
    "RV32EMC": TargetVariant(
        name="RV32EMC",
        gcc_march="rv32emc",
        gcc_mabi="ilp32e",
        tvm_target="llvm -mtriple=riscv32-unknown-elf -mabi=ilp32e -mcpu=generic-rv32",
        bitness=32,
        experimental=True,
        notes="32-bit embedded profile with 16 registers. Experimental target variant.",
    ),
}

DEFAULT_TARGET = "RV64GC"

# Set of standard TVM Relax operators supported by the bare-metal RISC-V targets.
#
# Every name here must be a real Relax operator with the "relax." prefix stripped,
# because that is exactly what `analyze_graph` compares against. Seven entries used to
# be Relay spellings that Relax never emits, so they could not match anything:
#
#   concatenate -> concat          nn.dense, nn.bias_add, nn.batch_matmul -> matmul/add
#   cast -> astype                 transpose -> permute_dims              unsqueeze -> expand_dims
#
# Six of those were harmless padding, but `concatenate` was not: the Relax name is
# `concat`, so every model containing a Concat -- which is most non-trivial graphs --
# was reported unsupported by a backend that compiles and runs it fine.
#
# The second group below was found the same way and is the larger half of the problem.
# This list was written by hand and never checked against the compiler, so it under-
# reported the backend by twenty-five operators -- including nn.conv2d and
# nn.max_pool2d, which is to say every convolutional model was told its operators were
# unsupported and offered a repair, by a toolchain that cross-compiles and runs it
# correctly. That is the "unsupported ops with a rewrite that does nothing" report.
#
# Nothing goes in this set on inspection any more. Every name here is covered by a
# corpus model that compiles for RV64GC, runs under QEMU and matches onnxruntime to
# 1e-4; `test_no_runnable_model_is_reported_unsupported` re-derives the whole claim
# from those runs, so an entry that stops being true fails the suite.
SUPPORTED_OPS = {
    "nn.relu",
    "nn.softmax",
    "nn.layer_norm",
    "matmul",
    "permute_dims",
    "astype",
    "add",
    "subtract",
    "multiply",
    "divide",
    "reshape",
    "squeeze",
    "concat",
    "split",
    "mean",
    "sum",
    "tanh",
    "sigmoid",
    "erf",
    "sqrt",
    "strided_slice",
    "take",
    "expand_dims",
    "full",
    "clip",
    "shape_of",
    "shape_to_tensor",
    "less",
    "where",
    "power",
    # Elementwise maths TVM lowers to a scalar C loop like any other.
    "exp",
    "log",
    "sin",
    "cos",
    "floor",
    "ceil",
    "round",
    "sign",
    "negative",
    "isnan",
    "mod",
    # Comparisons and boolean logic. These produce an int8 tensor, not a float one.
    "equal",
    "greater",
    "logical_and",
    "logical_not",
    "left_shift",
    # Binary reductions, and the shape plumbing TVM emits around them.
    "max",
    "min",
    "stack",
    "broadcast_to",
    # Activations with a closed form, lowered by TVM rather than rewritten by repair.
    "nn.gelu",
    "nn.leakyrelu",
    "nn.softplus",
    # Convolution and pooling: the reason this list mattered. Conv at one and three
    # spatial dimensions and the transposed form are here too -- an exported CNN is made
    # of these, and every one of them was refused by a backend that emits a real kernel
    # (conv1d six nested for-loops, conv2d nine, conv3d twelve).
    "nn.conv2d",
    "nn.conv1d",
    "nn.conv3d",
    "nn.conv2d_transpose",
    "nn.max_pool2d",
    "nn.avg_pool2d",
    "nn.prelu",
    "nn.log_softmax",
    # `variance` is what InstanceNormalization decomposes to, `argmax`/`one_hot` what
    # Hardmax does, and `image.resize2d` is Resize. None is an operator a user writes;
    # each was reported to them under a name their file does not contain.
    "variance",
    "argmax",
    "one_hot",
    "triu",
    "image.resize2d",
}




class ModelIR:
    """
    Wrapper for imported TVM Relax representations of models.
    """
    def __init__(self, mod: Any, params: Any, metadata: dict[str, Any]):
        self.mod = mod
        self.params = params
        self.metadata = metadata


@dataclass
class GraphReport:
    """
    Report containing metrics and analysis of a model's computation graph.
    """
    total_ops: int
    op_histogram: dict[str, int]
    has_transformer_bottleneck: bool
    unsupported_ops: list[str]


def c_safe_name(name: str) -> str:
    """
    Turn a tensor name into a valid C identifier.

    ONNX places no restrictions on tensor names, and real exporters use that freedom:
    PyTorch emits '/layers.0/attn/MatMul_output_0', and names with ':', '-' or a space
    turn up routinely. TVM's own sanitizer only replaces '.', so everything else
    survives into the generated C -- `var_in/put_0` is not an identifier, and the build
    fails in gcc with a syntax error pointing at a file the user never wrote.

    This is the single definition; runner.c_identifier delegates to it. Two sanitizers
    that disagree would let the host reference and the on-target harness classify the
    same tensor differently, which is the bug input_fill_kind's normalization exists to
    prevent.
    """
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    if not safe or safe[0].isdigit():
        safe = "t_" + safe
    return safe


def rename_inputs_to_c_identifiers(onnx_model: Any) -> dict[str, str]:
    """
    Rewrite graph input names in place so every one is a valid C identifier.

    Returns {safe name: original name} for the inputs that moved, empty if none did.

    Only graph inputs are touched, because only they become parameters of the emitted
    C function -- TVM names its own intermediates. Initializers are left alone even
    when an old-style graph lists them as inputs: they are folded to constants and
    never reach a signature.

    Renaming is safe for the parity check because the host feed is built from the
    original file by its original names, and input_fill_kind() normalizes both sides
    through c_safe_name() before deciding what to fill a tensor with.
    """
    graph = onnx_model.graph
    initializer_names = {init.name for init in graph.initializer}

    mapping: dict[str, str] = {}
    taken = {inp.name for inp in graph.input} | initializer_names
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        safe = c_safe_name(inp.name)
        if safe == inp.name:
            continue
        # Two different names can sanitize to the same identifier ('a.b' and 'a/b').
        # Silently merging them would wire one tensor to two inputs.
        candidate, n = safe, 1
        while candidate in taken:
            candidate, n = f"{safe}_{n}", n + 1
        taken.add(candidate)
        mapping[candidate] = inp.name

    if not mapping:
        return {}

    renames = {original: safe for safe, original in mapping.items()}
    for inp in graph.input:
        if inp.name in renames:
            inp.name = renames[inp.name]
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in renames:
                node.input[i] = renames[name]
    # A graph whose output is its own input passes the tensor straight through, so the
    # output carries the same name and has to move with it.
    for out in graph.output:
        if out.name in renames:
            out.name = renames[out.name]

    return mapping


def resolve_input_shapes(onnx_model: Any) -> dict[str, tuple[int, ...]]:
    """
    Resolve every graph input to a concrete static shape.

    Bare-metal codegen cannot emit dynamic dimensions, so symbolic dims are bound
    here: anything that looks like a sequence axis becomes 32, everything else
    (batch, and any other unnamed symbol) becomes 1.

    This is the single source of truth for input shapes. The host reference run and
    the on-target benchmark harness both go through it, so they always agree -- if
    they disagreed, the numerical parity check would compare two different problems.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for inp in onnx_model.graph.input:
        dims: list[int] = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param") and ("seq" in dim.dim_param or "length" in dim.dim_param):
                dims.append(32)
            else:
                dims.append(1)
        shapes[inp.name] = tuple(dims)
    return shapes


# TVM reports a refused graph as
#     The following operators are not supported for frontend ONNX: Dropout, LSTM
# so the operator names are everything after the final colon. The previous pattern
# searched for the literal string "UnsupportedOp", which only ever matched the
# fabricated fixture in models/ -- every real model came back as 'unknown', and stage
# 03 told the user an operator called "unknown" was blocking their build.
_UNSUPPORTED_LIST_RE = re.compile(r"not supported for frontend[^:]*:\s*(.+)", re.IGNORECASE)
_NO_OP_REGISTERED_RE = re.compile(r"No Op registered[^\w]*(?:for\s+)?[`'\"]?([A-Za-z_][\w.]*)")


def unsupported_operator_names(err_msg: str) -> list[str]:
    """
    Pull the operator names out of a TVM frontend rejection.

    Returns an empty list when the message names nothing, so the caller can say
    "unknown" once, deliberately, rather than every time.
    """
    match = _UNSUPPORTED_LIST_RE.search(err_msg)
    if match:
        # Only the first line: TVM appends a context block after the list.
        names = [n.strip() for n in match.group(1).splitlines()[0].split(",")]
        names = [n for n in names if n]
        if names:
            return names

    match = _NO_OP_REGISTERED_RE.search(err_msg)
    if match:
        return [match.group(1)]

    return []


def import_model(path: str, input_shapes: dict[str, tuple[int, ...]] | None = None) -> ModelIR:
    """
    Import an on-disk model (ONNX) into a TVM Relax module.
    Raises ImportInProgressError for PyTorch and TensorFlow formats.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        raise ImportInProgressError(
            "PyTorch import is in progress and not yet available; export to ONNX for now."
        )
    elif ext in (".pb", ".h5", ".savedmodel"):
        raise ImportInProgressError(
            "TensorFlow/Keras import is in progress and not yet available; export to ONNX for now."
        )
    elif ext != ".onnx":
        raise ValueError(f"Unsupported model format: '{ext}'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # The cache is keyed on file content only, so it is valid exclusively for the
    # default-shape import. Caller-supplied shapes neither read from nor write to it;
    # otherwise a custom-shape import would be handed back to a later default import.
    use_cache = input_shapes is None
    if use_cache:
        from tatva._cache import GLOBAL_SESSION_CACHE
        cached_ir = GLOBAL_SESSION_CACHE.get_model_ir(path)
        if cached_ir is not None:
            return cached_ir

    # Lazy imports to keep startup fast and independent
    import onnx
    from tvm.relax.frontend.onnx import from_onnx

    onnx_model = onnx.load(path)

    # Before anything reads a name off this graph. onnx.load returns a fresh object, so
    # this rewrites our copy and never the file on disk.
    renamed = rename_inputs_to_c_identifiers(onnx_model)

    if input_shapes is None:
        input_shapes = resolve_input_shapes(onnx_model)
    elif renamed:
        # A caller-supplied dict is keyed by the names in the file, which are the ones
        # they can see. Move it onto the names the graph now uses.
        originals = {original: safe for safe, original in renamed.items()}
        input_shapes = {originals.get(k, k): v for k, v in input_shapes.items()}

    try:
        mod = from_onnx(onnx_model, shape_dict=input_shapes)
    except UnsupportedOperatorError:
        raise
    except Exception as e:
        err_msg = str(e)
        names = unsupported_operator_names(err_msg)
        if names or "not supported for frontend" in err_msg or "No Op registered" in err_msg:
            raise UnsupportedOperatorError(
                operator_name=", ".join(names) if names else "unknown",
                details=f"Model contains operators unsupported by TVM frontend: {err_msg}"
            ) from e
        # Anything else the frontend throws is still an import failure, and it has to
        # leave here typed. A bare TypeError or AttributeError out of TVM's converters
        # (BitShift raises AttributeError: 'str' object has no attribute 'decode')
        # reaches classify_failure() as an unrecognised type and gets diagnosed as
        # "unknown" -- the user is shown a Python-internals message about a model they
        # can see nothing wrong with.
        raise CompilationError(
            stage="import",
            command=f"tvm.relax.frontend.onnx.from_onnx({os.path.basename(path)})",
            stderr=err_msg,
            details=(
                "The TVM ONNX frontend failed while converting this model. This is a frontend "
                "limitation, not a problem with the file: onnx.checker may well accept it. "
                f"Underlying error: {type(e).__name__}: {err_msg}"
            ),
        ) from e

    metadata = {
        "input_shapes": input_shapes,
        "file_size_bytes": os.path.getsize(path),
        "format": "ONNX",
        # Kept so later passes can go back to the source graph -- the quantizer
        # calibrates activation ranges by running the original model on the host.
        "source_path": os.path.abspath(path),
        # {name in the IR: name in the file}, for the inputs that had to be renamed.
        # Anything reporting an input back to the user should show the original: the
        # renamed one does not appear anywhere in the model they gave us.
        "renamed_inputs": renamed,
    }

    # Relax packs parameters directly as constants inside the module, so we pass None for params
    model_ir = ModelIR(mod, None, metadata)

    if use_cache:
        from tatva._cache import GLOBAL_SESSION_CACHE
        GLOBAL_SESSION_CACHE.put_model_ir(path, model_ir)

    return model_ir


# Relax's ways of calling a function rather than naming an operator. None of these is a
# model operator, and none belongs in a support diff -- see visit_call_ below.
_CALL_INTO_TIR = frozenset({
    "call_tir",
    "call_tir_inplace",
    "call_tir_with_grad",
    "call_dps_packed",
    "call_pure_packed",
    "call_builtin_with_ctx",
})


def analyze_graph(model_ir: ModelIR) -> GraphReport:
    """
    Analyze a TVM Relax module to gather graph statistics, count operators,
    detect attention/softmax subgraphs, and list unsupported operators.
    """
    import tvm.ir
    from tvm import relax

    @relax.expr_functor.visitor
    class OpCounter(relax.PyExprVisitor):
        def __init__(self):
            super().__init__()
            self.op_counts = {}
            self.total_ops = 0
            self.unsupported_ops = set()

        def visit_call_(self, call):
            super().visit_call_(call)
            if isinstance(call.op, tvm.ir.Op):
                # Report the same spelling SUPPORTED_OPS uses. The histogram used to
                # keep TVM's "relax." prefix while the support check stripped it, so
                # `tatva analyze` printed "relax.nn.softmax" against a documented
                # supported-op list that says "nn.softmax" -- two names, one operator.
                op_name = call.op.name
                if op_name.startswith("relax."):
                    op_name = op_name[len("relax."):]

                if op_name in _CALL_INTO_TIR:
                    # Not an operator. `call_tir` is how Relax calls a TIR function, and
                    # the ONNX frontend emits it for Pad, Tile and Einsum among others --
                    # the whole computation arrives as a PrimFunc with no operator node
                    # at all. Diffing the name against SUPPORTED_OPS told the user their
                    # model used an unsupported operator called 'call_tir', which is not
                    # something they can look up, remove or replace: it names TVM's
                    # calling convention, not anything in their file. It is also
                    # unconditionally lowerable, since TVM generated the callee itself.
                    #
                    # The callee's name is the useful thing, and it is what the generated
                    # C kernel and the profile will call it, so that is what goes in the
                    # histogram.
                    callee = call.args[0] if call.args else None
                    name_hint = getattr(callee, "name_hint", None)
                    op_name = str(name_hint) if name_hint else op_name
                    self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
                    self.total_ops += 1
                    return

                self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
                self.total_ops += 1
                if op_name not in SUPPORTED_OPS:
                    self.unsupported_ops.add(op_name)

    visitor = OpCounter()
    visitor.visit_expr(model_ir.mod["main"])

    has_softmax = any("softmax" in op_name for op_name in visitor.op_counts)
    has_matmul = any("matmul" in op_name or "dense" in op_name for op_name in visitor.op_counts)
    has_transformer_bottleneck = has_softmax and has_matmul

    return GraphReport(
        total_ops=visitor.total_ops,
        op_histogram=visitor.op_counts,
        has_transformer_bottleneck=has_transformer_bottleneck,
        unsupported_ops=list(visitor.unsupported_ops),
    )


# TODO: Implement TVM schedule optimization for RISC-V targets
