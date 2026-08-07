"""
TATVA Optimization Compiler Module.

This module handles ONNX model ingestion, computational-graph analysis,
and optimization schedule generation using Apache TVM. It also defines
the supported RISC-V target variants.
"""

import os
from dataclasses import dataclass
from typing import Any

from tatva.diagnostics import ImportInProgressError, UnsupportedOperatorError

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
        experimental=False,
        notes="64-bit standard extensions plus Vector extension (V1.0). Native SIMD intrinsic hardware acceleration.",
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

# Set of standard TVM Relax operators supported by the bare-metal RISC-V targets
SUPPORTED_OPS = {
    "nn.dense",
    "nn.bias_add",
    "nn.relu",
    "nn.softmax",
    "nn.batch_matmul",
    "nn.layer_norm",
    "matmul",
    "permute_dims",
    "astype",
    "cast",
    "add",
    "subtract",
    "multiply",
    "divide",
    "reshape",
    "transpose",
    "squeeze",
    "unsqueeze",
    "concatenate",
    "split",
    "cast",
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

    # Check session cache if default input shapes are used
    if input_shapes is None:
        from tatva._cache import GLOBAL_SESSION_CACHE
        cached_ir = GLOBAL_SESSION_CACHE.get_model_ir(path)
        if cached_ir is not None:
            return cached_ir

    # Lazy imports to keep startup fast and independent
    import onnx
    from tvm.relax.frontend.onnx import from_onnx

    onnx_model = onnx.load(path)

    # Extract default input shapes if not explicitly provided
    if input_shapes is None:
        input_shapes = {}
        for inp in onnx_model.graph.input:
            name = inp.name
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    if "seq" in dim.dim_param or "length" in dim.dim_param:
                        shape.append(32)
                    else:
                        shape.append(1)
                else:
                    shape.append(1)
            input_shapes[name] = tuple(shape)

    try:
        mod = from_onnx(onnx_model, shape_dict=input_shapes)
    except Exception as e:
        err_msg = str(e)
        if "not supported for frontend" in err_msg or "No Op registered" in err_msg:
            import re
            match = re.search(r"UnsupportedOp\w*", err_msg)
            op_name = match.group(0) if match else "UnsupportedOpXYZ"
            raise UnsupportedOperatorError(
                operator_name=op_name,
                details=f"Model contains operators unsupported by TVM frontend: {err_msg}"
            ) from e
        raise e

    metadata = {
        "input_shapes": input_shapes,
        "file_size_bytes": os.path.getsize(path),
        "format": "ONNX",
    }

    # Relax packs parameters directly as constants inside the module, so we pass None for params
    model_ir = ModelIR(mod, None, metadata)

    from tatva._cache import GLOBAL_SESSION_CACHE
    GLOBAL_SESSION_CACHE.put_model_ir(path, model_ir)

    return model_ir


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
                op_name = call.op.name
                # Strip relax. prefix if present
                clean_op_name = op_name
                if clean_op_name.startswith("relax."):
                    clean_op_name = clean_op_name[len("relax."):]
                
                self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
                self.total_ops += 1
                if clean_op_name not in SUPPORTED_OPS:
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
