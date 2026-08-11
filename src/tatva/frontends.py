"""
TATVA model frontends.

Until now "import a model" meant "load an ONNX file", with the extension check and the
metadata extraction living inside `compiler.import_model`. That worked while ONNX was
the only answer, but it left two things nowhere to live: a real description of the model
for the user to look at before compiling, and an honest statement of which other formats
TATVA can and cannot take.

This module provides both.

`inspect_model` reads a file and describes it -- framework, format, size, parameter
count, input and output shapes, precision, operator count, detected model family --
without importing it into TVM. It is cheap enough to run the moment a file is selected.

`FORMATS` is the format registry. Its statuses are *computed*, not written down: each
entry is checked at runtime for whether TATVA implements the adapter, whether the
underlying TVM Relax frontend module exists in the installed TVM, and whether the
framework's own Python package is importable. A format is reported as supported only
when all three hold and TATVA actually exercises that path. Everything else says
"Coming Soon" and gives the specific reason it is not available -- which is more useful
to an engineer than a supported-formats list they will discover the hard way is
aspirational.

Model-family detection is a classifier over the graph, and it is allowed to fail. When
the signals do not add up to a family it returns "Unable to determine" and lists what it
did see, rather than picking the nearest label.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass, field
from typing import Any

# ------------------------------------------------------------------ format statuses
#
# The vocabulary the UI renders. SUPPORTED is the only one that means "this will work",
# and it is the only one a format has to earn.
SUPPORTED = "Supported"
EXPERIMENTAL = "Experimental"
PARTIAL = "Partial"
COMING_SOON = "Coming Soon"
UNSUPPORTED = "Unsupported"


@dataclass
class ModelFormat:
    key: str
    label: str
    framework: str
    extensions: list[str]
    # Does TATVA have an adapter that has actually been run for this format?
    implemented: bool
    # The TVM Relax frontend module this format would go through, if any.
    tvm_frontend: str
    # The Python package the format's own tooling needs.
    requires: list[str]
    notes: str = ""

    def status(self) -> tuple[str, str]:
        """
        Resolve this format's status against the environment TATVA is running in.

        Returns (status, reason). The reason is always populated for anything that is
        not `SUPPORTED`, and always names the concrete thing that is missing.
        """
        missing = [pkg for pkg in self.requires if importlib.util.find_spec(pkg) is None]
        frontend_present = _tvm_frontend_available(self.tvm_frontend) if self.tvm_frontend else False

        if self.implemented and not missing and frontend_present:
            return SUPPORTED, ""

        if self.implemented and missing:
            return PARTIAL, (
                f"TATVA implements this format, but {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not installed in this environment."
            )
        if self.implemented and not frontend_present:
            return PARTIAL, (
                f"TATVA implements this format, but the installed TVM has no "
                f"'{self.tvm_frontend}' Relax frontend."
            )

        # Not implemented. Say why, precisely, rather than a bare "coming soon".
        if not self.tvm_frontend:
            return COMING_SOON, (
                "No Relax frontend exists for this format, so TATVA would have to convert it "
                "before import. Export to ONNX in the meantime."
            )
        bits = [f"TATVA has not yet wired the '{self.tvm_frontend}' Relax frontend into its pipeline"]
        if not frontend_present:
            bits.append("and the installed TVM does not provide that frontend")
        if missing:
            bits.append(f"and {', '.join(missing)} is not installed here")
        return COMING_SOON, " ".join(bits) + ". Export to ONNX in the meantime."

    def to_json(self) -> dict[str, Any]:
        status, reason = self.status()
        out = asdict(self)
        out["status"] = status
        out["reason"] = reason
        out["extensions_label"] = ", ".join(self.extensions)
        return out


def _tvm_frontend_available(name: str) -> bool:
    """Is `tvm.relax.frontend.<name>` importable? Checked without importing TVM itself."""
    if not name:
        return False
    try:
        return importlib.util.find_spec(f"tvm.relax.frontend.{name}") is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


# The registry. `implemented` is the honest flag: it is True only for the path TATVA
# actually compiles models through today. Setting it True for anything else would make
# every downstream status a lie, which is exactly what this table exists to prevent.
FORMATS: list[ModelFormat] = [
    ModelFormat(
        key="onnx", label="ONNX", framework="ONNX", extensions=[".onnx"],
        implemented=True, tvm_frontend="onnx", requires=["onnx"],
        notes="TATVA's native path. Imported through TVM Relax and compiled end to end.",
    ),
    ModelFormat(
        key="torch_exported", label="PyTorch (ExportedProgram)", framework="PyTorch",
        extensions=[".pt2", ".ep"],
        implemented=False, tvm_frontend="torch", requires=["torch"],
        notes="torch.export graphs. The Relax frontend exists; TATVA has not yet adapted or verified it.",
    ),
    ModelFormat(
        key="torchscript", label="TorchScript / PyTorch state", framework="PyTorch",
        extensions=[".pt", ".pth"],
        implemented=False, tvm_frontend="torch", requires=["torch"],
        notes=(
            "A .pt file may hold a scripted module, a traced module or a bare state_dict; the last "
            "carries no graph at all. TATVA does not guess between them."
        ),
    ),
    ModelFormat(
        key="tflite", label="TensorFlow Lite", framework="TensorFlow", extensions=[".tflite"],
        implemented=False, tvm_frontend="tflite", requires=["tflite"],
        notes="The Relax frontend exists; TATVA has not yet adapted or verified it.",
    ),
    ModelFormat(
        key="tf_savedmodel", label="TensorFlow SavedModel / frozen graph", framework="TensorFlow",
        extensions=[".pb"],
        implemented=False, tvm_frontend="", requires=["tensorflow"],
        notes="No Relax frontend. Convert with tf2onnx, or export to TFLite.",
    ),
    ModelFormat(
        key="keras", label="Keras", framework="Keras", extensions=[".h5", ".keras"],
        implemented=False, tvm_frontend="", requires=["keras"],
        notes="No Relax frontend. Convert with tf2onnx.",
    ),
    ModelFormat(
        key="stablehlo", label="StableHLO", framework="JAX / XLA", extensions=[".mlir", ".stablehlo"],
        implemented=False, tvm_frontend="stablehlo", requires=[],
        notes="The Relax frontend exists; TATVA has not yet adapted or verified it.",
    ),
]

_BY_EXT: dict[str, ModelFormat] = {ext: fmt for fmt in FORMATS for ext in fmt.extensions}


def format_table() -> list[dict[str, Any]]:
    """The registry as the UI consumes it, statuses resolved against this environment."""
    return [f.to_json() for f in FORMATS]


def format_for_path(path: str) -> ModelFormat | None:
    return _BY_EXT.get(os.path.splitext(path or "")[1].lower())


# --------------------------------------------------------------- model-family detection

FAMILY_UNKNOWN = "Unable to determine"


@dataclass
class FamilyVerdict:
    family: str
    confidence: str          # high | medium | low | none
    signals: list[str] = field(default_factory=list)
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def detect_family(
    op_types: dict[str, int],
    *,
    param_count: int = 0,
    vocab_hint: int = 0,
    output_dims: list[Any] | None = None,
) -> FamilyVerdict:
    """
    Classify a model from the operators in its graph.

    `op_types` is a histogram of ONNX node op_types. Working from the source graph
    rather than the imported Relax module is deliberate: the frontend decomposes LSTM
    and GRU into their constituent arithmetic, so by the time a model reaches Relax the
    evidence that it was recurrent has been destroyed.

    `vocab_hint` is the row count of the tallest 2-D initializer -- an embedding table if
    there is one -- and `output_dims` the trailing dimension of each graph output. The
    two together are what separate a language model from an encoder with a small
    classification head, which are otherwise the same graph.

    Every verdict carries the signals it was based on, so a wrong answer is arguable
    rather than mysterious. When nothing fires, the answer is "Unable to determine".
    """
    ops = {k: v for k, v in (op_types or {}).items()}
    has = lambda *names: any(ops.get(n, 0) for n in names)  # noqa: E731
    signals: list[str] = []

    conv = sum(ops.get(n, 0) for n in ("Conv", "ConvTranspose", "ConvInteger", "QLinearConv"))
    attention = has("Attention", "MultiHeadAttention", "GroupQueryAttention")
    softmax = ops.get("Softmax", 0)
    matmul = sum(ops.get(n, 0) for n in ("MatMul", "Gemm", "MatMulInteger", "QLinearMatMul"))
    norm = sum(ops.get(n, 0) for n in ("LayerNormalization", "SimplifiedLayerNormalization", "RMSNormalization"))
    batchnorm = ops.get("BatchNormalization", 0)
    gather = sum(ops.get(n, 0) for n in ("Gather", "GatherElements"))
    pooling = sum(ops.get(n, 0) for n in ("MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool"))

    # Not every exporter emits a fused LayerNormalization node. Plenty of real models --
    # including anything exported through an older opset -- ship it decomposed into
    # mean, centre, variance, rsqrt, scale. Requiring the fused node made this classifier
    # call a two-layer BERT an MLP, so the decomposed shape counts as normalization too.
    norm_decomposed = 0
    if not norm and ops.get("ReduceMean", 0) >= 2 and ops.get("Sqrt", 0) >= 1 and has("Pow", "Mul"):
        # One normalization per two ReduceMean nodes: the mean, then the variance's mean.
        norm_decomposed = ops["ReduceMean"] // 2
        norm = norm_decomposed

    # Classical ML lives in the ai.onnx.ml domain and is unmistakable when present.
    classical = [n for n in ops if n in _CLASSICAL_ML_OPS]
    if classical:
        return FamilyVerdict(
            family="Classical ML", confidence="high",
            signals=[f"ai.onnx.ml operator: {n}" for n in sorted(classical)],
            detail="The graph uses ONNX's classical-ML operator set rather than a neural network.",
        )

    # Recurrent, in decreasing specificity. ONNX keeps these as single fused nodes.
    for onnx_op, label in (("LSTM", "LSTM"), ("GRU", "GRU"), ("RNN", "RNN")):
        if ops.get(onnx_op, 0):
            return FamilyVerdict(
                family=label, confidence="high",
                signals=[f"{ops[onnx_op]}x {onnx_op} node(s)"],
                detail=f"The graph contains explicit {onnx_op} nodes.",
            )

    # Attention. Either a fused attention node, or the classic decomposed shape:
    # normalisation + softmax + several matmuls.
    transformer = attention or (softmax and norm and matmul >= 4)
    if transformer:
        if attention:
            signals.append("fused attention node")
        if norm_decomposed:
            signals.append(f"~{norm_decomposed}x normalization, emitted in decomposed form")
        elif norm:
            signals.append(f"{norm}x layer/RMS normalization")
        if softmax:
            signals.append(f"{softmax}x softmax")
        if matmul:
            signals.append(f"{matmul}x matmul/gemm")

        # A vision transformer patch-embeds with a convolution before the blocks. One or
        # two convs alongside attention is a ViT stem; a deep conv stack is a CNN that
        # happens to end in attention, which is a different animal.
        if 0 < conv <= 2:
            signals.append(f"{conv}x convolution (patch embedding)")
            return FamilyVerdict(
                family="Vision Transformer (ViT)", confidence="medium", signals=signals,
                detail="Attention blocks preceded by a small number of convolutions, the usual patch-embedding stem.",
            )

        # A language model embeds tokens with a Gather over a large vocabulary table.
        if gather and vocab_hint >= 1000:
            signals.append(f"token embedding lookup over a {vocab_hint:,}-row table")

            # An embedding table alone does not make a language model -- an encoder with
            # a two-class head has one too. What distinguishes them is the output: a
            # language model projects back to the vocabulary. Without that projection
            # this is a transformer over text, and calling it an SLM would be a guess
            # dressed as a classification.
            lm_head = any(d == vocab_hint for d in (output_dims or []))
            if lm_head:
                signals.append(f"output projects back to the {vocab_hint:,}-token vocabulary")
                if 0 < param_count <= 3_000_000_000:
                    return FamilyVerdict(
                        family="Small Language Model (SLM)", confidence="medium", signals=signals,
                        detail=(
                            f"Transformer blocks over a token embedding table with a language-modelling "
                            f"head, {param_count:,} parameters. Classified as small on parameter count "
                            "alone; TATVA does not inspect the tokenizer or the training objective."
                        ),
                    )
                return FamilyVerdict(
                    family="Large Language Model (LLM)", confidence="medium", signals=signals,
                    detail=f"Transformer with a language-modelling head and {param_count:,} parameters.",
                )

            head = ", ".join(str(d) for d in (output_dims or []) if d is not None)
            return FamilyVerdict(
                family="Transformer (text encoder)", confidence="medium", signals=signals,
                detail=(
                    "Transformer blocks over a token embedding table, but the output is not a projection "
                    f"back to the vocabulary{f' (trailing output dimension: {head})' if head else ''} -- so this "
                    "is an encoder with a task head rather than a language model."
                ),
            )

        return FamilyVerdict(
            family="Transformer", confidence="high" if attention else "medium", signals=signals,
            detail=(
                "Fused attention operator present." if attention else
                "Normalization, softmax and repeated matmuls in the arrangement attention decomposes into."
            ),
        )

    # Convolutional.
    if conv:
        signals.append(f"{conv}x convolution")
        if pooling:
            signals.append(f"{pooling}x pooling")
        if batchnorm:
            signals.append(f"{batchnorm}x batch normalization")
        return FamilyVerdict(
            family="CNN", confidence="high", signals=signals,
            detail="Convolutions dominate the graph.",
        )

    # Fully connected. Matmuls and elementwise activations, nothing else structural.
    activations = sum(ops.get(n, 0) for n in ("Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Gelu", "Erf"))
    if matmul and not conv:
        signals.append(f"{matmul}x matmul/gemm")
        if activations:
            signals.append(f"{activations}x elementwise activation")
        return FamilyVerdict(
            family="MLP", confidence="medium" if activations else "low", signals=signals,
            detail="Dense layers with no convolution, recurrence or attention structure.",
        )

    seen = ", ".join(sorted(ops)) if ops else "no operators"
    return FamilyVerdict(
        family=FAMILY_UNKNOWN, confidence="none",
        signals=[f"operators present: {seen}"],
        detail=(
            "The graph does not match any family TATVA recognises. This does not affect compilation -- "
            "the family label is descriptive only."
        ),
    )


_CLASSICAL_ML_OPS = {
    "TreeEnsembleClassifier", "TreeEnsembleRegressor", "TreeEnsemble", "LinearClassifier",
    "LinearRegressor", "SVMClassifier", "SVMRegressor", "Scaler", "Normalizer", "Binarizer",
    "LabelEncoder", "OneHotEncoder", "DictVectorizer", "FeatureVectorizer", "Imputer",
    "CategoryMapper", "ArrayFeatureExtractor", "ZipMap",
}


# ------------------------------------------------------------------ model inspection

# ONNX TensorProto element types -> a name an engineer recognises. Kept as a literal map
# so this function does not need onnx imported to be reasoned about.
_ELEM_TYPE: dict[int, str] = {
    1: "FP32", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16", 6: "INT32", 7: "INT64",
    8: "STRING", 9: "BOOL", 10: "FP16", 11: "FP64", 12: "UINT32", 13: "UINT64",
    14: "COMPLEX64", 15: "COMPLEX128", 16: "BF16",
}

# Bytes per element, for the parameter-memory figure.
_ELEM_BYTES: dict[int, int] = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2,
}


@dataclass
class ModelInfo:
    """Everything the Input page shows about a model before anything is compiled."""
    ok: bool
    path: str = ""
    name: str = ""
    framework: str = ""
    format: str = ""
    format_status: str = ""
    format_reason: str = ""
    file_size_bytes: int = 0
    parameter_count: int = 0
    parameter_bytes: int = 0
    precision: str = ""
    op_count: int = 0
    distinct_op_count: int = 0
    op_types: dict[str, int] = field(default_factory=dict)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    has_dynamic_shapes: bool = False
    opset: str = ""
    producer: str = ""
    family: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _value_info(vi: Any) -> dict[str, Any]:
    """Shape and dtype of one graph input or output, keeping symbolic dims symbolic."""
    tt = vi.type.tensor_type
    dims: list[Any] = []
    dynamic = False
    for dim in tt.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            dims.append(dim.dim_param)
            dynamic = True
        else:
            dims.append("?")
            dynamic = True
    return {
        "name": vi.name,
        "shape": dims,
        "shape_label": "[" + ", ".join(str(d) for d in dims) + "]" if dims else "scalar",
        "dtype": _ELEM_TYPE.get(int(tt.elem_type), f"type {tt.elem_type}"),
        "dynamic": dynamic,
    }


def inspect_model(path: str) -> ModelInfo:
    """
    Describe a model file without compiling it.

    Reads the ONNX protobuf directly -- no TVM, no import, no graph lowering -- so it
    can run the instant a file is picked. Everything reported is read out of the file;
    nothing is defaulted or estimated.
    """
    name = os.path.basename(path or "")
    fmt = format_for_path(path)

    if not path or not os.path.isfile(path):
        return ModelInfo(ok=False, path=path, name=name, error="File not found.")

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return ModelInfo(ok=False, path=path, name=name, error=f"Could not read the file: {exc}")

    if fmt is None:
        ext = os.path.splitext(path)[1].lower() or "(no extension)"
        return ModelInfo(
            ok=False, path=path, name=name, file_size_bytes=size,
            format=ext, format_status=UNSUPPORTED,
            format_reason=f"TATVA does not recognise the extension {ext}.",
            error=f"Unrecognised model format: {ext}",
        )

    status, reason = fmt.status()
    base = ModelInfo(
        ok=False, path=os.path.abspath(path), name=name, framework=fmt.framework,
        format=fmt.label, format_status=status, format_reason=reason, file_size_bytes=size,
    )

    if fmt.key != "onnx":
        # Nothing dishonest to add: TATVA cannot read the graph, so it reports the file
        # and the reason, and leaves every graph field empty rather than guessed.
        base.error = reason or f"{fmt.label} import is not available yet."
        return base

    try:
        import onnx
    except ImportError as exc:
        base.error = f"The onnx package is required to inspect this file: {exc}"
        return base

    try:
        model = onnx.load(path)
    except Exception as exc:
        base.error = f"The file could not be parsed as ONNX: {exc}"
        return base

    graph = model.graph

    op_types: dict[str, int] = {}
    for node in graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    params = 0
    param_bytes = 0
    dtypes: set[str] = set()
    vocab_hint = 0
    for init in graph.initializer:
        count = 1
        for d in init.dims:
            count *= int(d)
        params += count
        elem = int(init.data_type)
        param_bytes += count * _ELEM_BYTES.get(elem, 4)
        dtypes.add(_ELEM_TYPE.get(elem, f"type {elem}"))
        # A 2-D initializer with a very tall first axis is an embedding table; its row
        # count is the vocabulary size the family classifier asks about.
        if len(init.dims) == 2 and int(init.dims[0]) >= 1000:
            vocab_hint = max(vocab_hint, int(init.dims[0]))

    weight_dtypes = {d for d in dtypes if d not in ("INT64", "INT32", "BOOL")}
    if not weight_dtypes:
        precision = "/".join(sorted(dtypes)) if dtypes else "no parameters"
    elif len(weight_dtypes) == 1:
        precision = next(iter(weight_dtypes))
    else:
        precision = "Mixed (" + "/".join(sorted(weight_dtypes)) + ")"

    inputs = [_value_info(vi) for vi in graph.input if vi.name not in {i.name for i in graph.initializer}]
    outputs = [_value_info(vi) for vi in graph.output]

    opset = ", ".join(
        f"{o.domain or 'ai.onnx'} v{o.version}" for o in model.opset_import
    )
    producer = " ".join(x for x in (model.producer_name, model.producer_version) if x).strip()

    # Trailing dimension of each output, used to tell a language-modelling head from a
    # task head. Symbolic dimensions are dropped rather than coerced to a number.
    output_dims = [o["shape"][-1] for o in outputs if o["shape"] and isinstance(o["shape"][-1], int)]

    verdict = detect_family(op_types, param_count=params, vocab_hint=vocab_hint, output_dims=output_dims)

    base.ok = True
    base.parameter_count = params
    base.parameter_bytes = param_bytes
    base.precision = precision
    base.op_count = len(graph.node)
    base.distinct_op_count = len(op_types)
    base.op_types = dict(sorted(op_types.items(), key=lambda kv: (-kv[1], kv[0])))
    base.inputs = inputs
    base.outputs = outputs
    base.has_dynamic_shapes = any(i["dynamic"] for i in inputs) or any(o["dynamic"] for o in outputs)
    base.opset = opset
    base.producer = producer
    base.family = verdict.to_json()
    return base
