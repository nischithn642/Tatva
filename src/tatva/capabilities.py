"""
TATVA target capability database.

One place that answers, for a given operator on a given RISC-V target: is there a
lowering, what does the generated code actually look like, is it on the hot path, does
an optimization pass touch it, and is there a graph rewrite that can repair it when it
is missing.

This exists because the same answers used to be written inline in three places -- a
chain of `if "softmax" in op` branches in the GUI bridge, a flat `SUPPORTED_OPS` set in
the compiler, and prose in the UI -- so the three could disagree and did. Everything
that reports operator status now reads from here, including the frontend: the studio
renders whatever this module returns rather than carrying its own copy of the table.

Nothing here is aspirational. An operator appears as MAPPED only if it is in
`SUPPORTED_OPS`, which is the set the bare-metal C backend has actually been exercised
against, and `auto_fix` is populated only from the rewrite rules that
`tatva.repair` genuinely implements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tatva.compiler import SUPPORTED_OPS, TargetVariant

# Operator status vocabulary, kept as constants so the backend and the UI cannot drift
# into using different spellings of the same state.
STATUS_MAPPED = "MAPPED"
STATUS_UNMAPPED = "UNMAPPED"

# How the generated code treats an operator. This drives the badge colour in the studio
# and nothing else -- it is a description of the emitted C, not a quality judgement.
KIND_FUSED = "fused"      # replaced wholesale by a hand-written kernel
KIND_HOT = "hot"          # dominates the cycle count on this target
KIND_PLAIN = "plain"      # ordinary scalar C from TVM
KIND_BLOCKED = "blocked"  # no lowering at all


@dataclass
class OpCapability:
    """
    Everything TATVA knows about one operator on one target.

    `impact` and `fix` are deliberately separate from `status`: an operator can be
    mapped and still be the reason a model is slow, and an unmapped operator can be
    either repairable or a hard stop. Collapsing those into a single "supported"
    boolean is what made the old mapping table unactionable.
    """
    op: str
    status: str
    supported: bool
    kind: str
    lowering: str
    impact: str
    constraints: list[str] = field(default_factory=list)
    optimization: str = ""
    auto_fix_available: bool = False
    auto_fix_summary: str = ""
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- lowerings
#
# Keyed by the operator spelling `analyze_graph` reports (TVM's name with the "relax."
# prefix stripped). Each entry is (lowering, kind, impact, constraints, optimization).
#
# The wording describes what `runner.compile_model` emits today. When the softmax
# injection or the quantizer changes what it touches, this table changes with it --
# see `tatva.runner.inject_optimized_softmax` and `tatva.optimizer.quantize`.
_LOWERINGS: dict[str, tuple[str, str, str, list[str], str]] = {
    "nn.softmax": (
        "Single-pass fused kernel (Schraudolph fast exp) replaces TVM's three-pass reduction.",
        KIND_FUSED,
        "Dominant cost in attention blocks; the fusion pass targets exactly this operator.",
        ["float32 only", "reduction must be over the last axis"],
        "softmax fusion",
    ),
    "matmul": (
        # Relax has one matmul. Batched and fully-connected forms both land here -- there
        # is no separate nn.dense or nn.batch_matmul to describe.
        "Scalar C loop nest emitted by TVM Relax, one nest per batch index when the operands are batched.",
        KIND_HOT,
        "Usually the largest single share of the cycle count on a scalar target.",
        [],
        "INT8 quantization",
    ),
    "nn.layer_norm": (
        "Scalar C reduction over the feature axis, mean and variance in one pass each.",
        KIND_HOT,
        "Two full passes over the feature axis per call; second only to matmul on transformers.",
        [],
        "",
    ),
    "nn.relu": ("Scalar C max(x, 0) elementwise loop.", KIND_PLAIN, "Negligible.", [], ""),
    "tanh": ("Scalar C call to libm tanhf.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "sigmoid": ("Scalar C call to libm expf plus a reciprocal.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "erf": ("Scalar C call to libm erff.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "sqrt": ("Scalar C call to libm sqrtf.", KIND_PLAIN, "Cheap on targets with hardware FP.", [], ""),
    "power": ("Scalar C call to libm powf.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "add": ("Scalar C elementwise loop with broadcasting.", KIND_PLAIN, "Memory bound, negligible.", [], ""),
    "subtract": ("Scalar C elementwise loop with broadcasting.", KIND_PLAIN, "Memory bound, negligible.", [], ""),
    "multiply": ("Scalar C elementwise loop with broadcasting.", KIND_PLAIN, "Memory bound, negligible.", [], ""),
    "divide": ("Scalar C elementwise loop with broadcasting.", KIND_PLAIN, "Memory bound, negligible.", [], ""),
    "less": ("Scalar C elementwise comparison producing a bool tensor.", KIND_PLAIN, "Negligible.", [], ""),
    "where": ("Scalar C elementwise select.", KIND_PLAIN, "Negligible.", [], ""),
    "clip": ("Scalar C elementwise clamp.", KIND_PLAIN, "Negligible.", [], ""),
    "mean": ("Scalar C reduction loop.", KIND_PLAIN, "Proportional to the reduced axis.", [], ""),
    "sum": ("Scalar C reduction loop.", KIND_PLAIN, "Proportional to the reduced axis.", [], ""),
    "astype": ("Scalar C cast loop.", KIND_PLAIN, "Negligible.", [], ""),
    "full": ("Constant fill loop, or a static initializer when the value is known.", KIND_PLAIN, "Negligible.", [], ""),
    # Layout-only operators. TVM may still emit a copy, so they are not free, but they
    # never dominate.
    "reshape": ("Layout change only; TVM emits a copy when the buffer cannot be aliased.", KIND_PLAIN, "Copy cost at worst.", [], ""),
    "permute_dims": ("Index permutation with a strided copy.", KIND_PLAIN, "Copy cost, cache-unfriendly on large tensors.", [], ""),
    "squeeze": ("Layout change only.", KIND_PLAIN, "Negligible.", [], ""),
    "expand_dims": ("Layout change only.", KIND_PLAIN, "Negligible.", [], ""),
    "concat": ("Sequential copies into one output buffer.", KIND_PLAIN, "Copy cost.", [], ""),
    "split": ("Sequential copies out of one input buffer.", KIND_PLAIN, "Copy cost.", [], ""),
    "strided_slice": ("Strided copy loop.", KIND_PLAIN, "Copy cost.", [], ""),
    "take": ("Gather loop over the index tensor.", KIND_PLAIN, "Proportional to the index count.", [], ""),
    "shape_of": ("Resolved at compile time; the shape is a constant in the generated C.", KIND_PLAIN, "Free.", [], ""),
    "shape_to_tensor": ("Resolved at compile time; emitted as a constant tensor.", KIND_PLAIN, "Free.", [], ""),
    # Transcendental and rounding maths. All of these were missing from SUPPORTED_OPS
    # and so were reported as unsupported operators with a repair offered; they compile
    # and run unchanged. See the note on SUPPORTED_OPS in compiler.py.
    "exp": ("Scalar C call to libm expf.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "log": ("Scalar C call to libm logf.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "sin": ("Scalar C call to libm sinf.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "cos": ("Scalar C call to libm cosf.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "floor": ("Scalar C call to libm floorf.", KIND_PLAIN, "Negligible.", [], ""),
    "ceil": ("Scalar C call to libm ceilf.", KIND_PLAIN, "Negligible.", [], ""),
    "round": (
        "Scalar C call to libm roundf.",
        KIND_PLAIN,
        "Negligible.",
        ["rounds half away from zero, where ONNX Round rounds half to even"],
        "",
    ),
    "sign": ("Scalar C comparison chain.", KIND_PLAIN, "Negligible.", [], ""),
    "negative": ("Scalar C negation loop.", KIND_PLAIN, "Memory bound, negligible.", [], ""),
    "isnan": ("Scalar C self-comparison per element.", KIND_PLAIN, "Negligible.", [], ""),
    "mod": ("Scalar C call to libm fmodf.", KIND_PLAIN, "Negligible.", [], ""),
    # Comparisons and boolean logic. The result is an int8 tensor, so a model ending in
    # one of these prints 0.0/1.0 rather than a float.
    "equal": ("Scalar C elementwise comparison producing a bool tensor.", KIND_PLAIN, "Negligible.", [], ""),
    "greater": ("Scalar C elementwise comparison producing a bool tensor.", KIND_PLAIN, "Negligible.", [], ""),
    "logical_and": ("Scalar C elementwise bitwise-and over bool tensors.", KIND_PLAIN, "Negligible.", [], ""),
    "logical_not": ("Scalar C elementwise negation over a bool tensor.", KIND_PLAIN, "Negligible.", [], ""),
    "left_shift": ("Scalar C shift loop over integer tensors.", KIND_PLAIN, "Negligible.", [], ""),
    # Binary min/max, and the shape plumbing TVM emits to broadcast their operands.
    "max": ("Scalar C elementwise maximum.", KIND_PLAIN, "Memory bound, negligible.", [], ""),
    "min": ("Scalar C elementwise minimum.", KIND_PLAIN, "Memory bound, negligible.", [], ""),
    "stack": ("Sequential copies into one output buffer along a new axis.", KIND_PLAIN, "Copy cost.", [], ""),
    "broadcast_to": ("Copy loop that repeats the input along the broadcast axes.", KIND_PLAIN, "Copy cost.", [], ""),
    # Activations TVM lowers directly. `repair` also knows how to rewrite these into
    # simpler operators; that rewrite is now an optimization rather than a necessity.
    "nn.gelu": ("Scalar C erff-based closed form.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "nn.leakyrelu": ("Scalar C select against zero.", KIND_PLAIN, "Negligible.", [], ""),
    "nn.softplus": ("Scalar C log1pf/expf pair.", KIND_PLAIN, "Transcendental call per element.", [], ""),
    "nn.conv2d": (
        "Direct scalar C loop nest over (batch, out channel, y, x, in channel, ky, kx). No im2col, no Winograd.",
        KIND_HOT,
        "Dominates any convolutional model on a scalar target.",
        ["NCHW layout only"],
        "INT8 quantization",
    ),
    "nn.conv1d": (
        "Direct scalar C loop nest over (batch, out channel, x, in channel, kx). Six nested loops.",
        KIND_HOT,
        "Dominates 1-D convolutional models on a scalar target.",
        ["NCW layout only"],
        "INT8 quantization",
    ),
    "nn.conv3d": (
        "Direct scalar C loop nest over (batch, out channel, z, y, x, in channel, kz, ky, kx). Twelve nested loops.",
        KIND_HOT,
        "Volumetric convolution is the most expensive operator TATVA lowers; expect the bulk of the cycle count here.",
        ["NCDHW layout only"],
        "INT8 quantization",
    ),
    "nn.conv2d_transpose": (
        "Direct scalar C loop nest scattering each input element across the kernel window.",
        KIND_HOT,
        "Comparable to the forward convolution of the same shape.",
        ["NCHW layout only"],
        "",
    ),
    "nn.max_pool2d": (
        "Scalar C loop nest with a running maximum over the window.",
        KIND_PLAIN,
        "Proportional to output elements times window area.",
        ["NCHW layout only"],
        "",
    ),
    "nn.avg_pool2d": (
        "Scalar C loop nest with a running sum over the window, divided by the window area.",
        KIND_PLAIN,
        "Proportional to output elements times window area.",
        ["NCHW layout only"],
        "",
    ),
    "nn.prelu": ("Scalar C select against zero with a per-channel slope.", KIND_PLAIN, "Negligible.", [], ""),
    "nn.log_softmax": (
        "Scalar C max-shift, expf reduction and logf, in TVM's three passes over the axis.",
        KIND_PLAIN,
        "Transcendental call per element; the softmax fusion pass does not touch this form.",
        [],
        "",
    ),
    # Not operators a user writes. `variance` is what InstanceNormalization decomposes
    # to, `argmax`/`one_hot` what Hardmax does, `image.resize2d` is Resize and `triu` is
    # Trilu -- so each was reported unsupported under a name absent from the user's file.
    "variance": ("Scalar C two-pass reduction over the normalised axes.", KIND_PLAIN, "Proportional to the reduced axis.", [], ""),
    "argmax": ("Scalar C reduction tracking the running maximum's index.", KIND_PLAIN, "Proportional to the reduced axis.", [], ""),
    "one_hot": ("Constant fill followed by a scatter of the on-value.", KIND_PLAIN, "Negligible.", [], ""),
    "triu": ("Scalar C copy loop with a row/column predicate.", KIND_PLAIN, "Copy cost.", [], ""),
    "image.resize2d": (
        "Scalar C loop nest computing each output element's source coordinate.",
        KIND_PLAIN,
        "Proportional to output elements.",
        ["nearest and bilinear only"],
        "",
    ),
}

_DEFAULT_MAPPED = (
    "Generic scalar C emitted by TVM Relax.",
    KIND_PLAIN,
    "Not separately characterised on this target.",
    [],
    "",
)


def supported_ops_for(variant: TargetVariant) -> set[str]:
    """
    The operator set that has a lowering on `variant`.

    Support is currently uniform across the RISC-V variants: every target goes through
    the same TVM C backend and the same generated harness, so the set does not vary
    with `march`. The signature takes the target anyway because that is the honest
    shape of the question -- when a target-specific kernel does appear, it changes
    here and every caller already asks the right way.
    """
    return set(SUPPORTED_OPS)


def _vector_note(variant: TargetVariant) -> str:
    """
    The one place a target genuinely changes the answer.

    RV64GCV advertises the vector extension, but the C backend emits scalar loops and
    the only hand-written vector kernel is the softmax one. Saying "hot path" without
    this note on a "V" target implies vectorised code that is not there.
    """
    if "v" in variant.gcc_march.rsplit("c", 1)[-1] or variant.gcc_march.endswith("v"):
        return "Emitted as scalar C on this target; the vector unit is available but not targeted by codegen."
    return ""


def capability_for(op: str, variant: TargetVariant, *, count: int = 0) -> OpCapability:
    """
    Describe one operator on one target.

    `count` is carried through untouched so callers can build a mapping row without a
    second lookup; it does not affect any of the other fields.
    """
    from tatva.repair import repair_rule_for  # imported here: repair imports this module

    supported = op in supported_ops_for(variant)
    rule = repair_rule_for(op)

    if supported:
        lowering, kind, impact, constraints, optimization = _LOWERINGS.get(op, _DEFAULT_MAPPED)
        constraints = list(constraints)
        note = _vector_note(variant)
        if note and kind in (KIND_HOT, KIND_FUSED):
            constraints = [*constraints, note]
        return OpCapability(
            op=op,
            status=STATUS_MAPPED,
            supported=True,
            kind=kind,
            lowering=lowering,
            impact=impact,
            constraints=constraints,
            optimization=optimization,
            # A mapped operator is never auto-fixed. Rewriting something that already
            # compiles can only lose precision.
            auto_fix_available=False,
            auto_fix_summary="",
            reason="",
        )

    if rule is not None:
        return OpCapability(
            op=op,
            status=STATUS_UNMAPPED,
            supported=False,
            kind=KIND_BLOCKED,
            lowering="No direct lowering. A graph rewrite can express it with operators that do have one.",
            impact=f"Blocks code generation on {variant.name} until it is rewritten or removed from the model.",
            constraints=list(rule.constraints),
            optimization="",
            auto_fix_available=True,
            auto_fix_summary=rule.summary,
            reason=f"'{op}' has no kernel in TATVA's bare-metal operator set.",
        )

    return OpCapability(
        op=op,
        status=STATUS_UNMAPPED,
        supported=False,
        kind=KIND_BLOCKED,
        lowering="No lowering in TATVA's operator set -- compilation stops here.",
        impact=f"Blocks code generation on {variant.name}.",
        constraints=[],
        optimization="",
        auto_fix_available=False,
        auto_fix_summary="",
        reason=_unfixable_reason(op),
    )


# Why a particular unmapped operator has no automatic rewrite. Anything not named here
# gets the generic answer; inventing a specific-sounding reason for an operator nobody
# has looked at would be worse than admitting the general case.
# Nothing in here may name an operator that is in SUPPORTED_OPS -- an entry that does is
# a confident, specific explanation of why something cannot work, about something that
# demonstrably does. Five entries were exactly that (equal, exp, log, nn.conv2d and
# nn.max_pool2d) until the supported set was measured against the backend instead of
# recalled. `test_no_unfixable_reason_describes_a_supported_operator` is the guard.
#
# The three convolution and pooling entries that used to be here went the same way, and
# they were the ones that mattered: they told a user with a CNN, in specific and
# confident prose, that the operator at the centre of their model could never work. All
# three legalize to a real C loop nest and are now run end to end by the corpus.
#
# What stays here is measured the same way. An operator earns an entry by surviving
# `LegalizeOps` -- the relax op still standing after legalization, with no PrimFunc
# generated for it -- which is the actual condition under which no kernel exists.
_UNFIXABLE_REASONS: dict[str, str] = {
    "cumsum": (
        "TVM has no lowering rule for cumsum, so the operator survives legalization with no kernel "
        "generated for it, and there is nothing for the bare-metal harness to call. A prefix sum has "
        "no exact form in the supported set either: it is sequential by definition, and the "
        "elementwise and reduction operators available here cannot express the carry between elements."
    ),
}


def _unfixable_reason(op: str) -> str:
    return _UNFIXABLE_REASONS.get(
        op,
        f"'{op}' has no kernel in TATVA's bare-metal operator set and no rewrite rule that expresses it "
        f"exactly using the operators that do.",
    )


def capability_table(variant: TargetVariant) -> list[dict[str, Any]]:
    """
    The full supported-operator table for a target, for the capability browser in the
    studio. The UI renders whatever this returns; it holds no operator list of its own.
    """
    rows = [capability_for(op, variant).to_json() for op in sorted(supported_ops_for(variant))]
    # Sort hot paths and fused kernels to the top -- those are the rows an engineer
    # evaluating the target actually wants to read first.
    order = {KIND_FUSED: 0, KIND_HOT: 1, KIND_PLAIN: 2, KIND_BLOCKED: 3}
    rows.sort(key=lambda r: (order.get(r["kind"], 9), r["op"]))
    return rows


def repairable_ops() -> list[dict[str, Any]]:
    """Every operator the repair engine can currently rewrite, with its rule summary."""
    from tatva.repair import REPAIR_RULES

    return [
        {"op": op, "summary": rule.summary, "replacement_ops": list(rule.replacement_ops),
         "constraints": list(rule.constraints), "exact": rule.exact}
        for op, rule in sorted(REPAIR_RULES.items())
    ]
