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
        "Scalar C loop nest emitted by TVM Relax.",
        KIND_HOT,
        "Usually the largest single share of the cycle count on a scalar target.",
        [],
        "INT8 quantization",
    ),
    "nn.batch_matmul": (
        "Scalar C loop nest emitted by TVM Relax, one matmul per batch index.",
        KIND_HOT,
        "Usually the largest single share of the cycle count on a scalar target.",
        [],
        "INT8 quantization",
    ),
    "nn.dense": (
        "Scalar C loop nest emitted by TVM Relax.",
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
    "nn.bias_add": ("Scalar C add broadcast along the bias axis.", KIND_PLAIN, "Negligible.", [], ""),
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
    "cast": ("Scalar C cast loop.", KIND_PLAIN, "Negligible.", [], ""),
    "full": ("Constant fill loop, or a static initializer when the value is known.", KIND_PLAIN, "Negligible.", [], ""),
    # Layout-only operators. TVM may still emit a copy, so they are not free, but they
    # never dominate.
    "reshape": ("Layout change only; TVM emits a copy when the buffer cannot be aliased.", KIND_PLAIN, "Copy cost at worst.", [], ""),
    "transpose": ("Index permutation with a strided copy.", KIND_PLAIN, "Copy cost, cache-unfriendly on large tensors.", [], ""),
    "permute_dims": ("Index permutation with a strided copy.", KIND_PLAIN, "Copy cost, cache-unfriendly on large tensors.", [], ""),
    "squeeze": ("Layout change only.", KIND_PLAIN, "Negligible.", [], ""),
    "unsqueeze": ("Layout change only.", KIND_PLAIN, "Negligible.", [], ""),
    "expand_dims": ("Layout change only.", KIND_PLAIN, "Negligible.", [], ""),
    "concatenate": ("Sequential copies into one output buffer.", KIND_PLAIN, "Copy cost.", [], ""),
    "split": ("Sequential copies out of one input buffer.", KIND_PLAIN, "Copy cost.", [], ""),
    "strided_slice": ("Strided copy loop.", KIND_PLAIN, "Copy cost.", [], ""),
    "take": ("Gather loop over the index tensor.", KIND_PLAIN, "Proportional to the index count.", [], ""),
    "shape_of": ("Resolved at compile time; the shape is a constant in the generated C.", KIND_PLAIN, "Free.", [], ""),
    "shape_to_tensor": ("Resolved at compile time; emitted as a constant tensor.", KIND_PLAIN, "Free.", [], ""),
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
_UNFIXABLE_REASONS: dict[str, str] = {
    "equal": (
        "No exact decomposition into the supported set. A less/where chain reproduces it for ordinary "
        "values but disagrees on NaN, so rewriting it would silently change the model's meaning."
    ),
    "not_equal": (
        "No exact decomposition into the supported set, for the same NaN reason as 'equal'."
    ),
    "exp": "No exponential kernel in the bare-metal operator set, and no exact decomposition into the ones that exist.",
    "log": "No logarithm kernel in the bare-metal operator set, and no exact decomposition into the ones that exist.",
    "nn.conv1d": "Convolution has no kernel in the bare-metal operator set. Nothing in the supported set expresses it without a rewrite that changes cost by orders of magnitude.",
    "nn.conv2d": "Convolution has no kernel in the bare-metal operator set. Nothing in the supported set expresses it without a rewrite that changes cost by orders of magnitude.",
    "nn.conv3d": "Convolution has no kernel in the bare-metal operator set. Nothing in the supported set expresses it without a rewrite that changes cost by orders of magnitude.",
    "nn.max_pool2d": "Pooling has no kernel in the bare-metal operator set.",
    "nn.avg_pool2d": "Pooling has no kernel in the bare-metal operator set.",
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
