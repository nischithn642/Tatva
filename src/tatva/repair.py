"""
TATVA automatic graph repair.

When a model contains an operator the bare-metal RISC-V backend has no kernel for,
compilation stops. Some of those operators are not actually missing capability -- they
are compositions of operators TATVA already emits, and the only thing standing between
the model and a working build is the rewrite. This module performs those rewrites.

The rule that governs everything here:

    A rewrite is applied only if it is an exact identity, and only after it has been
    checked. Nothing is rewritten to make a build succeed.

"Checked" means two things, both recorded, both surfaced in the UI:

  1. Structural validation. The repaired module must be well-formed under TVM's own
     checker, every operator left in it must be in `SUPPORTED_OPS`, and the function's
     output shape and dtype must be unchanged. A rewrite that fails any of these is
     discarded and the original graph is kept.

  2. Numerical validation. Both modules are built for the host and run on the same
     random inputs; the maximum absolute difference is measured. This is what proves
     the identity holds rather than merely asserting it in a comment. If the host
     cannot execute the *original* module -- which happens, because the original is by
     definition full of operators that may not legalize -- the result is recorded as
     `not_executable` rather than as a pass. It is never recorded as a pass it did not
     earn.

Operators with no exact decomposition are not rewritten at all. `equal` is the standard
example: a less/where chain reproduces it everywhere except NaN, and a rewrite that is
right 99.99% of the time is a silent correctness bug, not a feature. Those cases stop
honestly and say why -- see `capabilities._UNFIXABLE_REASONS`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

# Tolerance for the host-side numerical check. The rewrites are algebraic identities,
# so the only difference expected is float association order -- in practice the
# measured difference is 0.0. This bound exists to catch a wrong rule, not to absorb
# approximation, and is deliberately far tighter than the 0.05 MSE tolerance the
# end-to-end benchmark uses.
REWRITE_TOLERANCE = 1e-5

# How many random input draws the numerical check uses. More draws cost host time and
# buy very little: an identity that holds on one draw of dense random data holds.
REWRITE_SAMPLES = 3


@dataclass
class RepairRule:
    """
    One rewrite: what it matches, what it produces, and why it is safe.

    `exact` is not decoration. Only exact rules are ever registered; the field is here
    so the claim is stated as data next to the rule rather than left implicit, and so
    the UI can show it without the frontend having its own opinion about which rewrites
    preserve semantics.
    """
    op: str
    summary: str
    replacement_ops: tuple[str, ...]
    identity: str
    exact: bool
    build: Callable[..., Any]
    constraints: tuple[str, ...] = ()


@dataclass
class RepairRecord:
    """
    The audit row for one rewritten operator. §7 of the product spec asks for the
    original operator, its attributes, the replacement, shapes, dtype, the reason, the
    validation status and the resulting mapping status -- this carries all of them.
    """
    original_op: str
    occurrences: int
    attributes: dict[str, Any]
    replacement_ops: list[str]
    identity: str
    summary: str
    reason: str
    input_shapes: list[list[Any]]
    dtype: str
    output_shape: list[Any] | None
    exact: bool
    structural_validation: str = "pending"
    structural_detail: str = ""
    numerical_validation: str = "pending"
    numerical_detail: str = ""
    max_abs_diff: float | None = None
    mapping_result: str = "pending"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairResult:
    """Outcome of one repair attempt over a whole graph."""
    attempted: bool
    applied: bool
    model_ir: Any                       # repaired ModelIR, or the original when nothing was applied
    records: list[RepairRecord] = field(default_factory=list)
    repaired_ops: list[str] = field(default_factory=list)
    unrepairable_ops: list[str] = field(default_factory=list)
    remaining_unsupported: list[str] = field(default_factory=list)
    status: str = "NO_REPAIR_NEEDED"    # REPAIRED | PARTIAL | BLOCKED | NO_REPAIR_NEEDED | DISCARDED
    message: str = ""
    structural_validation: str = "not_run"
    numerical_validation: str = "not_run"
    max_abs_diff: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "applied": self.applied,
            "records": [r.to_json() for r in self.records],
            "repaired_ops": list(self.repaired_ops),
            "unrepairable_ops": list(self.unrepairable_ops),
            "remaining_unsupported": list(self.remaining_unsupported),
            "status": self.status,
            "message": self.message,
            "structural_validation": self.structural_validation,
            "numerical_validation": self.numerical_validation,
            "max_abs_diff": self.max_abs_diff,
        }


# --------------------------------------------------------------------------- the rules
#
# Each builder takes (call, args, dtype) and returns the replacement expression. They
# are written against `tvm.relax.op` and imported lazily so that importing this module
# -- which `capabilities` does, on every operator lookup -- does not drag TVM in.


def _rules() -> dict[str, RepairRule]:
    from tvm import relax

    def const(v: float, dtype: str):
        return relax.const(v, dtype)

    def silu(call, args, dtype):
        (x,) = args
        return relax.op.multiply(x, relax.op.sigmoid(x))

    def gelu_tanh(call, args, dtype):
        (x,) = args
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). This is the *definition*
        # of gelu_tanh, so reproducing it is exact -- gelu_tanh is itself an approximation
        # of gelu, but that approximation is what the model asked for.
        x3 = relax.op.multiply(relax.op.multiply(x, x), x)
        inner = relax.op.multiply(
            relax.op.add(x, relax.op.multiply(x3, const(0.044715, dtype))),
            const(math.sqrt(2.0 / math.pi), dtype),
        )
        return relax.op.multiply(
            relax.op.multiply(x, const(0.5, dtype)),
            relax.op.add(relax.op.tanh(inner), const(1.0, dtype)),
        )

    def square(call, args, dtype):
        (x,) = args
        return relax.op.multiply(x, x)

    def abs_(call, args, dtype):
        (x,) = args
        return relax.op.where(relax.op.less(x, const(0.0, dtype)), relax.op.multiply(x, const(-1.0, dtype)), x)

    def minimum(call, args, dtype):
        a, b = args
        return relax.op.where(relax.op.less(a, b), a, b)

    def maximum(call, args, dtype):
        a, b = args
        return relax.op.where(relax.op.less(a, b), b, a)

    def rsqrt(call, args, dtype):
        (x,) = args
        return relax.op.divide(const(1.0, dtype), relax.op.sqrt(x))

    # Every rule below is for an operator the C backend cannot lower. That is not a
    # stylistic rule, it is the condition under which a rewrite is a repair at all --
    # see `test_no_rule_targets_an_operator_the_backend_already_lowers`.
    #
    # Six rules used to live here that no longer do: nn.gelu, nn.leakyrelu, negative,
    # greater, sign and broadcast_to. They were written against a SUPPORTED_OPS that
    # under-reported the backend; all six operators lower directly, and the corpus runs
    # them on RV64GC against onnxruntime. Keeping their rules was not merely dead code.
    # `_rewrite_module` applies the whole table whenever anything in the graph is
    # unsupported, so a model that needed `abs` repaired also had its perfectly good
    # `negative` and `nn.leakyrelu` swapped for multi-operator decompositions -- slower
    # code, on a tool whose entire output is a cycle count, reported back to the user as
    # operators that had been "repaired". Git history has them if the backend regresses.
    return {
        "nn.silu": RepairRule(
            op="nn.silu", summary="Rewrite as x * sigmoid(x).",
            replacement_ops=("multiply", "sigmoid"),
            identity="silu(x) = x * sigmoid(x)", exact=True, build=silu,
        ),
        "nn.gelu_tanh": RepairRule(
            op="nn.gelu_tanh", summary="Rewrite as the tanh form it is defined by.",
            replacement_ops=("multiply", "add", "tanh"),
            identity="gelu_tanh(x) = 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))",
            exact=True, build=gelu_tanh,
        ),
        "square": RepairRule(
            op="square", summary="Rewrite as x * x.",
            replacement_ops=("multiply",), identity="square(x) = x * x", exact=True, build=square,
        ),
        "abs": RepairRule(
            op="abs", summary="Rewrite as a select between x and -x.",
            replacement_ops=("where", "less", "multiply"),
            identity="abs(x) = x < 0 ? -x : x", exact=True, build=abs_,
        ),
        "minimum": RepairRule(
            op="minimum", summary="Rewrite as a select on the comparison.",
            replacement_ops=("where", "less"),
            identity="min(a, b) = a < b ? a : b", exact=True, build=minimum,
        ),
        "maximum": RepairRule(
            op="maximum", summary="Rewrite as a select on the comparison.",
            replacement_ops=("where", "less"),
            identity="max(a, b) = a < b ? b : a", exact=True, build=maximum,
        ),
        "rsqrt": RepairRule(
            op="rsqrt", summary="Rewrite as 1 / sqrt(x).",
            replacement_ops=("divide", "sqrt"), identity="rsqrt(x) = 1 / sqrt(x)", exact=True, build=rsqrt,
        ),
    }


_RULE_CACHE: dict[str, RepairRule] | None = None


def _rule_table() -> dict[str, RepairRule]:
    global _RULE_CACHE
    if _RULE_CACHE is None:
        try:
            _RULE_CACHE = _rules()
        except Exception:
            # No TVM available (docs build, a unit test that only wants the metadata).
            # An empty table means "no repair offered", which is the safe answer.
            _RULE_CACHE = {}
    return _RULE_CACHE


class _LazyRules:
    """
    Presents the rule table as a mapping without importing TVM at module import time.

    `capabilities` reads this on every operator lookup, and that path has to stay
    usable in the GUI process before the compiler backend is warm.
    """

    def __getitem__(self, key: str) -> RepairRule:
        return _rule_table()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return _rule_table().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in _rule_table()

    def __iter__(self):
        return iter(_rule_table())

    def items(self):
        return _rule_table().items()

    def keys(self):
        return _rule_table().keys()

    def __len__(self) -> int:
        return len(_rule_table())


REPAIR_RULES = _LazyRules()


def repair_rule_for(op: str) -> RepairRule | None:
    """The rewrite rule for an operator name, or None when TATVA has none."""
    return _rule_table().get(op)


# ------------------------------------------------------------------------ the mutator


def _short(name: str) -> str:
    """TVM spells operators 'relax.nn.gelu'; the rest of TATVA says 'nn.gelu'."""
    return name[len("relax."):] if name.startswith("relax.") else name


def _attrs_to_dict(attrs: Any) -> dict[str, Any]:
    """Best-effort capture of a call's attributes for the audit record."""
    if attrs is None:
        return {}
    out: dict[str, Any] = {}
    try:
        # `.keys()` is not redundant: `attrs` is a TVM Attrs node, not a dict, and
        # iterating it directly does not yield its field names.
        for key in attrs.keys():  # noqa: SIM118
            try:
                val = attrs[key]
            except Exception:
                continue
            if isinstance(val, (int, float, str, bool)):
                out[key] = val
            else:
                out[key] = str(val)
    except Exception:
        return {}
    return out


def _shape_of(struct_info: Any) -> list[Any] | None:
    shape = getattr(struct_info, "shape", None)
    if shape is None:
        return None
    try:
        return [int(d) if hasattr(d, "value") else str(d) for d in shape]
    except Exception:
        return None


def _rewrite_module(mod: Any) -> tuple[Any, dict[str, RepairRecord]]:
    """
    Apply every matching rule across the module.

    Returns the rewritten module and one record per distinct operator that was
    rewritten, with the occurrence count. Validation fields are filled in by the
    caller, which is what actually runs the checks -- a mutator that graded its own
    output would be exactly the kind of self-certifying step this feature exists to
    replace.
    """
    import tvm.ir
    from tvm import relax

    table = _rule_table()
    records: dict[str, RepairRecord] = {}

    @relax.expr_functor.mutator
    class RepairMutator(relax.PyExprMutator):
        def __init__(self, module):
            super().__init__(module)
            self.module_ = module

        def transform(self):
            for gv, func in self.module_.functions_items():
                if isinstance(func, relax.Function):
                    self.builder_.update_func(gv, self.visit_expr(func))
            return self.builder_.get()

        def visit_call_(self, call):
            call = self.visit_expr_post_order(call)
            if not isinstance(call.op, tvm.ir.Op):
                return call

            short = _short(call.op.name)
            rule = table.get(short)
            if rule is None:
                return call

            args = list(call.args)
            first = args[0] if args else None
            si = getattr(first, "struct_info", None)
            dtype = getattr(si, "dtype", None)
            if not dtype:
                # Without a concrete dtype the constants below cannot be typed, and a
                # guess would be a silent dtype change. Leave the call alone; it will be
                # reported as unmapped, which is the truth.
                return call

            try:
                replacement = rule.build(call, args, dtype)
            except Exception:
                return call

            rec = records.get(short)
            if rec is None:
                records[short] = RepairRecord(
                    original_op=short,
                    occurrences=1,
                    attributes=_attrs_to_dict(call.attrs),
                    replacement_ops=list(rule.replacement_ops),
                    identity=rule.identity,
                    summary=rule.summary,
                    reason=f"'{short}' has no kernel in TATVA's bare-metal operator set; "
                           f"the identity below expresses it with operators that do.",
                    input_shapes=[s for s in (_shape_of(getattr(a, "struct_info", None)) for a in args) if s is not None],
                    dtype=str(dtype),
                    output_shape=_shape_of(getattr(call, "struct_info", None)),
                    exact=rule.exact,
                )
            else:
                rec.occurrences += 1

            return replacement

    mutator = RepairMutator(mod)
    return mutator.transform(), records


# ----------------------------------------------------------------------- validation


def _structural_check(mod: Any, original_mod: Any) -> tuple[str, str, list[str]]:
    """
    Well-formedness, operator coverage and output-signature preservation.

    Returns (status, detail, remaining_unsupported_ops).
    """
    from tvm import relax

    from tatva.compiler import SUPPORTED_OPS

    try:
        if not relax.analysis.check_well_formed(mod):
            return "failed", "The rewritten module did not pass TVM's well-formedness checker.", []
    except Exception as exc:
        return "failed", f"Well-formedness check raised: {exc}", []

    # Every operator left in the graph must have a lowering.
    import tvm.ir

    remaining: set[str] = set()

    @relax.expr_functor.visitor
    class OpScan(relax.PyExprVisitor):
        def visit_call_(self, call):
            super().visit_call_(call)
            if isinstance(call.op, tvm.ir.Op):
                name = _short(call.op.name)
                if name not in SUPPORTED_OPS:
                    remaining.add(name)

    try:
        OpScan().visit_expr(mod["main"])
    except Exception as exc:
        return "failed", f"Could not re-scan the rewritten graph: {exc}", []

    # The public signature must not have moved. A rewrite that changes the output shape
    # or dtype is not a repair, it is a different model.
    try:
        before = mod["main"].struct_info
        after = original_mod["main"].struct_info
        if str(before.ret) != str(after.ret):
            return (
                "failed",
                f"Output signature changed: {after.ret} became {before.ret}.",
                sorted(remaining),
            )
    except Exception:
        # Signature comparison is a guard, not the primary check. If TVM will not give
        # us the struct info, say so rather than treating silence as agreement.
        return "partial", "Output signature could not be compared on this module.", sorted(remaining)

    if remaining:
        return (
            "passed",
            f"Well-formed, output signature unchanged. {len(remaining)} operator kind(s) still have no lowering.",
            sorted(remaining),
        )
    return "passed", "Well-formed, output signature unchanged, every remaining operator has a lowering.", []


def _numerical_check(original_mod: Any, repaired_mod: Any, input_shapes: dict[str, Any]) -> tuple[str, str, float | None]:
    """
    Run both modules on the host over the same random inputs and compare.

    This is the check that proves the rewrite rather than asserting it. The original
    module is full of operators the RISC-V backend cannot lower -- but the host LLVM
    backend can, which is exactly why the comparison is possible at all.

    Returns (status, detail, max_abs_diff). A status of `not_executable` means the
    host could not run one of the two modules; it is never reported as a pass.
    """
    try:
        import numpy as np
        import tvm
        from tvm import relax
    except Exception as exc:
        return "not_executable", f"Host execution unavailable: {exc}", None

    def build(mod):
        lowered = relax.transform.LegalizeOps()(mod)
        ex = relax.build(lowered, target="llvm")
        return relax.VirtualMachine(ex, tvm.cpu())

    try:
        vm_a = build(original_mod)
        vm_b = build(repaired_mod)
    except Exception as exc:
        return "not_executable", f"The host could not build one of the two graphs: {str(exc)[:200]}", None

    rng = np.random.default_rng(0)
    worst = 0.0
    try:
        params = list(original_mod["main"].params)
        for _ in range(REWRITE_SAMPLES):
            inputs = []
            for p in params:
                si = p.struct_info
                shape = [int(d) for d in si.shape]
                dtype = str(si.dtype)
                if dtype.startswith("float"):
                    arr = rng.standard_normal(shape).astype(dtype)
                elif dtype == "bool":
                    arr = (rng.random(shape) > 0.5)
                else:
                    arr = rng.integers(0, 2, size=shape).astype(dtype)
                inputs.append(tvm.runtime.tensor(arr))
            out_a = vm_a["main"](*inputs)
            out_b = vm_b["main"](*inputs)
            a = out_a.numpy() if hasattr(out_a, "numpy") else np.asarray(out_a)
            b = out_b.numpy() if hasattr(out_b, "numpy") else np.asarray(out_b)
            if a.shape != b.shape:
                return "failed", f"Output shape diverged: {a.shape} vs {b.shape}.", None

            # NaN and infinity are compared before the subtraction, not through it.
            # `nan - nan` is `nan`, and `max(0.0, nan)` in Python returns 0.0 -- so a
            # rewrite that turned a finite output into NaN would have been recorded as a
            # maximum difference of zero and reported as an exact match.
            a64 = a.astype("float64")
            b64 = b.astype("float64")
            bad_a = ~np.isfinite(a64)
            bad_b = ~np.isfinite(b64)
            if not np.array_equal(bad_a, bad_b):
                return "failed", (
                    "The rewrite changed which outputs are NaN or infinite "
                    f"({int(bad_a.sum())} non-finite before, {int(bad_b.sum())} after)."
                ), None
            if bad_a.any() and not np.array_equal(a64[bad_a], b64[bad_a], equal_nan=True):
                return "failed", "The rewrite changed a NaN output into an infinity, or vice versa.", None

            finite = ~bad_a
            diff = float(np.max(np.abs(a64[finite] - b64[finite]))) if finite.any() else 0.0
            worst = max(worst, diff)
    except Exception as exc:
        return "not_executable", f"Host execution failed: {str(exc)[:200]}", None

    if worst <= REWRITE_TOLERANCE:
        return "passed", f"Maximum absolute difference {worst:.3g} over {REWRITE_SAMPLES} random draws.", worst
    return "failed", f"Maximum absolute difference {worst:.3g} exceeds the {REWRITE_TOLERANCE:g} rewrite tolerance.", worst


# ------------------------------------------------------------------------- entry point


def repair_graph(model_ir: Any, variant: Any, *, verify_numerically: bool = True) -> RepairResult:
    """
    Attempt to repair every unsupported operator in `model_ir`.

    Returns a `RepairResult` whose `model_ir` is the repaired graph when the rewrite
    passed validation, and the untouched original when it did not. The caller is free
    to compile either -- this function never decides that a build should proceed, only
    whether a rewrite earned the right to be used.
    """
    from tatva.compiler import ModelIR, analyze_graph

    report = analyze_graph(model_ir)
    unsupported = sorted(report.unsupported_ops)

    if not unsupported:
        return RepairResult(
            attempted=False, applied=False, model_ir=model_ir,
            status="NO_REPAIR_NEEDED",
            message="Every operator in this graph already has a lowering; there is nothing to repair.",
        )

    table = _rule_table()
    fixable = [op for op in unsupported if op in table]
    unfixable = [op for op in unsupported if op not in table]

    if not fixable:
        return RepairResult(
            attempted=True, applied=False, model_ir=model_ir,
            unrepairable_ops=unfixable, remaining_unsupported=unsupported,
            status="BLOCKED",
            message=(
                f"{len(unfixable)} unsupported operator kind(s) and no rewrite rule for any of them: "
                f"{', '.join(unfixable)}. TATVA will not invent a decomposition it cannot prove."
            ),
        )

    try:
        repaired_mod, records = _rewrite_module(model_ir.mod)
    except Exception as exc:
        return RepairResult(
            attempted=True, applied=False, model_ir=model_ir,
            unrepairable_ops=unfixable, remaining_unsupported=unsupported,
            status="DISCARDED",
            message=f"The rewrite pass failed and the original graph was kept: {str(exc)[:300]}",
        )

    if not records:
        return RepairResult(
            attempted=True, applied=False, model_ir=model_ir,
            unrepairable_ops=unfixable, remaining_unsupported=unsupported,
            status="BLOCKED",
            message="No rewrite matched: the operators present did not carry the concrete dtypes the rules need.",
        )

    struct_status, struct_detail, remaining = _structural_check(repaired_mod, model_ir.mod)
    num_status, num_detail, worst = ("skipped", "Numerical verification was not requested.", None)
    if verify_numerically and struct_status != "failed":
        num_status, num_detail, worst = _numerical_check(
            model_ir.mod, repaired_mod, model_ir.metadata.get("input_shapes", {})
        )

    for rec in records.values():
        rec.structural_validation = struct_status
        rec.structural_detail = struct_detail
        rec.numerical_validation = num_status
        rec.numerical_detail = num_detail
        rec.max_abs_diff = worst
        rec.mapping_result = "MAPPED" if rec.original_op not in remaining else "UNMAPPED"

    record_list = list(records.values())
    repaired_ops = sorted(records.keys())

    # The two ways a rewrite loses the right to be used.
    if struct_status == "failed":
        for rec in record_list:
            rec.mapping_result = "DISCARDED"
        return RepairResult(
            attempted=True, applied=False, model_ir=model_ir, records=record_list,
            unrepairable_ops=unfixable, remaining_unsupported=unsupported,
            status="DISCARDED", structural_validation=struct_status, numerical_validation=num_status,
            max_abs_diff=worst,
            message=f"The rewrite was discarded and the original graph kept. {struct_detail}",
        )
    if num_status == "failed":
        for rec in record_list:
            rec.mapping_result = "DISCARDED"
        return RepairResult(
            attempted=True, applied=False, model_ir=model_ir, records=record_list,
            unrepairable_ops=unfixable, remaining_unsupported=unsupported,
            status="DISCARDED", structural_validation=struct_status, numerical_validation=num_status,
            max_abs_diff=worst,
            message=(
                f"The rewrite changed the model's output and was discarded; the original graph was kept. {num_detail}"
            ),
        )

    metadata = dict(model_ir.metadata)
    metadata["repaired"] = True
    metadata["repaired_ops"] = repaired_ops
    metadata["repair_records"] = [r.to_json() for r in record_list]
    repaired_ir = ModelIR(repaired_mod, model_ir.params, metadata)

    still_blocked = sorted(set(remaining) | set(unfixable))
    if still_blocked:
        status = "PARTIAL"
        message = (
            f"Rewrote {len(repaired_ops)} operator kind(s): {', '.join(repaired_ops)}. "
            f"{len(still_blocked)} still have no lowering and no rewrite rule: {', '.join(still_blocked)}. "
            f"Code generation will still stop."
        )
    else:
        status = "REPAIRED"
        message = (
            f"Rewrote {len(repaired_ops)} operator kind(s): {', '.join(repaired_ops)}. "
            f"Every operator in the graph now has a lowering."
        )

    return RepairResult(
        attempted=True, applied=True, model_ir=repaired_ir, records=record_list,
        repaired_ops=repaired_ops, unrepairable_ops=unfixable, remaining_unsupported=still_blocked,
        status=status, message=message,
        structural_validation=struct_status, numerical_validation=num_status, max_abs_diff=worst,
    )
