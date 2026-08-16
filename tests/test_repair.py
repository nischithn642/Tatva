"""
Tests for the graph repair engine.

The engine's whole claim is that it never applies a rewrite it has not proved. These
tests exercise that claim from both directions: every shipped rule is run against a real
Relax module and checked for an exact match on the host, and a deliberately wrong rule
is injected to confirm the engine throws its own output away rather than compiling it.

A rule with no test is a rule whose identity has only ever been asserted in a comment.
"""

import numpy as np
import pytest

tvm = pytest.importorskip("tvm")
relax = pytest.importorskip("tvm.relax")

from tatva.compiler import SUPPORTED_OPS, TARGETS, ModelIR, analyze_graph  # noqa: E402
from tatva.repair import REPAIR_RULES, REWRITE_TOLERANCE, repair_graph  # noqa: E402

VARIANT = TARGETS["RV64GC"]


def _module(fn, arity=1, shape=(2, 4), dtype="float32"):
    """A one-call Relax module: `main(x0, ...) -> fn(x0, ...)`."""
    bb = relax.BlockBuilder()
    params = [relax.Var(f"x{i}", relax.TensorStructInfo(shape, dtype)) for i in range(arity)]
    with bb.function("main", params):
        out = bb.emit(fn(*params))
        bb.emit_func_output(out)
    return bb.get()


def _ir(mod):
    return ModelIR(mod, {}, {"input_shapes": {}})


# One entry per rule in REPAIR_RULES. `test_every_rule_has_a_case` fails if a rule is
# added without one, so this table cannot quietly fall behind the engine.
#
# rsqrt is fed |x| because rsqrt of a negative is NaN on both sides of the comparison,
# and a test whose inputs make both graphs produce NaN proves nothing about the rewrite.
CASES: dict[str, tuple] = {
    "nn.silu": (lambda x: relax.op.nn.silu(x), 1),
    "nn.gelu_tanh": (lambda x: relax.op.nn.gelu_tanh(x), 1),
    "square": (lambda x: relax.op.square(x), 1),
    "abs": (lambda x: relax.op.abs(x), 1),
    "minimum": (lambda a, b: relax.op.minimum(a, b), 2),
    "maximum": (lambda a, b: relax.op.maximum(a, b), 2),
    "rsqrt": (lambda x: relax.op.rsqrt(relax.op.abs(x)), 1),
}


@pytest.mark.unit
def test_every_rule_has_a_case() -> None:
    """A new repair rule must arrive with a case in this file, not without one."""
    assert set(CASES) == set(REPAIR_RULES.keys())


@pytest.mark.unit
def test_no_rule_targets_an_operator_the_backend_already_lowers() -> None:
    """
    A rewrite is a repair only for an operator that cannot be lowered. For one that can,
    it is a pessimisation, and `_rewrite_module` applies the whole table as soon as
    anything in the graph is unsupported -- so a rule here for a supported operator does
    not sit dormant, it silently decomposes working code in every repair that runs.

    This is not hypothetical. SUPPORTED_OPS was hand-written and under-reported the
    backend by twenty-five operators; six of them had rules, and a model needing `abs`
    repaired came back with its `negative` and `nn.leakyrelu` rewritten too, reported as
    operators that had been fixed. Widening SUPPORTED_OPS is what creates the overlap, so
    the guard belongs here, where it fails the moment the two disagree.
    """
    overlap = sorted(set(REPAIR_RULES) & set(SUPPORTED_OPS))
    assert overlap == [], (
        f"{overlap} are in SUPPORTED_OPS and still carry a repair rule. If the backend "
        f"lowers them, drop the rule; if it does not, drop them from SUPPORTED_OPS."
    )


@pytest.mark.unit
@pytest.mark.parametrize("op", sorted(CASES))
def test_rule_repairs_and_matches_the_original_on_the_host(op) -> None:
    """
    Each rule turns its operator into supported ones and does not change the answer.

    Both graphs are built and run by TVM's host backend over the same random draws --
    the same check the engine performs before it accepts a rewrite, run here against a
    module built specifically to contain that one operator.
    """
    fn, arity = CASES[op]
    result = repair_graph(_ir(_module(fn, arity)), VARIANT)

    assert result.status == "REPAIRED", result.message
    assert result.applied is True
    assert op in result.repaired_ops
    assert result.structural_validation == "passed"
    assert result.numerical_validation == "passed", result.message
    assert result.max_abs_diff is not None
    assert result.max_abs_diff <= REWRITE_TOLERANCE
    assert result.remaining_unsupported == []

    # The record has to carry enough for someone to check the claim by hand.
    rec = next(r for r in result.records if r.original_op == op)
    assert rec.identity
    assert rec.replacement_ops
    assert rec.mapping_result == "MAPPED"
    assert rec.occurrences >= 1


@pytest.mark.unit
@pytest.mark.parametrize("op", sorted(CASES))
def test_repaired_graph_contains_only_supported_operators(op) -> None:
    """The point of the rewrite: nothing left in the graph blocks code generation."""
    fn, arity = CASES[op]
    result = repair_graph(_ir(_module(fn, arity)), VARIANT)
    report = analyze_graph(result.model_ir)
    assert report.unsupported_ops == []
    assert all(name in SUPPORTED_OPS for name in report.op_histogram)


@pytest.mark.unit
def test_clean_graph_is_left_alone() -> None:
    """Nothing to repair means nothing is touched, and the status says so."""
    mod = _module(lambda a, b: relax.op.add(a, b), 2)
    result = repair_graph(_ir(mod), VARIANT)

    assert result.status == "NO_REPAIR_NEEDED"
    assert result.attempted is False
    assert result.applied is False
    assert result.records == []
    assert result.model_ir.mod is mod


@pytest.mark.unit
def test_operator_with_no_rule_blocks_instead_of_being_guessed_at() -> None:
    """
    An unsupported operator TATVA has no proven identity for is reported, not invented.

    `nn.attention` has no lowering and no rewrite rule; the engine must return BLOCKED
    with the original graph, rather than producing something that merely compiles.
    """
    mod = _module(
        lambda q, k, v: relax.op.nn.attention(q, k, v),
        3, shape=(1, 4, 2, 8),
    )
    result = repair_graph(_ir(mod), VARIANT)

    assert result.status == "BLOCKED"
    assert result.applied is False
    assert result.model_ir.mod is mod
    assert "nn.attention" in result.unrepairable_ops
    assert "nn.attention" in result.remaining_unsupported


@pytest.mark.unit
def test_a_rewrite_that_changes_the_answer_is_discarded(monkeypatch) -> None:
    """
    The engine must throw away its own output when validation does not hold.

    A rule is swapped for one that is structurally fine -- `square` rewritten with
    supported operators -- but numerically wrong. Nothing about the resulting graph
    would stop it compiling, so only the numerical check can catch it. If the engine
    ever reported this as REPAIRED it would be shipping a silently different model.
    """
    from tatva import repair as repair_mod

    real = repair_mod._rule_table()
    broken = dict(real)
    broken["square"] = repair_mod.RepairRule(
        op="square",
        summary="Deliberately wrong rewrite used by the test suite.",
        replacement_ops=("multiply",),
        identity="square(x) = x * 3   (false)",
        exact=True,
        build=lambda call, args, dtype: relax.op.multiply(args[0], relax.const(3.0, dtype)),
    )
    monkeypatch.setattr(repair_mod, "_rule_table", lambda: broken)

    result = repair_graph(_ir(_module(lambda x: relax.op.square(x))), VARIANT)

    assert result.status == "DISCARDED"
    assert result.applied is False
    assert result.numerical_validation == "failed"
    assert result.max_abs_diff is not None and result.max_abs_diff > REWRITE_TOLERANCE
    assert all(r.mapping_result == "DISCARDED" for r in result.records)
    # The original graph is what the caller gets back, unchanged.
    assert analyze_graph(result.model_ir).unsupported_ops == ["square"]


@pytest.mark.unit
def test_a_rewrite_that_emits_an_unsupported_operator_is_discarded(monkeypatch) -> None:
    """
    Trading one unmapped operator for another is not a repair.

    This is a regression test: an early `broadcast_to` rule emitted `zeros`, which is
    also outside SUPPORTED_OPS. The rewrite ran, validated numerically, and left the
    build just as blocked as before.
    """
    from tatva import repair as repair_mod

    broken = dict(repair_mod._rule_table())
    broken["square"] = repair_mod.RepairRule(
        op="square",
        summary="Rewrite that swaps one unsupported operator for another.",
        replacement_ops=("erf",),
        identity="square(x) = square(x)  (still unsupported)",
        exact=True,
        # `nn.attention` is not in SUPPORTED_OPS either, so the structural scan must
        # notice the graph is no better off than it started.
        build=lambda call, args, dtype: relax.op.nn.attention(
            relax.op.reshape(args[0], (1, 2, 1, 4)),
            relax.op.reshape(args[0], (1, 2, 1, 4)),
            relax.op.reshape(args[0], (1, 2, 1, 4)),
        ),
    )
    monkeypatch.setattr(repair_mod, "_rule_table", lambda: broken)

    result = repair_graph(_ir(_module(lambda x: relax.op.square(x))), VARIANT)

    # Either the structural scan rejects it outright, or it is reported as PARTIAL with
    # the operator still named as unmapped. What must never happen is REPAIRED.
    assert result.status != "REPAIRED"
    if result.applied:
        assert result.remaining_unsupported


@pytest.mark.unit
def test_nan_divergence_is_not_reported_as_an_exact_match() -> None:
    """
    A rewrite that turns a finite output into NaN must fail the numerical check.

    `nan - nan` is `nan`, and Python's `max(0.0, nan)` returns 0.0 -- so subtracting
    first and taking the maximum afterwards recorded a NaN divergence as a maximum
    absolute difference of exactly zero, the same value an exact rewrite produces.
    """
    from tatva.repair import _numerical_check

    finite = _module(lambda x: relax.op.multiply(x, relax.const(1.0, "float32")))
    # sqrt of -|x| is NaN everywhere the input is non-zero.
    nan_producing = _module(
        lambda x: relax.op.sqrt(relax.op.multiply(relax.op.abs(x), relax.const(-1.0, "float32")))
    )

    status, detail, _worst = _numerical_check(finite, nan_producing, {})
    assert status == "failed", f"{status}: {detail}"
    assert "NaN" in detail or "infinite" in detail


@pytest.mark.unit
def test_numerical_check_accepts_two_graphs_that_agree() -> None:
    """The same check, run over a pair that genuinely matches, must pass."""
    from tatva.repair import _numerical_check

    a = _module(lambda x: relax.op.multiply(x, x))
    b = _module(lambda x: relax.op.square(x))
    status, detail, worst = _numerical_check(a, b, {})
    assert status == "passed", f"{status}: {detail}"
    assert worst is not None and worst <= REWRITE_TOLERANCE


@pytest.mark.unit
def test_repair_records_the_dtype_and_shapes_it_saw() -> None:
    """The audit row has to describe the actual call, not the rule in the abstract."""
    result = repair_graph(_ir(_module(lambda x: relax.op.nn.silu(x), 1, shape=(3, 5))), VARIANT)
    rec = result.records[0]
    assert rec.dtype == "float32"
    assert rec.input_shapes == [[3, 5]]
    assert rec.output_shape == [3, 5]
    assert rec.exact is True


@pytest.mark.unit
def test_repeated_operator_is_counted_not_duplicated() -> None:
    """Three silu calls are one record with a count of three, not three records."""
    def three(x):
        return relax.op.nn.silu(relax.op.nn.silu(relax.op.nn.silu(x)))

    result = repair_graph(_ir(_module(three)), VARIANT)
    assert result.repaired_ops == ["nn.silu"]
    assert len(result.records) == 1
    assert result.records[0].occurrences == 3


@pytest.mark.unit
def test_gelu_tanh_rewrite_matches_the_reference_definition() -> None:
    """
    Checked against the closed form written out in numpy rather than against TVM, so
    both sides of the engine's own comparison would have to be wrong in the same way for
    this to pass by accident.

    gelu_tanh is the most involved rewrite left in the table, and the only one whose
    identity is a formula rather than a rearrangement -- it is worth pinning to something
    outside the engine. (The same test used to cover the erf-form gelu; that rule is gone
    because the backend lowers nn.gelu directly, and the corpus runs it on RV64GC against
    onnxruntime, which is a stronger check than this one.)
    """
    result = repair_graph(_ir(_module(lambda x: relax.op.nn.gelu_tanh(x))), VARIANT)
    assert result.applied

    lowered = relax.transform.LegalizeOps()(result.model_ir.mod)
    vm = relax.VirtualMachine(relax.build(lowered, target="llvm"), tvm.cpu())

    x = np.random.default_rng(7).standard_normal((2, 4)).astype("float32")
    got = vm["main"](tvm.runtime.tensor(x)).numpy()
    expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    assert np.max(np.abs(got - expected)) <= 1e-6
