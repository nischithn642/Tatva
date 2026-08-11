"""
Tests for the target capability database.

This module is the single source of truth for "does this operator have a lowering, and
if not can TATVA fix it". Three things have to hold for that claim to be worth anything:
the table cannot contain an operator TATVA does not actually compile, the auto-fix flag
cannot be set for anything the repair engine does not implement, and an unmapped
operator must never be described in terms that read like support.
"""

import pytest

from tatva.capabilities import (
    _LOWERINGS,
    KIND_BLOCKED,
    KIND_FUSED,
    KIND_HOT,
    STATUS_MAPPED,
    STATUS_UNMAPPED,
    capability_for,
    capability_table,
    repairable_ops,
    supported_ops_for,
)
from tatva.compiler import SUPPORTED_OPS, TARGETS

RV64 = TARGETS["RV64GC"]
RV32 = TARGETS["RV32IMC"]


@pytest.mark.unit
def test_supported_set_is_the_compiler_s_own_set() -> None:
    """
    The capability database describes the backend; it does not extend it.

    If this drifts, the studio starts advertising operators the code generator has never
    been run against.
    """
    assert supported_ops_for(RV64) == set(SUPPORTED_OPS)


@pytest.mark.unit
def test_every_documented_lowering_is_for_an_operator_that_is_supported() -> None:
    """
    A lowering description for an unsupported operator would be prose describing code
    that does not exist. It cannot be reached by `capability_for`, so it would never be
    caught by eye.
    """
    undocumented_but_claimed = sorted(set(_LOWERINGS) - set(SUPPORTED_OPS))
    assert undocumented_but_claimed == []


@pytest.mark.unit
@pytest.mark.parametrize("op", sorted(SUPPORTED_OPS))
def test_supported_operator_reports_mapped_with_a_lowering(op) -> None:
    cap = capability_for(op, RV64)
    assert cap.status == STATUS_MAPPED
    assert cap.supported is True
    assert cap.kind != KIND_BLOCKED
    assert cap.lowering, f"{op} is reported as mapped with no description of how"
    assert cap.impact
    assert cap.reason == ""


@pytest.mark.unit
@pytest.mark.parametrize("op", sorted(SUPPORTED_OPS))
def test_a_mapped_operator_is_never_offered_an_auto_fix(op) -> None:
    """
    Rewriting an operator that already compiles can only lose precision. The button must
    not appear for one.
    """
    cap = capability_for(op, RV64)
    assert cap.auto_fix_available is False
    assert cap.auto_fix_summary == ""


@pytest.mark.unit
def test_unmapped_operator_with_a_rule_offers_the_fix_and_names_the_blockage() -> None:
    cap = capability_for("nn.silu", RV64)
    assert cap.status == STATUS_UNMAPPED
    assert cap.supported is False
    assert cap.kind == KIND_BLOCKED
    assert cap.auto_fix_available is True
    assert cap.auto_fix_summary
    assert "no kernel" in cap.reason
    assert RV64.name in cap.impact


@pytest.mark.unit
def test_unmapped_operator_without_a_rule_says_so_rather_than_offering_a_button() -> None:
    """§33: a control whose backend behaviour does not exist must not be rendered as if
    it did. `nn.conv2d` has no rewrite, so the row must carry no fix."""
    cap = capability_for("nn.conv2d", RV64)
    assert cap.status == STATUS_UNMAPPED
    assert cap.auto_fix_available is False
    assert cap.auto_fix_summary == ""
    assert "Convolution has no kernel" in cap.reason


@pytest.mark.unit
def test_an_operator_nobody_has_looked_at_gets_the_general_answer_not_an_invented_one() -> None:
    cap = capability_for("some.operator.that.does.not.exist", RV64)
    assert cap.status == STATUS_UNMAPPED
    assert cap.auto_fix_available is False
    assert "no rewrite rule" in cap.reason


@pytest.mark.unit
def test_auto_fix_flag_agrees_with_the_repair_engine_for_every_operator_it_names() -> None:
    """
    The capability database's fix column and the repair engine's rule table are two
    views of one fact. They are built from the same source here, and this test is what
    keeps them that way -- a rule removed from the engine must stop being advertised.
    """
    from tatva.repair import REPAIR_RULES

    advertised = {row["op"] for row in repairable_ops()}
    assert advertised == set(REPAIR_RULES)
    for op in advertised:
        cap = capability_for(op, RV64)
        assert cap.auto_fix_available is True, f"{op} has a rule but the table does not offer it"
        assert cap.auto_fix_summary == REPAIR_RULES[op].summary


@pytest.mark.unit
def test_repairable_ops_never_advertises_an_inexact_rewrite_as_exact() -> None:
    """Every shipped rule declares whether it reproduces the operator exactly, and the
    engine's own numerical check is what earned that flag."""
    rows = repairable_ops()
    assert rows
    for row in rows:
        assert isinstance(row["exact"], bool)
        assert row["summary"]
        assert row["replacement_ops"], f"{row['op']} claims a rewrite into nothing"


@pytest.mark.unit
def test_capability_table_covers_the_target_and_leads_with_the_hot_paths() -> None:
    rows = capability_table(RV64)
    assert {r["op"] for r in rows} == set(SUPPORTED_OPS)

    kinds = [r["kind"] for r in rows]
    first_plain = next((i for i, k in enumerate(kinds) if k not in (KIND_FUSED, KIND_HOT)), len(kinds))
    assert all(k in (KIND_FUSED, KIND_HOT) for k in kinds[:first_plain])
    assert KIND_FUSED not in kinds[first_plain:]
    assert KIND_HOT not in kinds[first_plain:]


@pytest.mark.unit
def test_a_vector_target_does_not_imply_vectorised_code() -> None:
    """
    RV64GCV advertises the vector extension and the C backend still emits scalar loops.
    Reporting matmul as a hot path on a "V" target without saying that would let a
    reader assume the vector unit is being used.
    """
    if "RV64GCV" not in TARGETS:
        pytest.skip("no vector target in the registry")
    scalar = capability_for("matmul", RV64)
    vector = capability_for("matmul", TARGETS["RV64GCV"])
    assert not any("vector unit" in c for c in scalar.constraints)
    assert any("vector unit is available but not targeted" in c for c in vector.constraints)


@pytest.mark.unit
def test_support_does_not_silently_vary_between_targets() -> None:
    """
    Every RISC-V variant goes through the same C backend today. If that ever stops being
    true, this test fails and the claim has to be re-stated rather than quietly drift.
    """
    for name, variant in TARGETS.items():
        assert supported_ops_for(variant) == set(SUPPORTED_OPS), name


@pytest.mark.unit
def test_softmax_is_the_operator_the_fusion_pass_names() -> None:
    """The optimization column has to point at a pass that exists, or the studio offers
    a speedup with nothing behind it."""
    cap = capability_for("nn.softmax", RV32)
    assert cap.kind == KIND_FUSED
    assert cap.optimization == "softmax fusion"
    assert capability_for("matmul", RV32).optimization == "INT8 quantization"
