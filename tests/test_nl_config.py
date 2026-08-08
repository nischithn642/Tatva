"""
Tests for the natural-language configuration layer.

The properties worth pinning down are not "does it understand English" -- it is a phrase
matcher and makes no such claim. They are:

  * it never invents a configuration the user could not have clicked,
  * it leaves untouched anything the sentence does not speak to,
  * it says so when it understood nothing, rather than quietly returning defaults,
  * an explicit instruction beats an inferred priority,
  * and every change it makes is accompanied by the phrase that caused it.
"""

from __future__ import annotations

import pytest

from tatva.compiler import TARGETS
from tatva.nl_config import (
    DEFAULT_TOLERANCE,
    STRICT_TOLERANCE,
    ConfigIntent,
    interpret,
    summarise,
)


class TestEmptyAndUnrecognised:
    def test_empty_string_changes_nothing(self):
        got = interpret("", target="RV32IMC", fuse=False, quantize=True)
        assert got.target == "RV32IMC"
        assert got.fuse is False
        assert got.quantize is True
        assert got.matched is False
        assert got.reasons == []

    @pytest.mark.parametrize("text", ["", "   ", "hello there", "please make it good"])
    def test_nothing_recognised_reports_unmatched(self, text):
        got = interpret(text, target="RV64GC", fuse=True, quantize=False)
        assert got.matched is False, f"{text!r} should not have matched anything"

    def test_unrecognised_preserves_caller_state_exactly(self):
        got = interpret("do the needful", target="RV64GCV", fuse=False, quantize=True)
        assert (got.target, got.fuse, got.quantize) == ("RV64GCV", False, True)

    def test_unknown_incoming_target_falls_back_to_a_real_one(self):
        got = interpret("", target="RV128Q")
        assert got.target in TARGETS


class TestPriorities:
    def test_size_turns_quantization_on(self):
        got = interpret("make the binary as small as possible")
        assert got.quantize is True
        assert got.matched is True

    def test_sram_constraint_reads_as_size(self):
        got = interpret("it has to fit in SRAM")
        assert got.quantize is True

    def test_speed_leaves_quantization_off(self):
        # INT8 is a measured regression on a scalar core; asking for speed must not
        # switch on the pass that makes it slower.
        got = interpret("I want the lowest latency", quantize=True)
        assert got.quantize is False
        assert got.fuse is True

    def test_accuracy_tightens_tolerance_and_disables_quantization(self):
        got = interpret("accuracy is what matters", quantize=True)
        assert got.quantize is False
        assert got.accuracy_tolerance == STRICT_TOLERANCE

    def test_default_tolerance_when_accuracy_not_mentioned(self):
        assert interpret("smallest binary").accuracy_tolerance == DEFAULT_TOLERANCE

    def test_conflicting_priorities_are_reported_not_hidden(self):
        got = interpret("smallest and fastest")
        assert got.conflicts, "a size/speed conflict should be surfaced"
        # Size wins, and the conflict text has to say that rather than leaving the user
        # to work out which one was applied.
        assert got.quantize is True
        assert "size was taken as the priority" in got.conflicts[0].lower()


class TestExplicitInstructions:
    def test_negation_beats_the_keyword_it_contains(self):
        # "no quantization" contains "quantization"; a naive matcher turns the pass on.
        got = interpret("no quantization")
        assert got.quantize is False

    @pytest.mark.parametrize(
        "text", ["no quantization", "without int8", "dont quantize", "fp32 only", "skip quantisation"]
    )
    def test_quantization_off_phrasings(self, text):
        assert interpret(text, quantize=True).quantize is False

    @pytest.mark.parametrize("text", ["use int8", "quantize it", "8-bit weights"])
    def test_quantization_on_phrasings(self, text):
        assert interpret(text).quantize is True

    @pytest.mark.parametrize("text", ["no fusion", "without fusion", "dont fuse", "skip fusion"])
    def test_fusion_off_phrasings(self, text):
        assert interpret(text, fuse=True).fuse is False

    def test_explicit_instruction_overrides_inferred_priority(self):
        # "smallest" infers quantization on; the explicit refusal has to win.
        got = interpret("smallest possible, but no quantization")
        assert got.quantize is False
        # And both steps are visible, so the user can see the override happened.
        assert len(got.reasons) >= 2

    def test_apostrophes_do_not_defeat_matching(self):
        assert interpret("don't fuse").fuse is False


class TestTargets:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("build for RV32IMC", "RV32IMC"),
            ("target rv64gc please", "RV64GC"),
            ("rv32imac", "RV32IMAC"),
            ("rv64imafdc", "RV64IMAFDC"),
            ("use the vector extension", "RV64GCV"),
            ("a 32-bit microcontroller", "RV32IMC"),
        ],
    )
    def test_target_aliases(self, text, expected):
        assert interpret(text).target == expected

    def test_specific_name_is_not_swallowed_by_the_generic_bucket(self):
        # "rv32imc" contains "rv32"; the specific target must win.
        assert interpret("rv32imc").target == "RV32IMC"

    def test_every_resolved_target_is_a_real_target(self):
        for text in ["rv32imc", "rv64gc", "vector", "32-bit", "embedded profile"]:
            assert interpret(text).target in TARGETS


class TestReasoningAndSummary:
    def test_every_change_carries_the_phrase_that_caused_it(self):
        got = interpret("smallest binary for rv32imc")
        assert got.reasons
        joined = " ".join(got.reasons).lower()
        assert "rv32imc" in joined
        assert "smallest" in joined

    def test_summary_names_target_passes_and_tolerance(self):
        s = summarise(interpret("lowest latency on rv64gc"))
        assert "RV64GC" in s
        assert "softmax fusion" in s
        assert "0.05" in s

    def test_summary_says_no_passes_rather_than_showing_an_empty_list(self):
        got = ConfigIntent(target="RV64GC", fuse=False, quantize=False)
        assert "no passes" in summarise(got)


class TestBridge:
    """The GUI bridge must not raise on anything a user can type into the box."""

    @pytest.mark.parametrize(
        "text",
        ["", "smallest", "rv32imc no quantization", "!!!", "🙂 make it fast", "a" * 5000],
    )
    def test_interpret_config_always_returns_a_result(self, text):
        from tatva.gui import TatvaPyBridge

        r = TatvaPyBridge().interpret_config(text, "RV64GC", True, False)
        assert r["success"] is True
        assert r["target"] in TARGETS
        assert isinstance(r["fuse"], bool)
        assert isinstance(r["quantize"], bool)
        assert isinstance(r["reasons"], list)
