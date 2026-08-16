"""
The standing answer to "can TATVA be pointed at an arbitrary ONNX file?"

Every other test in this suite runs a model that was chosen because it works. This one
sweeps the whole synthetic corpus in tests/onnx_corpus.py -- 88 small graphs covering
every operator TATVA claims plus a deliberate handful it does not -- and holds it to two
things, one for each half of that split.

The corpus is partitioned by NOT_RUNNABLE below. Eight models are listed there, with the
reason each gives; RUNNABLE is the other 80, defined as the rest rather than enumerated,
so a model added to the corpus is claimed to work until someone says otherwise.

For RUNNABLE, the bar is end to end and numerical, not "did not raise":

    the model cross-compiles for RV64GC, runs under QEMU, and its first logits match
    onnxruntime to 1e-4.

For NOT_RUNNABLE, the bar is that the refusal is usable:

    the pipeline reaches a typed verdict, `classify_failure` does not file it under
    'unknown', and the message names the operator or the limitation.

Both halves were failing, in ways that only a sweep like this exposes.

Unsupported models were unactionable: the regex that read TVM's rejection message only
matched the fabricated operator in models/model_unsupported.onnx, so every genuinely
unsupported model reported 'unknown', and a BLOCKED repair could come back with an empty
list of blocking operators.

Supported models were worse, because they looked fine. SUPPORTED_OPS was a hand-written
list that had never been checked against the backend, and it under-reported it by
thirty-six operators -- including nn.conv2d and nn.max_pool2d, then conv at one and three
dimensions, the transposed form, average pooling, PRelu, LogSoftmax and Resize. Every
convolutional model was told its operators were unsupported and offered a repair, by a
toolchain that cross-compiles and runs it correctly. (An earlier instance of the same
mistake: the list carried the Relay spelling `concatenate` against a Relax backend that
emits `concat`.) `test_supported_ops_all_exist_in_relax` catches a name that is not real;
`test_no_runnable_model_is_reported_unsupported` catches a backend capability the list
has forgotten to claim; and `test_model_compiles_runs_and_matches_the_host` is what makes
either claim worth anything, by re-deriving the whole list from models that actually ran.

That last test is also the only thing standing between a user and a measured wrong
answer, and it earned its keep twice. The first time, `gv = lv` bindings meant LayerNorm,
Identity and a constant-folded Gather never wrote their output buffer. The second was
CumSum: TVM has no lowering rule for `relax.cumsum`, so the operator survived
legalization, the harness emitter skipped the binding it could not turn into a call, and
the model compiled, booted, reported a cycle count and returned a tensor of 0.0. Neither
was reachable from any test that only asks whether the pipeline raised.

The end-to-end sweep needs the cross-compiler and QEMU, so it is marked `integration`.
Everything else here is `unit` and runs in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tatva.compiler import SUPPORTED_OPS, unsupported_operator_names
from tatva.diagnostics import (
    CompilationError,
    ImportInProgressError,
    MemoryLimitExceededError,
    SimulationTimeoutError,
    UnsupportedOperatorError,
    classify_failure,
)
from tests.onnx_corpus import CASE_NAMES, build_corpus

# The exceptions TATVA is allowed to raise from the front of the pipeline. Anything
# outside this set is, by definition, an unhandled error reaching the user.
DIAGNOSED_ERRORS = (
    UnsupportedOperatorError,
    CompilationError,
    MemoryLimitExceededError,
    SimulationTimeoutError,
    ImportInProgressError,
)

REPAIR_STATUSES = {"REPAIRED", "PARTIAL", "BLOCKED", "NO_REPAIR_NEEDED", "DISCARDED"}

# The models TATVA cannot run, and the reason each one gives. Everything else in the
# corpus must cross-compile, execute under QEMU and agree with onnxruntime -- see
# RUNNABLE below, which is simply the rest.
#
# Stating the exceptions rather than the successes is deliberate. A list of things that
# work grows stale in the safe direction: it silently stops covering whatever was added
# since. A list of things that do not work fails the moment a new model joins the corpus
# without being run, which is when someone should look.
#
# The value is a substring the diagnosis must contain. A limitation that cannot say what
# it is, in the user's terms, is indistinguishable from a crash.
NOT_RUNNABLE = {
    # TVM's ONNX frontend has no converter for these three, so they never import.
    "op_dropout": "Dropout",
    "op_lstm": "LSTM",
    "op_fabricated": "UnsupportedOpXYZ",
    # relax.abs legalizes to tirx.fabs, which the C backend does not implement. `repair`
    # rewrites it into less/multiply/where and that does run -- see
    # test_a_repaired_graph_compiles_and_runs.
    "unary_Abs": "could not lower",
    # Operators returning several tensors: the harness allocates one output buffer.
    "op_batchnorm": "tuple",
    "struct_multi_io": "tuple",
    # A data-dependent output shape, which a statically allocated build cannot size.
    "op_nonzero": "symbolic dimension",
    # LegalizeOps has no rule for relax.cumsum, so the op survives legalization and no
    # kernel is ever generated for it. This is the one that proves the entry above it is
    # worth its runtime: CumSum compiled, booted under QEMU, reported a cycle count and
    # returned a tensor of 0.0, because the harness emitter skipped the binding it could
    # not turn into a call. It now refuses -- see the raise in runner's harness emitter.
    "op_cumsum": "cumsum",
}

# Every other model in the corpus. The claim is the strong one -- compiles for RV64GC,
# runs to completion under QEMU, and matches the host to 1e-4 -- and it is checked
# that way by test_model_compiles_runs_and_matches_the_host.
RUNNABLE = sorted(set(CASE_NAMES) - set(NOT_RUNNABLE))


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> dict[str, Path]:
    """
    Build the corpus once for the whole session.

    Session scope on purpose: the models are pure functions of onnx_corpus.py, so
    rebuilding them per test would cost ~70 file writes per case for no isolation
    benefit. The IR cache, which is the thing that genuinely leaks between tests, is
    cleared per test by the autouse fixture in conftest.
    """
    return build_corpus(tmp_path_factory.mktemp("onnx_corpus"))


def _front_of_pipeline(path: Path) -> dict:
    """
    Push one model through import -> analyze -> repair and describe what happened.

    Deliberately catches BaseException. The question this file exists to answer is what
    an unknown model does to the pipeline, and "it raised something that is not an
    Exception" is an answer the assertions need to see rather than a reason to error
    out of the test with no diagnosis attached.
    """
    from tatva.compiler import TARGETS, analyze_graph, import_model
    from tatva.repair import repair_graph

    try:
        ir = import_model(str(path))
    except BaseException as e:
        return {"stage": "import", "error": e}

    try:
        report = analyze_graph(ir)
    except BaseException as e:
        return {"stage": "analyze", "error": e}

    rec = {
        "stage": "analyze",
        "error": None,
        "ops": sorted(report.op_histogram),
        "unsupported": sorted(report.unsupported_ops),
    }
    if not report.unsupported_ops:
        rec["verdict"] = "CLEAN"
        return rec

    try:
        result = repair_graph(ir, TARGETS["RV64GC"], verify_numerically=True)
    except BaseException as e:
        return {**rec, "stage": "repair", "error": e}

    rec["stage"] = "repair"
    rec["verdict"] = result.status
    rec["result"] = result
    return rec


@pytest.mark.unit
@pytest.mark.parametrize("case", CASE_NAMES)
def test_every_model_reaches_a_diagnosed_verdict(case: str, corpus) -> None:
    """
    No model in the corpus may take the pipeline down or stop it somewhere undiagnosed.

    Either the graph gets a verdict from `repair_graph`, or it raises one of TATVA's own
    diagnostic errors -- and in the second case `classify_failure` must recognise it,
    because the studio's stage-03 panel renders from that classification and its
    'unknown' branch is a bare stringified exception.
    """
    pytest.importorskip("tvm")
    rec = _front_of_pipeline(corpus[case])
    err = rec["error"]

    if err is None:
        assert rec["verdict"] in REPAIR_STATUSES | {"CLEAN"}, (
            f"{case}: {rec['stage']} produced an unrecognised verdict {rec['verdict']!r}"
        )
        return

    assert isinstance(err, DIAGNOSED_ERRORS), (
        f"{case}: {rec['stage']} raised an undiagnosed {type(err).__name__}: {err}"
    )
    diagnosis = classify_failure(err)
    assert diagnosis.error_type != "unknown", (
        f"{case}: {type(err).__name__} classified as 'unknown' -- the user is shown a raw "
        f"Python message: {err}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("case", CASE_NAMES)
def test_a_refusal_always_names_the_operator(case: str, corpus) -> None:
    """
    Whichever way TATVA says no, it has to say which operator it is saying no about.

    Rahul's report was that the mapping table shows operators as unsupported and offers
    a rewrite that does nothing -- which is what "unsupported operator: 'unknown'" and
    an empty `unrepairable_ops` list look like from the studio. An unnamed blocker is
    not something a user can act on.
    """
    pytest.importorskip("tvm")
    rec = _front_of_pipeline(corpus[case])
    err = rec["error"]

    if isinstance(err, UnsupportedOperatorError):
        assert err.operator_name and err.operator_name != "unknown", (
            f"{case}: refused without naming the operator. TVM's own message was: {err.details}"
        )
        return

    if err is not None or rec["verdict"] == "CLEAN":
        return

    result = rec["result"]
    accounted = set(result.repaired_ops) | set(result.unrepairable_ops)
    assert accounted, (
        f"{case}: analyze_graph flagged {rec['unsupported']} as unsupported, but repair "
        f"returned {result.status} accounting for no operator at all"
    )
    if result.status in ("BLOCKED", "PARTIAL"):
        assert result.unrepairable_ops, (
            f"{case}: status {result.status} with an empty unrepairable_ops -- the studio "
            f"has nothing to put in the 'blocked by' column"
        )
        assert all(op for op in result.unrepairable_ops), f"{case}: blank operator name in {result.unrepairable_ops}"


@pytest.mark.unit
@pytest.mark.parametrize("case", RUNNABLE)
def test_no_runnable_model_is_reported_unsupported(case: str, corpus) -> None:
    """
    A model TATVA can run must not be told its operators are unsupported.

    This is the mapping-table contract, and the whole of rahul's report. SUPPORTED_OPS
    is a hand-written list that `analyze_graph` diffs the graph against, and nothing
    used to tie it to what the compiler does -- so it drifted, in the direction that
    costs the user most. Twenty-six operators, `nn.conv2d` and `nn.max_pool2d` among
    them, were reported unsupported with a repair offered, by a toolchain that
    cross-compiles and runs them correctly. From the studio that looks exactly like
    "unsupported ops, and the rewrite does nothing": there was nothing to rewrite.

    Every case here is one test_model_compiles_runs_and_matches_the_host proves runs,
    so this cannot be satisfied by widening SUPPORTED_OPS on optimism.
    """
    pytest.importorskip("tvm")
    rec = _front_of_pipeline(corpus[case])
    assert rec["error"] is None, f"{case}: {rec['stage']} raised {rec['error']!r}"
    assert rec["unsupported"] == [], (
        f"{case}: reported {rec['unsupported']} as unsupported, but this model compiles "
        f"and runs. The graph is {rec['ops']}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(NOT_RUNNABLE))
def test_a_model_that_cannot_run_says_why(case: str, corpus) -> None:
    """
    The seven models TATVA cannot run must each fail with a diagnosis naming the cause.

    Not just a typed exception -- the specific reason, in words about the model. Two of
    these used to be raw crashes out of code generation: BatchNorm produced
    `KeyError: 'lv'` and NonZero a `TypeError` from int(), both naming variables invented
    by TVM that appear nowhere in the file the user handed over.

    The three tuple/shape cases fail in harness generation and the abs case in TVM
    lowering, all before the cross-compiler is looked for, so this needs no toolchain.
    """
    pytest.importorskip("tvm")
    import tempfile

    from tatva.compiler import TARGETS, import_model
    from tatva.runner import compile_model

    expected = NOT_RUNNABLE[case]
    with pytest.raises(DIAGNOSED_ERRORS) as excinfo:
        ir = import_model(str(corpus[case]))
        with tempfile.TemporaryDirectory() as build_dir:
            compile_model(ir, TARGETS["RV64GC"], build_dir, warmup_count=1, timed_count=1)

    err = excinfo.value
    assert classify_failure(err).error_type != "unknown", f"{case}: classified as 'unknown'"
    assert expected.lower() in str(err).lower(), (
        f"{case}: the failure never mentions {expected!r}, so the user cannot tell what "
        f"about their model TATVA objects to. It said: {err}"
    )


def _first_logits(raw_output: str) -> list[float]:
    """
    The output values the harness printed, parsed the way the studio parses them.

    float() accepts 'nan' and 'inf', which the target now emits for non-finite results.
    It used to print (uint64_t)NaN, i.e. "18446744073709551615.//////", and a run whose
    model merely overflowed looked like a broken emulator.
    """
    for line in raw_output.splitlines():
        if "FIRST_LOGITS:" in line:
            return [float(tok) for tok in line.split(":", 1)[1].split()]
    raise AssertionError(f"the harness printed no FIRST_LOGITS line:\n{raw_output}")


@pytest.mark.integration
@pytest.mark.parametrize("case", RUNNABLE)
def test_model_compiles_runs_and_matches_the_host(case: str, corpus, skip_if_no_toolchain) -> None:
    """
    The whole pipeline, on every model the corpus says should survive it.

    Import, analyze, cross-compile for RV64GC, boot it under QEMU, and check the values
    the target printed against onnxruntime on the same inputs. Nothing short of this
    catches a wrong answer: the model builds, the emulator exits cleanly, the studio
    reports a cycle count, and the numbers are simply not the model's.

    That was not hypothetical. `tvmgen_default_run` only emitted code for call_tir
    bindings, so a graph whose output was a rename (`gv = lv`, which is what the ONNX
    frontend emits for LayerNorm), a pass-through (`gv = x`), or a constant-folded
    operator (Gather) never wrote its output buffer at all and returned a tensor of
    zeros -- successfully, with a measurement attached.

    The first check below is what makes this test worth its runtime: a parity assertion
    that only compares zeros to zeros proves nothing, and several of these models
    legitimately produce zeros on the target.
    """
    import tempfile

    import numpy as np

    from tatva.compiler import TARGETS, import_model
    from tatva.runner import ExecutionEnvironment, compile_model, reference_output, run_and_measure

    reference = np.asarray(reference_output(str(corpus[case])), dtype=np.float64)

    ir = import_model(str(corpus[case]))
    with tempfile.TemporaryDirectory() as build_dir:
        artifact = compile_model(ir, TARGETS["RV64GC"], build_dir, warmup_count=1, timed_count=1)
        measurement = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)

    target = np.asarray(_first_logits(measurement.raw_output), dtype=np.float64)
    assert target.size, f"{case}: the target printed no values"

    n = min(target.size, reference.size, 5)
    # unary_Sqrt and unary_Log are fed a ramp that crosses zero, so both sides are NaN
    # over part of the output -- which is itself worth checking, because printing a
    # non-finite value is where the harness used to produce unparseable output.
    assert np.allclose(target[:n], reference[:n], rtol=1e-4, atol=1e-4, equal_nan=True), (
        f"{case}: the target computed something else.\n"
        f"  target: {list(target[:n])}\n"
        f"  host:   {list(reference[:n])}"
    )


@pytest.mark.integration
def test_a_repaired_graph_compiles_and_runs(corpus, skip_if_no_toolchain) -> None:
    """
    A repair that reports REPAIRED must produce a graph that actually builds.

    "The rewrite option is not working" is half of what was reported, and a REPAIRED
    status that has only been verified numerically on the host does not answer it: the
    rewritten graph still has to survive lowering, cross-compilation and the target.

    `unary_Abs` is the case that needs it. relax.abs legalizes to tirx.fabs, which the C
    backend has no implementation for, so the model genuinely cannot be compiled as
    written -- and the repair rewriting it to less/multiply/where is the only thing that
    makes it runnable at all.
    """
    import tempfile

    import numpy as np

    from tatva.compiler import TARGETS, import_model
    from tatva.repair import repair_graph
    from tatva.runner import ExecutionEnvironment, compile_model, reference_output, run_and_measure

    ir = import_model(str(corpus["unary_Abs"]))
    result = repair_graph(ir, TARGETS["RV64GC"], verify_numerically=True)
    assert result.status == "REPAIRED", f"expected a clean repair, got {result.status}"
    assert not result.remaining_unsupported

    with tempfile.TemporaryDirectory() as build_dir:
        artifact = compile_model(
            result.model_ir, TARGETS["RV64GC"], build_dir, warmup_count=1, timed_count=1
        )
        measurement = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)

    target = np.asarray(_first_logits(measurement.raw_output), dtype=np.float64)
    reference = np.asarray(reference_output(str(corpus["unary_Abs"])), dtype=np.float64)
    n = min(target.size, reference.size, 5)
    assert np.allclose(target[:n], reference[:n], rtol=1e-4, atol=1e-4), (
        f"the repaired graph built and ran but computes something other than abs: "
        f"{list(target[:n])} against {list(reference[:n])}"
    )


@pytest.mark.unit
def test_supported_ops_all_exist_in_relax() -> None:
    """
    Every name in SUPPORTED_OPS must be a real Relax operator.

    `analyze_graph` compares TVM's operator names, minus the "relax." prefix, against
    this set. A name that no Relax operator has can therefore never match -- it is not a
    harmless extra entry, it is a claim of support that silently does nothing, and in
    the case of `concatenate` it also meant the operator TVM *does* emit fell through to
    the unsupported list.
    """
    tvm = pytest.importorskip("tvm")
    pytest.importorskip("tvm.relax")

    registry = set(tvm.ir.Op.list_op_names())
    phantom = sorted(n for n in SUPPORTED_OPS if f"relax.{n}" not in registry and n not in registry)
    assert phantom == [], (
        f"SUPPORTED_OPS names operators Relax does not have: {phantom}. "
        f"These can never match anything analyze_graph reports."
    )


@pytest.mark.unit
def test_lowering_table_covers_exactly_the_supported_set() -> None:
    """
    The capability browser must describe every operator TATVA claims, and no others.

    test_capabilities already pins one direction (no lowering for an unsupported op).
    This pins the other: a supported operator with no entry falls back to the generic
    "not separately characterised" text, which is how `concat` would have gone on
    looking half-documented even after the spelling was fixed.
    """
    from tatva.capabilities import _LOWERINGS

    assert set(_LOWERINGS) == set(SUPPORTED_OPS), (
        f"undocumented: {sorted(SUPPORTED_OPS - set(_LOWERINGS))}; "
        f"documented but unsupported: {sorted(set(_LOWERINGS) - SUPPORTED_OPS)}"
    )


@pytest.mark.unit
def test_import_failure_is_not_explained_as_a_linker_problem() -> None:
    """
    A frontend refusal must be explained as one.

    Import failures are CompilationErrors so that `classify_failure` can type them, but
    they never reached a compiler. The generic compilation advice -- check your gcc
    flags, read the linker output, look at link.ld -- would send someone hunting through
    a toolchain that was never invoked.
    """
    from tatva.diagnostics import get_offline_explanation

    err = CompilationError(
        stage="import",
        command="tvm.relax.frontend.onnx.from_onnx(model.onnx)",
        stderr="AttributeError: 'str' object has no attribute 'decode'",
        details="The TVM ONNX frontend failed while converting this model.",
    )
    text = get_offline_explanation(classify_failure(err))

    assert "frontend" in text.lower()
    for misleading in ("link.ld", "linker", "cross-compiler options"):
        assert misleading not in text, f"import-stage advice still mentions {misleading!r}"


# The exact strings TVM emits. Kept verbatim so that if a TVM upgrade rewords them, this
# test fails here -- loudly and in one place -- instead of the whole corpus quietly going
# back to reporting 'unknown'.
@pytest.mark.unit
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("The following operators are not supported for frontend ONNX: Dropout", ["Dropout"]),
        ("The following operators are not supported for frontend ONNX: Dropout, LSTM", ["Dropout", "LSTM"]),
        (
            "The following operators are not supported for frontend ONNX: UnsupportedOpXYZ\n"
            "Some downstream context TVM likes to append",
            ["UnsupportedOpXYZ"],
        ),
        ("No Op registered for Foo", ["Foo"]),
        ("something else entirely", []),
    ],
)
def test_unsupported_operator_names_reads_tvms_message(message: str, expected: list[str]) -> None:
    """Pure parsing, no TVM needed -- so this still guards the message format on a bare checkout."""
    assert unsupported_operator_names(message) == expected
