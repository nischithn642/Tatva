"""
Tests for per-kernel cycle attribution (compile_model(profile=True)).

Two things need pinning, and they pull in opposite directions.

The first is that the profiler actually measures: it must emit one entry per
distinct generated kernel and its per-kernel totals must add up to very nearly
the same window the RUN_CYCLES samples cover. A profiler that reported plausible
but unattached numbers would be worse than no profiler at all.

The second is that it stays *off* unless asked. The latency figures TATVA
reports come from unprofiled builds, so instrumentation leaking into a default
build would quietly inflate every number the product prints. The guard test
below asserts the generated sources contain no trace of the feature when
profile=False, which is what keeps that guarantee from silently rotting.
"""

import pytest

from tatva.compiler import TARGETS, import_model
from tatva.optimizer import quantize
from tatva.runner import compile_model, run_and_measure


@pytest.mark.integration
def test_profiled_build_attributes_cycles_to_kernels(skip_if_no_toolchain, tmp_path) -> None:
    """A profiled run must explain nearly all of the window it measured."""
    ir = import_model("models/model.onnx")
    artifact = compile_model(
        ir, TARGETS["RV64GC"], str(tmp_path / "prof"), warmup_count=1, timed_count=2, profile=True
    )
    result = run_and_measure(artifact)

    assert result.kernel_profiles, "profile=True produced no KERNEL_CYCLES lines"
    # One entry per distinct kernel name, not per call site: `cast` is called twice
    # per run by this graph and must still appear exactly once.
    names = [k.name for k in result.kernel_profiles]
    assert len(names) == len(set(names)), f"duplicate kernel entries: {names}"

    # Coverage is never exactly 1.0 -- the residue is tvmgen_default_run's own
    # prologue, the DLTensor stores, and the rdcycle pair bracketing each call.
    assert 0.95 < result.attribution_coverage < 1.0, (
        f"coverage {result.attribution_coverage} outside the plausible band; "
        f"attributed {result.attributed_cycles} of {result.total_cycles}"
    )
    assert result.attributed_cycles == sum(k.cycles for k in result.kernel_profiles)

    # Heaviest first, so the top entry is the thing worth optimising.
    cycles = [k.cycles for k in result.kernel_profiles]
    assert cycles == sorted(cycles, reverse=True)

    # calls is a total across the timed runs, so it must be a multiple of timed_count.
    for k in result.kernel_profiles:
        assert k.calls % 2 == 0, f"{k.name} called {k.calls} times over 2 timed runs"
        assert k.cycles_per_call == pytest.approx(k.cycles / k.calls)


@pytest.mark.integration
def test_unprofiled_build_contains_no_instrumentation(skip_if_no_toolchain, tmp_path) -> None:
    """
    The default build must carry no trace of the profiler.

    This is the guardrail behind the claim that TATVA's reported latency is never
    measured with instrumentation switched on.
    """
    ir = import_model("models/model.onnx")
    compile_model(ir, TARGETS["RV64GC"], str(tmp_path / "plain"), warmup_count=1, timed_count=1)

    main_c = (tmp_path / "plain" / "main.c").read_text()
    model_run_c = (tmp_path / "plain" / "model_run.c").read_text()

    assert "@TATVA_" not in main_c, "harness template markers were left unsubstituted"
    for marker in ("tatva_kernel_", "tatva_read_cycles", "tatva_profile_reset", "KERNEL_CYCLES"):
        assert marker not in main_c, f"{marker} leaked into an unprofiled main.c"
        assert marker not in model_run_c, f"{marker} leaked into an unprofiled model_run.c"


@pytest.mark.integration
def test_unprofiled_result_reports_zero_attribution(skip_if_no_toolchain, tmp_path) -> None:
    """An unprofiled run must be distinguishable from one whose table explained nothing."""
    ir = import_model("models/model.onnx")
    artifact = compile_model(ir, TARGETS["RV64GC"], str(tmp_path / "plain2"), warmup_count=1, timed_count=2)
    result = run_and_measure(artifact)

    assert result.kernel_profiles == []
    assert result.attributed_cycles == 0
    assert result.attribution_coverage == 0.0
    # total_cycles is still populated, which is exactly why a consumer has to read
    # kernel_profiles to tell "profiling was off" from "nothing was attributed".
    assert result.total_cycles > 0


@pytest.mark.integration
def test_profile_attributes_int8_overhead_to_quantize_kernels(skip_if_no_toolchain, tmp_path) -> None:
    """
    The INT8 regression must be attributable, not merely observable.

    This is the measurement that justifies how the quantize pass is described: the
    quantize/dequantize kernels exist only in the INT8 build, and the kernels shared
    with FP32 must be unchanged -- in particular `matmul`, which stays FP32 and so
    gains nothing from the pass.
    """
    tgt = TARGETS["RV64GC"]

    fp32 = run_and_measure(
        compile_model(
            import_model("models/model.onnx"), tgt, str(tmp_path / "p32"), warmup_count=1, timed_count=2, profile=True
        )
    )
    int8 = run_and_measure(
        compile_model(
            quantize(import_model("models/model.onnx")),
            tgt,
            str(tmp_path / "p8"),
            warmup_count=1,
            timed_count=2,
            profile=True,
        )
    )

    fp32_cycles = {k.name: k.cycles for k in fp32.kernel_profiles}
    int8_cycles = {k.name: k.cycles for k in int8.kernel_profiles}

    # The overhead kernels exist only in the INT8 build.
    assert "quantize" in int8_cycles and "quantize" not in fp32_cycles
    assert "dequantize" in int8_cycles and "dequantize" not in fp32_cycles

    # The dominant kernel is untouched: quantization does not make the matmul integer.
    assert int8_cycles["matmul"] == fp32_cycles["matmul"], (
        "matmul cycles changed under quantization; the pass is no longer pure fake-quant"
    )

    # And the added kernels account for essentially the whole regression.
    delta = int8.total_cycles - fp32.total_cycles
    assert delta > 0, "INT8 is expected to be slower than FP32 on scalar RV64GC"
    overhead = int8_cycles["quantize"] + int8_cycles["dequantize"]
    assert overhead >= 0.9 * delta, (
        f"quantize+dequantize ({overhead}) explain less than 90% of the {delta}-cycle regression"
    )
