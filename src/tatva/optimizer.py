"""
TATVA Precision Quantization and Kernel Selection Module.

Holds the INT8 quantization transform on Relax IRModules, the optimized-softmax
kernel selection pass, and the configuration comparison benchmark.
"""

from typing import Any

import numpy as np

from tatva.compiler import ModelIR, TargetVariant

# INT8 symmetric quantization uses the signed range [-127, 127]; -128 is dropped so
# that negating a quantized value cannot overflow.
INT8_QMAX = 127.0

# Used when activation calibration is unavailable. It is a guess, and callers are
# told so via metadata["activation_scale_source"] rather than being left to assume
# the number came from the model.
FALLBACK_ACTIVATION_SCALE = 0.05

# Clipping percentile for activation calibration. Measured over the five fixtures in
# models/ (3 input seeds each, normalized MSE against the onnxruntime reference):
#
#   fixed 0.05  0.0142      p99.9   0.0098   <- chosen
#   max         0.0233      p99.5   0.0095
#   p99.99      0.0227      p99     0.0137
#
# Calibrating on the true max is the *worst* option: one outlier sets the step size
# for every other value. p99.9 clips a thousandth of the values and buys back the
# resolution, and it beats or ties the old hardcoded 0.05 on every fixture.
CALIBRATION_PERCENTILE = 99.9


def calibrate_activation_scale(
    onnx_path: str,
    inputs: dict[str, Any] | None = None,
    percentile: float = CALIBRATION_PERCENTILE,
) -> tuple[float, str]:
    """
    Measure the activation range of a model by running it once on the host.

    Only the tensors the quantizer actually touches are measured: the non-constant
    inputs of MatMul and Gemm nodes. Those are temporarily promoted to graph outputs,
    the model is run with the same dummy inputs the benchmark harness uses, and the
    `percentile` of their pooled magnitudes sets a symmetric INT8 scale.

    Two things are deliberate here. Restricting to matmul inputs keeps an unrelated
    tensor from setting the scale -- on models/model_pretrained.onnx the peak over
    all activations is 208 while the peak over matmul inputs is 11.4. And clipping at
    a percentile rather than the max keeps a lone outlier from coarsening the step
    for everything else; see CALIBRATION_PERCENTILE for the measurements.

    Returns (scale, source) where `source` describes how the number was obtained, so
    a caller can tell a measured scale from a fallback.
    """
    import onnx
    import onnxruntime as ort

    from tatva.runner import default_inputs_for

    try:
        model = onnx.load(onnx_path)
        initializers = {i.name for i in model.graph.initializer}
        produced = {name for node in model.graph.node for name in node.output if name}

        wanted = {
            inp
            for node in model.graph.node
            if node.op_type in ("MatMul", "Gemm")
            for inp in node.input
            if inp and inp not in initializers and inp in produced
        }
        scope = "matmul inputs"
        if not wanted:
            # No matmul takes a computed activation (all operands are weights or
            # graph inputs). Fall back to the widest evidence available.
            wanted = produced
            scope = "all intermediates"

        # A bare ValueInfoProto with only a name lets onnxruntime infer the type
        # itself. Declaring TensorProto.UNDEFINED instead is rejected outright
        # ("Invalid tensor data type 0"), which silently sent every model down the
        # fallback path.
        existing = {o.name for o in model.graph.output}
        for name in sorted(wanted):
            if name not in existing:
                vi = onnx.ValueInfoProto()
                vi.name = name
                model.graph.output.append(vi)

        sess = ort.InferenceSession(model.SerializeToString())
        feed = inputs if inputs is not None else default_inputs_for(onnx_path)
        expected = {i.name for i in sess.get_inputs()}
        outputs = sess.run(None, {k: v for k, v in feed.items() if k in expected})
        out_names = [o.name for o in sess.get_outputs()]
    except Exception as e:
        return FALLBACK_ACTIVATION_SCALE, f"fallback (calibration run failed: {e})"

    magnitudes: list[np.ndarray] = []
    # strict=False: onnxruntime returns one array per declared output, but pairing them
    # up is best-effort calibration -- a mismatch should skip a tensor, not abort the run.
    for name, arr in zip(out_names, outputs, strict=False):
        if name not in wanted:
            continue
        arr = np.asarray(arr)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.floating):
            continue
        finite = np.abs(arr[np.isfinite(arr)]).ravel()
        if finite.size:
            magnitudes.append(finite)

    if not magnitudes:
        return FALLBACK_ACTIVATION_SCALE, "fallback (no finite float activations observed)"

    pooled = np.concatenate(magnitudes)
    cutoff = float(np.percentile(pooled, percentile))
    if cutoff <= 0.0:
        # Everything below the percentile is zero; fall back to the true peak so the
        # scale is at least non-degenerate.
        cutoff = float(pooled.max())
    if cutoff <= 0.0:
        return FALLBACK_ACTIVATION_SCALE, "fallback (all observed activations were zero)"

    return cutoff / INT8_QMAX, (
        f"calibrated on host ({scope}, p{percentile:g} |activation| = {cutoff:.6g}, "
        f"max = {float(pooled.max()):.6g})"
    )


def _weight_scale(arr: np.ndarray) -> float:
    """
    Symmetric per-tensor INT8 scale for one weight tensor, from its own values.

    A fixed scale (this used to be 0.02) either wastes most of the INT8 range on a
    small-magnitude tensor or clips a large one; neither failure is visible in the
    graph, only in the output.
    """
    finite = arr[np.isfinite(arr)] if arr.size else arr
    peak = float(np.abs(finite).max()) if finite.size else 0.0
    if peak <= 0.0:
        return 1.0 / INT8_QMAX
    return peak / INT8_QMAX


def _fold_fake_quant(arr: np.ndarray, scale: float) -> np.ndarray:
    """
    Evaluate a quantize -> dequantize pair on a constant tensor at compile time.

    Reproduces in float32 what relax.quantize and relax.dequantize lower to in the
    C backend TATVA cross-compiles:

        quantize:    v = roundf(x / scale)
                     v = v < 127.0f ? v : 127.0f
                     q = (int8_t)(v > -128.0f ? v : -128.0f)
        dequantize:  out = (float)(int32_t)q * scale

    Two details are load-bearing and neither is what the obvious implementation does.

    C's roundf() rounds halves away from zero while numpy.round rounds them to even,
    so the ties are corrected explicitly rather than delegated. That is not a
    hypothetical difference: TVM's own LLVM host backend lowers te.round to a
    half-to-even instruction, so relax.transform.FoldConstant would fold this graph
    differently from the way the shipped RISC-V binary computes it. The tempting
    one-liner floor(|x| + 0.5) is wrong for a second, independent reason -- it
    double-rounds, and for x = 0.49999997f (the float32 just below 0.5) the float32
    sum reaches 1.0f, so floor returns 1 where roundf returns 0.

    And the clipping uses np.where rather than np.clip so the comparison order matches
    the emitted ternaries. It only shows up on non-finite input -- a NaN weight clips
    to 127 on target, where np.minimum would propagate the NaN into an out-of-range
    int8 cast -- but a quantizer that disagrees with its own kernel about edge cases is
    a bug waiting for a model that has them.

    Folding against the C semantics rather than the host's is what keeps the hoisted
    build close to the runtime one it replaces. It is NOT bit-identical, and the
    reason is worth knowing: TVM's C codegen prints the scale with %.6e, so the
    generated binary divides by a slightly different float32 than `scale`. Measured on
    models/model.onnx, one logit moves 0.221983 -> 0.221984 -- three orders of
    magnitude inside the 0.05 MSE tolerance compare_configs enforces. Round `scale`
    through '%.6e' first if bit-identical output against a pre-hoist build is ever
    required; that is a printf artifact of one backend, so it is not the default.
    """
    s = np.float32(scale)
    scaled = arr.astype(np.float32) / s
    # Round normally, then override the exact ties to go away from zero.
    with np.errstate(invalid="ignore"):
        ties = np.abs(scaled - np.trunc(scaled)) == np.float32(0.5)
    rounded = np.where(
        ties, np.trunc(scaled) + np.copysign(np.float32(1.0), scaled), np.round(scaled)
    ).astype(np.float32)
    clipped = np.where(rounded < np.float32(127.0), rounded, np.float32(127.0))
    clipped = np.where(clipped > np.float32(-128.0), clipped, np.float32(-128.0))
    q = clipped.astype(np.float32).astype(np.int8)
    return (q.astype(np.int32).astype(np.float32) * s).astype(np.float32)


def quantize(model_ir: ModelIR, activation_scale: float | None = None) -> ModelIR:
    """
    Insert INT8 Quantize-Dequantize pairs around the inputs of MatMul and Dense.

    Scales are derived from data, not assumed. Each weight tensor gets its own
    symmetric scale from its own values; the activation scale is measured by running
    the source model once on the host (see calibrate_activation_scale). The previous
    version hardcoded 0.05 for activations and 0.02 for weights regardless of the
    model, which is what "dynamic quantization" was never doing.

    Pass `activation_scale` to skip calibration and use an explicit value.
    """
    import tvm
    from tvm import relax

    mod = model_ir.mod
    metadata = model_ir.metadata.copy()

    if activation_scale is not None:
        act_scale, act_source = float(activation_scale), "caller-supplied"
    else:
        source_path = metadata.get("source_path")
        if source_path:
            act_scale, act_source = calibrate_activation_scale(source_path)
        else:
            act_scale, act_source = (
                FALLBACK_ACTIVATION_SCALE,
                "fallback (no source model path recorded on this IR)",
            )

    weight_scales: list[float] = []
    folded_weight_scales: list[float] = []

    @relax.expr_functor.mutator
    class QuantizationMutator(relax.PyExprMutator):
        def visit_call_(self, call: relax.Call) -> relax.Expr:
            # We first recursively visit the arguments
            call = super().visit_call_(call)

            if isinstance(call.op, tvm.ir.Op) and call.op.name in ("relax.matmul", "relax.nn.dense"):
                arg0 = call.args[0]
                arg1 = call.args[1]

                scale0 = relax.const(act_scale, "float32")
                zp0 = relax.const(0, "int32")
                q0 = self.builder_.emit(
                    relax.op.quantize(arg0, scale0, zp0, out_dtype="int8"),
                    name_hint="quant_act"
                )
                dq0 = self.builder_.emit(
                    relax.op.dequantize(q0, scale0, zp0, out_dtype="float32"),
                    name_hint="dequant_act"
                )

                # Per-tensor weight scale, measured when the weights are a constant.
                # A non-constant right-hand side (batched attention matmul) has no
                # values to measure here, so it falls back to the activation scale --
                # both sides of that product are activations anyway.
                wt_arr = arg1.data.numpy() if isinstance(arg1, relax.Constant) else None
                wt_scale = _weight_scale(wt_arr) if wt_arr is not None else act_scale
                weight_scales.append(wt_scale)

                if wt_arr is not None:
                    # The weight is known at compile time, so its quantize/dequantize
                    # pair is a constant expression. Evaluating it here instead of
                    # re-deriving the same bytes on every inference is the single
                    # largest cost in this pass: on models/model_pretrained.onnx these
                    # constants are 76.7% of every element the quantizer touches, and
                    # the generated binary was paying for all of them on every call.
                    #
                    # Both halves fold, not just the quantize. What the matmul consumes
                    # is the dequantized float32 tensor, so that is what is emitted: a
                    # float32 constant whose values have been round-tripped through
                    # int8. The arithmetic the model sees is unchanged -- this is the
                    # same fake-quant as before -- but the weight leg now costs nothing
                    # at inference time.
                    #
                    # Note what this does not do. The weights stay float32 on the
                    # device, so this buys latency and activation RAM, not flash. A
                    # hoisted weight occupies exactly the bytes it did before the
                    # quantizer ran; anything that reports this pass as a footprint
                    # reduction is reporting a number this code does not deliver.
                    folded_weight_scales.append(wt_scale)
                    dq1 = relax.const(_fold_fake_quant(wt_arr, wt_scale), "float32")
                else:
                    # A computed right-hand side -- the batched attention matmul, whose
                    # second operand is an activation -- has no compile-time value, so
                    # it still has to be quantized at runtime.
                    scale1 = relax.const(wt_scale, "float32")
                    zp1 = relax.const(0, "int32")
                    q1 = self.builder_.emit(
                        relax.op.quantize(arg1, scale1, zp1, out_dtype="int8"),
                        name_hint="quant_wt"
                    )
                    dq1 = self.builder_.emit(
                        relax.op.dequantize(q1, scale1, zp1, out_dtype="float32"),
                        name_hint="dequant_wt"
                    )

                return relax.Call(call.op, [dq0, dq1], call.attrs, call.sinfo_args, call.span)

            return call

    # Pass the module mod to initialize builder_ properly
    mutator = QuantizationMutator(mod)
    mutated_func = mutator.visit_expr(mod["main"])

    # Resolve global variable for 'main'
    gv = None
    for g_var in mod.get_global_vars():
        if g_var.name_hint == "main":
            gv = g_var
            break
    if gv is None:
        raise ValueError("Could not find global variable 'main' in the module.")

    bb = relax.BlockBuilder()
    bb.update_func(gv, mutated_func)
    mutated_mod = bb.finalize()

    metadata["quantized"] = True
    metadata["activation_scale"] = act_scale
    metadata["activation_scale_source"] = act_source
    metadata["weight_scales"] = weight_scales
    metadata["quantized_ops"] = len(weight_scales)
    metadata["hoisted_weight_ops"] = len(folded_weight_scales)

    return ModelIR(mutated_mod, None, metadata)


def check_softmax_fusable(model_ir: ModelIR) -> tuple[bool, str]:
    """
    Decide whether TATVA's replacement softmax kernel is valid for this graph.

    The kernel assumes float32 data reduced along the last axis. Injecting it into a
    graph that violates either assumption yields a binary that runs faster and computes
    the wrong answer, which is worse than not optimizing at all -- so the pass refuses
    rather than guesses. Returns (fusable, reason_if_not).

    There is deliberately NO input-range check here. The kernel's Schraudolph
    exponential is only valid while (x - row_max) stays above about -88, but that
    range is a property of runtime activations -- typically a matmul output plus an
    additive attention mask -- and Relax StructInfo carries dtype and shape only, no
    value bounds. A gate that cannot be evaluated is worse than no gate, because it
    reads as protection that is not there. Safety is enforced in the kernel instead:
    both SOFTMAX_SCALAR and SOFTMAX_VECTOR clamp (x - row_max) < -15.0f to exactly
    zero, which bounds the Schraudolph integer to [883820160, 1065353216] -- always
    positive, never saturating -- so every exp term is a positive float in roughly
    [3.2e-7, 1.0], the row max always contributes exactly 1.0, and the sum is
    therefore always >= 1.0. The kernel is unconditionally safe for any finite input
    and needs no gate.
    """
    import tvm
    from tvm import relax

    seen = 0
    for block in model_ir.mod["main"].body.blocks:
        for binding in block.bindings:
            call = binding.value
            if not isinstance(call, relax.Call) or not isinstance(call.op, tvm.ir.Op):
                continue
            if "softmax" not in call.op.name:
                continue

            seen += 1
            sinfo = call.args[0].struct_info
            dtype = str(getattr(sinfo, "dtype", ""))
            if dtype != "float32":
                return False, f"softmax input is {dtype}; the TATVA kernel is float32-only"

            shape = getattr(sinfo, "shape", None)
            if shape is None:
                return False, "softmax input has no static shape"
            ndim = len(shape)
            axis = int(getattr(call.attrs, "axis", -1))
            if axis not in (-1, ndim - 1):
                return False, (
                    f"softmax reduces axis {axis} of a {ndim}-D tensor; "
                    "the TATVA kernel only handles the last axis"
                )

    if seen == 0:
        return False, "the graph contains no softmax operator to replace"
    return True, ""


def select_fast_softmax_kernel(model_ir: ModelIR) -> ModelIR:
    """
    Request TATVA's Schraudolph fast-exponential softmax kernel for this model.

    This does NOT fuse anything and does not rewrite the Relax graph -- the older
    name `fuse_attention_softmax` described an operation that never happened. What it
    does is set a flag that runner.compile_model acts on, replacing TVM's generated
    softmax kernels in the emitted C with TATVA's own hand-written one.

    The returned IR carries either metadata["softmax_optimized"] = True or
    metadata["softmax_fusion_skipped"] = <reason>; it never declines silently.
    """
    from tatva.compiler import analyze_graph

    metadata = model_ir.metadata.copy()
    metadata.pop("softmax_optimized", None)
    metadata.pop("softmax_fusion_skipped", None)

    report = analyze_graph(model_ir)
    if not report.has_transformer_bottleneck:
        metadata["softmax_fusion_skipped"] = (
            "no attention bottleneck detected (needs both a softmax and a matmul/dense)"
        )
        return ModelIR(model_ir.mod, None, metadata)

    fusable, reason = check_softmax_fusable(model_ir)
    if not fusable:
        metadata["softmax_fusion_skipped"] = reason
        return ModelIR(model_ir.mod, None, metadata)

    metadata["softmax_optimized"] = True
    return ModelIR(model_ir.mod, None, metadata)


# Kept so existing callers and saved scripts keep working. New code should use
# select_fast_softmax_kernel, which says what the pass actually does.
fuse_attention_softmax = select_fast_softmax_kernel


def compare_configs(
    onnx_path: str,
    variant: TargetVariant,
    configs: list[str],
    passes: list[str] | None = None,
    model_ir: Any = None,
) -> dict[str, Any]:
    """
    Compare baseline (FP32) and optimized models on the same imported model instance.
    Reuses the same imported instance to avoid redundant loading, and utilizes session cache.

    `model_ir` lets a caller supply an already-imported module instead of loading the
    file again. The graph-repair engine uses it to compile the *rewritten* graph: the
    repaired module exists only in memory, so without this the compile below would
    silently re-import the unrepaired file from disk and the repair would have no effect
    on anything that was built.
    """
    from tatva._cache import GLOBAL_SESSION_CACHE
    from tatva.compiler import import_model
    from tatva.runner import ExecutionEnvironment, compile_model, reference_output, run_and_measure

    pass_key = ",".join(sorted(passes or [])) + ":" + ",".join(sorted(configs or []))
    # A repaired graph and the original file share a path, so without this the cache
    # would hand a repaired run's results back to an unrepaired one, and vice versa.
    if model_ir is not None:
        repaired = model_ir.metadata.get("repaired_ops") if hasattr(model_ir, "metadata") else None
        pass_key += ":repaired=" + (",".join(sorted(repaired)) if repaired else "supplied")
    target_key = variant.name if hasattr(variant, "name") else str(variant)

    cached_res = GLOBAL_SESSION_CACHE.get_artifact(onnx_path, pass_key, target_key)
    if cached_res is not None:
        return cached_res

    if model_ir is None:
        model_ir = import_model(onnx_path)
    results = {}
    skipped: dict[str, str | None] = {}

    if "baseline" in configs:
        artifact_bl = compile_model(model_ir, variant, warmup_count=2, timed_count=5)
        res_bl = run_and_measure(artifact_bl, environment=ExecutionEnvironment.QEMU_SIM)

        bl_logits = []
        for line in res_bl.raw_output.splitlines():
            if "FIRST_LOGITS:" in line:
                parts = line.strip().split(":")[1].strip().split()
                bl_logits = [float(x) for x in parts[:5]]
                break
        results["baseline"] = {"latency": res_bl, "logits": bl_logits, "build_dir": artifact_bl.build_dir}

    if "quantized" in configs:
        quant_model_ir = quantize(model_ir)
        artifact_quant = compile_model(quant_model_ir, variant, warmup_count=2, timed_count=5)
        res_quant = run_and_measure(artifact_quant, environment=ExecutionEnvironment.QEMU_SIM)

        quant_logits = []
        for line in res_quant.raw_output.splitlines():
            if "FIRST_LOGITS:" in line:
                parts = line.strip().split(":")[1].strip().split()
                quant_logits = [float(x) for x in parts[:5]]
                break
        results["quantized"] = {"latency": res_quant, "logits": quant_logits, "build_dir": artifact_quant.build_dir}

    if "fused" in configs:
        fused_model_ir = select_fast_softmax_kernel(model_ir)
        skipped["fused"] = fused_model_ir.metadata.get("softmax_fusion_skipped")
        artifact_fused = compile_model(fused_model_ir, variant, warmup_count=2, timed_count=5)
        res_fused = run_and_measure(artifact_fused, environment=ExecutionEnvironment.QEMU_SIM)

        fused_logits = []
        for line in res_fused.raw_output.splitlines():
            if "FIRST_LOGITS:" in line:
                parts = line.strip().split(":")[1].strip().split()
                fused_logits = [float(x) for x in parts[:5]]
                break
        results["fused"] = {"latency": res_fused, "logits": fused_logits, "build_dir": artifact_fused.build_dir}

    if "optimized" in configs:
        opt_passes = passes if passes is not None else ["fuse"]
        opt_model_ir = model_ir
        # Order of execution: fuse first (recommmended), then quantize
        if "fuse" in opt_passes:
            opt_model_ir = select_fast_softmax_kernel(opt_model_ir)
            skipped["optimized"] = opt_model_ir.metadata.get("softmax_fusion_skipped")
        if "quantize" in opt_passes:
            opt_model_ir = quantize(opt_model_ir)

        artifact_opt = compile_model(opt_model_ir, variant, warmup_count=2, timed_count=5)
        res_opt = run_and_measure(artifact_opt, environment=ExecutionEnvironment.QEMU_SIM)

        opt_logits = []
        for line in res_opt.raw_output.splitlines():
            if "FIRST_LOGITS:" in line:
                parts = line.strip().split(":")[1].strip().split()
                opt_logits = [float(x) for x in parts[:5]]
                break
        results["optimized"] = {"latency": res_opt, "logits": opt_logits, "build_dir": artifact_opt.build_dir}

    # Ground truth is the host ONNX Runtime result, not the QEMU baseline.
    # Scoring an optimized build against the baseline build only measures how far
    # the two RISC-V binaries drift from each other; if the baseline itself were
    # wrong, every optimized config would score a perfect zero MSE against it.
    #
    # Note this reads the file on disk, so when `model_ir` is a repaired graph the
    # reference is still the *original, unrepaired* model. That is deliberate: a rewrite
    # is only legitimate if the binary it produces still matches the model the user gave
    # us. Referencing the repaired graph against itself would make every rewrite pass.
    accuracy_tolerance = 0.05
    ref_logits = [float(x) for x in reference_output(onnx_path)[:5]]
    ref_arr = np.array(ref_logits)

    def mse_vs_reference(logits: list[float]) -> float:
        arr = np.array(logits)
        if arr.shape != ref_arr.shape:
            return float("inf")
        return float(np.mean((ref_arr - arr) ** 2))

    comparison: dict[str, Any] = {
        "reference_logits": ref_logits,
        "accuracy_reference": "host onnxruntime",
        "tolerance": accuracy_tolerance,
    }

    if "baseline" in results:
        bl_data = results["baseline"]
        mse_bl = mse_vs_reference(bl_data["logits"])
        comparison.update({
            "baseline_mean_ms": bl_data["latency"].mean_ms,
            "baseline_median_ms": bl_data["latency"].median_ms,
            "baseline_p95_ms": bl_data["latency"].p95_ms,
            "baseline_accuracy_delta_mse": mse_bl,
            "baseline_accuracy_ok": mse_bl <= accuracy_tolerance,
        })
        if mse_bl > accuracy_tolerance:
            comparison["error_baseline"] = (
                f"Baseline diverges from the host reference ({mse_bl:.5f} > {accuracy_tolerance}); "
                "the optimized deltas below are measured against a baseline that is itself wrong."
            )

        for cfg, prefix, err_key in (
            ("quantized", "quantized", "error"),
            ("fused", "fused", "error_fused"),
            ("optimized", "opt", "error_opt"),
        ):
            if cfg not in results:
                continue
            data = results[cfg]
            mse = mse_vs_reference(data["logits"])
            ok = mse <= accuracy_tolerance

            comparison.update({
                f"{prefix}_mean_ms": data["latency"].mean_ms,
                f"{prefix}_median_ms": data["latency"].median_ms,
                f"{prefix}_p95_ms": data["latency"].p95_ms,
                f"{prefix}_mean_delta_ms": data["latency"].mean_ms - bl_data["latency"].mean_ms,
                f"{prefix}_median_delta_ms": data["latency"].median_ms - bl_data["latency"].median_ms,
                f"{prefix}_p95_delta_ms": data["latency"].p95_ms - bl_data["latency"].p95_ms,
                f"{prefix}_accuracy_delta_mse": mse,
                f"{prefix}_accuracy_ok": ok,
            })
            if skipped.get(cfg):
                comparison[f"{prefix}_optimization_skipped"] = skipped[cfg]
            if not ok:
                comparison[err_key] = (
                    f"{cfg.capitalize()} accuracy degradation exceeds tolerance "
                    f"({mse:.5f} > {accuracy_tolerance})."
                )

        # Back-compat aliases: the quantize path historically used unprefixed keys.
        if "quantized" in results:
            comparison["latency_delta_ms"] = comparison["quantized_mean_delta_ms"]
            comparison["accuracy_delta_mse"] = comparison["quantized_accuracy_delta_mse"]
            comparison["accuracy_ok"] = comparison["quantized_accuracy_ok"]

    final_res = {"results": results, "comparison": comparison}
    GLOBAL_SESSION_CACHE.put_artifact(onnx_path, pass_key, target_key, final_res)
    return final_res
