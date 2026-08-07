"""
TATVA Precision Quantization and Schedule Optimization Module.

This module handles precision quantization transforms (FP32 -> INT8) on Relax
IRModules, attention/softmax fusion optimizations, and provides configuration
comparison benchmarking utilities.
"""

from typing import Any, List

import numpy as np

from tatva.compiler import ModelIR, TargetVariant





def quantize(model_ir: ModelIR) -> ModelIR:
    """
    Simulate quantization on the Relax IRModule by inserting Quantize-Dequantize (QDQ)
    operators around inputs and weights of MatMul and Dense operations.
    """
    import tvm
    from tvm import relax

    mod = model_ir.mod

    @relax.expr_functor.mutator
    class QuantizationMutator(relax.PyExprMutator):
        def visit_call_(self, call: relax.Call) -> relax.Expr:
            # We first recursively visit the arguments
            call = super().visit_call_(call)

            if isinstance(call.op, tvm.ir.Op) and call.op.name in ("relax.matmul", "relax.nn.dense"):
                arg0 = call.args[0]
                arg1 = call.args[1]

                # Inserting QDQ for activation input (arg0) using self.builder_.emit with unique name_hint
                scale0 = relax.const(0.05, "float32")
                zp0 = relax.const(0, "int32")
                q0 = self.builder_.emit(
                    relax.op.quantize(arg0, scale0, zp0, out_dtype="int8"),
                    name_hint="quant_act"
                )
                dq0 = self.builder_.emit(
                    relax.op.dequantize(q0, scale0, zp0, out_dtype="float32"),
                    name_hint="dequant_act"
                )

                # Inserting QDQ for weights input (arg1) using self.builder_.emit with unique name_hint
                scale1 = relax.const(0.02, "float32")
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

    metadata = model_ir.metadata.copy()
    metadata["quantized"] = True

    return ModelIR(mutated_mod, None, metadata)


def check_softmax_fusable(model_ir: ModelIR) -> tuple[bool, str]:
    """
    Decide whether TATVA's replacement softmax kernel is valid for this graph.

    The kernel assumes float32 data reduced along the last axis. Injecting it into a
    graph that violates either assumption yields a binary that runs faster and computes
    the wrong answer, which is worse than not optimizing at all -- so the pass refuses
    rather than guesses. Returns (fusable, reason_if_not).
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


def fuse_attention_softmax(model_ir: ModelIR) -> ModelIR:
    """
    Request TATVA's Schraudolph fast-exponential softmax kernel for this model.

    This does not rewrite the Relax graph. It sets a flag that runner.compile_model
    acts on, replacing TVM's generated softmax kernels in the emitted C with TATVA's
    own. The name is kept for API compatibility; see `softmax_fusion_skipped` in the
    returned metadata when the pass declines to apply.
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


def compare_configs(
    onnx_path: str, variant: TargetVariant, configs: List[str], passes: List[str] = None
) -> dict[str, Any]:
    """
    Compare baseline (FP32) and optimized models on the same imported model instance.
    Reuses the same imported instance to avoid redundant loading, and utilizes session cache.
    """
    from tatva._cache import GLOBAL_SESSION_CACHE
    from tatva.compiler import import_model
    from tatva.runner import ExecutionEnvironment, compile_model, reference_output, run_and_measure

    pass_key = ",".join(sorted(passes or [])) + ":" + ",".join(sorted(configs or []))
    target_key = variant.name if hasattr(variant, "name") else str(variant)

    cached_res = GLOBAL_SESSION_CACHE.get_artifact(onnx_path, pass_key, target_key)
    if cached_res is not None:
        return cached_res

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
        fused_model_ir = fuse_attention_softmax(model_ir)
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
            opt_model_ir = fuse_attention_softmax(opt_model_ir)
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
