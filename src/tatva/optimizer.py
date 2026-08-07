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


def fuse_attention_softmax(model_ir: ModelIR) -> ModelIR:
    """
    Fuses attention and softmax subgraphs into a custom optimized register-based single-pass
    implementation. Applies the optimization only if the attention/softmax bottleneck pattern
    is detected.
    """
    from tatva.compiler import analyze_graph

    # Analyse the module to see if the attention/softmax bottleneck exists
    report = analyze_graph(model_ir)
    if not report.has_transformer_bottleneck:
        # Return untouched if pattern not detected
        return model_ir

    # Copy metadata and enable softmax_optimized flag
    metadata = model_ir.metadata.copy()
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
    from tatva.runner import ExecutionEnvironment, compile_model, run_and_measure

    pass_key = ",".join(sorted(passes or [])) + ":" + ",".join(sorted(configs or []))
    target_key = variant.name if hasattr(variant, "name") else str(variant)

    cached_res = GLOBAL_SESSION_CACHE.get_artifact(onnx_path, pass_key, target_key)
    if cached_res is not None:
        return cached_res

    model_ir = import_model(onnx_path)
    results = {}

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

    comparison = {}
    if "baseline" in results:
        bl_data = results["baseline"]
        bl_logits_arr = np.array(bl_data["logits"])

        if "quantized" in results:
            quant_data = results["quantized"]
            quant_logits_arr = np.array(quant_data["logits"])

            # Calculate Mean Squared Error (MSE) accuracy delta
            mse_quant = float(np.mean((bl_logits_arr - quant_logits_arr) ** 2))
            accuracy_tolerance = 0.05
            accuracy_ok_quant = mse_quant <= accuracy_tolerance

            comparison.update({
                "baseline_mean_ms": bl_data["latency"].mean_ms,
                "quantized_mean_ms": quant_data["latency"].mean_ms,
                "latency_delta_ms": quant_data["latency"].mean_ms - bl_data["latency"].mean_ms,
                "accuracy_delta_mse": mse_quant,
                "accuracy_ok": accuracy_ok_quant,
                "tolerance": accuracy_tolerance,
            })

            if not accuracy_ok_quant:
                comparison["error"] = (
                    f"Accuracy degradation exceeds tolerance ({mse_quant:.5f} > {accuracy_tolerance})."
                )

        if "fused" in results:
            fused_data = results["fused"]
            fused_logits_arr = np.array(fused_data["logits"])

            # Calculate Mean Squared Error (MSE) accuracy delta for fusion
            mse_fused = float(np.mean((bl_logits_arr - fused_logits_arr) ** 2))
            # Schraudolph's exponential approximation has slightly more numerical error,
            # but is expected to fall well within our 0.05 limit (typically ~0.0001)
            accuracy_tolerance = 0.05
            accuracy_ok_fused = mse_fused <= accuracy_tolerance

            comparison.update({
                "fused_mean_ms": fused_data["latency"].mean_ms,
                "fused_median_ms": fused_data["latency"].median_ms,
                "fused_p95_ms": fused_data["latency"].p95_ms,
                "fused_mean_delta_ms": fused_data["latency"].mean_ms - bl_data["latency"].mean_ms,
                "fused_median_delta_ms": fused_data["latency"].median_ms - bl_data["latency"].median_ms,
                "fused_p95_delta_ms": fused_data["latency"].p95_ms - bl_data["latency"].p95_ms,
                "fused_accuracy_delta_mse": mse_fused,
                "fused_accuracy_ok": accuracy_ok_fused,
            })

            if not accuracy_ok_fused:
                comparison["error_fused"] = (
                    f"Fused accuracy degradation exceeds tolerance ({mse_fused:.5f} > {accuracy_tolerance})."
                )

        if "optimized" in results:
            opt_data = results["optimized"]
            opt_logits_arr = np.array(opt_data["logits"])

            # Calculate Mean Squared Error (MSE) accuracy delta for optimized configuration
            mse_opt = float(np.mean((bl_logits_arr - opt_logits_arr) ** 2))
            accuracy_tolerance = 0.05
            accuracy_ok_opt = mse_opt <= accuracy_tolerance

            comparison.update({
                "opt_mean_ms": opt_data["latency"].mean_ms,
                "opt_median_ms": opt_data["latency"].median_ms,
                "opt_p95_ms": opt_data["latency"].p95_ms,
                "opt_mean_delta_ms": opt_data["latency"].mean_ms - bl_data["latency"].mean_ms,
                "opt_median_delta_ms": opt_data["latency"].median_ms - bl_data["latency"].median_ms,
                "opt_p95_delta_ms": opt_data["latency"].p95_ms - bl_data["latency"].p95_ms,
                "opt_accuracy_delta_mse": mse_opt,
                "opt_accuracy_ok": accuracy_ok_opt,
            })

            if not accuracy_ok_opt:
                comparison["error_opt"] = (
                    f"Optimized accuracy degradation exceeds tolerance ({mse_opt:.5f} > {accuracy_tolerance})."
                )

    final_res = {"results": results, "comparison": comparison}
    GLOBAL_SESSION_CACHE.put_artifact(onnx_path, pass_key, target_key, final_res)
    return final_res
