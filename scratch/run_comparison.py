import os
import sys

# Ensure src/ is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tatva.compiler import TARGETS
from tatva.optimizer import compare_configs


def main() -> None:
    model_path = "models/model.onnx"
    variant = TARGETS["RV64GC"]

    print("Running simulation and measuring latency for baseline vs fused...")
    res = compare_configs(model_path, variant, ["baseline", "fused"])

    print("\nBaseline raw output:")
    print(res["results"]["baseline"]["latency"].raw_output)

    print("\nFused raw output:")
    print(res["results"]["fused"]["latency"].raw_output)

    comp = res["comparison"]
    print("\n=== Config Comparison Results ===")
    print(f"Baseline Latency (Mean):  {res['results']['baseline']['latency'].mean_ms:.4f} ms")
    print(f"Fused Latency (Mean):     {comp['fused_mean_ms']:.4f} ms")
    print(f"Fused Latency (Median):   {comp['fused_median_ms']:.4f} ms")
    print(f"Fused Latency (p95):      {comp['fused_p95_ms']:.4f} ms")
    print(f"Mean Latency Delta:       {comp['fused_mean_delta_ms']:.4f} ms")
    print(f"Median Latency Delta:     {comp['fused_median_delta_ms']:.4f} ms")
    print(f"p95 Latency Delta:        {comp['fused_p95_delta_ms']:.4f} ms")
    print(f"Accuracy Delta (MSE):     {comp['fused_accuracy_delta_mse']:.6f}")
    print(f"Accuracy within Tol:      {comp['fused_accuracy_ok']}")

    if "error_fused" in comp:
        print(f"Error Message:            {comp['error_fused']}")


if __name__ == "__main__":
    main()
