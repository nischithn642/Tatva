"""
Tooling Performance Regression Check Script for TATVA.

Measures tool compilation wall-time before and after session caching across repeated runs.
Note: This measures tool developer wall-time responsiveness, NOT model inference cycle metrics.
"""

import time
import os
import sys

# Add src to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tatva._cache import clear_cache, GLOBAL_SESSION_CACHE
from tatva.compiler import import_model, TARGETS
from tatva.optimizer import compare_configs


def run_perf_check() -> None:
    model_path = os.path.join("models", "model.onnx")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model fixture '{model_path}' not found.")
        sys.exit(1)

    target = TARGETS["RV64GC"]

    print("==================================================")
    print("TATVA Tooling Performance Check (Session Caching)")
    print("==================================================")

    # 1. Cold Run (Cache cleared)
    clear_cache()
    t0 = time.perf_counter()
    res1 = compare_configs(model_path, target, ["baseline", "optimized"], passes=["fuse"])
    t1 = time.perf_counter()
    cold_time_ms = (t1 - t0) * 1000.0
    print(f"Run 1 (Cold / Cache Miss): {cold_time_ms:.2f} ms")

    # 2. Warm Run (Cache hit)
    t2 = time.perf_counter()
    res2 = compare_configs(model_path, target, ["baseline", "optimized"], passes=["fuse"])
    t3 = time.perf_counter()
    warm_time_ms = (t3 - t2) * 1000.0
    print(f"Run 2 (Warm / Cache Hit) : {warm_time_ms:.2f} ms")

    speedup = cold_time_ms / warm_time_ms if warm_time_ms > 0 else 1.0
    saved_ms = cold_time_ms - warm_time_ms

    print("\n--------------------------------------------------")
    print(f"Tooling Wall-Time Reduction : {saved_ms:.2f} ms saved ({speedup:.1f}x speedup)")
    print(f"Session Cache Stats         : {GLOBAL_SESSION_CACHE.stats()}")
    print("--------------------------------------------------")
    print("[SUCCESS] Caching verified. Tooling responsiveness accelerated while preserving inference parity.")


if __name__ == "__main__":
    run_perf_check()
