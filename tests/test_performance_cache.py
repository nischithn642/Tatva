"""
Performance & Session Caching Test Suite for TATVA.

Verifies:
1. Lazy module loading (importing cli.py does not load tvm into sys.modules).
2. Session cache hit on repeated identical model imports / compilation runs.
3. Cache busting on input changes (file content hash, pass configuration, or target variant).
4. Cache clearing via clear_cache().
"""

import os
import sys
import tempfile

import pytest

from tatva._cache import GLOBAL_SESSION_CACHE, clear_cache
from tatva.compiler import import_model


@pytest.mark.unit
def test_lazy_import_cli_does_not_load_tvm() -> None:
    """
    LAZY IMPORT TEST:
    Assert that importing tatva.cli does NOT load heavy dependencies (tvm, onnxruntime)
    into sys.modules at top-level module load time.
    """
    modules_to_check = ["tvm", "onnxruntime"]

    # Pop modules temporarily if already imported by earlier tests in the process
    saved_modules = {}
    for mod in modules_to_check:
        if mod in sys.modules:
            saved_modules[mod] = sys.modules.pop(mod)

    try:
        if "tatva.cli" in sys.modules:
            import importlib
            importlib.reload(sys.modules["tatva.cli"])
        else:
            import tatva.cli  # noqa: F401

        for mod_name in modules_to_check:
            assert mod_name not in sys.modules, (
                f"Heavy module '{mod_name}' was loaded into sys.modules upon importing cli.py!"
            )
    finally:
        sys.modules.update(saved_modules)


@pytest.mark.unit
def test_session_cache_hit_and_invalidation() -> None:
    """
    SESSION CACHE & INVALIDATION TEST:
    Assert that:
    - Identical model imports return cached ModelIR.
    - Changing file content (hashing) invalidates model IR cache.
    - Artifact caching invalidates on pass_config or target changes.
    """
    clear_cache()

    # Create dummy ONNX model fixture
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
        # Write dummy ONNX content
        import onnx
        from onnx import TensorProto, helper

        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "test_graph", [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
        ], [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
        ])
        model = helper.make_model(graph, producer_name="test")
        onnx.save(model, tmp_path)

    try:
        # 1. First import (Cache Miss)
        stats_before = GLOBAL_SESSION_CACHE.stats()
        ir1 = import_model(tmp_path)
        stats_after_first = GLOBAL_SESSION_CACHE.stats()

        assert stats_after_first["misses"] > stats_before["misses"]
        assert stats_after_first["cached_models"] == 1

        # 2. Second import (Cache Hit)
        ir2 = import_model(tmp_path)
        stats_after_second = GLOBAL_SESSION_CACHE.stats()

        assert stats_after_second["hits"] > stats_after_first["hits"]
        assert ir1 is ir2

        # 3. Cache Busting on Content Change (save valid updated ONNX model)
        model_v2 = helper.make_model(graph, producer_name="test_v2")
        onnx.save(model_v2, tmp_path)

        ir3 = import_model(tmp_path)
        stats_after_bust = GLOBAL_SESSION_CACHE.stats()

        assert ir3 is not ir1
        assert stats_after_bust["cached_models"] == 2

        # 4. Artifact Cache Invalidation on Pass/Target change
        dummy_result_a = {"status": "ok", "mean_ms": 150.0}

        GLOBAL_SESSION_CACHE.put_artifact(tmp_path, "fuse", "RV64GC", dummy_result_a)

        # Hit on identical config
        hit = GLOBAL_SESSION_CACHE.get_artifact(tmp_path, "fuse", "RV64GC")
        assert hit == dummy_result_a

        # Miss on pass change ("fuse,quantize")
        miss_pass = GLOBAL_SESSION_CACHE.get_artifact(tmp_path, "fuse,quantize", "RV64GC")
        assert miss_pass is None

        # Miss on target change ("RV64GCV")
        miss_target = GLOBAL_SESSION_CACHE.get_artifact(tmp_path, "fuse", "RV64GCV")
        assert miss_target is None

        # 5. Clear Cache Test
        clear_cache()
        stats_cleared = GLOBAL_SESSION_CACHE.stats()
        assert stats_cleared["cached_models"] == 0
        assert stats_cleared["cached_artifacts"] == 0

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
