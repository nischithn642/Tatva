"""
Headless Smoke Tests for TATVA Desktop GUI Application (Milestone M5).

Verifies splash screen lifecycle, main window instantiation, panel notebook creation,
target registry dropdown filtering, and widget event handlers without executing full
heavy compilations.
"""

import contextlib
import tkinter as tk

import pytest

from tatva.compiler import DEFAULT_TARGET, TARGETS
from tatva.gui import TatvaApp, _backend, load_backend_libraries


@pytest.fixture(scope="module")
def gui_app():
    """
    Module-scoped fixture instantiating TatvaApp once for GUI smoke tests.
    """
    if "TARGETS" not in _backend:
        _backend["TARGETS"] = TARGETS
        _backend["DEFAULT_TARGET"] = DEFAULT_TARGET

    try:
        app = TatvaApp(is_test_mode=True)
    except tk.TclError:
        pytest.skip("Tkinter display not available in headless environment.")

    yield app

    with contextlib.suppress(Exception):
        app.destroy()


@pytest.mark.unit
def test_backend_libraries_loading() -> None:
    """
    Assert that heavy backend libraries load into _backend container.
    """
    loaded = False
    err = ""

    def callback(success: bool, error_msg: str) -> None:
        nonlocal loaded, err
        loaded = success
        err = error_msg

    load_backend_libraries(callback)
    assert loaded is True
    assert err == ""
    assert "import_model" in _backend
    assert "TARGETS" in _backend
    assert "verify_target" in _backend


@pytest.mark.unit
def test_gui_main_window_instantiation(gui_app) -> None:
    """
    Assert that TatvaApp main window instantiates all multi-panel tabs cleanly.
    """
    # Assert title and window properties
    assert "TATVA" in gui_app.title()

    # Assert chat view and artifacts notebook exist
    assert hasattr(gui_app, "txt_chat")
    assert hasattr(gui_app, "notebook_art")
    assert hasattr(gui_app, "entry_model")
    assert hasattr(gui_app, "cbo_targets")
    assert hasattr(gui_app, "cb_quantize")
    assert hasattr(gui_app, "cb_fuse_softmax")

    assert hasattr(gui_app, "lbl_badge")
    assert "SIMULATION" in gui_app.lbl_badge.cget("text")


@pytest.mark.unit
def test_gui_target_dropdown_population(gui_app) -> None:
    """
    Assert that target dropdown reflects TARGETS registry and experimental gating works.
    """
    # Default state: allow_exp = False. RV64GCV is behind the gate along with RV32EMC --
    # codegen emits scalar C, so the vector extension is enabled in the ISA string and
    # then never used. Offering it as a production target advertises a speedup that does
    # not exist.
    gui_app.cb_allow_exp.set(False)
    gui_app._update_target_dropdown()
    values_default = gui_app.cbo_targets["values"]
    assert "RV64GC" in values_default
    assert not any("RV64GCV" in v for v in values_default)
    assert not any("RV32EMC" in v for v in values_default)

    # Gated state: allow_exp = True
    gui_app.cb_allow_exp.set(True)
    gui_app._update_target_dropdown()
    values_exp = gui_app.cbo_targets["values"]
    assert "RV64GC" in values_exp
    assert any("RV64GCV" in v for v in values_exp)
    assert any("RV32EMC" in v for v in values_exp)


@pytest.mark.unit
def test_gui_model_entry_and_quant_notice(gui_app) -> None:
    """
    Assert that model entry text and quantization notice toggle handlers work correctly.
    """
    # Model entry test
    gui_app.entry_model.delete(0, tk.END)
    gui_app.entry_model.insert(0, "models/model.onnx")
    assert gui_app.entry_model.get() == "models/model.onnx"

    # Quantization toggle test
    assert gui_app.cb_quantize.get() is False
    gui_app.cb_quantize.set(True)
    gui_app._on_quantize_toggled()
    assert gui_app.cb_quantize.get() is True
