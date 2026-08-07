"""
Tests for TatvaPyBridge, the js_api object the PyWebView GUI calls into.

Every method here is reachable from JavaScript in website/index.html. Nothing in that
HTML can catch a Python exception -- a method that raises leaves the button spinning
forever with no error anywhere the user can see. So the contract these tests enforce is
mostly "returns a dict, always, even when everything it depends on is missing".
"""

import json
from unittest.mock import patch

import pytest

from tatva import __version__
from tatva.gui import TatvaPyBridge


@pytest.fixture
def bridge() -> TatvaPyBridge:
    return TatvaPyBridge()


@pytest.mark.unit
def test_build_info_matches_the_package_version(bridge) -> None:
    """
    The badge in the GUI must report the version the package actually is.

    website/index.html used to hardcode "v1.2.0-gemini" in five places while
    pyproject.toml said 0.2.0. This is the wire that keeps them honest.
    """
    info = bridge.get_build_info()

    assert info["version"] == f"v{__version__}"
    assert info["label"] == f"v{__version__}"


@pytest.mark.unit
def test_bridge_results_survive_json_serialization(bridge) -> None:
    """
    PyWebView marshals return values through JSON. A numpy scalar or a Path in there
    fails at the bridge boundary, not in Python, so it is invisible in a traceback.
    """
    for result in (
        bridge.get_build_info(),
        bridge.get_toolchain_health(),
        bridge.scan_hardware_boards(),
        bridge.validate_model_file("does/not/exist.onnx"),
        bridge.verify_toolchain_configuration("NOT_A_TARGET"),
    ):
        json.dumps(result)


@pytest.mark.unit
def test_toolchain_health_always_has_every_key_the_gui_reads(bridge) -> None:
    """
    The GUI indexes these directly. A missing key is a JS TypeError, not a red badge.
    """
    health = bridge.get_toolchain_health()

    for key in ("gcc", "gcc_name", "gcc_path", "qemu", "qemu_name", "qemu_path", "status_badge"):
        assert key in health, f"get_toolchain_health() dropped '{key}'"

    assert isinstance(health["gcc"], bool)
    assert isinstance(health["qemu"], bool)


@pytest.mark.unit
def test_toolchain_health_reports_failure_rather_than_a_green_badge(bridge) -> None:
    """
    If probing itself blows up, the answer is "I could not tell", never "all good".
    """
    with patch("scaffolding.executor.ToolchainManager.get_health_status", side_effect=OSError("boom")):
        health = bridge.get_toolchain_health()

    assert health["gcc"] is False
    assert health["qemu"] is False
    assert "boom" in health["error"]


@pytest.mark.unit
def test_validate_model_file_rejects_a_missing_path(bridge) -> None:
    res = bridge.validate_model_file("no/such/model.onnx")

    assert res["valid"] is False
    assert "no/such/model.onnx" in res["error"]


@pytest.mark.unit
def test_validate_model_file_rejects_an_empty_path(bridge) -> None:
    """The GUI passes "" when the user has not picked a file yet."""
    res = bridge.validate_model_file("")

    assert res["valid"] is False
    assert res["error"]


@pytest.mark.unit
def test_validate_model_file_reads_the_real_graph(bridge, baseline_model_path) -> None:
    """
    layer_count and has_bottleneck come from the ONNX graph, not from a guess.
    """
    res = bridge.validate_model_file(str(baseline_model_path))

    assert res["valid"] is True
    assert res["framework"] == "ONNX IR Model"
    assert res["layer_count"].endswith("Ops")
    assert int(res["layer_count"].split()[0]) > 0
    assert isinstance(res["has_bottleneck"], bool)
    assert res["sha256"].startswith("0x")
    assert res["size_mb"] > 0
    assert res["error"] == ""


@pytest.mark.unit
def test_validate_model_file_says_unknown_rather_than_inventing_a_layer_count(bridge, tmp_path) -> None:
    """
    A non-ONNX file cannot be parsed, so the op count stays "unknown" and the
    bottleneck verdict stays None. Reporting a plausible number here would be worse
    than reporting nothing.
    """
    fake = tmp_path / "weights.pt"
    fake.write_bytes(b"not really a torchscript archive")

    res = bridge.validate_model_file(str(fake))

    assert res["valid"] is True
    assert res["framework"] == "PyTorch TorchScript"
    assert res["layer_count"] == "unknown"
    assert res["has_bottleneck"] is None


@pytest.mark.unit
def test_validate_model_file_flags_an_unparseable_onnx_instead_of_claiming_success(bridge, tmp_path) -> None:
    corrupt = tmp_path / "broken.onnx"
    corrupt.write_bytes(b"\x00\x01\x02 definitely not a protobuf")

    res = bridge.validate_model_file(str(corrupt))

    assert res["valid"] is True  # the file exists and is readable
    assert res["layer_count"] == "unknown"
    assert "graph could not be parsed" in res["error"]
    assert res["status"] == "Loaded (graph unreadable)"


@pytest.mark.unit
def test_verify_toolchain_configuration_names_the_targets_it_knows(bridge) -> None:
    res = bridge.verify_toolchain_configuration("RV128GQ")

    assert res["status"] == "error"
    assert "RV128GQ" in res["error"]
    assert "RV64GC" in res["error"]


@pytest.mark.unit
def test_verify_toolchain_configuration_picks_the_right_qemu_bitness(bridge) -> None:
    """
    A 32-bit target must probe qemu-system-riscv32. Getting this backwards produced a
    green checkmark followed by a boot that hangs forever.
    """
    with patch("tatva.runner.find_riscv_gcc", return_value=("gcc", "/fake/gcc")), \
         patch("tatva.runner.find_qemu", return_value=("qemu", "")) as mock_qemu:
        res = bridge.verify_toolchain_configuration("RV32IMC")

    mock_qemu.assert_called_once_with(32)
    assert res["status"] == "error"
    assert "qemu-system-riscv32" in res["error"]


@pytest.mark.unit
def test_analyze_model_reports_a_missing_file_without_raising(bridge) -> None:
    res = bridge.analyze_model("no/such/model.onnx")

    assert "error" in res
    assert "no/such/model.onnx" in res["error"]


@pytest.mark.unit
def test_analyze_model_histogram_uses_the_documented_op_names(bridge, baseline_model_path) -> None:
    """
    The histogram must spell operators the same way SUPPORTED_OPS does. It used to keep
    TVM's "relax." prefix while the support check stripped it, so the GUI listed
    "relax.nn.softmax" against a supported-op list containing "nn.softmax".
    """
    res = bridge.analyze_model(str(baseline_model_path))

    assert "error" not in res
    assert res["total_ops"] > 0
    assert res["op_histogram"]
    assert not any(op.startswith("relax.") for op in res["op_histogram"])


@pytest.mark.unit
def test_ask_assistant_refuses_an_empty_prompt(bridge) -> None:
    res = bridge.ask_assistant("")

    assert res["success"] is False
    assert res["error"] == "Empty prompt."


@pytest.mark.unit
def test_ask_assistant_says_no_llm_is_configured_rather_than_faking_a_reply(bridge) -> None:
    """
    This panel used to answer any question with one canned sentence about "Gemini".
    There is no Gemini integration in this project; with nothing reachable it must
    admit that.
    """
    with patch.object(TatvaPyBridge, "get_ollama_models", return_value=[]):
        res = bridge.ask_assistant("How do I target RV32IMC?")

    assert res["success"] is False
    assert res["reply"] == ""
    assert "No LLM is configured" in res["error"]


@pytest.mark.unit
def test_ask_assistant_defaults_to_the_first_local_ollama_model(bridge) -> None:
    with patch.object(TatvaPyBridge, "get_ollama_models", return_value=["qwen2.5-coder:7b"]), \
         patch("scaffolding.llm_provider.LLMProvider.query", return_value=("42 cycles", 0.0)) as mock_query:
        res = bridge.ask_assistant("How many cycles?")

    assert res["success"] is True
    assert res["reply"] == "42 cycles"
    assert res["cost_usd"] == 0.0
    assert mock_query.call_args.kwargs["model_name"] == "Ollama: qwen2.5-coder:7b (Local / Free)"


@pytest.mark.unit
def test_ask_assistant_returns_provider_errors_instead_of_raising(bridge) -> None:
    with patch.object(TatvaPyBridge, "get_ollama_models", return_value=["qwen2.5-coder:7b"]), \
         patch("scaffolding.llm_provider.LLMProvider.query", side_effect=RuntimeError("HTTP 429 Rate Limit Exceeded")):
        res = bridge.ask_assistant("Anything")

    assert res["success"] is False
    assert "429" in res["error"]


@pytest.mark.unit
def test_get_ollama_models_returns_a_list_when_nothing_is_running(bridge) -> None:
    with patch("scaffolding.llm_provider.get_local_ollama_models", side_effect=OSError("connection refused")):
        assert bridge.get_ollama_models() == []


@pytest.mark.unit
def test_fetch_nvidia_models_without_a_key_reports_failure(bridge, monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("TATVA_NVIDIA_KEY", raising=False)

    res = bridge.fetch_nvidia_models("")

    assert res["success"] is False
    assert res["models"] == []
    assert res["error"]
    # get_nvidia_models is the thin list-returning wrapper over the same call.
    assert bridge.get_nvidia_models("") == []


@pytest.mark.unit
def test_scan_hardware_boards_does_not_claim_real_silicon(bridge) -> None:
    """
    There is no board-detection code in this project. Every number TATVA prints comes
    from QEMU, and this method is the one place the GUI says so.
    """
    res = bridge.scan_hardware_boards()

    assert res["found"] is False
    assert "SIMULATION" in res["status"].upper()
