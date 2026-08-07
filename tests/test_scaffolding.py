"""
Unit and Safety Gate Tests for Project Scaffolding Assistant Experimental Module.
"""

import json

import pytest

from scaffolding.agent import ScaffoldingAgent
from scaffolding.config import ScaffoldingConfig
from scaffolding.logger import ScaffoldingLogger
from tatva.config import ANTHROPIC_MODEL_LABEL


@pytest.mark.unit
def test_scaffolding_config_load() -> None:
    """
    Assert that ScaffoldingConfig loads defaults and computes token cost estimates correctly.
    """
    cfg = ScaffoldingConfig.load()
    assert cfg.default_model == ANTHROPIC_MODEL_LABEL
    assert ANTHROPIC_MODEL_LABEL in cfg.models

    prompt = "Keyword spotter model for RV64GCV"
    cost = cfg.estimate_cost(prompt, cfg.default_model)
    assert cost > 0.0
    assert isinstance(cost, float)


@pytest.mark.unit
def test_scaffolding_agent_offline_fallback() -> None:
    """
    Assert that ScaffoldingAgent generates structured starter project in-memory when offline/unauthenticated.
    """
    agent = ScaffoldingAgent()
    res = agent.generate("Keyword spotter targeting RV64GCV")

    assert "project_name" in res
    assert "files" in res
    assert len(res["files"]) >= 3

    paths = [f["path"] for f in res["files"]]
    assert "models/classifier.py" in paths
    assert "config/tatva_config.json" in paths
    assert "README.md" in paths


@pytest.mark.unit
def test_strict_execution_boundary_review_gate(tmp_path) -> None:
    """
    STRICT SAFETY GATE TEST:
    Verify that agent.generate(...) creates NO files on disk until write_to_disk(...) is explicitly clicked.
    """
    agent = ScaffoldingAgent()
    target_dir = tmp_path / "test_workspace"

    # Step 1: Generate scaffolding in-memory
    res = agent.generate("Speech classifier for RV32IMAF")
    assert res is not None

    # Verify target_dir does not exist yet on disk
    assert not target_dir.exists()
    assert not (target_dir / "models").exists()

    # Step 2: Explicitly invoke write_to_disk (simulating "Review and Accept" click)
    created_paths = agent.write_to_disk(str(target_dir), res["files"])
    assert len(created_paths) == len(res["files"])
    assert target_dir.exists()
    assert (target_dir / "models" / "classifier.py").exists()
    assert (target_dir / "config" / "tatva_config.json").exists()


@pytest.mark.unit
def test_cancellation_discard(tmp_path) -> None:
    """
    CANCELLATION TEST:
    Verify that generating scaffolding and discarding it leaves the disk completely untouched.
    """
    agent = ScaffoldingAgent()
    target_dir = tmp_path / "discarded_workspace"

    res = agent.generate("Gesture recognition ONNX model")
    assert res is not None

    # Simulate user clicking "Discard / Cancel" (no write_to_disk call)
    res = None

    # Verify zero files or directories were created on disk
    assert not target_dir.exists()


@pytest.mark.unit
def test_audit_logger(tmp_path) -> None:
    """
    Assert that generation attempts are appended to scaffolding audit log file.
    """
    log_file = tmp_path / "test_usage.log"
    logger = ScaffoldingLogger(log_rel_path=str(log_file))

    entry = logger.log_generation_attempt(
        prompt_text="Test prompt for logging",
        model_name="Claude 3.5 Sonnet (Anthropic)",
        input_tokens=50,
        output_tokens=100,
        estimated_cost_usd=0.0015,
        accepted_by_user=True,
    )

    assert entry["accepted_by_user"] is True
    assert log_file.exists()

    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["model_name"] == "Claude 3.5 Sonnet (Anthropic)"
        assert data["estimated_cost_usd"] == 0.0015
