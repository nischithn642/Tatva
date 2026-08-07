"""
Backend AI Agent Engine for Project Scaffolding Assistant.

Supports:
  - 6-file complete project scaffold generation (models, requirements, train, tests, config, README)
  - Automated AST syntax validation before human review
  - Multi-turn conversational iteration with chat history
  - Cumulative session cost tracking
  - Strict in-memory-only safety gate (no disk writes without explicit approval)
"""

import ast
import json
import os
from typing import Any

from scaffolding.config import ScaffoldingConfig
from scaffolding.executor import ScaffoldingExecutor
from scaffolding.llm_provider import LLMProvider
from scaffolding.logger import ScaffoldingLogger
from scaffolding.loop_agent import LoopAgent


def _ast_check(filename: str, content: str) -> tuple[bool, str]:
    """
    Run ast.parse() on Python file content.
    Returns (passed: bool, message: str).
    """
    if not filename.endswith(".py"):
        return True, "N/A (non-Python file)"
    try:
        ast.parse(content)
        return True, "PASSED"
    except SyntaxError as e:
        return False, f"FAILED — {e.msg} at line {e.lineno}"


class ScaffoldingAgent:
    """
    Decoupled AI Scaffolding Agent. Generates a complete, runnable RISC-V starter project.
    Supports multi-turn iteration, local Ollama execution, and autonomous closed-loop self-correction.
    """

    def __init__(self, config: ScaffoldingConfig | None = None) -> None:
        self.config = config or ScaffoldingConfig.load()
        self.logger = ScaffoldingLogger(self.config.log_file)
        self.chat_history: list[dict[str, str]] = []
        self.cumulative_cost_usd: float = 0.0
        self.llm_provider = LLMProvider()
        self.executor = ScaffoldingExecutor()
        self.loop_agent = LoopAgent(self.llm_provider, self.executor)

    def reset_session(self) -> None:
        """Reset conversation history and session cost for a fresh session."""
        self.chat_history = []
        self.cumulative_cost_usd = 0.0

    def generate(
        self,
        prompt_text: str,
        model_name: str | None = None,
        custom_api_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate structured scaffolding files from user prompt.
        Appends to multi-turn chat history. Returns file list with AST validation results.
        """
        model_name = model_name or self.config.default_model
        api_key = custom_api_key or self.config.get_api_key()
        estimated_cost = self.config.estimate_cost(prompt_text, model_name)

        if not prompt_text.strip():
            raise ValueError("Prompt text cannot be empty.")

        self.chat_history.append({"role": "user", "content": prompt_text})

        # Try Anthropic API if key is present
        if api_key and "Anthropic" in model_name:
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
                messages_payload = [
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                    for msg in self.chat_history
                ]
                from tatva.config import ANTHROPIC_MODEL

                response = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=4096,
                    system=self.config.system_prompt,
                    messages=messages_payload,
                )
                raw_text = response.content[0].text
                self.chat_history.append({"role": "assistant", "content": raw_text})

                parsed_result = self._parse_json_response(raw_text, estimated_cost)
                actual_cost = self._compute_cost(
                    response.usage.input_tokens, response.usage.output_tokens, model_name
                )
                self.cumulative_cost_usd += actual_cost
                parsed_result["cumulative_cost_usd"] = self.cumulative_cost_usd

                self.logger.log_generation_attempt(
                    prompt_text=prompt_text,
                    model_name=model_name,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    estimated_cost_usd=estimated_cost,
                    accepted_by_user=False,
                )
                self._run_ast_validation(parsed_result)
                return parsed_result
            except Exception as e:
                self.logger.log_generation_attempt(
                    prompt_text=prompt_text,
                    model_name=model_name,
                    input_tokens=len(prompt_text) // 4,
                    output_tokens=0,
                    estimated_cost_usd=estimated_cost,
                    accepted_by_user=False,
                    error=str(e),
                )
                result = self._generate_fallback_template(
                    prompt_text, estimated_cost, note=f"API Note: {e}"
                )
                self._run_ast_validation(result)
                self.cumulative_cost_usd += estimated_cost
                result["cumulative_cost_usd"] = self.cumulative_cost_usd
                return result

        # Deterministic offline fallback template
        result = self._generate_fallback_template(prompt_text, estimated_cost)
        self.logger.log_generation_attempt(
            prompt_text=prompt_text,
            model_name=model_name,
            input_tokens=len(prompt_text) // 4,
            output_tokens=600,
            estimated_cost_usd=estimated_cost,
            accepted_by_user=False,
        )
        self._run_ast_validation(result)
        self.cumulative_cost_usd += estimated_cost
        result["cumulative_cost_usd"] = self.cumulative_cost_usd
        return result

    def iterate(
        self,
        follow_up_prompt: str,
        model_name: str | None = None,
        custom_api_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Multi-turn: append a follow-up instruction to existing history and regenerate.
        All safety gates remain in place — result is still in-memory only.
        """
        return self.generate(follow_up_prompt, model_name, custom_api_key)

    def _compute_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Compute actual API cost from token counts."""
        input_rate = 3.0 / 1_000_000  # $3 per M input tokens (Claude 3.5 Sonnet)
        output_rate = 15.0 / 1_000_000  # $15 per M output tokens
        return round(input_tokens * input_rate + output_tokens * output_rate, 6)

    def _run_ast_validation(self, result: dict[str, Any]) -> None:
        """
        Run ast.parse() on every .py file in the result.
        Adds 'ast_check' field to each file_info dict: {'passed': bool, 'message': str}.
        """
        for file_info in result.get("files", []):
            passed, msg = _ast_check(file_info["path"], file_info["content"])
            file_info["ast_check"] = {"passed": passed, "message": msg}

    def _parse_json_response(self, text: str, estimated_cost: float) -> dict[str, Any]:
        """Parse raw LLM response text into structured dictionary."""
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx : end_idx + 1]
            data = json.loads(json_str)
            if "files" in data:
                data["estimated_cost_usd"] = estimated_cost
                return data

        raise ValueError("Response did not contain valid JSON files payload.")

    def _load_doc_context(self) -> str:
        """
        Load read-only context from Tatva's BASELINE.md and OPTIMIZATION.md documentation.
        """
        ctx_parts = []
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for doc_name in ["BASELINE.md", "OPTIMIZATION.md"]:
            doc_path = os.path.join(project_root, doc_name)
            if os.path.exists(doc_path):
                try:
                    with open(doc_path, encoding="utf-8") as f:
                        text = f.read()
                        ctx_parts.append(f"--- {doc_name} ---\n{text[:1500]}")
                except Exception:
                    pass
        return "\n\n".join(ctx_parts)

    def _generate_fallback_template(
        self, prompt_text: str, estimated_cost: float, note: str = "", model_context: str = ""
    ) -> dict[str, Any]:
        """
        Generate complete deterministic 8-file template project for offline execution or fallback.
        Files: models/classifier.py, requirements.txt, train.py, preprocess.py, Dockerfile,
               tests/test_model.py, config/tatva_config.json, README.md
        """
        proj_name = "riscv_transformer_starter"
        target = "RV32IMC" if "rv32imc" in prompt_text.lower() or "rv32" in prompt_text.lower() else "RV64GCV"

        files = [
            {
                "path": "models/classifier.py",
                "content": '''\
"""
PyTorch Transformer Classifier — RISC-V ONNX Export.
Generated by TATVA Project Scaffolding Assistant.
"""

import torch
import torch.nn as nn


class RISCVTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 32, num_classes: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=64, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.encoder_layer(x)
        logits = self.fc(x[:, 0, :])
        return logits


def export_onnx(output_path: str = "models/model.onnx") -> None:
    """Export model to ONNX format (opset 11) for TATVA compilation."""
    model = RISCVTransformerClassifier()
    model.eval()
    dummy_ids = torch.randint(0, 1000, (1, 32))
    dummy_mask = torch.ones((1, 32), dtype=torch.int64)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=11,
    )
    print(f"[OK] Exported ONNX model to {output_path}")


if __name__ == "__main__":
    export_onnx()
''',
            },
            {
                "path": "requirements.txt",
                "content": """\
# TATVA RISC-V Starter Project — Python Dependencies
torch>=2.0.0
onnx>=1.14.0
onnxruntime>=1.16.0
numpy>=1.24.0
pytest>=7.0.0
""",
            },
            {
                "path": "preprocess.py",
                "content": '''\
"""
Data Preprocessing & Augmentation Pipeline for RISC-V Model Training.
Generated by TATVA Project Scaffolding Assistant.
"""

import numpy as np
import torch


def apply_data_augmentation(tokens: np.ndarray, mask_prob: float = 0.15) -> np.ndarray:
    """Apply random token masking augmentation for robust transformer training."""
    augmented = tokens.copy()
    mask = np.random.rand(*tokens.shape) < mask_prob
    augmented[mask] = 0  # 0 is the [MASK] token
    return augmented


def prepare_features(text_samples: list[str], max_len: int = 32) -> torch.Tensor:
    """Convert raw text samples into padded tensor features."""
    encoded = []
    for sample in text_samples:
        tokens = [ord(c) % 1000 for c in sample[:max_len]]
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        encoded.append(tokens)
    return torch.tensor(encoded, dtype=torch.int64)


if __name__ == "__main__":
    samples = ["hello riscv", "tatva optimization studio", "transformer compiler"]
    features = prepare_features(samples)
    print(f"[PREPROCESS] Prepared feature batch shape: {features.shape}")
''',
            },
            {
                "path": "train.py",
                "content": '''\
"""
Training Script Skeleton for RISC-V Transformer Classifier with Data Augmentation.
Generated by TATVA Project Scaffolding Assistant.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from models.classifier import RISCVTransformerClassifier
from preprocess import apply_data_augmentation, prepare_features


def train(
    num_epochs: int = 5,
    batch_size: int = 4,
    vocab_size: int = 1000,
    seq_len: int = 32,
    num_classes: int = 5,
    lr: float = 1e-3,
) -> None:
    """Train classifier on preprocessed/augmented data and export final ONNX model."""
    model = RISCVTransformerClassifier(vocab_size=vocab_size, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"[TRAIN] Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        inputs_np = torch.randint(0, vocab_size, (batch_size, seq_len)).numpy()
        augmented_np = apply_data_augmentation(inputs_np, mask_prob=0.10)
        inputs = torch.from_numpy(augmented_np).long()
        masks = torch.ones((batch_size, seq_len), dtype=torch.int64)
        labels = torch.randint(0, num_classes, (batch_size,))

        optimizer.zero_grad()
        outputs = model(inputs, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"  Epoch [{epoch + 1}/{num_epochs}]  Loss: {loss.item():.4f}")

    print("[TRAIN] Training complete.")
    from models.classifier import export_onnx
    export_onnx("models/model.onnx")
    print("[TRAIN] Run: tatva optimize models/model.onnx --target RV64GCV")


if __name__ == "__main__":
    train()
''',
            },
            {
                "path": "Dockerfile",
                "content": """\
# Dockerfile for TATVA RISC-V Cross-Compilation Environment
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-riscv64-unknown-elf \
    qemu-system-riscv64 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
""",
            },
            {
                "path": "tests/test_model.py",
                "content": '''\
"""
Smoke Tests for RISC-V Transformer Classifier.
Confirms the model loads and runs a forward pass without error.
Generated by TATVA Project Scaffolding Assistant.
"""

import pytest
import torch

from models.classifier import RISCVTransformerClassifier


def test_model_forward_pass():
    """Model forward pass produces correct output shape."""
    model = RISCVTransformerClassifier(vocab_size=100, hidden_dim=16, num_classes=3)
    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, 100, (1, 16))
        attention_mask = torch.ones((1, 16), dtype=torch.int64)
        logits = model(input_ids, attention_mask)
    assert logits.shape == (1, 3), f"Unexpected output shape: {logits.shape}"


def test_model_output_is_finite():
    """Model output values should all be finite (no NaN/Inf)."""
    model = RISCVTransformerClassifier(vocab_size=100, hidden_dim=16, num_classes=3)
    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, 100, (2, 8))
        attention_mask = torch.ones((2, 8), dtype=torch.int64)
        logits = model(input_ids, attention_mask)
    assert torch.isfinite(logits).all(), "Model output contains NaN or Inf values"


def test_model_batch_independence():
    """Batch size 1 and batch size 4 should produce valid outputs."""
    model = RISCVTransformerClassifier()
    model.eval()
    for bs in [1, 4]:
        with torch.no_grad():
            ids = torch.randint(0, 1000, (bs, 32))
            mask = torch.ones((bs, 32), dtype=torch.int64)
            out = model(ids, mask)
        assert out.shape[0] == bs
''',
            },
            {
                "path": "config/tatva_config.json",
                "content": json.dumps(
                    {
                        "project_name": proj_name,
                        "target_variant": target,
                        "input_shapes": {
                            "input_ids": [1, 32],
                            "attention_mask": [1, 32],
                        },
                        "passes": ["fuse", "quantize"],
                        "optimization_level": 3,
                        "output_dir": "build_opt",
                    },
                    indent=2,
                ),
            },
            {
                "path": "README.md",
                "content": f"""\
# {proj_name.upper()} — TATVA RISC-V Starter Project

> Generated by [TATVA Project Scaffolding Assistant](https://tatva.dev)
> Prompt: *"{prompt_text}"*
> Target: `{target}`
{f"> Context: {model_context}" if model_context else ""}
{f"> Note: {note}" if note else ""}

## Project Structure

```
{proj_name}/
├── models/
│   ├── classifier.py      # PyTorch model definition + ONNX exporter
│   └── model.onnx         # Generated after running models/classifier.py
├── preprocess.py          # Data preprocessing & augmentation
├── train.py               # Training loop script
├── Dockerfile             # Container build recipe
├── tests/
│   └── test_model.py      # Smoke tests (forward pass, finite outputs)
├── config/
│   └── tatva_config.json  # TATVA optimization pipeline config
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Preprocessing & Augmentation
```bash
python preprocess.py
```

### 3. Train & Export ONNX Model
```bash
python train.py
# Writes: models/model.onnx
```

### 4. Run Tests
```bash
pytest tests/test_model.py -v
```

### 5. Optimize with TATVA
```bash
tatva optimize models/model.onnx --target {target}
```

---
*Generated by TATVA — Bare-Metal RISC-V Model Optimization Studio*
""",
            },
        ]

        return {
            "project_name": proj_name,
            "estimated_cost_usd": estimated_cost,
            "files": files,
            "note": note,
        }

    def write_to_disk(self, target_dir: str, files_data: list[dict[str, str]]) -> list[str]:
        """
        STRICT SAFETY GATE: Writes files to target_dir ONLY when explicitly invoked by user click.
        Returns list of absolute file paths created.
        """
        created_paths = []
        os.makedirs(target_dir, exist_ok=True)

        for file_info in files_data:
            rel_path = file_info["path"]
            content = file_info["content"]

            abs_path = os.path.normpath(os.path.join(target_dir, rel_path))
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)

            created_paths.append(abs_path)

        return created_paths

