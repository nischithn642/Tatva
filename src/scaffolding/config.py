"""
Configuration management for the Project Scaffolding Assistant.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "scaffolding_config.json")


@dataclass
class ScaffoldingConfig:
    provider: str = "NVIDIA NIM"
    nvidia_api_key: str = ""
    selected_model: str = ""
    default_model: str = "Claude 3.5 Sonnet (Anthropic)"
    models: List[str] = field(default_factory=lambda: [
        "Claude 3.5 Sonnet (Anthropic)",
        "DeepSeek-R1 (Local/API)",
        "Custom Endpoint"
    ])
    cost_rates_per_1k_chars: Dict[str, float] = field(default_factory=lambda: {
        "Claude 3.5 Sonnet (Anthropic)": 0.003,
        "DeepSeek-R1 (Local/API)": 0.0005,
        "Custom Endpoint": 0.001
    })
    api_endpoint: str = "https://api.anthropic.com/v1/messages"
    log_file: str = "logs/scaffolding_usage.log"
    system_prompt: str = (
        "You are Tatva's RISC-V Project Scaffolding Assistant. Generate a valid starter project "
        "for RISC-V target architectures. Output MUST be valid JSON."
    )

    @classmethod
    def load(cls) -> "ScaffoldingConfig":
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except Exception:
                return cls()
        return cls()

    def save(self) -> None:
        """Persist configuration settings to disk."""
        try:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "provider": self.provider,
                    "nvidia_api_key": self.nvidia_api_key,
                    "selected_model": self.selected_model,
                    "default_model": self.default_model,
                    "models": self.models,
                    "cost_rates_per_1k_chars": self.cost_rates_per_1k_chars,
                    "api_endpoint": self.api_endpoint,
                    "log_file": self.log_file,
                    "system_prompt": self.system_prompt,
                }, f, indent=2)
        except Exception:
            pass

    def get_api_key(self) -> str:
        """
        Retrieve API key from environment variable via centralized tatva.config.
        """
        from tatva.config import get_anthropic_api_key

        return get_anthropic_api_key() or ""

    def get_nvidia_api_key(self) -> str:
        """
        Retrieve NVIDIA API key from local config or environment variable.
        """
        from tatva.config import get_nvidia_api_key

        return self.nvidia_api_key or get_nvidia_api_key() or ""

    def estimate_cost(self, prompt_text: str, model_name: str) -> float:
        """
        Estimate USD cost based on prompt character length and model rate.
        """
        rate = self.cost_rates_per_1k_chars.get(model_name, 0.001)
        char_count = len(prompt_text)
        return round((char_count / 1000.0) * rate, 5)
