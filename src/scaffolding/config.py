"""
Configuration management for the Project Scaffolding Assistant.
"""

import json
import os
from dataclasses import dataclass, field, fields

from tatva.config import ANTHROPIC_MODEL_LABEL


def _config_path() -> str:
    """
    Where the user's scaffolding preferences live.

    Not in the repo. The old location was `<project>/config/scaffolding_config.json`,
    which was committed to git *and* written back by save() -- so the first person to
    type their NVIDIA key into the GUI had it staged for their next commit. It also
    resolves inside site-packages for an installed wheel, which is not writable on most
    systems. TATVA_CONFIG_DIR overrides.
    """
    root = os.environ.get("TATVA_CONFIG_DIR")
    if not root:
        base = (
            os.environ.get("APPDATA")
            or os.environ.get("XDG_CONFIG_HOME")
            or os.path.join(os.path.expanduser("~"), ".config")
        )
        root = os.path.join(base, "tatva")
    return os.path.join(root, "scaffolding_config.json")


# Snapshot for anything that wants to show the user where the file lives. load() and
# save() deliberately call _config_path() again rather than using this, because a path
# frozen at import time ignores TATVA_CONFIG_DIR set later -- which is how the test
# suite ended up writing into the developer's real %APPDATA% and then reading its own
# leftovers back on the next run.
CONFIG_PATH = _config_path()


@dataclass
class ScaffoldingConfig:
    provider: str = "NVIDIA NIM"
    # Session-only. Never written to disk by save(); read from the environment via
    # get_nvidia_api_key() when it is blank.
    nvidia_api_key: str = ""
    selected_model: str = ""
    default_model: str = ANTHROPIC_MODEL_LABEL
    models: list[str] = field(default_factory=lambda: [
        ANTHROPIC_MODEL_LABEL,
        "DeepSeek-R1 (Local/API)",
        "Custom Endpoint"
    ])
    cost_rates_per_1k_chars: dict[str, float] = field(default_factory=lambda: {
        ANTHROPIC_MODEL_LABEL: 0.003,
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
        path = _config_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                # Ignore keys this dataclass no longer has rather than raising TypeError
                # and silently resetting every preference the user set.
                known = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in data.items() if k in known})
            except Exception:
                return cls()
        return cls()

    def save(self) -> None:
        """
        Persist preferences to disk.

        Deliberately does not write nvidia_api_key. A secret in a JSON file is a secret
        that gets committed, backed up and shared; the key belongs in the environment
        (TATVA_NVIDIA_KEY / NVIDIA_API_KEY) or in the session only.
        """
        path = _config_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "provider": self.provider,
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
