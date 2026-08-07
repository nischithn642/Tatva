"""
Audit Logging module for Project Scaffolding Assistant attempts and cost tracking.
"""

import hashlib
import json
import os
import time
from typing import Dict, Any, Optional

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class ScaffoldingLogger:
    """
    Appends audit log records to logs/scaffolding_usage.log.
    """

    def __init__(self, log_rel_path: str = "logs/scaffolding_usage.log") -> None:
        self.log_path = os.path.join(PROJECT_DIR, log_rel_path)

    def log_generation_attempt(
        self,
        prompt_text: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        accepted_by_user: bool = False,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a generation attempt to log file.
        """
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_hash": prompt_hash,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "accepted_by_user": accepted_by_user,
            "error": error,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        return entry
