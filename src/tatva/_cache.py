"""
Session-Level Content-Hash Caching Module for TATVA.

Caches parsed ModelIR objects and compiled build artifacts to prevent redundant
TVM re-imports and re-compilations within a session.
Keys are derived from SHA256 content hashes of source model files, pass configuration
strings, and target architecture variants.
"""

import hashlib
import os
from collections import OrderedDict
from typing import Any


class SessionCache:
    """
    Bounded session cache storing ModelIR instances and compilation build artifacts.
    Uses LRU eviction policy to prevent unbounded memory growth.
    """

    def __init__(self, max_models: int = 16, max_artifacts: int = 32) -> None:
        self.max_models = max_models
        self.max_artifacts = max_artifacts
        self._model_cache: OrderedDict[str, Any] = OrderedDict()
        self._artifact_cache: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def get_file_sha256(file_path: str) -> str:
        """
        Compute SHA256 content hash of a model file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for hashing: {file_path}")

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_model_ir(self, file_path: str) -> Any | None:
        """
        Retrieve cached ModelIR for a file if content SHA256 matches.
        """
        if not os.path.exists(file_path):
            return None

        file_hash = self.get_file_sha256(file_path)
        if file_hash in self._model_cache:
            self._model_cache.move_to_end(file_hash)
            self.hits += 1
            return self._model_cache[file_hash]

        self.misses += 1
        return None

    def put_model_ir(self, file_path: str, model_ir: Any) -> None:
        """
        Cache a parsed ModelIR keyed by content SHA256.
        """
        if not os.path.exists(file_path):
            return

        file_hash = self.get_file_sha256(file_path)
        if file_hash in self._model_cache:
            self._model_cache.move_to_end(file_hash)
        self._model_cache[file_hash] = model_ir

        # LRU Eviction if capacity exceeded
        while len(self._model_cache) > self.max_models:
            self._model_cache.popitem(last=False)

    def get_artifact(self, file_path: str, pass_config: str, target: str) -> dict[str, Any] | None:
        """
        Retrieve cached compilation artifact data if (file_hash, pass_config, target) matches.
        """
        if not os.path.exists(file_path):
            return None

        file_hash = self.get_file_sha256(file_path)
        key = (file_hash, pass_config.strip().lower(), target.strip().upper())

        if key in self._artifact_cache:
            artifact = self._artifact_cache[key]
            # Verify build output directory if specified
            build_dir = artifact.get("build_dir")
            if build_dir is None or os.path.exists(build_dir):
                self._artifact_cache.move_to_end(key)
                self.hits += 1
                return artifact

        self.misses += 1
        return None

    def put_artifact(
        self, file_path: str, pass_config: str, target: str, artifact_data: dict[str, Any]
    ) -> None:
        """
        Cache compilation artifact metadata keyed by (file_hash, pass_config, target).
        """
        if not os.path.exists(file_path):
            return

        file_hash = self.get_file_sha256(file_path)
        key = (file_hash, pass_config.strip().lower(), target.strip().upper())

        if key in self._artifact_cache:
            self._artifact_cache.move_to_end(key)
        self._artifact_cache[key] = artifact_data

        # LRU Eviction if capacity exceeded
        while len(self._artifact_cache) > self.max_artifacts:
            self._artifact_cache.popitem(last=False)

    def clear(self) -> None:
        """
        Clear all cached models and compilation artifacts.
        """
        self._model_cache.clear()
        self._artifact_cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int]:
        """
        Return current cache metrics.
        """
        return {
            "cached_models": len(self._model_cache),
            "cached_artifacts": len(self._artifact_cache),
            "hits": self.hits,
            "misses": self.misses,
        }


# Global session cache instance
GLOBAL_SESSION_CACHE = SessionCache()


def clear_cache() -> None:
    """
    Public utility function to clear global session cache.
    """
    GLOBAL_SESSION_CACHE.clear()
