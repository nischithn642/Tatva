"""
Project Scaffolding Assistant Experimental Module.

Decoupled AI-assisted starter code generator for RISC-V targets.
"""

from scaffolding.agent import ScaffoldingAgent
from scaffolding.config import ScaffoldingConfig
from scaffolding.executor import ExecutionResult, ScaffoldingExecutor, ToolchainManager
from scaffolding.llm_provider import LLMProvider
from scaffolding.logger import ScaffoldingLogger
from scaffolding.loop_agent import LoopAgent

__all__ = [
    "ExecutionResult",
    "LLMProvider",
    "LoopAgent",
    "ScaffoldingAgent",
    "ScaffoldingConfig",
    "ScaffoldingExecutor",
    "ScaffoldingLogger",
    "ToolchainManager",
]
