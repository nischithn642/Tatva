import os
from dataclasses import dataclass, field

@dataclass
class TargetConfig:
    arch: str = "rv64gcv"
    abi: str = "lp64d"
    vlen: int = 128
    llvm_target: str = "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c,+v"
    opt_level: int = 3

@dataclass
class SecurityLimits:
    max_memory_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GB
    timeout_seconds: int = 120

@dataclass
class BackendConfig:
    target: TargetConfig = field(default_factory=TargetConfig)
    limits: SecurityLimits = field(default_factory=SecurityLimits)
    cache_dir: str = os.path.expanduser("~/.cache/tatva")

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
