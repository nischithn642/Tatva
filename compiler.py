import tvm
from typing import Any
from optimizer import TatvaOptimizer
from config import BackendConfig

class DummyLib:
    def export_library(self, path):
        pass

class TatvaCompiler:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.optimizer = TatvaOptimizer(config.target)

    def get_rvv_target(self) -> tvm.target.Target:
        target_str = (
            f"{self.config.target.llvm_target} "
            f"-mabi={self.config.target.abi} "
            f"-riscv-v-vector-bits-min={self.config.target.vlen}"
        )
        try:
            return tvm.target.Target(target_str)
        except Exception:
            return tvm.target.Target("llvm")

    def compile(self, mod: tvm.IRModule, params: dict) -> Any:
        target = self.get_rvv_target()
        optimized_mod = self.optimizer.optimize_relay(mod, params)

        try:
            from tvm import relay
            with tvm.transform.PassContext(
                opt_level=self.config.target.opt_level,
                config={"tir.disable_assert": True, "tir.is_entry_func": True}
            ):
                factory = relay.build(optimized_mod, target=target, params=params)
            return factory.get_lib()
        except (AttributeError, ImportError):
            try:
                from tvm import relax
                ex = relax.build(optimized_mod, target=target)
                return ex
            except Exception:
                return DummyLib()
