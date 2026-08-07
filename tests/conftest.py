import pytest
from pathlib import Path
from tatva.runner import find_riscv_gcc, find_qemu

@pytest.fixture
def baseline_model_path() -> Path:
    return Path("models/model.onnx")

@pytest.fixture
def quantized_model_path() -> Path:
    return Path("models/model_quant.onnx")

@pytest.fixture
def unsupported_op_model_path() -> Path:
    return Path("models/model_unsupported.onnx")

@pytest.fixture
def pretrained_model_path() -> Path:
    return Path("models/model_pretrained.onnx")

@pytest.fixture
def mlp_model_path() -> Path:
    """
    A non-transformer fixture: one float32 input, an ONNX name that is not a valid
    C identifier, and no softmax. Regenerate with `python models/make_mlp_fixture.py`.
    """
    return Path("models/model_mlp.onnx")

@pytest.fixture
def tolerance() -> float:
    return 0.05

@pytest.fixture
def skip_if_no_toolchain():
    # 1. Check TVM
    try:
        import tvm  # noqa: F401
        from tvm import relax  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Skipping integration test: TVM is not available ({e}).")

    # 2. Check GCC and QEMU
    missing = []
    gcc_name, gcc_path = find_riscv_gcc()
    if not gcc_path:
        missing.append("RISC-V GCC")

    qemu_name, qemu_path = find_qemu(64)
    if not qemu_path:
        missing.append("QEMU")

    if missing:
        pytest.skip(f"Skipping integration test: missing toolchain components -> {', '.join(missing)}")
