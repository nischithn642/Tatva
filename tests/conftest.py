from pathlib import Path

import pytest

from tatva._cache import clear_cache
from tatva.runner import find_qemu, find_riscv_gcc


@pytest.fixture(autouse=True)
def isolated_session_cache():
    """
    Give every test a cold session cache.

    GLOBAL_SESSION_CACHE is a module-level singleton keyed on file content, so without
    this a test silently inherits the imported IR and compiled artifacts of whichever
    tests ran before it. That makes a failure depend on the rest of the suite, which is
    the single worst property a test can have.
    """
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path, monkeypatch):
    """
    Point the scaffolding config at a throwaway directory.

    Without this, ScaffoldingConfig.save() writes to the real per-user location
    (%APPDATA%/tatva on Windows, ~/.config/tatva elsewhere) and load() reads it back on
    the next run. That did happen: the suite left a config behind naming a model that no
    longer exists, and a later test asserting on the *default* model failed against a
    file it had written itself weeks earlier.
    """
    monkeypatch.setenv("TATVA_CONFIG_DIR", str(tmp_path / "config"))
    yield


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
    _gcc_name, gcc_path = find_riscv_gcc()
    if not gcc_path:
        missing.append("RISC-V GCC")

    _qemu_name, qemu_path = find_qemu(64)
    if not qemu_path:
        missing.append("QEMU")

    if missing:
        pytest.skip(f"Skipping integration test: missing toolchain components -> {', '.join(missing)}")
