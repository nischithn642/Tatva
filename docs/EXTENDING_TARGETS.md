# Tutorial: Extending TATVA Target Architectures

This guide explains step-by-step how to add support for a new RISC-V target architecture variant (e.g. adding `RV64GCV` or `RV32IMAC`) to the TATVA toolchain.

---

## Target Registration Architecture

Target variants in TATVA are defined using the `TargetVariant` data class in `src/tatva/compiler.py`:

```python
@dataclass(frozen=True)
class TargetVariant:
    name: str          # Variant key (e.g. 'RV64GCV')
    march: str         # GCC architecture string (e.g. 'rv64gcv')
    mabi: str          # GCC ABI string (e.g. 'lp64d')
    abi: str           # Human-readable ABI description
    description: str   # Target variant description
    tvm_target: str    # TVM C codegen target string
    has_vector: bool = False   # True if hardware vector extensions are supported
    experimental: bool = False # True if target requires --allow-experimental flag
```

---

## Step-by-Step Target Extension Procedure

### Step 1: Add Entry to `TARGETS` Registry (`src/tatva/compiler.py`)

Open `src/tatva/compiler.py` and append your target variant definition to the `TARGETS` dictionary:

```python
TARGETS: dict[str, TargetVariant] = {
    # Existing target variants...
    "RV64GCV": TargetVariant(
        name="RV64GCV",
        march="rv64gcv",
        mabi="lp64d",
        abi="64-bit Hard Float with Vector Extensions",
        description="64-bit RISC-V with IMAFDC and Vector extension 1.0",
        tvm_target="c -keys=cpu -mcpu=generic-rv64",
        has_vector=True,
        experimental=False,
    ),
}
```

---

### Step 2: Configure Cross-Compiler & QEMU Parameters (`src/tatva/runner.py`)

In `src/tatva/runner.py`, update `compile_model` and `run_and_measure` to handle target-specific cross-compiler flags and QEMU emulation parameters:

1. **GCC Compiler Flags:** Ensure `riscv-none-elf-gcc` receives `-march=<march> -mabi=<mabi> -O3`.
2. **QEMU CPU Flag:** Ensure `qemu-system-riscv64` receives appropriate CPU options (e.g., `-cpu rv64,v=true,vext_spec=v1.0`).

```python
# Inside runner.py qemu execution parameters setup:
if variant.name == "RV64GCV":
    qemu_args.extend(["-cpu", "rv64,v=true,vext_spec=v1.0"])
```

---

### Step 3: Verify Target Registry (`tatva targets --verify`)

Run the CLI target verification command to validate your new target registration:

```bash
tatva targets --verify
```

Expected Output:
```text
=== TATVA Target Registry ===
Target Variant: RV64GC (GCC: -march=rv64gc -mabi=lp64d) [Default]
Target Variant: RV32EMC (GCC: -march=rv32emc -mabi=ilp32e)
Target Variant: RV64GCV (GCC: -march=rv64gcv -mabi=lp64d) [Vector Enabled]
[OK] Target variant 'RV64GCV' verified successfully.
```

---

### Step 4: Run Hello-World Compilation & QEMU Baseline Test

Run a baseline test to verify compilation, linking, OpenSBI firmware startup, and QEMU execution:

```bash
tatva baseline-test models/model.onnx --target RV64GCV
```

Expected Output:
```text
=== TATVA Baseline Execution Test ===
Target Variant: RV64GCV
Model:          models/model.onnx
Simulated Latency: 0.73090 ms
Parity Status:     PASS (MSE = 0.000000)
```

---

### Step 5: Add Unit Test Coverage (`tests/test_targets.py`)

Add a unit test in `tests/test_targets.py` verifying that your new target variant is present in the `TARGETS` registry and exposes valid GCC flags:

```python
def test_rv64gcv_target_definition() -> None:
    from tatva.compiler import TARGETS
    assert "RV64GCV" in TARGETS
    target = TARGETS["RV64GCV"]
    assert target.march == "rv64gcv"
    assert target.has_vector is True
```
