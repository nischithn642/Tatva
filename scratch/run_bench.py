import os
import sys

# Ensure src/ is in the python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tatva.compiler import TARGETS, import_model
from tatva.runner import ExecutionEnvironment, compile_model, run_and_measure


def main() -> None:
    model_path = "models/model.onnx"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Importing model from {model_path}...")
    model_ir = import_model(model_path)

    variant = TARGETS["RV64GC"]
    print(f"Compiling model for target {variant.name}...")
    artifact = compile_model(model_ir, variant, warmup_count=2, timed_count=5)
    print(f"Artifact compiled to: {artifact.elf_path}")

    print("Running simulation and measuring latency...")
    result = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)

    print("\n=== MeasurementResult JSON ===")
    print(result.to_json())


if __name__ == "__main__":
    main()
