# Tatva Model Fixtures

This directory contains standardized ONNX model fixtures used across unit and integration test suites.

- **`model.onnx`**: Synthetic baseline model containing standard operators (MatMul, Softmax) for regression testing.
- **`model_quant.onnx`**: Dynamically quantized int8 version generated via ONNX Runtime quantization utilities.
- **`model_unsupported.onnx`**: Modified graph fixture injected with an `UnsupportedOpXYZ` operator to test graceful exception gatekeeping.
- **`model_pretrained.onnx`**: Compact model fixture adapted for verification workflows under Apache License 2.0.
