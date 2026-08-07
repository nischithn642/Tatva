# TATVA M1 Baseline Metrics

> [!IMPORTANT]
> **Environment Verification:**
> All values listed below were measured using **QEMU system-mode simulation** on the host CPU. They do **NOT** represent real, physical RISC-V hardware execution.

## Model Details
* **Source Model:** `model.onnx`
* **Architecture:** Transformer (BERT-derived)
* **Configuration:** FP32 Baseline Execution (No quantization, standard softmax scheduling)
* **Target Variant:** `RV64GC` (RISC-V Bare-metal, openSBI Virt)
* **GCC Compiler Flags:** `-march=rv64gc -mabi=lp64d -O3`

## Measured Latency Parameters
* **Target CPU frequency:** 100 MHz (Nominal cycle-to-time conversion frequency)
* **Emulation Environment:** QEMU simulated: True
* **Mean Inference Latency:** **`0.7309` ms**
* **Median Inference Latency:** **`0.7309` ms**
* **p95 Inference Latency:** **`0.7309` ms**

---

## Parity Verification Results

Out-of-box correctness was validated by performing host-side ONNX Runtime inference and comparing output logits against target-side RISC-V bare-metal output.

* **Parity Check Output Status:** **PASS**
* **Allowed Tolerance Threshold:** `1e-4`
* **Reference Host Logits (Top 5):** `[0.7062111496925354, 0.21168753504753113, 3.785907054520976e-08, 0.08209286630153656, 8.445604180451483e-06]`
* **RISC-V Bare-Metal Logits (Top 5):** `[0.706211, 0.211687, 0.0, 0.082092, 8e-06]`

---

## Replication Command
```bash
tatva baseline-test models/model.onnx --target RV64GC
```
