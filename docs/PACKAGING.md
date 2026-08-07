# Tatva Desktop GUI Packaging & Standalone Installer Guide (Milestone M5)

This guide documents the standalone packaging workflow for Tatva Desktop GUI and the Optimization Studio pipeline using PyInstaller.

---

## 1. Executive Summary & Build Artifacts

| Parameter | Metric / Value |
|---|---|
| **Primary Platform** | Windows 10/11 x64 (`.exe`) |
| **Bundling Mechanism** | PyInstaller 6.x (`tatva.spec`) |
| **Build Automation Script** | `python scratch/build_app.py` |
| **Output Directory** | `dist/tatva/` |
| **Executable Path** | `dist/tatva/tatva.exe` |
| **Build Time** | ~80 - 90 seconds |
| **Total Bundle Size** | **105.90 MB** (Includes TVM Relax legalizer, ONNX Runtime C++ DLLs, Scaffolding Agent) |

---

## 2. Decoupled Architecture & External Tool Strategy

The Tatva Desktop GUI application bundles the Python runtime, PyTorch/ONNX IR converters, and TVM Relax legalizer into a single standalone distribution directory.

> [!IMPORTANT]
> **Native Toolchain Isolation Rule:**
> Native cross-compilers (`riscv-none-elf-gcc` / `riscv64-unknown-elf-gcc`) and QEMU emulators (`qemu-system-riscv64` / `qemu-system-riscv32`) are **NOT** bundled inside `tatva.exe`.
> 
> When launched, `tatva.exe` reuses `tatva doctor` and `verify_target` runtime discovery logic to detect native tools on PATH or in local `qemu/bin/` directories.
> If external tools are missing on the target host machine, `tatva.exe` **never crashes**. Instead, it renders an in-app guidance modal detailing step-by-step installation instructions.

---

## 3. How to Build the Standalone Executable

### Prerequisites
1. Python 3.10–3.13 installed in virtualenv `.venv`.
2. PyInstaller installed:
   ```bash
   pip install pyinstaller
   ```

### Command to Build
Run the automated build script:
```bash
python scratch/build_app.py
```

Or execute PyInstaller directly:
```bash
pyinstaller tatva.spec --noconfirm --clean
```

---

## 4. How to Deploy and Run the Standalone App

1. Copy or compress the `dist/tatva/` folder to the target machine.
2. Double-click `tatva.exe` (or run `dist/tatva/tatva.exe` from PowerShell / CMD).
3. The splash window (`SplashWindow`) will load heavy backend libraries (`tvm`, `onnxruntime`, `onnx`) in a background thread while displaying progress.
4. Once loaded, the 7-panel desktop GUI reveals automatically.

---

## 5. Cross-Platform Build Notes (macOS & Linux)

While Tatva is primarily compiled and tested on Windows 10/11 x64 in this milestone, PyInstaller supports generating native macOS application bundles (`.app`) and Linux ELF binaries using the same `tatva.spec` specification:

- **Linux (Ubuntu 22.04+):**
  ```bash
  pyinstaller tatva.spec --noconfirm --clean
  # Generates dist/tatva/tatva binary
  ```
- **macOS (Apple Silicon / Intel):**
  ```bash
  pyinstaller tatva.spec --noconfirm --clean --windowed
  # Generates dist/tatva.app bundle
  ```
