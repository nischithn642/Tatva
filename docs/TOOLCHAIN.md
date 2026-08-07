# TOOLCHAIN: RISC-V & QEMU Installation Guide

This document describes how to set up the necessary external toolchains for cross-compiling and simulating RISC-V binaries.

## Target Architecture Configuration

The compiler targets standard RISC-V configurations:
- **32-Bit Target**: `rv32imc` or `rv32imac` with application binary interface `ilp32`.
- **64-Bit Target**: `rv64gc` or `rv64imafdc` with application binary interface `lp64d`.

---

## 1. Installation on Linux (Ubuntu/Debian)

Run the following commands to install the RISC-V GCC cross-compiler and QEMU system/user emulators:

```bash
# Update package index
sudo apt update

# Install the RISC-V toolchain (riscv64-unknown-elf-gcc)
sudo apt install -y gcc-riscv64-unknown-elf

# Install QEMU with RISC-V system and user-mode emulation
sudo apt install -y qemu-system-misc qemu-user
```

---

## 2. Installation on macOS

Using Homebrew, run:

```bash
# Install the RISC-V cross-compiler toolchain
brew tap riscv-software-src/riscv
brew install riscv-gnu-toolchain

# Install QEMU
brew install qemu
```

---

## 3. Note for Windows

For Windows development, **WSL2 (Windows Subsystem for Linux)** running Ubuntu is highly recommended. Follow the Linux installation instructions above inside your WSL2 instance.

Alternatively, for local native Windows development, pre-compiled portable binaries can be extracted to a local folder and added to your system PATH:
- **GCC**: [xPack GNU RISC-V Embedded GCC](https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases) (`riscv-none-elf-gcc`)
- **QEMU**: [xPack QEMU RISC-V](https://github.com/xpack-dev-tools/qemu-riscv-xpack/releases) (`qemu-system-riscv64`)
