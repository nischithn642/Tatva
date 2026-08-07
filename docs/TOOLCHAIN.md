# TOOLCHAIN: RISC-V & QEMU Installation Guide

TATVA needs two external tools it does not ship: a RISC-V cross-compiler
(`riscv-none-elf-gcc`) and a RISC-V system emulator (`qemu-system-riscv64`). Together
they are about 520 MB, which is why they are not in the wheel.

`tatva doctor` tells you which one is missing and every directory it searched.

---

## 1. The easy way: `tatva setup`

```bash
tatva setup
```

This downloads pinned [xPack](https://xpack.github.io/) prebuilt binaries for your
platform and unpacks them into a per-user directory:

| Platform | Install location |
| :--- | :--- |
| Windows | `%LOCALAPPDATA%\tatva\toolchains` |
| macOS | `~/Library/Application Support/tatva/toolchains` |
| Linux | `~/.local/share/tatva/toolchains` (or `$XDG_DATA_HOME`) |

Set `TATVA_TOOLS_DIR` to install somewhere else.

Supported hosts: `win32-x64`, `linux-x64`, `linux-arm64`, `darwin-x64`, `darwin-arm64`.
On anything else — Windows on ARM, a RISC-V host, 32-bit x86 — xPack publishes no build,
and `tatva setup` says so instead of downloading the wrong asset. Use section 2.

Useful flags:

```bash
tatva setup --dry-run              # print the exact URLs and destinations, download nothing
tatva setup --component qemu       # just one of the two
tatva setup --force                # re-download over an existing install
tatva setup --yes                  # skip the confirmation prompt (CI)
```

Nothing is downloaded until you confirm the size, and re-running is free — an existing
install is detected and skipped.

Pinned versions: GCC `15.2.0-1`, QEMU `9.2.4-1`. They are pinned so that a cycle count
measured on your machine means the same thing as one measured on someone else's.

---

## 2. Bringing your own toolchain

Anything on `PATH` wins over what `tatva setup` installed, so a system package or a
hand-built toolchain is used in preference without any configuration.

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y gcc-riscv64-unknown-elf qemu-system-misc
```

### macOS (Homebrew)

```bash
brew tap riscv-software-src/riscv
brew install riscv-gnu-toolchain qemu
```

### Windows

`tatva setup` works natively — WSL2 is not required. If you would rather install by
hand, download from the xPack releases and put the `bin` directory on `PATH`:

- **GCC**: [xPack GNU RISC-V Embedded GCC](https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases) (`riscv-none-elf-gcc`)
- **QEMU**: [xPack QEMU RISC-V](https://github.com/xpack-dev-tools/qemu-riscv-xpack/releases) (`qemu-system-riscv64`)

TATVA accepts any of `riscv-none-elf-gcc`, `riscv64-unknown-elf-gcc`, or
`riscv64-linux-gnu-gcc` as the cross-compiler. A host `gcc` is never used as a fallback:
it would be invoked with `-march=rv64gc` and fail in a confusing way.

---

## 3. Where TATVA looks, in order

1. `PATH`
2. `<tools dir>/riscv-none-elf-gcc/bin` and `<tools dir>/qemu-riscv/bin` (section 1)
3. `<repo>/riscv-toolchain/bin` and `<repo>/qemu/bin` — the legacy in-repo location used
   by the old `setup_env.py`, still searched so an existing checkout keeps working

`tatva doctor` prints this list with the paths resolved for your machine.

---

## Target Architecture Configuration

The compiler targets standard RISC-V configurations:

- **32-bit**: `rv32imc` or `rv32imac`, ABI `ilp32`.
- **64-bit**: `rv64gc` (default) or `rv64imafdc`, ABI `lp64d`.

`rv64gcv` and `rv32emc` are marked experimental — see `tatva targets`. The vector
extension in particular is *selectable but not exploited*: code generation is scalar C,
so `rv64gcv` currently buys nothing over `rv64gc`.
