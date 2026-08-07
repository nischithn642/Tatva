"""
TATVA: RISC-V Transformer Optimization Toolchain.
"""

# The single source of truth for the version. pyproject.toml reads this file via
# hatch's dynamic version hook, the CLI's --version reads it, and the GUI's build
# badge reads it. Previously there were four copies and they had all drifted apart:
# pyproject said 0.2.0 while the package said 1.2.0-gemini.
__version__ = "0.3.0"
