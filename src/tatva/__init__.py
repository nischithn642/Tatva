"""
TATVA: RISC-V Transformer Optimization Toolchain.
"""

# The single source of truth for the version. pyproject.toml reads this file via
# hatch's dynamic version hook, the CLI's --version reads it, and the GUI's build
# badge reads it. Previously there were four copies and they had all drifted apart:
# pyproject said 0.2.0 while the package said 1.2.0-gemini.
#
# "2.0.0b1" is PEP 440 -- what hatchling, pip and the wheel filename require. It is
# the same release everything else calls Beta 2.0; DISPLAY_VERSION below is the name
# it goes by on the badge, in the zip filename and in the README, because "2.0.0b1"
# is not what anyone was told to expect.
__version__ = "2.0.0b1"
DISPLAY_VERSION = "Beta 2.0"
