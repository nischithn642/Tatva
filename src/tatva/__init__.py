"""
TATVA: RISC-V Transformer Optimization Toolchain.
"""

# The single source of truth for the version. pyproject.toml reads this file via
# hatch's dynamic version hook, the CLI's --version reads it, and the GUI's build
# badge reads it. Previously there were four copies and they had all drifted apart:
# pyproject said 0.2.0 while the package said 1.2.0-gemini.
#
# "2.1.0" is PEP 440 -- what hatchling, pip and the wheel filename require.
# DISPLAY_VERSION is the name the release goes by on the badge, in the artifact
# filenames and in the README. For 2.0 the two differed ("2.0.0b1" vs "Beta 2.0");
# they agree now, and the pair is kept so that a future pre-release can diverge
# again without every consumer having to learn a second spelling.
__version__ = "2.1.0"
DISPLAY_VERSION = "2.1"
