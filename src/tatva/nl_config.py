"""
Natural-language configuration.

The beta scope lists this as an experimental feature: state a priority in plain English
and have it resolve to a build configuration. It is a *convenience layer over the same
switches the checkboxes set* -- nothing here can produce a configuration you could not
have selected by hand, and nothing here touches a measurement.

Two properties matter more than coverage:

  * It is deterministic and offline. No model, no network, no API key. The same sentence
    always resolves to the same configuration, which is what makes it safe to put in
    front of a build that takes minutes.
  * It explains itself. Every decision carries the phrase that caused it, so a user can
    see why quantization went on and disagree with it. A configuration layer that
    silently reinterprets your intent is worse than no configuration layer.

Anything it does not recognise is left at the caller's current setting rather than
guessed at -- `interpret` reports what it matched and what it ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tatva.compiler import DEFAULT_TARGET, TARGETS

# Phrases are matched on word boundaries against a lowercased, punctuation-normalised
# string. Ordering within a rule matters only for which phrase gets quoted back to the
# user; the resulting configuration is the same.
#
# Negations are checked before their positive counterparts, because "no quantization"
# contains "quantization" and would otherwise turn the pass on.

_QUANT_OFF = (
    "no quantization", "no quantisation", "without quantization", "without quantisation",
    "no int8", "without int8", "dont quantize", "do not quantize", "dont quantise",
    "skip quantization", "skip quantisation", "fp32 only", "float only", "keep it fp32",
)
_QUANT_ON = (
    "quantize", "quantise", "quantization", "quantisation", "int8", "8-bit", "8 bit",
)
_FUSE_OFF = (
    "no fusion", "no fuse", "without fusion", "without fusing", "disable fusion",
    "dont fuse", "do not fuse", "skip fusion", "unfused",
)
_FUSE_ON = ("fusion", "fuse", "fused")

# Priorities. These are the phrases someone actually types when they have not read the
# pass list -- "make it as small as possible", "I care about speed".
_PRIORITY_SIZE = (
    "smallest", "smaller", "small", "size", "binary size", "footprint", "memory", "ram",
    "sram", "flash", "fit on", "fits in", "fit in", "space", "compact", "tiny",
    "constrained", "low memory", "least memory", "lightweight", "shrink",
)
_PRIORITY_SPEED = (
    "fastest", "faster", "fast", "speed", "latency", "quick", "quickest", "throughput",
    "performance", "real-time", "real time", "responsive", "low latency",
)
_PRIORITY_ACCURACY = (
    "accuracy", "accurate", "precision", "precise", "exact", "lossless", "no loss",
    "numerically", "quality", "correctness", "faithful",
)

# Target aliases. The canonical names are what the rest of the app uses; these are the
# ways people refer to them in a sentence.
_TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "RV32IMC": ("rv32imc", "rv32 imc"),
    "RV32IMAC": ("rv32imac", "rv32 imac"),
    "RV64GC": ("rv64gc", "rv64 gc", "64-bit", "64 bit", "rv64"),
    "RV64IMAFDC": ("rv64imafdc", "rv64 imafdc"),
    "RV64GCV": ("rv64gcv", "rv64 gcv", "vector", "rvv"),
    "RV32EMC": ("rv32emc", "rv32 emc", "embedded profile"),
    # Checked after the specific names above, so "rv32imc" is not swallowed by "rv32".
    "_RV32_GENERIC": ("rv32", "32-bit", "32 bit", "microcontroller", "mcu"),
}

# Every apostrophe a Windows text field can hand over: ASCII, the typographic one Word
# and phone keyboards autocorrect to, and the modifier letter. Built from code points
# rather than literals -- as characters they are indistinguishable in source.
_APOSTROPHES = frozenset(["'", chr(0x2019), chr(0x02BC)])

# The tolerance the parity check uses. Stated once here so the prose and the returned
# value cannot drift apart.
DEFAULT_TOLERANCE = 0.05
STRICT_TOLERANCE = 0.01


@dataclass
class ConfigIntent:
    """A build configuration resolved from a sentence, with the reasoning kept."""

    target: str = DEFAULT_TARGET
    fuse: bool = True
    quantize: bool = False
    accuracy_tolerance: float = DEFAULT_TOLERANCE
    # One line per decision, each naming the phrase that drove it.
    reasons: list[str] = field(default_factory=list)
    # True when nothing in the sentence was recognised, so every value above is just
    # the caller's starting configuration handed back. The UI says so rather than
    # implying the sentence was understood.
    matched: bool = False
    # Phrases that pull in opposite directions, e.g. "smallest and fastest".
    conflicts: list[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    """
    Lowercase and flatten punctuation so phrase matching is not defeated by commas.

    Apostrophes are deleted rather than replaced with a space, so "don't fuse" collapses
    to "dont fuse" and matches the alias spelled that way. Replacing them would have
    produced "don t fuse", which matches nothing. Both the ASCII apostrophe and the
    typographic one are handled -- Windows text fields hand over whichever the user's
    keyboard or autocorrect produced.
    """
    stripped = "".join(c for c in text.lower() if c not in _APOSTROPHES)
    return re.sub(r"[^a-z0-9+\- ]+", " ", stripped)


def _find(haystack: str, phrases: tuple[str, ...]) -> str | None:
    """Return the first phrase present as a whole word/phrase, or None."""
    for p in phrases:
        if re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", haystack):
            return p
    return None


def interpret(
    text: str,
    *,
    target: str = DEFAULT_TARGET,
    fuse: bool = True,
    quantize: bool = False,
) -> ConfigIntent:
    """
    Resolve a plain-English priority into a build configuration.

    The `target`, `fuse` and `quantize` arguments are the caller's *current* settings.
    Anything the sentence does not speak to is left exactly as it was -- this is a way to
    adjust a configuration, not to replace one with a set of defaults.
    """
    intent = ConfigIntent(target=target if target in TARGETS else DEFAULT_TARGET,
                          fuse=fuse, quantize=quantize)
    s = _normalise(text or "")
    if not s.strip():
        return intent

    # --- target ---------------------------------------------------------------
    for name, aliases in _TARGET_ALIASES.items():
        hit = _find(s, aliases)
        if not hit:
            continue
        # The generic 32-bit bucket has no single right answer; RV32IMC is the smallest
        # non-experimental 32-bit target, so it is the safe reading of "a 32-bit MCU".
        resolved = "RV32IMC" if name == "_RV32_GENERIC" else name
        intent.target = resolved
        intent.matched = True
        intent.reasons.append(f'Target set to {resolved} — you said "{hit}".')
        break

    # --- explicit pass instructions ------------------------------------------
    # These are stated intentions about a specific pass and outrank the priority
    # heuristics below, so they are applied last. Record them now.
    quant_off = _find(s, _QUANT_OFF)
    quant_on = None if quant_off else _find(s, _QUANT_ON)
    fuse_off = _find(s, _FUSE_OFF)
    fuse_on = None if fuse_off else _find(s, _FUSE_ON)

    # --- priorities -----------------------------------------------------------
    size = _find(s, _PRIORITY_SIZE)
    speed = _find(s, _PRIORITY_SPEED)
    accuracy = _find(s, _PRIORITY_ACCURACY)

    if size and speed:
        intent.conflicts.append(
            f'"{size}" and "{speed}" pull in opposite directions here: INT8 quantization '
            "is what shrinks the binary, and on a scalar RISC-V core it is measurably "
            "slower, not faster. Size was taken as the priority."
        )
    if size and accuracy:
        intent.conflicts.append(
            f'"{size}" and "{accuracy}" pull in opposite directions: quantization trades '
            "numerical accuracy for space. Size was taken as the priority, with the "
            "parity check left on to catch it if the trade goes too far."
        )

    if size:
        intent.matched = True
        intent.quantize = True
        intent.fuse = True
        intent.reasons.append(
            f'INT8 quantization on — you said "{size}", and quantization is the pass that '
            "reduces the binary and the working set. It costs latency on a scalar core."
        )
    elif speed:
        intent.matched = True
        intent.quantize = False
        intent.fuse = True
        intent.reasons.append(
            f'Softmax fusion on, INT8 quantization off — you said "{speed}". Fusion removes '
            "a pass over the attention output; quantization is a measured regression on a "
            "core with no INT8 dot-product instruction."
        )
    elif accuracy:
        intent.matched = True
        intent.quantize = False
        intent.fuse = True
        intent.accuracy_tolerance = STRICT_TOLERANCE
        intent.reasons.append(
            f'INT8 quantization off and the parity tolerance tightened to {STRICT_TOLERANCE} '
            f'— you said "{accuracy}". Softmax fusion stays on: it is an algebraic rewrite '
            "and does not change the result."
        )

    # --- explicit instructions win -------------------------------------------
    if quant_off:
        intent.matched = True
        intent.quantize = False
        intent.reasons.append(f'INT8 quantization off — you said "{quant_off}".')
    elif quant_on:
        intent.matched = True
        intent.quantize = True
        intent.reasons.append(f'INT8 quantization on — you said "{quant_on}".')

    if fuse_off:
        intent.matched = True
        intent.fuse = False
        intent.reasons.append(f'Softmax fusion off — you said "{fuse_off}".')
    elif fuse_on:
        intent.matched = True
        intent.fuse = True
        intent.reasons.append(f'Softmax fusion on — you said "{fuse_on}".')

    return intent


def summarise(intent: ConfigIntent) -> str:
    """One-line description of the resolved configuration, for a status row."""
    passes = [n for n, on in (("softmax fusion", intent.fuse), ("INT8 quantization", intent.quantize)) if on]
    return (
        f"{intent.target} · "
        + (" + ".join(passes) if passes else "no passes")
        + f" · parity tolerance {intent.accuracy_tolerance}"
    )
