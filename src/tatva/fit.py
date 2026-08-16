"""
Which RISC-V targets a given model actually fits.

Stage 01 offered six targets as six equal choices and let the user find out at stage 05
that the one they picked could not load their model, or would emulate it for twenty
minutes because it has no FPU. The chip is not a free choice once a model is loaded: the
model's weights decide whether the linked image fits under the emulator's ceiling, and
the model's precision decides whether every multiply becomes a soft-float library call.

Everything here is read out of the model file by `frontends.inspect_model` -- no import,
no TVM, no compilation -- and compared against the same constants the build itself uses
(`tatva.runner`). Nothing is predicted: the image figure is a *floor*, because
activations are not known until TVM has legalized the graph, and it is reported as one.

The verdicts:

  FITS      the image floor is under the emulator's ceiling and the target has hardware
            floating point for the precision this model is in
  SLOW      it will build and run, but this model's arithmetic is emulated in software
            on this target -- a measured 20.5x the guest cycles, 3.6x the wall clock
  BLOCKED   the weights alone put the image past what the bundled emulator can load

`SLOW` is not a refusal. A user targeting an RV32IMC part needs that build; they need to
know what it costs before they wait for it, not after.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from tatva.compiler import DEFAULT_TARGET, TARGETS, TargetVariant
from tatva.runner import (
    MAX_LOADABLE_IMAGE_BYTES,
    STACK_BYTES,
    WORKSPACE_POOL_BYTES,
    _has_hardware_float,
    _ram_region_bytes,
)

VERDICT_FITS = "FITS"
VERDICT_SLOW = "SLOW"
VERDICT_BLOCKED = "BLOCKED"

# Order in which equally-viable targets are preferred for the "best fit" mark. RV64GC
# leads because it is the default and the one every measurement in this repository was
# taken on; the rest follow by how much of the ISA the generated code can use.
_PREFERENCE = ["RV64GC", "RV64IMAFDC", "RV32IMAC", "RV32IMC", "RV64GCV", "RV32EMC"]

# Element sizes for the graph's own inputs and outputs, which are linked as fixed
# buffers by the harness. Same table as the parameter figure uses, keyed on the dtype
# name `inspect_model` reports rather than the ONNX enum.
_DTYPE_BYTES: dict[str, int] = {
    "FP32": 4, "FP64": 8, "FP16": 2, "BF16": 2,
    "INT8": 1, "UINT8": 1, "INT16": 2, "UINT16": 2,
    "INT32": 4, "UINT32": 4, "INT64": 8, "UINT64": 8, "BOOL": 1,
}

# Precisions whose arithmetic goes through the FPU. A model made entirely of these is
# the one that pays the soft-float penalty on a target without an F or D extension.
_FLOAT_PRECISIONS = {"FP32", "FP64", "FP16", "BF16"}


@dataclass
class TargetFit:
    """One target judged against one model."""
    target: str
    march: str
    mabi: str
    bitness: int
    experimental: bool
    verdict: str
    headline: str
    reasons: list[str] = field(default_factory=list)
    best: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_bytes(tensors: list[dict[str, Any]]) -> int:
    """
    Linked size of a list of graph inputs or outputs.

    A symbolic dimension counts as 1. That is deliberately the smallest thing it can be:
    every figure in this module is a floor, and a floor that guessed high would block a
    model that would have compiled.
    """
    total = 0
    for t in tensors or []:
        count = 1
        for dim in t.get("shape") or []:
            if isinstance(dim, int) and dim > 0:
                count *= dim
        total += count * _DTYPE_BYTES.get(str(t.get("dtype") or ""), 4)
    return total


def image_floor_bytes(info: Any) -> int:
    """
    The smallest RAM region this model could possibly be linked into.

    Weights, the graph's own IO buffers, the fixed workspace pool and the stack -- every
    pool `runner.compile_model` sizes the linker script from, except the activation
    arena, which does not exist until TVM has legalized the graph. So the real region is
    always this or larger, never smaller, which is what makes a BLOCKED verdict here
    safe to state: it cannot become unblocked later.
    """
    accounted = (
        int(getattr(info, "parameter_bytes", 0) or 0)
        + _tensor_bytes(getattr(info, "inputs", []))
        + _tensor_bytes(getattr(info, "outputs", []))
        + WORKSPACE_POOL_BYTES
        + STACK_BYTES
    )
    return _ram_region_bytes(accounted)


def _mib(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MiB"


def _float_precisions(info: Any) -> set[str]:
    """
    The floating-point precisions this model's weights are in.

    `precision` is either one dtype name or "Mixed (A/B)"; both are parsed here so a
    mixed FP32/INT8 model is still recognised as one that needs an FPU.
    """
    raw = str(getattr(info, "precision", "") or "")
    body = raw.split("(", 1)[1].rstrip(")") if "(" in raw else raw
    return {p.strip() for p in body.split("/")} & _FLOAT_PRECISIONS


def fit_for(info: Any, variant: TargetVariant, *, floor: int | None = None) -> TargetFit:
    """Judge one target against one inspected model."""
    floor = image_floor_bytes(info) if floor is None else floor
    floats = _float_precisions(info)
    reasons: list[str] = []

    if floor > MAX_LOADABLE_IMAGE_BYTES:
        return TargetFit(
            target=variant.name, march=variant.gcc_march, mabi=variant.gcc_mabi,
            bitness=variant.bitness, experimental=variant.experimental,
            verdict=VERDICT_BLOCKED,
            headline=f"Too large for the bundled emulator by at least "
                     f"{_mib(floor - MAX_LOADABLE_IMAGE_BYTES)}.",
            reasons=[
                f"Weights and IO alone link into {_mib(floor)}, and the bundled QEMU "
                f"`virt` board can load {_mib(MAX_LOADABLE_IMAGE_BYTES)} before the image "
                f"runs into the device tree. Activations are on top of that figure.",
                "This is a property of the emulator, not of the chip -- every target here "
                "has the same ceiling.",
            ],
        )

    soft_float = bool(floats) and not _has_hardware_float(variant)
    verdict = VERDICT_SLOW if soft_float else VERDICT_FITS

    if soft_float:
        reasons.append(
            f"{variant.gcc_march} has no F or D extension, so this model's "
            f"{'/'.join(sorted(floats))} arithmetic runs as soft-float library calls. "
            "Measured on the same model and region: 20.5x the guest cycles of RV64GC and "
            "3.6x the wall clock."
        )
    if "FP64" in floats and variant.bitness == 32:
        reasons.append("The model carries FP64 tensors, which a 32-bit target emulates in "
                       "software even where FP32 is native.")
    if variant.name == "RV64GCV":
        reasons.append("The vector extension is in the ABI but code generation is scalar C, "
                       "so this produces the same cycle count as RV64GC.")
    if variant.experimental:
        reasons.append(f"Experimental target. {variant.notes}".strip())

    # Everything appended so far is a caveat. A target with none of them gets the clean
    # verdict and the figures behind it; a target with one says so in the headline,
    # because a caveat three lines down is a caveat most people will not read.
    caveats = len(reasons)
    if not caveats:
        arithmetic = (f"Hardware floating point for this model's {'/'.join(sorted(floats))} weights"
                      if floats else "No floating-point weights, so nothing here needs an FPU")
        reasons.append(
            f"{arithmetic}, and the image floor of {_mib(floor)} is inside the "
            f"{_mib(MAX_LOADABLE_IMAGE_BYTES)} the bundled emulator can load."
        )

    headline = ("Builds and runs, but this model's arithmetic is emulated in software." if soft_float
                else "Fits this model, with caveats." if caveats
                else "Fits this model.")
    return TargetFit(
        target=variant.name, march=variant.gcc_march, mabi=variant.gcc_mabi,
        bitness=variant.bitness, experimental=variant.experimental,
        verdict=verdict, headline=headline, reasons=reasons,
    )


def fit_targets(model_path: str) -> dict[str, Any]:
    """
    Every target judged against one model, with the best fit marked.

    "Best" is the first target in `_PREFERENCE` that is neither blocked, slow nor
    experimental -- so it moves with the model rather than being a constant. When
    nothing qualifies, none is marked and the summary says why, because marking the
    least-bad option as a recommendation would be a claim this module cannot support.
    """
    from tatva.frontends import inspect_model

    if not model_path or not os.path.isfile(model_path):
        return {"success": False, "error": f"Model file not found: '{model_path}'", "targets": []}

    info = inspect_model(model_path)
    if not getattr(info, "ok", False):
        return {
            "success": False,
            "error": getattr(info, "error", "") or "The model could not be read.",
            "targets": [],
        }

    floor = image_floor_bytes(info)
    fits = [fit_for(info, v, floor=floor) for v in TARGETS.values()]
    by_name = {f.target: f for f in fits}

    best = ""
    for name in _PREFERENCE:
        f = by_name.get(name)
        if f is not None and f.verdict == VERDICT_FITS and not f.experimental:
            f.best = True
            best = name
            break

    blocked = [f.target for f in fits if f.verdict == VERDICT_BLOCKED]
    slow = [f.target for f in fits if f.verdict == VERDICT_SLOW]

    if blocked:
        summary = (f"{info.name} does not fit any target here: weights and IO alone need "
                   f"{_mib(floor)} and the bundled emulator loads {_mib(MAX_LOADABLE_IMAGE_BYTES)}.")
    elif best:
        summary = f"{best} fits {info.name} best."
        if slow:
            summary += (f" {', '.join(slow)} will build and run it, with the floating-point "
                        "work emulated in software.")
    else:
        summary = (f"No non-experimental target fits {info.name} without a caveat. "
                   "Read the notes on each before picking one.")

    return {
        "success": True,
        "error": "",
        "model": info.name,
        "precision": info.precision,
        "parameter_bytes": int(info.parameter_bytes or 0),
        "image_floor_bytes": floor,
        "image_limit_bytes": MAX_LOADABLE_IMAGE_BYTES,
        "default_target": DEFAULT_TARGET,
        "best": best,
        "summary": summary,
        "targets": [f.to_json() for f in fits],
    }
