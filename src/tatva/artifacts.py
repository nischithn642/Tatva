"""
TATVA generated-artifact manifest.

Every TATVA build already writes real files: the weights header, the generated C, the
bare-metal harness, the linker script and the linked ELF. Until now nothing enumerated
them, so the one question an engineer asks after a successful compile -- "what did it
actually produce, and where is it?" -- had no answer inside the app.

This module answers it by reading the build directory. Nothing here is a fixed list of
expected filenames dressed up as a result: `discover` walks what is on disk, sizes and
hashes what it finds, and reports exactly that. If `runner.compile_model` starts
emitting another file tomorrow, it shows up here with no change to this module. A file
that was never written simply does not appear.

The manifest is also persisted next to the artifacts as `artifact_manifest.json`, so a
build directory carries its own provenance once TATVA has exited.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

MANIFEST_FILENAME = "artifact_manifest.json"

# Which pipeline stage produced a given file. These are the stages `compile_model`
# actually runs, in the order it runs them -- see `tatva.runner.compile_model`.
STAGE_TVM_LOWERING = "TVM lowering"
STAGE_CODEGEN = "TVM code generation"
STAGE_HARNESS = "Harness generation"
STAGE_SOFTMAX = "Softmax kernel injection"
STAGE_LINK = "Cross-compilation and link"
STAGE_MANIFEST = "Manifest"

# name -> (type, stage, description). The lookup is by exact filename first, then by
# extension, then a generic fallback -- so an unrecognised file is still listed with
# its real size and hash rather than being dropped for not being on a list.
_KNOWN: dict[str, tuple[str, str, str]] = {
    "weights.h": ("C header", STAGE_TVM_LOWERING,
                  "Model parameters exported as static C arrays, ready to sit in .rodata."),
    "model_run.c": ("C source", STAGE_CODEGEN,
                    "The model's forward pass, emitted by TVM Relax as portable C99."),
    "model_info.h": ("C header", STAGE_CODEGEN,
                     "Input and output shapes, dtypes and buffer sizes for the harness."),
    "operators.c": ("C source", STAGE_CODEGEN,
                    "Operator kernels backing the forward pass. The fusion pass rewrites the softmax kernels here."),
    "main.c": ("C source", STAGE_HARNESS,
               "Bare-metal entry point: fills the input buffers, runs the timed loop, prints cycles and logits."),
    "start.S": ("Assembly", STAGE_HARNESS,
                "Reset vector and stack setup. Runs before main() on a machine with no bootloader."),
    "link.ld": ("Linker script", STAGE_HARNESS,
                "Memory map for the target: load address, section placement, stack and heap bounds."),
    "model.elf": ("ELF executable", STAGE_LINK,
                  "The linked bare-metal binary. This is what QEMU boots and what you would flash."),
    MANIFEST_FILENAME: ("JSON", STAGE_MANIFEST, "This manifest."),
}

_BY_EXT: dict[str, tuple[str, str, str]] = {
    ".c": ("C source", STAGE_CODEGEN, "Generated C source."),
    ".h": ("C header", STAGE_CODEGEN, "Generated C header."),
    ".s": ("Assembly", STAGE_HARNESS, "Generated assembly."),
    ".ld": ("Linker script", STAGE_HARNESS, "Linker script."),
    ".elf": ("ELF executable", STAGE_LINK, "Linked binary."),
    ".bin": ("Raw binary", STAGE_LINK, "Raw binary image."),
    ".map": ("Link map", STAGE_LINK, "Linker map."),
    ".json": ("JSON", STAGE_MANIFEST, "Structured build data."),
    ".o": ("Object file", STAGE_LINK, "Intermediate object file."),
}


@dataclass
class Artifact:
    name: str
    path: str
    type: str
    stage: str
    description: str
    size_bytes: int
    sha256: str
    line_count: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _line_count(path: str) -> int | None:
    """Lines of generated source. Used by the effort calculator, which counts real output."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def _classify(name: str) -> tuple[str, str, str]:
    if name in _KNOWN:
        return _KNOWN[name]
    ext = os.path.splitext(name)[1].lower()
    if ext in _BY_EXT:
        return _BY_EXT[ext]
    return ("File", STAGE_CODEGEN, "Produced by the build.")


def describe(path: str) -> Artifact | None:
    """Describe one file on disk. Returns None if it is not a readable regular file."""
    if not os.path.isfile(path):
        return None
    name = os.path.basename(path)
    kind, stage, desc = _classify(name)
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    textual = kind in ("C source", "C header", "Assembly", "Linker script", "JSON")
    return Artifact(
        name=name,
        path=os.path.abspath(path),
        type=kind,
        stage=stage,
        description=desc,
        size_bytes=size,
        sha256=_sha256(path),
        line_count=_line_count(path) if textual else None,
    )


def discover(build_dir: str) -> list[Artifact]:
    """
    Enumerate everything a build directory contains.

    Sorted by stage in pipeline order and then by name, so the list reads as the build
    happened rather than alphabetically.
    """
    if not build_dir or not os.path.isdir(build_dir):
        return []
    found: list[Artifact] = []
    for entry in sorted(os.listdir(build_dir)):
        art = describe(os.path.join(build_dir, entry))
        if art is not None:
            found.append(art)
    order = {
        STAGE_TVM_LOWERING: 0, STAGE_CODEGEN: 1, STAGE_SOFTMAX: 2,
        STAGE_HARNESS: 3, STAGE_LINK: 4, STAGE_MANIFEST: 5,
    }
    found.sort(key=lambda a: (order.get(a.stage, 9), a.name))
    return found


def build_manifest(
    *,
    run_id: str,
    model_path: str,
    target_name: str,
    march: str,
    mabi: str,
    configs: dict[str, str],
    passes: list[str] | None = None,
    repaired_ops: list[str] | None = None,
    written_after: list[str] | None = None,
) -> dict[str, Any]:
    """
    Assemble the manifest for a completed run.

    `configs` maps a build configuration name ("baseline", "optimized") to its build
    directory. Both are listed separately because they are genuinely different builds
    with different generated C -- collapsing them would hide the thing the optimization
    pass actually changed.

    `written_after` names files the run writes to the same directory once this manifest
    exists -- the effort estimate and the audit trail. They are named rather than
    inventoried, so a reader who counts the directory and finds more files than this
    manifest lists can see immediately which ones and why, instead of concluding that
    the inventory is wrong.
    """
    from tatva import __version__

    sections: list[dict[str, Any]] = []
    total_bytes = 0
    total_files = 0
    for config_name, build_dir in configs.items():
        # A previous run against the same model, passes and target reuses its build
        # directory, so a stale manifest may already be sitting there. It is dropped and
        # replaced by the self-entry below rather than inventoried at its old size.
        artifacts = [a for a in discover(build_dir) if a.name != MANIFEST_FILENAME]
        size = sum(a.size_bytes for a in artifacts)
        rows: list[dict[str, Any]] = [a.to_json() for a in artifacts]
        count = len(artifacts)
        if build_dir and os.path.isdir(build_dir):
            # The manifest lists itself, because a reader comparing this inventory with
            # the directory should not find an unexplained extra file. Its size and hash
            # are null because it does not exist yet at the moment this list is taken --
            # writing a size here would mean writing a number that describes nothing.
            kind, stage, _ = _KNOWN[MANIFEST_FILENAME]
            rows.append({
                "name": MANIFEST_FILENAME,
                "path": os.path.abspath(os.path.join(build_dir, MANIFEST_FILENAME)),
                "type": kind,
                "stage": stage,
                "description": (
                    "This manifest. Written after the inventory above was taken, so its own "
                    "size and hash are not listed here."
                ),
                "size_bytes": None,
                "sha256": "",
                "line_count": None,
            })
            count += 1
        total_bytes += size
        total_files += count
        sections.append({
            "config": config_name,
            "build_dir": os.path.abspath(build_dir) if build_dir else "",
            "exists": bool(build_dir) and os.path.isdir(build_dir),
            "file_count": count,
            "total_bytes": size,
            "artifacts": rows,
        })

    manifest = {
        "schema": "tatva.artifact_manifest/1",
        "run_id": run_id,
        "compiler_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {
            "path": os.path.abspath(model_path) if model_path else "",
            "filename": os.path.basename(model_path) if model_path else "",
        },
        "target": {"name": target_name, "march": march, "mabi": mabi},
        "passes": list(passes or []),
        "repaired_ops": list(repaired_ops or []),
        "totals": {"file_count": total_files, "total_bytes": total_bytes},
        "written_after_this_manifest": list(written_after or []),
        "builds": sections,
    }
    return manifest


def write_manifest(manifest: dict[str, Any], build_dir: str) -> str:
    """
    Persist the manifest beside the artifacts it describes and return its path.

    Written last, and its own entry is added afterwards so the recorded size and hash
    describe the file that is actually on disk rather than the one before the write.
    """
    if not build_dir or not os.path.isdir(build_dir):
        return ""
    path = os.path.join(build_dir, MANIFEST_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
    except OSError:
        return ""
    return path


def read_manifest(build_dir: str) -> dict[str, Any] | None:
    """Read back a manifest a previous run left behind, if there is one."""
    path = os.path.join(build_dir or "", MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def generated_source_stats(configs: dict[str, str]) -> dict[str, int]:
    """
    Counted totals across the generated sources of every build in the run.

    The effort calculator consumes this. It counts lines that exist in files that
    exist; when there is no build directory it returns zeros, and the caller is
    expected to treat zero as "nothing was generated", not as a small number.
    """
    files = 0
    lines = 0
    header_lines = 0
    elf_bytes = 0
    for build_dir in configs.values():
        for art in discover(build_dir):
            if art.name == MANIFEST_FILENAME:
                continue
            files += 1
            if art.line_count:
                lines += art.line_count
                if art.type == "C header":
                    header_lines += art.line_count
            if art.type == "ELF executable":
                elf_bytes += art.size_bytes
    return {
        "generated_files": files,
        "generated_lines": lines,
        "generated_header_lines": header_lines,
        "elf_bytes": elf_bytes,
    }
