"""
Tests for tatva compilation and latency measurement runner.
"""

import os
import re
from pathlib import Path

import pytest

from tatva.compiler import TARGETS, import_model
from tatva.runner import (
    MIN_RAM_BYTES,
    RAM_GRANULE_BYTES,
    ExecutionEnvironment,
    compile_model,
    find_qemu,
    find_riscv_gcc,
    run_and_measure,
)


@pytest.mark.integration
def test_compile_and_measure_e2e(skip_if_no_toolchain) -> None:
    """
    Assert that we can compile and run a model under QEMU system mode
    and get deterministic simulated latency measurements.
    """
    _gcc_name, gcc_path = find_riscv_gcc()
    _qemu_name, qemu_path = find_qemu(64)

    if not gcc_path or not qemu_path:
        pytest.skip("Skipping runner e2e test because GCC or QEMU-64 is missing.")

    # Load small ONNX model
    model_path = "models/model.onnx"
    model_ir = import_model(model_path)

    # Compile for default target RV64GC
    variant = TARGETS["RV64GC"]
    artifact = compile_model(model_ir, variant, warmup_count=1, timed_count=3)
    assert artifact is not None
    assert artifact.elf_path.endswith(".elf")

    # Run in simulator
    result = run_and_measure(artifact, environment=ExecutionEnvironment.QEMU_SIM)
    assert result is not None
    assert result.environment == "QEMU_SIM"
    assert result.simulated is True
    # timed_count is a ceiling, not a quota: the on-target loop stops as soon as two
    # consecutive runs report the same cycle count, because under -icount shift=0 no
    # later sample can differ. Assert the ceiling is respected and that any truncation
    # was earned -- a short sample set is only legitimate if the samples agree.
    assert 2 <= len(result.raw_samples_ms) <= 3
    if len(result.raw_samples_ms) < 3:
        assert len(set(result.raw_samples_ms)) == 1
    assert result.mean_ms > 0.0
    assert result.median_ms > 0.0
    assert result.p95_ms > 0.0


@pytest.mark.integration
def test_weights_are_emitted_as_a_blob_not_as_c_source(skip_if_no_toolchain) -> None:
    """
    The change that made large models compilable: constant tensors are written to
    weights.bin and pulled in by `.incbin`, rather than spelled out as a C initializer
    list. The initializer form cost ~4.1 source bytes per model byte and took cc1 past
    3 GB of RSS on a 116 MB model, so a regression here is invisible on the small models
    in `models/` and fatal on the ones this release exists for.

    This compiles the small model and inspects what was emitted -- the assertions are
    about form, so they hold at any size.
    """
    _gcc_name, gcc_path = find_riscv_gcc()
    if not gcc_path:
        pytest.skip("Skipping: RISC-V GCC is missing.")

    model_ir = import_model("models/model.onnx")
    artifact = compile_model(model_ir, TARGETS["RV64GC"], warmup_count=1, timed_count=1)

    build_dir = Path(artifact.build_dir)
    blob_path = build_dir / "weights.bin"
    assert blob_path.exists(), "weights.bin was not emitted"
    blob_size = blob_path.stat().st_size
    assert blob_size > 0

    weights_s = (build_dir / "weights.S").read_text()
    weights_h = (build_dir / "weights.h").read_text()

    # Every tensor is an .incbin of a range that exists inside the blob, and each one
    # starts on the boundary its .balign claims -- a mismatch between the padding written
    # into the blob and the alignment asserted in the source silently shifts a tensor.
    directives = re.findall(r'\.incbin\s+"([^"]+)",\s*(\d+),\s*(\d+)', weights_s)
    assert directives, "no .incbin directives -- weights are not coming from the blob"

    previous_end = 0
    for path, start_text, length_text in directives:
        start, length = int(start_text), int(length_text)
        assert start % 16 == 0, f"tensor at {start} is not 16-aligned"
        assert start >= previous_end, "tensor ranges overlap"
        assert start + length <= blob_size, "tensor range runs off the end of weights.bin"
        previous_end = start + length

        # GNU as reads escapes inside this string, so a Windows path would be mangled.
        assert "\\" not in path, f"backslash in .incbin path: {path}"
        assert os.path.isabs(path), "a relative path makes the build depend on the CWD"
        assert os.path.exists(path)

    assert weights_s.count(".balign 16") == len(directives)
    assert weights_s.count(".globl constant_data_") == len(directives)

    # The header declares the symbols and holds no data. The failure being guarded
    # against is a partial revert leaving initializer lists behind.
    assert weights_h.count("extern ") == len(directives)
    assert "{" not in weights_h, "weights.h contains an initializer list"
    assert len(weights_h) < 100 * 1024, "weights.h should be declarations only"

    # The RAM region is computed from what was emitted, and the small model must still
    # land on the historical floor.
    assert artifact.ram_bytes % RAM_GRANULE_BYTES == 0
    assert artifact.ram_bytes >= MIN_RAM_BYTES
    assert artifact.ram_bytes > blob_size, "the region has to hold the weights it links"

    link_ld = (build_dir / "link.ld").read_text()
    assert f"LENGTH = {artifact.ram_bytes}" in link_ld
    assert "@" not in link_ld, "an unsubstituted template marker reached the linker"


@pytest.mark.integration
def test_a_region_overflow_is_reported_as_a_memory_limit(skip_if_no_toolchain, monkeypatch) -> None:
    """
    When the model does not fit, `ld` says `region \\`RAM' overflowed by N bytes` and the
    build must surface that as a memory-limit failure carrying the byte count -- not as a
    generic CompilationError advising the user to check link.ld for address collisions,
    which is neither the cause nor a fix.

    The only test of this path so far fed a hand-constructed exception to the diagnostics
    classifier, which cannot catch the regex drifting from what the linker actually
    prints. This provokes a real overflow by shrinking the computed region, so the string
    being matched comes from `ld` itself.
    """
    from tatva import runner as runner_module
    from tatva.diagnostics import MemoryLimitExceededError

    _gcc_name, gcc_path = find_riscv_gcc()
    if not gcc_path:
        pytest.skip("Skipping: RISC-V GCC is missing.")

    model_ir = import_model("models/model.onnx")

    # 64 KiB cannot hold any model's weights, so the link is certain to overflow.
    tiny = 64 * 1024
    monkeypatch.setattr(runner_module, "_ram_region_bytes", lambda _accounted: tiny)

    with pytest.raises(MemoryLimitExceededError) as caught:
        compile_model(model_ir, TARGETS["RV64GC"], warmup_count=1, timed_count=1)

    error = caught.value
    assert error.limit_bytes == tiny
    assert error.required_bytes > tiny, "the overflow byte count was not added to the limit"

    # The details are what the user reads, and they must break the budget down rather
    # than repeat whichever section the linker happened to be placing.
    assert "RV64GC" in error.details
    for label in ("weights", "activations", "workspace", "stack"):
        assert label in error.details


@pytest.mark.integration
def test_a_model_split_into_external_data_compiles_to_the_same_weights(
    skip_if_no_toolchain, tmp_path
) -> None:
    """
    Above roughly 2 GB an ONNX file cannot be a single protobuf message, so large models
    ship as `model.onnx` plus a sidecar `model.onnx_data`. That is the form the models
    this release targets actually arrive in, and until now nothing tested it at all.

    Both forms of the same model must produce byte-identical weights: the split is a
    storage detail and must not reach the compiled image. Asserted here on a small model,
    where the property is the same and the test finishes in seconds.
    """
    import shutil

    import onnx

    _gcc_name, gcc_path = find_riscv_gcc()
    if not gcc_path:
        pytest.skip("Skipping: RISC-V GCC is missing.")

    single = tmp_path / "single.onnx"
    shutil.copy("models/model.onnx", single)

    # The same graph, rewritten with its initializers in a sidecar file. Only raw_data
    # tensors can move out of the graph, and this fixture stores float_data, so they are
    # rebuilt first -- large real models are already raw_data, being far past the point
    # where the field-per-value encoding is practical.
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    split = split_dir / "model.onnx"
    model = onnx.load(str(single))
    for initializer in model.graph.initializer:
        initializer.CopyFrom(
            onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(initializer), initializer.name)
        )
    onnx.save_model(
        model,
        str(split),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=0,
    )
    assert (split_dir / "model.onnx_data").exists(), "the sidecar was not written"
    assert split.stat().st_size < single.stat().st_size, "weights did not move out of the graph"

    blobs = []
    for path in (single, split):
        artifact = compile_model(
            import_model(str(path)), TARGETS["RV64GC"], warmup_count=1, timed_count=1
        )
        blobs.append((Path(artifact.build_dir) / "weights.bin").read_bytes())

    assert blobs[0] == blobs[1], "the external-data form produced different weights"
    assert len(blobs[0]) > 0
