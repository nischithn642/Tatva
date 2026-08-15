"""
Tests for the two limits that stop a large or slow model from running, and for the
diagnosis each one produces.

Both of these used to reach the user as "Unexpected Compilation Failure" followed by a
QEMU command line. Neither is a compilation failure:

  * The image outgrew the address range QEMU's `virt` board leaves below its device
    tree. The ELF is valid; only the emulator refuses it.
  * The run outlived its wall-clock ceiling, which on a soft-float target it will do
    long before anything is wrong.

The byte counts and addresses asserted here were measured against the bundled
toolchain, not derived from the source, so a change to a constant has to be a
deliberate edit here too.
"""

import struct
import subprocess

import pytest

from tatva import runner as runner_module
from tatva.compiler import TARGETS
from tatva.diagnostics import (
    DiagnosisContext,
    EmulatorImageLimitError,
    MemoryLimitExceededError,
    SimulationTimeoutError,
    classify_failure,
    get_offline_explanation,
    whitelist_payload,
)
from tatva.runner import (
    MAX_LOADABLE_IMAGE_BYTES,
    MIN_RAM_BYTES,
    QEMU_FDT_BASE,
    QEMU_SECONDS_PER_MIB_PER_RUN,
    QEMU_SOFT_FLOAT_SECONDS_PER_MIB_PER_RUN,
    RAM_ORIGIN,
    CompiledArtifact,
    _elf_load_span,
    _has_hardware_float,
    _qemu_timeout_seconds,
    run_and_measure,
)

MIB = 1024 * 1024

# QEMU's own words, copied from a real refused load of a 1.013 GB model. Kept verbatim
# because the backstop parses it -- a paraphrase would test the paraphrase.
REAL_OVERLAP_STDERR = """qemu-system-riscv64.exe: Some ROM regions are overlapping
These ROM regions might have been loaded by direct user request or by default.
They could be BIOS/firmware images, a guest kernel, initrd or some other file loaded into guest memory.
Check whether you intended to load all this guest code, and whether it has been built to load to the correct addresses.

The following two regions overlap (in the memory address space):
  build1040\\model.elf ELF program header segment 1 (addresses 0x0000000080200000 - 0x00000000c122a258)
  fdt (addresses 0x00000000bfe00000 - 0x00000000bfe012c4)
"""


def _fake_elf(path, *, is_64: bool, paddr: int, memsz: int) -> str:
    """Write a minimal ELF with a single PT_LOAD segment, and nothing else."""
    ident = b"\x7fELF" + bytes([2 if is_64 else 1, 1, 1]) + b"\x00" * 9
    if is_64:
        ehsize, phentsize, phoff = 64, 56, 64
        header = ident + struct.pack(
            "<HHIQQQIHHHHHH", 2, 243, 1, 0, phoff, 0, 0, ehsize, phentsize, 1, 0, 0, 0
        )
        phdr = struct.pack("<IIQQQQQQ", 1, 5, 0, paddr, paddr, memsz, memsz, 0x1000)
    else:
        ehsize, phentsize, phoff = 52, 32, 52
        header = ident + struct.pack(
            "<HHIIIIIHHHHHH", 2, 243, 1, 0, phoff, 0, 0, ehsize, phentsize, 1, 0, 0, 0
        )
        phdr = struct.pack("<IIIIIIII", 1, 0, paddr, paddr, memsz, memsz, 5, 0x1000)

    path.write_bytes(header + phdr)
    return str(path)


def _artifact(elf_path: str, target: str = "RV64GC", ram_bytes: int = MIN_RAM_BYTES) -> CompiledArtifact:
    return CompiledArtifact(
        elf_path=elf_path,
        build_dir="",
        variant=TARGETS[target],
        ram_bytes=ram_bytes,
        run_count=7,
    )


@pytest.mark.unit
def test_the_loadable_window_is_the_gap_below_the_device_tree() -> None:
    """
    The ceiling that actually binds. Measured against the bundled qemu-system-riscv64:
    the FDT sat at 0xBFE00000 for every `-m` from 1100M to 8192M, so the space between
    the load address and that address is all an image ever gets, whatever `-m` says.

    1020 MiB is lower than the linker's ~2 GiB relocation reach and lower than
    protobuf's 2 GB, which is why it is the number that matters.
    """
    assert QEMU_FDT_BASE == 0xBFE00000
    assert RAM_ORIGIN == 0x80200000
    assert MAX_LOADABLE_IMAGE_BYTES == QEMU_FDT_BASE - RAM_ORIGIN
    assert MAX_LOADABLE_IMAGE_BYTES == 1020 * MIB == 1_069_547_520


@pytest.mark.unit
def test_the_measured_boundary_falls_on_the_right_side_of_the_window() -> None:
    """
    Both ends were compiled and handed to QEMU: a 1008.1 MiB image ran, a 1036 MiB one
    was refused. The window has to agree with what the emulator actually did.
    """
    ran = int(1008.1 * MIB)
    refused = 1036 * MIB
    assert ran < MAX_LOADABLE_IMAGE_BYTES
    assert refused > MAX_LOADABLE_IMAGE_BYTES

    # The 1 GB model shipped in the release notes clears the wall by under 12 MiB. If a
    # future change eats that margin, this is the test that should say so.
    assert MAX_LOADABLE_IMAGE_BYTES - ran == pytest.approx(11.9 * MIB, abs=0.1 * MIB)


@pytest.mark.unit
def test_hardware_float_detection_matches_every_shipped_target() -> None:
    """`g` is shorthand for `imafd`, so it counts; `c` is compression and must not."""
    expected = {
        "RV32IMC": False,
        "RV32IMAC": False,
        "RV32EMC": False,
        "RV64GC": True,
        "RV64IMAFDC": True,
        "RV64GCV": True,
    }
    assert set(expected) == set(TARGETS)
    for name, has_fpu in expected.items():
        assert _has_hardware_float(TARGETS[name]) is has_fpu, name


@pytest.mark.unit
def test_soft_float_targets_are_given_the_time_they_actually_need() -> None:
    """
    all_minilm_l6_v2.onnx, 128 MiB region, 7 counted inferences. On RV64GC it finished
    in 220.5 s and passed. On RV32IMC the identical model needed 807.4 s -- 20.5x the
    guest cycles, because every FP32 multiply becomes a soft-float call -- and was
    killed at the 537 s hardware-float ceiling.
    """
    hard = _qemu_timeout_seconds(MIN_RAM_BYTES, 7, TARGETS["RV64GC"])
    soft = _qemu_timeout_seconds(MIN_RAM_BYTES, 7, TARGETS["RV32IMC"])

    assert hard == 537
    assert soft > 807, "the ceiling must outlast the run that was measured under it"
    assert soft == int(128 * 7 * QEMU_SOFT_FLOAT_SECONDS_PER_MIB_PER_RUN)


@pytest.mark.unit
def test_omitting_the_variant_keeps_the_ceiling_it_always_had() -> None:
    """The parameter is optional so existing callers are unaffected by its arrival."""
    for ram, runs in ((MIN_RAM_BYTES, 7), (1040 * MIB, 2), (512 * MIB, 1)):
        assert _qemu_timeout_seconds(ram, runs) == _qemu_timeout_seconds(ram, runs, TARGETS["RV64GC"])

    assert QEMU_SOFT_FLOAT_SECONDS_PER_MIB_PER_RUN > QEMU_SECONDS_PER_MIB_PER_RUN


@pytest.mark.unit
@pytest.mark.parametrize("is_64", [True, False])
def test_elf_load_span_reads_the_program_headers(tmp_path, is_64: bool) -> None:
    """
    Read from PT_LOAD, not from the file size: .bss occupies addresses without occupying
    bytes, so the span is the larger of the two and it is the span QEMU overlap-checks.

    Cross-checked against the bundled `riscv-none-elf-readelf -lW` on real linked images
    of all_minilm_l6_v2.onnx, which is where the ELF32/ELF64 split matters:

      RV32IMC  LOAD 0x80200000  filesz 0x5648750  memsz 0x6a16f9c
      RV64GC   LOAD 0x80200000  filesz 0x564b2d8  memsz 0x6a19b38

    `_elf_load_span` returned exactly those addresses for both, and both spans exceed the
    file on disk by about 20 MiB.
    """
    elf = _fake_elf(tmp_path / "m.elf", is_64=is_64, paddr=RAM_ORIGIN, memsz=64 * MIB)
    assert _elf_load_span(elf) == (RAM_ORIGIN, RAM_ORIGIN + 64 * MIB)


@pytest.mark.unit
def test_elf_load_span_returns_none_rather_than_guessing(tmp_path) -> None:
    """A parse failure must cost a worse message, never a wrong answer."""
    junk = tmp_path / "not.elf"
    junk.write_bytes(b"this is not an ELF file")
    assert _elf_load_span(str(junk)) is None
    assert _elf_load_span(str(tmp_path / "missing.elf")) is None


@pytest.mark.unit
def test_an_oversized_image_is_refused_before_qemu_is_even_started(tmp_path, monkeypatch) -> None:
    """
    The check exists so the failure names the model rather than the device tree. It also
    has to run before the process is spawned -- there is no point paying for a load that
    is known to be refused.
    """
    elf = _fake_elf(tmp_path / "big.elf", is_64=True, paddr=RAM_ORIGIN, memsz=MAX_LOADABLE_IMAGE_BYTES + 1)
    monkeypatch.setattr(runner_module, "find_qemu", lambda _bits: ("qemu", "qemu-system-riscv64"))

    def _never(*_args, **_kwargs):
        raise AssertionError("QEMU was started for an image already known not to load")

    monkeypatch.setattr(runner_module.subprocess, "run", _never)

    with pytest.raises(EmulatorImageLimitError) as excinfo:
        run_and_measure(_artifact(elf))

    err = excinfo.value
    assert err.limit_bytes == MAX_LOADABLE_IMAGE_BYTES
    assert err.required_bytes == MAX_LOADABLE_IMAGE_BYTES + 1
    assert err.fdt_address == QEMU_FDT_BASE
    # Still catchable as the memory failure it is.
    assert isinstance(err, MemoryLimitExceededError)


@pytest.mark.unit
def test_an_image_that_just_fits_is_not_refused(tmp_path, monkeypatch) -> None:
    """The boundary is exclusive: reaching the device tree address is the failure."""
    elf = _fake_elf(tmp_path / "fits.elf", is_64=True, paddr=RAM_ORIGIN, memsz=MAX_LOADABLE_IMAGE_BYTES)
    monkeypatch.setattr(runner_module, "find_qemu", lambda _bits: ("qemu", "qemu-system-riscv64"))
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, stdout="RUN_CYCLES: 100\n", stderr=""),
    )

    result = run_and_measure(_artifact(elf))
    assert result.mean_ms > 0


@pytest.mark.unit
def test_the_overlap_backstop_reads_qemus_own_addresses(tmp_path, monkeypatch) -> None:
    """
    If the pre-check cannot see the problem -- an unparsable ELF, or a QEMU that moves
    the device tree -- the emulator's own complaint is still classified rather than
    dumped. The addresses come from its message, not from the constants.
    """
    elf = tmp_path / "x.elf"
    elf.write_bytes(b"not an elf")
    monkeypatch.setattr(runner_module, "find_qemu", lambda _bits: ("qemu", "qemu-system-riscv64"))
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 1, stdout="", stderr=REAL_OVERLAP_STDERR),
    )

    with pytest.raises(EmulatorImageLimitError) as excinfo:
        run_and_measure(_artifact(str(elf)))

    err = excinfo.value
    assert err.fdt_address == 0xBFE00000
    assert err.required_bytes == 0xC122A258 - 0x80200000
    assert err.limit_bytes == MAX_LOADABLE_IMAGE_BYTES


@pytest.mark.unit
def test_a_timeout_is_reported_as_a_timeout(tmp_path, monkeypatch) -> None:
    """
    The failure the user actually hit. A killed run is not a compilation failure, and
    on a soft-float target it is the expected outcome for an FP32 transformer.
    """
    elf = _fake_elf(tmp_path / "m.elf", is_64=False, paddr=RAM_ORIGIN, memsz=64 * MIB)
    monkeypatch.setattr(runner_module, "find_qemu", lambda _bits: ("qemu", "qemu-system-riscv32"))

    def _timeout(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="qemu", timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(runner_module.subprocess, "run", _timeout)

    with pytest.raises(SimulationTimeoutError) as excinfo:
        run_and_measure(_artifact(elf, target="RV32IMC"))

    err = excinfo.value
    assert err.soft_float is True
    assert err.target == "RV32IMC"
    assert err.run_count == 7
    assert err.timeout_seconds == _qemu_timeout_seconds(MIN_RAM_BYTES, 7, TARGETS["RV32IMC"])
    assert "Unexpected" not in str(err)


@pytest.mark.unit
def test_a_hardware_float_timeout_does_not_blame_the_fpu(tmp_path, monkeypatch) -> None:
    """RV64GC has an FPU, so a timeout there needs a different explanation."""
    elf = _fake_elf(tmp_path / "m.elf", is_64=True, paddr=RAM_ORIGIN, memsz=64 * MIB)
    monkeypatch.setattr(runner_module, "find_qemu", lambda _bits: ("qemu", "qemu-system-riscv64"))
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd="qemu", timeout=1)),
    )

    with pytest.raises(SimulationTimeoutError) as excinfo:
        run_and_measure(_artifact(elf, target="RV64GC"))

    assert excinfo.value.soft_float is False
    diagnosis = get_offline_explanation(classify_failure(excinfo.value))
    assert "no hardware floating-point unit" not in diagnosis


@pytest.mark.unit
def test_the_emulator_limit_is_diagnosed_as_the_emulators() -> None:
    """
    The mitigation has to be true. Raising `-m` does not help -- the address was
    measured at 0xBFE00000 for every memory size tried -- and the ELF itself is fine.
    """
    err = EmulatorImageLimitError(
        limit_bytes=MAX_LOADABLE_IMAGE_BYTES,
        required_bytes=1036 * MIB,
        fdt_address=QEMU_FDT_BASE,
        details="measured",
    )
    context = classify_failure(err)
    assert context.error_type == "emulator_image_limit"
    assert context.metadata["fdt_address"] == QEMU_FDT_BASE

    diagnosis = get_offline_explanation(context)
    assert "1020.0 MiB" in diagnosis
    assert "1036.0 MiB" in diagnosis
    assert "not of RISC-V" in diagnosis
    assert "Do not raise QEMU's `-m`" in diagnosis
    # It must not be described as the target running out of memory.
    assert "workspace footprint" not in diagnosis


@pytest.mark.unit
def test_the_soft_float_diagnosis_names_the_target_that_works() -> None:
    err = SimulationTimeoutError(
        timeout_seconds=537, target="RV32IMC", run_count=7, soft_float=True, details="measured"
    )
    context = classify_failure(err)
    assert context.error_type == "simulation_timeout"

    diagnosis = get_offline_explanation(context)
    assert "537" in diagnosis
    assert "RV32IMC" in diagnosis
    assert "RV64GC" in diagnosis
    assert "no hardware floating-point unit" in diagnosis
    assert "nothing crashed" in diagnosis


@pytest.mark.unit
def test_the_new_metadata_survives_the_whitelist() -> None:
    """
    The whitelist is what leaves the machine when an API key is configured. A field it
    does not know is dropped silently, which would strip the addresses out of the
    diagnosis without any sign that it happened.
    """
    emulator = whitelist_payload(
        "emulator_image_limit",
        {"limit_bytes": 1, "required_bytes": 2, "fdt_address": 3, "details": "d", "weights": [1] * 99},
    )
    assert emulator == {"limit_bytes": 1, "required_bytes": 2, "fdt_address": 3, "details": "d"}

    timeout = whitelist_payload(
        "simulation_timeout",
        {"timeout_seconds": 537, "target": "RV32IMC", "run_count": 7, "soft_float": True, "details": "d"},
    )
    assert timeout == {
        "timeout_seconds": 537,
        "target": "RV32IMC",
        "run_count": 7,
        "soft_float": True,
        "details": "d",
    }


@pytest.mark.unit
def test_unknown_error_types_still_fall_through_unchanged() -> None:
    """The generic branch is the safety net; adding two types must not disturb it."""
    diagnosis = get_offline_explanation(
        DiagnosisContext(error_type="something_new", metadata={"message": "boom"})
    )
    assert diagnosis.startswith("Unexpected Compilation Failure:")
