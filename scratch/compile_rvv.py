import os
import subprocess
from tatva.runner import find_riscv_gcc, find_qemu, START_S, LINK_LD

def test_rvv_build():
    gcc_name, gcc_path = find_riscv_gcc()
    qemu_name, qemu_path = find_qemu(64)
    print(f"GCC: {gcc_path}")
    print(f"QEMU: {qemu_path}")

    scratch_dir = "scratch"
    os.makedirs(scratch_dir, exist_ok=True)

    with open("scratch/start.S", "w") as f:
        f.write(START_S)
    with open("scratch/link.ld", "w") as f:
        f.write(LINK_LD)

    compile_cmd = [
        gcc_path,
        "-O2",
        "-march=rv64gcv",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-ffreestanding",
        "-nostdlib",
        "-T", "scratch/link.ld",
        "scratch/start.S",
        "scratch/test_rvv.c",
        "-o", "scratch/test_rvv.elf",
        "-lgcc"
    ]

    print("Compiling RVV binary...")
    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Compilation FAILED:\n{res.stderr}")
        return

    print("Compilation SUCCESS!")

    qemu_cmd = [
        qemu_path,
        "-M", "virt",
        "-cpu", "rv64,v=true,vlen=128",
        "-kernel", "scratch/test_rvv.elf",
        "-nographic",
        "-icount", "shift=0"
    ]

    print("Running in QEMU with RVV enabled...")
    qemu_res = subprocess.run(qemu_cmd, capture_output=True, text=True, timeout=10)
    print(f"QEMU Output:\n{qemu_res.stdout}")
    if "RVV_VECTOR_SUCCESS" in qemu_res.stdout:
        print(">>> RVV VECTOR HARDWARE EMULATION VERIFIED WORKING! <<<")

if __name__ == "__main__":
    test_rvv_build()
