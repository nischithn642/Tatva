import os
import sys
import urllib.request
import zipfile
import subprocess
import shutil

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
TOOLCHAIN_DIR = os.path.join(PROJECT_DIR, "riscv-toolchain")
VENV_DIR = os.path.join(PROJECT_DIR, ".venv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

# Toolchain URL (xPack RISC-V GCC 15.2.0-1)
TOOLCHAIN_URL = "https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v15.2.0-1/xpack-riscv-none-elf-gcc-15.2.0-1-win32-x64.zip"
TOOLCHAIN_ZIP = os.path.join(PROJECT_DIR, "riscv-toolchain.zip")

# Tiny Random BERT ONNX Model URL from Hugging Face
MODEL_URL = "https://huggingface.co/optimum-intel-internal-testing/tiny-random-bert/resolve/main/onnx/model.onnx"

def log(msg):
    print(f"[TATVA SETUP] {msg}")

def download_file(url, dest):
    if os.path.exists(dest):
        log(f"File already exists: {dest}, skipping download.")
        return
    log(f"Downloading {url} to {dest}...")
    
    # Custom User-Agent to prevent HTTP 403 Forbidden from Hugging Face/GitHub
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    log("Download completed successfully.")

def setup_toolchain():
    if os.path.exists(TOOLCHAIN_DIR):
        log("Toolchain directory already exists. Skipping toolchain setup.")
        return
    
    download_file(TOOLCHAIN_URL, TOOLCHAIN_ZIP)
    
    log(f"Extracting toolchain ZIP to temp location...")
    temp_extract_dir = os.path.join(PROJECT_DIR, "toolchain_temp")
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(TOOLCHAIN_ZIP, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    # xPack zips contain a top-level folder like xpack-riscv-none-elf-gcc-15.2.0-1
    extracted_folders = [f for f in os.listdir(temp_extract_dir) if os.path.isdir(os.path.join(temp_extract_dir, f))]
    if not extracted_folders:
        raise RuntimeError("No folder found in toolchain zip archive")
    
    source_dir = os.path.join(temp_extract_dir, extracted_folders[0])
    log(f"Moving toolchain from {source_dir} to {TOOLCHAIN_DIR}...")
    shutil.move(source_dir, TOOLCHAIN_DIR)
    
    log("Cleaning up temporary zip and folder...")
    os.remove(TOOLCHAIN_ZIP)
    shutil.rmtree(temp_extract_dir)
    log("Toolchain setup complete.")

def setup_venv():
    if os.path.exists(VENV_DIR):
        log("Virtual environment already exists.")
    else:
        log("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
        log("Virtual environment created.")
        
    # Get venv python path
    if sys.platform == "win32":
        venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
        venv_pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        venv_python = os.path.join(VENV_DIR, "bin", "python")
        venv_pip = os.path.join(VENV_DIR, "bin", "pip")
        
    log("Upgrading pip...")
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    log("Installing dependencies in virtual environment...")
    # Install dependencies individually to handle any potential version issues
    subprocess.run([venv_pip, "install", "numpy>=1.24.0", "onnx>=1.14.0", "onnxruntime>=1.15.0", "click>=8.0.0"], check=True)
    
    log("Installing apache-tvm in virtual environment...")
    try:
        subprocess.run([venv_pip, "install", "apache-tvm"], check=True)
        log("apache-tvm installed successfully.")
    except Exception as e:
        log(f"Failed to install apache-tvm via pip: {e}. We may need to build from source or check python version compatibility.")
        raise e
        
    log("Installing tatva package in editable mode...")
    subprocess.run([venv_pip, "install", "-e", "."], check=True)
    log("Virtual environment setup completed.")

def setup_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_file(MODEL_URL, MODEL_PATH)
    log(f"Sample model is available at {MODEL_PATH}")

if __name__ == "__main__":
    try:
        log("Starting TATVA Step 1 Environment Setup...")
        setup_toolchain()
        setup_model()
        setup_venv()
        log("Setup script finished successfully!")
    except Exception as e:
        log(f"ERROR during setup: {e}")
        sys.exit(1)
