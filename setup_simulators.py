import os
import sys
import urllib.request
import zipfile
import shutil

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
QEMU_DIR = os.path.join(PROJECT_DIR, "qemu")
RENODE_DIR = os.path.join(PROJECT_DIR, "renode")

QEMU_URL = "https://github.com/xpack-dev-tools/qemu-riscv-xpack/releases/download/v9.2.4-1/xpack-qemu-riscv-9.2.4-1-win32-x64.zip"
RENODE_URL = "https://github.com/renode/renode/releases/download/v1.16.1/renode-1.16.1.windows-portable-dotnet.zip"

QEMU_ZIP = os.path.join(PROJECT_DIR, "qemu.zip")
RENODE_ZIP = os.path.join(PROJECT_DIR, "renode.zip")

def log(msg):
    print(f"[TATVA SIM SETUP] {msg}")

def download_file(url, dest):
    if os.path.exists(dest):
        log(f"File already exists: {dest}, skipping download.")
        return
    log(f"Downloading {url} to {dest}...")
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    log("Download completed successfully.")

def setup_qemu():
    if os.path.exists(QEMU_DIR):
        log("QEMU directory already exists. Skipping setup.")
        return
    
    download_file(QEMU_URL, QEMU_ZIP)
    
    log("Extracting QEMU ZIP to temp location...")
    temp_extract_dir = os.path.join(PROJECT_DIR, "qemu_temp")
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(QEMU_ZIP, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    extracted_folders = [f for f in os.listdir(temp_extract_dir) if os.path.isdir(os.path.join(temp_extract_dir, f))]
    if not extracted_folders:
        raise RuntimeError("No folder found in QEMU zip archive")
        
    source_dir = os.path.join(temp_extract_dir, extracted_folders[0])
    log(f"Moving QEMU from {source_dir} to {QEMU_DIR}...")
    shutil.move(source_dir, QEMU_DIR)
    
    log("Cleaning up temporary zip and folder...")
    os.remove(QEMU_ZIP)
    shutil.rmtree(temp_extract_dir)
    log("QEMU setup complete.")

def setup_renode():
    if os.path.exists(RENODE_DIR):
        log("Renode directory already exists. Skipping setup.")
        return
    
    download_file(RENODE_URL, RENODE_ZIP)
    
    log("Extracting Renode ZIP to temp location...")
    temp_extract_dir = os.path.join(PROJECT_DIR, "renode_temp")
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(RENODE_ZIP, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    extracted_folders = [f for f in os.listdir(temp_extract_dir) if os.path.isdir(os.path.join(temp_extract_dir, f))]
    if not extracted_folders:
        raise RuntimeError("No folder found in Renode zip archive")
        
    source_dir = os.path.join(temp_extract_dir, extracted_folders[0])
    log(f"Moving Renode from {source_dir} to {RENODE_DIR}...")
    shutil.move(source_dir, RENODE_DIR)
    
    log("Cleaning up temporary zip and folder...")
    os.remove(RENODE_ZIP)
    shutil.rmtree(temp_extract_dir)
    log("Renode setup complete.")

if __name__ == "__main__":
    try:
        setup_qemu()
        setup_renode()
        log("Simulators setup finished successfully!")
    except Exception as e:
        log(f"ERROR during simulator setup: {e}")
        sys.exit(1)
