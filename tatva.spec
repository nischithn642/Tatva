# -*- coding: utf-8 -*-
#
# PyInstaller spec for TATVA.exe — the primary way the app launches.
#
# Build:  pyinstaller tatva.spec        (or: python build_exe.py, which also zips it)
# Output: dist/TATVA/TATVA.exe
#
# One-FOLDER, not one-file, and that is deliberate. A one-file build unpacks the whole
# payload -- Apache TVM's native libraries included, which is most of a gigabyte -- into
# a temp directory on every single launch. That turns a double-click into a thirty-second
# wait and leaves the temp copy behind. The folder build starts in a couple of seconds,
# and it zips just as well, which is the shape this ships in.
#
# The one thing PyInstaller cannot work out on its own is TVM, in two separate ways:
#
#   1. Its .dll/.so files are opened by ctypes at runtime, so no import statement points
#      at them and following the import graph finds nothing. collect_dynamic_libs fixes
#      that -- it is what makes a frozen build able to actually compile a model rather
#      than only render the UI.
#   2. Its Python side imports backend modules by name at runtime. Static analysis sees
#      none of them, so the first frozen build failed on the very first analyze with
#      "No module named 'tvm.backend.cuda.operator.intrinsics'". collect_submodules
#      sweeps in all ~800 of them; they are pure Python and cost almost nothing.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.getcwd())

# Forward slashes on purpose: this spec is a Python file and has to build on Linux and
# macOS too, where 'src\\tatva\\gui.py' is a single filename containing backslashes.
ENTRY = os.path.join('src', 'tatva', 'gui.py')

binaries = []
datas = []
runtime_hidden = []

for pkg in ('tvm', 'tvm_ffi'):
    try:
        binaries += collect_dynamic_libs(pkg)
        datas += collect_data_files(pkg)
        runtime_hidden += collect_submodules(pkg)
    except Exception:
        # Not installed in this build environment. Fail at runtime with a real message
        # rather than turning the whole build into an import error here.
        pass

# Everything the app reads off disk at runtime. Paths are (source, dest-in-bundle).
#
# The UI files are listed one by one rather than as ('website', 'website'). That folder
# also holds an unrelated Node web app -- server.js, package-lock.json and 32 MB of
# node_modules -- none of which the desktop window loads, and all of which a whole-folder
# rule would have shipped inside the exe.
datas += [
    (os.path.join('website', 'index.html'), 'website'),
    (os.path.join('website', 'studio.css'), 'website'),
    (os.path.join('website', 'studio.js'), 'website'),
    (os.path.join('website', 'assets'), os.path.join('website', 'assets')),
]

# Brand assets, named individually for the same reason as the UI files above: shipping
# all of assets/ also ships rebuilt_gui_screenshot.png, half a megabyte of a screenshot
# of a GUI that no longer exists, inside every copy anyone downloads.
for art in ('logo-dark.png', 'logo-light.png', 'icon.png', 'tatva.ico'):
    if os.path.exists(os.path.join(ROOT, 'assets', art)):
        datas.append((os.path.join('assets', art), 'assets'))

# Sample models, so a first run has something to compile without hunting for a file.
# Only the ones list_sample_models() offers: shipping all of models/ would add the
# 17 MB pretrained checkpoint and a pile of .pth/.pb fixtures nobody opens from the GUI.
for sample in ('model_mlp.onnx', 'model.onnx', 'model_nano.onnx', 'model_medium.onnx'):
    src = os.path.join('models', sample)
    if os.path.exists(os.path.join(ROOT, src)):
        datas.append((src, 'models'))

# Docs the diagnostics panel quotes. Optional -- a missing one should not fail the build.
for doc in ('BASELINE.md', 'OPTIMIZATION.md', 'README.md'):
    if os.path.exists(os.path.join(ROOT, doc)):
        datas.append((doc, '.'))

a = Analysis(
    [ENTRY],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'tatva',
        'tatva._cache',
        'tatva._output',
        'tatva.cli',
        'tatva.compiler',
        'tatva.config',
        'tatva.diagnostics',
        'tatva.logging_setup',
        'tatva.optimizer',
        'tatva.runner',
        'tatva.toolchain',
        'scaffolding',
        'scaffolding.agent',
        'scaffolding.config',
        'scaffolding.executor',
        'scaffolding.llm_provider',
        'scaffolding.logger',
        'scaffolding.loop_agent',
        # pywebview picks its GUI backend at runtime by importing it, so the Windows
        # (EdgeChromium) backend is invisible to static analysis.
        'webview',
        'webview.platforms.winforms',
        'webview.platforms.edgechromium',
        'clr_loader',
        'pythonnet',
        'click',
        'dotenv',
        'onnx',
        'onnxruntime',
        'numpy',
        'tvm',
        *runtime_hidden,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Pulled in transitively and never used by the app. Dropping them saves a few
        # hundred megabytes in the shipped folder.
        'matplotlib',
        'notebook',
        'IPython',
        'jupyter',
        'pytest',
        'sphinx',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # one-folder: binaries go to COLLECT, not into the exe
    name='TATVA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is off on purpose. It corrupts some of TVM's native DLLs, and the failure
    # shows up as an unhelpful load error at runtime rather than at build time.
    upx=False,
    console=False,              # no console window behind the app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join('assets', 'tatva.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TATVA',
)
