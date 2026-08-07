# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for the TATVA desktop GUI.
#
# Build with: pyinstaller tatva.spec   (pip install pyinstaller first)
#
# Caveat worth knowing before you rely on this: Apache TVM ships large native
# libraries that PyInstaller does not discover by following imports. A frozen build
# starts, and the GUI, analyze and doctor panels work, but anything that actually
# compiles a model needs tvm's shared objects collected too -- see `collect_tvm`
# below. Treat this as the "no Python on the target machine" escape hatch, not as
# the primary distribution channel. The wheel (`uv build`) is the supported one.

import os

from PyInstaller.utils.hooks import collect_dynamic_libs

# Forward slashes on purpose: this spec is a Python file and has to build on
# Linux and macOS too, where the previous 'src\\tatva\\gui.py' resolved to a
# single filename containing backslashes and failed with "script not found".
block_cipher = None

# TVM's .so/.dll/.dylib files are loaded by ctypes at runtime, so no import
# statement points at them and PyInstaller cannot infer them.
try:
    collect_tvm = collect_dynamic_libs('tvm') + collect_dynamic_libs('tvm_ffi')
except Exception:  # tvm not installed in the build env -- fail at runtime, not here
    collect_tvm = []

a = Analysis(
    [os.path.join('src', 'tatva', 'gui.py')],
    pathex=['src'],
    binaries=collect_tvm,
    datas=[
        ('website', 'website'),
        ('assets', 'assets'),
        ('BASELINE.md', '.'),
        ('OPTIMIZATION.md', '.'),
    ],
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
        'webview',
        'click',
        'dotenv',
        'onnx',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Tatva',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[os.path.join('assets', 'logo.png')],
)
