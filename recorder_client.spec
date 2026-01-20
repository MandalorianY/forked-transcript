# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for spch2txt Recorder Client (Windows)

Usage:
    pyinstaller recorder_client.spec

This creates a standalone executable for the recording client without
the server components (no Flask, no ML models).
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

block_cipher = None

# Get the project root directory
project_root = Path.cwd()

# Copy package metadata (required for Streamlit)
datas = copy_metadata('streamlit')
datas += copy_metadata('altair')
datas += copy_metadata('pillow')
datas += copy_metadata('requests')

# Copy Streamlit's static files (HTML, CSS, JS)
datas += collect_data_files('streamlit')

# Data files to include
datas += [
    (str(project_root / 'src' / 'audio' / '__init__.py'), 'src/audio'),
    (str(project_root / 'src' / 'audio' / 'capture.py'), 'src/audio'),
    (str(project_root / 'src' / 'audio' / 'utils.py'), 'src/audio'),
    (str(project_root / 'src' / 'client' / '__init__.py'), 'src/client'),
    (str(project_root / 'src' / 'client' / 'api_client.py'), 'src/client'),
    (str(project_root / 'src' / 'config.py'), 'src'),
    (str(project_root / 'src' / 'ui' / 'app_new.py'), 'src/ui'),
    (str(project_root / '.env.example'), '.'),
]

# Hidden imports (modules not automatically detected)
hiddenimports = [
    'streamlit',
    'streamlit.runtime',
    'streamlit.runtime.scriptrunner',
    'streamlit.runtime.scriptrunner.script_runner',
    'streamlit.runtime.scriptrunner.magic_funcs',
    'streamlit.runtime.state',
    'streamlit.runtime.state.session_state',
    'streamlit.web',
    'streamlit.web.cli',
    'streamlit.web.server',
    'streamlit.web.server.server',
    'streamlit.components.v1',
    'streamlit.logger',
    'sounddevice',
    'pyaudiowpatch',  # Windows-specific
    'requests',
    'dotenv',
    'altair',
    'altair.vegalite.v5',
    'altair.vegalite.v5.schema',
    'click',
    'tornado',
    'tornado.web',
    'tornado.ioloop',
    'watchdog',
    'watchdog.observers',
    'watchdog.observers.polling',
    'validators',
    'toml',
    'tzlocal',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
]

# Analysis
a = Analysis(
    ['launcher.py'],  # Use launcher script instead of app_new.py directly
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchaudio',
        'tensorflow',
        'whisper',
        'openai',
        'pyannote',
        'flask',
        'flask_cors',
        'werkzeug',
        'transformers',
        'matplotlib',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpchToText_Recorder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Enable console for troubleshooting
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add your icon path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SpchToText_Recorder',
)
