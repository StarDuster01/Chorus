# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
import site

# Add site-packages to path so PyInstaller can find all modules
site_packages = site.getsitepackages()
for site_package in site_packages:
    sys.path.insert(0, site_package)

a = Analysis(
    ['ragbot-backend/wrapper.py'],
    pathex=[],
    binaries=[],
    datas=[
        (str(Path('ragbot-frontend/build').absolute()), 'frontend_build'),
        ('production_config.json', '.'),
        ('ragbot-backend/.env', '.'),
        ('ragbot-backend/users.json', '.'),
        ('ragbot-backend/datasets', 'datasets'),
        ('ragbot-backend/bots', 'bots'),
        ('ragbot-backend/conversations', 'conversations'),
        ('ragbot-backend/chorus', 'chorus'),
        ('ragbot-backend/uploads', 'uploads'),
        ('ragbot-backend/chroma_db', 'chroma_db')
    ],
    hiddenimports=[
        'engineio.async_drivers.eventlet',
        'flask_cors',
        'anthropic',
        'openai',
        'PyPDF2',
        'chromadb',
        'pptx',
        'docx',
        'PIL',
        'bcrypt',
        'jwt'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RAGBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RAGBot',
)
