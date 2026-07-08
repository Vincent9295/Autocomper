# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['autocomper.py'],
    pathex=[],
    binaries=[],
    datas=[('ffmpeg', 'ffmpeg'), ('img', 'img'), ('models', 'models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchaudio', 'llvmlite', 'pyarrow', 'scipy', 'imageio_ffmpeg', 'pandas', 'matplotlib', 'numba', 'sqlalchemy', 'cryptography', 'psycopg2', 'lxml'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoComper Enhanced',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name='AutoComper Enhanced',
)
