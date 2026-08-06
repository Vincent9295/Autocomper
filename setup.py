import sys
from pathlib import Path

import numpy

sys.setrecursionlimit(5000)
from cx_Freeze import setup, Executable
from config import VERSION
from packaging_helpers import numpy_dll_include_files

base = 'Win32GUI' if sys.platform == 'win32' else None

includefiles = [('ffmpeg/windows/', 'ffmpeg/windows'), 'img/', 'models/',
    ('C:/Windows/System32/vcomp140.dll', 'vcomp140.dll'),
    ('C:/Windows/System32/msvcp140.dll', 'msvcp140.dll'),
    ('C:/Windows/System32/vcruntime140.dll', 'vcruntime140.dll'),
    ('C:/Windows/System32/vcruntime140_1.dll', 'vcruntime140_1.dll')]

# NumPy's Windows extension depends on OpenBLAS DLLs stored outside the
# package directory. cx_Freeze includes the .pyd file but does not discover
# these sibling DLLs automatically.
numpy_site_dir = Path(numpy.__file__).resolve().parent.parent
includefiles.extend(numpy_dll_include_files(numpy_site_dir))
includes = ['yt_dlp.utils._deprecated']
excludes = ['Tkinter', 'torch', 'torchaudio', 'scipy', 'pandas',
            'numba', 'llvmlite', 'matplotlib', 'pyarrow', 'imageio_ffmpeg',
            'sklearn', 'sqlalchemy', 'cryptography', 'psycopg2', 'lxml']
packages = ['moviepy']

setup(
    name='AutoComper',
    version=VERSION,
    description='Automatic Comp Creation Tool',
    author='wz-bff',
    options={'build_exe': {'includes': includes, 'excludes': excludes,
                           'packages': packages, 'include_files': includefiles}},
    executables=[Executable('autocomper.py',
                            base=base)]
)
