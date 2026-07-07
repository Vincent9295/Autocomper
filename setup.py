import sys
sys.setrecursionlimit(10000)
from cx_Freeze import setup, Executable
from config import VERSION

base = 'Win32GUI' if sys.platform == 'win32' else None

includefiles = ['ffmpeg/', 'img/', 'models/']
includes = ['yt_dlp.utils._deprecated']
excludes = ['Tkinter', 'numba', 'scikit-learn']
packages = ['moviepy', 'librosa', 'soundfile', 'scipy']

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
