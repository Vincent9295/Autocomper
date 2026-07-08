import sys
sys.setrecursionlimit(10000)
from cx_Freeze import setup, Executable
from config import VERSION

base = 'Win32GUI' if sys.platform == 'win32' else None

includefiles = ['ffmpeg/', 'img/', 'models/',
    ('C:/Windows/System32/vcomp140.dll', 'vcomp140.dll'),
    ('C:/Windows/System32/msvcp140.dll', 'msvcp140.dll'),
    ('C:/Windows/System32/vcruntime140.dll', 'vcruntime140.dll'),
    ('C:/Windows/System32/vcruntime140_1.dll', 'vcruntime140_1.dll'),
    ('C:/Windows/System32/concrt140.dll', 'concrt140.dll')]
includes = ['yt_dlp.utils._deprecated']
excludes = ['Tkinter']
packages = ['moviepy', 'librosa', 'scipy', 'sklearn']

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
