from pathlib import Path


def numpy_dll_include_files(numpy_site_dir):
    """Return NumPy's Windows DLLs with the frozen app's lib destination."""
    numpy_site_dir = Path(numpy_site_dir)
    include_files = []
    for numpy_lib_dir in (numpy_site_dir / "numpy.libs", numpy_site_dir / "numpy" / ".libs"):
        if numpy_lib_dir.is_dir():
            include_files.extend(
                (str(dll), (Path("lib") / dll.name).as_posix())
                for dll in numpy_lib_dir.glob("*.dll")
            )
    return include_files
