import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_TARGET = Path(__file__).resolve().parent / "AutoComper Enhanced v1.1.0"
OLD_BUILD_DIRECTORY = "exe.win-amd64-3.11"
EXACT_ARTIFACTS = {
    "ffmpeg",
    "img",
    "models",
    "lib",
    "autocomper.exe",
    "frozen_application_license.txt",
}
RUNTIME_USER_FILES = {"preferences.ini", "autocomper_presets.json"}


def is_generated_artifact(path):
    """Return whether a top-level path is a known cx_Freeze artifact."""
    path = Path(path)
    name = path.name.lower()
    if name in {artifact.lower() for artifact in EXACT_ARTIFACTS}:
        return True
    if name == OLD_BUILD_DIRECTORY.lower():
        return True
    if name.startswith("python") and name.endswith(".dll"):
        return True
    return name.endswith((".zip", ".manifest"))


def cleanup_generated_artifacts(target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in target.iterdir():
        if not is_generated_artifact(path):
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def cleanup_target(target, preserve_user=False):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if not preserve_user:
        for path in target.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        return
    cleanup_generated_artifacts(target)


def build_command(project_root, target):
    del project_root
    return [
        sys.executable,
        "setup.py",
        "build_exe",
        "--build-exe",
        str(Path(target)),
    ]


def validate_target(target):
    target = Path(target)
    required = (target / "autocomper.exe", target / "ffmpeg" / "windows" / "ffmpeg.exe")
    missing = [str(path) for path in required if not path.is_file()]
    forbidden = [
        str(target / "ffmpeg" / platform)
        for platform in ("linux", "osx")
        if (target / "ffmpeg" / platform).exists()
    ]
    forbidden_user_files = [
        str(target / name)
        for name in RUNTIME_USER_FILES
        if (target / name).exists()
    ]
    stray_dlls = [
        str(path)
        for path in target.glob("*.dll")
        if path.name.lower() not in {"python3.dll", "python311.dll"}
    ]
    if missing or forbidden or forbidden_user_files or stray_dlls:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if forbidden:
            details.append("forbidden platform artifacts: " + ", ".join(forbidden))
        if forbidden_user_files:
            details.append("runtime user files: " + ", ".join(forbidden_user_files))
        if stray_dlls:
            details.append("dlls outside lib: " + ", ".join(stray_dlls))
        raise RuntimeError("Invalid package output (" + "; ".join(details) + ")")
    return True


def cleanup_runtime_user_files(target):
    """Remove per-user runtime state from a distributable package."""
    target = Path(target)
    for name in RUNTIME_USER_FILES:
        path = target / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def run_packaging(target=DEFAULT_TARGET, kill=True, preserve_user=False, runner=subprocess.run):
    project_root = Path(__file__).resolve().parent
    target = Path(target)
    if kill:
        runner(
            ["taskkill", "/F", "/T", "/IM", "autocomper.exe"],
            cwd=project_root,
            check=False,
        )

    cleanup_target(target, preserve_user=preserve_user)
    result = runner(
        build_command(project_root, target),
        cwd=project_root,
        check=False,
    )
    build_dir = project_root / "build"
    if build_dir.exists():
        cleanup_generated_artifacts(build_dir)
    if result.returncode:
        return result.returncode
    cleanup_runtime_user_files(target)
    try:
        validate_target(target)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the Windows AutoComper package.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--no-kill", action="store_true", help="Skip taskkill for testing/debugging.")
    parser.add_argument(
        "--preserve-user",
        action="store_true",
        help="Preserve user files in the existing target for local testing.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_packaging(
        args.target, kill=not args.no_kill, preserve_user=args.preserve_user)


if __name__ == "__main__":
    raise SystemExit(main())
