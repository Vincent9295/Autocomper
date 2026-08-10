"""Local, deterministic caches for remote media processing."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


_SAFE_METADATA_KEYS = {
    "format", "format_id", "codec", "acodec", "abr", "tbr",
    "sample_rate", "channels",
}
_TOKEN_KEY = re.compile(r"cookie|token|secret|authorization|credential", re.IGNORECASE)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _number(value: float) -> str:
    return format(float(value), ".12g")


class CacheStore:
    """Manage cache paths and atomic local persistence without network access."""

    def __init__(self, root: str | os.PathLike[str] | None = None):
        if root is None:
            root = os.getenv("LOCALAPPDATA")
            root = Path(root) / "AutoComper" / "cache" if root else Path.home() / ".cache" / "autocomper"
        self.root = Path(root).expanduser().resolve()
        self.detection_root = self.root / "detection"
        self.audio_root = self.root / "audio"
        self.segment_root = self.root / "segments"

    def ensure_ready(self) -> Path:
        """Create the cache tree and verify that the selected root is writable."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            for directory in (self.detection_root, self.audio_root, self.segment_root):
                directory.mkdir(parents=True, exist_ok=True)
            probe = self.root / ".write-test"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            raise OSError(f"Remote cache root is not writable: {self.root} ({exc})") from exc
        return self.root

    def _is_safe_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root)
        except ValueError:
            return False
        return True

    def get_cache_size(self) -> int:
        """Return the total size of regular files stored below the cache root."""
        if not self.root.is_dir():
            return 0
        total = 0
        for path in self.root.rglob("*"):
            if path.is_symlink() or not path.is_file() or not self._is_safe_path(path):
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def clear(self, kind: str | None = None) -> None:
        """Delete cache contents while retaining the root and category directories."""
        normalized = "all" if kind is None else str(kind).strip().lower()
        roots = {
            "detection": self.detection_root,
            "audio": self.audio_root,
            "segments": self.segment_root,
        }
        if normalized == "all":
            selected = tuple(roots.values())
        elif normalized in roots:
            selected = (roots[normalized],)
        else:
            raise ValueError("kind must be one of: all, detection, audio, segments")

        for directory in selected:
            if not self._is_safe_path(directory) or not directory.exists():
                continue
            for path in sorted(directory.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if not self._is_safe_path(path):
                    continue
                try:
                    if path.is_dir() and not path.is_symlink():
                        path.rmdir()
                    else:
                        path.unlink()
                except OSError:
                    continue

    def _key_path(self, directory: Path, key: Any, suffix: str) -> Path:
        return directory / (_canonical_hash(key) + suffix)

    def get_detection_cache_path(
        self,
        platform: str,
        source_id: str,
        model: str,
        precision: str,
        block_size: int,
        threshold: float,
        focus_idx: int | None,
        relevant_flags: Mapping[str, Any] | None = None,
    ) -> Path:
        key = {
            "platform": str(platform).strip().lower(),
            "source_id": str(source_id).strip(),
            "model": str(model),
            "precision": str(precision),
            "block_size": int(block_size),
            "threshold": float(threshold),
            "focus_idx": focus_idx,
            "relevant_flags": dict(relevant_flags or {}),
        }
        return self._key_path(self.detection_root, key, ".json")

    def save_detection_result(self, *args: Any) -> Path:
        result = args[-1]
        path = self.get_detection_cache_path(*args[:-1])
        self.save_json(path, result)
        return path

    def load_detection_result(self, *args: Any) -> Any | None:
        return self.read_json(self.get_detection_cache_path(*args))

    def get_detection_failure_path(self, *args: Any) -> Path:
        path = self.get_detection_cache_path(*args)
        return path.with_suffix(".failed.json")

    def save_detection_failure(self, *args: Any) -> Path:
        result = args[-1]
        path = self.get_detection_failure_path(*args[:-1])
        self.save_json(path, result)
        return path

    def load_detection_failure(self, *args: Any) -> Any | None:
        return self.read_json(self.get_detection_failure_path(*args))

    def has_detection_failure(self, *args: Any) -> bool:
        return self.get_detection_failure_path(*args).is_file()

    def clear_detection_failure(self, *args: Any) -> None:
        self.get_detection_failure_path(*args).unlink(missing_ok=True)

    def get_audio_cache_path(
        self,
        source_identity: str,
        audio_url: str,
        audio_format: str,
        format_identity: Mapping[str, Any] | None = None,
    ) -> Path:
        extension = str(audio_format).strip().lower().lstrip(".") or "audio"
        key = {"source_identity": str(source_identity), "format": extension}
        for name in ("format_id", "codec", "acodec", "abr", "tbr", "sample_rate", "channels"):
            if format_identity and format_identity.get(name) is not None:
                key[name] = format_identity[name]
        return self._key_path(self.audio_root, key, "." + extension)

    def has_audio_cache(
        self, source_identity: str, audio_url: str, audio_format: str,
        format_identity: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.resolve_audio_cache(
            source_identity, audio_url, audio_format, format_identity=format_identity
        ) is not None

    def save_audio_cache(
        self,
        source_identity: str,
        audio_url: str,
        audio_format: str,
        data: bytes,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.get_audio_cache_path(
            source_identity, audio_url, audio_format, format_identity=metadata
        )
        self.save_file(path, data)
        safe_metadata = {"source_identity": str(source_identity), "format": str(audio_format).lstrip(".")}
        for key, value in (metadata or {}).items():
            if key in _SAFE_METADATA_KEYS and not _TOKEN_KEY.search(str(key)):
                safe_metadata[key] = value
        self.save_json(path.with_suffix(".json"), safe_metadata)
        return path

    def save_audio_cache_file(
        self,
        source_identity: str,
        audio_url: str,
        audio_format: str,
        source_path: str | os.PathLike[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically register a completed audio file without loading it into RAM."""
        path = self.get_audio_cache_path(
            source_identity, audio_url, audio_format, format_identity=metadata
        )
        source = Path(source_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, path)
        safe_metadata = {"source_identity": str(source_identity), "format": str(audio_format).lstrip(".")}
        for key, value in (metadata or {}).items():
            if key in _SAFE_METADATA_KEYS and not _TOKEN_KEY.search(str(key)):
                safe_metadata[key] = value
        self.save_json(path.with_suffix(".json"), safe_metadata)
        return path

    def resolve_audio_cache(
        self,
        source_identity: str,
        audio_url: str,
        audio_format: str,
        format_identity: Mapping[str, Any] | None = None,
    ) -> Path | None:
        path = self.get_audio_cache_path(
            source_identity, audio_url, audio_format, format_identity=format_identity
        )
        if path.is_file():
            return path

        extension = str(audio_format).strip().lower().lstrip(".") or "audio"
        source_key = str(source_identity)
        for metadata_path in self.audio_root.glob("*.json"):
            metadata = self.read_json(metadata_path)
            if not isinstance(metadata, Mapping):
                continue
            if (metadata.get("source_identity") != source_key
                    or str(metadata.get("format", "")).lstrip(".").lower() != extension):
                continue
            legacy_path = metadata_path.with_suffix("." + extension)
            if legacy_path.is_file():
                return legacy_path
        return None

    def get_segment_cache_path(
        self, source_identity: str, start: float, end: float, padding: float,
        extension: str = "mp4", media_type: str = "video"
    ) -> Path:
        ext = str(extension).lstrip(".") or "mp4"
        key = {
            "source_identity": str(source_identity), "start": _number(start),
            "end": _number(end), "padding": _number(padding),
            "media_type": str(media_type).lower(),
        }
        return self._key_path(self.segment_root, key, "." + ext)

    def save_segment_cache(
        self, source_identity: str, start: float, end: float, padding: float,
        data: bytes, extension: str = "mp4", media_type: str = "video"
    ) -> Path:
        path = self.get_segment_cache_path(
            source_identity, start, end, padding, extension, media_type
        )
        self.save_file(path, data)
        self.save_json(path.with_suffix(".json"), {
            "source_identity": str(source_identity),
            "start": float(start), "end": float(end), "padding": float(padding),
            "media_type": str(media_type).lower(),
        })
        return path

    def find_cached_segment(
        self, source_identity: str, start: float, end: float, padding: float,
        extension: str = "mp4", media_type: str = "video"
    ) -> Path | None:
        source_key = str(source_identity)
        requested_start = float(start) - float(padding)
        requested_end = float(end) + float(padding)
        for metadata_path in self.segment_root.glob("*.json"):
            metadata = self.read_json(metadata_path)
            if (not isinstance(metadata, Mapping) or metadata.get("source_identity") != source_key
                    or metadata.get("media_type", "video") != str(media_type).lower()):
                continue
            try:
                cached_start = float(metadata["start"])
                cached_end = float(metadata["end"])
            except (TypeError, ValueError):
                continue
            if cached_start <= requested_start and cached_end >= requested_end:
                ext = str(extension).lstrip(".") or "mp4"
                data_path = self.get_segment_cache_path(
                    source_identity, cached_start, cached_end,
                    metadata.get("padding", 0), ext, media_type
                )
                if data_path.is_file():
                    return data_path
        return None

    @staticmethod
    def save_file(path: str | os.PathLike[str], data: bytes) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".cache-", suffix=".tmp", dir=str(destination.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def save_json(cls, path: str | os.PathLike[str], value: Any) -> None:
        data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cls.save_file(path, data)

    @staticmethod
    def read_json(path: str | os.PathLike[str]) -> Any | None:
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError, TypeError):
            return None
