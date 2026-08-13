"""Metadata resolution and input expansion for remote VOD sources."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tempfile
import os
import re
import subprocess
import threading
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from utils import FFMPEG_PATH, run_tracked, run_tracked_progress
from remote_prefetch import (
    SpeedMonitor,
    iter_hls_bytes,
    iter_range_bytes,
    supports_hls_prefetch,
    supports_range_prefetch,
)
from progress import format_transfer_progress


class RemoteMediaError(Exception):
    """Base error for remote media input and resolution failures."""


class SourceResolveError(RemoteMediaError):
    """A single source could not be resolved."""


class SourceExpansionError(RemoteMediaError):
    """An input value could not be interpreted or expanded."""


class SegmentFetchError(RemoteMediaError):
    """A requested remote media segment could not be materialized."""


_FFMPEG_DETAIL_MAX = 400


def _sanitize_ffmpeg_detail(detail):
    """Trim a raw ffmpeg error block for safe, compact logging.

    FFmpeg progress output adds dozens of lines per failed remote fetch and
    signed stream URLs carry expiring tokens. Keep only the tail of the text
    (the actual error line lives at the end) and redact every URL query
    string so logs stay readable, token-free, and cheap to render.
    """
    text = str(detail or "")
    text = re.sub(r"(https?://[^\s?]+)\?[^\s]*", r"\1?[redacted]", text)
    if len(text) > _FFMPEG_DETAIL_MAX:
        text = "..." + text[-_FFMPEG_DETAIL_MAX:]
    return text


_BROWSER_COOKIE_NAMES = ("firefox", "chrome", "edge")
_COOKIES_FILE_PREFIX = "cookiesfile:"
_BILIBILI_HOST_MARKERS = ("bilibili.com", "b23.tv")
_BILIBILI_CDN_HOSTS = (
    "upos-sz-mirrorcos.bilivideo.com",
    "upos-sz-mirroraliov.bilivideo.com",
    "upos-sz-mirroralib.bilivideo.com",
)
_REMOTE_SEEK_PAD = 10.0


def _segment_is_http(url) -> bool:
    return str(url).startswith(("http://", "https://"))


@dataclass
class MediaSource:
    platform: str
    source_url: str
    source_id: str
    display_name: str = ""
    duration: float | None = None
    audio_url: str = ""
    video_url: str = ""
    http_headers: dict[str, str] = field(default_factory=dict)
    audio_headers: dict[str, str] = field(default_factory=dict)
    video_headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    audio_candidates: list[dict[str, Any]] = field(default_factory=list)
    video_candidates: list[dict[str, Any]] = field(default_factory=list)
    max_height: int | None = None
    resolved_at: float | None = None


@dataclass
class PlaylistEntry:
    """Flat metadata for one playlist item; no stream URL is resolved."""

    platform: str
    entry_id: str
    title: str
    webpage_url: str
    duration: float | None = None
    upload_date: str = ""
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlaylistDescriptor:
    """Bounded, paged flat metadata for a playlist-like input."""

    platform: str
    source_url: str
    title: str
    total_count: int
    page_size: int = 30
    _entries: list[PlaylistEntry] = field(default_factory=list, repr=False)
    _hydrate_entry: Callable[[PlaylistEntry], Mapping[str, Any] | None] | None = field(
        default=None, repr=False
    )
    _hydrated_indices: set[int] = field(default_factory=set, repr=False)

    def load_page(self, page_index: int, hydrate: bool = True) -> list[PlaylistEntry]:
        try:
            page_index = int(page_index)
        except (TypeError, ValueError):
            return []
        if page_index < 0:
            return []
        start = page_index * self.page_size
        page = self._entries[start:start + self.page_size]
        if hydrate:
            for entry in page:
                self.hydrate_entry(entry)
        return list(page)

    def needs_hydration(self, entry: PlaylistEntry) -> bool:
        return bool(
            self._hydrate_entry
            and entry.index not in self._hydrated_indices
        )

    def hydrate_entry(self, entry: PlaylistEntry) -> PlaylistEntry:
        if not self._hydrate_entry or entry.index in self._hydrated_indices:
            return entry
        try:
            hydrated = self._hydrate_entry(entry)
        except Exception:
            hydrated = None
        if hydrated:
            _merge_entry_metadata(entry, hydrated)
            entry.metadata["_resolved_info"] = dict(hydrated)
            self._hydrated_indices.add(entry.index)
            entry.metadata["hydration_failed"] = False
        else:
            entry.metadata["hydration_failed"] = True
        return entry


def apply_audio_candidate(source: MediaSource, candidate: Mapping[str, Any]) -> MediaSource:
    """Apply a real yt-dlp audio candidate without changing source identity."""
    url = str(candidate.get("url") or "")
    if not url:
        raise ValueError("audio candidate has no url")
    source.audio_url = url
    headers = candidate.get("http_headers") or {}
    source.audio_headers = {str(key): str(value) for key, value in headers.items()}
    for key in ("filesize", "filesize_approx", "clen"):
        source.metadata.pop(key, None)
    for key in ("filesize", "filesize_approx", "clen"):
        if candidate.get(key) is not None:
            source.metadata[key] = candidate.get(key)
    return source


def probe_audio_candidate(
    source: MediaSource,
    candidate: Mapping[str, Any],
    duration: float = 3,
    timeout: float = 5,
    run_func: Callable[..., Any] | None = None,
) -> float | None:
    """Probe a bounded portion of an audio candidate and return FFmpeg speed.

    ``timeout`` is deliberately short (5s): a probe is only a latency heuristic
    for candidate selection. On a large batch, every candidate that hangs eats
    a full timeout per URL (90 VODs × 12 URLs = up to ~6h of pure timeouts),
    which also looks like a CDN rate-limit stall in the logs.
    """
    url = str(candidate.get("url") or "")
    if not url:
        return None
    try:
        duration_value = float(duration)
        timeout_value = float(timeout)
    except (TypeError, ValueError):
        return None
    if duration_value <= 0 or timeout_value <= 0:
        return None
    headers = candidate.get("http_headers") or source.audio_headers or source.http_headers
    command = [
        str(FFMPEG_PATH), "-hide_banner", "-loglevel", "info", "-t", format(duration_value, ".12g"),
    ]
    if headers:
        command.extend(["-headers", _format_http_headers(headers)])
    command.extend(["-i", url, "-vn", "-f", "null", "-"])
    runner = run_func or run_tracked
    try:
        result = runner(command, timeout=timeout_value, text=True)
    except Exception:
        return None
    if isinstance(result, (int, float)):
        return float(result)
    if getattr(result, "returncode", 0) not in (0, None):
        return None
    output = "\n".join(str(value or "") for value in (
        getattr(result, "stderr", ""), getattr(result, "stdout", "")
    ))
    match = re.search(r"speed\s*=\s*([0-9]+(?:\.[0-9]+)?)x", output, re.IGNORECASE)
    return float(match.group(1)) if match else None


def select_audio_candidate(
    source: MediaSource,
    min_realtime_speed: float = 1.0,
    probe_duration: float = 3,
    run_func: Callable[..., Any] | None = None,
    log_func: Callable[[str], Any] | None = None,
) -> dict[str, Any] | None:
    """Keep the best audio unless a bounded probe shows a slower candidate is needed.

    If every candidate probe fails (timeout / 403), the source is left untouched
    (its resolve-time ``audio_url``) and ``None`` is returned instead of silently
    re-applying the first candidate — a probe that can't complete tells us nothing
    about which CDN is usable, and forcing a URL here can pin a stale/rate-limited
    one for the whole batch.
    """
    candidates = list(source.audio_candidates or [])
    if not candidates:
        return None
    try:
        threshold = float(min_realtime_speed)
    except (TypeError, ValueError):
        threshold = 1.0
    index = 0
    chose = False
    chosen_candidate = None
    while index < len(candidates):
        candidate = candidates[index]
        group = [candidate]
        if candidate.get("cdn_variant_index") is not None:
            index += 1
            while index < len(candidates) and candidates[index].get("format_id") == candidate.get("format_id"):
                group.append(candidates[index])
                index += 1
        else:
            index += 1

        measured = []
        for candidate in group:
            speed = probe_audio_candidate(source, candidate, duration=probe_duration, run_func=run_func)
            measured.append((speed, candidate))
            if log_func is not None:
                format_id = candidate.get("format_id") or "unknown"
                abr = candidate.get("abr") or candidate.get("tbr") or "unknown"
                speed_text = f"{speed:g}x" if speed is not None else "unknown"
                cdn = candidate.get("cdn_host")
                suffix = f" cdn={cdn}" if cdn else ""
                log_func(f"Audio candidate {format_id} abr={abr}{suffix} speed={speed_text}")

        usable = [(speed, candidate) for speed, candidate in measured if speed is not None]
        if not usable:
            continue
        fastest_speed, fastest_candidate = max(usable, key=lambda item: item[0])
        if fastest_speed >= threshold:
            apply_audio_candidate(source, fastest_candidate)
            chose = True
            chosen_candidate = fastest_candidate
            if log_func is not None and len(group) > 1:
                log_func(
                    f"Bilibili CDN selected: {fastest_candidate.get('cdn_host')} "
                    f"speed={fastest_speed:g}x"
                )
            break
        if log_func is not None:
            log_func(f"Audio candidate speed below {threshold:.1f}x; trying next candidate")
    if not chose:
        return None
    return chosen_candidate


def build_audio_cache_command(source: MediaSource, output_file: str | Path) -> list[str]:
    """Build an argv-only FFmpeg command that stores compressed remote audio."""
    if not source.audio_url:
        raise ValueError("MediaSource has no audio_url")

    command = [
        str(FFMPEG_PATH),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    headers = source.audio_headers or source.http_headers
    if headers:
        command.extend(["-headers", _format_http_headers(headers)])
    selected_codec = ""
    for candidate in source.audio_candidates or []:
        if str(candidate.get("url") or "") == source.audio_url:
            selected_codec = str(
                candidate.get("acodec") or candidate.get("codec") or ""
            ).lower()
            break
    output_suffix = Path(output_file).suffix.lower()
    copy_aac = output_suffix in {".m4a", ".mp4"} and (
        selected_codec.startswith("aac") or selected_codec.startswith("mp4a")
    )
    command.extend([
        "-i",
        source.audio_url,
        "-vn",
        "-c:a",
        "copy" if copy_aac else "aac",
    ])
    if not copy_aac:
        command.extend(["-b:a", "128k"])
    command.extend([
        "-movflags",
        "+faststart",
        str(output_file),
    ])
    return command


def build_hls_remux_command(input_file: str | Path, output_file: str | Path) -> list[str]:
    return [
        str(FFMPEG_PATH), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_file), "-vn", "-c:a", "copy", "-movflags", "+faststart",
        str(output_file),
    ]


def _audio_cache_timeout(source: MediaSource, timeout: float | None) -> float:
    if timeout is not None:
        return timeout
    try:
        duration = float(source.duration) if source.duration is not None else 0
    except (TypeError, ValueError):
        duration = 0
    return max(1800.0, duration * 3) if duration > 0 else 1800.0


def _audio_cache_format_identity(source: MediaSource) -> dict[str, Any]:
    """Return stable fields for the currently selected audio format."""
    for candidate in source.audio_candidates or []:
        if str(candidate.get("url") or "") != source.audio_url:
            continue
        identity = {
            key: candidate.get(key)
            for key in ("format_id", "abr", "tbr", "sample_rate", "channels")
            if candidate.get(key) is not None
        }
        codec = candidate.get("acodec") or candidate.get("codec")
        if codec is not None:
            identity["acodec"] = codec
            identity["codec"] = codec
        return identity
    return {}


def resolve_cached_audio(
    source: MediaSource,
    cache_store: Any,
    output_file: str | Path | None = None,
) -> Path | None:
    """Return the cached audio path if it already exists, without probing the network.

    Mirrors ``fetch_audio_cache``'s cache-hit lookup so callers can skip the
    per-candidate CDN speed probes when the audio is already cached (large Audio
    Cache batches would otherwise fire hundreds of probes every run and can trip
    bilibili's CDN rate limiting).
    """
    if cache_store is None or not source.audio_url:
        return None
    requested_format = Path(output_file).suffix.lstrip(".") if output_file else "m4a"
    audio_format = requested_format or "m4a"
    return cache_store.resolve_audio_cache(
        stable_source_id(source), source.audio_url, audio_format,
        format_identity=_audio_cache_format_identity(source),
    )



def _audio_cache_retry_reason(error: SegmentFetchError) -> str:
    detail = str(error)
    if "timed out after" in detail.lower():
        return "timeout"
    if "ffmpeg audio cache fetch failed (rc=" in detail.lower():
        return detail.split(":", 1)[0]
    if "without output" in detail.lower():
        return "no output"
    return "error"


def fetch_audio_cache(
    source: MediaSource,
    cache_store: Any,
    output_file: str | Path | None = None,
    run_func: Callable[..., Any] | None = None,
    timeout: float | None = None,
    retries: int = 2,
    logger: Callable[[str], Any] | None = None,
    log_func: Callable[[str], Any] | None = None,
    refresh_func: Callable[[MediaSource], MediaSource | None] | None = None,
    progress_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> Path:
    """Fetch one compressed audio stream, persist it, and reuse it thereafter."""
    if cache_store is None:
        raise ValueError("cache_store is required for audio caching")
    if retries < 0:
        raise ValueError("retries must not be negative")
    if not source.audio_url:
        raise SegmentFetchError(f"Could not cache audio for {source.source_url}: no audio_url")

    requested_format = Path(output_file).suffix.lstrip(".") if output_file else "m4a"
    audio_format = requested_format or "m4a"
    identity = stable_source_id(source)
    format_identity = _audio_cache_format_identity(source)
    log = log_func or logger
    cache_label = f"{source.platform or 'unknown'}:{source.source_id or 'unknown'}"

    def report(message: str) -> None:
        if log is not None:
            log(message)

    cached = cache_store.resolve_audio_cache(
        identity, source.audio_url, audio_format, format_identity=format_identity
    )
    if cached is not None:
        report(f"Audio cache hit for {cache_label}: {Path(cached).name}")
        return Path(cached)

    destination = Path(output_file) if output_file else cache_store.get_audio_cache_path(
        identity, source.audio_url, audio_format, format_identity=format_identity
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".remote-audio-", suffix="." + audio_format,
                                     dir=str(destination.parent))
    os.close(fd)
    Path(temporary).unlink(missing_ok=True)
    temporary_path = Path(temporary)
    transport_path = None
    runner = run_func or run_tracked
    effective_timeout = _audio_cache_timeout(source, timeout)
    try:
        duration_value = float(source.duration) if source.duration is not None else None
    except (TypeError, ValueError):
        duration_value = None
    duration_label = f"{duration_value:g}s" if duration_value is not None else "unknown"
    last_error: Exception | None = None
    platform_label = "YouTube" if source.platform == "youtube" else "Bilibili"

    if source.platform in {"youtube", "bilibili"} and supports_range_prefetch(source):
        try:
            report(f"{platform_label} audio cache prefetch for {cache_label}")
            speed_monitor = SpeedMonitor()

            def recover_slow_prefetch(stale_source, speed):
                report(
                    f"{platform_label} audio cache speed low ({speed:.2f} MB/s); "
                    "refreshing source before continuing"
                )
                if refresh_func is None:
                    return
                updated = refresh_func(stale_source)
                if isinstance(updated, MediaSource) and updated is not stale_source:
                    stale_source.__dict__.update(updated.__dict__)

            resume_limit = 5
            for resume_attempt in range(resume_limit):
                partial_size = temporary_path.stat().st_size if temporary_path.exists() else 0
                try:
                    with temporary_path.open("ab" if partial_size else "wb") as handle:
                        for chunk in iter_range_bytes(
                            source, chunk_size=4 * 1024 * 1024, concurrency=4, logger=log,
                            progress_callback=progress_callback,
                            speed_monitor=speed_monitor,
                            slow_callback=recover_slow_prefetch,
                            start_offset=partial_size,
                        ):
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                    break
                except Exception:
                    if resume_attempt >= resume_limit - 1:
                        raise
                    report(
                        f"{platform_label} audio cache interrupted at {partial_size} bytes; "
                        f"refreshing and resuming ({resume_attempt + 1}/{resume_limit - 1})"
                    )
                    if refresh_func is not None:
                        updated = refresh_func(source)
                        if isinstance(updated, MediaSource) and updated is not source:
                            source.__dict__.update(updated.__dict__)
            if temporary_path.stat().st_size == 0:
                raise SegmentFetchError(f"{platform_label} audio cache prefetch produced no output")
            registered = cache_store.save_audio_cache_file(
                identity, source.audio_url, audio_format, temporary_path,
                metadata=format_identity,
            )
            report(
                f"Audio cache completed for {cache_label}: "
                f"{registered.stat().st_size} bytes"
            )
            return Path(registered)
        except Exception as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            report(f"{platform_label} audio cache prefetch failed for {cache_label}; falling back to FFmpeg")

    if source.platform == "twitch" and supports_hls_prefetch(source):
        fd, transport = tempfile.mkstemp(prefix=".remote-hls-", suffix=".ts", dir=str(destination.parent))
        os.close(fd)
        transport_path = Path(transport)
        try:
            report(f"Twitch HLS audio cache prefetch for {cache_label}")
            with transport_path.open("wb") as handle:
                for chunk in iter_hls_bytes(
                    source, concurrency=4, logger=log, progress_callback=progress_callback
                ):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if not transport_path.is_file() or transport_path.stat().st_size == 0:
                raise SegmentFetchError("Twitch HLS audio cache prefetch produced no output")
            result = runner(build_hls_remux_command(transport_path, temporary_path),
                            timeout=effective_timeout, text=True)
            if getattr(result, "returncode", 0) != 0:
                detail = _sanitize_ffmpeg_detail(
                    getattr(result, "stderr", "") or getattr(result, "stdout", "") or "unknown error"
                )
                raise SegmentFetchError(f"FFmpeg HLS remux failed (rc={result.returncode}): {detail}")
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                raise SegmentFetchError("FFmpeg HLS remux completed without output")
            registered = cache_store.save_audio_cache_file(
                identity, source.audio_url, audio_format, temporary_path,
                metadata=format_identity,
            )
            report(f"Audio cache completed for {cache_label}: {registered.stat().st_size} bytes")
            return Path(registered)
        except Exception as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            report(f"Twitch HLS audio cache prefetch failed for {cache_label}; falling back to FFmpeg")
        finally:
            transport_path.unlink(missing_ok=True)

    refreshed = False
    attempt = 0
    allowed_attempts = int(retries) + 1
    while attempt < allowed_attempts:
        try:
            report(
                f"Audio cache starting download for {cache_label} "
                f"(attempt {attempt + 1}/{int(retries) + 1}; "
                f"duration={duration_label}; timeout={effective_timeout:g}s)"
            )
            command = build_audio_cache_command(source, temporary_path)
            stop_monitor = threading.Event()
            monitor_started = time.monotonic()

            def monitor_file():
                while not stop_monitor.wait(0.75):
                    if progress_callback is None:
                        continue
                    try:
                        current_size = temporary_path.stat().st_size
                    except FileNotFoundError:
                        current_size = 0
                    progress_callback(format_transfer_progress(
                        current_size, None, time.monotonic() - monitor_started))

            monitor = threading.Thread(target=monitor_file, name="audio-cache-progress", daemon=True)
            monitor.start()
            try:
                result = runner(command, timeout=effective_timeout, text=True)
            finally:
                stop_monitor.set()
                monitor.join(timeout=1)
            return_code = getattr(result, "returncode", 0)
            if return_code != 0:
                detail = _sanitize_ffmpeg_detail(
                    getattr(result, "stderr", "") or getattr(result, "stdout", "") or "unknown error"
                )
                raise SegmentFetchError(f"FFmpeg audio cache fetch failed (rc={return_code}): {detail}")
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                raise SegmentFetchError(
                    f"FFmpeg audio cache fetch completed without output: {temporary_path}"
                )
            registered = cache_store.save_audio_cache_file(
                identity, source.audio_url, audio_format, temporary_path,
                metadata=format_identity,
            )
            report(
                f"Audio cache completed for {cache_label}: "
                f"{registered.stat().st_size} bytes"
            )
            return Path(registered)
        except subprocess.TimeoutExpired:
            last_error = SegmentFetchError(
                f"Audio cache timed out after {effective_timeout:g} seconds"
            )
            if refresh_func is not None and not refreshed:
                refreshed = True
                try:
                    updated = refresh_func(source)
                    if isinstance(updated, MediaSource) and updated is not source:
                        source.__dict__.update(updated.__dict__)
                    report("Audio cache refreshed remote source; retrying")
                    allowed_attempts += 1
                    continue
                except Exception:
                    report("Audio cache source refresh failed; continuing with the existing retry policy")
            if attempt < int(retries):
                report(
                    f"Audio cache retry for {cache_label} "
                    f"({attempt + 1}/{int(retries)}): timeout"
                )
                time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            last_error = exc if isinstance(exc, SegmentFetchError) else SegmentFetchError(
                f"Could not cache remote audio for {source.source_url}: {exc}"
            )
            if refresh_func is not None and not refreshed:
                refreshed = True
                try:
                    updated = refresh_func(source)
                    if isinstance(updated, MediaSource) and updated is not source:
                        source.__dict__.update(updated.__dict__)
                    report("Audio cache refreshed remote source; retrying")
                    allowed_attempts += 1
                    continue
                except Exception:
                    report("Audio cache source refresh failed; continuing with the existing retry policy")
            if attempt < int(retries):
                report(
                    f"Audio cache retry for {cache_label} "
                    f"({attempt + 1}/{int(retries)}): "
                    f"{_audio_cache_retry_reason(last_error)}"
                )
                time.sleep(min(2 ** attempt, 8))
        finally:
            temporary_path.unlink(missing_ok=True)
        attempt += 1
    report(
        f"Audio cache failed for {cache_label} after "
        f"{int(retries) + 1} attempt(s): "
        f"{_audio_cache_retry_reason(last_error) if last_error else 'error'}"
    )
    raise last_error or SegmentFetchError(
        f"Could not cache remote audio for {source.source_url}"
    )


def _format_http_headers(headers: Mapping[str, str]) -> str:
    values = []
    for key, value in headers.items():
        key = str(key)
        value = str(value)
        if any(char in key or char in value for char in ("\r", "\n")):
            raise ValueError("HTTP headers must not contain newline characters")
        values.append(f"{key}: {value}")
    return "\r\n".join(values) + ("\r\n" if values else "")


def _segment_number(value: float, name: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if number < 0:
        raise ValueError(f"{name} must not be negative")
    return format(number, ".12g")


def build_segment_command(
    source: MediaSource,
    start: float,
    end: float,
    output_file: str | Path,
    audio_only: bool = False,
    codec: str | None = None,
) -> list[str]:
    """Build an argv-only FFmpeg command for one remote time interval."""
    start_value = float(start)
    end_value = float(end)
    if start_value < 0 or end_value <= start_value:
        raise ValueError("segment end must be greater than a non-negative start")

    if audio_only and not source.audio_url:
        raise ValueError("MediaSource has no audio_url")
    if not audio_only and not source.video_url:
        stream_name = "audio_url" if audio_only else "video_url"
        raise ValueError(f"MediaSource has no {stream_name}")

    duration = end_value - start_value
    seek_start = max(0.0, start_value - _REMOTE_SEEK_PAD)
    trim_offset = start_value - seek_start
    input_duration = trim_offset + duration
    seek_text = _segment_number(seek_start, "seek start")
    input_duration_text = _segment_number(input_duration, "input duration")
    trim_start_text = _segment_number(trim_offset, "trim offset")
    trim_end_text = _segment_number(trim_offset + duration, "trim end")
    duration_text = _segment_number(duration, "duration")

    command = [
        str(FFMPEG_PATH),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        seek_text,
        "-t",
        input_duration_text,
    ]
    input_headers = (source.audio_headers or source.http_headers) if audio_only else (
        source.video_headers or source.http_headers
    )
    if input_headers:
        command.extend(["-headers", _format_http_headers(input_headers)])
    embedded_audio = not audio_only and source_has_embedded_audio(source)
    stream_url = str(source.audio_url if audio_only else source.video_url)
    if _segment_is_http(stream_url):
        # 签名 URL 过期/CDN 半连接时 FFmpeg 会永久挂起而不退出：
        # 给远程网络输入加读写超时，让 FFmpeg 自行中止并暴露错误给上层 refresh 重试。
        command.extend(["-rw_timeout", "60000000"])
    command.extend(["-i", stream_url])
    if not audio_only and source.audio_url and not embedded_audio:
        audio_headers = source.audio_headers or source.http_headers
        if audio_headers:
            command.extend(["-headers", _format_http_headers(audio_headers)])
        command.extend([
            "-ss", seek_text,
            "-t", input_duration_text,
            "-i", str(source.audio_url),
        ])
    if audio_only:
        command.extend([
            "-af",
            f"asetpts=PTS-STARTPTS,atrim=start={trim_start_text}:end={trim_end_text},"
            "asetpts=PTS-STARTPTS",
        ])
    else:
        command.extend([
            "-vf",
            f"setpts=PTS-STARTPTS,trim=start={trim_start_text}:end={trim_end_text},"
            "setpts=PTS-STARTPTS",
        ])
        if source.audio_url or embedded_audio:
            command.extend([
                "-af",
                f"asetpts=PTS-STARTPTS,atrim=start={trim_start_text}:end={trim_end_text},"
                "asetpts=PTS-STARTPTS",
            ])
    command.extend(["-t", duration_text])
    if audio_only:
        command.append("-vn")
        if codec:
            command.extend(["-c:a", str(codec)])
    else:
        command.extend(["-map", "0:v:0"])
        if source.audio_url or embedded_audio:
            command.extend([
                "-map", "0:a:0" if embedded_audio else "1:a:0", "-c:a", "aac", "-shortest"
            ])
        else:
            command.extend(["-map", "0:a:0?"])
        if codec:
            command.extend(["-c:v", str(codec)])
    command.extend([
        "-avoid_negative_ts", "make_zero",
        "-reset_timestamps", "1",
    ])
    if Path(output_file).suffix.lower() in {".mp4", ".m4a", ".mov"}:
        # moov 前置（faststart）：materialized clip 之后的每次 ffmpeg -i
        # 校验/探测/seek 都只需读文件头，避免大段 clip（moov 在尾部）被
        # 整段读取，造成 prepare→compile 收尾时数秒卡顿。
        command.extend(["-movflags", "+faststart"])
    command.append(str(output_file))
    return command


def _segment_has_stream(path, want_video=True):
    """Return True if the segment file contains a readable video/audio stream.

    ffmpeg exits 0 even for some empty/truncated outputs, so the raw exit code
    and the file-size check alone are not enough to trust a downloaded segment.
    """
    try:
        out = run_tracked([FFMPEG_PATH, "-hide_banner", "-i", str(path)],
                          timeout=10, text=True)
    except Exception:
        return False
    stderr = getattr(out, "stderr", None)
    if stderr is None:
        return True
    if not isinstance(stderr, str):
        stderr = str(stderr)
    kind = r"Video:" if want_video else r"Audio:"
    return bool(re.search(r"Stream #\d+:\d+.*" + kind, stderr))


def _segment_duration(path) -> float | None:
    """Return the container duration of a downloaded segment in seconds.

    Uses the same bounded ffmpeg probe as `_segment_has_stream` so a corrupt or
    truncated download (which ffmpeg may still exit 0 for) can be detected
    before it enters the compile stage.
    """
    try:
        out = run_tracked(
            [FFMPEG_PATH, "-hide_banner", "-i", str(path),
             "-probesize", "32M", "-analyzeduration", "100M"],
            timeout=10, text=True)
        stderr = getattr(out, "stderr", None)
    except Exception:
        return None
    if not isinstance(stderr, str):
        return None
    m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", stderr)
    if not m:
        return None
    h, mi, s, ms = map(int, m.groups())
    return h * 3600 + mi * 60 + s + ms / 100.0


def _refresh_with_backoff(
    refresh_func: Callable[[MediaSource], MediaSource | None],
    source: MediaSource,
    logger: Callable[[str], Any] | None = None,
    attempts: int = 3,
) -> MediaSource | None:
    """Retry a refresh with rate-limit backoff so one rate-limited resolve does not
    cascade into hard-retrying an expired URL. Re-raises the last failure."""
    from remote_rate import LimitedRefresher, ResolveLimiter
    if isinstance(refresh_func, LimitedRefresher):
        return refresh_func(source)
    limited = LimitedRefresher(
        refresh_func, limiter=ResolveLimiter(), retries=attempts, logger=logger
    )
    return limited(source)


def fetch_segment(
    source: MediaSource,
    start: float,
    end: float,
    output_file: str | Path,
    cache_store: Any | None = None,
    padding_before: float = 0,
    padding_after: float = 0,
    timeout: float = 600,
    retries: int = 2,
    run_func: Callable[..., Any] | None = None,
    audio_only: bool = False,
    codec: str | None = None,
    allow_covering_cache: bool = True,
    refresh_func: Callable[[MediaSource], MediaSource | None] | None = None,
    logger: Callable[[str], Any] | None = None,
    progress_callback: Callable[..., Any] | None = None,
    stall_timeout: float = 30,
) -> Path:
    """Fetch a requested remote interval, optionally reusing covering cache."""
    if padding_before < 0 or padding_after < 0:
        raise ValueError("segment padding must not be negative")
    if retries < 0:
        raise ValueError("retries must not be negative")

    identity = stable_source_id(source)
    padding = float(padding_before) + float(padding_after)
    extension = Path(output_file).suffix.lstrip(".") or ("m4a" if audio_only else "mp4")
    media_type = "audio" if audio_only else "video"
    if cache_store is not None:
        if allow_covering_cache:
            cached = cache_store.find_cached_segment(
                identity, start, end, padding, extension=extension, media_type=media_type
            )
            if cached is not None:
                return Path(cached)

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".remote-segment-", suffix="." + extension,
                                     dir=str(destination.parent))
    os.close(fd)
    Path(temporary).unlink(missing_ok=True)
    temporary_path = Path(temporary)
    runner = run_func or run_tracked

    def run_command(command):
        if progress_callback is None:
            return runner(command, timeout=timeout, text=True)
        expected_duration = max(0.0, (float(end) + float(padding_after)) -
                                max(0.0, float(start) - float(padding_before)))

        def report(current, total, elapsed):
            progress_callback(current, total or expected_duration, elapsed)

        return run_tracked_progress(
            command, duration=expected_duration, timeout=timeout,
            progress_callback=report, stall_timeout=stall_timeout,
        )
    last_error: Exception | None = None
    refreshed = False
    attempt = 0
    allowed_attempts = int(retries) + 1
    while attempt < allowed_attempts:
        try:
            command = build_segment_command(
                source,
                max(0.0, float(start) - float(padding_before)),
                float(end) + float(padding_after),
                temporary_path,
                audio_only=audio_only,
                codec=codec,
            )
            result = run_command(command)
            return_code = getattr(result, "returncode", 0)
            if return_code != 0:
                raw_detail = (getattr(result, "stderr", "")
                              or getattr(result, "stdout", "") or "unknown error")
                has_403 = "403" in str(raw_detail)
                detail = _sanitize_ffmpeg_detail(raw_detail)
                if has_403:
                    detail = (
                        f"{detail} (The remote source URL likely expired or is "
                        f"rate-limited; AutoComper will try refreshing the source.)"
                    )
                raise SegmentFetchError(f"FFmpeg segment fetch failed (rc={return_code}): {detail}")
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                raise SegmentFetchError(f"FFmpeg segment fetch completed without output: {temporary_path}")
            if run_func is None and not _segment_has_stream(
                    temporary_path, want_video=not audio_only):
                raise SegmentFetchError(
                    f"FFmpeg segment fetch produced an unreadable segment "
                    f"(no {'video' if not audio_only else 'audio'} stream): {temporary_path}")
            if run_func is None:
                # 时长校验：下载部分成功但内容不完整/损坏时（URL 过期、CDN
                # 半连接），ffmpeg 可能退出 0 且流存在，但返回的 segment 远短于
                # 请求区间。明显偏短的 segment 直接当作损坏走 refresh/retry，
                # 阻止损坏数据进入 compile（超长 reverify 合并 clip 最容易踩中）。
                expected_duration = (
                    float(end) + float(padding_after)
                ) - max(0.0, float(start) - float(padding_before))
                actual_duration = _segment_duration(temporary_path)
                if (expected_duration > 0 and actual_duration is not None
                        and actual_duration > 0
                        and actual_duration < expected_duration * 0.5):
                    raise SegmentFetchError(
                        f"FFmpeg segment fetch produced a truncated segment "
                        f"({actual_duration:g}s vs expected ~{expected_duration:g}s): "
                        f"{temporary_path}")
            data = temporary_path.read_bytes()
            if cache_store is not None:
                fetched_start = max(0.0, float(start) - float(padding_before))
                fetched_end = float(end) + float(padding_after)
                cache_store.save_segment_cache(
                    identity, fetched_start, fetched_end, 0, data,
                    extension=extension, media_type=media_type)
            cache_store_class = cache_store if cache_store is not None else type("Atomic", (), {})
            saver = getattr(cache_store_class, "save_file", None)
            if saver is None:
                from remote_cache import CacheStore
                saver = CacheStore.save_file
            saver(destination, data)
            return destination
        except Exception as exc:
            last_error = exc if isinstance(exc, SegmentFetchError) else SegmentFetchError(
                f"Could not fetch remote segment {start}-{end}: "
                f"{_sanitize_ffmpeg_detail(exc)}"
            )
            if refresh_func is not None and not refreshed:
                refreshed = True
                try:
                    updated = _refresh_with_backoff(refresh_func, source, logger=logger)
                    if isinstance(updated, MediaSource) and updated is not source:
                        source.__dict__.update(updated.__dict__)
                    if logger is not None:
                        logger("Remote segment source refreshed; retrying")
                    allowed_attempts += 1
                    continue
                except Exception:
                    if logger is not None:
                        logger("Remote segment source refresh failed; continuing with the existing retry policy")
            if attempt < int(retries):
                time.sleep(min(2 ** attempt, 8))
        finally:
            temporary_path.unlink(missing_ok=True)
        attempt += 1

    raise last_error or SegmentFetchError(f"Could not fetch remote segment {start}-{end}")


def stable_source_id(source: MediaSource) -> str:
    """Return an identity based on platform and VOD identity, not stream URLs."""
    platform = (source.platform or "unknown").strip().lower()
    source_id = (source.source_id or "").strip()
    if source_id:
        return f"{platform}:{source_id}"

    parts = urlsplit(source.source_url.strip())
    canonical_url = urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, "", "")
    )
    return f"{platform}:{canonical_url}"


def source_has_embedded_audio(source: MediaSource) -> bool:
    """Return whether the selected video stream contains an audio codec."""
    acodec = source.metadata.get("acodec")
    return bool(acodec) and str(acodec).strip().lower() != "none"


def parse_url_list(text: str) -> list[str]:
    """Parse a URL list while retaining first-seen order."""
    result = []
    seen = set()
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("#") or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_browser_cookies(browser_cookies: str | None) -> str | None:
    if browser_cookies is None:
        return None
    value = str(browser_cookies).strip()
    if value.casefold() == "none":
        return None
    if value.startswith(_COOKIES_FILE_PREFIX):
        cookies_path = value[len(_COOKIES_FILE_PREFIX):].strip()
        if not cookies_path:
            raise ValueError("cookies file path is empty")
        return f"{_COOKIES_FILE_PREFIX}{cookies_path}"
    lowered = value.casefold()
    if lowered not in (*_BROWSER_COOKIE_NAMES, "auto"):
        raise ValueError(
            "browser_cookies must be None, firefox, chrome, edge, auto, or a cookies file"
        )
    return lowered


def _ydl_options(
    browser_cookies: str | None = None,
    extract_flat: bool = False,
) -> dict[str, Any]:
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 30,
    }
    if extract_flat:
        options["extract_flat"] = True
    normalized = _normalize_browser_cookies(browser_cookies)
    if normalized is None:
        return options
    if normalized.startswith(_COOKIES_FILE_PREFIX):
        options["cookies"] = normalized[len(_COOKIES_FILE_PREFIX):]
    elif normalized in _BROWSER_COOKIE_NAMES:
        options["cookiesfrombrowser"] = (normalized,)
    return options


def _make_ydl(
    ydl_factory: Callable[..., Any] | None,
    browser_cookies: str | None = None,
    extract_flat: bool = False,
):
    if ydl_factory is not None:
        return ydl_factory(_ydl_options(browser_cookies, extract_flat=extract_flat))
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise SourceResolveError("yt-dlp is required to resolve remote media") from exc
    return YoutubeDL(_ydl_options(browser_cookies, extract_flat=extract_flat))


def _extract_info(
    url: str,
    ydl_factory: Callable[..., Any] | None,
    browser_cookies: str | None = None,
    extract_flat: bool = False,
    resolve_timeout: float = 90,
) -> Mapping[str, Any]:
    """Resolve one URL with yt-dlp, enforcing a wall-clock timeout.

    yt-dlp's ``socket_timeout`` covers individual socket reads, but a resolve
    can still stall across many retries or in platform-specific handlers. A
    watchdog thread aborts after ``resolve_timeout`` so an unstable network can
    never hang the whole run forever.
    """

    def _do_extract() -> Mapping[str, Any]:
        ydl = _make_ydl(
            ydl_factory,
            browser_cookies=browser_cookies,
            extract_flat=extract_flat,
        )
        if hasattr(ydl, "__enter__"):
            with ydl as active_ydl:
                return active_ydl.extract_info(url, download=False)
        return ydl.extract_info(url, download=False)

    import queue as _queue
    import threading as _threading

    result_queue = _queue.Queue(maxsize=1)

    def worker():
        try:
            result_queue.put(_do_extract())
        except BaseException as exc:  # noqa: BLE001 - surface any resolve error
            result_queue.put(exc)

    thread = _threading.Thread(target=worker, name="ytdlp-resolve", daemon=True)
    thread.start()
    try:
        item = result_queue.get(timeout=resolve_timeout)
    except _queue.Empty:
        raise SourceResolveError(
            f"Could not resolve source {url}: timed out after "
            f"{resolve_timeout:g}s (network stalled or rate-limited)"
        )
    if isinstance(item, BaseException):
        if isinstance(item, SourceResolveError):
            raise item
        raise SourceResolveError(_readable_resolve_error(url, item)) from item
    info = item
    if not isinstance(info, Mapping):
        raise SourceResolveError(f"Could not resolve source {url}: invalid metadata")
    return info


def _readable_resolve_error(url: str, exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.lower()
    if any(marker in lowered for marker in ("dpapi", "decrypt", "browser", "not installed", "not logged")):
        return (
            f"Could not read browser cookies for {url}: {detail}. "
            "On Windows, only Firefox is reliable. Chrome 127+ encrypts its cookies "
            "(App-Bound) and Edge locks its database while the browser is open. "
            "Close Edge, switch to Firefox, or import a cookies.txt file "
            "('Get cookies.txt LOCALLY' browser extension)."
        )
    return f"Could not resolve source {url}: {detail}"


def _is_bilibili_url(url: str) -> bool:
    hostname = (urlsplit(url).hostname or "").lower()
    return any(hostname == marker or hostname.endswith("." + marker) for marker in _BILIBILI_HOST_MARKERS)


def _is_http_412_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return "412" in detail or "precondition failed" in detail


def _should_auto_use_cookies(url: str, exc: Exception) -> bool:
    return _is_bilibili_url(url) or _is_http_412_error(exc)


def _short_cookie_failure(browser: str, exc: Exception) -> str:
    detail = str(exc).strip().replace("\n", " ") or exc.__class__.__name__
    if len(detail) > 120:
        detail = detail[:117] + "..."
    return f"{browser}: {detail}"


_cookie_failure_last_printed: dict[str, float] = {}


def _cookie_failure_should_print(browser: str, url: str) -> bool:
    """Throttle per-browser cookie failure logs to avoid refresh-storm spam."""
    key = f"{browser}\x00{url}"
    now = time.monotonic()
    last = _cookie_failure_last_printed.get(key)
    if last is not None and now - last < 30:
        return False
    _cookie_failure_last_printed[key] = now
    return True


def _extract_with_cookie_policy(
    url: str,
    ydl_factory: Callable[..., Any] | None,
    browser_cookies: str | None,
    extract_flat: bool = False,
) -> Mapping[str, Any]:
    normalized = _normalize_browser_cookies(browser_cookies)
    if normalized != "auto":
        return _extract_info(url, ydl_factory, normalized, extract_flat=extract_flat)

    try:
        return _extract_info(url, ydl_factory, extract_flat=extract_flat)
    except SourceResolveError as initial_error:
        if not _should_auto_use_cookies(url, initial_error):
            raise
        failures = []
        for browser in _BROWSER_COOKIE_NAMES:
            try:
                return _extract_info(url, ydl_factory, browser, extract_flat=extract_flat)
            except SourceResolveError as exc:
                failures.append(_short_cookie_failure(browser, exc))
                if _cookie_failure_should_print(browser, url):
                    print(f"Remote browser cookies failed ({url}): {failures[-1]}")
        detail = "; ".join(failures)
        raise SourceResolveError(
            f"Could not resolve source {url}: {detail}. "
            "请登录 Bilibili 浏览器或选择 cookies"
        ) from initial_error


def preflight_cookie_source(browser_cookies: str | None) -> str | None:
    """Return a human-readable failure reason for an unusable cookie source, or None.

    Only checks sources that can be validated locally without a network request:
    a cookies.txt file must exist and be readable; no full resolve is performed.
    ``None``/``auto``/browser names are considered usable here because they fall
    back to direct access at resolve time.
    """
    normalized = _normalize_browser_cookies(browser_cookies)
    if normalized is None:
        return None
    if normalized == "auto":
        return None
    if normalized.startswith(_COOKIES_FILE_PREFIX):
        cookies_path = normalized[len(_COOKIES_FILE_PREFIX):]
        if not os.path.isfile(cookies_path):
            return (
                f"Cookies file not found: {cookies_path}. "
                "Re-choose the file in Remote Settings (Remote Browser Cookies → Cookies File…)."
            )
        try:
            with open(cookies_path, "rb"):
                pass
        except OSError as exc:
            return f"Cookies file is not readable: {cookies_path} ({exc})"
        return None
    return None


def _stream_url(info: Mapping[str, Any], audio: bool) -> str:
    formats = info.get("requested_formats") or []
    for fmt in formats:
        if not isinstance(fmt, Mapping) or not fmt.get("url"):
            continue
        has_audio = fmt.get("acodec") not in (None, "none")
        has_video = fmt.get("vcodec") not in (None, "none")
        if (audio and has_audio and not has_video) or (not audio and has_video):
            return str(fmt["url"])

    direct_url = info.get("url")
    if direct_url and (audio or info.get("vcodec") not in (None, "none")):
        return str(direct_url)
    return ""


def _bilibili_cdn_variants(url: str) -> list[str]:
    """Return the original signed URL plus known same-resource CDN hosts."""
    parsed = urlsplit(str(url or ""))
    if not parsed.scheme or not parsed.netloc:
        return [str(url)] if url else []
    variants = []
    for host in (parsed.netloc,) + _BILIBILI_CDN_HOSTS:
        candidate = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
        if candidate not in variants:
            variants.append(candidate)
    return variants


def _expand_bilibili_audio_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for candidate in candidates:
        for index, url in enumerate(_bilibili_cdn_variants(candidate.get("url", ""))):
            item = dict(candidate)
            item["url"] = url
            item["cdn_variant_index"] = index
            item["cdn_host"] = urlsplit(url).netloc
            expanded.append(item)
    return expanded


def _candidate_sort_key(candidate: Mapping[str, Any], audio: bool) -> tuple[Any, ...]:
    def number(name: str) -> float:
        try:
            value = float(candidate.get(name))
            return value if value >= 0 else 0
        except (TypeError, ValueError):
            return 0

    if audio:
        abr = number("abr")
        tbr = number("tbr")
        return (
            max(abr, tbr),
            1 if candidate.get("_requested") else 0,
            abr + tbr,
            str(candidate.get("format_id") or ""),
        )
    return (
        1 if candidate.get("_requested") else 0,
        number("height"),
        number("tbr"),
        str(candidate.get("format_id") or ""),
    )


def _stream_candidates(info: Mapping[str, Any], audio: bool) -> list[dict[str, Any]]:
    requested_formats = info.get("requested_formats") or []
    all_formats = info.get("formats") or []
    if not isinstance(requested_formats, (list, tuple)):
        requested_formats = []
    if not isinstance(all_formats, (list, tuple)):
        all_formats = []
    base_headers = info.get("http_headers") or {}
    if not isinstance(base_headers, Mapping):
        base_headers = {}
    candidates = []
    seen_format_ids = set()
    seen_urls = set()
    for fmt, requested in [
        *((item, True) for item in requested_formats),
        *((item, False) for item in all_formats),
    ]:
        if not isinstance(fmt, Mapping) or not fmt.get("url"):
            continue
        format_id = fmt.get("format_id")
        format_id_key = str(format_id) if format_id is not None else ""
        url = str(fmt["url"])
        if (format_id_key and format_id_key in seen_format_ids) or url in seen_urls:
            continue
        if format_id_key:
            seen_format_ids.add(format_id_key)
        seen_urls.add(url)
        has_audio = fmt.get("acodec") not in (None, "none")
        has_video = fmt.get("vcodec") not in (None, "none")
        if not ((audio and has_audio and not has_video) or (not audio and has_video)):
            continue
        headers = dict(base_headers)
        format_headers = fmt.get("http_headers") or {}
        if isinstance(format_headers, Mapping):
            headers.update(format_headers)
        candidates.append({
            "url": url,
            "acodec": fmt.get("acodec"),
            "abr": fmt.get("abr"),
            "tbr": fmt.get("tbr"),
            "height": fmt.get("height"),
            "http_headers": {str(key): str(value) for key, value in headers.items()},
            "format_id": format_id,
            "filesize": fmt.get("filesize"),
            "filesize_approx": fmt.get("filesize_approx"),
            "clen": fmt.get("clen"),
            "source_format": "requested_formats" if requested else "formats",
            "_requested": requested,
        })
    candidates.sort(key=lambda item: _candidate_sort_key(item, audio), reverse=True)
    for candidate in candidates:
        candidate.pop("_requested", None)
    return candidates


def _selected_format_headers(info: Mapping[str, Any]) -> dict[str, str]:
    """Merge selected stream headers over the extractor-level fallbacks."""
    headers = info.get("http_headers") or {}
    merged = dict(headers) if isinstance(headers, Mapping) else {}
    formats = info.get("requested_formats") or info.get("formats") or []
    for fmt in formats:
        if not isinstance(fmt, Mapping) or not fmt.get("url"):
            continue
        has_audio = fmt.get("acodec") not in (None, "none")
        has_video = fmt.get("vcodec") not in (None, "none")
        if not (has_audio or has_video):
            continue
        format_headers = fmt.get("http_headers") or {}
        if isinstance(format_headers, Mapping):
            merged.update(format_headers)
    return {str(key): str(value) for key, value in merged.items()}


def _metadata_without_cookie_values(info: Mapping[str, Any]) -> dict[str, Any]:
    """Keep cookies out of general metadata while retaining stream headers separately."""
    return {
        key: value
        for key, value in info.items()
        if str(key).lower() not in {
            "cookie", "cookies", "cookie_header", "http_headers",
            "formats", "requested_formats",
        }
    }


def _entry_url(entry: Mapping[str, Any]) -> str:
    for key in ("webpage_url", "original_url", "url"):
        value = entry.get(key)
        if value:
            return str(value)
    return ""


def _normalize_date(value: Any) -> str:
    if value is None or value == "":
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        if len(text) == 8:
            return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
        try:
            stamp = float(text)
            if stamp > 100000000000:
                stamp /= 1000
            return datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return ""
    candidate = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).date().isoformat()
    except ValueError:
        match = re.match(r"^(\d{4})[-/]?(\d{2})[-/]?(\d{2})", text)
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else ""


def _normalized_duration(entry: Mapping[str, Any]) -> float | None:
    value = entry.get("duration")
    if value is None:
        value = entry.get("duration_ms")
        if value is not None:
            try:
                return float(value) / 1000
            except (TypeError, ValueError):
                return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalized_upload_date(entry: Mapping[str, Any]) -> str:
    for key in ("upload_date", "release_date", "timestamp", "release_timestamp", "created_at", "pubdate"):
        date = _normalize_date(entry.get(key))
        if date:
            return date
    return ""


def _bilibili_title_date(title: Any) -> str:
    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", str(title or ""))
    if not match:
        return ""
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _bilibili_bvid(entry: PlaylistEntry) -> str:
    match = re.search(r"\b(BV[0-9A-Za-z]+)\b", f"{entry.entry_id} {entry.webpage_url}")
    return match.group(1) if match else str(entry.entry_id)


def _default_bilibili_view_request(bvid: str) -> Mapping[str, Any]:
    endpoint = "https://api.bilibili.com/x/web-interface/view?bvid=" + quote(str(bvid), safe="")
    request = Request(endpoint, headers={"User-Agent": "AutoComper/1.0"})
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("code", 0) != 0:
        raise RuntimeError("Bilibili view API returned an error")
    return payload


def _entry_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in entry.items()
        if str(key).lower() not in {
            "formats", "requested_formats", "url", "manifest_url", "audio_url", "video_url",
        }
    }


def _playlist_platform(url: str, info: Mapping[str, Any]) -> str:
    forced = _forced_playlist_platform(url)
    if forced:
        return forced
    platform = str(info.get("extractor_key") or info.get("extractor") or "unknown").lower()
    if platform in {"youtubetab", "youtubeplaylist"}:
        return "youtube"
    return platform


def _forced_playlist_platform(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/")
    query = parse_qs(parsed.query)
    if _is_bilibili_url(url):
        if re.fullmatch(r"/[^/]+/lists/[^/]+", path) and query.get("type") == ["series"]:
            return "bilibili-series"
        if re.fullmatch(r"/list/[^/]+", path) and "sid" in query:
            return "bilibili-collection"
        if re.fullmatch(r"/[^/]+/upload/video", path):
            return "bilibili-uploads"
        if re.fullmatch(r"/[^/]+", path) and (parsed.hostname or "").lower().startswith("space."):
            return "bilibili-uploads"
    hostname = (parsed.hostname or "").lower()
    if hostname == "twitch.tv" or hostname.endswith(".twitch.tv"):
        if re.fullmatch(r"/[^/]+/videos", path):
            return "twitch-vods"
    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if path == "/playlist" and "list" in query:
            return "youtube"
        if re.fullmatch(r"/@[^/]+/(?:videos|streams|shorts)", path):
            return "youtube-uploads"
        if re.fullmatch(r"/(?:channel|user|c)/[^/]+/(?:videos|streams|shorts)", path):
            return "youtube-uploads"
    return ""


def _flat_entry(platform: str, entry: Mapping[str, Any], index: int) -> PlaylistEntry | None:
    webpage_url = _entry_url(entry)
    if not webpage_url:
        return None
    entry_id = str(entry.get("id") or webpage_url)
    duration = _normalized_duration(entry)
    metadata = _entry_metadata(entry)
    return PlaylistEntry(
        platform=platform,
        entry_id=entry_id,
        title=str(entry.get("title") or "Unknown"),
        webpage_url=webpage_url,
        duration=duration,
        upload_date=_normalized_upload_date(entry),
        index=index,
        metadata=metadata,
    )


def _merge_entry_metadata(entry: PlaylistEntry, hydrated: Mapping[str, Any]) -> None:
    if entry.title == "Unknown" and hydrated.get("title"):
        entry.title = str(hydrated["title"])
    if entry.duration is None:
        entry.duration = _normalized_duration(hydrated)
    if not entry.upload_date:
        entry.upload_date = _normalized_upload_date(hydrated)
    if not entry.upload_date:
        entry.upload_date = _bilibili_title_date(entry.title)
    entry.metadata.update(_entry_metadata(hydrated))


def _descriptor_from_info(
    url: str,
    info: Mapping[str, Any],
    ydl_factory: Callable[..., Any] | None = None,
    browser_cookies: str | None = None,
    bilibili_view_request: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> PlaylistDescriptor:
    platform = _playlist_platform(url, info)
    entries = []
    for index, raw_entry in enumerate(info.get("entries") or []):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = _flat_entry(platform, raw_entry, index)
        if entry is not None:
            entries.append(entry)
        if len(entries) >= 1000:
            break
    if platform.casefold().startswith("bilibili") and len(entries) > 1:
        part_count = len(entries)
        for part_number, entry in enumerate(entries, 1):
            entry.metadata["part_number"] = part_number
            entry.metadata["part_count"] = part_count
    hydrate_entry = None
    if platform.startswith("bilibili"):
        def hydrate_entry(entry: PlaylistEntry) -> Mapping[str, Any] | None:
            if entry.title != "Unknown" and entry.duration is not None and entry.upload_date:
                return None
            if bilibili_view_request is not None:
                try:
                    response = bilibili_view_request(_bilibili_bvid(entry))
                    data = response.get("data", response) if isinstance(response, Mapping) else None
                    if isinstance(data, Mapping):
                        api_metadata = {
                            "id": entry.entry_id,
                            "title": data.get("title"),
                            "duration": data.get("duration"),
                            "pubdate": data.get("pubdate"),
                            "webpage_url": entry.webpage_url,
                        }
                        if any(value not in (None, "") for key, value in api_metadata.items()
                               if key not in {"id", "webpage_url"}):
                            return api_metadata
                except Exception:
                    pass
            return _extract_with_cookie_policy(
                entry.webpage_url, ydl_factory, browser_cookies, extract_flat=False
            )
    elif platform.startswith("youtube") or platform in {"twitch", "twitch-vods"}:
        def hydrate_entry(entry: PlaylistEntry) -> Mapping[str, Any] | None:
            if entry.title != "Unknown" and entry.duration is not None and entry.upload_date:
                return None
            return _extract_with_cookie_policy(
                entry.webpage_url, ydl_factory, browser_cookies, extract_flat=False
            )

    return PlaylistDescriptor(
        platform=platform,
        source_url=url,
        title=str(info.get("title") or url),
        total_count=len(entries),
        _entries=entries,
        _hydrate_entry=hydrate_entry,
    )


def _single_entry_from_info(url: str, info: Mapping[str, Any]) -> PlaylistEntry:
    platform = str(info.get("extractor_key") or info.get("extractor") or "unknown").lower()
    entry = _flat_entry(platform, info, 0)
    if entry is None:
        entry = PlaylistEntry(platform, str(info.get("id") or url), str(info.get("title") or url), url)
    return entry


def _text_descriptor(urls: list[str], source_url: str) -> PlaylistDescriptor:
    entries = [
        PlaylistEntry("text", url, url, url, index=index, metadata={"source": "text-list"})
        for index, url in enumerate(urls[:1000])
    ]
    return PlaylistDescriptor("text", source_url, "URL list", len(entries), _entries=entries)


def describe_input(
    value: str,
    ydl_factory: Callable[..., Any] | None = None,
    browser_cookies: str | None = None,
    bilibili_view_request: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> PlaylistEntry | PlaylistDescriptor:
    """Describe one VOD or playlist without resolving stream URLs."""
    input_value = normalize_youtube_playlist_url(value.strip())
    if not input_value:
        raise SourceExpansionError("Input is empty")
    if not _is_url(input_value):
        path = Path(input_value)
        if not path.is_file():
            raise SourceExpansionError(f"Input does not exist: {input_value}")
        return _text_descriptor(parse_url_list(path.read_text(encoding="utf-8")), input_value)
    try:
        info = _extract_with_cookie_policy(
            input_value, ydl_factory, browser_cookies, extract_flat=True
        )
    except ValueError as exc:
        raise SourceExpansionError(str(exc)) from exc
    if bilibili_view_request is None and ydl_factory is None and _is_bilibili_url(input_value):
        bilibili_view_request = _default_bilibili_view_request
    if info.get("_type") in ("playlist", "multi_video") or _forced_playlist_platform(input_value):
        return _descriptor_from_info(
            input_value,
            info,
            ydl_factory=ydl_factory,
            browser_cookies=browser_cookies,
            bilibili_view_request=bilibili_view_request,
        )
    return _single_entry_from_info(input_value, info)


def normalize_youtube_playlist_url(url: str) -> str:
    """Convert a YouTube watch URL carrying a playlist into its playlist URL."""
    parsed = urlsplit(url)
    if parsed.netloc.lower() not in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return url
    query = parse_qs(parsed.query)
    playlist_id = query.get("list", [""])[0]
    if not playlist_id or parsed.path != "/watch":
        return url
    return urlunsplit((parsed.scheme, parsed.netloc, "/playlist", f"list={playlist_id}", ""))


def _source_from_info(url: str, info: Mapping[str, Any], max_height: int | None = None) -> MediaSource:
    if info.get("_type") in ("playlist", "multi_video"):
        raise SourceResolveError(f"Source is a playlist, not a single VOD: {url}")

    platform = str(info.get("extractor_key") or info.get("extractor") or "unknown").lower()
    source_url = str(info.get("webpage_url") or info.get("original_url") or url)
    source_id = str(info.get("id") or "")
    duration = info.get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    headers = _selected_format_headers(info)
    metadata = _metadata_without_cookie_values(info)
    audio_candidates = _stream_candidates(info, audio=True)
    if platform in {"bilibili", "bilibiliweb"}:
        audio_candidates = _expand_bilibili_audio_candidates(audio_candidates)
    video_candidates = _stream_candidates(info, audio=False)
    audio_url = audio_candidates[0]["url"] if audio_candidates else _stream_url(info, audio=True)
    video_url = video_candidates[0]["url"] if video_candidates else _stream_url(info, audio=False)
    audio_headers = audio_candidates[0]["http_headers"] if audio_candidates else dict(headers)
    video_headers = video_candidates[0]["http_headers"] if video_candidates else dict(headers)
    if video_candidates and "acodec" in video_candidates[0]:
        metadata["acodec"] = video_candidates[0]["acodec"]
    if audio_candidates:
        for key in ("filesize", "filesize_approx", "clen"):
            if audio_candidates[0].get(key) is not None:
                metadata[key] = audio_candidates[0][key]
    return apply_video_quality_limit(
        MediaSource(
            platform=platform,
            source_url=source_url,
            source_id=source_id,
            display_name=str(info.get("title") or source_id or source_url),
            duration=duration,
            audio_url=audio_url,
            video_url=video_url,
            http_headers=headers,
            audio_headers=audio_headers,
            video_headers=video_headers,
            metadata=metadata,
            audio_candidates=audio_candidates,
            video_candidates=video_candidates,
            resolved_at=time.monotonic(),
        ),
        max_height,
    )


def _info_has_streams(info: Mapping[str, Any]) -> bool:
    """Return whether an info dict carries enough data to build a MediaSource.

    Bilibili playlist hydration uses the lightweight view API which returns
    only metadata (id/title/duration/pubdate/webpage_url) with no stream
    candidates. Reusing that as a full info would produce an empty-URL
    MediaSource, so import must fall back to a full resolve when no usable
    stream data is present.
    """
    if not isinstance(info, Mapping):
        return False
    for key in ("formats", "requested_formats"):
        value = info.get(key)
        if isinstance(value, (list, tuple)) and value:
            return True
    for key in ("url", "audio_url", "video_url", "manifest_url"):
        if str(info.get(key) or "").strip():
            return True
    return False


def source_from_hydrated_entry(entry: PlaylistEntry, max_height: int | None = None) -> MediaSource:
    """Build a MediaSource from an entry hydrated earlier, avoiding a second
    full yt-dlp extraction on import. Falls back to the single-VOD path when
    the cached info is a playlist or lacks stream candidates."""
    info = (entry.metadata or {}).get("_resolved_info")
    if not isinstance(info, Mapping):
        raise SourceResolveError(f"Source was not hydrated: {entry.webpage_url}")
    info = dict(info)
    info.setdefault("webpage_url", entry.webpage_url)
    info.setdefault("id", entry.entry_id)
    if info.get("_type") in ("playlist", "multi_video"):
        raise SourceResolveError(f"Source is a playlist, not a single VOD: {entry.webpage_url}")
    if not _info_has_streams(info):
        raise SourceResolveError(
            f"Source metadata is not resolvable without a fresh extraction: {entry.webpage_url}"
        )
    source = _source_from_info(entry.webpage_url, info, max_height)
    if not (source.audio_url or source.video_url):
        raise SourceResolveError(
            f"Source has no usable stream URLs: {entry.webpage_url}"
        )
    return source


def apply_video_quality_limit(source: MediaSource, max_height: int | None) -> MediaSource:
    """Limit remote video segment quality to the configured maximum height.

    Prefers the highest video candidate at or below ``max_height``. If no
    candidate is at or below the limit, falls back to the lowest available
    candidate. ``None`` keeps the current highest-quality behavior. Audio
    candidates are never touched.
    """
    source.max_height = max_height
    candidates = list(source.video_candidates or [])
    if max_height is None or not candidates:
        return source
    try:
        max_height = int(max_height)
    except (TypeError, ValueError):
        return source

    def height_of(candidate):
        try:
            value = int(candidate.get("height"))
            return value if value > 0 else 0
        except (TypeError, ValueError):
            return 0

    at_or_below = [
        candidate for candidate in candidates
        if height_of(candidate) > 0 and height_of(candidate) <= max_height
    ]
    if at_or_below:
        selected = max(at_or_below, key=height_of)
    else:
        with_height = [candidate for candidate in candidates if height_of(candidate) > 0]
        selected = min(with_height, key=height_of) if with_height else candidates[0]
    if selected.get("url"):
        source.video_url = str(selected["url"])
        if selected.get("http_headers"):
            source.video_headers = {str(k): str(v) for k, v in selected["http_headers"].items()}
    return source


def resolve_source(
    url: str,
    ydl_factory: Callable[..., Any] | None = None,
    browser_cookies: str | None = None,
    max_height: int | None = None,
) -> MediaSource:
    """Resolve one VOD with yt-dlp metadata extraction only."""
    value = url.strip()
    if not value:
        raise SourceResolveError("Cannot resolve an empty source URL")
    try:
        info = _extract_with_cookie_policy(value, ydl_factory, browser_cookies)
    except ValueError as exc:
        raise SourceResolveError(str(exc)) from exc
    source = _source_from_info(value, info)
    return apply_video_quality_limit(source, max_height)


def _is_url(value: str) -> bool:
    return urlsplit(value).scheme in {"http", "https"}


def _dedupe_sources(sources: list[MediaSource]) -> list[MediaSource]:
    result = []
    seen = set()
    for source in sources:
        identity = stable_source_id(source)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(source)
    return result


def expand_input(
    value: str,
    ydl_factory: Callable[..., Any] | None = None,
    browser_cookies: str | None = None,
    failure_logger: Callable[[str], Any] | None = None,
) -> list[MediaSource]:
    """Expand a single URL or text file into independently resolved VODs."""
    input_value = value.strip()
    if not input_value:
        raise SourceExpansionError("Input is empty")

    if not _is_url(input_value):
        descriptor = describe_input(
            input_value, ydl_factory=ydl_factory, browser_cookies=browser_cookies
        )
    else:
        descriptor = describe_input(
            input_value, ydl_factory=ydl_factory, browser_cookies=browser_cookies
        )
    entries = [descriptor] if isinstance(descriptor, PlaylistEntry) else descriptor._entries
    sources = []
    failures = []
    for entry in entries:
        entry_url = entry.webpage_url
        try:
            sources.append(resolve_source(
                entry_url,
                ydl_factory=ydl_factory,
                browser_cookies=browser_cookies,
            ))
        except SourceResolveError as exc:
            failures.append(f"{entry_url}: {exc}")
            message = f"Remote source failed ({entry_url}): {exc}"
            (failure_logger or print)(message)
    if not sources:
        detail = "; ".join(failures) or "no sources were resolved"
        raise SourceExpansionError(f"No remote sources could be resolved: {detail}")
    return _dedupe_sources(sources)
