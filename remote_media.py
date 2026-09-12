"""Metadata resolution and input expansion for remote VOD sources."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import tempfile
import os
import re
import subprocess
import threading
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from utils import (FFMPEG_PATH, run_tracked, run_tracked_progress,
                   is_twitch_platform)
from remote_prefetch import (
    SpeedMonitor,
    _probe_real_size,
    _parse_hls_playlist,
    iter_hls_bytes,
    iter_range_bytes,
    supports_hls_prefetch,
    supports_range_prefetch,
)
from progress import format_transfer_progress, format_hls_progress

# "Preparing clips"（远程片段下载）阶段的看门狗参数：
#   - progress_stall_timeout：out_time 连续这么久不前进即判定卡死并重试
#     （正常的慢速 CDN 只要有数据前进就会重置计时器）。
#   - heartbeat：只在"确实卡了"（进度停滞 ≥30s）时每 30s 打一条，并附一条
#     成因提示。以前每 60s 无条件打一条，慢线路下满屏都是心跳；现在正常的
#     （哪怕偏慢但有前进的）下载完全不刷日志。
_SEGMENT_PROGRESS_STALL_TIMEOUT = 120.0
_SEGMENT_HEARTBEAT_INTERVAL = 30.0
_SEGMENT_HEARTBEAT_STALL_THRESHOLD = 30.0
_SEGMENT_HEARTBEAT_HINT = (
    "Note: a clip fetch this slow is usually caused by Max Download Concurrency "
    "being too high for your connection, or by an unstable network path to the CDN. "
    "Lowering Max Download Concurrency in Remote Settings (or waiting for a stable "
    "network) normally fixes it. The clip is retried automatically, the run is not "
    "frozen.")
# 单次取片段的超时不再一律 600s：按"这个 clip 预计要下多少字节"算，5 秒的
# clip 不该独占 10 分钟。下限保证慢但稳定的线路仍能跑完（约 0.15 MB/s ≈
# 1.2 Mbit/s 的持续吞吐就来得及），上限维持原来的 600s。
_SEGMENT_TIMEOUT_FLOOR = 120.0
_SEGMENT_TIMEOUT_CAP = 600.0
_SEGMENT_MIN_THROUGHPUT = 0.15 * 1024 * 1024          # bytes/second
# 与 utils.check_compile_disk_space 同一套码率估算（每秒钟媒体约多少字节）。
_QUALITY_BYTES_PER_SECOND = ((480, 200 * 1024), (720, 350 * 1024), (1080, 500 * 1024))
_DEFAULT_BYTES_PER_SECOND = 500 * 1024
# Audio Cache 的 FFmpeg 回落下载是"单文件、无续传"的：卡死判定后重试会从 0 重新
# 下载整个 VOD，所以阈值放宽（无输出 30s 看门狗仍然生效，那是真·连接挂死）。
_AUDIO_CACHE_FALLBACK_STALL_TIMEOUT = 300.0
_PLATFORM_LABELS = {"youtube": "YouTube", "bilibili": "Bilibili",
                    "bilibiliweb": "Bilibili", "twitch": "Twitch"}


def platform_display_name(platform) -> str:
    """Human-readable platform name for logs/progress.

    yt-dlp 报的 Twitch key 是 ``twitchvod`` / ``twitchstream`` / ``twitchclips``
    等，本项目描述符还有 ``twitch-vods``：精确查表会让这些来源在日志/心跳里
    显示成原始 key，所以按平台族兜底。
    """
    key = str(platform or "").strip()
    label = _PLATFORM_LABELS.get(key.lower())
    if label:
        return label
    if is_twitch_platform(key):
        return "Twitch"
    return key or "Remote"
# 交付比请求短超过这个秒数就重取一次（最后一次尝试仍短则接受并记录）。
# 以前静默接受：clip 被对齐裁到实际交付长度，结尾比 padding 预期早、切点生硬。
_SHORT_SEGMENT_TOLERANCE = 0.3
_short_delivery_records: list[dict[str, Any]] = []
_short_delivery_lock = threading.Lock()


def reset_short_delivery_records() -> None:
    """Clear the per-run short-delivery log (called before materialize)."""
    with _short_delivery_lock:
        _short_delivery_records.clear()


def short_delivery_records() -> list[dict[str, Any]]:
    """Clips whose downloaded segment was shorter than the requested window."""
    with _short_delivery_lock:
        return list(_short_delivery_records)


def _log_short_delivery(source, start, end, actual, expected) -> None:
    shortfall = max(0.0, float(expected) - float(actual))
    platform = getattr(source, "platform", None) or "remote"
    source_id = getattr(source, "source_id", None) or "?"
    label = f"{platform}:{source_id}"
    record = {"name": label, "start": float(start), "end": float(end),
              "actual": float(actual), "expected": float(expected),
              "shortfall": shortfall}
    with _short_delivery_lock:
        _short_delivery_records.append(record)
    print(f"  {label} clip {float(start):g}-{float(end):g}s was delivered "
          f"{shortfall:.2f}s short ({actual:.2f}s of {expected:.2f}s); "
          f"the clip will end earlier than the padding suggests.")


class RemoteMediaError(Exception):
    """Base error for remote media input and resolution failures."""


class SourceResolveError(RemoteMediaError):
    """A single source could not be resolved."""


class LiveBroadcastError(SourceResolveError):
    """The URL points at an ongoing broadcast instead of a finished replay.

    直播/正在进行的 HLS 只有一个几分钟的滑动窗口：检测能"找到片段"（音频按实时
    读取），但 compile 阶段按绝对时间取片段时，窗口外的片段全部取不到——用户看到
    的是"找到的片段全部抓取失败"。所以在解析阶段就明确拒绝，让批次跳过该源并给出
    可操作提示，而不是先花几小时检测再全量失败。
    """


# 只把明确"正在进行中"的标记当作直播：yt-dlp 对回放给 was_live/post_live，
# 对直播给 is_live。post_live（刚结束、回放仍在转码）不拒绝，只提示。
_LIVE_STATUSES = {"is_live", "live"}


def _live_flags(value) -> tuple[bool, bool]:
    """Return (is_live, still_processing) from an info dict or MediaSource."""
    metadata = getattr(value, "metadata", None)
    if metadata is None:
        metadata = value if isinstance(value, Mapping) else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    status = str(metadata.get("live_status") or "").strip().lower()
    is_live = metadata.get("is_live") is True or status in _LIVE_STATUSES
    processing = status in {"post_live", "is_upcoming"}
    return bool(is_live), bool(processing)


def source_is_live(value) -> bool:
    """Whether the resolved metadata describes an ongoing broadcast."""
    return _live_flags(value)[0]


def source_is_still_processing(value) -> bool:
    """Whether the broadcast just ended / has not started (replay may be partial)."""
    return _live_flags(value)[1]


class SourceExpansionError(RemoteMediaError):
    """An input value could not be interpreted or expanded."""


class SegmentFetchError(RemoteMediaError):
    """A requested remote media segment could not be materialized."""


class _CdnSwitchSignal(Exception):
    """Internal signal: an Audio Cache download switched CDN mid-transfer and
    the caller should resume from the already-written bytes."""


# Audio Cache: consecutive slow samples before probing/switching CDN, and the
# max number of CDN switches per source (avoids infinite re-probing when every
# CDN host is slow).
_AUDIO_CACHE_SLOW_STREAK_SWITCH = 3
_AUDIO_CACHE_MAX_CDN_SWITCHES = 3
# Candidate probing: bail out of select_audio_candidate after this many
# consecutive candidate-groups all fail to produce a speed (network/CDN
# unhealthy) instead of wasting ~5s per probe across all candidates.
_PROBE_MAX_FAILED_GROUPS = 2


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
    # B 站国际 CDN（Akamai 全球网络）：海外用户直连国内 upos 慢时的主力选项；
    # 探测按实测速度选，国内用户探得慢就不会被选中。
    "upos-hz-mirrorakam.akamaized.net",
)
_REMOTE_SEEK_PAD = 10.0
# 输入读取窗余量（秒）：覆盖 HLS 分片粒度导致的 seek 落点偏差（实测 ~1s），
# 保证 trim 终点前的内容一定被读到。余量只扩大下载范围，不改变输出内容。
_REMOTE_READ_MARGIN = 3.0
# Playlist 条目 hydration 的解析超时。普通 resolve 用 90s，但 playlist 浏览时
# 每个条目都可能触发一次 hydration（4 并发），慢网络下 90s/条会堆积大量后台
# 线程；缩短到 30s 避免翻页时资源膨胀。
_PLAYLIST_HYDRATION_TIMEOUT = 30.0
# 大型合集列表实测 2000+。导入走 extract_flat（轻量、无逐视频 metadata），
# 旧 1000 上限是全量 metadata 时代的保守值；仍保留一个天花板防止病态列表
# 冻死选择树 UI。autocomper.MAX_PLAYLIST_ENTRIES 即本常量的再导出。
MAX_PLAYLIST_ENTRIES = 5000


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
        error = None
        try:
            hydrated = self._hydrate_entry(entry)
        except Exception as exc:
            hydrated = None
            error = _classify_hydration_error(exc)
        if hydrated:
            _merge_entry_metadata(entry, hydrated)
            entry.metadata["_resolved_info"] = dict(hydrated)
            self._hydrated_indices.add(entry.index)
            entry.metadata["hydration_failed"] = False
            entry.metadata.pop("hydration_error", None)
        else:
            entry.metadata["hydration_failed"] = True
            if error:
                entry.metadata["hydration_error"] = error
        return entry

    def failed_hydration_entries(self) -> list[PlaylistEntry]:
        """Entries whose last hydration attempt failed (retry candidates)."""
        return [entry for entry in self._entries
                if getattr(entry, "metadata", {}).get("hydration_failed")]


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


def _probe_failure_detail(result_or_exc) -> str:
    """Return a concise failure detail (returncode + stderr tail) for a probe."""
    rc = getattr(result_or_exc, "returncode", None)
    stderr = getattr(result_or_exc, "stderr", None) or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    lines = [str(line).strip() for line in str(stderr).splitlines() if str(line).strip()]
    tail = "\n".join(lines[-3:])[-300:]
    detail = f"rc={rc}" if rc is not None else type(result_or_exc).__name__
    if tail:
        detail += f" stderr={tail}"
    return detail


def probe_audio_candidate(
    source: MediaSource,
    candidate: Mapping[str, Any],
    duration: float = 3,
    timeout: float = 5,
    run_func: Callable[..., Any] | None = None,
    log_func: Callable[[str], Any] | None = None,
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
    if not (math.isfinite(duration_value) and math.isfinite(timeout_value)):
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
    except Exception as exc:
        if log_func is not None:
            log_func(f"Audio probe failed ({_probe_failure_detail(exc)}); skipping candidate")
        return None
    if isinstance(result, (int, float)):
        return float(result)
    if getattr(result, "returncode", 0) not in (0, None):
        if log_func is not None:
            log_func(f"Audio probe failed ({_probe_failure_detail(result)}); skipping candidate")
        return None
    output = "\n".join(str(value or "") for value in (
        getattr(result, "stderr", ""), getattr(result, "stdout", "")
    ))
    match = re.search(r"speed\s*=\s*([0-9]+(?:\.[0-9]+)?)x", output, re.IGNORECASE)
    return float(match.group(1)) if match else None


class ProbeCooldown:
    """跨源探测连败冷却：CDN 持续 403 时放慢探测节奏，等限流窗口过去。

    连续 ``threshold`` 个源的候选组全部探测失败（403/超时）时，下一个源
    探测前先等待 ``cooldown`` 秒；任何一次有候选测出有效速度即清零。
    不减少镜像探测覆盖，只在大批次尾部自动放慢，避免越跑越多 403。
    """

    def __init__(self, threshold: int = 2, cooldown: float = 30.0):
        self.threshold = max(1, int(threshold))
        self.cooldown = float(cooldown)
        self._streak = 0

    def wait_if_needed(self, sleep_func=time.sleep, log_func=None) -> bool:
        if self._streak < self.threshold:
            return False
        if log_func is not None:
            log_func(
                f"Bilibili CDN probes failed for {self._streak} sources in a row; "
                f"cooling down {self.cooldown:g}s before the next source")
        sleep_func(self.cooldown)
        self._streak = 0
        return True

    def observe(self, stats: Mapping[str, int] | None) -> None:
        if not stats:
            return
        groups = int(stats.get("groups") or 0)
        failed = int(stats.get("failed_groups") or 0)
        if groups <= 0:
            return
        if failed >= groups:
            self._streak += 1
        else:
            self._streak = 0


def select_audio_candidate(
    source: MediaSource,
    min_realtime_speed: float = 1.0,
    probe_duration: float = 3,
    run_func: Callable[..., Any] | None = None,
    log_func: Callable[[str], Any] | None = None,
    stats: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """Keep the best audio unless a bounded probe shows a slower candidate is needed.

    If every candidate probe fails (timeout / 403), the source is left untouched
    (its resolve-time ``audio_url``) and ``None`` is returned instead of silently
    re-applying the first candidate — a probe that can't complete tells us nothing
    about which CDN is usable, and forcing a URL here can pin a stale/rate-limited
    one for the whole batch. ``stats`` optionally receives probe outcome counts
    (``groups`` measured, ``failed_groups`` fully failed) for callers that adapt
    pacing to CDN health.
    """
    candidates = list(source.audio_candidates or [])
    if str(source.platform or "").strip().lower() not in {"bilibili", "bilibiliweb"}:
        # audio_candidates 对非 Bilibili 只是不同码率的音频格式，不是 CDN 镜像；
        # 对它们做"速度探测 + 选路"毫无意义，还可能凭空多打几次 CDN 请求
        # （YouTube 每个格式一次探测），并产生误导性的 "CDN probe failed
        # repeatedly"。CDN 候选切换（apply_audio_candidate 改 audio_url）只对
        # Bilibili 的多 CDN 候选有效。
        return None
    if not candidates:
        return None
    try:
        threshold = float(min_realtime_speed)
    except (TypeError, ValueError):
        threshold = 1.0
    stats_state = {"groups": 0, "failed_groups": 0}
    index = 0
    chose = False
    chosen_candidate = None
    probe_fail_groups = 0
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
        stats_state["groups"] += 1
        for candidate in group:
            speed = probe_audio_candidate(
                source, candidate, duration=probe_duration, run_func=run_func,
                log_func=log_func)
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
            # 整组候选探测全失败（unknown/timeout/403）：连续失败说明网络或
            # CDN 不健康，继续测剩余候选只会白等（每个超时 5s，×12 候选/源
            # 在大批次上累计数小时）。连续 2 组失败即提前放弃探测。
            probe_fail_groups += 1
            stats_state["failed_groups"] += 1
            if probe_fail_groups >= _PROBE_MAX_FAILED_GROUPS:
                if log_func is not None:
                    log_func(
                        "CDN probe failed repeatedly; keeping the default audio "
                        "URL (download may be slower)"
                    )
                break
            continue
        probe_fail_groups = 0
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
    if stats is not None:
        stats.update(stats_state)
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
        # 网络读守卫：半开的 CDN 连接不能永久挂住 worker（与取段命令对齐）。
        "-rw_timeout",
        "60000000",
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
    # 非/无限时长（JSON 缓存可往返 Infinity 字面量）按未知处理；
    # 上限 24h：inf*3 若不加守卫会让 run_tracked 的 timeout 失效（永不超时）。
    if duration <= 0 or not math.isfinite(duration):
        return 1800.0
    return min(24 * 3600.0, max(1800.0, duration * 3))


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
    prefetch_chunk_size: int = 4 * 1024 * 1024,
    prefetch_concurrency: int = 4,
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
    platform_label = platform_display_name(source.platform)

    if source.platform in {"youtube", "bilibili"} \
            and not supports_range_prefetch(source) \
            and str(source.audio_url or "").startswith(("http://", "https://")) \
            and run_func is None:
        # yt-dlp 元数据缺 filesize 时 supports_range_prefetch 为 False，会静默
        # 回落 FFmpeg 单连接慢路径（用户表现为 Audio Cache 龟速）。先用
        # bytes=0-0 探测真实流大小写入候选元数据；拿到就走并行预取，探测
        # 失败才回落（行为同旧版）。
        try:
            probed_size = _probe_real_size(
                source.audio_url,
                dict(getattr(source, "audio_headers", None)
                     or getattr(source, "http_headers", None) or {}))
        except Exception:
            probed_size = None
        if probed_size:
            for item in (getattr(source, "audio_candidates", None) or []):
                if isinstance(item, dict) and str(item.get("url") or "") == str(source.audio_url):
                    item["filesize"] = probed_size
                    break
            else:
                metadata = getattr(source, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["filesize"] = probed_size
            report(f"{platform_label} audio stream size probed: {probed_size} bytes "
                   f"for {cache_label}")

    if source.platform in {"youtube", "bilibili"} and supports_range_prefetch(source):
        try:
            report(f"{platform_label} audio cache prefetch for {cache_label}")
            speed_monitor = SpeedMonitor()

            slow_streak = [0]
            cdn_switches = [0]

            def recover_slow_prefetch(stale_source, speed):
                report(
                    f"{platform_label} audio cache speed low ({speed:.2f} MB/s); "
                    "refreshing source before continuing"
                )
                # 先尝试刷新签名 URL（URL 过期/限流时刷新可能直接恢复）。
                if refresh_func is not None:
                    try:
                        updated = refresh_func(stale_source)
                        if isinstance(updated, MediaSource) and updated is not stale_source:
                            stale_source.__dict__.update(updated.__dict__)
                    except Exception:
                        pass
                # 连续低速达到阈值 → 重新探测候选组并切换 CDN 节点
                # （bilibili 的 audio_candidates 常含多个 cdn_host）。
                # 切换成功后抛信号中断当前下载，让外层 resume 从已写字节续传，
                # 100-500MB 的音频不从头重下。
                slow_streak[0] += 1
                if (slow_streak[0] >= _AUDIO_CACHE_SLOW_STREAK_SWITCH
                        and cdn_switches[0] < _AUDIO_CACHE_MAX_CDN_SWITCHES):
                    slow_streak[0] = 0
                    cdn_switches[0] += 1
                    try:
                        chosen = select_audio_candidate(
                            stale_source, min_realtime_speed=1.0,
                            probe_duration=3, log_func=report)
                        if chosen is not None:
                            host = chosen.get("cdn_host") or chosen.get("url", "").split("/")[2] or "?"
                            report(
                                f"{platform_label} audio cache switched CDN "
                                f"({cdn_switches[0]}/{_AUDIO_CACHE_MAX_CDN_SWITCHES}) "
                                f"to {host}")
                            raise _CdnSwitchSignal()
                    except _CdnSwitchSignal:
                        raise
                    except Exception as exc:
                        report(f"CDN switch probe failed: {type(exc).__name__}")

            resume_limit = 5
            for resume_attempt in range(resume_limit):
                partial_size = temporary_path.stat().st_size if temporary_path.exists() else 0
                try:
                    with temporary_path.open("ab" if partial_size else "wb") as handle:
                        for chunk in iter_range_bytes(
                            source, chunk_size=prefetch_chunk_size,
                            concurrency=prefetch_concurrency, logger=log,
                            progress_callback=progress_callback,
                            speed_monitor=speed_monitor,
                            slow_callback=recover_slow_prefetch,
                            start_offset=partial_size,
                            refresher=refresh_func,
                        ):
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                    break
                except _CdnSwitchSignal:
                    # CDN 已切换：续传（不消耗 resume_attempt 次数）
                    if resume_attempt >= resume_limit - 1:
                        raise
                    report(
                        f"{platform_label} audio cache CDN switched; "
                        f"resuming at {partial_size} bytes"
                    )
                except InterruptedError:
                    raise
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
        except InterruptedError:
            temporary_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            report(f"{platform_label} audio cache prefetch failed for {cache_label}; falling back to FFmpeg")

    if is_twitch_platform(source.platform) and supports_hls_prefetch(source):
        fd, transport = tempfile.mkstemp(prefix=".remote-hls-", suffix=".ts", dir=str(destination.parent))
        os.close(fd)
        transport_path = Path(transport)
        try:
            report(f"Twitch HLS audio cache prefetch for {cache_label}")
            if progress_callback is not None:
                # 立刻切到当前源的进度卡（此前 Twitch 只在 FFmpeg 回落路径
                # 才有字节监视，HLS 预取期间界面完全没有动静）。
                progress_callback(format_hls_progress(0.0, duration_value, 0.0))
            with transport_path.open("wb") as handle:
                for chunk in iter_hls_bytes(
                    source, concurrency=prefetch_concurrency, logger=log,
                    progress_callback=progress_callback,
                    refresher=refresh_func,
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
            # 已知源时长且是生产路径时，用 FFmpeg 自己的 -progress（out_time =
            # 已写入媒体时间）报百分比/实时倍速/ETA：Twitch 没有 Content-Length，
            # 字节监视只能显示 MB 与 MB/s，界面看起来"没有进度"。
            use_media_progress = bool(
                progress_callback is not None and run_func is None
                and duration_value and duration_value > 0)

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

            monitor = None
            if not use_media_progress:
                monitor = threading.Thread(target=monitor_file,
                                           name="audio-cache-progress", daemon=True)
                monitor.start()
            try:
                if use_media_progress:
                    def report_media_progress(current_seconds, total_seconds, elapsed):
                        progress_callback(format_hls_progress(
                            current_seconds, total_seconds or duration_value, elapsed))
                    result = run_tracked_progress(
                        command, duration=duration_value, timeout=effective_timeout,
                        progress_callback=report_media_progress, stall_timeout=30,
                        progress_stall_timeout=_AUDIO_CACHE_FALLBACK_STALL_TIMEOUT,
                        heartbeat_label=f"{platform_label}:{source.source_id or '?'}",
                        heartbeat_interval=_SEGMENT_HEARTBEAT_INTERVAL,
                        heartbeat_verb="downloading")
                else:
                    result = runner(command, timeout=effective_timeout, text=True)
            finally:
                stop_monitor.set()
                if monitor is not None:
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


# ── fMP4 HLS（Twitch 新版切片）窗口抓取 ─────────────────────────────────────
# Twitch 的一部分 VOD 用 fMP4 打包 HLS（playlist 带 #EXT-X-MAP 初始化段、分片是
# .mp4）而不是老的 MPEG-TS。对这种 playlist，ffmpeg 的网络 seek 会退化：它打开
# 正确分片后仍逐片顺序读取、几十秒内吐不出任何帧（实测 605s 处的 clip：60s 内
# 输出 0 字节，而同一个分片单独下载只需 0.4s），于是每个 clip 都被 stall/超时
# 看门狗判失败 → 用户看到"找到的片段全部抓取失败"，且原因落进汇总的
# "other error" 桶（测试者 2026-09-11 的报告，VOD 2870466234）。
# 对策：不信 ffmpeg 的网络 seek——自己按 playlist 的 EXTINF 定位分片，下载
# init + 覆盖窗口的分片，先 copy remux 成本地 mp4，再按普通本地文件精确裁剪
# （本地 -ss 不走 HLS demuxer 的 seek 路径）。窗口数学已逐像素验证：本地裁剪的
# 帧序列与"直接从目标分片取参考帧"完全一致（0/276480 字节差异）。
_HLS_SEGMENT_SUFFIXES = (".mp4", ".m4s", ".cmfv", ".cmfa", ".m4v")
_HLS_PLAN_TTL = 300.0
_HLS_SEGMENT_TIMEOUT = 60.0
_hls_manifest_cache: dict[str, tuple[float, tuple]] = {}
_hls_manifest_lock = threading.Lock()


def _is_hls_manifest(url) -> bool:
    return urlsplit(str(url or "")).path.lower().endswith((".m3u8", ".m3u"))


def _segment_file_suffix(url) -> str:
    suffix = Path(urlsplit(str(url or "")).path).suffix.lower()
    return suffix if suffix in _HLS_SEGMENT_SUFFIXES else ".bin"


def _http_read_bytes(url, headers, timeout=_HLS_SEGMENT_TIMEOUT) -> bytes:
    request = Request(str(url), headers={str(k): str(v) for k, v in (headers or {}).items()})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def hls_playlist_entries(playlist_url, headers=None, refresh=False):
    """Parsed HLS playlist for a VOD: ([(url, media_seconds), ...], total).

    Cached per URL for ``_HLS_PLAN_TTL`` seconds: a compile fetches many clips
    from one source and the playlist itself is small.
    """
    key = str(playlist_url)
    now = time.monotonic()
    if not refresh:
        with _hls_manifest_lock:
            cached = _hls_manifest_cache.get(key)
        if cached is not None and now - cached[0] <= _HLS_PLAN_TTL:
            return cached[1]
    manifest = _http_read_bytes(playlist_url, headers).decode("utf-8", "replace")
    entries = _parse_hls_playlist(
        manifest, str(playlist_url),
        lambda url: _http_read_bytes(url, headers))
    with _hls_manifest_lock:
        if len(_hls_manifest_cache) >= 64:
            _hls_manifest_cache.clear()
        _hls_manifest_cache[key] = (now, entries)
    return entries


def hls_is_fragmented_mp4(entries) -> bool:
    """Whether the playlist is fMP4 (init segment via #EXT-X-MAP / .mp4 parts)."""
    if not entries:
        return False
    for url, seconds in entries:
        if seconds <= 0:                     # #EXT-X-MAP init segment
            return True
        if _segment_file_suffix(url) in _HLS_SEGMENT_SUFFIXES:
            return True
    return False


def plan_hls_window(entries, start, end):
    """Segments needed for [start, end] (plus seek/read margins), or None.

    Returns ``None`` for TS playlists (ffmpeg seeks those fine over the network)
    and for windows that cannot be located.
    """
    if not hls_is_fragmented_mp4(entries):
        return None
    seek_start = max(0.0, float(start) - _REMOTE_SEEK_PAD)
    window_end = float(end) + _REMOTE_READ_MARGIN
    picked, offset = [], 0.0
    for url, seconds in entries:
        if seconds <= 0:
            picked.append((url, offset, 0.0))       # init segment, no timeline
            continue
        if offset + seconds > seek_start and offset < window_end:
            picked.append((url, offset, seconds))
        offset += seconds
    media = [item for item in picked if item[2] > 0.0]
    if not media:
        return None
    first_offset = media[0][1]
    inner = max(0.0, seek_start - first_offset)
    return {
        "entries": picked,
        "first_offset": first_offset,
        "inner_seek": inner,
        # 窗口文件里第 0 秒对应的时间轴位置（本地裁剪的换算基准）
        "window_start": first_offset + inner,
        "media_total": offset,
    }


def hls_window_plan(playlist_url, headers, start, end):
    """fMP4 window plan for a stream URL, or None when not applicable.

    Any failure here (playlist unreadable, TS playlist, window not found) just
    means "use the normal network command" — this must never turn a working
    source into a failing one.
    """
    if not _is_hls_manifest(playlist_url):
        return None
    try:
        entries, _total = hls_playlist_entries(playlist_url, headers)
    except Exception:
        return None
    plan = plan_hls_window(entries, start, end)
    if plan is not None:
        plan["playlist_url"] = str(playlist_url)
        plan["headers"] = {str(k): str(v) for k, v in (headers or {}).items()}
    return plan


class _HlsWindow:
    """A local file holding the needed HLS segments, plus its cleanup.

    ``file_start`` is the media time of the file's first frame (the playlist
    offset of the first downloaded segment) and ``inner_seek`` the position
    inside the file that corresponds to the requested read origin — the cut has
    to be expressed relative to those two, not to the clip start.
    """

    def __init__(self, path: Path, temp_dir: Path, file_start: float,
                 inner_seek: float, seconds: float):
        self.path = Path(path)
        self.temp_dir = Path(temp_dir)
        self.file_start = float(file_start)
        self.inner_seek = float(inner_seek)
        self.seconds = float(seconds)

    @property
    def read_origin(self) -> float:
        """Media time of the frame the local seek lands on."""
        return self.file_start + self.inner_seek

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def materialize_hls_window(plan, work_dir, timeout=_HLS_SEGMENT_TIMEOUT,
                           runner=None, retries=1, remux_timeout=None):
    """Download a window's segments and remux them into one local MP4.

    ``timeout`` bounds ONE segment read (a segment is a few MB, 60s is generous
    even on a slow line); ``remux_timeout`` bounds the local remux that assembles
    them, which is proportional to the whole window, so callers pass the clip's
    adaptive budget instead of the flat 60s.

    Raises SegmentFetchError on failure: at this point we know the source is
    fMP4, where the network-seek command cannot work, so the caller's existing
    retry/refresh ladder is the right response (falling back would burn a stall
    watchdog timeout for nothing).
    """
    remux_timeout = timeout if remux_timeout is None else float(remux_timeout)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".remote-hls-window-", dir=str(work_dir)))
    headers = dict(plan.get("headers") or {})
    try:
        lines = ["#EXTM3U", "#EXT-X-VERSION:7", "#EXT-X-TARGETDURATION:10",
                 "#EXT-X-PLAYLIST-TYPE:VOD"]
        seconds_total = 0.0
        for index, (url, _offset, seconds) in enumerate(plan["entries"]):
            name = f"s{index:05d}{_segment_file_suffix(url)}"
            last_error = None
            for attempt in range(max(1, int(retries) + 1)):
                try:
                    data = _http_read_bytes(url, headers, timeout=timeout)
                    if not data:
                        raise SegmentFetchError(f"empty HLS segment: {url}")
                    (temp_dir / name).write_bytes(data)
                    last_error = None
                    break
                except Exception as exc:                            # noqa: BLE001
                    last_error = exc
                    if attempt < int(retries):
                        time.sleep(min(2 ** attempt, 4))
            if last_error is not None:
                raise SegmentFetchError(
                    f"Could not download HLS segment {name}: {last_error}")
            if seconds <= 0:
                lines.append(f'#EXT-X-MAP:URI="{name}"')
            else:
                lines.append(f"#EXTINF:{seconds:.3f},")
                lines.append(name)
                seconds_total += seconds
        lines.append("#EXT-X-ENDLIST")
        playlist_path = temp_dir / "window.m3u8"
        playlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        window_file = temp_dir / "window.mp4"
        command = [
            str(FFMPEG_PATH), "-y", "-hide_banner", "-loglevel", "error",
            "-allowed_extensions", "ALL",
            "-i", str(playlist_path),
            "-c", "copy", "-movflags", "+faststart",
            str(window_file),
        ]
        result = (runner or run_tracked)(command, timeout=remux_timeout, text=True)
        return_code = getattr(result, "returncode", 0)
        if return_code != 0 or not window_file.is_file() or window_file.stat().st_size == 0:
            detail = _sanitize_ffmpeg_detail(
                getattr(result, "stderr", "") or getattr(result, "stdout", "")
                or "no output")
            raise SegmentFetchError(
                f"Could not assemble the HLS window (rc={return_code}): {detail}")
        return _HlsWindow(window_file, temp_dir, plan["first_offset"],
                          plan["inner_seek"], seconds_total)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_local_window_command(window_file, file_start, inner_seek, start, end,
                               output_file, audio_only=False, codec=None,
                               has_audio=True):
    """Cut [start, end] out of a local window file (exact local seek).

    ``file_start`` is the media time of the window file's first frame and
    ``inner_seek`` the file-relative position to seek to (the plan's read
    origin). The trim window is then expressed relative to that origin, which is
    what makes the cut land exactly on [start, end].

    Output options mirror ``build_segment_command`` so the produced clip is
    interchangeable with the network path's result.
    """
    start_value, end_value = float(start), float(end)
    if end_value <= start_value or start_value < 0:
        raise ValueError("segment end must be greater than a non-negative start")
    duration = end_value - start_value
    inner = max(0.0, float(inner_seek))
    origin = float(file_start) + inner
    trim_start = max(0.0, start_value - origin)
    trim_end = trim_start + duration
    read_span = trim_end + _REMOTE_READ_MARGIN
    command = [
        str(FFMPEG_PATH), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", _segment_number(inner, "window seek"),
        "-t", _segment_number(read_span, "window read span"),
        "-i", str(window_file),
    ]
    trim_start_text = _segment_number(trim_start, "trim start")
    trim_end_text = _segment_number(trim_end, "trim end")
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
        if has_audio:
            command.extend([
                "-af",
                f"asetpts=PTS-STARTPTS,atrim=start={trim_start_text}:end={trim_end_text},"
                "asetpts=PTS-STARTPTS",
            ])
    command.extend(["-t", _segment_number(duration, "duration")])
    if audio_only:
        command.append("-vn")
        if codec:
            command.extend(["-c:a", str(codec)])
    else:
        command.extend(["-map", "0:v:0"])
        if has_audio:
            command.extend(["-map", "0:a:0", "-c:a", "aac", "-shortest"])
        else:
            command.extend(["-map", "0:a:0?"])
        if codec:
            command.extend(["-c:v", str(codec)])
    command.extend([
        "-avoid_negative_ts", "make_zero",
        "-reset_timestamps", "1",
    ])
    if Path(output_file).suffix.lower() in {".mp4", ".m4a", ".mov"}:
        command.extend(["-movflags", "+faststart"])
    command.append(str(output_file))
    return command


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
    if (start_value < 0 or end_value <= start_value
            or not math.isfinite(start_value) or not math.isfinite(end_value)):
        raise ValueError("segment end must be greater than a non-negative start")

    if audio_only and not source.audio_url:
        raise ValueError("MediaSource has no audio_url")
    if not audio_only and not source.video_url:
        stream_name = "audio_url" if audio_only else "video_url"
        raise ValueError(f"MediaSource has no {stream_name}")

    duration = end_value - start_value
    seek_start = max(0.0, start_value - _REMOTE_SEEK_PAD)
    trim_offset = start_value - seek_start
    # 输入读取窗余量：HLS/TS 的输入 -t 按 seek "落点"计时，而落点可能比分片
    # 边界晚（Twitch ~1s 分片）→ 读取在 trim 终点前 δ(~1s) 处停止，段尾缺失。
    # 实测 58/678 个缓存 Twitch 段少交付 ~1.0s，该缺口在 concat 逐段累积成
    # 全局 A/V 失步。加固定余量保证 trim 永远拿得到完整窗口；trim 过滤器
    # 与输出 -t 仍精确裁剪，内容不受余量影响。
    input_duration = trim_offset + duration + _REMOTE_READ_MARGIN
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
    探测超时/异常属于"不确定"而非"无流"——重探一次再下结论，防止把好
    segment 误判成 unreadable（慢盘/AV 扫描可能让 10s 探测超时）。
    """
    kind = r"Video:" if want_video else r"Audio:"
    pattern = re.compile(r"Stream #\d+:\d+.*" + kind)
    for attempt in (1, 2):
        try:
            out = run_tracked([FFMPEG_PATH, "-hide_banner", "-i", str(path)],
                              timeout=10, text=True)
        except Exception:
            if attempt == 2:
                return False
            time.sleep(1)
            continue
        stderr = getattr(out, "stderr", None)
        if stderr is None:
            return True
        if not isinstance(stderr, str):
            stderr = str(stderr)
        # 无匹配是确定性结果（真无流），不重试
        return bool(pattern.search(stderr))
    return False


def _segment_duration(path) -> float | None:
    """Return the container duration of a downloaded segment in seconds.

    Uses the same bounded ffmpeg probe as `_segment_has_stream` so a corrupt or
    truncated download (which ffmpeg may still exit 0 for) can be detected
    before it enters the compile stage. 探测失败重试一次：返回 None 会同时关掉
    50% 截断检查和短交付检查，在并发抓取（磁盘忙）时不该静默失效。
    """
    stderr = None
    for probe_attempt in range(2):
        try:
            out = run_tracked(
                [FFMPEG_PATH, "-hide_banner", "-i", str(path),
                 "-probesize", "32M", "-analyzeduration", "100M"],
                timeout=10, text=True)
            stderr = getattr(out, "stderr", None)
        except Exception:
            stderr = None
        if isinstance(stderr, str) and "Duration:" in stderr:
            break
        if probe_attempt == 0:
            time.sleep(0.5)
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


def _rotate_video_candidate(source: MediaSource) -> bool:
    """无视频流失败后轮换到下一个视频候选（不同 CDN 边缘/格式）。

    视频输入拿不到帧而音频输入正常（音频走另一个 host）→ 产出只有音轨的
    片段。给该源一次换线机会；没有其他不同 URL 的候选时返回 False。
    """
    candidates = [c for c in (getattr(source, "video_candidates", None) or [])
                  if isinstance(c, dict) and str(c.get("url") or "")]
    current = str(source.video_url or "")
    urls = [str(c.get("url")) for c in candidates]
    try:
        current_index = urls.index(current)
    except ValueError:
        current_index = -1
    for step in range(1, len(urls) + 1):
        idx = (current_index + step) % len(urls)
        if urls[idx] == current:
            continue
        source.video_url = urls[idx]
        headers = candidates[idx].get("http_headers")
        if headers:
            source.video_headers = dict(headers)
        return True
    return False


# ═══ 片段落点校验（placement verification）═══════════════════════════════════
# 取回的区间不一定是请求的区间。"内容整体挪了、时长却完全正确"是同一类故障：
#   * Twitch（fMP4 HLS 视频轨，VOD 2840821927 @15501.40s）：窗口里的内容比请求
#     位置晚 0.61s，而同一位置的 audio-only 轨是准的（+0.01s）。同一 VOD 在
#     15493.40/15503.40 却完全准确——说明是"某个请求位置刚好落进坏区间"
#     （playlist 段偏移与段内媒体时间戳不一致），不是整条流的固定偏移。
#   * Bilibili Audio Cache（BV1NG4y1r7nK）：缓存的检测音频与线上流整体差
#     3.25s，时间戳在缓存里是对的，落到流上就早了 3.25s（与请求位置无关）。
# 用户看到的就是"clip 太早停/开头被吃掉"。这类问题只能在内容层面测：把 clip
# 开头的音频与"检测时间轴"做逐延迟归一化互相关，量出真实落点偏移 δ 再修。
#   1. |δ| ≤ 容差 → 不动（绝大多数 clip，只多两次本地解码）。
#   2. 按 start−δ 重取（Bilibili 这类整体偏移，实测一次就落到目标上）。
#   3. 仍不准 → 带 pre-roll 重取，按量出来的偏移在本地裁（Twitch 这类"请求
#      位置落进坏区间"的情况，实测 pre-roll 后 δ=-0.02s，裁出来 δ=-0.00s）。
# 修不动就保留原片段并记录（可见性优先），校验本身的任何异常都不让 clip 失败。
_PLACEMENT_SR = 8000                # 互相关采样率
_PLACEMENT_TOLERANCE = 0.15         # |δ| ≤ 该值视为对齐（≈2 帧 @60fps 余量）
_PLACEMENT_PROBE_SECONDS = 4.0
_PLACEMENT_MAX_LAG = 8.0            # 在"意图位置 ± 该值"范围内找峰值
_PLACEMENT_STABLE_SPREAD = 0.25     # 头/尾两次测量的最大分歧（超过即视为内部错乱）
_PLACEMENT_PREROLL = 8.0            # 修复用 pre-roll / post-roll
_PLACEMENT_MAX_SPAN = 180.0         # 参与测量的最大跨度（超长合并 clip 不必整段进 FFT）
_PLACEMENT_MIN_CORRELATION = 0.5    # 峰值质量门槛：低于它不修（宁可不动）
_PLACEMENT_PEAK_MARGIN = 0.92       # 最佳峰须明显高于次佳峰（挖掉 ±0.5s 后的）
_PLACEMENT_MIN_LEVEL = 40.0         # int16 RMS 下限：静音区间测不了
_PLACEMENT_CACHE_SPREAD = 1.5       # 缓存抽查：两个采样点允许的最大偏移分歧
_PLACEMENT_CACHE_KEY = "cache_placement"   # 缓存 sidecar 里的抽查结果字段
_CACHE_START_TOLERANCE = 0.05       # 片段缓存命中：起点允许的偏差
_PLACEMENT_DECODE_TIMEOUT = 120.0
_placement_records: list[dict[str, Any]] = []
_placement_lock = threading.Lock()


def reset_placement_records() -> None:
    """Clear the per-run placement log (called before materialize)."""
    with _placement_lock:
        _placement_records.clear()


def placement_records() -> list[dict[str, Any]]:
    """Per-clip placement measurements taken while fetching remote segments."""
    with _placement_lock:
        return list(_placement_records)


def _record_placement(record: dict[str, Any]) -> None:
    with _placement_lock:
        _placement_records.append(record)


def _placement_numpy():
    """numpy 是打包依赖，但只在真正做落点校验时才需要（惰性导入）。"""
    try:
        import numpy
    except Exception:                                    # pragma: no cover - 环境问题
        return None
    return numpy


def _placement_source_label(source) -> str:
    return f"{getattr(source, 'platform', None) or 'remote'}:" \
           f"{getattr(source, 'source_id', None) or '?'}"


def _cached_reference_path(source, cache_store) -> Path | None:
    """The cached detection audio for ``source`` when it is already on disk."""
    if cache_store is None:
        return None
    try:
        cached = resolve_cached_audio(source, cache_store)
    except Exception:                                    # noqa: BLE001
        return None
    if cached is None:
        return None
    path = Path(cached)
    return path if path.is_file() else None


def _decode_placement_pcm(path, start=None, duration=None):
    """Decode a window of a local file to mono int16 samples (None on failure).

    Uses a plain ``subprocess.run`` (like ``compile._decode_audio_region``): this
    is a short local decode that runs two or three times per clip, and
    ``run_tracked``'s watchdog polling costs ~0.3s per call — measurable at this
    frequency. Cancellation stays responsive because the network fetches and the
    repair cut (the long steps) all go through the tracked runners.
    """
    numpy = _placement_numpy()
    if numpy is None:
        return None
    command = [str(FFMPEG_PATH), "-v", "error"]
    if start is not None:
        command += ["-ss", f"{max(0.0, float(start)):.3f}"]
    if duration is not None:
        command += ["-t", f"{max(0.05, float(duration)):.3f}"]
    command += ["-i", str(path), "-vn", "-map", "0:a:0?", "-ac", "1",
                "-ar", str(int(_PLACEMENT_SR)), "-f", "s16le", "-"]
    try:
        result = subprocess.run(command, capture_output=True,
                                timeout=_PLACEMENT_DECODE_TIMEOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:                                    # noqa: BLE001
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    samples = numpy.frombuffer(result.stdout, dtype=numpy.int16).astype(numpy.float64)
    return samples if len(samples) else None


def _placement_level(samples) -> float:
    if samples is None or len(samples) == 0:
        return 0.0
    return math.sqrt(float((samples * samples).mean()))


def _placement_span(duration) -> float:
    """How much of the clip is measured.

    A merged/very long clip does not need its whole length in the correlation:
    the head and a second probe inside the measured span are enough to tell a
    uniform offset from an internally scrambled window, and capping the span
    keeps the reference window, the decode and the FFT bounded.
    """
    return max(0.0, min(float(duration), _PLACEMENT_MAX_SPAN))


def _slice_reference(samples, base, at, seconds):
    """Slice ``seconds`` of reference audio that starts at absolute time ``at``."""
    numpy = _placement_numpy()
    if numpy is None or samples is None:
        return None
    start = int(round((float(at) - float(base)) * _PLACEMENT_SR))
    length = int(round(float(seconds) * _PLACEMENT_SR))
    if start < 0 or length <= 0 or start + length > len(samples):
        return None
    return samples[start:start + length]


def _locate_probe(probe, region, region_base, intended, max_lag=_PLACEMENT_MAX_LAG):
    """Locate ``probe`` inside ``region`` near absolute time ``intended``.

    Returns ``(correlation, measured_seconds)`` or None when the match is too
    weak or ambiguous to trust. Per-lag energy normalisation is required: with a
    global normalisation a short probe against a long region peaks at about
    sqrt(len(probe)/len(region)) no matter where it actually belongs, which is
    how an earlier investigation wrongly concluded "the content is uncorrelated".
    """
    numpy = _placement_numpy()
    if numpy is None or probe is None or region is None:
        return None
    count = int(len(probe))
    if count < _PLACEMENT_SR // 2 or len(region) < count:
        return None
    if _placement_level(probe) < _PLACEMENT_MIN_LEVEL:
        return None
    lag = int(round(max(0.0, float(max_lag)) * _PLACEMENT_SR))
    center = int(round((float(intended) - float(region_base)) * _PLACEMENT_SR))
    low = max(0, center - lag)
    high = min(len(region) - count, center + lag)
    if high <= low:
        return None
    window = region[low:high + count]
    base = float(region_base) + low / _PLACEMENT_SR
    centered_probe = probe - probe.mean()
    centered = window - window.mean()
    energy = float((centered_probe ** 2).sum())
    if energy <= 0:
        return None
    size = 1 << int(numpy.ceil(numpy.log2(len(centered) + count)))
    cross = numpy.fft.irfft(
        numpy.fft.rfft(centered, size) * numpy.conj(numpy.fft.rfft(centered_probe, size)),
        size)[:len(centered) - count + 1]
    cumulative = numpy.concatenate(([0.0], numpy.cumsum(centered ** 2)))
    denominator = numpy.sqrt((cumulative[count:] - cumulative[:-count]) * energy)
    values = numpy.divide(cross, denominator, out=numpy.zeros_like(cross),
                          where=denominator > 0)
    index = int(numpy.argmax(values))
    peak = float(values[index])
    if peak < _PLACEMENT_MIN_CORRELATION:
        return None
    # 歧义检查：把最佳峰 ±0.5s 挖掉后，次佳峰不能接近最佳峰（音乐/重复段落会
    # 出现多个等高峰，此时任何"纠正"都是猜）。
    guard = max(1, int(0.5 * _PLACEMENT_SR))
    others = numpy.concatenate((values[:max(0, index - guard)],
                                values[min(len(values), index + guard + 1):]))
    if len(others) and float(others.max()) > peak * _PLACEMENT_PEAK_MARGIN:
        return None
    return peak, base + index / _PLACEMENT_SR


class _PlacementReference:
    """Audio of the detection timeline, used to locate a clip's real content.

    Audio Cache batches detect on the cached file, so that file is the authority
    (a local full download: no seek involved, absolutely trustworthy). Remote
    Stream batches detect on the live audio-only rendition, so a small window of
    it is fetched — a few hundred KB, ~1% of the video clip it validates.
    """

    def __init__(self, source, mode, cache_store, scratch_dir, logger=None):
        self.source = source
        self.mode = str(mode or "auto")
        self.cache_store = cache_store
        self.scratch_dir = Path(scratch_dir) if scratch_dir else Path(tempfile.gettempdir())
        self.logger = logger
        self.cached = (_cached_reference_path(source, cache_store)
                       if self.mode in ("cache", "auto") else None)
        self._windows: dict[tuple[float, float], Any] = {}

    @property
    def available(self) -> bool:
        if self.cached is not None:
            return True
        # Audio Cache 批次的时间戳定义在缓存音频上：缓存不在时绝不能用线上流当
        # 参照（两者本来就可能差好几秒，那会把 clip 对齐到错误的时间轴），
        # 而是放弃校验、保留原片段。Remote Stream / auto 才允许现取线上音频。
        if self.mode == "cache":
            return False
        return bool(getattr(self.source, "audio_url", None))

    @property
    def absolute(self) -> bool:
        """True when the reference needs no self-check (a local full download)."""
        return self.cached is not None

    def read(self, start, seconds):
        key = (round(float(start), 2), round(float(seconds), 2))
        if key not in self._windows:
            self._windows[key] = self._read(start, seconds)
        return self._windows[key]

    def _read(self, start, seconds):
        start = max(0.0, float(start))
        seconds = float(seconds)
        if seconds <= 0:
            return None
        if self.cached is not None:
            samples = _decode_placement_pcm(self.cached, start, seconds)
            if samples is not None:
                return samples, start
            return None
        if self.mode == "cache" or not getattr(self.source, "audio_url", None):
            return None
        try:
            handle, name = tempfile.mkstemp(prefix=".placement-ref-", suffix=".wav",
                                            dir=str(self.scratch_dir))
            os.close(handle)
        except OSError:
            return None
        window_path = Path(name)
        try:
            fetch_segment(self.source, start, start + seconds, window_path,
                          timeout=None, retries=1, audio_only=True,
                          codec="pcm_s16le", verify_placement=False,
                          allow_covering_cache=False, logger=None)
            samples = _decode_placement_pcm(window_path)
        except InterruptedError:
            raise
        except Exception:                                # noqa: BLE001
            return None
        finally:
            try:
                window_path.unlink(missing_ok=True)
            except OSError:
                pass
        if samples is None or len(samples) == 0:
            return None
        return samples, start


def measure_clip_placement(path, nominal_start, duration, read_reference):
    """Measure where a local clip file's content actually sits on the timeline.

    ``nominal_start`` is the media time the file's first sample is supposed to
    hold. Returns ``{"delta", "head", "tail", "spread", "stable",
    "correlation", "probes"}`` in seconds — ``delta`` > 0 means the content sits
    that much LATER than intended (the beginning of the moment got cut off) —
    or None when it cannot be measured (silence, no reference, weak/ambiguous
    match, too short).
    """
    if read_reference is None:
        return None
    duration = float(duration)
    span = _placement_span(duration)
    if span < 2.0:
        return None
    probe_seconds = min(_PLACEMENT_PROBE_SECONDS, max(1.0, span * 0.4))
    head_at = 0.2
    tail_at = max(head_at, span - probe_seconds - 0.2)
    head = _decode_placement_pcm(path, head_at, probe_seconds)
    if head is None or len(head) < _PLACEMENT_SR // 2:
        return None
    tail = _decode_placement_pcm(path, tail_at, probe_seconds) \
        if tail_at > head_at + 0.5 else None
    span_start = max(0.0, float(nominal_start) - _PLACEMENT_MAX_LAG)
    span_seconds = (float(nominal_start) + span + _PLACEMENT_MAX_LAG) - span_start
    reference = read_reference(span_start, span_seconds)
    if reference is None:
        return None
    samples, base = reference
    found = []
    for name, samples_probe, at in (("head", head, float(nominal_start) + head_at),
                                    ("tail", tail, float(nominal_start) + tail_at)):
        if samples_probe is None:
            continue
        hit = _locate_probe(samples_probe, samples, base, at)
        if hit is not None:
            found.append((name, hit[0], hit[1] - at))
    if not found:
        return None
    head_delta = next((delta for name, _corr, delta in found if name == "head"), None)
    tail_delta = next((delta for name, _corr, delta in found if name == "tail"), None)
    deltas = [delta for _name, _corr, delta in found]
    spread = (max(deltas) - min(deltas)) if len(deltas) > 1 else 0.0
    # 只有尾部探针可测时不算"一致"：便宜的做法（改请求区间）动的是整段内容，
    # 而尾部偏移并不代表开头（开头可能是淡入/静音，量不出来）。头部单点可测
    # 就够了——落点偏移本来就是按开头定义的。
    stable = spread <= _PLACEMENT_STABLE_SPREAD and (len(found) > 1 or head_delta is not None)
    return {"delta": sum(deltas) / len(deltas),
            "head": head_delta, "tail": tail_delta, "spread": spread,
            "stable": stable,
            "correlation": max(corr for _name, corr, _delta in found),
            "probes": len(found)}


def _placement_reference_stable(reference, nominal_start, duration) -> bool:
    """Is the reference's own timeline self-consistent?

    Only Remote Stream mode needs this: there the reference is a freshly fetched
    window of the same rendition family the clip comes from, so a rendition that
    misplaces content on seek could bias the measurement (and "correcting" a
    good clip is worse than doing nothing). Re-read the same span with a
    pre-roll — a different window plan — and require the overlap to agree.
    """
    span_start = max(0.0, float(nominal_start) - _PLACEMENT_MAX_LAG)
    span_seconds = ((float(nominal_start) + _placement_span(duration)
                     + _PLACEMENT_MAX_LAG) - span_start)
    first = reference.read(span_start, span_seconds)
    if first is None:
        return False
    shifted = reference.read(max(0.0, span_start - _PLACEMENT_PREROLL),
                             span_seconds + _PLACEMENT_PREROLL)
    if shifted is None:
        return False
    probe_at = span_start + min(3.0, span_seconds / 3.0)
    probe = _slice_reference(first[0], first[1], probe_at, _PLACEMENT_PROBE_SECONDS)
    if probe is None:
        return False
    hit = _locate_probe(probe, shifted[0], shifted[1], probe_at)
    if hit is None:
        return False
    return abs(hit[1] - probe_at) <= _PLACEMENT_TOLERANCE


def _placement_fetch(source, fetch_start, fetch_end, destination, *, timeout,
                     stall_timeout, progress_callback, logger, refresh_func=None,
                     audio_only=False, codec=None) -> bool:
    """Fetch one repair window into ``destination`` (False when it failed).

    Written to its own path so the already-fetched clip stays usable: a repair
    that fails must leave the original segment untouched. Cache is bypassed on
    purpose — a repair must read the stream, not the very data being checked.
    ``timeout`` is normally None so the fetch sizes its own budget from the
    repair window (the pre-roll window is bigger than the clip it repairs).
    """
    if fetch_start < 0 or fetch_end <= fetch_start:
        return False
    destination = Path(destination)
    destination.unlink(missing_ok=True)
    try:
        fetch_segment(source, float(fetch_start), float(fetch_end), destination,
                      cache_store=None, timeout=timeout, retries=1, logger=logger,
                      progress_callback=progress_callback,
                      stall_timeout=stall_timeout, verify_placement=False,
                      allow_covering_cache=False, refresh_func=refresh_func,
                      audio_only=bool(audio_only), codec=codec)
    except InterruptedError:
        raise
    except Exception as exc:                             # noqa: BLE001
        if logger is not None:
            logger(f"Clip placement repair fetch failed "
                   f"({fetch_start:g}-{fetch_end:g}s): {_sanitize_ffmpeg_detail(exc)}")
        return False
    return destination.is_file() and destination.stat().st_size > 0


def _placement_cut(window_file, offset, duration, output_file, *, has_audio,
                   timeout, logger=None, audio_only=False, codec=None) -> bool:
    """Cut [offset, offset+duration] out of a local window file (exact trim).

    Reuses the fMP4 window path's cut command, so a repaired clip is byte-for-byte
    the same kind of file as a normal one (compile cannot tell them apart).
    """
    command = build_local_window_command(
        window_file, 0.0, 0.0, float(offset), float(offset) + float(duration),
        output_file, audio_only=bool(audio_only), codec=codec,
        has_audio=bool(has_audio))
    try:
        result = run_tracked(command, timeout=timeout, text=True)
    except InterruptedError:
        raise
    except Exception as exc:                             # noqa: BLE001
        if logger is not None:
            logger(f"Clip placement cut failed: {_sanitize_ffmpeg_detail(exc)}")
        return False
    return_code = getattr(result, "returncode", 0)
    if return_code != 0:
        if logger is not None:
            logger(f"Clip placement cut failed (rc={return_code}): "
                   f"{_sanitize_ffmpeg_detail(getattr(result, 'stderr', '') or '')}")
        return False
    return Path(output_file).is_file() and Path(output_file).stat().st_size > 0


def cached_segment_is_aligned(source, path, nominal_start, duration, *, cache_store=None,
                              mode="auto", logger=None, scratch_dir=None) -> bool | None:
    """Is a cached segment's content still on the detected moment?

    True/False when it can be measured, None when it cannot (silence, a missing
    reference, a weak match) — in which case the caller keeps using the cache.
    ``scratch_dir`` should be a temp directory: for a stream reference the
    fetcher writes a window file, and the cache tree is not the place for it.
    """
    reference = _PlacementReference(source, mode, cache_store,
                                    scratch_dir or Path(path).parent, logger=logger)
    if not reference.available:
        return None
    measured = measure_clip_placement(path, nominal_start, duration, reference.read)
    if measured is None:
        return None
    return abs(float(measured["delta"])) <= _PLACEMENT_TOLERANCE


def cached_segment_span(path):
    """(nominal start, duration) a cached segment claims, or None if unknown.

    ``CacheStore.find_cached_segment`` may return a *covering* entry — a
    reverify/preview window that spans the requested clip — so the file's head is
    not necessarily the requested window's head. Measuring against the wrong
    origin would report a bogus offset and re-download a perfectly good clip, so
    the entry's own sidecar metadata is the only safe origin.
    """
    try:
        metadata = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
        start = float(metadata["start"])
        end = float(metadata["end"])
    except Exception:                                    # noqa: BLE001
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end - start < 2.0:
        return None
    return start, end - start


def correct_clip_placement(source, path, nominal_start, duration, *, cache_store=None,
                           mode="auto", logger=None, timeout=None, stall_timeout=30,
                           progress_callback=None, refresh_func=None,
                           has_audio=None, audio_only=False, codec=None) -> dict[str, Any]:
    """Measure a fetched clip's content position and repair it when it drifted.

    Always returns a record (also appended to ``placement_records()`` for the
    run summary) and never raises for a repair problem: the calling fetch is
    already successful, a clip with imperfect placement beats no clip at all.
    """
    path = Path(path)
    record: dict[str, Any] = {
        "name": _placement_source_label(source),
        "start": float(nominal_start),
        "end": float(nominal_start) + float(duration),
        "delta": None, "head": None, "tail": None, "correlation": None,
        "action": "unmeasurable", "corrected": False, "mislocated": False,
        "delta_before": None, "reason": "",
    }
    reference = _PlacementReference(source, mode, cache_store, path.parent, logger=logger)
    if not reference.available:
        record["reason"] = "no-reference"
        _record_placement(record)
        return record
    measured = measure_clip_placement(path, nominal_start, duration, reference.read)
    if measured is None:
        record["reason"] = "unmeasurable"
        _record_placement(record)
        return record
    record.update({key: measured[key] for key in
                   ("delta", "head", "tail", "correlation", "spread")})
    # 修复成功后 delta 会被替换成"修复后"的值；原始偏移单独留一份给汇总。
    record["delta_before"] = record["delta"]
    delta = float(measured["delta"])
    if abs(delta) <= _PLACEMENT_TOLERANCE:
        record["action"] = "aligned"
        _record_placement(record)
        return record
    record["mislocated"] = True
    if not measured["stable"]:
        # 头尾量出来的偏移不一致：窗口内部被拉伸/错乱，没有单一的平移量可修。
        record["action"] = "unstable"
        _record_placement(record)
        return record
    if not reference.absolute and not _placement_reference_stable(
            reference, float(nominal_start), float(duration)):
        record["action"] = "reference-unstable"
        _record_placement(record)
        return record
    if logger is not None:
        logger(f"Clip {record['start']:g}-{record['end']:g}s content sits "
               f"{delta:+.2f}s off the detection timeline "
               f"(correlation {measured['correlation']:.2f}); re-aligning")
    # 修复的代价与 clip 长度成正比（重取同长度窗口，pre-roll 那条再多 16s）：这是
    # 把整段内容搬到正确位置的必要成本，且只有偏移的 clip 才会付。每次修复都用
    # 自己的自适应 timeout 取，取不到就退回"保留原片段 + 汇报"，不会把 clip 弄丢。
    if has_audio is None:
        has_audio = source_has_embedded_audio(source)
    # 候选文件名必须保留原扩展名：裁剪命令的输出格式由扩展名决定（.mp4/.m4a/.wav）。
    candidate = path.with_name(f"{path.stem}.placement{path.suffix}")
    duration = float(duration)
    # 修法 1：整体平移请求区间。Bilibili Audio Cache 那种"缓存与流整体差 X 秒"
    # 一次就够（实测请求 +3.25s 后内容正好落在目标上）。
    if _placement_fetch(source, float(nominal_start) - delta,
                        float(nominal_start) - delta + duration, candidate,
                        timeout=None, stall_timeout=stall_timeout,
                        progress_callback=progress_callback, logger=logger,
                        refresh_func=refresh_func, audio_only=audio_only, codec=codec):
        re_measured = measure_clip_placement(candidate, nominal_start, duration,
                                             reference.read)
        if re_measured is not None and abs(re_measured["delta"]) <= _PLACEMENT_TOLERANCE:
            os.replace(candidate, path)
            record.update({"action": "shifted", "corrected": True,
                           "delta": re_measured["delta"], "head": re_measured["head"],
                           "tail": re_measured["tail"],
                           "correlation": re_measured["correlation"]})
            _record_placement(record)
            return record
        candidate.unlink(missing_ok=True)
    # 修法 2：带 pre-roll 重取，按量出来的偏移在本地裁。Twitch 那种"某个请求
    # 位置落进坏区间"的偏移（±1-2s 的窄带）平移请求救不了，pre-roll 能绕开。
    pre = _PLACEMENT_PREROLL
    pre_start = max(0.0, float(nominal_start) - pre)
    pre_end = float(nominal_start) + duration + pre
    if _placement_fetch(source, pre_start, pre_end, candidate, timeout=None,
                        stall_timeout=stall_timeout,
                        progress_callback=progress_callback, logger=logger,
                        refresh_func=refresh_func, audio_only=audio_only, codec=codec):
        pre_duration = _segment_duration(candidate) or (pre_end - pre_start)
        inside = measure_clip_placement(candidate, pre_start, pre_duration, reference.read)
        # 窗口开头的静音会让 head 探针过不了 level 门限（head=None → stable=False），
        # 但窗口本身可能已经完全对齐：Typicatly 那 19 段就是卡在这里——pre-roll
        # 窗口每个可测点都在 −0.021s，按名义偏移裁出来就是对的，却因为 head 不可测
        # 放弃了整个修法 2。所以"head 不可测、其余探针一致对齐"时按窗口原点处理
        # （head 视为 0.0），照常本地裁；裁完仍要重测通过，猜错不会被留下。
        usable_head = None if inside is None else inside["head"]
        if (usable_head is None and inside is not None and inside["tail"] is not None
                and abs(float(inside["delta"])) <= _PLACEMENT_TOLERANCE):
            if logger is not None:
                logger("Pre-roll window's head probe is silent but the rest of the "
                       "window is aligned; cutting it at the nominal offset")
            usable_head = 0.0
        if (inside is not None and usable_head is not None
                and (inside["stable"] or usable_head == 0.0)):
            # 窗口内"意图起点"的位置：窗口自身的落点偏移是 head，窗口里偏移 o 的
            # 内容对应的媒体时间 = pre_start + o + head，令它等于意图起点即可。
            # （实测 Twitch：pre_start=15495.40、head=-0.02 → 偏移 6.02s。）
            offset = (float(nominal_start) - pre_start) - float(usable_head)
            if offset >= 0.0 and offset + duration <= pre_duration + _PLACEMENT_TOLERANCE:
                cut = path.with_name(f"{path.stem}.cut{path.suffix}")
                cut.unlink(missing_ok=True)
                if _placement_cut(candidate, offset, duration, cut, has_audio=has_audio,
                                  timeout=timeout, logger=logger, audio_only=audio_only,
                                  codec=codec):
                    final = measure_clip_placement(cut, nominal_start, duration,
                                                   reference.read)
                    if (final is not None
                            and abs(final["delta"]) <= _PLACEMENT_TOLERANCE):
                        os.replace(cut, path)
                        record.update({"action": "preroll", "corrected": True,
                                       "offset": offset, "delta": final["delta"],
                                       "head": final["head"], "tail": final["tail"],
                                       "correlation": final["correlation"]})
                        _record_placement(record)
                        return record
                cut.unlink(missing_ok=True)
        candidate.unlink(missing_ok=True)
    record["action"] = "unfixed"
    if logger is not None:
        logger(f"Clip {record['start']:g}-{record['end']:g}s could not be re-aligned "
               f"({delta:+.2f}s off); keeping it as fetched")
    _record_placement(record)
    return record


def _placement_stream_self_consistent(reference, position, probe_seconds) -> bool:
    """Is the live reference's own timeline self-consistent at ``position``?

    Second opinion before telling the user their cached audio is out of sync:
    re-read the same span with a pre-roll (a different window plan) and require
    the overlap to agree. A rendition whose seek is unreliable at that position
    would otherwise produce a false "your cache drifted" warning.
    """
    first = reference.read(max(0.0, float(position) - _PLACEMENT_MAX_LAG),
                           float(probe_seconds) + 2 * _PLACEMENT_MAX_LAG)
    if first is None:
        return False
    probe = _slice_reference(first[0], first[1], float(position), float(probe_seconds))
    if probe is None:
        return False
    shifted = reference.read(
        max(0.0, float(position) - _PLACEMENT_MAX_LAG - _PLACEMENT_PREROLL),
        float(probe_seconds) + 2 * _PLACEMENT_MAX_LAG + _PLACEMENT_PREROLL)
    if shifted is None:
        return False
    hit = _locate_probe(probe, shifted[0], shifted[1], float(position))
    return hit is not None and abs(hit[1] - float(position)) <= _PLACEMENT_TOLERANCE


def _cache_drift_lines(source, path, at, delta, checked, spread=None) -> list[str]:
    """The warning shown when a cached detection audio is off the stream.

    Deliberately hedged about the cause: a cache↔stream mismatch means either the
    cached download has duplicated/missing audio (re-download it) or the
    rendition's own seek is unreliable at that position (re-downloading changes
    nothing). The clips are re-aligned individually either way.
    """
    varies = ""
    if spread is not None and spread > _PLACEMENT_CACHE_SPREAD:
        varies = (f" The offset varies by {spread:.1f}s between the sampled "
                  f"positions, which points at duplicated/missing audio inside the "
                  f"cached download rather than a single constant shift.")
    return [
        f"Cached audio for {_placement_source_label(source)} disagrees with the live "
        f"stream by up to {delta:+.2f}s (worst at {at / 60:.0f}min, "
        f"{checked} position(s) checked).{varies}",
        f"Clips from this source are re-aligned individually while fetching, so they "
        f"still land on the detected moment. If the exported timestamps look shifted "
        f"when you check them against the VOD, delete {path.name} from the remote "
        f"cache so it is downloaded again; if they look right, this is the stream's "
        f"own seek rather than your cache and nothing needs to be done.",
    ]


def _load_cache_check(cache_path) -> dict[str, Any] | None:
    """The previous verification of this exact cache file, if it still applies.

    The Audio Cache pre-pass is deliberately probe-free on cache hits (probing
    every source tripped bilibili's CDN rate limit before). The result is stored
    in the cache's own sidecar metadata keyed by size+mtime, so a source is
    checked once per downloaded file: repeat runs cost no requests, a re-download
    invalidates the record, and a known drift is re-reported for free.
    """
    marker = Path(cache_path).with_suffix(".json")
    try:
        record = json.loads(marker.read_text(encoding="utf-8")).get(_PLACEMENT_CACHE_KEY)
        stat = Path(cache_path).stat()
    except Exception:                                    # noqa: BLE001
        return None
    if not isinstance(record, dict):
        return None
    try:
        if int(record.get("size", -1)) != int(stat.st_size):
            return None
        if abs(float(record.get("mtime", 0.0)) - stat.st_mtime) > 1.0:
            return None
    except (TypeError, ValueError):
        return None
    return record


def _save_cache_check(cache_path, *, checked, drift=None, at=None, spread=None) -> None:
    """Persist a verification result next to the cache (never creates the file)."""
    marker = Path(cache_path).with_suffix(".json")
    temporary = marker.with_name(marker.name + ".placement.tmp")
    try:
        if not marker.is_file():
            return
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return
        stat = Path(cache_path).stat()
        metadata[_PLACEMENT_CACHE_KEY] = {
            "size": int(stat.st_size), "mtime": float(stat.st_mtime),
            "checked": int(checked), "drift": None if drift is None else float(drift),
            "at": None if at is None else float(at),
            "spread": None if spread is None else float(spread),
        }
        temporary.write_text(json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                             encoding="utf-8")
        os.replace(temporary, marker)
    except Exception:                                    # noqa: BLE001
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def verify_cached_audio_alignment(source, cache_path, *, cache_store=None,
                                  samples=2, probe_seconds=6.0, logger=None,
                                  scratch_dir=None, refresh=False) -> list[str]:
    """Spot-check that a cached detection audio still matches the live stream.

    Audio Cache batches detect on the cached file and compile from the stream, so
    a cache that drifted from the stream shifts every timestamp it produced (the
    per-clip placement check still re-aligns the clips, but the user should know
    the cache is off, because re-downloading it is the real fix). Returns
    printable lines; an empty list means "checked and aligned" or "not checked".

    Each cache file is verified once: the result is stored in its sidecar
    metadata (see ``_load_cache_check``) and replayed from there afterwards, so a
    repeat run neither re-probes the CDN nor loses a known warning.
    """
    path = Path(cache_path)
    if not path.is_file() or not getattr(source, "audio_url", None):
        return []
    if not refresh:
        previous = _load_cache_check(path)
        if previous is not None:
            if previous.get("drift") is None:
                return []
            return _cache_drift_lines(source, path, float(previous.get("at") or 0.0),
                                      float(previous["drift"]),
                                      int(previous.get("checked") or 0),
                                      spread=previous.get("spread"))
    duration = _segment_duration(path)
    if not duration or duration < 300.0:
        return []
    reference = _PlacementReference(source, "stream", cache_store,
                                    scratch_dir or Path(tempfile.gettempdir()),
                                    logger=logger)
    if not reference.available:
        return []
    margin = 30.0
    count = max(1, int(samples))
    span = duration - 2 * margin
    positions = [margin + span * (index + 1) / (count + 1) for index in range(count)]
    checked, deltas, worst = 0, [], None
    for position in positions:
        probe = _decode_placement_pcm(path, position, probe_seconds)
        if probe is None:
            continue
        window = reference.read(max(0.0, position - _PLACEMENT_MAX_LAG),
                                probe_seconds + 2 * _PLACEMENT_MAX_LAG)
        if window is None:
            continue
        hit = _locate_probe(probe, window[0], window[1], position)
        if hit is None:
            continue
        checked += 1
        delta = hit[1] - position
        deltas.append(delta)
        if worst is None or abs(delta) > abs(worst[1]):
            worst = (position, delta, hit[0])
    if not checked:
        return []
    if all(abs(delta) <= _PLACEMENT_TOLERANCE for delta in deltas):
        _save_cache_check(path, checked=checked)
        return []
    worst_at, worst_delta, _worst_corr = worst
    spread = (max(deltas) - min(deltas)) if len(deltas) > 1 else 0.0
    # 报之前先确认这不是"线上流自己 seek 抖动"：多个采样点一致（缓存整体错位就是
    # 这样）就够了；采样点之间分歧较大时（缓存可能中间有重复/缺失音频），必须再
    # 确认最坏那一点上流自身是自洽的。证据不足就不报——误报会让用户白重下几十 GB。
    if spread > _PLACEMENT_CACHE_SPREAD and not _placement_stream_self_consistent(
            reference, worst_at, probe_seconds):
        return []
    _save_cache_check(path, checked=checked, drift=worst_delta, at=worst_at,
                      spread=spread)
    lines = _cache_drift_lines(source, path, worst_at, worst_delta, checked,
                               spread=spread)
    if logger is not None:
        for line in lines:
            logger(line)
    return lines


def estimate_segment_fetch_timeout(source, duration, explicit=None) -> float:
    """Per-clip fetch timeout derived from the clip's expected download size.

    以前每次尝试固定 600s：一个 5 秒的 clip 卡在网络里也要独占 10 分钟。现在
    按 "预计字节数 / 可接受的最低持续吞吐" 计算，这样慢但稳定的线路（≥0.15 MB/s）
    依然来得及下完，只有真正没进展的才会更早被放弃。要看更早的失败由
    progress-stall 看门狗（120s 无进展）负责。显式传入的 timeout 优先。
    """
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    per_second = _DEFAULT_BYTES_PER_SECOND
    height = None
    try:
        value = int(getattr(source, "max_height", None) or 0)
        height = value if value > 0 else None
    except (TypeError, ValueError):
        height = None
    if height is None:
        for candidate in (getattr(source, "video_candidates", None) or []):
            try:
                value = int(candidate.get("height") or 0)
            except (TypeError, ValueError, AttributeError):
                value = 0
            if value > 0:
                height = value
                break
    if height:
        for limit, bytes_per_second in _QUALITY_BYTES_PER_SECOND:
            if height <= limit:
                per_second = bytes_per_second
                break
        else:
            per_second = 800 * 1024
    expected_bytes = max(1.0, float(duration)) * per_second
    return min(_SEGMENT_TIMEOUT_CAP,
               max(_SEGMENT_TIMEOUT_FLOOR, expected_bytes / _SEGMENT_MIN_THROUGHPUT))


def fetch_segment(
    source: MediaSource,
    start: float,
    end: float,
    output_file: str | Path,
    cache_store: Any | None = None,
    padding_before: float = 0,
    padding_after: float = 0,
    timeout: float | None = None,
    retries: int = 2,
    run_func: Callable[..., Any] | None = None,
    audio_only: bool = False,
    codec: str | None = None,
    allow_covering_cache: bool = True,
    refresh_func: Callable[[MediaSource], MediaSource | None] | None = None,
    logger: Callable[[str], Any] | None = None,
    progress_callback: Callable[..., Any] | None = None,
    stall_timeout: float = 30,
    max_total_duration: float = 0,
    verify_placement: bool = True,
    placement_reference: str = "auto",
) -> Path:
    """Fetch a requested remote interval, optionally reusing covering cache.

    ``timeout`` bounds ONE ffmpeg attempt. When omitted it is derived from the
    clip's expected size (see ``estimate_segment_fetch_timeout``) instead of the
    old flat 600s.

    ``max_total_duration`` optionally bounds the WHOLE segment fetch (all
    retries, source refreshes and backoff sleeps). It defaults to 0 (unbounded):
    a clip whose download is merely slow (but steadily producing data) must be
    allowed to finish rather than being killed by an arbitrary wall-clock
    budget — the per-attempt ``stall_timeout`` already abandons a genuinely
    hung connection (no data). Only a caller with a specific reason may pass a
    positive budget.

    ``verify_placement`` measures the fetched clip's real content position and
    repairs it when the rendition handed back a shifted window (see the
    placement block above); ``placement_reference`` selects what "the detection
    timeline" is: ``"cache"`` (Audio Cache batches: the cached file detection ran
    on), ``"stream"`` (Remote Stream batches: the live audio-only rendition) or
    ``"auto"`` (prefer the cache). Repair fetches pass ``verify_placement=False``
    so a check never recurses into itself.
    """
    if padding_before < 0 or padding_after < 0:
        raise ValueError("segment padding must not be negative")
    if retries < 0:
        raise ValueError("retries must not be negative")
    expected_seconds = ((float(end) + float(padding_after))
                        - max(0.0, float(start) - float(padding_before)))
    timeout = estimate_segment_fetch_timeout(source, expected_seconds, explicit=timeout)
    _fetch_started = time.monotonic()

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
                # 命中条目可能比请求区间宽：`find_cached_segment` 是"覆盖即命中"
                # （上次跑用了更大的 padding、或旧版本的扩展窗口都会产生这种条目），
                # 而 materialize 交给 compile 的语义是"这个文件就是整个 clip"
                # （时间戳 [0, duration]）——从更宽的条目里按 clip 时长切，内容就会
                # 从条目自己的起点开始，整体错位。所以只接受"起点一致"的命中
                # （尾部多余无所谓：compile 按 clip 时长裁掉）。
                requested_start = max(0.0, float(start) - float(padding_before))
                span = cached_segment_span(cached)
                starts_exact = (span is not None
                                and abs(span[0] - requested_start)
                                <= _CACHE_START_TOLERANCE)
                if not starts_exact:
                    if logger is not None:
                        logger("Cached clip starts at a different position than "
                               "requested; re-fetching the exact window")
                elif (verify_placement and run_func is None
                        and not (max_total_duration and max_total_duration > 0)
                        and _fetch_has_audio(source, audio_only)):
                    aligned = cached_segment_is_aligned(
                        source, cached, span[0], span[1], cache_store=cache_store,
                        mode=placement_reference, logger=logger,
                        scratch_dir=Path(output_file).parent)
                    if aligned is False:
                        if logger is not None:
                            logger("Cached clip content is off the detected moment; "
                                   "re-fetching it instead of reusing the cache")
                    else:
                        return Path(cached)
                else:
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
            # "Preparing clips"（materialize 下载）阶段的卡死：ffmpeg 仍在按
            # 0.5s 打 progress、out_time 却不动时，原有"30s 无输出"看门狗永远
            # 不触发，界面就停在 "Preparing clips: n/N · ETA 00:00"（测试者反馈）。
            # 进度不前进超时就抛 ProgressStallTimeout（TimeoutExpired 子类，
            # 走既有的 retry/refresh 阶梯）。
            progress_stall_timeout=_SEGMENT_PROGRESS_STALL_TIMEOUT,
            # 心跳只在进度停滞 ≥30s 后每 30s 打一条（正常的慢下载不再刷屏），
            # 并附一条成因提示（并发过高 / 线路不稳），让用户知道不是卡死。
            heartbeat_label=(f"{source.platform or 'remote'}:"
                             f"{source.source_id or '?'} {start:g}-{end:g}s"),
            heartbeat_interval=_SEGMENT_HEARTBEAT_INTERVAL,
            heartbeat_verb="fetching",
            heartbeat_stall_threshold=_SEGMENT_HEARTBEAT_STALL_THRESHOLD,
            heartbeat_hint=_SEGMENT_HEARTBEAT_HINT,
        )
    last_error: Exception | None = None
    refreshed = False
    attempt = 0
    allowed_attempts = int(retries) + 1
    # 这一轮交付的片段是否"短但可用"（用于跳过缓存写入，见下）。
    short_delivery = False
    # 持续失败预算：只针对"反复失败无进展"的片段。慢速但稳定产出的下载
    # 有数据（stall 不触发）不会被误杀；连续失败累计超过该秒数则放弃，
    # 避免 materialize 卡在单个坏片段上无限 refresh 探测。
    _fail_budget_started: float | None = None
    while attempt < allowed_attempts:
        if (max_total_duration and max_total_duration > 0
                and time.monotonic() - _fetch_started > max_total_duration):
            raise SegmentFetchError(
                f"Segment fetch for {start}-{end} exceeded "
                f"{max_total_duration:g}s budget; giving up on this clip "
                f"(the VOD stream for this interval may be unavailable).")
        try:
            fetch_start = max(0.0, float(start) - float(padding_before))
            fetch_end = float(end) + float(padding_after)
            # fMP4 HLS（Twitch 新版切片）：ffmpeg 的网络 seek 在这种 playlist 上
            # 会退化成逐片顺序读取、吐不出帧，所以改成"下载窗口分片 + 本地裁剪"。
            # 任何准备阶段的失败都退回原来的网络命令（不能把本来能用的源搞坏）。
            hls_window = None
            if run_func is None:
                embedded_audio = source_has_embedded_audio(source)
                if audio_only:
                    window_url = source.audio_url
                    window_headers = source.audio_headers or source.http_headers
                    window_has_audio = True
                elif source.video_url and (embedded_audio or not source.audio_url):
                    # 分片窗口只覆盖一个 playlist：音视频分开的源（视频轨无音轨 +
                    # 独立 audio_url）继续走网络命令，避免窗口里没有音轨。
                    window_url = source.video_url
                    window_headers = source.video_headers or source.http_headers
                    window_has_audio = embedded_audio
                else:
                    window_url = ""
                    window_headers = {}
                    window_has_audio = False
                if window_url:
                    plan = hls_window_plan(window_url, window_headers, fetch_start, fetch_end)
                    if plan is not None:
                        # 分片读取用固定的 60s/次读（单个分片几 MB，够宽松）；窗口
                        # remux 是整个窗口的本地活，用本 clip 的自适应 timeout
                        # （长合并 clip 的窗口可能几百 MB，60s 会误杀并触发整窗重下）。
                        hls_window = materialize_hls_window(
                            plan, Path(temporary_path).parent, remux_timeout=timeout)
            try:
                if hls_window is not None:
                    command = build_local_window_command(
                        hls_window.path, hls_window.file_start, hls_window.inner_seek,
                        fetch_start, fetch_end, temporary_path,
                        audio_only=audio_only, codec=codec,
                        has_audio=window_has_audio,
                    )
                else:
                    command = build_segment_command(
                        source,
                        fetch_start,
                        fetch_end,
                        temporary_path,
                        audio_only=audio_only,
                        codec=codec,
                    )
                result = run_command(command)
            finally:
                if hls_window is not None:
                    hls_window.cleanup()
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
                error = SegmentFetchError(
                    f"FFmpeg segment fetch produced an unreadable segment "
                    f"(no {'video' if not audio_only else 'audio'} stream): "
                    f"{temporary_path}"
                    + ("" if audio_only else
                       " (the VOD's video track may end before this clip's "
                       "start, or the video CDN edge is currently unreachable "
                       "from this network)"))
                if not audio_only:
                    error.no_video_stream = True
                raise error
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
                        and actual_duration > 0):
                    if actual_duration < expected_duration * 0.5:
                        raise SegmentFetchError(
                            f"FFmpeg segment fetch produced a truncated segment "
                            f"({actual_duration:g}s vs expected ~{expected_duration:g}s): "
                            f"{temporary_path}")
                    shortfall = expected_duration - actual_duration
                    # 有整段预算的调用（预览：max_total_duration>0）本来就不该
                    # 为一个窗口再抓一遍，也不该往 materialize 的短交付汇总里写
                    # 预览记录。
                    if (shortfall > _SHORT_SEGMENT_TOLERANCE and not audio_only
                            and not (max_total_duration and max_total_duration > 0)):
                        # 交付比请求短（但过了 50% 门槛）：以前静默接受，clip 会被
                        # 对齐裁到实际交付长度 → 结尾比 padding 预期早、切点生硬
                        # （测试者反馈）。先按失败重试（走 refresh 阶梯）；最后一次
                        # 尝试仍短就接受，宁可短一点也不丢整个 clip。
                        #
                        # 但"连续失败 90s 就放弃"的预算不能把短交付算进去：慢 CDN 上
                        # 两次 40-50s 的抓取就够触发，结果是"接受一个短 clip"变成
                        # "丢掉整个 clip"。预算已耗尽时直接走接受分支。
                        budget_exhausted = (
                            _fail_budget_started is not None
                            and time.monotonic() - _fail_budget_started > 90)
                        if attempt + 1 < allowed_attempts and not budget_exhausted:
                            raise SegmentFetchError(
                                f"FFmpeg segment delivered {actual_duration:g}s of the "
                                f"requested {expected_duration:g}s "
                                f"({shortfall:g}s short); re-fetching")
                        _log_short_delivery(source, start, end, actual_duration,
                                            expected_duration)
                        short_delivery = True
            if (verify_placement and run_func is None and not short_delivery
                    and _fetch_has_audio(source, audio_only)
                    and not (max_total_duration and max_total_duration > 0)):
                # 落点校验：取回的内容是否真的落在请求的位置上（见 placement
                # 说明块）。只对有音轨的取回做（音频是唯一便宜的内容锚点）；
                # 任何异常都不影响已经成功的这次取回。
                try:
                    correct_clip_placement(
                        source, temporary_path, fetch_start,
                        float(end) + float(padding_after) - fetch_start,
                        cache_store=cache_store, mode=placement_reference,
                        logger=logger, timeout=timeout, stall_timeout=stall_timeout,
                        progress_callback=progress_callback, refresh_func=refresh_func,
                        audio_only=audio_only, codec=codec)
                except InterruptedError:
                    raise
                except Exception as exc:                     # noqa: BLE001
                    if logger is not None:
                        logger("Clip placement check skipped: "
                               f"{_sanitize_ffmpeg_detail(exc)}")
            data = temporary_path.read_bytes()
            if cache_store is not None and not short_delivery:
                # 短交付的片段不写缓存：缓存命中是按"声称覆盖了请求区间"判断的，
                # 存进去以后下一轮会直接命中，既不重抓也不报告，可见性只在第一轮
                # 存在（重新抓一次还会顺带刷新签名 URL）。
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
            _fail_budget_started = None
            return destination
        except InterruptedError:
            raise
        except Exception as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                # 这条超时是按 clip 体量算出来的（见 estimate_segment_fetch_timeout），
                # 不再把整条 ffmpeg 命令行灌进日志；直接说清楚"允许多久 + 可能原因"。
                last_error = SegmentFetchError(
                    f"Segment fetch for {start:g}-{end:g}s timed out after "
                    f"{timeout:g}s (this clip's size allows {timeout:g}s); the CDN "
                    f"delivered no usable data in time")
            else:
                last_error = exc if isinstance(exc, SegmentFetchError) else SegmentFetchError(
                    f"Could not fetch remote segment {start}-{end}: "
                    f"{_sanitize_ffmpeg_detail(exc)}"
                )
            # 持续失败预算：连续失败累计超 90s 仍无成功 → 放弃该片段。
            # 防止单个坏片段（VOD 区间流不可用）无限 refresh 探测，拖住
            # 整个 materialize（Addendum 17 移除总时长预算后无兜底）。
            if _fail_budget_started is None:
                _fail_budget_started = time.monotonic()
            elif time.monotonic() - _fail_budget_started > 90:
                raise SegmentFetchError(
                    f"Segment fetch for {start}-{end} kept failing for 90s; "
                    f"giving up on this clip (the VOD stream for this interval "
                    f"may be unavailable).")
            # 无视频流（音频正常）→ 大概率是当前 video_url 的 CDN 边缘对本
            # 网络不可用：轮换到下一个视频候选换线重试（不消耗 refresh 名额；
            # 刷新的重新解析可能选回同一个坏边缘）。
            if isinstance(exc, SegmentFetchError) and getattr(exc, "no_video_stream", False):
                if _rotate_video_candidate(source) and logger is not None:
                    logger("Remote segment produced no video stream; "
                           "retrying with the next video candidate")
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


def _fetch_has_audio(source: MediaSource, audio_only: bool) -> bool:
    """Will this segment fetch produce a file with an audio track?

    Audio is the only cheap content anchor, so this gates the placement check:
    an audio-only fetch always has it, a video rendition with embedded audio has
    it, and a video-only rendition plus a separate ``audio_url`` gets the audio
    muxed in as a second input (``build_segment_command`` maps ``1:a:0``) — which
    is exactly the Bilibili/YouTube shape whose cached audio drifts from the
    stream.
    """
    if audio_only:
        return True
    return source_has_embedded_audio(source) or bool(source.audio_url)


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
    url: str | None = None,
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
    # 注意：这里不预注入 youtube player_client（如 default,web_embedded）。
    # 实测预注入会让 web_embedded 客户端给出需要 PO token 的媒体 URL，
    # 元数据解析成功但下载全部 403（Audio Cache 卡 0 MB/s）。
    # tv_downgraded 坏客户端的 page-reload 错误由
    # _extract_with_cookie_policy 的匿名回退梯子处理。
    return options


def _make_ydl(
    ydl_factory: Callable[..., Any] | None,
    browser_cookies: str | None = None,
    extract_flat: bool = False,
    url: str | None = None,
):
    if ydl_factory is not None:
        return ydl_factory(_ydl_options(browser_cookies, extract_flat=extract_flat, url=url))
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
            url=url,
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


_YOUTUBE_HOST_MARKERS = ("youtube.com", "youtu.be", "youtube-nocookie.com")


def _is_youtube_url(url: str) -> bool:
    hostname = (urlsplit(url).hostname or "").lower()
    return any(hostname == marker or hostname.endswith("." + marker)
               for marker in _YOUTUBE_HOST_MARKERS)


def _is_http_412_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return "412" in detail or "precondition failed" in detail


# YouTube bot-check / Twitch 鉴权类错误：无 cookie 的重试永远不会成功，
# 带 cookie 重试通常直接解决——这是 YouTube "Metadata failed" 的最常见根因。
_AUTH_REQUIRED_ERROR_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "please sign in",
    "login required",
    "oauth",
    "authentication",
)


def _looks_like_auth_required_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return any(marker in detail for marker in _AUTH_REQUIRED_ERROR_MARKERS)


def _is_yt_page_reload_error(exc: Exception) -> bool:
    """yt-dlp #17389：带 cookie 时 tv_downgraded 客户端已坏的标志性错误。"""
    return "page needs to be reloaded" in str(exc).lower()


def _should_auto_use_cookies(url: str, exc: Exception) -> bool:
    if _is_bilibili_url(url) or _is_http_412_error(exc):
        return True
    return _looks_like_auth_required_error(exc)


def classify_resolve_failure(exc: Exception) -> tuple[str, str]:
    """Return (short_reason, actionable_hint) for a failed remote resolve.

    测试者报的"YouTube sources failing to resolve constantly"有两类完全不同的
    成因，日志里必须能一眼区分：
      - 解析太多被限流/机器人验证（等一会儿/cookies/降速就能恢复）；
      - yt-dlp 客户端被平台改坏（换新版包才能修，2026 年已发生两次：
        android_vr 403、tv_downgraded "page needs to be reloaded"）。
    """
    detail = str(exc)
    lowered = detail.lower()
    from remote_rate import is_throttling_error
    if isinstance(exc, LiveBroadcastError):
        return ("live broadcast (no replay yet)",
                "A live playlist only keeps the last few minutes, so clips found "
                "now could not be fetched later - AutoComper skipped this source "
                "instead of losing every clip. Wait until the stream ends and the "
                "replay is processed, then use the replay URL "
                "(e.g. twitch.tv/videos/<id>).")
    if "page needs to be reloaded" in lowered:
        return ("youtube page-reload error (yt-dlp client regression)",
                "Update AutoComper: this is fixed by the bundled yt-dlp "
                "(the tester build must be the latest package).")
    if is_throttling_error(exc):
        return ("rate limited / bot check",
                "AutoComper is slowing resolves down automatically; if it keeps "
                "failing, wait 10-30 minutes, or configure browser cookies / a "
                "cookies.txt in Remote Settings (signed-in requests get a much "
                "higher quota).")
    if _looks_like_auth_required_error(exc):
        return ("sign-in required",
                "Configure browser cookies or a cookies.txt file in Remote "
                "Settings.")
    for marker in ("unable to extract", "failed to extract any player response",
                   "nsig extraction failed", "player response",
                   "signature extraction failed"):
        if marker in lowered:
            return ("youtube extraction failed (yt-dlp may be outdated)",
                    "Update AutoComper to the newest package; if it already is, "
                    "this is a platform-side extractor break.")
    if "timed out" in lowered or "timeout" in lowered:
        return ("timed out",
                "The network path to the platform is slow/unstable; retry or "
                "use a different route.")
    if "unavailable" in lowered or "private" in lowered or "removed" in lowered:
        return ("unavailable",
                "The video is private, removed, region-blocked or members-only.")
    short = detail.strip().replace("\n", " ")
    if len(short) > 90:
        short = short[:87] + "..."
    return (short or exc.__class__.__name__, "")


def _classify_hydration_error(exc: Exception) -> str:
    """Short failure reason for a playlist-picker status cell (~110-240px 列宽)。
    操作性建议（如配置 cookies）放在 cookie 回退耗尽后的报错与 README，不塞窄列。"""
    detail = str(exc).strip().replace("\n", " ")
    lowered = detail.lower()
    if _looks_like_auth_required_error(exc):
        return "sign-in required"
    if _is_yt_page_reload_error(exc):
        return "youtube page reload error"
    if _is_http_412_error(exc) or "429" in lowered or "too many requests" in lowered:
        return "rate limited"
    if "timed out" in lowered or "timeout" in lowered:
        return "timed out"
    if "unavailable" in lowered or "private" in lowered or "removed" in lowered:
        return "unavailable"
    if len(detail) > 60:
        detail = detail[:57] + "..."
    return detail or exc.__class__.__name__


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
    resolve_timeout: float = 90,
) -> Mapping[str, Any]:
    normalized = _normalize_browser_cookies(browser_cookies)
    if normalized not in (None, "auto"):
        try:
            return _extract_info(url, ydl_factory, normalized, extract_flat=extract_flat,
                                 resolve_timeout=resolve_timeout)
        except SourceResolveError as exc:
            # yt-dlp #17389：YouTube 带 cookie 时 tv_downgraded 客户端对部分
            # 用户已坏。公开视频匿名解析正常（issue 原报告证实）——回退匿名
            # 重试一次；仍失败则抛原始错误（原因更明确）。
            if _is_youtube_url(url) and _is_yt_page_reload_error(exc):
                try:
                    return _extract_info(url, ydl_factory, None, extract_flat=extract_flat,
                                         resolve_timeout=resolve_timeout)
                except SourceResolveError:
                    raise exc
            raise

    try:
        return _extract_info(url, ydl_factory, extract_flat=extract_flat,
                             resolve_timeout=resolve_timeout)
    except SourceResolveError as initial_error:
        if not _should_auto_use_cookies(url, initial_error):
            raise
        failures = []
        for browser in _BROWSER_COOKIE_NAMES:
            try:
                return _extract_info(url, ydl_factory, browser, extract_flat=extract_flat,
                                     resolve_timeout=resolve_timeout)
            except SourceResolveError as exc:
                failures.append(_short_cookie_failure(browser, exc))
                if _cookie_failure_should_print(browser, url):
                    print(f"Remote browser cookies failed ({url}): {failures[-1]}")
        detail = "; ".join(failures)
        raise SourceResolveError(
            f"Could not resolve source {url}: {detail}. "
            "Sign-in appears to be required; configure browser cookies or a "
            "cookies.txt file in Remote Settings."
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


# B 站 P2P/边缘慢节点（参考社区共识）：被调度到这些 host 的链接通常远慢于
# upos 正式节点。黑名单只影响候选排序前的过滤，全部被过滤时保留原列表兜底。
_BILIBILI_SLOW_CDN_MARKERS = ("mcdn", "pcdn", "szbdyd.com")


def _is_slow_bilibili_cdn_host(host: str) -> bool:
    lowered = str(host or "").lower()
    # upos 官方快节点族（upcdn.bilivideo.com）子串里含 "pcdn"，不能误伤
    if "upcdn" in lowered:
        return False
    return any(marker in lowered for marker in _BILIBILI_SLOW_CDN_MARKERS)


def _expand_bilibili_audio_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for candidate in candidates:
        for index, url in enumerate(_bilibili_cdn_variants(candidate.get("url", ""))):
            item = dict(candidate)
            item["url"] = url
            item["cdn_variant_index"] = index
            item["cdn_host"] = urlsplit(url).netloc
            expanded.append(item)
    filtered = [item for item in expanded
                if not _is_slow_bilibili_cdn_host(item.get("cdn_host"))]
    return filtered if filtered else expanded


_BILIBILI_CDN_SUFFIXES = (".bilivideo.com", ".bilivideo.cn", ".akamaized.net",
                          ".szbdyd.com")
# Full Download 的主机选择策略：只有原节点实测"明显慢"、且流足够大（探测量相对
# 整段下载可以忽略）时才去测镜像，最多 3 个候选；任何异常都保留原 URL。
_FAST_HOST_SKIP_SPEED = 3.0
_FAST_HOST_MIN_STREAM = 32 * 1024 * 1024
_FAST_HOST_MAX_CANDIDATES = 3


def is_bilibili_cdn_host(host: str) -> bool:
    """True for a Bilibili media CDN host (mirror substitution is safe there)."""
    lowered = str(host or "").lower().split(":")[0]
    return bool(lowered) and any(lowered.endswith(suffix)
                                 for suffix in _BILIBILI_CDN_SUFFIXES)


def select_fastest_bilibili_url(url, headers=None, size=None, log_func=None,
                                measure_func=None, cancel_check=None):
    """Return the fastest reachable mirror for one Bilibili stream URL.

    All known mirrors serve the same signed path+query, so only the host changes
    (exactly what the Audio Cache candidate pool does for audio). The assigned
    node is kept unless it measures clearly slow, so the common case costs one
    bounded read and no extra requests.
    """
    original = str(url or "")
    parsed = urlsplit(original)
    if not is_bilibili_cdn_host(parsed.netloc):
        return original
    variants = _bilibili_cdn_variants(original)
    alternatives = [item for item in variants
                    if urlsplit(item).netloc != parsed.netloc
                    and not _is_slow_bilibili_cdn_host(urlsplit(item).netloc)]
    if not alternatives:
        return original
    if size is not None:
        try:
            if int(size) < _FAST_HOST_MIN_STREAM:
                return original
        except (TypeError, ValueError):
            pass
    if measure_func is None:
        from remote_prefetch import measure_stream_throughput as measure_func  # noqa: N813
    probe_bytes = 4 * 1024 * 1024
    if size:
        try:
            probe_bytes = max(1024 * 1024, min(probe_bytes, int(size) // 64))
        except (TypeError, ValueError):
            pass
    if callable(cancel_check) and cancel_check():
        return original
    try:
        measured = measure_func(original, dict(headers or {}), probe_bytes=probe_bytes)
    except Exception:
        return original
    if measured is not None and measured >= _FAST_HOST_SKIP_SPEED:
        return original
    best_url = original
    best_speed = measured if measured is not None else -1.0
    for candidate in alternatives[:_FAST_HOST_MAX_CANDIDATES]:
        if callable(cancel_check) and cancel_check():
            break
        try:
            speed = measure_func(candidate, dict(headers or {}),
                                 probe_bytes=probe_bytes)
        except Exception:
            speed = None
        if speed is not None and speed > best_speed:
            best_speed = speed
            best_url = candidate
    if best_url != original and log_func is not None:
        old_speed = f"{measured:.2f} MB/s" if measured is not None else "unmeasurable"
        log_func(f"Bilibili CDN: {urlsplit(best_url).netloc} ({best_speed:.2f} MB/s) "
                 f"chosen over {parsed.netloc} ({old_speed})")
    return best_url


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
    resolve_timeout: float = 90,
) -> PlaylistDescriptor:
    platform = _playlist_platform(url, info)
    entries = []
    seen_entry_ids = set()
    for index, raw_entry in enumerate(info.get("entries") or []):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = _flat_entry(platform, raw_entry, index)
        if entry is None:
            continue
        # 同一 playlist 里 entry_id 重复（yt-dlp flat 提取偶发）会破坏 treeview
        # 的 iid 唯一性并让 selection 字典互相覆盖；保留首次出现的条目。
        if entry.entry_id in seen_entry_ids:
            continue
        seen_entry_ids.add(entry.entry_id)
        entries.append(entry)
        if len(entries) >= MAX_PLAYLIST_ENTRIES:
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
                entry.webpage_url, ydl_factory, browser_cookies, extract_flat=False,
                resolve_timeout=resolve_timeout
            )
    elif platform.startswith("youtube") or is_twitch_platform(platform):
        def hydrate_entry(entry: PlaylistEntry) -> Mapping[str, Any] | None:
            if entry.title != "Unknown" and entry.duration is not None and entry.upload_date:
                return None
            return _extract_with_cookie_policy(
                entry.webpage_url, ydl_factory, browser_cookies, extract_flat=False,
                resolve_timeout=resolve_timeout
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
        for index, url in enumerate(urls[:MAX_PLAYLIST_ENTRIES])
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
            resolve_timeout=_PLAYLIST_HYDRATION_TIMEOUT,
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
    if source_is_live(info):
        # 直播中：HLS 只有几分钟滑动窗口，检测完再取片段必然大面积失败。
        # 这里直接拒绝（批次会跳过该源并给出提示），而不是花几小时检测。
        raise LiveBroadcastError(
            f"{url} is still live, not a finished replay. A live playlist only "
            "keeps the last few minutes, so clips detected now cannot be fetched "
            "afterwards. Wait until the stream ends and the replay is processed, "
            "then process the replay URL (e.g. twitch.tv/videos/<id>).")

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
