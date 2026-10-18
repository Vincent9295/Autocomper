"""Bounded in-memory HTTP range prefetching for YouTube audio streams."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import re
import threading
import time
from typing import Mapping
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from progress import format_transfer_progress, format_hls_progress
from utils import is_twitch_platform, pause_pending, wait_while_paused


DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 30
DEFAULT_RANGE_ATTEMPTS = 5
# 退避随次数增长，封顶 60s：大文件允许更多重试，但不会无限空等。
DEFAULT_RETRY_DELAYS = (2, 5, 10, 20, 40, 60, 60, 60, 60, 60, 60, 60)
# Twitch HLS 单个分片重试：3 次小退避，防单个片段失败毁掉整个缓存下载。
_HLS_FRAGMENT_ATTEMPTS = 3
_HLS_FRAGMENT_BACKOFF = (1, 2, 4)
# A re-resolve is only worth its 1.5-2 s (plus the shared 2 s limiter) when the
# URL is actually rejected. Slow-but-working ranges must never trigger one.
_REFRESH_MIN_INTERVAL = 5.0
_EXPIRED_ERROR_MARKERS = ("403", "forbidden", "expired", "412", "precondition failed")


def _looks_like_expired_error(exc) -> bool:
    text = str(exc or "").lower()
    return any(marker in text for marker in _EXPIRED_ERROR_MARKERS)


class SpeedMonitor:
    """Detect sustained low throughput after an initial connection warmup."""

    def __init__(self, threshold=1.0, warmup_samples=2, slow_samples=2):
        self.threshold = float(threshold)
        self.warmup_samples = int(warmup_samples)
        self.slow_samples = int(slow_samples)
        self.samples = 0
        self.slow_streak = 0

    def observe(self, megabytes_per_second):
        self.samples += 1
        if self.samples <= self.warmup_samples:
            return False
        if float(megabytes_per_second) < self.threshold:
            self.slow_streak += 1
        else:
            self.slow_streak = 0
        return self.slow_streak >= self.slow_samples

    def reset(self):
        self.samples = 0
        self.slow_streak = 0


class RangePrefetchError(Exception):
    """A range request could not safely provide the compressed stream."""


class CdnTooSlowError(RangePrefetchError):
    """Every mirror stayed below the usable floor; skip the source, do not crawl.

    The caller should record the source as failed and move on (the batch keeps
    going). It must not silently fall back to the single-connection FFmpeg path:
    that path would be just as slow on the same CDN and would occupy the run for
    hours instead of skipping one source.
    """


def _size_from(source):
    metadata = getattr(source, "metadata", {}) or {}
    candidate = getattr(source, "audio_candidates", []) or []
    active = next((item for item in candidate
                   if str(item.get("url") or "") == str(getattr(source, "audio_url", ""))), None)
    for item in ((active,) if active is not None else (metadata,)):
        for key in ("filesize", "filesize_approx", "clen"):
            value = item.get(key)
            # int(inf) 抛 OverflowError（不是 ValueError）；先做有限性过滤，
            # 防 JSON 缓存往返的 Infinity 字面量炸掉调用方。
            if isinstance(value, float) and not math.isfinite(value):
                continue
            try:
                value = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if value > 0:
                return value
    return None


def supports_range_prefetch(source):
    platform = str(getattr(source, "platform", "")).lower()
    url = str(getattr(source, "audio_url", ""))
    return platform in {"youtube", "bilibili"} and url.startswith(("http://", "https://")) and _size_from(source) is not None


def supports_hls_prefetch(source):
    """Twitch HLS prefetch eligibility.

    用平台族判断（``twitchvod``/``twitchstream``/``twitchclips``…）而不是裸平台名
    相等比较：yt-dlp 从不返回裸的 ``twitch``，精确比较会让 Twitch 的 HLS 分片
    预取（含签名 URL 刷新与媒体秒进度）永远不生效。
    """
    platform = str(getattr(source, "platform", "")).lower()
    url = str(getattr(source, "audio_url", ""))
    return is_twitch_platform(platform) and url.startswith(("http://", "https://"))


def _default_request(url, start, end, headers, timeout):
    request_headers = dict(headers or {})
    request_headers["Range"] = f"bytes={start}-{end}"
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        response_headers = {str(key): str(value) for key, value in response.headers.items()}
        return response.read(), int(getattr(response, "status", response.getcode())), response_headers


# 有界读取的分片大小：每次 read1 拿到多少算多少，绝不 read() 等整包。
_BOUNDED_READ_CHUNK = 64 * 1024


def read_range_bounded(url, start, end, headers=None, deadline=3.0,
                       socket_timeout=1.0, request_func=None, stall_timeout=None):
    """Read one HTTP byte range with a hard wall-clock deadline.

    ``_default_request`` uses ``response.read()``, which waits for the whole body
    and resets the socket timeout on every byte. A CDN that trickles (or a host
    that accepts the connection and then goes quiet) therefore blocks forever:
    measured on a real mirror, one 16 MiB range sat for six minutes while other
    ranges had already completed. This reader takes 64 KiB pieces, uses a 1 s
    socket timeout, and stops at ``deadline`` even when bytes keep arriving
    slowly. It returns ``(data, elapsed, status, content_range)``; partial data
    is returned, the caller decides whether a partial read is useful.

    ``stall_timeout`` switches the bound from "total wall clock" to "no bytes for
    N seconds": a slow-but-steady range is then allowed to finish (the offset
    writer uses 45 s), while a silent connection is still abandoned.

    ``request_func`` is the test seam: it must return ``(data, status, headers)``
    and is trusted to be bounded by the caller.
    """
    start = int(start)
    end = int(end)
    expected = end - start + 1
    if expected <= 0:
        raise RangePrefetchError("range read requires end >= start")
    if request_func is not None:
        started = time.monotonic()
        data, status, response_headers = request_func(
            url, start, end, dict(headers or {}), deadline)
        content_range = str((response_headers or {}).get("content-range")
                            or (response_headers or {}).get("Content-Range") or "")
        return bytes(data or b""), time.monotonic() - started, int(status), content_range

    request = Request(
        url, headers={**dict(headers or {}), "Range": f"bytes={start}-{end}"})
    started = time.monotonic()
    last_byte = started
    data = bytearray()
    status = 0
    content_range = ""
    with urlopen(request, timeout=socket_timeout) as response:
        status = int(getattr(response, "status", response.getcode()))
        content_range = str(response.headers.get("Content-Range") or "")
        if status != 206:
            return b"", time.monotonic() - started, status, content_range
        while len(data) < expected:
            now = time.monotonic()
            if stall_timeout is not None:
                if now - last_byte > stall_timeout:
                    break
            elif now - started >= deadline:
                break
            try:
                piece = response.read1(_BOUNDED_READ_CHUNK)
            except (TimeoutError, OSError):
                piece = b""
            if not piece:
                if stall_timeout is not None:
                    if time.monotonic() - last_byte > stall_timeout:
                        break
                elif time.monotonic() - started >= deadline:
                    break
                continue
            data.extend(piece)
            last_byte = time.monotonic()
    return bytes(data), time.monotonic() - started, status, content_range


def read_range_stream(url, start, end, sink, headers=None, stall_timeout=45.0,
                      socket_timeout=1.0, expected_start=None,
                      expected_total=None, stop_check=None):
    """Stream one HTTP range into ``sink`` as the bytes arrive.

    The offset writer needs two things the buffered reader cannot give it:
    live progress (a 16 MiB range at 100 KiB/s otherwise reports nothing for
    minutes) and a way to keep the bytes of a range that was cut short. ``sink``
    is called with each 64 KiB piece, after the status/Content-Range has been
    validated, so a 403 or a wrong-offset response never reaches the file.

    Returns ``(received, elapsed, status, content_range)``. A trickle that keeps
    the no-byte watchdog happy (the average stays below ``_RANGE_MIN_RATE_BYTES``
    after ``_RANGE_MIN_RATE_AFTER`` seconds) is aborted; the writer resumes the
    range from its own offset.
    """
    start = int(start)
    end = int(end)
    expected = end - start + 1
    request = Request(
        url, headers={**dict(headers or {}), "Range": f"bytes={start}-{end}"})
    started = time.monotonic()
    last_byte = started
    received = 0
    status = 0
    content_range = ""
    with urlopen(request, timeout=socket_timeout) as response:
        status = int(getattr(response, "status", response.getcode()))
        content_range = str(response.headers.get("Content-Range") or "")
        if status != 206:
            return 0, time.monotonic() - started, status, content_range
        if content_range:
            match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)",
                             content_range.strip(), re.IGNORECASE)
            if match:
                if (expected_start is not None
                        and int(match.group(1)) != int(expected_start)):
                    raise RangePrefetchError(
                        "range response does not start at the requested offset")
                total = None if match.group(3) == "*" else int(match.group(3))
                if (expected_total is not None and total is not None
                        and total != int(expected_total)):
                    raise RangePrefetchError(
                        f"stream size changed (server reports {total}, expected "
                        f"{expected_total}); refusing to mix objects")
        while received < expected:
            if stop_check is not None and stop_check():
                break
            now = time.monotonic()
            if now - last_byte > stall_timeout:
                break
            if now - started > _RANGE_MIN_RATE_AFTER:
                rate = received / max(now - started, 0.001)
                if rate < _RANGE_MIN_RATE_BYTES:
                    break
            try:
                piece = response.read1(_BOUNDED_READ_CHUNK)
            except (TimeoutError, OSError):
                piece = b""
            if not piece:
                if time.monotonic() - last_byte > stall_timeout:
                    break
                continue
            if len(piece) > expected - received:
                piece = piece[:expected - received]
            sink(piece)
            received += len(piece)
            last_byte = time.monotonic()
    return received, time.monotonic() - started, status, content_range


def _probe_real_size(url, headers, timeout=15, request_func=None):
    """Return the real byte size of a range-capable stream, or None on failure.

    bilibili (and occasionally YouTube) ``filesize_approx`` can be noticeably
    off from the actual DASH stream size. Using the estimate to slice ranges
    then walks past the real end: the boundary block returns a short 206 (which
    fails the length check) and every later block 416s, stalling the download at
    ~70-80% for a long time. Probing ``bytes=0-0`` gives the authoritative size
    from ``Content-Range`` in one cheap request.
    """
    if request_func is None:
        # 只读 1 字节：旧实现走 _default_request，它在**检查状态码之前**就
        # response.read()，服务器忽略 Range 直接回 200 + 整个文件时整段进内存
        # （实测 40MB 响应 → 42MB 常驻）。与 _probe_stream_size 同样的做法。
        request = Request(url, headers={**dict(headers or {}), "Range": "bytes=0-0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                response_headers = {str(key): str(value)
                                    for key, value in response.headers.items()}
                response.read(1)
        except Exception:
            return None
    else:
        try:
            _data, status, response_headers = request_func(url, 0, 0,
                                                           dict(headers), timeout)
        except Exception:
            return None
    if int(status) != 206:
        return None
    if not isinstance(response_headers, Mapping):
        return None
    content_range = str(response_headers.get("content-range") or response_headers.get("Content-Range") or "")
    m = re.match(r"bytes \d+-\d+/(\d+)", content_range)
    if not m:
        return None
    try:
        total = int(m.group(1))
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


def _probe_stream_size(url, headers, timeout=15, request_func=None):
    """Authoritative stream size from a one-byte range request.

    Same idea as ``_probe_real_size`` but it never buffers a body: Full Download
    moves gigabyte video streams, and a server that ignores ``Range`` and answers
    200 with the whole file would otherwise be read into memory before the status
    check. Only a compliant 206 with a single byte is accepted, so this doubles
    as the "does this stream support range transfer at all" gate.
    """
    if request_func is not None:
        try:
            result = request_func(url, 0, 0, dict(headers or {}), timeout)
        except Exception:
            return None
        if not isinstance(result, tuple) or len(result) < 2:
            return None
        data, status = result[0], result[1]
        response_headers = result[2] if len(result) == 3 and result[2] else {}
        try:
            if int(status) != 206 or len(bytes(data or b"")) != 1:
                return None
        except (TypeError, ValueError):
            return None
        content_range = str(response_headers.get("content-range")
                            or response_headers.get("Content-Range") or "")
    else:
        request = Request(url, headers={**dict(headers or {}), "Range": "bytes=0-0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                content_range = str(response.headers.get("Content-Range") or "")
                response.read(1)
        except Exception:
            return None
        if status != 206:
            return None
    m = re.match(r"bytes \d+-\d+/(\d+)", content_range)
    if not m:
        return None
    try:
        total = int(m.group(1))
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


def _default_hls_request(url, headers, timeout):
    request = Request(url, headers=dict(headers or {}))
    with urlopen(request, timeout=timeout) as response:
        return response.read(), int(getattr(response, "status", response.getcode()))


def _log(logger, message):
    if logger is None:
        return
    if hasattr(logger, "log"):
        logger.log(message)
    elif callable(logger):
        logger(message)


def _scaled_range_attempts(source, base=DEFAULT_RANGE_ATTEMPTS, maximum=12):
    """Retry budget grows with source duration (base + hours, capped)."""
    try:
        duration = float(getattr(source, "duration", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    # 非/无限（含 JSON 往返的 Infinity）按未知处理：回落到 base，
    # 避免旧的 int(inf) OverflowError 逃逸窄 except。
    if not math.isfinite(duration) or duration < 0:
        duration = 0.0
    hours = max(0.0, duration / 3600.0)
    return min(maximum, base + int(hours))


def _content_range_total(response_headers):
    """服务器自报的真实总长（206 和 416 都带它），取不到返回 None。

    416 的 ``Content-Range`` 形如 ``bytes */12345``，206 的形如
    ``bytes 0-99/12345``：两种都要认，否则"服务器说到此为止"与"下载被截断"
    就无法区分（两者都会让生成器正常结束）。
    """
    content_range = str((response_headers or {}).get("content-range")
                        or (response_headers or {}).get("Content-Range") or "")
    m = re.match(r"bytes (?:\d+-\d+|\*)/(\d+)", content_range, re.IGNORECASE)
    if not m:
        return None
    try:
        total = int(m.group(1))
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


def iter_range_bytes(source, chunk_size=DEFAULT_CHUNK_SIZE, concurrency=DEFAULT_CONCURRENCY,
                     request_func=None, logger=None, progress_callback=None,
                     max_attempts=None,
                     retry_delays=DEFAULT_RETRY_DELAYS, sleep_func=time.sleep,
                     speed_monitor=None, slow_callback=None, start_offset=0,
                     refresher=None, stats=None):
    """Yield a known-size YouTube audio stream in byte order with bounded prefetch.

    ``stats``（可选 dict，原地填写）让调用方能核对完整性：
      * ``size``：划分 ranges 用的流大小；
      * ``size_probed``：这个大小是探测到的真实值（而不是元数据里的估计值）；
      * ``server_total``：服务器在 Content-Range 里自报的真实总长；
      * ``written``：实际 yield 出去的字节数。
    服务器提前结束（416 / 空 206）时生成器**正常结束**，所以"下完"和"被截断"
    只有在调用方按这些数字核对时才分得清（本函数的历史行为必须保持：sound_reader
    的检测预取依赖"正常结束"来收尾）。
    """
    try:
        chunk_size = int(chunk_size)
        concurrency = int(concurrency)
    except (TypeError, ValueError) as exc:
        raise RangePrefetchError("invalid prefetch parameters") from exc
    size = _size_from(source)
    if chunk_size <= 0 or concurrency <= 0 or size is None:
        raise RangePrefetchError("range prefetch requires a positive known audio size")
    if not supports_range_prefetch(source):
        raise RangePrefetchError("source does not support range prefetch")

    # 用真实流大小校正预估大小：filesize_approx 对 bilibili 偏差可能偏大
    # （越界到实际末尾之后 → 末尾块长度不匹配 + 后续块 416，卡在 ~70-80%）
    # 也可能偏小（漏掉实际尾部 → 下载文件缺尾部 box，ffprobe 报损坏）。
    # 只要探测到的真实大小与预估不同，一律用真实值划分 ranges。
    platform = str(getattr(source, "platform", "")).lower()
    request = request_func or _default_request
    headers = dict(getattr(source, "audio_headers", {}) or {})
    probed_size = None
    if platform == "bilibili":
        # 传**原始** request_func，不是包装后的 `request`：_probe_real_size 只在
        # request_func is None 时走"urlopen + read(1)"的安全路径。传 _default_request
        # 会让它在**检查状态码之前**就 response.read()——忽略 Range、直接回 200 +
        # 整个文件的 CDN 会把整段流读进内存（实测 40MB 响应 → 42MB 常驻），
        # 而这段守卫本来就是为了防它。调用方注入假请求时行为完全不变。
        real_size = _probe_real_size(source.audio_url, headers, request_func=request_func)
        if real_size is not None and real_size > 0:
            probed_size = real_size
            if real_size != size:
                size = real_size

    try:
        start_offset = int(start_offset)
    except (TypeError, ValueError) as exc:
        raise RangePrefetchError("invalid range resume offset") from exc
    if start_offset < 0 or start_offset > size:
        raise RangePrefetchError("range resume offset is outside the source")

    if stats is not None:
        stats.clear()
        stats.update({"size": size, "size_probed": probed_size is not None,
                      "server_total": None, "written": start_offset})

    if max_attempts is None:
        max_attempts = _scaled_range_attempts(source)
    max_attempts = max(1, int(max_attempts))
    retry_delays = tuple(retry_delays or ())

    platform_label = "YouTube" if platform == "youtube" else "Bilibili"
    # chunk 上限保护：超大 chunk（或用户误设成接近整文件）会让单个 range 请求变成
    # 全文件级下载，一旦挂起就要等满 timeout 才能重试，历史上造成"永久卡住"。
    # 这里只压上限到 64MiB；小 chunk 永远更安全，不需要下限。
    chunk_size = min(int(chunk_size), 64 * 1024 * 1024)
    _log(logger, f"{platform_label} memory prefetch: {concurrency} workers, {chunk_size // (1024 * 1024)}MiB chunks")
    ranges = [(start, min(size - 1, start + chunk_size - 1))
              for start in range(start_offset, size, chunk_size)]
    worker_concurrency = max(1, int(concurrency))
    window = max(1, worker_concurrency * 2)
    refresh_lock = threading.Lock()
    refresh_last = [0.0]

    def refresh_once(exc):
        """Refresh the signed URL once per few seconds, only for expiry errors.

        The old code called ``refresher`` from every worker on every exception.
        ``LimitedRefresher`` serializes calls behind a 2 s limiter and retries
        three times, so eight failing ranges turned into ~20 s with no download
        at all, repeated for every slow sample. A slow range is not an expired
        URL; only 403/412/expired should pay for a re-resolve.
        """
        if refresher is None or not _looks_like_expired_error(exc):
            return False
        with refresh_lock:
            now = time.monotonic()
            if now - refresh_last[0] < _REFRESH_MIN_INTERVAL:
                return False
            refresh_last[0] = now
            try:
                updated = refresher(source)
                if updated is not None and updated is not source:
                    source.__dict__.update(updated.__dict__)
                headers.clear()
                headers.update(dict(getattr(source, "audio_headers", {}) or {}))
                _log(logger, f"{platform_label} range request failed; retrying "
                             f"with refreshed URL")
                return True
            except Exception:                                  # noqa: BLE001
                return False

    def fetch(index, start, end):
        last_error = None
        started_at = time.monotonic()
        for attempt in range(max(1, int(max_attempts))):
            try:
                # Snapshot per attempt: another worker may refresh the source
                # while this request is in flight; mixing a new URL with old
                # headers must not happen.
                url = str(getattr(source, "audio_url", "") or "")
                request_headers = dict(headers)
                result = request(url, start, end, request_headers, DEFAULT_TIMEOUT)
                response_headers = {}
                if not isinstance(result, tuple) or len(result) not in (2, 3):
                    raise RangePrefetchError("invalid range response")
                data, status = result[:2]
                if len(result) == 3 and result[2]:
                    response_headers = {str(key).lower(): str(value) for key, value in result[2].items()}
                if int(status) != 206:
                    if int(status) == 416:
                        # 请求区间超出实际流末尾：后续所有块也会 416，直接按
                        # "末尾已到"提前结束，不重试（filesize_approx 虚高场景）。
                        # 服务器自报的总长只能经 stats 带出去：消费者按 3 元组解包，
                        # 这里多返回一个值会直接 ValueError（round 48 终检发现）。
                        if stats is not None:
                            total = _content_range_total(response_headers)
                            if total:
                                stats["server_total"] = total
                        return index, b"", -1
                    raise RangePrefetchError("range server returned a non-partial response")
                expected_length = end - start + 1
                short_final = platform == "bilibili" and index == len(ranges) - 1 and len(data) < expected_length
                content_range = response_headers.get("content-range", "")
                # 长度不匹配但 Content-Range 前缀合法：说明这是实际末尾块（预估
                # filesize 虚高），CDN 只返回了剩余真实数据。按实际长度接受，
                # 避免长度校验把它当成坏块反复重试卡在 ~70-80%。
                cr_matches = content_range.lower().startswith(f"bytes {start}-{start + len(data) - 1}/")
                if len(data) != expected_length and not short_final and not cr_matches:
                    raise RangePrefetchError("range response length mismatch")
                if short_final:
                    if content_range and not cr_matches:
                        raise RangePrefetchError("range response Content-Range mismatch")
                if len(data) == 0:
                    # 实际末尾已全部收完，后续块同样为空/416，提前正常结束
                    return index, b"", -1
                elapsed = max(time.monotonic() - started_at, 0.001)
                speed = len(data) / elapsed / (1024 * 1024)
                return index, bytes(data), speed
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, int(max_attempts)):
                    # 签名 URL 过期（403）是 YouTube 批量检测失败的常见原因：
                    # 先刷新 source（重新解析出新的 audio_url/audio_headers）再重试。
                    refresh_once(exc)
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)] if retry_delays else 0
                    if delay:
                        sleep_func(delay)
        raise RangePrefetchError("range request failed after retry") from last_error

    executor = ThreadPoolExecutor(max_workers=worker_concurrency)
    futures = {}
    pending_index = 0
    next_index = 0
    buffered = {}
    started_at = time.monotonic()
    completed_bytes = start_offset
    end_reached = False
    try:
        while pending_index < len(ranges) and len(futures) < window:
            start, end = ranges[pending_index]
            futures[executor.submit(fetch, pending_index, start, end)] = pending_index
            pending_index += 1
        while futures:
            completed = next(as_completed(tuple(futures)))
            index = futures.pop(completed)
            result_index, data, speed = completed.result()
            buffered[result_index] = data
            if speed is not None and speed > 0 and speed_monitor is not None and speed_monitor.observe(speed):
                if slow_callback is not None:
                    slow_callback(source, speed)
                speed_monitor.reset()
            if speed == -1:
                end_reached = True
            while next_index in buffered:
                data = buffered.pop(next_index)
                completed_bytes += len(data)
                if progress_callback is not None:
                    progress_callback(format_transfer_progress(
                        completed_bytes, size, time.monotonic() - started_at))
                if data:
                    yield data
                next_index += 1
            if not end_reached:
                while pending_index < len(ranges) and len(futures) < window:
                    start, end = ranges[pending_index]
                    futures[executor.submit(fetch, pending_index, start, end)] = pending_index
                    pending_index += 1
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _bounded_default_request(url, start, end, headers, timeout):
    """Production range request: bounded pieces + no-byte stall watchdog."""
    data, _elapsed, status, content_range = read_range_bounded(
        url, start, end, headers, deadline=timeout, socket_timeout=1.0,
        stall_timeout=timeout)
    return data, status, {"content-range": content_range}


_SWITCH_VERIFY_BYTES = 256 * 1024
_SWITCH_VERIFY_MIN_BYTES = 64 * 1024
# Aggregate throughput below this for two consecutive windows means the host is
# not merely fluctuating: try a mirror switch (never on a single noisy sample).
_WRITER_MIN_MBPS = 0.35
_WRITER_WATCHDOG_SECONDS = 10.0
_WRITER_HOPELESS_MBPS = 0.08
_WRITER_HOPELESS_WINDOWS = 6
# A connection that keeps trickling a byte every few seconds defeats the no-byte
# watchdog. After this long, require at least this average rate or abort the
# range (the writer then resumes it, possibly on another host).
_RANGE_MIN_RATE_BYTES = 32 * 1024
_RANGE_MIN_RATE_AFTER = 30.0
_RANGE_REPORT_SECONDS = 0.5


def download_stream_to_file(url, destination, headers=None, size=None,
                            chunk_size=DEFAULT_CHUNK_SIZE,
                            concurrency=DEFAULT_CONCURRENCY, request_func=None,
                            logger=None, progress_callback=None, max_attempts=None,
                            retry_delays=DEFAULT_RETRY_DELAYS, sleep_func=time.sleep,
                            cancel_check=None, stall_timeout=None,
                            refresh_func=None, switch_func=None, max_switches=2,
                            stats=None, abort_on_hopeless=False):
    """Write one http(s) stream to ``destination`` with parallel Range requests.

    Full Download used to be plain yt-dlp: for Bilibili/YouTube the chosen
    formats are ``protocol=https`` with no fragments, so yt-dlp transfers them on
    ONE connection and Bilibili's per-connection throttle caps it around
    1 MB/s (measured 0.9-1.8 MB/s single vs 20-23 MB/s with four range requests
    on the same URL). This is the transfer the Audio Cache already had.

    Every range is written at its own offset, with a per-range byte count (the
    file grows with the committed ranges and is truncated to ``size`` only after
    every range is verified; it is not preallocated). That is what makes a slow
    range harmless: the other ranges keep committing and progress keeps moving
    (the old ordered reader froze the whole file behind one slow range and
    buffered up to ``window x chunk`` in RAM). A range is resumed from its own
    offset after a
    short read, and the transfer is only accepted when **every** range is full:
    the old code truncated to the total byte count, which could leave a zero hole
    and lose the tail while returning success.

    ``refresh_func``/``switch_func`` take no arguments and return
    ``(url, headers)`` (or None). A switch verifies that the new URL serves the
    same first 256 KiB that are already on disk before it is accepted, because
    different mirrors of one VOD are byte-identical but a wrong object is not.
    """
    try:
        chunk_size = int(chunk_size)
        concurrency = int(concurrency)
    except (TypeError, ValueError) as exc:
        raise RangePrefetchError("invalid parallel download parameters") from exc
    if chunk_size <= 0 or concurrency <= 0:
        raise RangePrefetchError("parallel download requires positive chunk size and concurrency")
    header_map = dict(headers or {})
    # 总是先探测真实大小：① 顺带确认服务端真的支持 Range（不回 206 就直接
    # 放弃，交回 yt-dlp）；② yt-dlp 的 filesize_approx 对 Bilibili 可能偏大或
    # 偏小，按错的 size 划分区间会卡在尾部（越界 416 / 少下尾部 box）。
    probed = _probe_stream_size(url, header_map, request_func=request_func)
    size_verified = probed is not None
    declared_size = size
    if probed is not None:
        size = probed
    try:
        size = int(size)
    except (TypeError, ValueError) as exc:
        raise RangePrefetchError("unknown stream size") from exc
    if size <= 0:
        raise RangePrefetchError("unknown stream size")

    # 与 iter_range_bytes 同样的上限保护：超大 chunk 会让单个请求退化成
    # 全文件级请求，一旦挂起就要等满 timeout 才重试。
    chunk_size = min(chunk_size, 64 * 1024 * 1024)
    ranges = [(start, min(size - 1, start + chunk_size - 1))
              for start in range(0, size, chunk_size)]
    lengths = [end - start + 1 for start, end in ranges]
    if max_attempts is None:
        max_attempts = DEFAULT_RANGE_ATTEMPTS
    max_attempts = max(1, int(max_attempts))
    retry_delays = tuple(retry_delays or ())
    stall_timeout = float(stall_timeout or 45.0)

    _log(logger, f"Parallel transfer: {max(1, int(concurrency))} connections, "
                 f"{chunk_size // (1024 * 1024)}MiB chunks, "
                 f"{size / (1024 * 1024):.1f}MB total")

    request = request_func or _bounded_default_request
    state_lock = threading.Lock()
    refresh_lock = threading.Lock()
    switch_lock = threading.Lock()
    url_holder = [str(url)]
    headers_holder = [dict(header_map or {})]
    done = [0] * len(ranges)
    written_total = [0]
    refreshed_at = [0.0]
    switches = [0]
    abort_event = threading.Event()
    abort_reason = [""]
    started_at = time.monotonic()
    last_report_at = [0.0]

    # No preallocation: the file grows with the highest committed offset and is
    # truncated to the exact size only after every range is verified. NTFS
    # sparse-extends a file on truncate anyway, so preallocation did not reserve
    # space reliably but did make Explorer show the full size from the first
    # second (read as "already finished" by the user). Random-offset writes extend
    # the file on demand.
    handle = open(destination, "w+b")

    def report(force=False):
        if progress_callback is None:
            return
        now = time.monotonic()
        if not force and now - last_report_at[0] < _RANGE_REPORT_SECONDS:
            return
        last_report_at[0] = now
        with state_lock:
            done_bytes = written_total[0]
        progress_callback(done_bytes, size, now - started_at)

    def _apply_update(candidate):
        if not candidate:
            return False
        new_url, new_headers = candidate
        if not new_url or str(new_url) == url_holder[0]:
            return False
        url_holder[0] = str(new_url)
        headers_holder[0] = dict(new_headers or {})
        return True

    def refresh_once(exc):
        if refresh_func is None or not _looks_like_expired_error(exc):
            return False
        with refresh_lock:
            now = time.monotonic()
            if now - refreshed_at[0] < _REFRESH_MIN_INTERVAL:
                return False
            refreshed_at[0] = now
            try:
                updated = _apply_update(refresh_func())
            except Exception:                                  # noqa: BLE001
                return False
            if updated:
                _log(logger, "Range request failed; retrying with a refreshed URL")
            return updated

    def _switch_serves_the_same_prefix(new_url, new_headers, n):
        """Compare the first n bytes already on disk with the new mirror."""
        if n <= 0:
            return True
        try:
            with state_lock:
                handle.flush()
            with open(destination, "rb") as disk:
                prefix = disk.read(n)
            if len(prefix) < n:
                return True                 # nothing committed at offset 0 yet
            if request_func is not None:
                data, status, _response_headers = request(
                    new_url, 0, n - 1, dict(new_headers or {}), stall_timeout)
            else:
                data, _elapsed, status, _crange = read_range_bounded(
                    new_url, 0, n - 1, new_headers, deadline=stall_timeout,
                    socket_timeout=1.0, stall_timeout=stall_timeout)
            return int(status) == 206 and bytes(data)[:n] == prefix
        except Exception:                                      # noqa: BLE001
            return False

    def switch_once():
        if switch_func is None:
            return False
        with switch_lock:
            if switches[0] >= max(0, int(max_switches)):
                return False
            try:
                candidate = switch_func()
            except Exception:                                  # noqa: BLE001
                return False
            if not candidate:
                return False
            new_url, new_headers = candidate
            if str(new_url) == url_holder[0]:
                return False
            with state_lock:
                committed = done[0] if done else 0
            verify_bytes = min(_SWITCH_VERIFY_BYTES, committed)
            if verify_bytes >= _SWITCH_VERIFY_MIN_BYTES and not _switch_serves_the_same_prefix(
                    str(new_url), new_headers, verify_bytes):
                _log(logger, "Mirror serves different bytes; keeping the current host")
                return False
            switches[0] += 1
            url_holder[0] = str(new_url)
            headers_holder[0] = dict(new_headers or {})
            _log(logger, f"Switched range host after {switches[0]} switch(es)")
            return True

    def fetch(index, start, end):
        length = lengths[index]
        attempts = 0
        rounds = 0
        max_rounds = max(8, max_attempts * 8)
        last_error = None
        while rounds < max_rounds:
            rounds += 1
            if abort_event.is_set():
                raise CdnTooSlowError(abort_reason[0])
            if callable(cancel_check) and cancel_check():
                raise InterruptedError("Parallel transfer cancelled by user.")
            if pause_pending():
                if not wait_while_paused(cancel_check=cancel_check):
                    raise InterruptedError("Parallel transfer cancelled by user.")
            with state_lock:
                offset_in_range = done[index]
            if offset_in_range >= length:
                return index
            offset = start + offset_in_range
            before = offset_in_range
            try:
                if request_func is None:
                    # Production path: commit each 64 KiB piece as it arrives, so
                    # the UI moves during a slow range and a cut-off range keeps
                    # everything it already received.
                    def commit(piece):
                        with state_lock:
                            handle.seek(start + done[index])
                            handle.write(piece)
                            handle.flush()
                            done[index] += len(piece)
                            written_total[0] += len(piece)
                        report()

                    received, status, content_range = read_range_stream(
                        url_holder[0], offset, end, commit,
                        headers=dict(headers_holder[0]),
                        stall_timeout=stall_timeout,
                        expected_start=offset, expected_total=size,
                        stop_check=lambda: (abort_event.is_set()
                                            or pause_pending()
                                            or (callable(cancel_check)
                                                and cancel_check())))
                    if int(status) == 416:
                        # The requested offset is past the real end. If a real
                        # prefix of this range is on disk, the stream ended there.
                        if done[index] > 0:
                            with state_lock:
                                done[index] = length
                            report(force=True)
                            return index
                        raise RangePrefetchError("range server rejected the first range")
                    if int(status) != 206:
                        raise RangePrefetchError(
                            "range server returned a non-partial response")
                    if callable(cancel_check) and cancel_check():
                        raise InterruptedError("Parallel transfer cancelled by user.")
                    if pause_pending():
                        # The range was cut at a piece boundary; the next loop
                        # iteration blocks in wait_while_paused and resumes from
                        # the committed offset when the user continues.
                        attempts = 0
                        continue
                    if received <= 0:
                        raise RangePrefetchError("empty range response")
                    if abort_event.is_set():
                        raise CdnTooSlowError(abort_reason[0])
                    if done[index] >= length:
                        report(force=True)
                        return index
                    attempts = 0
                else:
                    data, status, response_headers = request(
                        url_holder[0], offset, end, dict(headers_holder[0]),
                        stall_timeout)
                    if not isinstance(data, (bytes, bytearray)):
                        raise RangePrefetchError("invalid range response")
                    data = bytes(data)
                    if int(status) == 416:
                        # The requested offset is past the real end. If a real prefix
                        # of this range is already on disk, the stream ended there.
                        if offset_in_range > 0:
                            with state_lock:
                                done[index] = length
                            report(force=True)
                            return index
                        raise RangePrefetchError("range server rejected the first range")
                    if int(status) != 206:
                        raise RangePrefetchError("range server returned a non-partial response")
                    response_headers = {str(key).lower(): str(value)
                                        for key, value in (response_headers or {}).items()}
                    content_range = str(response_headers.get("content-range", "")).lower()
                    if content_range:
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise RangePrefetchError(
                                "range response does not start at the requested offset")
                        total = _content_range_total(response_headers)
                        if total is not None and total != size:
                            raise RangePrefetchError(
                                f"stream size changed (server reports {total}, "
                                f"expected {size}); refusing to mix objects")
                    remaining = length - offset_in_range
                    if len(data) > remaining:
                        data = data[:remaining]
                    if not data:
                        raise RangePrefetchError("empty range response")
                    with state_lock:
                        handle.seek(offset)
                        handle.write(data)
                        # Flush per committed range: a crash leaves the bytes on
                        # disk, and progress can be read back by another handle.
                        handle.flush()
                        done[index] = offset_in_range + len(data)
                        written_total[0] += len(data)
                    report()
                    if done[index] >= length:
                        report(force=True)
                        return index
                    # Progress but not complete: resume this range immediately.
                    attempts = 0
            except InterruptedError:
                raise
            except CdnTooSlowError:
                raise
            except Exception as exc:                           # noqa: BLE001
                last_error = exc
                with state_lock:
                    progressed = done[index] > before
                if progressed:
                    attempts = 0
                    continue
                attempts += 1
                refreshed = refresh_once(exc)
                if attempts >= max_attempts:
                    if switch_once():
                        attempts = 0
                        continue
                    raise RangePrefetchError(
                        f"range {start}-{end} failed after {max_attempts} "
                        f"attempts without progress") from exc
                if not refreshed and retry_delays:
                    sleep_func(retry_delays[min(attempts - 1, len(retry_delays) - 1)])
        raise RangePrefetchError(
            f"range {start}-{end} did not finish ({done[index]}/{length} bytes)") from last_error

    executor = ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
    futures = {}
    window = max(1, int(concurrency)) * 2
    next_index = 0
    watchdog_stop = threading.Event()
    last_sample = [time.monotonic(), 0]
    low_windows = [0]
    hopeless_windows = [0]
    failed_switch_rounds = [0]

    def speed_watchdog():
        """Switch hosts, or abort a hopeless transfer, on aggregate throughput.

        Per-range samples are noisy and a single slow connection is normal; two
        consecutive 10 s windows under ``_WRITER_MIN_MBPS`` trigger one switch
        probe. If mirrors do not help and the rate stays under
        ``_WRITER_HOPELESS_MBPS`` for ``_WRITER_HOPELESS_WINDOWS`` windows, the
        transfer is abandoned with ``CdnTooSlowError`` so the batch can skip the
        source instead of crawling for hours.
        """
        while not watchdog_stop.wait(_WRITER_WATCHDOG_SECONDS):
            now = time.monotonic()
            with state_lock:
                written = written_total[0]
            if written >= size or abort_event.is_set():
                return
            if pause_pending():
                # No bytes arriving while paused is not a slow host: reset the
                # windows so a pause never triggers a mirror switch.
                last_sample[0] = time.monotonic()
                last_sample[1] = written
                low_windows[0] = hopeless_windows[0] = failed_switch_rounds[0] = 0
                continue
            elapsed = max(now - last_sample[0], 0.001)
            rate = (written - last_sample[1]) / elapsed / (1024 * 1024)
            last_sample[0] = now
            last_sample[1] = written
            if elapsed < _WRITER_WATCHDOG_SECONDS - 1.0:
                continue
            hopeless_windows[0] = (hopeless_windows[0] + 1
                                   if rate < _WRITER_HOPELESS_MBPS else 0)
            low_windows[0] = (low_windows[0] + 1
                              if rate < _WRITER_MIN_MBPS else 0)
            if rate >= _WRITER_MIN_MBPS:
                failed_switch_rounds[0] = 0
            if low_windows[0] >= 2:
                low_windows[0] = 0
                if switch_once():
                    last_sample[0] = time.monotonic()
                    last_sample[1] = written
                    hopeless_windows[0] = 0
                    failed_switch_rounds[0] = 0
                    continue
                failed_switch_rounds[0] += 1
            # Two consecutive rescue rounds that could not find a faster mirror,
            # while the current transfer is still under the usable floor. By
            # default the writer NEVER aborts on slowness (the user wants every
            # source downloaded, however slow); callers that opt in get
            # ``CdnTooSlowError`` so they can park the source and retry later.
            if abort_on_hopeless and (
                    hopeless_windows[0] >= _WRITER_HOPELESS_WINDOWS
                    or (failed_switch_rounds[0] >= 2
                        and rate < _WRITER_MIN_MBPS)):
                abort_reason[0] = (
                    f"CDN throughput stayed at {rate:.2f} MB/s "
                    f"(no mirror reached {_WRITER_MIN_MBPS:g} MB/s; "
                    f"{failed_switch_rounds[0]} rescue round(s) failed)")
                abort_event.set()
                return

    watchdog = threading.Thread(target=speed_watchdog, name="range-speed-watchdog",
                                daemon=True)
    watchdog.start()
    try:
        while next_index < len(ranges) and len(futures) < window:
            start, end = ranges[next_index]
            futures[executor.submit(fetch, next_index, start, end)] = next_index
            next_index += 1
        while futures:
            completed = next(as_completed(tuple(futures)))
            futures.pop(completed)
            completed.result()
            while next_index < len(ranges) and len(futures) < window:
                start, end = ranges[next_index]
                futures[executor.submit(fetch, next_index, start, end)] = next_index
                next_index += 1
    except InterruptedError:
        for future in futures:
            future.cancel()
        raise
    except Exception as exc:
        for future in futures:
            future.cancel()
        if isinstance(exc, CdnTooSlowError):
            raise
        raise RangePrefetchError("parallel transfer failed") from exc
    finally:
        watchdog_stop.set()
        if watchdog.is_alive():
            watchdog.join(timeout=2)
        executor.shutdown(wait=True, cancel_futures=True)
        try:
            handle.flush()
            os.fsync(handle.fileno())
        except OSError:
            pass
        handle.close()

    report()
    with state_lock:
        total_written = written_total[0]
    incomplete = [index for index, length in enumerate(lengths)
                  if done[index] < length]
    if incomplete:
        # A hole is never acceptable: hand the caller a failure so it can retry
        # or fall back, instead of truncating and calling a broken file done.
        raise RangePrefetchError(
            f"parallel transfer left {len(incomplete)} incomplete range(s) "
            f"({total_written} of {size} bytes)")
    if total_written != size:
        raise RangePrefetchError(
            f"parallel transfer wrote {total_written} of {size} bytes")
    if not size_verified and total_written != declared_size:
        # 探测失败时 size 只是调用方的 filesize_approx。写不够说明是**截断**：以前
        # 这里照样返回成功，文件被登记为"已下载"并永久复用——文件顶部注释里
        # "bilibili filesize_approx 可能偏小（缺尾部 box）"说的就是它。宁可失败，
        # 让上层退回 yt-dlp 重下（那才是正确字节）。
        raise RangePrefetchError(
            f"parallel transfer wrote {total_written} of the estimated "
            f"{declared_size} bytes; the size could not be verified")
    # Every range is on disk; make the final size exact (a no-op when the last
    # range already extended the file to `size`).
    try:
        with open(destination, "r+b") as exact:
            exact.truncate(size)
    except OSError as exc:
        raise RangePrefetchError(
            f"could not finalize the transfer size: {exc}") from exc
    if stats is not None:
        stats.update({"size": size, "written": total_written,
                      "ranges": len(ranges), "incomplete": 0,
                      "switches": switches[0], "size_probed": size_verified})
    return total_written


def measure_stream_throughput(url, headers=None, probe_bytes=4 * 1024 * 1024,
                             deadline=6.0, request_func=None):
    """MB/s from one bounded, wall-clock-capped range read (None = unmeasurable).

    Bilibili nodes vary far more than they should: the same VOD measured minutes
    apart on different mirrors gave 0.4-0.6 MB/s on one and 13-20 MB/s on
    another, so picking a host is worth one bounded probe before committing to a
    multi-gigabyte download.
    """
    try:
        probe_bytes = int(probe_bytes)
    except (TypeError, ValueError):
        return None
    if probe_bytes <= 0:
        return None
    try:
        data, elapsed, status, _content_range = read_range_bounded(
            url, 0, probe_bytes - 1, headers, deadline=deadline,
            request_func=request_func)
    except Exception:                                          # noqa: BLE001
        return None
    if int(status) != 206 or not data:
        return None
    return len(data) / max(elapsed, 0.001) / (1024 * 1024)


def _hls_attribute(line, name):
    prefix = name + '="'
    start = line.find(prefix)
    if start < 0:
        return ""
    start += len(prefix)
    end = line.find('"', start)
    return line[start:end] if end >= 0 else ""


def iter_hls_bytes(source, concurrency=DEFAULT_CONCURRENCY, request_func=None, logger=None,
                   progress_callback=None, refresher=None):
    """Yield Twitch HLS init and media fragments in playlist order.

    ``refresher`` (optional) is called when a fragment keeps failing after its
    per-fragment retries: the signed HLS URL/token is renewed and the manifest
    is re-read, then the stream continues. Without it an expired Twitch URL
    could only fail the whole prefetch and fall back to the slow FFmpeg path
    (the tester saw the legacy "(attempt 1/3)" download with no live progress).
    """
    try:
        concurrency = int(concurrency)
    except (TypeError, ValueError) as exc:
        raise RangePrefetchError("invalid HLS concurrency") from exc
    if concurrency <= 0 or not supports_hls_prefetch(source):
        raise RangePrefetchError("source does not support HLS prefetch")
    request = request_func or _default_hls_request
    headers = dict(getattr(source, "audio_headers", {}) or {})
    started_at = time.monotonic()
    bytes_done = 0
    refresh_state = {"done": False, "playlist": None, "duration": None}

    def refresh_playlist(expected_len, log_message):
        """Renew the signed URL once, re-read the manifest and keep it.

        分片是并发预取的（window = concurrency*2 个请求都绑在同一个签名 token
        上），token 一旦过期通常同时失败多个分片——所以刷新结果必须**留存**给
        后续失败复用；只按"刷新过一次"就放弃会让整个预取失败、退回慢速 FFmpeg
        路径（这个特性等于没生效）。
        """
        if refresh_state["playlist"] is not None:
            return refresh_state["playlist"], refresh_state["duration"]
        if refresher is None or refresh_state["done"]:
            return None
        refresh_state["done"] = True
        try:
            updated = refresher(source)
            if isinstance(updated, object) and updated is not source:
                source.__dict__.update(getattr(updated, "__dict__", {}) or {})
            headers.clear()
            headers.update(dict(getattr(source, "audio_headers", {}) or {}))
            manifest = fetch(source.audio_url).decode("utf-8")
            playlist, duration = _parse_hls_playlist(manifest, source.audio_url, fetch)
        except Exception:
            return None
        # 位置必须一一对应：刷新拿到的是同一个 VOD 的同一段 DVR 窗口，
        # 长度不同说明窗口/变体变了，按旧索引续跑会静默错位（重复或漏段）。
        if expected_len and len(playlist) != expected_len:
            _log(logger, "Twitch HLS refresh returned a different playlist length; "
                         "not resuming from stale indices")
            return None
        _log(logger, log_message)
        refresh_state["playlist"] = playlist
        refresh_state["duration"] = duration
        return playlist, duration

    def fetch(url):
        last_error = None
        for attempt in range(_HLS_FRAGMENT_ATTEMPTS):
            try:
                result = request(url, dict(headers), DEFAULT_TIMEOUT)
                if not isinstance(result, tuple) or len(result) < 2 or int(result[1]) < 200 or int(result[1]) >= 300:
                    raise RangePrefetchError("HLS request returned an invalid response")
                return bytes(result[0])
            except Exception as exc:
                last_error = exc
                if attempt + 1 < _HLS_FRAGMENT_ATTEMPTS:
                    time.sleep(_HLS_FRAGMENT_BACKOFF[attempt])
        raise RangePrefetchError("HLS request failed") from last_error

    try:
        manifest = fetch(source.audio_url).decode("utf-8")
        playlist, media_duration = _parse_hls_playlist(manifest, source.audio_url, fetch)
        _log(logger, f"Twitch HLS prefetch: {len(playlist)} fragments"
                     + (f", {media_duration:.1f}s media" if media_duration else ""))
        executor = ThreadPoolExecutor(max_workers=concurrency)
        futures = {}
        next_submit = 0
        next_yield = 0
        buffered = {}
        fetched_media = 0.0
        window = max(1, concurrency * 2)
        try:
            while next_submit < len(playlist) and len(futures) < window:
                future = executor.submit(fetch, playlist[next_submit][0])
                futures[future] = next_submit
                next_submit += 1
            while futures:
                completed_future = next(as_completed(tuple(futures)))
                index = futures.pop(completed_future)
                try:
                    buffered[index] = completed_future.result()
                except Exception:
                    # 分片彻底失败：刷新签名 URL 并用新 manifest 续跑。
                    # 刷新结果会留存，同一 token 上并发失败的其它分片复用同一次
                    # 刷新（否则第二个失败就把整段预取打回 FFmpeg 慢路径）。
                    renewed = refresh_playlist(
                        len(playlist),
                        "Twitch HLS fragment failed; refreshing the signed URL")
                    if renewed is None:
                        raise
                    playlist, media_duration = renewed
                    if index >= len(playlist):
                        raise
                    buffered[index] = fetch(playlist[index][0])
                while next_yield in buffered:
                    data = buffered.pop(next_yield)
                    bytes_done += len(data)
                    # 刷新后新 manifest 可能更短：已按旧索引完成的在途分片不能
                    # 用新 playlist 取时长（越界会 IndexError，把整段预取降级成
                    # FFmpeg 回落）。越界时按 0 计，只影响进度显示。
                    if next_yield < len(playlist):
                        fetched_media += playlist[next_yield][1]
                    if progress_callback is not None:
                        progress_callback(format_hls_progress(
                            fetched_media, media_duration, time.monotonic() - started_at,
                            bytes_done))
                    yield data
                    next_yield += 1
                while next_submit < len(playlist) and len(futures) < window:
                    future = executor.submit(fetch, playlist[next_submit][0])
                    futures[future] = next_submit
                    next_submit += 1
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
    except RangePrefetchError:
        raise
    except Exception as exc:
        raise RangePrefetchError("HLS manifest processing failed") from exc


def _parse_hls_playlist(manifest, base_url, fetch):
    """Return ([(url, media_seconds), ...], total_media_seconds)."""
    lines = [line.strip() for line in manifest.splitlines() if line.strip()]
    variants = [(line, lines[index + 1]) for index, line in enumerate(lines[:-1])
                if line.startswith("#EXT-X-STREAM-INF:") and not lines[index + 1].startswith("#")]
    if variants:
        audio = [item for item in variants if "AUDIO=" in item[0].upper()]
        uri = max(audio or variants, key=lambda item: item[0].count("BANDWIDTH="))[1]
        child_url = urljoin(base_url, uri)
        return _parse_hls_playlist(fetch(child_url).decode("utf-8"), child_url, fetch)
    result = []
    pending_duration = 0.0
    for line in lines:
        if line.startswith("#EXTINF:"):
            try:
                pending_duration = max(0.0, float(line.split(":", 1)[1].split(",")[0]))
            except (TypeError, ValueError, IndexError):
                pending_duration = 0.0
        elif line.startswith("#EXT-X-MAP:"):
            uri = _hls_attribute(line, "URI")
            if uri:
                result.append((urljoin(base_url, uri), 0.0))
            # MAP（init segment）不消耗前一个 #EXTINF：不重置的话那段时长会
            # 记到下一个真正分片上，总时长偏大、百分比永远到不了 100。
            pending_duration = 0.0
        elif not line.startswith("#"):
            result.append((urljoin(base_url, line), pending_duration))
            pending_duration = 0.0
    if not result:
        raise RangePrefetchError("HLS manifest contains no media segments")
    return result, sum(item[1] for item in result)
