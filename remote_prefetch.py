"""Bounded in-memory HTTP range prefetching for YouTube audio streams."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
from typing import Mapping
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from progress import format_transfer_progress


DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 30
DEFAULT_RANGE_ATTEMPTS = 5
# 退避随次数增长，封顶 60s：大文件允许更多重试，但不会无限空等。
DEFAULT_RETRY_DELAYS = (2, 5, 10, 20, 40, 60, 60, 60, 60, 60, 60, 60)
# Twitch HLS 单个分片重试：3 次小退避，防单个片段失败毁掉整个缓存下载。
_HLS_FRAGMENT_ATTEMPTS = 3
_HLS_FRAGMENT_BACKOFF = (1, 2, 4)


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


def _size_from(source):
    metadata = getattr(source, "metadata", {}) or {}
    candidate = getattr(source, "audio_candidates", []) or []
    active = next((item for item in candidate
                   if str(item.get("url") or "") == str(getattr(source, "audio_url", ""))), None)
    for item in ((active,) if active is not None else (metadata,)):
        for key in ("filesize", "filesize_approx", "clen"):
            value = item.get(key)
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def supports_range_prefetch(source):
    platform = str(getattr(source, "platform", "")).lower()
    url = str(getattr(source, "audio_url", ""))
    return platform in {"youtube", "bilibili"} and url.startswith(("http://", "https://")) and _size_from(source) is not None


def supports_hls_prefetch(source):
    platform = str(getattr(source, "platform", "")).lower()
    url = str(getattr(source, "audio_url", ""))
    return platform == "twitch" and url.startswith(("http://", "https://"))


def _default_request(url, start, end, headers, timeout):
    request_headers = dict(headers or {})
    request_headers["Range"] = f"bytes={start}-{end}"
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        response_headers = {str(key): str(value) for key, value in response.headers.items()}
        return response.read(), int(getattr(response, "status", response.getcode())), response_headers


def _probe_real_size(url, headers, timeout=15, request_func=None):
    """Return the real byte size of a range-capable stream, or None on failure.

    bilibili (and occasionally YouTube) ``filesize_approx`` can be noticeably
    off from the actual DASH stream size. Using the estimate to slice ranges
    then walks past the real end: the boundary block returns a short 206 (which
    fails the length check) and every later block 416s, stalling the download at
    ~70-80% for a long time. Probing ``bytes=0-0`` gives the authoritative size
    from ``Content-Range`` in one cheap request.
    """
    request = request_func or _default_request
    try:
        _data, status, response_headers = request(url, 0, 0, dict(headers), timeout)
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
    hours = max(0.0, duration / 3600.0)
    return min(maximum, base + int(hours))


def iter_range_bytes(source, chunk_size=DEFAULT_CHUNK_SIZE, concurrency=DEFAULT_CONCURRENCY,
                     request_func=None, logger=None, progress_callback=None,
                     max_attempts=None,
                     retry_delays=DEFAULT_RETRY_DELAYS, sleep_func=time.sleep,
                     speed_monitor=None, slow_callback=None, start_offset=0,
                     refresher=None):
    """Yield a known-size YouTube audio stream in byte order with bounded prefetch."""
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
    if platform == "bilibili":
        real_size = _probe_real_size(source.audio_url, headers, request_func=request)
        if real_size is not None and real_size > 0 and real_size != size:
            size = real_size

    try:
        start_offset = int(start_offset)
    except (TypeError, ValueError) as exc:
        raise RangePrefetchError("invalid range resume offset") from exc
    if start_offset < 0 or start_offset > size:
        raise RangePrefetchError("range resume offset is outside the source")

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

    def fetch(index, start, end):
        last_error = None
        started_at = time.monotonic()
        for attempt in range(max(1, int(max_attempts))):
            try:
                result = request(source.audio_url, start, end, dict(headers), DEFAULT_TIMEOUT)
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
                    if refresher is not None:
                        try:
                            updated = refresher(source)
                            if updated is not None and updated is not source:
                                source.__dict__.update(updated.__dict__)
                            nonlocal_headers = dict(getattr(source, "audio_headers", {}) or {})
                            headers.clear()
                            headers.update(nonlocal_headers)
                            _log(logger, f"{platform_label} range request failed; retrying with refreshed URL")
                        except Exception:
                            pass
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


def _hls_attribute(line, name):
    prefix = name + '="'
    start = line.find(prefix)
    if start < 0:
        return ""
    start += len(prefix)
    end = line.find('"', start)
    return line[start:end] if end >= 0 else ""


def iter_hls_bytes(source, concurrency=DEFAULT_CONCURRENCY, request_func=None, logger=None,
                   progress_callback=None):
    """Yield Twitch HLS init and media fragments in playlist order."""
    try:
        concurrency = int(concurrency)
    except (TypeError, ValueError) as exc:
        raise RangePrefetchError("invalid HLS concurrency") from exc
    if concurrency <= 0 or not supports_hls_prefetch(source):
        raise RangePrefetchError("source does not support HLS prefetch")
    request = request_func or _default_hls_request
    headers = dict(getattr(source, "audio_headers", {}) or {})

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
        playlist_urls = _parse_hls_playlist(manifest, source.audio_url, fetch)
        _log(logger, f"Twitch HLS prefetch: {len(playlist_urls)} fragments")
        executor = ThreadPoolExecutor(max_workers=concurrency)
        futures = {}
        next_submit = 0
        next_yield = 0
        buffered = {}
        completed = 0
        window = max(1, concurrency * 2)
        try:
            while next_submit < len(playlist_urls) and len(futures) < window:
                future = executor.submit(fetch, playlist_urls[next_submit])
                futures[future] = next_submit
                next_submit += 1
            while futures:
                completed_future = next(as_completed(tuple(futures)))
                index = futures.pop(completed_future)
                buffered[index] = completed_future.result()
                while next_yield in buffered:
                    data = buffered.pop(next_yield)
                    completed += len(data)
                    if progress_callback is not None:
                        progress_callback(format_transfer_progress(completed, None, 0))
                    yield data
                    next_yield += 1
                while next_submit < len(playlist_urls) and len(futures) < window:
                    future = executor.submit(fetch, playlist_urls[next_submit])
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
    lines = [line.strip() for line in manifest.splitlines() if line.strip()]
    variants = [(line, lines[index + 1]) for index, line in enumerate(lines[:-1])
                if line.startswith("#EXT-X-STREAM-INF:") and not lines[index + 1].startswith("#")]
    if variants:
        audio = [item for item in variants if "AUDIO=" in item[0].upper()]
        uri = max(audio or variants, key=lambda item: item[0].count("BANDWIDTH="))[1]
        child_url = urljoin(base_url, uri)
        return _parse_hls_playlist(fetch(child_url).decode("utf-8"), child_url, fetch)
    result = []
    for line in lines:
        if line.startswith("#EXT-X-MAP:"):
            uri = _hls_attribute(line, "URI")
            if uri:
                result.append(urljoin(base_url, uri))
        elif not line.startswith("#"):
            result.append(urljoin(base_url, line))
    if not result:
        raise RangePrefetchError("HLS manifest contains no media segments")
    return result
