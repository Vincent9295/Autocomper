#!/usr/bin/env python
import hashlib
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
import onnxruntime as ort
from typing import Generator, Any, Dict, Tuple
from collections import OrderedDict

from utils import (FFMPEG_PATH, run_tracked, register_proc, unregister_proc,
                   is_twitch_platform)
from proglog import default_bar_logger
from remote_media import MediaSource, stable_source_id
from remote_prefetch import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONCURRENCY,
    RangePrefetchError,
    iter_range_bytes,
    supports_range_prefetch,
)
from progress import format_block_progress

SAMPLE_RATE = 32000
is_windows = sys.platform.startswith('win')

DEFAULT_STALL_TIMEOUT = 60.0
_BASE_RETRY_ATTEMPTS = 5
_MAX_RETRY_ATTEMPTS = 12
_RETRY_BACKOFF = (2, 5, 10, 20, 40)


def scaled_retry_attempts(duration=None, base=_BASE_RETRY_ATTEMPTS,
                          maximum=_MAX_RETRY_ATTEMPTS):
    """Retry budget grows with source length: 5 + hours, capped at 12.

    Large VODs (many blocks/segments) hit more transient CDN failures, so a
    fixed small retry count is not enough; small files stay snappy.

    非有限值（inf/NaN）按 0 处理：JSON 往返会带上 Infinity 字面量（见
    remote_prefetch._scaled_range_attempts 里同一处坑），而 ``int(inf)`` 抛的是
    **OverflowError**——不在下面的 except 里，会直接炸掉整次检测（一个字节都没读）。
    """
    base = max(1, int(base))
    maximum = max(base, int(maximum))
    try:
        value = float(duration)
        hours = max(0.0, value / 3600.0) if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        hours = 0.0
    return min(maximum, base + int(hours))


def retry_backoff(attempt, delays=_RETRY_BACKOFF, cap=60.0):
    index = max(0, int(attempt))
    if index < len(delays):
        return float(delays[index])
    return float(cap)


class RemoteAudioIncompleteError(Exception):
    """Remote audio ended before all blocks implied by its known duration."""


class RemoteAudioStallError(Exception):
    """A remote audio read produced no data for the stall timeout; URL may be
    expired or the network connection hung. Callers should refresh and retry."""


def _open_wav_writer(path: str, sample_rate: int = SAMPLE_RATE):
    """Open a WAV file for streaming 16-bit mono PCM appends."""
    import struct
    f = open(path, "wb")
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0, b"WAVE", b"fmt ", 16, 1, 1, sample_rate,
        sample_rate * 2, 2, 16, b"data", 0,
    )
    f.write(header)
    return f


def _finalize_wav(f, data_size: int):
    """Patch the RIFF/data sizes once all PCM has been appended."""
    import struct
    f.seek(4)
    f.write(struct.pack("<I", 36 + data_size))
    f.seek(40)
    f.write(struct.pack("<I", data_size))
    f.close()



def subsample(frame: np.ndarray, scale_factor: int) -> np.ndarray:
    subframe = frame[:len(frame) - (len(frame) % scale_factor)].reshape(-1, scale_factor)
    subframe_mean = subframe.max(axis=1)
    subsample = subframe_mean
    if len(frame) % scale_factor != 0:
        residual_frame = frame[len(frame) - (len(frame) % scale_factor):]
        residual_mean = residual_frame.max()
        subsample = np.append(subsample, residual_mean)
    return subsample


def get_segments(scores: np.ndarray, precision: int, threshold: float, offset: int):
    seq_iter = iter(np.where(scores > threshold)[0])
    try:
        seq = next(seq_iter)
        pred = scores[seq]
        segment = {'start': seq, 'end': seq, 'pred': pred}
    except StopIteration:
        return
    for seq in seq_iter:
        pred = scores[seq]
        if seq - 1 == segment['end']:
            segment['end'] = seq
            segment['pred'] = max(segment['pred'], pred)
        else:
            yield segment
            segment = {'start': seq, 'end': seq, 'pred': pred}
    yield segment


def compute_timestamps(framewise_output, precision, threshold, focus_idx, offset):
    if not (0 <= focus_idx < framewise_output.shape[1]):
        raise ValueError(f"focus_idx {focus_idx} out of range "
                         f"(model has {framewise_output.shape[1]} classes)")
    focus = framewise_output[:, focus_idx]
    subsampled_scores = subsample(focus, precision)
    segments = []
    for segment in get_segments(subsampled_scores, precision, threshold, offset):
        # 峰值帧 argmax 检查：focus 类必须是 527 类最高分。
        # 怪声音（假阳）常有竞争类更高 → suspect=True；真 burp 几乎总是 argmax
        # （实测：920/920 干净合集 + 5/5 噪声直播真 burp 全部 argmax==focus）。
        f0 = segment['start'] * precision
        f1 = min(framewise_output.shape[0], (segment['end'] + 1) * precision)
        peak = f0 + int(np.argmax(framewise_output[f0:f1, focus_idx]))
        top1_idx = int(np.argmax(framewise_output[peak, :]))
        peak_scores = framewise_output[peak, :]
        top1_score = float(peak_scores[top1_idx])
        runner_up = float(np.partition(peak_scores, -2)[-2])
        segments.append({
            'start': segment['start'] * precision / 100 + offset,
            'end': segment['end'] * precision / 100 + offset + 1,
            'pred': round(float(segment['pred']), 6),
            'suspect': top1_idx != focus_idx,
            'top1_idx': top1_idx,
            'top1_score': round(top1_score, 6),
            'runner_up': round(runner_up, 6),
        })
    return segments


def pad_array_if_needed(arr, desired_size, pad_value=0):
    current_size = arr.shape[0]
    if current_size < desired_size:
        padding_needed = desired_size - current_size
        return np.pad(arr, (0, padding_needed), "constant", constant_values=(pad_value,))
    return arr


def _source_input(source_or_file):
    if isinstance(source_or_file, MediaSource):
        if not source_or_file.audio_url:
            raise ValueError("MediaSource has no audio_url")
        return source_or_file.audio_url, source_or_file.audio_headers or source_or_file.http_headers
    return source_or_file, {}


def _format_http_headers(headers):
    values = []
    for key, value in headers.items():
        key = str(key)
        value = str(value)
        if any(char in key or char in value for char in ('\r', '\n')):
            raise ValueError("HTTP headers must not contain newline characters")
        values.append(f"{key}: {value}")
    return "\r\n".join(values) + ("\r\n" if values else "")


def build_audio_command(source_or_file, sample_rate, frame_count, output_pipe=True,
                        input_source=None, input_headers=None,
                        start_time=None, duration=None):
    """Build an FFmpeg argv list for local or remote audio input."""
    source, headers = _source_input(source_or_file)
    if input_source is not None:
        source = input_source
        headers = input_headers or {}
    command = [FFMPEG_PATH, '-hide_banner', '-loglevel', 'warning']
    if headers:
        command.extend(['-headers', _format_http_headers(headers)])
    if start_time is not None:
        command.extend(['-ss', str(start_time)])
    if duration is not None:
        command.extend(['-t', str(duration)])
    if _is_http_input(source):
        # 签名 URL 过期/CDN 半连接时 FFmpeg 会永久挂起而不退出：
        # 给远程网络输入加读写超时，让 FFmpeg 自行中止并暴露错误给上层重试。
        command.extend(['-rw_timeout', '60000000'])
    command.extend(['-i', source])
    if output_pipe:
        command.extend([
            '-filter_complex', '[0:a]aresample=32000:async=1,asetpts=PTS-STARTPTS,atempo=1,aformat=channel_layouts=stereo,pan=mono|c0=0.5*c0+0.5*c1[audio]',
            '-map', '[audio]', '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate), '-ac', '1', '-bufsize', '128k', '-'
        ])
    return command


def _is_http_input(source) -> bool:
    return str(source).startswith(("http://", "https://"))


def load_audio(file: str | MediaSource, sr: int, frame_count: int,
               prefetch_chunk_size=DEFAULT_CHUNK_SIZE,
               prefetch_concurrency=DEFAULT_CONCURRENCY, progress_callback=None,
               refresh_func=None, select_candidate_func=None):
    # Bilibili range responses can fail after earlier PCM has already been
    # yielded. Use the restartable direct FFmpeg path for Remote Stream.
    if (isinstance(file, MediaSource)
            and str(file.platform).lower() == "youtube"
            and supports_range_prefetch(file)):
        try:
            yield from _load_audio_prefetched(
                file, sr, frame_count, prefetch_chunk_size, prefetch_concurrency,
                progress_callback, refresh_func
            )
            return
        except RangePrefetchError as exc:
            if not getattr(exc, "can_fallback", True):
                raise
            # 签名 URL 可能已过期：刷新 source 后再走直接路径，避免拿过期 URL 重试。
            if refresh_func is not None:
                try:
                    updated = refresh_func(file)
                    if isinstance(updated, MediaSource) and updated is not file:
                        file.__dict__.update(updated.__dict__)
                except Exception:
                    pass
            logging.getLogger(__name__).warning("Remote memory prefetch fallback: %s", str(exc))
    if isinstance(file, MediaSource) and str(file.platform).lower() == "bilibili":
        duration = _get_audio_duration(file)
        if duration is not None and duration > 0:
            yield from _load_audio_bilibili_blocks(
                file, sr, frame_count, duration, progress_callback, refresh_func,
                select_candidate_func
            )
            return
    if isinstance(file, MediaSource) and is_twitch_platform(file.platform):
        # Twitch（含 yt-dlp 的 twitchvod/twitchstream 等 key）HLS 签名/片段 URL
        # 也会过期：给裸的 _load_audio_direct 补上停滞超时 + refresh 重试，
        # 避免静默卡死。平台判断必须用平台族：yt-dlp 不返回裸的 "twitch"。
        # 只用于重试次数缩放，直接用元数据时长，避免额外的 ffmpeg 探测。
        duration = file.duration if file.duration is not None else None
        yield from _load_audio_twitch_retry(
            file, sr, frame_count, duration, refresh_func
        )
        return
    if isinstance(file, MediaSource):
        # 其余远程源（如 YouTube 预取失败后的直接路径）：同样启用停滞超时，
        # 避免签名 URL 过期时 FFmpeg 永久挂起；本地文件仍走无超时路径。
        yield from _load_audio_direct(
            file, sr, frame_count, stall_timeout=DEFAULT_STALL_TIMEOUT
        )
        return
    # 字符串输入（既包括裸 URL，也包括 **Audio Cache 模式下的本地缓存 m4a 路径**）。
    # 两者都必须带停滞看门狗：
    #   * 数据在中间断掉（截断的 .m4a）时 ffmpeg 在坏点之后一个字节都不再输出，
    #     没有超时就会永久阻塞在 read() 上——实测一个截断的 Bilibili 缓存文件
    #     （容器 2158s、实际只解出 1010s）挂住 10 分钟以上，表现就是检测阶段卡在
    #     某个源不动、界面还"响应中"；
    #   * 缓存文件是本地文件，但它的内容来自网络下载，所以同样会截断。
    # 停滞时抛 AudioDecodeError 而不是 RemoteAudioStallError：后者在上层有自己的
    # 分支，只把该源标记为"跳过"，不会清理缓存文件；而 AudioDecodeError 会落到通用
    # 异常分支，那里对本地缓存音频会删除损坏的 m4a + 同名 json 并重新下载一次再重试
    # ——损坏的缓存由此自愈，而不是留在盘上让下次运行再撞一次。
    try:
        yield from _load_audio_direct(file, sr, frame_count,
                                      stall_timeout=DEFAULT_STALL_TIMEOUT)
    except RemoteAudioStallError as exc:
        raise AudioDecodeError(-1, [str(file)],
                               stderr=f"audio stalled: {exc}") from exc


def _load_audio_twitch_retry(source, sr, frame_count, duration, refresh_func=None):
    """Twitch HLS 读取 + 过期 URL 重试。

    **重试只能发生在"一个字节都还没产出去"的时候。** 旧实现无条件 `yield from`：
    中途 stall / URL 过期时，错误发生在已经产出若干块之后，重试会从字节 0 重新读，
    把消费方已经计入的 PCM **再喂一遍**——片段重复、之后每个片段的时间轴整体后移
    （重复块数 × block_size），而 `processed_blocks < block_count` 的完整性检查因为
    计数只增不减，完全看不到这个问题（实测：4 秒源 stall 后变成 6 块、片段报在
    0-4/1-5/…/5-9s）。已经产出过数据就如实抛错，交给上层按"这个源失败"处理。
    """
    attempts = scaled_retry_attempts(duration)
    for attempt in range(attempts):
        emitted = False
        try:
            for block in _load_audio_direct(
                source, sr, frame_count, stall_timeout=DEFAULT_STALL_TIMEOUT
            ):
                emitted = True
                yield block
            return
        except Exception:
            if emitted:
                # 已经交付过音频：重试会重复喂数据，宁可让上层看到失败。
                raise
            if attempt >= attempts - 1:
                raise
            time.sleep(retry_backoff(attempt))
            if refresh_func is not None:
                updated = refresh_func(source)
                if isinstance(updated, MediaSource) and updated is not source:
                    source.__dict__.update(updated.__dict__)


def _subprocess_options():
    subprocess_options = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE}
    if is_windows:
        subprocess_options['creationflags'] = subprocess.CREATE_NO_WINDOW
    return subprocess_options


class AudioDecodeError(subprocess.CalledProcessError):
    """FFmpeg exited non-zero while loading audio; carries the exit code and a
    stderr tail so the caller can see the real reason (403 / timeout / crash)."""

    def __str__(self):
        tail = getattr(self, "stderr", None)
        extra = ""
        if tail:
            text = tail[-1024:] if isinstance(tail, bytes) else str(tail)[-1024:]
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            extra = f"\n  ffmpeg stderr tail: {text.strip()}"
        return f"ffmpeg exited with status {self.returncode} while loading audio{extra}"


def _read_process_stderr(process, max_bytes=2048):
    """Collect up to ``max_bytes`` of process stderr without risking a hang
    (on Windows a grandchild holding the pipe handle can delay EOF)."""
    parts = []

    def _collect():
        try:
            while True:
                chunk = process.stderr.read(65536)
                if not chunk:
                    break
                parts.append(chunk)
        except Exception:
            pass

    thread = threading.Thread(target=_collect, daemon=True)
    thread.start()
    thread.join(timeout=2)
    return b"".join(parts)[-max_bytes:]


def _load_audio_direct(file, sr, frame_count, start_time=None, duration=None,
                       stall_timeout=None):
    cmd = build_audio_command(
        file, sr, frame_count, start_time=start_time, duration=duration
    )
    chunk_size = frame_count * 2
    process = subprocess.Popen(cmd, bufsize=1, **_subprocess_options())
    register_proc(process)
    try:
        if stall_timeout is not None and stall_timeout > 0:
            yield from _read_audio_with_stall(process, chunk_size, stall_timeout, cmd)
        else:
            yield from _read_audio_plain(process, chunk_size, cmd)
    finally:
        unregister_proc(process)


def _read_audio_plain(process, chunk_size, cmd):
    try:
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            yield chunk
    except GeneratorExit:
        process.terminate()
        process.wait()
        return
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise AudioDecodeError(return_code, cmd, stderr=_read_process_stderr(process))


def _read_audio_with_stall(process, chunk_size, stall_timeout, cmd):
    """Read FFmpeg stdout with a no-data stall watchdog.

    If no bytes arrive within ``stall_timeout`` seconds the process is killed
    and ``RemoteAudioStallError`` is raised so callers can refresh the (possibly
    expired) signed URL and retry instead of hanging forever.

    队列**必须有界**：消费者每个 block 要跑一次 527 类模型（实测约 10× 实时），
    而 ffmpeg 解码本地/远程音频可以快一两个数量级。无界队列会把整条流解码进内存
    （3 小时 VOD ≈ 690 MB，10 小时 ≈ 2.3 GB）。maxsize=2 时读取线程在消费者落后时
    自然阻塞，ffmpeg 的 stdout 管道被填满后也随之等待——看门狗语义不变（有积压时
    ``get(timeout=...)`` 立即返回，只有真的没数据才会超时）。
    """
    import queue as _queue
    import threading as _threading

    items = _queue.Queue(maxsize=2)

    def reader():
        try:
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    items.put(None)
                    break
                items.put(chunk)
        except BaseException as exc:  # noqa: BLE001 - surface any read error
            try:
                items.put(exc)
            except BaseException:  # noqa: BLE001 - consumer已经走了
                pass

    thread = _threading.Thread(target=reader, name="audio-reader", daemon=True)
    thread.start()
    cancelled = False
    try:
        while True:
            try:
                item = items.get(timeout=stall_timeout)
            except _queue.Empty:
                process.kill()
                raise RemoteAudioStallError(
                    f"No audio data for {stall_timeout:g}s; remote URL likely expired "
                    f"or the connection hung")
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    except GeneratorExit:
        cancelled = True
        process.terminate()
        process.wait()
        thread.join(timeout=1)
        return
    finally:
        if thread.is_alive():
            thread.join(timeout=1)
        try:
            process.stdout.close()
        except OSError:
            pass
        if cancelled:
            return
        return_code = process.wait()
        if return_code and not _exception_in_flight():
            raise AudioDecodeError(return_code, cmd, stderr=_read_process_stderr(process))


def _exception_in_flight() -> bool:
    import sys
    return sys.exc_info()[0] is not None


def _load_audio_bilibili_blocks(source, sr, frame_count, duration,
                                progress_callback=None, refresh_func=None,
                                select_candidate_func=None):
    block_duration = frame_count / sr
    block_count = _duration_block_count(duration, block_duration)
    attempts = scaled_retry_attempts(duration)
    # 连续失败达到阈值才切换 CDN（CDN 探测有成本，频繁探测反而触发限流）。
    # 每次失败先 refresh（换签名），连续失败说明不是签名问题而是节点慢。
    cdn_switches = 0
    consecutive_failures = 0
    for index in range(block_count):
        start_time = index * block_duration
        segment_duration = min(block_duration, max(0, duration - start_time))
        if segment_duration <= 0:
            break
        last_error = None
        for attempt in range(attempts):
            try:
                block = bytearray()
                for chunk in _load_audio_direct(
                    source, sr, frame_count,
                    start_time=start_time, duration=segment_duration,
                    stall_timeout=DEFAULT_STALL_TIMEOUT,
                ):
                    block.extend(chunk)
                for chunk_start in range(0, len(block), frame_count * 2):
                    yield bytes(block[chunk_start:chunk_start + frame_count * 2])
                last_error = None
                consecutive_failures = 0
                break
            except Exception as exc:
                last_error = exc
                consecutive_failures += 1
                if attempt >= attempts - 1:
                    raise
                time.sleep(retry_backoff(attempt))
                if refresh_func is not None:
                    updated = refresh_func(source)
                    if isinstance(updated, MediaSource) and updated is not source:
                        source.__dict__.update(updated.__dict__)
                # 连续失败且仍可切换 CDN → 重新探测候选组换节点
                if (select_candidate_func is not None
                        and consecutive_failures >= 3
                        and cdn_switches < 2):
                    cdn_switches += 1
                    consecutive_failures = 0
                    try:
                        select_candidate_func(source)
                    except Exception as exc:
                        logging.getLogger(__name__).warning(
                            "Block CDN switch probe failed: %s", str(exc))
        if last_error is not None:
            raise last_error


class _LookaheadBlocks:
    """Yield audio blocks while decoding the next one in the background.

    Detection is strictly serial today: read a block, run the model on it, read the next one. On a
    long VOD that means the network and the CPU take turns, even though the model needs roughly 3x
    real time while the audio download manages 100x - the link sits idle for most of the run.

    This wraps the block iterator in one background thread and a **bounded** queue (``depth``
    blocks, default 1): the worker keeps one block ahead, so inference of block N overlaps the
    download of block N+1. The bound is what keeps memory honest - a 600 s block of PCM is about
    106 MB, so depth 1 costs one extra block, and a slow consumer simply stops the reader.

    Ordering, error propagation and cleanup are preserved: the worker pulls blocks in order, an
    exception surfaces on the consumer (as before, at the point the block would have been read),
    and closing the wrapper closes the underlying generator so its ffmpeg process is terminated.
    """

    def __init__(self, blocks, depth=1):
        self._blocks = blocks
        self._depth = max(1, int(depth))
        self._queue = None
        self._thread = None
        self._error = None
        self._closed = False
        self._started = False

    def __len__(self):
        return len(self._blocks)

    def _start(self):
        if self._started:
            return
        self._started = True
        import queue as _queue
        import threading as _threading
        self._queue = _queue.Queue(maxsize=self._depth)

        def worker():
            try:
                for block in self._blocks:
                    if self._closed:
                        break
                    self._queue.put(block)            # 队列满就在这里等：读到的块数受 depth 限制
            except BaseException as exc:              # noqa: BLE001 - 原样交给消费方
                self._error = exc
            finally:
                try:
                    self._queue.put(None)
                except BaseException:
                    pass

        self._thread = _threading.Thread(target=worker, name="audio-lookahead", daemon=True)
        self._thread.start()

    def __iter__(self):
        self._start()
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def close(self):
        """Stop the reader and close the source iterator (which stops its ffmpeg).

        只对**支持 close 的**迭代器调用 close：包装对象可能只是普通迭代器（生成器有
        close，`_SizedIterable` 这类没有），强行调用会抛 AttributeError 把收尾搞崩。
        """
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._blocks, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:                                  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __del__(self):                                        # pragma: no cover - 兜底
        try:
            self.close()
        except Exception:                                      # noqa: BLE001
            pass


def _load_audio_prefetched(source, sr, frame_count, prefetch_chunk_size,
                           prefetch_concurrency, progress_callback=None,
                           refresh_func=None):
    cmd = build_audio_command(source, sr, frame_count, input_source="pipe:0")
    chunk_size = frame_count * 2
    process = subprocess.Popen(cmd, bufsize=1, stdin=subprocess.PIPE, **_subprocess_options())
    register_proc(process)
    writer_error = []

    def writer():
        try:
            for compressed in iter_range_bytes(
                source, chunk_size=prefetch_chunk_size, concurrency=prefetch_concurrency,
                progress_callback=progress_callback, refresher=refresh_func
            ):
                process.stdin.write(compressed)
        except Exception as exc:
            writer_error.append(exc)
        finally:
            try:
                process.stdin.close()
            except Exception:
                pass

    thread = threading.Thread(target=writer, name="youtube-memory-prefetch", daemon=True)
    thread.start()
    yielded = False
    try:
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            yielded = True
            yield chunk
        thread.join(timeout=1)
        if thread.is_alive():
            process.terminate()
            thread.join(timeout=1)
        process.stdout.close()
        return_code = process.wait()
        if writer_error:
            error = RangePrefetchError("prefetch writer failed")
            error.can_fallback = not yielded
            raise error
        if return_code:
            error = RangePrefetchError("prefetch ffmpeg failed")
            error.can_fallback = not yielded
            raise error
    except GeneratorExit:
        process.terminate()
        process.wait()
        thread.join(timeout=1)
        return
    except Exception as exc:
        process.terminate()
        process.wait()
        thread.join(timeout=1)
        if not yielded:
            if isinstance(exc, RangePrefetchError):
                raise exc
            error = RangePrefetchError("prefetch reader failed")
            error.can_fallback = True
            raise error from exc
        error = RangePrefetchError("prefetch stream failed")
        error.can_fallback = False
        raise error from exc
    finally:
        unregister_proc(process)


def hash_file(file_path, algorithm='sha256', chunk_size=8192) -> str:
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _get_audio_duration(file):
    """用 ffmpeg（非 ffprobe）快速探测时长。

    注意 `build_audio_command` 固定带 `-loglevel warning`，而 `Duration:` 是 info
    级输出——所以这里必须自己拼一条**带 info 日志**的探测命令，否则正则永远匹配
    不到，fallback 等于死代码（本地文件永远拿不到时长 → 进度只显示 1 个 block，
    而且远程源在没有 metadata 时长时连"读少了"的截断保护都会失效）。
    用 `-f null -` 让 ffmpeg 正常结束，避免"没有输出文件"导致的 rc=1。
    """
    if isinstance(file, MediaSource):
        try:
            duration = float(file.duration)
            if math.isfinite(duration) and duration >= 0:
                return duration
        except (TypeError, ValueError):
            pass
    try:
        input_source, headers = _source_input(file)
        cmd = [FFMPEG_PATH, '-hide_banner', '-nostdin']
        if headers:
            cmd += ['-headers', _format_http_headers(headers)]
        if _is_http_input(input_source):
            # 与 build_audio_command 一致：签名 URL 过期时不要让探测永久挂住
            cmd += ['-rw_timeout', '60000000']
        cmd += ['-i', str(input_source), '-f', 'null', '-']
        out = run_tracked(cmd, timeout=30, text=True)
        m = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', out.stderr or '')
        if m:
            h, mi, s, ms = map(int, m.groups())
            return h * 3600 + mi * 60 + s + ms / 100
    except Exception:
        pass
    return None


def _duration_block_count(duration, block_size):
    if duration is None:
        return 1
    return max(1, math.ceil(float(duration) / block_size))


def _log_remote_progress(logger, duration, block_size):
    block_count = _duration_block_count(duration, block_size)
    message = f"Remote Stream blocks: {block_count} (block size: {block_size}s)"
    print(message)
    if hasattr(logger, "log"):
        logger.log(message)
    elif callable(logger):
        logger(message)
    return block_count


class _SizedIterable:
    def __init__(self, gen, total):
        self._gen = gen
        self._total = total
    def __iter__(self):
        return self._gen
    def __len__(self):
        return self._total


MAX_CACHE_SIZE = 20
timestamps_dict: 'OrderedDict[Tuple[str, int, int, float, str], Dict[str, Any]]' = OrderedDict()

def _detection_cache_args(source, model, precision, block_size, threshold, focus_idx):
    """检测缓存与 .failed.json 的键。

    这里**故意**用裸 ``source.source_id``（不是 stable_source_id）：键同时决定磁盘上
    的缓存文件名，改成 URL 兜底会让升级后所有既有检测缓存失配——12 小时 VOD 会全部
    重新检测一遍，代价远大于"极少数没有 id 的源可能撞键"的风险。yt-dlp 对三大平台
    一律给出 id；只有元数据不完整的 PlaylistEntry 才会为空。
    """
    return (
        source.platform,
        source.source_id,
        model,
        str(precision),
        block_size,
        threshold,
        focus_idx,
        {},
    )


_CHECKPOINT_VERSION = 1
_CHECKPOINT_SUFFIX = ".progress.json"


def _checkpoint_path(cache_store, args):
    """按检测缓存同样的键取检查点路径（同目录、同哈希、不同后缀）。"""
    return cache_store.get_detection_cache_path(*args).with_suffix(_CHECKPOINT_SUFFIX)


def _save_checkpoint(cache_store, args, blocks_done, timestamps, duration, block_size):
    """把一个源已完成的块原子落盘，供中断后续跑。

    以前只有整源跑完才写结果（`save_detection_result`），所以一个 6 小时源在中途失败
    （Twitch 签名 URL 过期、网络抖动）会丢掉**全部**推理成果，重跑又从第 1 块开始。
    这里每完成一块就记录"已完成到第几块"以及累计的时间戳，重跑时跳过已完成的块。
    """
    if cache_store is None:
        return
    payload = {
        "version": _CHECKPOINT_VERSION,
        "source_id": args[1],
        "block_size": int(block_size),
        "duration": (round(float(duration), 3) if duration else None),
        "blocks_done": int(blocks_done),
        "timestamps": list(timestamps),
    }
    try:
        cache_store.save_json(_checkpoint_path(cache_store, args), payload)
    except Exception:                                            # noqa: BLE001
        # 检查点只是"省时间"的优化：写不进去不该让检测失败。
        pass


def _load_checkpoint(cache_store, args, duration, block_size):
    """读回可用的检查点；不可用/不匹配返回 None。

    时长不匹配就丢弃：源换了或长度变了，块边界不再对应同一段音频，硬续会错位。
    """
    if cache_store is None:
        return None
    try:
        data = cache_store.read_json(_checkpoint_path(cache_store, args))
    except Exception:                                            # noqa: BLE001
        return None
    if not isinstance(data, dict) or data.get("version") != _CHECKPOINT_VERSION:
        return None
    if data.get("source_id") != args[1] or int(data.get("block_size") or 0) != int(block_size):
        return None
    if duration is None:
        return None                        # 没有时长就没法确认块边界，宁可不续
    try:
        if abs(float(data.get("duration") or 0.0) - float(duration)) > 1.0:
            return None
    except (TypeError, ValueError):
        return None
    blocks_done = int(data.get("blocks_done") or 0)
    timestamps = data.get("timestamps") or []
    if blocks_done <= 0 or not isinstance(timestamps, list):
        return None
    return {"blocks_done": blocks_done, "timestamps": timestamps}


def _clear_checkpoint(cache_store, args):
    if cache_store is None:
        return
    try:
        _checkpoint_path(cache_store, args).unlink(missing_ok=True)
    except Exception:                                            # noqa: BLE001
        pass


def get_timestamps(file, precision=100, block_size=600, threshold=0.90, focus_idx=58,
                   model="bdetectionmodel_05_01_23", logger=None, ort_session=None,
                   use_gpu=True, cache_store=None, progress_callback=None,
                   refresh_func=None, save_audio_path=None,
                   select_candidate_func=None, prefetch_concurrency=None):
    # 必须是"正数"：0 会让 subsample 除零（precision=0 → ZeroDivisionError 卡在
    # 检测中途），block_size=0 会让 frame_count/chunk_size 归零（AudioDecodeError）。
    # GUI 的数字校验只挡非数字，用户手打 0 是可能的。
    if precision <= 0:
        raise Exception("Precision must be a positive number!")
    if not (threshold >= 0 and threshold <= 1):
        raise Exception("Threshold must be between 0 and 1!")
    if block_size <= 0:
        raise Exception("Block size must be a positive number!")

    is_remote = isinstance(file, MediaSource)
    if is_remote:
        cache_key = (stable_source_id(file), precision, block_size, threshold, model, focus_idx)
        if cache_store is not None:
            cached = cache_store.load_detection_result(
                *_detection_cache_args(file, model, precision, block_size, threshold, focus_idx)
            )
            if cached is not None:
                cached['filename'] = file
                return cached, True
    else:
        file_hash = hash_file(file)
        cache_key = (file_hash, precision, block_size, threshold, model, focus_idx)

    if ort_session is None and cache_key in timestamps_dict:
        previous_data = timestamps_dict[cache_key]
        previous_data['filename'] = file
        if logger:
            bar_logger = default_bar_logger(logger)
            _dur = _get_audio_duration(file)
            block_count = (_duration_block_count(_dur, block_size)
                           if is_remote else
                           (max(1, int(_dur / block_size) + 1) if _dur is not None else 1))
            if is_remote and _dur is not None:
                _log_remote_progress(logger, _dur, block_size)
            for _ in bar_logger.iter_bar(block=range(block_count)):
                if progress_callback is not None and is_remote:
                    progress_callback(format_block_progress(
                        _ + 1, block_count, 0, source_duration=_dur))
        return previous_data, True

    if ort_session is None:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if use_gpu:
            try:
                ort_session = ort.InferenceSession(model, sess_options,
                                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            except Exception:
                ort_session = ort.InferenceSession(model, sess_options,
                                                   providers=['CPUExecutionProvider'])
        else:
            ort_session = ort.InferenceSession(model, sess_options,
                                               providers=['CPUExecutionProvider'])

    offset = 0
    blocks = load_audio(
        file, SAMPLE_RATE, SAMPLE_RATE * block_size,
        prefetch_concurrency=(prefetch_concurrency if prefetch_concurrency
                              else DEFAULT_CONCURRENCY),
        progress_callback=progress_callback if is_remote else None,
        refresh_func=refresh_func if is_remote else None,
        select_candidate_func=select_candidate_func if is_remote else None,
    )
    _dur = _get_audio_duration(file)
    block_count = 1
    if _dur is not None:
        block_count = (_duration_block_count(_dur, block_size)
                       if is_remote else max(1, int(_dur / block_size) + 1))
        if is_remote:
            _log_remote_progress(logger, _dur, block_size)
        blocks = _SizedIterable(blocks, block_count)
    else:
        blocks = _SizedIterable(blocks, 1)

    info = {'filename': file, 'timestamps': []}
    frame_count = SAMPLE_RATE * block_size
    processed_blocks = 0
    started_at = time.monotonic()

    # 续跑：上次中断前已完成的块直接复用，不重复推理。只对远端源做（本地文件没有
    # 中断/过期这回事，且每次都会变），并且检查点里的时长必须与本次一致。
    resume_args = None
    resumed_blocks = 0
    if is_remote and cache_store is not None:
        resume_args = _detection_cache_args(file, model, precision, block_size, threshold, focus_idx)
        checkpoint = _load_checkpoint(cache_store, resume_args, _dur, block_size)
        if checkpoint is not None:
            resumed_blocks = min(checkpoint["blocks_done"], max(0, block_count - 1))
            if resumed_blocks > 0:
                info['timestamps'] = list(checkpoint["timestamps"])
                offset = resumed_blocks * block_size
                processed_blocks = resumed_blocks
                print(f"Remote Stream resumed: {resumed_blocks}/{block_count} blocks "
                      f"already done, continuing from block {resumed_blocks + 1}")

    if logger:
        bar_logger = default_bar_logger(logger)
        blocks = bar_logger.iter_bar(block=blocks)

    # 单块预读：推理第 N 块的同时，后台线程把第 N+1 块读好。只对远端源做——本地文件读取
    # 本来就不是瓶颈，而且多一个线程只会添乱。队列深度 1（多一块 PCM），见 _LookaheadBlocks。
    lookahead = None
    if is_remote:
        lookahead = _LookaheadBlocks(blocks, depth=1)
        blocks = lookahead

    wav_file = None
    wav_data_size = 0
    if is_remote and save_audio_path:
        wav_file = _open_wav_writer(save_audio_path)
    try:
        for block_index, block in enumerate(blocks, 1):
            if block_index <= resumed_blocks:
                continue          # 已完成的块：音频照读（要按序解码），但不重复推理
            processed_blocks = block_index
            if wav_file is not None:
                wav_file.write(block)
                wav_data_size += len(block)
            if is_remote:
                if progress_callback is not None:
                    progress_callback(
                        format_block_progress(
                            block_index, block_count, time.monotonic() - started_at,
                            source_duration=_dur,
                        )
                    )
            samples = np.frombuffer(block, dtype=np.int16)
            samples = pad_array_if_needed(samples, frame_count)
            samples = samples.reshape(1, -1)
            samples = samples / (2**15)
            samples = samples.astype(np.float32)
            ort_inputs = {"input": samples}
            framewise_output = ort_session.run(["output"], ort_inputs)[0]
            preds = framewise_output[0]
            info["timestamps"].extend(compute_timestamps(preds, precision, threshold, focus_idx, offset))
            offset += block_size
            # 每完成一块就落一次检查点：中途失败（URL 过期、网络抖动）时前面的推理
            # 成果不会丢，重跑从这一块之后继续。
            if resume_args is not None:
                _save_checkpoint(cache_store, resume_args, block_index,
                                 info["timestamps"], _dur, block_size)
    finally:
        if wav_file is not None:
            _finalize_wav(wav_file, wav_data_size)
        if lookahead is not None:
            lookahead.close()       # 收尾：停掉预读线程并关掉底层生成器（终止其 ffmpeg）


    if is_remote and _dur is not None:
        if processed_blocks < block_count:
            message = (
                f"Remote Stream incomplete: processed {processed_blocks}/{block_count} blocks "
                f"({file.display_name or file.source_id or 'remote source'})"
            )
            print(f"Remote Stream incomplete: processed {processed_blocks}/{block_count} blocks")
            raise RemoteAudioIncompleteError(message)
        print(f"Remote Stream completed: {processed_blocks}/{block_count} blocks")

    if is_remote and cache_store is not None:
        cache_result = dict(info)
        cache_result['filename'] = file.source_url
        cache_store.save_detection_result(
            *_detection_cache_args(file, model, precision, block_size, threshold, focus_idx),
            cache_result,
        )
        if resume_args is not None:
            _clear_checkpoint(cache_store, resume_args)   # 整源结果已落盘，检查点作废

    if len(timestamps_dict) >= MAX_CACHE_SIZE:
        timestamps_dict.popitem(last=False)
    timestamps_dict[cache_key] = info
    return info, False
