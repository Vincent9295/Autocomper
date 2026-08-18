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

from utils import FFMPEG_PATH, run_tracked, register_proc, unregister_proc
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
    """
    base = max(1, int(base))
    maximum = max(base, int(maximum))
    try:
        hours = max(0.0, float(duration) / 3600.0)
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
    if isinstance(file, MediaSource) and str(file.platform).lower() == "twitch":
        # Twitch HLS 签名/片段 URL 也会过期：给裸的 _load_audio_direct 补上
        # 停滞超时 + refresh 重试，避免静默卡死。
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
    yield from _load_audio_direct(file, sr, frame_count)


def _load_audio_twitch_retry(source, sr, frame_count, duration, refresh_func=None):
    attempts = scaled_retry_attempts(duration)
    for attempt in range(attempts):
        try:
            yield from _load_audio_direct(
                source, sr, frame_count, stall_timeout=DEFAULT_STALL_TIMEOUT
            )
            return
        except Exception as exc:
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
    """
    import queue as _queue
    import threading as _threading

    items = _queue.Queue()

    def reader():
        try:
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    items.put(None)
                    break
                items.put(chunk)
        except BaseException as exc:  # noqa: BLE001 - surface any read error
            items.put(exc)

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
    """用 ffmpeg（非 ffprobe）快速探测时长。"""
    if isinstance(file, MediaSource):
        try:
            duration = float(file.duration)
            if math.isfinite(duration) and duration >= 0:
                return duration
        except (TypeError, ValueError):
            pass
    try:
        cmd = build_audio_command(file, SAMPLE_RATE, 0, output_pipe=False)
        out = run_tracked(cmd, timeout=10, text=True)
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


def get_timestamps(file, precision=100, block_size=600, threshold=0.90, focus_idx=58,
                   model="bdetectionmodel_05_01_23", logger=None, ort_session=None,
                   use_gpu=True, cache_store=None, progress_callback=None,
                   refresh_func=None, save_audio_path=None,
                   select_candidate_func=None):
    if precision < 0:
        raise Exception("Precision must be a positive number!")
    if not (threshold >= 0 and threshold <= 1):
        raise Exception("Threshold must be between 0 and 1!")
    if block_size < 0:
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

    if logger:
        bar_logger = default_bar_logger(logger)
        blocks = bar_logger.iter_bar(block=blocks)

    wav_file = None
    wav_data_size = 0
    if is_remote and save_audio_path:
        wav_file = _open_wav_writer(save_audio_path)
    try:
        for block_index, block in enumerate(blocks, 1):
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
    finally:
        if wav_file is not None:
            _finalize_wav(wav_file, wav_data_size)


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

    if len(timestamps_dict) >= MAX_CACHE_SIZE:
        timestamps_dict.popitem(last=False)
    timestamps_dict[cache_key] = info
    return info, False
