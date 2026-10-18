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
                   is_twitch_platform, pause_or_cancel)
from proglog import default_bar_logger
from remote_media import MediaSource, stable_source_id
from remote_prefetch import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONCURRENCY,
    RangePrefetchError,
    iter_range_bytes,
    supports_range_prefetch,
)
from progress import format_block_progress, format_resume_progress

SAMPLE_RATE = 32000
is_windows = sys.platform.startswith('win')

DEFAULT_STALL_TIMEOUT = 60.0
# 看门狗按"多少秒没有**字节**到达"计时，所以读取线程要按小块读（read1）：一次
# read(chunk_size) 会一直等到**整块** PCM 到齐，那样 60 秒的超时实际是在要求一个
# 600 秒块（约 9.6 MB 压缩音频）在 60 秒内传完，也就是持续 ≥1.3 Mbit/s；慢一点
# 的线路每一块都会被误判成"URL 过期"而整源跳过（round 47 审计 P1-4）。
_STALL_READ_CHUNK = 1 << 16
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


def _cached_audio_input(file, cache_store) -> bool:
    """True when the input is AutoComper's own cached audio file.

    Such a file is stable across runs (keyed by source identity + audio format), which is why
    it is the one local input that can be length-checked and self-healed: a cached audio that
    decodes short was cut off during download, while a user's own local file must never be
    deleted by us.
    """
    if cache_store is None or isinstance(file, MediaSource):
        return False
    try:
        root = os.path.realpath(str(cache_store.audio_root))
        path = os.path.realpath(str(file))
        if not root:
            return False
        return os.path.commonpath([root, path]) == root
    except (OSError, ValueError):
        return False


def can_start_at_block(file) -> bool:
    """Whether ``load_audio`` can begin at an arbitrary block for this source.

    Only the Bilibili loader fetches every block with its own ranged request, so only it can
    skip the blocks a resume has already finished. Every other loader consumes one ordered
    stream and must read from the beginning.

    Measured on a real 3.38 h Bilibili VOD (BV1434y1T77d, 21 blocks of 600 s): reading blocks
    0..4 in order took 212 s (42 s/block average, Bilibili throttles a single connection),
    while fetching block 4 directly took 7.3 s - and the two payloads were **byte-identical**
    (38,399,780 bytes, same sha256), because the resumed block is the very same ranged request
    either way. So the fast path cannot change the audio; it only stops downloading and
    discarding the finished blocks (a 34-block resume saved ~24 minutes in that measurement).
    """
    if not isinstance(file, MediaSource) or str(file.platform).lower() != "bilibili":
        return False
    try:
        duration = _get_audio_duration(file)
    except Exception:                                          # noqa: BLE001
        return False
    return duration is not None and duration > 0


def load_audio(file: str | MediaSource, sr: int, frame_count: int,
               prefetch_chunk_size=DEFAULT_CHUNK_SIZE,
               prefetch_concurrency=DEFAULT_CONCURRENCY, progress_callback=None,
               refresh_func=None, select_candidate_func=None, start_block=0):
    """Yield audio blocks for a source.

    ``start_block`` is honoured only by the loader that can seek per block (Bilibili, see
    ``can_start_at_block``); the stream-based loaders ignore it and always start at 0, so a
    caller must keep skipping the leading items itself for those.
    """
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
                select_candidate_func, start_block=start_block
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

    The watchdog times **bytes arriving from ffmpeg**, so ``stall_timeout`` really means
    "the connection has been silent for N seconds". Timing completed blocks instead would
    turn a healthy but slow line into a failure: a block is ``chunk_size = frame_count * 2``
    bytes (600 s of audio is about 9.6 MB compressed), so 60 s per block demands a sustained
    ~1.3 Mbit/s, and Bilibili's per-block fetch makes every block pay it (round 47 audit
    P1-4: sources were skipped with "remote URL likely expired" on slow lines).

    The queue stays **bounded** (maxsize=2): the consumer runs a 527-class model per block
    (~10x realtime measured) while ffmpeg decodes orders of magnitude faster, so an
    unbounded queue would decode the whole stream into memory (3 h VOD ~690 MB, 10 h ~2.3 GB).
    The reader loop is also cancellable, so an abandoned read cannot leave a thread blocked
    in ``put`` holding two blocks of PCM forever (round 47 audit, the per-cancel leak).
    """
    import queue as _queue
    import threading as _threading

    items = _queue.Queue(maxsize=2)
    stopped = _threading.Event()
    last_byte_at = [time.monotonic()]

    def put(item):
        """Enqueue without ever blocking forever, so a cancel always wins."""
        while not stopped.is_set():
            try:
                items.put(item, timeout=0.25)
                return True
            except _queue.Full:
                continue
        return False

    def reader():
        buffer = bytearray()
        try:
            while not stopped.is_set():
                # read1 returns as soon as *some* bytes are available (a plain read() waits
                # for the whole chunk), which is what lets the watchdog time byte silence.
                data = process.stdout.read1(_STALL_READ_CHUNK)
                if not data:
                    break
                last_byte_at[0] = time.monotonic()
                buffer.extend(data)
                while len(buffer) >= chunk_size:
                    if not put(bytes(buffer[:chunk_size])):
                        return
                    del buffer[:chunk_size]
            if buffer and not stopped.is_set():
                put(bytes(buffer))
            put(None)
        except BaseException as exc:  # noqa: BLE001 - surface any read error
            if not stopped.is_set():
                put(exc)

    thread = _threading.Thread(target=reader, name="audio-reader", daemon=True)
    thread.start()
    cancelled = False
    try:
        while True:
            try:
                item = items.get(timeout=0.25)
            except _queue.Empty:
                if time.monotonic() - last_byte_at[0] > stall_timeout:
                    process.kill()
                    raise RemoteAudioStallError(
                        f"No audio bytes for {stall_timeout:g}s; the remote URL may have "
                        f"expired or the connection hung")
                continue
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    except GeneratorExit:
        cancelled = True
        stopped.set()
        process.terminate()
        process.wait()
        thread.join(timeout=1)
        return
    finally:
        stopped.set()
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
                                select_candidate_func=None, start_block=0):
    """Fetch one ranged request per block, optionally starting at ``start_block``.

    Each index is its own ffmpeg call with `-ss index * block_duration`, so a resume can begin
    at the block it stopped at: the finished blocks' requests are simply never issued, and the
    resumed block's audio is byte-identical to what the sequential read would have produced.
    """
    block_duration = frame_count / sr
    # 防御性取整：block_size 正常来自 GUI 的 IntVar，但预设文件/调用方可能给 float，
    # 而切片和 range() 都要求整数（否则一个 TypeError 会被下面的重试阶梯放大成几十秒）。
    frame_count = int(frame_count)
    block_count = _duration_block_count(duration, block_duration)
    start_block = max(0, min(int(start_block or 0), block_count))
    attempts = scaled_retry_attempts(duration)
    # 连续失败达到阈值才切换 CDN（CDN 探测有成本，频繁探测反而触发限流）。
    # 每次失败先 refresh（换签名），连续失败说明不是签名问题而是节点慢。
    cdn_switches = 0
    consecutive_failures = 0
    for index in range(start_block, block_count):
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
                if not block:
                    # A range request that returns nothing means the audio stream ends here.
                    # Bilibili's metadata duration is routinely a little longer than the real
                    # audio track, so the last (sometimes the last two) block indices point past
                    # EOF. That is the end of the stream, not a failure: stop asking for more
                    # instead of burning the retry ladder on ranges that cannot exist.
                    print(f"Remote audio ended before block {index + 1}/{block_count}; "
                          f"continuing with what the stream has.")
                    return
                # 一个索引只产出**一个**元素，并且不超过一块的采样数。以前这里按
                # frame_count 切片（range(0, len(block), frame_count * 2)），一次读多了
                # 就会产出两个元素：消费方按"一个元素 = 一整块 block_size"推进时间轴，
                # 于是 UI 显示 "Block: 14 / 13"、后面所有时间戳整体后移一个 block
                # （Twitch 那条注释警告过的"重复块数 x block_size"家族）。读多了就截断。
                yield bytes(block[:frame_count * 2])
                last_error = None
                consecutive_failures = 0
                break
            except Exception as exc:
                last_error = exc
                consecutive_failures += 1
                if attempt >= attempts - 1:
                    # 注意：这里**不能**把"最后一块报错"当成流结束。空读是"音频到此为止"
                    # 的硬证据（ffmpeg 越过 EOF 会写出 0 字节且 rc=0）；而 403/超时/解码
                    # 失败只说明这一刻没拿到数据，静默当成结束会把真实音频丢掉，甚至让
                    # 单块源"成功检测出 0 个片段"。报错照旧上抛，让上层按源失败处理。
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
        # 取消/放弃时用它把卡在 put 上的 worker 放出来：否则线程会带着一整块 PCM
        # （600 秒块约 106 MB）永久停在那里（round 48 终检发现）。
        import threading as _threading
        self._stopped = _threading.Event()

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
                    if self._closed or self._stopped.is_set():
                        break
                    if not self._put(block):
                        return
            except BaseException as exc:              # noqa: BLE001 - 原样交给消费方
                self._error = exc
            finally:
                if not self._stopped.is_set():
                    self._put(None)

        self._thread = _threading.Thread(target=worker, name="audio-lookahead", daemon=True)
        self._thread.start()

    def _put(self, item):
        """入队但绝不无限等待：消费者放弃/取消后必须能退出（见 _stopped 的注释）。"""
        while not self._stopped.is_set():
            try:
                self._queue.put(item, timeout=0.25)
                return True
            except Exception:                         # noqa: BLE001 - queue.Full
                continue
        return False

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
        # 先放行卡在 put 上的 worker，再关底层生成器并等它退出：
        # 否则消费方放弃时线程会带着一整块 PCM 永远停在那次 put 上。
        self._stopped.set()
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
                   select_candidate_func=None, prefetch_concurrency=None,
                   prefetch_chunk_size=None, prepare_source_func=None):
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

    # 读前准备（目前是"逐源挑选最快的音频 CDN 候选"）：只有两个缓存都没命中、确实要读
    # 音频时才做。以前调用方在调用本函数**之前**就无条件探测，于是每个"已有检测结果"的
    # 源也要白测一遍候选（每个候选 5 秒超时 + 1 秒限流），几百个源的批次就是十几分钟的
    # 无用网络活动；失败只记一行日志，加载器会退回默认 URL（与旧行为一致）。
    if prepare_source_func is not None:
        try:
            prepare_source_func(file)
        except Exception as exc:                                  # noqa: BLE001
            print(f"Audio candidate probe skipped: {type(exc).__name__}")

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
    _dur = _get_audio_duration(file)
    block_count = 1
    if _dur is not None:
        block_count = (_duration_block_count(_dur, block_size)
                       if is_remote else max(1, int(_dur / block_size) + 1))
        if is_remote:
            _log_remote_progress(logger, _dur, block_size)

    info = {'filename': file, 'timestamps': []}
    frame_count = SAMPLE_RATE * block_size
    processed_blocks = 0
    started_at = time.monotonic()

    # 续跑：上次中断前已完成的块直接复用，不重复推理。只对远端源做（本地文件没有
    # 中断/过期这回事，且每次都会变），并且检查点里的时长必须与本次一致。
    resume_args = None
    resumed_blocks = 0
    skip_blocks = 0
    if is_remote and cache_store is not None:
        resume_args = _detection_cache_args(file, model, precision, block_size, threshold, focus_idx)
        checkpoint = _load_checkpoint(cache_store, resume_args, _dur, block_size)
        if checkpoint is not None:
            resumed_blocks = min(checkpoint["blocks_done"], max(0, block_count - 1))
            if resumed_blocks > 0:
                info['timestamps'] = list(checkpoint["timestamps"])
                offset = resumed_blocks * block_size
                processed_blocks = resumed_blocks
                # B 站按块取流，可以直接从恢复的那一块开始读，不用把已完成的块重下一遍；
                # 其他平台是单条顺序流，只能读完再用 continue 丢掉（下面循环里那一支）。
                # 实测（BV1434y1T77d，3.38 h）：顺序读 5 块 212 秒 vs 直取第 5 块 7.3 秒，
                # 两者音频逐字节相同 —— 因为本来就是同一个 ranged 请求。
                skip_blocks = resumed_blocks if can_start_at_block(file) else 0
                print(f"Remote Stream resumed: {resumed_blocks}/{block_count} blocks "
                      f"already done, continuing from block {resumed_blocks + 1}")
                if skip_blocks:
                    print(f"  Fast resume: the {skip_blocks} finished block(s) are skipped, "
                          f"not re-read (this platform fetches every block on its own).")
                else:
                    # 明说"会重扫一遍"：已完成的块不重新推理，但音频必须按序再读一遍，
                    # 对长 VOD 就是几分钟到几十分钟的静默期。以前什么都不说，用户看到
                    # 的就是"卡在这一步"（测试者实测约 10 分钟）。见 round 47。
                    reread_hours = (resumed_blocks * block_size) / 3600.0
                    print(f"  Note: the audio for those {resumed_blocks} finished block(s) "
                          f"(about {reread_hours:.1f} h) is read again before detection "
                          f"continues. The model is not re-run and nothing is lost, but this "
                          f"can take a while on a long VOD; the panel shows "
                          f"\"Re-scan: re-reading already-detected audio\" with an ETA.")

    blocks = load_audio(
        file, SAMPLE_RATE, SAMPLE_RATE * block_size,
        prefetch_chunk_size=(prefetch_chunk_size if prefetch_chunk_size
                             else DEFAULT_CHUNK_SIZE),
        prefetch_concurrency=(prefetch_concurrency if prefetch_concurrency
                              else DEFAULT_CONCURRENCY),
        progress_callback=progress_callback if is_remote else None,
        refresh_func=refresh_func if is_remote else None,
        select_candidate_func=select_candidate_func if is_remote else None,
        start_block=skip_blocks,
    )
    if _dur is not None:
        blocks = _SizedIterable(blocks, max(1, block_count - skip_blocks))
    else:
        blocks = _SizedIterable(blocks, 1)

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
        for block_index, block in enumerate(blocks, skip_blocks + 1):
            if block_index <= resumed_blocks:
                # 只有"单条顺序流"的平台会走到这里（B 站已由 start_block 跳过它们）：
                # 这些块必须按序读出来才能到达恢复点，读出来直接丢掉。
                if is_remote and progress_callback is not None:
                    progress_callback(
                        format_resume_progress(
                            block_index, resumed_blocks, time.monotonic() - started_at,
                            source_duration=_dur,
                        )
                    )
                continue          # 已完成的块：音频照读（要按序解码），但不重复推理
            # 任务边界（按块）：暂停在这里等待，取消优先。这一刻没有子进程正在被
            # 读取——音频预读线程最多再往前读一块就会阻塞在有界队列上，所以不会
            # 触发"长时间无输出"看门狗；暂停期间这一块只被取出、尚未推理。
            if pause_or_cancel():
                raise InterruptedError("Detection cancelled by user.")
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
        missing = block_count - processed_blocks
        if missing == 1:
            # 元数据时长通常比音频流本身略长（Bilibili 的音频轨尤其常见），所以最后一块
            # 取不到是"流自然结束"，不是损坏。整源按已完成处理：否则这个源每次运行都必然
            # 被判 incomplete 而跳过，检查点又停在 block_count-1，形成永远撞同一面墙的
            # 重试循环（见 round 47 的复现测试）。
            print(f"Remote Stream audio ended one block early "
                  f"({processed_blocks}/{block_count}); treating the source as complete.")
        elif missing > 0:
            message = (
                f"Remote Stream incomplete: processed {processed_blocks}/{block_count} blocks "
                f"({file.display_name or file.source_id or 'remote source'})"
            )
            print(f"Remote Stream incomplete: processed {processed_blocks}/{block_count} blocks")
            raise RemoteAudioIncompleteError(message)
        elif processed_blocks > block_count:
            print(f"Remote Stream completed: {block_count}/{block_count} blocks "
                  f"({processed_blocks - block_count} extra tail block(s) past the metadata "
                  f"duration)")
        else:
            print(f"Remote Stream completed: {processed_blocks}/{block_count} blocks")
    elif _dur is not None and _cached_audio_input(file, cache_store):
        # 缓存音频过去完全不查长度：截断的 m4a（容器时长照旧报完整、mdat 提前结束）会被
        # 当成"检测完成"，缺失的那段片段永远不出现，而且坏缓存会被一直复用。本地分支的
        # block_count 自带 +1 余量（int(dur/block)+1），所以"少 1 块"是正常尾巴，
        # 少 2 块以上才算截断，抛 AudioDecodeError 走既有的损坏缓存自愈分支。
        missing = block_count - processed_blocks
        if missing > 1:
            raise AudioDecodeError(
                -1, [str(file)],
                stderr=(f"cached audio is short: decoded {processed_blocks}/"
                        f"{block_count} blocks"))
        if missing == 1:
            print(f"Cached audio ended one block early ({processed_blocks}/{block_count}); "
                  f"treating the source as complete.")
    elif _dur is not None and block_count - processed_blocks > 1:
        # 用户自己的本地文件：只报告，绝不删除（那不是我们的缓存）。
        print(f"Audio file is shorter than its container duration: processed "
              f"{processed_blocks}/{block_count} blocks ({file})")

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
