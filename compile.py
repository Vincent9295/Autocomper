#!/usr/bin/env python
"""compile.py — native FFmpeg video/audio compilation."""

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from colorama import Fore, Style

from progress import format_compile_progress
from utils import (FFMPEG_PATH, run_tracked, run_tracked_progress,
                   cancel_pending)
import sys
import os

MERGE_THRESHOLD = 2  # seconds

# 音频输出编码组。concat 链条里的任何 AAC 输入都会让 concat filter 的音频时间线
# 相对视频每段拉伸 ~20ms（AAC priming 残留，实测真实远程段 A-V 中位数 +21.3ms，
# 940 段批次尾部累积近 20s——"整体逐渐错位"根因）。因此：
#   - 进入后续 concat 的所有中间产物（cut 临时 clip、_batchL* 批次文件）
#     必须写 FLAC（无编码延迟、时长精确）；
#   - 只有最终输出的那一次编码才用 AAC。
_FLAC_AUDIO = ['-c:a', 'flac']
_AAC_AUDIO_OUT = ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']

_FFMPEG_DETAIL_MAX = 400


def _sanitize_ffmpeg_detail(detail):
    """Trim a raw ffmpeg error block for safe, compact logging."""
    text = str(detail or "")
    text = re.sub(r"(https?://[^\s?]+)\?[^\s]*", r"\1?[redacted]", text)
    if len(text) > _FFMPEG_DETAIL_MAX:
        text = "..." + text[-_FFMPEG_DETAIL_MAX:]
    return text


def _source_key(filename, entry=None):
    metadata = (entry or {}).get("source_metadata") or {}
    if metadata.get("source_id"):
        return f"{str(metadata.get('platform') or 'unknown').lower()}:{metadata['source_id']}"
    return os.path.normcase(os.path.normpath(str(filename)))

# 输入 -ss 在 MPEG-TS（尤其直播录像，PTS 不连续）上按字节估算落点，
# 视频可能落在目标数秒之外（实测 +3.5s），而音频落点正确。
# 此时 -vsync cfr 会用首帧填满 0..pts 空隙，-shortest 又在音频长度处截断，
# 导致整段 clip 画面冻结（用户报"corrupted"）。
# 对策：输入多回退 _SEEK_PAD 秒，再用输出 -ss/-to 精确定位（PTS 相对值不变）。
_SEEK_PAD = 10.0  # seconds


def _run_ffmpeg(command, timeout, progress_callback=None, duration=None, stage="FFmpeg"):
    if progress_callback is None:
        return run_tracked(command, timeout=timeout, text=True)

    def report(current, total, elapsed):
        progress_callback(format_compile_progress(current, total, elapsed, stage))

    return run_tracked_progress(
        command, duration=duration, timeout=timeout, progress_callback=report
    )


def clip_safe_end(duration, entry=None):
    """Return the usable end time without shortening generated remote clips."""
    metadata = (entry or {}).get('source_metadata') or {}
    if metadata.get('materialized_remote_segment'):
        return duration
    return max(0, duration - 0.5)


_PROBE_CACHE = {}


def _ffprobe(input_file: str):
    """ffmpeg -i 探测容器元数据（包内无 ffprobe）。

    -probesize/-analyzeduration 限制探测读取量，加快大批量片段的容器解析。
    """
    key = os.path.normcase(os.path.normpath(str(input_file)))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    try:
        out = run_tracked([FFMPEG_PATH, '-hide_banner', '-i', input_file,
                           '-probesize', '32M', '-analyzeduration', '100M'],
                          timeout=10, text=True)
        result = out.stderr or ''
    except Exception:
        result = ''
    _PROBE_CACHE[key] = result
    return result


def _get_video_size(input_file: str):
    """Return (width, height) of the first video stream, or (None, None)."""
    stderr = _ffprobe(input_file)
    m = re.search(r'Stream #0:\d+.*Video:.*?(\d{2,})x(\d{2,})', stderr)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _get_frame_rate(input_file: str):
    """Return frame rate (fps) of the first video stream, or default 30."""
    stderr = _ffprobe(input_file)
    m = re.search(r'(\d+(?:\.\d+)?)\s*fps', stderr)
    if m:
        return float(m.group(1))
    return 30.0


def _mixed_resolution_target(file_list):
    """混合分辨率检测（_ffmpeg_concat / _ffmpeg_concat_batched 共用）。

    返回目标 (w, h)；无混合或全部探测失败时返回 None。目标取出现次数最多的
    尺寸。显著低于目标的输入会被放大拼接——放大无法恢复细节，成片里这些
    片段偏软是源本身分辨率/码率不足（Bilibili 部分回放只给 360x640 低码率流），
    不是编码问题；日志明确点名，避免误判为编译退化。
    """
    sizes = {}
    for fp in file_list:
        w, h = _get_video_size(fp)
        if w and h:
            sizes.setdefault((w, h), []).append(os.path.basename(fp))
    if not sizes:
        print("  No resolvable input resolutions; keeping source size.")
        return None
    if len(sizes) == 1:
        return None
    size_text = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
    print(f"{Fore.YELLOW}Mixed resolutions ({size_text}) -> re-encoding for sync...")
    # 目标取出现次数最多的尺寸；平票取面积更大的（宁可放大低清源，
    # 也不要把原生高分辨率内容降采样丢细节）。
    target = max(sizes, key=lambda k: (len(sizes[k]), k[0] * k[1]))
    print(f"  Target resolution: {target[0]}x{target[1]}")
    area = target[0] * target[1]
    upscaled = [name for (w, h), names in sizes.items()
                if w * h * 2 < area for name in names]
    if upscaled:
        preview = ", ".join(upscaled[:3]) + ("..." if len(upscaled) > 3 else "")
        print(f"{Fore.YELLOW}  NOTE: {len(upscaled)} clip(s) are much lower-resolution "
              f"sources and will be upscaled; their softness is a source limitation, "
              f"not an encoding issue: {preview}{Style.RESET_ALL}")
    return target


def _get_video_duration(input_file: str):
    """Return total duration of the media file in seconds, or None."""
    stderr = _ffprobe(input_file)
    m = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', stderr)
    if m:
        h, mi, s, ms = map(int, m.groups())
        return h * 3600 + mi * 60 + s + ms / 100
    return None


# ═══ VIDEO cut / concat ═══════════════════════════════════════════════

_VIDEO_CODEC_CACHE = None

def get_video_codec():
    """NVENC 可用则用，否则回退 libx264。结果缓存。"""
    global _VIDEO_CODEC_CACHE
    if _VIDEO_CODEC_CACHE is not None:
        return list(_VIDEO_CODEC_CACHE)
    # 必须显式质量控制：无 -b:v/-cq 时 nvenc 默认码率极低（实测 1080p 仅 ~570 kb/s），
    # 高动态画面严重糊化/块状损坏。CQ 模式 + maxrate 封顶，兼顾画质与体积。
    nvenc = ['-c:v', 'h264_nvenc', '-preset', '3', '-pix_fmt', 'yuv420p',
             '-rc', 'vbr', '-cq', '20', '-b:v', '0',
             '-maxrate', '12M', '-bufsize', '24M',
             '-rc-lookahead', '20', '-sar', '1:1']
    x264 = list(_X264_CODEC)
    # 探针帧必须 >= 256x144：新款 GPU（如 RTX 50 系）NVENC 最小编码尺寸 > 64x64，
    # 用 64x64 探测会误报 "Frame Dimension less than the minimum supported value"。
    try:
        r = run_tracked([FFMPEG_PATH, '-hide_banner', '-loglevel', 'error',
                         '-f', 'lavfi', '-i', 'color=black:s=256x144:d=0.1']
                        + nvenc + ['-f', 'null', '-'], timeout=15)
        _VIDEO_CODEC_CACHE = nvenc if r.returncode == 0 else x264
    except Exception:
        _VIDEO_CODEC_CACHE = x264
    if _VIDEO_CODEC_CACHE is x264:
        # 正常回退：质量与 NVENC 一致（CRF18），仅速度较慢
        print(f"{Fore.YELLOW}NVENC unavailable, using libx264 (CPU). "
              f"This is fine - same output quality, just slower.{Style.RESET_ALL}")
    return list(_VIDEO_CODEC_CACHE)


_X264_CODEC = ['-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
               '-crf', '18', '-maxrate', '12M', '-bufsize', '24M', '-sar', '1:1']

# ── 逐段 A=V 时长强制相等（修 "next sound starts early" 的累积漂移）──
# Twitch 段实测：视频帧时间线 = req + 17~33ms（HLS 头偏移 + 帧粒度），
# 音频内容 = req − 45ms（priming 编辑）→ 每段 A−V ≈ −78ms，concat 逐段
# 累积 → 视频时间轴越来越落后于音频（"画面移后/转场提前"）。
# 修复：copy_cut 音频 asetpts 重基 + atrim 截尾 + apad 补齐到视频时长；
# re-encode 同理（CFR 视频 = eff，音频 atrim+apad 到 eff）→ 每段 A=V
# 精确相等，漂移归零。Bilibili 段容器 V=req 无头偏移，本就无此问题
# （与测试者"只影响 Twitch"的观察一致）。
_TAIL_WIN_S = 0.04                 # 包络窗 40ms
_TAIL_QUIET_FLOOR = 90.0           # "可闻"判定绝对下限（8k RMS int16 量纲）
_TAIL_QUIET_RATIO = 0.06           # "可闻"相对阈值：区域峰值 × 6%
_TAIL_FADE_S = 0.1                 # 截断处淡出时长


def _decode_audio_region(path, offset, duration, sr=8000):
    """解码 [offset, offset+duration) 的单声道音频，返回 float32 采样；
    失败返回 None。"""
    cmd = [FFMPEG_PATH, '-v', 'error', '-ss', f'{offset:.3f}',
           '-t', f'{duration:.3f}', '-i', path, '-map', '0:a:0?',
           '-ac', '1', '-ar', str(sr), '-f', 's16le', '-']
    r = subprocess.run(cmd, capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32)


def tail_audible_end_seconds(samples, sr,
                             quiet_floor=_TAIL_QUIET_FLOOR,
                             quiet_ratio=_TAIL_QUIET_RATIO,
                             fade_s=_TAIL_FADE_S):
    """垫后/尾部分析：最后一个"可闻窗"的末端 + fade 边距 → clip 的有效
    终点（秒，相对 samples[0]）。全部静音 → fade_s；退化输入 → None。"""
    samples = np.asarray(samples, dtype=np.float32)
    win = max(1, int(_TAIL_WIN_S * sr))
    total = len(samples) // win
    if total < 2:
        return None
    env = np.abs(samples[:total * win].reshape(total, win)).mean(axis=1)
    peak = float(env.max())
    floor = max(quiet_floor, peak * quiet_ratio)
    audible = [i for i, v in enumerate(env) if v >= floor]
    if not audible:
        return fade_s
    return (audible[-1] + 1) * win / sr + fade_s


def _fallback_to_x264():
    """NVENC 编码中途失败时，永久回退 libx264（更新缓存）。"""
    global _VIDEO_CODEC_CACHE
    _VIDEO_CODEC_CACHE = _X264_CODEC
    print(f"{Fore.YELLOW}NVENC encode failed; falling back to libx264 (CPU).{Style.RESET_ALL}")


def _ffmpeg_cut(input_file, timestamps, output_file, res=None, normalize=False,
                fps=None, preserve_duration=False, progress_callback=None,
                duration=None, batch_size=6):
    if not timestamps:
        return False

    duration = duration or _get_video_duration(input_file)
    if duration:
        safe_end = duration if preserve_duration else max(0, duration - 0.5)
        timestamps = [(max(0, s), min(e, safe_end)) for s, e in timestamps
                       if s < safe_end and (min(e, safe_end) - max(0, s)) >= 1.0]
    else:
        timestamps = [(max(0, s), e) for s, e in timestamps if e - s >= 1.0]
    n = len(timestamps)
    if n == 0:
        return False

    video_codec = get_video_codec()
    if fps and fps > 0:
        video_codec += ['-r', str(int(fps))]
    audio_codec_tmp = ['-c:a', 'flac']      # FLAC 无编码延迟，时长精确
    mem_opts = ['-threads', '2']

    if n == 1:
        s, e = timestamps[0]
        dur = e - s
        pad = min(_SEEK_PAD, s)
        # A=V 强制相等：视频 CFR 填满到 dur（-t dur），音频 atrim+apad 到 dur
        af = (f'atrim={pad}:{pad + dur},asetpts=PTS-STARTPTS,'
              f'apad=whole_dur={dur:.6f}')
        if normalize:
            af = (f'atrim={pad}:{pad + dur},asetpts=PTS-STARTPTS,loudnorm,'
                  f'apad=whole_dur={dur:.6f}')

        def build_cmd(codec):
            # -t 而不是 -shortest：-shortest 会在音频 filtergraph EOF 时立即
            # 中止调度，而 NVENC lookahead 还压着 ~15 帧（实测两流各被截短
            # 恰好 500ms——切点与输入 EOF 重合时必现）。显式 -t {dur} 让
            # 编码器自然排水并精确封顶：视频 CFR 填满到 dur、音频 atrim
            # 精确到 dur → A=V=dur，无截短、无漂移。
            c = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error'] + mem_opts + [
                '-accurate_seek',
                '-ss', str(s - pad),
                '-i', input_file,
                '-ss', str(pad), '-to', str(pad + dur),
                '-af', af,
                '-avoid_negative_ts', 'make_zero',
                '-vsync', 'cfr', '-t', f'{dur:.6f}',
            ] + codec + audio_codec_tmp
            if res:
                w, h = res
                c.extend(['-vf', f'scale={w}:{h}:force_original_aspect_ratio=decrease,'
                                 f'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2'])
            c.append(output_file)
            return c

        result = _run_ffmpeg(build_cmd(video_codec), timeout=600,
                             progress_callback=progress_callback, duration=dur,
                             stage="Encoding clip")
        if result.returncode != 0 and 'h264_nvenc' in video_codec:
            # 取消导致的 rc!=0 不是 NVENC 故障：不能触发 x264 回退重试，
            # 否则取消后编译会以 CPU 编码继续跑完（用户报告的假取消）。
            if cancel_pending():
                raise InterruptedError("Compile cancelled by user.")
            _fallback_to_x264()
            codec = list(_X264_CODEC)
            if fps and fps > 0:
                codec += ['-r', str(int(fps))]
            result = _run_ffmpeg(build_cmd(codec), timeout=600,
                                 progress_callback=progress_callback, duration=dur,
                                 stage="Encoding clip")
        if result.returncode != 0:
            raise Exception(f"FFmpeg cut failed for {os.path.basename(input_file)}"
                            f"\n  rc={result.returncode}\n  stderr: "
                            f"{_sanitize_ffmpeg_detail(result.stderr)}\n  stdout: "
                            f"{_sanitize_ffmpeg_detail(result.stdout)}")
        return True

    # n > 1：多段切分——先逐段切（重基到 0 的 re-encode），再 batched concat。
    # 段循环的递归调用 n=1 → 终止于 re-encode，无递归风险。
    seg_files = []
    seg_dir = os.path.dirname(output_file)
    # 文件名带 output stem：音频模式 max_parallel=5 时多个 source 并发切分，
    # 共享目录下同名 _segN.mp4 会互相覆盖（产物静默串源）。
    seg_stem = os.path.splitext(os.path.basename(output_file))[0]
    try:
        for i, (s, e) in enumerate(timestamps):
            seg_file = os.path.join(seg_dir, f"_{seg_stem}_seg{i}.mp4")
            seg_files.append(seg_file)
            _ffmpeg_cut(input_file, [(s, e)], seg_file, res=None,
                        normalize=normalize, fps=fps,
                        preserve_duration=preserve_duration,
                        progress_callback=progress_callback, duration=duration)
        # 多段合并：产物会继续进入 compile_vid 的最终 concat，必须写 FLAC
        # 音频（AAC 中间产物会让 concat 音频时间线逐边界拉伸 ~20ms）。
        _ffmpeg_concat_batched(seg_files, output_file, res=res, normalize=normalize,
                               fps=fps, progress_callback=progress_callback,
                               total_duration=duration, batch_size=batch_size,
                               audio_out=_FLAC_AUDIO)
    finally:
        for sf in seg_files:
            try:
                os.remove(sf)
            except OSError:
                pass
    return True


def _ffmpeg_concat(file_list, output_file, res=None, normalize=False, fps=None,
                   progress_callback=None, total_duration=None, audio_out=None):
    """Concatenate using concat FILTER (frame-level, not demuxer)."""
    if not file_list:
        return False
    if len(file_list) == 1 and not res and not normalize:
        shutil.copy2(file_list[0], output_file)
        os.remove(file_list[0])
        return True

    if not res and len(file_list) > 1:
        res = _mixed_resolution_target(file_list)

    n = len(file_list)
    parts = []
    if res:
        w, h = res
        for i in range(n):
            parts.append(f'[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,'
                         f'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1:1,'
                         f'setpts=PTS-STARTPTS[v{i}]')
            parts.append(f'[{i}:a]asetpts=PTS-STARTPTS[a{i}]')
    else:
        for i in range(n):
            parts.append(f'[{i}:v]setsar=1:1,setpts=PTS-STARTPTS[v{i}]')
            parts.append(f'[{i}:a]asetpts=PTS-STARTPTS[a{i}]')
    v_srcs = ''.join(f'[v{i}]' for i in range(n))
    a_srcs = ''.join(f'[a{i}]' for i in range(n))
    parts.append(f'{v_srcs}concat=n={n}:v=1:a=0[outv]')
    # aresample 嵌入 filter_complex（不能放 -af，与复杂滤波器图冲突）
    if normalize:
        parts.append(f'{a_srcs}concat=n={n}:v=0:a=1,loudnorm,aresample=async=1:first_pts=0[outa]')
    else:
        parts.append(f'{a_srcs}concat=n={n}:v=0:a=1,aresample=async=1:first_pts=0[outa]')
    filter_complex = ';'.join(parts)

    def build_cmd(codec):
        c = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error', '-threads', '2']
        for fp in file_list:
            c += ['-i', fp]
        c += ['-filter_complex', filter_complex,
              '-map', '[outv]', '-map', '[outa]'] + codec + \
             (audio_out or _AAC_AUDIO_OUT)
        if fps and fps > 0:
            c += ['-r', str(int(fps))]
        # -t 而不是 -shortest（与切段路径同理）：-shortest 在任一 filtergraph
        # EOF 时经 sync queue 中止，NVENC lookahead 压着的尾帧会让末尾被截。
        # 输入在切段修复后 A=V 精确等长 → 输出自然在 Σdur 结束；-t 只是
        # probe 总和 +1s 的安全封顶（探测不全时退回 -shortest）。
        if probes_complete and total_dur > 0:
            c += ['-t', f'{total_dur + 1.0:.3f}']
        else:
            c += ['-shortest']
        c += ['-vsync', 'cfr', output_file]
        return c

    # 验证所有输入文件有视频+音频流；顺带累加总时长用于动态超时
    # （固定 1200s 会误杀多小时合集的健康编码）和 -t 封顶。
    # 缺音频的输入会让下方 filter_complex 绑定 [i:a] 失败——在昂贵的
    # 合并启动前点名报错，而不是让 ffmpeg 抛出难懂的 stream specifier 错误。
    total_dur = 0.0
    probes_complete = True
    for i, fp in enumerate(file_list):
        if not os.path.exists(fp):
            raise Exception(f"FFmpeg concat: file {i} missing: {fp}")
        probe = _ffprobe(fp)
        if re.search(r'Stream #0:\d+.*Video:.*?(\d{2,})x(\d{2,})', probe) is None:
            raise Exception(f"FFmpeg concat: file {i} has no video stream: {fp}")
        if re.search(r'Stream #0:\d+.*Audio:', probe) is None:
            raise Exception(f"FFmpeg concat: file {i} has no audio stream: {fp}")
        dm = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', probe)
        if dm:
            h, mi, s, ms = map(int, dm.groups())
            total_dur += h * 3600 + mi * 60 + s + ms / 100
        else:
            probes_complete = False
    concat_timeout = max(1800, int(total_dur * 2) + 600)
    # 进度显示用调用方传入的真实总长（含 padding），否则 probe 到的片段时长
    # 会低于实际输出，让 Encoded: X / Y 看起来像超时。
    progress_dur = total_duration if total_duration else total_dur

    video_codec = get_video_codec()
    try:
        result = _run_ffmpeg(build_cmd(video_codec), timeout=concat_timeout,
                             progress_callback=progress_callback, duration=progress_dur,
                             stage="Final merge")
    except subprocess.TimeoutExpired:
        if 'h264_nvenc' not in video_codec:
            raise
        # NVENC 卡死（如睡眠唤醒后驱动死锁）：永久回退 x264 重试一次
        result = None
    if result is None or (result.returncode != 0 and 'h264_nvenc' in video_codec):
        # 同上：取消导致的失败不是 NVENC 故障，禁止整段合并的 x264 重试。
        if cancel_pending():
            raise InterruptedError("Compile cancelled by user.")
        _fallback_to_x264()
        result = _run_ffmpeg(build_cmd(list(_X264_CODEC)), timeout=concat_timeout,
                             progress_callback=progress_callback, duration=total_dur,
                             stage="Final merge")
    if result.returncode != 0:
        missing = [fp for fp in file_list if not os.path.exists(fp)]
        no_video = []
        for fp in file_list:
            if os.path.exists(fp):
                w, h = _get_video_size(fp)
                if w is None:
                    no_video.append(os.path.basename(fp))
        detail = f"\nFiles: {len(file_list)}, missing: {len(missing)}"
        if missing and len(missing) <= 5:
            detail += f"\nMissing: {missing}"
        if no_video and len(no_video) <= 5:
            detail += f"\nNo video stream: {no_video}"
        raise Exception(f"FFmpeg concat failed:{detail}\n[stderr]\n{_sanitize_ffmpeg_detail(result.stderr)}\n[stdout]\n{_sanitize_ffmpeg_detail(result.stdout)}")
    return True


def _ffmpeg_concat_batched(file_list, output_file, res=None, normalize=False, batch_size=6,
                           fps=None, _lvl=0, progress_callback=None,
                           temp_dir=None, total_duration=None, audio_out=None):
    """Batched concat for large file lists. 批数仍超 batch_size 时递归分批，
    保证任意 clip 数量下单条 ffmpeg 命令行都不会爆 Windows 32767 上限。

    中间产物 _batchL* 一律写 FLAC 音频（AAC 中间件会让下一层 concat 的音频
    时间线每边界拉伸 ~20ms，逐层累积成"整体逐渐错位"）；只有产出 output_file
    的那一次编码使用 audio_out（调用方不传则为最终输出的 AAC）。

    batch_size < 2 时无法收缩工作集（单元素批次递归自身）→ 实测无限递归
    直到 RecursionError/数小时 IO 空转。入口强制钳到 ≥2。
    """
    if not file_list:
        return False
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        batch_size = 6
    if batch_size < 2:
        batch_size = 2
    if len(file_list) == 1 and not res and not normalize:
        shutil.copy2(file_list[0], output_file)
        os.remove(file_list[0])
        return True

    if not res and len(file_list) > 1:
        res = _mixed_resolution_target(file_list)

    if len(file_list) <= batch_size:
        return _ffmpeg_concat(file_list, output_file, res=res, normalize=normalize, fps=fps,
                              progress_callback=progress_callback,
                              total_duration=total_duration, audio_out=audio_out)

    # 中间文件默认与输出同盘；调用方可传入 compile 的临时目录，避免大批量
    # 合并时 _batchL* 中间件 flood 用户输出文件夹。
    temp_dir = temp_dir or os.path.dirname(output_file) or os.path.dirname(file_list[0])
    batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]
    batch_files = []
    try:
        for bi, batch in enumerate(batches):
            batch_out = os.path.join(temp_dir, f"_batchL{_lvl}_{bi}.mp4")
            batch_files.append(batch_out)
            print(f"  Batch {bi + 1}/{len(batches)} ({len(batch)} files)...")
            if progress_callback is not None:
                progress_callback(format_compile_progress(
                    0, None, 0, f"Batch {bi + 1}/{len(batches)}: starting"
                ))
            ok = _ffmpeg_concat(batch, batch_out, res=res, normalize=normalize, fps=fps,
                                progress_callback=progress_callback,
                                total_duration=total_duration,
                                audio_out=_FLAC_AUDIO)
            if not ok:
                raise Exception(f"Batch {bi + 1} failed")
        print(f"  Final merge ({len(batch_files)} files)...")
        if progress_callback is not None:
            progress_callback(format_compile_progress(0, None, 0, "Final merge: starting"))
        if len(batch_files) > batch_size:
            # 批数仍过多，继续递归分批（层级文件名避免覆盖同层）
            return _ffmpeg_concat_batched(batch_files, output_file, res=res,
                                          normalize=normalize, batch_size=batch_size,
                                          fps=fps, _lvl=_lvl + 1,
                                          progress_callback=progress_callback,
                                          temp_dir=temp_dir,
                                          total_duration=total_duration,
                                          audio_out=audio_out)
        _ffmpeg_concat(batch_files, output_file, res=res, normalize=normalize, fps=fps,
                       progress_callback=progress_callback, total_duration=total_duration,
                       audio_out=audio_out)
    finally:
        for bf in batch_files:
            try:
                os.remove(bf)
            except OSError:
                pass
    return True


# ═══ AUDIO cut / concat ═══════════════════════════════════════════════

def _ffmpeg_cut_audio(input_file, timestamps, output_file, normalize=False,
                      progress_callback=None, duration=None):
    if not timestamps:
        return False

    duration = duration or _get_video_duration(input_file)
    if duration:
        safe_end = max(0, duration - 0.5)
        timestamps = [(max(0, s), min(e, safe_end)) for s, e in timestamps
                       if s < safe_end and (min(e, safe_end) - max(0, s)) >= 1.0]
    else:
        timestamps = [(max(0, s), e) for s, e in timestamps if e - s >= 1.0]
    n = len(timestamps)
    if n == 0:
        return False

    audio_codec = ['-c:a', 'libmp3lame', '-b:a', '192k', '-ar', '44100']

    if n == 1:
        s, e = timestamps[0]
        pad = min(_SEEK_PAD, s)
        cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error', '-threads', '2',
               '-ss', str(s - pad), '-i', input_file,
               '-ss', str(pad), '-to', str(pad + (e - s)),
               '-avoid_negative_ts', 'make_zero'] + audio_codec
        if normalize:
            cmd.extend(['-af', 'loudnorm'])
        cmd.append(output_file)
        result = _run_ffmpeg(cmd, timeout=600, progress_callback=progress_callback,
                             duration=e - s, stage="Encoding audio clip")
        if result.returncode != 0:
            raise Exception(f"FFmpeg audio cut failed for {os.path.basename(input_file)}:\n{result.stderr}")
        return True

    # 多段：用 FLAC 中间格式（无 encoder priming），避免 MP3 每段引入的
    # encoder delay 在 concat 时逐段累积（实测 20 段 +0.85s → 0）。
    seg_files = []
    seg_dir = os.path.dirname(output_file)
    # 文件名必须带 output stem 唯一化：音频模式 max_parallel=5，两个以上
    # 多段 source 并发时共享目录里同名 _asegN.flac / concat_list.txt 会互相
    # 覆盖 → 成片静默串入别的 VOD 的音频。这是审计发现的高危缺陷。
    seg_stem = os.path.splitext(os.path.basename(output_file))[0]
    try:
        for i, (s, e) in enumerate(timestamps):
            seg_file = os.path.join(seg_dir, f"_{seg_stem}_aseg{i}.flac")
            seg_files.append(seg_file)
            cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error', '-threads', '2',
                   '-ss', str(s - min(_SEEK_PAD, s)), '-i', input_file,
                   '-ss', str(min(_SEEK_PAD, s)), '-to', str(min(_SEEK_PAD, s) + (e - s)),
                   '-map', '0:a:0', '-c:a', 'flac', '-ar', '44100',
                   '-avoid_negative_ts', 'make_zero', seg_file]
            if normalize:
                cmd = cmd[:-1] + ['-af', 'loudnorm', seg_file]
            result = _run_ffmpeg(cmd, timeout=600,
                                 progress_callback=progress_callback,
                                 duration=e - s, stage="Encoding audio clip")
            if result.returncode != 0:
                raise Exception(
                    f"FFmpeg audio cut failed for {os.path.basename(input_file)}:\n{result.stderr}")
        _ffmpeg_concat_audio(seg_files, output_file, normalize=normalize,
                             progress_callback=progress_callback)
    finally:
        for sf in seg_files:
            try:
                os.remove(sf)
            except OSError:
                pass
    return True


def _ffmpeg_concat_audio(file_list, output_file, normalize=False, progress_callback=None):
    if not file_list:
        return False
    temp_dir = os.path.dirname(file_list[0])
    # 列表文件随首个输入 stem 唯一化（并发 audio cut 时共享的 concat_list.txt
    # 会被兄弟任务截断/删除，见 _ffmpeg_cut_audio 的 stem 注释）。
    first_stem = os.path.splitext(os.path.basename(file_list[0]))[0]
    list_path = os.path.join(temp_dir, f"{first_stem}_concat_list.txt")
    durations = []
    for fp in file_list:
        dur = _get_video_duration(fp)
        durations.append(dur)
    with open(list_path, 'w', encoding='utf-8') as fh:
        fh.write('ffconcat version 1.0\n')
        for i, fp in enumerate(file_list):
            fh.write(f"file '{fp}'\n")
            if durations[i]:
                fh.write(f"duration {durations[i]:.6f}\n")

    # 输入为 FLAC 中间片段（无 encoder priming），统一重编码为 mp3，
    # 否则 -c:a copy 无法把 FLAC 直接封进 .mp3。
    cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error', '-threads', '2',
           '-fflags', '+genpts+igndts', '-f', 'concat', '-safe', '0', '-copytb', '0',
           '-i', list_path]
    if normalize:
        cmd += ['-c:a', 'libmp3lame', '-b:a', '192k', '-ar', '44100', '-af', 'loudnorm']
    else:
        cmd += ['-c:a', 'libmp3lame', '-b:a', '192k', '-ar', '44100']
    cmd.append(output_file)

    total_duration = sum((duration or 0) for duration in durations)
    result = _run_ffmpeg(cmd, timeout=600, progress_callback=progress_callback,
                         duration=total_duration, stage="Final audio merge")
    try:
        os.remove(list_path)
    except OSError:
        pass
    if result.returncode != 0:
        missing = [fp for fp in file_list if not os.path.exists(fp)]
        detail = f"\nFiles: {len(file_list)}, missing: {len(missing)}"
        if missing and len(missing) <= 5:
            detail += f"\nMissing: {missing}"
        raise Exception(f"FFmpeg audio concat failed:{detail}\n[stderr]\n{_sanitize_ffmpeg_detail(result.stderr)}\n[stdout]\n{_sanitize_ffmpeg_detail(result.stdout)}")
    return True


# ═══ Public API ═══════════════════════════════════════════════════════

def compile_vid(dict_list, output, merge_clips=True, combine_vids=True,
                res=None, logger=None, normalize=False, is_video=True, padding=None,
                excluded=None, progress_callback=None, batch_size=6):
    output_format = ".mp4" if is_video else ".mp3"

    # 同进程二次运行会复用旧路径的 probe 结果（如 _seg0.mp4 已重写），
    # 清空缓存避免时长/尺寸用旧值。
    _PROBE_CACHE.clear()

    # 清理上次崩溃/强杀残留的中间文件（只删我们自己的命名模式，不碰用户文件）
    _out_dir = output if os.path.isdir(output) else os.path.dirname(output)
    if _out_dir and os.path.isdir(_out_dir):
        for _stale in os.listdir(_out_dir):
            # 新命名带 output stem（并发唯一化）：_<stem>_segN/_asegN；
            # 旧命名保留匹配，覆盖升级前崩溃的残留。
            if re.fullmatch(r'(_batchL\d+_\d+|(?:_[^\\/]{0,120})?_seg\d+|'
                            r'(?:_[^\\/]{0,120})?_aseg\d+)\.(mp4|mp3|flac)', _stale):
                try:
                    os.remove(os.path.join(_out_dir, _stale))
                except OSError:
                    pass
        for _stale in os.listdir(_out_dir):
            if _stale == 'concat_list.txt' or (
                    _stale.startswith('_') and _stale.endswith('_concat_list.txt')):
                try:
                    os.remove(os.path.join(_out_dir, _stale))
                except OSError:
                    pass

    if is_video:
        cut_func, concat_func = _ffmpeg_cut, _ffmpeg_concat
        max_parallel = 1
    else:
        cut_func, concat_func = _ffmpeg_cut_audio, _ffmpeg_concat_audio
        max_parallel = 5

    tempfiles = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = []

            # 固定输出帧率 30fps，防止 VFR / 25fps 导致的 A/V 偏移
            fps = 30 if is_video else None

            # 并行预取容器元数据（填充 _PROBE_CACHE），避免 queuing 阶段串行
            # 探测大量片段文件导致 UI freeze。已带 duration 的 entry 直接跳过。
            probe_files = [
                elt["filename"]
                for elt in dict_list
                if not elt.get("duration")
            ]
            if probe_files:
                with ThreadPoolExecutor(max_workers=min(len(probe_files), 5)) as probe_executor:
                    list(probe_executor.map(_get_video_duration, probe_files))

            for n, elt in enumerate(dict_list):
                filename = elt["filename"]
                filename_stripped = os.path.basename(str(filename))
                timestamps = [(d["start"], d["end"]) for d in elt["timestamps"]]

                print(f"{Fore.GREEN}[{n + 1}/{len(dict_list)}]{Style.RESET_ALL} "
                      f"Queuing {filename_stripped}...")

                if padding:
                    before, after = padding
                    if before < 0 or after < 0:
                        raise Exception("Clip padding cannot be negative!")
                    for i, ts in enumerate(timestamps):
                        timestamps[i] = (ts[0] - before, ts[1] + after)

                # Review 取消勾选 / Strict FP 丢弃的片段：merge 不允许跨过这些
                # 区间桥接，否则被删片段的内容会随桥接重新出现在成片里（~1s 残影）。
                blocked = []
                if excluded:
                    blocked = [(s, e) for s, e in excluded.get(_source_key(filename, elt), []) if e > s]
                    # 与保留片段重叠的排除段不是用户删除（如 reverify 新增 clip
                    # 覆盖后被 auto-deselect 的 original），不能当禁区
                    blocked = [r for r in blocked
                               if not any(r[0] < te and r[1] > ts
                                          for ts, te in timestamps)]

                if merge_clips:
                    i = 0
                    while i < len(timestamps) - 1:
                        gap_s, gap_e = timestamps[i][1], timestamps[i + 1][0]
                        gap_blocked = (gap_e > gap_s and
                                       any(r[0] < gap_e and r[1] > gap_s
                                           for r in blocked))
                        if (gap_e - gap_s < MERGE_THRESHOLD) and not gap_blocked:
                            timestamps[i] = (timestamps[i][0],
                                             max(timestamps[i][1], timestamps[i + 1][1]))
                            del timestamps[i + 1]
                        else:
                            i += 1

                dur = elt.get('duration') or _get_video_duration(filename)
                if dur:
                    safe_end = clip_safe_end(dur, elt)
                    orig_count = len(timestamps)
                    timestamps = [(max(0, s), min(e, safe_end)) for s, e in timestamps
                                   if s < safe_end and (min(e, safe_end) - max(0, s)) >= 1.0]
                    if len(timestamps) < orig_count:
                        print(f"{Fore.YELLOW}  Warning: {orig_count - len(timestamps)} segment(s) clipped")
                else:
                    orig_count = len(timestamps)
                    timestamps = [(max(0, s), e) for s, e in timestamps if e - s >= 1.0]
                    if len(timestamps) < orig_count:
                        print(f"{Fore.YELLOW}  Warning: {orig_count - len(timestamps)} segment(s) too short")

                # ── 修剪相邻 clip 重叠（编译阶段最后防线）──
                # padding 后相邻 clip 可能重叠（紧邻 clip 各自加了 before/after）。
                # 只修剪重叠部分，而不是删除整个 clip：删除会让边界处的
                # speech/burp 整段丢失（cut off），也不该让内容重复播放。
                if len(timestamps) > 1:
                    timestamps.sort(key=lambda x: x[0])
                    trimmed = []
                    for ts in timestamps:
                        s, e = ts
                        if trimmed and s < trimmed[-1][1]:
                            # 重叠：把当前 clip 起点推到上一 clip 终点，
                            # 保留两个 clip 且恰好衔接。
                            s = trimmed[-1][1]
                        if e - s >= 1.0:
                            trimmed.append((s, e))
                    timestamps = trimmed

                if not timestamps:
                    print(f"{Fore.YELLOW}No timestamps found for this video!")
                    continue

                if combine_vids:
                    temp = os.path.join(temp_dir, f"{n}{output_format}")
                    tempfiles.append(temp)
                else:
                    base = os.path.basename(filename).rsplit('.', 1)[0]
                    temp = os.path.join(output, f"{base}_comped{output_format}")

                cut_res = res if (is_video and not combine_vids and res is not None) else None
                preserve_duration = bool(
                    (elt.get('source_metadata') or {}).get(
                        'materialized_remote_segment'))
                tasks.append((n, filename, filename_stripped, timestamps, temp,
                              cut_res, preserve_duration, dur))

            if not tasks:
                raise Exception("No timestamps found for any input media!")
            if progress_callback is not None:
                progress_callback(format_compile_progress(0, None, 0, "Compile: preparing"))

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), max_parallel)) as executor:
                running = {}
                for task in tasks:
                    if cancel_pending():
                        executor.shutdown(cancel_futures=True)
                        raise InterruptedError("Compile cancelled by user.")
                    n, fn, fn_stripped, ts, tmp, cr, preserve_duration, dur = task
                    f = executor.submit(cut_func, fn, ts, tmp,
                                        **({'res': cr, 'fps': fps,
                                            'preserve_duration': preserve_duration,
                                            'duration': dur,
                                            'batch_size': batch_size,
                                            'progress_callback': progress_callback}
                                            if is_video else {}),
                                       normalize=normalize,
                                       **({'progress_callback': progress_callback}
                                          if not is_video else {}),
                                       **({'duration': dur} if not is_video else {}))
                    running[f] = (n, fn_stripped)

                cut_failures = []
                cut_done = 0
                cut_total = len(running)
                for future in concurrent.futures.as_completed(running):
                    if cancel_pending():
                        executor.shutdown(cancel_futures=True)
                        raise InterruptedError("Compile cancelled by user.")
                    n, fn_stripped = running[future]
                    try:
                        future.result()
                        cut_done += 1
                        print(f"{Fore.GREEN}[{cut_done}/{cut_total}] Done writing all clips for {fn_stripped}.")
                    except Exception as ex:
                        print(f"{Fore.RED}Failed writing clips for {fn_stripped}: {ex}")
                        cut_failures.append((fn_stripped, ex))
                if cut_failures:
                    skipped = ", ".join(name for name, _ in cut_failures)
                    print(f"{Fore.YELLOW}{len(cut_failures)} clip(s) failed to write "
                          f"and were skipped: {skipped}{Style.RESET_ALL}")

            if combine_vids:
                tempfiles = [t for t in [task[4] for task in tasks] if os.path.exists(t)]
                if not tempfiles:
                    raise Exception("All clip writes failed; nothing to combine.")
                # 切片完成后、合并启动前的边界检查：此刻取消不应再开始
                # 一段可能长达数小时的整片重编码。
                if cancel_pending():
                    raise InterruptedError("Compile cancelled by user.")
                print("Combining individual media, please do not close the program...", end="")
                if is_video:
                    # 真实总长 = 所有 clip 的已裁剪时长之和（含 padding），
                    # 让进度显示 Encoded: X / X 而不是 probe 的未含 padding 估值。
                    total_duration = sum(
                        e - s for task in tasks for s, e in task[3]
                    )
                    _ffmpeg_concat_batched(
                        tempfiles, output, res=res, normalize=normalize, fps=fps,
                        progress_callback=progress_callback,
                        temp_dir=temp_dir, total_duration=total_duration,
                        batch_size=batch_size,
                    )
                else:
                    concat_func(tempfiles, output, normalize=normalize,
                                progress_callback=progress_callback)
                print(f"{Fore.GREEN}Done combining media.")

    except Exception:
        raise
    finally:
        for f in tempfiles:
            try:
                os.remove(f)
            except (FileNotFoundError, OSError):
                continue
