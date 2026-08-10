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

from colorama import Fore, Style

from progress import format_compile_progress
from utils import FFMPEG_PATH, run_tracked, run_tracked_progress
import sys
import os

MERGE_THRESHOLD = 2  # seconds


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


def _fallback_to_x264():
    """NVENC 编码中途失败时，永久回退 libx264（更新缓存）。"""
    global _VIDEO_CODEC_CACHE
    _VIDEO_CODEC_CACHE = _X264_CODEC
    print(f"{Fore.YELLOW}NVENC encode failed; falling back to libx264 (CPU).{Style.RESET_ALL}")


def _ffmpeg_cut(input_file, timestamps, output_file, res=None, normalize=False,
                fps=None, preserve_duration=False, progress_callback=None,
                duration=None, copy_cut=False):
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

    if copy_cut and n == 1:
        s, e = timestamps[0]
        # materialized clip 已是精确区间：视频流直接 copy（快），音频重编为
        # FLAC（无编码延迟、时长精确），避免 aac priming 在 concat 边界产生
        # audio overlap/desync。帧率/音量等归一化交给后续 concat（combine 模式）。
        full_range = s <= 0.05 and (duration is None or e >= duration - 0.1)
        if full_range:
            cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error',
                   '-i', input_file, '-map', '0:v:0', '-map', '0:a:0?',
                   '-c:v', 'copy', '-c:a', 'flac',
                   '-avoid_negative_ts', 'make_zero',
                   '-movflags', '+faststart', output_file]
            result = _run_ffmpeg(cmd, timeout=300,
                                 progress_callback=progress_callback,
                                 duration=duration, stage="Remuxing clip")
            if result.returncode != 0:
                raise Exception(
                    f"FFmpeg remux failed for {os.path.basename(input_file)}"
                    f"\n  rc={result.returncode}\n  stderr: {result.stderr}"
                    f"\n  stdout: {result.stdout}")
            return True

    video_codec = get_video_codec()
    if fps and fps > 0:
        video_codec += ['-r', str(int(fps))]
    audio_codec_tmp = ['-c:a', 'flac']      # FLAC 无编码延迟，时长精确
    audio_codec_out = ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']
    mem_opts = ['-threads', '2']

    if n == 1:
        s, e = timestamps[0]
        dur = e - s
        pad = min(_SEEK_PAD, s)
        af = f'atrim={pad}:{pad + dur}'
        if normalize:
            af += ',loudnorm'

        def build_cmd(codec):
            c = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error'] + mem_opts + [
                '-accurate_seek',
                '-ss', str(s - pad),
                '-i', input_file,
                '-ss', str(pad), '-to', str(pad + dur),
                '-af', af,
                '-avoid_negative_ts', 'make_zero',
                '-vsync', 'cfr', '-shortest',
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
            _fallback_to_x264()
            codec = list(_X264_CODEC)
            if fps and fps > 0:
                codec += ['-r', str(int(fps))]
            result = _run_ffmpeg(build_cmd(codec), timeout=600,
                                 progress_callback=progress_callback, duration=dur,
                                 stage="Encoding clip")
        if result.returncode != 0:
            raise Exception(f"FFmpeg cut failed for {os.path.basename(input_file)}"
                            f"\n  rc={result.returncode}\n  stderr: {result.stderr}\n  stdout: {result.stdout}")
        return True

    seg_files = []
    seg_dir = os.path.dirname(output_file)
    try:
        for i, (s, e) in enumerate(timestamps):
            seg_file = os.path.join(seg_dir, f"_seg{i}.mp4")
            seg_files.append(seg_file)
            _ffmpeg_cut(input_file, [(s, e)], seg_file, res=None,
                        normalize=normalize, fps=fps,
                        preserve_duration=preserve_duration,
                        progress_callback=progress_callback, duration=duration)
        # 必须走 batched：单视频上千 segment 时 _ffmpeg_concat 会把所有 -i
        # 和 filter 塞进一条命令，超过 Windows 命令行上限 → WinError 206
        _ffmpeg_concat_batched(seg_files, output_file, res=res, normalize=normalize,
                               fps=fps, progress_callback=progress_callback)
    finally:
        for sf in seg_files:
            try:
                os.remove(sf)
            except OSError:
                pass
    return True


def _ffmpeg_concat(file_list, output_file, res=None, normalize=False, fps=None,
                    progress_callback=None):
    """Concatenate using concat FILTER (frame-level, not demuxer)."""
    if not file_list:
        return False
    if len(file_list) == 1 and not res and not normalize:
        shutil.copy2(file_list[0], output_file)
        os.remove(file_list[0])
        return True

    if not res and len(file_list) > 1:
        sizes = set()
        for fp in file_list:
            w, h = _get_video_size(fp)
            if w and h:
                sizes.add((w, h))
        if len(sizes) > 1:
            size_text = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
            print(f"{Fore.YELLOW}Mixed resolutions ({size_text}) -> re-encoding for sync...")
            size_counts = {}
            for fp in file_list:
                sz = _get_video_size(fp)
                if sz and sz[0] and sz[1]:
                    size_counts[sz] = size_counts.get(sz, 0) + 1
            if size_counts:
                res = max(size_counts, key=size_counts.get)
                print(f"  Target resolution: {res[0]}x{res[1]}")
            else:
                print("  No resolvable input resolutions; keeping source size.")

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
             ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']
        if fps and fps > 0:
            c += ['-r', str(int(fps))]
        c += ['-vsync', 'cfr', '-shortest', output_file]
        return c

    # 验证所有输入文件有视频流；顺带累加总时长用于动态超时
    # （固定 1200s 会误杀多小时合集的健康编码）
    total_dur = 0.0
    for i, fp in enumerate(file_list):
        if not os.path.exists(fp):
            raise Exception(f"FFmpeg concat: file {i} missing: {fp}")
        probe = _ffprobe(fp)
        if re.search(r'Stream #0:\d+.*Video:.*?(\d{2,})x(\d{2,})', probe) is None:
            raise Exception(f"FFmpeg concat: file {i} has no video stream: {fp}")
        dm = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', probe)
        if dm:
            h, mi, s, ms = map(int, dm.groups())
            total_dur += h * 3600 + mi * 60 + s + ms / 100
    concat_timeout = max(1800, int(total_dur * 2) + 600)

    video_codec = get_video_codec()
    try:
        result = _run_ffmpeg(build_cmd(video_codec), timeout=concat_timeout,
                             progress_callback=progress_callback, duration=total_dur,
                             stage="Final merge")
    except subprocess.TimeoutExpired:
        if 'h264_nvenc' not in video_codec:
            raise
        # NVENC 卡死（如睡眠唤醒后驱动死锁）：永久回退 x264 重试一次
        result = None
    if result is None or (result.returncode != 0 and 'h264_nvenc' in video_codec):
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
        raise Exception(f"FFmpeg concat failed:{detail}\n[stderr]\n{result.stderr}\n[stdout]\n{result.stdout}")
    return True


def _ffmpeg_concat_batched(file_list, output_file, res=None, normalize=False, batch_size=6,
                           fps=None, _lvl=0, progress_callback=None):
    """Batched concat for large file lists. 批数仍超 batch_size 时递归分批，
    保证任意 clip 数量下单条 ffmpeg 命令行都不会爆 Windows 32767 上限。"""
    if not file_list:
        return False
    if len(file_list) == 1 and not res and not normalize:
        shutil.copy2(file_list[0], output_file)
        os.remove(file_list[0])
        return True

    if not res and len(file_list) > 1:
        sizes = set()
        for fp in file_list:
            w, h = _get_video_size(fp)
            if w and h:
                sizes.add((w, h))
        if len(sizes) > 1:
            size_text = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
            print(f"{Fore.YELLOW}Mixed resolutions ({size_text}) -> re-encoding for sync...")
            size_counts = {}
            for fp in file_list:
                sz = _get_video_size(fp)
                if sz and sz[0] and sz[1]:
                    size_counts[sz] = size_counts.get(sz, 0) + 1
            if size_counts:
                res = max(size_counts, key=size_counts.get)
                print(f"  Target resolution: {res[0]}x{res[1]}")
            else:
                print("  No resolvable input resolutions; keeping source size.")

    if len(file_list) <= batch_size:
        return _ffmpeg_concat(file_list, output_file, res=res, normalize=normalize, fps=fps,
                              progress_callback=progress_callback)

    temp_dir = os.path.dirname(output_file) or os.path.dirname(file_list[0])
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
                                progress_callback=progress_callback)
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
                                          progress_callback=progress_callback)
        _ffmpeg_concat(batch_files, output_file, res=res, normalize=normalize, fps=fps,
                        progress_callback=progress_callback)
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
    try:
        for i, (s, e) in enumerate(timestamps):
            seg_file = os.path.join(seg_dir, f"_aseg{i}.flac")
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
    list_path = os.path.join(temp_dir, 'concat_list.txt')
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
        raise Exception(f"FFmpeg audio concat failed:{detail}\n[stderr]\n{result.stderr}\n[stdout]\n{result.stdout}")
    return True


# ═══ Public API ═══════════════════════════════════════════════════════

def compile_vid(dict_list, output, merge_clips=True, combine_vids=True,
                res=None, logger=None, normalize=False, is_video=True, padding=None,
                excluded=None, progress_callback=None):
    output_format = ".mp4" if is_video else ".mp3"

    # 清理上次崩溃/强杀残留的中间文件（只删我们自己的命名模式，不碰用户文件）
    _out_dir = output if os.path.isdir(output) else os.path.dirname(output)
    if _out_dir and os.path.isdir(_out_dir):
        for _stale in os.listdir(_out_dir):
            if re.fullmatch(r'(_batchL\d+_\d+|_seg\d+|_aseg\d+)\.(mp4|mp3)', _stale):
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
                      f"Queuing {filename_stripped}...", end="")

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
                copy_cut = bool(
                    is_video and combine_vids and preserve_duration
                    and len(timestamps) == 1)
                tasks.append((n, filename, filename_stripped, timestamps, temp,
                              cut_res, preserve_duration, dur, copy_cut))

            if not tasks:
                raise Exception("No timestamps found for any input media!")
            if progress_callback is not None:
                progress_callback(format_compile_progress(0, None, 0, "Compile: preparing"))

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), max_parallel)) as executor:
                running = {}
                for task in tasks:
                    n, fn, fn_stripped, ts, tmp, cr, preserve_duration, dur, copy_cut = task
                    f = executor.submit(cut_func, fn, ts, tmp,
                                        **({'res': cr, 'fps': fps,
                                            'preserve_duration': preserve_duration,
                                            'duration': dur,
                                            'copy_cut': copy_cut,
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
                print("Combining individual media, please do not close the program...", end="")
                if is_video:
                    _ffmpeg_concat_batched(
                        tempfiles, output, res=res, normalize=normalize, fps=fps,
                        progress_callback=progress_callback,
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
