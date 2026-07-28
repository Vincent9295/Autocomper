#!/usr/bin/env python
"""compile.py — native FFmpeg video/audio compilation."""

import concurrent.futures
import os
import re
import shutil
import sys
import tempfile

from colorama import Fore, Style

from utils import FFMPEG_PATH, run_tracked
import sys
import os

MERGE_THRESHOLD = 2  # seconds

# 输入 -ss 在 MPEG-TS（尤其直播录像，PTS 不连续）上按字节估算落点，
# 视频可能落在目标数秒之外（实测 +3.5s），而音频落点正确。
# 此时 -vsync cfr 会用首帧填满 0..pts 空隙，-shortest 又在音频长度处截断，
# 导致整段 clip 画面冻结（用户报"corrupted"）。
# 对策：输入多回退 _SEEK_PAD 秒，再用输出 -ss/-to 精确定位（PTS 相对值不变）。
_SEEK_PAD = 10.0  # seconds


def _ffprobe(input_file: str):
    """ffmpeg -i 探测容器元数据（包内无 ffprobe）。"""
    try:
        out = run_tracked([FFMPEG_PATH, '-hide_banner', '-i', input_file],
                          timeout=10, text=True)
        return out.stderr or ''
    except Exception:
        return ''


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
                fps=None):
    if not timestamps:
        return False

    duration = _get_video_duration(input_file)
    if duration:
        safe_end = max(0, duration - 0.5)
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

        result = run_tracked(build_cmd(video_codec), timeout=600, text=True)
        if result.returncode != 0 and 'h264_nvenc' in video_codec:
            _fallback_to_x264()
            codec = list(_X264_CODEC)
            if fps and fps > 0:
                codec += ['-r', str(int(fps))]
            result = run_tracked(build_cmd(codec), timeout=600, text=True)
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
            _ffmpeg_cut(input_file, [(s, e)], seg_file, res=None, normalize=normalize, fps=fps)
        # 必须走 batched：单视频上千 segment 时 _ffmpeg_concat 会把所有 -i
        # 和 filter 塞进一条命令，超过 Windows 命令行上限 → WinError 206
        _ffmpeg_concat_batched(seg_files, output_file, res=res, normalize=normalize, fps=fps)
    finally:
        for sf in seg_files:
            try:
                os.remove(sf)
            except OSError:
                pass
    return True


def _ffmpeg_concat(file_list, output_file, res=None, normalize=False, fps=None):
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
            print(f"{Fore.YELLOW}Mixed resolutions ({len(sizes)} types) -> re-encoding for sync...")
            size_counts = {}
            for fp in file_list:
                sz = _get_video_size(fp)
                if sz and sz[0] and sz[1]:
                    size_counts[sz] = size_counts.get(sz, 0) + 1
            if size_counts:
                res = max(size_counts, key=size_counts.get)
                print(f"  Target resolution: {res[0]}x{res[1]}")

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

    cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error', '-threads', '2']
    for fp in file_list:
        cmd += ['-i', fp]
    cmd += ['-filter_complex', filter_complex,
            '-map', '[outv]', '-map', '[outa]'] + get_video_codec() + \
           ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']
    if fps and fps > 0:
        cmd += ['-r', str(int(fps))]
    cmd += ['-vsync', 'cfr', '-shortest', output_file]

    # 验证所有输入文件有视频流
    for i, fp in enumerate(file_list):
        if not os.path.exists(fp):
            raise Exception(f"FFmpeg concat: file {i} missing: {fp}")
        sz = _get_video_size(fp)
        if sz[0] is None:
            raise Exception(f"FFmpeg concat: file {i} has no video stream: {fp}")

    result = run_tracked(cmd, timeout=1200, text=True)
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


def _ffmpeg_concat_batched(file_list, output_file, res=None, normalize=False, batch_size=6, fps=None, _lvl=0):
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
            print(f"{Fore.YELLOW}Mixed resolutions ({len(sizes)} types) -> re-encoding for sync...")
            size_counts = {}
            for fp in file_list:
                sz = _get_video_size(fp)
                if sz and sz[0] and sz[1]:
                    size_counts[sz] = size_counts.get(sz, 0) + 1
            if size_counts:
                res = max(size_counts, key=size_counts.get)
                print(f"  Target resolution: {res[0]}x{res[1]}")

    if len(file_list) <= batch_size:
        return _ffmpeg_concat(file_list, output_file, res=res, normalize=normalize, fps=fps)

    temp_dir = os.path.dirname(output_file) or os.path.dirname(file_list[0])
    batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]
    batch_files = []
    try:
        for bi, batch in enumerate(batches):
            batch_out = os.path.join(temp_dir, f"_batchL{_lvl}_{bi}.mp4")
            batch_files.append(batch_out)
            print(f"  Batch {bi + 1}/{len(batches)} ({len(batch)} files)...")
            ok = _ffmpeg_concat(batch, batch_out, res=res, normalize=normalize, fps=fps)
            if not ok:
                raise Exception(f"Batch {bi + 1} failed")
        print(f"  Final merge ({len(batch_files)} files)...")
        if len(batch_files) > batch_size:
            # 批数仍过多，继续递归分批（层级文件名避免覆盖同层）
            return _ffmpeg_concat_batched(batch_files, output_file, res=res,
                                          normalize=normalize, batch_size=batch_size,
                                          fps=fps, _lvl=_lvl + 1)
        _ffmpeg_concat(batch_files, output_file, res=res, normalize=normalize, fps=fps)
    finally:
        for bf in batch_files:
            try:
                os.remove(bf)
            except OSError:
                pass
    return True


# ═══ AUDIO cut / concat ═══════════════════════════════════════════════

def _ffmpeg_cut_audio(input_file, timestamps, output_file, normalize=False):
    if not timestamps:
        return False

    duration = _get_video_duration(input_file)
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
        result = run_tracked(cmd, timeout=600, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg audio cut failed for {os.path.basename(input_file)}:\n{result.stderr}")
        return True

    seg_files = []
    seg_dir = os.path.dirname(output_file)
    try:
        for i, (s, e) in enumerate(timestamps):
            seg_file = os.path.join(seg_dir, f"_aseg{i}.mp3")
            seg_files.append(seg_file)
            _ffmpeg_cut_audio(input_file, [(s, e)], seg_file, normalize=normalize)
        _ffmpeg_concat_audio(seg_files, output_file, normalize=normalize)
    finally:
        for sf in seg_files:
            try:
                os.remove(sf)
            except OSError:
                pass
    return True


def _ffmpeg_concat_audio(file_list, output_file, normalize=False):
    if not file_list:
        return False
    if len(file_list) == 1:
        shutil.copy2(file_list[0], output_file)
        os.remove(file_list[0])
        return True

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

    cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error', '-threads', '2',
           '-fflags', '+genpts+igndts', '-f', 'concat', '-safe', '0', '-copytb', '0',
           '-i', list_path]
    if normalize:
        cmd += ['-c:a', 'libmp3lame', '-b:a', '192k', '-ar', '44100', '-af', 'loudnorm']
    else:
        cmd += ['-c:a', 'copy']
    cmd.append(output_file)

    result = run_tracked(cmd, timeout=600, text=True)
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
                res=None, logger=None, normalize=False, is_video=True, padding=None):
    output_format = ".mp4" if is_video else ".mp3"

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

                if merge_clips:
                    i = 0
                    while i < len(timestamps) - 1:
                        if timestamps[i + 1][0] - timestamps[i][1] < MERGE_THRESHOLD:
                            timestamps[i] = (timestamps[i][0],
                                             max(timestamps[i][1], timestamps[i + 1][1]))
                            del timestamps[i + 1]
                        else:
                            i += 1

                dur = _get_video_duration(filename)
                if dur:
                    safe_end = max(0, dur - 0.5)
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
                if len(timestamps) > 1:
                    timestamps.sort(key=lambda x: x[0])
                    i = 0
                    while i < len(timestamps) - 1:
                        if timestamps[i][1] > timestamps[i + 1][0]:
                            dur_i = timestamps[i][1] - timestamps[i][0]
                            dur_j = timestamps[i + 1][1] - timestamps[i + 1][0]
                            if dur_i >= dur_j:
                                del timestamps[i + 1]
                            else:
                                del timestamps[i]
                        else:
                            i += 1
                    timestamps = [(s, e) for s, e in timestamps if e - s >= 1.0]

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
                tasks.append((n, filename, filename_stripped, timestamps, temp, cut_res))

            if not tasks:
                raise Exception("No timestamps found for any input media!")

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), max_parallel)) as executor:
                running = {}
                for task in tasks:
                    n, fn, fn_stripped, ts, tmp, cr = task
                    f = executor.submit(cut_func, fn, ts, tmp,
                                       **({'res': cr, 'fps': fps} if is_video else {}), normalize=normalize)
                    running[f] = (n, fn_stripped)

                for future in concurrent.futures.as_completed(running):
                    n, fn_stripped = running[future]
                    try:
                        future.result()
                        print(f"{Fore.GREEN}Done writing all clips for {fn_stripped}.")
                    except Exception as ex:
                        print(f"{Fore.RED}Failed writing clips for {fn_stripped}: {ex}")
                        raise

            if combine_vids:
                tempfiles = [t for t in [task[4] for task in tasks] if os.path.exists(t)]
                print("Combining individual media, please do not close the program...", end="")
                if is_video:
                    _ffmpeg_concat_batched(tempfiles, output, res=res, normalize=normalize, fps=fps)
                else:
                    concat_func(tempfiles, output, normalize=normalize)
                print(f"{Fore.GREEN}Done combining media.")

    except Exception:
        raise
    finally:
        for f in tempfiles:
            try:
                os.remove(f)
            except (FileNotFoundError, OSError):
                continue