#!/usr/bin/env python
"""compile.py — native FFmpeg video/audio compilation."""

import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile

from colorama import Fore, Style

from utils import FFMPEG_PATH
import sys
import os

MERGE_THRESHOLD = 2  # seconds

_subprocess_opts = {}
if sys.platform == 'win32':
    _subprocess_opts['creationflags'] = 0x08000000


def _ffprobe(input_file: str):
    """Run ffmpeg -i to probe container metadata (fast, reads header only)."""
    try:
        cmd = [FFMPEG_PATH, '-hide_banner', '-i', input_file]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, **_subprocess_opts)
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
    """Return total duration of the media file in seconds, or None.

    优先 stream=duration（精确到帧），回退 format=duration，最后 ffmpeg -i。
    """
    # 方法 1: ffprobe stream duration (most accurate, works with 2017+ FFmpeg)
    for stream in ('v:0', 'a:0'):
        try:
            cmd = [
                FFMPEG_PATH, '-v', 'error',
                '-select_streams', stream,
                '-show_entries', 'stream=duration',
                '-of', 'csv=p=0',
                input_file,
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, **_subprocess_opts)
            val = out.stdout.strip()
            if val and val != 'N/A':
                return float(val)
        except Exception:
            pass

    # 方法 2: format duration (container-level fallback)
    try:
        cmd = [
            FFMPEG_PATH, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            input_file,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, **_subprocess_opts)
        val = out.stdout.strip()
        if val and val != 'N/A':
            return float(val)
    except Exception:
        pass

    # 方法 3: ffmpeg -i (pre-2017 FFmpeg, WAV files, etc.)
    stderr = _ffprobe(input_file)
    m = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', stderr)
    if m:
        h, mi, s, ms = map(int, m.groups())
        return h * 3600 + mi * 60 + s + ms / 100
    return None


# ═══ VIDEO cut / concat ═══════════════════════════════════════════════

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

    video_codec = ['-c:v', 'h264_nvenc', '-preset', '3', '-pix_fmt', 'yuv420p',
                   '-rc-lookahead', '0', '-sar', '1:1']
    if fps and fps > 0:
        video_codec += ['-r', str(int(fps))]
    audio_codec_tmp = ['-c:a', 'flac']      # FLAC 无编码延迟，时长精确
    audio_codec_out = ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']
    mem_opts = ['-threads', '2']

    if n == 1:
        s, e = timestamps[0]
        dur = e - s
        af = f'atrim=0:{dur}'
        if normalize:
            af += ',loudnorm'
        cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error'] + mem_opts + [
            '-accurate_seek',
            '-ss', str(s), '-to', str(e),
            '-i', input_file,
            '-af', af,
            '-avoid_negative_ts', 'make_zero',
            '-vsync', 'cfr', '-shortest',
        ] + video_codec + audio_codec_tmp
        if res:
            w, h = res
            cmd.extend(['-vf', f'scale={w}:{h}:force_original_aspect_ratio=decrease,'
                               f'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2'])
        cmd.append(output_file)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, **_subprocess_opts)
        if result.returncode != 0:
            raise Exception(f"FFmpeg cut failed for {os.path.basename(input_file)}"
                            f"\n  rc={result.returncode}\n  stderr: {result.stderr}\n  stdout: {result.stdout}")
        return True

    # ── single-pass trim+concat filter graph ──
    parts = []
    if res:
        w, h = res
        for i, (s, e) in enumerate(timestamps):
            parts.append(f'[0:v]trim={s}:{e},scale={w}:{h}:force_original_aspect_ratio=decrease,'
                        f'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1:1,'
                        f'setpts=PTS-STARTPTS[v{i}]')
            parts.append(f'[0:a]atrim={s}:{e},asetpts=PTS-STARTPTS[a{i}]')
    else:
        for i, (s, e) in enumerate(timestamps):
            parts.append(f'[0:v]trim={s}:{e},setsar=1:1,setpts=PTS-STARTPTS[v{i}]')
            parts.append(f'[0:a]atrim={s}:{e},asetpts=PTS-STARTPTS[a{i}]')
    v_srcs = ''.join(f'[v{i}]' for i in range(n))
    a_srcs = ''.join(f'[a{i}]' for i in range(n))
    parts.append(f'{v_srcs}concat=n={n}:v=1:a=0[outv]')
    if normalize:
        parts.append(f'{a_srcs}concat=n={n}:v=0:a=1,loudnorm,aresample=async=1:first_pts=0[outa]')
    else:
        parts.append(f'{a_srcs}concat=n={n}:v=0:a=1,aresample=async=1:first_pts=0[outa]')
    filter_complex = ';'.join(parts)

    cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error', '-threads', '2',
           '-accurate_seek', '-i', input_file,
           '-filter_complex', filter_complex,
           '-map', '[outv]', '-map', '[outa]'] + video_codec + [
           '-c:a', 'aac', '-b:a', '128k', '-ar', '44100',
           '-avoid_negative_ts', 'make_zero',
           '-vsync', 'cfr', '-shortest']
    cmd.append(output_file)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, **_subprocess_opts)
    if result.returncode != 0:
        raise Exception(f"FFmpeg trim+concat failed for {os.path.basename(input_file)}"
                        f"\n  rc={result.returncode}\n  stderr: {result.stderr}\n  stdout: {result.stdout}")
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
            '-map', '[outv]', '-map', '[outa]',
            '-c:v', 'h264_nvenc', '-preset', '3', '-pix_fmt', 'yuv420p',
            '-rc-lookahead', '0', '-sar', '1:1',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '44100',
            '-vsync', 'cfr', '-shortest']
    if fps and fps > 0:
        cmd.insert(cmd.index('-shortest'), str(int(fps)))
        cmd.insert(cmd.index(str(int(fps))), '-r')
    cmd.append(output_file)

    # 验证所有输入文件有视频流
    for i, fp in enumerate(file_list):
        if not os.path.exists(fp):
            raise Exception(f"FFmpeg concat: file {i} missing: {fp}")
        sz = _get_video_size(fp)
        if sz[0] is None:
            raise Exception(f"FFmpeg concat: file {i} has no video stream: {fp}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, **_subprocess_opts)
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


def _ffmpeg_concat_batched(file_list, output_file, res=None, normalize=False, batch_size=6, fps=None):
    """Batched concat for large file lists."""
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
            batch_out = os.path.join(temp_dir, f"_batch{bi}.mp4")
            batch_files.append(batch_out)
            print(f"  Batch {bi + 1}/{len(batches)} ({len(batch)} files)...")
            ok = _ffmpeg_concat(batch, batch_out, res=res, normalize=normalize, fps=fps)
            if not ok:
                raise Exception(f"Batch {bi + 1} failed")
        print(f"  Final merge ({len(batch_files)} files)...")
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
        cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error', '-threads', '2',
               '-ss', str(s), '-to', str(e), '-i', input_file,
               '-avoid_negative_ts', 'make_zero'] + audio_codec
        if normalize:
            cmd.extend(['-af', 'loudnorm'])
        cmd.append(output_file)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, **_subprocess_opts)
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, **_subprocess_opts)
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

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            tempfiles = []
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
                            timestamps[i] = (timestamps[i][0], timestamps[i + 1][1])
                            timestamps.remove(timestamps[i + 1])
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