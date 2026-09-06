import sys
import os
import platform
import shutil
import re
import subprocess
import threading
import time
from pathlib import Path

from typing import Literal, Tuple, Dict, Any, Optional, List

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from yt_dlp.networking.exceptions import TransportError

_ACTIVE_PROCS = set()
_cancel_requested = False
_cancel_lock = threading.Lock()


def request_cancel():
    """Set the cooperative cancellation flag; long-running loops stop spawning work."""
    global _cancel_requested
    with _cancel_lock:
        _cancel_requested = True


def cancel_clear():
    """Clear the cooperative cancellation flag for a fresh run."""
    global _cancel_requested
    with _cancel_lock:
        _cancel_requested = False


def cancel_pending():
    """Whether a cancel was requested (thread-safe read)."""
    with _cancel_lock:
        return _cancel_requested


class InsufficientDiskSpaceError(RuntimeError):
    """Raised before a download when the destination lacks safe free space."""


def _size_value(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def estimate_download_size(info):
    """Estimate bytes for one media item without performing network I/O."""
    if not isinstance(info, dict):
        return None
    for key in ("filesize", "filesize_approx"):
        size = _size_value(info.get(key))
        if size is not None:
            return size

    entries = info.get("requested_formats")
    if not entries:
        entries = info.get("formats")
        if not isinstance(entries, list) or len(entries) != 2:
            entries = None
    if not isinstance(entries, list):
        return None
    sizes = [_size_value(item.get("filesize")) or _size_value(item.get("filesize_approx"))
             for item in entries if isinstance(item, dict)]
    if len(sizes) == len(entries) and all(size is not None for size in sizes):
        return sum(sizes)
    return None


def check_download_space(output_location, estimated_bytes, overhead=1.2,
                         reserve=512 * 1024 * 1024):
    """Ensure a conservative amount of free space is available."""
    if estimated_bytes is None:
        return
    try:
        free = shutil.disk_usage(output_location).free
    except (OSError, ValueError):
        return
    required = int(estimated_bytes * overhead + reserve)
    if free < required:
        raise InsufficientDiskSpaceError(
            f"Insufficient disk space: need {required:,} bytes, "
            f"but only {free:,} bytes are free at {output_location}"
        )


def check_compile_disk_space(output_location, total_seconds, max_height=None,
                             overhead=1.5, reserve=2 * 1024 * 1024 * 1024):
    """Preflight disk space before remote clip materialization + concat.

    ``total_seconds`` is the sum of every clip duration about to be downloaded.
    ``max_height`` (from the Max Download Quality setting) drives a
    conservative per-second byte estimate. Raising here - before any download
    starts - prevents a big batch from silently filling the temp disk and
    failing mid-way with Errno 28 (the entire compile then aborts with nothing).
    """
    if not total_seconds or total_seconds <= 0:
        return
    if max_height is None:
        bytes_per_sec = 600 * 1024
    elif max_height <= 480:
        bytes_per_sec = 200 * 1024
    elif max_height <= 720:
        bytes_per_sec = 350 * 1024
    elif max_height <= 1080:
        bytes_per_sec = 500 * 1024
    else:
        bytes_per_sec = 800 * 1024
    estimated_bytes = int(total_seconds * bytes_per_sec)
    check_download_space(output_location, estimated_bytes, overhead=overhead,
                         reserve=reserve)


def run_tracked(cmd, timeout=None, text=False):
    """subprocess.run 等价物，注册进程以便取消时统一终止。

    输出在 daemon reader 线程上读取，主循环用 ``queue.get(timeout=...)``
    非阻塞等待——避免 FFmpeg 探测命令在 CDN 半开连接上挂起时（无输出、
    进程不退出）永久阻塞 ``read``，使 timeout 永远不触发（曾导致 Audio
    Cache 的 CDN 探测卡死整个批次）。
    """
    opts = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE}
    if sys.platform == 'win32':
        opts['creationflags'] = 0x08000000
    p = subprocess.Popen(cmd, **opts)
    _ACTIVE_PROCS.add(p)
    started_at = time.monotonic()
    if not hasattr(p, "poll"):
        # 无 poll 的测试替身（FakeProcess）：communicate 语义保持不变，
        # 避免 reader 线程消耗掉替身的有限输出流。
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f"WARNING: process would not die after kill (stuck in driver call?), abandoning: {cmd[0]}")
            raise
        finally:
            _ACTIVE_PROCS.discard(p)
        if text:
            out = out.decode('utf-8', errors='replace') if out is not None else None
            err = err.decode('utf-8', errors='replace') if err is not None else None
        return subprocess.CompletedProcess(cmd, p.returncode, out, err)

    if timeout is None:
        # 快速路径（无 timeout 需求，如 reverify 本地 m4a 窗口提取）：
        # 主线程阻塞 read —— FFmpeg 退出后管道立即 EOF，read 立刻返回，
        # 无需 reader 线程 / queue 的固定开销。若走 reader 线程路径，
        # Windows 上 FFmpeg 退出后子进程继承的管道句柄会让 read 不即时
        # EOF，items.get(timeout=30) 会卡满整个 timeout（每窗口 30s 的
        # reverify 变慢根因）。
        out_buf = []
        err_buf = []
        while True:
            if cancel_pending():
                p.kill()
            try:
                chunk = p.stdout.read(65536)
            except Exception:
                chunk = b""
            if chunk:
                out_buf.append(chunk)
            else:
                try:
                    chunk_err = p.stderr.read(65536)
                except Exception:
                    chunk_err = b""
                if chunk_err:
                    err_buf.append(chunk_err)
            if p.poll() is not None:
                break
            time.sleep(0.02)
        p.wait()
        out = b"".join(out_buf)
        err = b"".join(err_buf)
        _ACTIVE_PROCS.discard(p)
        if text:
            out = out.decode('utf-8', errors='replace') if out is not None else None
            err = err.decode('utf-8', errors='replace') if err is not None else None
        return subprocess.CompletedProcess(cmd, p.returncode, out, err)

    import queue as _queue
    import threading as _threading

    items = _queue.Queue()

    def _reader(stream, name):
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    items.put((name, None))
                    break
                items.put((name, chunk))
        except BaseException as exc:  # noqa: BLE001 - surface any read error
            items.put((name, exc))

    for stream, name in ((p.stdout, "out"), (p.stderr, "err")):
        _threading.Thread(target=_reader, args=(stream, name), daemon=True).start()

    out_parts = []
    err_parts = []
    remaining = {"out", "err"}
    try:
        while True:
            if cancel_pending():
                p.kill()
            try:
                # 短轮询（0.1s）：进程退出后立即 break，不依赖 reader 线程的 EOF。
                # 旧实现未退出时 wait = timeout（如 10s），进程快速退出但 reader
                # EOF 有延迟（Windows 管道）时 items.get 会卡满整个 timeout——
                # 每个 ffprobe 卡 10s（concat 前 595 个片段 → 99 分钟假死）。
                # 短轮询下进程退出（p.poll() 非 None）在下一次迭代立即 break。
                wait = 0.1
                if timeout is not None:
                    wait = min(wait, max(0.0, timeout - (time.monotonic() - started_at)))
                item = items.get(timeout=wait)
            except _queue.Empty:
                if p.poll() is not None:
                    # 进程已退出但 reader 未 EOF：关闭管道唤醒后收尾。
                    break
                if timeout is not None and time.monotonic() - started_at >= timeout:
                    # 真实超时：到点仍无输出且进程未退出才杀。
                    p.kill()
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f"WARNING: process would not die after kill (stuck in driver call?), abandoning: {cmd[0]}")
                    raise subprocess.TimeoutExpired(cmd, timeout)
                # 进程仍在运行且未到 timeout：继续短轮询等待。0.1s 内无输出不
                # 代表挂起——NVENC 探测等命令全程无输出且需 >0.1s 才退出，
                # 这里直接 kill 会误判超时（曾导致 NVENC 探测 0.17s 被误杀、
                # 编译回退 x264）。
                continue
            name, data = item
            if data is None:
                remaining.discard(name)
            elif isinstance(data, BaseException):
                raise data
            else:
                (out_parts if name == "out" else err_parts).append(data)
            if p.poll() is not None:
                # 进程已退出：drain 已到达的剩余输出（reader 线程可能刚 put 完
                # 数据还没到 EOF），再收尾。短超时保证卡住的 reader 不阻塞。
                break
            if timeout is not None and time.monotonic() - started_at > timeout:
                p.kill()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"WARNING: process would not die after kill (stuck in driver call?), abandoning: {cmd[0]}")
                raise subprocess.TimeoutExpired(cmd, timeout)
        # 进程已退出：短暂 drain 剩余输出（reader 线程可能刚 put 完数据还没
        # 到 EOF）；短超时保证卡住的 reader 永不阻塞主流程。
        while True:
            try:
                item = items.get(timeout=0.1)
            except _queue.Empty:
                break
            name, data = item
            if data is None:
                remaining.discard(name)
                if not remaining:
                    break
            elif isinstance(data, BaseException):
                raise data
            else:
                (out_parts if name == "out" else err_parts).append(data)
    finally:
        # 进程已退出：关闭管道，让仍阻塞在 read 的 reader 线程立刻返回
        # （否则 daemon reader 永久占用，且可能拖住解释器收尾）。
        if p.poll() is not None:
            try:
                p.stdout.close()
            except Exception:
                pass
            try:
                p.stderr.close()
            except Exception:
                pass
        _ACTIVE_PROCS.discard(p)
    # 正常路径也可能在进程未退出时到达这里（双流 EOF 但进程仍存活，
    # 如 FFmpeg 关闭流后卡死在驱动调用）。有界等待 + 强杀兜底，
    # 与 kill 路径的加固一致，绝不永久阻塞主流程。
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"WARNING: process would not die after kill (stuck in driver call?), abandoning: {cmd[0]}")
    out = b"".join(out_parts)
    err = b"".join(err_parts)
    if text:
        out = out.decode('utf-8', errors='replace') if out is not None else None
        err = err.decode('utf-8', errors='replace') if err is not None else None
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def run_tracked_progress(cmd, duration=None, timeout=None, progress_callback=None,
                         stall_timeout=None):
    """Run FFmpeg while forwarding its machine-readable progress output.

    ``stall_timeout`` (seconds) adds a no-data watchdog: if FFmpeg produces no
    output for that long (e.g. a half-open network read), the process is killed
    and ``RemoteAudioStallError`` is raised so callers can refresh and retry
    instead of hanging forever. Reads happen on a daemon thread so a silent
    process can never block the timeout check.
    """
    command = list(cmd)
    if "-progress" not in command:
        command.extend(["-progress", "pipe:1", "-nostats"])
    opts = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "bufsize": 1,
    }
    if sys.platform == "win32":
        opts["creationflags"] = 0x08000000
    started_at = time.monotonic()
    state = {}
    output = []
    p = subprocess.Popen(command, **opts)
    _ACTIVE_PROCS.add(p)
    try:
        # 无显式 stall_timeout 时也走 stall 看门狗：readline 在 reader 线程，
        # 主循环非阻塞等待，FFmpeg 在 CDN 半开连接上挂起时（无输出）能在
        # stall_timeout 后 kill，而不是永远阻塞 timeout 检查。
        effective_stall = stall_timeout if (stall_timeout is not None and stall_timeout > 0) else 30
        return _run_progress_with_stall(p, state, output, duration,
                                        progress_callback, started_at,
                                        timeout, effective_stall, command)
    finally:
        _ACTIVE_PROCS.discard(p)


def _run_progress_with_stall(p, state, output, duration, progress_callback,
                             started_at, timeout, stall_timeout, command):
    import queue as _queue
    import threading as _threading

    items = _queue.Queue()

    def reader():
        try:
            while True:
                line = p.stdout.readline()
                if not line:
                    items.put(None)
                    break
                items.put(line)
        except BaseException as exc:  # noqa: BLE001 - surface any read error
            items.put(exc)

    thread = _threading.Thread(target=reader, name="ffmpeg-progress-reader", daemon=True)
    thread.start()
    try:
        while True:
            if cancel_pending():
                p.kill()
                raise InterruptedError("Operation cancelled by user.")
            try:
                item = items.get(timeout=stall_timeout)
            except _queue.Empty:
                p.kill()
                from sound_reader import RemoteAudioStallError
                raise RemoteAudioStallError(
                    f"No data from FFmpeg for {stall_timeout:g}s; remote URL likely "
                    f"expired or the connection hung")
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            text_line = item.decode("utf-8", errors="replace").strip()
            output.append(text_line)
            if "=" in text_line:
                key, value = text_line.split("=", 1)
                state[key] = value
                if key == "out_time_ms" and progress_callback is not None:
                    try:
                        current = float(value) / 1_000_000
                    except ValueError:
                        continue
                    progress_callback(current, duration, time.monotonic() - started_at)
            if timeout is not None and time.monotonic() - started_at > timeout:
                p.kill()
                raise subprocess.TimeoutExpired(command, timeout)
    finally:
        # U1 同款兜底：EOF 后进程仍可能楔死（v1.0.9 NVENC 驱动类）。裸
        # wait(10) 会把"已完成编码"误报为失败且永不清理进程。
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except OSError:
                pass
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("run_tracked_progress: wedged FFmpeg survived kill; abandoning it.")
        thread.join(timeout=1)
    return subprocess.CompletedProcess(command, p.returncode, "\n".join(output), "")


def register_proc(p):
    _ACTIVE_PROCS.add(p)


def unregister_proc(p):
    _ACTIVE_PROCS.discard(p)


def kill_tracked_procs():
    for p in list(_ACTIVE_PROCS):
        try:
            p.kill()
        except Exception:
            pass
    _ACTIVE_PROCS.clear()


DOWNLOAD_QUALITY_OPTIONS = ["No Limit", "144p", "240p", "360p",
                            "480p", "720p", "1080p", "1440p", "2160p", "4320p"]


def get_bundle_filepath(filepath: str) -> str:
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, filepath)
        else:
            return os.path.join(os.path.dirname(sys.executable), filepath)
    else:
        return os.path.join(Path.cwd(), filepath)


current_platform = platform.system()
if current_platform == "Windows":
    ffmpeg_path = r".\ffmpeg\windows\ffmpeg.exe"
    is_windows = True
elif current_platform == "Darwin":  # macOS
    ffmpeg_path = r"./ffmpeg/osx/ffmpeg"
else:  # Linux
    ffmpeg_path = r"./ffmpeg/linux/ffmpeg"

FFMPEG_PATH = get_bundle_filepath(ffmpeg_path)


def _is_youtube_download_url(url: str) -> bool:
    from urllib.parse import urlsplit
    hostname = (urlsplit(str(url)).hostname or "").lower()
    return any(hostname == marker or hostname.endswith("." + marker)
               for marker in ("youtube.com", "youtu.be", "youtube-nocookie.com"))


def _download_ydl_options(browser_cookies: str | None = None, url: str | None = None) -> dict:
    """Return yt-dlp cookie options without ever passing an anonymous Auto value.

    注意：不预注入 youtube player_client——web_embedded 等客户端给出的媒体
    URL 需要 PO token，解析成功但下载 403（Audio Cache 卡 0 MB/s 的教训）。
    tv_downgraded 的 page-reload 错误由 resolve 层的匿名回退处理。
    """
    value = str(browser_cookies or "").strip()
    if value.casefold() == "none":
        value = ""
    if value.casefold() == "auto":
        value = "firefox"
    if value.startswith("cookiesfile:"):
        cookies_path = value[len("cookiesfile:"):].strip()
        if cookies_path:
            return {"cookies": cookies_path}
        value = ""
    lowered = value.casefold()
    if lowered not in {"", "firefox", "chrome", "edge"}:
        raise ValueError("browser_cookies must be None, firefox, chrome, edge, auto, or a cookies file")
    return {"cookiesfrombrowser": (lowered,)} if lowered else {}


def _download_cookie_candidates(browser_cookies: str | None) -> list[str | None]:
    value = str(browser_cookies or "").strip()
    if value.startswith("cookiesfile:"):
        return [value]
    if value.casefold() == "auto":
        return ["firefox", "chrome", "edge", None]
    return [browser_cookies]


def _download_transfer_options(retries):
    return {
        "continuedl": True,
        "nopart": False,
        "retries": retries,
        "fragment_retries": retries,
        "file_access_retries": retries,
    }


def _supports_parallel_fragments(info):
    if not isinstance(info, dict):
        return False
    candidates = info.get("formats") or info.get("requested_formats") or [info]
    if isinstance(candidates, dict):
        candidates = [candidates]
    segmented_protocols = {"m3u8", "m3u8_native", "http_dash_segments", "dash", "f4m"}
    return any(
        isinstance(fmt, dict)
        and (fmt.get("protocol") in segmented_protocols or fmt.get("fragments"))
        for fmt in candidates
    )


def _download_log(logger, message):
    info = getattr(logger, "info", None)
    if callable(info):
        info(message)


def convert_quality_str_to_int(quality: str) -> int:
    if not quality:
        return None

    numbers = re.findall(r'\d+', quality)

    if len(numbers) == 1:
        # Case like '720p' where there is only one number
        return int(numbers[0])
    elif len(numbers) == 2:
        # Case like '256x144' where there are two numbers; return the smaller (height) number
        return min(tuple(map(int, numbers)))
    else:
        return None


def get_single_video_details(url, max_quality: str):
    max_height = convert_quality_str_to_int(max_quality)

    format_str = \
        f'bestvideo[height<={max_height}]+bestaudio/bestvideo[height<=720][fps<=60]+bestaudio/bestvideo[height<={max_height}]/best[height<={max_height}]' \
        if max_quality in DOWNLOAD_QUALITY_OPTIONS and max_quality != DOWNLOAD_QUALITY_OPTIONS[0] else 'bestvideo+bestaudio/best'

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'format': format_str,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)

    if info_dict is not None \
            and info_dict.get('title') not in [None, "[Private video]", "[Deleted video]"] \
            and (info_dict.get('uploader')) \
            and (info_dict.get('original_url') or info_dict.get('url')):
        return {
            'id': info_dict.get('id'),
            'title': info_dict.get('title'),
            'uploader': info_dict.get('uploader'),
            'url': info_dict.get('original_url') or info_dict.get('url'),
        }
    return None


def get_urls(base_url: str):
    def check_video(info_dict):
        if not info_dict.get('entries'):
            return [info_dict.get('original_url') or info_dict.get('url')]
        else:
            return [check_video(x) for x in info_dict['entries']]

    def flatten(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': 'in_playlist',  # Extract only metadata, not the video itself
    }
    attempts = 0
    while True:
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(base_url, download=False)
                break
        except TransportError:
            attempts += 1
            if attempts >= 3:
                raise
            time.sleep(1)

    return list(flatten(check_video(info_dict)))


def get_number_of_vids_in_playlist(playlist_url: str) -> int:
    return len(get_urls(playlist_url))


def is_valid_yt_dlp_url(base_url: str, max_quality: str = None):
    if max_quality and max_quality not in DOWNLOAD_QUALITY_OPTIONS:
        raise Exception("Invalid max quality specified")

    try:
        urls = get_urls(base_url)
    except DownloadError as e:
        cleaned_error = '.'.join(str(e).split(':')[1:])
        raise Exception(
            f"An error occured while retrieving URLs: {str(cleaned_error)}")
    except Exception as e:
        raise Exception(
            f"An unexpected error occured while retrieving URLs. Please try again.\nError: {str(e)}")

    for url in urls:
        try:
            vid_details = get_single_video_details(url, max_quality)
            if vid_details:
                yield vid_details
            else:
                yield Exception("Video is privated, deleted, or otherwise unavailable.\nIf you know the video is public, try raising your max allowed quality in settings.")
        except DownloadError as e:
            cleaned_error = '.'.join(str(e).split(':')[1:])
            if 'Requested format' in cleaned_error:
                yield Exception("No video found at or below the max allowable quality.\nTry raising your max quality in settings.")
            else:
                yield Exception(
                    f"An error occured while retrieving URLs: {str(cleaned_error)}")
        except Exception as e:
            yield Exception(
                f"An unexpected error occured while retrieving URLs. Please try again.\nError: {str(e)}")


def download_video(url: str, filename: str, output_location: str, max_quality: str, max_speed: int, logger, n_retries: int = 3, browser_cookies: str | None = None) -> Tuple[bool, str]:
    logger.reset_total_progress(100)
    os.makedirs(output_location, exist_ok=True)

    max_height = convert_quality_str_to_int(max_quality)

    format_str = \
        f'bestvideo[height<={max_height}]+bestaudio/bestvideo[height<=720][fps<=60]+bestaudio/bestvideo[height<={max_height}]/best[height<={max_height}]' \
        if max_quality in DOWNLOAD_QUALITY_OPTIONS and max_quality != DOWNLOAD_QUALITY_OPTIONS[0] else 'bestvideo+bestaudio/best'

    last_error = None
    with open(os.devnull, 'w') as devnull:
        for cookie_source in _download_cookie_candidates(browser_cookies):
            ydl_opts = {
                'outtmpl': f"{filename}.%(ext)s",
                'quiet': True,
                'logger': logger,
                'progress_hooks': [logger.hook],
                'ffmpeg_location': FFMPEG_PATH
            }
            ydl_opts.update(_download_transfer_options(n_retries))
            ydl_opts.update(_download_ydl_options(cookie_source, url=url))
            if max_speed > 0:
                ydl_opts['limit_rate'] = f"{max_speed}K"
            attempts = 0
            while attempts < n_retries:
                if attempts > 0:
                    ydl_opts.pop('concurrent_fragment_downloads', None)
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull

                try:
                    with YoutubeDL(ydl_opts) as ydl:
                        video_info = ydl.extract_info(url, download=False)
                    try:
                        estimated = estimate_download_size(video_info)
                    except Exception:
                        estimated = None
                    check_download_space(output_location, estimated)
                    ydl_opts['format'] = format_str
                    parallel = _supports_parallel_fragments(video_info)
                    if parallel and attempts == 0:
                        ydl_opts['concurrent_fragment_downloads'] = 4
                        _download_log(logger, 'Download resume enabled; parallel fragments enabled')
                    elif attempts > 0 and parallel:
                        _download_log(logger, 'Parallel download failed; retrying sequentially')
                    else:
                        _download_log(logger, 'Download resume enabled; sequential transfer')
                    has_video = any(
                        (fmt.get('vcodec') != 'none' and fmt.get('acodec') != 'none')
                        or (fmt.get('video_ext') != 'none' and fmt.get('audio_ext') != 'none')
                        for fmt in video_info.get('formats', [])
                    ) or (
                        video_info.get('vcodec') and video_info.get('vcodec') != 'none'
                    )
                    if not has_video:
                        return True, None
                    with YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(url, download=True)

                    file_ext = info_dict.get('ext', 'mp4')
                    source_file = f"{filename}.{file_ext}"
                    output_file = os.path.join(output_location, f"{os.path.basename(filename)}.{file_ext}")
                    shutil.move(source_file, output_file)
                    return True, output_file
                except Exception as e:
                    last_error = e
                    attempts += 1
                    if attempts >= n_retries:
                        break
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
    return False, str(last_error) if last_error else "download failed"


def download_audio(url: str, filename: str, output_location: str, max_speed: int, logger, n_retries: int = 10, browser_cookies: str | None = None) -> Tuple[bool, str]:
    logger.reset_total_progress(100)

    os.makedirs(output_location, exist_ok=True)
    last_error = None
    with open(os.devnull, 'w') as devnull:
        for cookie_source in _download_cookie_candidates(browser_cookies):
            ydl_opts = {
                'outtmpl': f"{filename}.%(ext)s",
                'format': 'bestaudio/best',
                'noplaylist': True,
                'quiet': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'progress_hooks': [logger.hook],
                'ffmpeg_location': FFMPEG_PATH
            }
            ydl_opts.update(_download_transfer_options(n_retries))
            ydl_opts.update(_download_ydl_options(cookie_source, url=url))
            if max_speed > 0:
                ydl_opts['limit_rate'] = f"{max_speed}K"
            attempts = 0
            while attempts < n_retries:
                if attempts > 0:
                    ydl_opts.pop('concurrent_fragment_downloads', None)
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull

                try:
                    with YoutubeDL(ydl_opts) as ydl:
                        metadata = ydl.extract_info(url, download=False)
                        try:
                            estimated = estimate_download_size(metadata)
                        except Exception:
                            estimated = None
                        check_download_space(output_location, estimated)
                        if _supports_parallel_fragments(metadata) and attempts == 0:
                            ydl_opts['concurrent_fragment_downloads'] = 4
                            _download_log(logger, 'Download resume enabled; parallel fragments enabled')
                        elif attempts > 0:
                            _download_log(logger, 'Parallel download failed; retrying sequentially')
                        else:
                            _download_log(logger, 'Download resume enabled; sequential transfer')
                        info_dict = ydl.extract_info(url, download=True)
                        file_ext = ydl.params['postprocessors'][0].get(
                            'preferredcodec', info_dict.get('ext', 'mp3'))
                        source_file = f"{filename}.{file_ext}"
                        output_file = os.path.join(output_location, f"{os.path.basename(filename)}.{file_ext}")
                        shutil.move(source_file, output_file)
                        return True, output_file
                except Exception as e:
                    last_error = e
                    attempts += 1
                    if attempts >= n_retries:
                        break
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
    return False, str(last_error) if last_error else "download failed"


class MediaUpload:
    def __init__(self, path: str, type: Literal['video', 'audio'], is_url: bool = False,
                 url: Optional[str] = None, source: Any = None):
        self.path = path
        self.type = type
        self.is_url = is_url
        self.url = url
        self.source = source

    def get_path(self) -> str:
        return self.path

    def set_path(self, path: str):
        self.path = path

    def get_type(self) -> Literal['video', 'audio']:
        return self.type

    def set_type(self, type: Literal['video', 'audio']):
        self.type = type

    def get_is_url(self) -> bool:
        return self.is_url

    def set_is_url(self, is_url: bool):
        self.is_url = is_url

    def get_url(self) -> Optional[str]:
        return self.url

    def get_source(self) -> Any:
        return self.source

    def set_source(self, source: Any):
        self.source = source
