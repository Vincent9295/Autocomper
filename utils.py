import sys
import os
import platform
import shutil
import re
import subprocess
import time
from pathlib import Path

from typing import Literal, Tuple, Dict, Any, Optional, List

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from yt_dlp.networking.exceptions import TransportError

_ACTIVE_PROCS = set()


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
    """subprocess.run 等价物，注册进程以便取消时统一终止。"""
    opts = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE}
    if sys.platform == 'win32':
        opts['creationflags'] = 0x08000000
    p = subprocess.Popen(cmd, **opts)
    _ACTIVE_PROCS.add(p)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # 进程卡在驱动层调用（如睡眠唤醒后 NVENC 死锁）时 TerminateProcess
            # 也杀不掉，wait 会永久挂起冻结整个 app。放弃僵尸进程照常报错，
            # 让上层回退/重试，僵尸由 OS 自行回收。
            print(f"WARNING: process would not die after kill (stuck in driver call?), abandoning: {cmd[0]}")
        raise
    finally:
        _ACTIVE_PROCS.discard(p)
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
        if stall_timeout is not None and stall_timeout > 0:
            return _run_progress_with_stall(p, state, output, duration,
                                            progress_callback, started_at,
                                            timeout, stall_timeout, command)
        while True:
            line = p.stdout.readline()
            if line:
                text_line = line.decode("utf-8", errors="replace").strip()
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
            elif p.poll() is not None:
                break
            if timeout is not None and time.monotonic() - started_at > timeout:
                p.kill()
                raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, p.returncode, "\n".join(output), "")
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
        p.wait(timeout=10)
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


def _download_ydl_options(browser_cookies: str | None = None) -> dict:
    """Return yt-dlp cookie options without ever passing an anonymous Auto value."""
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
        except Exception:
            raise

    return list(flatten(check_video(info_dict)))


def get_number_of_vids_in_playlist(playlist_url: str) -> int:
    try:
        return len(get_urls(playlist_url))
    except:
        raise


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
            ydl_opts.update(_download_ydl_options(cookie_source))
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
            ydl_opts.update(_download_ydl_options(cookie_source))
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
