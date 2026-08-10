import configparser
import os
import shutil
import queue
import re
import shutil
import sys
import subprocess
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


def runtime_lib_directory(executable=None, meipass=None, source_dir=None, frozen=False):
    """Return the bundled lib directory for frozen or source execution."""
    if meipass:
        return Path(meipass) / "lib"
    if frozen and executable:
        return Path(executable).resolve().parent / "lib"
    return Path(source_dir or Path(__file__).resolve().parent) / "lib"


def _register_dll_directory(path):
    if hasattr(os, "add_dll_directory") and os.path.isdir(path):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(path))


if hasattr(os, "add_dll_directory"):
    _register_dll_directory(runtime_lib_directory(
        executable=sys.executable,
        meipass=getattr(sys, "_MEIPASS", None),
        frozen=bool(getattr(sys, "frozen", False)),
    ))
    # Windows Store Python sandbox fix: add CUDA Toolkit bin to DLL search path
    _cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'
    _register_dll_directory(_cuda_bin)

import onnxruntime as ort

import threading
import tkinter as tk
import tkinter.font as tkfont
import webbrowser
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import proglog

import sv_ttk
from colorama import Fore, Style
from kthread import KThread
from PIL import Image, ImageTk
from proglog import ProgressBarLogger
from progress import (ProgressThrottle, ProgressWidgetAdapter,
                      compose_progress_title, format_compile_progress,
                      format_fetch_progress, format_transfer_progress)

from compile import _get_video_duration, compile_vid, get_video_codec
from config import VERSION, REPO_URL
from custom_tooltip import CustomHovertip
from sound_reader import (RemoteAudioIncompleteError, RemoteAudioStallError,
                          get_timestamps, _detection_cache_args)
from remote_media import (MediaSource, fetch_audio_cache, fetch_segment,
                           PlaylistDescriptor, PlaylistEntry, describe_input,
                           expand_input, resolve_source, select_audio_candidate,
                           stable_source_id, SourceResolveError,
                           apply_video_quality_limit,
                           source_from_hydrated_entry,
                           preflight_cookie_source,
                           _audio_cache_format_identity)
from remote_cache import CacheStore
from remote_rate import LimitedRefresher, ResolveLimiter
from utils import (DOWNLOAD_QUALITY_OPTIONS, FFMPEG_PATH, MediaUpload,
                     convert_quality_str_to_int, download_audio, download_video,
                     get_bundle_filepath, kill_tracked_procs, run_tracked,
                     run_tracked_progress)

VIDEO_INPUT = [("Video Files",  "*.mp4 *.avi *.mkv *.m4v *.mov")]
VIDEO_OUTPUT = [("Video Files", "*.mp4"), ("All Files", "*.*")]
AUDIO_INPUT = [("Audio Files",  "*.mp3 *.wav *.flac")]
AUDIO_OUTPUT = [("Audio Files", "*.mp3"), ("All Files", "*.*")]

DEFAULT_SETTINGS = {
    'keep_downloaded_vids': False,
    'download_path': "No location selected!",
    'max_quality': "No Limit",
    'max_download_speed': '0',
    'output_text_path': "No file selected!",
    'remote_cache_path': str(CacheStore().root),
}

REMOTE_MODES = ("Remote Stream", "Audio Cache", "Full Download")
REMOTE_BROWSER_COOKIES = ("Auto", "None", "Firefox", "Chrome", "Edge", "Cookies File…")
PLAYLIST_PAGE_SIZE = 30
MAX_PLAYLIST_ENTRIES = 1000
REMOTE_MODE_TOOLTIP_TEXT = (
    "Remote Stream: read remote audio directly and fetch video only when needed.\n"
    "Audio Cache: download compressed audio once, then detect from the local cache.\n"
    "Full Download: use the complete remote-video download workflow."
)
REMOTE_BROWSER_COOKIES_TOOLTIP_TEXT = (
    "Auto: try direct access first, then browser cookies when the site requires them.\n"
    "None: never read browser cookies.\n"
    "Firefox, Chrome, or Edge: read cookies from that browser. The browser must be\n"
    "installed and logged in to the account that can access the remote media.\n"
    "On Windows, only Firefox is reliable. Chrome 127+ encrypts its cookies\n"
    "(App-Bound) and Edge locks its database while the browser is open.\n"
    "Cookies File…: read cookies from a cookies.txt file (exported with the\n"
    "'Get cookies.txt LOCALLY' browser extension). Works with any browser.\n"
    "When Cookies File… is selected, a file picker opens to choose the file."
)
REMOTE_CONCURRENCY_TOOLTIP_TEXT = (
    "How many remote video clips to download at once while preparing clips.\n"
    "Each worker runs its own FFmpeg process, so higher values use more CPU/disk\n"
    "and can make the whole PC feel sluggish. On a busy or weaker machine, lower\n"
    "this to 1-2 if you notice lag during 'Preparing clips'.\n"
    "Higher values finish faster on fast connections; lower values save bandwidth.\n"
    "Default: 5. Disabled while a run is in progress."
)
REMOTE_CONCURRENCY_DEFAULT = 5
REMOTE_CONCURRENCY_MIN = 1
REMOTE_CONCURRENCY_MAX = 16
REMOTE_CACHE_TOOLTIP_TEXT = (
    "Remote Stream, Audio Cache, reverify, and downloaded segment files are stored here.\n"
    "Full Download does not necessarily use this cache. The cache can be cleared, but\n"
    "the next run may need to download or detect the remote media again.\n\n"
    "External audio files are not identified automatically from their contents.\n"
    "To use one safely, it must be associated with the matching VOD source ID and\n"
    "must cover the same VOD timeline. For Bilibili multi-part videos, each part\n"
    "has its own source ID. Audio files stay local and are never uploaded."
)
EXTERNAL_AUDIO_TOOLTIP_TEXT = (
    "Import audio downloaded separately for the selected Remote VOD.\n"
    "Select exactly one YouTube, Twitch, or Bilibili URL in the main list first.\n"
    "The audio must cover the same VOD timeline; AutoComper cannot identify its\n"
    "source from audio content alone. Bilibili multi-part videos are imported\n"
    "one part at a time. The file is converted to m4a and stays local."
)


def paginate_sources(sources, page_size=PLAYLIST_PAGE_SIZE):
    """Split playlist sources into ordered pages, rejecting oversized playlists."""
    if len(sources) > MAX_PLAYLIST_ENTRIES:
        raise ValueError(f"Playlist contains more than {MAX_PLAYLIST_ENTRIES} entries")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return [sources[index:index + page_size]
            for index in range(0, len(sources), page_size)]


def selected_sources(sources, selected_ids):
    """Return selected sources in their original order."""
    selected_ids = set(selected_ids)
    return [source for source in sources if stable_source_id(source) in selected_ids]


def selected_playlist_entries(descriptor, selected_ids, selected_order=None):
    """Return confirmed playlist entries in descriptor or explicit order."""
    selected_ids = set(selected_ids)
    entries = {entry.entry_id: entry for entry in descriptor._entries}
    if selected_order is not None:
        return [entries[entry_id] for entry_id in selected_order
                if entry_id in selected_ids and entry_id in entries]
    return [entry for entry in descriptor._entries if entry.entry_id in selected_ids]


def hydration_update_is_current(entry_id, page_index, generation, current_page,
                                 current_generation, visible_entry_ids):
    """Return whether a worker result may update the visible page."""
    return (page_index == current_page and generation == current_generation
            and entry_id in visible_entry_ids)


def playlist_order_map(selected_order):
    """Return one-based visible order numbers for selected playlist entries."""
    return {entry_id: index for index, entry_id in enumerate(selected_order, 1)}


def playlist_entry_display_date(entry):
    """Return the best date label for a playlist entry without network access."""
    upload_date = str(getattr(entry, "upload_date", "") or "")
    if upload_date:
        return upload_date
    platform = str(getattr(entry, "platform", "") or "").casefold()
    if "bilibili" in platform:
        match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", str(getattr(entry, "title", "") or ""))
        if match:
            return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return "Unknown"


def playlist_tree_values(entry, selected_ids, selected_order):
    """Build Tk-independent values for one playlist row."""
    metadata = getattr(entry, "metadata", {}) or {}
    part = playlist_entry_part_number(entry)
    part_count = metadata.get("part_count")
    if part is None:
        part_label = "Unknown"
    elif part_count is not None:
        part_label = f"{part}/{part_count}"
    else:
        part_label = part
    failed = bool(metadata.get("hydration_failed"))
    display_date = "Unknown" if failed else playlist_entry_display_date(entry)
    missing_metadata = (
        entry.title == "Unknown"
        or entry.duration is None
        or display_date == "Unknown"
    )
    loading = bool(metadata.get("hydration_pending")) and missing_metadata and not failed
    return (
        playlist_order_map(selected_order).get(entry.entry_id, "")
        if entry.entry_id in selected_ids else "",
        "[x]" if entry.entry_id in selected_ids else "[ ]",
        "Loading..." if loading else entry.title,
        part_label,
        format_playlist_duration(entry.duration),
        display_date,
        "Metadata failed" if failed else ("Loading..." if loading else "Ready"),
    )


def format_playlist_duration(duration):
    """Format playlist seconds as an unbounded HH:MM:SS label."""
    if duration is None:
        return "Unknown"
    try:
        total = max(0, int(float(duration)))
    except (TypeError, ValueError, OverflowError):
        return "Unknown"
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def extract_part_number(title):
    """Return a one-based part/episode number from a title, when recognizable."""
    text = str(title or "")
    patterns = (
        r"第\s*(\d+)\s*(?:部分|集)",
        r"\b[Pp]\s*0*(\d+)\b",
        r"\bPart\s+0*(\d+)\b",
        r"\b[Pp]\s*0*(\d+)\s*/\s*\d+\b",
        r"\bPart\s+0*(\d+)\s*/\s*\d+\b",
        r"第\s*(\d+)\s*/\s*\d+\s*部分",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def playlist_entry_part_number(entry):
    """Return an entry part number from metadata, then from its title."""
    metadata = getattr(entry, "metadata", {}) or {}
    if metadata.get("part_number") is not None:
        try:
            return int(metadata["part_number"])
        except (TypeError, ValueError):
            pass
    return extract_part_number(getattr(entry, "title", ""))


def sort_playlist_entries(entries):
    """Return entries sorted by group/date/title/part, preserving unknown-part index."""
    def normalized_title(title):
        text = str(title or "Unknown")
        text = re.sub(r"第\s*\d+\s*(?:部分|集)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b[Pp]\s*0*\d+\b", "", text)
        text = re.sub(r"\bPart\s+0*\d+\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b0*\d+\s*/\s*\d+\b", "", text)
        return " ".join(text.split()).casefold()

    def key(entry):
        metadata = getattr(entry, "metadata", {}) or {}
        group = str(metadata.get("group") or metadata.get("series") or "").casefold()
        date = str(getattr(entry, "upload_date", "") or "")
        raw_title = str(getattr(entry, "title", "") or "Unknown")
        title = normalized_title(raw_title)
        part = playlist_entry_part_number(entry)
        return (group, date, title, part is None, part if part is not None else entry.index, entry.index)

    return sorted(entries, key=key)


def resolve_playlist_entries(
    entries,
    selected_ids,
    resolver=None,
    expander=None,
    browser_cookies=None,
    failure_logger=None,
    status_logger=None,
    expansion_stats=None,
):
    """Resolve confirmed entries independently, keeping successful sources."""
    resolver = resolver or resolve_source
    expander = expander or expand_input
    selected_ids = set(selected_ids)
    sources = []
    seen_source_ids = set()
    expansion_stats = expansion_stats if expansion_stats is not None else {}
    expansion_stats.setdefault("expanded_entries", 0)
    expansion_stats.setdefault("expanded_parts", 0)

    def add_source(source):
        source_id = stable_source_id(source) if isinstance(source, MediaSource) else source
        if source_id not in seen_source_ids:
            seen_source_ids.add(source_id)
            sources.append(source)

    def resolve_entry(entry):
        """Reuse already-hydrated metadata when available; otherwise resolve fresh."""
        cached_info = (entry.metadata or {}).get("_resolved_info")
        if isinstance(cached_info, Mapping):
            try:
                return source_from_hydrated_entry(entry)
            except Exception:
                pass
        return resolver(entry.webpage_url, browser_cookies=browser_cookies)

    for entry in entries:
        if entry.entry_id not in selected_ids:
            continue
        try:
            add_source(resolve_entry(entry))
        except Exception as exc:
            is_bilibili_playlist = (
                isinstance(exc, SourceResolveError)
                and "bilibili" in str(getattr(entry, "platform", "")).casefold()
                and "source is a playlist" in str(exc).casefold()
            )
            if is_bilibili_playlist:
                try:
                    expanded = expander(
                        entry.webpage_url, browser_cookies=browser_cookies
                    )
                    expansion_stats["expanded_entries"] += 1
                    message = f"Expanded {entry.webpage_url} into {len(expanded)} parts"
                    (status_logger or print)(message)
                    source_count_before = len(sources)
                    for source in expanded:
                        add_source(source)
                    expansion_stats["expanded_parts"] += len(sources) - source_count_before
                    continue
                except Exception as expansion_error:
                    exc = expansion_error
            message = f"Remote source failed ({entry.webpage_url}): {exc}"
            (failure_logger or print)(message)
    return sources


def media_uploads_for_sources(sources, media_type="video"):
    """Wrap selected remote sources without resolving or downloading media."""
    return [MediaUpload(
        source.display_name or source.source_url,
        media_type,
        True,
        source.source_url,
        source,
    ) for source in sources]


def format_cache_size(size: int) -> str:
    """Format cache bytes for the compact settings display."""
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


def cache_open_command(platform: str, path: str) -> list[str]:
    """Return the non-Windows folder opener command for a platform."""
    if platform == "darwin":
        return ["open", path]
    return ["xdg-open", path]


def normalize_remote_settings(data):
    """Return preset data with safe defaults for remote settings."""
    normalized = dict(data)
    mode = normalized.get("remote_mode", "Remote Stream")
    cookies = normalized.get("remote_browser_cookies", "Auto")
    normalized["remote_mode"] = mode if mode in REMOTE_MODES else "Remote Stream"
    normalized["remote_browser_cookies"] = (
        cookies if _valid_cookie_label(cookies) else "Auto"
    )
    if "remote_download_concurrency" in normalized:
        try:
            concurrency = int(normalized["remote_download_concurrency"])
        except (TypeError, ValueError):
            concurrency = REMOTE_CONCURRENCY_DEFAULT
        if concurrency < REMOTE_CONCURRENCY_MIN:
            concurrency = REMOTE_CONCURRENCY_DEFAULT
        elif concurrency > REMOTE_CONCURRENCY_MAX:
            concurrency = REMOTE_CONCURRENCY_MAX
        normalized["remote_download_concurrency"] = concurrency
    normalized.pop("remote_cache_size", None)
    return normalized


def _valid_cookie_label(value):
    """Return whether a stored cookie label is a known choice or a file label."""
    if value in REMOTE_BROWSER_COOKIES:
        return True
    return str(value).casefold().startswith("cookies file")


def select_remote_cache_store(current_store: CacheStore, selected_path) -> CacheStore:
    """Return a validated store for a selected path, or the current store on cancel."""
    if not selected_path:
        return current_store
    store = CacheStore(selected_path)
    store.ensure_ready()
    return store


def restore_remote_cache_path(app, selected_path, warning_func=None) -> CacheStore:
    """Apply a preset cache path without replacing the current store on failure."""
    try:
        store = select_remote_cache_store(app.remote_cache_store, selected_path)
    except OSError as exc:
        warning = f"Could not use preset remote cache folder:\n{exc}"
        if warning_func is not None:
            warning_func(warning)
        else:
            print(f"WARNING: {warning}")
        return app.remote_cache_store

    app.remote_cache_store = store
    app.remote_cache_path.set(str(store.root))
    return store


def prepare_remote_cache_store(store: CacheStore) -> CacheStore:
    """Validate the selected cache store once for a processing session."""
    store.ensure_ready()
    return store

os.environ['FFMPEG_BINARY'] = FFMPEG_PATH


def get_photo_icon(path: str, width: int = 25, height: int = 25) -> ImageTk.PhotoImage:
    image_path = get_bundle_filepath(path)
    image = Image.open(image_path).convert(mode='RGBA')
    image = image.resize((width, height))
    return ImageTk.PhotoImage(image)


def clean_filename(filename: str, replacement: str = "_") -> str:
    unsafe_characters = r'[<>:"/\\|?*]'
    safe_name = re.sub(unsafe_characters, replacement, filename)
    safe_name = safe_name.strip()  # .replace(" ", replacement)
    return safe_name[:150]


def _elide_middle(text: str, max_pixels: int, measure) -> str:
    """Middle-truncate long names so head (e.g. part number) and tail (e.g. date) stay visible."""
    if measure(text) <= max_pixels:
        return text
    budget = (max_pixels - measure('…')) // 2
    head = ''
    for ch in text:
        if measure(head + ch) > budget:
            break
        head += ch
    tail = ''
    for ch in reversed(text):
        if measure(ch + tail) > budget:
            break
        tail = ch + tail
    return head + '…' + tail


_temp_dir_obj = tempfile.TemporaryDirectory()  # 保持引用防止 GC 立即删除
TEMP_DIR = _temp_dir_obj.name

# 超过该秒数（30 分钟）后，处理某个远程源前会重新解析一次，避免开头一次性
# 解析的大量 URL 在排到后面时已经过期。
REMOTE_REFRESH_THRESHOLD = 1800.0
# 预览前，若源已超过该秒数（5 分钟）未刷新则主动重新解析，避免用过期的
# signed URL 拉预览段导致 403。
PREVIEW_REFRESH_THRESHOLD = 300.0


def ensure_temp_dir():
    """Recreate the shared temporary root after a successful run cleans it."""
    os.makedirs(TEMP_DIR, exist_ok=True)


TEMP_CHILD_PREFIXES = ("remote-compile-", "reverify-", "preview-")


def cleanup_temp_children(temp_root=None, prefixes=TEMP_CHILD_PREFIXES):
    """Remove only application-owned temporary children below the session root."""
    root = Path(temp_root or TEMP_DIR)
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.name.startswith(tuple(prefixes)):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass


def processing_uploads_for_batch(uploaded_videos, local_entries, remote_entries):
    """Return resolved local and remote uploads in their original batch order."""
    local_ids = {id(upload) for upload in local_entries}
    remote_ids = {id(upload) for upload, _ in remote_entries}
    return [upload for upload in uploaded_videos
            if id(upload) in local_ids or id(upload) in remote_ids]


def create_inference_session(selected_model, use_gpu=True):
    """Create the one ONNX session shared by a processing batch."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if use_gpu:
        try:
            return ort.InferenceSession(
                selected_model, sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception:
            return ort.InferenceSession(
                selected_model, sess_options,
                providers=['CPUExecutionProvider'])
    return ort.InferenceSession(
        selected_model, sess_options,
        providers=['CPUExecutionProvider'])


def log_shared_session_providers(shared_session, log_func=print):
    """Report the providers actually selected by the shared ONNX session."""
    providers = shared_session.get_providers()
    label = "GPU (CUDA)" if "CUDAExecutionProvider" in providers else "CPU"
    log_func(f"ONNX inference provider: {label}")


def detection_block_size(remote_mode, user_block_size, is_remote=True):
    """Return the configured block size for every processing mode."""
    return user_block_size


def get_source_display_name(value):
    if isinstance(value, MediaSource):
        return value.display_name or value.source_url or stable_source_id(value)
    if isinstance(value, MediaUpload):
        source = value.get_source()
        if isinstance(source, MediaSource):
            return source.display_name or source.source_url or stable_source_id(source)
        return os.path.basename(str(value.get_path() or value.get_url() or value))
    return os.path.basename(str(value))


def get_source_persistence_name(value):
    if isinstance(value, MediaSource):
        return value.source_url or get_source_display_name(value)
    return str(value)


def _get_external_audio_codec(path):
    result = run_tracked([
        FFMPEG_PATH, "-hide_banner", "-i", str(path),
    ], timeout=60, text=True)
    output = (getattr(result, "stderr", "") or "") + (getattr(result, "stdout", "") or "")
    match = re.search(r"Audio:\s*([^,\s]+)", output, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def source_key(value):
    """Return a stable dictionary key for local paths and remote sources."""
    if isinstance(value, MediaSource):
        return stable_source_id(value)
    return os.path.normcase(os.path.normpath(str(value)))


def remote_mode_actions(mode):
    """Return (resolve_sources, cache_audio, full_download) for a UI mode."""
    return mode != "Full Download", mode == "Audio Cache", mode == "Full Download"


def browser_cookie_setting_value(label, cookies_file=""):
    """Convert the GUI browser-cookie label to the resolver setting."""
    value = str(label or "Auto").strip()
    lowered = value.casefold()
    if lowered == "none":
        return None
    if lowered.startswith("cookies file"):
        cookies_path = str(cookies_file or "").strip()
        if cookies_path:
            return f"cookiesfile:{cookies_path}"
        return None
    return lowered if lowered else None


def remote_detection_input(upload, mode, audio_cache_paths):
    """Select the detection input without changing the source used for compile."""
    if not upload.get_is_url():
        return upload.get_path()
    source = upload.get_source()
    if mode == "Audio Cache":
        return audio_cache_paths[stable_source_id(source)]
    if mode == "Remote Stream":
        return source
    return upload.get_path()


def preserve_remote_result(result, source):
    """Attach original remote identity after detection ran on cached audio."""
    preserved = dict(result)
    preserved["filename"] = source
    preserved["source_url"] = source.source_url
    preserved["source_metadata"] = {
        "platform": source.platform,
        "source_id": source.source_id,
        "display_name": get_source_display_name(source),
    }
    return preserved


def resolve_remote_uploads(
    uploaded_videos,
    resolver=None,
    logger=None,
    browser_cookies=None,
    max_height=None,
    limiter=None,
):
    """Resolve URL uploads without changing local uploads.

    Returns local uploads and ``(upload, source)`` pairs. A bad URL is isolated
    so the remaining batch can continue.
    """
    resolver = resolver or resolve_source
    local_entries = []
    remote_sources = []
    total_remote = sum(1 for upload in uploaded_videos if upload.get_is_url())
    resolved_count = 0
    for upload in uploaded_videos:
        if not upload.get_is_url():
            local_entries.append(upload)
            continue
        resolved_count += 1
        source = upload.get_source()
        try:
            if limiter is not None:
                limiter.wait()
            if source is None:
                source_url = upload.get_url() or upload.get_path()
                if browser_cookies is None:
                    if max_height is not None:
                        source = resolver(source_url, max_height=max_height)
                    else:
                        source = resolver(source_url)
                elif max_height is not None:
                    source = resolver(source_url, browser_cookies=browser_cookies,
                                      max_height=max_height)
                else:
                    source = resolver(source_url, browser_cookies=browser_cookies)
            elif max_height is not None and source.max_height != max_height:
                source = apply_video_quality_limit(source, max_height)
            upload.set_source(source)
            upload.set_path(source.display_name or source.source_url)
            remote_sources.append((upload, source))
            if logger is not None:
                logger(
                    f"Resolving remote source [{resolved_count}/{total_remote}]: "
                    f"{source.display_name or source.source_id or source.source_url}"
                )
        except Exception as exc:
            if logger is not None:
                logger(
                    f"Resolving remote source [{resolved_count}/{total_remote}] failed: {exc}"
                )
            message = f"Remote source failed ({upload.get_url() or upload.get_path()}): {exc}"
            (logger or print)(message)
    if total_remote and not remote_sources and not local_entries:
        raise RuntimeError("No remote sources could be resolved; see the preceding errors.")
    return local_entries, remote_sources


def select_remote_stream_audio(remote_entries, selector=None, logger=None):
    """Prefer fast remote audio candidates without blocking detection on probe errors."""
    selector = selector or select_audio_candidate
    log = logger or print
    for _, source in remote_entries:
        try:
            selected = selector(source, log_func=log)
            if selected is not None:
                log(
                    f"Remote Stream audio selected: format_id={selected.get('format_id') or 'unknown'} "
                    f"abr={selected.get('abr') or selected.get('tbr') or 'unknown'}"
                )
        except Exception as exc:
            log(f"Remote Stream audio probe skipped: {type(exc).__name__}")


def _mark_detection_failure(cache_store, source, precision, block_size, threshold,
                            focus_idx, model, reason):
    """Record a detection failure marker for a remote source so a re-run can retry it."""
    if not isinstance(source, MediaSource) or cache_store is None:
        return
    try:
        cache_store.save_detection_failure(
            *_detection_cache_args(source, model, precision, block_size, threshold, focus_idx),
            {"error": str(reason)[:500], "recorded_at": time.time()},
        )
    except Exception:
        pass


def refresh_remote_source(source, browser_cookies=None):
    """Resolve fresh stream URLs without logging signed URL or cookie details."""
    refreshed = resolve_source(source.source_url, browser_cookies=browser_cookies,
                               max_height=source.max_height)
    if refreshed.platform in {"bilibili", "bilibiliweb"}:
        try:
            select_audio_candidate(refreshed, probe_duration=1, log_func=None)
        except Exception:
            pass
    return refreshed


def materialize_remote_entries(entries, temp_dir, fetcher=fetch_segment,
                               selected_intervals=None, cache_store=None, padding=None,
                               is_video=True, refresh_func=None, failures=None,
                               progress_callback=None, max_parallel=5):
    """Return compile-ready entries with padding normalized exactly once.

    Remote video/audio clips are downloaded concurrently (``max_parallel``
    workers) while preserving input order in the returned list. A failed clip
    is isolated into ``failures`` without blocking the rest of the batch.
    """
    selected_intervals = selected_intervals or {}
    before, after = (padding or (0, 0))
    before, after = float(before), float(after)
    if before < 0 or after < 0:
        raise ValueError("Clip padding cannot be negative!")
    materialized: list[dict] = []
    remote_video_total = sum(
        1 for entry in entries if isinstance(entry.get('filename'), MediaSource)
    )
    refresh_lock = threading.Lock()
    state = {"total": 0, "completed": 0}
    completed_lock = threading.Lock()
    in_flight = {}
    in_flight_lock = threading.Lock()

    def emit(event):
        if progress_callback is not None:
            progress_callback(event)

    def fetch_task(task):
        (entry_index, interval_index, source, start, end, output,
         clip_index, clips_total, video_index, duration, pred) = task
        in_flight_key = (entry_index, interval_index)

        def fetch_progress(current, total_value, elapsed):
            with in_flight_lock:
                if not in_flight:
                    return
                display_key = min(in_flight, key=lambda key: in_flight[key]["order"])
                meta = in_flight[display_key]
            sample = format_fetch_progress(current, total_value, elapsed)
            done = state["completed"]
            total_count = state["total"]
            sample["title"] = (
                f"Preparing clips: {done}/{total_count} "
                f"({int(done / total_count * 100) if total_count else 0}%)"
            )
            sample["current_line"] = (
                f"Video {meta['video']}/{meta['videos_total']} · "
                f"clip {meta['clip']}/{meta['clips_total']} · "
                f"Range {meta['start']:g}-{meta['end']:g}s"
            )
            emit({"kind": "progress", **sample})

        with in_flight_lock:
            in_flight[in_flight_key] = {
                "video": video_index,
                "videos_total": remote_video_total,
                "clip": clip_index,
                "clips_total": clips_total,
                "start": start,
                "end": end,
                "order": (video_index, clip_index),
            }
        emit({
            "kind": "start",
            "video": video_index,
            "videos_total": remote_video_total,
            "clip": clip_index,
            "clips_total": clips_total,
            "start": start,
            "end": end,
            "elapsed": 0.0,
            "name": get_source_display_name(source),
            "completed": state["completed"],
            "total": state["total"],
        })
        try:
            fetch_kwargs = {}
            # 下载区间已在任务构建时按 padding 及相邻 clip 钳制好，
            # 这里直接以区间本身下载，不再追加 padding。
            if not is_video:
                fetch_kwargs["audio_only"] = True
            if refresh_func is not None:
                def guarded_refresh(source_arg):
                    with refresh_lock:
                        return refresh_func(source_arg)
                fetch_kwargs["refresh_func"] = guarded_refresh
            fetch_kwargs["progress_callback"] = fetch_progress
            if cache_store is None:
                fetched = fetcher(source, start, end, output, **fetch_kwargs)
            else:
                fetched = fetcher(source, start, end, output, cache_store=cache_store,
                                  **fetch_kwargs)
        except Exception as exc:
            with in_flight_lock:
                in_flight.pop(in_flight_key, None)
            with completed_lock:
                state["completed"] += 1
                current_completed = state["completed"]
            message = (
                f"Could not fetch {get_source_display_name(source)} "
                f"interval {start:g}-{end:g}: {exc}"
            )
            if failures is None:
                raise RuntimeError(message) from exc
            failures.append(message)
            try:
                os.remove(output)
            except OSError:
                pass
            emit({
                "kind": "complete",
                "video": video_index,
                "videos_total": remote_video_total,
                "clip": clip_index,
                "clips_total": clips_total,
                "start": start,
                "end": end,
                "elapsed": 0.0,
                "name": get_source_display_name(source),
                "completed": current_completed,
                "total": state["total"],
                "failed": True,
            })
            return None
        with in_flight_lock:
            in_flight.pop(in_flight_key, None)
        with completed_lock:
            state["completed"] += 1
            current_completed = state["completed"]
        emit({
            "kind": "complete",
            "video": video_index,
            "videos_total": remote_video_total,
            "clip": clip_index,
            "clips_total": clips_total,
            "start": start,
            "end": end,
            "elapsed": 0.0,
            "name": get_source_display_name(source),
            "completed": current_completed,
            "total": state["total"],
        })
        return {
            'filename': str(fetched),
            'timestamps': [{'start': 0.0, 'end': duration, 'pred': pred}],
            'source_url': source.source_url,
            'source_metadata': {
                'platform': source.platform,
                'source_id': source.source_id,
                'display_name': source.display_name or source.source_url,
                'materialized_remote_segment': True,
            },
            'duration': duration,
        }

    tasks = []
    result_plan = []
    remote_video_index = 0
    for entry_index, entry in enumerate(entries):
        source = entry.get('filename')
        if not isinstance(source, MediaSource):
            if padding is None:
                local_item = entry
            else:
                local_item = dict(entry)
                # 本地 clip 的 padding 同样按相邻 clip 钳制，避免紧邻 clip 的
                # padding 重叠（与远程 clip 的 eff_start/eff_end 行为一致）。
                ts_list = list(entry.get('timestamps', []))
                clamped = []
                prev_eff_end = None
                for idx, ts in enumerate(ts_list):
                    start = float(ts['start'])
                    end = float(ts['end'])
                    next_start = (
                        float(ts_list[idx + 1]['start'])
                        if idx + 1 < len(ts_list) else None
                    )
                    eff_end = (end + after) if next_start is None else min(end + after, next_start)
                    eff_start = start - before
                    if prev_eff_end is not None:
                        eff_start = max(eff_start, prev_eff_end)
                    prev_eff_end = eff_end
                    clamped.append(dict(ts, start=eff_start, end=eff_end))
                local_item['timestamps'] = clamped
            # 本地 entry 直接参与后续 assemble；远程 entry 等下载完成后才放行。
            # 必须在这里按原始顺序记录占位，否则本地全被提前、远程全部殿后，
            # 交错排列（如 [url, local, url]）的列表顺序会被破坏。
            result_plan.append(("local", local_item))
            continue
        remote_video_index += 1
        identity = stable_source_id(source)
        if is_video and source.max_height is not None:
            apply_video_quality_limit(source, source.max_height)
        requested = selected_intervals.get(identity)
        timestamps = entry.get('timestamps', [])
        if requested is not None:
            requested = {(round(start, 3), round(end, 3)) for start, end in requested}
            timestamps = [ts for ts in timestamps
                          if (round(ts['start'], 3), round(ts['end'], 3)) in requested]
        clips_total = len(timestamps)
        n_ts = len(timestamps)
        prev_eff_end = None
        for interval_index, ts in enumerate(timestamps):
            start, end = float(ts['start']), float(ts['end'])
            # 计算带 padding 但不会与相邻 clip 重叠的有效区间：
            # 1) after 最多伸到下一个 clip 的起点；2) before 受上一个
            # 有效区间终点约束。reverify 的 original/new 跨组不桥接，
            # 若不加钳制，gap<before+after 时两段 padding 会重叠，
            # 拼接边界处开头内容会重复播放。
            next_start = (
                float(timestamps[interval_index + 1]['start'])
                if interval_index + 1 < n_ts else None
            )
            eff_end = (end + after) if next_start is None else min(end + after, next_start)
            eff_start = start - before
            if prev_eff_end is not None:
                eff_start = max(eff_start, prev_eff_end)
            prev_eff_end = eff_end
            duration = eff_end - eff_start
            extension = "mp4" if is_video else "m4a"
            output = os.path.join(
                str(temp_dir), f"remote_{entry_index}_{interval_index}_"
                f"{stable_source_id(source).replace(':', '_')}.{extension}")
            state["total"] += 1
            tasks.append((entry_index, interval_index, source, eff_start, eff_end,
                          output, interval_index + 1, clips_total,
                          remote_video_index, duration, ts.get('pred', 0)))
            result_plan.append(("task", (entry_index, interval_index)))

    if not tasks:
        return [item for kind, item in result_plan]

    results_by_task = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), max(max_parallel, 1))) as executor:
        futures = {
            executor.submit(fetch_task, task): (task[0], task[1])
            for task in tasks
        }
        for future in futures:
            key = futures[future]
            try:
                result = future.result()
            except Exception:
                result = None
            if result is not None:
                results_by_task[key] = result

    materialized = []
    for kind, payload in result_plan:
        if kind == "local":
            materialized.append(payload)
        else:
            result = results_by_task.get(payload)
            if result is not None:
                materialized.append(result)
    return materialized

def convert_seconds_to_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(round((seconds % 3600) % 60))
    if remaining_seconds == 60:
        minutes += 1
        remaining_seconds = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"

def _save_selected_txt(dict_list, txt_path):
    """保存审核后勾选的片段到 {原名}_selected.txt（与 timestamps.txt 同目录）。"""
    if not txt_path or txt_path == "No file selected!" or not dict_list:
        return
    try:
        selected_path = txt_path.rsplit('.', 1)[0] + '_selected.txt'
        with open(selected_path, 'w', encoding='utf-8') as f:
            for entry in dict_list:
                f.write(f"{get_source_persistence_name(entry['filename'])}\n")
                for ts in entry['timestamps']:
                    s = convert_seconds_to_timestamp(ts['start'])
                    e = convert_seconds_to_timestamp(ts['end'])
                    f.write(f"{s} - {e}, confidence: {ts['pred']}\n")
                f.write("\n")
        print(f"{Fore.GREEN}Saved selected clips to {selected_path}")
    except Exception as e:
        print(f"{Fore.YELLOW}Could not save _selected.txt: {e}")


def _parse_timestamps_txt(txt_path):
    """Parse timestamps.txt -> (with videos, without videos)"""
    with_videos = []
    without_videos = []
    current_file = None
    current_ts = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_file:
                    target = with_videos if current_ts else without_videos
                    target.append({'filename': current_file, 'timestamps': current_ts})
                current_file = None
                current_ts = []
                continue
            m = re.match(
                r'(\d+):(\d{2}):(\d{2})\s*-\s*(\d+):(\d{2}):(\d{2}),\s*confidence:\s*([\d.]+)',
                line)
            if m:
                h1, m1, s1, h2, m2, s2, conf = m.groups()
                s = int(h1) * 3600 + int(m1) * 60 + int(s1)
                e = int(h2) * 3600 + int(m2) * 60 + int(s2)
                current_ts.append({'start': s, 'end': e, 'pred': float(conf)})
            else:
                current_file = line
    if current_file:
        target = with_videos if current_ts else without_videos
        target.append({'filename': current_file, 'timestamps': current_ts})
    return with_videos, without_videos




def _scan_reverify_audio(raw, scan_windows, precision, focus_idx, threshold,
                         ort_session, verify_block_size, direct_accept,
                         logger=None, timestamp_offset=0):
    """Scan PCM samples and return accepted new timestamps."""
    import numpy as _np
    from sound_reader import compute_timestamps as _compute_ts

    sample_rate = 32000
    checked = dskip = confirmed = rejected = 0
    if logger:
        windows_iter = proglog.default_bar_logger(logger).iter_bar(block=scan_windows)
    else:
        windows_iter = scan_windows
    found = []
    for ws, we in windows_iter:
        checked += 1
        si = max(0, int(ws * sample_rate))
        ei = min(len(raw), int(we * sample_rate))
        if ei - si < sample_rate:
            continue
        slice_audio = raw[si:ei].astype(_np.float32) / 32767.0
        frame_count = sample_rate * verify_block_size
        if len(slice_audio) < frame_count:
            continue
        num_blocks = len(slice_audio) // frame_count
        blocks = slice_audio[:num_blocks * frame_count].reshape(num_blocks, frame_count)
        for b_idx in range(num_blocks):
            block = blocks[b_idx]
            rms = _np.sqrt(_np.mean(block ** 2))
            if rms > 0.005:
                block = block.copy() * min(2.5, 0.25 / rms)
            preds = ort_session.run(["output"], {
                "input": block.reshape(1, -1).astype(_np.float32)
            })[0]
            block_offset = b_idx * verify_block_size + timestamp_offset
            for ts in _compute_ts(preds[0], precision, threshold, focus_idx, block_offset):
                if ts['pred'] > direct_accept:
                    ts['source'] = 'new'
                    found.append(ts)
                    dskip += 1
                elif not ts['suspect']:
                    ts['source'] = 'new'
                    found.append(ts)
                    confirmed += 1
                else:
                    rejected += 1
    return found, checked, dskip, confirmed, rejected


def _merge_reverify_timestamps(timestamps):
    def merge_group(group):
        if not group:
            return []
        group.sort(key=lambda item: item['start'])
        merged = [group[0]]
        for ts in group[1:]:
            if ts['start'] <= merged[-1]['end'] + 2.0:
                merged[-1]['end'] = max(merged[-1]['end'], ts['end'])
                merged[-1]['pred'] = max(merged[-1]['pred'], ts['pred'])
            else:
                merged.append(ts)
        return merged

    originals = merge_group([t for t in timestamps if t.get('source') == 'original'])
    news = merge_group([t for t in timestamps if t.get('source') == 'new'])
    result = originals + news
    result.sort(key=lambda item: item['start'])
    # 反复去重直到没有相邻重叠。删除区间后必须重新检查前一个区间，
    # 否则一次循环会漏掉删除后新暴露出的重叠（例如长 clip 与内部残留区间）。
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result) - 1:
            if result[i]['end'] > result[i + 1]['start']:
                duration_i = result[i]['end'] - result[i]['start']
                duration_j = result[i + 1]['end'] - result[i + 1]['start']
                # 保留更长的区间；等长时保留前一个（与历史行为一致）
                if duration_i >= duration_j:
                    result.pop(i + 1)
                else:
                    result.pop(i)
                changed = True
                continue
            i += 1
    return [t for t in result if t['end'] - t['start'] > 0.5]


def _read_wav_pcm(path):
    import wave
    import numpy as _np

    with wave.open(str(path), 'rb') as audio:
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        if width != 2:
            raise ValueError(f"remote reverify requires 16-bit WAV, got {width * 8}-bit")
        samples = _np.frombuffer(audio.readframes(audio.getnframes()), dtype=_np.int16)
    if channels > 1:
        samples = samples[:len(samples) - (len(samples) % channels)]
        samples = samples.reshape(-1, channels).mean(axis=1).astype(_np.int16)
    return samples


def _verify_and_expand(dict_list, selected_model, window=5.0,
                       precision=100, block_size=600, logger=None,
                       focus_idx=58, threshold=0.30, ort_session=None,
                       verify_block_size=10, direct_accept=0.75,
                       use_gpu=True, cache_store=None, refresh_func=None,
                       audio_cache_paths=None, progress_callback=None):
    """Reverify local files and remote sources without downloading remote video."""
    if not dict_list:
        return dict_list
    ensure_temp_dir()

    if ort_session is None:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        try:
            ort_session = ort.InferenceSession(selected_model, sess_options, providers=providers)
        except Exception:
            ort_session = ort.InferenceSession(selected_model, sess_options, providers=['CPUExecutionProvider'])

    sample_rate = 32000
    checked = dskip = confirmed = rejected = 0
    for entry in dict_list:
        filename = entry['filename']
        cached_audio = None
        if isinstance(filename, MediaSource) and audio_cache_paths:
            cached_audio = audio_cache_paths.get(stable_source_id(filename))
            if cached_audio is not None and not Path(cached_audio).is_file():
                cached_audio = None
        scan_filename = cached_audio or filename
        original_ts = entry.get('timestamps', [])
        if not original_ts:
            continue
        for ts in original_ts:
            ts.setdefault('source', 'original')

        try:
            if isinstance(filename, MediaSource) and cached_audio is None:
                if not filename.audio_url:
                    raise ValueError("MediaSource has no audio_url")
                scan_windows = []
                for ts in original_ts:
                    ws = max(0.0, float(ts['start']) - window)
                    we = float(ts['end']) + window
                    if filename.duration is not None:
                        we = min(we, float(filename.duration))
                    if we <= ws:
                        continue
                    if scan_windows and ws <= scan_windows[-1][1] + 1:
                        scan_windows[-1] = (scan_windows[-1][0], max(scan_windows[-1][1], we))
                    else:
                        scan_windows.append((ws, we))

                new_timestamps = []
                window_total = len(scan_windows)
                for window_index, (ws, we) in enumerate(scan_windows):
                    print(
                        f"{Fore.CYAN}Verification: fetching audio "
                        f"[{window_index + 1}/{window_total}] "
                        f"({ws:g}-{we:g}s) for "
                        f"{get_source_display_name(filename)}")
                    if progress_callback is not None:
                        progress_callback({
                            "title": "Verification",
                            "current_line": (
                                f"Fetching verification audio "
                                f"[{window_index + 1}/{window_total}]"
                            ),
                            "percent_line": (
                                f"Progress: "
                                f"{int((window_index + 1) / window_total * 100)}%"
                            ),
                            "speed_line": "Speed: --",
                            "eta_line": "ETA: --",
                        })

                    def fetch_progress(current, total, elapsed):
                        if progress_callback is not None:
                            sample = format_fetch_progress(current, total, elapsed)
                            sample["title"] = (
                                f"Verification: audio "
                                f"[{window_index + 1}/{window_total}]"
                            )
                            progress_callback(sample)

                    temporary = tempfile.NamedTemporaryFile(
                        suffix='.wav', prefix='reverify-', dir=TEMP_DIR, delete=False)
                    temporary.close()
                    try:
                        audio_path = fetch_segment(
                            filename, ws, we, temporary.name,
                            cache_store=cache_store, audio_only=True, codec='pcm_s16le',
                            refresh_func=refresh_func, logger=print,
                            progress_callback=fetch_progress)
                        raw = _read_wav_pcm(audio_path)
                        duration = len(raw) / sample_rate
                        found, scans, skipped, accepted, refused = _scan_reverify_audio(
                            raw, [(0.0, duration)], precision, focus_idx, threshold,
                            ort_session, verify_block_size, direct_accept, logger)
                        for ts in found:
                            ts['start'] += ws
                            ts['end'] += ws
                        new_timestamps.extend(found)
                        checked += scans
                        dskip += skipped
                        confirmed += accepted
                        rejected += refused
                    finally:
                        try:
                            os.remove(temporary.name)
                        except OSError:
                            pass
                entry['timestamps'] = _merge_reverify_timestamps(original_ts + new_timestamps)
                continue

            full_audio = tempfile.NamedTemporaryFile(
                suffix='.pcm', prefix='reverify-', dir=TEMP_DIR, delete=False)
            full_audio.close()
            try:
                extract_cmd = [
                    os.environ.get('FFMPEG_BINARY', 'ffmpeg'), '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(scan_filename), '-vn', '-f', 's16le', '-acodec', 'pcm_s16le',
                    '-ar', str(sample_rate), '-ac', '1', full_audio.name]
                source_label = (
                    get_source_display_name(filename)
                    if isinstance(filename, MediaSource) else os.path.basename(str(filename))
                )
                print(f"{Fore.CYAN}Verification: extracting full audio for {source_label}...")
                if progress_callback is not None:
                    progress_callback({
                        "title": "Verification",
                        "current_line": f"Extracting full audio for {source_label}...",
                        "percent_line": "Progress: --",
                        "speed_line": "Speed: --",
                        "eta_line": "ETA: --",
                    })
                run_tracked(extract_cmd)
                if os.path.getsize(full_audio.name) == 0:
                    continue
                raw = __import__('numpy').memmap(full_audio.name, dtype='int16', mode='r')
                if len(raw) == 0:
                    continue
                max_dur = len(raw) / sample_rate
                scan_windows = []
                for ts in original_ts:
                    ws = max(0, ts['start'] - window)
                    we = min(ts['end'] + window, max_dur - 0.1)
                    if we <= ws:
                        continue
                    if scan_windows and ws <= scan_windows[-1][1] + 1:
                        scan_windows[-1] = (scan_windows[-1][0], max(scan_windows[-1][1], we))
                    else:
                        scan_windows.append((ws, we))
                new_timestamps = []
                for ws, we in scan_windows:
                    found, scans, skipped, accepted, refused = _scan_reverify_audio(
                        raw, [(ws, we)], precision, focus_idx, threshold,
                        ort_session, verify_block_size, direct_accept, logger,
                        timestamp_offset=ws)
                    new_timestamps.extend(found)
                    checked += scans
                    dskip += skipped
                    confirmed += accepted
                    rejected += refused
                entry['timestamps'] = _merge_reverify_timestamps(original_ts + new_timestamps)
                if cached_audio is not None:
                    preserved = preserve_remote_result(entry, filename)
                    entry.clear()
                    entry.update(preserved)
            finally:
                try:
                    del raw
                except (NameError, UnboundLocalError):
                    pass
                try:
                    os.remove(full_audio.name)
                except OSError:
                    pass
        except Exception as exc:
            if isinstance(filename, MediaSource):
                print(f"{Fore.YELLOW}  Verify scan failed for remote source "
                      f"{get_source_display_name(filename)}: {exc}")
            else:
                print(f"{Fore.YELLOW}  Verify scan failed for {os.path.basename(filename)}: {exc}")

    if checked > 0:
        print(f"{Fore.CYAN}Verification: scanned {checked} window(s), "
              f"confirmed {confirmed} new, DRC-skip {dskip}, rejected {rejected}.")
    return dict_list


def _smart_sort_key(filepath):
    """Sort by: folder → date/title → part → natural."""
    name = os.path.basename(filepath)
    folder = os.path.basename(os.path.dirname(filepath)).lower()
    name_no_ext = os.path.splitext(name)[0]

    # --- folder key: extract date+time for both folder names and loose filenames
    _h = lambda s: ('{:02d}'.format(int(m.group(1))) if (m := re.search(r'(\d{1,2})点场?', s)) else '\uffff')
    _src = folder if re.search(r'\d{4}年', folder) else name
    fm = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', _src)
    if fm:
        fkey = (0, f'{fm[1]}{int(fm[2]):02d}{int(fm[3]):02d}', _h(_src))
    else:
        fm = re.search(r'(\d{4})年(\d{1,2})月', _src)
        if fm:
            fkey = (0, f'{fm[1]}{int(fm[2]):02d}', _h(_src))
        else:
            fm = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', _src)
            if fm:
                fkey = (0, f'{fm[1]}{fm[2]}{fm[3]}', _h(_src))
            elif re.search(r'(\d{4})[-_](\d{2})', _src):
                fm = re.search(r'(\d{4})[-_](\d{2})', _src)
                fkey = (0, f'{fm[1]}{fm[2]}', _h(_src))
            else:
                fp = re.split(r'(\d+)', folder)
                fkey = (1, tuple(int(p) if p.isdigit() else p.lower() for p in fp), folder)

    # --- file key
    _of = lambda n: 1 if re.search(r'\bOriginal\b', n) else 0
    _pp = lambda n: int(re.search(r'^p?(\d{1,2})[\s_\-]', n).group(1)) if re.search(r'^p?(\d{1,2})[\s_\-]', n) else (int(re.search(r'part\s*(\d+)', n, re.I).group(1)) if re.search(r'part\s*(\d+)', n, re.I) else 0)
    # 1. Chinese live stream date
    m = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2})点场?', name)
    if m:
        return (fkey, 0, f'{m[1]}{int(m[2]):02d}{int(m[3]):02d}_{int(m[4]):02d}', _of(name), _pp(name))
    # 2. ISO date: "2022-07-19" or "2022_07_19"
    m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', name)
    if m:
        return (fkey, 0, f'{m[1]}{m[2]}{m[3]}', _of(name), _pp(name))
    # 3a. Part FIRST: "p0-title" or "0-title" (Bilibili)
    m = re.search(r'^p?(\d{1,2})[\s_\-]+(.+)$', name_no_ext, re.IGNORECASE)
    if m:
        return (fkey, 1, m[2].strip().lower(), int(m[1]))
    # 3b. Part LAST: "video_p1", "video part 2", "movie (3)"
    m = re.search(r'^(.*?)[\s_\-]+p(?:art[\s_]*)?(\d+)$', name_no_ext, re.IGNORECASE)
    if not m:
        m = re.search(r'^(.*?)\s*\((\d+)\)\s*$', name_no_ext)
    if m:
        base = re.sub(r'[\s_\-]+$', '', m[1].strip().lower())
        return (fkey, 1, base, int(m[2]))
    # 4. Natural sort fallback
    parts = re.split(r'(\d+)', name)
    return (fkey, 2, tuple(int(p) if p.isdigit() else p.lower() for p in parts))


def preview_playable_path(fetch_result, fallback_path):
    """Return the fetched preview path, or the generated fallback path."""
    if fetch_result is None:
        return str(fallback_path)
    if os.fspath(fetch_result) == os.fspath(fallback_path):
        return str(fallback_path)
    return str(fetch_result)


def _release_grab(window):
    """Safely release a Tk grab, tolerating already-destroyed windows."""
    try:
        if window is not None and window.winfo_exists():
            window.grab_release()
    except tk.TclError:
        pass


class ReviewDialog:
    """片段审核对话框 —— Treeview + 音频/视频预览 + 勾选/取消。"""

    def __init__(self, parent, dict_list, padding, output_path,
                 use_verify=False, txt_path=None, cache_store=None):
        self.parent = parent
        self.dict_list = dict_list
        self.padding = padding or (0, 0)
        self.output_path = output_path
        self.txt_path = txt_path
        self.cache_store = cache_store or CacheStore()
        self.result = None
        self.checks = []
        self._preview_paths = []

        self.flat = []
        for entry in dict_list:
            fn = entry['filename']
            for ts in entry.get('timestamps', []):
                self.flat.append({
                    'filename': fn,
                    'start': ts['start'], 'end': ts['end'],
                    'pred': ts.get('pred', 0),
                    'source': ts.get('source', 'original'),
                    'suspect': ts.get('suspect', False),
                })

        self.win = tk.Toplevel(parent)
        self.win.title("Review Clips")
        self.win.geometry("800x500")
        self.win.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.win.transient(parent)
        self.win.lift()
        self.win.focus_force()

        # 深色主题 Treeview 可读性修复
        style = ttk.Style()
        style.configure('Review.Treeview',
                        background='#2d2d2d',
                        foreground='#e0e0e0',
                        fieldbackground='#2d2d2d')
        style.map('Review.Treeview',
                  background=[('selected', '#444444')],
                  foreground=[('selected', '#ffffff')])

        info_frame = ttk.Frame(self.win)
        info_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        total = len(self.flat)
        new_count = sum(1 for f in self.flat if f.get('source') == 'new')
        suspect_count = sum(1 for f in self.flat if f.get('suspect'))
        t = f"Total: {total} segments"
        if use_verify and new_count > 0:
            t += f"  |  New: {new_count}"
        if suspect_count > 0:
            t += f"  |  Suspect: {suspect_count}"
        ttk.Label(info_frame, text=t).pack(side=tk.LEFT)
        ttk.Label(info_frame,
                  text="Right-click for preview  |  Click row to toggle",
                  foreground='gray').pack(side=tk.RIGHT)

        tree_frame = ttk.Frame(self.win)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(tree_frame,
                                  columns=('sel', 'time', 'file', 'conf', 'status'),
                                  show='headings', selectmode='extended')
        self.tree.heading('sel', text='\u2611')
        self.tree.heading('time', text='Time')
        self.tree.heading('file', text='Source File')
        self.tree.heading('conf', text='Confidence')
        self.tree.heading('status', text='Status')
        self.tree.column('sel', width=40, anchor='center')
        self.tree.column('time', width=140)
        self.tree.column('file', width=340)
        self.tree.column('conf', width=80)
        self.tree.column('status', width=120)

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for i, f in enumerate(self.flat):
            suspect = f.get('suspect', False)
            cv = tk.BooleanVar(value=not suspect)  # suspect 预取消勾选（软标记，可勾回）
            self.checks.append(cv)
            s_str = self._fmt(f['start'])
            e_str = self._fmt(f['end'])
            bn = get_source_display_name(f['filename'])
            if suspect:
                st = 'Suspect'
            else:
                st = 'New' if f.get('source') == 'new' else 'Original'
            tag = 'checked'
            self.tree.insert('', tk.END, iid=str(i),
                             values=('☑' if cv.get() else '☐',
                                     f"{s_str} - {e_str}", bn,
                                     f"{f['pred']:.2f}", st), tags=(tag,))
            self._style(str(i), cv.get())
            if suspect:
                self.tree.set(str(i), 0, '☐')
                self.tree.item(str(i), tags=('unchecked',))

        # Auto-deselect originals that overlap with new reverify clips
        for i, f in enumerate(self.flat):
            if f.get('source') != 'original':
                continue
            for j, g in enumerate(self.flat):
                if g.get('source') != 'new':
                    continue
                if f['filename'] != g['filename']:
                    continue
                # deselect original only if new fully covers it
                if g['start'] <= f['start'] and g['end'] >= f['end']:
                    self.checks[i].set(False)
                    self._style(str(i), False)
                    self.tree.set(str(i), 0, '\u2610')
                    self.tree.item(str(i), tags=('unchecked',))
                    break

        self.tree.tag_configure('checked', foreground='#4caf50')
        self.tree.tag_configure('unchecked', foreground='#888888')

        self.ctx_menu = tk.Menu(self.win, tearoff=0)
        self.ctx_menu.add_command(label="Play Audio",
                                   command=self._play_audio)
        self.ctx_menu.add_command(label="Open Video Clip",
                                   command=self._play_video)
        self.tree.bind("<Button-3>", self._on_right_click)
        self.tree.bind("<Button-1>", self._on_click)
        self.tree.bind("<Double-1>", self._on_double_click)

        bf = ttk.Frame(self.win)
        bf.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(bf, text="Select All", command=self._sel_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Deselect All", command=self._desel_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Invert", command=self._invert).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Compile", command=self._on_compile).pack(side=tk.RIGHT, padx=2)
        ttk.Button(bf, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=2)

        self.win.wait_window()

    def _fmt(self, sec):
        h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
        return f"{h}:{m:02d}:{s:02d}"

    def _style(self, iid, checked):
        self.tree.item(iid, tags=('checked',) if checked else ('unchecked',))

    def _on_click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid and iid.isdigit():
            col = self.tree.identify_column(event.x)
            if col == '#1':  # 'sel' column
                idx = int(iid)
                cv = self.checks[idx]
                cv.set(not cv.get())
                self._style(iid, cv.get())
                self.tree.set(iid, 'sel', '\u2611' if cv.get() else '\u2610')

    def _on_right_click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid and iid.isdigit():
            self.tree.selection_set(iid)
            self.ctx_menu.post(event.x_root, event.y_root)

    def _on_double_click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid and iid.isdigit():
            self._edit_times(int(iid))

    def _edit_times(self, idx):
        f = self.flat[idx]
        dlg = tk.Toplevel(self.win)
        dlg.title("Edit Times")
        dlg.transient(self.win)
        dlg.geometry("420x180")
        dlg.resizable(True, True)
        dlg.minsize(350, 160)

        ttk.Label(dlg, text="Start (HH:MM:SS or seconds):").pack(padx=10, pady=(10, 0))
        start_var = tk.StringVar(value=self._fmt(f['start']))
        ttk.Entry(dlg, textvariable=start_var, width=25).pack(padx=10, pady=2)

        ttk.Label(dlg, text="End (HH:MM:SS or seconds):").pack(padx=10, pady=(5, 0))
        end_var = tk.StringVar(value=self._fmt(f['end']))
        ttk.Entry(dlg, textvariable=end_var, width=25).pack(padx=10, pady=2)

        def _parse(s):
            s = s.strip()
            m = re.match(r'^(\d+):(\d{2}):(\d{2})$', s)
            if m:
                return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            try:
                return float(s)
            except ValueError:
                return None

        def _apply_real():
            ns = _parse(start_var.get())
            ne = _parse(end_var.get())
            if ns is None or ne is None or ns >= ne:
                messagebox.showwarning("Invalid", "Invalid time values (start must be < end).")
                return
            old_s, old_e = f['start'], f['end']
            f['start'] = ns
            f['end'] = ne
            s_str = self._fmt(ns)
            e_str = self._fmt(ne)
            self.tree.set(str(idx), 'time', f"{s_str} - {e_str}")
            for entry in self.dict_list:
                if entry['filename'] != f['filename']:
                    continue
                for ts in entry.get('timestamps', []):
                    if abs(ts['start'] - old_s) < 0.01 and abs(ts['end'] - old_e) < 0.01:
                        ts['start'] = ns
                        ts['end'] = ne
                        break
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(pady=10, fill=tk.X, padx=20)
        apply_btn = ttk.Button(bf, text="Apply", command=_apply_real, style='Accent.TButton')
        apply_btn.pack(side=tk.LEFT, padx=10, ipadx=20, ipady=4)
        cancel_btn = ttk.Button(bf, text="Cancel", command=dlg.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=10, ipadx=20, ipady=4)

    def _sel_idx(self):
        sel = self.tree.selection()
        if sel and sel[0].isdigit():
            return int(sel[0])
        return None

    def _play_audio(self):
        idx = self._sel_idx()
        if idx is not None:
            self._preview(idx, video=False)

    def _play_video(self):
        idx = self._sel_idx()
        if idx is not None:
            self._preview(idx, video=True)

    def _preview(self, idx, video):
        f = self.flat[idx]
        bf, af = self.padding
        ss = max(0, f['start'] - bf)
        dur = (f['end'] + af) - ss
        self._preview_paths = getattr(self, '_preview_paths', [])
        tmp_path = None
        try:
            suf = '.mp4' if video else '.wav'
            tmp = tempfile.NamedTemporaryFile(
                suffix=suf, prefix='preview-', dir=TEMP_DIR, delete=False)
            tmp.close()
            tmp_path = tmp.name
            if isinstance(f['filename'], MediaSource):
                cache_store = getattr(self, 'cache_store', None) or CacheStore()
                source = f['filename']
                source_resolved_at = getattr(source, "resolved_at", None)
                if (source_resolved_at is not None
                        and time.monotonic() - source_resolved_at > PREVIEW_REFRESH_THRESHOLD):
                    try:
                        updated = refresh_remote_source(
                            source,
                            browser_cookies=browser_cookie_setting_value(
                                self.remote_browser_cookies.get(),
                                cookies_file=self.remote_cookies_file.get(),
                            ),
                        )
                        if updated is not None and updated is not source:
                            source.__dict__.update(updated.__dict__)
                        print(f"Preview source refreshed ({source.display_name})")
                    except Exception as exc:
                        print(f"{Fore.YELLOW}Preview source refresh failed: {exc}")
                fetch_result = fetch_segment(
                    f['filename'], ss, ss + dur, tmp.name,
                    cache_store=cache_store, allow_covering_cache=False,
                    audio_only=not video,
                    codec='pcm_s16le' if not video else None,
                    refresh_func=lambda source: refresh_remote_source(
                        source,
                        browser_cookies=browser_cookie_setting_value(
                            self.remote_browser_cookies.get(),
                            cookies_file=self.remote_cookies_file.get(),
                        ),
                    ),
                    logger=print)
                playable = preview_playable_path(fetch_result, tmp.name)
                if sys.platform == 'win32':
                    os.startfile(playable)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', playable])
                else:
                    subprocess.Popen(['xdg-open', playable])
                if playable == tmp.name:
                    getattr(self, '_preview_paths', []).append(tmp.name)
                else:
                    Path(tmp.name).unlink(missing_ok=True)
                return
            ff = os.environ.get('FFMPEG_BINARY', 'ffmpeg')
            if video:
                cmd = ([ff, '-y', '-hide_banner', '-loglevel', 'error',
                        '-ss', str(ss), '-t', str(dur), '-i', f['filename']]
                       + get_video_codec() +
                       ['-c:a', 'aac', '-b:a', '128k', tmp.name])
            else:
                cmd = [ff, '-y', '-hide_banner', '-loglevel', 'error',
                       '-ss', str(ss), '-t', str(dur), '-i', f['filename'],
                       '-vn', '-acodec', 'pcm_s16le', '-ar', '32000', '-ac', '1',
                       tmp.name]
            run_tracked(cmd, timeout=30)
            if sys.platform == 'win32':
                os.startfile(tmp.name)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', tmp.name])
            else:
                subprocess.Popen(['xdg-open', tmp.name])
            getattr(self, '_preview_paths', []).append(tmp.name)
        except Exception as e:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            print(f"{Fore.YELLOW}Preview failed: {e}")

    def cleanup_preview_files(self):
        for path in getattr(self, '_preview_paths', []):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        self._preview_paths = []

    def _sel_all(self):
        for i, cv in enumerate(self.checks):
            cv.set(True)
            self._style(str(i), True)
            self.tree.set(str(i), 'sel', '\u2611')

    def _desel_all(self):
        for i, cv in enumerate(self.checks):
            cv.set(False)
            self._style(str(i), False)
            self.tree.set(str(i), 'sel', '\u2610')

    def _invert(self):
        for i, cv in enumerate(self.checks):
            nv = not cv.get()
            cv.set(nv)
            self._style(str(i), nv)
            self.tree.set(str(i), 'sel', '\u2611' if nv else '\u2610')

    def _on_compile(self):
        result = []
        for entry in self.dict_list:
            fn = entry['filename']
            kept = []
            for ts in entry.get('timestamps', []):
                for i, f in enumerate(self.flat):
                    if (f['filename'] == fn and
                        abs(f['start'] - ts['start']) < 0.01 and
                        abs(f['end'] - ts['end']) < 0.01 and
                        self.checks[i].get()):
                        kept.append({'start': ts['start'], 'end': ts['end'],
                                     'pred': ts.get('pred', 0)})
                        break
            if kept:
                result.append({'filename': fn, 'timestamps': kept})
                if isinstance(fn, MediaSource):
                    result[-1]['source_url'] = fn.source_url
                    result[-1]['source_metadata'] = {
                        'platform': fn.platform,
                        'source_id': fn.source_id,
                        'display_name': get_source_display_name(fn),
                    }
        self.result = result

        # 保存勾选的 timestamps 到 _selected.txt
        if self.txt_path and self.txt_path != "No file selected!":
            selected_path = self.txt_path.rsplit('.txt', 1)[0] + '_selected.txt'
            lines = []
            for entry in result:
                lines.append(get_source_persistence_name(entry['filename']))
                for ts in entry['timestamps']:
                    s = ts['start']
                    e = ts['end']
                    start_str = self._fmt(s)
                    end_str = self._fmt(e)
                    lines.append(f"{start_str} - {end_str}, confidence: {ts.get('pred', 0):.2f}")
                lines.append('')
            if lines:
                with open(selected_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print(f"{Fore.GREEN}Saved selected timestamps to {selected_path}")

        self.cleanup_preview_files()
        self.win.destroy()

    def _on_cancel(self):
        self.result = None
        self.cleanup_preview_files()
        self.win.destroy()


class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Autocomper')

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 线程安全 UI 更新通道：后台线程只往队列投递，主线程单点轮询执行，
        # 避免并发 root.after 直接调用干扰 Tcl 事件循环导致窗口未响应。
        self._ui_queue = queue.Queue()
        self.root.after(50, self._poll_ui)

        # Set initial window size
        self.root.geometry('1150x800')

        # Enforce minimum window size
        self.root.resizable(True, True)
        
        self.root.wm_minsize(1050, 760)

        # Create a grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=0, minsize=400)

        # Left Column Widgets
        self.left_frame = ttk.Frame(root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=30, sticky="nsew")

        # Create a vertical separator
        separator = ttk.Separator(root, orient='vertical')
        separator.grid(row=0, column=1, sticky='ns')

        self.models_dir = get_bundle_filepath("models")
        self.preferences_file = 'preferences.ini'

        self.preferences = configparser.ConfigParser()
        try:
            self.preferences.read(self.preferences_file)
        except configparser.Error:
            messagebox.showwarning("Error", "Failed to load preferences.")

        if 'Settings' not in self.preferences:
            self.preferences['Settings'] = DEFAULT_SETTINGS
            with open(self.preferences_file, 'w') as configfile:
                self.preferences.write(configfile)
        else:
            for key, value in DEFAULT_SETTINGS.items():
                if key not in self.preferences['Settings']:
                    self.preferences.set('Settings', key, value)
                    with open(self.preferences_file, 'w') as configfile:
                        self.preferences.write(configfile)

        self.precision = tk.IntVar(value=100)
        self.block_size = tk.IntVar(value=600)
        self.threshold = tk.DoubleVar(value=0.90)
        self.focus_idx = tk.IntVar(value=58)
        self.model = tk.StringVar(value="bdetectionmodel_05_01_23.onnx")
        self.merge_clips = tk.BooleanVar(value=True)
        self.combine_vids = tk.BooleanVar(value=True)
        self.normalize_audio = tk.BooleanVar()
        self.remote_mode = tk.StringVar(value="Remote Stream")
        self.remote_browser_cookies = tk.StringVar(value="Auto")
        self.remote_cookies_file = tk.StringVar()
        configured_cache_path = self.preferences.get(
            "Settings", "remote_cache_path", fallback=str(CacheStore().root))
        self.remote_cache_store = CacheStore(configured_cache_path)
        try:
            self.remote_cache_store.ensure_ready()
        except OSError as exc:
            messagebox.showwarning(
                "Remote Cache",
                f"Saved remote cache location is unavailable:\n{exc}\n\nUsing the default cache location.")
            self.remote_cache_store = CacheStore()
            self.remote_cache_store.ensure_ready()
        self.remote_cache_path = tk.StringVar(value=str(self.remote_cache_store.root))
        self.remote_cache_size = tk.StringVar()

        self.keep_downloaded_vids = tk.BooleanVar(value=False)
        self.download_video_path = tk.StringVar()
        self.max_quality = tk.StringVar()
        self.max_download_speed = tk.IntVar()
        
        self.output_text_path = tk.StringVar()

        try:
            _kdv = self.preferences.getboolean("Settings", "keep_downloaded_vids")
        except (configparser.Error, ValueError):
            _kdv = False
        self.keep_downloaded_vids.set(_kdv)

        self.download_video_path.set(
            self.preferences.get("Settings", "download_path"))

        self.max_quality.set(
            self.preferences.get("Settings", "max_quality"))
        
        try:
            self.max_download_speed.set(int(
                self.preferences.get("Settings", "max_download_speed")))
        except (configparser.Error, ValueError, tk.TclError):
            self.max_download_speed.set(0)

        self.output_text_path.set(
            self.preferences.get("Settings", "output_text_path"))

        # Create a list to store uploaded video file paths
        self.uploaded_videos = []
        self.sort_ascending = False  # first click → ascending

        self.filelist_frame = ttk.Frame(self.left_frame)

        self.media_toggle_frame = ttk.Frame(self.filelist_frame)

        self.is_video = True

        def check_number(char):
            return char.isdigit() or char == ""
        
        def check_decimal(char):
            if char == "":
                return True
            try:
                float(char)
                return True
            except ValueError:
                return False
        
        self.num_check = (self.root.register(check_number), '%P')
        self.decimal_check = (self.root.register(check_decimal), '%P')

        def toggle_media():
            if self.is_video:
                self.toggle_button.config(text='Audio')
            else:
                self.toggle_button.config(text='Video')
            self.is_video = not self.is_video
            self.uploaded_videos.clear()
            self.update_listbox()
            self.clear_output()
            self.populate_add_button()

        # Settings Button
        settings_photo = get_photo_icon(os.path.join("img", "settings.png"))

        self.settings_button = ttk.Button(
            self.media_toggle_frame, image=settings_photo, width=5, padding=0, command=self.open_settings_modal)
        self.settings_button.image = settings_photo

        self.settings_button.pack(side=tk.LEFT, anchor=tk.NW)

        ttk.Label(self.media_toggle_frame, text="Input Media Type:",
                  font=(None, 12, "bold")).pack()

        self.toggle_button = ttk.Button(
            self.media_toggle_frame, text="Video", width=20, command=toggle_media)
        self.toggle_button.pack(pady=10)

        self.media_toggle_frame.pack(fill=tk.BOTH)

        self.video_listbox = ttk.Treeview(
            self.filelist_frame, selectmode=tk.EXTENDED, columns="#1", show='')
        self.video_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._listbox_font = tkfont.nametofont("TkDefaultFont")
        self._listbox_resize_after = None
        self.video_listbox.bind('<Configure>', self._on_listbox_resize)

        scrollbar = ttk.Scrollbar(self.filelist_frame, orient="vertical")
        scrollbar.config(command=self.video_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.video_listbox.config(yscrollcommand=scrollbar.set)

        self.filelist_frame.pack(fill=tk.BOTH)

        self.filelist_buttons_frame = ttk.Frame(self.left_frame)

        # Create buttons for adding and removing videos
        # self.add_button = ttk.Button(
        #     self.filelist_buttons_frame, text="Add Media", command=self.add_video)

        self.add_button = ttk.Button(
            self.filelist_buttons_frame, text="Add Media", style="TButton")

        self.media_menu = tk.Menu(
            root, tearoff=0, font=(None, 11, "bold"),
            bg="#333333", fg="#ffffff", activebackground="#555555", activeforeground="#ffffff"
        )

        self.media_menu.add_command(
            label=" Add Video Files ", command=self.add_video)
        self.media_menu.add_command(
            label=" Add URL ", command=self.add_video_url)
        self.media_menu.add_command(
            label=" Add Folder ", command=self.add_video_folder)

        def show_menu(event):
            x = self.add_button.winfo_rootx()
            y = self.add_button.winfo_rooty() + self.add_button.winfo_height()
            # menu.post(event.x_root, event.y_root)
            self.media_menu.post(x, y)

        self.add_button.bind("<Button-1>", show_menu)

        self.up_arrow = ttk.Button(
            self.filelist_buttons_frame, text="↑", width=3, command=self.move_selected_up)
        self.down_arrow = ttk.Button(
            self.filelist_buttons_frame, text="↓", width=3, command=self.move_selected_down)
        self.sort_button = ttk.Button(
            self.filelist_buttons_frame, text="⇅", width=3, command=self.sort_filelist)

        self.remove_button = ttk.Button(
            self.filelist_buttons_frame, text="Remove Selected", command=self.remove_selected)
        self.clear_button = ttk.Button(
            self.filelist_buttons_frame, text="Clear All", command=self.clear_list)

        self.add_button.pack(pady=5, padx=1, side=tk.LEFT)
        self.up_arrow.pack(pady=5, padx=3, side=tk.LEFT)
        self.down_arrow.pack(pady=5, padx=3, side=tk.LEFT)
        self.sort_button.pack(pady=5, padx=3, side=tk.LEFT)

        # Drag and drop
        root.drop_target_register(DND_FILES)
        root.dnd_bind('<<Drop>>', self._on_drop)

        self.clear_button.pack(pady=5, side=tk.RIGHT)
        self.remove_button.pack(pady=5, padx=5, side=tk.RIGHT)

        self.filelist_buttons_frame.pack(after=self.filelist_frame, fill=tk.X)

        ttk.Separator(self.left_frame, orient="horizontal").pack(
            fill=tk.X, pady=15)

        self.options_viewport = ttk.Frame(self.left_frame)
        self.options_viewport.pack(fill=tk.BOTH, expand=True)
        self.options_canvas = tk.Canvas(
            self.options_viewport, highlightthickness=0, borderwidth=0)
        self.options_scrollbar = ttk.Scrollbar(
            self.options_viewport, orient=tk.VERTICAL,
            command=self.options_canvas.yview)
        self.options_canvas.configure(
            yscrollcommand=self.options_scrollbar.set)
        self.options_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.options_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.options_row = ttk.Frame(self.options_canvas)
        self.options_canvas_window = self.options_canvas.create_window(
            (0, 0), window=self.options_row, anchor="nw")

        def update_options_scrollregion(event=None):
            self.options_canvas.configure(
                scrollregion=self.options_canvas.bbox("all"))

        def resize_options_window(event):
            self.options_canvas.itemconfigure(
                self.options_canvas_window, width=event.width)
            update_options_scrollregion()

        self.options_row.bind("<Configure>", update_options_scrollregion)
        self.options_canvas.bind("<Configure>", resize_options_window)
        self.options_row.grid_columnconfigure(0, weight=1)
        self.options_row.grid_columnconfigure(1, weight=1)
        self.options_row.grid_columnconfigure(2, weight=1)
        self.options_row.grid_rowconfigure(0, weight=1)

        self.remote_settings_frame = ttk.LabelFrame(self.options_row, text="Remote Settings")
        self.remote_settings_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(10, 8))

        ttk.Label(self.remote_settings_frame, text="Remote Processing:").pack()
        self.remote_mode_dropdown = ttk.Combobox(
            self.remote_settings_frame, textvariable=self.remote_mode,
            values=["Remote Stream", "Audio Cache", "Full Download"],
            state="readonly", width=27)
        self.remote_mode_dropdown.pack(pady=(0, 8))

        ttk.Label(self.remote_settings_frame, text="Remote Browser Cookies:").pack()
        self.remote_browser_cookies_dropdown = ttk.Combobox(
            self.remote_settings_frame,
            textvariable=self.remote_browser_cookies,
            values=list(REMOTE_BROWSER_COOKIES),
            state="readonly",
            width=27,
        )
        self.remote_browser_cookies_dropdown.pack(pady=(0, 8))
        self.remote_browser_cookies_dropdown.bind(
            "<<ComboboxSelected>>", self._on_cookie_selection
        )
        self.remote_mode_tooltip = CustomHovertip(
            self.remote_mode_dropdown, REMOTE_MODE_TOOLTIP_TEXT
        )
        self.remote_browser_cookies_tooltip = CustomHovertip(
            self.remote_browser_cookies_dropdown, REMOTE_BROWSER_COOKIES_TOOLTIP_TEXT
        )

        ttk.Label(self.remote_settings_frame, text="Max Download Concurrency:").pack()
        self.remote_download_concurrency = tk.IntVar(value=REMOTE_CONCURRENCY_DEFAULT)
        self.remote_concurrency_spinbox = ttk.Spinbox(
            self.remote_settings_frame,
            from_=REMOTE_CONCURRENCY_MIN, to=REMOTE_CONCURRENCY_MAX, width=5,
            textvariable=self.remote_download_concurrency,
        )
        self.remote_concurrency_spinbox.pack(pady=(0, 8))
        self.remote_concurrency_tooltip = CustomHovertip(
            self.remote_concurrency_spinbox, REMOTE_CONCURRENCY_TOOLTIP_TEXT
        )

        self.remote_cache_frame = ttk.LabelFrame(self.remote_settings_frame, text="Remote Cache")
        self.remote_cache_frame.pack(fill=tk.X, padx=4, pady=(0, 8))
        ttk.Label(self.remote_cache_frame, text="Cache Location:").pack(anchor="w", padx=6, pady=(5, 0))
        self.remote_cache_path_entry = ttk.Entry(
            self.remote_cache_frame, textvariable=self.remote_cache_path,
            state="readonly", width=32)
        self.remote_cache_path_entry.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Label(
            self.remote_cache_frame, textvariable=self.remote_cache_size).pack(
                anchor="w", padx=6, pady=(0, 4))
        remote_cache_buttons = ttk.Frame(self.remote_cache_frame)
        remote_cache_buttons.pack(fill=tk.X, pady=(0, 6))
        remote_cache_first_row = ttk.Frame(remote_cache_buttons)
        remote_cache_first_row.pack()
        self.remote_cache_choose_button = ttk.Button(
            remote_cache_first_row, text="Choose Cache Folder", command=self.choose_remote_cache)
        self.remote_cache_choose_button.pack(side=tk.LEFT, padx=2)
        self.remote_cache_open_button = ttk.Button(
            remote_cache_first_row, text="Open Cache Folder", command=self.open_remote_cache)
        self.remote_cache_open_button.pack(side=tk.LEFT, padx=2)
        remote_cache_import_row = ttk.Frame(remote_cache_buttons)
        remote_cache_import_row.pack()
        self.import_external_audio_button = ttk.Button(
            remote_cache_import_row, text="Import External Audio",
            command=self.import_external_audio)
        self.import_external_audio_button.pack(padx=2)
        self.external_audio_tooltip = CustomHovertip(
            self.import_external_audio_button, EXTERNAL_AUDIO_TOOLTIP_TEXT)
        remote_cache_clear_row = ttk.Frame(remote_cache_buttons)
        remote_cache_clear_row.pack()
        self.remote_cache_clear_button = ttk.Button(
            remote_cache_clear_row, text="Clear Cache", command=self.clear_remote_cache)
        self.remote_cache_clear_button.pack(padx=2)
        self.remote_cache_tooltip = CustomHovertip(
            self.remote_cache_frame, REMOTE_CACHE_TOOLTIP_TEXT)
        self.refresh_remote_cache_size()

        self.text_options_frame = ttk.Frame(self.options_row)
        self.text_options_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=(10, 8))

        ttk.Label(self.text_options_frame, text="Model Options:",
                  font=(None, 11, "bold")).pack(pady=(10, 10))

        # Model Dropdown
        # First, get list of available models
        models = os.listdir(self.models_dir)

        # Filter out directories, keep only onnx files
        models = [item for item in models if os.path.isfile(
            os.path.join(self.models_dir, item))]

        models = [item for item in models if item.endswith('.onnx')]

        if len(models) == 0:
            raise Exception(f"No models found in directory {self.models_dir}")

        ttk.Label(self.text_options_frame, text="Model:", font=(
            None, 10, "bold")).pack(pady=(0, 1))

        self.model_dropdown = ttk.Combobox(
            self.text_options_frame, values=models, textvariable=self.model, state="readonly", width=30)

        self.model_dropdown.current(0)  # default dropdown option

        self.model_dropdown.pack()

        # Focus Index
        ttk.Label(self.text_options_frame, text="Focus Index:",
                  font=(None, 10, "bold")).pack(pady=(10, 1))
        self.focus_idx_entry = ttk.Entry(self.text_options_frame, textvariable=self.focus_idx, width=5)
        self.focus_idx_entry.pack()
        def _idx_tooltip():
            CustomHovertip(self.focus_idx_entry, "58=burp, 60=fart. Don't change unless custom model!")
        self.focus_idx_entry.after(200, _idx_tooltip)

        # Precision Entry
        ttk.Label(self.text_options_frame, text="Precision:",
                  font=(None, 10, "bold")).pack(pady=(10, 1))
        self.precision_entry = ttk.Entry(
            self.text_options_frame, textvariable=self.precision, validate='key', validatecommand=self.num_check)
        self.precision_entry.pack()

        # Block Size Entry
        ttk.Label(self.text_options_frame, text="Block Size (CAUTION):", font=(
            None, 10, "bold")).pack(pady=(10, 1))
        self.block_size_entry = ttk.Entry(
            self.text_options_frame, textvariable=self.block_size, validate='key', validatecommand=self.num_check)
        self.block_size_entry.pack()

        # Threshold Entry
        ttk.Label(self.text_options_frame, text="Threshold:",
                  font=(None, 10, "bold")).pack(pady=(10, 1))
        self.threshold_entry = ttk.Entry(
            self.text_options_frame, textvariable=self.threshold, validate='key', validatecommand=self.decimal_check)
        self.threshold_entry.pack()

        # preset save/load
        self.preset_frame = ttk.Frame(self.text_options_frame)
        self.preset_frame.pack(pady=(10, 0))
        self.save_preset_btn = ttk.Button(self.preset_frame, text="Save Preset",
                                           command=self.save_preset)
        self.save_preset_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.preset_combo_var = tk.StringVar(value="Load Preset")
        self.preset_combo = ttk.Combobox(self.preset_frame, textvariable=self.preset_combo_var,
                                          values=[], state="readonly", width=12)
        self.preset_combo.pack(side=tk.LEFT)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self.video_options_frame = ttk.Frame(self.options_row)
        self.video_options_frame.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=(10, 8))

        self.checkbox_frame = ttk.Frame(self.video_options_frame)
        self.checkbox_frame.pack(anchor=tk.W)

        ttk.Label(self.checkbox_frame, text="Video/Audio Options:",
                  font=(None, 11, "bold")).pack(pady=(10, 10), padx=0)

        # Merge Clips Checkbox
        self.merge_clips_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Merge Nearby Clips", variable=self.merge_clips)
        self.merge_clips_checkbox.pack(anchor=tk.W)

        # Merge Clips Checkbox
        self.combine_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Combine Input Media", variable=self.combine_vids, command=self.clear_output)
        self.combine_checkbox.pack(anchor=tk.W)

        # Normalize audio checkbox
        self.normalize_audio_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Normalize Audio", variable=self.normalize_audio)
        self.normalize_audio_checkbox.pack(anchor=tk.W)

        # Save timestamps to file checkbox
        self.save_txt = tk.BooleanVar()

        self.txt_output_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Save Timestamps to File", variable=self.save_txt)
        self.txt_output_checkbox.pack(anchor=tk.W)

        self.skip_detection_auto = tk.BooleanVar(value=True)
        self.skip_auto_checkbox = ttk.Checkbutton(
            self.checkbox_frame, text="Auto-use existing timestamps (skip prompt)",
            variable=self.skip_detection_auto)
        self.skip_auto_checkbox.pack(anchor=tk.W)

        # Create a Checkbutton for custom resolution
        self.use_custom_resolution = tk.BooleanVar()

        self.use_custom_padding = tk.BooleanVar()

        self.custom_resolution_width_var = tk.IntVar()
        self.custom_resolution_width_var.set(1920)

        self.custom_resolution_height_var = tk.IntVar()
        self.custom_resolution_height_var.set(1080)

        self.custom_padding_before = tk.IntVar()
        self.custom_padding_before.set(0)

        self.custom_padding_after = tk.IntVar()
        self.custom_padding_after.set(0)

        self.checkbox_frame_three = ttk.Frame(self.video_options_frame)
        self.checkbox_frame_three.pack(anchor=tk.W)
        self.custom_resolution_checkbox = ttk.Checkbutton(
            self.checkbox_frame_three, text="Use Custom Output Resolution", variable=self.use_custom_resolution, command=self.toggle_text_boxes)
        self.custom_resolution_checkbox.pack(anchor=tk.W)

        self.container_frame = ttk.Frame(self.checkbox_frame_three)

        # Create text input boxes (initially hidden)
        self.res_width_label = ttk.Label(self.container_frame, text="Width:")
        self.res_width_entry = ttk.Entry(
            self.container_frame, textvariable=self.custom_resolution_width_var, width=5, validate='key', validatecommand=self.num_check)

        self.res_height_label = ttk.Label(self.container_frame, text="Height:")
        self.res_height_entry = ttk.Entry(
            self.container_frame, textvariable=self.custom_resolution_height_var, width=5, validate='key', validatecommand=self.num_check)

        self.res_width_label.pack(side=tk.LEFT)
        self.res_width_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.res_height_label.pack(side=tk.LEFT)
        self.res_height_entry.pack(side=tk.LEFT)

        self.checkbox_frame_four = ttk.Frame(self.video_options_frame)
        self.checkbox_frame_four.pack(anchor=tk.W)
        self.use_clip_padding_checkbox = ttk.Checkbutton(
            self.checkbox_frame_four, text="Add Padding Time (Seconds)", variable=self.use_custom_padding, command=self.toggle_padding_text_boxes)
        self.use_clip_padding_checkbox.pack(anchor=tk.W)

        self.padding_container_frame = ttk.Frame(self.checkbox_frame_four)

        # Create text input boxes (initially hidden)
        self.padding_before_label = ttk.Label(
            self.padding_container_frame, text="Before:")
        self.padding_before_entry = ttk.Entry(
            self.padding_container_frame, textvariable=self.custom_padding_before, width=5, validate='key', validatecommand=self.decimal_check)

        self.padding_after_label = ttk.Label(
            self.padding_container_frame, text="After:")
        self.padding_after_entry = ttk.Entry(
            self.padding_container_frame, textvariable=self.custom_padding_after, width=5, validate='key', validatecommand=self.decimal_check)

        self.padding_before_label.pack(side=tk.LEFT)
        self.padding_before_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.padding_after_label.pack(side=tk.LEFT)
        self.padding_after_entry.pack(side=tk.LEFT)

        # 二次验证 checkbox
        self.checkbox_frame_five = ttk.Frame(self.video_options_frame)
        self.checkbox_frame_five.pack(anchor=tk.W)
        self.use_verify = tk.BooleanVar()
        self.verify_window_var = tk.DoubleVar(value=30.0)
        self.verify_checkbox = ttk.Checkbutton(
            self.checkbox_frame_five, text="Re-verify clips (scan near segments for missed clips)",
            variable=self.use_verify)
        self.verify_checkbox.pack(anchor=tk.W)

        # 审核 checkbox
        self.checkbox_frame_six = ttk.Frame(self.video_options_frame)
        self.checkbox_frame_six.pack(anchor=tk.W)
        self.use_review = tk.BooleanVar()
        self.review_checkbox = ttk.Checkbutton(
            self.checkbox_frame_six, text="Review clips before compile",
            variable=self.use_review)
        self.review_checkbox.pack(anchor=tk.W)

        # 严格假阳过滤 checkbox（argmax 门，默认关）
        self.checkbox_frame_seven = ttk.Frame(self.video_options_frame)
        self.checkbox_frame_seven.pack(anchor=tk.W)
        self.use_strict_fp = tk.BooleanVar()
        self.strict_fp_checkbox = ttk.Checkbutton(
            self.checkbox_frame_seven,
            text="Strict FP filter (drop clips where burp is not the top class)",
            variable=self.use_strict_fp)
        self.strict_fp_checkbox.pack(anchor=tk.W)

        # Right Column Widgets
        right_frame = ttk.Frame(root, width=400)
        right_frame.grid(row=0, column=2, padx=10, pady=30, sticky="nsew")
        right_frame.grid_propagate(False)
        right_frame.pack_propagate(False)

        self.output_video_path = tk.StringVar()
        self.output_video_path.set("No location selected!")

        # Output Location Selector
        ttk.Label(right_frame, text="Output Location:",
                  font=(None, 12, "bold")).pack()
        self.output_location_label = ttk.Label(
            right_frame, textvariable=self.output_video_path)
        self.output_location_label.pack(pady=10)
        self.output_location_button = ttk.Button(
            right_frame, text="Select Output File", width=34, command=self.select_output_location)
        self.output_location_button.pack(pady=(15, 2.5))

        # CPU/GPU toggle (prominent placement for testers)
        self.use_gpu = tk.BooleanVar(value=True)
        self.use_gpu_checkbox = ttk.Checkbutton(
            right_frame, text="Use GPU (CUDA)",
            variable=self.use_gpu)
        self.use_gpu_checkbox.pack()

        self.process_cancel_frame = ttk.Frame(right_frame)
        self.process_cancel_frame.pack()

        # Process Video Button
        self.process_button = ttk.Button(
            self.process_cancel_frame, text='Process Videos', width=30, padding=4.5, command=self.process_videos_multi)
        self.process_button.grid(row=0, column=0, pady=(5, 20), padx=(0, 2.5))

        # Cancel Button
        stop_photo = get_photo_icon(os.path.join("img", "stop.png"))

        self.cancel_button = ttk.Button(
            self.process_cancel_frame, image=stop_photo, width=5, padding=0, command=self.confirm_stop_process)
        self.cancel_button.image = stop_photo

        self.cancel_button.grid(row=0, column=1, pady=(5, 20), padx=(2.5, 0))

        self.cancel_button["state"] = tk.DISABLED

        # Progress bar for final render and remote transfers
        self.ui_bar = ttk.Progressbar(right_frame, orient='horizontal')
        self.ui_bar.pack(fill=tk.X, padx=10, pady=(4, 4))

        self.remote_progress_frame = ttk.Frame(right_frame, height=120)
        self.remote_progress_frame.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.remote_progress_frame.pack_propagate(False)
        self.remote_progress_text = tk.StringVar(value="")
        self.remote_progress_label = ttk.Label(
            self.remote_progress_frame, textvariable=self.remote_progress_text,
            anchor="nw", justify="left")
        self.remote_progress_label.pack(fill=tk.BOTH, expand=True)
        self.transfer_progress = ProgressWidgetAdapter(
            self.root, self.ui_bar, self.remote_progress_text,
            ui_queue=self._ui_queue)

        self.final_bar = FinalRenderBar(
            ui=self.ui_bar, progress_callback=self._queue_transfer_progress)

        self.stdout_frame = ttk.Frame(right_frame, width=200, height=100)

        # Text widget to display stdout
        self.stdout_text = tk.Text(
            self.stdout_frame, wrap="word", relief=tk.FLAT, fg="white")
        self.stdout_text.grid(row=0, column=0, sticky="nsew")

        text_scrollbar = ttk.Scrollbar(self.stdout_frame, orient="vertical")
        text_scrollbar.config(command=self.stdout_text.yview)
        text_scrollbar.grid(row=0, column=1, sticky="ns")

        self.stdout_text.config(yscrollcommand=text_scrollbar.set)

        # Configure grid weights to make the text widget expand
        self.stdout_frame.grid_rowconfigure(0, weight=1)
        self.stdout_frame.grid_columnconfigure(0, weight=1)

        self.stdout_frame.pack(fill=tk.BOTH, expand=True)
        self.stdout_frame.pack_propagate(False)

        # Redirect stdout to the Text widget
        sys.stdout = StdoutRedirector(self.stdout_text, self.root)

        self.active_thread = None

        self.dont_show_again_var = tk.BooleanVar(value=False)

        # Tooltips galore
        prec_tooltip = CustomHovertip(
            self.precision_entry, 'Precision (in ms) of the timestamp selection process (higher is less precise)')
        block_tooltip = CustomHovertip(
            self.block_size_entry, 'Amount of seconds (of samples) to process at once.\nLarger sizes offer better performance, but will consume significantly more memory.\nWARNING: Setting this too high for very long videos will use up a LOT of memory; only turn this up if you know your computer can handle it.')
        thres_tooltip = CustomHovertip(
            self.threshold_entry, 'The confidence threshold for a sound to be reported from 0-1.')
        merge_tooltip = CustomHovertip(
            self.merge_clips_checkbox, 'If timestamps are close together, combine them into one longer clip')
        comb_tooltip = CustomHovertip(
            self.combine_checkbox, 'Combine everything into one output video.\nIf unchecked, you will instead select a directory, and output\nvideos will be saved as (original_title)_comped inside the directory.')
        res_tooltip = CustomHovertip(self.custom_resolution_checkbox,
                                     '(BUGGY) Sets the resolution of the output video(s).\nMost useful when combining videos\nof different resolutions. Only applicable if the input media is video.')
        norm_tooltip = CustomHovertip(
            self.normalize_audio_checkbox, 'Normalizes the audio of each clip to 0 dB. Use this if your clips have wildly different volumes.')
        output_tooltip = CustomHovertip(
            self.output_location_label, f"{self.output_video_path.get()}")
        cancel_tooltip = CustomHovertip(
            self.cancel_button, 'Cancel the current compilation process.')
        gpu_tooltip = CustomHovertip(
            self.use_gpu_checkbox, 'Run AI detection on your NVIDIA GPU (much faster).\nUncheck to run on CPU instead — slower, but keeps your GPU quiet and cool.\nHas no effect if you don\'t have CUDA set up.')
        timestamps_tooltip = CustomHovertip(
            self.txt_output_checkbox, 'Save the timestamps to a txt file (by default `timestamps.txt` in the output directory).\nYou can change the file name in settings.')
        padding_tooltip = CustomHovertip(
            self.use_clip_padding_checkbox, 'Add extra time before and after each individual clip. Values are in seconds.\nIf using this option, Iit\'s recommended to enable \'Merge Nearby Clips\' to avoid duplicate clips.'
        )
        verify_tooltip = CustomHovertip(
            self.verify_checkbox, 'After AI detection, scan near each clip with a lower threshold to find clips the AI may have missed.')
        review_tooltip = CustomHovertip(
            self.review_checkbox, 'Before compiling, open a dialog to preview, check/uncheck, and edit each clip individually.')
        strict_fp_tooltip = CustomHovertip(
            self.strict_fp_checkbox, 'Drop clips where another sound class (speech/scream/etc.) scores higher than burp.\nReduces false positives for noisy streamers, but may rarely miss real burps mixed with loud talking.\nSuspect clips are also shown pre-deselected in Review regardless of this option.')
        skip_auto_tooltip = CustomHovertip(
            self.skip_auto_checkbox, 'When a timestamps.txt file already exists, automatically use it without showing the confirmation dialog.')
        settings_tooltip = CustomHovertip(self.settings_button, "Settings")

        self.disable_while_processing = [
            self.add_button,
            self.remove_button,
            self.up_arrow,
            self.down_arrow,
            self.sort_button,
            self.clear_button,
            self.process_button,
            self.model_dropdown,
            self.precision_entry,
            self.block_size_entry,
            self.threshold_entry,
            self.merge_clips_checkbox,
            self.combine_checkbox,
            self.custom_resolution_checkbox,
            self.res_height_entry,
            self.res_width_entry,
            self.output_location_button,
            self.normalize_audio_checkbox,
            self.toggle_button,
            self.txt_output_checkbox,
            self.settings_button,
            self.use_clip_padding_checkbox,
            self.padding_before_entry,
            self.padding_after_entry,
            self.verify_checkbox,
            self.review_checkbox,
            self.strict_fp_checkbox,
            self.skip_auto_checkbox,
            self.focus_idx_entry,
            self.preset_combo,
            self.save_preset_btn,
            self.use_gpu_checkbox,
            self.remote_mode_dropdown,
            self.remote_browser_cookies_dropdown,
            self.remote_concurrency_spinbox,
            self.remote_cache_open_button,
            self.remote_cache_clear_button,
            self.remote_cache_choose_button,
            self.import_external_audio_button,
        ]
        
        self.enable_while_processing = [
            self.cancel_button
        ]

        self._refresh_preset_combo()

    def _queue_transfer_progress(self, sample, label_prefix=""):
        if not hasattr(self, "transfer_progress"):
            return
        display = dict(sample)
        title = compose_progress_title(display, label_prefix)
        if title:
            display["text"] = f"{title}: {display.get('text', '')}"
            display["title"] = title
        text = display.get("text", "").lower()
        is_compile = title.lower() == "compile"
        self.transfer_progress.submit(
            display, force=is_compile or ("completed" in text or "failed" in text or "cancelled" in text)
        )

    def _poll_ui(self):
        # 单轮最多处理有限条 UI 更新，避免一次性排空大量积压事件
        # 阻塞主线程事件循环导致窗口假死（与 StdoutRedirector 的渲染上限对齐）。
        processed = 0
        try:
            while processed < 100:
                func, args = self._ui_queue.get_nowait()
                try:
                    func(*args)
                except Exception:
                    pass
                processed += 1
        except queue.Empty:
            pass
        self.root.after(50, self._poll_ui)

    def _schedule_ui(self, func, *args):
        """Run func(*args) on the main thread via the thread-safe UI queue."""
        if hasattr(self, "_ui_queue"):
            self._ui_queue.put((func, args))
        else:
            func(*args)

    def clear_transfer_progress(self, text=""):
        # 合并写：多次调用只保留最新 text，且同一时刻至多入队 1 条待执行更新，
        # 防止高频进度事件把 _ui_queue 灌爆。
        if not hasattr(self, "_progress_pending"):
            self._progress_pending = None
        self._progress_pending = text
        if getattr(self, "_progress_flush_queued", False):
            return
        self._progress_flush_queued = True

        def flush():
            self._progress_flush_queued = False
            text = self._progress_pending
            self._progress_pending = None
            if hasattr(self, "remote_progress_text"):
                self.remote_progress_text.set(text)
            if hasattr(self, "ui_bar"):
                self.ui_bar.__setitem__("value", 0)

        self._schedule_ui(flush)

    def _show_remote_clip_progress(self, event):
        # 材料化期间 FFmpeg 进度事件高频到达，progress 事件节流刷新，
        # start/complete/failed 等状态事件强制立即展示。
        if not hasattr(self, "_clip_progress_throttle"):
            self._clip_progress_throttle = ProgressThrottle(
                self._render_remote_clip_progress, interval=0.75)
        forced = event.get("kind", "") != "progress"
        self._clip_progress_throttle.update(event, force=forced)

    def _render_remote_clip_progress(self, event):
        kind = event.get("kind", "")
        completed = int(event.get("completed", 0) or 0)
        total = int(event.get("total", 0) or 0)
        percent = int(completed / total * 100) if total else 0
        if kind == "progress":
            lines = [
                event.get("title", ""),
                event.get("current_line", ""),
                event.get("speed_line", ""),
                event.get("eta_line", ""),
            ]
        else:
            state = str(kind).capitalize()
            lines = [
                f"Preparing clips: {completed}/{total} ({percent}%)",
                f"Video {event.get('video')}/{event.get('videos_total')} · "
                f"clip {event.get('clip')}/{event.get('clips_total')} · "
                f"Range: {event.get('start', 0):g}-{event.get('end', 0):g}s",
                f"Status: {state}",
            ]
        self.clear_transfer_progress("\n".join(line for line in lines if line))

    def disable_objects(self):
        for elt in self.disable_while_processing:
            elt["state"] = tk.DISABLED
        
        for elt in self.enable_while_processing:
            elt["state"] = tk.NORMAL

    def reenable_disabled_objects(self):
        for elt in self.disable_while_processing:
            if elt == self.model_dropdown:
                elt["state"] = "readonly"
            else:
                elt["state"] = tk.NORMAL
        
        for elt in self.enable_while_processing:
            elt["state"] = tk.DISABLED

    def populate_add_button(self):
        self.media_menu.delete(0, 'end')
        if self.is_video:
            self.media_menu.add_command(
                label=" Add Video Files ", command=self.add_video)
            self.media_menu.add_command(
                label=" Add URL ", command=self.add_video_url)
            self.media_menu.add_command(
                label=" Add Folder ", command=self.add_video_folder)
        else:
            self.media_menu.add_command(
                label=" Add Audio Files ", command=self.add_video)
            self.media_menu.add_command(
                label=" Add URL ", command=self.add_video_url)
            self.media_menu.add_command(
                label=" Add Folder ", command=self.add_video_folder)

    def clear_output(self):
        self.output_video_path.set("No location selected!")
        self.output_tooltip = CustomHovertip(
            self.output_location_label, f"{self.output_video_path.get()}")

    def toggle_text_boxes(self):
        # Toggle the visibility of text boxes based on the checkbox state
        if self.use_custom_resolution.get():  # Checkbox is checked
            self.container_frame.pack(after=self.custom_resolution_checkbox)
        else:  # Checkbox is unchecked
            self.container_frame.pack_forget()

    def toggle_padding_text_boxes(self):
        # Toggle the visibility of text boxes based on the checkbox state
        if self.use_custom_padding.get():  # Checkbox is checked
            self.padding_container_frame.pack(
                after=self.use_clip_padding_checkbox)
        else:  # Checkbox is unchecked
            self.padding_container_frame.pack_forget()

    def custom_warning_dialog(self, parent, title, message):
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.grab_set()
        dialog.minsize(width=400, height=200)
        dialog.resizable(False, False)
        x = parent.winfo_x() + 15
        y = parent.winfo_y() + 15
        dialog.geometry(f"+{x}+{y}")

        result = {"action": None, "dont_show_again": False}

        message_label = ttk.Label(dialog, text=message, wraplength=280)
        message_label.pack(pady=10, padx=10)

        dont_show_again_check = ttk.Checkbutton(
            dialog, text="Don't show this again", variable=self.dont_show_again_var
        )
        dont_show_again_check.pack(pady=5)

        def on_continue():
            result["action"] = "continue"
            result["dont_show_again"] = self.dont_show_again_var.get()
            dialog.destroy()

        # Function to handle Cancel button click
        def on_cancel():
            result["action"] = "cancel"
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        continue_button = ttk.Button(
            button_frame, text="Continue", command=on_continue)
        continue_button.pack(side="left", padx=5)

        cancel_button = ttk.Button(
            button_frame, text="Stop", command=on_cancel)
        cancel_button.pack(side="right", padx=5)

        parent.wait_window(dialog)

        return result

    def add_video(self):
        input_formats = VIDEO_INPUT if self.is_video else AUDIO_INPUT

        file_names = filedialog.askopenfilenames(
            filetypes=input_formats)
        if file_names:
            for file in file_names:
                self.uploaded_videos.append(MediaUpload(
                    file, 'video' if self.is_video else 'audio'))
            self.update_listbox(scroll_to_bottom=True)

    def _on_cookie_selection(self, event=None):
        """Handle a browser-cookie dropdown change; open a picker for Cookies File…."""
        label = str(self.remote_browser_cookies.get() or "").strip()
        if label.casefold() != "cookies file…" and not label.casefold().startswith("cookies file"):
            return
        path = filedialog.askopenfilename(
            title="Select a cookies.txt file",
            filetypes=[("Cookie files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.remote_cookies_file.set(path)
            self.remote_browser_cookies.set(
                f"Cookies File: {os.path.basename(path)}"
            )
        else:
            previous = "Cookies File…" if self.remote_cookies_file.get() else "Auto"
            self.remote_browser_cookies.set(previous)


    def _pick_playlist_sources(self, descriptor):
        """Show a lazy paged playlist review and return confirmed entries."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Review Playlist Entries")
        dialog.transient(self.root)
        dialog.geometry("900x600")
        dialog.minsize(900, 600)
        dialog.resizable(True, True)

        selected_ids = set()
        selected_order = []
        page_index = [0]
        generation = [0]
        closed = [False]
        sort_status = [""]
        hydration_pending = set()
        hydration_executor = ThreadPoolExecutor(max_workers=4)
        page_label = ttk.Label(dialog)
        page_label.pack(pady=(10, 4))
        entries_frame = ttk.Frame(dialog)
        entries_frame.pack(fill="both", expand=True, padx=12)
        tree = ttk.Treeview(
            entries_frame,
            columns=("order", "select", "title", "part", "duration", "date", "status"),
            show="headings",
            selectmode="browse",
        )
        tree.heading("order", text="Order")
        tree.heading("select", text="Select")
        tree.heading("title", text="Title")
        tree.heading("part", text="Part")
        tree.heading("duration", text="Duration")
        tree.heading("date", text="Date")
        tree.heading("status", text="Status")
        tree.column("order", width=55, anchor="center", stretch=False)
        tree.column("select", width=70, anchor="center", stretch=False)
        tree.column("title", width=470, anchor="w")
        tree.column("part", width=55, anchor="center", stretch=False)
        tree.column("duration", width=90, anchor="e", stretch=False)
        tree.column("date", width=110, anchor="center", stretch=False)
        tree.column("status", width=110, anchor="center", stretch=False)
        scrollbar = ttk.Scrollbar(entries_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        controls = ttk.Frame(dialog)
        controls.pack(pady=(8, 4))
        actions = ttk.Frame(dialog)
        actions.pack(pady=(2, 10))

        def render_page():
            generation[0] += 1
            render_generation = generation[0]
            for item in tree.get_children():
                tree.delete(item)
            current_page = page_index[0]
            page = descriptor.load_page(current_page, hydrate=False)
            visible_ids = {entry.entry_id for entry in page}
            page_label.config(
                text=(f"{descriptor.title}  |  Page {current_page + 1} of "
                      f"{max(1, (descriptor.total_count + descriptor.page_size - 1) // descriptor.page_size)}"
                      f"  |  Selected: {len(selected_ids)}{sort_status[0]}")
            )
            for entry in page:
                needs_hydration = descriptor.needs_hydration(entry)
                if needs_hydration:
                    entry.metadata["hydration_pending"] = True
                tree.insert("", "end", iid=entry.entry_id,
                            values=playlist_tree_values(entry, selected_ids, selected_order))
                if needs_hydration and entry.entry_id not in hydration_pending:
                    hydration_pending.add(entry.entry_id)
                    hydration_executor.submit(
                        hydrate_page_entry, entry, current_page, render_generation, visible_ids
                    )
            previous_button.config(state=tk.NORMAL if page_index[0] else tk.DISABLED)
            next_button.config(
                state=(tk.NORMAL if (page_index[0] + 1) * descriptor.page_size < descriptor.total_count
                       else tk.DISABLED)
            )

        def hydrate_page_entry(entry, render_page_index, render_generation, visible_ids):
            try:
                descriptor.hydrate_entry(entry)
            finally:
                hydration_pending.discard(entry.entry_id)
            if closed[0]:
                return
            if entry.metadata.get("hydration_failed"):
                retry_count = int(entry.metadata.get("hydration_retries", 0) or 0)
                if retry_count < 3:
                    entry.metadata["hydration_retries"] = retry_count + 1
                    backoff = (5, 15, 45)[retry_count]
                    if retry_count == 0:
                        try:
                            self.root.after(0, lambda: apply_hydration_update(
                                entry, render_page_index, render_generation, visible_ids))
                        except tk.TclError:
                            pass
                    try:
                        self.root.after(
                            int(backoff * 1000),
                            lambda: schedule_hydration_retry(
                                entry, render_page_index, render_generation, visible_ids),
                        )
                    except tk.TclError:
                        pass
                else:
                    entry.metadata.pop("hydration_retries", None)
                    try:
                        self.root.after(0, lambda: apply_hydration_update(
                            entry, render_page_index, render_generation, visible_ids))
                    except tk.TclError:
                        pass
                return
            try:
                self.root.after(0, lambda: apply_hydration_update(
                    entry, render_page_index, render_generation, visible_ids))
            except tk.TclError:
                pass

        def schedule_hydration_retry(entry, render_page_index, render_generation, visible_ids):
            if closed[0]:
                return
            entry.metadata["hydration_failed"] = False
            if entry.entry_id not in hydration_pending:
                hydration_pending.add(entry.entry_id)
                hydration_executor.submit(
                    hydrate_page_entry, entry, render_page_index, render_generation, visible_ids
                )

        def apply_hydration_update(entry, render_page_index, render_generation, visible_ids):
            entry.metadata["hydration_pending"] = False
            if closed[0] or not hydration_update_is_current(
                    entry.entry_id, render_page_index, render_generation,
                    page_index[0], generation[0], visible_ids):
                return
            if tree.exists(entry.entry_id):
                retry_count = int(entry.metadata.get("hydration_retries", 0) or 0)
                values = playlist_tree_values(entry, selected_ids, selected_order)
                if entry.metadata.get("hydration_failed") and retry_count and retry_count < 3:
                    values = list(values)
                    values[6] = f"Retrying ({retry_count}/3)..."
                    values = tuple(values)
                tree.item(entry.entry_id, values=values)

        def select_page(value):
            for entry in descriptor.load_page(page_index[0], hydrate=False):
                if value:
                    selected_ids.add(entry.entry_id)
                    if entry.entry_id not in selected_order:
                        selected_order.append(entry.entry_id)
                else:
                    selected_ids.discard(entry.entry_id)
                    if entry.entry_id in selected_order:
                        selected_order.remove(entry.entry_id)
            render_page()

        def toggle_entry(event):
            row = tree.identify_row(event.y)
            column = tree.identify_column(event.x)
            if not row or column != "#2":
                return
            if row in selected_ids:
                selected_ids.remove(row)
                selected_order.remove(row)
            else:
                selected_ids.add(row)
                selected_order.append(row)
            render_page()

        def confirm():
            selected = selected_playlist_entries(descriptor, selected_ids, selected_order)
            if not selected:
                messagebox.showwarning(
                    "Nothing selected", "Select at least one playlist entry.", parent=dialog
                )
                return
            result["sources"] = selected
            closed[0] = True
            generation[0] += 1
            hydration_executor.shutdown(wait=False, cancel_futures=True)
            _release_grab(dialog)
            dialog.destroy()

        def cancel():
            closed[0] = True
            generation[0] += 1
            hydration_executor.shutdown(wait=False, cancel_futures=True)
            _release_grab(dialog)
            dialog.destroy()

        previous_button = ttk.Button(
            controls, text="Previous", command=lambda: move_page(-1)
        )
        previous_button.pack(side="left", padx=2)
        next_button = ttk.Button(
            controls, text="Next", command=lambda: move_page(1)
        )
        next_button.pack(side="left", padx=2)
        ttk.Button(
            controls, text="Select Page", command=lambda: select_page(True)
        ).pack(side="left", padx=2)
        ttk.Button(
            controls, text="Deselect Page", command=lambda: select_page(False)
        ).pack(side="left", padx=2)
        ttk.Button(
            controls, text="Sort Selected",
            command=lambda: sort_selected(),
        ).pack(side="left", padx=2)
        ttk.Label(controls, text="Jump to page:").pack(side="left", padx=(10, 2))
        _total_pages = max(
            1, (descriptor.total_count + descriptor.page_size - 1) // descriptor.page_size
        )
        _jump_page = tk.IntVar(value=1)

        def go_to_page():
            try:
                target = int(_jump_page.get())
            except (TypeError, ValueError, tk.TclError):
                target = 1
            target = max(1, min(target, _total_pages))
            page_index[0] = target - 1
            render_page()

        jump_spin = ttk.Spinbox(
            controls, from_=1, to=_total_pages, width=4,
            textvariable=_jump_page,
        )
        jump_spin.pack(side="left", padx=(0, 2))
        jump_spin.bind("<Return>", lambda event: go_to_page())
        ttk.Button(controls, text="Go", command=go_to_page).pack(side="left", padx=2)
        ttk.Button(actions, text="Confirm Selected", command=confirm).pack(side="left", padx=4)
        ttk.Button(actions, text="Cancel", command=cancel).pack(side="left", padx=4)

        result = {"sources": None}

        def move_page(delta):
            page_index[0] = max(0, page_index[0] + delta)
            render_page()

        def sort_selected():
            selected_entries = selected_playlist_entries(descriptor, selected_ids, selected_order)
            selected_order[:] = [entry.entry_id for entry in sort_playlist_entries(selected_entries)]
            sort_status[0] = f"  |  Sorted {len(selected_order)} selected entries"
            render_page()

        tree.bind("<Button-1>", toggle_entry)
        render_page()
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        _release_grab(getattr(self, "entry_window", None))
        dialog.lift()
        dialog.focus_force()
        dialog.grab_set()
        self.root.wait_window(dialog)
        return result["sources"]

    def add_video_url(self):
        self.entry_window = tk.Toplevel(self.root)
        x = self.root.winfo_x() + 15
        y = self.root.winfo_y() + 15
        self.entry_window.geometry(f"400x130+{x}+{y}")
        self.entry_window.title("Enter URL")
        self.entry_window.resizable(False, False)
        self.entry_window.transient(self.root)

        entry_label = ttk.Label(
            self.entry_window, font=(None, 12, "bold"), text="Enter a URL and Press Enter:")
        entry_label.pack(pady=10)

        entry_label = ttk.Label(
            self.entry_window, font=(None, 10), text="Please be patient when submitting playlists")
        entry_label.pack(pady=(5, 0))

        url_entry = ttk.Entry(self.entry_window, width=50)
        url_entry.pack(pady=5)

        self.thread_active = False

        def check_url():
            url = url_entry.get()
            try:
                described = describe_input(
                    url,
                    browser_cookies=browser_cookie_setting_value(
                        self.remote_browser_cookies.get(),
                        cookies_file=self.remote_cookies_file.get(),
                    ),
                )
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Invalid URL: {url}\nError: {exc}"
                ))
                self.root.after(0, lambda: url_entry.config(state=tk.NORMAL))
                self.root.after(0, lambda: setattr(self, "thread_active", False))
                return

            def finish_description():
                if isinstance(described, PlaylistDescriptor):
                    selected_entries = self._pick_playlist_sources(described)
                else:
                    selected_entries = [described]
                if selected_entries is None:
                    _release_grab(self.entry_window)
                    self.entry_window.destroy()
                    self.thread_active = False
                    return
                failures = []
                expansion_stats = {}
                cookies = browser_cookie_setting_value(
                    self.remote_browser_cookies.get(),
                    cookies_file=self.remote_cookies_file.get(),
                )

                def resolve_selected():
                    sources = resolve_playlist_entries(
                        selected_entries,
                        {entry.entry_id for entry in selected_entries},
                        browser_cookies=cookies,
                        failure_logger=failures.append,
                        status_logger=print,
                        expansion_stats=expansion_stats,
                    )

                    def finish_import():
                        for media_obj in media_uploads_for_sources(
                            sources, "video" if self.is_video else "audio"
                        ):
                            self.uploaded_videos.append(media_obj)
                            self.update_listbox_add_video(scroll_to_bottom=True)
                        count = len(sources)
                        message = f"Successfully imported {count} video" + ("." if count == 1 else "s.")
                        if expansion_stats.get("expanded_parts"):
                            message += f" Expanded {expansion_stats['expanded_parts']} parts."
                        if failures:
                            messagebox.showwarning(
                                "Import completed",
                                f"{message}\nSkipped {len(failures)} item(s).",
                            )
                        elif count:
                            messagebox.showinfo("Success", message)
                        else:
                            messagebox.showwarning("Import failed", "No selected entries could be resolved.")
                        _release_grab(self.entry_window)
                        self.entry_window.destroy()
                        self.thread_active = False

                    self.root.after(0, finish_import)

                threading.Thread(target=resolve_selected).start()

            self.root.after(0, finish_description)

        def close_add_url(event=None):
            if self.thread_active:
                confirm = messagebox.askyesno("Confirm Cancellation",
                                              f"The current job will be cancelled, but any previously parsed URLs will be kept. Would you like to cancel?")
                if confirm:
                    _release_grab(self.entry_window)
                    self.entry_window.destroy()
            else:
                _release_grab(self.entry_window)
                self.entry_window.destroy()

        def check_url_threaded(event=None):
            self.thread_active = True
            url_entry["state"] = tk.DISABLED
            thread = threading.Thread(target=check_url)
            thread.start()

        self.entry_window.protocol("WM_DELETE_WINDOW", close_add_url)

        url_entry.bind("<Return>", check_url_threaded)
        url_entry.bind("<Escape>", close_add_url)
        url_entry.focus_set()

        self.entry_window.grab_set()
        self.root.wait_window(self.entry_window)

    def _on_listbox_resize(self, event):
        if self._listbox_resize_after is not None:
            self.root.after_cancel(self._listbox_resize_after)
        self._listbox_resize_after = self.root.after(200, self.update_listbox)

    def _display_name(self, video_path: str) -> str:
        """Listbox 显示名：超长文件名中间省略，保住开头序号与结尾日期。"""
        name = str(os.path.basename(video_path))
        w = self.video_listbox.winfo_width()
        if w < 50:  # 尚未渲染，按主窗口左栏常见宽度兜底
            w = 700
        return _elide_middle(name, w - 30, self._listbox_font.measure)

    def update_listbox(self, scroll_to_bottom: bool = False):
        self.video_listbox.delete(*self.video_listbox.get_children())

        for video in self.uploaded_videos:
            video_path = video.get_path()
            item_number = len(self.video_listbox.get_children())
            if video.get_is_url():
                self.video_listbox.insert("", "end", item_number, values=(
                    str(video_path),))
            else:
                self.video_listbox.insert("", "end", item_number, values=(
                    self._display_name(video_path),))

        if scroll_to_bottom:
            self.video_listbox.yview_moveto(1.0)

        self.video_listbox.pack()

    def update_listbox_add_video(self, scroll_to_bottom: bool = False):
        current_items = {self.video_listbox.item(item_id, 'values')[
            0]: item_id for item_id in self.video_listbox.get_children()}

        for video in self.uploaded_videos:
            video_path = video.get_path()
            video_key = str(video_path) if video.get_is_url() else self._display_name(video_path)

            if video_key not in current_items:
                item_number = len(self.video_listbox.get_children())
                self.video_listbox.insert(
                    "", "end", item_number, values=(video_key,))
            else:
                del current_items[video_key]

        for video_key, item_id in current_items.items():
            self.video_listbox.delete(item_id)

        if scroll_to_bottom:
            self.video_listbox.yview_moveto(1.0)

        self.video_listbox.pack()

    def move_selected_up(self):
        selected_index = self.video_listbox.selection()
        selected_index = tuple(int(x) for x in selected_index)
        if selected_index and len(selected_index) != len(self.uploaded_videos):
            for i in sorted(selected_index):
                if i != 0:
                    self.uploaded_videos[i -
                                         1], self.uploaded_videos[i] = self.uploaded_videos[i], self.uploaded_videos[i - 1]
            self.update_listbox()

        self.video_listbox.selection_clear()
        for i in selected_index:
            if i != 0:
                self.video_listbox.selection_add(str(i - 1))
            else:
                self.video_listbox.selection_add(str(i))

    def move_selected_down(self):
        selected_index = self.video_listbox.selection()
        selected_index = tuple(int(x) for x in selected_index)
        if selected_index and len(selected_index) != len(self.uploaded_videos):
            for i in reversed(sorted(selected_index)):
                if i != len(self.uploaded_videos) - 1:
                    self.uploaded_videos[i], self.uploaded_videos[i +
                                                                  1] = self.uploaded_videos[i + 1], self.uploaded_videos[i]
            self.update_listbox()

        self.video_listbox.selection_clear()
        for i in selected_index:
            if i != len(self.uploaded_videos) - 1:
                self.video_listbox.selection_add(str(i + 1))
            else:
                self.video_listbox.selection_add(str(i))

    def remove_selected(self):
        selected_index = self.video_listbox.selection()
        selected_index = tuple(int(x) for x in selected_index)
        if selected_index:
            for i in reversed(sorted(selected_index)):
                del self.uploaded_videos[i]
            self.update_listbox()

    def clear_list(self):
        self.uploaded_videos = []
        self.update_listbox()

    def sort_filelist(self):
        """Toggle sort: first click ascending (0→1), second descending (1→0)."""
        self.sort_ascending = not self.sort_ascending
        self.uploaded_videos.sort(key=lambda x: _smart_sort_key(x.get_path()),
                                  reverse=not self.sort_ascending)
        self.sort_button.config(text="↑" if self.sort_ascending else "↓")
        self.update_listbox()

    def _on_drop(self, event):
        """Handle drag-and-drop of video files."""
        files = self.root.tk.splitlist(event.data)
        for f in files:
            video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.ts')
            audio_exts = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')
            if f.lower().endswith(video_exts + audio_exts):
                existing = [x.get_path() if hasattr(x, 'get_path') else x for x in self.uploaded_videos]
                if f not in existing:
                    mtype = 'video' if f.lower().endswith(video_exts) else 'audio'
                    self.uploaded_videos.append(MediaUpload(f, mtype))
        self.uploaded_videos.sort(key=lambda x: _smart_sort_key(x.get_path()))
        self.update_listbox()

    def _preset_path(self):
        """Path to autocomper_presets.json — adjacent to the .exe or cwd."""
        import sys, os as _os
        if getattr(sys, 'frozen', False):
            return _os.path.join(_os.path.dirname(sys.executable), 'autocomper_presets.json')
        return _os.path.join(_os.getcwd(), 'autocomper_presets.json')

    def _collect_all_vars(self):
        """Collect all tkinter variable values into a dict."""
        import tkinter as tk
        preset = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            try:
                v = getattr(self, attr)
            except Exception:
                continue
            if isinstance(v, tk.BooleanVar):
                preset[attr] = v.get()
            elif isinstance(v, tk.IntVar):
                preset[attr] = v.get()
            elif isinstance(v, tk.DoubleVar):
                preset[attr] = v.get()
            elif isinstance(v, tk.StringVar):
                preset[attr] = v.get()
        # remove transient/state vars
        for k in ['active_thread', 'thread_active', 'remote_cache_size']:
            preset.pop(k, None)
        return preset

    def _apply_all_vars(self, data):
        """Restore tkinter variable values from a dict."""
        import tkinter as tk
        data = normalize_remote_settings(data)
        for attr, val in data.items():
            if attr == "remote_cache_path":
                continue
            try:
                v = getattr(self, attr)
            except Exception:
                continue
            if isinstance(v, (tk.BooleanVar, tk.IntVar, tk.DoubleVar, tk.StringVar)):
                try:
                    v.set(val)
                except Exception:
                    pass
        if "remote_cache_path" in data:
            restore_remote_cache_path(
                self,
                data["remote_cache_path"],
                lambda warning: messagebox.showwarning("Remote Cache", warning),
            )

    def save_preset(self):
        """Save current settings to autocomper_presets.json with a custom name."""
        import json
        from tkinter import simpledialog
        name = simpledialog.askstring("Save Preset", "Preset name:")
        if not name:
            return
        pp = self._preset_path()
        try:
            with open(pp, 'r') as f:
                presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            presets = {}
        presets[name] = self._collect_all_vars()
        with open(pp, 'w') as f:
            json.dump(presets, f, indent=2)
        print(f"{Fore.GREEN}Preset '{name}' saved.")
        self._refresh_preset_combo()

    def _refresh_preset_combo(self):
        """Reload preset list into the ComboBox."""
        import json
        try:
            with open(self._preset_path(), 'r') as f:
                presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            presets = {}
        names = list(presets.keys())
        self.preset_combo["values"] = names
        if names:
            self.preset_combo_var.set(names[0])
        else:
            self.preset_combo_var.set("Load Preset")

    def _on_preset_selected(self, event=None):
        """ComboBox callback: load the selected preset."""
        import json
        name = self.preset_combo_var.get()
        if not name or name == "Load Preset":
            return
        try:
            with open(self._preset_path(), 'r') as f:
                presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        data = presets.get(name)
        if not data:
            return
        self._apply_all_vars(data)
        # sync padding UI with loaded values
        self.toggle_padding_text_boxes()
        print(f"{Fore.GREEN}Preset '{name}' loaded.")

    def refresh_remote_cache_size(self):
        self.remote_cache_size.set(
            f"Current size: {format_cache_size(self.remote_cache_store.get_cache_size())}")

    def choose_remote_cache(self):
        selected_path = filedialog.askdirectory(title="Choose Cache Folder")
        if not selected_path:
            return
        try:
            store = select_remote_cache_store(self.remote_cache_store, selected_path)
        except OSError as exc:
            messagebox.showerror("Remote Cache", f"Could not use cache folder:\n{exc}")
            return
        self.remote_cache_store = store
        self.remote_cache_path.set(str(store.root))
        self.save_settings()
        self.refresh_remote_cache_size()

    def import_external_audio(self):
        selected = self.video_listbox.selection()
        if len(selected) != 1:
            messagebox.showwarning(
                "Import External Audio",
                "Select exactly one remote VOD in the main list first.",
            )
            return
        try:
            upload = self.uploaded_videos[int(selected[0])]
        except (ValueError, IndexError):
            messagebox.showerror("Import External Audio", "Could not identify the selected VOD.")
            return
        if not upload.get_is_url():
            messagebox.showwarning(
                "Import External Audio",
                "The selected item is a local file. Select a YouTube, Twitch, or Bilibili URL.",
            )
            return

        audio_path = filedialog.askopenfilename(
            title="Select External Audio or Container",
            filetypes=[
                ("Audio and Media Files", "*.m4a *.m4s *.mp4 *.webm *.opus *.mp3 *.wav *.flac"),
                ("All Files", "*.*"),
            ],
        )
        if not audio_path:
            return
        self.import_external_audio_button.configure(state=tk.DISABLED)
        self.clear_transfer_progress("Inspecting external media...")
        threading.Thread(
            target=self._import_external_audio_worker,
            args=(upload.get_source(), upload.get_url() or upload.get_path(), audio_path,
                  self.remote_browser_cookies.get()),
            daemon=True,
        ).start()

    def _import_external_audio_worker(self, existing_source, source_url, audio_path, cookies):
        temporary_path = None
        try:
            source = existing_source if isinstance(existing_source, MediaSource) else resolve_source(
                source_url, browser_cookies=cookies)
            if not source.audio_url:
                raise ValueError("The selected VOD has no audio stream.")
            source_duration = float(source.duration) if source.duration else None
            audio_duration = _get_video_duration(audio_path)
            audio_codec = _get_external_audio_codec(audio_path)
            if not audio_codec:
                raise ValueError("The selected file does not contain a readable audio stream.")
            if source_duration and audio_duration:
                tolerance = max(5.0, source_duration * 0.01)
                if abs(source_duration - audio_duration) > tolerance:
                    raise ValueError(
                        f"Audio duration ({audio_duration:.1f}s) does not match "
                        f"VOD duration ({source_duration:.1f}s)."
                    )

            fd, temporary = tempfile.mkstemp(
                prefix="external-audio-", suffix=".m4a", dir=str(TEMP_DIR)
            )
            os.close(fd)
            temporary_path = Path(temporary)

            def report(current, total, elapsed):
                sample = format_compile_progress(
                    current, total, elapsed, "Converting external audio"
                )
                self.root.after(0, self._queue_transfer_progress, sample, "Import")

            codec_args = ["-c:a", "copy"] if audio_codec in {"aac", "mp4a"} else [
                "-c:a", "aac", "-b:a", "192k"
            ]
            result = run_tracked_progress([
                FFMPEG_PATH, "-hide_banner", "-loglevel", "warning", "-y",
                "-i", str(audio_path), "-vn", *codec_args,
                "-movflags", "+faststart", str(temporary_path),
            ], duration=audio_duration, timeout=600, progress_callback=report)
            if result.returncode != 0 and codec_args == ["-c:a", "copy"]:
                temporary_path.unlink(missing_ok=True)
                result = run_tracked_progress([
                    FFMPEG_PATH, "-hide_banner", "-loglevel", "warning", "-y",
                    "-i", str(audio_path), "-vn", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(temporary_path),
                ], duration=audio_duration, timeout=600, progress_callback=report)
            if result.returncode != 0 or not temporary_path.is_file():
                detail = getattr(result, "stdout", "") or getattr(result, "stderr", "")
                raise RuntimeError(f"Audio conversion failed: {detail}")
            metadata = _audio_cache_format_identity(source)
            self.remote_cache_store.save_audio_cache_file(
                stable_source_id(source), source.audio_url, "m4a", temporary_path,
                metadata=metadata,
            )
            temporary_path = None
            self.root.after(0, self._finish_external_audio_import,
                            source.display_name or source.source_id)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.root.after(0, self._fail_external_audio_import, str(exc))

    def _finish_external_audio_import(self, display_name):
        self._restore_external_audio_button_state()
        self.refresh_remote_cache_size()
        self.clear_transfer_progress("External audio imported.")
        messagebox.showinfo(
            "Import External Audio",
            f"External audio registered for:\n{display_name}",
        )

    def _fail_external_audio_import(self, error):
        self._restore_external_audio_button_state()
        self.clear_transfer_progress("External audio import failed.")
        messagebox.showerror("Import External Audio", error)

    def _restore_external_audio_button_state(self):
        # import 完成后本应恢复按钮；但若此刻 batch 正在运行（disable_objects
        # 已禁用该按钮），则保持禁用，等运行结束时 reenable_disabled_objects 再恢复。
        try:
            if str(self.import_external_audio_button.cget("state")) != tk.DISABLED:
                self.import_external_audio_button.configure(state=tk.NORMAL)
        except tk.TclError:
            self.import_external_audio_button.configure(state=tk.NORMAL)

    def open_remote_cache(self):
        path = str(self.remote_cache_store.root)
        try:
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(cache_open_command(sys.platform, path))
        except (OSError, AttributeError, subprocess.SubprocessError) as exc:
            messagebox.showerror("Open Cache Folder", f"Could not open cache folder: {exc}")

    def clear_remote_cache(self):
        if not messagebox.askyesno(
                "Clear Remote Cache",
                "Clear all remote cache files? The next run may need to download or detect media again."):
            return
        try:
            self.remote_cache_store.clear()
            self.refresh_remote_cache_size()
        except OSError as exc:
            messagebox.showerror("Clear Remote Cache", f"Could not clear cache: {exc}")

    def remove_urls_from_list(self):
        self.uploaded_videos = [
            x for x in self.uploaded_videos if os.path.dirname(x.get_path()) != TEMP_DIR]
        self.update_listbox()

    def select_output_location(self):
        if self.combine_vids.get():
            output_formats = VIDEO_OUTPUT if self.is_video else AUDIO_OUTPUT
            file_name = filedialog.asksaveasfilename(
                defaultextension=".mp4", filetypes=output_formats)
            if file_name:
                self.output_video_path.set(file_name)
                self.output_tooltip = CustomHovertip(
                    self.output_location_label, f"{self.output_video_path.get()}")
        else:
            folder_path = filedialog.askdirectory()
            if folder_path:
                self.output_video_path.set(folder_path)
                self.output_tooltip = CustomHovertip(
                    self.output_location_label, f"{self.output_video_path.get()}")


    def add_video_folder(self):
        """Recursively add all video/audio files from a folder"""
        folder = filedialog.askdirectory(title="Select Folder")
        if not folder:
            return
        extensions = ('.mp4', '.avi', '.mkv', '.m4v', '.mov') if self.is_video else ('.mp3', '.wav', '.flac')
        found = 0
        for root, _, files in os.walk(folder):
            for f in files:
                if any(f.lower().endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, f)
                    self.uploaded_videos.append(MediaUpload(full_path, 'video' if self.is_video else 'audio'))
                    found += 1
        self.uploaded_videos.sort(key=lambda x: _smart_sort_key(x.get_path()))
        self.update_listbox(scroll_to_bottom=True)
        print(f"{Fore.GREEN}Added {found} files from {folder}")


    def process_videos_multi(self):
        # Run video processing in new thread so the app doesn't hang
        self.active_thread = KThread(target=self.process_videos)
        self.active_thread.start()

    def is_thread_active(self):
        return type(self.active_thread) is KThread and self.active_thread.is_alive()

    def confirm_stop_process(self):
        # Check if there is a thread running
        if not self.is_thread_active():
            messagebox.showerror("Error", "No process is currently running!")
            return False
        else:
            confirm = messagebox.askyesno("Confirm Cancellation",
                                          f"The current job will be cancelled, losing all progress. Would you like to cancel?")
            if confirm:
                try:
                    kill_tracked_procs()
                    self.active_thread.terminate()
                finally:
                    print(
                        f"\n{Fore.RED}FAILURE: Operation cancelled by user.")
                    cleanup_temp_children()
                    self.clear_transfer_progress("Cancelled")
                    self.reenable_disabled_objects()
                    return True
            return False

    def on_closing(self):
        if self.is_thread_active():
            if self.confirm_stop_process():
                self.root.destroy()
        else:
            self.root.destroy()

    def save_settings(self):
        self.preferences.set(
            "Settings", "keep_downloaded_vids", str(self.keep_downloaded_vids.get()))
        self.preferences.set(
            "Settings", "download_path", self.download_video_path.get())
        self.preferences.set(
            "Settings", "max_quality", self.max_quality.get())
        self.preferences.set(
            "Settings", "max_download_speed", str(self.max_download_speed.get()))
        self.preferences.set(
            "Settings", "output_text_path", self.output_text_path.get())
        self.preferences.set(
            "Settings", "remote_cache_path", str(self.remote_cache_store.root))

        with open(self.preferences_file, 'w') as configfile:
            self.preferences.write(configfile)

    def reset_preferences_to_file(self):
        try:
            _kdv = self.preferences.getboolean("Settings", "keep_downloaded_vids")
        except (configparser.Error, ValueError):
            _kdv = False
        self.keep_downloaded_vids.set(_kdv)
        self.download_video_path.set(self.preferences.get(
            "Settings", "download_path"
        ))
        self.max_quality.set(self.preferences.get(
            "Settings", "max_quality"
        ))
        try:
            self.max_download_speed.set(int(self.preferences.get("Settings", "max_download_speed")))
        except (configparser.Error, ValueError, tk.TclError):
            self.max_download_speed.set(0)
        self.output_text_path.set(self.preferences.get(
            "Settings", "output_text_path"
        ))

    def open_settings_modal(self):
        modal = tk.Toplevel(self.root)
        modal.title("Settings")
        modal.geometry("640x480")
        modal.resizable(False, False)

        x = self.root.winfo_x() + 15
        y = self.root.winfo_y() + 15

        # Set the modal's position relative to the parent window
        modal.geometry(f"640x480+{x}+{y}")

        def on_close_save(event=None):
            self.save_settings()
            on_close()

        def on_close_no_save(event=None):
            self.reset_preferences_to_file()
            on_close()

        def on_close(event=None):
            modal.grab_release()
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", on_close_no_save)

        # Set all local variables to the stored values
        # in preferences.ini to maintain consistency
        self.reset_preferences_to_file()

        # DOWNLOAD SETTINGS

        ttk.Label(modal, text="Download Settings",
                  font=(None, 14, "bold")).pack(pady=(20, 5))

        def toggle_download_button():
            if self.keep_downloaded_vids.get():
                self.download_location_button.config(state="normal")
                self.download_location_text.config(state="readonly")
                self.clear_download_location_button.config(state="normal")
            else:
                self.download_location_button.config(state="disabled")
                self.download_location_text.config(state="disabled")
                self.clear_download_location_button.config(state="disabled")

        self.keep_saved_vids_checkbox = ttk.Checkbutton(
            modal, text="Keep Media Downloaded By URL", variable=self.keep_downloaded_vids,
            command=toggle_download_button)
        self.keep_saved_vids_checkbox.pack()

        download_settings_frame = ttk.Frame(modal)

        def get_download_location():
            folder_path = filedialog.askdirectory()
            if folder_path:
                self.download_video_path.set(folder_path)

        def clear_download_location():
            self.download_video_path.set("No location selected!")

        location_label_frame = ttk.Frame(download_settings_frame)
        self.download_location_label = ttk.Label(
            location_label_frame, text="Download Location:", font=(None, 11, "bold"))
        self.download_location_text = ttk.Entry(
            location_label_frame, textvariable=self.download_video_path, width=25, state="readonly")

        self.download_location_label.pack(side="left", padx=5, pady=5)
        self.download_location_text.pack(side="left", padx=5, pady=5)

        download_location_photo = get_photo_icon(
            os.path.join("img", "folder.png"))

        self.download_location_button = ttk.Button(
            location_label_frame, image=download_location_photo, width=5, padding=0, command=get_download_location)
        self.download_location_button.image = download_location_photo
        self.download_location_button.pack(side="left", padx=5, pady=5)

        stop_photo = get_photo_icon(
            os.path.join("img", "stop.png"))

        self.clear_download_location_button = ttk.Button(
            location_label_frame, image=stop_photo, width=5, padding=0, command=clear_download_location)
        # self.clear_download_location_button.image = stop_photo
        self.clear_download_location_button.pack(side="left", padx=5, pady=5)

        location_label_frame.pack()

        max_quality_frame = ttk.Frame(download_settings_frame)

        self.max_quality_label = ttk.Label(
            max_quality_frame, text="Max Download Quality:", font=(None, 11, "bold"))

        self.max_quality_dropdown = ttk.Combobox(
            max_quality_frame, textvariable=self.max_quality, values=DOWNLOAD_QUALITY_OPTIONS, state="readonly")

        self.max_quality_label.pack(side="left", padx=5, pady=5)
        self.max_quality_dropdown.pack(side="left", padx=5, pady=5)

        max_quality_frame.pack()
        
        max_download_speed_frame = ttk.Frame(download_settings_frame)

        self.max_download_speed_label = ttk.Label(
            max_download_speed_frame, text="Max Download Speed (KB/S):", font=(None, 11, "bold"))

        self.max_download_speed_entry = ttk.Entry(
            max_download_speed_frame, textvariable=self.max_download_speed, validate='key', validatecommand=self.num_check)

        self.max_download_speed_label.pack(side="left", padx=5, pady=5)
        self.max_download_speed_entry.pack(side="left", padx=5, pady=5)

        max_download_speed_frame.pack()

        download_settings_frame.pack()

        toggle_download_button()

        ttk.Separator(modal, orient="horizontal").pack(
            fill=tk.X, pady=5)

        # OUTPUT SETTINGS

        ttk.Label(modal, text="Output Settings",
                  font=(None, 14, "bold")).pack(pady=(20, 5))

        output_settings_frame = ttk.Frame(modal)

        def get_text_output_location():
            file_name = filedialog.asksaveasfilename(
                defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
            if file_name:
                self.output_text_path.set(file_name)

        def clear_text_output_location():
            self.output_text_path.set("No file selected!")

        text_output_frame = ttk.Frame(output_settings_frame)
        self.text_output_label = ttk.Label(
            text_output_frame, text="Timestamp Output File:", font=(None, 11, "bold"))
        self.text_output_text = ttk.Entry(
            text_output_frame, textvariable=self.output_text_path, width=25, state="readonly")

        self.text_output_label.pack(side="left", padx=5, pady=5)
        self.text_output_text.pack(side="left", padx=5, pady=5)

        self.text_location_button = ttk.Button(
            text_output_frame, image=download_location_photo, width=5, padding=0, command=get_text_output_location)
        self.text_location_button.image = download_location_photo
        self.text_location_button.pack(side="left", padx=5, pady=5)

        self.clear_text_output_location_button = ttk.Button(
            text_output_frame, image=stop_photo, width=5, padding=0, command=clear_text_output_location)
        self.clear_text_output_location_button.pack(
            side="left", padx=5, pady=5)

        text_output_frame.pack()

        output_settings_frame.pack()

        ttk.Separator(modal, orient="horizontal").pack(
            fill=tk.X, pady=5)

        style = ttk.Style()
        style.configure("Custom.TButton", font=("Helvetica", 14))
        ttk.Button(modal, text="Save Settings", command=on_close_save,
                   style="Custom.TButton").pack(pady=20)

        self.version_label = ttk.Label(
            modal, text=f"Autocomper v{VERSION}", font=(None, 10, "normal"), cursor="hand2")

        self.version_label.pack(side="bottom", padx=5, pady=(5, 15))
        
        def open_latest_release(event):
            webbrowser.open_new(REPO_URL)

        self.version_label.bind("<Button-1>", open_latest_release)

        modal.bind("<Return>", on_close_save)
        modal.bind("<Escape>", on_close_no_save)

        folder_tooltip = CustomHovertip(
            self.download_location_button, 'Choose Output Location')
        clear_tooltip = CustomHovertip(
            self.clear_download_location_button, 'Clear Output Location')
        
        max_speed_tooltip = CustomHovertip(
            self.max_download_speed_entry, 'Max allowable download speed in kilobytes per second. 0 means no limit.')

        folder_tooltip_two = CustomHovertip(
            self.text_location_button, 'Choose Timestamp TXT Output Location')
        clear_tooltip_two = CustomHovertip(
            self.clear_text_output_location_button, 'Clear Timestamp TXT Output Location')
        timestamp_output_label_tooltip = CustomHovertip(
            self.text_output_label, "Output file to save timestamps, if applicable.\nIf not chosen, they will be saved to 'timestamps.txt' in the selected output directory."
        )

        modal.transient(self.root)
        modal.grab_set()
        modal.focus_set()
        self.root.wait_window(modal)

    def handle_url_downloads(self):
        keep_downloaded_vids = self.keep_downloaded_vids.get()
        download_path = self.download_video_path.get()
        browser_cookies = browser_cookie_setting_value(
            self.remote_browser_cookies.get(),
            cookies_file=self.remote_cookies_file.get(),
        )

        # 防御：如果配置的下载目录不存在，回退到 TEMP_DIR
        if download_path and download_path != "No location selected!":
            if not os.path.isdir(download_path):
                print(f"{Fore.YELLOW}Configured download directory missing: {download_path}")
                print(f"{Fore.YELLOW}Falling back to temporary directory.")
                download_path = TEMP_DIR

        if not keep_downloaded_vids:
            download_path = TEMP_DIR

        if keep_downloaded_vids and (not download_path or download_path == "No location selected!"):
            raise Exception(
                "Please set a directory to save downloaded media. You can do this by clicking the gear in the top left.")

        indices_to_delete = []
        for i, video in enumerate(self.uploaded_videos):
            media_type = video.get_type()
            media_path = video.get_path()
            media_url = video.get_url()

            print(
                f"{Fore.GREEN}[{i + 1}/{len(self.uploaded_videos)}]{Style.RESET_ALL} Downloading {media_path}")

            if not video.get_is_url():
                print(f"{Fore.YELLOW}Not a URL, skipping...")
                continue

            if video.get_source() is None:
                try:
                    video.set_source(resolve_source(media_url, browser_cookies=browser_cookies))
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not resolve remote source {media_url}: {exc}") from exc

            output_path = os.path.join(
                download_path,
                str(media_path) +
                (".mp4" if media_type == "video" else ".mp3")
            )
            if os.path.exists(output_path):
                if messagebox.askyesno(
                    title="Media Already Exists",
                    message=f"The media '{media_path}' already exists in the download directory. Would you like to use the existing file? If not, the media will be redownloaded and overwrite the existing file."""
                ):
                    self.uploaded_videos[i].set_path(output_path)
                    self.uploaded_videos[i].set_is_url(False)
                    print(f"{Fore.GREEN}Done!")
                    continue

            if media_type == 'video':
                success, result = download_video(
                    media_url, media_path, download_path, self.max_quality.get(), self.max_download_speed.get(), self.final_bar,
                    browser_cookies=browser_cookies)
                if success:
                    if result:
                        self.uploaded_videos[i].set_path(result)
                        self.uploaded_videos[i].set_is_url(False)
                    else:
                        indices_to_delete.append(i)
                        print(f"{Fore.YELLOW}No video found, skipping")
                else:
                    raise Exception(
                        f"Failed to download {media_path}: {result}\nPress 'Process' again and it should start from where you left off.")
            elif media_type == 'audio':
                success, result = download_audio(
                    media_url, media_path, download_path, self.max_download_speed.get(), self.final_bar,
                    browser_cookies=browser_cookies)
                if success:
                    self.uploaded_videos[i].set_path(result)
                    self.uploaded_videos[i].set_is_url(False)
                else:
                    raise Exception(
                        f"Failed to download {media_path}: {result}\nPress 'Process' again and it should start from where you left off.")

            print(f"{Fore.GREEN}Done!")

        for idx in reversed(indices_to_delete):
            del self.uploaded_videos[idx]
        self.update_listbox()

    def process_videos(self):
        self.clear_transfer_progress()
        self.disable_objects()
        self.final_bar.reset_total_progress(1)
        cleanup_temp_children()

        self.reset_preferences_to_file()

        try:
            precision = self.precision.get()
            block_size = self.block_size.get()
            threshold = self.threshold.get()
            selected_model = os.path.join(self.models_dir, self.model.get())
            if not os.path.isfile(selected_model):
                available = [f for f in os.listdir(self.models_dir) if f.endswith('.onnx')]
                if not available:
                    raise Exception(f"No models found in directory {self.models_dir}")
                self.model.set(available[0])
                selected_model = os.path.join(self.models_dir, available[0])
            merge_clips = self.merge_clips.get()
            combine = self.combine_vids.get()
            normalize = self.normalize_audio.get()
            save_timestamps = self.save_txt.get()

            # Parse focus index from dropdown
            focus_idx = self.focus_idx.get()  # user-configurable sound class

            # Get model location if in a compiled app
            selected_model = get_bundle_filepath(selected_model)
            use_gpu = self.use_gpu.get()
            cache_store = prepare_remote_cache_store(self.remote_cache_store)
            shared_session = create_inference_session(selected_model, use_gpu=use_gpu)

            self.stdout_text["state"] = tk.NORMAL
            self.stdout_text.delete("1.0", tk.END)
            self.stdout_text["state"] = tk.DISABLED
            self.root.update_idletasks()
            log_shared_session_providers(shared_session)




            if not self.uploaded_videos:
                raise Exception("Please pick some videos to compile.")

            if not self.output_video_path.get() or self.output_video_path.get() == "No location selected!":
                raise Exception("Please specify an output location.")

            output_video_path = self.output_video_path.get()

            dict_list = []
            incomplete_failures = []
            # filename -> [(s,e)]：被排除的片段（Review 取消勾选 / Strict FP 丢弃），
            # 传给 compile_vid 防止 merge 桥接把已删片段带回成片
            excluded = {}

            if combine and os.path.exists(output_video_path):
                if not messagebox.askyesno("Confirm Overwrite",
                                           f"Output file \'{output_video_path}\' already exists and will be overwritten. Would you like to continue?"):
                    raise (Exception("Operation cancelled."))

            if not combine:
                for video in self.uploaded_videos:
                    video = video.get_path()
                    print(video)
                    temp = str(video.split('/')[-1]).rsplit('.', 1)
                    temp = '.'.join(temp[:-1])
                    temp = str(output_video_path + '/' + temp + "_comped.mp4")
                    if os.path.exists(temp):
                        if not messagebox.askyesno("Confirm Overwrite",
                                                   f"Output file \'{video}\' already exists and will be overwritten. Would you like to continue?"):
                            raise (Exception("Operation cancelled."))

            resolve_remote, audio_cache, full_download = remote_mode_actions(self.remote_mode.get())
            detection_size = detection_block_size(self.remote_mode.get(), block_size)
            if self.remote_mode.get() == "Remote Stream":
                print(f"Remote Stream block size: {detection_size}s")
            browser_cookies = browser_cookie_setting_value(
                self.remote_browser_cookies.get(),
                cookies_file=self.remote_cookies_file.get(),
            )
            preflight_failure = preflight_cookie_source(browser_cookies)
            if preflight_failure:
                print(f"{Fore.YELLOW}Cookie preflight warning: {preflight_failure}")
            max_height = convert_quality_str_to_int(self.max_quality.get())
            _resolve_limiter = ResolveLimiter()
            refresh_func = LimitedRefresher(
                lambda source: refresh_remote_source(
                    source, browser_cookies=browser_cookies
                ),
                limiter=_resolve_limiter,
                retries=3,
                logger=print,
            )
            audio_cache_paths = {}
            resolved_local_entries = []
            remote_entries = []
            remote_failures = []
            if any(x.get_is_url() for x in self.uploaded_videos) and full_download:
                self.handle_url_downloads()
            elif any(x.get_is_url() for x in self.uploaded_videos) and resolve_remote and not audio_cache:
                resolved_local_entries, remote_entries = resolve_remote_uploads(
                    self.uploaded_videos, browser_cookies=browser_cookies,
                    max_height=max_height, limiter=_resolve_limiter, logger=print,
                )
            elif any(x.get_is_url() for x in self.uploaded_videos) and audio_cache:
                resolved_local_entries, remote_entries = resolve_remote_uploads(
                    self.uploaded_videos, browser_cookies=browser_cookies,
                    max_height=max_height, limiter=_resolve_limiter, logger=print,
                )
                failed_remote_ids = set()
                for cache_index, (upload, source) in enumerate(remote_entries):
                    try:
                        # 逐源选择 audio candidate：只在真正缓存该源前探测，
                        # 避免开跑前对所有源一次性探测导致排尾 URL 过期。
                        try:
                            select_audio_candidate(source, log_func=print)
                        except Exception as exc:
                            print(f"{Fore.YELLOW}  Audio candidate probe skipped: {type(exc).__name__}")
                        audio_cache_paths[stable_source_id(source)] = fetch_audio_cache(
                            source, cache_store, log_func=print, refresh_func=refresh_func,
                            progress_callback=lambda sample, source=source, cache_index=cache_index:
                                self._queue_transfer_progress(
                                    sample,
                                    f"Audio Cache [{cache_index + 1}/{len(remote_entries)}] "
                                    f"{get_source_display_name(source)}",
                                ),
                        )
                    except Exception as exc:
                        message = (
                            f"Could not cache audio for {get_source_display_name(source)}: {exc}"
                        )
                        remote_failures.append(message)
                        failed_remote_ids.add(stable_source_id(source))
                        print(f"{Fore.YELLOW}Remote VOD skipped: {message}")
                remote_entries = [
                    (upload, source) for upload, source in remote_entries
                    if stable_source_id(source) not in failed_remote_ids
                ]

                # 上一轮某些远程源检测失败（或整批只跑到一半被取消）：本次如果重新
                # 解析到了这些源且 detection cache 里有失败标记，就询问是否重试。
                if remote_entries:
                    previously_failed = []
                    for upload, source in remote_entries:
                        if source is None:
                            continue
                        try:
                            if cache_store.has_detection_failure(
                                    *_detection_cache_args(
                                        source, selected_model, precision,
                                        detection_block_size(self.remote_mode.get(), block_size, True),
                                        threshold, focus_idx,
                                    )):
                                previously_failed.append(source)
                        except Exception:
                            continue
                    if previously_failed:
                        retry_choice = messagebox.askyesno(
                            "Retry Failed Sources",
                            f"{len(previously_failed)} remote source(s) failed detection in a "
                            f"previous run and are now resolved again:\n\n"
                            + "\n".join(f"  - {get_source_display_name(s)}" for s in previously_failed[:8])
                            + ("\n  ..." if len(previously_failed) > 8 else "")
                            + "\n\nRe-detect them now? (Yes re-runs detection for these; "
                              "No skips them this run.)",
                        )
                        if retry_choice:
                            for source in previously_failed:
                                try:
                                    cache_store.clear_detection_failure(
                                        *_detection_cache_args(
                                            source, selected_model, precision,
                                            detection_block_size(self.remote_mode.get(), block_size, True),
                                            threshold, focus_idx,
                                        )
                                    )
                                except Exception:
                                    pass
                            print(
                                f"{Fore.CYAN}Re-detecting {len(previously_failed)} previously "
                                f"failed remote source(s).")
                        else:
                            drop_ids = {stable_source_id(s) for s in previously_failed}
                            remote_entries = [
                                (u, s) for u, s in remote_entries
                                if s is None or stable_source_id(s) not in drop_ids
                            ]
                            print(
                                f"{Fore.YELLOW}Skipping {len(drop_ids)} previously failed "
                                f"remote source(s) this run.")


            res = ()
            if self.use_custom_resolution.get():
                res = (
                    self.custom_resolution_width_var.get(),
                    self.custom_resolution_height_var.get()
                )
            else:
                res = None

            padding = ()
            if self.use_custom_padding.get():
                padding = (
                    self.custom_padding_before.get(),
                    self.custom_padding_after.get()
                )
            else:
                padding = None

            self.clear_transfer_progress("Transfers complete; getting timestamps...")

            # --- Check for existing timestamps.txt ---
            # 候选路径与保存逻辑一致：设置里的路径优先，其次是默认位置
            # （<输出目录>/timestamps.txt）——否则未设置 txt 路径时上次保存的
            # timestamps.txt 永远找不到，用户会被迫重跑检测
            txt_candidates = []
            _cfg_txt = self.output_text_path.get()
            if _cfg_txt and _cfg_txt != "No file selected!":
                txt_candidates.append(_cfg_txt)
            if os.path.isdir(output_video_path):
                txt_candidates.append(os.path.join(output_video_path, "timestamps.txt"))
            else:
                txt_candidates.append(os.path.join(os.path.dirname(output_video_path), "timestamps.txt"))
            txt_path = next((p for p in txt_candidates if os.path.exists(p)), None)
            if txt_path:
                auto_use = self.skip_detection_auto.get()
                if auto_use or messagebox.askyesno(
                    "Skip Detection",
                    f"Found existing timestamps file:\n{txt_path}\n\nSkip AI detection and use saved timestamps directly?"
                ):
                    print(f"{Fore.GREEN}Loading timestamps from {txt_path}...")
                    with_videos, _ = _parse_timestamps_txt(txt_path)

                    # Build basename -> [paths] map (handle duplicates)
                    basename_map = {}
                    remote_name_map = {}
                    for v in self.uploaded_videos:
                        if v.get_source() is not None:
                            remote_name_map[v.get_source().source_url] = v.get_source()
                            continue
                        base = os.path.basename(v.get_path())
                        basename_map.setdefault(base, []).append(v.get_path())

                    dict_list = []
                    loaded = 0
                    for entry in with_videos:
                        if entry['filename'] in remote_name_map:
                            entry['filename'] = remote_name_map[entry['filename']]
                            dict_list.append(entry)
                            loaded += 1
                            continue
                        base = os.path.basename(entry['filename'])
                        candidates = basename_map.get(base, [])
                        if len(candidates) == 1:
                            # Unique basename: direct match
                            entry['filename'] = candidates[0]
                            dict_list.append(entry)
                            loaded += 1
                        elif len(candidates) > 1:
                            # Duplicate basename: try exact path match first
                            entry_norm = os.path.normpath(entry['filename'])
                            matched = False
                            for c in candidates:
                                if os.path.normpath(c) == entry_norm:
                                    entry['filename'] = c
                                    dict_list.append(entry)
                                    loaded += 1
                                    matched = True
                                    break
                            if not matched:
                                # Fallback: use first candidate
                                print(f"{Fore.YELLOW}  Ambiguous: {base} (using first match)")
                                entry['filename'] = candidates[0]
                                dict_list.append(entry)
                                loaded += 1
                        else:
                            print(f"{Fore.YELLOW}  Not in list: {base}")

                    print(f"{Fore.GREEN}Loaded {loaded} of {len(with_videos)} video(s).")
                    if loaded == 0:
                        raise Exception("No videos from timestamps matched the current list.")
                    _uploaded_remote = sum(
                        1 for v in self.uploaded_videos if v.get_is_url())
                    if _uploaded_remote > loaded:
                        print(
                            f"{Fore.YELLOW}WARNING: {_uploaded_remote - loaded} remote source(s) "
                            f"in the current list have no timestamp data in {txt_path}. "
                            f"They were likely skipped in the earlier run. If you want to recover "
                            f"them, cancel and re-run WITHOUT skipping detection, so the missed "
                            f"sources are re-detected.")

                    total_progress = ((4 if combine else 2) if self.is_video else (2 if combine else 1)) * (loaded + (1 if combine and loaded > 1 else 0)) * 100
                    self.final_bar.reset_total_progress(total_progress)

                    print(f"Compiling and writing to {output_video_path.split('/')[-1]}...")
                    if self.use_verify.get():
                        print(f"{Fore.CYAN}Running verification scan...")
                        dict_list = _verify_and_expand(
                            dict_list, selected_model,
                            window=self.verify_window_var.get(),
                            focus_idx=focus_idx,
                            logger=self.final_bar,
                            threshold=threshold * 0.6,
                            use_gpu=use_gpu, ort_session=shared_session,
                            cache_store=cache_store, refresh_func=refresh_func,
                            audio_cache_paths=audio_cache_paths,
                            progress_callback=self._queue_transfer_progress)
                    # auto-deselect originals fully covered by new reverify clips
                    for entry in dict_list:
                        ts = entry.get('timestamps', [])
                        news = [t for t in ts if t.get('source') == 'new']
                        if not news:
                            continue
                        filtered = []
                        for t in ts:
                            if t.get('source') == 'original':
                                covered = any(g['start'] <= t['start'] and g['end'] >= t['end'] for g in news)
                                if covered:
                                    continue
                            filtered.append(t)
                        entry['timestamps'] = filtered
                    pre_review = {source_key(e['filename']): [(t['start'], t['end']) for t in e.get('timestamps', [])]
                                  for e in dict_list}
                    if self.use_review.get():
                        dlg = ReviewDialog(self.root, dict_list, padding,
                                          output_video_path,
                                          use_verify=self.use_verify.get(),
                                          txt_path=self.output_text_path.get(),
                                          cache_store=cache_store)
                        if dlg.result is None:
                            print(f"{Fore.YELLOW}Review cancelled.")
                            self.reenable_disabled_objects()
                            return
                        if not dlg.result:
                            raise Exception("No segments selected after review.")
                        dict_list = dlg.result
                        for entry in dict_list:
                            fn = entry['filename']
                            sel = {(round(t['start'], 3), round(t['end'], 3)) for t in entry['timestamps']}
                            removed = [iv for iv in pre_review.get(source_key(fn), [])
                                       if (round(iv[0], 3), round(iv[1], 3)) not in sel]
                            if removed:
                                excluded.setdefault(source_key(fn), []).extend(removed)
                        _save_selected_txt(dict_list, txt_path)
                    ensure_temp_dir()
                    remote_temp = tempfile.mkdtemp(dir=TEMP_DIR, prefix='remote-compile-')
                    remote_failures = []
                    # 检测可能已耗时 1-3 小时：compile 前对所有远程源做一次前瞻刷新，
                    # 保证 materialize 使用的 video_url/audio_url 是新鲜的（即使 detection 命中了缓存）。
                    for entry in dict_list:
                        source = entry.get('filename')
                        if not isinstance(source, MediaSource):
                            continue
                        source_resolved_at = getattr(source, "resolved_at", None)
                        if (source_resolved_at is not None
                                and time.monotonic() - source_resolved_at > REMOTE_REFRESH_THRESHOLD):
                            print(
                                f"{Fore.CYAN}Refreshing stale remote source before compile for "
                                f"{get_source_display_name(source)}...")
                            try:
                                updated = refresh_func(source)
                                if isinstance(updated, MediaSource) and updated is not source:
                                    source.__dict__.update(updated.__dict__)
                            except Exception as exc:
                                print(f"{Fore.YELLOW}  Source refresh failed ({exc}); using existing URL.")
                    try:
                        self.clear_transfer_progress("Preparing remote clips...")
                        compile_entries = materialize_remote_entries(
                            dict_list, remote_temp, cache_store=cache_store, padding=padding,
                            is_video=self.is_video, failures=remote_failures,
                            refresh_func=refresh_func,
                            progress_callback=self._show_remote_clip_progress,
                            max_parallel=self.remote_download_concurrency.get())
                        for failure in remote_failures:
                            print(f"{Fore.YELLOW}Remote VOD skipped: {failure}")
                        if not compile_entries:
                            raise RuntimeError("No compile-ready VOD results remain after remote fetch failures.")
                        self.clear_transfer_progress("Starting compile...")
                        compile_vid(compile_entries, output_video_path, merge_clips,
                                    combine, res, self.final_bar, normalize,
                                    self.is_video, None, excluded=excluded,
                                    progress_callback=lambda sample: self._queue_transfer_progress(
                                        sample, "Compile"))
                    except Exception as exc:
                        raise Exception(f"Remote segment materialization/compile failed: {exc}") from exc
                    finally:
                        shutil.rmtree(remote_temp, ignore_errors=True)
                    print(f"{Fore.GREEN}Wrote final video to {output_video_path.split('/')[-1]}.")
                    messagebox.showinfo("Info", f"Video(s) exported to {output_video_path}. Enjoy!")
                    print(f"{Fore.GREEN}SUCCESS!")
                    self.reenable_disabled_objects()
                    return

            try:
                vids_with_clips = 0
                self.final_bar.reset_total_progress(
                    (len(self.uploaded_videos) * 100 * 2))

                if resolve_remote:
                    processing_uploads = processing_uploads_for_batch(
                        self.uploaded_videos, resolved_local_entries, remote_entries)
                else:
                    processing_uploads = list(self.uploaded_videos)
                def run_detection(upload, i):
                    """Run detection for one upload; returns True on success."""
                    nonlocal vids_with_clips
                    input_video_path = remote_detection_input(
                        upload, self.remote_mode.get(), audio_cache_paths
                    )
                    current_block_size = detection_block_size(
                        self.remote_mode.get(), block_size,
                        isinstance(input_video_path, MediaSource),
                    )
                    print(
                        f"{Fore.GREEN}[{i + 1}/{len(processing_uploads)}]{Style.RESET_ALL} Getting timestamps for "
                        f"{get_source_display_name(input_video_path)}")
                    save_audio_path = None
                    if (self.use_verify.get()
                            and self.remote_mode.get() == "Remote Stream"
                            and isinstance(input_video_path, MediaSource)):
                        ensure_temp_dir()
                        _fd, save_audio_path = tempfile.mkstemp(
                            suffix=".wav", prefix="reverify-stream-", dir=TEMP_DIR)
                        os.close(_fd)
                        os.remove(save_audio_path)
                    if isinstance(input_video_path, MediaSource):
                        # 超大批次在开头一次性解析所有 URL，排到后面的源可能早已过期：
                        # 检测前若解析已超阈值则重新 resolve 一次新鲜 URL。
                        source_resolved_at = getattr(input_video_path, "resolved_at", None)
                        if (source_resolved_at is not None
                                and time.monotonic() - source_resolved_at > REMOTE_REFRESH_THRESHOLD):
                            print(
                                f"{Fore.CYAN}Refreshing stale remote source for "
                                f"{get_source_display_name(input_video_path)}...")
                            try:
                                updated = refresh_func(input_video_path)
                                if isinstance(updated, MediaSource) and updated is not input_video_path:
                                    input_video_path.__dict__.update(updated.__dict__)
                            except Exception as exc:
                                print(f"{Fore.YELLOW}  Source refresh failed ({exc}); using existing URL.")
                        # 逐源选择 audio candidate：只在真正检测该源前探测，
                        # 保证用的是最新解析的候选池和当时的网络状况，且 audio_url 只在读取前一刻钉死。
                        try:
                            select_audio_candidate(input_video_path, log_func=print)
                        except Exception as exc:
                            print(f"{Fore.YELLOW}  Audio candidate probe skipped: {type(exc).__name__}")
                    try:
                        timestamps, used_existing_data = get_timestamps(
                            input_video_path, precision, current_block_size, threshold, focus_idx, selected_model, self.final_bar,
                            use_gpu=use_gpu, ort_session=shared_session, cache_store=cache_store,
                            refresh_func=refresh_func, save_audio_path=save_audio_path,
                            progress_callback=lambda sample, upload=upload, i=i:
                                self._queue_transfer_progress(
                                    sample,
                                    f"Remote Stream [{i + 1}/{len(processing_uploads)}] "
                                    f"{get_source_display_name(upload)}",
                                ),
                        )
                    except RemoteAudioIncompleteError as exc:
                        failure = f"{get_source_display_name(input_video_path)}: {exc}"
                        incomplete_failures.append(failure)
                        print(f"{Fore.YELLOW}Remote VOD skipped: {failure}")
                        _mark_detection_failure(cache_store, input_video_path,
                                                 precision, current_block_size, threshold,
                                                 focus_idx, selected_model, str(exc))
                        return False
                    except RemoteAudioStallError as exc:
                        failure = f"{get_source_display_name(input_video_path)}: {exc}"
                        incomplete_failures.append(failure)
                        print(f"{Fore.YELLOW}Remote VOD skipped (stalled): {failure}")
                        _mark_detection_failure(cache_store, input_video_path,
                                                 precision, current_block_size, threshold,
                                                 focus_idx, selected_model, str(exc))
                        return False
                    except Exception as exc:
                        if isinstance(input_video_path, MediaSource):
                            failure = f"{get_source_display_name(input_video_path)}: {exc}"
                            incomplete_failures.append(failure)
                            print(f"{Fore.YELLOW}Remote VOD skipped: {failure}")
                            _mark_detection_failure(cache_store, input_video_path,
                                                     precision, current_block_size, threshold,
                                                     focus_idx, selected_model, str(exc))
                            return False
                        raise
                    if (save_audio_path and os.path.isfile(save_audio_path)
                            and isinstance(input_video_path, MediaSource)):
                        audio_cache_paths[stable_source_id(input_video_path)] = save_audio_path
                    timestamps = {'filename': timestamps['filename'],
                                  'timestamps': [dict(t) for t in timestamps['timestamps']]}
                    source_for_result = upload.get_source() if audio_cache and upload.get_is_url() else input_video_path
                    if isinstance(source_for_result, MediaSource):
                        timestamps = preserve_remote_result(timestamps, source_for_result)
                    if self.use_strict_fp.get():
                        before = len(timestamps['timestamps'])
                        dropped_ts = [t for t in timestamps['timestamps'] if t.get('suspect')]
                        if dropped_ts:
                            excluded.setdefault(source_key(timestamps['filename']), []).extend(
                                (t['start'], t['end']) for t in dropped_ts)
                        timestamps['timestamps'] = [
                            t for t in timestamps['timestamps'] if not t.get('suspect')]
                        dropped = before - len(timestamps['timestamps'])
                        if dropped:
                            print(f"{Fore.CYAN}Strict FP filter: dropped {dropped} suspect clip(s).")
                    dict_list.append(timestamps)
                    if isinstance(input_video_path, MediaSource):
                        try:
                            cache_store.clear_detection_failure(
                                *_detection_cache_args(
                                    input_video_path, selected_model, precision,
                                    current_block_size, threshold, focus_idx,
                                )
                            )
                        except Exception:
                            pass
                    if used_existing_data: print(f"{Fore.GREEN}Using existing timestamp data from previous run.")
                    num_found = len(timestamps['timestamps'])
                    if num_found > 1:
                        print(
                            f"{Fore.GREEN}Found {len(timestamps['timestamps'])} clips.")
                        vids_with_clips += 1
                    elif num_found == 1:
                        print(
                            f"{Fore.GREEN}Found 1 clip.")
                        vids_with_clips += 1
                    else:
                        print(
                            f"{Fore.YELLOW}Could not find any clips.")
                    return True

                failed_uploads = []
                for i, upload in enumerate(processing_uploads):
                    ok = run_detection(upload, i)
                    if not ok and upload.get_is_url():
                        failed_uploads.append(upload)

                # 第二遍：网络/限流是暂时性的，等退避后仅对失败的远程源重试一次，
                # 避免整批最后直接跳过本来能恢复的源。
                if failed_uploads:
                    print(
                        f"{Fore.CYAN}Retrying {len(failed_uploads)} failed remote source(s) "
                        f"after a short pause...")
                    time.sleep(10)
                    recovered = []
                    for retry_index, upload in enumerate(failed_uploads):
                        ok = run_detection(upload, len(processing_uploads) + retry_index)
                        if not ok:
                            print(
                                f"{Fore.YELLOW}  Retry still failed: "
                                f"{get_source_display_name(upload)}")
                        else:
                            recovered.append(get_source_display_name(upload))
                    if recovered:
                        print(
                            f"{Fore.GREEN}Retry pass recovered {len(recovered)} previously "
                            f"failed source(s); remaining failures listed below.")

                if incomplete_failures:
                    print(
                        f"{Fore.YELLOW}{len(incomplete_failures)} remote VOD(s) were skipped:"
                    )
                    for failure in incomplete_failures:
                        print(f"{Fore.YELLOW}  - {failure}")

                if not dict_list:
                    raise Exception(
                        "No timestamps found for any input media. "
                        "Every remote source failed to resolve/detect (see the skipped VOD list above)."
                    )

                # Set values for progress bar
                # If saving individually, or there is only one video
                if not combine or vids_with_clips == 1:
                    if self.is_video:
                        total_progress = 4 * vids_with_clips * 100
                    else:
                        total_progress = 2 * vids_with_clips * 100

                    self.final_bar.reset_total_progress(total_progress)
                else:
                    if self.is_video:
                        total_progress = 4 * (vids_with_clips + 1) * 100
                    else:
                        total_progress = 2 * (vids_with_clips + 1) * 100

                self.final_bar.reset_total_progress(total_progress)

                # Save txt file with timestamp info
                if save_timestamps:
                    try:
                        if self.output_text_path.get() != "No file selected!":
                            txt_path = self.output_text_path.get()
                            # 防御：如果保存路径的目录不存在（比如从别的 PC 继承的配置），回退
                            txt_dir = os.path.dirname(txt_path) or '.'
                            if not os.path.isdir(txt_dir):
                                print(f"{Fore.YELLOW}Configured output directory missing: {txt_dir}")
                                print(f"{Fore.YELLOW}Falling back to output video location.")
                                txt_path = os.path.join(os.path.dirname(output_video_path), "timestamps.txt")
                        elif os.path.isdir(output_video_path):
                            txt_path = os.path.join(
                                output_video_path, "timestamps.txt")
                        else:
                            txt_path = os.path.join(os.path.dirname(
                                output_video_path), "timestamps.txt")

                        timestamps_text = ""
                        found_timestamps = False
                        for file in dict_list:
                            timestamps_text += f"{get_source_persistence_name(file['filename'])}\n"

                            for ts in file['timestamps']:
                                timestamps_text += f"{convert_seconds_to_timestamp(ts['start'])} - {convert_seconds_to_timestamp(ts['end'])}, confidence: {ts['pred']}\n"
                                found_timestamps = True

                            timestamps_text += "\n"

                        if found_timestamps:
                            with open(txt_path, 'w', encoding="utf-8") as file:
                                file.write(timestamps_text)
                            print(
                                f"{Fore.GREEN}Saved timestamps to {txt_path}!")
                    except:
                        raise

                # --- re-verify and/or review before final compile ---
                if self.use_verify.get():
                    print(f"{Fore.CYAN}Running verification scan...")
                    dict_list = _verify_and_expand(
                        dict_list, selected_model,
                        window=self.verify_window_var.get(),
                        focus_idx=focus_idx,
                        logger=self.final_bar,
                        threshold=threshold * 0.6,
                        use_gpu=use_gpu, ort_session=shared_session,
                        cache_store=cache_store, refresh_func=refresh_func,
                        audio_cache_paths=audio_cache_paths,
                        progress_callback=self._queue_transfer_progress)
                # auto-deselect originals fully covered by new reverify clips
                for entry in dict_list:
                    ts = entry.get('timestamps', [])
                    news = [t for t in ts if t.get('source') == 'new']
                    if not news:
                        continue
                    filtered = []
                    for t in ts:
                        if t.get('source') == 'original':
                            covered = any(g['start'] <= t['start'] and g['end'] >= t['end'] for g in news)
                            if covered:
                                continue
                        filtered.append(t)
                    entry['timestamps'] = filtered
                pre_review = {source_key(e['filename']): [(t['start'], t['end']) for t in e.get('timestamps', [])]
                              for e in dict_list}
                if self.use_review.get():
                    dlg = ReviewDialog(self.root, dict_list, padding,
                                      output_video_path,
                                      use_verify=self.use_verify.get(),
                                      txt_path=txt_path,
                                      cache_store=cache_store)
                    if dlg.result is None:
                        print(f"{Fore.YELLOW}Review cancelled.")
                        self.reenable_disabled_objects()
                        return
                    if not dlg.result:
                        raise Exception("No segments selected after review.")
                    dict_list = dlg.result
                    for entry in dict_list:
                        fn = entry['filename']
                        sel = {(round(t['start'], 3), round(t['end'], 3)) for t in entry['timestamps']}
                        removed = [iv for iv in pre_review.get(source_key(fn), [])
                                   if (round(iv[0], 3), round(iv[1], 3)) not in sel]
                        if removed:
                            excluded.setdefault(source_key(fn), []).extend(removed)
                    # 保存 _selected.txt（仅勾选的片段）
                    _save_selected_txt(dict_list, txt_path)

                print(
                    f"Compiling and writing to {output_video_path.split('/')[-1]}...")
                ensure_temp_dir()
                remote_temp = tempfile.mkdtemp(dir=TEMP_DIR, prefix='remote-compile-')
                remote_failures = []
                try:
                    self.clear_transfer_progress("Preparing remote clips...")
                    compile_entries = materialize_remote_entries(
                            dict_list, remote_temp, cache_store=cache_store, padding=padding,
                            is_video=self.is_video, failures=remote_failures,
                            progress_callback=self._show_remote_clip_progress,
                            max_parallel=self.remote_download_concurrency.get())
                    for failure in remote_failures:
                        print(f"{Fore.YELLOW}Remote VOD skipped: {failure}")
                    if not compile_entries:
                        raise RuntimeError("No compile-ready VOD results remain after remote fetch failures.")
                    self.clear_transfer_progress("Starting compile...")
                    compile_vid(compile_entries, output_video_path, merge_clips,
                                combine, res, self.final_bar, normalize,
                                self.is_video, None, excluded=excluded,
                                progress_callback=lambda sample: self._queue_transfer_progress(
                                    sample, "Compile"))
                except Exception as exc:
                    raise Exception(f"Remote segment materialization/compile failed: {exc}") from exc
                finally:
                    shutil.rmtree(remote_temp, ignore_errors=True)
                print(
                    f"{Fore.GREEN}Wrote final video to {output_video_path.split('/')[-1]}.")
                messagebox.showinfo(
                    "Info", f"Video(s) exported to {output_video_path}. Enjoy!")
            except Exception as e:
                raise Exception(
                    "Encountered error during video processing: " + str(e))

            print(f"{Fore.GREEN}SUCCESS!")

            try:
                cleanup_temp_children()
            except:
                # Sometimes deleting the temp dir can fail
                # OSes will auto-delete this directory anyway
                # so this isn't a huge problem
                pass

            if not self.keep_downloaded_vids.get():
                self.remove_urls_from_list()

            self.root.update_idletasks()
            self.clear_transfer_progress("Completed")
            self.reenable_disabled_objects()

        except Exception as e:
            messagebox.showerror("Error", e)
            print(f"\n{Fore.RED}FAILURE: " + str(e))
            cleanup_temp_children()
            self.clear_transfer_progress("Failed")
            self.reenable_disabled_objects()
            return


class StdoutRedirector:

    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root
        self.queue = queue.Queue()
        self.text_widget["state"] = tk.DISABLED
        self.root.after(100, self._poll)

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        return

    def _poll(self):
        # 每轮最多渲染 120 条，突发日志（如 queuing/done 批量打印）分摊到
        # 多个 poll 周期，避免主线程一次性大插入导致 UI 卡顿。
        rendered = 0
        try:
            while rendered < 120:
                text = self.queue.get_nowait()
                try:
                    self._render(text)
                    rendered += 1
                except Exception:
                    pass
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _render(self, text):
        # 仅在用户停留在底部时自动滚动，长日志每次渲染不再强制全量重排
        at_bottom = self.text_widget.yview()[1] >= 0.999
        self.text_widget["state"] = tk.NORMAL

        # Check/apply colorama colors
        # This is the worst code ever written but hey it works
        if Fore.RED in text:
            text = text.replace(Fore.RED, "")
            self.text_widget.tag_configure("red", foreground="red")
            self.text_widget.insert(tk.END, text, "red")
        elif Fore.YELLOW in text:
            text = text.replace(Fore.YELLOW, "")
            self.text_widget.tag_configure("yellow", foreground="yellow")
            self.text_widget.insert(tk.END, text, "yellow")
        elif Fore.GREEN in text:
            r, g, b = 144, 238, 144
            light_green = f"#{r:02x}{g:02x}{b:02x}"
            text = text.replace(Fore.GREEN, "")

            if Style.RESET_ALL in text:
                middle_index = text.find(Style.RESET_ALL)

                text = text.replace(Style.RESET_ALL, "")

                self.text_widget.insert(tk.END, text)

                start_index = self.text_widget.index("end-1c linestart")
                middle_index = self.text_widget.index(
                    f"{start_index}+{middle_index}c")

                self.text_widget.tag_configure(
                    light_green, foreground=light_green)
                self.text_widget.tag_configure(
                    "white", foreground="white")

                self.text_widget.tag_add(
                    light_green, start_index, middle_index)
                self.text_widget.tag_add("white", middle_index, tk.END)
            else:
                self.text_widget.tag_configure(
                    light_green, foreground=light_green)
                self.text_widget.insert(tk.END, text, light_green)
        else:
            self.text_widget.insert(tk.END, text)

        if at_bottom:
            self.text_widget.see(tk.END)  # Scroll to the end of the text
        self.text_widget["state"] = tk.DISABLED


class FinalRenderBar(ProgressBarLogger):
    def __init__(self, ui, init_state=None, bars=None, ignored_bars=None,
                 logged_bars='all', min_time_interval=0, ignore_bars_under=0,
                 progress_callback=None):
        self.ui = ui
        self.progress_callback = progress_callback
        self.reset_total_progress(100)

        super().__init__(init_state, bars, ignored_bars,
                         logged_bars, min_time_interval, ignore_bars_under)

    def set_current_progress(self, current_progress):
        self.current_progress = current_progress

    def reset_total_progress(self, max_value):
        self.max_value = max_value
        self.current_progress = 0
        self.total_progress = 0

        self.ui['value'] = self.total_progress
        self.ui['maximum'] = self.max_value

    def callback(self, **changes):
        for (parameter, value) in changes.items():
            # print ('Parameter %s is now %s' % (parameter, value))
            return

    # Normal proglog callback
    def bars_callback(self, bar, attr, value, old_value=None):
        self.current_progress = (value / self.bars[bar]['total']) * 100

        if self.current_progress >= 100:
            self.total_progress += self.current_progress
            self.current_progress = 0

        self.ui['value'] = self.total_progress + self.current_progress

    # YT-DLP progress hook stuff
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def hook(self, d):
        if d.get('status') == 'downloading':
            current = d.get('downloaded_bytes') or 0
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            elapsed = d.get('elapsed') or 0
            if total:
                self.current_progress = min(100.0, float(current) / float(total) * 100)
            else:
                percent_str = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_percent_str', ''))
                try:
                    self.current_progress = float(percent_str.strip('%'))
                except ValueError:
                    pass
            if self.progress_callback is not None:
                sample = format_transfer_progress(current, total, elapsed)
                self.progress_callback(sample, "Full Download")
            else:
                self.ui['value'] = self.current_progress
        elif d.get('status') in ('finished', 'error') and self.progress_callback is not None:
            self.progress_callback({
                "current": 1 if d.get('status') == 'finished' else 0, "total": 1,
                "percent": 100 if d.get('status') == 'finished' else 0,
                "text": "Full Download completed" if d.get('status') == 'finished' else "Full Download failed",
            })


def main():
    root = TkinterDnD.Tk()
    sv_ttk.set_theme("dark")

    app = VideoProcessorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
