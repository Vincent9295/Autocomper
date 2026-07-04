import configparser
import os
import re
import shutil
import sys
import subprocess
import tempfile

# 必须在导入 moviepy 之前设置 FFmpeg 路径
_ffmpeg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg')
if sys.platform == 'win32':
    os.environ['FFMPEG_BINARY'] = os.path.join(_ffmpeg_dir, 'windows', 'ffmpeg.exe')
elif sys.platform == 'darwin':
    os.environ['FFMPEG_BINARY'] = os.path.join(_ffmpeg_dir, 'osx', 'ffmpeg')
else:
    os.environ['FFMPEG_BINARY'] = os.path.join(_ffmpeg_dir, 'linux', 'ffmpeg')

import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

import sv_ttk
from colorama import Fore, Style
from kthread import KThread
from PIL import Image, ImageTk
from proglog import ProgressBarLogger

from compile import compile_vid
from config import VERSION, REPO_URL
from custom_tooltip import CustomHovertip
from sound_reader import get_timestamps
from utils import (DOWNLOAD_QUALITY_OPTIONS, FFMPEG_PATH, MediaUpload,
                   download_audio, download_video, get_bundle_filepath,
                   get_number_of_vids_in_playlist, is_valid_yt_dlp_url)

VIDEO_INPUT = [("Video Files",  "*.mp4 *.avi *.mkv *.m4v *.mov")]
VIDEO_OUTPUT = [("Video Files", "*.mp4"), ("All Files", "*.*")]
AUDIO_INPUT = [("Audio Files",  "*.mp3 *.wav *.flac")]
AUDIO_OUTPUT = [("Audio Files", "*.mp3"), ("All Files", "*.*")]

DEFAULT_SETTINGS = {
    'keep_downloaded_vids': False,
    'download_path': "No location selected!",
    'max_quality': "No Limit",
    'max_download_speed': '0',
    'output_text_path': "No file selected!"
}

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


TEMP_DIR = tempfile.TemporaryDirectory().name
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




def _verify_and_expand(dict_list, selected_model, window=5.0,
                       precision=100, block_size=600, logger=None):
    """在每个原片段周围用更低的阈值重扫描，补充漏掉的声音片段。

    新发现的片段标记 source='new'，原始片段标记 source='original'。
    审核对话框中可凭此快速定位需要检查的新增片段。
    """
    if not dict_list:
        return dict_list

    # 预加载 ONNX 模型一次（所有窗口共享，省去每窗口 3-5s 的加载时间）
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    verify_session = ort.InferenceSession(selected_model, sess_options,
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    checked = 0
    new_found = 0

    for entry in dict_list:
        filename = entry['filename']
        original_ts = entry['timestamps']
        if not original_ts:
            continue

        # 标记所有原始片段
        for ts in original_ts:
            ts.setdefault('source', 'original')

        # 合并重叠的扫描窗口
        scan_windows = []
        for ts in original_ts:
            ws = max(0, ts['start'] - window)
            we = ts['end'] + window
            if scan_windows and ws <= scan_windows[-1][1] + 1:
                scan_windows[-1] = (scan_windows[-1][0], max(scan_windows[-1][1], we))
            else:
                scan_windows.append((ws, we))

        # 用 FFmpeg 提取每段短音频 → 低阈值重跑 ONNX
        merged_timestamps = list(original_ts)
        for ws, we in scan_windows:
            checked += 1
            try:
                tmp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp_audio.close()
                extract_cmd = [
                    os.environ.get('FFMPEG_BINARY', 'ffmpeg'),
                    '-y', '-hide_banner', '-loglevel', 'error',
                    '-ss', str(ws), '-t', str(we - ws),
                    '-i', filename,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '32000', '-ac', '1',
                    tmp_audio.name
                ]
                _opts = {}
                if sys.platform == 'win32':
                    _opts['creationflags'] = 0x08000000
                subprocess.run(extract_cmd, capture_output=True, timeout=30, **_opts)

                scan_result, _ = get_timestamps(
                    tmp_audio.name, precision=precision, block_size=block_size,
                    threshold=0.30, focus_idx=58, model=selected_model,
                    logger=logger, ort_session=verify_session
                )
                for ts in scan_result['timestamps']:
                    ts['start'] += ws
                    ts['end'] += ws
                    ts['source'] = 'new'
                    merged_timestamps.append(ts)
                    new_found += 1
            except Exception as e:
                print(f"{Fore.YELLOW}  Verify scan failed for [{ws}-{we}]: {e}")
            finally:
                try:
                    os.remove(tmp_audio.name)
                except Exception:
                    pass

        # 合并相邻/重叠片段（2s 内视为同一声事件）
        if merged_timestamps:
            merged_timestamps.sort(key=lambda x: x['start'])
            deduped = [merged_timestamps[0]]
            for ts in merged_timestamps[1:]:
                if ts['start'] <= deduped[-1]['end'] + 2.0:
                    deduped[-1]['end'] = max(deduped[-1]['end'], ts['end'])
                    deduped[-1]['pred'] = max(deduped[-1]['pred'], ts['pred'])
                else:
                    deduped.append(ts)
            entry['timestamps'] = deduped

    if checked > 0:
        print(f"{Fore.CYAN}Verification: scanned {checked} window(s), "
              f"found {new_found} additional segment(s).")

    return dict_list


class ReviewDialog:
    """片段审核对话框 —— Treeview + 音频/视频预览 + 勾选/取消。"""

    def __init__(self, parent, dict_list, padding, output_path,
                 use_verify=False, txt_path=None):
        self.parent = parent
        self.dict_list = dict_list
        self.padding = padding or (0, 0)
        self.output_path = output_path
        self.txt_path = txt_path
        self.result = None
        self.checks = []

        self.flat = []
        for entry in dict_list:
            fn = entry['filename']
            for ts in entry.get('timestamps', []):
                self.flat.append({
                    'filename': fn,
                    'start': ts['start'], 'end': ts['end'],
                    'pred': ts.get('pred', 0),
                    'source': ts.get('source', 'original'),
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
        t = f"Total: {total} segments"
        if use_verify and new_count > 0:
            t += f"  |  New: {new_count}"
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
            cv = tk.BooleanVar(value=True)
            self.checks.append(cv)
            s_str = self._fmt(f['start'])
            e_str = self._fmt(f['end'])
            bn = os.path.basename(f['filename'])
            st = 'New' if f.get('source') == 'new' else 'Original'
            tag = 'checked'
            self.tree.insert('', tk.END, iid=str(i),
                             values=('\u2611' if cv.get() else '\u2610',
                                     f"{s_str} - {e_str}", bn,
                                     f"{f['pred']:.2f}", st), tags=(tag,))
            self._style(str(i), cv.get())

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
        try:
            suf = '.mp4' if video else '.wav'
            tmp = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
            tmp.close()
            ff = os.environ.get('FFMPEG_BINARY', 'ffmpeg')
            if video:
                cmd = [ff, '-y', '-hide_banner', '-loglevel', 'error',
                       '-ss', str(ss), '-t', str(dur), '-i', f['filename'],
                       '-c:v', 'h264_nvenc', '-preset', '3',
                       '-c:a', 'aac', '-b:a', '128k', tmp.name]
            else:
                cmd = [ff, '-y', '-hide_banner', '-loglevel', 'error',
                       '-ss', str(ss), '-t', str(dur), '-i', f['filename'],
                       '-vn', '-acodec', 'pcm_s16le', '-ar', '32000', '-ac', '1',
                       tmp.name]
            _opts = {}
            if sys.platform == 'win32':
                _opts['creationflags'] = 0x08000000
            subprocess.run(cmd, capture_output=True, timeout=30, **_opts)
            if sys.platform == 'win32':
                os.startfile(tmp.name)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', tmp.name])
            else:
                subprocess.Popen(['xdg-open', tmp.name])
        except Exception as e:
            print(f"{Fore.YELLOW}Preview failed: {e}")

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
        self.result = result

        # 保存勾选的 timestamps 到 _selected.txt
        if self.txt_path and self.txt_path != "No file selected!":
            selected_path = self.txt_path.replace('.txt', '_selected.txt')
            lines = []
            for entry in result:
                lines.append(entry['filename'])
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

        self.win.destroy()

    def _on_cancel(self):
        self.result = None
        self.win.destroy()


class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Autocomper')

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Set initial window size
        self.root.geometry('1000x800')

        # Enforce minimum window size
        self.root.resizable(True, True)
        
        self.root.wm_minsize(875, 675)

        # Create a grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)

        # Left Column Widgets
        self.left_frame = ttk.Frame(root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=30, sticky="nsew")

        # Create a vertical separator
        separator = ttk.Separator(root, orient='vertical')
        separator.grid(row=0, column=1, sticky='ns')

        self.models_dir = "models/"
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
        self.model = tk.StringVar(value="bdetectionmodel_05_01_23.onnx")
        self.merge_clips = tk.BooleanVar(value=True)
        self.combine_vids = tk.BooleanVar(value=True)
        self.normalize_audio = tk.BooleanVar()

        self.keep_downloaded_vids = tk.BooleanVar(value=False)
        self.download_video_path = tk.StringVar()
        self.max_quality = tk.StringVar()
        self.max_download_speed = tk.IntVar()
        
        self.output_text_path = tk.StringVar()

        self.keep_downloaded_vids.set(bool(
            self.preferences.get("Settings", "keep_downloaded_vids")))

        self.download_video_path.set(
            self.preferences.get("Settings", "download_path"))

        self.max_quality.set(
            self.preferences.get("Settings", "max_quality"))
        
        self.max_download_speed.set(int(
            self.preferences.get("Settings", "max_download_speed")))

        self.output_text_path.set(
            self.preferences.get("Settings", "output_text_path"))

        # Create a list to store uploaded video file paths
        self.uploaded_videos = []

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

        self.remove_button = ttk.Button(
            self.filelist_buttons_frame, text="Remove Selected", command=self.remove_selected)
        self.clear_button = ttk.Button(
            self.filelist_buttons_frame, text="Clear All", command=self.clear_list)

        self.add_button.pack(pady=5, padx=1, side=tk.LEFT)
        self.up_arrow.pack(pady=5, padx=3, side=tk.LEFT)
        self.down_arrow.pack(pady=5, padx=3, side=tk.LEFT)

        self.clear_button.pack(pady=5, side=tk.RIGHT)
        self.remove_button.pack(pady=5, padx=5, side=tk.RIGHT)

        self.filelist_buttons_frame.pack(after=self.filelist_frame, fill=tk.X)

        ttk.Separator(self.left_frame, orient="horizontal").pack(
            fill=tk.X, pady=15)

        self.text_options_frame = ttk.Frame(self.left_frame)

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

        self.text_options_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        separator = ttk.Separator(self.left_frame, orient='vertical')
        separator.pack(side='left', fill='y', padx=(0, 15), pady=0)

        self.checkbox_frame = ttk.Frame(self.left_frame)
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

        self.checkbox_frame_three = ttk.Frame(self.left_frame)
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

        self.checkbox_frame_four = ttk.Frame(self.left_frame)
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
        self.checkbox_frame_five = ttk.Frame(self.left_frame)
        self.checkbox_frame_five.pack(anchor=tk.W)
        self.use_verify = tk.BooleanVar()
        self.verify_window_var = tk.DoubleVar(value=5.0)
        self.verify_checkbox = ttk.Checkbutton(
            self.checkbox_frame_five, text="Re-verify clips (scan near segments for missed clips)",
            variable=self.use_verify)
        self.verify_checkbox.pack(anchor=tk.W)

        # 审核 checkbox
        self.checkbox_frame_six = ttk.Frame(self.left_frame)
        self.checkbox_frame_six.pack(anchor=tk.W)
        self.use_review = tk.BooleanVar()
        self.review_checkbox = ttk.Checkbutton(
            self.checkbox_frame_six, text="Review clips before compile",
            variable=self.use_review)
        self.review_checkbox.pack(anchor=tk.W)

        # Right Column Widgets
        right_frame = ttk.Frame(root, width=300)
        right_frame.grid(row=0, column=2, padx=10, pady=30, sticky="nsew")
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

        # Progress bar for final render
        self.ui_bar = ttk.Progressbar(right_frame, orient='horizontal')
        self.ui_bar.pack(fill=tk.X, padx=10, pady=10)

        self.final_bar = FinalRenderBar(ui=self.ui_bar)

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

        # Redirect stdout to the Text widget
        sys.stdout = StdoutRedirector(self.stdout_text)  # TODO: Change this

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
        timestamps_tooltip = CustomHovertip(
            self.txt_output_checkbox, 'Save the timestamps to a txt file (by default `timestamps.txt` in the output directory).\nYou can change the file name in settings.')
        padding_tooltip = CustomHovertip(
            self.use_clip_padding_checkbox, 'Add extra time before and after each individual clip. Values are in seconds.\nIf using this option, Iit\'s recommended to enable \'Merge Nearby Clips\' to avoid duplicate clips.'
        )
        verify_tooltip = CustomHovertip(
            self.verify_checkbox, 'After AI detection, scan near each clip with a lower threshold to find clips the AI may have missed.')
        review_tooltip = CustomHovertip(
            self.review_checkbox, 'Before compiling, open a dialog to preview, check/uncheck, and edit each clip individually.')
        skip_auto_tooltip = CustomHovertip(
            self.skip_auto_checkbox, 'When a timestamps.txt file already exists, automatically use it without showing the confirmation dialog.')

        settings_tooltip = CustomHovertip(self.settings_button, "Settings")

        self.disable_while_processing = [
            self.add_button,
            self.remove_button,
            self.up_arrow,
            self.down_arrow,
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
            self.skip_auto_checkbox
        ]
        
        self.enable_while_processing = [
            self.cancel_button
        ]

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

    def add_video_url(self):
        self.root.grab_set()
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
            self.thread_active = True
            number_vids_label = ttk.Label(
                self.entry_window, font=(None, 10), text="Getting info...")
            number_vids_label.pack(pady=(5, 10))

            url_entry["state"] = tk.DISABLED
            self.entry_window.update_idletasks()
            url = url_entry.get()

            try:
                n_videos = get_number_of_vids_in_playlist(url)
            except Exception as e:
                number_vids_label.pack_forget()
                messagebox.showerror(
                    "Error",  f"Invalid URL: {url}\nError: {str(e)}")
                url_entry["state"] = tk.ACTIVE
                self.thread_active = False
                return

            valid_videos = []
            errors = []
            for i, vid_details in enumerate(is_valid_yt_dlp_url(url, self.max_quality.get())):
                number_vids_label.config(
                    text=f"Parsing video {i + 1}/{n_videos}")
                if isinstance(vid_details, Exception):
                    if not self.dont_show_again_var.get():
                        result = self.custom_warning_dialog(
                            self.entry_window, "Warning", f"Unable to add a video from the playlist: {str(vid_details)}\nContinue anyway?")
                        if result['action'] == 'cancel':
                            errors = [
                                "Cancelled by user" for _ in range(n_videos)]
                            self.thread_active = False
                            break
                    errors.append(str(vid_details))
                else:
                    valid_videos.append(vid_details)
                    title = vid_details.get('title', 'unknown title')
                    uploader = vid_details.get(
                        'uploader', 'unknown uploader')
                    url = vid_details.get('url')
                    if not url:
                        errors.append("URL not found")
                        continue

                    cleaned_uploader = clean_filename(uploader, "")
                    cleaned_title = clean_filename(title, "")
                    # just in case an uploader uses the same title twice
                    cleaned_title += f" [{vid_details.get('id', 'unknown ID')}]"
                    media_obj = MediaUpload(
                        f"{cleaned_uploader} - {cleaned_title}", 'video' if self.is_video else 'audio', True, url)
                    self.uploaded_videos.append(media_obj)
                    self.update_listbox_add_video(scroll_to_bottom=True)

            if len(errors) == n_videos:
                number_vids_label.pack_forget()
                messagebox.showerror(
                    "Error",  f"Invalid URL: {errors[0] if errors else 'Unknown error'}")
                url_entry["state"] = tk.ACTIVE
                return
            elif len(errors) > 0:
                messagebox.showwarning(
                    "Warning", f"Unable to add {len(errors)}/{n_videos} videos from the playlist.\n\nCommon issues include:\n- The playlist contains deleted or private videos\n- The max allowable quality is too low\n- The playlist contains TikTok photo slideshows or other non-video media\n- If you've used this tool frequently, you may be flagged as a bot. Wait a few minutes and try again")
            elif len(errors) == 0:
                message = f"Successfully imported {n_videos} videos." if n_videos != 1 else "Successfully imported 1 video."
                messagebox.showinfo(
                    "Success", message)
            self.entry_window.destroy()
            self.thread_active = False

        def close_add_url(event=None):
            if self.thread_active:
                confirm = messagebox.askyesno("Confirm Cancellation",
                                              f"The current job will be cancelled, but any previously parsed URLs will be kept. Would you like to cancel?")
                if confirm:
                    self.entry_window.destroy()
            else:
                self.entry_window.destroy()

        def check_url_threaded(event=None):
            thread = threading.Thread(target=check_url)
            thread.start()

        self.entry_window.protocol("WM_DELETE_WINDOW", close_add_url)

        url_entry.bind("<Return>", check_url_threaded)
        url_entry.bind("<Escape>", close_add_url)
        url_entry.focus_set()

        self.entry_window.grab_set()
        self.root.wait_window(self.entry_window)

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
                    str(os.path.basename(video_path)).replace(" ", "\ ")))

        if scroll_to_bottom:
            self.video_listbox.yview_moveto(1.0)

        self.video_listbox.pack()

    def update_listbox_add_video(self, scroll_to_bottom: bool = False):
        current_items = {self.video_listbox.item(item_id, 'values')[
            0]: item_id for item_id in self.video_listbox.get_children()}

        for video in self.uploaded_videos:
            video_path = video.get_path()
            video_key = str(video_path) if video.get_is_url() else str(
                os.path.basename(video_path)).replace(" ", "\ ")

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
                    self.active_thread.terminate()
                finally:
                    print(
                        f"\n{Fore.RED}FAILURE: Operation cancelled by user.")
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

        with open(self.preferences_file, 'w') as configfile:
            self.preferences.write(configfile)

    def reset_preferences_to_file(self):
        self.keep_downloaded_vids.set(self.preferences.get(
            "Settings", "keep_downloaded_vids"))
        self.download_video_path.set(self.preferences.get(
            "Settings", "download_path"
        ))
        self.max_quality.set(self.preferences.get(
            "Settings", "max_quality"
        ))
        self.max_download_speed.set(self.preferences.get(
            "Settings", "max_download_speed"
        ))
        self.output_text_path.set(self.preferences.get(
            "Settings", "output_text_path"
        ))

    def open_settings_modal(self):
        self.root.grab_set()
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
            modal.destroy()
            self.root.grab_release()

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
                    media_url, media_path, download_path, self.max_quality.get(), self.max_download_speed.get(), self.final_bar)
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
                    media_url, media_path, download_path, self.max_download_speed.get(), self.final_bar)
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
        self.disable_objects()
        self.final_bar.reset_total_progress(1)

        self.reset_preferences_to_file()

        try:
            precision = self.precision.get()
            block_size = self.block_size.get()
            threshold = self.threshold.get()
            selected_model = os.path.join(self.models_dir, self.model.get())
            merge_clips = self.merge_clips.get()
            combine = self.combine_vids.get()
            normalize = self.normalize_audio.get()
            save_timestamps = self.save_txt.get()

            # Get model location if in a compiled app
            selected_model = get_bundle_filepath(selected_model)

            self.stdout_text["state"] = tk.NORMAL
            self.stdout_text.delete("1.0", tk.END)
            self.stdout_text["state"] = tk.DISABLED
            self.root.update_idletasks()




            if not self.uploaded_videos:
                raise Exception("Please pick some videos to compile.")

            if not self.output_video_path.get() or self.output_video_path.get() == "No location selected!":
                raise Exception("Please specify an output location.")

            output_video_path = self.output_video_path.get()

            dict_list = []

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

            if any(x.get_is_url() for x in self.uploaded_videos):
                self.handle_url_downloads()

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
            # --- Check for existing timestamps.txt ---
            txt_path = self.output_text_path.get()
            if txt_path and txt_path != "No file selected!" and os.path.exists(txt_path):
                auto_use = self.skip_detection_auto.get()
                if auto_use or messagebox.askyesno(
                    "Skip Detection",
                    f"Found existing timestamps file:\n{txt_path}\n\nSkip AI detection and use saved timestamps directly?"
                ):
                    print(f"{Fore.GREEN}Loading timestamps from {txt_path}...")
                    with_videos, _ = _parse_timestamps_txt(txt_path)

                    # Build basename -> [paths] map (handle duplicates)
                    basename_map = {}
                    for v in self.uploaded_videos:
                        base = os.path.basename(v.get_path())
                        basename_map.setdefault(base, []).append(v.get_path())

                    dict_list = []
                    loaded = 0
                    for entry in with_videos:
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

                    total_progress = ((4 if combine else 2) if self.is_video else (2 if combine else 1)) * (loaded + (1 if combine and loaded > 1 else 0)) * 100
                    self.final_bar.reset_total_progress(total_progress)

                    print(f"Compiling and writing to {output_video_path.split('/')[-1]}...")
                    if self.use_verify.get():
                        print(f"{Fore.CYAN}Running verification scan...")
                        dict_list = _verify_and_expand(
                            dict_list, selected_model,
                            window=self.verify_window_var.get(),
                            logger=self.final_bar)
                    if self.use_review.get():
                        dlg = ReviewDialog(self.root, dict_list, padding,
                                          output_video_path,
                                          use_verify=self.use_verify.get(),
                                          txt_path=self.output_text_path.get())
                        if dlg.result is None:
                            print(f"{Fore.YELLOW}Review cancelled.")
                            self.reenable_disabled_objects()
                            return
                        if not dlg.result:
                            raise Exception("No segments selected after review.")
                        dict_list = dlg.result
                    compile_vid(dict_list, output_video_path, merge_clips, combine, res, self.final_bar, normalize, self.is_video, padding)
                    print(f"{Fore.GREEN}Wrote final video to {output_video_path.split('/')[-1]}.")
                    messagebox.showinfo("Info", f"Video(s) exported to {output_video_path}. Enjoy!")
                    print(f"{Fore.GREEN}SUCCESS!")
                    self.reenable_disabled_objects()
                    return

            try:
                vids_with_clips = 0
                self.final_bar.reset_total_progress(
                    (len(self.uploaded_videos) * 100 * 2))

                for i, input_video_path in enumerate(self.uploaded_videos):
                    input_video_path = input_video_path.get_path()
                    print(
                        f"{Fore.GREEN}[{i + 1}/{len(self.uploaded_videos)}]{Style.RESET_ALL} Getting timestamps for {os.path.basename(input_video_path)}")
                    timestamps, used_existing_data = get_timestamps(
                        input_video_path, precision, block_size, threshold, 58, selected_model, self.final_bar)
                    dict_list.append(timestamps)
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

                        def convert_seconds_to_timestamp(seconds: float) -> str:
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            remaining_seconds = int(
                                round((seconds % 3600) % 60))

                            if remaining_seconds == 60:
                                minutes += 1
                                remaining_seconds = 0

                            if minutes == 60:
                                hours += 1
                                minutes = 0

                            timestamp = f"{hours}:{minutes:02}:{remaining_seconds:02}"
                            return timestamp

                        timestamps_text = ""
                        found_timestamps = False
                        for file in dict_list:
                            timestamps_text += f"{file['filename']}\n"

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
                        logger=self.final_bar)
                if self.use_review.get():
                    _txt_path = txt_path if 'txt_path' in dir() else self.output_text_path.get()
                    dlg = ReviewDialog(self.root, dict_list, padding,
                                      output_video_path,
                                      use_verify=self.use_verify.get(),
                                      txt_path=_txt_path)
                    if dlg.result is None:
                        print(f"{Fore.YELLOW}Review cancelled.")
                        self.reenable_disabled_objects()
                        return
                    if not dlg.result:
                        raise Exception("No segments selected after review.")
                    dict_list = dlg.result

                print(
                    f"Compiling and writing to {output_video_path.split('/')[-1]}...")
                compile_vid(dict_list, output_video_path, merge_clips,
                            combine, res, self.final_bar, normalize, self.is_video, padding)
                print(
                    f"{Fore.GREEN}Wrote final video to {output_video_path.split('/')[-1]}.")
                messagebox.showinfo(
                    "Info", f"Video(s) exported to {output_video_path}. Enjoy!")
            except Exception as e:
                raise Exception(
                    "Encountered error during video processing: " + str(e))

            print(f"{Fore.GREEN}SUCCESS!")

            try:
                shutil.rmtree(TEMP_DIR)
            except:
                # Sometimes deleting the temp dir can fail
                # OSes will auto-delete this directory anyway
                # so this isn't a huge problem
                pass

            if not self.keep_downloaded_vids.get():
                self.remove_urls_from_list()

            self.root.update_idletasks()
            self.reenable_disabled_objects()

        except Exception as e:
            messagebox.showerror("Error", e)
            print(f"\n{Fore.RED}FAILURE: " + str(e))
            self.reenable_disabled_objects()
            return


class StdoutRedirector:

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget["state"] = tk.DISABLED

    def write(self, text):
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

        self.text_widget.see(tk.END)  # Scroll to the end of the text
        self.text_widget["state"] = tk.DISABLED

    def flush(self):
        return


class FinalRenderBar(ProgressBarLogger):
    def __init__(self, ui, init_state=None, bars=None, ignored_bars=None, logged_bars='all', min_time_interval=0, ignore_bars_under=0):
        self.ui = ui
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
        if d['status'] == 'downloading':
            percent_str = re.sub(r'\x1b\[[0-9;]*m', '', d['_percent_str'])
            percent = float(percent_str.strip('%'))
            self.current_progress = percent
        self.ui['value'] = self.current_progress


def main():
    root = root = tk.Tk()
    sv_ttk.set_theme("dark")

    app = VideoProcessorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()