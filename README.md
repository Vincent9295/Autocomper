# AutoComper (Enhanced)

> Forked from [wz-bff/AutoComper](https://github.com/wz-bff/AutoComper) — a GUI frontend for AI-powered sound detection video clipping.

This enhanced version adds **clip review, re-verification, editing, batch processing**, and a **native FFmpeg GPU pipeline** — improving both speed and usability.

---

## 🚀 What's New (vs. Original)

### New Features

| Feature | Description |
|---------|-------------|
| **Skip Detection** | Load existing `timestamps.txt` to skip AI detection entirely. Auto-use mode suppresses the confirmation dialog. |
| **Review Dialog** | After detection, preview and check/uncheck every clip before compiling. Right-click for audio/video preview. |
| **Edit Times** | Double-click any row in the review dialog to manually adjust start/end times (HH:MM:SS or seconds). |
| **Re-verify Clips** | DRC scan near each clip to find missed sounds. Threshold syncs to main detection. New/original clips shown separately. High-score DRC hits skip P3 confirmation. |
| **Add Folder** | Recursively scan a folder for video/audio files — no need to pick files one by one. |
| **Save Selected** | Review dialog exports checked clips to `{original}_selected.txt` for future re-use. |
| **Audio Mode** | Full audio-only pipeline with native FFmpeg concat. |
| **CPU/GPU Toggle** | One-click switch between CUDA and CPU inference — keeps your GPU quiet during overnight runs. Saved in presets. |

### Technical Improvements

| Area | Original | Enhanced |
|------|----------|----------|
| **Video pipeline** | MoviePy (`libx264` CPU) | Native FFmpeg subprocess (`h264_nvenc` GPU) |
| **Inference** | `onnxruntime` (CPU) | `onnxruntime-gpu` (CUDA) — falls back to CPU automatically; **CPU/GPU toggle** for quiet overnight runs |
| **Audio loading** | `list()` full memory load | Streaming generator + LRU cache |
| **Frame rate** | Inherit from source | Fixed **30 fps** output (prevents VFR desync) |
| **Audio sample rate** | Variable | Fixed **44100 Hz** output |
| **Concat method** | Concat demuxer (timestamp bugs) | Concat **filter** (frame-level, no drift) |
| **Mixed resolutions** | Not handled | Auto-detect → scale/pad all to mode resolution |
| **Clip overlap** | Possible audio bleed | Keep longer clip, discard shorter; compile never bridges gaps containing removed clips |
| **Stereo input** | Left channel only | **50/50 L+R mix** |
| **Re-verify** | None | DRC **+8dB**, threshold syncs to main, >0.75 direct accept, argmax gate on mid-score hits |
| **False positives** | None | Optional **Strict FP filter** (drop clips where burp isn't the top class); suspect clips pre-deselected in Review |
| **Memory** | Unbounded | `-threads 2`, batched concat (6 files/batch), segment-by-segment encoding |

---

## 📋 System Requirements

- **Windows** (primary target)
- **NVIDIA GPU** with updated drivers (for `h264_nvenc` + `onnxruntime-gpu`)
- **Python 3.10+** (for building from source)
- FFmpeg binary placed at `ffmpeg/windows/ffmpeg.exe`

---

## 🔧 Installation

### Pre-built (Recommended)

1. Download the latest release from [Releases](../../releases)
2. Extract the zip — `autocomper.exe` is ready to run
3. Place your model (`.onnx`) in the `models/` folder next to the exe
4. FFmpeg is bundled — no extra setup needed

### Build from Source

```powershell
# Windows PowerShell
python -m venv .env
.\.env\Scripts\Activate.ps1
pip install -r requirements.txt
python setup.py build
```

The executable is at `build/exe.win-*/autocomper.exe`. Copy `ffmpeg/`, `img/`, and `models/` into the build directory.

---

## 🎮 NVIDIA GPU Acceleration (Optional)

The app works fine on CPU, but NVIDIA GPU users can speed up inference by installing:

1. **[CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)** (~3 GB)
   - Uncheck "Nsight VSE" and "Visual Studio Integration" — not needed.
2. **[cuDNN 8.9.7 for CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-archive)** (requires free NVIDIA account)
   - Download `cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip`
   - Extract and copy files:
     - `bin\*.dll` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\`
     - `include\*.h` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\`
     - `lib\x64\*.lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\`

If CUDA isn't installed, the app falls back to CPU automatically.

**Prefer silence over speed?** Uncheck **"Use GPU (CUDA)"** (next to the Process button) to run detection purely on CPU — slower, but your GPU stays quiet for overnight runs.

---

## 📖 Usage

1. **Add Videos** — pick files or use **Add Folder** to scan a directory
2. **Configure** — set Precision / Block Size / Threshold. Use tooltips for guidance.
3. **Optional: Add Padding** — extend each clip by N seconds before/after detection
4. **Optional: Re-verify** — rescan near detected clips to catch missed sounds
5. **Optional: Review** — preview each clip, check/uncheck, double-click to edit times
6. **Select Output File** — choose where to save the compiled video(s)
7. **Process Videos** — compile!

### Re-verify Details

Re-verify uses DRC (Dynamic Range Compression) to boost quiet sounds near detected clips. Key behaviors:

- **DRC gain**: up to +8 dB (boosts quiet burps buried in music)
- **Threshold**: automatically syncs to your main detection threshold (no mismatch)
- **High-score skip**: DRC hits with confidence > 0.75 bypass P3 confirmation
- **New/original**: DRC-discovered clips shown separately — no automatic boundary expansion
- **Output**: `Verification: scanned X window(s), confirmed Y new, DRC-skip Z, rejected W.`

### Timestamps Format

```
/path/to/video.mp4
0:00:05 - 0:00:10, confidence: 0.95
0:01:15 - 0:01:20, confidence: 0.88
```

## 🙏 Credits

- Original [wz-bff/AutoComper](https://github.com/wz-bff/AutoComper) — the foundation
- [onnxruntime](https://onnxruntime.ai/) for inference
- [sv-ttk](https://github.com/rdbende/Sun-Valley-ttk-theme) for modern UI theme
- [Boletus Edulis](https://www.youtube.com/@BoletusEdulis79) for extensive testing <-- GOATED Person