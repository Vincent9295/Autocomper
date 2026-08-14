# AutoComper Enhanced

> Original [wz-bff/AutoComper](https://github.com/wz-bff/AutoComper) — the foundation.

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
| **Remote VOD Processing** | Process Bilibili, YouTube, and Twitch VODs from remote audio; fetch only selected video segments. |
| **Remote Network Modes** | Choose Remote Stream, Audio Cache, or Full Download for slow or unstable networks. |
| **Compile Progress Monitoring** | Live remote clip preparation, FFmpeg encoding progress, merge progress, speed, and ETA. |
| **Max Download Concurrency** | Control how many remote clips are fetched at once while preparing a compilation (default 5). |
| **Video-Name Timestamps** | Timestamps files now derive from your output video name — e.g. `myvideo_timestamps.txt`, `myvideo_timestamps_reverified.txt`, `myvideo_timestamps_selected.txt` — so detection, re-verify, and review selections stay grouped together. |
| **Merge Batch Size** | Adjust how many clips each FFmpeg merge batch combines before the final concat. Lower it for laptops/weak CPUs; higher is faster on strong machines. |
| **Improved UI** | Scrollable settings panel, stable Settings layout, clearer remote clip progress, and repositioned tooltips. |

### Technical Improvements vs. the Original

The following are improvements in this Enhanced version compared with the original project:

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
| **Re-verify** | None | DRC **+8dB**, scans ±5s around each clip, min confidence **0.40**, >0.75 direct accept, argmax gate + margin + energy floor on mid-score hits |
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

1. **Add Videos** — pick files, use **Add Folder**, or add a Bilibili, YouTube, or Twitch VOD URL/playlist.
2. **Configure** — set Precision / Block Size / Threshold. Use tooltips for guidance.
3. **Choose Remote Processing** for URL inputs:
   - **Remote Stream** reads remote audio directly and fetches video only for previews/selected clips.
   - **Audio Cache** downloads only compressed audio, then detects locally; useful on unstable networks.
   - **Full Download** uses the existing complete-download workflow.
4. **Optional: Add Padding** — extend each clip by N seconds before/after detection.
5. **Optional: Re-verify** — rescan near detected clips to catch missed sounds.
6. **Optional: Review** — preview and check/uncheck every clip before compiling. Remote video previews are fetched on demand.
7. **Select Output File** — choose where to save the compiled video(s).
8. **Process Videos** — compile!

### Remote VOD Notes

Remote processing lets AutoComper analyze a VOD without first downloading the complete video. It still has to read the compressed audio for the full VOD, because the detector must search the entire timeline.

#### Which Remote Processing Mode Should I Use?

| Mode | What it does | Best for | Trade-offs |
|------|--------------|----------|------------|
| **Remote Stream** | Reads remote audio directly while ONNX analyzes it in blocks. Video is fetched only for previews and selected clips. | A first pass when disk space is limited | Sensitive to CDN interruptions and network speed; a failed block is retried before the VOD is skipped. |
| **Audio Cache** | Downloads only compressed audio into the persistent cache, then runs detection locally from that file. | Repeated runs, reverify, or unstable networks | The first run still depends on the remote CDN; later runs reuse the cached audio. |
| **Full Download** | Downloads the complete video using the normal yt-dlp workflow, then processes the local file. | Sources that do not work reliably with streaming, or users who need a local copy | Requires the most disk space and the longest initial download. |

Remote Stream reports the current detection block, for example `Block: 6 / 13`. Audio Cache reports 4 MiB download chunks and the current transfer speed. A slow transfer does not necessarily mean that AutoComper is frozen: Bilibili, YouTube, and Twitch can change CDN throughput during a long request.

While a compilation is being prepared, the UI reports `Preparing clips: N / total` together with the current VOD, clip number, time range, and live download speed/ETA. During the FFmpeg compile the same panel shows encoding progress, speed, and ETA, and the log lists each clip as it is written and each concat batch as it is merged. The UI stays responsive even for large compilations.

#### Skipped Clips and the Skipped-Clips Report

Each remote clip is downloaded on its own. If one clip cannot be fetched (expired signed URL, CDN connection failure, disk full, or an unreachable source), AutoComper records the failure, skips that clip, and continues with the next one; the final video is compiled from the clips that did succeed. The log then prints a compact summary grouped by failure cause (URL expired/rate-limited, connection failed, disk full, other) instead of one giant error wall per clip.

For the full detail on every skipped clip, AutoComper writes a `_skipped_clips.txt` file next to your output video (for example `MyVideo_skipped_clips.txt`). It lists each skipped clip's name, full source URL, time range, and the exact failure reason, so you can see exactly what was left out. The file is only created when at least one clip was skipped. If nothing is skipped, no file is produced.

Before downloading any clips, AutoComper checks that the temp drive has enough free space for the selected clips plus the final merge (based on your **Max Download Quality**). If the disk would run out mid-compile, it stops with a clear "insufficient disk space" error instead of filling the drive and failing partway.

#### Remote Settings

The **Remote Settings** panel controls how URL inputs are resolved and cached:

- **Remote Processing** selects `Remote Stream`, `Audio Cache`, or `Full Download`.
- **Remote Browser Cookies** selects how yt-dlp authenticates when a platform requires a browser session:
  - **Auto** starts without cookies and tries browser cookies when the platform reports that they may help.
  - **None** never reads browser cookies. Use this for public videos when you do not want browser access.
  - **Firefox**, **Chrome**, or **Edge** explicitly reads cookies from that browser.
- **Cache Location** shows where persistent remote data is stored. The default Windows location is `%LOCALAPPDATA%\AutoComper\cache`.
- **Choose Cache Folder** changes the cache root. A removable drive is supported when it is mounted and writable.
- **Open Cache Folder** opens the current cache location in File Explorer.
- **Clear Cache** removes AutoComper's detection, audio, and segment cache entries. It does not delete the original videos or ordinary user files.
- **Max Download Concurrency** sets how many remote video clips are fetched at once while preparing a compilation (default 5). Each worker runs its own FFmpeg process, so higher values finish faster on fast connections but use more CPU/disk; lower this to 1-2 if the PC feels sluggish during "Preparing clips". The control is disabled while a run is in progress.
- **Audio Cache Chunk Size (MiB)** sets how much audio each chunk download requests while an Audio Cache file is being fetched (default 4 MiB). Larger chunks mean fewer requests but slower restart on a dropped connection; smaller chunks resume more cheaply on flaky networks.
- **Audio Cache Download Concurrency** controls how many chunks are fetched in parallel while building an Audio Cache file. More workers finish faster on fast links but raise CDN rate-limit pressure; lower it if you see HTTP 412 / 429 errors from Bilibili on big batches.

#### Playlist Imports and Platform Rate Limits

Importing a large playlist (for example 100+ entries) can hit the platform's API rate limit and show `Metadata failed` in the entry list, or skip items during import. This is a platform limit, not a file or cache problem, and a short wait (typically 10-30 minutes) clears it.

AutoComper already reduces how often it hits the API:

- the review dialog can jump straight to a page instead of paging through every earlier page;
- an entry whose metadata failed is retried automatically with a backoff instead of being marked failed forever;
- importing reuses metadata already hydrated in the review dialog instead of re-resolving every selected entry.

To keep imports reliable:

- import in smaller batches (20-30 entries at a time);
- wait for each page's entries to show `Ready` before confirming;
- if you still see `Metadata failed`, wait 10-30 minutes for the rate limit to clear, then retry;
- staying logged in (browser cookies) raises the platform quota.

#### Import External Audio



**Import External Audio** is available in the Remote Cache button group. It lets you use audio downloaded outside AutoComper as the Audio Cache for a selected remote VOD.

Use it as follows:

1. Select exactly one YouTube, Twitch, or Bilibili remote URL in the main media list.
2. Click **Import External Audio**.
3. Choose one local media file.
4. Let AutoComper inspect the file, verify its audio stream and duration, and register it for the selected VOD.

Supported input containers include `.m4s`, `.mp4`, `.m4a`, `.webm`, `.opus`, `.mp3`, `.wav`, and `.flac`. An `.mp4` is accepted only when it contains an audio stream; its video stream is ignored. A fragmented `.m4s` file must contain enough initialization data for FFmpeg to read it.

Only one audio file is imported at a time. This is intentional: automatic multi-file ordering can silently create the wrong timeline or audio overlap. Bilibili multi-part videos must be imported one part at a time, with the matching part selected in the media list.

AutoComper converts non-AAC audio to AAC in an `.m4a` cache file. If the input already contains AAC/MP4A audio, it uses a direct copy/remux path instead of re-encoding, which is substantially faster. The original file is not moved, deleted, or uploaded. The imported audio must cover the same timeline as the selected VOD; a filename alone cannot identify its source.

#### Why Browser Cookies May Be Needed

Browser cookies are not required for every public VOD. They can help when a platform requires a signed-in session for:

- age-restricted, private, members-only, or region-limited videos;
- higher-quality formats that are hidden from anonymous requests;
- Bilibili requests that return HTTP 412 or reject an anonymous CDN request;
- playlist or channel metadata that is incomplete without a session.

Cookies are read locally through yt-dlp and are not written into AutoComper's cache metadata or normal logs. If a browser reports a DPAPI decryption error, AutoComper tries another configured source when using **Auto**, and the user can select a specific browser that is open, installed, and logged in. A browser must already be signed in; AutoComper does not log in for the user.

#### Why Remote Downloads Can Be Slow

Remote speed depends on more than the local internet connection. Common causes include:

- CDN throttling after a connection has transferred for a while;
- the route chosen by the ISP, VPN, proxy, or region;
- an expired or refreshed signed URL;
- a temporary HTTP error or interrupted range request;
- a platform selecting a slower audio CDN or format.

AutoComper mitigates transient failures by retrying remote chunks, monitoring throughput, refreshing signed URLs when sustained speed drops, resuming Audio Cache downloads from the completed byte offset, and retrying Bilibili Remote Stream blocks independently. These measures cannot remove a persistent ISP/CDN speed cap. If a source remains slow, try another network route, disable or change the VPN, use **Audio Cache** so later runs do not redownload audio, or use **Full Download**.

#### Cookies, GPU, and Network Responsibilities

- **Remote Browser Cookies** affect source access and available formats; they do not speed up ONNX inference.
- **Use GPU (CUDA)** controls ONNX inference only. It does not control CDN download speed, remote audio decoding, or FFmpeg encoding.
- **Remote Stream** and **Audio Cache** need the full remote audio timeline for detection, even though they avoid downloading the full video.
- Remote **Re-verify** downloads only audio windows around detected candidates, reuses the same ONNX/DRC logic, and maps new timestamps back to the original VOD timeline.

Active live streams, DRM-protected media, and every private or region-restricted source are not guaranteed to work.

### Re-verify Details

Re-verify uses DRC (Dynamic Range Compression) to boost quiet sounds near detected clips. Key behaviors:

- **DRC gain**: up to +8 dB (boosts quiet burps buried in music)
- **Scan window**: ±5 seconds around each clip (catches burps the model missed right next to a hit)
- **Threshold floor**: never scans below **0.40**, even when your main detection threshold is set low (0.2–0.5) — keeps the scan out of the high-false-positive noise band
- **High-score skip**: DRC hits with confidence > 0.75 bypass P3 confirmation
- **Argmax margin**: mid-score hits are accepted only when burp clearly beats the next-best sound class (1.15×), not merely "happens to be top class"
- **Energy floor**: rejects clips whose raw (un-boosted) audio is near-silent — kills "no sound" false positives
- **New/original**: DRC-discovered clips shown separately — no automatic boundary expansion
- **Output**: `Verification: scanned X window(s), confirmed Y new, DRC-skip Z, rejected W.`

Any remote source must download its full audio timeline before detection can run — this is true for both **Remote Stream** and **Audio Cache**, even though neither downloads the video. What happens during **Re-verify** depends on the mode:

- **Remote Stream** keeps no stored audio, so the re-verify scan re-downloads only the small window around each detected clip (no full re-download).
- **Audio Cache** keeps the stored audio and reuses it directly for the re-verify scan, so nothing extra is downloaded.

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
