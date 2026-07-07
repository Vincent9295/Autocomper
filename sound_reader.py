#!/usr/bin/env python
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import numpy as np
import onnxruntime as ort
from typing import Generator, Any, Dict, Tuple
from collections import OrderedDict

from utils import FFMPEG_PATH
from proglog import default_bar_logger

SAMPLE_RATE = 32000
is_windows = sys.platform.startswith('win')


def subsample(frame: np.ndarray, scale_factor: int) -> np.ndarray:
    subframe = frame[:len(frame) - (len(frame) % scale_factor)].reshape(-1, scale_factor)
    subframe_mean = subframe.max(axis=1)
    subsample = subframe_mean
    if len(frame) % scale_factor != 0:
        residual_frame = frame[len(frame) - (len(frame) % scale_factor):]
        residual_mean = residual_frame.max()
        subsample = np.append(subsample, residual_mean)
    return subsample


def get_segments(scores: np.ndarray, precision: int, threshold: float, offset: int):
    seq_iter = iter(np.where(scores > threshold)[0])
    try:
        seq = next(seq_iter)
        pred = scores[seq]
        segment = {'start': seq, 'end': seq, 'pred': pred}
    except StopIteration:
        return
    for seq in seq_iter:
        pred = scores[seq]
        if seq - 1 == segment['end']:
            segment['end'] = seq
            segment['pred'] = max(segment['pred'], pred)
        else:
            yield segment
            segment = {'start': seq, 'end': seq, 'pred': pred}
    yield segment


def compute_timestamps(framewise_output, precision, threshold, focus_idx, offset):
    focus = framewise_output[:, focus_idx]
    subsampled_scores = subsample(focus, precision)
    segments = map(lambda segment: {
        'start': segment['start'] * precision / 100 + offset,
        'end': segment['end'] * precision / 100 + offset + 1,
        'pred': round(float(segment['pred']), 6)
    }, get_segments(subsampled_scores, precision, threshold, offset))
    return segments


def pad_array_if_needed(arr, desired_size, pad_value=0):
    current_size = arr.shape[0]
    if current_size < desired_size:
        padding_needed = desired_size - current_size
        return np.pad(arr, (0, padding_needed), "constant", constant_values=(pad_value,))
    return arr


def load_audio(file: str, sr: int, frame_count: int):
    cmd = [FFMPEG_PATH, '-hide_banner', '-loglevel', 'warning', '-i', file,
            '-filter_complex', '[0:a]aresample=32000:async=1,asetpts=PTS-STARTPTS,atempo=1,pan=mono|c0=0.5*c0+0.5*c1[audio]',
           '-map', '[audio]', '-f', 's16le', '-acodec', 'pcm_s16le',
           '-ar', str(sr), '-ac', '1', '-bufsize', '128k', '-']
    subprocess_options = {'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE}
    if is_windows:
        subprocess_options['creationflags'] = subprocess.CREATE_NO_WINDOW
    chunk_size = frame_count * 2
    process = subprocess.Popen(cmd, bufsize=1, **subprocess_options)
    try:
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            yield chunk
    except GeneratorExit:
        process.terminate()
        process.wait()
        return
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        if process.returncode != 0:
            raise Exception("Failed to process the file. Either the file does not exist or is corrupted.")
        raise subprocess.CalledProcessError(return_code, cmd)


def hash_file(file_path, algorithm='sha256', chunk_size=8192) -> str:
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _get_audio_duration(file):
    """用 ffmpeg（非 ffprobe）快速探测时长。"""
    try:
        cmd = [FFMPEG_PATH, '-hide_banner', '-i', file]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                             **({'creationflags': subprocess.CREATE_NO_WINDOW} if is_windows else {}))
        m = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', out.stderr or '')
        if m:
            h, mi, s, ms = map(int, m.groups())
            return h * 3600 + mi * 60 + s + ms / 100
    except Exception:
        pass
    return None


class _SizedIterable:
    def __init__(self, gen, total):
        self._gen = gen
        self._total = total
    def __iter__(self):
        return self._gen
    def __len__(self):
        return self._total


MAX_CACHE_SIZE = 20
timestamps_dict: 'OrderedDict[Tuple[str, int, int, float, str], Dict[str, Any]]' = OrderedDict()


def _separate_vocals(input_file, temp_dir):
    """Extract percussive component via HPSS — isolates short impulse sounds (e.g. burps)
    from sustained harmonic content (music/singing). Returns percussive.wav path or None."""
    try:
        import librosa
    except ImportError:
        print("Warning: librosa not installed. Run: pip install librosa")
        return None
    import tempfile as _tempfile
    try:
        audio_tmp = _tempfile.NamedTemporaryFile(suffix='.wav', dir=temp_dir, delete=False)
        audio_tmp.close()
        extract_cmd = [FFMPEG_PATH, '-y', '-hide_banner', '-loglevel', 'error',
                       '-i', input_file, '-vn', '-acodec', 'pcm_s16le',
                       '-ar', '32000', '-ac', '1', audio_tmp.name]
        _opts = {'creationflags': subprocess.CREATE_NO_WINDOW} if is_windows else {}
        if subprocess.run(extract_cmd, capture_output=True, timeout=120, **_opts).returncode != 0:
            os.remove(audio_tmp.name)
            return None
        y, sr = librosa.load(audio_tmp.name, sr=32000, mono=True)
        _, y_percussive = librosa.effects.hpss(y)
        out_path = _tempfile.NamedTemporaryFile(suffix='.wav', dir=temp_dir, delete=False)
        out_path.close()
        import soundfile as sf
        sf.write(out_path.name, y_percussive, sr)
        os.remove(audio_tmp.name)
        return out_path.name
    except Exception:
        pass
    return None


def get_timestamps(file, precision=100, block_size=600, threshold=0.90, focus_idx=58,
                   model="bdetectionmodel_05_01_23", logger=None, ort_session=None,
                   use_vocal_sep=False):
    if precision < 0:
        raise Exception("Precision must be a positive number!")
    if not (threshold >= 0 and threshold <= 1):
        raise Exception("Threshold must be between 0 and 1!")
    if block_size < 0:
        raise Exception("Block size must be a positive number!")

    file_hash = hash_file(file)
    cache_key = (file_hash, precision, block_size, threshold, model, focus_idx)
    if ort_session is None and cache_key in timestamps_dict:
        previous_data = timestamps_dict[cache_key]
        previous_data['filename'] = file
        if logger:
            bar_logger = default_bar_logger(logger)
            _dur = _get_audio_duration(file)
            block_count = max(1, int(_dur / block_size) + 1) if _dur is not None else 1
            for _ in bar_logger.iter_bar(block=range(block_count)):
                pass
        return previous_data, True

    if ort_session is None:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(model, sess_options,
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # --- percussive isolation (HPSS) ---
    actual_file = file
    _vocals_cleanup = None
    if use_vocal_sep:
        print("Running HPSS percussive isolation...")
        _vocals_cleanup = _separate_vocals(file, os.path.dirname(file) or tempfile.gettempdir())
        if _vocals_cleanup:
            actual_file = _vocals_cleanup
        else:
            print("Warning: HPSS failed, using original audio")

    offset = 0
    blocks = load_audio(actual_file, SAMPLE_RATE, SAMPLE_RATE * block_size)
    _dur = _get_audio_duration(actual_file)
    if _dur is not None:
        blocks = _SizedIterable(blocks, max(1, int(_dur / block_size) + 1))
    else:
        blocks = _SizedIterable(blocks, 1)

    info = {'filename': file, 'timestamps': []}
    frame_count = SAMPLE_RATE * block_size

    if logger:
        bar_logger = default_bar_logger(logger)
        blocks = bar_logger.iter_bar(block=blocks)

    for block in blocks:
        samples = np.frombuffer(block, dtype=np.int16)
        samples = pad_array_if_needed(samples, frame_count)
        samples = samples.reshape(1, -1)
        samples = samples / (2**15)
        samples = samples.astype(np.float32)
        ort_inputs = {"input": samples}
        framewise_output = ort_session.run(["output"], ort_inputs)[0]
        preds = framewise_output[0]
        info["timestamps"].extend(compute_timestamps(preds, precision, threshold, focus_idx, offset))
        offset += block_size

    if len(timestamps_dict) >= MAX_CACHE_SIZE:
        timestamps_dict.popitem(last=False)
    timestamps_dict[cache_key] = info
    return info, False