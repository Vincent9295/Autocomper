"""Pure transfer-progress formatting and thread-safe UI adaptation helpers."""

import time


def _clock_text(seconds):
    seconds = max(0, int(seconds or 0))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _size_text(value):
    value = float(value or 0)
    units = ("B", "KB", "MB", "GB", "TB")
    size = value
    unit = units[0]
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def format_transfer_progress(current, total, elapsed, source_duration=None):
    current = max(0.0, float(current or 0))
    elapsed = max(0.0, float(elapsed or 0))
    total_value = None if total in (None, 0) else max(0.0, float(total))
    speed = current / elapsed if elapsed > 0 else 0.0
    speed_mb_s = speed / (1024 * 1024) if total_value is not None else None
    realtime = None
    try:
        duration = float(source_duration)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0 and elapsed > 0:
        realtime = (current / total_value * duration) / elapsed if total_value else duration / elapsed

    percent = None
    eta = None
    if total_value is not None and total_value > 0:
        percent = min(100.0, current / total_value * 100)
        if speed > 0:
            eta = max(0.0, (total_value - current) / speed)

    parts = []
    if percent is not None:
        parts.append(f"{percent:.0f}%")
    if realtime is not None:
        parts.append(f"{realtime:.2f}x realtime")
    elif speed_mb_s is not None:
        parts.append(f"{speed_mb_s:.2f} MB/s")
    else:
        parts.append(f"elapsed {_clock_text(elapsed)}")
    if total_value is None and speed > 0 and realtime is None:
        parts.append(f"{speed / (1024 * 1024):.2f} MB/s")
    if eta is not None:
        parts.append(f"ETA {_clock_text(eta)}")
    if total_value is not None:
        parts.append(f"elapsed {_clock_text(elapsed)}")
    stage = "completed" if percent == 100 else "transferring"
    title = stage.title()
    current_line = (
        f"Progress: {_size_text(current)} / {_size_text(total_value)}"
        if total_value is not None else f"Progress: {_size_text(current)}"
    )
    percent_line = f"Progress: {percent:.0f}%" if percent is not None else "Progress: --"
    speed_line = (
        f"Speed: {realtime:.2f}x realtime"
        if realtime is not None else f"Speed: {speed / (1024 * 1024):.2f} MB/s"
    )
    eta_line = f"ETA: {_clock_text(eta)}" if eta is not None else "ETA: --"
    return {
        "current": current,
        "total": total_value,
        "elapsed": elapsed,
        "percent": percent,
        "speed_mb_s": speed_mb_s if speed_mb_s is not None else speed / (1024 * 1024),
        "eta_seconds": eta,
        "realtime": realtime,
        "stage": stage,
        "title": title,
        "current_line": current_line,
        "percent_line": percent_line,
        "speed_line": speed_line,
        "eta_line": eta_line,
        "text": " | ".join(parts),
    }


def format_block_progress(current, total, elapsed, source_duration=None):
    current = max(0, int(current or 0))
    total = max(1, int(total or 1))
    elapsed = max(0.0, float(elapsed or 0))
    percent = min(100.0, current / total * 100)
    realtime = None
    try:
        duration = float(source_duration)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0 and elapsed > 0:
        realtime = (current / total * duration) / elapsed
    speed_text = f"Speed: {realtime:.2f}x realtime" if realtime is not None else "Speed: --"
    return {
        "current": current,
        "total": total,
        "elapsed": elapsed,
        "percent": percent,
        "speed_mb_s": None,
        "eta_seconds": None,
        "realtime": realtime,
        "stage": "completed" if percent == 100 else "transferring",
        "title": "Block progress",
        "current_line": f"Block: {current} / {total}",
        "percent_line": f"Progress: {percent:.0f}%",
        "speed_line": speed_text,
        "eta_line": "ETA: --",
        "text": f"Block: {current} / {total} | Progress: {percent:.0f}% | {speed_text}",
    }


def format_compile_progress(current, total, elapsed, stage="FFmpeg"):
    current = max(0.0, float(current or 0))
    total_value = max(0.0, float(total or 0))
    elapsed = max(0.0, float(elapsed or 0))
    percent = min(100.0, current / total_value * 100) if total_value else None
    speed = current / elapsed if elapsed > 0 else 0.0
    eta = ((total_value - current) / speed) if total_value and speed > 0 else None
    total_text = _clock_text(total_value) if total_value else "--:--"
    current_text = _clock_text(current)
    return {
        "current": current,
        "total": total_value or None,
        "elapsed": elapsed,
        "percent": percent,
        "speed_mb_s": None,
        "eta_seconds": eta,
        "realtime": speed or None,
        "stage": "completed" if percent == 100 else "compiling",
        "title": str(stage),
        "current_line": f"Encoded: {current_text} / {total_text}",
        "percent_line": f"Progress: {percent:.0f}%" if percent is not None else "Progress: --",
        "speed_line": f"Speed: {speed:.2f}x realtime" if speed else "Speed: --",
        "eta_line": f"ETA: {_clock_text(eta)}" if eta is not None else "ETA: --",
        "text": f"{stage}: Encoded {current_text} / {total_text}",
    }


class ProgressThrottle:
    def __init__(self, callback, interval=0.75, clock=None):
        self.callback = callback
        self.interval = max(0.5, float(interval))
        self.clock = clock or time.monotonic
        self.last_emit = None

    def update(self, sample, force=False):
        now = self.clock()
        if force or self.last_emit is None or now - self.last_emit >= self.interval:
            self.last_emit = now
            self.callback(sample)
            return True
        return False


class ProgressWidgetAdapter:
    def __init__(self, root, bar, label, interval=0.75, clock=None):
        self.root = root
        self.bar = bar
        self.label = label
        self.interval = max(0.5, float(interval))
        self.clock = clock or time.monotonic
        self.last_submit = None
        self.pending = None
        self.scheduled = False

    def submit(self, sample, force=False):
        now = self.clock()
        if not force and self.last_submit is not None and now - self.last_submit < self.interval:
            return False
        self.last_submit = now
        self.pending = sample
        if self.scheduled:
            return True
        self.scheduled = True
        self.root.after(0, self._flush)
        return True

    def _flush(self):
        sample = self.pending
        self.pending = None
        self.scheduled = False
        if sample is None:
            return
        total = sample.get("total")
        current = sample.get("current", 0)
        if total:
            self.bar["value"] = min(100.0, float(current) / float(total) * 100)
        lines = [sample.get("title", ""), sample.get("current_line", ""),
                 sample.get("percent_line", ""), sample.get("speed_line", ""),
                 sample.get("eta_line", "")]
        if any(lines):
            self.label.set("\n".join(line for line in lines if line))
        else:
            self.label.set(sample.get("text", ""))
