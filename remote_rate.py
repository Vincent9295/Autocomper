"""Shared rate limiting and backoff for remote yt-dlp resolves and refreshes.

Bilibili rate-limits repeated yt-dlp extractions. A batch that resolves many
sources (or a refresh storm after an expired URL) can trigger rate limiting and
make every later resolve fail. These helpers enforce a minimum interval between
resolves and add exponential backoff when a rate-limit / HTTP 412 is detected.
"""

import threading
import time
from typing import Callable

_MIN_RESOLVE_INTERVAL = 2.0
_RATE_LIMIT_BACKOFF = (5.0, 15.0, 45.0, 90.0)
# 批内自适应降速：连续被限流时把 resolve 间隔拉到这些秒数（成功一次就回落）
_THROTTLE_STEPS = (15.0, 30.0, 60.0, 120.0)
_RATE_LIMIT_MARKERS = ("412", "precondition failed", "rate limit", "rate-limit",
                       "too many requests", "429")
# YouTube 的"机器人验证"同一个成因（短时间解析太多），必须同样降速
_BOT_CHECK_MARKERS = ("not a bot", "sign in to confirm", "confirm you're not a bot",
                      "unusual traffic", "captcha")


class ResolveLimiter:
    """Serialize yt-dlp resolves with a minimum interval between calls."""

    def __init__(self, min_interval: float = _MIN_RESOLVE_INTERVAL):
        self._base_interval = float(min_interval)
        self._min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._last = 0.0
        self._throttle_index = 0

    @property
    def min_interval(self) -> float:
        with self._lock:
            return self._min_interval

    def wait(self) -> None:
        """Block until the minimum interval since the last resolve has passed."""
        with self._lock:
            elapsed = time.monotonic() - self._last
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last = time.monotonic()

    def penalize(self) -> float:
        """Slow down after a throttling failure; returns the new interval."""
        with self._lock:
            self._throttle_index = min(self._throttle_index + 1,
                                       len(_THROTTLE_STEPS))
            self._min_interval = _THROTTLE_STEPS[self._throttle_index - 1]
            return self._min_interval

    def note_success(self) -> float:
        """Decay back toward the base interval after a successful resolve."""
        with self._lock:
            if self._throttle_index <= 0:
                self._min_interval = self._base_interval
                return self._min_interval
            self._throttle_index -= 1
            self._min_interval = (_THROTTLE_STEPS[self._throttle_index - 1]
                                  if self._throttle_index > 0
                                  else self._base_interval)
            return self._min_interval


def is_rate_limit_error(exc: Exception) -> bool:
    """Return whether an exception looks like a platform rate-limit / 412."""
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def is_bot_check_error(exc: Exception) -> bool:
    """YouTube's "confirm you're not a bot" / unusual-traffic interstitial."""
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _BOT_CHECK_MARKERS)


def is_throttling_error(exc: Exception) -> bool:
    """Either explicit rate limiting or a bot-check — both mean "slow down"."""
    return is_rate_limit_error(exc) or is_bot_check_error(exc)


def throttle_step(index: int) -> float:
    """Cooldown for the Nth consecutive throttled resolve (0-based)."""
    position = max(0, int(index))
    if position >= len(_THROTTLE_STEPS):
        return _THROTTLE_STEPS[-1]
    return _THROTTLE_STEPS[position]


def rate_limit_backoff(attempt_index: int) -> float:
    """Return the sleep seconds for a rate-limit backoff attempt (0-based)."""
    index = max(0, int(attempt_index))
    if index >= len(_RATE_LIMIT_BACKOFF):
        return _RATE_LIMIT_BACKOFF[-1]
    return _RATE_LIMIT_BACKOFF[index]


def backoff_for_attempt(attempt_index: int, cap: float = 60.0) -> float:
    """Return an exponential backoff (2^attempt) capped at ``cap`` seconds."""
    index = max(0, int(attempt_index))
    return min(float(2 ** index), float(cap))


class LimitedRefresher:
    """Wrap a refresh_func with the shared resolver limiter and rate-limit retry.

    ``retries`` failed refresh attempts (capped) are allowed, sleeping with
    rate-limit backoff between attempts, before the last failure is re-raised.
    """

    def __init__(
        self,
        refresh_func: Callable[..., object],
        limiter: ResolveLimiter | None = None,
        retries: int = 3,
        logger: Callable[[str], object] | None = None,
    ):
        self._refresh_func = refresh_func
        self._limiter = limiter or ResolveLimiter()
        self._retries = max(1, int(retries))
        self._logger = logger or (lambda _message: None)

    def __call__(self, source):
        last_error: Exception | None = None
        for attempt in range(self._retries):
            self._limiter.wait()
            try:
                return self._refresh_func(source)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self._retries:
                    wait = rate_limit_backoff(attempt)
                    self._logger(
                        f"Remote refresh failed ({type(exc).__name__}); "
                        f"retrying in {wait:.0f}s"
                    )
                    time.sleep(wait)
        if last_error is not None:
            raise last_error
        raise RuntimeError("refresh failed without an error")
