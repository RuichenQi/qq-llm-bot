"""Sliding-window per-user rate limit (in-memory)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from config import CONFIG


class RateLimiter:
    def __init__(self, per_minute: int | None = None) -> None:
        self._limit = per_minute or CONFIG.limits.rate_limit_per_min
        self._hits: Dict[int, Deque[float]] = defaultdict(deque)

    def check(self, user_id: int) -> bool:
        """Return True if the user is below the per-minute cap; record the hit."""
        if user_id in CONFIG.superusers:
            return True
        now = time.monotonic()
        window = self._hits[user_id]
        # drop entries older than 60s
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._limit:
            return False
        window.append(now)
        return True
