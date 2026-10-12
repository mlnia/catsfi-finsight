"""Thread-safe TTL + LRU cache (answer cache: identical question/filters/prompt -> zero LLM cost)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, maxsize: int = 512, ttl_s: float = 600):
        self.maxsize, self.ttl = maxsize, ttl_s
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            ts, val = item
            if time.monotonic() - ts > self.ttl:
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return val

    def set(self, key: str, val: Any) -> None:
        with self._lock:
            self._d[key] = (time.monotonic(), val)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)
