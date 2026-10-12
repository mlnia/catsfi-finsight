"""API-key authentication with roles (viewer < admin) and per-key token-bucket rate limiting."""

from __future__ import annotations

import hashlib
import threading
import time

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from finsight.observability.metrics import GUARDRAIL_EVENTS

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
ROLE_RANK = {"viewer": 1, "admin": 2}


class RateLimiter:
    def __init__(self, per_minute: int):
        self.rate = per_minute / 60.0
        self.capacity = max(per_minute, 1)
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, ts = self._b.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - ts) * self.rate)
            if tokens < 1:
                self._b[key] = (tokens, now)
                return False
            self._b[key] = (tokens - 1, now)
            return True


class Principal:
    def __init__(self, key_id: str, role: str):
        self.key_id, self.role = key_id, role


def require(role: str):
    def dep(request: Request, api_key: str | None = Depends(api_key_header)) -> Principal:
        c = request.app.state.container
        s = c.settings
        if not s.auth_enabled:
            principal = Principal("anonymous", "admin" if s.env != "prod" else "viewer")
        else:
            got = s.api_key_roles.get(api_key or "")
            if got is None:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key")
            principal = Principal(hashlib.sha256(api_key.encode()).hexdigest()[:10], got)
        if ROLE_RANK.get(principal.role, 0) < ROLE_RANK[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role '{role}' required")
        limiter: RateLimiter = request.app.state.rate_limiter
        if not limiter.allow(principal.key_id + (request.client.host if request.client and not s.auth_enabled else "")):
            GUARDRAIL_EVENTS.labels("rate_limit", "reject").inc()
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
        return principal

    return dep
