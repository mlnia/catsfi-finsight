"""Market data adapters (no API keys): Binance USDT-M futures klines for crypto, Yahoo Finance
chart API for equities. Short in-process TTL cache keeps the agent cheap and rate-limit friendly."""

from __future__ import annotations

import time

import httpx

BINANCE_URL = "https://fapi.binance.com/fapi/v1/klines"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_RANGE = {"1h": "1mo", "1d": "1y", "1wk": "5y"}
_cache: dict[tuple[str, str], tuple[float, list[float]]] = {}


def fetch_closes(symbol: str, interval: str = "1d", limit: int = 250, client: httpx.Client | None = None,
                 ttl_s: int = 300) -> list[float]:
    key = (symbol.upper(), interval)
    if key in _cache and time.time() - _cache[key][0] < ttl_s:
        return _cache[key][1]
    c = client or httpx.Client(timeout=15, headers={"User-Agent": "Mozilla/5.0 (CATsFi FinSight)"})
    sym = symbol.upper()
    if sym.endswith("USDT"):
        iv = {"1wk": "1w"}.get(interval, interval)
        r = c.get(BINANCE_URL, params={"symbol": sym, "interval": iv, "limit": limit})
        r.raise_for_status()
        closes = [float(k[4]) for k in r.json()]
    else:
        r = c.get(YAHOO_URL.format(symbol=sym), params={"interval": interval, "range": YAHOO_RANGE.get(interval, "1y")})
        r.raise_for_status()
        res = r.json()["chart"]["result"]
        if not res:
            raise ValueError(f"no market data for {sym}")
        closes = [x for x in res[0]["indicators"]["quote"][0]["close"] if x is not None][-limit:]
    if len(closes) < 60:
        raise ValueError(f"not enough bars for {sym} {interval}: {len(closes)}")
    _cache[key] = (time.time(), closes)
    return closes
