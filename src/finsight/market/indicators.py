"""CATsFi technical-analysis core, ported from the CATsFi scanners (`rsi_divergence.py`,
`high_density_areas_module.py`, `trend_reversal_breakout_scanner*.py`) into pure, side-effect
free, unit-tested functions (no plotting, no network, no globals)."""

from __future__ import annotations

import numpy as np

RSI_PERIOD = 14
PIVOT_LEFT = PIVOT_RIGHT = 2
DIVERGENCE_LOOKBACK = 50
MAX_PIVOT_DISTANCE = 25
DENSITY_BINS = 20
DENSITY_THRESHOLD = 1.5
MIN_PEAKS_IN_ZONE = 3
PEAK_DISTANCE = 3


def calculate_rsi(prices, period: int = RSI_PERIOD) -> np.ndarray | None:
    """Wilder RSI; first `period` values are NaN."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 1:
        return None
    delta = np.diff(prices)
    gains, losses = np.where(delta > 0, delta, 0.0), np.where(delta < 0, -delta, 0.0)
    avg_gain, avg_loss = np.zeros(len(delta)), np.zeros(len(delta))
    avg_gain[period - 1], avg_loss[period - 1] = gains[:period].mean(), losses[:period].mean()
    for i in range(period, len(delta)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    rsi = np.full(len(prices), np.nan)
    for i in range(period, len(prices)):
        loss = avg_loss[i - 1]
        rsi[i] = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain[i - 1] / loss)
    return rsi


def _pivots(values, left: int, right: int, high: bool) -> list[int]:
    v = np.asarray(values, dtype=float)
    out = []
    for i in range(left, len(v) - right):
        cur, lv, rv = v[i], v[i - left : i], v[i + 1 : i + right + 1]
        if np.isnan(cur) or np.isnan(lv).any() or np.isnan(rv).any():
            continue
        if high and cur > lv.max() and cur >= rv.max():
            out.append(i)
        if not high and cur < lv.min() and cur <= rv.min():
            out.append(i)
    return out


def find_pivot_lows(values, left=PIVOT_LEFT, right=PIVOT_RIGHT) -> list[int]:
    return _pivots(values, left, right, high=False)


def find_pivot_highs(values, left=PIVOT_LEFT, right=PIVOT_RIGHT) -> list[int]:
    return _pivots(values, left, right, high=True)


def _divergence(prices, rsi, pivots, kind: str, lookback=DIVERGENCE_LOOKBACK, max_distance=MAX_PIVOT_DISTANCE) -> dict | None:
    """kind: bullish (LL price / HL rsi), bearish (HH / LH), hidden_bullish (HL / LL), hidden_bearish (LH / HH)."""
    n = len(prices)
    recent = [i for i in pivots if max(0, n - lookback) <= i < n]
    if len(recent) < 2:
        return None
    if kind == "bullish":
        cur = min(recent, key=lambda i: prices[i])
    elif kind == "bearish":
        cur = max(recent, key=lambda i: prices[i])
    else:
        cur = max(recent)
    if np.isnan(rsi[cur]):
        return None
    for prev in sorted((i for i in recent if i < cur and cur - i <= max_distance), reverse=True):
        if np.isnan(rsi[prev]):
            continue
        p0, p1, r0, r1 = prices[prev], prices[cur], rsi[prev], rsi[cur]
        ok = {
            "bullish": p1 < p0 and r1 > r0,
            "bearish": p1 > p0 and r1 < r0,
            "hidden_bullish": p1 > p0 and r1 < r0,
            "hidden_bearish": p1 < p0 and r1 > r0,
        }[kind]
        if ok:
            return {"type": kind, "previous_index": int(prev), "current_index": int(cur),
                    "previous_price": float(p0), "current_price": float(p1),
                    "previous_rsi": round(float(r0), 2), "current_rsi": round(float(r1), 2), "bars_ago": int(n - 1 - cur)}
    return None


def find_divergences(prices, rsi=None) -> list[dict]:
    prices = np.asarray(prices, dtype=float)
    rsi = calculate_rsi(prices) if rsi is None else rsi
    if rsi is None:
        return []
    lows, highs = find_pivot_lows(prices), find_pivot_highs(prices)
    out = []
    for kind, piv in (("bullish", lows), ("hidden_bullish", lows), ("bearish", highs), ("hidden_bearish", highs)):
        d = _divergence(prices, rsi, piv, kind)
        if d:
            out.append(d)
    return out


def _peaks(x: np.ndarray, distance: int) -> list[int]:
    """Local maxima separated by >= distance bars (scipy.find_peaks equivalent for our use)."""
    cand = [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1]]
    cand.sort(key=lambda i: x[i], reverse=True)
    kept: list[int] = []
    for i in cand:
        if all(abs(i - k) >= distance for k in kept):
            kept.append(i)
    return sorted(kept)


def density_zones(prices, bins=DENSITY_BINS, threshold=DENSITY_THRESHOLD, min_peaks=MIN_PEAKS_IN_ZONE) -> list[dict]:
    """Support/resistance = price bins where swing highs/lows cluster `threshold`x above uniform."""
    close = np.asarray(prices, dtype=float)
    turning = np.concatenate([close[_peaks(close, PEAK_DISTANCE)], close[_peaks(-close, PEAK_DISTANCE)]])
    if len(turning) < bins:
        return []
    counts, edges = np.histogram(turning, bins=bins)
    expected = len(turning) / bins
    last = close[-1]
    zones = []
    for j in range(bins):
        dens = counts[j] / expected
        if dens > threshold and counts[j] >= min_peaks:
            lo, hi = float(edges[j]), float(edges[j + 1])
            zones.append({"low": round(lo, 6), "high": round(hi, 6), "density": round(float(dens), 2),
                          "touches": int(counts[j]),
                          "role": "support" if hi < last else "resistance" if lo > last else "current"})
    zones.sort(key=lambda z: abs((z["low"] + z["high"]) / 2 - last))
    return zones


def trend_state(prices, sma_len: int = 50, window: int = 20) -> dict:
    """CATsFi trend filter: >=85% of the last `window` closes above/below SMA, plus Kaufman efficiency ratio."""
    p = np.asarray(prices, dtype=float)
    if len(p) < sma_len:
        return {"trend": "unknown", "efficiency_ratio": 0.0, "sma": None}
    sma = float(p[-sma_len:].mean())
    sample = p[-window:]
    above = float(np.mean(sample > sma))
    trend = "up" if above >= 0.85 else "down" if above <= 0.15 else "sideways"
    path = float(np.abs(np.diff(sample)).sum())
    er = abs(sample[-1] - sample[0]) / path if path else 0.0
    return {"trend": trend, "efficiency_ratio": round(er, 3), "sma": round(sma, 6), "share_above_sma": round(above, 2)}


def snapshot(prices) -> dict:
    p = np.asarray(prices, dtype=float)
    rsi = calculate_rsi(p)
    return {
        "last_close": float(p[-1]),
        "change_pct_20": round(float((p[-1] / p[-21] - 1) * 100), 2) if len(p) > 21 else None,
        "rsi": round(float(rsi[-1]), 2) if rsi is not None else float("nan"),
        "rsi_period": RSI_PERIOD,
        **trend_state(p),
        "divergences": find_divergences(p, rsi),
        "sr_zones": density_zones(p)[:5],
    }
