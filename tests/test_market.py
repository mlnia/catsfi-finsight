import math

import numpy as np

from finsight.market.fundamentals import annual_series, fmt_value
from finsight.market.indicators import (
    calculate_rsi,
    density_zones,
    find_divergences,
    find_pivot_highs,
    find_pivot_lows,
    snapshot,
    trend_state,
)

from .conftest import FACTS, synthetic_closes


def test_rsi_bounds_and_extremes():
    up = np.arange(1, 60, dtype=float)
    assert calculate_rsi(up)[-1] == 100.0
    down = up[::-1]
    assert calculate_rsi(down)[-1] < 1.0
    r = calculate_rsi(synthetic_closes())
    assert np.isnan(r[:14]).all() and np.nanmin(r) >= 0 and np.nanmax(r) <= 100
    assert calculate_rsi([1, 2, 3]) is None


def test_pivots():
    x = [1, 2, 5, 2, 1, 0, 1, 3, 1]
    assert find_pivot_highs(x) == [2]
    assert find_pivot_lows(x) == [5]


def test_bullish_divergence_detected():
    # lower low in price while momentum (RSI) makes a higher low
    p = list(np.linspace(120, 100, 30)) + list(np.linspace(100, 108, 6)) + list(np.linspace(108, 98.5, 4)) + [99.5, 101, 102]
    kinds = {d["type"] for d in find_divergences(p)}
    assert "bullish" in kinds


def test_density_zones_and_trend():
    closes = synthetic_closes()
    ranging = [100 + 4 * math.sin(i / 4) for i in range(250)]  # range-bound: swings cluster at the extremes
    zones = density_zones(ranging)
    assert zones and all(z["low"] < z["high"] for z in zones)
    t = trend_state(closes)
    assert t["trend"] in {"up", "sideways"} and 0 <= t["efficiency_ratio"] <= 1


def test_snapshot_contract():
    s = snapshot(synthetic_closes())
    assert {"last_close", "rsi", "rsi_period", "trend", "efficiency_ratio", "divergences", "sr_zones"} <= set(s)


def test_annual_series_uses_fy_10k_full_year_values():
    rows = annual_series(FACTS, "Revenues", years=3)
    assert [r["fy"] for r in rows] == [2023, 2024, 2025]
    assert rows[-1]["value"] == 2_418_000_000  # the quarterly 600M fact inside the 10-K is ignored
    assert annual_series(FACTS, "EarningsPerShareDiluted")[-1]["unit"] == "USD/shares"
    assert annual_series(FACTS, "Assets") == []


def test_fmt_value():
    assert fmt_value(2_418_000_000, "USD") == "$2.42B"
    assert fmt_value(286_000_000, "USD") == "$286.0M"
    assert fmt_value(2.31, "USD/shares") == "$2.31"
