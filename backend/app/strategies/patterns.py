from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev

import polars as pl


def _vals(history: pl.DataFrame, column: str) -> list[float]:
    if column not in history.columns:
        return []
    return [
        float(value)
        for value in history[column].to_list()
        if value is not None
    ]


def ma_at(history: pl.DataFrame, index: int, window: int, column: str = "close") -> float | None:
    if index < window - 1:
        return None
    values = _vals(history, column)
    if index >= len(values):
        return None
    window_values = values[index - window + 1 : index + 1]
    return fmean(window_values) if len(window_values) == window else None


def atr_at(history: pl.DataFrame, index: int, window: int = 14) -> float | None:
    if index < 1:
        return None
    highs = _vals(history, "high")
    lows = _vals(history, "low")
    closes = _vals(history, "close")
    if max(index, len(highs) - 1) >= len(highs):
        return None
    start = max(1, index - window + 1)
    ranges: list[float] = []
    for cursor in range(start, index + 1):
        prev_close = closes[cursor - 1]
        ranges.append(
            max(
                highs[cursor] - lows[cursor],
                abs(highs[cursor] - prev_close),
                abs(lows[cursor] - prev_close),
            )
        )
    if not ranges:
        return None
    return fmean(ranges)


def returns(history: pl.DataFrame, periods: int) -> list[float | None]:
    closes = _vals(history, "close")
    result: list[float | None] = [None] * len(closes)
    for index in range(periods, len(closes)):
        base = closes[index - periods]
        if base:
            result[index] = closes[index] / base - 1
    return result


def swing_high(
    history: pl.DataFrame, end_index: int, window: int
) -> dict | None:
    """Highest high in the W bars before the end index, excluding the end bar."""
    if end_index < 3:
        return None
    start = max(0, end_index - window)
    stop = end_index  # exclusive
    if stop - start < 3:
        return None
    highs = _vals(history, "high")
    lows = _vals(history, "low")
    if stop > len(highs):
        return None
    # The latest occurrence wins ties ("最后一个交易日").
    rel = max(range(start, stop), key=lambda i: (highs[i], i))
    swing_high_value = highs[rel]
    if swing_high_value <= 0:
        return None
    if (rel > 0 and swing_high_value <= highs[rel - 1]) or (
        rel + 1 < len(highs) and swing_high_value <= highs[rel + 1]
    ):
        return None
    # Include the as-of bar in the pullback measurement; the high itself is
    # taken from the preceding window so the current bar cannot define it.
    after = lows[rel + 1 : end_index + 1] or [lows[rel]]
    depth = (swing_high_value - min(after)) / swing_high_value
    return {
        "swing_index": rel,
        "swing_high": swing_high_value,
        "duration": end_index - rel,
        "depth": depth,
        "pullback_low": min(after),
    }


def linear_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = fmean(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(n))
    return numerator / denominator if denominator else 0.0


def _touch_positions(
    values: list[float],
    threshold: float,
    *,
    above: bool,
) -> list[int]:
    return [
        index
        for index, value in enumerate(values)
        if (value >= threshold if above else value <= threshold)
    ]


def _touch_structure(indices: list[int], spacing: int, window: int) -> bool:
    if len(indices) < 3:
        return False
    spaced = [indices[0]]
    for index in indices[1:]:
        if index - spaced[-1] >= spacing:
            spaced.append(index)
    return len(spaced) >= 3 and (spaced[-1] - spaced[0]) >= window / 2


@dataclass
class BoxStats:
    upper: float
    lower: float
    width: float
    normalized_slope: float
    atr: float
    upper_touches: int
    lower_touches: int


def box_stats(
    history: pl.DataFrame,
    *,
    window: int,
    upper_quantile: float,
    lower_quantile: float,
    touch_atr: float,
    min_touches: int,
    touch_spacing: int,
    width_min: float,
    width_max: float,
    slope_max: float | None = None,
    slope_min: float | None = None,
    abnormal_atr: float | None = None,
) -> BoxStats | None:
    """Detect a frozen box from a window that ends *before* the breakout day."""
    if history.height < window + 1:
        return None
    window_frame = history.tail(window)
    highs = _vals(window_frame, "high")
    lows = _vals(window_frame, "low")
    closes = _vals(window_frame, "close")
    if len(highs) < window or len(lows) < window or len(closes) < window:
        return None
    atr = atr_at(history, history.height - 1)
    if atr is None or atr <= 0:
        return None
    upper = float(pl.Series(highs).quantile(upper_quantile, interpolation="linear"))
    lower = float(pl.Series(lows).quantile(lower_quantile, interpolation="linear"))
    if lower <= 0:
        return None
    width = (upper - lower) / lower
    if not (width_min <= width <= width_max):
        return None
    mean_close = fmean(closes)
    if mean_close <= 0:
        return None
    normalized_slope = linear_slope(closes) / mean_close
    if slope_max is not None and abs(normalized_slope) > slope_max:
        return None
    if slope_min is not None and normalized_slope < slope_min:
        return None
    upper_hits = _touch_positions(highs, upper - touch_atr * atr, above=True)
    lower_hits = _touch_positions(lows, lower + touch_atr * atr, above=False)
    if not _touch_structure(upper_hits, touch_spacing, window):
        return None
    if not _touch_structure(lower_hits, touch_spacing, window):
        return None
    if len(upper_hits) < min_touches or len(lower_hits) < min_touches:
        return None
    if abnormal_atr is not None:
        for high, low in zip(highs[-5:], lows[-5:], strict=False):
            if high > upper + abnormal_atr * atr or low < lower - abnormal_atr * atr:
                return None
    return BoxStats(
        upper=upper,
        lower=lower,
        width=width,
        normalized_slope=normalized_slope,
        atr=atr,
        upper_touches=len(upper_hits),
        lower_touches=len(lower_hits),
    )


def two_day_structure(
    history: pl.DataFrame, d2_index: int
) -> dict | None:
    if d2_index < 2 or d2_index >= history.height:
        return None
    rows = history.slice(d2_index - 2, 3).to_dicts()
    first_prior, first, second = rows
    lows_ok = (
        first_prior["low"] is not None
        and first["low"] is not None
        and second["low"] is not None
        and float(first["low"]) >= float(first_prior["low"])
        and float(second["low"]) >= float(first["low"])
    )
    if not lows_ok:
        return None
    return {
        "frozen_high": max(float(first["high"]), float(second["high"])),
        "observation_low": min(float(first["low"]), float(second["low"])),
        "d1_index": d2_index - 1,
        "d2_index": d2_index,
    }


def stabilization_bar(row: dict, previous_close: float | None) -> bool:
    open_ = row.get("open")
    close = row.get("close")
    high = row.get("high")
    low = row.get("low")
    if None in (open_, close, high, low):
        return False
    body = abs(float(close) - float(open_))
    lower_shadow = min(float(open_), float(close)) - float(low)
    prev_ok = previous_close is None or float(close) >= previous_close
    return float(close) >= float(open_) and (prev_ok or lower_shadow >= body)


def zscore(values: list[float], value: float) -> float | None:
    if len(values) < 5:
        return None
    deviation = pstdev(values)
    if deviation == 0:
        return None
    return (value - fmean(values)) / deviation


def drawdown_from_high(history: pl.DataFrame, window: int) -> float | None:
    if history.height < window:
        return None
    frame = history.tail(window)
    highs = _vals(frame, "high")
    closes = _vals(frame, "close")
    if not highs or not closes or max(highs) <= 0:
        return None
    return (max(highs) - closes[-1]) / max(highs)
