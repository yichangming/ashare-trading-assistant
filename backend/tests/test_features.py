from datetime import date, timedelta

import polars as pl
import pytest

from app.indicators.features import FeatureBuilder


def _bars(days: int = 80) -> pl.DataFrame:
    rows = []
    start = date(2023, 10, 1)
    for index in range(days):
        trade_date = start + timedelta(days=index)
        close = 10.0 + index * 0.1
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": close - 0.05,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "pre_close": close - 0.1,
                "vol": 100000.0 + index * 1000,
                "amount": 1000000.0 + index * 10000,
                "adj_factor": 1.0,
                "up_limit": close * 1.1,
                "down_limit": close * 0.9,
                "turnover_rate": 2.0,
                "circ_mv": 10000000000.0,
                "is_suspended": False,
            }
        )
    return pl.DataFrame(rows)


def test_build_features_on_asof():
    bars = _bars()
    as_of = bars["trade_date"][-1]
    features = FeatureBuilder().build(bars, as_of)
    assert features.height == 1
    row = features.row(0, named=True)
    assert row["ma20"] == pytest.approx(
        bars["close"].tail(20).mean()
    )
    assert row["ma60"] == pytest.approx(
        bars["close"].tail(60).mean()
    )
    assert row["ret_20d"] == pytest.approx(
        bars["close"][-1] / bars["close"][-21] - 1
    )
    assert row["close_to_high_60d"] == pytest.approx(
        bars["close"][-1] / bars["high"].tail(60).max()
    )
    assert row["board_height"] == 0
    assert row["rs_20d_market"] == pytest.approx(1.0)


def test_lookahead_truncation_is_invariant():
    bars = _bars()
    as_of = bars["trade_date"][-1]
    full = FeatureBuilder().build(_bars(days=100), as_of)
    truncated = FeatureBuilder().build(
        bars.filter(pl.col("trade_date") <= as_of), as_of
    )
    assert full.to_dicts() == truncated.to_dicts()


def _weekday_bars(
    days: int = 80,
    *,
    peak_index: int | None = None,
    limit_up_index: int | None = None,
) -> pl.DataFrame:
    dates: list[date] = []
    current = date(2023, 10, 2)
    while len(dates) < days:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    rows = []
    for index, trade_date in enumerate(dates):
        close = 10.0 + index * 0.1
        if peak_index is not None and index > peak_index:
            close -= (index - peak_index) * 0.2
        up_limit = close if index == limit_up_index else close * 1.1
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": close - 0.05,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "pre_close": close - 0.1,
                "vol": 100000.0 + index * 1000,
                "amount": 1000000.0 + index * 10000,
                "adj_factor": 1.0,
                "up_limit": up_limit,
                "down_limit": close * 0.9,
                "turnover_rate": 2.0,
                "circ_mv": 10000000000.0,
                "is_suspended": False,
            }
        )
    return pl.DataFrame(rows)


def test_pullback_days_counts_trading_rows():
    bars = _weekday_bars(peak_index=74)
    as_of = bars["trade_date"][-1]
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["pullback_days"] == 5
    assert (as_of - bars["trade_date"][74]).days == 7


def test_days_since_limit_up_counts_trading_rows():
    bars = _weekday_bars(limit_up_index=74)
    as_of = bars["trade_date"][-1]
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["days_since_limit_up"] == 5
    assert (as_of - bars["trade_date"][74]).days == 7
