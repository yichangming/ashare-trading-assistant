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
    bars = _bars(days=100)
    as_of = bars["trade_date"][79]
    future = pl.int_range(0, pl.len()) >= 80
    shocked = bars.with_columns(
        pl.when(future)
        .then(pl.col("close") * 0.5)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(future)
        .then(pl.col("high") * 0.5)
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(future)
        .then(pl.col("low") * 0.5)
        .otherwise(pl.col("low"))
        .alias("low"),
        pl.when(future)
        .then(pl.col("adj_factor") * 10.0)
        .otherwise(pl.col("adj_factor"))
        .alias("adj_factor"),
    )
    full = FeatureBuilder().build(shocked, as_of)
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


def _series_bars(
    ts_code: str,
    days: int,
    *,
    base: float = 10.0,
    step: float = 0.1,
) -> pl.DataFrame:
    start = date(2023, 10, 1) + timedelta(days=80 - days)
    rows = []
    for index in range(days):
        close = base + index * step
        rows.append(
            {
                "ts_code": ts_code,
                "trade_date": start + timedelta(days=index),
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


def _with_limit_ups(
    bars: pl.DataFrame, indices: tuple[int, ...]
) -> pl.DataFrame:
    return bars.with_columns(
        pl.when(pl.int_range(0, pl.len()).is_in(list(indices)))
        .then(pl.col("close"))
        .otherwise(pl.col("up_limit"))
        .alias("up_limit")
    )


def test_prev_close_uses_exchange_pre_close():
    bars = _weekday_bars()
    as_of = bars["trade_date"][-1]
    bars = bars.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 79)
        .then(123.45)
        .otherwise(pl.col("pre_close"))
        .alias("pre_close")
    )
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["prev_close"] == pytest.approx(123.45)

    fallback = FeatureBuilder().build(
        bars.drop("pre_close"), as_of
    ).row(0, named=True)
    assert fallback["prev_close"] == pytest.approx(bars["close"][-2])


def test_prev_is_limit_up_reflects_prior_day():
    bars = _with_limit_ups(_weekday_bars(), (78,))
    as_of = bars["trade_date"][-1]
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["is_limit_up"] is False
    assert row["prev_is_limit_up"] is True


def test_board_height_streak_and_reset():
    as_of = _weekday_bars()["trade_date"][79]
    bars = _with_limit_ups(_weekday_bars(), (74, 75, 76, 77, 78, 79))
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["board_height"] == 6

    bars = _with_limit_ups(_weekday_bars(), (74, 75, 76))
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["board_height"] == 0

    bars = _with_limit_ups(_weekday_bars(), (74, 75, 76, 78))
    row = FeatureBuilder().build(bars, bars["trade_date"][78]).row(0, named=True)
    assert row["board_height"] == 1


def test_atr14_and_raw():
    bars = _bars()
    row = FeatureBuilder().build(
        bars, bars["trade_date"][-1]
    ).row(0, named=True)
    assert row["atr14"] == pytest.approx(0.4)
    assert row["atr14_raw"] == pytest.approx(0.4)


def test_adjusted_raw_columns_use_adj_factor():
    bars = _bars().with_columns(pl.lit(2.0).alias("adj_factor"))
    as_of = bars["trade_date"][-1]
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["adj_factor"] == 2.0
    assert row["ma20_raw"] == pytest.approx(bars["close"].tail(20).mean())
    assert row["atr14_raw"] == pytest.approx(0.4)
    assert row["low_5d_raw"] == pytest.approx(bars["low"].tail(5).min())
    assert row["high_60d_raw"] == pytest.approx(bars["high"].tail(60).max())


def test_limit_flags_and_one_word():
    bars = _with_limit_ups(_weekday_bars(), (79,))
    as_of = bars["trade_date"][-1]
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["is_limit_up"] is True
    assert row["touched_limit_up"] is True
    assert row["one_word"] is False

    zero_range = bars.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 79)
        .then(pl.col("close"))
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.int_range(0, pl.len()) == 79)
        .then(pl.col("close"))
        .otherwise(pl.col("low"))
        .alias("low"),
    )
    row = FeatureBuilder().build(zero_range, as_of).row(0, named=True)
    assert row["one_word"] is True
    assert row["is_limit_up"] is True
    assert row["touched_limit_up"] is True
    assert row["stabilization"] is False

    down = _weekday_bars().with_columns(
        pl.when(pl.int_range(0, pl.len()) == 79)
        .then(pl.col("close"))
        .otherwise(pl.col("down_limit"))
        .alias("down_limit")
    )
    row = FeatureBuilder().build(down, as_of).row(0, named=True)
    assert row["is_limit_down"] is True


def test_non_positive_limit_price_never_flags():
    idx = pl.int_range(0, pl.len())
    bars = _weekday_bars().with_columns(
        pl.when(idx == 79)
        .then(0.0)
        .otherwise(pl.col("up_limit"))
        .alias("up_limit")
    )
    as_of = bars["trade_date"][-1]
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["is_limit_up"] is False
    assert row["touched_limit_up"] is False

    null_bars = bars.with_columns(
        pl.when(idx == 79)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("up_limit"))
        .alias("up_limit")
    )
    row = FeatureBuilder().build(null_bars, as_of).row(0, named=True)
    assert row["is_limit_up"] is False
    assert row["touched_limit_up"] is False


def test_rs_20d_market_ignores_null_returns():
    a = _series_bars("000001.SZ", 80, base=10.0, step=0.1)
    b = _series_bars("000002.SZ", 80, base=20.0, step=0.05)
    c = _series_bars("000003.SZ", 10, base=30.0, step=0.1)
    bars = pl.concat([a, b, c])
    as_of = a["trade_date"][-1]
    features = FeatureBuilder().build(bars, as_of)
    by_code = {row["ts_code"]: row for row in features.iter_rows(named=True)}
    assert by_code["000003.SZ"]["ret_20d"] is None
    assert by_code["000001.SZ"]["rs_20d_market"] == pytest.approx(1.0)
    assert by_code["000002.SZ"]["rs_20d_market"] == pytest.approx(0.5)
    assert by_code["000003.SZ"]["rs_20d_market"] is None


def test_insufficient_history_uses_sentinels():
    bars = _bars(days=10)
    row = FeatureBuilder().build(
        bars, bars["trade_date"][-1]
    ).row(0, named=True)
    assert row["days_since_limit_up"] == 999
    assert row["pullback_days"] == 999
    assert row["ret_20d"] is None
    assert row["atr14"] is None


def test_trading_day_offsets_count_suspended_rows():
    bars = _with_limit_ups(_weekday_bars(peak_index=70), (70,))
    idx = pl.int_range(0, pl.len())
    suspended = (idx >= 71) & (idx <= 74)
    bars = bars.with_columns(
        pl.when(suspended).then(0.0).otherwise(pl.col("vol")).alias("vol"),
        suspended.alias("is_suspended"),
    )
    as_of = bars["trade_date"][-1]
    row = FeatureBuilder().build(bars, as_of).row(0, named=True)
    assert row["pullback_days"] == 9
    assert row["days_since_limit_up"] == 9
    assert (as_of - bars["trade_date"][70]).days > 9
