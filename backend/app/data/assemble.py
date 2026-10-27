from __future__ import annotations

from datetime import date

import polars as pl

from app.core.errors import DataNotReadyError
from app.data.store import ParquetStore

_STOCK_REQUIRED = ("ts_code", "name", "market")
_ST_NAME_PATTERN = "ST|退"


def _normalize_date_column(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    dtype = frame.schema[column]
    if dtype == pl.Date:
        return frame
    if dtype == pl.Utf8:
        text = pl.col(column)
        parsed = (
            pl.when(text.str.len_chars() == 8)
            .then(text.str.strptime(pl.Date, "%Y%m%d", strict=False))
            .otherwise(text.str.strptime(pl.Date, "%Y-%m-%d", strict=False))
        )
        return frame.with_columns(parsed.alias(column))
    return frame.with_columns(pl.col(column).cast(pl.Date).alias(column))


def _normalize_daily(frame: pl.DataFrame) -> pl.DataFrame:
    numeric = ["open", "high", "low", "close", "pre_close", "vol"]
    normalized = _normalize_date_column(frame, "trade_date")
    return normalized.with_columns(
        [
            *[pl.col(column).cast(pl.Float64) for column in numeric],
            (pl.col("amount").cast(pl.Float64) * 1000).alias("amount"),
        ]
    )


def _normalize_keyed(frame: pl.DataFrame, numeric: list[str]) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    normalized = _normalize_date_column(frame, "trade_date")
    casts = [
        pl.col(column).cast(pl.Float64)
        for column in numeric
        if column in normalized.columns
    ]
    return normalized.with_columns(casts)


def _validate_stock_basic(stocks: pl.DataFrame) -> None:
    if stocks.is_empty():
        raise DataNotReadyError("stock_basic reference is missing or empty")
    missing = [column for column in _STOCK_REQUIRED if column not in stocks.columns]
    if missing:
        raise DataNotReadyError(
            f"stock_basic missing required columns: {', '.join(missing)}"
        )


def _st_flags(
    stocks: pl.DataFrame, namechange: pl.DataFrame, trade_date: date
) -> pl.DataFrame:
    current = stocks.select(
        "ts_code",
        pl.col("name")
        .str.contains(_ST_NAME_PATTERN)
        .fill_null(False)
        .alias("_current_st"),
        pl.col("market").fill_null("主板").alias("market"),
    )
    history_columns = {"ts_code", "name", "start_date"}
    if namechange.is_empty() or not history_columns <= set(namechange.columns):
        return current.with_columns(pl.col("_current_st").alias("is_st")).select(
            "ts_code", "market", "is_st"
        )

    history = _normalize_date_column(namechange, "start_date")
    if "end_date" in history.columns:
        history = _normalize_date_column(history, "end_date")
    else:
        history = history.with_columns(pl.lit(None, dtype=pl.Date).alias("end_date"))
    active = history.filter(
        (pl.col("start_date") <= pl.lit(trade_date))
        & (pl.col("end_date").is_null() | (pl.col("end_date") >= pl.lit(trade_date)))
    )
    historical = active.group_by("ts_code").agg(
        pl.col("name")
        .str.contains(_ST_NAME_PATTERN)
        .fill_null(False)
        .any()
        .alias("_historical_st")
    )
    return (
        current.join(historical, on="ts_code", how="left")
        .with_columns(
            pl.coalesce(
                [pl.col("_historical_st"), pl.col("_current_st")]
            ).alias("is_st")
        )
        .select("ts_code", "market", "is_st")
    )


def _derive_limits(bars: pl.DataFrame, st_flags: pl.DataFrame) -> pl.DataFrame:
    market = pl.col("market").fill_null("主板")
    is_st = pl.col("is_st").fill_null(False)
    limit_bps = (
        pl.when(is_st & (market == "主板"))
        .then(500)
        .when(market == "创业板")
        .then(2000)
        .when(market == "科创板")
        .then(2000)
        .when(market == "北交所")
        .then(3000)
        .otherwise(1000)
    )
    joined = bars.join(st_flags, on="ts_code", how="left")
    pre_close_cents = (pl.col("pre_close") * 100).round(0).cast(pl.Int64)
    up_cents = (pre_close_cents * (10000 + limit_bps) + 5000) // 10000
    down_cents = (pre_close_cents * (10000 - limit_bps) + 5000) // 10000
    return joined.with_columns(
        (up_cents.cast(pl.Float64) / 100).alias("up_limit"),
        (down_cents.cast(pl.Float64) / 100).alias("down_limit"),
        pl.lit(True).alias("limit_derived"),
    ).drop(["market", "is_st"])


def _apply_suspension(bars: pl.DataFrame, suspend: pl.DataFrame) -> pl.DataFrame:
    suspended_codes: set[str] = set()
    if not suspend.is_empty() and "ts_code" in suspend.columns:
        if "suspend_type" in suspend.columns:
            suspend = suspend.filter(pl.col("suspend_type") == "S")
        if "suspend_timing" in suspend.columns:
            # Partial-day suspensions still trade for part of the session;
            # only null/empty timing denotes a full-day suspension.
            timing = (
                pl.col("suspend_timing")
                .cast(pl.String, strict=False)
                .str.strip_chars()
            )
            suspend = suspend.filter(
                pl.col("suspend_timing").is_null() | (timing == "")
            )
        suspended_codes = set(suspend["ts_code"].to_list())
    return bars.with_columns(
        (
            pl.col("ts_code").is_in(sorted(suspended_codes))
            | (pl.col("vol") == 0)
            | (pl.col("amount") == 0)
        )
        .fill_null(False)
        .alias("is_suspended")
    )


class DataAssembler:
    def __init__(self, store: ParquetStore):
        self.store = store

    def build_day(self, trade_date: date) -> pl.DataFrame:
        key = trade_date.strftime("%Y%m%d")
        daily = self.store.read_raw("daily", key)
        if daily.is_empty():
            raise DataNotReadyError(f"daily data missing for {trade_date}")
        adj = self.store.read_raw("adj_factor", key)
        if adj.is_empty():
            raise DataNotReadyError(f"adj_factor data missing for {trade_date}")
        basic = self.store.read_raw("daily_basic", key)
        if basic.is_empty():
            raise DataNotReadyError(f"daily_basic data missing for {trade_date}")
        index_raw = self.store.read_raw("index_daily", key)
        if index_raw.is_empty():
            raise DataNotReadyError(f"index_daily data missing for {trade_date}")

        stocks = self.store.read_reference("stock_basic")
        _validate_stock_basic(stocks)

        bars = _normalize_daily(daily)
        adj = _normalize_keyed(adj, ["adj_factor"])
        bars = bars.join(adj, on=["ts_code", "trade_date"], how="left")

        basic = _normalize_keyed(basic, ["turnover_rate", "circ_mv"])
        basic = basic.with_columns(
            (pl.col("turnover_rate") / 100).alias("turnover_rate"),
            (pl.col("circ_mv") * 10000).alias("circ_mv"),
        )
        bars = bars.join(
            basic.select(["ts_code", "trade_date", "turnover_rate", "circ_mv"]),
            on=["ts_code", "trade_date"],
            how="left",
        )

        limits = self.store.read_raw("stk_limit", key)
        if not limits.is_empty():
            limits = _normalize_keyed(limits, ["up_limit", "down_limit"])
            bars = bars.join(
                limits.select(["ts_code", "trade_date", "up_limit", "down_limit"]),
                on=["ts_code", "trade_date"],
                how="left",
            ).with_columns(pl.lit(False).alias("limit_derived"))
        else:
            namechange = self.store.read_reference("namechange")
            st_flags = _st_flags(stocks, namechange, trade_date)
            bars = _derive_limits(bars, st_flags)

        suspend = self.store.read_raw("suspend_d", key)
        bars = _apply_suspension(bars, suspend)

        index_bars = _normalize_keyed(
            index_raw, ["open", "high", "low", "close", "vol", "amount"]
        )
        index_bars = index_bars.with_columns(
            (pl.col("amount") * 1000).alias("amount")
        )

        limit_list = self.store.read_raw("limit_list_d", key)
        if not limit_list.is_empty():
            limit_list = _normalize_date_column(limit_list, "trade_date")

        self.store.write_processed("bars", trade_date, bars)
        self.store.write_processed("index_bars", trade_date, index_bars)
        if not limit_list.is_empty():
            self.store.write_processed("limit_list", trade_date, limit_list)
        return bars
