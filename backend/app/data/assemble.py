from __future__ import annotations

from datetime import date

import polars as pl

from app.core.errors import DataNotReadyError
from app.data.store import ParquetStore


def _normalize_daily(frame: pl.DataFrame) -> pl.DataFrame:
    numeric = ["open", "high", "low", "close", "pre_close", "vol"]
    return frame.with_columns(
        [
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y%m%d", strict=False)
            .alias("trade_date"),
            *[pl.col(column).cast(pl.Float64) for column in numeric],
            (pl.col("amount").cast(pl.Float64) * 1000).alias("amount"),
        ]
    )


def _normalize_keyed(
    frame: pl.DataFrame, numeric: list[str], date_columns: list[str] | None = None
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    expressions = [
        pl.col("trade_date")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("trade_date"),
        *[pl.col(column).cast(pl.Float64) for column in numeric if column in frame.columns],
    ]
    for column in date_columns or []:
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias(column)
            )
    return frame.with_columns(expressions)


def _derive_limits(bars: pl.DataFrame, stocks: pl.DataFrame) -> pl.DataFrame:
    market = pl.col("market").fill_null("主板")
    is_st = pl.col("name").str.contains("ST").fill_null(False)
    base_limit_pct = (
        pl.when(market == "创业板")
        .then(0.20)
        .when(market == "科创板")
        .then(0.20)
        .when(market == "北交所")
        .then(0.30)
        .otherwise(0.10)
    )
    limit_pct = (
        pl.when(is_st & (market == "主板"))
        .then(0.05)
        .otherwise(base_limit_pct)
    )
    joined = bars.join(
        stocks.select(["ts_code", "name", "market"]),
        on="ts_code",
        how="left",
    )
    return joined.with_columns(
        limit_pct.alias("limit_pct"),
        (pl.col("pre_close") * (1 + limit_pct)).round(2).alias("up_limit"),
        (pl.col("pre_close") * (1 - limit_pct)).round(2).alias("down_limit"),
        pl.lit(True).alias("limit_derived"),
    )


class DataAssembler:
    def __init__(self, store: ParquetStore):
        self.store = store

    def build_day(self, trade_date: date) -> pl.DataFrame:
        key = trade_date.strftime("%Y%m%d")
        daily = self.store.read_raw("daily", key)
        if daily.is_empty():
            raise DataNotReadyError(f"daily data missing for {trade_date}")

        bars = _normalize_daily(daily)
        adj = self.store.read_raw("adj_factor", key)
        if not adj.is_empty():
            adj = _normalize_keyed(adj, ["adj_factor"])
            bars = bars.join(adj, on=["ts_code", "trade_date"], how="left")
        else:
            bars = bars.with_columns(pl.lit(1.0).alias("adj_factor"))
        bars = bars.with_columns(pl.col("adj_factor").fill_null(1.0))

        basic = self.store.read_raw("daily_basic", key)
        if not basic.is_empty():
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
        else:
            bars = bars.with_columns(
                pl.lit(None).cast(pl.Float64).alias("turnover_rate"),
                pl.lit(None).cast(pl.Float64).alias("circ_mv"),
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
            stocks = self.store.read_reference("stock_basic")
            bars = _derive_limits(bars, stocks)

        suspend = self.store.read_raw("suspend_d", key)
        if not suspend.is_empty():
            suspended_codes = suspend["ts_code"].unique().to_list()
            bars = bars.with_columns(
                pl.col("ts_code").is_in(suspended_codes).alias("is_suspended")
            )
        else:
            bars = bars.with_columns((pl.col("vol") == 0).alias("is_suspended"))

        self.store.write_processed("bars", trade_date, bars)

        index_raw = self.store.read_raw("index_daily", key)
        if not index_raw.is_empty():
            index_bars = _normalize_keyed(
                index_raw, ["open", "high", "low", "close", "vol", "amount"]
            )
            self.store.write_processed("index_bars", trade_date, index_bars)

        limit_list = self.store.read_raw("limit_list_d", key)
        if not limit_list.is_empty():
            normalized = limit_list.with_columns(
                pl.col("trade_date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias("trade_date")
            )
            self.store.write_processed("limit_list", trade_date, normalized)
        return bars
