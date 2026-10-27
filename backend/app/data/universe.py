from __future__ import annotations

from datetime import date

import polars as pl

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig


def _is_st_at(namechange: pl.DataFrame, ts_code: str, as_of: date) -> bool:
    if namechange.is_empty():
        return False
    rows = namechange.filter(pl.col("ts_code") == ts_code)
    for row in rows.iter_rows(named=True):
        start = row.get("start_date")
        end = row.get("end_date")
        if start is None:
            continue
        if start <= as_of and (end is None or end >= as_of):
            name = str(row.get("name", ""))
            if "ST" in name or "退" in name:
                return True
    return False


class UniverseBuilder:
    def __init__(self, calendar: TradingCalendar):
        self.calendar = calendar

    def build(
        self,
        as_of: date,
        features: pl.DataFrame,
        stocks: pl.DataFrame,
        namechange: pl.DataFrame,
        config: StrategyConfig,
    ) -> pl.DataFrame:
        if features.is_empty() or stocks.is_empty():
            return pl.DataFrame()
        merged = features.join(
            stocks.select(
                [
                    "ts_code",
                    "name",
                    "market",
                    "industry",
                    "list_date",
                    "delist_date",
                ]
            ),
            on="ts_code",
            how="left",
        )
        listing_days = []
        is_st = []
        for row in merged.iter_rows(named=True):
            list_date = row.get("list_date")
            listing_days.append(
                self.calendar.count_between(list_date, as_of)
                if list_date is not None
                else 0
            )
            name = str(row.get("name", ""))
            name_is_st = "ST" in name or "退" in name
            is_st.append(
                _is_st_at(namechange, row["ts_code"], as_of) or name_is_st
            )
        merged = merged.with_columns(
            pl.Series("listing_days", listing_days, dtype=pl.Int64),
            pl.Series("is_st", is_st, dtype=pl.Boolean),
        )
        template = config.template_a
        return merged.filter(
            pl.col("market").is_in(template.allowed_boards)
            & (pl.col("listing_days") >= template.min_listing_days)
            & (~pl.col("is_st"))
            & (~pl.col("is_suspended"))
            & (pl.col("amount_20d_avg") >= template.min_avg_amount)
            & (pl.col("close") >= template.universe_min_price)
        )
