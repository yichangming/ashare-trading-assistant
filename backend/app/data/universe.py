from __future__ import annotations

from datetime import date, datetime

import polars as pl

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig
from app.core.errors import DataNotReadyError

_NAME_CHANGE_COLUMNS = ("ts_code", "name", "start_date", "end_date")


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def _validate_namechange(namechange: pl.DataFrame) -> None:
    missing = [
        column
        for column in _NAME_CHANGE_COLUMNS
        if column not in namechange.columns
    ]
    if missing:
        raise ValueError(
            f"namechange missing required columns: {', '.join(missing)}"
        )


def _is_st_at(
    namechange: pl.DataFrame, ts_code: str, as_of: date
) -> bool | None:
    if namechange.is_empty():
        return None
    _validate_namechange(namechange)
    rows = namechange.filter(pl.col("ts_code") == ts_code)
    if rows.is_empty():
        return None
    active_st = False
    for row in rows.iter_rows(named=True):
        start = _parse_date(row.get("start_date"))
        end = _parse_date(row.get("end_date"))
        if start is None:
            continue
        if start <= as_of and (end is None or end >= as_of):
            name = str(row.get("name", ""))
            active_st = active_st or "ST" in name or "退" in name
    return active_st


def _empty_universe(features: pl.DataFrame) -> pl.DataFrame:
    return features.head(0).with_columns(
        pl.Series("listing_days", [], dtype=pl.Int64),
        pl.Series("is_st", [], dtype=pl.Boolean),
    )


class UniverseBuilder:
    def __init__(self, calendar: TradingCalendar):
        self.calendar = calendar

    def _check_calendar(self, as_of: date, min_days: int) -> None:
        if not self.calendar.dates:
            raise DataNotReadyError("trading calendar is empty")
        coverage = self.calendar.count_between(
            self.calendar.dates[0], as_of
        )
        if coverage < min_days:
            raise DataNotReadyError(
                f"trading calendar has {coverage} trading days before "
                f"{as_of}; need at least {min_days}"
            )

    def build(
        self,
        as_of: date,
        features: pl.DataFrame,
        stocks: pl.DataFrame,
        namechange: pl.DataFrame,
        config: StrategyConfig,
    ) -> pl.DataFrame:
        template = config.template_a
        self._check_calendar(as_of, template.min_listing_days)
        if "trade_date" not in features.columns:
            raise ValueError("features missing required column: trade_date")
        features = features.filter(pl.col("trade_date") == as_of)
        if features.is_empty() or stocks.is_empty():
            return _empty_universe(features)

        stock_columns = ["ts_code", "name", "market", "industry", "list_date"]
        if "delist_date" in stocks.columns:
            stock_columns.append("delist_date")
        reference = stocks.select(stock_columns)
        if "delist_date" in reference.columns:
            reference = reference.filter(
                pl.col("delist_date").is_null()
                | (pl.col("delist_date") > as_of)
            )
        reference = reference.unique(
            subset=["ts_code"], keep="first", maintain_order=True
        )
        merged = features.join(reference, on="ts_code", how="left")

        listing_days = []
        is_st = []
        for row in merged.iter_rows(named=True):
            list_date = _parse_date(row.get("list_date"))
            listing_days.append(
                self.calendar.count_between(list_date, as_of)
                if list_date is not None
                else 0
            )
            historical_st = _is_st_at(namechange, row["ts_code"], as_of)
            if historical_st is None:
                name = str(row.get("name", ""))
                historical_st = "ST" in name or "退" in name
            is_st.append(historical_st)
        merged = merged.with_columns(
            pl.Series("listing_days", listing_days, dtype=pl.Int64),
            pl.Series("is_st", is_st, dtype=pl.Boolean),
        )
        return merged.filter(
            pl.col("market").is_in(template.allowed_boards)
            & (pl.col("listing_days") >= template.min_listing_days)
            & (~pl.col("is_st"))
            & (~pl.col("is_suspended"))
            & (pl.col("amount_20d_avg") >= template.min_avg_amount)
            & (pl.col("close") >= template.universe_min_price)
        )
