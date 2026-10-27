from __future__ import annotations

from datetime import date

import polars as pl

from app.data.store import ParquetStore

_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d")


def _normalize_date_columns(
    frame: pl.DataFrame, columns: tuple[str, ...]
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column in columns:
        dtype = frame.schema[column]
        if dtype == pl.Date:
            continue
        if dtype == pl.String:
            expressions.append(
                pl.coalesce(
                    pl.col(column).str.strptime(
                        pl.Date, _DATE_FORMATS[0], strict=False
                    ),
                    pl.col(column).str.strptime(
                        pl.Date, _DATE_FORMATS[1], strict=False
                    ),
                ).alias(column)
            )
        else:
            expressions.append(
                pl.col(column).cast(pl.Date, strict=False).alias(column)
            )
    if not expressions:
        return frame
    return frame.with_columns(expressions)


class ParquetMarketDataProvider:
    def __init__(self, store: ParquetStore):
        self.store = store

    def load_bars(self, start: date, end: date) -> pl.DataFrame:
        return self.store.read_processed("bars", start, end)

    def load_index_bars(
        self, index_code: str, start: date, end: date
    ) -> pl.DataFrame:
        frame = self.store.read_processed("index_bars", start, end)
        if frame.is_empty():
            return frame
        return frame.filter(pl.col("ts_code") == index_code)

    def load_stock_basic(self, as_of: date) -> pl.DataFrame:
        frame = self.store.read_reference("stock_basic")
        if frame.is_empty():
            return frame
        frame = _normalize_date_columns(frame, ("list_date", "delist_date"))
        return frame.filter(
            (pl.col("list_date") <= as_of)
            & (pl.col("delist_date").is_null() | (pl.col("delist_date") > as_of))
        )

    def load_namechange(self) -> pl.DataFrame:
        return self.store.read_reference("namechange")

    def load_trade_cal(self, start: date, end: date) -> pl.DataFrame:
        frame = self.store.read_reference("trade_cal")
        if frame.is_empty():
            return frame
        frame = _normalize_date_columns(frame, ("cal_date",))
        return frame.filter((pl.col("cal_date") >= start) & (pl.col("cal_date") <= end))

    def load_index_member(self) -> pl.DataFrame:
        return self.store.read_reference("index_member")
