from __future__ import annotations

from datetime import date

import polars as pl

from app.data.store import ParquetStore


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
        list_date = pl.col("list_date").cast(pl.Date, strict=False)
        delist_date = pl.col("delist_date").cast(pl.Date, strict=False)
        return frame.filter(
            (list_date <= as_of)
            & (delist_date.is_null() | (delist_date > as_of))
        )

    def load_namechange(self) -> pl.DataFrame:
        return self.store.read_reference("namechange")

    def load_trade_cal(self, start: date, end: date) -> pl.DataFrame:
        frame = self.store.read_reference("trade_cal")
        if frame.is_empty():
            return frame
        cal_date = pl.col("cal_date").cast(pl.Date, strict=False)
        return frame.filter((cal_date >= start) & (cal_date <= end))

    def load_index_member(self) -> pl.DataFrame:
        return self.store.read_reference("index_member")
