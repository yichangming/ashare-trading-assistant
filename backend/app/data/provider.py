from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.data.store import ParquetStore

_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d")
_SUSPEND_SCHEMA = {
    "trade_date": pl.Date,
    "ts_code": pl.String,
    "suspend_type": pl.String,
    "suspend_timing": pl.String,
}


def _empty_suspend() -> pl.DataFrame:
    return pl.DataFrame(schema=_SUSPEND_SCHEMA)


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

    def load_suspend(self, start: date, end: date) -> pl.DataFrame | None:
        """Load ``suspend_d`` raw partitions for ``[start, end]``.

        Returns ``None`` only when no raw partition exists anywhere in the
        requested range, which signals that the optional endpoint has not been
        synced.  When at least one partition exists, returns a typed frame
        containing only ``suspend_type == "S"`` rows; the result may be empty
        (for example an R-only or zero-suspension day), which means the exact
        expected universe is known and has no suspended names.  ``trade_date``
        is normalized to :class:`polars.Date`.
        """
        if start > end:
            return None
        partition_exists = False
        frames: list[pl.DataFrame] = []
        current = start
        while current <= end:
            key = current.strftime("%Y%m%d")
            if self.store.raw_path("suspend_d", key).exists():
                partition_exists = True
                frame = self.store.read_raw("suspend_d", key)
                if not frame.is_empty():
                    frames.append(frame)
            current += timedelta(days=1)
        if not partition_exists:
            return None
        if not frames:
            return _empty_suspend()
        merged = pl.concat(frames, how="vertical_relaxed")
        if "trade_date" not in merged.columns or "ts_code" not in merged.columns:
            return _empty_suspend()
        merged = _normalize_date_columns(merged, ("trade_date",))
        merged = merged.filter(
            (pl.col("trade_date") >= start) & (pl.col("trade_date") <= end)
        )
        if "suspend_type" in merged.columns:
            merged = merged.filter(pl.col("suspend_type") == "S")
        if "suspend_timing" in merged.columns:
            merged = merged.with_columns(
                pl.col("suspend_timing")
                .cast(pl.String, strict=False)
                .alias("suspend_timing")
            )
        if merged.is_empty():
            return _empty_suspend()
        return merged

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

    def load_index_classify(self) -> pl.DataFrame:
        return self.store.read_reference("index_classify")

    def load_limit_list(self, start: date, end: date) -> pl.DataFrame:
        return self.store.read_processed("limit_list", start, end)
