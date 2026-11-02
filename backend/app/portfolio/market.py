"""Normalised market points shared by valuation and position decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from app.domain.enums import ValuationStatus

SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class PricePoint:
    ts_code: str
    open: float
    high: float
    low: float
    close: float
    price: float
    data_time: datetime | None
    status: ValuationStatus = ValuationStatus.OK
    issues: tuple[str, ...] = ()
    source: str = "daily"
    delayed: bool = False

    @property
    def usable(self) -> bool:
        return self.status == ValuationStatus.OK and self.price > 0


def points_from_bars(
    frame: pl.DataFrame,
    as_of: date,
    *,
    source: str = "daily",
) -> dict[str, PricePoint]:
    if frame.is_empty() or "ts_code" not in frame.columns:
        return {}
    points: dict[str, PricePoint] = {}
    data_time = datetime.combine(as_of, time(15, 0), tzinfo=SHANGHAI)
    for row in frame.iter_rows(named=True):
        code = str(row.get("ts_code") or "")
        if not code:
            continue
        try:
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if close <= 0:
            continue
        open_ = float(row.get("open") or close)
        high = float(row.get("high") or close)
        low = float(row.get("low") or close)
        points[code] = PricePoint(
            ts_code=code,
            open=open_,
            high=high,
            low=low,
            close=close,
            price=close,
            data_time=data_time,
            status=ValuationStatus.OK,
            source=source,
        )
    return points
