"""Normalised market points shared by valuation and position decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from app.core.calendar import TradingCalendar
from app.domain.enums import ValuationStatus

SHANGHAI = ZoneInfo("Asia/Shanghai")
MARKET_OPEN = time(9, 30)
MORNING_CLOSE = time(11, 30)
AFTERNOON_OPEN = time(13, 0)
MARKET_CLOSE = time(15, 0)


def market_phase(
    now: datetime, calendar: TradingCalendar | None = None
) -> str:
    """Return the current A-share session phase.

    ``unknown`` deliberately covers a calendar coverage gap; callers must not
    treat it as a confirmed holiday.
    """
    moment = now.astimezone(SHANGHAI) if now.tzinfo else now.replace(tzinfo=SHANGHAI)
    day = moment.date()
    if calendar is not None:
        if not calendar.covers(day):
            return "unknown"
        if not calendar.is_trading_day(day):
            return "holiday"
    elif day.weekday() >= 5:
        return "weekend"
    clock = moment.time()
    if clock < MARKET_OPEN:
        return "pre_open"
    if clock <= MORNING_CLOSE:
        return "trading"
    if clock < AFTERNOON_OPEN:
        return "lunch"
    if clock <= MARKET_CLOSE:
        return "trading"
    return "post_close"


def is_market_open_phase(phase: str) -> bool:
    return phase == "trading"


def next_session_open(
    day: date, calendar: TradingCalendar | None = None
) -> datetime:
    """First 09:30 after ``day`` where the protection line may take effect."""
    candidate = day
    for _ in range(14):
        candidate = candidate + timedelta(days=1)
        if calendar is not None and calendar.covers(candidate):
            if calendar.is_trading_day(candidate):
                return datetime.combine(candidate, MARKET_OPEN, tzinfo=SHANGHAI)
            continue
        if candidate.weekday() < 5:
            return datetime.combine(candidate, MARKET_OPEN, tzinfo=SHANGHAI)
    return datetime.combine(candidate, MARKET_OPEN, tzinfo=SHANGHAI)


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
