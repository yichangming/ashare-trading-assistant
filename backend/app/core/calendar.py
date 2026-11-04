from __future__ import annotations

from bisect import bisect_left
from datetime import date, timedelta

import polars as pl

from app.core.dates import parse_date


class TradingCalendar:
    def __init__(
        self,
        dates: list[date],
        *,
        covered_start: date | None = None,
        covered_end: date | None = None,
        closed_dates: list[date] | None = None,
    ):
        self.dates = sorted(set(dates))
        self._index = {item: i for i, item in enumerate(self.dates)}
        self._closed = set(closed_dates or [])
        all_dates = set(self.dates) | self._closed
        self.covered_start = covered_start or (min(all_dates) if all_dates else None)
        self.covered_end = covered_end or (max(all_dates) if all_dates else None)

    def is_trading_day(self, day: date) -> bool:
        return day in self._index

    def covers(self, day: date) -> bool:
        """Return whether the stored calendar explicitly covers ``day``.

        ``is_trading_day`` alone cannot distinguish a holiday from a date
        outside the downloaded calendar range.  Callers must check
        ``covers`` before treating a missing date as a confirmed closure.
        """
        if self.covered_start is None or self.covered_end is None:
            return False
        return self.covered_start <= day <= self.covered_end

    def is_explicit_holiday(self, day: date) -> bool:
        return self.covers(day) and day not in self._index

    def is_covered_trading_day(self, day: date) -> bool:
        return self.covers(day) and day in self._index

    def add_trading_days(self, day: date, count: int) -> date:
        if day not in self._index:
            raise ValueError(f"{day} is not a trading day")
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        target = self._index[day] + count
        if target >= len(self.dates):
            raise ValueError(
                f"calendar coverage ends at {self.dates[-1]}; "
                f"cannot add {count} trading days to {day}"
            )
        return self.dates[target]

    def count_between(self, start: date, end: date) -> int:
        if start > end:
            return 0
        left = bisect_left(self.dates, start)
        right = bisect_left(self.dates, end + timedelta(days=1))
        return max(0, right - left)

    @classmethod
    def from_trade_cal(cls, frame: pl.DataFrame) -> TradingCalendar:
        if frame.is_empty() or "cal_date" not in frame.columns:
            return cls([])
        cal_dates = [parse_date(item) for item in frame["cal_date"].to_list()]
        if "is_open" in frame.columns:
            open_dates = [
                parse_date(cal_date)
                for cal_date, is_open in zip(
                    frame["cal_date"].to_list(),
                    frame["is_open"].to_list(),
                    strict=True,
                )
                if is_open == 1
            ]
        else:
            open_dates = list(cal_dates)
        open_set = set(open_dates)
        return cls(
            open_dates,
            covered_start=min(cal_dates),
            covered_end=max(cal_dates),
            closed_dates=[item for item in cal_dates if item not in open_set],
        )
