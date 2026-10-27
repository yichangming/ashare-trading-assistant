from __future__ import annotations

from bisect import bisect_left
from datetime import date, timedelta

import polars as pl

from app.core.dates import parse_date


class TradingCalendar:
    def __init__(self, dates: list[date]):
        self.dates = sorted(set(dates))
        self._index = {item: i for i, item in enumerate(self.dates)}

    def is_trading_day(self, day: date) -> bool:
        return day in self._index

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
        if frame.is_empty():
            return cls([])
        dates = [
            parse_date(item)
            for item in frame.filter(pl.col("is_open") == 1)["cal_date"].to_list()
        ]
        return cls(dates)
