"""B2 5-minute signal replay.

Only completed bars supplied by the caller are used.  When a stock has no
minute data the code is listed in ``missing`` instead of being replayed from
daily bars.  Same-day PnL is not invented because A-share T+1 forbids selling
on the entry day; execution simulation belongs to the next-day daily engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
from pydantic import BaseModel

from app.core.full_config import FullSystemConfig
from app.strategies.b2_intraday import detect_b2


class MinuteSignalRecord(BaseModel):
    ts_code: str
    signal_time: str
    entry_price: float
    stop_price: float
    frozen_low: float
    frozen_high: float
    vwap: float
    actionable: bool
    valid_until: str


class MinuteReplayResult(BaseModel):
    day: date
    signals: list[MinuteSignalRecord] = []
    missing: list[str] = []
    notes: list[str] = []


@dataclass
class MinuteReplay:
    config: FullSystemConfig

    def run(
        self,
        day: date,
        frames: dict[str, pl.DataFrame],
        *,
        completed_until=None,
    ) -> MinuteReplayResult:
        result = MinuteReplayResult(
            day=day,
            notes=[
                "only completed 5-minute bars are used",
                "same-day exits are not simulated (T+1)",
            ],
        )
        for code in sorted(frames):
            bars = frames[code]
            if bars.is_empty():
                result.missing.append(code)
                continue
            signal = detect_b2(
                code,
                bars,
                self.config.strategies.b,
                completed_until=completed_until,
            )
            if signal is None:
                continue
            result.signals.append(
                MinuteSignalRecord(
                    ts_code=code,
                    signal_time=signal.signal_time.isoformat(),
                    entry_price=signal.entry_price,
                    stop_price=signal.stop_price,
                    frozen_low=signal.frozen_low,
                    frozen_high=signal.frozen_high,
                    vwap=signal.vwap,
                    actionable=signal.actionable,
                    valid_until=signal.valid_until.isoformat(),
                )
            )
        return result
