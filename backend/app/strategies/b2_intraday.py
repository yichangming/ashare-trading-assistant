"""5-minute B2 state machine.

The implementation is deliberately bar-close driven: no intrabar path is
invented.  A signal becomes actionable only after a completed bar closes above
the frozen high and above the day's cumulative VWAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from statistics import fmean
from zoneinfo import ZoneInfo

import polars as pl

from app.core.full_config import StrategyBConfig
from app.strategies.patterns import stabilization_bar

SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass
class B2Signal:
    ts_code: str
    signal_time: datetime
    entry_price: float
    frozen_low: float
    frozen_high: float
    vwap: float
    atr5: float
    stop_price: float
    stop_distance_pct: float
    actionable: bool
    vwap_approximate: bool
    reason: str
    valid_until: datetime


def _as_datetime(value) -> datetime | None:
    moment: datetime | None = None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
    if moment is None:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=SHANGHAI)


def _session_ok(moment: datetime) -> bool:
    current = moment.time()
    return (
        time(9, 30) <= current <= time(11, 30)
        or time(13, 0) <= current <= time(15, 0)
    )


def _atr5(rows: list[dict], end_index: int, window: int = 14) -> float:
    start = max(1, end_index - window + 1)
    ranges: list[float] = []
    for index in range(start, end_index + 1):
        row = rows[index]
        prev = rows[index - 1]
        ranges.append(
            max(
                float(row["high"]) - float(row["low"]),
                abs(float(row["high"]) - float(prev["close"])),
                abs(float(row["low"]) - float(prev["close"])),
            )
        )
    return fmean(ranges) if ranges else max(0.01, float(rows[end_index]["close"]) * 0.002)


def detect_b2(
    ts_code: str,
    bars: pl.DataFrame,
    cfg: StrategyBConfig,
    *,
    completed_until: datetime | None = None,
) -> B2Signal | None:
    if bars.is_empty():
        return None
    if completed_until is not None and completed_until.tzinfo is None:
        completed_until = completed_until.replace(tzinfo=SHANGHAI)
    required = {"trade_time", "open", "high", "low", "close", "vol", "amount"}
    if not required <= set(bars.columns):
        return None
    rows: list[dict] = []
    for row in bars.sort("trade_time").iter_rows(named=True):
        moment = _as_datetime(row.get("trade_time"))
        if moment is None or not _session_ok(moment):
            continue
        if completed_until is not None and moment > completed_until:
            continue
        row = dict(row)
        row["trade_time"] = moment
        rows.append(row)
    if len(rows) < 4:
        return None

    cumulative_amount = 0.0
    cumulative_volume = 0.0
    vwap_values: list[float] = []
    approximate = False
    for row in rows:
        volume = float(row.get("vol") or 0.0)
        amount = float(row.get("amount") or 0.0)
        cumulative_amount += amount
        cumulative_volume += volume
        if cumulative_volume > 0:
            vwap = cumulative_amount / cumulative_volume
        else:
            vwap = float(row["close"])
        vwap_values.append(vwap)
        close = float(row["close"])
        if close > 0 and cumulative_volume > 0:
            ratio = amount / volume if volume else close
            if abs(ratio / close - 1) > cfg.vwap_ratio_tolerance:
                approximate = True

    attempts = 0
    divergence_index: int | None = None
    freeze: dict | None = None
    stabilization: list[dict] = []

    for index, row in enumerate(rows):
        moment = row["trade_time"]
        if moment.time() > time.fromisoformat(cfg.b2_last_signal_time):
            break
        close = float(row["close"])
        vwap = vwap_values[index]
        if freeze is None:
            if divergence_index is None:
                if close < vwap:
                    divergence_index = index
                continue
            if index <= divergence_index:
                continue
            previous_close = (
                float(rows[index - 1]["close"]) if index > 0 else None
            )
            if stabilization_bar(row, previous_close):
                stabilization.append(row)
                if len(stabilization) >= cfg.b2_stabilize_count:
                    window = stabilization[-cfg.b2_stabilize_count :]
                    freeze = {
                        "low": min(float(item["low"]) for item in window),
                        "high": max(float(item["high"]) for item in window),
                        "index": index,
                    }
                    stabilization = []
            else:
                stabilization = []
            continue

        # Only a completed bar closing below the frozen low invalidates the
        # structure; a wick alone is noise.
        if close < float(freeze["low"]):
            attempts += 1
            freeze = None
            divergence_index = None
            stabilization = []
            if attempts >= cfg.b2_max_attempts:
                break
            continue
        if close > float(freeze["high"]) and close > vwap:
            atr5 = _atr5(rows, index)
            stop_price = float(freeze["low"]) - 0.5 * atr5
            stop_distance = (close - stop_price) / close if close else 1.0
            actionable = cfg.b2_stop_min <= stop_distance <= cfg.b2_stop_max
            valid_until = min(
                moment + timedelta(minutes=cfg.b2_valid_minutes),
                datetime.combine(
                    moment.date(), time(15, 0), tzinfo=moment.tzinfo
                ),
            )
            return B2Signal(
                ts_code=ts_code,
                signal_time=moment,
                entry_price=close,
                frozen_low=float(freeze["low"]),
                frozen_high=float(freeze["high"]),
                vwap=vwap,
                atr5=atr5,
                stop_price=stop_price,
                stop_distance_pct=stop_distance,
                actionable=actionable,
                vwap_approximate=approximate,
                reason=(
                    "分歧后两根止跌，收盘突破冻结高点并站上 VWAP"
                    if actionable
                    else "结构确认但止损距离超出 B2 范围"
                ),
                valid_until=valid_until,
            )
    return None
