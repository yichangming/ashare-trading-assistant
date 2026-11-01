from __future__ import annotations

from statistics import fmean

import polars as pl

from app.domain.enums import Strategy, SubMode
from app.domain.models import Candidate
from app.strategies.base import (
    StrategyContext,
    base_pool,
    make_candidate,
    prior_high,
)
from app.strategies.patterns import BoxStats, box_stats


def _platform(history: pl.DataFrame, end_index: int, cfg) -> BoxStats | None:
    """Platform ending at ``end_index - 1`` (the bar before the breakout)."""
    if end_index < cfg.platform_window + 2:
        return None
    before = history.slice(0, end_index)
    return box_stats(
        before,
        window=cfg.platform_window,
        upper_quantile=cfg.upper_quantile,
        lower_quantile=cfg.lower_quantile,
        touch_atr=cfg.touch_atr,
        min_touches=cfg.min_touches,
        touch_spacing=cfg.touch_spacing,
        width_min=cfg.width_min,
        width_max=cfg.width_max,
        slope_max=cfg.slope_max,
        abnormal_atr=cfg.abnormal_atr,
    )


def _volume_ratio(history: pl.DataFrame, index: int, window: int = 20) -> float:
    if index < window:
        return 1.0
    volumes = history["vol"].to_list()
    base = [float(value) for value in volumes[index - window : index] if value is not None]
    if not base:
        return 1.0
    average = fmean(base)
    return float(volumes[index]) / average if average > 0 and volumes[index] is not None else 1.0


def _is_c1(history: pl.DataFrame, index: int, box: BoxStats, cfg) -> bool:
    row = history.row(index, named=True)
    if row.get("close") is None:
        return False
    return (
        float(row["close"]) > box.upper + cfg.breakout_atr * box.atr
        and _volume_ratio(history, index) >= cfg.volume_ratio_min
    )


def _score(box: BoxStats, row: dict, confirmed: bool) -> float:
    width_score = max(0.0, 100.0 - box.width / 0.15 * 100.0)
    touch_score = min(100.0, (box.upper_touches + box.lower_touches) / 8 * 100.0)
    volume = min(100.0, float(row.get("vol_ratio_today_20") or 1.0) / 3 * 100.0)
    structure = 100.0 if confirmed else 55.0
    return 0.25 * width_score + 0.25 * touch_score + 0.25 * volume + 0.25 * structure


def generate(context: StrategyContext) -> list[Candidate]:
    cfg = context.config.strategies.c
    universe = context.config.universe
    candidates: list[Candidate] = []
    for code in context.codes:
        row = context.feature_row(code)
        history = context.history(code)
        if row is None or history is None or history.height < cfg.platform_window + 5:
            continue
        if not base_pool(
            context,
            row,
            strategy=Strategy.C,
            min_amount=universe.min_avg_amount,
            cap_min=universe.market_cap_min,
            cap_max=universe.market_cap_max,
            price_min=universe.min_price,
            price_max=universe.max_price,
            exclude_limit_up=True,
        ):
            continue
        atr = float(row.get("atr14_raw") or 0)
        close = float(row["close"])
        if atr <= 0:
            continue
        last = history.height - 1
        added = False

        # C1: today is the breakout day, platform frozen before today.
        box = _platform(history, last, cfg)
        if box is not None and _is_c1(history, last, box, cfg):
            score = _score(box, row, True)
            leader = context.leader_row(code) or {}
            leader_score = float(leader.get("leader_score") or 0.0)
            total = 0.7 * leader_score + 0.3 * score if leader else score
            candidate = make_candidate(
                context,
                ts_code=code,
                strategy=Strategy.C,
                sub_mode=SubMode.C1,
                setup_type="platform_breakout_c1",
                horizon="mid",
                score=total,
                reasons=[
                    f"平台 {box.upper:.2f}/{box.lower:.2f}，宽度 {box.width:.1%}",
                    f"上下沿接触 {box.upper_touches}/{box.lower_touches} 次",
                    f"收盘突破上沿 +{cfg.breakout_atr:.2f} ATR",
                ],
                evidence={
                    "stage": "confirmed",
                    "box_upper": box.upper,
                    "box_lower": box.lower,
                    "box_width": box.width,
                    "box_atr": box.atr,
                    "breakout_index": last,
                    "confirmed": True,
                },
                feature_extras={
                    "box_upper": box.upper,
                    "box_lower": box.lower,
                    "box_width": box.width,
                },
            )
            if candidate is not None:
                candidates.append(candidate)
                added = True

        # C2: a prior C1 breakout, then a pullback and today's confirmation.
        best: dict | None = None
        for breakout_index in range(
            max(cfg.platform_window + 2, last - cfg.c2_observe_days),
            last,
        ):
            prior_box = _platform(history, breakout_index, cfg)
            if prior_box is None or not _is_c1(history, breakout_index, prior_box, cfg):
                continue
            zone = cfg.c2_zone_atr * prior_box.atr
            pullback_seen = False
            pullback_low: float | None = None
            invalid = False
            for index in range(breakout_index + 1, last):
                bar = history.row(index, named=True)
                if bar.get("close") is None or bar.get("low") is None:
                    continue
                if float(bar["close"]) < prior_box.upper - zone:
                    invalid = True
                    break
                if float(bar["low"]) <= prior_box.upper + zone:
                    pullback_seen = True
                    low = float(bar["low"])
                    pullback_low = low if pullback_low is None else min(pullback_low, low)
            if invalid or not pullback_seen:
                continue
            prev = prior_high(history, last)
            confirmed = (
                prev is not None
                and close > prev
                and close > prior_box.upper
            )
            current_low = float(row["low"])
            observing = (
                abs(current_low - prior_box.upper) <= zone
                and close >= prior_box.upper - zone
            )
            if not confirmed and not observing:
                continue
            best = {
                "box": prior_box,
                "breakout_index": breakout_index,
                "confirmed": confirmed,
                "pullback_seen": pullback_seen,
                "pullback_low": pullback_low,
            }
        if best is not None:
            prior_box = best["box"]
            confirmed = bool(best["confirmed"])
            score = _score(prior_box, row, confirmed)
            leader = context.leader_row(code) or {}
            leader_score = float(leader.get("leader_score") or 0.0)
            total = 0.7 * leader_score + 0.3 * score if leader else score
            candidate = make_candidate(
                context,
                ts_code=code,
                strategy=Strategy.C,
                sub_mode=SubMode.C2,
                setup_type="platform_pullback_c2",
                horizon="mid",
                score=total,
                reasons=[
                    f"平台 {prior_box.upper:.2f}/{prior_box.lower:.2f}",
                    "突破后回踩平台上沿区域",
                    "收盘突破前日高点并站上平台上沿" if confirmed else "回踩区域内等待确认",
                ],
                evidence={
                    "stage": "confirmed" if confirmed else "observing",
                    "box_upper": prior_box.upper,
                    "box_lower": prior_box.lower,
                    "box_width": prior_box.width,
                    "box_atr": prior_box.atr,
                    "breakout_index": best["breakout_index"],
                    "pullback_low": best["pullback_low"],
                    "confirmed": confirmed,
                },
                feature_extras={
                    "box_upper": prior_box.upper,
                    "box_lower": prior_box.lower,
                    "box_width": prior_box.width,
                },
            )
            if candidate is not None:
                candidates.append(candidate)
                added = True
        _ = added
    return candidates
