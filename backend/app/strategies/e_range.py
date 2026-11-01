from __future__ import annotations

from app.domain.enums import Strategy, SubMode
from app.domain.models import Candidate
from app.strategies.base import (
    StrategyContext,
    base_pool,
    make_candidate,
    prior_high,
)
from app.strategies.patterns import BoxStats, box_stats


def _range(history, end_index: int, cfg) -> BoxStats | None:
    if end_index < cfg.range_window + 2:
        return None
    before = history.slice(0, end_index)
    return box_stats(
        before,
        window=cfg.range_window,
        upper_quantile=cfg.upper_quantile,
        lower_quantile=cfg.lower_quantile,
        touch_atr=cfg.touch_atr,
        min_touches=cfg.min_touches,
        touch_spacing=cfg.touch_spacing,
        width_min=cfg.width_min,
        width_max=cfg.width_max,
        slope_min=cfg.slope_min,
    )


def generate(context: StrategyContext) -> list[Candidate]:
    cfg = context.config.strategies.e
    universe = context.config.universe
    costs = context.config.costs
    candidates: list[Candidate] = []
    for code in context.codes:
        row = context.feature_row(code)
        history = context.history(code)
        if row is None or history is None or history.height < cfg.range_window + 5:
            continue
        if not base_pool(
            context,
            row,
            strategy=Strategy.E,
            min_amount=universe.min_avg_amount,
            cap_min=universe.market_cap_min,
            cap_max=universe.market_cap_max,
            price_min=universe.min_price,
            price_max=universe.max_price,
        ):
            continue
        last = history.height - 1
        box = _range(history, last, cfg)
        if box is None:
            continue
        # The range must not have already broken in the recent window.
        recent = history.slice(max(0, last - 5), last)
        broken = any(
            float(close) > box.upper or float(close) < box.lower
            for close in recent["close"].to_list()
            if close is not None
        )
        if broken:
            continue
        close = float(row["close"])
        atr = float(row.get("atr14_raw") or 0)
        if atr <= 0:
            continue
        entry_threshold = box.lower + cfg.entry_zone_pct * (box.upper - box.lower)
        if close > entry_threshold:
            continue
        structure = close >= box.lower - cfg.stop_atr * atr
        if not structure:
            continue
        midpoint = (box.upper + box.lower) / 2
        prev = prior_high(history, last)
        confirmed = (
            prev is not None
            and close > prev
            and close <= midpoint
        )
        observing = close <= entry_threshold
        if not confirmed and not observing:
            continue
        stop_ref = box.lower - cfg.stop_atr * atr
        r_value = close - stop_ref
        if r_value <= 0:
            continue
        room = box.upper - close - close * costs.round_trip_cost_pct
        if room / r_value < cfg.min_rr:
            continue
        score = (
            40.0
            + min(20.0, (box.upper_touches + box.lower_touches) * 3)
            + min(20.0, (1 - abs(close - midpoint) / max(midpoint - box.lower, 1e-9)) * 20)
            + (20.0 if confirmed else 5.0)
        )
        leader = context.leader_row(code) or {}
        leader_score = float(leader.get("leader_score") or 0.0)
        total = 0.7 * leader_score + 0.3 * score if leader else score
        candidate = make_candidate(
            context,
            ts_code=code,
            strategy=Strategy.E,
            sub_mode=SubMode.E_RANGE,
            setup_type="range_trade",
            horizon="short",
            score=total,
            reasons=[
                f"区间 {box.lower:.2f}–{box.upper:.2f}，宽度 {box.width:.1%}",
                f"上下沿接触 {box.upper_touches}/{box.lower_touches} 次",
                f"进入区间下部 20%，上方空间/风险 {room / r_value:.2f}",
                "收盘突破前日高点" if confirmed else "等待突破前日高点",
            ],
            evidence={
                "stage": "confirmed" if confirmed else "observing",
                "range_upper": box.upper,
                "range_lower": box.lower,
                "range_width": box.width,
                "range_atr": box.atr,
                "stop_ref": stop_ref,
                "r_value": r_value,
                "upside_room_ratio": room / r_value,
                "confirmed": confirmed,
            },
            feature_extras={
                "range_upper": box.upper,
                "range_lower": box.lower,
                "range_width": box.width,
                "upside_room_ratio": room / r_value,
            },
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates
