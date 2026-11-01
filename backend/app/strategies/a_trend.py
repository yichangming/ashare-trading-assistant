from __future__ import annotations

from app.domain.enums import Strategy, SubMode
from app.domain.models import Candidate
from app.strategies.base import (
    StrategyContext,
    base_pool,
    make_candidate,
)
from app.strategies.patterns import atr_at, ma_at, swing_high


def _touch(
    history,
    index: int,
    *,
    ma10: float,
    ma20: float,
    atr: float,
    touch_atr: float,
) -> str | None:
    low = float(history.row(index, named=True)["low"])
    if abs(low - ma10) <= touch_atr * atr and low > ma20 - touch_atr * atr:
        return "shallow"
    if abs(low - ma20) <= touch_atr * atr:
        return "deep"
    return None


def _pattern_score(
    row: dict, duration: int, atr: float, ma10: float, ma20: float
) -> float:
    low = float(row["low"])
    vol_ratio = float(row.get("vol_ratio_5_20") or 1.0)
    volume = max(0.0, min(100.0, (1.0 - vol_ratio) * 400.0))
    distance = min(abs(low - ma10), abs(low - ma20)) / max(atr, 1e-9)
    ma_score = max(0.0, min(100.0, (1.0 - distance / 0.75) * 100.0))
    pullback = max(0.0, min(100.0, 100.0 - abs(duration - 5.5) / 2.5 * 100.0))
    open_ = float(row["open"])
    close = float(row["close"])
    high = float(row["high"])
    body = abs(close - open_)
    lower_shadow = min(open_, close) - low
    stabilization = (
        100.0
        if close > open_ or lower_shadow >= body or close >= low + 0.5 * (high - low)
        else 40.0
    )
    return 0.4 * volume + 0.2 * ma_score + 0.2 * pullback + 0.2 * stabilization


def generate(context: StrategyContext) -> list[Candidate]:
    cfg = context.config.strategies.a
    universe = context.config.universe
    candidates: list[Candidate] = []
    for code in context.codes:
        row = context.feature_row(code)
        history = context.history(code)
        if row is None or history is None or history.height < 70:
            continue
        if not base_pool(
            context,
            row,
            strategy=Strategy.A,
            min_amount=universe.min_avg_amount,
            cap_min=universe.market_cap_min,
            cap_max=universe.market_cap_max,
            price_min=universe.min_price,
            price_max=universe.max_price,
            exclude_limit_up=True,
        ):
            continue
        ma20 = row.get("ma20_raw")
        ma60 = row.get("ma60_raw")
        ma10 = row.get("ma10_raw")
        atr = row.get("atr14_raw")
        close = row.get("close")
        low = row.get("low")
        slope = row.get("ma20_slope")
        rs = row.get("rs_20d_market")
        if None in (ma10, ma20, ma60, atr, close, low, slope, rs):
            continue
        # ma20_raw/ma60_raw are in the same raw (unadjusted) scale as close;
        # comparing them with hfq_close would bypass the trend filter whenever
        # adj_factor != 1.
        if not (float(close) > float(ma20) > float(ma60)):
            continue
        if float(slope) <= 0 or float(rs) < cfg.rs_min:
            continue
        if float(atr) <= 0:
            continue
        swing = swing_high(history, history.height - 1, cfg.pullback_window)
        if swing is None:
            continue
        if not (cfg.pullback_min_days <= swing["duration"] <= cfg.pullback_max_days):
            continue
        if not (cfg.pullback_depth_min <= swing["depth"] <= cfg.pullback_depth_max):
            continue
        structure_ok = (
            float(close) >= float(ma20) * (1 - cfg.structure_ma_tolerance)
            and float(low) >= float(ma20) - cfg.touch_atr * float(atr)
            and float(close) > float(ma60)
        )
        if not structure_ok:
            continue
        if float(row.get("vol_ratio_5_20") or 1.0) >= cfg.volume_shrink_ratio:
            continue

        last = history.height - 1
        observation_index: int | None = None
        current_touch = _touch(
            history,
            last,
            ma10=float(ma10),
            ma20=float(ma20),
            atr=float(atr),
            touch_atr=cfg.touch_atr,
        )
        previous_touch = None
        for offset in (1, 2):
            index = last - offset
            if index < 20:
                continue
            ma10_at = ma_at(history, index, cfg.shallow_ma)
            ma20_at = ma_at(history, index, cfg.deep_ma)
            atr_at_value = atr_at(history, index)
            if None in (ma10_at, ma20_at, atr_at_value):
                continue
            touch = _touch(
                history,
                index,
                ma10=float(ma10_at),
                ma20=float(ma20_at),
                atr=float(atr_at_value),
                touch_atr=cfg.touch_atr,
            )
            if touch is not None:
                previous_touch = touch
                if observation_index is None:
                    observation_index = index
                break
        if observation_index is None and current_touch is not None:
            observation_index = last
        if current_touch is None and previous_touch is None:
            continue
        # Classify from the whole pullback's minimum low, not just today's bar.
        pullback_low = float(swing["pullback_low"])
        sub: str | None
        if (
            abs(pullback_low - float(ma10)) <= cfg.touch_atr * float(atr)
            and pullback_low > float(ma20) - cfg.touch_atr * float(atr)
        ):
            sub = "shallow"
        elif abs(pullback_low - float(ma20)) <= cfg.touch_atr * float(atr):
            sub = "deep"
        else:
            sub = current_touch or previous_touch
        if sub is None:
            continue
        reference_ma = float(ma10) if sub == "shallow" else float(ma20)
        observation_high: float | None
        if observation_index == last and current_touch is not None and last > 0:
            # Intraday touch + close back above the prior high is treated as a
            # same-session observation/confirmation at the close.
            observation_high = float(history.row(last - 1, named=True)["high"])
            confirmed = (
                float(close) > observation_high
                and float(close) > reference_ma
            )
        else:
            observation_high = (
                float(history.row(observation_index, named=True)["high"])
                if observation_index is not None
                else None
            )
            confirmed = (
                observation_index is not None
                and observation_index < last
                and observation_high is not None
                and float(close) > observation_high
                and float(close) > reference_ma
            )
        stage = "confirmed" if confirmed else "observing"
        pattern = _pattern_score(row, swing["duration"], float(atr), float(ma10), float(ma20))
        leader = context.leader_row(code) or {}
        leader_score = float(leader.get("leader_score") or 0.0)
        score = 0.7 * leader_score + 0.3 * pattern if leader else pattern
        pullback_label = "浅" if sub == "shallow" else "深"
        reasons = [
            f"趋势 MA20>MA60，MA20 斜率 {float(slope):.3f}",
            f"{pullback_label}回踩 {swing['duration']} 日，"
            f"深度 {swing['depth']:.1%}",
            f"缩量比 {float(row.get('vol_ratio_5_20') or 1):.2f}",
        ]
        if confirmed:
            reasons.append("收盘突破前一日高点且站上参考均线")
        else:
            reasons.append("已触及参考均线，等待突破确认")
        candidate = make_candidate(
            context,
            ts_code=code,
            strategy=Strategy.A,
            sub_mode=SubMode.A_SHALLOW if sub == "shallow" else SubMode.A_DEEP,
            setup_type=f"trend_pullback_{sub}",
            horizon="mid",
            score=score,
            reasons=reasons,
            evidence={
                "stage": stage,
                "pullback_type": sub,
                "swing_high": swing["swing_high"],
                "pullback_days": swing["duration"],
                "pullback_depth": swing["depth"],
                "pullback_low": swing["pullback_low"],
                "reference_ma": reference_ma,
                "confirmed": confirmed,
            },
            feature_extras={
                "pullback_duration": swing["duration"],
                "pullback_depth": swing["depth"],
                "pattern_score": pattern,
            },
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates
