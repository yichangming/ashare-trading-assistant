from __future__ import annotations

from datetime import date, datetime

from app.domain.enums import AccountMode, Strategy, SubMode
from app.domain.models import Candidate
from app.strategies.base import (
    StrategyContext,
    base_pool,
    make_candidate,
)
from app.strategies.patterns import (
    drawdown_from_high,
    returns,
    two_day_structure,
    zscore,
)


def _negative_event(context: StrategyContext, code: str, cfg) -> bool:
    for event in context.events_for(code):
        if event.get("direction") != "negative":
            continue
        if event.get("retracted"):
            continue
        if event.get("verification_status") != "verified":
            continue
        event_day = event.get("ann_date") or event.get("published_at")
        if isinstance(event_day, str):
            try:
                event_day = date.fromisoformat(event_day[:10])
            except ValueError:
                return True
        if isinstance(event_day, datetime):
            event_day = event_day.date()
        if event_day is None:
            return True
        # Only recent major negative events block D; stale news is priced in.
        if context.trading_days_since(event_day) <= cfg.event_risk_days:
            return True
    return False


def generate(context: StrategyContext) -> list[Candidate]:
    cfg = context.config.strategies.d
    universe = context.config.universe
    if not context.events_available:
        return []
    candidates: list[Candidate] = []
    for code in context.codes:
        row = context.feature_row(code)
        history = context.history(code)
        if row is None or history is None or history.height < 65:
            continue
        if not base_pool(
            context,
            row,
            strategy=Strategy.D,
            min_amount=universe.min_avg_amount,
            cap_min=universe.market_cap_min,
            cap_max=universe.market_cap_max,
            price_min=universe.min_price,
            price_max=universe.max_price,
        ):
            continue
        if _negative_event(context, code, cfg):
            continue
        closes = [float(value) for value in history["close"].to_list() if value is not None]
        if len(closes) < 65:
            continue
        ret5 = closes[-1] / closes[-6] - 1 if closes[-6] else None
        ret10 = closes[-1] / closes[-11] - 1 if closes[-11] else None
        if ret5 is None or ret10 is None:
            continue
        five_day_returns = [
            value for value in returns(history, 5)[-60:] if value is not None
        ]
        z_value = zscore(five_day_returns, ret5)
        drawdown = drawdown_from_high(history, 20)
        trend_up = bool(
            float(row.get("ma20_raw") or 0) > float(row.get("ma60_raw") or 0)
            and float(row.get("close") or 0) > float(row.get("ma60_raw") or 0)
        )
        if (
            not trend_up
            and context.account_mode == AccountMode.REAL
            and not cfg.downtrend_real_enabled
        ):
            continue
        sector = context.sector(str(row.get("sector") or ""))
        industry_relative = None
        if sector is not None and row.get("ret_5d") is not None:
            industry_relative = float(row["ret_5d"]) - sector.return_5d
        conditions = 0
        if z_value is not None and z_value <= cfg.z_min:
            conditions += 1
        if trend_up:
            if ret5 <= cfg.drop5_uptrend:
                conditions += 1
            if ret10 <= cfg.drop10_uptrend:
                conditions += 1
            if drawdown is not None and drawdown >= cfg.drawdown20_uptrend:
                conditions += 1
        else:
            if ret5 <= cfg.drop5_downtrend:
                conditions += 1
            if ret10 <= cfg.drop10_downtrend:
                conditions += 1
            if drawdown is not None and drawdown >= cfg.drawdown20_downtrend:
                conditions += 1
        if industry_relative is not None and industry_relative <= cfg.industry_relative:
            conditions += 1
        if conditions < cfg.min_conditions:
            continue

        last = history.height - 1
        best: dict | None = None
        for d2_index in range(last, max(1, last - cfg.structure_wait_days - 2), -1):
            structure = two_day_structure(history, d2_index)
            if structure is None:
                continue
            invalid = False
            for index in range(d2_index + 1, last + 1):
                bar = history.row(index, named=True)
                low = bar.get("low")
                close = bar.get("close")
                if low is None or close is None:
                    continue
                if (
                    float(low) < structure["observation_low"]
                    or float(close) < structure["observation_low"]
                ):
                    invalid = True
                    break
            if invalid:
                continue
            close_today = float(row["close"])
            confirmed = d2_index < last and close_today > structure["frozen_high"]
            observing = d2_index == last
            if not confirmed and not observing:
                continue
            candidate_structure = {
                **structure,
                "confirmed": confirmed,
                "d2_index": d2_index,
            }
            if confirmed:
                best = candidate_structure
                break
            if best is None:
                best = candidate_structure
        if best is None:
            continue
        sub = SubMode.D_UPTREND if trend_up else SubMode.D_DOWNTREND
        score = (
            min(100.0, conditions * 20)
            + min(20.0, max(0.0, -(z_value or 0) * 5))
            + (20.0 if best["confirmed"] else 5.0)
        ) / 1.4
        leader = context.leader_row(code) or {}
        leader_score = float(leader.get("leader_score") or 0.0)
        total = 0.7 * leader_score + 0.3 * score if leader else score
        trend_label = "上升趋势" if trend_up else "下降趋势"
        reasons = [
            f"{trend_label}，满足 {conditions} 项超跌条件",
            f"两日不创新低，冻结高点 {best['frozen_high']:.2f}，"
            f"观察低点 {best['observation_low']:.2f}",
            "收盘突破冻结高点" if best["confirmed"] else "结构形成，等待突破冻结高点",
        ]
        candidate = make_candidate(
            context,
            ts_code=code,
            strategy=Strategy.D,
            sub_mode=sub,
            setup_type="oversold_repair",
            horizon="short" if not trend_up else "mid",
            score=total,
            reasons=reasons,
            evidence={
                "stage": "confirmed" if best["confirmed"] else "observing",
                "trend_context": "uptrend" if trend_up else "downtrend",
                "conditions": conditions,
                "ret5": ret5,
                "ret10": ret10,
                "zscore5": z_value,
                "drawdown20": drawdown,
                "industry_relative": industry_relative,
                "frozen_high": best["frozen_high"],
                "observation_low": best["observation_low"],
                "confirmed": best["confirmed"],
                "event_data_incomplete": not context.events_available,
            },
            feature_extras={
                "oversold_conditions": conditions,
                "drawdown20": drawdown or 0.0,
                "frozen_high": best["frozen_high"],
                "observation_low": best["observation_low"],
            },
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates
