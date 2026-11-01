"""Daily position protection review.

This service only updates protective stops and raises notifications.  It never
sells automatically; every exit remains a human action in the broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.calendar import TradingCalendar
from app.core.full_config import FullSystemConfig
from app.data.interfaces import MarketDataProvider
from app.db.repositories import NotificationRepository, PlanRepository, PositionRepository
from app.domain.enums import Strategy


class ExitReviewResult(BaseModel):
    checked: int = 0
    stop_alerts: int = 0
    tightened: int = 0
    target_alerts: int = 0
    time_exits: int = 0


@dataclass
class ExitReviewService:
    provider: MarketDataProvider
    config: FullSystemConfig
    account_id: str = "default"

    def review(self, as_of: date, session: Session) -> ExitReviewResult:
        result = ExitReviewResult()
        try:
            bars = self.provider.load_bars(as_of, as_of)
        except Exception:  # noqa: BLE001 - review is best effort
            return result
        if bars.is_empty() or "ts_code" not in bars.columns:
            return result
        by_code = {row["ts_code"]: row for row in bars.iter_rows(named=True)}
        positions = PositionRepository.list_positions(session, self.account_id)
        try:
            calendar = TradingCalendar.from_trade_cal(
                self.provider.load_trade_cal(as_of - timedelta(days=400), as_of)
            )
        except Exception:  # noqa: BLE001
            calendar = TradingCalendar([])
        for position in positions:
            row = by_code.get(position.ts_code)
            if row is None:
                continue
            result.checked += 1
            close = float(row["close"])
            high = float(row["high"])
            entry = float(position.cost_price)
            stop = float(position.current_stop)
            position.current_price = Decimal(str(close))
            plan = (
                PlanRepository.get(session, position.plan_id)
                if position.plan_id
                else None
            )
            payload = plan.plan_json if plan else {}
            initial_stop = float(
                (payload.get("stop_plan") or {}).get("initial_stop") or stop
            )
            risk = max(0.0, entry - initial_stop)

            if close <= stop:
                result.stop_alerts += 1
                NotificationRepository.add(
                    session,
                    level="critical",
                    code="exit_stop",
                    title=f"{position.ts_code} 触及保护线",
                    message=(
                        f"收盘 {close:.2f} ≤ 保护线 {stop:.2f}；请人工复核退出。"
                    ),
                    dedupe_key=f"exit-stop:{as_of.isoformat()}:{position.ts_code}",
                    trade_date=as_of,
                )
                continue

            if risk > 0 and position.primary_strategy in (
                Strategy.A.value,
                Strategy.C.value,
            ):
                new_stop = stop
                if high >= entry + 2 * risk:
                    new_stop = max(stop, entry + risk)
                elif high >= entry + risk:
                    new_stop = max(stop, entry)
                if new_stop > stop:
                    position.current_stop = Decimal(str(round(new_stop, 4)))
                    result.tightened += 1
                    NotificationRepository.add(
                        session,
                        level="info",
                        code="stop_tightened",
                        title=f"{position.ts_code} 保护线上移",
                        message=f"保护线 {stop:.2f} → {new_stop:.2f}",
                        dedupe_key=(
                            f"stop-tightened:{as_of.isoformat()}:"
                            f"{position.ts_code}:{new_stop:.2f}"
                        ),
                        trade_date=as_of,
                    )

            for target in (payload.get("take_profit_plan") or {}).get(
                "targets", []
            ):
                target_price = float(target.get("price") or 0)
                if target_price > 0 and high >= target_price:
                    result.target_alerts += 1
                    NotificationRepository.add(
                        session,
                        level="warning",
                        code="take_profit_hit",
                        title=f"{position.ts_code} 触及止盈目标",
                        message=(
                            f"最高 {high:.2f} ≥ 目标 {target_price:.2f}；"
                            "请人工复核分批止盈。"
                        ),
                        dedupe_key=(
                            f"take-profit:{as_of.isoformat()}:"
                            f"{position.ts_code}:{target_price:.2f}"
                        ),
                        trade_date=as_of,
                    )
                    break

            time_stop = int(
                (payload.get("stop_plan") or {}).get("time_stop_days") or 0
            )
            if time_stop > 0 and position.opened_at is not None and calendar.dates:
                held = calendar.count_between(
                    position.opened_at.date(), as_of
                )
                if held >= time_stop and close < entry:
                    result.time_exits += 1
                    NotificationRepository.add(
                        session,
                        level="warning",
                        code="time_stop",
                        title=f"{position.ts_code} 时间止损",
                        message=(
                            f"持有 {held} 个交易日且收盘低于成本，"
                            "请人工复核退出。"
                        ),
                        dedupe_key=(
                            f"time-stop:{as_of.isoformat()}:{position.ts_code}"
                        ),
                        trade_date=as_of,
                    )
        session.flush()
        return result
