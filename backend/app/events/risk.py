"""Negative/retracted event risk: cancel pending entries and flag exits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.full_config import FullSystemConfig
from app.db.models import EventRow
from app.db.repositories import (
    NotificationRepository,
    PlanRepository,
    PositionRepository,
)
from app.domain.enums import PlanStatus


@dataclass
class EventRiskService:
    session: Session
    config: FullSystemConfig
    account_id: str = "default"

    def apply(self, as_of: date) -> dict[str, int]:
        window = self.config.strategies.d.event_risk_days
        rows = list(
            self.session.scalars(
                select(EventRow).where(
                    EventRow.verification_status == "verified",
                    EventRow.ann_date.is_not(None),
                    EventRow.ann_date >= as_of - timedelta(days=window),
                    EventRow.ann_date <= as_of,
                    or_(
                        EventRow.retracted.is_(True),
                        EventRow.direction == "negative",
                    ),
                )
            )
        )
        codes = {
            link.ts_code
            for row in rows
            for link in row.links
            if link.ts_code
        }
        cancelled = 0
        flagged = 0
        for code in codes:
            for plan in PlanRepository.list_open(self.session, self.account_id):
                if plan.ts_code != code:
                    continue
                if plan.status not in (
                    PlanStatus.PENDING_EXECUTION.value,
                    PlanStatus.PENDING_ENTRY.value,
                    PlanStatus.PARTIAL_FILL.value,
                ):
                    continue
                PlanRepository.update_status(
                    self.session,
                    plan.id,
                    PlanStatus.CANCELLED,
                    "negative or retracted event",
                )
                cancelled += 1
            position = PositionRepository.get(self.session, self.account_id, code)
            if position is None:
                continue
            NotificationRepository.add(
                self.session,
                level="critical",
                code="event_risk_exit",
                title=f"{code} 负面/撤回事件",
                message=(
                    "持仓关联事件为负面或被撤回；请人工复核是否退出，"
                    "系统不会自动卖出。"
                ),
                dedupe_key=f"event-risk-exit:{as_of.isoformat()}:{code}",
                trade_date=as_of,
            )
            flagged += 1
        self.session.flush()
        return {"events": len(rows), "cancelled_plans": cancelled, "flagged_positions": flagged}
