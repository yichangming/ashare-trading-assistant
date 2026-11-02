"""Daily position protection review.

This service is a thin daily adapter over the shared position decision
service.  It never sells automatically; every exit remains a human action in
the broker.  Missing/stale data raises a data-anomaly decision instead of
silently defaulting to a safe hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.full_config import FullSystemConfig
from app.data.interfaces import MarketDataProvider
from app.db.repositories import PositionRepository
from app.domain.enums import ExitSession, PositionAction
from app.domain.models import PositionDecision
from app.portfolio.decisions import PositionDecisionService
from app.portfolio.market import points_from_bars
from app.portfolio.service import AccountService


class ExitReviewResult(BaseModel):
    checked: int = 0
    stop_alerts: int = 0
    tightened: int = 0
    target_alerts: int = 0
    time_exits: int = 0
    data_anomalies: int = 0
    reduces: int = 0
    decisions: list[PositionDecision] = []


@dataclass
class ExitReviewService:
    provider: MarketDataProvider
    config: FullSystemConfig
    account_id: str = "default"

    def review(self, as_of: date, session: Session) -> ExitReviewResult:
        result = ExitReviewResult()
        positions = PositionRepository.list_positions(session, self.account_id)
        if not positions:
            return result
        points = {}
        try:
            bars = self.provider.load_bars(as_of, as_of)
            points = points_from_bars(bars, as_of)
        except Exception:  # noqa: BLE001 - review records the data anomaly
            points = {}

        before_stops = {
            position.ts_code: position.current_stop for position in positions
        }
        AccountService(session, self.config, self.account_id).revalue(
            as_of, points
        )
        decisions = PositionDecisionService(
            session,
            self.config,
            account_id=self.account_id,
            provider=self.provider,
        ).evaluate_all(
            as_of,
            points=points,
            session_type=ExitSession.DAILY,
            persist=True,
            notify=True,
        )
        result.checked = len(decisions)
        result.decisions = decisions
        for decision in decisions:
            if decision.action == PositionAction.EXIT:
                if decision.reason_code == "hard_stop":
                    result.stop_alerts += 1
                elif decision.reason_code == "time_stop":
                    result.time_exits += 1
                elif decision.reason_code.startswith("target"):
                    result.target_alerts += 1
            elif decision.action == PositionAction.REDUCE:
                result.reduces += 1
                result.target_alerts += 1
            elif decision.action == PositionAction.DATA_ANOMALY:
                result.data_anomalies += 1
            before = before_stops.get(decision.ts_code)
            if (
                before is not None
                and decision.proposed_stop is not None
                and decision.proposed_stop > before
            ):
                result.tightened += 1
        session.flush()
        return result
