from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.core.full_config import load_full_config
from app.db.models import PlanRow
from app.db.repositories import (
    AccountRepository,
    NotificationRepository,
    PositionRepository,
)
from app.db.session import init_db, reset_engine, session_scope
from app.jobs.exits import ExitReviewService


class _Provider:
    def __init__(self, close: float, high: float):
        self.close = close
        self.high = high

    def load_bars(self, start, end):
        return pl.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": end,
                    "open": self.close,
                    "high": self.high,
                    "low": self.close,
                    "close": self.close,
                    "vol": 1_000_000.0,
                }
            ]
        )

    def load_trade_cal(self, start, end):
        days = [
            start + timedelta(days=index)
            for index in range((end - start).days + 1)
        ]
        return pl.DataFrame({"cal_date": days, "is_open": [1 for _ in days]})


def _position(session, *, stop: float = 9.5, plan_id: str | None = "plan-1"):
    AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
    PositionRepository.upsert(
        session,
        account_id="default",
        ts_code="000001.SZ",
        shares=1000,
        available_shares=1000,
        cost_price=10,
        current_stop=stop,
        current_price=10,
        primary_strategy="A",
        plan_id=plan_id,
        opened_at=date(2026, 10, 1),
    )
    session.add(
        PlanRow(
            id="plan-1",
            account_id="default",
            trade_date=date(2026, 10, 1),
            ts_code="000001.SZ",
            sector="银行",
            strategy="A",
            status="OPEN",
            expires_on=date(2026, 10, 30),
            plan_json={
                "stop_plan": {"initial_stop": "9.5", "time_stop_days": 10},
                "take_profit_plan": {"targets": [{"price": "11.5"}]},
            },
        )
    )
    session.flush()


def test_exit_review_alerts_on_stop():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        _position(session)
        result = ExitReviewService(
            _Provider(close=9.4, high=9.5), load_full_config()
        ).review(date(2026, 10, 30), session)
        assert result.stop_alerts == 1
        assert any(
            item.code == "exit_stop"
            for item in NotificationRepository.list_recent(session)
        )


def test_exit_review_tightens_protection_and_flags_target():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        _position(session)
        result = ExitReviewService(
            _Provider(close=11.2, high=11.6), load_full_config()
        ).review(date(2026, 10, 30), session)
        assert result.tightened == 1
        assert result.target_alerts == 1
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert float(position.current_stop) == 9.5
        assert position.pending_stop is not None
        assert float(position.pending_stop) >= 10.5
        assert position.pending_stop_effective_at is not None
