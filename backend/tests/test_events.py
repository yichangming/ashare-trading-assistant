from __future__ import annotations

from datetime import date

import polars as pl

from app.db.repositories import EventRepository
from app.db.session import init_db, reset_engine, session_scope
from app.events.ingest import EventIngestor


def _forecast() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20261030"],
            "end_date": ["20261231"],
            "type": ["预增"],
            "p_change_min": [15.0],
            "p_change_max": [20.0],
            "first_ann_date": ["20261030"],
            "summary": ["预计增长"],
            "change_reason": ["经营改善"],
        }
    )


def test_earnings_event_is_verified_and_revised():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        ingestor = EventIngestor(session)
        assert ingestor.ingest_earnings(
            _forecast(), sub_type="forecast", source="forecast_vip"
        ) == 1
        events = EventRepository.list_recent(session, verified_only=True)
        assert len(events) == 1
        assert events[0].direction == "positive"
        assert events[0].verification_status == "verified"
        assert events[0].ann_date == date(2026, 10, 30)
        assert ingestor.ingest_earnings(
            _forecast(), sub_type="forecast", source="forecast_vip"
        ) == 0
        assert EventRepository.list_recent(session)[0].revision == 1


def test_manual_contract_stays_pending_until_verified():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        ingestor = EventIngestor(session)
        imported = ingestor.ingest_manual(
            [
                {
                    "event_type": "contract",
                    "ts_code": "000001.SZ",
                    "ann_date": "2026-10-30",
                    "title": "重大合同",
                    "direction": "positive",
                    "stage": "formal",
                    "certainty": "formal",
                    "verification_status": "pending",
                    "materiality": 0.9,
                }
            ]
        )
        assert imported == 1
        event = EventRepository.list_recent(session)[0]
        assert event.verification_status == "pending"
        assert not EventRepository.list_recent(session, verified_only=True)
        EventRepository.verify(session, event.id, "verified")
        assert EventRepository.list_recent(session, verified_only=True)


def test_repurchase_and_holder_trade_ingestion():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        ingestor = EventIngestor(session)
        repurchase = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20261030"],
                "end_date": [None],
                "proc": ["预案"],
                "amount": [100000000.0],
                "low_limit": [10.0],
                "high_limit": [12.0],
            }
        )
        holder = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20261030"],
                "holder_name": ["测试股东"],
                "in_de": ["IN"],
                "change_vol": [100000.0],
                "change_ratio": [0.01],
            }
        )
        assert ingestor.ingest_repurchase(repurchase, source="repurchase") == 1
        assert ingestor.ingest_holder_trade(holder, source="stk_holdertrade") == 1
        events = EventRepository.list_recent(session, verified_only=True)
        assert len(events) == 2
        assert {event.direction for event in events} == {"positive"}
        assert any(event.sub_type == "holder_trade" for event in events)


def test_negative_event_cancels_pending_plan_and_flags_position():
    from app.core.full_config import load_full_config
    from app.db.models import PlanRow
    from app.db.repositories import PlanRepository, PositionRepository
    from app.domain.enums import PlanStatus
    from app.events.risk import EventRiskService

    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        EventIngestor(session).ingest_manual(
            [
                {
                    "event_type": "contract",
                    "ts_code": "000001.SZ",
                    "ann_date": "2026-10-30",
                    "title": "立案调查",
                    "direction": "negative",
                    "stage": "formal",
                    "certainty": "formal",
                    "verification_status": "verified",
                    "materiality": 0.9,
                }
            ]
        )
        session.add(
            PlanRow(
                id="plan-event",
                account_id="default",
                trade_date=date(2026, 10, 30),
                ts_code="000001.SZ",
                sector="银行",
                strategy="C",
                status=PlanStatus.PENDING_EXECUTION.value,
                expires_on=date(2026, 11, 5),
                plan_json={"status": PlanStatus.PENDING_EXECUTION.value},
            )
        )
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=9.5,
            current_price=10.5,
        )
        session.flush()
        result = EventRiskService(session, load_full_config()).apply(
            date(2026, 10, 31)
        )
        assert result["cancelled_plans"] == 1
        assert result["flagged_positions"] == 1
        assert PlanRepository.get(session, "plan-event").status == "CANCELLED"


def test_event_retraction_is_stored():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        ingestor = EventIngestor(session)
        rows = [
            {
                "event_type": "contract",
                "ts_code": "000001.SZ",
                "ann_date": "2026-10-30",
                "title": "重大合同",
                "direction": "positive",
                "verification_status": "verified",
            }
        ]
        ingestor.ingest_manual(rows)
        rows[0]["retracted"] = True
        ingestor.ingest_manual(rows)
        event = EventRepository.list_recent(session)[0]
        assert event.retracted is True
        assert event.revision == 1
        assert len(event.revisions) == 1
        assert event.revisions[0].revision == 0
        assert event.revisions[0].retracted is False
