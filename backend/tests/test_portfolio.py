from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.core.full_config import load_full_config
from app.db.repositories import AccountRepository
from app.db.session import init_db, reset_engine, session_scope
from app.domain.enums import ExecutionSide
from app.portfolio.csv_io import import_csv, write_templates
from app.portfolio.service import AccountService, PortfolioError


def _service(session, account_id: str = "default") -> AccountService:
    service = AccountService(session, load_full_config(), account_id)
    service.configure_risk(equity=1_000_000, cash=1_000_000)
    return service


def test_buy_t_plus_one_sell_and_account_state():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        execution = service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2024, 1, 2, 10, 0),
            external_id="buy-1",
        )
        assert execution.shares == 1000
        state = service.state()
        assert len(state.positions) == 1
        assert state.positions[0].available_shares == 0
        with pytest.raises(PortfolioError, match="T\\+1"):
            service.record_execution(
                ts_code="000001.SZ",
                side=ExecutionSide.SELL,
                price=Decimal("11"),
                shares=1000,
                executed_at=datetime(2024, 1, 2, 14, 0),
            )
        service.refresh_available(date(2024, 1, 3))
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.SELL,
            price=Decimal("11"),
            shares=500,
            executed_at=datetime(2024, 1, 3, 10, 0),
            external_id="sell-1",
        )
        state = service.state()
        assert state.positions[0].shares == 500
        assert state.positions[0].available_shares == 500
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.SELL,
            price=Decimal("11"),
            shares=500,
            executed_at=datetime(2024, 1, 3, 14, 0),
            external_id="sell-2",
        )
        assert service.state().positions == []


def test_duplicate_execution_is_idempotent_and_cash_flow_and_action():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2024, 1, 2, 10, 0),
            external_id="same",
        )
        cash_before = service.account().cash
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2024, 1, 2, 10, 0),
            external_id="same",
        )
        assert service.account().cash == cash_before
        service.record_cash_flow(
            flow_type="deposit",
            amount=1000,
            occurred_at=datetime(2024, 1, 2, 15, 0),
            external_id="dep-1",
        )
        service.apply_corporate_action(
            ts_code="000001.SZ",
            ex_date=date(2024, 1, 5),
            action_type="cash_dividend",
            cash_per_share=Decimal("0.5"),
        )
        state = service.state()
        assert state.positions[0].shares == 1000
        assert service.account().cash > cash_before


def test_csv_imports_are_idempotent(tmp_path: Path):
    reset_engine()
    init_db("sqlite://")
    templates = write_templates(tmp_path)
    assert len(templates) == 5
    executions = tmp_path / "executions.csv"
    executions.write_text(
        "executed_at,ts_code,side,price,shares,fees,plan_id,source,external_id,note\n"
        "2024-01-02T10:00:00,000001.SZ,buy,10,1000,5.1,,csv,,\n",
        encoding="utf-8",
    )
    positions = tmp_path / "positions.csv"
    positions.write_text(
        "ts_code,sector,shares,available_shares,cost_price,current_stop,current_price,strategy,sub_mode,plan_id,opened_at\n"
        "000002.SZ,银行,500,500,20,18,21,A,a_shallow,,\n",
        encoding="utf-8",
    )
    cash_flows = tmp_path / "cash_flows.csv"
    cash_flows.write_text(
        "occurred_at,flow_type,amount,external_id,note\n"
        "2024-01-02T09:00:00,deposit,100000,cash-1,\n",
        encoding="utf-8",
    )
    with session_scope() as session:
        service = _service(session, "csv")
        first = import_csv(
            session=session,
            service=service,
            config=service.config,
            kind="executions",
            path=executions,
        )
        assert first.imported == 1
        second = import_csv(
            session=session,
            service=service,
            config=service.config,
            kind="executions",
            path=executions,
        )
        assert second.imported == 0 and second.duplicates == 1
        pos = import_csv(
            session=session,
            service=service,
            config=service.config,
            kind="positions",
            path=positions,
        )
        assert pos.imported == 1
        cash = import_csv(
            session=session,
            service=service,
            config=service.config,
            kind="cash_flows",
            path=cash_flows,
        )
        assert cash.imported == 1
        assert len(AccountRepository.to_state(session, "csv").positions) == 2
        # An imported position must carry lots so later sells can reduce FIFO.
        service.record_execution(
            ts_code="000002.SZ",
            side=ExecutionSide.SELL,
            price=Decimal("21"),
            shares=500,
            executed_at=datetime(2024, 1, 3, 10, 0),
            external_id="sell-imported",
        )
        assert all(
            position.shares == 0
            for position in AccountRepository.to_state(session, "csv").positions
            if position.ts_code == "000002.SZ"
        )


def test_future_corporate_action_is_applied_on_ex_date():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2024, 1, 2, 10, 0),
            external_id="buy-future",
        )
        service.apply_corporate_action(
            ts_code="000001.SZ",
            ex_date=date(2024, 2, 1),
            action_type="stock_dividend",
            share_ratio=0.5,
            apply_now=False,
        )
        assert service.state().positions[0].shares == 1000

        assert service.apply_due_corporate_actions(date(2024, 1, 31)) == 0
        assert service.state().positions[0].shares == 1000

        assert service.apply_due_corporate_actions(date(2024, 2, 1)) == 1
        assert service.state().positions[0].shares == 1500


def test_expired_plans_are_marked_and_removed_from_pending_risk():
    from app.db.models import PlanRow
    from app.db.repositories import PlanRepository
    from app.domain.enums import PlanStatus

    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        row = PlanRow(
            id="plan-expired",
            account_id="default",
            trade_date=date(2024, 1, 2),
            ts_code="000001.SZ",
            sector="银行",
            strategy="A",
            status=PlanStatus.PENDING_EXECUTION.value,
            expires_on=date(2024, 1, 3),
            plan_json={"status": PlanStatus.PENDING_EXECUTION.value},
        )
        session.add(row)
        session.flush()
        assert PlanRepository.pending_exposures(session) == []
        assert PlanRepository.expire_overdue(session, date(2024, 1, 4)) == 1
        assert row.status == PlanStatus.EXPIRED.value
        assert row.plan_json["cancel_reason"] == "expired"


def test_manual_execution_respects_plan_price_and_expiry():
    from app.db.repositories import PlanRepository
    from tests.test_backtest import _plan

    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        plan = _plan(date(2024, 1, 2))
        PlanRepository.save(session, plan)
        with pytest.raises(PortfolioError, match="最高接受价"):
            service.record_execution(
                ts_code="000001.SZ",
                side=ExecutionSide.BUY,
                price=Decimal("10.5"),
                shares=1000,
                executed_at=datetime(2024, 1, 3, 10, 0),
                plan_id=plan.id,
                strict_plan=True,
            )
        with pytest.raises(PortfolioError, match="有效期"):
            service.record_execution(
                ts_code="000001.SZ",
                side=ExecutionSide.BUY,
                price=Decimal("10.0"),
                shares=1000,
                executed_at=datetime(2024, 1, 10, 10, 0),
                plan_id=plan.id,
                strict_plan=True,
            )
        # A real fill that deviates from the plan is still recorded, with an
        # audit warning; refusing it would make the broker account unknowable.
        deviated = service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10.5"),
            shares=1000,
            executed_at=datetime(2024, 1, 3, 10, 0),
            plan_id=plan.id,
            external_id="deviation-1",
        )
        assert deviated.shares == 1000
        assert "最高接受价" in (deviated.plan_deviation or "")
        execution = service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10.0"),
            shares=1000,
            executed_at=datetime(2024, 1, 3, 10, 5),
            external_id="fill-2",
        )
        assert execution.shares == 1000
        assert deviated.strategy == "A"


def test_cancel_order_rejects_filled_and_is_idempotent():
    from app.db.models import OrderRow
    from app.portfolio.service import cancel_order

    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        AccountRepository.ensure(session)
        session.add(
            OrderRow(
                id="order-1",
                account_id="default",
                plan_id="p",
                ts_code="000001.SZ",
                side="buy",
                price=Decimal("10"),
                shares=1000,
                status="filled",
            )
        )
        session.flush()
        with pytest.raises(PortfolioError, match="filled"):
            cancel_order(session, "order-1", "user")
        session.get(OrderRow, "order-1").status = "pending"
        cancel_order(session, "order-1", "user")
        cancel_order(session, "order-1", "user")
        assert session.get(OrderRow, "order-1").status == "cancelled"
