from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from app.backtest.simulator import Simulator
from app.core.full_config import load_full_config
from app.db.repositories import (
    AccountRepository,
    NotificationRepository,
    PlanRepository,
    PositionRepository,
)
from app.db.session import init_db, reset_engine, session_scope
from app.domain.enums import (
    EntryType,
    ExecutionSide,
    ExitSession,
    PlanStatus,
    PositionAction,
    SentimentStage,
    Strategy,
    SubMode,
    Template,
    ValuationStatus,
)
from app.domain.models import (
    AccountState,
    Candidate,
    EntryPlan,
    ExitRules,
    PositionPlan,
    StopPlan,
    TakeProfitPlan,
    TakeProfitTarget,
    TradePlan,
)
from app.domain.money import money, price
from app.portfolio.decisions import PositionDecisionService
from app.portfolio.market import PricePoint
from app.portfolio.service import AccountService, PortfolioError
from app.strategies.b2_intraday import add_trading_minutes

SHANGHAI = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 10, 30)


def _plan(
    *,
    strategy: Strategy = Strategy.A,
    sub_mode: SubMode = SubMode.A_SHALLOW,
    entry: float = 10.0,
    stop: float = 9.5,
    shares: int = 1000,
    targets: list[tuple[float, float]] | None = None,
    is_add_on: bool = False,
    expires_on: date | None = None,
    exit_rules: ExitRules | None = None,
    code: str = "000001.SZ",
) -> TradePlan:
    candidate = Candidate(
        ts_code=code,
        trade_date=DAY,
        template=Template.B if strategy == Strategy.B else Template.A,
        setup_type="test",
        horizon="mid",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={"close": entry, "atr14_raw": 0.5},
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv",
        data_version="dv",
        strategy=strategy,
        sub_mode=sub_mode,
    )
    expiry = expires_on or DAY + timedelta(days=2)
    entry_plan = EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=price(entry),
        validity_days=2,
        max_chase_pct=0.01,
        cancel_conditions=[],
        expires_on=expiry,
    )
    stop_plan = StopPlan(
        initial_stop=price(stop),
        stop_distance_pct=(entry - stop) / entry,
        atr14=0.5,
        structure_ref="test",
        time_stop_days=10,
        trailing_rules=[],
        r_value=price(entry - stop),
    )
    position = PositionPlan(
        template=Template.B if strategy == Strategy.B else Template.A,
        risk_pct=0.025,
        risk_amount=money(25000),
        shares=shares,
        position_value=money(entry * shares),
        position_pct=0.025,
        heat_before=0.0,
        heat_after=0.0025,
        constraints_checked=[],
        adjustments=[],
        is_add_on=is_add_on,
    )
    target_rows = targets or [(entry + 2 * (entry - stop), 0.5)]
    take_profit = TakeProfitPlan(
        targets=[
            TakeProfitTarget(
                r_multiple=max(0.01, (target - entry) / (entry - stop)),
                price=price(target),
                fraction=fraction,
            )
            for target, fraction in target_rows
        ],
        trailing_rule="trail",
        time_exit_rule="time",
    )
    return TradePlan(
        candidate=candidate,
        entry_plan=entry_plan,
        stop_plan=stop_plan,
        position_plan=position,
        take_profit_plan=take_profit,
        expires_on=expiry,
        exit_rules=exit_rules,
        strategy_version="sv",
        data_version="dv",
        code_version="test",
        strategy=strategy,
        sub_mode=sub_mode,
        confirmation_price=price(entry),
        max_acceptable_price=price(entry * 1.01),
        execution_low=price(entry),
        execution_high=price(entry * 1.01),
    ).model_copy(update={"status": PlanStatus.PENDING_EXECUTION})


def _service(session, account_id: str = "default") -> AccountService:
    service = AccountService(session, load_full_config(), account_id)
    service.configure_risk(equity=1_000_000, cash=1_000_000)
    return service


def _point(
    code: str = "000001.SZ",
    *,
    open_: float = 10.0,
    high: float = 10.0,
    low: float = 10.0,
    close: float = 10.0,
    price_value: float | None = None,
    status: ValuationStatus = ValuationStatus.OK,
    data_time: datetime | None = None,
    issues: tuple[str, ...] = (),
) -> PricePoint:
    return PricePoint(
        ts_code=code,
        open=open_,
        high=high,
        low=low,
        close=close,
        price=close if price_value is None else price_value,
        data_time=data_time or datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
        status=status,
        issues=issues,
    )


def test_b_position_tightens_protection_and_suggests_target_reduction():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        plan = _plan(
            strategy=Strategy.B,
            sub_mode=SubMode.B2,
            stop=9.5,
            shares=1000,
            targets=[(11.0, 0.5)],
            exit_rules=ExitRules(
                trailing="breakeven_1r_2r_ma5_atr",
                time_exit="no_new_high",
                time_stop_days=3,
            ),
        )
        PlanRepository.save(session, plan, trade_date=DAY)
        execution = service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2026, 10, 30, 10, 0),
            plan_id=plan.id,
            external_id="b-buy",
        )
        assert execution.strategy == "B"
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert float(position.initial_r) == pytest.approx(0.5)
        decision = PositionDecisionService(
            session, load_full_config()
        ).evaluate_all(
            DAY,
            points={
                "000001.SZ": _point(
                    high=11.6, low=10.4, close=11.2, open_=10.2
                )
            },
            session_type=ExitSession.DAILY,
            now=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
        )[0]
        assert decision.action == PositionAction.REDUCE
        assert decision.reason_code.startswith("target")
        assert decision.proposed_stop is not None
        assert float(decision.proposed_stop) >= 10.0
        assert float(position.current_stop) == 9.5
        assert position.pending_stop is not None
        assert float(position.pending_stop) >= 10.0
        assert position.pending_stop_effective_at is not None
        assert decision.suggested_shares == 500
        assert decision.targets[0].status == "pending"


def test_missing_plan_or_stop_is_data_anomaly_not_safe_hold():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2026, 10, 30, 10, 0),
            external_id="unlinked-buy",
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert float(position.current_stop) == 0.0
        assert position.needs_reconciliation
        decision = PositionDecisionService(
            session, load_full_config()
        ).evaluate_all(
            DAY,
            points={"000001.SZ": _point(close=10.0, high=10.1, low=9.9)},
            now=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
        )[0]
        assert decision.action == PositionAction.DATA_ANOMALY
        assert decision.reason_code == "stop_missing"
        assert "待补全" in decision.explanation


def test_partial_initial_fills_are_one_batch_and_add_ons_are_counted():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        plan = _plan(shares=300)
        add_one = _plan(shares=100, is_add_on=True)
        add_two = _plan(shares=100, is_add_on=True)
        for item in (plan, add_one, add_two):
            PlanRepository.save(session, item, trade_date=DAY)
        for index in range(3):
            service.record_execution(
                ts_code="000001.SZ",
                side=ExecutionSide.BUY,
                price=Decimal("10"),
                shares=100,
                executed_at=datetime(2026, 10, 30, 10, index),
                plan_id=plan.id,
                external_id=f"initial-{index}",
            )
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert position.add_on_count == 0
        assert position.initial_shares == 300
        assert position.primary_strategy == "A"
        assert sorted(lot.add_index for lot in position.lots) == [0, 0, 0]

        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10.5"),
            shares=100,
            executed_at=datetime(2026, 10, 31, 10, 0),
            plan_id=add_one.id,
            external_id="add-1",
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert position.add_on_count == 1
        assert position.initial_shares == 300
        assert position.primary_strategy == "A"
        assert position.plan_id == plan.id

        service.refresh_available(date(2026, 11, 1))
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.SELL,
            price=Decimal("10.8"),
            shares=100,
            executed_at=datetime(2026, 11, 1, 10, 0),
            external_id="sell-part",
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert position.add_on_count == 1
        assert position.initial_shares == 300

        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10.9"),
            shares=100,
            executed_at=datetime(2026, 11, 2, 10, 0),
            plan_id=add_two.id,
            external_id="add-2",
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert position.add_on_count == 2
        assert position.primary_strategy == "A"

        service.refresh_available(date(2026, 11, 3))
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.SELL,
            price=Decimal("11"),
            shares=position.shares,
            executed_at=datetime(2026, 11, 3, 10, 0),
            external_id="close-all",
        )
        assert PositionRepository.get(session, "default", "000001.SZ") is None

        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("12"),
            shares=200,
            executed_at=datetime(2026, 11, 4, 10, 0),
            plan_id=plan.id,
            external_id="reopen",
        )
        reopened = PositionRepository.get(session, "default", "000001.SZ")
        assert reopened.add_on_count == 0
        assert reopened.initial_shares == 200


def test_plan_linkage_validates_owner_and_security_and_records_deviation():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        plan = _plan(expires_on=date(2026, 10, 30))
        PlanRepository.save(session, plan, trade_date=DAY)
        with pytest.raises(PortfolioError, match="security"):
            service.record_execution(
                ts_code="000002.SZ",
                side=ExecutionSide.BUY,
                price=Decimal("10"),
                shares=100,
                executed_at=datetime(2026, 10, 30, 10, 0),
                plan_id=plan.id,
            )
        execution = service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10.5"),
            shares=100,
            executed_at=datetime(2026, 10, 31, 10, 0),
            plan_id=plan.id,
            external_id="late-fill",
        )
        assert execution.strategy == "A"
        assert execution.sector == "银行"
        assert execution.plan_deviation
        assert "最高接受价" in execution.plan_deviation
        assert "有效期" in execution.plan_deviation


def test_valuation_updates_equity_and_excludes_deposits_from_peak():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        AccountRepository.ensure(session, equity=100_000, cash=90_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=9,
            current_price=10,
            primary_strategy="A",
            initial_r=1,
            initial_shares=1000,
            opened_at=datetime(2026, 10, 25, 10, 0),
        )
        service = AccountService(session, load_full_config())
        service.revalue(
            DAY,
            {"000001.SZ": _point(close=12, high=12, low=11.9, open_=11)},
        )
        account = AccountRepository.get(session)
        assert float(account.equity) == pytest.approx(102_000)
        assert float(account.peak_equity) == pytest.approx(102_000)

        service.revalue(DAY, {})
        account = AccountRepository.get(session)
        assert float(account.equity) == pytest.approx(102_000)
        assert account.valuation_incomplete
        assert "000001.SZ" in (account.valuation_note or "")

        service.record_cash_flow(
            flow_type="deposit",
            amount=5000,
            occurred_at=datetime(2026, 10, 30, 16, 0),
            external_id="dep-1",
        )
        account = AccountRepository.get(session)
        state = AccountRepository.to_state(session)
        assert float(account.cash) == pytest.approx(95_000)
        assert float(state.equity) == pytest.approx(107_000)
        assert float(state.net_deposits) == pytest.approx(5_000)
        assert state.drawdown_pct == pytest.approx(0.0)
        assert float(state.peak_equity) == pytest.approx(102_000)


def test_live_and_backtest_use_same_exit_rule_without_future_stop():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        plan = _plan(stop=9.5, targets=[(11.5, 0.5)])
        PlanRepository.save(session, plan, trade_date=DAY)
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2026, 10, 30, 10, 0),
            plan_id=plan.id,
            external_id="future-stop",
        )
        point = _point(open_=10.2, high=11.6, low=9.6, close=11.2)
        decision = PositionDecisionService(
            session, load_full_config()
        ).evaluate_all(
            DAY,
            points={"000001.SZ": point},
            now=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
        )[0]
        assert decision.action == PositionAction.REDUCE
        assert decision.reason_code.startswith("target")
        assert float(decision.proposed_stop) >= 10.5

        sim = Simulator(load_full_config(), 1_000_000, 1_000_000, 1_000_000)
        sim.add_plans([plan], DAY - timedelta(days=2))
        entry_day = DAY - timedelta(days=1)
        sim.process_day(
            entry_day,
            pl.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": entry_day,
                        "open": 10.0,
                        "high": 10.1,
                        "low": 9.9,
                        "close": 10.0,
                        "vol": 1e8,
                        "amount": 1e10,
                    }
                ]
            ),
        )
        assert "000001.SZ" in sim.positions
        trades = sim.process_day(
            DAY,
            pl.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": DAY,
                        "open": 10.2,
                        "high": 11.6,
                        "low": 9.6,
                        "close": 11.2,
                        "vol": 1e8,
                        "amount": 1e10,
                    }
                ]
            ),
        )
        assert trades
        assert all(trade.exit_reason == "target" for trade in trades)
        assert sim.positions["000001.SZ"].current_stop == pytest.approx(9.5)
        assert sim.positions["000001.SZ"].pending_stop is not None
        assert sim.positions["000001.SZ"].pending_stop >= 10.5


def test_b2_previous_trading_day_watchlist_lunch_validity_and_restart():
    reset_engine()
    init_db("sqlite://")
    from app.core.full_config import load_full_config as load_config
    from app.db.repositories import AccountRepository, PlanRepository, SnapshotRepository
    from app.db.session import session_scope
    from app.domain.models import Candidate
    from app.jobs.b2 import B2Monitor
    from tests.strategy_helpers import make_bars, make_context
    from tests.test_b2_intraday import _bars, _MinuteSource

    friday = date(2026, 10, 30)
    monday = date(2026, 11, 2)

    class MondayMinutes(_MinuteSource):
        def load(self, ts_code, day):
            return _bars().with_columns(
                pl.col("trade_time").str.replace(
                    "2026-10-30", "2026-11-02"
                )
            )

    class Provider:
        def load_trade_cal(self, start, end):
            days = []
            cursor = start
            while cursor <= end:
                if cursor.weekday() < 5:
                    days.append(cursor)
                cursor += timedelta(days=1)
            return pl.DataFrame({"cal_date": days, "is_open": [1 for _ in days]})

    context = make_context(make_bars([10.0] * 40))
    candidate = Candidate(
        ts_code="000001.SZ",
        trade_date=friday,
        template=Template.A,
        setup_type="b2_watch",
        horizon="short",
        sector="测试",
        leader_rank=1,
        leader_score=80.0,
        score=80.0,
        reasons=["test"],
        features={"close": 10.0, "atr14_raw": 0.5},
        sentiment_stage=context.market.sentiment_stage,
        strategy_version="test",
        data_version="test",
        strategy=Strategy.B,
        sub_mode=SubMode.B2,
        evidence={"stage": "watchlist"},
    )
    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        state = AccountRepository.to_state(session)
        SnapshotRepository.save(
            session,
            account_id="default",
            trade_date=friday,
            state=state,
            detail={
                "candidates": [candidate.model_dump(mode="json")],
                "regime": context.market.model_dump(mode="json"),
            },
        )
    monitor = B2Monitor(Provider(), load_config(), MondayMinutes())
    now = datetime(2026, 11, 2, 10, 25, tzinfo=SHANGHAI)
    with session_scope() as session:
        result = monitor.check(monday, now=now, session=session)
        assert result.watchlist_date == friday
        assert result.signals == ["000001.SZ"]
        first_plans = PlanRepository.list_for_date(session, monday)
        assert len(first_plans) == 1
    # A restarted monitor must not create a second plan for the same signal.
    monitor2 = B2Monitor(Provider(), load_config(), MondayMinutes())
    with session_scope() as session:
        result2 = monitor2.check(monday, now=now, session=session)
        assert result2.signals == ["000001.SZ"]
        assert len(PlanRepository.list_for_date(session, monday)) == 1

    signal = datetime(2026, 10, 30, 11, 25, tzinfo=SHANGHAI)
    assert add_trading_minutes(signal, 10).time() == datetime.strptime(
        "13:05", "%H:%M"
    ).time()


def test_intraday_rejects_stale_and_missing_time_and_keeps_other_positions():
    reset_engine()
    init_db("sqlite://")

    class Provider:
        def load_trade_cal(self, start, end):
            days = [
                date(2026, 11, 2),
            ]
            return pl.DataFrame({"cal_date": days, "is_open": [1 for _ in days]})

    class Quotes:
        def get_quote(self, code):
            if code == "000001.SZ":
                return 9.0, datetime(2020, 1, 2, 10, 0, tzinfo=SHANGHAI)
            if code == "000002.SZ":
                return 10.0, datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI)
            return None

    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        for code, stop in (("000001.SZ", 10.0), ("000002.SZ", 9.0)):
            PositionRepository.upsert(
                session,
                account_id="default",
                ts_code=code,
                shares=1000,
                available_shares=1000,
                cost_price=10,
                current_stop=stop,
                current_price=10.5,
                primary_strategy="A",
                initial_r=0.5,
                initial_shares=1000,
                opened_at=datetime(2026, 10, 30, 10, 0),
            )
    from app.jobs.intraday import IntradayMonitor

    monitor = IntradayMonitor(
        Quotes(), calendar_provider=Provider(), config=load_full_config()
    )
    result = monitor.check(
        date(2026, 11, 2),
        now=datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI),
    )
    assert result["checked"] == 1
    assert result["degraded"]
    with session_scope() as session:
        stale = PositionRepository.get(session, "default", "000001.SZ")
        fresh = PositionRepository.get(session, "default", "000002.SZ")
        assert float(stale.current_price) == 10.5
        assert float(fresh.current_price) == 10.0
        codes = {item.code for item in NotificationRepository.list_recent(session)}
        assert "hard_stop_breached" not in codes


def test_intraday_deduplicates_alert_and_emits_recovery():
    reset_engine()
    init_db("sqlite://")

    class Provider:
        def load_trade_cal(self, start, end):
            return pl.DataFrame(
                {"cal_date": [date(2026, 11, 2)], "is_open": [1]}
            )

    class Quotes:
        price = 9.5

        def get_quote(self, code):
            return self.price, datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI)

    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=10,
            current_price=10.5,
            primary_strategy="A",
            initial_r=0.5,
            initial_shares=1000,
            opened_at=datetime(2026, 10, 30, 10, 0),
        )
    from app.jobs.intraday import IntradayMonitor

    quotes = Quotes()
    monitor = IntradayMonitor(
        quotes, calendar_provider=Provider(), config=load_full_config()
    )
    now = datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI)
    monitor.check(date(2026, 11, 2), now=now)
    monitor.check(date(2026, 11, 2), now=now)
    with session_scope() as session:
        stops = [
            item
            for item in NotificationRepository.list_recent(session)
            if item.code == "hard_stop_breached"
        ]
        assert len(stops) == 1
    quotes.price = 10.6
    monitor.check(date(2026, 11, 2), now=now)
    with session_scope() as session:
        assert any(
            item.code == "position_recovered"
            for item in NotificationRepository.list_recent(session)
        )


def test_backtest_keeps_unfilled_plan_until_expiry():
    config = load_full_config()
    sim = Simulator(config, 100_000, 100_000, 100_000)
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    d2 = date(2024, 1, 4)
    plan = _plan(shares=1000, expires_on=d2)
    # The plan is created after the close on d0; it is valid through d2.
    sim.add_plans([plan], d0)
    sim.process_day(
        d1,
        pl.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": d1,
                    "open": 9.6,
                    "high": 9.9,
                    "low": 9.5,
                    "close": 9.8,
                    "vol": 1e8,
                    "amount": 1e10,
                }
            ]
        ),
    )
    assert "000001.SZ" not in sim.positions
    assert len(sim.pending) == 1
    sim.process_day(
        d2,
        pl.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": d2,
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "vol": 1e8,
                    "amount": 1e10,
                }
            ]
        ),
    )
    assert "000001.SZ" in sim.positions
    assert sim.pending == []


def test_backtest_fees_reconcile_with_equity_and_cancel_is_explicit():
    config = load_full_config()
    sim = Simulator(config, 100_000, 100_000, 100_000)
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    d2 = date(2024, 1, 4)
    sim.add_plans([_plan(shares=1000)], d0)
    sim.process_day(
        d1,
        pl.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": d1,
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "vol": 1e8,
                    "amount": 1e10,
                }
            ]
        ),
    )
    sim.process_day(
        d2,
        pl.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": d2,
                    "open": 9.4,
                    "high": 9.6,
                    "low": 9.3,
                    "close": 9.4,
                    "vol": 1e8,
                    "amount": 1e10,
                }
            ]
        ),
    )
    metrics = sim.metrics()
    assert metrics["entry_fees_paid"] > 0
    assert metrics["exit_fees_paid"] > 0
    assert metrics["final_equity"] == pytest.approx(
        sim.initial_equity + metrics["realized_pnl"] + metrics["unrealized_pnl"],
        abs=1e-4,
    )

    sim2 = Simulator(config, 100_000, 100_000, 100_000)
    plan = _plan(shares=1000)
    sim2.add_plans([plan], d0)
    assert sim2.cancel_plan(plan.id, d0, "user cancel")
    assert sim2.pending == []
    assert sim2.cancelled_orders


def test_daily_revalues_before_pipeline_and_does_not_overwrite_with_stale_state():
    from types import SimpleNamespace

    from app.db.repositories import SnapshotRepository
    from app.domain.enums import GlobalRiskMode
    from app.jobs.daily import DailyJobService

    reset_engine()
    init_db("sqlite://")

    class Provider:
        def load_bars(self, start, end):
            return pl.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": end,
                        "open": 11.9,
                        "high": 12.0,
                        "low": 11.8,
                        "close": 12.0,
                        "vol": 1e8,
                        "amount": 1e10,
                    }
                ]
            )

        def load_trade_cal(self, start, end):
            days = [
                start + timedelta(days=index)
                for index in range((end - start).days + 1)
            ]
            return pl.DataFrame({"cal_date": days, "is_open": [1 for _ in days]})

    class Regime(SimpleNamespace):
        def model_dump(self, **_kwargs):
            return {
                "global_risk": self.global_risk.value,
                "degraded_flags": [],
                "data_version": "dv",
            }

    class Pipeline:
        def __init__(self):
            self.account = None

        def run(self, day, account, **kwargs):
            self.account = account
            stale = AccountState(
                equity=Decimal("100000"),
                cash=Decimal("90000"),
                peak_equity=Decimal("100000"),
            )
            return SimpleNamespace(
                regime=Regime(
                    global_risk=GlobalRiskMode.NORMAL,
                    degraded_flags=[],
                    data_version="dv",
                ),
                account=stale,
                plans=[],
                candidates=[],
                observations=[],
                rejected=[],
                by_strategy={},
            )

    class Reports:
        def generate(self, result, **kwargs):
            return {"markdown": "", "html": "", "markdown_path": "", "html_path": ""}

    with session_scope() as session:
        AccountRepository.ensure(session, equity=100_000, cash=90_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=9,
            current_price=10,
            primary_strategy="A",
            initial_r=1,
            initial_shares=1000,
            opened_at=datetime(2026, 10, 25, 10, 0),
        )
    pipeline = Pipeline()
    service = DailyJobService(
        provider=Provider(),
        config=load_full_config(),
        pipeline=pipeline,
        reports=Reports(),
    )
    outcome = service.run(DAY)
    assert outcome.status == "succeeded"
    assert pipeline.account is not None
    assert float(pipeline.account.equity) == pytest.approx(102_000)
    with session_scope() as session:
        assert float(AccountRepository.get(session).equity) == pytest.approx(102_000)
        snapshot = SnapshotRepository.list_recent(session, "default", limit=1)[-1]
        assert float(snapshot.equity) == pytest.approx(102_000)
        assert snapshot.detail_json["decisions"]


def test_data_update_reports_unavailable_without_faking_success(tmp_path, monkeypatch):
    from app.data.store import ParquetStore
    from app.jobs.data_update import DataUpdateService

    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    service = DataUpdateService(
        store=store, config=load_full_config(), token=None
    )
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        result = service.run(date(2026, 11, 2), session)
    assert not result.data_ready
    statuses = {stage.name: stage.status for stage in result.stages}
    assert statuses["reference"] == "unavailable"
    assert statuses["incremental_sync"] == "unavailable"
    assert statuses["assemble"] == "failed"
    assert statuses["events"] == "unavailable"
    assert any("关键数据未就绪" in issue for issue in result.issues)
    assert not any(
        stage.status == "succeeded"
        for stage in result.stages
        if stage.name in ("reference", "incremental_sync")
    )


def test_event_sync_distinguishes_no_new_events_from_unavailable(tmp_path):
    from app.data.store import ParquetStore
    from app.db.repositories import SettingRepository
    from app.jobs.data_update import DataUpdateService

    class EmptyClient:
        def fetch(self, endpoint, **kwargs):
            return pl.DataFrame()

    class BrokenClient:
        def fetch(self, endpoint, **kwargs):
            raise RuntimeError("permission denied")

    reset_engine()
    init_db("sqlite://")
    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    service = DataUpdateService(
        store=store,
        config=load_full_config(),
        token="x",
        event_client_factory=lambda _token: EmptyClient(),
    )
    with session_scope() as session:
        status, counts, available = service._event_sync(
            session, date(2026, 11, 3)
        )
        assert status == "no_new_events"
        assert available is True
        assert SettingRepository.get(session, "event_sync")["status"] == (
            "no_new_events"
        )
    broken = DataUpdateService(
        store=store,
        config=load_full_config(),
        token="x",
        event_client_factory=lambda _token: BrokenClient(),
    )
    with session_scope() as session:
        status, counts, available = broken._event_sync(
            session, date(2026, 11, 3)
        )
        assert status == "failed"
        assert available is False
        assert counts["error"]


def test_daily_stops_new_signals_but_keeps_protection_when_data_not_ready():
    from types import SimpleNamespace

    from app.jobs.daily import DailyJobService
    from app.jobs.data_update import DataUpdateResult, StageResult

    reset_engine()
    init_db("sqlite://")

    class Update:
        def run(self, day, session):
            return DataUpdateResult(
                as_of=day,
                status="incomplete",
                data_ready=False,
                stages=[StageResult(name="quality", status="failed")],
                issues=["关键数据未就绪"],
                events_available=False,
            )

    class Pipeline:
        def __init__(self):
            self.calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("pipeline must not run")

    class Reports:
        def generate(self, result, **kwargs):
            return {"markdown": "", "html": "", "markdown_path": "", "html_path": ""}

    pipeline = Pipeline()
    service = DailyJobService(
        provider=SimpleNamespace(store=None),
        config=load_full_config(),
        pipeline=pipeline,
        reports=Reports(),
        data_update=Update(),
    )
    outcome = service.run(DAY, run_data_update=True)
    assert outcome.status == "succeeded"
    assert outcome.incomplete
    assert pipeline.calls == 0
    assert outcome.data_stages
    with session_scope() as session:
        assert any(
            item.code == "daily_data_incomplete"
            for item in NotificationRepository.list_recent(session)
        )


def test_b2_multiple_signals_use_updated_account_risk_budget():
    from app.db.repositories import SnapshotRepository
    from app.domain.models import PositionPlan
    from app.jobs.b2 import B2Monitor
    from tests.test_b2_intraday import _bars

    reset_engine()
    init_db("sqlite://")

    class Provider:
        def load_trade_cal(self, start, end):
            days = []
            cursor = start
            while cursor <= end:
                if cursor.weekday() < 5:
                    days.append(cursor)
                cursor += timedelta(days=1)
            return pl.DataFrame({"cal_date": days, "is_open": [1 for _ in days]})

    class Minutes:
        def load(self, ts_code, day):
            return _bars().with_columns(
                pl.lit(ts_code).alias("ts_code"),
                pl.col("trade_time").str.replace(
                    "2026-10-30", "2026-11-02"
                ),
            )

    class RecordingSizer:
        def __init__(self):
            self.pending_seen = []

        def size(self, candidate, entry, stop, account, regime, config):
            self.pending_seen.append(len(account.pending_plans))
            return PositionPlan(
                template=Template.A,
                risk_pct=0.02,
                risk_amount=Decimal("1000"),
                shares=100,
                position_value=Decimal("1000"),
                position_pct=0.001,
                heat_before=account.heat,
                heat_after=account.heat + 0.001,
            )

    from tests.strategy_helpers import make_bars, make_context

    context = make_context(make_bars([10.0] * 40))
    friday = date(2026, 10, 30)
    monday = date(2026, 11, 2)
    candidates = [
        Candidate(
            ts_code=code,
            trade_date=friday,
            template=Template.A,
            setup_type="b2_watch",
            horizon="short",
            sector="测试",
            leader_rank=1,
            leader_score=80.0,
            score=80.0,
            reasons=["test"],
            features={"close": 10.0, "atr14_raw": 0.5},
            sentiment_stage=context.market.sentiment_stage,
            strategy_version="test",
            data_version="test",
            strategy=Strategy.B,
            sub_mode=SubMode.B2,
            evidence={"stage": "watchlist"},
        )
        for code in ("000001.SZ", "000002.SZ")
    ]
    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        state = AccountRepository.to_state(session)
        SnapshotRepository.save(
            session,
            account_id="default",
            trade_date=friday,
            state=state,
            detail={
                "candidates": [item.model_dump(mode="json") for item in candidates],
                "regime": context.market.model_dump(mode="json"),
            },
        )
    sizer = RecordingSizer()
    monitor = B2Monitor(
        provider=Provider(),
        config=load_full_config(),
        minute_source=Minutes(),
        sizer=sizer,
    )
    with session_scope() as session:
        result = monitor.check(
            monday,
            now=datetime(2026, 11, 2, 10, 25, tzinfo=SHANGHAI),
            session=session,
        )
    assert result.signals == ["000001.SZ", "000002.SZ"]
    assert sizer.pending_seen == [0, 1]


def test_minute_source_rate_limit_is_global_across_symbols(monkeypatch):
    import app.jobs.b2 as b2_module

    class FakeClock:
        now = 0.0
        sleeps: list[float] = []

        @classmethod
        def monotonic(cls):
            return cls.now

        @classmethod
        def sleep(cls, seconds):
            cls.sleeps.append(seconds)
            cls.now += seconds

    monkeypatch.setattr(b2_module.clock, "monotonic", FakeClock.monotonic)
    monkeypatch.setattr(b2_module.clock, "sleep", FakeClock.sleep)
    source = b2_module.TushareMinuteSource(
        token="token", min_interval_seconds=60
    )
    monkeypatch.setattr(
        source, "_fetch", lambda ts_code, day, token: pl.DataFrame()
    )
    source.load("000001.SZ", DAY)
    source.load("000002.SZ", DAY)
    assert source.requests_made == 2
    assert FakeClock.sleeps == [60, 60]


def test_report_contains_position_actions_from_shared_decisions(tmp_path):
    from app.domain.models import PositionDecision
    from app.jobs.reports import ReportService

    decision = PositionDecision(
        ts_code="000001.SZ",
        sector="银行",
        primary_strategy=Strategy.A,
        sub_mode=SubMode.A_SHALLOW,
        rule_version="sv",
        action=PositionAction.REDUCE,
        reason_code="target_1",
        explanation="触及分批止盈目标，建议减仓 500 股。",
        total_shares=1000,
        available_shares=1000,
        suggested_shares=500,
        current_stop=Decimal("10.5"),
        proposed_stop=Decimal("10.5"),
        targets=[
            {
                "price": "11.0",
                "fraction": 0.5,
                "planned_shares": 500,
                "filled_shares": 0,
                "status": "pending",
            }
        ],
        valuation_status=ValuationStatus.OK,
    )
    account = AccountState(
        equity=Decimal("102000"),
        cash=Decimal("90000"),
        peak_equity=Decimal("102000"),
    )
    report = ReportService(tmp_path).generate(
        None,
        as_of=DAY,
        account=account,
        decisions=[decision],
        incomplete=["测试数据不完整"],
        position_reviewed=True,
    )
    assert "持仓行动建议" in report["markdown"]
    assert "000001.SZ" in report["markdown"]
    assert "触及分批止盈目标 1" in report["markdown"]
    assert "减仓" in report["markdown"]
    assert "数据不完整" in report["markdown"]
    assert "000001.SZ" in report["html"]
    assert "测试数据不完整" in report["html"]
    assert "<h2>今日摘要</h2>" in report["html"]
    assert "<table>" in report["html"]
    assert "<pre>" not in report["html"]


def test_user_max_drawdown_participates_in_plan_governance():
    from app.planning.errors import PositionRejectedError
    from app.planning.strategy_sizing import StrategyPositionSizer
    from app.risk.strategy_governor import StrategyRiskGovernor
    from app.strategies import c_platform
    from tests.strategy_helpers import make_context
    from tests.test_full_system import _c_bars

    context = make_context(_c_bars())
    candidate = next(
        item for item in c_platform.generate(context) if item.strategy == Strategy.C
    )
    from app.core.calendar import TradingCalendar
    from app.planning.strategy_planner import StrategyPlanner

    calendar = TradingCalendar(
        [context.as_of + timedelta(days=index) for index in range(10)]
    )
    parts = StrategyPlanner().plan_parts(candidate, calendar, context.config)
    account = AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal("1000000"),
        mode="real",
        risk_config_configured=True,
        max_drawdown=0.08,
    ).model_copy(update={"equity": Decimal("900000")})
    with pytest.raises(PositionRejectedError, match="drawdown"):
        StrategyPositionSizer().size(
            candidate,
            parts.entry,
            parts.stop,
            account,
            context.market,
            context.config,
        )
    plan = TradePlan(
        candidate=candidate,
        entry_plan=parts.entry,
        stop_plan=parts.stop,
        position_plan=PositionPlan(
            template=Template.A,
            risk_pct=0.02,
            risk_amount=Decimal("1000"),
            shares=100,
            position_value=Decimal("1000"),
            position_pct=0.001,
            heat_before=0.0,
            heat_after=0.001,
        ),
        take_profit_plan=parts.take_profit,
        expires_on=parts.entry.expires_on,
        strategy_version="sv",
        data_version=context.data_version,
        code_version="test",
        strategy=candidate.strategy,
        sub_mode=candidate.sub_mode,
    ).model_copy(update={"status": PlanStatus.DRAFT})
    outcome = StrategyRiskGovernor().apply(
        [plan], account, context.market, context.config
    )
    assert outcome[0].status == PlanStatus.REJECTED
    assert "drawdown" in " ".join(outcome[0].risk_decision.reasons)


def test_full_chain_plan_partial_fills_valuation_reduce_and_close():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        plan = _plan(shares=1000, targets=[(11.0, 0.5)])
        PlanRepository.save(session, plan, trade_date=DAY)
        for index in range(2):
            service.record_execution(
                ts_code="000001.SZ",
                side=ExecutionSide.BUY,
                price=Decimal("10"),
                shares=400,
                executed_at=datetime(2026, 10, 30, 10, index),
                plan_id=plan.id,
                external_id=f"chain-buy-{index}",
            )
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert position.shares == 800
        assert position.add_on_count == 0
        service.revalue(
            DAY,
            {"000001.SZ": _point(open_=10.3, high=11.2, low=10.1, close=11.0)},
        )
        decision = PositionDecisionService(
            session, load_full_config()
        ).evaluate_all(
            DAY,
            points={
                "000001.SZ": _point(
                    open_=10.3, high=11.2, low=10.1, close=11.0
                )
            },
            now=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
        )[0]
        assert decision.action == PositionAction.REDUCE
        assert decision.suggested_shares == 400
        service.refresh_available(DAY + timedelta(days=1))
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.SELL,
            price=Decimal("11.0"),
            shares=400,
            executed_at=datetime(2026, 10, 31, 10, 0),
            external_id="chain-reduce",
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert position.shares == 400
        assert position.add_on_count == 0
        state = AccountRepository.to_state(session)
        assert float(state.positions[0].targets[0].filled_shares) == 400
        decision2 = PositionDecisionService(
            session, load_full_config()
        ).evaluate_all(
            DAY + timedelta(days=1),
            points={
                "000001.SZ": _point(
                    open_=11.0, high=11.1, low=10.8, close=10.9
                )
            },
            now=datetime(2026, 10, 31, 15, 0, tzinfo=SHANGHAI),
        )[0]
        assert decision2.action in (
            PositionAction.HOLD,
            PositionAction.REDUCE,
            PositionAction.ADD_CONDITIONAL,
        )
        service.refresh_available(DAY + timedelta(days=2))
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.SELL,
            price=Decimal("10.9"),
            shares=400,
            executed_at=datetime(2026, 11, 1, 10, 0),
            external_id="chain-close",
        )
        assert PositionRepository.get(session, "default", "000001.SZ") is None


def test_backtest_distinguishes_invalidation_and_unconfirmed_cancel():
    config = load_full_config()
    sim = Simulator(config, 100_000, 100_000, 100_000)
    d0 = date(2024, 1, 2)
    plan_a = _plan(shares=1000)
    plan_b = _plan(shares=1000, code="000002.SZ")
    sim.add_plans([plan_a, plan_b], d0)
    assert sim.invalidate_plan(plan_a.id, d0, "structure gone")
    assert sim.invalidated_orders
    assert [order.plan.id for order in sim.pending] == [plan_b.id]
    assert sim.mark_cancel_unconfirmed(plan_b.id, d0, "request sent")
    assert sim.pending[0].state == "cancel_unconfirmed"
    assert sim.reserved_cash() > 0
    assert sim.confirm_cancel(plan_b.id, d0, "confirmed")
    assert sim.pending == []
    assert sim.cancelled_orders


def test_intraday_missing_and_future_timestamps_are_unknown_and_holiday_skips():
    reset_engine()
    init_db("sqlite://")

    class Provider:
        def __init__(self, open_day: bool):
            self.open_day = open_day

        def load_trade_cal(self, start, end):
            return pl.DataFrame(
                {
                    "cal_date": [date(2026, 11, 2)],
                    "is_open": [1 if self.open_day else 0],
                }
            )

    class Quotes:
        def __init__(self, value):
            self.value = value

        def get_quote(self, code):
            return self.value

    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=9,
            current_price=10.5,
            primary_strategy="A",
            initial_r=1,
            initial_shares=1000,
            opened_at=datetime(2026, 10, 30, 10, 0),
        )
    from app.jobs.intraday import IntradayMonitor, Quote

    now = datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI)
    missing = IntradayMonitor(
        Quotes(Quote(9.0, None, now)), calendar_provider=Provider(True)
    ).check(date(2026, 11, 2), now=now)
    assert missing["positions"][0]["status"] == "unknown"
    assert missing["alerts"] == 1  # data anomaly, not a stop alert
    future = IntradayMonitor(
        Quotes(
            Quote(9.0, datetime(2026, 11, 2, 11, 0, tzinfo=SHANGHAI), now)
        ),
        calendar_provider=Provider(True),
    ).check(date(2026, 11, 2), now=now)
    assert future["positions"][0]["status"] == "unknown"
    holiday = IntradayMonitor(
        Quotes(Quote(9.0, now, now)), calendar_provider=Provider(False)
    ).check(date(2026, 11, 2), now=now)
    assert holiday["skipped"] is True


def test_intraday_plan_expires_at_lunch_boundary():
    from app.db.models import PlanRow

    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        session.add(
            PlanRow(
                id="b2-lunch",
                account_id="default",
                trade_date=DAY,
                ts_code="000001.SZ",
                sector="银行",
                strategy="B",
                sub_mode="b2",
                status=PlanStatus.PENDING_EXECUTION.value,
                expires_on=DAY,
                expires_at=datetime(DAY.year, DAY.month, DAY.day, 13, 5, tzinfo=SHANGHAI),
                plan_json={"status": PlanStatus.PENDING_EXECUTION.value},
            )
        )
        session.flush()
        later = datetime(DAY.year, DAY.month, DAY.day, 13, 6, tzinfo=SHANGHAI)
        assert PlanRepository.expire_intraday(session, later) == 1
        assert session.get(PlanRow, "b2-lunch").status == PlanStatus.EXPIRED.value


def test_intraday_minute_fallback_is_explicitly_delayed():
    reset_engine()
    init_db("sqlite://")
    from app.jobs.intraday import IntradayMonitor
    from tests.test_b2_intraday import _bars

    class Provider:
        def load_trade_cal(self, start, end):
            return pl.DataFrame(
                {"cal_date": [date(2026, 11, 2)], "is_open": [1]}
            )

    class NoQuote:
        def get_quote(self, code):
            return None

    class Minutes:
        def load(self, code, day):
            return _bars().with_columns(
                pl.col("trade_time").str.replace(
                    "2026-10-30", "2026-11-02"
                )
            )

    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=9,
            current_price=10.0,
            primary_strategy="A",
            initial_r=1,
            initial_shares=1000,
            opened_at=datetime(2026, 10, 30, 10, 0),
        )
    monitor = IntradayMonitor(
        NoQuote(),
        minute_source=Minutes(),
        calendar_provider=Provider(),
        config=load_full_config(),
    )
    result = monitor.check(
        date(2026, 11, 2),
        now=datetime(2026, 11, 2, 10, 21, tzinfo=SHANGHAI),
    )
    assert result["checked"] == 1
    assert result["positions"][0]["source"] == "minute_delayed"
    assert result["positions"][0]["data_time"].endswith("10:20:00+08:00")


def test_valuation_does_not_use_bars_from_before_position_opened():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        AccountRepository.ensure(session, equity=100_000, cash=90_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            shares=1000,
            available_shares=0,
            cost_price=10,
            current_stop=0,
            current_price=10,
            needs_reconciliation=True,
            opened_at=datetime(2026, 11, 2, 10, 0),
        )
        service = AccountService(session, load_full_config())
        service.revalue(
            date(2026, 10, 27),
            {
                "000001.SZ": _point(
                    close=11.47,
                    high=11.6,
                    low=11.3,
                    open_=11.4,
                    data_time=datetime(2026, 10, 27, 15, 0, tzinfo=SHANGHAI),
                )
            },
        )
        account = AccountRepository.get(session)
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert float(position.current_price) == 10.0
        assert account.valuation_incomplete
        assert float(account.equity) == pytest.approx(100_000)


def test_f_position_exits_when_negative_event_is_verified():
    from app.db.repositories import EventRepository

    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        plan = _plan(
            strategy=Strategy.F,
            sub_mode=SubMode.F_CONTRACT,
            targets=[(11.0, 0.5)],
            exit_rules=ExitRules(
                trailing="breakeven_1r_2r_atr",
                time_exit="event_expiry",
                time_stop_days=5,
                event_expiry_days=5,
            ),
        )
        PlanRepository.save(session, plan, trade_date=DAY)
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2026, 10, 30, 10, 0),
            plan_id=plan.id,
            external_id="f-buy",
        )
        EventRepository.upsert(
            session,
            event_type="contract",
            title="立案调查",
            source="manual",
            dedupe_key="f-negative",
            ts_code="000001.SZ",
            ann_date=DAY,
            direction="negative",
            verification_status="verified",
        )
        decision = PositionDecisionService(
            session, load_full_config()
        ).evaluate_all(
            DAY,
            points={"000001.SZ": _point(close=10.2, high=10.3, low=10.0)},
            now=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
        )[0]
        assert decision.action == PositionAction.EXIT
        assert decision.reason_code == "event_invalidated"
        assert decision.suggested_shares == 1000


def test_real_account_with_incomplete_valuation_is_rejected():
    from app.core.calendar import TradingCalendar
    from app.planning.errors import PositionRejectedError
    from app.planning.strategy_planner import StrategyPlanner
    from app.planning.strategy_sizing import StrategyPositionSizer
    from app.strategies import c_platform
    from tests.strategy_helpers import make_context
    from tests.test_full_system import _c_bars

    context = make_context(_c_bars())
    candidate = next(
        item for item in c_platform.generate(context) if item.strategy == Strategy.C
    )
    calendar = TradingCalendar(
        [context.as_of + timedelta(days=index) for index in range(10)]
    )
    parts = StrategyPlanner().plan_parts(candidate, calendar, context.config)
    account = AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal("1000000"),
        mode="real",
        risk_config_configured=True,
        valuation_incomplete=True,
    )
    with pytest.raises(PositionRejectedError, match="valuation incomplete"):
        StrategyPositionSizer().size(
            candidate,
            parts.entry,
            parts.stop,
            account,
            context.market,
            context.config,
        )
