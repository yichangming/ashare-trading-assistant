from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from app.backtest.simulator import Simulator
from app.core.calendar import TradingCalendar
from app.core.full_config import load_full_config
from app.db.models import OrderRow
from app.db.repositories import (
    AccountRepository,
    PlanRepository,
    PositionRepository,
)
from app.db.session import init_db, reset_engine, session_scope
from app.domain.enums import (
    EntryType,
    ExecutionSide,
    ExitSession,
    OrderStatus,
    PlanStatus,
    PositionAction,
    SentimentStage,
    Strategy,
    SubMode,
    Template,
    ValuationStatus,
)
from app.domain.models import (
    Candidate,
    EntryPlan,
    ExitRules,
    PositionDecision,
    PositionPlan,
    StopPlan,
    TakeProfitPlan,
    TakeProfitTarget,
    TradePlan,
)
from app.domain.money import money, price
from app.jobs.data_update import DataUpdateService
from app.jobs.health import DataHealthService
from app.jobs.intraday import IntradayMonitor, Quote
from app.portfolio.decisions import PositionDecisionService
from app.portfolio.exit_rules import (
    ExitObservation,
    ExitPolicy,
    evaluate_exit,
)
from app.portfolio.market import PricePoint
from app.portfolio.service import (
    AccountService,
    cancel_order,
    cancel_plan,
    confirm_order_cancel,
    record_order_fill,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 10, 30)  # Friday
MONDAY = date(2026, 11, 2)


def _plan(
    *,
    strategy: Strategy = Strategy.A,
    sub_mode: SubMode = SubMode.A_SHALLOW,
    entry: float = 10.0,
    stop: float = 9.5,
    shares: int = 1000,
    targets: list[tuple[float, float]] | None = None,
    exit_rules: ExitRules | None = None,
    evidence: dict | None = None,
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
        evidence=evidence or {},
        data_time=datetime.combine(DAY, datetime.min.time(), tzinfo=SHANGHAI),
    )
    expiry = DAY + timedelta(days=2)
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
    position_plan = PositionPlan(
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
        is_add_on=False,
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
        position_plan=position_plan,
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
    *,
    open_: float = 10.0,
    high: float = 10.0,
    low: float = 10.0,
    close: float = 10.0,
    data_time: datetime | None = None,
    status: ValuationStatus = ValuationStatus.OK,
) -> PricePoint:
    return PricePoint(
        ts_code="000001.SZ",
        open=open_,
        high=high,
        low=low,
        close=close,
        price=close,
        data_time=data_time or datetime.combine(DAY, datetime.min.time(), tzinfo=SHANGHAI)
        + timedelta(hours=15),
        status=status,
    )


def _weekday_calendar(
    start: date = date(2026, 10, 19),
    end: date = date(2026, 11, 30),
) -> TradingCalendar:
    days = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return TradingCalendar(
        days,
        covered_start=start,
        covered_end=end,
        closed_dates=[
            item
            for item in (
                start + timedelta(days=index)
                for index in range((end - start).days + 1)
            )
            if item not in set(days)
        ],
    )


class _CalendarProvider:
    def __init__(self, calendar: TradingCalendar | None = None):
        self.calendar = calendar or _weekday_calendar()

    def load_trade_cal(self, start: date, end: date) -> pl.DataFrame:
        dates = [
            item
            for item in (
                self.calendar.covered_start + timedelta(days=index)
                for index in range(
                    (self.calendar.covered_end - self.calendar.covered_start).days
                    + 1
                )
            )
            if start <= item <= end
        ]
        return pl.DataFrame(
            {
                "cal_date": dates,
                "is_open": [1 if item in self.calendar.dates else 0 for item in dates],
            }
        )

    def load_bars(self, start: date, end: date) -> pl.DataFrame:
        return pl.DataFrame()


# 4. calendar coverage -------------------------------------------------------
def test_calendar_distinguishes_coverage_from_confirmed_holiday():
    calendar = TradingCalendar.from_trade_cal(
        pl.DataFrame(
            {
                "cal_date": [DAY, DAY + timedelta(days=1)],
                "is_open": [1, 0],
            }
        )
    )
    saturday = DAY + timedelta(days=1)
    monday = DAY + timedelta(days=3)
    assert calendar.covers(DAY)
    assert calendar.is_trading_day(DAY)
    assert calendar.covers(saturday)
    assert not calendar.is_trading_day(saturday)
    assert calendar.is_explicit_holiday(saturday)
    assert not calendar.covers(monday)
    assert not calendar.is_explicit_holiday(monday)


def test_data_update_tries_calendar_refresh_when_coverage_ends_before_monday(tmp_path):
    from app.data.store import ParquetStore

    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    store.write_reference(
        "trade_cal",
        pl.DataFrame({"cal_date": [DAY], "is_open": [1]}),
    )
    store.write_reference(
        "stock_basic",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "list_date": [date(2000, 1, 1)],
                "delist_date": [None],
            }
        ),
    )
    calls: list[date] = []

    class FakeSync:
        def sync_reference(self, start: date, end: date):
            calls.append(end)
            dates = []
            cursor = start
            while cursor <= end:
                if cursor.weekday() < 5:
                    dates.append(cursor)
                cursor += timedelta(days=1)
            store.write_reference(
                "trade_cal",
                pl.DataFrame(
                    {
                        "cal_date": dates,
                        "is_open": [1 for _ in dates],
                    }
                ),
            )
            return {"trade_cal": len(dates)}

        def sync_incremental(self, as_of: date):
            return []

    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        result = DataUpdateService(
            store,
            load_full_config(),
            token=None,
            sync_factory=lambda: FakeSync(),
        ).run(MONDAY, session)
    assert calls, "calendar refresh must be attempted for uncovered Monday"
    assert calls[0] >= MONDAY
    trading_stage = next(
        stage for stage in result.stages if stage.name == "trading_day"
    )
    assert trading_stage.status == "succeeded"
    assert result.status != "skipped"


def test_data_update_marks_unknown_when_refresh_fails_instead_of_skipping(tmp_path):
    from app.data.store import ParquetStore

    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    store.write_reference(
        "trade_cal",
        pl.DataFrame({"cal_date": [DAY], "is_open": [1]}),
    )

    class BrokenSync:
        def sync_reference(self, start: date, end: date):
            raise RuntimeError("network unavailable")

    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        result = DataUpdateService(
            store,
            load_full_config(),
            token=None,
            sync_factory=lambda: BrokenSync(),
        ).run(MONDAY, session)
    trading_stage = next(
        stage for stage in result.stages if stage.name == "trading_day"
    )
    assert trading_stage.status == "unknown"
    assert result.status != "skipped"
    assert any("未覆盖" in issue or "未知" in issue for issue in result.issues)


# 10. anomaly priority -------------------------------------------------------
def test_stop_missing_has_priority_but_missing_quote_alone_is_data_unavailable():
    stop_missing = ExitPolicy(
        strategy=Strategy.A,
        entry_price=10.0,
        initial_stop=None,
        current_stop=None,
        initial_r=0.5,
    )
    observation = ExitObservation(
        close=9.0, high=9.1, low=8.9, data_ok=False, data_issues=("缺少行情",)
    )
    decision = evaluate_exit(
        stop_missing, observation, total_shares=1000, available_shares=1000
    )
    assert decision.reason_code == "stop_missing"

    data_missing = ExitPolicy(
        strategy=Strategy.A,
        entry_price=10.0,
        initial_stop=9.5,
        current_stop=9.5,
        initial_r=0.5,
    )
    decision = evaluate_exit(
        data_missing, observation, total_shares=1000, available_shares=1000
    )
    assert decision.reason_code == "data_unavailable"

    fresh = ExitObservation(close=9.8, high=9.9, low=9.7, data_ok=True)
    decision = evaluate_exit(
        stop_missing, fresh, total_shares=1000, available_shares=1000
    )
    assert decision.reason_code == "stop_missing"
    decision = evaluate_exit(
        data_missing, fresh, total_shares=1000, available_shares=1000
    )
    assert decision.action == PositionAction.HOLD


# 3. pending protection line ------------------------------------------------
def test_daily_stop_is_effective_next_session_and_repeated_runs_are_idempotent():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        service = _service(session)
        plan = _plan(shares=1000, targets=[(20.0, 0.5)])
        PlanRepository.save(session, plan, trade_date=DAY)
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2026, 10, 30, 10, 0),
            plan_id=plan.id,
            external_id="stop-timing",
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        evaluator = PositionDecisionService(session, load_full_config())
        friday_point = _point(
            open_=10.2,
            high=11.6,
            low=10.4,
            close=11.3,
            data_time=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
        )
        first = evaluator.evaluate_all(
            DAY,
            points={"000001.SZ": friday_point},
            now=datetime(2026, 10, 30, 15, 5, tzinfo=SHANGHAI),
        )[0]
        assert first.action in (
            PositionAction.HOLD,
            PositionAction.ADD_CONDITIONAL,
        )
        assert first.current_stop == Decimal("9.5")
        assert first.pending_stop is not None and first.pending_stop >= Decimal("10.5")
        assert position.pending_stop_effective_at is not None
        assert position.pending_stop_effective_at.date() == MONDAY
        pending_before = position.pending_stop

        second = evaluator.evaluate_all(
            DAY,
            points={"000001.SZ": friday_point},
            now=datetime(2026, 10, 30, 15, 6, tzinfo=SHANGHAI),
        )[0]
        assert second.action in (
            PositionAction.HOLD,
            PositionAction.ADD_CONDITIONAL,
        )
        assert second.reason_code in ("hold", "conditional_add")
        assert float(position.current_stop) == 9.5
        assert position.pending_stop == pending_before

        monday_point = _point(
            open_=11.1,
            high=11.3,
            low=10.9,
            close=11.1,
            data_time=datetime(2026, 11, 2, 15, 0, tzinfo=SHANGHAI),
        )
        promoted = evaluator.evaluate_all(
            MONDAY,
            points={"000001.SZ": monday_point},
            now=datetime(2026, 11, 2, 15, 5, tzinfo=SHANGHAI),
        )[0]
        assert promoted.action in (
            PositionAction.HOLD,
            PositionAction.ADD_CONDITIONAL,
        )
        assert float(position.current_stop) >= 10.5
        assert position.pending_stop is None


def test_intraday_tick_does_not_create_a_next_session_stop():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=9.5,
            current_price=10,
            primary_strategy="A",
            initial_r=0.5,
            initial_shares=1000,
            opened_at=datetime(2026, 10, 30, 10, 0),
            highest_price=10,
            last_new_high_on=DAY,
        )
        evaluator = PositionDecisionService(session, load_full_config())
        evaluator.evaluate_all(
            DAY,
            points={
                "000001.SZ": _point(
                    high=12.5,
                    low=12.4,
                    close=12.5,
                    data_time=datetime(2026, 10, 30, 10, 5, tzinfo=SHANGHAI),
                )
            },
            session_type=ExitSession.INTRADAY,
            now=datetime(2026, 10, 30, 10, 5, tzinfo=SHANGHAI),
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert position.pending_stop is None


# 5. no-new-high time exit ---------------------------------------------------
def test_no_new_high_uses_trading_days_since_last_high_and_resets_on_new_high():
    rules = ExitRules(
        trailing="breakeven_1r_2r_ma5_atr",
        time_exit="no_new_high",
        time_stop_days=3,
    )
    policy = ExitPolicy(
        strategy=Strategy.B,
        entry_price=10.0,
        initial_stop=9.5,
        current_stop=9.5,
        initial_r=0.5,
        highest_price=10.8,
        exit_rules=rules,
    )
    old_position = ExitObservation(
        close=10.4,
        high=10.5,
        low=10.3,
        days_held=10,
        days_since_new_high=3,
        last_new_high_on=DAY - timedelta(days=5),
        data_time=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
    )
    decision = evaluate_exit(
        policy, old_position, total_shares=1000, available_shares=1000
    )
    assert decision.action == PositionAction.EXIT
    assert decision.reason_code == "time_stop"

    before_threshold = ExitObservation(
        close=10.4,
        high=10.5,
        low=10.3,
        days_held=10,
        days_since_new_high=2,
        last_new_high_on=DAY - timedelta(days=4),
        data_time=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
    )
    assert (
        evaluate_exit(
            policy, before_threshold, total_shares=1000, available_shares=1000
        ).action
        == PositionAction.HOLD
    )


def test_service_records_last_new_high_date_and_time_exit_resets():
    reset_engine()
    init_db("sqlite://")
    calendar = _weekday_calendar(date(2026, 10, 19), date(2026, 11, 30))
    rules = ExitRules(
        trailing="breakeven_1r_2r_ma5_atr",
        time_exit="no_new_high",
        time_stop_days=3,
    )
    with session_scope() as session:
        service = _service(session)
        plan = _plan(
            strategy=Strategy.B,
            sub_mode=SubMode.B1,
            shares=1000,
            targets=[(20.0, 0.5)],
            exit_rules=rules,
        )
        PlanRepository.save(session, plan, trade_date=DAY)
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2026, 10, 30, 10, 0),
            plan_id=plan.id,
            external_id="high-reset",
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        evaluator = PositionDecisionService(session, load_full_config())
        days = [
            (MONDAY, 10.8),
            (date(2026, 11, 3), 10.7),
            (date(2026, 11, 4), 10.6),
            (date(2026, 11, 5), 10.5),
        ]
        decisions = []
        for day, high in days:
            point = _point(
                open_=high,
                high=high,
                low=high - 0.1,
                close=high - 0.05,
                data_time=datetime.combine(day, datetime.min.time(), tzinfo=SHANGHAI)
                + timedelta(hours=15),
            )
            decisions.append(
                evaluator.evaluate_position(
                    position,
                    as_of=day,
                    point=point,
                    session_type=ExitSession.DAILY,
                    now=datetime.combine(day, datetime.min.time(), tzinfo=SHANGHAI)
                    + timedelta(hours=15, minutes=5),
                    calendar=calendar,
                )
            )
        assert position.last_new_high_on == MONDAY
        assert decisions[0].days_since_new_high == 0
        assert decisions[-1].action == PositionAction.EXIT
        assert decisions[-1].reason_code == "time_stop"


# 7. partial sell order coverage --------------------------------------------
def _position_with_stop(session, *, stop: float = 9.5) -> None:
    AccountRepository.ensure(session, equity=1_000_000, cash=900_000)
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
        initial_r=0.5,
        initial_shares=1000,
        opened_at=datetime(2026, 10, 29, 10, 0),
        highest_price=10,
        last_new_high_on=DAY - timedelta(days=1),
    )


def test_small_pending_sell_order_does_not_hide_remaining_hard_stop_risk():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        _position_with_stop(session)
        session.add(
            OrderRow(
                id="sell-1",
                account_id="default",
                plan_id="plan-1",
                ts_code="000001.SZ",
                side=ExecutionSide.SELL.value,
                price=Decimal("9.4"),
                shares=100,
                filled_shares=0,
                status=OrderStatus.PENDING.value,
            )
        )
        session.flush()
        evaluator = PositionDecisionService(session, load_full_config())
        decision = evaluator.evaluate_all(
            DAY,
            points={
                "000001.SZ": _point(open_=9.5, high=9.6, low=9.2, close=9.3)
            },
            now=datetime(2026, 10, 30, 15, 5, tzinfo=SHANGHAI),
        )[0]
        assert decision.action == PositionAction.EXIT
        assert decision.pending_shares == 100
        assert decision.covered_shares == 100
        assert decision.suggested_shares == 900
        assert "尚未覆盖 900" in decision.explanation

        record_order_fill(session, "sell-1", filled_shares=60)
        partial = evaluator.evaluate_all(
            DAY,
            points={
                "000001.SZ": _point(open_=9.5, high=9.6, low=9.2, close=9.3)
            },
            now=datetime(2026, 10, 30, 15, 6, tzinfo=SHANGHAI),
        )[0]
        assert partial.pending_shares == 40
        assert partial.suggested_shares == 960
        assert partial.pending_sell_orders[0]["remaining_shares"] == 40
        assert partial.pending_sell_orders[0]["filled_shares"] == 60
        assert partial.pending_sell_orders[0]["status"] == "partial"
        assert Decimal(partial.pending_sell_orders[0]["price"]) == Decimal("9.4")


def test_fully_covered_exit_keeps_risk_action_with_zero_uncovered_quantity():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        _position_with_stop(session)
        session.add(
            OrderRow(
                id="sell-all",
                account_id="default",
                plan_id="plan-1",
                ts_code="000001.SZ",
                side=ExecutionSide.SELL.value,
                price=Decimal("9.4"),
                shares=1000,
                filled_shares=0,
                status=OrderStatus.PENDING.value,
            )
        )
        session.flush()
        decision = PositionDecisionService(session, load_full_config()).evaluate_all(
            DAY,
            points={
                "000001.SZ": _point(open_=9.5, high=9.6, low=9.2, close=9.3)
            },
            now=datetime(2026, 10, 30, 15, 5, tzinfo=SHANGHAI),
        )[0]
        assert decision.action == PositionAction.EXIT
        assert decision.suggested_shares == 0
        assert decision.covered_shares == 1000


# 8. plan/order cancellation ------------------------------------------------
def test_plan_cancel_and_order_cancel_keep_exposure_until_confirmed():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        _service(session)
        plan = _plan(shares=500)
        PlanRepository.save(session, plan, trade_date=DAY)
        session.add(
            OrderRow(
                id="buy-order",
                account_id="default",
                plan_id=plan.id,
                ts_code="000001.SZ",
                side=ExecutionSide.BUY.value,
                price=Decimal("10"),
                shares=500,
                filled_shares=0,
                status=OrderStatus.PENDING.value,
            )
        )
        session.flush()
        exposures = PlanRepository.pending_exposures(session)
        assert sum(item.shares for item in exposures) == 500

        result = cancel_plan(session, plan.id)
        assert result["exposure_retained"] is True
        assert sum(item.shares for item in PlanRepository.pending_exposures(session)) == 500

        cancel_order(session, "buy-order", "user requested")
        assert session.get(OrderRow, "buy-order").status == "cancel_requested"
        assert sum(item.shares for item in PlanRepository.pending_exposures(session)) == 500

        record_order_fill(session, "buy-order", filled_shares=100)
        assert sum(item.shares for item in PlanRepository.pending_exposures(session)) == 400

        confirm_order_cancel(session, "buy-order")
        assert sum(item.shares for item in PlanRepository.pending_exposures(session)) == 0


def test_expired_plan_does_not_release_active_order_exposure():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        _service(session)
        plan = _plan(shares=500)
        PlanRepository.save(session, plan, trade_date=DAY)
        session.add(
            OrderRow(
                id="expired-order",
                account_id="default",
                plan_id=plan.id,
                ts_code="000001.SZ",
                side=ExecutionSide.BUY.value,
                price=Decimal("10"),
                shares=500,
                filled_shares=0,
                status=OrderStatus.PENDING.value,
            )
        )
        session.flush()
        PlanRepository.expire_overdue(session, DAY + timedelta(days=5))
        assert PlanRepository.get(session, plan.id).status == PlanStatus.EXPIRED.value
        assert sum(item.shares for item in PlanRepository.pending_exposures(session)) == 500


# 2/6. intraday valuation and freshness -------------------------------------
class _Quotes:
    def __init__(self, values: dict[str, Quote | None]):
        self.values = values

    def get_quote(self, code: str):
        return self.values.get(code)


def test_intraday_revalues_before_b2_and_keeps_missing_quote_valuation():
    reset_engine()
    init_db("sqlite://")
    calendar = _weekday_calendar()
    seen: list[float] = []

    class FakeB2:
        def check(self, day, *, now, session):
            state = AccountRepository.to_state(session)
            seen.append(float(state.equity))

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
            opened_at=datetime(2026, 10, 29, 10, 0),
            highest_price=10,
            last_new_high_on=DAY - timedelta(days=1),
        )
    now = datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI)
    monitor = IntradayMonitor(
        _Quotes(
            {
                "000001.SZ": Quote(
                    11.0, datetime(2026, 11, 2, 9, 59, tzinfo=SHANGHAI), now
                )
            }
        ),
        b2=FakeB2(),
        calendar_provider=_CalendarProvider(calendar),
    )
    result = monitor.check(MONDAY, now=now)
    assert result["checked"] == 1
    with session_scope() as session:
        state = AccountRepository.to_state(session)
        assert float(state.cash) == 90_000
        assert float(state.positions_value) == 11_000
        assert float(state.equity) == 101_000
        assert not state.valuation_incomplete
    assert seen == [101_000]

    missing = IntradayMonitor(
        _Quotes({"000001.SZ": None}),
        calendar_provider=_CalendarProvider(calendar),
    ).check(MONDAY, now=now)
    assert missing["degraded"]
    with session_scope() as session:
        state = AccountRepository.to_state(session)
        assert float(state.positions_value) == 11_000
        assert state.valuation_incomplete
        assert "000001.SZ" in (state.valuation_note or "")


def test_same_day_expired_quote_is_not_usable_and_health_matches():
    reset_engine()
    init_db("sqlite://")
    calendar = _weekday_calendar()
    with session_scope() as session:
        _position_with_stop(session, stop=9)
    now = datetime(2026, 11, 2, 10, 5, tzinfo=SHANGHAI)
    monitor = IntradayMonitor(
        _Quotes(
            {
                "000001.SZ": Quote(
                    9.8, datetime(2026, 11, 2, 9, 55, tzinfo=SHANGHAI), now
                )
            }
        ),
        calendar_provider=_CalendarProvider(calendar),
    )
    result = monitor.check(MONDAY, now=now)
    assert result["positions"][0]["status"] == "stale"
    assert result["positions"][0]["usable"] is False
    assert result["checked"] == 0
    assert result["degraded"] is True
    with session_scope() as session:
        position = PositionRepository.get(session, "default", "000001.SZ")
        assert float(position.current_price) == 10
        health = DataHealthService(_CalendarProvider(calendar)).check(
            MONDAY, session
        )
        assert health.quote_freshness == "stale"
        assert health.ok is False


# 1. page/API decision precedence -------------------------------------------
def test_current_decisions_prefer_newer_intraday_exit_over_daily_hold():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        _position_with_stop(session, stop=9.5)
        position = PositionRepository.get(session, "default", "000001.SZ")
        intraday = PositionDecision(
            ts_code="000001.SZ",
            action=PositionAction.EXIT,
            reason_code="hard_stop",
            explanation="盘中跌破保护线",
            trigger_price=Decimal("9.0"),
            data_time=datetime(2026, 11, 2, 9, 55, tzinfo=SHANGHAI),
            valid_until=datetime(2026, 11, 2, 10, 5, tzinfo=SHANGHAI),
            current_stop=Decimal("9.5"),
            total_shares=1000,
            available_shares=1000,
            suggested_shares=1000,
            decision_source="intraday",
            valuation_status=ValuationStatus.OK,
        )
        position.last_decision_json = intraday.model_dump(mode="json")
        session.flush()
        decisions = PositionDecisionService(
            session, load_full_config()
        ).current_all(
            DAY,
            points={
                "000001.SZ": _point(
                    open_=10.0,
                    high=10.1,
                    low=9.9,
                    close=10.0,
                    data_time=datetime(2026, 10, 30, 15, 0, tzinfo=SHANGHAI),
                )
            },
            now=datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI),
        )
        assert len(decisions) == 1
        assert decisions[0].action == PositionAction.EXIT
        assert decisions[0].decision_source == "intraday"
        assert decisions[0].trigger_price == Decimal("9.0")
        assert decisions[0].is_current is True


def test_expired_intraday_hold_becomes_data_anomaly_not_safe_hold():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        _position_with_stop(session)
        position = PositionRepository.get(session, "default", "000001.SZ")
        stale_hold = PositionDecision(
            ts_code="000001.SZ",
            action=PositionAction.HOLD,
            reason_code="hold",
            explanation="盘中未触发",
            trigger_price=Decimal("10.1"),
            data_time=datetime(2026, 11, 2, 9, 50, tzinfo=SHANGHAI),
            valid_until=datetime(2026, 11, 2, 9, 53, tzinfo=SHANGHAI),
            current_stop=Decimal("9.5"),
            total_shares=1000,
            available_shares=1000,
            decision_source="intraday",
            valuation_status=ValuationStatus.OK,
        )
        position.last_decision_json = stale_hold.model_dump(mode="json")
        session.flush()
        decision = PositionDecisionService(
            session, load_full_config()
        ).current_all(
            DAY,
            now=datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI),
        )[0]
        assert decision.action == PositionAction.DATA_ANOMALY
        assert decision.reason_code == "quote_stale"
        assert decision.is_current is False
        assert decision.stale_reason == "intraday_quote_expired"


# 9. backtest exit inputs ----------------------------------------------------
def _flat_bar(day: date, price_value: float = 10.1) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": day,
                "open": price_value,
                "high": price_value,
                "low": price_value,
                "close": price_value,
                "vol": 1e8,
                "amount": 1e10,
            }
        ]
    )


def _run_sim_days(sim: Simulator, start: date, count: int) -> list:
    trades = []
    for index in range(count):
        day = start + timedelta(days=index)
        trades.extend(sim.process_day(day, _flat_bar(day)))
    return trades


def test_backtest_f_event_expiry_uses_known_event_date_and_data_time():
    calendar = _weekday_calendar()
    plan = _plan(
        strategy=Strategy.F,
        sub_mode=SubMode.F_EARNINGS,
        targets=[(20.0, 0.5)],
        exit_rules=ExitRules(
            trailing="breakeven_1r_2r_atr",
            time_exit="event_expiry",
            time_stop_days=2,
            event_expiry_days=2,
        ),
        evidence={
            "event_id": "event-1",
            "event_ann_date": "2026-10-29",
            "direction": "positive",
            "verification_status": "verified",
        },
    )
    sim = Simulator(
        load_full_config(),
        1_000_000,
        1_000_000,
        1_000_000,
        calendar=calendar,
        events_by_code={
            "000001.SZ": [
                {
                    "event_id": "event-1",
                    "ann_date": "2026-10-29",
                    "direction": "positive",
                    "verification_status": "verified",
                }
            ]
        },
    )
    sim.add_plans([plan], DAY - timedelta(days=1))
    sim.process_day(DAY, _flat_bar(DAY, 10.0))
    assert "000001.SZ" in sim.positions
    trades = _run_sim_days(sim, MONDAY, 3)
    assert trades
    assert trades[0].exit_reason == "event"


def test_backtest_f_retraction_invalidates_and_missing_events_is_labeled():
    calendar = _weekday_calendar()
    plan = _plan(
        strategy=Strategy.F,
        sub_mode=SubMode.F_EARNINGS,
        targets=[(20.0, 0.5)],
        exit_rules=ExitRules(
            trailing="breakeven_1r_2r_atr",
            time_exit="event_expiry",
            time_stop_days=5,
            event_expiry_days=5,
        ),
        evidence={
            "event_id": "event-1",
            "event_ann_date": "2026-10-29",
            "direction": "positive",
            "verification_status": "verified",
        },
    )
    sim = Simulator(
        load_full_config(),
        1_000_000,
        1_000_000,
        1_000_000,
        calendar=calendar,
        events_by_code={
            "000001.SZ": [
                {
                    "event_id": "event-1",
                    "ann_date": "2026-10-29",
                    "direction": "positive",
                    "verification_status": "verified",
                },
                {
                    "event_id": "event-1",
                    "ann_date": "2026-10-29",
                    "first_seen_at": "2026-11-02T09:00:00+08:00",
                    "direction": "positive",
                    "verification_status": "verified",
                    "retracted": True,
                },
            ]
        },
    )
    sim.add_plans([plan], DAY - timedelta(days=1))
    sim.process_day(DAY, _flat_bar(DAY, 10.0))
    trades = sim.process_day(MONDAY, _flat_bar(MONDAY))
    assert trades and trades[0].exit_reason == "event"

    missing = Simulator(
        load_full_config(),
        1_000_000,
        1_000_000,
        1_000_000,
        calendar=calendar,
        events_by_code=None,
    )
    missing.add_plans([plan], DAY - timedelta(days=1))
    missing.process_day(DAY, _flat_bar(DAY, 10.0))
    assert any("历史事件覆盖缺失" in note for note in missing.missing_data)


def test_live_and_backtest_time_exit_reach_same_decision():
    reset_engine()
    init_db("sqlite://")
    calendar = _weekday_calendar()
    rules = ExitRules(
        trailing="breakeven_1r_2r_ma5_atr",
        time_exit="no_new_high",
        time_stop_days=3,
    )
    with session_scope() as session:
        service = _service(session)
        plan = _plan(
            strategy=Strategy.B,
            sub_mode=SubMode.B1,
            shares=1000,
            targets=[(20.0, 0.5)],
            exit_rules=rules,
        )
        PlanRepository.save(session, plan, trade_date=DAY)
        service.record_execution(
            ts_code="000001.SZ",
            side=ExecutionSide.BUY,
            price=Decimal("10"),
            shares=1000,
            executed_at=datetime(2026, 10, 30, 10, 0),
            plan_id=plan.id,
            external_id="live-bt-compare",
        )
        position = PositionRepository.get(session, "default", "000001.SZ")
        evaluator = PositionDecisionService(session, load_full_config())
        highs = [
            (MONDAY, 10.4),
            (date(2026, 11, 3), 10.3),
            (date(2026, 11, 4), 10.2),
            (date(2026, 11, 5), 10.1),
        ]
        live_decisions = []
        for day, high in highs:
            live_decisions.append(
                evaluator.evaluate_position(
                    position,
                    as_of=day,
                    point=_point(
                        open_=10.0,
                        high=high,
                        low=9.9,
                        close=high - 0.05,
                        data_time=datetime.combine(
                            day, datetime.min.time(), tzinfo=SHANGHAI
                        )
                        + timedelta(hours=15),
                    ),
                    session_type=ExitSession.DAILY,
                    now=datetime.combine(
                        day, datetime.min.time(), tzinfo=SHANGHAI
                    )
                    + timedelta(hours=15, minutes=5),
                    calendar=calendar,
                )
            )
    sim = Simulator(
        load_full_config(),
        1_000_000,
        1_000_000,
        1_000_000,
        calendar=calendar,
    )
    sim.add_plans([plan], DAY - timedelta(days=1))
    sim.process_day(DAY, _flat_bar(DAY, 10.0))
    sim_trades = []
    for day, high in highs:
        sim_trades.extend(
            sim.process_day(day, _flat_bar(day, high))
        )
    assert live_decisions[-1].action == PositionAction.EXIT
    assert live_decisions[-1].reason_code == "time_stop"
    assert sim_trades
    assert sim_trades[0].exit_reason == "time"


def test_backtest_ma5_range_and_atr_inputs_are_updated():
    calendar = _weekday_calendar()
    plan_b = _plan(
        strategy=Strategy.B,
        sub_mode=SubMode.B1,
        entry=10.0,
        stop=9.0,
        targets=[(20.0, 0.5)],
        exit_rules=ExitRules(
            trailing="breakeven_1r_2r_ma5_atr",
            time_exit="no_new_high",
            time_stop_days=10,
        ),
    )
    sim = Simulator(
        load_full_config(),
        1_000_000,
        1_000_000,
        1_000_000,
        calendar=calendar,
    )
    sim.add_plans([plan_b], DAY - timedelta(days=1))
    sim.process_day(DAY, _flat_bar(DAY, 10.0))
    prices = [11.2, 11.8, 12.4, 12.8, 13.2]
    for index, value in enumerate(prices):
        day = MONDAY + timedelta(days=index)
        sim.process_day(day, _flat_bar(day, value))
    position = sim.positions["000001.SZ"]
    assert position.ma5 is not None and position.ma5 > 10.0
    assert position.pending_stop is not None
    assert position.pending_stop >= position.ma5 - 1e-6
    pending_before = position.pending_stop
    next_day = MONDAY + timedelta(days=len(prices))
    sim.process_day(next_day, _flat_bar(next_day, 12.6))
    assert position.current_stop == pytest.approx(pending_before)

    plan_e = _plan(
        strategy=Strategy.E,
        sub_mode=SubMode.E_RANGE,
        entry=10.0,
        stop=9.0,
        targets=[(20.0, 0.5)],
        exit_rules=ExitRules(
            trailing="breakeven_1r_2r_range_atr",
            time_exit="below_entry",
            time_stop_days=10,
        ),
        evidence={"range_upper": 12.4, "range_lower": 9.6},
    )
    sim_e = Simulator(
        load_full_config(),
        1_000_000,
        1_000_000,
        1_000_000,
        calendar=calendar,
    )
    sim_e.add_plans([plan_e], DAY - timedelta(days=1))
    sim_e.process_day(DAY, _flat_bar(DAY, 10.0))
    for index, value in enumerate([11.0, 12.0, 12.8]):
        day = MONDAY + timedelta(days=index)
        sim_e.process_day(day, _flat_bar(day, value))
    position_e = sim_e.positions["000001.SZ"]
    assert position_e.range_midpoint == pytest.approx(11.0)
    assert position_e.pending_stop is not None
    assert position_e.pending_stop >= 11.0
