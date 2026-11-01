from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.core.calendar import TradingCalendar
from app.core.full_config import load_full_config
from app.db.repositories import (
    AccountRepository,
    EventRepository,
    JobRepository,
    NotificationRepository,
    SettingRepository,
)
from app.db.session import init_db, reset_engine, session_scope
from app.domain.enums import AccountMode, PlanStatus, Strategy
from app.domain.models import AccountState, PositionState, TradePlan
from app.planning.errors import PositionRejectedError
from app.planning.strategy_planner import StrategyPlanner
from app.planning.strategy_sizing import StrategyPositionSizer
from app.risk.strategy_governor import StrategyRiskGovernor
from app.strategies import c_platform, f_event
from tests.strategy_helpers import make_bars, make_context


def _c_bars():
    closes = [10.2, 10.3, 10.1, 10.4, 10.2] * 5
    highs = [10.5 if index % 5 == 0 else 10.35 for index in range(25)]
    lows = [10.0 if index % 5 == 2 else 10.15 for index in range(25)]
    opens = [10.2] * 25
    volumes = [20_000_000.0] * 25
    return make_bars(
        closes + [10.8],
        opens=opens + [10.7],
        highs=highs + [10.85],
        lows=lows + [10.7],
        volumes=volumes + [40_000_000.0],
    )


def _account(mode: AccountMode = AccountMode.PAPER, configured: bool = True):
    return AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal("1000000"),
        mode=mode,
        risk_config_configured=configured,
    )


def test_c1_candidate_becomes_plan_and_is_approved():
    bars = _c_bars()
    context = make_context(bars)
    candidate = next(
        item for item in c_platform.generate(context) if item.strategy == Strategy.C
    )
    calendar = TradingCalendar(
        [context.as_of + timedelta(days=index) for index in range(10)]
    )
    parts = StrategyPlanner().plan_parts(candidate, calendar, context.config)
    position = StrategyPositionSizer().size(
        candidate,
        parts.entry,
        parts.stop,
        _account(),
        context.market,
        context.config,
    )
    plan = TradePlan(
        candidate=candidate,
        entry_plan=parts.entry,
        stop_plan=parts.stop,
        position_plan=position,
        take_profit_plan=parts.take_profit,
        expires_on=parts.entry.expires_on,
        strategy_version=context.config.version,
        data_version=context.data_version,
        code_version="test",
        strategy=candidate.strategy,
        sub_mode=candidate.sub_mode,
        confirmation_price=parts.confirmation_price,
        max_acceptable_price=parts.max_acceptable_price,
        execution_low=parts.execution_low,
        execution_high=parts.execution_high,
        data_time=candidate.data_time,
        signal_time=candidate.data_time,
    )
    result = StrategyRiskGovernor().apply(
        [plan], _account(), context.market, context.config
    )
    assert result[0].status == PlanStatus.PENDING_EXECUTION
    assert result[0].risk_decision.outcome.value == "approve"
    assert result[0].position_plan.shares > 0


def test_real_account_without_risk_configuration_is_rejected():
    bars = _c_bars()
    context = make_context(bars)
    candidate = next(
        item for item in c_platform.generate(context) if item.strategy == Strategy.C
    )
    calendar = TradingCalendar(
        [context.as_of + timedelta(days=index) for index in range(10)]
    )
    parts = StrategyPlanner().plan_parts(candidate, calendar, context.config)
    with pytest.raises(PositionRejectedError, match="risk configuration incomplete"):
        StrategyPositionSizer().size(
            candidate,
            parts.entry,
            parts.stop,
            _account(AccountMode.REAL, configured=False),
            context.market,
            context.config,
        )


def test_same_stock_second_strategy_is_rejected_by_governor():
    bars = _c_bars()
    context = make_context(
        bars,
        events={
            "000001.SZ": [
                {
                    "event_id": "evt-1",
                    "event_type": "contract",
                    "title": "重大合同",
                    "source": "manual",
                    "ann_date": bars["trade_date"][-1],
                    "verification_status": "verified",
                    "direction": "positive",
                    "materiality": 0.8,
                    "retracted": False,
                }
            ]
        },
    )
    config = load_full_config().model_copy(deep=True)
    config.strategies.f.price_mode = "c1"
    context.config = config
    c_candidate = next(
        item for item in c_platform.generate(context) if item.strategy == Strategy.C
    )
    f_candidate = next(
        item for item in f_event.generate(context) if item.strategy == Strategy.F
    )
    calendar = TradingCalendar(
        [context.as_of + timedelta(days=index) for index in range(10)]
    )
    plans = []
    for candidate in (c_candidate, f_candidate):
        parts = StrategyPlanner().plan_parts(candidate, calendar, config)
        position = StrategyPositionSizer().size(
            candidate,
            parts.entry,
            parts.stop,
            _account(),
            context.market,
            config,
        )
        plans.append(
            TradePlan(
                candidate=candidate,
                entry_plan=parts.entry,
                stop_plan=parts.stop,
                position_plan=position,
                take_profit_plan=parts.take_profit,
                expires_on=parts.entry.expires_on,
                strategy_version=config.version,
                data_version=context.data_version,
                code_version="test",
                strategy=candidate.strategy,
                sub_mode=candidate.sub_mode,
                confirmation_price=parts.confirmation_price,
                max_acceptable_price=parts.max_acceptable_price,
                execution_low=parts.execution_low,
                execution_high=parts.execution_high,
                data_time=candidate.data_time,
                signal_time=candidate.data_time,
            )
        )
    result = StrategyRiskGovernor().apply(
        plans, _account(), context.market, config
    )
    assert sum(1 for plan in result if plan.status == PlanStatus.PENDING_EXECUTION) == 1
    assert any("same stock already planned" in " ".join(
        plan.risk_decision.reasons
    ) for plan in result if plan.status == PlanStatus.REJECTED)


def test_repository_roundtrip_and_idempotency():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        account = AccountRepository.ensure(
            session, equity=1_000_000, cash=1_000_000
        )
        event, created = EventRepository.upsert(
            session,
            event_type="contract",
            title="test",
            source="manual",
            dedupe_key="evt-1",
            ts_code="000001.SZ",
            verification_status="verified",
            direction="positive",
        )
        assert created
        event2, created2 = EventRepository.upsert(
            session,
            event_type="contract",
            title="test revised",
            source="manual",
            dedupe_key="evt-1",
            ts_code="000001.SZ",
            verification_status="verified",
            direction="positive",
        )
        assert not created2
        assert event2.revision == 1
        notification, created3 = NotificationRepository.add(
            session,
            level="risk",
            code="risk",
            title="risk",
            message="message",
            dedupe_key="n-1",
        )
        assert created3
        _, created4 = NotificationRepository.add(
            session,
            level="risk",
            code="risk",
            title="risk",
            message="message",
            dedupe_key="n-1",
        )
        assert not created4
        job = JobRepository.start(
            session,
            job_name="daily",
            idempotency_key="2024-01-02:full",
            trade_date=date(2024, 1, 2),
        )
        JobRepository.finish(session, job.id, status="succeeded")
        SettingRepository.set(session, "risk_profile", {"equity": 1_000_000})
        assert SettingRepository.get(session, "risk_profile") == {
            "equity": 1_000_000
        }
        assert account.id == "default"
        assert notification.id


def _account_with_c_position(
    *,
    add_on_count: int = 0,
    strategy: Strategy = Strategy.C,
    risk_configured: bool = True,
):
    return AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal("1000000"),
        positions=[
            PositionState(
                ts_code="000001.SZ",
                sector="测试",
                shares=1000,
                available_shares=1000,
                cost_price=Decimal("10"),
                current_stop=Decimal("9.5"),
                current_price=Decimal("11"),
                strategy=strategy,
                initial_shares=1000,
                add_on_count=add_on_count,
            )
        ],
        risk_config_configured=risk_configured,
    )


def test_add_on_is_capped_and_survives_position_limits():
    bars = _c_bars()
    context = make_context(bars)
    candidate = next(
        item for item in c_platform.generate(context) if item.strategy == Strategy.C
    )
    calendar = TradingCalendar(
        [context.as_of + timedelta(days=index) for index in range(10)]
    )
    parts = StrategyPlanner().plan_parts(candidate, calendar, context.config)
    config = context.config.model_copy(deep=True)
    config.risk.max_positions = 1  # an add-on does not open a new position
    account = _account_with_c_position()
    position = StrategyPositionSizer().size(
        candidate, parts.entry, parts.stop, account, context.market, config
    )
    assert position.shares <= 500
    assert "add_on_rules" in position.constraints_checked

    plan = TradePlan(
        candidate=candidate,
        entry_plan=parts.entry,
        stop_plan=parts.stop,
        position_plan=position,
        take_profit_plan=parts.take_profit,
        expires_on=parts.entry.expires_on,
        strategy_version=config.version,
        data_version=context.data_version,
        code_version="test",
        strategy=candidate.strategy,
        sub_mode=candidate.sub_mode,
        confirmation_price=parts.confirmation_price,
        max_acceptable_price=parts.max_acceptable_price,
        execution_low=parts.execution_low,
        execution_high=parts.execution_high,
        data_time=candidate.data_time,
        signal_time=candidate.data_time,
    )
    result = StrategyRiskGovernor().apply(
        [plan], account, context.market, config
    )
    assert result[0].status == PlanStatus.PENDING_EXECUTION


def test_add_on_rejects_max_count_and_disallowed_strategy():
    bars = _c_bars()
    context = make_context(bars)
    candidate = next(
        item for item in c_platform.generate(context) if item.strategy == Strategy.C
    )
    calendar = TradingCalendar(
        [context.as_of + timedelta(days=index) for index in range(10)]
    )
    parts = StrategyPlanner().plan_parts(candidate, calendar, context.config)
    with pytest.raises(PositionRejectedError, match="maximum add-ons"):
        StrategyPositionSizer().size(
            candidate,
            parts.entry,
            parts.stop,
            _account_with_c_position(add_on_count=2),
            context.market,
            context.config,
        )
    config = context.config.model_copy(deep=True)
    config.risk.add_on_strategies = [Strategy.A]
    with pytest.raises(PositionRejectedError, match="add-on not allowed"):
        StrategyPositionSizer().size(
            candidate,
            parts.entry,
            parts.stop,
            _account_with_c_position(),
            context.market,
            config,
        )


def test_stale_real_account_blocks_new_plans():
    bars = _c_bars()
    context = make_context(bars)
    candidate = next(
        item for item in c_platform.generate(context) if item.strategy == Strategy.C
    )
    calendar = TradingCalendar(
        [context.as_of + timedelta(days=index) for index in range(10)]
    )
    parts = StrategyPlanner().plan_parts(candidate, calendar, context.config)
    account = _account(AccountMode.REAL, configured=True).model_copy(
        update={"stale": True}
    )
    with pytest.raises(PositionRejectedError, match="stale"):
        StrategyPositionSizer().size(
            candidate,
            parts.entry,
            parts.stop,
            account,
            context.market,
            context.config,
        )


def test_plan_state_machine_sequence_and_invalid_transition():
    from app.planning.state_machine import (
        InvalidTransition,
        validate_transition,
    )

    sequence = [
        PlanStatus.OBSERVING,
        PlanStatus.PENDING_TRIGGER,
        PlanStatus.CONFIRMED,
        PlanStatus.PENDING_EXECUTION,
        PlanStatus.PARTIAL_FILL,
        PlanStatus.OPEN,
        PlanStatus.CLOSED,
    ]
    for current, target in zip(sequence, sequence[1:], strict=False):
        validate_transition(current, target)
    with pytest.raises(InvalidTransition):
        validate_transition(PlanStatus.CLOSED, PlanStatus.OPEN)
    with pytest.raises(InvalidTransition):
        validate_transition(PlanStatus.REJECTED, PlanStatus.PENDING_EXECUTION)
