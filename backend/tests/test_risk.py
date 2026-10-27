from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.core.config import StrategyConfig, load_strategy_config
from app.domain.enums import (
    EntryType,
    PlanStatus,
    RiskOutcome,
    SentimentStage,
    Template,
    TrendRegime,
)
from app.domain.models import (
    AccountState,
    Candidate,
    EntryPlan,
    MarketRegime,
    PendingExposure,
    PositionPlan,
    PositionState,
    RegimeIndicators,
    StopPlan,
    TakeProfitPlan,
    TakeProfitTarget,
    TemplatePolicy,
    TradePlan,
)
from app.domain.money import money, price
from app.planning.errors import PositionRejectedError
from app.planning.sizing import PositionSizer


def _regime(
    a: TemplatePolicy | None = None,
    b: TemplatePolicy | None = None,
    sentiment_stage: SentimentStage = SentimentStage.FERMENT,
) -> MarketRegime:
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime=TrendRegime.ATTACK,
        sentiment_stage=sentiment_stage,
        risk_factor=1.0,
        template_availability={
            Template.A: a
            if a is not None
            else TemplatePolicy(enabled=True, mode="normal", max_positions=3),
            Template.B: b
            if b is not None
            else TemplatePolicy(enabled=True, mode="normal", max_positions=2),
        },
        indicators=RegimeIndicators(
            limit_up_count=80,
            limit_down_count=0,
            max_board_height=5,
            promotion_rate=0.5,
            blow_rate=0.1,
            red_rate=0.6,
            limit_up_premium=0.02,
            breadth=0.6,
            index_close=4000.0,
            index_ma20=3900.0,
            index_ma60=3800.0,
        ),
        data_version="dv_test",
    )


def _account(
    positions: list[PositionState] | None = None,
    peak: float = 1_000_000,
    *,
    equity: float = 1_000_000,
    cash: float | None = None,
    pending: list[PendingExposure] | None = None,
    trading_halted: bool = False,
    drawdown_half_latched: bool = False,
) -> AccountState:
    equity_value = Decimal(str(equity))
    return AccountState(
        equity=equity_value,
        cash=Decimal(str(cash)) if cash is not None else equity_value,
        peak_equity=Decimal(str(peak)),
        positions=positions or [],
        pending_plans=pending or [],
        trading_halted=trading_halted,
        drawdown_half_latched=drawdown_half_latched,
    )


def _candidate(
    code: str = "000001.SZ",
    sector: str = "银行",
    template: Template = Template.A,
    score: float = 90.0,
    data_version: str = "dv_test",
) -> Candidate:
    return Candidate(
        ts_code=code,
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector=sector,
        leader_rank=1,
        leader_score=90.0,
        score=score,
        reasons=["test"],
        features={"close": 10.0},
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version=data_version,
    )


def _entry(entry_price: float = 10.0) -> EntryPlan:
    return EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=price(entry_price),
        entry_zone_low=price(entry_price * 0.98),
        entry_zone_high=price(entry_price * 1.02),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=[],
        expires_on=date(2024, 1, 5),
    )


def _stop(entry_price: float = 10.0, stop_price: float = 9.0) -> StopPlan:
    return StopPlan(
        initial_stop=price(stop_price),
        stop_distance_pct=(entry_price - stop_price) / entry_price,
        atr14=0.5,
        structure_ref="test",
        time_stop_days=10,
        trailing_rules=[],
        r_value=price(entry_price - stop_price),
    )


def _plan(
    code: str,
    sector: str,
    template: Template = Template.A,
    score: float = 90.0,
    shares: int = 10_000,
    data_version: str = "dv_test",
    entry_price: float = 10.0,
    stop_price: float = 9.0,
) -> TradePlan:
    candidate = _candidate(
        code=code,
        sector=sector,
        template=template,
        score=score,
        data_version=data_version,
    )
    entry = _entry(entry_price)
    stop = _stop(entry_price, stop_price)
    position = PositionPlan(
        template=template,
        risk_pct=0.025,
        risk_amount=money(25_000),
        shares=shares,
        position_value=money(entry_price * shares),
        position_pct=entry_price * shares / 1_000_000,
        heat_before=0.0,
        heat_after=shares / 1_000_000,
        constraints_checked=[],
        adjustments=[],
    )
    take_profit = TakeProfitPlan(
        targets=[
            TakeProfitTarget(r_multiple=2.0, price=price(12.0), fraction=0.5)
        ],
        trailing_rule="test",
        time_exit_rule="test",
    )
    return TradePlan(
        candidate=candidate,
        entry_plan=entry,
        stop_plan=stop,
        position_plan=position,
        take_profit_plan=take_profit,
        expires_on=date(2024, 1, 5),
        strategy_version="sv_test",
        data_version=data_version,
        code_version="test",
    )


def _position(
    ts_code: str = "000009.SZ",
    sector: str = "地产",
    shares: int = 50_000,
    current_stop: float = 9.6,
    current_price: float = 11.0,
    template: Template = Template.A,
) -> PositionState:
    return PositionState(
        ts_code=ts_code,
        sector=sector,
        shares=shares,
        available_shares=shares,
        cost_price=Decimal("10.0"),
        current_stop=Decimal(str(current_stop)),
        current_price=Decimal(str(current_price)),
        template=template,
    )


def _pending(
    ts_code: str = "000008.SZ",
    sector: str = "地产",
    entry_price: float = 10.0,
    stop_price: float = 9.0,
    shares: int = 100,
    template: Template = Template.A,
) -> PendingExposure:
    return PendingExposure(
        ts_code=ts_code,
        sector=sector,
        entry_price=price(entry_price),
        stop_price=price(stop_price),
        shares=shares,
        template=template,
    )


def _size(
    account: AccountState,
    *,
    regime: MarketRegime | None = None,
    config: StrategyConfig | None = None,
    template: Template = Template.A,
    entry_price: float = 10.0,
    stop_price: float = 9.0,
) -> PositionPlan:
    candidate = _candidate(template=template)
    return PositionSizer().size(
        candidate,
        _entry(entry_price),
        _stop(entry_price, stop_price),
        account,
        regime if regime is not None else _regime(),
        config if config is not None else load_strategy_config(),
    )


def _config_with_risk(**updates: object) -> StrategyConfig:
    config = load_strategy_config()
    return config.model_copy(
        update={"risk": config.risk.model_copy(update=updates)}
    )


def _apply(
    plans: list[TradePlan],
    account: AccountState | None = None,
    regime: MarketRegime | None = None,
    config: StrategyConfig | None = None,
    correlations: dict[str, dict[str, float]] | None = None,
    correlation_unavailable: set[str] | None = None,
) -> list[TradePlan]:
    from app.risk.governor import RiskGovernor

    return RiskGovernor().apply(
        plans,
        account if account is not None else _account(),
        regime if regime is not None else _regime(),
        config if config is not None else load_strategy_config(),
        correlations=correlations,
        correlation_unavailable=correlation_unavailable,
    )


def _reasons(plan: TradePlan) -> list[str]:
    assert plan.risk_decision is not None
    return plan.risk_decision.reasons


# --- mandated baseline cases -------------------------------------------------


def test_capacity_and_priority():
    from app.risk.governor import RiskGovernor

    config = load_strategy_config()
    plans = [
        _plan(f"00000{i}.SZ", f"行业{i}", score=90 - i)
        for i in range(4)
    ]
    result = RiskGovernor().apply(plans, _account(), _regime(), config)
    assert sum(1 for plan in result if plan.status == PlanStatus.PENDING_ENTRY) == 3
    assert sum(1 for plan in result if plan.status == PlanStatus.REJECTED) == 1
    approved = [
        plan.candidate.ts_code
        for plan in result
        if plan.status == PlanStatus.PENDING_ENTRY
    ]
    assert approved == ["000000.SZ", "000001.SZ", "000002.SZ"]


def test_b_max_and_disabled_template():
    from app.risk.governor import RiskGovernor

    config = load_strategy_config()
    b_plans = [
        _plan(f"00010{i}.SZ", f"行业{i}", template=Template.B)
        for i in range(3)
    ]
    result = RiskGovernor().apply(b_plans, _account(), _regime(), config)
    assert sum(1 for plan in result if plan.status == PlanStatus.PENDING_ENTRY) == 2

    regime = _regime().model_copy(
        update={
            "template_availability": {
                Template.A: TemplatePolicy(
                    enabled=False, mode="observe", max_positions=0
                ),
                Template.B: TemplatePolicy(
                    enabled=False, mode="observe", max_positions=0
                ),
            }
        }
    )
    result = RiskGovernor().apply(b_plans[:1], _account(), regime, config)
    assert result[0].status == PlanStatus.REJECTED
    assert result[0].risk_decision.outcome == RiskOutcome.REJECT


def test_heat_reduces_and_version_mismatch_rejects():
    from app.risk.governor import RiskGovernor

    config = load_strategy_config()
    position = PositionState(
        ts_code="000009.SZ",
        sector="地产",
        shares=50000,
        available_shares=50000,
        cost_price=Decimal("10.0"),
        current_stop=Decimal("9.6"),
        current_price=Decimal("11.0"),
        template=Template.A,
    )
    plan = _plan("000001.SZ", "银行", shares=10_000)
    result = RiskGovernor().apply([plan], _account(positions=[position]), _regime(), config)
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[0].risk_decision.outcome == RiskOutcome.REDUCE
    assert result[0].position_plan.shares == 5000
    assert _reasons(result[0]) == ["heat cap reduced size"]
    # The input plan is not mutated.
    assert plan.status == PlanStatus.DRAFT
    assert plan.position_plan.shares == 10_000

    bad = _plan("000002.SZ", "银行", data_version="old")
    result = RiskGovernor().apply([bad], _account(), _regime(), config)
    assert result[0].status == PlanStatus.REJECTED


def test_exposure_cap_reduces_total_value():
    from app.risk.governor import RiskGovernor

    config = load_strategy_config()
    plans = [
        _plan(f"00020{i}.SZ", f"行业{i}", shares=40_000)
        for i in range(3)
    ]
    result = RiskGovernor().apply(plans, _account(), _regime(), config)
    approved = [
        plan
        for plan in result
        if plan.status == PlanStatus.PENDING_ENTRY
    ]
    total = sum(
        float(plan.position_plan.position_value) for plan in approved
    )
    assert total <= 900_000


# --- account drawdown latches ------------------------------------------------


def test_half_latch_triggers_at_exact_ten_percent():
    account = _account(equity=900_000, peak=1_000_000)
    # The float property exhibits the usual representation artifact; the
    # Decimal-based latch must still trigger at exactly 10%.
    assert account.drawdown_pct < 0.10
    latched = account.with_risk_latches(stop_at=0.15, half_at=0.10)
    assert latched.drawdown_half_latched is True
    assert latched.trading_halted is False


def test_stop_latch_triggers_at_exact_fifteen_percent():
    account = _account(equity=850_000, peak=1_000_000)
    latched = account.with_risk_latches(stop_at=0.15, half_at=0.10)
    assert latched.drawdown_half_latched is True
    assert latched.trading_halted is True


def test_half_latch_persists_between_recovery_and_half_at():
    account = _account(equity=900_000, peak=1_000_000).with_risk_latches(
        stop_at=0.15, half_at=0.10
    )
    assert account.drawdown_half_latched is True

    # 8% drawdown: above the 5% recovery line, so the latch persists.
    recovered = account.model_copy(
        update={
            "equity": Decimal("920000"),
            "cash": Decimal("920000"),
        }
    )
    still_latched = recovered.with_risk_latches(stop_at=0.15, half_at=0.10)
    assert still_latched.drawdown_half_latched is True


def test_half_latch_clears_only_below_five_percent():
    account = _account(equity=900_000, peak=1_000_000).with_risk_latches(
        stop_at=0.15, half_at=0.10
    )
    # Exactly 5% is not below the recovery line.
    at_five = account.model_copy(
        update={
            "equity": Decimal("950000"),
            "cash": Decimal("950000"),
        }
    )
    assert at_five.with_risk_latches(
        stop_at=0.15, half_at=0.10
    ).drawdown_half_latched is True

    below_five = account.model_copy(
        update={
            "equity": Decimal("960000"),
            "cash": Decimal("960000"),
        }
    )
    cleared = below_five.with_risk_latches(stop_at=0.15, half_at=0.10)
    assert cleared.drawdown_half_latched is False
    assert cleared.trading_halted is False


def test_trading_halt_never_auto_clears():
    halted = _account(equity=850_000, peak=1_000_000).with_risk_latches(
        stop_at=0.15, half_at=0.10
    )
    assert halted.trading_halted is True

    recovered = halted.model_copy(
        update={
            "equity": Decimal("960000"),
            "cash": Decimal("960000"),
        }
    )
    still_halted = recovered.with_risk_latches(stop_at=0.15, half_at=0.10)
    assert still_halted.trading_halted is True
    assert still_halted.drawdown_half_latched is False

    # Manual restart is an explicit reset of the latch.
    restarted = still_halted.model_copy(update={"trading_halted": False})
    assert restarted.with_risk_latches(
        stop_at=0.15, half_at=0.10
    ).trading_halted is False


def test_sizer_halves_risk_when_half_latched_at_low_drawdown():
    account = _account(
        equity=920_000, peak=1_000_000, drawdown_half_latched=True
    )
    plan = _size(account)
    assert plan.risk_amount == money(11_500)
    assert plan.shares == 11_100

    unlatched = _account(equity=920_000, peak=1_000_000)
    assert _size(unlatched).risk_amount == money(23_000)


def test_sizer_restores_full_risk_after_latch_clears():
    account = _account(
        equity=900_000, peak=1_000_000, drawdown_half_latched=True
    )
    recovered = account.model_copy(
        update={
            "equity": Decimal("960000"),
            "cash": Decimal("960000"),
        }
    ).with_risk_latches(stop_at=0.15, half_at=0.10)
    assert recovered.drawdown_half_latched is False
    plan = _size(recovered)
    assert plan.risk_amount == money(24_000)
    assert plan.shares == 23_300


def test_sizer_rejects_when_trading_halted():
    account = _account(trading_halted=True)
    with pytest.raises(PositionRejectedError, match="halted|circuit breaker"):
        _size(account)


def test_sizer_rejects_when_halt_never_clears():
    halted = _account(equity=850_000, peak=1_000_000).with_risk_latches(
        stop_at=0.15, half_at=0.10
    )
    recovered = halted.model_copy(
        update={
            "equity": Decimal("960000"),
            "cash": Decimal("960000"),
        }
    ).with_risk_latches(stop_at=0.15, half_at=0.10)
    with pytest.raises(PositionRejectedError, match="halted|circuit breaker"):
        _size(recovered)


# --- batch template policy and capacity -------------------------------------


def test_a_policy_cap_applies_across_batch():
    config = _config_with_risk(max_positions=5)
    regime = _regime(
        a=TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    )
    plans = [
        _plan(f"00000{i}.SZ", f"行业{i}", score=90 - i)
        for i in range(4)
    ]
    result = _apply(plans, regime=regime, config=config)
    approved = [
        plan.candidate.ts_code
        for plan in result
        if plan.status == PlanStatus.PENDING_ENTRY
    ]
    assert approved == ["000000.SZ", "000001.SZ", "000002.SZ"]
    rejected = [
        plan for plan in result if plan.status == PlanStatus.REJECTED
    ]
    assert len(rejected) == 1
    assert "max A policy" in _reasons(rejected[0])[0]


def test_b_policy_cap_below_config_max_applies_across_batch():
    config = load_strategy_config()
    assert config.risk.max_b_positions == 2
    regime = _regime(
        b=TemplatePolicy(enabled=True, mode="normal", max_positions=1)
    )
    plans = [
        _plan(f"00010{i}.SZ", f"行业{i}", template=Template.B, score=90 - i)
        for i in range(2)
    ]
    result = _apply(plans, regime=regime, config=config)
    approved = [
        plan for plan in result if plan.status == PlanStatus.PENDING_ENTRY
    ]
    rejected = [
        plan for plan in result if plan.status == PlanStatus.REJECTED
    ]
    assert len(approved) == 1
    assert approved[0].candidate.ts_code == "000100.SZ"
    assert len(rejected) == 1
    reason = _reasons(rejected[0])[0]
    assert "max B positions" in reason
    assert "config 2" in reason
    assert "policy 1" in reason
    assert "policy binds" in reason


def test_b_cap_reason_names_config_binding_source():
    config = load_strategy_config()
    regime = _regime(
        b=TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    )
    plans = [
        _plan(f"00012{i}.SZ", f"行业{i}", template=Template.B, score=90 - i)
        for i in range(3)
    ]
    result = _apply(plans, regime=regime, config=config)
    rejected = [
        plan for plan in result if plan.status == PlanStatus.REJECTED
    ]
    assert len(rejected) == 1
    reason = _reasons(rejected[0])[0]
    assert "config 2" in reason
    assert "policy 3" in reason
    assert "config binds" in reason


def test_b_config_cap_applies_when_policy_allows_more():
    config = load_strategy_config()
    regime = _regime(
        b=TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    )
    plans = [
        _plan(f"00011{i}.SZ", f"行业{i}", template=Template.B, score=90 - i)
        for i in range(3)
    ]
    result = _apply(plans, regime=regime, config=config)
    assert sum(
        1 for plan in result if plan.status == PlanStatus.PENDING_ENTRY
    ) == 2


def test_pending_plans_consume_batch_capacity():
    account = _account(
        pending=[
            _pending(ts_code=f"00030{i}.SZ", sector=f"行业{i}")
            for i in range(3)
        ]
    )
    result = _apply([_plan("000001.SZ", "银行")], account=account)
    assert result[0].status == PlanStatus.REJECTED
    assert "max positions" in _reasons(result[0])[0]


def test_industry_cap_applies_across_batch():
    plans = [
        _plan(f"00040{i}.SZ", "银行", score=90 - i) for i in range(3)
    ]
    result = _apply(plans)
    approved = [
        plan for plan in result if plan.status == PlanStatus.PENDING_ENTRY
    ]
    rejected = [
        plan for plan in result if plan.status == PlanStatus.REJECTED
    ]
    assert len(approved) == 2
    assert len(rejected) == 1
    assert "industry" in _reasons(rejected[0])[0]


def test_missing_template_policy_rejects():
    regime = _regime().model_copy(
        update={
            "template_availability": {
                Template.A: TemplatePolicy(
                    enabled=True, mode="normal", max_positions=3
                )
            }
        }
    )
    result = _apply(
        [_plan("000001.SZ", "银行", template=Template.B)], regime=regime
    )
    assert result[0].status == PlanStatus.REJECTED
    assert "policy" in _reasons(result[0])[0]


@pytest.mark.parametrize("template", [Template.A, Template.B])
def test_governor_rejects_observe_mode(template: Template):
    observe = TemplatePolicy(enabled=True, mode="observe", max_positions=3)
    regime = (
        _regime(a=observe) if template == Template.A else _regime(b=observe)
    )
    result = _apply(
        [_plan("000001.SZ", "银行", template=template)], regime=regime
    )
    assert result[0].status == PlanStatus.REJECTED
    assert "observe" in _reasons(result[0])[0]


@pytest.mark.parametrize("template", [Template.A, Template.B])
def test_sizer_rejects_observe_mode(template: Template):
    observe = TemplatePolicy(enabled=True, mode="observe", max_positions=3)
    regime = (
        _regime(a=observe) if template == Template.A else _regime(b=observe)
    )
    with pytest.raises(PositionRejectedError, match="observe"):
        _size(_account(), regime=regime, template=template)


def test_start_stage_gives_template_b_priority():
    config = _config_with_risk(max_positions=1)
    regime = _regime(sentiment_stage=SentimentStage.START)
    plans = [
        _plan("000001.SZ", "银行", template=Template.A, score=99.0),
        _plan("000002.SZ", "科技", template=Template.B, score=80.0),
    ]
    result = _apply(plans, regime=regime, config=config)
    by_code = {plan.candidate.ts_code: plan for plan in result}
    assert by_code["000002.SZ"].status == PlanStatus.PENDING_ENTRY
    assert by_code["000001.SZ"].status == PlanStatus.REJECTED


def test_equal_scores_tie_break_by_ts_code():
    config = _config_with_risk(max_positions=1)
    # Reversed input order: deterministic priority must still prefer the
    # lexicographically smaller ts_code.
    plans = [
        _plan("000002.SZ", "科技", score=90.0),
        _plan("000001.SZ", "银行", score=90.0),
    ]
    result = _apply(plans, config=config)
    assert [plan.candidate.ts_code for plan in result] == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[1].status == PlanStatus.REJECTED


# --- correlation -------------------------------------------------------------


def test_correlation_against_existing_position_rejects():
    result = _apply(
        [_plan("000001.SZ", "银行")],
        account=_account(positions=[_position()]),
        correlations={"000001.SZ": {"000009.SZ": 0.81}},
    )
    assert result[0].status == PlanStatus.REJECTED
    assert "correlation" in _reasons(result[0])[0]
    assert "000009.SZ" in _reasons(result[0])[0]


def test_correlation_against_pending_plan_rejects():
    result = _apply(
        [_plan("000001.SZ", "银行")],
        account=_account(pending=[_pending(ts_code="000008.SZ")]),
        correlations={"000001.SZ": {"000008.SZ": 0.90}},
    )
    assert result[0].status == PlanStatus.REJECTED
    assert "correlation" in _reasons(result[0])[0]


@pytest.mark.parametrize(
    "correlations",
    [
        {"000002.SZ": {"000001.SZ": 0.90}},
        {"000001.SZ": {"000002.SZ": 0.90}},
    ],
)
def test_correlation_keeps_higher_scoring_approved_plan(
    correlations: dict[str, dict[str, float]],
):
    plans = [
        _plan("000001.SZ", "行业1", score=95.0),
        _plan("000002.SZ", "行业2", score=90.0),
    ]
    result = _apply(plans, correlations=correlations)
    by_code = {plan.candidate.ts_code: plan for plan in result}
    assert by_code["000001.SZ"].status == PlanStatus.PENDING_ENTRY
    assert by_code["000002.SZ"].status == PlanStatus.REJECTED
    assert "correlation" in _reasons(by_code["000002.SZ"])[0]


def test_start_template_priority_wins_over_score_for_correlations():
    # Template B is processed first in START, so it wins the correlated pair
    # even though the Template A plan scores higher.
    regime = _regime(sentiment_stage=SentimentStage.START)
    plans = [
        _plan("000001.SZ", "银行", template=Template.A, score=99.0),
        _plan("000002.SZ", "科技", template=Template.B, score=80.0),
    ]
    result = _apply(
        plans,
        regime=regime,
        correlations={"000001.SZ": {"000002.SZ": 0.90}},
    )
    by_code = {plan.candidate.ts_code: plan for plan in result}
    assert by_code["000002.SZ"].status == PlanStatus.PENDING_ENTRY
    assert by_code["000001.SZ"].status == PlanStatus.REJECTED
    assert "correlation" in _reasons(by_code["000001.SZ"])[0]


def test_approval_reason_marks_correlation_unavailable():
    result = _apply([_plan("000001.SZ", "银行")])
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[0].risk_decision.outcome == RiskOutcome.APPROVE
    assert _reasons(result[0]) == ["correlation unavailable"]

    with_correlations = _apply(
        [_plan("000001.SZ", "银行")], correlations={}
    )
    assert _reasons(with_correlations[0]) == ["all checks passed"]


def test_correlation_unavailable_set_marks_approved_reason():
    result = _apply(
        [_plan("000001.SZ", "银行")],
        correlations={},
        correlation_unavailable={"000001.SZ"},
    )
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert _reasons(result[0]) == [
        "all checks passed",
        "correlation unavailable",
    ]


def test_correlation_unavailable_set_appends_to_reduced_decision():
    config = _config_with_risk(max_single_position_pct=0.05)
    result = _apply(
        [_plan("000001.SZ", "银行")],
        config=config,
        correlations={},
        correlation_unavailable={"000001.SZ"},
    )
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[0].risk_decision.outcome == RiskOutcome.REDUCE
    assert "correlation unavailable" in _reasons(result[0])


def test_correlation_threshold_is_exclusive():
    result = _apply(
        [_plan("000001.SZ", "银行")],
        account=_account(positions=[_position()]),
        correlations={"000001.SZ": {"000009.SZ": 0.80}},
    )
    assert result[0].status == PlanStatus.PENDING_ENTRY


def test_correlation_ignores_unheld_codes_and_missing_mapping():
    result = _apply(
        [_plan("000001.SZ", "银行")],
        account=_account(positions=[_position()]),
        correlations={"000001.SZ": {"999999.SZ": 0.99}},
    )
    assert result[0].status == PlanStatus.PENDING_ENTRY

    result = _apply([_plan("000001.SZ", "银行")])
    assert result[0].status == PlanStatus.PENDING_ENTRY


@pytest.mark.parametrize("value", [float("nan"), 1.5, -1.5])
def test_invalid_correlation_rejects(value: float):
    result = _apply(
        [_plan("000001.SZ", "银行")],
        account=_account(positions=[_position()]),
        correlations={"000001.SZ": {"000009.SZ": value}},
    )
    assert result[0].status == PlanStatus.REJECTED
    assert "invalid correlation" in _reasons(result[0])[0]


# --- reduce reason attribution ----------------------------------------------


def test_reduce_reason_names_single_position_cap():
    result = _apply([_plan("000001.SZ", "银行", shares=50_000)])
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[0].risk_decision.outcome == RiskOutcome.REDUCE
    assert result[0].position_plan.shares == 40_000
    assert _reasons(result[0]) == ["single-position cap reduced size"]


def test_reduce_reason_names_cash_reserve():
    result = _apply(
        [_plan("000001.SZ", "银行", shares=20_000)],
        account=_account(cash=200_000),
    )
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[0].risk_decision.outcome == RiskOutcome.REDUCE
    assert result[0].position_plan.shares == 10_000
    assert _reasons(result[0]) == ["cash reserve reduced size"]


def test_reduce_reason_names_exposure_cap():
    position = _position(
        ts_code="000009.SZ",
        sector="地产",
        shares=80_000,
        current_stop=7.5,
        current_price=8.0,
    )
    result = _apply(
        [_plan("000001.SZ", "银行", shares=30_000, stop_price=9.5)],
        account=_account(positions=[position]),
    )
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[0].risk_decision.outcome == RiskOutcome.REDUCE
    assert result[0].position_plan.shares == 26_000
    assert _reasons(result[0]) == ["exposure cap reduced size"]


def test_reduce_reason_names_every_binding_cap():
    result = _apply(
        [_plan("000001.SZ", "银行", shares=50_000)],
        account=_account(cash=200_000),
    )
    assert result[0].risk_decision.outcome == RiskOutcome.REDUCE
    assert result[0].position_plan.shares == 10_000
    assert _reasons(result[0]) == [
        "single-position cap reduced size",
        "cash reserve reduced size",
    ]


def test_batch_heat_before_chains_approved_plans():
    plans = [
        _plan("000001.SZ", "行业1", score=95.0),
        _plan("000002.SZ", "行业2", score=90.0),
    ]
    result = _apply(plans)
    assert all(plan.status == PlanStatus.PENDING_ENTRY for plan in result)
    first, second = result
    assert first.position_plan.heat_before == 0.0
    assert second.position_plan.heat_before == pytest.approx(
        first.position_plan.heat_after
    )
    assert second.position_plan.heat_after == pytest.approx(0.02)


def test_below_one_lot_reason_names_caps_and_shares():
    result = _apply(
        [_plan("000001.SZ", "银行", shares=10_000)],
        account=_account(cash=100_050),
    )
    assert result[0].status == PlanStatus.REJECTED
    reason = _reasons(result[0])[0]
    assert "below one lot" in reason
    assert "0 shares remaining" in reason
    assert "cash reserve" in reason


# --- drawdown halt at governor level ----------------------------------------


def test_governor_rejects_all_when_trading_halted():
    plans = [_plan("000001.SZ", "银行"), _plan("000002.SZ", "科技")]
    result = _apply(plans, account=_account(trading_halted=True))
    assert all(plan.status == PlanStatus.REJECTED for plan in result)
    for plan in result:
        reasons = " ".join(_reasons(plan))
        assert "circuit breaker" in reasons
        assert "manual restart required" in reasons


def test_governor_rejects_all_at_drawdown_stop():
    plans = [_plan("000001.SZ", "银行"), _plan("000002.SZ", "科技")]
    result = _apply(
        plans, account=_account(equity=850_000, peak=1_000_000)
    )
    assert all(plan.status == PlanStatus.REJECTED for plan in result)
    for plan in result:
        assert "manual restart required" in " ".join(_reasons(plan))


def test_governor_may_approve_while_half_latched():
    result = _apply(
        [_plan("000001.SZ", "银行")],
        account=_account(
            equity=920_000, peak=1_000_000, drawdown_half_latched=True
        ),
    )
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[0].risk_decision.outcome == RiskOutcome.APPROVE


def test_governor_approves_after_manual_restart():
    halted = _account(equity=850_000, peak=1_000_000).with_risk_latches(
        stop_at=0.15, half_at=0.10
    )
    recovered = halted.model_copy(
        update={
            "equity": Decimal("960000"),
            "cash": Decimal("960000"),
        }
    ).with_risk_latches(stop_at=0.15, half_at=0.10)
    assert recovered.trading_halted is True
    result = _apply([_plan("000001.SZ", "银行")], account=recovered)
    assert result[0].status == PlanStatus.REJECTED

    restarted = recovered.model_copy(update={"trading_halted": False})
    result = _apply([_plan("000001.SZ", "银行")], account=restarted)
    assert result[0].status == PlanStatus.PENDING_ENTRY


# --- plan validation ---------------------------------------------------------


def test_plan_with_entry_not_above_stop_rejects():
    plan = _plan("000001.SZ", "银行")
    bad = plan.model_copy(
        update={
            "stop_plan": plan.stop_plan.model_copy(
                update={"initial_stop": price(10.0)}
            )
        }
    )
    result = _apply([bad])
    assert result[0].status == PlanStatus.REJECTED
    assert "entry price" in _reasons(result[0])[0]


def test_plan_with_zero_shares_rejects():
    result = _apply([_plan("000001.SZ", "银行", shares=0)])
    assert result[0].status == PlanStatus.REJECTED
    assert "shares" in _reasons(result[0])[0]


def test_plan_with_non_positive_position_value_rejects():
    plan = _plan("000001.SZ", "银行")
    bad = plan.model_copy(
        update={
            "position_plan": plan.position_plan.model_copy(
                update={"position_value": money(0)}
            )
        }
    )
    result = _apply([bad])
    assert result[0].status == PlanStatus.REJECTED
    assert "position value" in _reasons(result[0])[0]


@pytest.mark.parametrize(
    ("field", "changes", "reason"),
    [
        (
            "entry_plan",
            {
                "trigger_price": Decimal("-1"),
                "entry_zone_low": Decimal("-1"),
                "entry_zone_high": Decimal("-1"),
            },
            "entry price",
        ),
        ("stop_plan", {"initial_stop": Decimal("-1")}, "stop"),
    ],
)
def test_plan_with_non_positive_prices_rejects(
    field: str, changes: dict[str, Decimal], reason: str
):
    plan = _plan("000001.SZ", "银行")
    nested = getattr(plan, field).model_copy(update=changes)
    result = _apply([plan.model_copy(update={field: nested})])
    assert result[0].status == PlanStatus.REJECTED
    assert reason in _reasons(result[0])[0]


def test_non_lot_aligned_shares_reject():
    result = _apply([_plan("000001.SZ", "银行", shares=10_050)])
    assert result[0].status == PlanStatus.REJECTED
    assert "multiple of 100" in _reasons(result[0])[0]


def test_position_template_mismatch_rejects():
    plan = _plan("000001.SZ", "银行")
    bad = plan.model_copy(
        update={
            "position_plan": plan.position_plan.model_copy(
                update={"template": Template.B}
            )
        }
    )
    result = _apply([bad])
    assert result[0].status == PlanStatus.REJECTED
    assert "template" in _reasons(result[0])[0]


def test_non_draft_status_rejects():
    plan = _plan("000001.SZ", "银行")
    bad = plan.model_copy(update={"status": PlanStatus.PENDING_ENTRY})
    result = _apply([bad])
    assert result[0].status == PlanStatus.REJECTED
    assert "DRAFT" in _reasons(result[0])[0]


def test_expired_plan_rejects():
    plan = _plan("000001.SZ", "银行")
    bad = plan.model_copy(update={"expires_on": date(2024, 1, 1)})
    result = _apply([bad])
    assert result[0].status == PlanStatus.REJECTED
    assert "expired" in _reasons(result[0])[0]


def test_duplicate_code_within_batch_rejects_later_plan():
    plans = [
        _plan("000001.SZ", "银行", score=95.0),
        _plan("000001.SZ", "科技", score=90.0),
    ]
    result = _apply(plans)
    assert result[0].candidate.score == 95.0
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[1].status == PlanStatus.REJECTED
    assert "duplicate" in _reasons(result[1])[0]


def test_duplicate_code_with_existing_position_rejects():
    result = _apply(
        [_plan("000001.SZ", "银行")],
        account=_account(positions=[_position(ts_code="000001.SZ")]),
    )
    assert result[0].status == PlanStatus.REJECTED
    assert "duplicate" in _reasons(result[0])[0]


def test_disabled_first_plan_does_not_block_later_valid_plan():
    regime = _regime(
        a=TemplatePolicy(enabled=False, mode="observe", max_positions=0)
    )
    plans = [
        _plan("000001.SZ", "银行", template=Template.A),
        _plan("000001.SZ", "银行", template=Template.B),
    ]
    result = _apply(plans, regime=regime)
    by_template = {plan.candidate.template: plan for plan in result}
    assert by_template[Template.A].status == PlanStatus.REJECTED
    assert by_template[Template.B].status == PlanStatus.PENDING_ENTRY


def test_invalid_first_plan_does_not_block_later_valid_plan():
    plans = [
        _plan("000001.SZ", "银行", shares=0),
        _plan("000001.SZ", "银行", shares=10_000),
    ]
    result = _apply(plans)
    assert result[0].status == PlanStatus.REJECTED
    assert result[1].status == PlanStatus.PENDING_ENTRY
