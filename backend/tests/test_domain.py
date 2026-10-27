from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.enums import PlanStatus, SentimentStage, Template, TrendRegime
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


def _candidate(**overrides) -> Candidate:
    data = {
        "ts_code": "000001.SZ",
        "trade_date": date(2024, 1, 2),
        "template": Template.A,
        "setup_type": "mainline_pullback",
        "horizon": "mid",
        "sector": "银行",
        "leader_rank": 1,
        "leader_score": 90.0,
        "score": 88.0,
        "reasons": ["mainline"],
        "features": {"close": 10.0},
        "sentiment_stage": SentimentStage.FERMENT,
        "strategy_version": "sv_test",
        "data_version": "dv_test",
    }
    data.update(overrides)
    return Candidate(**data)


def _entry_plan(**overrides) -> EntryPlan:
    data = {
        "type": "pullback",
        "entry_zone_low": Decimal("9.80"),
        "entry_zone_high": Decimal("10.20"),
        "validity_days": 3,
        "max_chase_pct": 0.03,
        "expires_on": date(2024, 1, 5),
    }
    data.update(overrides)
    return EntryPlan(**data)


def _stop_plan(**overrides) -> StopPlan:
    data = {
        "initial_stop": Decimal("9.00"),
        "stop_distance_pct": 0.05,
        "atr14": 0.30,
        "structure_ref": "ma20",
        "time_stop_days": 10,
        "trailing_rules": ["break below MA20"],
        "r_value": Decimal("1.00"),
    }
    data.update(overrides)
    return StopPlan(**data)


def _trade_plan(**overrides) -> TradePlan:
    data = {
        "candidate": _candidate(),
        "entry_plan": _entry_plan(),
        "stop_plan": _stop_plan(),
        "position_plan": PositionPlan(
            template=Template.A,
            risk_pct=0.01,
            risk_amount=Decimal("1000"),
            shares=1000,
            position_value=Decimal("10000"),
            position_pct=0.01,
            heat_before=0.0,
            heat_after=0.01,
        ),
        "take_profit_plan": TakeProfitPlan(
            targets=[TakeProfitTarget(r_multiple=2.0, price=Decimal("12.00"), fraction=0.5)],
            trailing_rule="trail at 2R",
            time_exit_rule="exit after 10 days",
        ),
        "expires_on": date(2024, 1, 5),
        "strategy_version": "sv_test",
        "data_version": "dv_test",
        "code_version": "cv_test",
    }
    data.update(overrides)
    return TradePlan(**data)


def test_price_rounds_half_up():
    assert price("10.005") == Decimal("10.01")
    assert price(9.994) == Decimal("9.99")
    assert money("1234.567") == Decimal("1234.57")


def test_candidate_requires_reasons():
    with pytest.raises(ValidationError):
        Candidate(
            ts_code="000001.SZ",
            trade_date=date(2024, 1, 2),
            template=Template.A,
            setup_type="mainline_pullback",
            horizon="mid",
            sector="银行",
            leader_rank=1,
            leader_score=90.0,
            score=88.0,
            reasons=[],
            features={"close": 10.0},
            sentiment_stage=SentimentStage.FERMENT,
            strategy_version="sv_test",
            data_version="dv_test",
        )


def test_market_regime_requires_both_templates():
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    with pytest.raises(ValidationError):
        MarketRegime(
            trade_date=date(2024, 1, 2),
            trend_regime=TrendRegime.ATTACK,
            sentiment_stage=SentimentStage.FERMENT,
            risk_factor=1.0,
            template_availability={Template.A: policy},
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


def test_entry_plan_zone_midpoint():
    plan = EntryPlan(
        type="pullback",
        entry_zone_low=Decimal("9.80"),
        entry_zone_high=Decimal("10.20"),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=["close < MA20"],
        expires_on=date(2024, 1, 5),
    )
    assert plan.entry_price == Decimal("10.00")


def test_entry_plan_rejects_reversed_zone():
    with pytest.raises(ValidationError, match="entry_zone_low must be <= entry_zone_high"):
        _entry_plan(entry_zone_low=Decimal("10.20"), entry_zone_high=Decimal("9.80"))


def test_entry_plan_requires_trigger_or_full_zone():
    with pytest.raises(ValidationError, match="trigger_price or a full entry zone is required"):
        _entry_plan(entry_zone_low=None, entry_zone_high=None)


def test_entry_plan_rejects_partial_zone_with_trigger():
    with pytest.raises(
        ValidationError, match="both entry_zone_low and entry_zone_high are required"
    ):
        _entry_plan(trigger_price=Decimal("10.00"), entry_zone_high=None)


def test_entry_plan_trigger_price_is_quantized():
    plan = _entry_plan(
        entry_zone_low=None,
        entry_zone_high=None,
        trigger_price=Decimal("10.005"),
    )
    assert plan.entry_price == Decimal("10.01")


def test_entry_plan_rejects_non_positive_prices():
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        _entry_plan(trigger_price=Decimal("0"))
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        _entry_plan(entry_zone_low=Decimal("-1"))


def test_stop_plan_rejects_non_positive_initial_stop():
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        _stop_plan(initial_stop=Decimal("0"))


def test_candidate_rejects_unknown_field():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _candidate(unexpected="value")


def test_trade_plan_defaults():
    plan = _trade_plan()
    assert plan.status is PlanStatus.DRAFT
    assert len(plan.id) == 32
    assert all(char in "0123456789abcdef" for char in plan.id)
    assert plan.created_at.tzinfo is not None
    assert plan.created_at.utcoffset() is not None


def test_trade_plan_rejects_entry_at_or_below_stop():
    with pytest.raises(ValidationError, match="entry price must be above initial stop"):
        _trade_plan(stop_plan=_stop_plan(initial_stop=Decimal("10.00")))


def test_pending_exposure_rejects_stop_at_or_above_entry():
    with pytest.raises(ValidationError, match="stop_price must be below entry_price"):
        PendingExposure(
            ts_code="000002.SZ",
            sector="银行",
            entry_price=Decimal("20.00"),
            stop_price=Decimal("20.00"),
            shares=5000,
            template=Template.A,
        )


def test_account_heat_includes_pending():
    account = AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("500000"),
        peak_equity=Decimal("1000000"),
        positions=[
            PositionState(
                ts_code="000001.SZ",
                sector="银行",
                shares=10000,
                available_shares=10000,
                cost_price=Decimal("10.00"),
                current_stop=Decimal("9.00"),
                current_price=Decimal("11.00"),
                template=Template.A,
            )
        ],
        pending_plans=[
            PendingExposure(
                ts_code="000002.SZ",
                sector="银行",
                entry_price=Decimal("20.00"),
                stop_price=Decimal("18.00"),
                shares=5000,
                template=Template.A,
            )
        ],
    )
    assert account.heat == pytest.approx((2.0 * 10000 + 2.0 * 5000) / 1000000)


def test_plan_status_defaults():
    assert PlanStatus.DRAFT.value == "DRAFT"
