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
    PositionState,
    RegimeIndicators,
    TemplatePolicy,
)
from app.domain.money import money, price


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
