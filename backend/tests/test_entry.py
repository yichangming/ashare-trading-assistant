from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig, load_strategy_config
from app.domain.enums import EntryType, SentimentStage, Template
from app.domain.models import (
    Candidate,
    EntryPlan,
    MarketRegime,
    RegimeIndicators,
    TemplatePolicy,
)
from app.domain.money import price
from app.planning.entry import EntryPlanner
from app.planning.errors import EntryInvalidError

_A_CANCEL_CONDITIONS = [
    "close below entry zone low",
    "close below MA20",
    "gap below zone low by more than 3%",
]
_B_CANCEL_CONDITIONS = [
    "close below stop",
    "close below MA5/MA10 without recovery",
    "sentiment stage turns ebb",
]


def _calendar() -> TradingCalendar:
    return TradingCalendar(
        [date(2024, 1, d) for d in (2, 3, 4, 5, 6, 7)]
    )


def _regime() -> MarketRegime:
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime="attack",
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={
            Template.A: policy,
            Template.B: policy,
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


def _candidate(template: Template, features: dict) -> Candidate:
    return Candidate(
        ts_code="000001.SZ",
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={key: float(value) for key, value in features.items()},
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


def _plan(
    candidate: Candidate,
    *,
    calendar: TradingCalendar | None = None,
    as_of: date = date(2024, 1, 2),
    config: StrategyConfig | None = None,
) -> EntryPlan:
    return EntryPlanner().plan(
        candidate,
        candidate.features,
        _regime(),
        calendar if calendar is not None else _calendar(),
        as_of,
        config if config is not None else load_strategy_config(),
    )


def _config_a(**updates: float) -> StrategyConfig:
    config = load_strategy_config()
    return config.model_copy(
        update={"template_a": config.template_a.model_copy(update=updates)}
    )


def _config_b(**updates: float) -> StrategyConfig:
    config = load_strategy_config()
    return config.model_copy(
        update={"template_b": config.template_b.model_copy(update=updates)}
    )


def _b_features(**overrides: float) -> dict[str, float]:
    features = {
        "ma5_raw": 10.5,
        "ma10_raw": 10.0,
        "board_height": 1,
        "days_since_limit_up": 0,
    }
    features.update(overrides)
    return features


def test_a_pullback_zone():
    config = load_strategy_config()
    candidate = _candidate(
        Template.A, {"ma20_raw": 10.0, "ma10_raw": 9.9}
    )
    plan = EntryPlanner().plan(
        candidate, candidate.features, _regime(), _calendar(),
        date(2024, 1, 2), config,
    )
    assert plan.type == EntryType.PULLBACK
    assert float(plan.entry_zone_low) == 9.80
    assert float(plan.entry_zone_high) == 10.20
    assert plan.expires_on == date(2024, 1, 5)


def test_b_divergence_and_confirmation():
    config = load_strategy_config()
    base = {
        "ma5_raw": 10.5,
        "ma10_raw": 10.0,
        "board_height": 1,
        "days_since_limit_up": 0,
    }
    candidate = _candidate(Template.B, base)
    plan = EntryPlanner().plan(
        candidate, candidate.features, _regime(), _calendar(),
        date(2024, 1, 2), config,
    )
    assert plan.type == EntryType.DIVERGENCE_LOW

    candidate2 = _candidate(
        Template.B,
        {**base, "board_height": 2},
    )
    plan2 = EntryPlanner().plan(
        candidate2, candidate2.features, _regime(), _calendar(),
        date(2024, 1, 2), config,
    )
    assert plan2.type == EntryType.CONFIRMATION


def test_b_invalid_ma_order():
    config = load_strategy_config()
    candidate = _candidate(
        Template.B,
        {
            "ma5_raw": 10.0,
            "ma10_raw": 10.5,
            "board_height": 1,
            "days_since_limit_up": 0,
        },
    )
    with pytest.raises(EntryInvalidError):
        EntryPlanner().plan(
            candidate, candidate.features, _regime(), _calendar(),
            date(2024, 1, 2), config,
        )


def test_a_plan_metadata():
    plan = _plan(_candidate(Template.A, {"ma20_raw": 10.0}))

    assert plan.trigger_price == Decimal("10.00")
    assert plan.entry_zone_low == Decimal("9.80")
    assert plan.entry_zone_high == Decimal("10.20")
    assert plan.entry_price == Decimal("10.00")
    assert plan.validity_days == 3
    assert plan.max_chase_pct == pytest.approx(0.03)
    assert plan.cancel_conditions == _A_CANCEL_CONDITIONS
    assert plan.expires_on == date(2024, 1, 5)


def test_a_zone_rounds_to_cents():
    plan = _plan(_candidate(Template.A, {"ma20_raw": 10.005}))

    assert plan.trigger_price == Decimal("10.01")
    assert plan.entry_zone_low == Decimal("9.80")
    assert plan.entry_zone_high == Decimal("10.21")
    assert plan.entry_price == Decimal("10.01")
    assert plan.entry_zone_low <= plan.entry_zone_high


def test_a_half_cent_zone_uses_decimal_rounding():
    plan = _plan(_candidate(Template.A, {"ma20_raw": 7.25}))

    assert plan.entry_zone_low == Decimal("7.11")
    assert plan.entry_zone_high == Decimal("7.40")
    assert plan.trigger_price == Decimal("7.25")
    assert plan.entry_zone_low <= plan.trigger_price <= plan.entry_zone_high
    assert plan.entry_price == price(
        (plan.entry_zone_low + plan.entry_zone_high) / 2
    )
    assert plan.entry_price == Decimal("7.26")


def test_a_rejects_zone_that_rounds_to_zero():
    candidate = _candidate(Template.A, {"ma20_raw": 0.001})

    with pytest.raises(EntryInvalidError, match="zone"):
        _plan(candidate)


@pytest.mark.parametrize("ma20", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_a_rejects_invalid_ma20(ma20: float | None) -> None:
    features = {} if ma20 is None else {"ma20_raw": ma20}
    candidate = _candidate(Template.A, features)
    with pytest.raises(EntryInvalidError):
        _plan(candidate)


def test_a_rejects_over_extended_reference_price():
    candidate = _candidate(Template.A, {"ma20_raw": 10.0, "close": 10.81})

    with pytest.raises(EntryInvalidError, match="extension"):
        _plan(candidate)


@pytest.mark.parametrize("close", [float("nan"), float("inf"), 0.0, -1.0])
def test_a_rejects_non_finite_or_non_positive_close(close: float) -> None:
    candidate = _candidate(Template.A, {"ma20_raw": 10.0, "close": close})

    with pytest.raises(EntryInvalidError, match="close"):
        _plan(candidate)


def test_a_allows_reference_price_at_extension_cap():
    candidate = _candidate(Template.A, {"ma20_raw": 10.0, "close": 10.8})

    plan = _plan(candidate)

    assert plan.type == EntryType.PULLBACK


@pytest.mark.parametrize("valid_days", [0, -1])
def test_a_rejects_non_positive_validity_days(valid_days: int) -> None:
    candidate = _candidate(Template.A, {"ma20_raw": 10.0})

    with pytest.raises(EntryInvalidError, match="validity"):
        _plan(candidate, config=_config_a(entry_valid_days=valid_days))


def test_a_rejects_non_trading_as_of():
    candidate = _candidate(Template.A, {"ma20_raw": 10.0})

    with pytest.raises(EntryInvalidError, match="not a trading day"):
        _plan(candidate, as_of=date(2024, 1, 8))


def test_a_rejects_calendar_coverage_shortfall():
    candidate = _candidate(Template.A, {"ma20_raw": 10.0})

    with pytest.raises(EntryInvalidError, match="calendar coverage"):
        _plan(candidate, as_of=date(2024, 1, 5))


def test_a_rejects_validity_exhausting_calendar():
    candidate = _candidate(Template.A, {"ma20_raw": 10.0})

    with pytest.raises(EntryInvalidError, match="calendar coverage"):
        _plan(
            candidate,
            as_of=date(2024, 1, 7),
            config=_config_a(entry_valid_days=1),
        )


def test_b_divergence_plan_metadata():
    candidate = _candidate(Template.B, _b_features())
    plan = _plan(candidate)

    assert plan.type == EntryType.DIVERGENCE_LOW
    assert plan.trigger_price is None
    assert plan.entry_zone_low == Decimal("9.80")
    assert plan.entry_zone_high == Decimal("10.71")
    assert plan.entry_price == Decimal("10.26")
    assert plan.validity_days == 2
    assert plan.max_chase_pct == pytest.approx(0.03)
    assert plan.cancel_conditions == _B_CANCEL_CONDITIONS
    assert plan.expires_on == date(2024, 1, 4)


def test_b_confirmation_plan_metadata():
    candidate = _candidate(Template.B, _b_features(board_height=2))
    plan = _plan(candidate)

    assert plan.type == EntryType.CONFIRMATION
    assert plan.trigger_price == Decimal("10.50")
    assert plan.entry_zone_low == Decimal("10.40")
    assert plan.entry_zone_high == Decimal("10.61")
    assert plan.entry_price == Decimal("10.51")
    assert plan.validity_days == 2
    assert plan.expires_on == date(2024, 1, 4)


def test_b_confirmation_half_cent_zone_uses_decimal_rounding():
    candidate = _candidate(
        Template.B,
        _b_features(ma5_raw=20.5, ma10_raw=20.0, board_height=2),
    )
    plan = _plan(candidate)

    assert plan.type == EntryType.CONFIRMATION
    assert plan.trigger_price == Decimal("20.50")
    assert plan.entry_zone_low == Decimal("20.30")
    assert plan.entry_zone_high == Decimal("20.71")
    assert plan.entry_zone_low <= plan.trigger_price <= plan.entry_zone_high
    assert plan.entry_price == Decimal("20.51")


def test_b_divergence_half_cent_zone_uses_decimal_rounding():
    candidate = _candidate(
        Template.B,
        _b_features(ma5_raw=8.75, ma10_raw=8.25, board_height=1),
    )
    plan = _plan(candidate)

    assert plan.type == EntryType.DIVERGENCE_LOW
    assert plan.trigger_price is None
    assert plan.entry_zone_low == Decimal("8.09")
    assert plan.entry_zone_high == Decimal("8.93")
    assert plan.entry_price == Decimal("8.51")


@pytest.mark.parametrize(
    "overrides",
    [
        {"ma5_raw": None},
        {"ma10_raw": None},
        {"ma5_raw": 0.0},
        {"ma10_raw": -1.0},
        {"ma5_raw": float("nan")},
        {"ma10_raw": float("inf")},
    ],
)
def test_b_rejects_invalid_mas(overrides: dict[str, float | None]) -> None:
    features = _b_features()
    for key, value in overrides.items():
        if value is None:
            features.pop(key, None)
        else:
            features[key] = value
    candidate = _candidate(Template.B, features)

    with pytest.raises(EntryInvalidError, match="MA5/MA10"):
        _plan(candidate)


@pytest.mark.parametrize(
    "board_height",
    [None, 0.0, -1.0, 1.5, 3.0, float("nan")],
)
def test_b_rejects_invalid_board_height(
    board_height: float | None,
) -> None:
    features = _b_features()
    if board_height is None:
        features.pop("board_height")
    else:
        features["board_height"] = board_height
    candidate = _candidate(Template.B, features)

    with pytest.raises(EntryInvalidError, match="board height"):
        _plan(candidate)


@pytest.mark.parametrize("board_height", [1, 2])
def test_b_rejects_zones_that_round_to_zero(board_height: int) -> None:
    candidate = _candidate(
        Template.B,
        _b_features(ma5_raw=0.001, ma10_raw=0.001, board_height=board_height),
    )

    with pytest.raises(EntryInvalidError):
        _plan(candidate)


@pytest.mark.parametrize("valid_days", [0, -1])
def test_b_rejects_non_positive_validity_days(valid_days: int) -> None:
    candidate = _candidate(Template.B, _b_features())

    with pytest.raises(EntryInvalidError, match="validity"):
        _plan(candidate, config=_config_b(entry_valid_days=valid_days))


def test_b_rejects_non_trading_as_of():
    candidate = _candidate(Template.B, _b_features())

    with pytest.raises(EntryInvalidError, match="not a trading day"):
        _plan(candidate, as_of=date(2024, 1, 8))


def test_b_rejects_calendar_coverage_shortfall():
    candidate = _candidate(Template.B, _b_features())

    with pytest.raises(EntryInvalidError, match="calendar coverage"):
        _plan(candidate, as_of=date(2024, 1, 6))


@pytest.mark.parametrize(
    ("template", "features"),
    [(Template.A, {}), (Template.B, {})],
)
def test_entry_errors_identify_candidate_and_template(
    template: Template,
    features: dict[str, float],
) -> None:
    candidate = _candidate(template, features)

    with pytest.raises(EntryInvalidError) as excinfo:
        _plan(candidate)

    message = str(excinfo.value)
    assert "000001.SZ" in message
    assert f"({template.value})" in message
