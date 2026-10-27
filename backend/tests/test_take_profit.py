from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.core.config import StrategyConfig, load_strategy_config
from app.domain.enums import EntryType, SentimentStage, Template
from app.domain.models import Candidate, EntryPlan, StopPlan, TakeProfitPlan
from app.domain.money import price
from app.planning.errors import PoorRiskRewardError
from app.planning.take_profit import TakeProfitPlanner


def _candidate(template: Template, resistance: float | None) -> Candidate:
    features = {"close": 20.0}
    if resistance is not None:
        features["high_60d_raw"] = resistance
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
        features=features,
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


def _entry(price_value: float) -> EntryPlan:
    return EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=price(price_value),
        entry_zone_low=price(price_value * 0.98),
        entry_zone_high=price(price_value * 1.02),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=[],
        expires_on=date(2024, 1, 5),
    )


def _stop(
    entry_price: float,
    stop_price: float,
    time_stop_days: int = 10,
) -> StopPlan:
    return StopPlan(
        initial_stop=price(stop_price),
        stop_distance_pct=(entry_price - stop_price) / entry_price,
        atr14=0.5,
        structure_ref="test",
        time_stop_days=time_stop_days,
        trailing_rules=[],
        r_value=price(entry_price - stop_price),
    )


def _plan(
    template: Template,
    *,
    resistance: float | None = None,
    entry_price: float = 20.0,
    stop_price: float = 18.6,
    time_stop_days: int = 10,
    config: StrategyConfig | None = None,
) -> TakeProfitPlan:
    return TakeProfitPlanner().plan(
        _candidate(template, resistance),
        _entry(entry_price),
        _stop(entry_price, stop_price, time_stop_days),
        config if config is not None else load_strategy_config(),
    )


def _config_a(**take_profit_updates: object) -> StrategyConfig:
    config = load_strategy_config()
    take_profit = config.template_a.take_profit.model_copy(update=take_profit_updates)
    template_a = config.template_a.model_copy(update={"take_profit": take_profit})
    return config.model_copy(update={"template_a": template_a})


def _config_b(**take_profit_updates: object) -> StrategyConfig:
    config = load_strategy_config()
    take_profit = config.template_b.take_profit.model_copy(update=take_profit_updates)
    template_b = config.template_b.model_copy(update={"take_profit": take_profit})
    return config.model_copy(update={"template_b": template_b})


def test_a_targets():
    config = load_strategy_config()
    plan = TakeProfitPlanner().plan(
        _candidate(Template.A, resistance=30.0),
        _entry(20.0),
        _stop(20.0, 18.6),
        config,
    )
    assert [float(target.price) for target in plan.targets] == [22.80, 24.20]
    assert plan.targets[0].fraction == pytest.approx(1 / 3, rel=1e-3)


def test_a_poor_risk_reward_rejected():
    config = load_strategy_config()
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=21.0),
            _entry(20.0),
            _stop(20.0, 18.6),
            config,
        )


def test_b_target():
    config = load_strategy_config()
    plan = TakeProfitPlanner().plan(
        _candidate(Template.B, resistance=None),
        _entry(10.0),
        _stop(10.0, 9.5),
        config,
    )
    assert float(plan.targets[0].price) == 11.00
    assert plan.targets[0].fraction == 0.5


@pytest.mark.parametrize("resistance", [None, float("nan"), float("inf")])
def test_a_absent_or_non_finite_resistance_uses_r_targets(resistance):
    plan = _plan(Template.A, resistance=resistance)
    assert [float(target.price) for target in plan.targets] == [22.80, 24.20]


def test_a_empty_features_uses_r_targets():
    candidate = _candidate(Template.A, resistance=None).model_copy(
        update={"features": {}}
    )
    plan = TakeProfitPlanner().plan(
        candidate,
        _entry(20.0),
        _stop(20.0, 18.6),
        load_strategy_config(),
    )
    assert [float(target.price) for target in plan.targets] == [22.80, 24.20]


@pytest.mark.parametrize("resistance", [19.5, 20.0])
def test_a_resistance_at_or_below_entry_is_ignored(resistance):
    plan = _plan(Template.A, resistance=resistance)
    assert [float(target.price) for target in plan.targets] == [22.80, 24.20]


def test_a_resistance_exactly_one_and_a_half_r_passes_and_caps():
    plan = _plan(Template.A, resistance=22.1)
    assert [float(target.price) for target in plan.targets] == [22.10, 22.10]
    assert [target.r_multiple for target in plan.targets] == pytest.approx([1.5, 1.5])


def test_a_targets_cap_at_resistance_inside_second_target():
    plan = _plan(Template.A, resistance=22.24)  # 1.6R above entry
    assert [float(target.price) for target in plan.targets] == [22.24, 22.24]
    assert [target.r_multiple for target in plan.targets] == pytest.approx([1.6, 1.6])


def test_a_first_target_caps_at_resistance_between_r2_and_r3():
    plan = _plan(Template.A, resistance=23.50)  # 2.5R above entry
    assert [float(target.price) for target in plan.targets] == [22.80, 23.50]
    assert [target.r_multiple for target in plan.targets] == pytest.approx([2.0, 2.5])


def test_a_resistance_beyond_r3_does_not_cap():
    plan = _plan(Template.A, resistance=24.90)  # 3.5R above entry
    assert [float(target.price) for target in plan.targets] == [22.80, 24.20]
    assert [target.r_multiple for target in plan.targets] == pytest.approx([2.0, 3.0])


def test_a_resistance_just_below_one_and_a_half_r_rejected():
    with pytest.raises(PoorRiskRewardError):
        _plan(Template.A, resistance=22.09)


def test_a_fractions_and_r_multiples():
    config = load_strategy_config()
    plan = _plan(Template.A, config=config)
    take_profit = config.template_a.take_profit
    assert [target.r_multiple for target in plan.targets] == pytest.approx(
        [take_profit.r2, take_profit.r3]
    )
    assert plan.targets[0].fraction == pytest.approx(take_profit.first_fraction)
    assert plan.targets[1].fraction == pytest.approx(take_profit.second_fraction)
    assert all(0 < target.fraction <= 1 for target in plan.targets)
    assert all(target.price > 0 for target in plan.targets)


def test_a_configured_r_levels_drive_targets_and_text():
    config = _config_a(r1=1.5, r2=2.5, r3=4.0)
    plan = _plan(Template.A, config=config)
    assert [float(target.price) for target in plan.targets] == [23.50, 25.60]
    assert [target.r_multiple for target in plan.targets] == pytest.approx([2.5, 4.0])
    assert plan.trailing_rule == (
        "sell 33% at 23.50 (+2.5R); sell 33% at 25.60 (+4.0R); "
        "remainder trend; +1.5R move stop to breakeven; "
        "+2.5R move stop to +1.5R; trail by close - 1.5*ATR14"
    )
    assert plan.time_exit_rule == (
        "exit after 10 trading days without +1.5R and below entry"
    )


def test_b_configured_r_level_drives_target_and_text():
    config = _config_b(r2=3.0)
    plan = _plan(Template.B, entry_price=10.0, stop_price=9.5, config=config)
    assert float(plan.targets[0].price) == 11.50
    assert plan.targets[0].r_multiple == pytest.approx(3.0)
    assert plan.trailing_rule == (
        "sell 50% at 11.50 (+3.0R); remainder trail; "
        "+1R move stop to breakeven; +3R move stop to +1R; "
        "trail remainder by MA5/1.5*ATR14"
    )


def test_b_configured_fraction_drives_fraction_and_text():
    config = _config_b(second_fraction=0.4)
    plan = _plan(Template.B, entry_price=10.0, stop_price=9.5, config=config)
    assert plan.targets[0].fraction == 0.4
    assert "sell 40% at 11.00 (+2.0R)" in plan.trailing_rule


def test_unordered_targets_rejected():
    with pytest.raises(PoorRiskRewardError):
        _plan(Template.A, config=_config_a(r2=3.0, r3=2.0))


def test_b_fraction_r_multiple_and_trailing_remainder():
    plan = _plan(Template.B, entry_price=10.0, stop_price=9.5)
    assert len(plan.targets) == 1
    assert plan.targets[0].r_multiple == 2.0
    assert plan.targets[0].fraction == 0.5
    assert "sell 50% at 11.00 (+2.0R)" in plan.trailing_rule
    assert "trail remainder by MA5/1.5*ATR14" in plan.trailing_rule


def test_b_ignores_resistance_gate():
    plan = _plan(Template.B, resistance=10.2, entry_price=10.0, stop_price=9.5)
    assert float(plan.targets[0].price) == 11.00
    assert plan.targets[0].r_multiple == pytest.approx(2.0)


def test_a_trailing_rule():
    plan = _plan(Template.A)
    assert plan.trailing_rule == (
        "sell 33% at 22.80 (+2.0R); sell 33% at 24.20 (+3.0R); "
        "remainder trend; +1R move stop to breakeven; "
        "+2R move stop to +1R; trail by close - 1.5*ATR14"
    )


@pytest.mark.parametrize(
    ("resistance", "expected_sells"),
    [
        (22.1, "sell 33% at 22.10 (+1.5R); sell 33% at 22.10 (+1.5R)"),
        (22.24, "sell 33% at 22.24 (+1.6R); sell 33% at 22.24 (+1.6R)"),
        (23.5, "sell 33% at 22.80 (+2.0R); sell 33% at 23.50 (+2.5R)"),
    ],
)
def test_a_trailing_rule_uses_capped_targets(resistance, expected_sells):
    plan = _plan(Template.A, resistance=resistance)
    assert plan.trailing_rule == (
        f"{expected_sells}; remainder trend; "
        "+1R move stop to breakeven; +2R move stop to +1R; "
        "trail by close - 1.5*ATR14"
    )
    assert plan.time_exit_rule == (
        "exit after 10 trading days without +1R and below entry"
    )


def test_a_text_uses_configured_r1():
    config = _config_a(r1=1.5)
    plan = _plan(Template.A, config=config)
    assert plan.trailing_rule == (
        "sell 33% at 22.80 (+2.0R); sell 33% at 24.20 (+3.0R); "
        "remainder trend; +1.5R move stop to breakeven; "
        "+2R move stop to +1.5R; trail by close - 1.5*ATR14"
    )
    assert plan.time_exit_rule == (
        "exit after 10 trading days without +1.5R and below entry"
    )


def test_b_text_uses_configured_r1():
    config = _config_b(r1=1.5)
    plan = _plan(Template.B, entry_price=10.0, stop_price=9.5, config=config)
    assert plan.trailing_rule == (
        "sell 50% at 11.00 (+2.0R); remainder trail; "
        "+1.5R move stop to breakeven; +2R move stop to +1.5R; "
        "trail remainder by MA5/1.5*ATR14"
    )


@pytest.mark.parametrize(
    ("template", "entry_price", "stop_price", "time_stop_days"),
    [(Template.A, 20.0, 18.6, 10), (Template.B, 10.0, 9.5, 3)],
)
def test_time_exit_uses_stop_time_stop_days(
    template, entry_price, stop_price, time_stop_days
):
    plan = _plan(
        template,
        entry_price=entry_price,
        stop_price=stop_price,
        time_stop_days=time_stop_days,
    )
    assert f"exit after {time_stop_days} trading days" in plan.time_exit_rule


def test_time_exit_is_not_hardcoded():
    plan = _plan(Template.A, time_stop_days=12)
    assert "exit after 12 trading days" in plan.time_exit_rule


def test_targets_round_half_up_once():
    stop = _stop(10.0, 9.5).model_copy(update={"r_value": Decimal("0.3325")})
    plan = TakeProfitPlanner().plan(
        _candidate(Template.A, resistance=None),
        _entry(10.0),
        stop,
        load_strategy_config(),
    )
    assert [target.price for target in plan.targets] == [
        Decimal("10.67"),
        Decimal("11.00"),
    ]


@pytest.mark.parametrize(
    "r_value",
    [
        Decimal("0"),
        Decimal("-0.5"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_invalid_r_rejected_with_context(r_value):
    stop = _stop(20.0, 18.6).model_copy(update={"r_value": r_value})
    with pytest.raises(PoorRiskRewardError) as excinfo:
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=None),
            _entry(20.0),
            stop,
            load_strategy_config(),
        )
    message = str(excinfo.value)
    assert "000001.SZ" in message
    assert "(A)" in message


def test_non_positive_entry_price_rejected():
    entry = EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=Decimal("0.004"),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=[],
        expires_on=date(2024, 1, 5),
    )
    assert entry.entry_price == Decimal("0.00")
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=None),
            entry,
            _stop(20.0, 18.6),
            load_strategy_config(),
        )


@pytest.mark.parametrize("trigger_price", [Decimal("NaN"), Decimal("Infinity")])
def test_non_finite_entry_price_rejected(trigger_price):
    entry = _entry(20.0).model_copy(
        update={
            "trigger_price": trigger_price,
            "entry_zone_low": None,
            "entry_zone_high": None,
        }
    )
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=None),
            entry,
            _stop(20.0, 18.6),
            load_strategy_config(),
        )


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5])
def test_a_invalid_fraction_rejected(fraction):
    with pytest.raises(PoorRiskRewardError):
        _plan(Template.A, config=_config_a(first_fraction=fraction))


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_b_invalid_fraction_rejected(fraction):
    with pytest.raises(PoorRiskRewardError):
        _plan(
            Template.B,
            entry_price=10.0,
            stop_price=9.5,
            config=_config_b(second_fraction=fraction),
        )


@pytest.mark.parametrize("r_value", [Decimal("1e100"), Decimal("1e999999")])
def test_huge_r_rejected_as_planning_error(r_value):
    stop = _stop(20.0, 18.6).model_copy(update={"r_value": r_value})
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=None),
            _entry(20.0),
            stop,
            load_strategy_config(),
        )


def test_huge_r_overflow_in_resistance_gate_is_planning_error():
    stop = _stop(20.0, 18.6).model_copy(update={"r_value": Decimal("9e999999")})
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=30.0),
            _entry(20.0),
            stop,
            load_strategy_config(),
        )


def test_sub_cent_r_rejected_when_target_not_above_entry():
    stop = _stop(10.0, 9.5).model_copy(update={"r_value": Decimal("0.001")})
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=None),
            _entry(10.0),
            stop,
            load_strategy_config(),
        )


def test_stop_above_entry_rejected():
    stop = _stop(10.0, 9.5).model_copy(update={"initial_stop": price(10.5)})
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=None),
            _entry(10.0),
            stop,
            load_strategy_config(),
        )


@pytest.mark.parametrize(
    "initial_stop", [Decimal("NaN"), Decimal("Infinity")]
)
def test_non_finite_initial_stop_rejected(initial_stop):
    stop = _stop(10.0, 9.5).model_copy(update={"initial_stop": initial_stop})
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=None),
            _entry(10.0),
            stop,
            load_strategy_config(),
        )
