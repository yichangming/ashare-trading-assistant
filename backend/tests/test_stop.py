from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.core.config import StrategyConfig, load_strategy_config
from app.domain.enums import EntryType, SentimentStage, Template
from app.domain.models import Candidate, EntryPlan, StopPlan
from app.domain.money import price
from app.planning.errors import StopTooWideError
from app.planning.stop import StopPlanner

_A_TRAILING_RULES = [
    "move to breakeven at +1R",
    "move to +1R at +2R",
    "trail by close - 1.5*ATR14",
]
_B_TRAILING_RULES = [
    "move to breakeven at +1R",
    "move to +1R at +2R",
    "trail by close - 1.5*ATR14 and MA5",
]


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
    template: Template,
    features: dict,
    *,
    entry_price: float = 10.0,
    config: StrategyConfig | None = None,
) -> StopPlan:
    candidate = _candidate(template, features)
    return StopPlanner().plan(
        candidate,
        _entry(entry_price),
        candidate.features,
        config if config is not None else load_strategy_config(),
    )


def _config_a(**updates: object) -> StrategyConfig:
    config = load_strategy_config()
    return config.model_copy(
        update={"template_a": config.template_a.model_copy(update=updates)}
    )


def _config_b(**updates: object) -> StrategyConfig:
    config = load_strategy_config()
    return config.model_copy(
        update={"template_b": config.template_b.model_copy(update=updates)}
    )


def test_a_structure_stop():
    config = load_strategy_config()
    candidate = _candidate(
        Template.A,
        {
            "low_10d_raw": 9.5,
            "ma20_raw": 10.5,
            "atr14_raw": 0.5,
        },
    )
    stop = StopPlanner().plan(candidate, _entry(), candidate.features, config)
    assert float(stop.initial_stop) == 9.25
    assert stop.stop_distance_pct == pytest.approx(0.075)
    assert float(stop.r_value) == 0.75


def test_a_too_wide_rejected():
    config = load_strategy_config()
    candidate = _candidate(
        Template.A,
        {
            "low_10d_raw": 8.0,
            "ma20_raw": 10.5,
            "atr14_raw": 0.5,
        },
    )
    with pytest.raises(StopTooWideError):
        StopPlanner().plan(candidate, _entry(), candidate.features, config)


def test_b_clamps_stop_distance():
    config = load_strategy_config()
    wide = _candidate(
        Template.B,
        {
            "low_5d_raw": 9.0,
            "ma5_raw": 9.0,
            "ma10_raw": 8.9,
            "atr14_raw": 0.3,
        },
    )
    stop = StopPlanner().plan(wide, _entry(), wide.features, config)
    assert float(stop.initial_stop) == 9.50

    tight = _candidate(
        Template.B,
        {
            "low_5d_raw": 9.95,
            "ma5_raw": 9.98,
            "ma10_raw": 9.96,
            "atr14_raw": 0.05,
        },
    )
    stop2 = StopPlanner().plan(tight, _entry(), tight.features, config)
    assert float(stop2.initial_stop) == 9.70


def test_a_atr_fallback_without_support():
    stop = _plan(
        Template.A,
        {"low_10d_raw": 10.5, "ma20_raw": 10.8, "atr14_raw": 0.5},
    )
    assert stop.structure_ref == "ATR fallback"
    assert float(stop.initial_stop) == 9.25
    assert float(stop.r_value) == 0.75
    assert stop.stop_distance_pct == pytest.approx(0.075)
    assert stop.atr14 == pytest.approx(0.5)


def test_a_structure_stop_at_or_above_entry_uses_atr_fallback():
    # A negative buffer makes support - buffer * ATR exceed the entry price.
    # Config validation forbids this, so bypass it with model_copy to cover
    # the defensive fallback branch.
    config = _config_a(atr_buffer=-1.0)
    stop = _plan(
        Template.A,
        {"low_10d_raw": 9.9, "ma20_raw": 10.2, "atr14_raw": 0.5},
        config=config,
    )
    assert stop.structure_ref == "ATR fallback"
    assert float(stop.initial_stop) == 9.25
    assert float(stop.r_value) == 0.75


def test_a_min_stop_distance_is_widened():
    stop = _plan(
        Template.A,
        {"low_10d_raw": 9.98, "ma20_raw": 10.5, "atr14_raw": 0.04},
    )
    assert float(stop.initial_stop) == 9.70
    assert stop.structure_ref == "support=low_10d_raw:9.98; widened to min stop"
    assert stop.stop_distance_pct == pytest.approx(0.03)


def test_a_exact_min_boundary_is_kept():
    # (10.00 - (9.80 - 0.5 * 0.20)) / 10.00 == 3%.
    stop = _plan(
        Template.A,
        {"low_10d_raw": 9.8, "ma20_raw": 10.5, "atr14_raw": 0.2},
    )
    assert float(stop.initial_stop) == 9.70
    assert stop.structure_ref == "support=low_10d_raw:9.8"


def test_a_exact_max_boundary_is_allowed():
    # (10.00 - (9.40 - 0.5 * 0.40)) / 10.00 == 8%.
    stop = _plan(
        Template.A,
        {"low_10d_raw": 9.4, "ma20_raw": 10.5, "atr14_raw": 0.4},
    )
    assert float(stop.initial_stop) == 9.20
    assert stop.stop_distance_pct == pytest.approx(0.08)
    assert stop.structure_ref == "support=low_10d_raw:9.4"


def test_a_multiple_below_entry_supports_choose_highest():
    # Both levels are eligible; MA20 at 9.60 is the higher one.
    stop = _plan(
        Template.A,
        {"low_10d_raw": 9.2, "ma20_raw": 9.6, "atr14_raw": 0.2},
    )
    assert stop.structure_ref == "support=ma20_raw:9.6"
    assert float(stop.initial_stop) == 9.50
    assert stop.stop_distance_pct == pytest.approx(0.05)


def test_a_just_over_max_is_rejected_with_context():
    candidate = _candidate(
        Template.A,
        {"low_10d_raw": 9.39, "ma20_raw": 10.5, "atr14_raw": 0.4},
    )
    with pytest.raises(StopTooWideError) as excinfo:
        StopPlanner().plan(
            candidate, _entry(), candidate.features, load_strategy_config()
        )
    message = str(excinfo.value)
    assert candidate.ts_code in message
    assert "(A)" in message
    assert "8.10%" in message
    assert "8.00%" in message


def test_b_exact_max_boundary_is_kept():
    # (10.00 - (9.60 - 0.5 * 0.20)) / 10.00 == 5%, so no clamp is applied.
    stop = _plan(
        Template.B,
        {"low_5d_raw": 9.6, "ma5_raw": 10.5, "ma10_raw": 10.4, "atr14_raw": 0.2},
    )
    assert float(stop.initial_stop) == 9.50
    assert stop.structure_ref == "support=low_5d_raw:9.6"
    assert stop.stop_distance_pct == pytest.approx(0.05)


def test_b_wide_stop_is_clamped_to_max():
    stop = _plan(
        Template.B,
        {"low_5d_raw": 9.8, "ma5_raw": 10.6, "ma10_raw": 10.7, "atr14_raw": 0.6},
        entry_price=10.5,
    )
    # 10.50 * (1 - 0.05) == 9.975, which rounds half-up to 9.98.
    assert float(stop.initial_stop) == 9.98
    assert stop.structure_ref == "support=low_5d_raw:9.8; clamped to max stop"
    assert float(stop.r_value) == 0.52
    assert stop.stop_distance_pct == pytest.approx(0.52 / 10.5)


def test_b_exact_min_boundary_is_kept_and_highest_support_is_used():
    # Supports 9.70, 9.80, 9.90 are all below entry; 9.90 is the highest.
    stop = _plan(
        Template.B,
        {"low_5d_raw": 9.7, "ma5_raw": 9.8, "ma10_raw": 9.9, "atr14_raw": 0.4},
    )
    assert float(stop.initial_stop) == 9.70
    assert stop.structure_ref == "support=ma10_raw:9.9"
    assert stop.stop_distance_pct == pytest.approx(0.03)


def test_b_min_stop_distance_is_widened():
    stop = _plan(
        Template.B,
        {"low_5d_raw": 9.99, "ma5_raw": 10.5, "ma10_raw": 10.4, "atr14_raw": 0.02},
    )
    assert float(stop.initial_stop) == 9.70
    assert stop.structure_ref == "support=low_5d_raw:9.99; widened to min stop"
    assert stop.stop_distance_pct == pytest.approx(0.03)


def test_b_atr_fallback_without_support():
    stop = _plan(
        Template.B,
        {"low_5d_raw": 10.2, "ma5_raw": 10.5, "ma10_raw": 10.4, "atr14_raw": 0.3},
    )
    assert stop.structure_ref == "ATR fallback"
    assert float(stop.initial_stop) == 9.55
    assert float(stop.r_value) == 0.45
    assert stop.stop_distance_pct == pytest.approx(0.045)


def test_fallback_is_widened_for_a():
    stop = _plan(Template.A, {"atr14_raw": 0.1})
    assert stop.structure_ref == "ATR fallback; widened to min stop"
    assert float(stop.initial_stop) == 9.70
    assert stop.stop_distance_pct == pytest.approx(0.03)


def test_fallback_is_rejected_for_a():
    candidate = _candidate(Template.A, {"atr14_raw": 1.0})
    with pytest.raises(StopTooWideError, match="A stop distance"):
        StopPlanner().plan(
            candidate, _entry(), candidate.features, load_strategy_config()
        )


def test_fallback_is_clamped_for_b():
    stop = _plan(Template.B, {"atr14_raw": 0.5})
    assert stop.structure_ref == "ATR fallback; clamped to max stop"
    assert float(stop.initial_stop) == 9.50
    assert stop.stop_distance_pct == pytest.approx(0.05)


def test_b_atr_buffer_comes_from_config():
    features = {
        "low_5d_raw": 9.8,
        "ma5_raw": 10.5,
        "ma10_raw": 10.4,
        "atr14_raw": 0.2,
    }
    assert float(_plan(Template.B, features).initial_stop) == 9.70
    wide_stop = _plan(Template.B, features, config=_config_b(atr_buffer=1.0))
    assert float(wide_stop.initial_stop) == 9.60


def test_decimal_half_cent_structure_stop_rounds_half_up():
    # In binary floats 10.94 - 0.5 * 0.01 is 10.934999999999999, which would
    # round to 10.93. Decimal arithmetic keeps the exact 10.935 half-cent.
    stop = _plan(
        Template.B,
        {"low_5d_raw": 10.94, "ma5_raw": 11.5, "ma10_raw": 11.6, "atr14_raw": 0.01},
        entry_price=11.3,
    )
    assert stop.initial_stop == Decimal("10.94")
    assert stop.structure_ref == "support=low_5d_raw:10.94"
    assert float(stop.r_value) == 0.36
    assert stop.stop_distance_pct == pytest.approx(0.36 / 11.3)


@pytest.mark.parametrize(
    ("template", "features", "entry_price", "expected_stop", "nudge"),
    [
        (
            Template.A,
            {"low_10d_raw": 1.01, "ma20_raw": 1.2, "atr14_raw": 0.001},
            1.03,
            0.99,
            "nudged down to min stop",
        ),
        (
            Template.B,
            {
                "low_5d_raw": 1.01,
                "ma5_raw": 1.2,
                "ma10_raw": 1.3,
                "atr14_raw": 0.001,
            },
            1.03,
            0.99,
            "nudged down to min stop",
        ),
        (
            Template.A,
            {"low_10d_raw": 0.985, "ma20_raw": 1.2, "atr14_raw": 0.001},
            1.07,
            0.99,
            "nudged up to max stop",
        ),
        (
            Template.B,
            {
                "low_5d_raw": 1.0551,
                "ma5_raw": 1.2,
                "ma10_raw": 1.3,
                "atr14_raw": 0.001,
            },
            1.11,
            1.06,
            "nudged up to max stop",
        ),
    ],
)
def test_final_stop_is_nudged_into_band(
    template: Template,
    features: dict,
    entry_price: float,
    expected_stop: float,
    nudge: str,
) -> None:
    stop = _plan(template, features, entry_price=entry_price)
    assert float(stop.initial_stop) == expected_stop
    assert nudge in stop.structure_ref
    high = 0.08 if template == Template.A else 0.05
    assert 0.03 <= stop.stop_distance_pct <= high
    assert stop.stop_distance_pct == pytest.approx(
        (entry_price - expected_stop) / entry_price
    )


@pytest.mark.parametrize("template", [Template.A, Template.B])
def test_impossible_cent_grid_stop_is_rejected(template: Template) -> None:
    # At an entry of 0.02 no cent-aligned stop sits 3%-8%/3%-5% below: 0.02
    # is 0% away and 0.01 is 50% away.
    candidate = _candidate(template, {"atr14_raw": 0.001})
    with pytest.raises(StopTooWideError, match="outside"):
        StopPlanner().plan(
            candidate, _entry(0.02), candidate.features, load_strategy_config()
        )


def test_support_must_be_strictly_below_entry():
    # low_10d_raw equals the entry price, so it is not a support.
    stop = _plan(
        Template.A,
        {"low_10d_raw": 10.0, "ma20_raw": 10.5, "atr14_raw": 0.5},
    )
    assert stop.structure_ref == "ATR fallback"
    assert float(stop.initial_stop) == 9.25


@pytest.mark.parametrize("template", [Template.A, Template.B])
@pytest.mark.parametrize("support", [9.9999, 10.0])
def test_support_rounding_to_entry_is_excluded(
    template: Template, support: float
) -> None:
    # 9.9999 is below entry in raw terms but rounds to 10.00 on the price
    # grid, so it is not a tradable support, just like an exact 10.00 level.
    if template == Template.A:
        features = {"low_10d_raw": support, "ma20_raw": 10.5, "atr14_raw": 0.5}
        expected_stop = 9.25
    else:
        features = {
            "low_5d_raw": support,
            "ma5_raw": 10.5,
            "ma10_raw": 10.6,
            "atr14_raw": 0.3,
        }
        expected_stop = 9.55
    stop = _plan(template, features)
    assert stop.structure_ref == "ATR fallback"
    assert float(stop.initial_stop) == expected_stop


@pytest.mark.parametrize("template", [Template.A, Template.B])
def test_zero_and_negative_supports_fall_back(template: Template) -> None:
    if template == Template.A:
        features = {"low_10d_raw": 0.0, "ma20_raw": -1.0, "atr14_raw": 0.5}
        expected_stop = 9.25
    else:
        features = {
            "low_5d_raw": 0.0,
            "ma5_raw": -1.0,
            "ma10_raw": -0.5,
            "atr14_raw": 0.3,
        }
        expected_stop = 9.55
    stop = _plan(template, features)
    assert stop.structure_ref == "ATR fallback"
    assert float(stop.initial_stop) == expected_stop
    assert stop.initial_stop > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("template", [Template.A, Template.B])
def test_non_finite_supports_are_ignored(
    template: Template, bad: float
) -> None:
    features = (
        {"low_10d_raw": bad, "ma20_raw": bad, "atr14_raw": 0.5}
        if template == Template.A
        else {"low_5d_raw": bad, "ma5_raw": 10.5, "ma10_raw": 10.4, "atr14_raw": 0.3}
    )
    stop = _plan(template, features)
    assert stop.structure_ref == "ATR fallback"
    expected = 9.25 if template == Template.A else 9.55
    assert float(stop.initial_stop) == expected


@pytest.mark.parametrize(
    ("template", "features", "expected_stop"),
    [
        (
            Template.A,
            {"low_10d_raw": None, "ma20_raw": None, "atr14_raw": 0.5},
            9.25,
        ),
        (
            Template.B,
            {
                "low_5d_raw": None,
                "ma5_raw": None,
                "ma10_raw": None,
                "atr14_raw": 0.3,
            },
            9.55,
        ),
    ],
)
def test_null_supports_are_ignored(
    template: Template, features: dict, expected_stop: float
) -> None:
    candidate = _candidate(template, {"atr14_raw": features["atr14_raw"]})
    stop = StopPlanner().plan(
        candidate, _entry(), features, load_strategy_config()
    )
    assert stop.structure_ref == "ATR fallback"
    assert float(stop.initial_stop) == expected_stop
    assert float(stop.initial_stop) > 0


@pytest.mark.parametrize("template", [Template.A, Template.B])
def test_missing_atr_is_rejected_with_context(template: Template) -> None:
    support = (
        {"low_10d_raw": 9.5, "ma20_raw": 10.5}
        if template == Template.A
        else {"low_5d_raw": 9.8, "ma5_raw": 10.2, "ma10_raw": 10.1}
    )
    candidate = _candidate(template, support)
    with pytest.raises(StopTooWideError) as excinfo:
        StopPlanner().plan(
            candidate, _entry(), candidate.features, load_strategy_config()
        )
    message = str(excinfo.value)
    assert candidate.ts_code in message
    assert f"({template.value})" in message
    assert "ATR" in message


@pytest.mark.parametrize("atr", [0.0, -0.5, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("template", [Template.A, Template.B])
def test_invalid_atr_is_rejected_with_context(
    template: Template, atr: float
) -> None:
    support = (
        {"low_10d_raw": 9.5, "ma20_raw": 10.5}
        if template == Template.A
        else {"low_5d_raw": 9.8, "ma5_raw": 10.2, "ma10_raw": 10.1}
    )
    candidate = _candidate(template, {**support, "atr14_raw": atr})
    with pytest.raises(StopTooWideError) as excinfo:
        StopPlanner().plan(
            candidate, _entry(), candidate.features, load_strategy_config()
        )
    message = str(excinfo.value)
    assert candidate.ts_code in message
    assert f"({template.value})" in message
    assert "ATR" in message


def test_invalid_stop_bounds_are_rejected_with_context():
    config = _config_a(stop_min_pct=0.09, stop_max_pct=0.08)
    candidate = _candidate(
        Template.A,
        {"low_10d_raw": 9.5, "ma20_raw": 10.5, "atr14_raw": 0.5},
    )
    with pytest.raises(StopTooWideError) as excinfo:
        StopPlanner().plan(candidate, _entry(), candidate.features, config)
    message = str(excinfo.value)
    assert candidate.ts_code in message
    assert "(A)" in message
    assert "stop" in message.lower()


def test_non_positive_final_stop_is_rejected():
    config = _config_b(stop_max_pct=1.0)
    candidate = _candidate(
        Template.B,
        {"low_5d_raw": 9.0, "ma5_raw": 10.5, "ma10_raw": 10.4, "atr14_raw": 40.0},
    )
    with pytest.raises(StopTooWideError, match="final stop must be positive"):
        StopPlanner().plan(candidate, _entry(), candidate.features, config)


def test_non_positive_r_is_rejected():
    # The pre-round stop is 0.09; rounding it from 0.093 puts the distance at
    # 10%, so it is nudged up to the entry price and R becomes zero.
    candidate = _candidate(
        Template.A,
        {"low_10d_raw": 0.0935, "ma20_raw": 0.12, "atr14_raw": 0.001},
    )
    with pytest.raises(StopTooWideError, match="non-positive R"):
        StopPlanner().plan(
            candidate, _entry(0.10), candidate.features, load_strategy_config()
        )


@pytest.mark.parametrize("template", [Template.A, Template.B])
def test_non_positive_time_stop_days_is_rejected_by_planner(
    template: Template,
) -> None:
    # model_copy bypasses config validation, so the planner must guard too.
    if template == Template.A:
        config = _config_a(time_stop_days=0)
        features = {"low_10d_raw": 9.5, "ma20_raw": 10.5, "atr14_raw": 0.5}
    else:
        config = _config_b(time_stop_days=0)
        features = {
            "low_5d_raw": 9.8,
            "ma5_raw": 10.2,
            "ma10_raw": 10.1,
            "atr14_raw": 0.3,
        }
    candidate = _candidate(template, features)
    with pytest.raises(StopTooWideError) as excinfo:
        StopPlanner().plan(candidate, _entry(), candidate.features, config)
    message = str(excinfo.value)
    assert candidate.ts_code in message
    assert f"({template.value})" in message
    assert "time stop" in message


def test_time_stop_and_trailing_rules_follow_template():
    config = load_strategy_config()
    a_stop = _plan(
        Template.A,
        {"low_10d_raw": 9.5, "ma20_raw": 10.5, "atr14_raw": 0.5},
        config=config,
    )
    assert a_stop.time_stop_days == config.template_a.time_stop_days == 10
    assert a_stop.trailing_rules == _A_TRAILING_RULES

    b_stop = _plan(
        Template.B,
        {"low_5d_raw": 10.2, "ma5_raw": 10.5, "ma10_raw": 10.4, "atr14_raw": 0.3},
        config=config,
    )
    assert b_stop.time_stop_days == config.template_b.time_stop_days == 3
    assert b_stop.trailing_rules == _B_TRAILING_RULES


def test_time_stop_days_come_from_config():
    stop = _plan(
        Template.A,
        {"low_10d_raw": 9.5, "ma20_raw": 10.5, "atr14_raw": 0.5},
        config=_config_a(time_stop_days=7),
    )
    assert stop.time_stop_days == 7


@pytest.mark.parametrize(
    ("template", "features"),
    [
        (Template.A, {"low_10d_raw": 9.5, "ma20_raw": 10.5, "atr14_raw": 0.5}),
        (
            Template.B,
            {"low_5d_raw": 9.8, "ma5_raw": 10.4, "ma10_raw": 10.3, "atr14_raw": 0.6},
        ),
    ],
)
def test_r_value_matches_entry_minus_stop(template: Template, features: dict) -> None:
    stop = _plan(template, features)
    entry_price = float(_entry().entry_price)
    assert float(stop.r_value) == pytest.approx(
        entry_price - float(stop.initial_stop)
    )
    assert stop.stop_distance_pct == pytest.approx(
        float(stop.r_value) / entry_price
    )
