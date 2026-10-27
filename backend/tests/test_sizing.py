from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.core.config import StrategyConfig, load_strategy_config
from app.domain.enums import EntryType, SentimentStage, Template
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
    TemplatePolicy,
)
from app.domain.money import money, price
from app.planning.errors import PositionRejectedError
from app.planning.sizing import PositionSizer

_A_CHECKS = [
    "risk_budget",
    "max_positions",
    "max_a_policy",
    "industry_concentration",
    "duplicate",
    "correlation_unavailable",
    "single_position_cap",
    "cash_reserve",
    "total_exposure",
    "portfolio_heat",
]


def _account(
    equity: float = 1_000_000,
    cash: float = 1_000_000,
    peak: float | None = None,
    positions: list[PositionState] | None = None,
    pending: list[PendingExposure] | None = None,
) -> AccountState:
    return AccountState(
        equity=Decimal(str(equity)),
        cash=Decimal(str(cash)),
        peak_equity=Decimal(str(peak if peak is not None else equity)),
        positions=positions or [],
        pending_plans=pending or [],
    )


def _policy(
    enabled: bool = True, max_positions: int = 3, mode: str = "normal"
) -> TemplatePolicy:
    return TemplatePolicy(enabled=enabled, mode=mode, max_positions=max_positions)


def _regime(
    risk_factor: float = 1.0,
    a: TemplatePolicy | None = None,
    b: TemplatePolicy | None = None,
) -> MarketRegime:
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime="attack",
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=risk_factor,
        template_availability={
            Template.A: a if a is not None else _policy(),
            Template.B: b if b is not None else _policy(),
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


def _candidate(
    template: Template = Template.A,
    sector: str = "银行",
    ts_code: str = "000001.SZ",
) -> Candidate:
    return Candidate(
        ts_code=ts_code,
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector=sector,
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={"close": 10.0},
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


def _stop(entry_price: float, stop_price: float) -> StopPlan:
    return StopPlan(
        initial_stop=price(stop_price),
        stop_distance_pct=(entry_price - stop_price) / entry_price,
        atr14=0.5,
        structure_ref="test",
        time_stop_days=10,
        trailing_rules=[],
        r_value=price(entry_price - stop_price),
    )


def _position(
    ts_code: str = "000009.SZ",
    sector: str = "地产",
    shares: int = 100,
    current_stop: float = 9.0,
    current_price: float = 10.0,
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
    *,
    candidate: Candidate | None = None,
    entry_price: float = 20.0,
    stop_price: float = 18.6,
    account: AccountState | None = None,
    regime: MarketRegime | None = None,
    config: StrategyConfig | None = None,
    correlations: dict[str, float] | None = None,
) -> PositionPlan:
    return PositionSizer().size(
        candidate if candidate is not None else _candidate(),
        _entry(entry_price),
        _stop(entry_price, stop_price),
        account if account is not None else _account(),
        regime if regime is not None else _regime(),
        config if config is not None else load_strategy_config(),
        correlations=correlations,
    )


def _config_with_risk(**updates: object) -> StrategyConfig:
    config = load_strategy_config()
    return config.model_copy(
        update={"risk": config.risk.model_copy(update=updates)}
    )


# --- mandated baseline cases -------------------------------------------------


def test_base_sizing():
    plan = _size()
    assert plan.shares == 17100
    assert float(plan.position_value) == pytest.approx(342000.0)
    assert plan.position_value == money(342000)
    assert plan.position_pct == pytest.approx(0.342)
    assert plan.risk_amount == money(25000)
    assert plan.risk_pct == 0.025
    assert plan.heat_before == 0.0
    assert plan.heat_after == pytest.approx(0.02394, rel=1e-4)
    assert plan.adjustments == []
    assert plan.constraints_checked == _A_CHECKS

    # Whole lots, sized to the risk budget including round-trip costs.
    per_share_risk = Decimal("1.40") + Decimal("20.00") * Decimal("0.003")
    assert per_share_risk * plan.shares <= plan.risk_amount
    assert per_share_risk * (plan.shares + 100) > plan.risk_amount


def test_single_position_cap():
    plan = _size(entry_price=10.0, stop_price=9.5)
    assert plan.shares == 40000
    assert plan.position_pct == pytest.approx(0.40)
    assert plan.position_value == money(400000)
    assert plan.adjustments == ["single_position_cap: 47100 -> 40000"]


def test_heat_cap_reduces_size():
    position = _position(shares=50000, current_stop=9.6, current_price=11.0)
    plan = _size(
        entry_price=10.0,
        stop_price=9.5,
        account=_account(positions=[position]),
    )
    assert plan.shares == 10000
    assert plan.heat_before == pytest.approx(0.07)
    assert plan.heat_after == pytest.approx(0.075)
    assert plan.adjustments == [
        "single_position_cap: 47100 -> 40000",
        "total_exposure: 40000 -> 35000",
        "portfolio_heat: 35000 -> 10000",
    ]


def test_drawdown_and_max_positions_reject():
    with pytest.raises(PositionRejectedError, match="drawdown"):
        _size(account=_account(equity=1_000_000, peak=1_200_000))

    positions = [
        _position(ts_code=f"00000{i}.SZ", sector=f"行业{i}") for i in range(3)
    ]
    with pytest.raises(PositionRejectedError, match="max positions"):
        _size(account=_account(positions=positions))


def test_correlation_above_threshold_rejects():
    position = _position()
    with pytest.raises(PositionRejectedError, match="correlation"):
        _size(
            entry_price=10.0,
            stop_price=9.5,
            account=_account(positions=[position]),
            correlations={"000009.SZ": 0.90},
        )


# --- regime risk factor and drawdown ----------------------------------------


def test_risk_factor_scales_risk():
    plan = _size(regime=_regime(risk_factor=0.5))
    assert plan.risk_amount == money(12500)
    assert plan.shares == 8500
    assert plan.heat_after == pytest.approx(0.0119)


def test_zero_risk_factor_rejects():
    with pytest.raises(PositionRejectedError, match="risk factor"):
        _size(regime=_regime(risk_factor=0.0))


@pytest.mark.parametrize("factor", [1.5, -0.1, float("nan"), float("inf")])
def test_invalid_risk_factor_rejects(factor: float):
    regime = _regime().model_copy(update={"risk_factor": factor})
    with pytest.raises(PositionRejectedError, match="risk factor"):
        _size(regime=regime)


def test_drawdown_boundaries():
    full = _size(account=_account(equity=950_000, cash=950_000, peak=1_000_000))
    half = _size(account=_account(equity=900_000, cash=900_000, peak=1_000_000))
    assert full.shares == 16200
    assert half.shares == 7700
    assert full.risk_amount == money(23750)
    assert half.risk_amount == money(11250)

    with pytest.raises(PositionRejectedError, match="drawdown"):
        _size(account=_account(equity=850_000, cash=850_000, peak=1_000_000))


# --- template policy and global slot caps -----------------------------------


def test_pending_plans_consume_global_slots():
    account = _account(
        positions=[
            _position(ts_code="000009.SZ", sector="地产"),
            _position(ts_code="000010.SZ", sector="科技"),
        ],
        pending=[_pending(ts_code="000011.SZ", sector="医药")],
    )
    with pytest.raises(PositionRejectedError, match="max positions"):
        _size(account=account)


def test_pending_plans_consume_template_slots():
    account = _account(
        positions=[_position(ts_code="000009.SZ", sector="地产")],
        pending=[_pending(ts_code="000011.SZ", sector="医药")],
    )
    with pytest.raises(PositionRejectedError, match="max A policy"):
        _size(account=account, regime=_regime(a=_policy(max_positions=2)))


def test_pending_plans_consume_b_policy_slots():
    account = _account(pending=[_pending(template=Template.B)])
    with pytest.raises(PositionRejectedError, match="max B policy"):
        _size(
            candidate=_candidate(template=Template.B),
            entry_price=10.0,
            stop_price=9.5,
            account=account,
            regime=_regime(b=_policy(max_positions=1)),
        )


def test_a_policy_cap_rejects_third_a():
    account = _account(
        positions=[
            _position(ts_code="000009.SZ", sector="地产"),
            _position(ts_code="000010.SZ", sector="科技"),
        ]
    )
    with pytest.raises(PositionRejectedError, match="max A policy"):
        _size(account=account, regime=_regime(a=_policy(max_positions=2)))


def test_b_policy_cap_overrides_config_max():
    b_candidate = _candidate(template=Template.B)
    # Without a held B position the config/max policy combination permits entry.
    plan = _size(candidate=b_candidate, entry_price=10.0, stop_price=9.5)
    assert plan.shares == 37700
    assert "max_b_positions" in plan.constraints_checked
    assert "max_b_policy" in plan.constraints_checked

    account = _account(
        positions=[
            _position(ts_code="000009.SZ", sector="地产", template=Template.B)
        ]
    )
    # config.risk.max_b_positions is 2, but the ferment regime policy caps B at 1.
    with pytest.raises(PositionRejectedError, match="max B policy"):
        _size(
            candidate=b_candidate,
            entry_price=10.0,
            stop_price=9.5,
            account=account,
            regime=_regime(b=_policy(max_positions=1)),
        )


def test_b_config_cap_rejects_third_b():
    account = _account(
        positions=[
            _position(ts_code="000009.SZ", sector="地产", template=Template.B),
            _position(ts_code="000010.SZ", sector="科技", template=Template.B),
        ]
    )
    with pytest.raises(PositionRejectedError, match="max B positions"):
        _size(
            candidate=_candidate(template=Template.B),
            entry_price=10.0,
            stop_price=9.5,
            account=account,
            regime=_regime(b=_policy(max_positions=3)),
        )


def test_b_policy_zero_rejects_any_b():
    with pytest.raises(PositionRejectedError, match="max B policy"):
        _size(
            candidate=_candidate(template=Template.B),
            entry_price=10.0,
            stop_price=9.5,
            regime=_regime(b=_policy(max_positions=0)),
        )


def test_legacy_over_cap_holding_does_not_block_other_template():
    # Both A holdings exceed the A policy cap (a legacy over-cap snapshot),
    # but global config slots remain and the B template has capacity.
    account = _account(
        positions=[
            _position(ts_code="000009.SZ", sector="地产", template=Template.A),
            _position(ts_code="000010.SZ", sector="科技", template=Template.A),
        ]
    )
    regime = _regime(a=_policy(max_positions=1), b=_policy(max_positions=1))
    plan = _size(
        candidate=_candidate(template=Template.B),
        entry_price=10.0,
        stop_price=9.5,
        account=account,
        regime=regime,
    )
    assert plan.shares == 37700
    assert plan.adjustments == []


def test_disabled_template_rejects():
    with pytest.raises(PositionRejectedError, match="disabled"):
        _size(regime=_regime(a=_policy(enabled=False, max_positions=0)))


def test_missing_template_policy_rejects():
    regime = _regime().model_copy(
        update={"template_availability": {Template.A: _policy()}}
    )
    with pytest.raises(PositionRejectedError, match="policy"):
        _size(regime=regime)


def test_negative_policy_max_rejects():
    bad = _policy().model_copy(update={"max_positions": -1})
    regime = _regime().model_copy(
        update={
            "template_availability": {Template.A: bad, Template.B: _policy()}
        }
    )
    with pytest.raises(PositionRejectedError, match="policy"):
        _size(regime=regime)


# --- industry, duplicate, correlation ---------------------------------------


def test_industry_cap_includes_pending():
    account = _account(
        positions=[_position(ts_code="000009.SZ", sector="银行")],
        pending=[_pending(ts_code="000011.SZ", sector="银行")],
    )
    with pytest.raises(PositionRejectedError, match="industry"):
        _size(account=account)


def test_industry_cap_allows_one_existing():
    account = _account(positions=[_position(ts_code="000009.SZ", sector="银行")])
    assert _size(account=account).shares == 17100


def test_duplicate_position_rejects():
    account = _account(positions=[_position(ts_code="000001.SZ", sector="银行")])
    with pytest.raises(PositionRejectedError, match="duplicate"):
        _size(account=account)


def test_duplicate_pending_rejects():
    account = _account(pending=[_pending(ts_code="000001.SZ", sector="银行")])
    with pytest.raises(PositionRejectedError, match="duplicate"):
        _size(account=account)


def test_correlation_threshold_is_exclusive():
    account = _account(positions=[_position()])
    plan = _size(account=account, correlations={"000009.SZ": 0.80})
    assert plan.shares == 17100


def test_correlation_ignores_unheld_codes():
    plan = _size(correlations={"999999.SZ": 0.99})
    assert plan.shares == 17100
    assert "correlation" in plan.constraints_checked
    assert "correlation_unavailable" not in plan.constraints_checked


def test_correlation_marker_when_not_supplied():
    plan = _size()
    assert "correlation_unavailable" in plan.constraints_checked
    assert "correlation" not in plan.constraints_checked


def test_correlation_includes_pending():
    account = _account(pending=[_pending()])
    with pytest.raises(PositionRejectedError, match="correlation"):
        _size(account=account, correlations={"000008.SZ": 0.81})


@pytest.mark.parametrize("correlation", [float("nan"), 1.5, -1.5])
def test_invalid_correlation_rejects(correlation: float):
    account = _account(positions=[_position()])
    with pytest.raises(PositionRejectedError, match="correlation"):
        _size(account=account, correlations={"000009.SZ": correlation})


# --- cash, exposure, heat caps ----------------------------------------------


def test_cash_reserve_reduces_size():
    plan = _size(account=_account(cash=200_000))
    assert plan.shares == 5000
    assert plan.position_value == money(100_000)
    assert plan.position_pct == pytest.approx(0.10)
    assert plan.adjustments == ["cash_reserve: 17100 -> 5000"]


def test_cash_reserve_includes_pending():
    account = _account(
        cash=200_000,
        pending=[
            _pending(entry_price=10.0, stop_price=9.9, shares=5_000)
        ],
    )
    plan = _size(entry_price=10.0, stop_price=9.5, account=account)
    assert plan.shares == 5000
    assert plan.adjustments == [
        "single_position_cap: 47100 -> 40000",
        "cash_reserve: 40000 -> 5000",
    ]


def test_cash_reserve_breach_rejects():
    with pytest.raises(PositionRejectedError, match="cash reserve"):
        _size(
            entry_price=10.0,
            stop_price=9.5,
            account=_account(cash=100_000),
        )


def test_total_exposure_reduces_size():
    position = _position(
        ts_code="000009.SZ",
        sector="地产",
        shares=80_000,
        current_stop=7.5,
        current_price=8.0,
    )
    plan = _size(account=_account(positions=[position]))
    assert plan.shares == 13000
    assert plan.position_value == money(260_000)
    assert plan.position_pct == pytest.approx(0.26)
    # positions 640k + new 260k = 900k = 90% of equity (the exact cap).
    assert float(plan.position_value) + 640_000 == pytest.approx(900_000)
    assert plan.adjustments == ["total_exposure: 17100 -> 13000"]


def test_exposure_guard_fires_for_inconsistent_snapshot():
    # cash + positions exceed equity, so the cash check passes while the
    # portfolio is already above the exposure cap.
    account = _account(
        cash=950_000,
        positions=[
            _position(
                ts_code="000009.SZ",
                sector="地产",
                shares=50_000,
                current_stop=9.0,
                current_price=10.0,
            )
        ],
        pending=[
            _pending(entry_price=10.0, stop_price=9.9, shares=45_000)
        ],
    )
    with pytest.raises(PositionRejectedError, match="exposure"):
        _size(entry_price=10.0, stop_price=9.5, account=account)


def test_total_exposure_includes_pending():
    account = _account(
        cash=1_500_000,
        pending=[
            _pending(entry_price=10.0, stop_price=9.9, shares=80_000)
        ],
    )
    plan = _size(entry_price=10.0, stop_price=8.0, account=account)
    assert plan.shares == 10000
    assert plan.position_value == money(100_000)
    assert plan.adjustments == ["total_exposure: 12300 -> 10000"]


def test_heat_only_adjustment_attribution():
    # A small high-risk legacy position leaves only the heat budget binding.
    position = _position(
        shares=516, current_stop=1.0, current_price=100.0
    )
    plan = _size(account=_account(positions=[position]))
    assert plan.shares == 17000
    assert plan.adjustments == ["portfolio_heat: 17100 -> 17000"]


def test_heat_cap_includes_pending():
    account = _account(
        pending=[
            _pending(entry_price=10.0, stop_price=8.0, shares=30_000)
        ]
    )
    plan = _size(entry_price=10.0, stop_price=9.5, account=account)
    assert plan.heat_before == pytest.approx(0.06)
    assert plan.shares == 30000
    assert plan.heat_after == pytest.approx(0.075)
    assert plan.adjustments == [
        "single_position_cap: 47100 -> 40000",
        "portfolio_heat: 40000 -> 30000",
    ]


def test_heat_at_cap_rejects():
    position = _position(shares=75_000, current_stop=9.0, current_price=10.0)
    with pytest.raises(PositionRejectedError, match="heat"):
        _size(account=_account(positions=[position]))


# --- lot sizing and accounting ----------------------------------------------


def test_share_rounding_to_whole_lots():
    plan = _size(entry_price=20.0, stop_price=18.0)
    assert plan.shares == 12100
    assert plan.shares % 100 == 0


def test_below_one_lot_rejects():
    with pytest.raises(PositionRejectedError, match="one lot"):
        _size(account=_account(equity=1_000, cash=1_000))


def test_risk_amount_and_heat_accounting():
    plan = _size(entry_price=20.0, stop_price=18.0)
    assert plan.risk_amount == money(Decimal("1000000") * Decimal("0.025"))
    assert plan.heat_after == pytest.approx(
        plan.heat_before + plan.shares * 2.0 / 1_000_000
    )


def test_b_risk_budget():
    plan = _size(
        candidate=_candidate(template=Template.B),
        entry_price=10.0,
        stop_price=9.5,
    )
    assert plan.template == Template.B
    assert plan.risk_pct == 0.02
    assert plan.risk_amount == money(20000)
    assert plan.shares == 37700


def test_string_template_is_treated_as_a():
    candidate = _candidate().model_copy(update={"template": "A"})
    plan = _size(candidate=candidate)
    assert plan.template == Template.A
    assert plan.risk_pct == 0.025
    assert plan.shares == 17100


# --- input and config validation --------------------------------------------


@pytest.mark.parametrize(
    "update",
    [
        {"equity": Decimal("0")},
        {"equity": Decimal("-1")},
        {"equity": Decimal("Infinity")},
        {"cash": Decimal("-1")},
        {"cash": Decimal("Infinity")},
        {"peak_equity": Decimal("0")},
        {"peak_equity": Decimal("Infinity")},
    ],
)
def test_invalid_account_rejects(update: dict[str, Decimal]):
    account = _account().model_copy(update=update)
    with pytest.raises(PositionRejectedError, match="account"):
        _size(account=account)


@pytest.mark.parametrize("stop_price", [20.0, 25.0])
def test_stop_not_below_entry_rejects(stop_price: float):
    entry = _entry(20.0)
    bad_stop = _stop(20.0, 18.6).model_copy(
        update={"initial_stop": price(stop_price)}
    )
    with pytest.raises(PositionRejectedError, match="stop"):
        PositionSizer().size(
            _candidate(),
            entry,
            bad_stop,
            _account(),
            _regime(),
            load_strategy_config(),
        )


def test_non_finite_stop_rejects():
    entry = _entry(20.0)
    bad_stop = _stop(20.0, 18.6).model_copy(
        update={"initial_stop": Decimal("Infinity")}
    )
    with pytest.raises(PositionRejectedError, match="stop"):
        PositionSizer().size(
            _candidate(),
            entry,
            bad_stop,
            _account(),
            _regime(),
            load_strategy_config(),
        )


@pytest.mark.parametrize("entry_price", [0.0, -1.0])
def test_non_positive_entry_rejects(entry_price: float):
    entry = _entry(20.0).model_copy(
        update={
            "entry_zone_low": Decimal(str(entry_price)),
            "entry_zone_high": Decimal(str(entry_price)),
        }
    )
    with pytest.raises(PositionRejectedError, match="entry price"):
        PositionSizer().size(
            _candidate(),
            entry,
            _stop(20.0, 18.6),
            _account(),
            _regime(),
            load_strategy_config(),
        )


def test_non_finite_entry_rejects():
    entry = _entry(20.0).model_copy(
        update={
            "entry_zone_low": Decimal("Infinity"),
            "entry_zone_high": Decimal("Infinity"),
        }
    )
    with pytest.raises(PositionRejectedError, match="entry price"):
        PositionSizer().size(
            _candidate(),
            entry,
            _stop(20.0, 18.6),
            _account(),
            _regime(),
            load_strategy_config(),
        )


@pytest.mark.parametrize(
    "risk_update",
    [
        {"max_positions": -1},
        {"max_positions": 0},
        {"max_b_positions": -1},
        {"max_single_position_pct": 0.0},
        {"max_heat": 0.0},
        {"max_heat": 1.5},
        {"min_cash_pct": -0.1},
        {"min_cash_pct": 1.5},
        {"max_industry_positions": -1},
        {"correlation_threshold": 1.5},
        {"round_trip_cost_pct": -0.01},
        {"drawdown_half_at": 0.15, "drawdown_stop_at": 0.10},
    ],
)
def test_invalid_risk_config_rejects(risk_update: dict[str, object]):
    config = _config_with_risk(**risk_update)
    with pytest.raises(PositionRejectedError, match="config"):
        _size(config=config)


def test_invalid_template_risk_pct_rejects():
    config = load_strategy_config()
    bad = config.model_copy(
        update={
            "template_a": config.template_a.model_copy(update={"risk_pct": 0.0})
        }
    )
    with pytest.raises(PositionRejectedError, match="config"):
        _size(config=bad)
