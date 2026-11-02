from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl

from app.backtest.engine import BacktestEngine
from app.backtest.simulator import Simulator
from app.core.full_config import load_full_config
from app.domain.enums import (
    EntryType,
    PlanStatus,
    SentimentStage,
    Template,
)
from app.domain.models import (
    Candidate,
    EntryPlan,
    PositionPlan,
    StopPlan,
    TakeProfitPlan,
    TakeProfitTarget,
    TradePlan,
)
from app.domain.money import money, price


def _plan(day: date = date(2024, 1, 2), shares: int = 1000) -> TradePlan:
    candidate = Candidate(
        ts_code="000001.SZ",
        trade_date=day,
        template=Template.A,
        setup_type="test",
        horizon="mid",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={"close": 10.0, "atr14_raw": 0.5},
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv",
        data_version="dv",
    )
    entry = EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=price(10.0),
        validity_days=2,
        max_chase_pct=0.01,
        cancel_conditions=[],
        expires_on=day + timedelta(days=2),
    )
    stop = StopPlan(
        initial_stop=price(9.5),
        stop_distance_pct=0.05,
        atr14=0.5,
        structure_ref="test",
        time_stop_days=10,
        trailing_rules=[],
        r_value=price(0.5),
    )
    position = PositionPlan(
        template=Template.A,
        risk_pct=0.025,
        risk_amount=money(25000),
        shares=shares,
        position_value=money(10 * shares),
        position_pct=0.025,
        heat_before=0.0,
        heat_after=0.0025,
        constraints_checked=[],
        adjustments=[],
    )
    take_profit = TakeProfitPlan(
        targets=[
            TakeProfitTarget(r_multiple=2.0, price=price(11.0), fraction=0.5)
        ],
        trailing_rule="trail",
        time_exit_rule="time",
    )
    plan = TradePlan(
        candidate=candidate,
        entry_plan=entry,
        stop_plan=stop,
        position_plan=position,
        take_profit_plan=take_profit,
        expires_on=day + timedelta(days=2),
        strategy_version="sv",
        data_version="dv",
        code_version="test",
        confirmation_price=price(10.0),
        max_acceptable_price=price(10.1),
        execution_low=price(10.0),
        execution_high=price(10.1),
    )
    return plan.model_copy(update={"status": PlanStatus.PENDING_EXECUTION})


def _bar(day: date, open_: float, high: float, low: float, close: float, vol: float = 1e8):
    return {
        "ts_code": "000001.SZ",
        "trade_date": day,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "vol": vol,
        "amount": vol * close,
    }


def test_simulator_t_plus_one_stop_and_target():
    config = load_full_config()
    sim = Simulator(config, 100_000, 100_000, 100_000)
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    d2 = date(2024, 1, 4)
    sim.add_plans([_plan(d0)], d0)
    # The entry fills, but a same-day stop touch is protected by T+1.
    sim.process_day(d1, pl.DataFrame([_bar(d1, 10.0, 10.2, 9.4, 9.6)]))
    assert "000001.SZ" in sim.positions
    assert sim.trades == []
    # Next trading day the stop is actionable.
    exited = sim.process_day(d2, pl.DataFrame([_bar(d2, 10.0, 10.05, 9.3, 9.4)]))
    assert exited
    assert sim.trades[-1].exit_reason == "stop"
    assert "000001.SZ" not in sim.positions


def test_simulator_partial_fill_and_target():
    config = load_full_config()
    sim = Simulator(config, 1_000_000, 1_000_000, 1_000_000)
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    sim.add_plans([_plan(d0, shares=1000)], d0)
    # Tushare ``vol`` is in lots (手): 100 lots = 10,000 shares.  The 1%
    # participation cap is 100 shares, so the order partially fills.
    sim.process_day(d1, pl.DataFrame([_bar(d1, 10.0, 10.2, 9.9, 10.1, vol=100)]))
    assert sim.positions["000001.SZ"].shares == 100
    assert sim.positions["000001.SZ"].add_on_count == 0
    d2 = date(2024, 1, 4)
    sim.process_day(d2, pl.DataFrame([_bar(d2, 10.1, 10.2, 10.0, 10.1, vol=100)]))
    assert sim.positions["000001.SZ"].shares == 200
    assert sim.positions["000001.SZ"].add_on_count == 0
    d3 = date(2024, 1, 5)
    sim.process_day(d3, pl.DataFrame([_bar(d3, 10.2, 11.2, 10.1, 11.0, vol=1e8)]))
    assert sim.trades
    assert any(trade.exit_reason == "target" for trade in sim.trades)


class _FakeProvider:
    def __init__(self):
        self.days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]

    def load_trade_cal(self, start, end):
        return pl.DataFrame(
            {
                "exchange": ["SSE"] * len(self.days),
                "cal_date": self.days,
                "is_open": [1] * len(self.days),
                "pretrade_date": [None] * len(self.days),
            }
        ).filter((pl.col("cal_date") >= start) & (pl.col("cal_date") <= end))

    def load_bars(self, start, end):
        return pl.DataFrame(
            [
                _bar(day, 10.0, 10.2, 9.9, 10.1)
                for day in self.days
                if start <= day <= end
            ]
        )


class _StubPipeline:
    def __init__(self):
        self.calls = 0

    def run(self, day, account, **kwargs):
        self.calls += 1
        return SimpleNamespace(plans=[_plan(day)] if self.calls == 1 else [])


def test_engine_reuses_pipeline_and_records_curve():
    provider = _FakeProvider()
    pipeline = _StubPipeline()
    engine = BacktestEngine(
        provider=provider,
        config=load_full_config(),
        pipeline=pipeline,
    )
    result = engine.run(date(2024, 1, 2), date(2024, 1, 4), initial_equity=100_000)
    assert pipeline.calls == 3
    assert result.equity_curve
    assert result.notes
    assert result.missing_data == []


def test_simulator_does_not_fill_suspended_or_limit_up_bars():
    config = load_full_config()
    sim = Simulator(config, 100_000, 100_000, 100_000)
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    sim.add_plans([_plan(d0)], d0)
    suspended = {**_bar(d1, 10.0, 10.2, 9.9, 10.1, vol=0), "is_suspended": True}
    sim.process_day(d1, pl.DataFrame([suspended]))
    assert "000001.SZ" not in sim.positions

    limit_up = {
        **_bar(d1, 11.0, 11.0, 11.0, 11.0),
        "pre_close": 10.0,
        "up_limit": 11.0,
        "down_limit": 9.0,
    }
    sim.process_day(d1, pl.DataFrame([limit_up]))
    assert "000001.SZ" not in sim.positions


def test_simulator_cannot_sell_into_limit_down():
    config = load_full_config()
    sim = Simulator(config, 100_000, 100_000, 100_000)
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    d2 = date(2024, 1, 4)
    sim.add_plans([_plan(d0)], d0)
    sim.process_day(d1, pl.DataFrame([_bar(d1, 10.0, 10.2, 9.9, 10.1)]))
    assert "000001.SZ" in sim.positions

    limit_down = {
        **_bar(d2, 9.0, 9.0, 9.0, 9.0),
        "pre_close": 10.0,
        "up_limit": 11.0,
        "down_limit": 9.0,
    }
    sim.process_day(d2, pl.DataFrame([limit_down]))
    assert "000001.SZ" in sim.positions
    assert sim.trades == []


def test_simulator_metrics_include_risk_adjusted_fields():
    config = load_full_config()
    sim = Simulator(config, 1_000_000, 1_000_000, 1_000_000)
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    sim.add_plans([_plan(d0, shares=1000)], d0)
    sim.process_day(d1, pl.DataFrame([_bar(d1, 10.0, 10.2, 9.9, 10.1)]))
    metrics = sim.metrics()
    for key in (
        "annualized_return",
        "sharpe",
        "sortino",
        "calmar",
        "expectancy",
        "profit_factor",
        "turnover",
        "fill_rate",
        "avg_holding_days",
        "capital_utilization",
        "by_sub_mode",
    ):
        assert key in metrics
    assert metrics["fill_rate"] > 0


def test_engine_walk_forward_ablation_and_cost_sensitivity():
    provider = _FakeProvider()
    engine = BacktestEngine(
        provider=provider,
        config=load_full_config(),
        pipeline=_StubPipeline(),
    )
    folds = engine.walk_forward(
        date(2024, 1, 2), date(2024, 1, 4), folds=2, initial_equity=100_000
    )
    assert len(folds) == 2
    assert all("total_return" in fold for fold in folds)

    ablation = engine.ablation(
        date(2024, 1, 2),
        date(2024, 1, 4),
        strategies=("A", "C"),
        initial_equity=100_000,
    )
    assert set(ablation) == {"all", "A", "C"}

    sensitivity = engine.cost_sensitivity(
        date(2024, 1, 2),
        date(2024, 1, 4),
        multipliers=(1.0, 2.0),
        initial_equity=100_000,
    )
    assert [item["multiplier"] for item in sensitivity] == [1.0, 2.0]
