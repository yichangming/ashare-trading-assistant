from __future__ import annotations

from datetime import timedelta

import polars as pl

from app.core.full_config import load_full_config
from app.domain.enums import (
    PermissionMode,
    SentimentStage,
    Strategy,
    TrendRegime,
)
from app.regime.permissions import assess_market
from tests.strategy_helpers import make_bars, make_context


def _index_closes() -> list[float]:
    # Early high volatility, recent low volatility -> current percentile is low.
    closes = [100.0]
    for index in range(1, 60):
        amplitude = 0.01 if index < 40 else 0.002
        step = amplitude if index % 2 else -amplitude * 0.9
        closes.append(closes[-1] * (1 + step))
    return closes


def _assess(trend: TrendRegime, sentiment: SentimentStage, breadth: float):
    context = make_context(make_bars([10.0] * 40))
    as_of = context.as_of
    regime = context.market.model_copy(
        update={"trend_regime": trend, "sentiment_stage": sentiment}
    )
    index = pl.DataFrame(
        {
            "trade_date": [as_of - timedelta(days=59 - i) for i in range(60)],
            "close": _index_closes(),
            "amount": [1.0e10] * 60,
        }
    )
    bars = pl.DataFrame(
        {
            "trade_date": [as_of] * 10,
            "pre_close": [10.0] * 10,
            "close": [10.0] * 10,
        }
    )
    return assess_market(
        regime,
        index_bars=index,
        bars=bars,
        breadth=breadth,
        config=load_full_config(),
    )


def test_defensive_narrow_permission_matrix():
    regime = _assess(TrendRegime.DEFENSIVE, SentimentStage.START, 0.15)
    modes = {
        strategy: permission.mode
        for strategy, permission in regime.strategy_permissions.items()
    }
    assert modes[Strategy.A] == PermissionMode.PAUSED
    assert modes[Strategy.B] == PermissionMode.PAUSED
    assert modes[Strategy.C] == PermissionMode.PAUSED
    assert modes[Strategy.D] == PermissionMode.REDUCED
    assert modes[Strategy.E] == PermissionMode.NORMAL
    assert modes[Strategy.F] == PermissionMode.REDUCED


def test_attack_broad_permission_matrix():
    regime = _assess(TrendRegime.ATTACK, SentimentStage.FERMENT, 0.65)
    modes = {
        strategy: permission.mode
        for strategy, permission in regime.strategy_permissions.items()
    }
    assert modes[Strategy.A] == PermissionMode.NORMAL
    assert modes[Strategy.B] == PermissionMode.NORMAL
    assert modes[Strategy.C] == PermissionMode.NORMAL
    assert modes[Strategy.D] == PermissionMode.REDUCED
    assert modes[Strategy.E] == PermissionMode.REDUCED
    assert modes[Strategy.F] == PermissionMode.NORMAL


def test_freeze_pauses_breakout_families():
    regime = _assess(TrendRegime.NEUTRAL, SentimentStage.FREEZE, 0.55)
    modes = {
        strategy: permission.mode
        for strategy, permission in regime.strategy_permissions.items()
    }
    assert modes[Strategy.A] == PermissionMode.PAUSED
    assert modes[Strategy.C] == PermissionMode.PAUSED
    assert modes[Strategy.E] == PermissionMode.PAUSED
    assert modes[Strategy.F] == PermissionMode.REDUCED


def test_minute_health_reduces_b_permission():
    context = make_context(make_bars([10.0] * 40))
    as_of = context.as_of
    regime = context.market.model_copy(update={"trend_regime": TrendRegime.NEUTRAL})
    index = pl.DataFrame(
        {
            "trade_date": [as_of - timedelta(days=59 - i) for i in range(60)],
            "close": _index_closes(),
            "amount": [1.0e10] * 60,
        }
    )
    bars = pl.DataFrame(
        {
            "trade_date": [as_of] * 10,
            "pre_close": [10.0] * 10,
            "close": [10.0] * 10,
        }
    )
    result = assess_market(
        regime,
        index_bars=index,
        bars=bars,
        breadth=0.6,
        config=load_full_config(),
        minute_health=False,
    )
    assert result.strategy_permissions[Strategy.B].mode == PermissionMode.REDUCED
