from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.core.config import load_strategy_config
from app.domain.enums import SentimentStage, Template, TrendRegime
from app.domain.models import MarketRegime
from app.regime.market import MarketRegimeService


def _index_bars(
    closes: list[float], start: date = date(2024, 1, 1)
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [
                start + timedelta(days=index) for index in range(len(closes))
            ],
            "close": closes,
        }
    )


def _rising() -> pl.DataFrame:
    return _index_bars([100.0 + index for index in range(61)])


def _neutral() -> pl.DataFrame:
    return _index_bars(
        [100.0 + 1.2 * index for index in range(51)]
        + [160.0 - 2.0 * (index - 50) for index in range(51, 61)]
    )


def _sentiment_row() -> dict:
    return {
        "limit_up_count": 80,
        "limit_down_count": 0,
        "max_board_height": 5,
        "promotion_rate": 0.5,
        "blow_rate": 0.1,
        "red_rate": 0.6,
        "limit_up_premium": 0.02,
    }


def _compute(
    index_bars: pl.DataFrame,
    *,
    stage: SentimentStage,
    breadth: float = 0.60,
    as_of: date = date(2024, 3, 1),
) -> MarketRegime:
    return MarketRegimeService().compute(
        as_of=as_of,
        index_bars=index_bars,
        sentiment_row=_sentiment_row(),
        stage=stage,
        breadth=breadth,
        sector_ranking=[],
        degraded_flags=[],
        data_version="dv_test",
        config=load_strategy_config(),
    )


def test_defense_overrides_sentiment():
    config = load_strategy_config()
    start = date(2024, 1, 1)
    indices = pl.DataFrame(
        {
            "trade_date": [start + timedelta(days=i) for i in range(61)],
            "close": [100.0 - i for i in range(61)],
        }
    )
    sentiment_row = {
        "limit_up_count": 80,
        "limit_down_count": 0,
        "max_board_height": 5,
        "promotion_rate": 0.5,
        "blow_rate": 0.1,
        "red_rate": 0.6,
        "limit_up_premium": 0.02,
    }
    regime = MarketRegimeService().compute(
        as_of=date(2024, 3, 1),
        index_bars=indices,
        sentiment_row=sentiment_row,
        stage=SentimentStage.FERMENT,
        breadth=0.60,
        sector_ranking=[],
        degraded_flags=[],
        data_version="dv_test",
        config=config,
    )
    assert regime.trend_regime == TrendRegime.DEFENSIVE
    assert regime.risk_factor == 0.0
    assert not regime.template_availability[Template.A].enabled
    assert not regime.template_availability[Template.B].enabled


def test_attack_trend_enables_both_templates():
    regime = _compute(_rising(), stage=SentimentStage.FERMENT)
    assert regime.trend_regime == TrendRegime.ATTACK
    assert regime.risk_factor == pytest.approx(1.0)
    a = regime.template_availability[Template.A]
    b = regime.template_availability[Template.B]
    assert (a.enabled, a.mode, a.max_positions) == (True, "normal", 3)
    assert (b.enabled, b.mode, b.max_positions) == (True, "normal", 2)
    assert regime.indicators.index_close == pytest.approx(160.0)
    assert regime.indicators.index_ma20 == pytest.approx(150.5)
    assert regime.indicators.index_ma60 == pytest.approx(130.5)
    assert regime.indicators.breadth == pytest.approx(0.60)
    assert regime.data_version == "dv_test"
    assert regime.sector_ranking == []


def test_neutral_trend_caps_ferment_b_position():
    regime = _compute(_neutral(), stage=SentimentStage.FERMENT)
    assert regime.trend_regime == TrendRegime.NEUTRAL
    assert regime.risk_factor == pytest.approx(0.5)
    assert regime.template_availability[Template.A].max_positions == 3
    b = regime.template_availability[Template.B]
    assert b.enabled
    assert b.mode == "normal"
    assert b.max_positions == 1


def test_climax_forces_pullback_only_and_disables_b():
    regime = _compute(_rising(), stage=SentimentStage.CLIMAX)
    a = regime.template_availability[Template.A]
    b = regime.template_availability[Template.B]
    assert (a.enabled, a.mode, a.max_positions) == (
        True,
        "pullback_only",
        3,
    )
    assert (b.enabled, b.mode, b.max_positions) == (False, "observe", 0)
    assert regime.risk_factor == pytest.approx(0.5)


@pytest.mark.parametrize("stage", [SentimentStage.FREEZE, SentimentStage.EBB])
def test_cold_stages_disable_both_templates(stage: SentimentStage):
    regime = _compute(_rising(), stage=stage)
    assert regime.trend_regime == TrendRegime.ATTACK
    assert regime.risk_factor == 0.0
    for template in (Template.A, Template.B):
        policy = regime.template_availability[template]
        assert not policy.enabled
        assert policy.mode == "observe"
        assert policy.max_positions == 0


def test_weak_breadth_forces_defensive_even_with_strong_index():
    regime = _compute(_rising(), stage=SentimentStage.FERMENT, breadth=0.30)
    assert regime.trend_regime == TrendRegime.DEFENSIVE
    assert regime.risk_factor == 0.0
    assert all(
        not policy.enabled
        for policy in regime.template_availability.values()
    )


def test_missing_index_data_raises():
    with pytest.raises(ValueError, match="index data missing"):
        _compute(
            _rising(),
            stage=SentimentStage.FERMENT,
            as_of=date(2024, 3, 2),
        )


def test_future_index_bars_do_not_change_regime():
    as_of = date(2024, 3, 1)
    full_bars = _index_bars(
        [100.0 + index if index <= 60 else 1.0 for index in range(70)]
    )
    full = _compute(full_bars, stage=SentimentStage.FERMENT, as_of=as_of)
    truncated = _compute(_rising(), stage=SentimentStage.FERMENT, as_of=as_of)
    assert full.model_dump() == truncated.model_dump()
