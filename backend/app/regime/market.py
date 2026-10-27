from __future__ import annotations

from datetime import date

import polars as pl

from app.core.config import StrategyConfig
from app.domain.enums import SentimentStage, Template, TrendRegime
from app.domain.models import (
    MarketRegime,
    RegimeIndicators,
    SectorStrength,
    TemplatePolicy,
)


class MarketRegimeService:
    def compute(
        self,
        as_of: date,
        index_bars: pl.DataFrame,
        sentiment_row: dict,
        stage: SentimentStage,
        breadth: float,
        sector_ranking: list[SectorStrength],
        degraded_flags: list[str],
        data_version: str,
        config: StrategyConfig,
    ) -> MarketRegime:
        fast_window = config.environment.ma_fast
        slow_window = config.environment.ma_slow
        prepared = index_bars.sort("trade_date").with_columns(
            pl.col("close")
            .rolling_mean(fast_window, min_samples=fast_window)
            .alias("ma_fast"),
            pl.col("close")
            .rolling_mean(slow_window, min_samples=slow_window)
            .alias("ma_slow"),
        )
        row = prepared.filter(pl.col("trade_date") == as_of).tail(1)
        if row.is_empty():
            raise ValueError(f"index data missing for {as_of}")
        close_value = row["close"][0]
        fast_value = row["ma_fast"][0]
        slow_value = row["ma_slow"][0]
        if fast_value is None or slow_value is None:
            raise ValueError(
                f"index history insufficient for {as_of}: "
                f"need {slow_window} bars"
            )
        index_close = float(close_value)
        ma_fast = float(fast_value)
        ma_slow = float(slow_value)

        if index_close < ma_slow or breadth < config.environment.breadth_defense:
            trend = TrendRegime.DEFENSIVE
            trend_cap = 0.0
        elif (
            index_close > ma_fast
            and breadth >= config.environment.breadth_attack
        ):
            trend = TrendRegime.ATTACK
            trend_cap = 1.0
        else:
            trend = TrendRegime.NEUTRAL
            trend_cap = 0.5

        stage_policy = getattr(config.switching, stage.value)
        risk_factor = min(stage_policy.risk_factor, trend_cap)
        policies = self._resolve_policies(stage, trend, config)

        indicators = RegimeIndicators(
            limit_up_count=int(sentiment_row["limit_up_count"]),
            limit_down_count=int(sentiment_row["limit_down_count"]),
            max_board_height=int(sentiment_row["max_board_height"]),
            promotion_rate=float(sentiment_row["promotion_rate"]),
            blow_rate=float(sentiment_row["blow_rate"]),
            red_rate=float(sentiment_row["red_rate"]),
            limit_up_premium=float(sentiment_row["limit_up_premium"]),
            breadth=breadth,
            index_close=index_close,
            index_ma20=ma_fast,
            index_ma60=ma_slow,
        )
        return MarketRegime(
            trade_date=as_of,
            trend_regime=trend,
            sentiment_stage=stage,
            risk_factor=risk_factor,
            template_availability=policies,
            indicators=indicators,
            sector_ranking=sector_ranking,
            degraded_flags=degraded_flags,
            data_version=data_version,
        )

    def _resolve_policies(
        self, stage: SentimentStage, trend: TrendRegime, config: StrategyConfig
    ) -> dict[Template, TemplatePolicy]:
        if trend == TrendRegime.DEFENSIVE:
            return {
                Template.A: TemplatePolicy(
                    enabled=False, mode="observe", max_positions=0
                ),
                Template.B: TemplatePolicy(
                    enabled=False, mode="observe", max_positions=0
                ),
            }
        stage_policy = getattr(config.switching, stage.value)
        a = TemplatePolicy(
            enabled=stage_policy.a.enabled,
            mode=stage_policy.a.mode,
            max_positions=stage_policy.a.max_positions,
        )
        max_b = stage_policy.b.max_positions
        if stage == SentimentStage.FERMENT and trend == TrendRegime.NEUTRAL:
            max_b = min(max_b, 1)
        b = TemplatePolicy(
            enabled=stage_policy.b.enabled,
            mode=stage_policy.b.mode,
            max_positions=max_b,
        )
        if stage == SentimentStage.CLIMAX:
            a = TemplatePolicy(
                enabled=True, mode="pullback_only", max_positions=3
            )
            b = TemplatePolicy(
                enabled=False, mode="observe", max_positions=0
            )
        return {Template.A: a, Template.B: b}
