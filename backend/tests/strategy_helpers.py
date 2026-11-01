from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.core.full_config import load_full_config
from app.domain.enums import (
    PermissionMode,
    SentimentStage,
    Strategy,
    Template,
    TrendRegime,
)
from app.domain.models import (
    MarketRegime,
    RegimeIndicators,
    SectorStrength,
    StrategyPermission,
    TemplatePolicy,
)
from app.indicators.features import FeatureBuilder
from app.strategies.base import StrategyContext


def make_bars(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    codes: list[str] | None = None,
    start: date = date(2024, 1, 1),
) -> pl.DataFrame:
    rows: list[dict] = []
    if codes is None:
        codes = ["000001.SZ"]
    for code in codes:
        previous: float | None = None
        for index, close in enumerate(closes):
            pre_close = previous if previous is not None else close
            open_ = opens[index] if opens is not None else close * 0.995
            high = highs[index] if highs is not None else close * 1.005
            low = lows[index] if lows is not None else close * 0.995
            volume = volumes[index] if volumes is not None else 1_000_000.0
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": start + timedelta(days=index),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "pre_close": pre_close,
                    "vol": volume,
                    "amount": volume * close,
                    "adj_factor": 1.0,
                    "up_limit": round(pre_close * 1.10, 2),
                    "down_limit": round(pre_close * 0.90, 2),
                    "turnover_rate": 0.08,
                    "circ_mv": 1.0e10,
                    "is_suspended": False,
                }
            )
            previous = close
    return pl.DataFrame(rows)


def make_context(
    bars: pl.DataFrame,
    *,
    as_of: date | None = None,
    config=None,
    events: dict[str, list[dict]] | None = None,
    events_available: bool = True,
    leader_score: float = 80.0,
) -> StrategyContext:
    code = bars["ts_code"][0]
    as_of = as_of or bars["trade_date"][-1]
    features = FeatureBuilder().build(bars, as_of).with_columns(
        pl.lit("测试").alias("sector"),
        pl.lit("主板").alias("market"),
        pl.lit(False).alias("is_st"),
        pl.lit(300).alias("listing_days"),
    )
    sectors = [
        SectorStrength(
            sector="测试",
            score=90.0,
            rank=1,
            return_20d=0.20,
            return_5d=0.03,
            breadth=0.8,
            amount_change=0.2,
            limit_up_share=0.3,
            limit_up_count=3,
            red_rate=0.6,
            stock_count=10,
            persistent_days=5,
        )
    ]
    leaders = pl.DataFrame(
        {
            "ts_code": [code],
            "sector": ["测试"],
            "leader_score": [leader_score],
            "leader_rank": [1],
            "is_follower": [False],
            "amount_20d_avg": [3e8],
        }
    )
    cfg = config or load_full_config()
    permissions = {
        strategy: StrategyPermission(
            strategy=strategy,
            mode=PermissionMode.NORMAL,
            risk_factor=1.0,
            reasons=[],
        )
        for strategy in Strategy
    }
    market = MarketRegime(
        trade_date=as_of,
        trend_regime=TrendRegime.ATTACK,
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={
            Template.A: TemplatePolicy(enabled=True, mode="normal", max_positions=3),
            Template.B: TemplatePolicy(enabled=True, mode="normal", max_positions=2),
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
        strategy_permissions=permissions,
    )
    return StrategyContext(
        as_of=as_of,
        bars=bars,
        features=features,
        sectors=sectors,
        market=market,
        config=cfg,
        data_version="dv_test",
        leaders=leaders,
        events=events or {},
        events_available=events_available,
    )
