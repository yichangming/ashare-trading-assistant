"""Four-dimensional market assessment and per-strategy permissions."""

from __future__ import annotations

from datetime import datetime, time
from statistics import fmean, pstdev
from zoneinfo import ZoneInfo

import polars as pl

from app.core.full_config import FullSystemConfig
from app.domain.enums import (
    GlobalRiskMode,
    PermissionMode,
    Strategy,
)
from app.domain.models import MarketRegime, StrategyPermission

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _index_metrics(index_bars: pl.DataFrame) -> tuple[float | None, float, float]:
    if index_bars.is_empty() or "close" not in index_bars.columns:
        return None, 1.0, 1.0
    frame = index_bars.sort("trade_date")
    closes = [float(value) for value in frame["close"].to_list() if value is not None]
    if len(closes) < 2:
        return None, 1.0, 1.0
    returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1]]
    one_day = returns[-1]
    vol_percentile = 0.5
    if len(returns) >= 40:
        window = 20
        rolling = [
            pstdev(returns[i - window + 1 : i + 1])
            for i in range(window - 1, len(returns))
        ]
        if rolling:
            current = rolling[-1]
            sample = rolling[-252:]
            vol_percentile = sum(1 for value in sample if value <= current) / len(sample)
    amount_ratio = 1.0
    if "amount" in frame.columns:
        amounts = [float(value) for value in frame["amount"].to_list() if value is not None]
        if len(amounts) >= 20:
            short = fmean(amounts[-5:])
            long = fmean(amounts[-20:])
            amount_ratio = short / long if long > 0 else 1.0
    return one_day, vol_percentile, amount_ratio


def _decline_diffusion(bars: pl.DataFrame, as_of) -> float:
    if bars.is_empty() or "pre_close" not in bars.columns:
        return 1.0
    frame = bars.filter(pl.col("trade_date") == as_of)
    if frame.is_empty():
        return 1.0
    up = frame.filter(pl.col("close") > pl.col("pre_close")).height
    down = frame.filter(pl.col("close") < pl.col("pre_close")).height
    if up == 0:
        return 10.0 if down else 1.0
    return down / up


def assess_market(
    regime: MarketRegime,
    *,
    index_bars: pl.DataFrame,
    bars: pl.DataFrame,
    breadth: float,
    config: FullSystemConfig,
    minute_health: bool = True,
    data_fault: bool = False,
) -> MarketRegime:
    cfg = config.market
    one_day, vol_percentile, amount_ratio = _index_metrics(index_bars)
    volatility = (
        "stress" if vol_percentile >= cfg.vol_stress_percentile else "normal"
    )
    if vol_percentile <= 0.35:
        volatility = "calm"
    liquidity = "dry" if amount_ratio < cfg.liquidity_dry_ratio else "normal"
    diffusion = _decline_diffusion(bars, regime.trade_date)
    breadth_label = "normal"
    if breadth >= cfg.breadth_attack:
        breadth_label = "broad"
    elif breadth < cfg.breadth_defense:
        breadth_label = "narrow"

    global_risk = GlobalRiskMode.NORMAL
    risk_reasons: list[str] = []
    if data_fault:
        global_risk = GlobalRiskMode.PAUSE_NEW
        risk_reasons.append("data fault: no new signals")
    elif one_day is not None and one_day <= -0.08:
        global_risk = GlobalRiskMode.CLEAR
        risk_reasons.append("index one-day drop <= -8%: clear-risk review")
    elif one_day is not None and one_day <= -0.05:
        global_risk = GlobalRiskMode.PAUSE_NEW
        risk_reasons.append("index one-day drop <= -5%: pause new")
    elif volatility == "stress" and breadth_label == "narrow":
        global_risk = GlobalRiskMode.REDUCE
        risk_reasons.append("volatility stress + narrow breadth: reduce risk")
    elif diffusion > cfg.decline_diffusion * 2:
        global_risk = GlobalRiskMode.REDUCE
        risk_reasons.append(f"decline diffusion {diffusion:.1f}: reduce risk")

    permissions: dict[Strategy, StrategyPermission] = {}
    for strategy in Strategy:
        rule = cfg.permissions[strategy]
        reasons: list[str] = []
        if not rule.enabled:
            mode = PermissionMode.PAUSED
            reasons.append("strategy disabled")
        elif global_risk in (GlobalRiskMode.PAUSE_NEW, GlobalRiskMode.CLEAR):
            mode = PermissionMode.PAUSED
            reasons.append("global risk switch")
        else:
            trend = regime.trend_regime.value
            sentiment = regime.sentiment_stage.value
            normal = (
                trend in rule.normal_trends
                and breadth_label in rule.normal_breadth
                and sentiment in rule.normal_sentiment
                and volatility in rule.normal_vol
            )
            reduced = (
                trend in rule.reduced_trends
                or breadth_label in rule.reduced_breadth
                or sentiment in rule.reduced_sentiment
                or volatility in rule.reduced_vol
            )
            if normal:
                mode = PermissionMode.NORMAL
            elif reduced:
                mode = PermissionMode.REDUCED
                reasons.append("market conditions reduced")
            else:
                mode = PermissionMode.PAUSED
                reasons.append("market conditions not permitted")
            if rule.requires_minute_health and not minute_health:
                mode = PermissionMode.REDUCED
                reasons.append("minute/realtime data degraded")
        if global_risk == GlobalRiskMode.REDUCE and mode == PermissionMode.NORMAL:
            mode = PermissionMode.REDUCED
            reasons.append("global reduce-risk switch")
        factor = (
            rule.risk_factor_normal
            if mode == PermissionMode.NORMAL
            else rule.risk_factor_reduced
            if mode == PermissionMode.REDUCED
            else 0.0
        )
        permissions[strategy] = StrategyPermission(
            strategy=strategy,
            mode=mode,
            risk_factor=factor,
            reasons=reasons,
            data_time=datetime.combine(
                regime.trade_date, time(15, 0), tzinfo=SHANGHAI
            ),
        )
    return regime.model_copy(
        update={
            "strategy_permissions": permissions,
            "global_risk": global_risk,
            "risk_reasons": risk_reasons,
            "breadth_label": breadth_label,
            "volatility_label": volatility,
            "liquidity_label": liquidity,
            "decline_diffusion": diffusion,
        }
    )
