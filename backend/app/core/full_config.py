"""Configuration for the six-strategy full system.

The legacy dual-template config remains in :mod:`app.core.config`; this module
is the v2 baseline described in ``docs/full-system/parameters.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import Strategy

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CostsConfig(StrictModel):
    commission_rate: float = Field(default=0.00025, ge=0, le=0.01)
    commission_min: float = Field(default=5.0, ge=0)
    stamp_duty_rate: float = Field(default=0.0005, ge=0, le=0.01)
    transfer_fee_rate: float = Field(default=0.00001, ge=0, le=0.01)
    slippage_a: float = Field(default=0.001, ge=0, le=0.05)
    slippage_b: float = Field(default=0.002, ge=0, le=0.05)
    round_trip_cost_pct: float = Field(default=0.003, ge=0, le=0.05)


class UniverseConfig(StrictModel):
    allowed_boards: list[str] = Field(
        default_factory=lambda: ["主板", "创业板", "科创板"]
    )
    min_listing_days: int = Field(default=120, ge=0)
    min_avg_amount: float = Field(default=100_000_000, ge=0)
    b_min_avg_amount: float = Field(default=50_000_000, ge=0)
    min_price: float = Field(default=3.0, gt=0)
    max_price: float = Field(default=200.0, gt=0)
    market_cap_min: float = Field(default=3_000_000_000, ge=0)
    market_cap_max: float = Field(default=80_000_000_000, ge=0)
    b_market_cap_min: float = Field(default=3_000_000_000, ge=0)
    b_market_cap_max: float = Field(default=30_000_000_000, ge=0)
    exclude_st: bool = True
    exclude_suspended: bool = True


class PermissionRuleConfig(StrictModel):
    enabled: bool = True
    normal_trends: list[str] = Field(default_factory=lambda: ["attack", "neutral"])
    reduced_trends: list[str] = Field(default_factory=lambda: [])
    normal_breadth: list[str] = Field(default_factory=lambda: ["broad", "normal"])
    reduced_breadth: list[str] = Field(default_factory=lambda: ["narrow"])
    normal_sentiment: list[str] = Field(
        default_factory=lambda: ["start", "ferment", "climax"]
    )
    reduced_sentiment: list[str] = Field(default_factory=lambda: [])
    normal_vol: list[str] = Field(default_factory=lambda: ["calm", "normal"])
    reduced_vol: list[str] = Field(default_factory=lambda: ["stress"])
    risk_factor_normal: float = Field(default=1.0, ge=0, le=1)
    risk_factor_reduced: float = Field(default=0.5, ge=0, le=1)
    requires_minute_health: bool = False
    requires_verified_event: bool = False


def _default_permissions() -> dict[Strategy, PermissionRuleConfig]:
    return {
        Strategy.A: PermissionRuleConfig(
            normal_trends=["attack", "neutral"],
            reduced_trends=[],
            normal_breadth=["broad", "normal"],
            reduced_breadth=[],
            normal_sentiment=["start", "ferment", "climax"],
            reduced_sentiment=[],
        ),
        Strategy.B: PermissionRuleConfig(
            normal_breadth=["broad", "normal"],
            reduced_breadth=[],
            normal_sentiment=["start", "ferment"],
            reduced_sentiment=[],
            requires_minute_health=True,
        ),
        Strategy.C: PermissionRuleConfig(
            normal_trends=["attack", "neutral"],
            normal_breadth=["broad", "normal"],
            reduced_breadth=[],
        ),
        Strategy.D: PermissionRuleConfig(
            normal_trends=["neutral"],
            reduced_trends=["attack", "defensive"],
            normal_vol=["calm", "normal", "stress"],
            normal_sentiment=["start", "ferment", "climax", "ebb", "freeze"],
            risk_factor_normal=1.0,
            risk_factor_reduced=0.5,
        ),
        Strategy.E: PermissionRuleConfig(
            normal_trends=["neutral", "defensive"],
            reduced_trends=["attack"],
            normal_breadth=["broad", "normal", "narrow"],
            reduced_breadth=[],
            normal_sentiment=["start", "ferment", "climax"],
            reduced_sentiment=[],
        ),
        Strategy.F: PermissionRuleConfig(
            normal_trends=["attack", "neutral"],
            reduced_trends=["defensive"],
            normal_sentiment=["start", "ferment", "climax"],
            reduced_sentiment=["ebb", "freeze"],
            requires_verified_event=True,
        ),
    }


class MarketConfig(StrictModel):
    index_code: str = "000985.CSI"
    fallback_index_code: str = "000300.SH"
    ma_fast: int = Field(default=20, ge=1)
    ma_slow: int = Field(default=60, ge=1)
    slope_window: int = Field(default=10, ge=1)
    breadth_attack: float = Field(default=0.55, ge=0, le=1)
    breadth_defense: float = Field(default=0.35, ge=0, le=1)
    breadth_exit: float = Field(default=0.45, ge=0, le=1)
    vol_stress_percentile: float = Field(default=0.85, ge=0, le=1)
    liquidity_dry_ratio: float = Field(default=0.70, ge=0)
    decline_diffusion: float = Field(default=1.5, ge=0)
    confirm_days: int = Field(default=2, ge=1)
    permissions: dict[Strategy, PermissionRuleConfig] = Field(
        default_factory=_default_permissions
    )


class StrategyAConfig(StrictModel):
    rs_min: float = Field(default=0.70, ge=0, le=1)
    pullback_window: int = Field(default=20, ge=5)
    pullback_min_days: int = Field(default=3, ge=1)
    pullback_max_days: int = Field(default=8, ge=1)
    pullback_depth_min: float = Field(default=0.02, ge=0, le=1)
    pullback_depth_max: float = Field(default=0.15, ge=0, le=1)
    touch_atr: float = Field(default=0.5, ge=0)
    shallow_ma: int = Field(default=10, ge=1)
    deep_ma: int = Field(default=20, ge=1)
    structure_ma_tolerance: float = Field(default=0.02, ge=0, le=0.2)
    volume_shrink_ratio: float = Field(default=1.0, gt=0)
    confirmation_days: int = Field(default=2, ge=1)
    valid_days: int = Field(default=2, ge=1)
    stop_atr: float = Field(default=0.5, ge=0)
    stop_min: float = Field(default=0.03, ge=0, le=0.5)
    stop_max: float = Field(default=0.08, ge=0, le=0.5)
    time_stop_days: int = Field(default=10, ge=1)
    r2: float = Field(default=2.0, gt=0)
    r3: float = Field(default=3.0, gt=0)
    first_fraction: float = Field(default=1 / 3, gt=0, lt=1)
    second_fraction: float = Field(default=1 / 3, gt=0, lt=1)
    min_rr: float = Field(default=1.5, ge=0)
    risk_pct: float = Field(default=0.025, gt=0, le=0.2)

    @model_validator(mode="after")
    def _fractions(self) -> StrategyAConfig:
        if self.first_fraction + self.second_fraction >= 1:
            raise ValueError("A take-profit fractions must leave a remainder")
        if self.pullback_min_days > self.pullback_max_days:
            raise ValueError("pullback_min_days must be <= pullback_max_days")
        if self.stop_min > self.stop_max:
            raise ValueError("stop_min must be <= stop_max")
        return self


class StrategyBConfig(StrictModel):
    board_min: int = Field(default=1, ge=1)
    board_max: int = Field(default=2, ge=1)
    observe_days: int = Field(default=3, ge=1)
    sector_top_n: int = Field(default=2, ge=1)
    sector_limit_up_min: int = Field(default=3, ge=0)
    sector_red_rate_min: float = Field(default=0.50, ge=0, le=1)
    blow_rate_max: float = Field(default=0.35, ge=0, le=1)
    max_board_min: int = Field(default=3, ge=0)
    max_20d_gain: float = Field(default=0.80, ge=0)
    stop_atr: float = Field(default=0.5, ge=0)
    stop_min: float = Field(default=0.03, ge=0, le=0.5)
    stop_max: float = Field(default=0.05, ge=0, le=0.5)
    time_stop_days: int = Field(default=3, ge=1)
    risk_pct: float = Field(default=0.02, gt=0, le=0.2)
    b2_bar_minutes: int = Field(default=5, ge=1)
    b2_stabilize_count: int = Field(default=2, ge=1)
    b2_valid_minutes: int = Field(default=10, ge=1)
    b2_max_attempts: int = Field(default=2, ge=1)
    b2_last_signal_time: str = "14:50"
    b2_stop_max: float = Field(default=0.03, ge=0, le=0.2)
    b2_stop_min: float = Field(default=0.015, ge=0, le=0.2)
    vwap_ratio_tolerance: float = Field(default=0.02, ge=0)

    @model_validator(mode="after")
    def _ranges(self) -> StrategyBConfig:
        if self.board_min > self.board_max:
            raise ValueError("board_min must be <= board_max")
        if self.stop_min > self.stop_max:
            raise ValueError("stop_min must be <= stop_max")
        if self.b2_stop_min > self.b2_stop_max:
            raise ValueError("b2_stop_min must be <= b2_stop_max")
        return self


class StrategyCConfig(StrictModel):
    platform_window: int = Field(default=20, ge=15, le=30)
    upper_quantile: float = Field(default=0.90, gt=0, lt=1)
    lower_quantile: float = Field(default=0.10, gt=0, lt=1)
    width_min: float = Field(default=0.04, ge=0)
    width_max: float = Field(default=0.15, ge=0)
    touch_atr: float = Field(default=0.25, ge=0)
    min_touches: int = Field(default=3, ge=1)
    touch_spacing: int = Field(default=2, ge=1)
    slope_max: float = Field(default=0.0015, ge=0)
    abnormal_atr: float = Field(default=1.5, ge=0)
    breakout_atr: float = Field(default=0.2, ge=0)
    volume_ratio_min: float = Field(default=1.2, ge=0)
    c1_valid_days: int = Field(default=1, ge=1)
    c2_observe_days: int = Field(default=10, ge=1)
    c2_zone_atr: float = Field(default=0.5, ge=0)
    c2_valid_days: int = Field(default=2, ge=1)
    stop_atr: float = Field(default=0.5, ge=0)
    stop_min: float = Field(default=0.03, ge=0, le=0.5)
    stop_max: float = Field(default=0.08, ge=0, le=0.5)
    time_stop_days: int = Field(default=10, ge=1)
    r2: float = Field(default=2.0, gt=0)
    r3: float = Field(default=3.0, gt=0)
    min_rr: float = Field(default=1.5, ge=0)
    risk_pct: float = Field(default=0.025, gt=0, le=0.2)

    @model_validator(mode="after")
    def _ranges(self) -> StrategyCConfig:
        if self.width_min > self.width_max:
            raise ValueError("width_min must be <= width_max")
        if self.stop_min > self.stop_max:
            raise ValueError("stop_min must be <= stop_max")
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("lower_quantile must be < upper_quantile")
        return self


class StrategyDConfig(StrictModel):
    z_min: float = Field(default=-2.0)
    drop5_uptrend: float = Field(default=-0.08, le=0)
    drop5_downtrend: float = Field(default=-0.12, le=0)
    drop10_uptrend: float = Field(default=-0.12, le=0)
    drop10_downtrend: float = Field(default=-0.18, le=0)
    drawdown20_uptrend: float = Field(default=0.15, ge=0, le=1)
    drawdown20_downtrend: float = Field(default=0.25, ge=0, le=1)
    industry_relative: float = Field(default=-0.03, le=0)
    min_conditions: int = Field(default=2, ge=1, le=4)
    structure_wait_days: int = Field(default=5, ge=1)
    stop_atr: float = Field(default=0.5, ge=0)
    stop_min: float = Field(default=0.03, ge=0, le=0.5)
    stop_max: float = Field(default=0.08, ge=0, le=0.5)
    time_stop_days: int = Field(default=5, ge=1)
    r2: float = Field(default=2.0, gt=0)
    risk_pct: float = Field(default=0.015, gt=0, le=0.2)
    downtrend_risk_factor: float = Field(default=0.5, ge=0, le=1)
    downtrend_real_enabled: bool = False
    event_risk_days: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def _ranges(self) -> StrategyDConfig:
        if self.stop_min > self.stop_max:
            raise ValueError("stop_min must be <= stop_max")
        return self


class StrategyEConfig(StrictModel):
    range_window: int = Field(default=30, ge=20, le=60)
    upper_quantile: float = Field(default=0.90, gt=0, lt=1)
    lower_quantile: float = Field(default=0.10, gt=0, lt=1)
    width_min: float = Field(default=0.04, ge=0)
    width_max: float = Field(default=0.15, ge=0)
    touch_atr: float = Field(default=0.25, ge=0)
    min_touches: int = Field(default=3, ge=1)
    touch_spacing: int = Field(default=3, ge=1)
    slope_min: float = Field(default=-0.0005)
    entry_zone_pct: float = Field(default=0.20, gt=0, le=0.5)
    stop_atr: float = Field(default=0.5, ge=0)
    stop_min: float = Field(default=0.03, ge=0, le=0.5)
    stop_max: float = Field(default=0.08, ge=0, le=0.5)
    valid_days: int = Field(default=2, ge=1)
    min_rr: float = Field(default=1.5, ge=0)
    time_stop_days: int = Field(default=10, ge=1)
    risk_pct: float = Field(default=0.015, gt=0, le=0.2)
    r2: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def _ranges(self) -> StrategyEConfig:
        if self.width_min > self.width_max:
            raise ValueError("width_min must be <= width_max")
        if self.stop_min > self.stop_max:
            raise ValueError("stop_min must be <= stop_max")
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("lower_quantile must be < upper_quantile")
        return self


class StrategyFConfig(StrictModel):
    event_window_days: int = Field(default=5, ge=1)
    require_verified: bool = True
    allow_ambiguous: bool = False
    ai_extraction: bool = False
    price_mode: Literal["c1", "c2"] = "c2"
    risk_pct: float = Field(default=0.02, gt=0, le=0.2)
    min_materiality: float = Field(default=0.0, ge=0, le=1)


class StrategyConfigs(StrictModel):
    a: StrategyAConfig = Field(default_factory=StrategyAConfig)
    b: StrategyBConfig = Field(default_factory=StrategyBConfig)
    c: StrategyCConfig = Field(default_factory=StrategyCConfig)
    d: StrategyDConfig = Field(default_factory=StrategyDConfig)
    e: StrategyEConfig = Field(default_factory=StrategyEConfig)
    f: StrategyFConfig = Field(default_factory=StrategyFConfig)


class RiskConfig(StrictModel):
    max_positions: int = Field(default=5, ge=1)
    max_per_strategy: dict[Strategy, int] = Field(
        default_factory=lambda: {
            Strategy.A: 2,
            Strategy.B: 2,
            Strategy.C: 2,
            Strategy.D: 1,
            Strategy.E: 1,
            Strategy.F: 2,
        }
    )
    max_b_positions: int = Field(default=2, ge=0)
    single_position_pct: float = Field(default=0.30, gt=0, le=1)
    max_industry_positions: int = Field(default=2, ge=1)
    min_cash_pct: float = Field(default=0.10, ge=0, lt=1)
    max_exposure_pct: float = Field(default=0.90, gt=0, le=1)
    max_heat: float = Field(default=0.075, gt=0, le=1)
    correlation_threshold: float = Field(default=0.80, ge=0, le=1)
    drawdown_half_at: float = Field(default=0.10, ge=0, le=1)
    drawdown_stop_at: float = Field(default=0.15, ge=0, le=1)
    add_on_max: int = Field(default=2, ge=0)
    add_on_fraction: float = Field(default=0.50, gt=0, le=1)
    add_on_strategies: list[Strategy] = Field(
        default_factory=lambda: [Strategy.A, Strategy.C, Strategy.F]
    )
    stress_gap_pct: float = Field(default=-0.10, le=0)


class EventConfig(StrictModel):
    event_window_days: int = Field(default=5, ge=1)
    require_human_verification: bool = True
    sources: list[str] = Field(
        default_factory=lambda: [
            "forecast_vip",
            "express_vip",
            "income_vip",
            "repurchase",
            "stk_holdertrade",
            "manual",
        ]
    )
    retraction_detection: bool = True


class BacktestConfig(StrictModel):
    initial_equity: float = Field(default=1_000_000, gt=0)
    min_history_days: int = Field(default=400, ge=120)
    walk_forward_train_years: int = Field(default=3, ge=1)
    walk_forward_test_years: int = Field(default=1, ge=1)
    demo: bool = True
    max_runtime_seconds: int = Field(default=600, ge=1)


class SchedulerConfig(StrictModel):
    timezone: str = "Asia/Shanghai"
    daily_time: str = "17:00"
    data_health_time: str = "16:30"
    intraday_interval_seconds: int = Field(default=60, ge=10)
    retry_count: int = Field(default=3, ge=0)
    retry_base_seconds: int = Field(default=30, ge=1)
    notification_retention_days: int = Field(default=90, ge=1)


class WebConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    session_cookie: str = "ashare_session"
    allow_demo: bool = True
    sse_heartbeat_seconds: int = Field(default=15, ge=5)
    sound_notifications: bool = True


class FullSystemConfig(StrictModel):
    name: str = "full_system_v1"
    version: str = "1.0"
    market: MarketConfig = Field(default_factory=MarketConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    strategies: StrategyConfigs = Field(default_factory=StrategyConfigs)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    events: EventConfig = Field(default_factory=EventConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @model_validator(mode="after")
    def _risk_consistency(self) -> FullSystemConfig:
        if self.risk.drawdown_half_at >= self.risk.drawdown_stop_at:
            raise ValueError("drawdown_half_at must be < drawdown_stop_at")
        if self.risk.min_cash_pct >= 1:
            raise ValueError("min_cash_pct must be < 1")
        if self.market.breadth_defense > self.market.breadth_attack:
            raise ValueError("breadth_defense must be <= breadth_attack")
        return self


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    return payload


def load_full_config(path: Path | str | None = None) -> FullSystemConfig:
    config_path = (
        Path(path)
        if path is not None
        else BACKEND_ROOT / "configs" / "strategies" / "full_system_v1.yaml"
    )
    if not config_path.exists():
        return FullSystemConfig()
    return FullSystemConfig.model_validate(_load_yaml(config_path))
