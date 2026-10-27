from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FreezeThresholds(StrictModel):
    limit_up_max: int = Field(ge=0)
    limit_down_min: int = Field(ge=0)
    blow_rate_min: float = Field(ge=0, le=1)
    max_board_max: int = Field(ge=0)
    red_rate_max: float = Field(ge=0, le=1)


class StartThresholds(StrictModel):
    limit_up_min: int = Field(ge=0)
    limit_up_max: int = Field(ge=0)
    max_board_min: int = Field(ge=0)
    max_board_max: int = Field(ge=0)
    blow_rate_min: float = Field(ge=0, le=1)
    blow_rate_max: float = Field(ge=0, le=1)
    red_rate_min: float = Field(ge=0, le=1)
    red_rate_max: float = Field(ge=0, le=1)


class FermentThresholds(StrictModel):
    limit_up_min: int = Field(ge=0)
    max_board_min: int = Field(ge=0)
    blow_rate_max: float = Field(ge=0, le=1)
    red_rate_min: float = Field(ge=0, le=1)


class ClimaxThresholds(StrictModel):
    limit_up_min: int = Field(ge=0)
    max_board_min: int = Field(ge=0)
    red_rate_min: float = Field(ge=0, le=1)
    blow_rate_rise_pp: float = Field(ge=0)
    premium_drop_pp: float = Field(ge=0)


class EbbThresholds(StrictModel):
    limit_up_max: int = Field(ge=0)
    limit_down_min: int = Field(ge=0)
    blow_rate_min: float = Field(ge=0, le=1)
    red_rate_max: float = Field(ge=0, le=1)


class EnvironmentThresholds(StrictModel):
    freeze: FreezeThresholds
    start: StartThresholds
    ferment: FermentThresholds
    climax: ClimaxThresholds
    ebb: EbbThresholds


class EnvironmentConfig(StrictModel):
    index_code: str
    fallback_index_code: str
    ma_fast: int = Field(default=20, ge=0)
    ma_slow: int = Field(default=60, ge=0)
    breadth_attack: float = Field(default=0.55, ge=0, le=1)
    breadth_defense: float = Field(default=0.35, ge=0, le=1)
    stage_confirm_days: int = Field(default=2, ge=0)
    thresholds: EnvironmentThresholds

    @model_validator(mode="after")
    def _check_windows_and_breadth(self) -> EnvironmentConfig:
        if self.ma_fast > self.ma_slow:
            raise ValueError("ma_fast must be <= ma_slow")
        if self.breadth_defense > self.breadth_attack:
            raise ValueError("breadth_defense must be <= breadth_attack")
        return self


class IndustryWeights(StrictModel):
    return_20d: float = Field(ge=0)
    return_5d: float = Field(ge=0)
    breadth: float = Field(ge=0)
    amount_change: float = Field(ge=0)
    limit_up_share: float = Field(ge=0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> IndustryWeights:
        total = (
            self.return_20d
            + self.return_5d
            + self.breadth
            + self.amount_change
            + self.limit_up_share
        )
        if not math.isfinite(total):
            raise ValueError("industry weights must be finite")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"industry weights must sum to 1.0, got {total}")
        return self


class IndustryConfig(StrictModel):
    top_n: int = Field(default=3, ge=0)
    persistence_days: int = Field(default=2, ge=0)
    min_stocks: int = Field(default=5, ge=0)
    source: Literal["auto", "sw", "stock_basic"] = "auto"
    weights: IndustryWeights


class LeaderWeights(StrictModel):
    rs_20d: float = Field(ge=0)
    rs_5d: float = Field(ge=0)
    amount_turnover: float = Field(ge=0)
    high_distance: float = Field(ge=0)
    limit_up_status: float = Field(ge=0)
    size_price_fit: float = Field(ge=0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> LeaderWeights:
        total = (
            self.rs_20d
            + self.rs_5d
            + self.amount_turnover
            + self.high_distance
            + self.limit_up_status
            + self.size_price_fit
        )
        if not math.isfinite(total):
            raise ValueError("leader weights must be finite")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"leader weights must sum to 1.0, got {total}")
        return self


class LeaderConfig(StrictModel):
    top_n_per_industry: int = Field(default=2, ge=0)
    follower_rank_cutoff: int = Field(default=3, ge=0)
    amount_vs_leader_min: float = Field(default=0.5, ge=0)
    weights: LeaderWeights


class ScoreWeights(StrictModel):
    leader: float = Field(ge=0)
    setup: float = Field(ge=0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> ScoreWeights:
        total = self.leader + self.setup
        if not math.isfinite(total):
            raise ValueError("score weights must be finite")
        if abs(total - 1.0) > 1e-6:
            raise ValueError("score weights must sum to 1.0")
        return self


class TakeProfitAConfig(StrictModel):
    r1: float = Field(default=1.0, gt=0)
    r2: float = Field(default=2.0, gt=0)
    r3: float = Field(default=3.0, gt=0)
    first_fraction: float = Field(default=1 / 3, gt=0, le=1)
    second_fraction: float = Field(default=1 / 3, gt=0, le=1)

    @model_validator(mode="after")
    def _check_levels_and_fractions(self) -> TakeProfitAConfig:
        if not self.r1 < self.r2 < self.r3:
            raise ValueError("take-profit levels must satisfy r1 < r2 < r3")
        if self.first_fraction + self.second_fraction >= 1.0:
            raise ValueError(
                "take-profit fractions must sum to less than 1.0 to leave a remainder"
            )
        return self


class TakeProfitBConfig(StrictModel):
    r1: float = Field(default=1.0, gt=0)
    r2: float = Field(default=2.0, gt=0)
    second_fraction: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def _check_levels(self) -> TakeProfitBConfig:
        if not self.r1 < self.r2:
            raise ValueError("take-profit levels must satisfy r1 < r2")
        if self.second_fraction >= 1.0:
            raise ValueError(
                "second_fraction must be less than 1.0 to leave a remainder"
            )
        return self


class TemplateAConfig(StrictModel):
    max_candidates: int = Field(default=15, ge=0)
    max_per_industry: int = Field(default=2, ge=0)
    allowed_boards: list[str] = ["主板", "创业板"]
    universe_min_price: float = Field(default=3.0, ge=0)
    min_listing_days: int = Field(default=120, ge=0)
    min_avg_amount: float = Field(default=100_000_000, ge=0)
    min_price: float = Field(default=5.0, ge=0)
    max_price: float = Field(default=100.0, ge=0)
    market_cap_min: float = Field(default=5_000_000_000, ge=0)
    market_cap_max: float = Field(default=50_000_000_000, ge=0)
    max_distance_from_60d_high: float = Field(default=0.10, ge=0, le=1)
    rs_20d_market_min: float = Field(default=0.70, ge=0, le=1)
    pullback_days_min: int = Field(default=3, ge=0)
    pullback_days_max: int = Field(default=8, ge=0)
    volume_shrink_ratio: float = Field(default=1.0, ge=0)
    ma_zone_pct: float = Field(default=0.02, ge=0, le=1)
    max_ma20_extension: float = Field(default=0.08, ge=0, le=1)
    entry_valid_days: int = Field(default=3, ge=0)
    max_gap_pct: float = Field(default=0.03, ge=0, le=1)
    stop_min_pct: float = Field(default=0.03, ge=0, le=1)
    stop_max_pct: float = Field(default=0.08, ge=0, le=1)
    atr_buffer: float = Field(default=0.5, ge=0)
    time_stop_days: int = Field(default=10, gt=0)
    risk_pct: float = Field(default=0.025, ge=0, le=1)
    take_profit: TakeProfitAConfig
    score: ScoreWeights

    @model_validator(mode="after")
    def _check_bounds(self) -> TemplateAConfig:
        if self.stop_min_pct > self.stop_max_pct:
            raise ValueError("stop_min_pct must be <= stop_max_pct")
        if self.pullback_days_min > self.pullback_days_max:
            raise ValueError("pullback_days_min must be <= pullback_days_max")
        if self.min_price > self.max_price:
            raise ValueError("min_price must be <= max_price")
        if self.market_cap_min > self.market_cap_max:
            raise ValueError("market_cap_min must be <= market_cap_max")
        return self


class TemplateBConfig(StrictModel):
    max_candidates: int = Field(default=8, ge=0)
    max_per_industry: int = Field(default=2, ge=0)
    min_limit_up_in_sector: int = Field(default=3, ge=0)
    sector_red_rate_min: float = Field(default=0.50, ge=0, le=1)
    blow_rate_max: float = Field(default=0.35, ge=0, le=1)
    max_board_min: int = Field(default=3, ge=0)
    volume_ratio_min: float = Field(default=2.0, ge=0)
    turnover_min: float = Field(default=0.05, ge=0, le=1)
    turnover_max: float = Field(default=0.20, ge=0, le=1)
    market_cap_min: float = Field(default=5_000_000_000, ge=0)
    market_cap_max: float = Field(default=20_000_000_000, ge=0)
    price_min: float = Field(default=5.0, ge=0)
    price_max: float = Field(default=50.0, ge=0)
    max_board: int = Field(default=4, ge=0)
    max_20d_gain: float = Field(default=0.80, ge=0, le=1)
    entry_valid_days: int = Field(default=2, ge=0)
    max_gap_pct: float = Field(default=0.03, ge=0, le=1)
    stop_min_pct: float = Field(default=0.03, ge=0, le=1)
    stop_max_pct: float = Field(default=0.05, ge=0, le=1)
    atr_buffer: float = Field(default=0.5, ge=0)
    time_stop_days: int = Field(default=3, gt=0)
    risk_pct: float = Field(default=0.02, ge=0, le=1)
    take_profit: TakeProfitBConfig
    score: ScoreWeights

    @model_validator(mode="after")
    def _check_bounds(self) -> TemplateBConfig:
        if self.stop_min_pct > self.stop_max_pct:
            raise ValueError("stop_min_pct must be <= stop_max_pct")
        if self.turnover_min > self.turnover_max:
            raise ValueError("turnover_min must be <= turnover_max")
        if self.price_min > self.price_max:
            raise ValueError("price_min must be <= price_max")
        if self.market_cap_min > self.market_cap_max:
            raise ValueError("market_cap_min must be <= market_cap_max")
        if self.max_board_min > self.max_board:
            raise ValueError("max_board_min must be <= max_board")
        return self


class RiskConfig(StrictModel):
    max_positions: int = Field(default=3, ge=0)
    max_b_positions: int = Field(default=2, ge=0)
    max_single_position_pct: float = Field(default=0.40, ge=0, le=1)
    min_cash_pct: float = Field(default=0.10, ge=0, le=1)
    max_heat: float = Field(default=0.075, ge=0, le=1)
    max_industry_positions: int = Field(default=2, ge=0)
    correlation_threshold: float = Field(default=0.80, ge=0, le=1)
    drawdown_half_at: float = Field(default=0.10, ge=0, le=1)
    drawdown_stop_at: float = Field(default=0.15, ge=0, le=1)
    round_trip_cost_pct: float = Field(default=0.003, ge=0, le=1)
    commission_rate: float = Field(default=0.00025, ge=0, le=1)
    commission_min: float = Field(default=5.0, ge=0)
    stamp_duty_rate: float = Field(default=0.0005, ge=0, le=1)
    transfer_fee_rate: float = Field(default=0.00001, ge=0, le=1)
    slippage_rate: float = Field(default=0.001, ge=0, le=1)
    b_slippage_rate: float = Field(default=0.002, ge=0, le=1)

    @model_validator(mode="after")
    def _check_drawdown_levels(self) -> RiskConfig:
        if self.drawdown_half_at >= self.drawdown_stop_at:
            raise ValueError("drawdown_half_at must be < drawdown_stop_at")
        return self


class TemplatePolicyConfig(StrictModel):
    enabled: bool
    mode: Literal["normal", "pullback_only", "observe"] = "normal"
    max_positions: int = Field(default=0, ge=0)


class StagePolicyConfig(StrictModel):
    a: TemplatePolicyConfig
    b: TemplatePolicyConfig
    risk_factor: float = Field(ge=0, le=1)


class SwitchingConfig(StrictModel):
    freeze: StagePolicyConfig
    start: StagePolicyConfig
    ferment: StagePolicyConfig
    climax: StagePolicyConfig
    ebb: StagePolicyConfig


class StrategyConfig(StrictModel):
    name: str
    version: str
    environment: EnvironmentConfig
    industry: IndustryConfig
    leader: LeaderConfig
    template_a: TemplateAConfig
    template_b: TemplateBConfig
    switching: SwitchingConfig
    risk: RiskConfig

    @model_validator(mode="after")
    def _check_stage_policies(self) -> StrategyConfig:
        for stage_name in ("freeze", "start", "ferment", "climax", "ebb"):
            stage = getattr(self.switching, stage_name)
            for template_name, limit in (
                ("a", self.risk.max_positions),
                ("b", self.risk.max_b_positions),
            ):
                policy = getattr(stage, template_name)
                if policy.max_positions > limit:
                    raise ValueError(
                        f"{stage_name}.{template_name}.max_positions must be <= risk max ({limit})"
                    )
                if not policy.enabled and policy.max_positions != 0:
                    raise ValueError(
                        f"{stage_name}.{template_name}.max_positions must be 0 when disabled"
                    )
        return self


class DataConfig(StrictModel):
    provider: Literal["tushare"] = "tushare"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    regime_dir: str = "data/regime"
    history_start: str = "2018-01-01"
    rate_limit_per_min: int = Field(default=180, ge=0)


class AppConfig(StrictModel):
    data: DataConfig
    strategy_path: str = "configs/strategies/dual_template_v1.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    return payload


def load_app_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else BACKEND_ROOT / "configs" / "app.yaml"
    return AppConfig.model_validate(_load_yaml(config_path))


def load_strategy_config(path: Path | str | None = None) -> StrategyConfig:
    config_path = (
        Path(path)
        if path is not None
        else BACKEND_ROOT / "configs" / "strategies" / "dual_template_v1.yaml"
    )
    return StrategyConfig.model_validate(_load_yaml(config_path))
