from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FreezeThresholds(StrictModel):
    limit_up_max: int
    limit_down_min: int
    blow_rate_min: float
    max_board_max: int
    red_rate_max: float


class StartThresholds(StrictModel):
    limit_up_min: int
    limit_up_max: int
    max_board_min: int
    max_board_max: int
    blow_rate_min: float
    blow_rate_max: float
    red_rate_min: float
    red_rate_max: float


class FermentThresholds(StrictModel):
    limit_up_min: int
    max_board_min: int
    blow_rate_max: float
    red_rate_min: float


class ClimaxThresholds(StrictModel):
    limit_up_min: int
    max_board_min: int
    red_rate_min: float
    blow_rate_rise_pp: float
    premium_drop_pp: float


class EbbThresholds(StrictModel):
    limit_up_max: int
    limit_down_min: int
    blow_rate_min: float
    red_rate_max: float


class EnvironmentThresholds(StrictModel):
    freeze: FreezeThresholds
    start: StartThresholds
    ferment: FermentThresholds
    climax: ClimaxThresholds
    ebb: EbbThresholds


class EnvironmentConfig(StrictModel):
    index_code: str
    fallback_index_code: str
    ma_fast: int = 20
    ma_slow: int = 60
    breadth_attack: float = 0.55
    breadth_defense: float = 0.35
    stage_confirm_days: int = 2
    thresholds: EnvironmentThresholds


class IndustryWeights(StrictModel):
    return_20d: float
    return_5d: float
    breadth: float
    amount_change: float
    limit_up_share: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> IndustryWeights:
        total = (
            self.return_20d
            + self.return_5d
            + self.breadth
            + self.amount_change
            + self.limit_up_share
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"industry weights must sum to 1.0, got {total}")
        return self


class IndustryConfig(StrictModel):
    top_n: int = 3
    persistence_days: int = 2
    min_stocks: int = 5
    source: Literal["auto", "sw", "stock_basic"] = "auto"
    weights: IndustryWeights


class LeaderWeights(StrictModel):
    rs_20d: float
    rs_5d: float
    amount_turnover: float
    high_distance: float
    limit_up_status: float
    size_price_fit: float

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
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"leader weights must sum to 1.0, got {total}")
        return self


class LeaderConfig(StrictModel):
    top_n_per_industry: int = 2
    follower_rank_cutoff: int = 3
    amount_vs_leader_min: float = 0.5
    weights: LeaderWeights


class ScoreWeights(StrictModel):
    leader: float
    setup: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> ScoreWeights:
        if abs(self.leader + self.setup - 1.0) > 1e-6:
            raise ValueError("score weights must sum to 1.0")
        return self


class TakeProfitAConfig(StrictModel):
    r1: float = 1.0
    r2: float = 2.0
    r3: float = 3.0
    first_fraction: float = 1 / 3
    second_fraction: float = 1 / 3


class TakeProfitBConfig(StrictModel):
    r1: float = 1.0
    r2: float = 2.0
    second_fraction: float = 0.5


class TemplateAConfig(StrictModel):
    max_candidates: int = 15
    max_per_industry: int = 2
    allowed_boards: list[str] = ["主板", "创业板"]
    universe_min_price: float = 3.0
    min_listing_days: int = 120
    min_avg_amount: float = 100_000_000
    min_price: float = 5.0
    max_price: float = 100.0
    market_cap_min: float = 5_000_000_000
    market_cap_max: float = 50_000_000_000
    max_distance_from_60d_high: float = 0.10
    rs_20d_market_min: float = 0.70
    pullback_days_min: int = 3
    pullback_days_max: int = 8
    volume_shrink_ratio: float = 1.0
    ma_zone_pct: float = 0.02
    max_ma20_extension: float = 0.08
    entry_valid_days: int = 3
    max_gap_pct: float = 0.03
    stop_min_pct: float = 0.03
    stop_max_pct: float = 0.08
    atr_buffer: float = 0.5
    time_stop_days: int = 10
    risk_pct: float = 0.025
    take_profit: TakeProfitAConfig
    score: ScoreWeights


class TemplateBConfig(StrictModel):
    max_candidates: int = 8
    max_per_industry: int = 2
    min_limit_up_in_sector: int = 3
    sector_red_rate_min: float = 0.50
    blow_rate_max: float = 0.35
    max_board_min: int = 3
    volume_ratio_min: float = 2.0
    turnover_min: float = 0.05
    turnover_max: float = 0.20
    market_cap_min: float = 5_000_000_000
    market_cap_max: float = 20_000_000_000
    price_min: float = 5.0
    price_max: float = 50.0
    max_board: int = 4
    max_20d_gain: float = 0.80
    entry_valid_days: int = 2
    max_gap_pct: float = 0.03
    stop_min_pct: float = 0.03
    stop_max_pct: float = 0.05
    time_stop_days: int = 3
    risk_pct: float = 0.02
    take_profit: TakeProfitBConfig
    score: ScoreWeights


class RiskConfig(StrictModel):
    max_positions: int = 3
    max_b_positions: int = 2
    max_single_position_pct: float = 0.40
    min_cash_pct: float = 0.10
    max_heat: float = 0.075
    max_industry_positions: int = 2
    correlation_threshold: float = 0.80
    drawdown_half_at: float = 0.10
    drawdown_stop_at: float = 0.15
    round_trip_cost_pct: float = 0.003
    commission_rate: float = 0.00025
    commission_min: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_rate: float = 0.001
    b_slippage_rate: float = 0.002


class TemplatePolicyConfig(StrictModel):
    enabled: bool
    mode: Literal["normal", "pullback_only", "observe"] = "normal"
    max_positions: int = 0


class StagePolicyConfig(StrictModel):
    a: TemplatePolicyConfig
    b: TemplatePolicyConfig
    risk_factor: float


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


class DataConfig(StrictModel):
    provider: str = "tushare"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    regime_dir: str = "data/regime"
    history_start: str = "2018-01-01"
    rate_limit_per_min: int = 180


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
