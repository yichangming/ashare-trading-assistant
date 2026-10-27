from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import (
    EntryType,
    PlanStatus,
    RiskOutcome,
    SentimentStage,
    Template,
    TrendRegime,
)
from app.domain.money import price


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemplatePolicy(DomainModel):
    enabled: bool
    mode: Literal["normal", "pullback_only", "observe"] = "normal"
    max_positions: int = Field(default=0, ge=0)


class RegimeIndicators(DomainModel):
    limit_up_count: int = Field(ge=0)
    limit_down_count: int = Field(ge=0)
    max_board_height: int = Field(ge=0)
    promotion_rate: float = Field(ge=0, le=1)
    blow_rate: float = Field(ge=0, le=1)
    red_rate: float = Field(ge=0, le=1)
    limit_up_premium: float
    breadth: float = Field(ge=0, le=1)
    index_close: float = Field(gt=0)
    index_ma20: float = Field(gt=0)
    index_ma60: float = Field(gt=0)


class SectorStrength(DomainModel):
    sector: str
    score: float = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    return_20d: float
    return_5d: float
    breadth: float = Field(ge=0, le=1)
    amount_change: float
    limit_up_share: float = Field(ge=0, le=1)
    limit_up_count: int = Field(ge=0)
    red_rate: float = Field(ge=0, le=1)
    stock_count: int = Field(ge=0)
    persistent_days: int = Field(ge=0)


class MarketRegime(DomainModel):
    trade_date: date
    trend_regime: TrendRegime
    sentiment_stage: SentimentStage
    risk_factor: float = Field(ge=0, le=1)
    template_availability: dict[Template, TemplatePolicy]
    indicators: RegimeIndicators
    sector_ranking: list[SectorStrength] = Field(default_factory=list)
    degraded_flags: list[str] = Field(default_factory=list)
    data_version: str

    @model_validator(mode="after")
    def _both_templates(self) -> MarketRegime:
        if set(self.template_availability) != {Template.A, Template.B}:
            raise ValueError("template_availability must contain A and B")
        return self


class Candidate(DomainModel):
    ts_code: str
    trade_date: date
    template: Template
    setup_type: str
    horizon: Literal["short", "mid"]
    sector: str
    leader_rank: int = Field(ge=1)
    leader_score: float = Field(ge=0, le=100)
    score: float = Field(ge=0, le=100)
    reasons: list[str] = Field(min_length=1)
    features: dict[str, float]
    sentiment_stage: SentimentStage
    strategy_version: str
    data_version: str


class EntryPlan(DomainModel):
    type: EntryType
    trigger_price: Decimal | None = Field(default=None, gt=0)
    entry_zone_low: Decimal | None = Field(default=None, gt=0)
    entry_zone_high: Decimal | None = Field(default=None, gt=0)
    validity_days: int = Field(gt=0)
    max_chase_pct: float = Field(ge=0)
    cancel_conditions: list[str] = Field(default_factory=list)
    expires_on: date

    @model_validator(mode="after")
    def _prices_present(self) -> EntryPlan:
        if (self.entry_zone_low is None) != (self.entry_zone_high is None):
            raise ValueError("both entry_zone_low and entry_zone_high are required")
        if self.trigger_price is None and self.entry_zone_low is None:
            raise ValueError("trigger_price or a full entry zone is required")
        if (
            self.entry_zone_low is not None
            and self.entry_zone_high is not None
            and self.entry_zone_low > self.entry_zone_high
        ):
            raise ValueError("entry_zone_low must be <= entry_zone_high")
        return self

    @property
    def entry_price(self) -> Decimal:
        if self.entry_zone_low is not None and self.entry_zone_high is not None:
            return price((self.entry_zone_low + self.entry_zone_high) / 2)
        if self.trigger_price is None:
            raise ValueError("entry price is unavailable")
        return price(self.trigger_price)


class StopPlan(DomainModel):
    initial_stop: Decimal = Field(gt=0)
    stop_distance_pct: float = Field(gt=0)
    atr14: float = Field(gt=0)
    structure_ref: str
    time_stop_days: int = Field(gt=0)
    trailing_rules: list[str]
    r_value: Decimal = Field(gt=0)


class PositionPlan(DomainModel):
    template: Template
    risk_pct: float = Field(gt=0, le=1)
    risk_amount: Decimal = Field(ge=0)
    shares: int = Field(ge=0)
    position_value: Decimal = Field(ge=0)
    position_pct: float = Field(ge=0, le=1)
    heat_before: float = Field(ge=0)
    heat_after: float = Field(ge=0)
    constraints_checked: list[str] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list)


class TakeProfitTarget(DomainModel):
    r_multiple: float = Field(gt=0)
    price: Decimal = Field(gt=0)
    fraction: float = Field(gt=0, le=1)


class TakeProfitPlan(DomainModel):
    targets: list[TakeProfitTarget] = Field(min_length=1)
    trailing_rule: str
    time_exit_rule: str


class RiskDecision(DomainModel):
    outcome: RiskOutcome
    reasons: list[str] = Field(min_length=1)
    adjusted_shares: int | None = Field(default=None, ge=0)


class PendingExposure(DomainModel):
    ts_code: str
    sector: str
    entry_price: Decimal = Field(gt=0)
    stop_price: Decimal = Field(gt=0)
    shares: int = Field(ge=0)
    template: Template

    @model_validator(mode="after")
    def _stop_below_entry(self) -> PendingExposure:
        if self.stop_price >= self.entry_price:
            raise ValueError("stop_price must be below entry_price")
        return self


class PositionState(DomainModel):
    ts_code: str
    sector: str
    shares: int = Field(ge=0)
    available_shares: int = Field(ge=0)
    cost_price: Decimal = Field(gt=0)
    current_stop: Decimal = Field(gt=0)
    current_price: Decimal = Field(gt=0)
    template: Template


class AccountState(DomainModel):
    equity: Decimal = Field(gt=0)
    cash: Decimal = Field(ge=0)
    peak_equity: Decimal = Field(gt=0)
    positions: list[PositionState] = Field(default_factory=list)
    pending_plans: list[PendingExposure] = Field(default_factory=list)

    @property
    def drawdown_pct(self) -> float:
        return max(0.0, 1.0 - float(self.equity / self.peak_equity))

    @property
    def positions_value(self) -> Decimal:
        return sum(
            (position.current_price * position.shares for position in self.positions),
            Decimal("0"),
        )

    @property
    def heat(self) -> float:
        equity = float(self.equity)
        if equity <= 0:
            return 0.0
        open_risk = sum(
            max(0.0, float(position.current_price - position.current_stop)) * position.shares
            for position in self.positions
        )
        pending_risk = sum(
            max(0.0, float(exposure.entry_price - exposure.stop_price)) * exposure.shares
            for exposure in self.pending_plans
        )
        return (open_risk + pending_risk) / equity

    def with_pending(self, exposure: PendingExposure) -> AccountState:
        return self.model_copy(update={"pending_plans": [*self.pending_plans, exposure]})


class TradePlan(DomainModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    candidate: Candidate
    entry_plan: EntryPlan
    stop_plan: StopPlan
    position_plan: PositionPlan
    take_profit_plan: TakeProfitPlan
    risk_decision: RiskDecision | None = None
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_on: date
    strategy_version: str
    data_version: str
    code_version: str
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _entry_above_stop(self) -> TradePlan:
        if self.entry_plan.entry_price <= self.stop_plan.initial_stop:
            raise ValueError("entry price must be above initial stop")
        return self
