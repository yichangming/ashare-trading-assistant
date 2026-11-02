from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import (
    AccountMode,
    EntryType,
    GlobalRiskMode,
    PermissionMode,
    PlanStatus,
    PositionAction,
    RiskOutcome,
    SentimentStage,
    Strategy,
    SubMode,
    Template,
    TrendRegime,
    ValuationStatus,
)
from app.domain.money import price

_ZERO = Decimal("0")
_ONE = Decimal("1")


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


class StrategyPermission(DomainModel):
    strategy: Strategy
    mode: PermissionMode
    risk_factor: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    data_time: datetime | None = None


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
    strategy_permissions: dict[Strategy, StrategyPermission] = Field(
        default_factory=dict
    )
    global_risk: GlobalRiskMode = GlobalRiskMode.NORMAL
    risk_reasons: list[str] = Field(default_factory=list)
    breadth_label: str = "normal"
    volatility_label: str = "normal"
    liquidity_label: str = "normal"
    decline_diffusion: float = 1.0

    @model_validator(mode="after")
    def _both_templates(self) -> MarketRegime:
        if set(self.template_availability) != {Template.A, Template.B}:
            raise ValueError("template_availability must contain A and B")
        return self


class Candidate(DomainModel):
    ts_code: str
    trade_date: date
    template: Template = Template.A
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
    strategy: Strategy | None = None
    sub_mode: SubMode | None = None
    permission: PermissionMode = PermissionMode.NORMAL
    evidence: dict[str, Any] = Field(default_factory=dict)
    data_time: datetime | None = None


class EntryPlan(DomainModel):
    type: EntryType
    trigger_price: Decimal | None = Field(default=None, gt=0)
    entry_zone_low: Decimal | None = Field(default=None, gt=0)
    entry_zone_high: Decimal | None = Field(default=None, gt=0)
    validity_days: int = Field(gt=0)
    validity_minutes: int | None = Field(default=None, gt=0)
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
    is_add_on: bool = False


class TakeProfitTarget(DomainModel):
    r_multiple: float = Field(gt=0)
    price: Decimal = Field(gt=0)
    fraction: float = Field(gt=0, le=1)


class TakeProfitPlan(DomainModel):
    targets: list[TakeProfitTarget] = Field(min_length=1)
    trailing_rule: str
    time_exit_rule: str


class ExitRules(DomainModel):
    """Executable exit rules.

    The planner writes these fields; runtime code never parses the display
    strings in ``trailing_rule``/``time_exit_rule`` to decide a trade.
    """

    trailing: Literal[
        "breakeven_1r_2r_atr",
        "breakeven_1r_2r_ma5_atr",
        "breakeven_1r_2r_range_atr",
        "breakeven_1r_atr",
    ] = "breakeven_1r_2r_atr"
    time_exit: Literal[
        "none",
        "below_entry",
        "no_new_high",
        "event_expiry",
    ] = "below_entry"
    time_stop_days: int = Field(default=10, ge=1)
    event_expiry_days: int | None = Field(default=None, ge=1)


class TargetState(DomainModel):
    price: Decimal = Field(gt=0)
    fraction: float = Field(gt=0, le=1)
    planned_shares: int = Field(ge=0)
    filled_shares: int = Field(ge=0)
    status: Literal["pending", "partial", "done"] = "pending"

    @property
    def remaining_shares(self) -> int:
        return max(0, self.planned_shares - self.filled_shares)


class PositionDecision(DomainModel):
    """Structured, auditable position action shared by every surface."""

    ts_code: str
    sector: str = "未知"
    primary_strategy: Strategy | None = None
    sub_mode: SubMode | None = None
    rule_version: str = ""
    action: PositionAction
    reason_code: str
    explanation: str
    trigger_price: Decimal | None = Field(default=None, ge=0)
    trigger_time: datetime | None = None
    data_time: datetime | None = None
    valid_until: datetime | None = None
    total_shares: int = Field(default=0, ge=0)
    available_shares: int = Field(default=0, ge=0)
    suggested_shares: int = Field(default=0, ge=0)
    restricted_shares: int = Field(default=0, ge=0)
    current_stop: Decimal | None = Field(default=None, ge=0)
    proposed_stop: Decimal | None = Field(default=None, ge=0)
    targets: list[TargetState] = Field(default_factory=list)
    invalidation: str = ""
    confirmed: bool = False
    executed_shares: int = Field(default=0, ge=0)
    pending_shares: int = Field(default=0, ge=0)
    valuation_status: ValuationStatus = ValuationStatus.UNKNOWN
    needs_reconciliation: bool = False
    data_issues: list[str] = Field(default_factory=list)

    @property
    def is_risk(self) -> bool:
        return self.action in {
            PositionAction.EXIT,
            PositionAction.REDUCE,
            PositionAction.DATA_ANOMALY,
        }


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
    template: Template = Template.A
    expires_on: date | None = None
    strategy: Strategy | None = None
    sub_mode: SubMode | None = None

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
    current_stop: Decimal = Field(ge=0)
    current_price: Decimal = Field(ge=0)
    template: Template = Template.A
    strategy: Strategy | None = None
    sub_mode: SubMode | None = None
    primary_strategy: Strategy | None = None
    strategy_version: str | None = None
    initial_r: Decimal | None = None
    initial_shares: int = Field(default=0, ge=0)
    add_on_count: int = Field(default=0, ge=0)
    needs_reconciliation: bool = False
    valuation_status: ValuationStatus = ValuationStatus.UNKNOWN
    valuation_at: datetime | None = None
    data_time: datetime | None = None
    targets: list[TargetState] = Field(default_factory=list)


class AccountState(DomainModel):
    equity: Decimal = Field(gt=0)
    cash: Decimal = Field(ge=0)
    peak_equity: Decimal = Field(gt=0)
    positions: list[PositionState] = Field(default_factory=list)
    pending_plans: list[PendingExposure] = Field(default_factory=list)
    trading_halted: bool = False
    drawdown_half_latched: bool = False
    mode: AccountMode = AccountMode.PAPER
    risk_config_configured: bool = False
    max_drawdown: float = Field(default=0.15, gt=0, le=1)
    net_deposits: Decimal = Decimal("0")
    valuation_incomplete: bool = False
    valuation_note: str | None = None
    last_synced_at: datetime | None = None
    last_valuation_at: datetime | None = None
    last_reconciled_at: datetime | None = None
    stale: bool = False

    @property
    def performance_equity(self) -> Decimal:
        """Equity excluding external cash flows.

        Deposits increase cash but are not profit; withdrawals are added
        back.  This is the documented cash-flow adjustment used by drawdown
        and peak tracking.
        """
        return self.equity - self.net_deposits

    @property
    def drawdown_pct(self) -> float:
        performance = float(self.performance_equity)
        peak = float(self.peak_equity)
        if peak <= 0:
            return 0.0
        return max(0.0, 1.0 - performance / peak)

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

    def with_risk_latches(
        self,
        stop_at: float | Decimal,
        half_at: float | Decimal,
        recovery_at: float | Decimal = 0.05,
    ) -> AccountState:
        """Return a copy with drawdown latches updated.

        Drawdown is computed with :class:`~decimal.Decimal` so the exact
        10% and 15% boundaries are not affected by float representation.
        The stop latch only ever transitions to ``True``; clearing it
        requires an explicit manual restart.  The half-risk latch sets at
        ``half_at`` and clears once drawdown falls below ``recovery_at``.
        """
        drawdown = max(
            _ZERO, _ONE - self.performance_equity / self.peak_equity
        )
        stop_threshold = Decimal(str(stop_at))
        half_threshold = Decimal(str(half_at))
        recovery_threshold = Decimal(str(recovery_at))

        trading_halted = self.trading_halted or drawdown >= stop_threshold
        if drawdown >= half_threshold:
            drawdown_half_latched = True
        elif drawdown < recovery_threshold:
            drawdown_half_latched = False
        else:
            drawdown_half_latched = self.drawdown_half_latched
        return self.model_copy(
            update={
                "trading_halted": trading_halted,
                "drawdown_half_latched": drawdown_half_latched,
            }
        )


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
    expires_at: datetime | None = None
    exit_rules: ExitRules | None = None
    strategy_version: str
    data_version: str
    code_version: str
    reasons: list[str] = Field(default_factory=list)
    strategy: Strategy | None = None
    sub_mode: SubMode | None = None
    signal_time: datetime | None = None
    data_time: datetime | None = None
    confirmation_price: Decimal | None = Field(default=None, gt=0)
    max_acceptable_price: Decimal | None = Field(default=None, gt=0)
    execution_low: Decimal | None = Field(default=None, gt=0)
    execution_high: Decimal | None = Field(default=None, gt=0)
    cancel_reason: str | None = None
    target_status: Literal["known", "unknown"] = "known"
    risk_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _entry_above_stop(self) -> TradePlan:
        if self.entry_plan.entry_price <= self.stop_plan.initial_stop:
            raise ValueError("entry price must be above initial stop")
        return self
