from __future__ import annotations

import math
from decimal import Decimal

from app.core.config import StrategyConfig
from app.domain.enums import PlanStatus, RiskOutcome, SentimentStage, Template
from app.domain.models import (
    AccountState,
    MarketRegime,
    PendingExposure,
    RiskDecision,
    TemplatePolicy,
    TradePlan,
)
from app.domain.money import money

_ZERO = Decimal("0")
_ONE = Decimal("1")
_LOT = Decimal("100")
_HALT_REASON = "drawdown circuit breaker active: manual restart required"


def _dec(value: float | int | str | Decimal) -> Decimal:
    return Decimal(str(value))


def _lots(amount: Decimal, unit: Decimal) -> int:
    """Largest 100-share-aligned quantity affordable by ``amount`` at ``unit``."""
    if amount <= 0 or unit <= 0:
        return 0
    return int(amount / unit // _LOT) * 100


class RiskGovernor:
    """Final portfolio-level review of a batch of trade plans.

    Plans are processed in priority order: Template B first during the START
    stage, then by descending candidate score.  Capacity and risk caps are
    evaluated against the evolving account, so plans approved earlier in the
    batch consume slots, heat, cash, and exposure for later plans.

    Template A is capped by its regime policy, Template B by
    ``min(config.risk.max_b_positions, policy.max_positions)``, and the whole
    batch by ``config.risk.max_positions``.  A set ``trading_halted`` account
    latch (or an instantaneous drawdown at the stop level) rejects every
    plan; a half-risk latch does not block approval because the position
    sizer has already halved the risk budget.

    ``correlations`` maps a candidate code to ``{other_code: correlation}``.
    A plan correlated above ``config.risk.correlation_threshold`` with an
    existing position, a pending plan, or an already-approved plan in the
    batch is rejected, so the higher-priority plan is retained.  Correlation
    is symmetric; a value supplied in either direction is honoured.

    Size reductions name the binding cap in ``RiskDecision.reasons``.
    """

    def apply(
        self,
        plans: list[TradePlan],
        account: AccountState,
        regime: MarketRegime,
        config: StrategyConfig,
        correlations: dict[str, dict[str, float]] | None = None,
    ) -> list[TradePlan]:
        if self._halted(account, config):
            return [self._reject(plan, _HALT_REASON) for plan in plans]

        ordered = sorted(plans, key=lambda plan: self._priority(plan, regime))
        working = account
        approved_codes: set[str] = set()
        seen_codes: set[str] = {
            item.ts_code for item in account.positions
        } | {item.ts_code for item in account.pending_plans}
        output: list[TradePlan] = []

        for plan in ordered:
            code = plan.candidate.ts_code
            if code in seen_codes:
                output.append(self._reject(plan, f"duplicate plan for {code}"))
                continue
            seen_codes.add(code)

            invalid = self._invalid_plan_reason(plan)
            if invalid is not None:
                output.append(self._reject(plan, invalid))
                continue
            if plan.data_version != regime.data_version:
                output.append(self._reject(plan, "data version mismatch"))
                continue

            template = plan.candidate.template
            policy = self._policy(regime, template)
            if policy is None:
                output.append(
                    self._reject(plan, "template policy missing from market regime")
                )
                continue
            if policy.max_positions < 0:
                output.append(self._reject(plan, "invalid template policy"))
                continue
            if not policy.enabled:
                output.append(self._reject(plan, "template disabled"))
                continue

            positions = working.positions
            pending = working.pending_plans
            occupied = len(positions) + len(pending)
            if occupied >= config.risk.max_positions:
                output.append(self._reject(plan, "max positions reached"))
                continue

            if template == Template.A:
                a_count = sum(
                    1 for item in positions if item.template == Template.A
                ) + sum(
                    1 for item in pending if item.template == Template.A
                )
                if a_count >= policy.max_positions:
                    output.append(
                        self._reject(plan, "max A policy positions reached")
                    )
                    continue
            elif template == Template.B:
                b_count = sum(
                    1 for item in positions if item.template == Template.B
                ) + sum(
                    1 for item in pending if item.template == Template.B
                )
                b_limit = min(
                    config.risk.max_b_positions, policy.max_positions
                )
                if b_count >= b_limit:
                    output.append(
                        self._reject(plan, "max B policy positions reached")
                    )
                    continue
            else:
                output.append(
                    self._reject(plan, f"unsupported template {template!r}")
                )
                continue

            industry_count = sum(
                1 for item in positions if item.sector == plan.candidate.sector
            ) + sum(
                1 for item in pending if item.sector == plan.candidate.sector
            )
            if industry_count >= config.risk.max_industry_positions:
                output.append(
                    self._reject(plan, "max industry positions reached")
                )
                continue

            correlation_reason = self._correlation_reason(
                plan, working, approved_codes, correlations, config
            )
            if correlation_reason is not None:
                output.append(self._reject(plan, correlation_reason))
                continue

            entry_price = Decimal(plan.entry_plan.entry_price)
            stop_price = Decimal(plan.stop_plan.initial_stop)
            equity = Decimal(working.equity)
            capped = self._capped_shares(
                plan, working, entry_price, stop_price, config
            )
            if isinstance(capped, str):
                output.append(self._reject(plan, capped))
                continue
            shares, binding_caps = capped
            if shares < 100:
                output.append(
                    self._reject(plan, "position below one lot after caps")
                )
                continue

            if shares < plan.position_plan.shares:
                decision = RiskDecision(
                    outcome=RiskOutcome.REDUCE,
                    reasons=[
                        f"{name} reduced size" for name in binding_caps
                    ],
                    adjusted_shares=shares,
                )
            else:
                decision = RiskDecision(
                    outcome=RiskOutcome.APPROVE,
                    reasons=["all checks passed"],
                )

            position_value = money(entry_price * shares)
            new_risk = (entry_price - stop_price) * shares
            heat_after = float(
                _dec(working.heat) + new_risk / equity
            )
            position = plan.position_plan.model_copy(
                update={
                    "shares": shares,
                    "position_value": position_value,
                    "position_pct": float(position_value) / float(equity),
                    "heat_after": heat_after,
                }
            )
            approved = plan.model_copy(
                update={
                    "position_plan": position,
                    "risk_decision": decision,
                    "status": PlanStatus.PENDING_ENTRY,
                }
            )
            output.append(approved)
            approved_codes.add(code)
            working = working.with_pending(
                PendingExposure(
                    ts_code=code,
                    sector=plan.candidate.sector,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    shares=shares,
                    template=template,
                )
            )
        return output

    @staticmethod
    def _capped_shares(
        plan: TradePlan,
        working: AccountState,
        entry_price: Decimal,
        stop_price: Decimal,
        config: StrategyConfig,
    ) -> tuple[int, list[str]] | str:
        """Return capped shares and the binding caps, or a rejection reason."""
        equity = Decimal(working.equity)
        cash = Decimal(working.cash)
        min_cash = equity * _dec(config.risk.min_cash_pct)
        shares = plan.position_plan.shares
        binding: list[str] = []

        max_single_value = equity * _dec(config.risk.max_single_position_pct)
        shares = RiskGovernor._apply_cap(
            shares,
            _lots(max_single_value, entry_price),
            "single-position cap",
            binding,
        )

        pending_notional = sum(
            (item.entry_price * item.shares for item in working.pending_plans),
            _ZERO,
        )
        max_cash_spend = cash - pending_notional - min_cash
        if max_cash_spend <= 0:
            return "cash reserve limit reached"
        shares = RiskGovernor._apply_cap(
            shares,
            _lots(max_cash_spend, entry_price),
            "cash reserve",
            binding,
        )

        existing_exposure = working.positions_value + pending_notional
        max_exposure = equity - min_cash
        remaining_exposure = max_exposure - existing_exposure
        if remaining_exposure <= 0:
            return "total exposure limit reached"
        shares = RiskGovernor._apply_cap(
            shares,
            _lots(remaining_exposure, entry_price),
            "exposure cap",
            binding,
        )

        max_heat = equity * _dec(config.risk.max_heat)
        current_heat = _dec(working.heat) * equity
        max_new_risk = max_heat - current_heat
        if max_new_risk <= 0:
            return "portfolio heat limit reached"
        shares = RiskGovernor._apply_cap(
            shares,
            _lots(max_new_risk, entry_price - stop_price),
            "heat cap",
            binding,
        )
        return shares, binding

    @staticmethod
    def _apply_cap(
        current: int,
        proposed: int,
        name: str,
        binding: list[str],
    ) -> int:
        if proposed >= current:
            return current
        binding.append(name)
        return proposed

    @staticmethod
    def _invalid_plan_reason(plan: TradePlan) -> str | None:
        if plan.position_plan.shares <= 0:
            return "invalid plan: shares must be positive"
        position_value = plan.position_plan.position_value
        if not position_value.is_finite() or position_value <= 0:
            return "invalid plan: position value must be positive"
        try:
            entry_price = Decimal(plan.entry_plan.entry_price)
        except (ArithmeticError, ValueError):
            return "invalid plan: entry price is unavailable"
        stop_price = Decimal(plan.stop_plan.initial_stop)
        if not entry_price.is_finite() or entry_price <= 0:
            return "invalid plan: entry price must be positive"
        if not stop_price.is_finite() or stop_price <= 0:
            return "invalid plan: stop must be positive"
        if entry_price <= stop_price:
            return "invalid plan: entry price must be above stop"
        return None

    @staticmethod
    def _policy(
        regime: MarketRegime, template: Template
    ) -> TemplatePolicy | None:
        return regime.template_availability.get(template)

    @staticmethod
    def _correlation_reason(
        plan: TradePlan,
        working: AccountState,
        approved_codes: set[str],
        correlations: dict[str, dict[str, float]] | None,
        config: StrategyConfig,
    ) -> str | None:
        if correlations is None:
            return None
        held_codes = (
            {item.ts_code for item in working.positions}
            | {item.ts_code for item in working.pending_plans}
            | approved_codes
        )
        candidate_code = plan.candidate.ts_code
        candidate_correlations = correlations.get(candidate_code, {})
        threshold = config.risk.correlation_threshold
        for held_code in sorted(held_codes):
            value = candidate_correlations.get(held_code)
            if value is None:
                value = correlations.get(held_code, {}).get(candidate_code)
            if value is None:
                continue
            if not math.isfinite(value) or not -1 <= value <= 1:
                return f"invalid correlation for {held_code}: {value!r}"
            if value > threshold:
                return (
                    f"correlation above threshold with {held_code}: "
                    f"{value} > {threshold}"
                )
        return None

    @staticmethod
    def _halted(account: AccountState, config: StrategyConfig) -> bool:
        if account.trading_halted:
            return True
        equity = Decimal(account.equity)
        peak_equity = Decimal(account.peak_equity)
        drawdown = max(_ZERO, _ONE - equity / peak_equity)
        return drawdown >= _dec(config.risk.drawdown_stop_at)

    @staticmethod
    def _priority(plan: TradePlan, regime: MarketRegime) -> tuple[int, float]:
        score = -plan.candidate.score
        if (
            regime.sentiment_stage == SentimentStage.START
            and plan.candidate.template == Template.B
        ):
            return (0, score)
        return (1, score)

    @staticmethod
    def _reject(plan: TradePlan, reason: str) -> TradePlan:
        return plan.model_copy(
            update={
                "status": PlanStatus.REJECTED,
                "risk_decision": RiskDecision(
                    outcome=RiskOutcome.REJECT, reasons=[reason]
                ),
            }
        )
