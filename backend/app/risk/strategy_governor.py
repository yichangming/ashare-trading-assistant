"""Portfolio gate for the six-strategy system."""

from __future__ import annotations

from app.core.full_config import FullSystemConfig
from app.domain.enums import (
    AccountMode,
    PermissionMode,
    PlanStatus,
    RiskOutcome,
    Strategy,
    Template,
)
from app.domain.models import (
    AccountState,
    MarketRegime,
    PendingExposure,
    RiskDecision,
    TradePlan,
)
from app.domain.money import money

STRATEGY_PRIORITY = {
    Strategy.A: 10,
    Strategy.C: 20,
    Strategy.E: 30,
    Strategy.F: 40,
    Strategy.D: 50,
    Strategy.B: 60,
}


def _lots(value: float) -> int:
    return int(value // 100 * 100)


class StrategyRiskGovernor:
    def apply(
        self,
        plans: list[TradePlan],
        account: AccountState,
        regime: MarketRegime,
        config: FullSystemConfig,
        correlations: dict[str, dict[str, float]] | None = None,
    ) -> list[TradePlan]:
        if account.trading_halted:
            return [
                self._reject(plan, "trading halted: manual restart required")
                for plan in plans
            ]
        if account.drawdown_pct >= config.risk.drawdown_stop_at:
            return [
                self._reject(plan, "drawdown circuit breaker active")
                for plan in plans
            ]
        if account.mode == AccountMode.REAL and account.stale:
            return [
                self._reject(
                    plan, "account data stale; reconcile positions first"
                )
                for plan in plans
            ]

        def priority(plan: TradePlan) -> tuple:
            strategy = plan.strategy or Strategy.A
            return (
                STRATEGY_PRIORITY.get(strategy, 99),
                -plan.candidate.score,
                plan.candidate.ts_code,
            )

        ordered = sorted(plans, key=priority)
        working = account
        approved_codes: set[str] = set()
        output: list[TradePlan] = []

        for plan in ordered:
            strategy = plan.strategy or Strategy.A
            permission = regime.strategy_permissions.get(strategy)
            if permission is not None and permission.mode == PermissionMode.PAUSED:
                output.append(self._reject(plan, "strategy permission paused"))
                continue
            if plan.data_version != regime.data_version:
                output.append(self._reject(plan, "data version mismatch"))
                continue
            if plan.candidate.ts_code in approved_codes:
                output.append(self._reject(plan, "same stock already planned"))
                continue
            if plan.status != PlanStatus.DRAFT:
                output.append(self._reject(plan, "plan is not DRAFT"))
                continue
            if plan.position_plan.shares % 100 != 0:
                output.append(self._reject(plan, "shares are not a full lot"))
                continue
            existing = next(
                (
                    item
                    for item in working.positions
                    if item.ts_code == plan.candidate.ts_code
                ),
                None,
            )
            is_add_on = existing is not None
            if is_add_on:
                assert existing is not None
                if strategy not in config.risk.add_on_strategies:
                    output.append(
                        self._reject(
                            plan,
                            f"add-on not allowed for strategy {strategy.value}",
                        )
                    )
                    continue
                if existing.add_on_count >= config.risk.add_on_max:
                    output.append(self._reject(plan, "maximum add-ons reached"))
                    continue
                if existing.current_price < existing.cost_price:
                    output.append(
                        self._reject(
                            plan,
                            "add-on requires the existing position to be profitable",
                        )
                    )
                    continue
                if plan.stop_plan.initial_stop < existing.current_stop:
                    output.append(
                        self._reject(
                            plan,
                            "add-on would lower the existing protection line",
                        )
                    )
                    continue
            else:
                total_positions = len(working.positions) + len(working.pending_plans)
                if total_positions >= config.risk.max_positions:
                    output.append(self._reject(plan, "max positions reached"))
                    continue
                strategy_count = sum(
                    1 for item in working.positions if item.strategy == strategy
                ) + sum(
                    1 for item in working.pending_plans if item.strategy == strategy
                )
                if strategy_count >= config.risk.max_per_strategy.get(strategy, 0):
                    output.append(
                        self._reject(
                            plan, f"max {strategy.value} positions reached"
                        )
                    )
                    continue
                if strategy == Strategy.B:
                    b_count = sum(
                        1 for item in working.positions if item.strategy == Strategy.B
                    ) + sum(
                        1
                        for item in working.pending_plans
                        if item.strategy == Strategy.B
                    )
                    if b_count >= config.risk.max_b_positions:
                        output.append(
                            self._reject(plan, "max B positions reached")
                        )
                        continue
                industry_count = sum(
                    1
                    for item in working.positions
                    if item.sector == plan.candidate.sector
                ) + sum(
                    1
                    for item in working.pending_plans
                    if item.sector == plan.candidate.sector
                )
                if industry_count >= config.risk.max_industry_positions:
                    output.append(
                        self._reject(plan, "max industry positions reached")
                    )
                    continue

            if correlations:
                inner = correlations.get(plan.candidate.ts_code)
                if inner is None:
                    output.append(
                        self._reject(plan, "correlation unavailable")
                    )
                    continue
                held_or_approved = {
                    item.ts_code for item in working.positions
                } | {
                    item.ts_code for item in working.pending_plans
                } | approved_codes
                correlated = [
                    code
                    for code, value in inner.items()
                    if code in held_or_approved
                    and value > config.risk.correlation_threshold
                ]
                if correlated:
                    output.append(
                        self._reject(
                            plan,
                            f"correlation above threshold: {correlated[0]}",
                        )
                    )
                    continue

            entry_price = float(plan.entry_plan.entry_price)
            stop_price = float(plan.stop_plan.initial_stop)
            equity = float(working.equity)
            max_new_risk = equity * config.risk.max_heat - working.heat * equity
            if max_new_risk <= 0:
                output.append(
                    self._reject(plan, "portfolio heat limit reached")
                )
                continue
            shares = min(
                plan.position_plan.shares,
                _lots(max_new_risk / (entry_price - stop_price)),
            )
            if is_add_on:
                assert existing is not None
                initial_shares = existing.initial_shares or existing.shares
                shares = min(
                    shares,
                    _lots(initial_shares * config.risk.add_on_fraction),
                )
            current_value = float(working.positions_value) + sum(
                float(item.entry_price * item.shares)
                for item in working.pending_plans
            )
            pending_value = sum(
                float(item.entry_price * item.shares)
                for item in working.pending_plans
            )
            existing_value = (
                float(existing.current_price * existing.shares)
                if is_add_on and existing is not None
                else 0.0
            )
            max_value = min(
                max(0.0, equity * config.risk.single_position_pct - existing_value),
                equity * config.risk.max_exposure_pct - current_value,
                float(working.cash)
                - equity * config.risk.min_cash_pct
                - pending_value,
            )
            if max_value <= 0:
                output.append(
                    self._reject(plan, "cash or exposure limit reached")
                )
                continue
            shares = min(shares, _lots(max_value / entry_price))
            if shares < 100:
                output.append(
                    self._reject(plan, "position below one lot after caps")
                )
                continue

            reasons = [
                reason
                for reason, binding in (
                    ("single-position cap", shares < plan.position_plan.shares),
                    ("cash reserve", max_value < equity * config.risk.single_position_pct),
                    ("exposure cap", max_value < equity * config.risk.max_exposure_pct),
                    ("heat cap", shares < plan.position_plan.shares),
                )
                if binding
            ]
            if shares < plan.position_plan.shares:
                decision = RiskDecision(
                    outcome=RiskOutcome.REDUCE,
                    reasons=reasons or ["risk cap reduced size"],
                    adjusted_shares=shares,
                )
            else:
                decision = RiskDecision(
                    outcome=RiskOutcome.APPROVE,
                    reasons=["all checks passed"]
                    if correlations
                    else ["all checks passed", "correlation unavailable"],
                )
            value = money(entry_price * shares)
            position = plan.position_plan.model_copy(
                update={
                    "shares": shares,
                    "position_value": value,
                    "position_pct": float(value) / equity,
                    "heat_before": working.heat,
                    "heat_after": working.heat
                    + (entry_price - stop_price) * shares / equity,
                }
            )
            approved = plan.model_copy(
                update={
                    "position_plan": position,
                    "risk_decision": decision,
                    "status": PlanStatus.PENDING_EXECUTION,
                }
            )
            output.append(approved)
            approved_codes.add(plan.candidate.ts_code)
            working = working.with_pending(
                PendingExposure(
                    ts_code=plan.candidate.ts_code,
                    sector=plan.candidate.sector,
                    entry_price=plan.entry_plan.entry_price,
                    stop_price=plan.stop_plan.initial_stop,
                    shares=shares,
                    template=Template.B
                    if strategy == Strategy.B
                    else Template.A,
                    expires_on=plan.expires_on,
                    strategy=strategy,
                    sub_mode=plan.sub_mode,
                )
            )
        return output

    def _reject(self, plan: TradePlan, reason: str) -> TradePlan:
        return plan.model_copy(
            update={
                "status": PlanStatus.REJECTED,
                "risk_decision": RiskDecision(
                    outcome=RiskOutcome.REJECT, reasons=[reason]
                ),
                "cancel_reason": reason,
            }
        )
