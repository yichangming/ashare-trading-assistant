"""Risk-budget sizing and portfolio constraints for A-F plans."""

from __future__ import annotations

from app.core.full_config import FullSystemConfig
from app.domain.enums import AccountMode, PermissionMode, Strategy, Template
from app.domain.models import (
    AccountState,
    Candidate,
    EntryPlan,
    MarketRegime,
    PositionPlan,
    StopPlan,
)
from app.domain.money import money
from app.planning.errors import PositionRejectedError


def _lots(value: float) -> int:
    return int(value // 100 * 100)


class StrategyPositionSizer:
    def size(
        self,
        candidate: Candidate,
        entry: EntryPlan,
        stop: StopPlan,
        account: AccountState,
        regime: MarketRegime,
        config: FullSystemConfig,
        correlations: dict[str, float] | None = None,
    ) -> PositionPlan:
        strategy = candidate.strategy or Strategy.A
        if account.mode == AccountMode.REAL and not account.risk_config_configured:
            raise PositionRejectedError("risk configuration incomplete")
        if account.mode == AccountMode.REAL and account.stale:
            raise PositionRejectedError(
                "account data stale; reconcile positions before new plans"
            )
        if account.trading_halted:
            raise PositionRejectedError("trading halted: manual restart required")
        permission = regime.strategy_permissions.get(strategy)
        if permission is not None and permission.mode == PermissionMode.PAUSED:
            raise PositionRejectedError("strategy permission paused")
        risk_factor = (
            permission.risk_factor
            if permission is not None
            else (0.5 if candidate.permission == PermissionMode.REDUCED else 1.0)
        )
        if candidate.permission == PermissionMode.PAUSED:
            raise PositionRejectedError("candidate permission paused")
        if risk_factor <= 0:
            raise PositionRejectedError("risk factor is zero")

        equity = float(account.equity)
        drawdown = account.drawdown_pct
        if drawdown >= config.risk.drawdown_stop_at:
            raise PositionRejectedError("drawdown circuit breaker active")
        drawdown_factor = (
            0.5
            if account.drawdown_half_latched
            or drawdown >= config.risk.drawdown_half_at
            else 1.0
        )
        risk_pct = {
            Strategy.A: config.strategies.a.risk_pct,
            Strategy.B: config.strategies.b.risk_pct,
            Strategy.C: config.strategies.c.risk_pct,
            Strategy.D: config.strategies.d.risk_pct,
            Strategy.E: config.strategies.e.risk_pct,
            Strategy.F: config.strategies.f.risk_pct,
        }[strategy]
        if strategy == Strategy.D and candidate.evidence.get("trend_context") == "downtrend":
            risk_pct *= config.strategies.d.downtrend_risk_factor
        risk_amount = equity * risk_pct * risk_factor * drawdown_factor
        if risk_amount <= 0:
            raise PositionRejectedError("risk amount is zero")
        entry_price = float(entry.entry_price)
        stop_price = float(stop.initial_stop)
        if entry_price <= stop_price:
            raise PositionRejectedError("entry price must exceed stop")
        per_share = (
            entry_price
            - stop_price
            + entry_price * config.costs.round_trip_cost_pct
        )
        shares = _lots(risk_amount / per_share) if per_share > 0 else 0
        original = shares
        adjustments: list[str] = []
        checks = ["risk_budget"]

        existing_position = next(
            (
                item
                for item in account.positions
                if item.ts_code == candidate.ts_code
            ),
            None,
        )
        if any(
            item.ts_code == candidate.ts_code for item in account.pending_plans
        ):
            raise PositionRejectedError("duplicate pending plan")
        is_add_on = existing_position is not None

        if is_add_on:
            assert existing_position is not None
            if strategy not in config.risk.add_on_strategies:
                raise PositionRejectedError(
                    f"add-on not allowed for strategy {strategy.value}"
                )
            if existing_position.add_on_count >= config.risk.add_on_max:
                raise PositionRejectedError("maximum add-ons reached")
            if existing_position.current_price < existing_position.cost_price:
                raise PositionRejectedError(
                    "add-on requires the existing position to be profitable"
                )
            if stop.initial_stop < existing_position.current_stop:
                raise PositionRejectedError(
                    "add-on would lower the existing protection line"
                )
            initial_shares = existing_position.initial_shares or existing_position.shares
            allowed_add = _lots(
                initial_shares * config.risk.add_on_fraction
            )
            if shares > allowed_add:
                shares = allowed_add
                adjustments.append(f"add-on capped at {allowed_add} shares")
            checks.append("add_on_rules")
        else:
            total_positions = len(account.positions) + len(account.pending_plans)
            if total_positions >= config.risk.max_positions:
                raise PositionRejectedError("max positions reached")
            strategy_count = sum(
                1 for item in account.positions if item.strategy == strategy
            ) + sum(
                1 for item in account.pending_plans if item.strategy == strategy
            )
            if strategy_count >= config.risk.max_per_strategy.get(strategy, 0):
                raise PositionRejectedError(
                    f"max {strategy.value} positions reached"
                )
            checks.extend(["max_positions", "max_per_strategy"])

            if strategy == Strategy.B:
                b_count = sum(
                    1 for item in account.positions if item.strategy == Strategy.B
                ) + sum(
                    1 for item in account.pending_plans if item.strategy == Strategy.B
                )
                if b_count >= config.risk.max_b_positions:
                    raise PositionRejectedError("max B positions reached")
                checks.append("max_b_positions")

            industry_count = sum(
                1 for item in account.positions if item.sector == candidate.sector
            ) + sum(
                1 for item in account.pending_plans if item.sector == candidate.sector
            )
            if industry_count >= config.risk.max_industry_positions:
                raise PositionRejectedError("max industry positions reached")
            checks.append("industry_concentration")

        if correlations:
            existing = {item.ts_code for item in account.positions} | {
                item.ts_code for item in account.pending_plans
            }
            for code, correlation in correlations.items():
                if (
                    code in existing
                    and correlation > config.risk.correlation_threshold
                ):
                    raise PositionRejectedError("correlation above threshold")
        checks.append("correlation")

        existing_value = (
            float(existing_position.current_price * existing_position.shares)
            if is_add_on and existing_position is not None
            else 0.0
        )
        max_value = max(0.0, equity * config.risk.single_position_pct - existing_value)
        shares = min(shares, _lots(max_value / entry_price))
        checks.append("single_position_cap")

        max_cash = float(account.cash) - equity * config.risk.min_cash_pct
        if max_cash <= 0:
            raise PositionRejectedError("cash reserve would be breached")
        shares = min(shares, _lots(max_cash / entry_price))
        checks.append("cash_reserve")

        max_exposure = equity * config.risk.max_exposure_pct
        current_value = float(account.positions_value) + sum(
            float(item.entry_price * item.shares)
            for item in account.pending_plans
        )
        shares = min(
            shares,
            _lots((max_exposure - current_value) / entry_price),
        )
        checks.append("total_exposure")

        max_new_risk = equity * config.risk.max_heat - account.heat * equity
        if max_new_risk <= 0:
            raise PositionRejectedError("portfolio heat limit reached")
        shares = min(
            shares, _lots(max_new_risk / (entry_price - stop_price))
        )
        checks.append("portfolio_heat")

        if shares < original:
            adjustments.append(f"shares reduced from {original} to {shares}")
        if shares < 100:
            raise PositionRejectedError("position below one lot")

        position_value = money(entry_price * shares)
        heat_after = account.heat + (entry_price - stop_price) * shares / equity
        return PositionPlan(
            template=Template.B if strategy == Strategy.B else Template.A,
            risk_pct=risk_pct,
            risk_amount=money(risk_amount),
            shares=shares,
            position_value=position_value,
            position_pct=float(position_value) / equity,
            heat_before=account.heat,
            heat_after=heat_after,
            constraints_checked=checks,
            adjustments=adjustments,
        )
