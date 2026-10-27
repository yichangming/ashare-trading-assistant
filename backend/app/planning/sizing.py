from __future__ import annotations

import math
from decimal import Decimal

from app.core.config import StrategyConfig
from app.domain.enums import Template
from app.domain.models import (
    AccountState,
    Candidate,
    EntryPlan,
    MarketRegime,
    PositionPlan,
    StopPlan,
    TemplatePolicy,
)
from app.domain.money import money
from app.planning.errors import PositionRejectedError

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HALF = Decimal("0.5")
_LOT = Decimal("100")


def _dec(value: float | int | str | Decimal) -> Decimal:
    return Decimal(str(value))


def _lots(amount: Decimal, unit: Decimal) -> int:
    """Largest 100-share-aligned share quantity affordable by ``amount`` at ``unit``."""
    if amount <= 0 or unit <= 0:
        return 0
    return int(amount / unit // _LOT) * 100


def _apply_cap(
    current: int,
    proposed: int,
    name: str,
    adjustments: list[str],
) -> int:
    """Apply a size cap, recording the binding cap and its share change."""
    if proposed >= current:
        return current
    adjustments.append(f"{name}: {current} -> {proposed}")
    return proposed


class PositionSizer:
    """Size a candidate with fixed-risk sizing and portfolio constraints.

    ``EntryPlan.entry_price`` (the zone midpoint when a full zone exists) and
    ``StopPlan.initial_stop`` are authoritative.  Per-share risk includes the
    configured round-trip cost, and shares are floored to 100-share lots.

    Existing positions and approved-but-unfilled pending plans both consume
    global slots, template slots, industry slots, cash, exposure, and heat.
    Global slots are bounded by ``config.risk.max_positions``; Template A is
    bounded by its regime policy, while Template B is bounded by
    ``min(config.risk.max_b_positions, policy.max_positions)``.  Policies
    whose mode is ``"observe"`` reject even when ``enabled`` is true.  A
    drawdown of
    at least ``drawdown_stop_at`` rejects; one of at least
    ``drawdown_half_at`` halves the risk budget.  A set
    ``AccountState.trading_halted`` latch rejects even when the instantaneous
    drawdown has recovered, and a set ``drawdown_half_latched`` latch keeps
    the risk budget halved until the caller clears it.

    When ``correlations`` is not supplied, ``correlation_unavailable`` is
    recorded in ``constraints_checked`` instead of ``correlation``.

    Caps that reduce size are reported in ``adjustments`` as
    ``"<cap>: <before> -> <after>"``.  Hard-constraint failures, invalid
    inputs, and a result below one lot raise
    :class:`~app.planning.errors.PositionRejectedError`.  Money and share
    arithmetic uses :class:`~decimal.Decimal`; floats appear only in ratio
    fields of the emitted plan.

    ``PositionPlan.risk_pct`` is the raw template risk ratio, and
    ``PositionPlan.risk_amount`` is the regime/drawdown-adjusted budget before
    position caps and lot rounding.  Neither is the effective risk of the
    final position.
    """

    def size(
        self,
        candidate: Candidate,
        entry: EntryPlan,
        stop: StopPlan,
        account: AccountState,
        regime: MarketRegime,
        config: StrategyConfig,
        correlations: dict[str, float] | None = None,
    ) -> PositionPlan:
        equity = _dec(account.equity)
        cash = _dec(account.cash)
        peak_equity = _dec(account.peak_equity)
        self._validate_account(equity, cash, peak_equity)

        try:
            entry_price = Decimal(entry.entry_price)
        except (ArithmeticError, ValueError) as exc:
            raise PositionRejectedError(
                f"invalid entry price for {candidate.ts_code}: {exc}"
            ) from exc
        stop_price = _dec(stop.initial_stop)
        if not entry_price.is_finite():
            raise PositionRejectedError(
                f"non-finite entry price for {candidate.ts_code}: {entry_price}"
            )
        if entry_price <= 0:
            raise PositionRejectedError(
                f"entry price must be positive for {candidate.ts_code}: "
                f"{entry_price}"
            )
        if not stop_price.is_finite():
            raise PositionRejectedError(
                f"non-finite stop for {candidate.ts_code}: {stop_price}"
            )
        if stop_price <= 0:
            raise PositionRejectedError(
                f"stop must be positive for {candidate.ts_code}: {stop_price}"
            )
        if stop_price >= entry_price:
            raise PositionRejectedError(
                f"stop {stop_price} must be below entry {entry_price} "
                f"for {candidate.ts_code}"
            )

        self._validate_config(config)
        if candidate.template not in (Template.A, Template.B):
            raise PositionRejectedError(
                f"unsupported template {candidate.template!r}"
            )
        policy_a, policy_b = self._policies(regime)
        policy = policy_a if candidate.template == Template.A else policy_b
        if not policy.enabled:
            raise PositionRejectedError(
                f"{candidate.template} template is disabled by regime policy"
            )
        if policy.mode == "observe":
            raise PositionRejectedError(
                f"{candidate.template} template is in observe mode"
            )

        factor_value = regime.risk_factor
        if not math.isfinite(factor_value) or not 0 <= factor_value <= 1:
            raise PositionRejectedError(f"invalid risk factor: {factor_value!r}")
        risk_factor = _dec(factor_value)
        if risk_factor == 0:
            raise PositionRejectedError("risk factor is zero")

        drawdown = max(_ZERO, _ONE - equity / peak_equity)
        drawdown_half_at = _dec(config.risk.drawdown_half_at)
        drawdown_stop_at = _dec(config.risk.drawdown_stop_at)
        if account.trading_halted:
            raise PositionRejectedError(
                "drawdown circuit breaker active: trading halted, "
                "manual restart required"
            )
        if drawdown >= drawdown_stop_at:
            raise PositionRejectedError(
                f"drawdown circuit breaker active: {drawdown:.4f} >= "
                f"{drawdown_stop_at:.4f}"
            )
        drawdown_factor = (
            _HALF
            if account.drawdown_half_latched or drawdown >= drawdown_half_at
            else _ONE
        )

        risk_pct_value = (
            config.template_a.risk_pct
            if candidate.template == Template.A
            else config.template_b.risk_pct
        )
        risk_pct = _dec(risk_pct_value)
        if not risk_pct.is_finite() or not 0 < risk_pct <= 1:
            raise PositionRejectedError(
                f"invalid template config: risk_pct must be in (0, 1], "
                f"got {risk_pct_value!r}"
            )
        risk_amount = equity * risk_pct * risk_factor * drawdown_factor
        per_share_risk = (entry_price - stop_price) + (
            entry_price * _dec(config.risk.round_trip_cost_pct)
        )
        shares = _lots(risk_amount, per_share_risk)
        original = shares
        checks = ["risk_budget"]
        adjustments: list[str] = []

        positions = account.positions
        pending = account.pending_plans

        occupied = len(positions) + len(pending)
        if occupied >= config.risk.max_positions:
            raise PositionRejectedError(
                f"max positions reached: {occupied} occupied, "
                f"cap {config.risk.max_positions}"
            )
        checks.append("max_positions")

        a_count = sum(
            1 for item in positions if item.template == Template.A
        ) + sum(1 for item in pending if item.template == Template.A)
        b_count = sum(
            1 for item in positions if item.template == Template.B
        ) + sum(1 for item in pending if item.template == Template.B)
        if candidate.template == Template.A:
            if a_count >= policy_a.max_positions:
                raise PositionRejectedError(
                    f"max A policy positions reached: {a_count} >= "
                    f"{policy_a.max_positions}"
                )
            checks.append("max_a_policy")
        else:
            if b_count >= config.risk.max_b_positions:
                raise PositionRejectedError(
                    f"max B positions reached: {b_count} >= "
                    f"{config.risk.max_b_positions}"
                )
            checks.append("max_b_positions")
            if b_count >= policy_b.max_positions:
                raise PositionRejectedError(
                    f"max B policy positions reached: {b_count} >= "
                    f"{policy_b.max_positions}"
                )
            checks.append("max_b_policy")

        industry_count = sum(
            1 for item in positions if item.sector == candidate.sector
        ) + sum(1 for item in pending if item.sector == candidate.sector)
        if industry_count >= config.risk.max_industry_positions:
            raise PositionRejectedError(
                f"max industry positions reached for {candidate.sector}: "
                f"{industry_count} >= {config.risk.max_industry_positions}"
            )
        checks.append("industry_concentration")

        held_codes = {item.ts_code for item in positions} | {
            item.ts_code for item in pending
        }
        if candidate.ts_code in held_codes:
            raise PositionRejectedError(
                f"duplicate position or plan for {candidate.ts_code}"
            )
        checks.append("duplicate")

        if correlations is None:
            checks.append("correlation_unavailable")
        else:
            threshold = config.risk.correlation_threshold
            for code, correlation in correlations.items():
                if not math.isfinite(correlation) or not -1 <= correlation <= 1:
                    raise PositionRejectedError(
                        f"invalid correlation for {code}: {correlation!r}"
                    )
                if code in held_codes and correlation > threshold:
                    raise PositionRejectedError(
                        f"correlation above threshold with {code}: "
                        f"{correlation} > {threshold}"
                    )
            checks.append("correlation")

        max_position_value = equity * _dec(config.risk.max_single_position_pct)
        shares = _apply_cap(
            shares,
            _lots(max_position_value, entry_price),
            "single_position_cap",
            adjustments,
        )
        checks.append("single_position_cap")

        pending_notional = sum(
            (item.entry_price * item.shares for item in pending), _ZERO
        )
        min_cash = equity * _dec(config.risk.min_cash_pct)
        available_cash = cash - pending_notional
        max_cash_spend = available_cash - min_cash
        if max_cash_spend <= 0:
            raise PositionRejectedError(
                f"cash reserve would be breached: available cash "
                f"{available_cash}, reserve {min_cash}"
            )
        shares = _apply_cap(
            shares,
            _lots(max_cash_spend, entry_price),
            "cash_reserve",
            adjustments,
        )
        checks.append("cash_reserve")

        # Defense in depth: for a consistent account snapshot the cash check
        # binds first (positions above 90% of equity imply cash below the
        # reserve).  This can only fire when cash, positions, and pending
        # commitments do not reconcile.
        existing_exposure = account.positions_value + pending_notional
        max_exposure = equity - min_cash
        remaining_exposure = max_exposure - existing_exposure
        if remaining_exposure <= 0:
            raise PositionRejectedError(
                f"total exposure limit reached: existing exposure "
                f"{existing_exposure}, cap {max_exposure}"
            )
        shares = _apply_cap(
            shares,
            _lots(remaining_exposure, entry_price),
            "total_exposure",
            adjustments,
        )
        checks.append("total_exposure")

        max_heat = equity * _dec(config.risk.max_heat)
        current_heat = _dec(account.heat) * equity
        max_new_risk = max_heat - current_heat
        if max_new_risk <= 0:
            raise PositionRejectedError(
                f"portfolio heat limit reached: heat {account.heat:.4f} >= "
                f"{config.risk.max_heat:.4f}"
            )
        shares = _apply_cap(
            shares,
            _lots(max_new_risk, entry_price - stop_price),
            "portfolio_heat",
            adjustments,
        )
        checks.append("portfolio_heat")

        # Fallback for any future size reduction that is not attributable to
        # one of the named caps above.
        if shares != original and not adjustments:
            adjustments.append(f"shares reduced from {original} to {shares}")
        if shares < 100:
            raise PositionRejectedError(
                f"position below one lot: {shares} shares for {candidate.ts_code}"
            )

        position_value = money(entry_price * shares)
        new_risk = (entry_price - stop_price) * shares
        heat_after = float(_dec(account.heat) + new_risk / equity)
        # risk_pct is the raw template ratio; risk_amount is the
        # regime/drawdown-adjusted budget before caps and lot rounding.
        return PositionPlan(
            template=candidate.template,
            risk_pct=risk_pct_value,
            risk_amount=money(risk_amount),
            shares=shares,
            position_value=position_value,
            position_pct=float(position_value) / float(equity),
            heat_before=account.heat,
            heat_after=heat_after,
            constraints_checked=checks,
            adjustments=adjustments,
        )

    @staticmethod
    def _validate_account(
        equity: Decimal, cash: Decimal, peak_equity: Decimal
    ) -> None:
        if not equity.is_finite() or equity <= 0:
            raise PositionRejectedError(
                f"invalid account state: equity must be finite and positive, "
                f"got {equity}"
            )
        if not cash.is_finite() or cash < 0:
            raise PositionRejectedError(
                f"invalid account state: cash must be finite and non-negative, "
                f"got {cash}"
            )
        if not peak_equity.is_finite() or peak_equity <= 0:
            raise PositionRejectedError(
                f"invalid account state: peak equity must be finite and positive, "
                f"got {peak_equity}"
            )

    @staticmethod
    def _validate_config(config: StrategyConfig) -> None:
        risk = config.risk
        if risk.max_positions <= 0:
            raise PositionRejectedError(
                "invalid risk config: max_positions must be positive"
            )
        if risk.max_b_positions < 0:
            raise PositionRejectedError(
                "invalid risk config: max_b_positions must be non-negative"
            )
        if not 0 < risk.max_single_position_pct <= 1:
            raise PositionRejectedError(
                "invalid risk config: max_single_position_pct must be in (0, 1]"
            )
        if not 0 <= risk.min_cash_pct <= 1:
            raise PositionRejectedError(
                "invalid risk config: min_cash_pct must be in [0, 1]"
            )
        if not 0 < risk.max_heat <= 1:
            raise PositionRejectedError(
                "invalid risk config: max_heat must be in (0, 1]"
            )
        if risk.max_industry_positions <= 0:
            raise PositionRejectedError(
                "invalid risk config: max_industry_positions must be positive"
            )
        if not 0 <= risk.correlation_threshold <= 1:
            raise PositionRejectedError(
                "invalid risk config: correlation_threshold must be in [0, 1]"
            )
        if not 0 <= risk.round_trip_cost_pct <= 1:
            raise PositionRejectedError(
                "invalid risk config: round_trip_cost_pct must be in [0, 1]"
            )
        if not 0 <= risk.drawdown_half_at < risk.drawdown_stop_at <= 1:
            raise PositionRejectedError(
                "invalid risk config: drawdown levels must satisfy "
                "0 <= drawdown_half_at < drawdown_stop_at <= 1"
            )

    @staticmethod
    def _policies(
        regime: MarketRegime,
    ) -> tuple[TemplatePolicy, TemplatePolicy]:
        policy_a = regime.template_availability.get(Template.A)
        policy_b = regime.template_availability.get(Template.B)
        if policy_a is None or policy_b is None:
            raise PositionRejectedError(
                "template policy missing from market regime"
            )
        if policy_a.max_positions < 0 or policy_b.max_positions < 0:
            raise PositionRejectedError(
                "invalid template policy: max_positions must be non-negative"
            )
        return policy_a, policy_b
