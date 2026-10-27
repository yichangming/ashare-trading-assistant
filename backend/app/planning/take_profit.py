from __future__ import annotations

import math
from decimal import Decimal

from app.core.config import StrategyConfig
from app.domain.enums import Template
from app.domain.models import (
    Candidate,
    EntryPlan,
    StopPlan,
    TakeProfitPlan,
    TakeProfitTarget,
)
from app.domain.money import price
from app.planning.errors import PoorRiskRewardError

_RESISTANCE_MIN_R = Decimal("1.5")


def _reject(candidate: Candidate, detail: str) -> PoorRiskRewardError:
    return PoorRiskRewardError(
        f"{candidate.ts_code} ({candidate.template.value}): {detail}"
    )


def _fraction(candidate: Candidate, name: str, value: float) -> float:
    if not math.isfinite(value) or not 0 < value <= 1:
        raise _reject(candidate, f"{name} must be in (0, 1], got {value}")
    return value


def _format_r(value: float) -> str:
    return f"{value:g}"


class TakeProfitPlanner:
    """Build take-profit targets and exit rules from the entry and stop plans.

    ``EntryPlan.entry_price`` (the zone midpoint when a full zone exists) and
    ``StopPlan.r_value`` are authoritative and must be finite and positive.
    Target prices are computed in :class:`~decimal.Decimal` as
    ``entry + rN * R`` and rounded half-up exactly once by ``price()``.  The
    configured ``r1``/``r2``/``r3`` levels drive both prices and rule text, so
    strategy overrides are reflected in the emitted plan.

    Template A sells ``first_fraction`` at ``r2``R and ``second_fraction`` at
    ``r3``R, leaving the remainder for the trend/trailing rule.  A known
    ``high_60d_raw`` resistance above entry must sit at least 1.5R away,
    otherwise the plan is rejected.  When it is far enough away, targets are
    capped at that level and their ``r_multiple`` reports the capped distance.
    Resistance at or below entry, or non-finite resistance, is ignored.

    Template B sells ``second_fraction`` at ``r2``R and trails the remainder by
    MA5/ATR with no fixed percentage target; it does not apply the resistance
    gate.  The time-exit text reflects ``StopPlan.time_stop_days`` so
    configuration changes flow through.

    ``high_60d_raw`` includes the current bar, so it is only a proxy for
    overhead supply rather than a true swing-high level.  Proper swing-high
    resistance detection is a follow-up task.

    All R and target arithmetic is guarded: decimal overflow or invalid
    operations are converted to :class:`PoorRiskRewardError` so callers never
    see a bare :class:`ArithmeticError`.
    """

    def plan(
        self,
        candidate: Candidate,
        entry: EntryPlan,
        stop: StopPlan,
        config: StrategyConfig,
    ) -> TakeProfitPlan:
        try:
            entry_price = entry.entry_price
        except (ArithmeticError, ValueError) as exc:
            raise _reject(candidate, f"entry price is invalid: {exc}") from exc
        if not entry_price.is_finite() or entry_price <= 0:
            raise _reject(
                candidate,
                f"entry price must be finite and positive, got {entry_price}",
            )

        r_value = stop.r_value
        if not r_value.is_finite() or r_value <= 0:
            raise _reject(candidate, f"R must be finite and positive, got {r_value}")
        if stop.initial_stop >= entry_price:
            raise _reject(
                candidate,
                f"initial stop {stop.initial_stop} must be below entry {entry_price}",
            )

        try:
            if candidate.template == Template.A:
                return self._plan_a(candidate, entry_price, r_value, stop, config)
            return self._plan_b(candidate, entry_price, r_value, stop, config)
        except ArithmeticError as exc:
            raise _reject(candidate, f"target arithmetic failed: {exc}") from exc

    def _plan_a(
        self,
        candidate: Candidate,
        entry_price: Decimal,
        r_value: Decimal,
        stop: StopPlan,
        config: StrategyConfig,
    ) -> TakeProfitPlan:
        take_profit = config.template_a.take_profit
        first_fraction = _fraction(
            candidate, "A first fraction", take_profit.first_fraction
        )
        second_fraction = _fraction(
            candidate, "A second fraction", take_profit.second_fraction
        )
        resistance = self._resistance_above_entry(candidate, entry_price)
        if (
            resistance is not None
            and resistance - entry_price < _RESISTANCE_MIN_R * r_value
        ):
            raise _reject(
                candidate,
                f"nearest resistance {resistance} is {resistance - entry_price} "
                f"above entry, closer than 1.5R ({_RESISTANCE_MIN_R * r_value})",
            )

        r1, r2, r3 = take_profit.r1, take_profit.r2, take_profit.r3
        targets = [
            self._target(
                candidate, entry_price, r_value, r2, first_fraction, resistance
            ),
            self._target(
                candidate, entry_price, r_value, r3, second_fraction, resistance
            ),
        ]
        if targets[1].price < targets[0].price:
            raise _reject(
                candidate,
                f"A targets are not ordered: {targets[0].price}, {targets[1].price}",
            )
        return TakeProfitPlan(
            targets=targets,
            trailing_rule=(
                f"+{_format_r(r1)}R move stop to breakeven; "
                f"+{_format_r(r2)}R sell {first_fraction:.0%}; "
                f"+{_format_r(r3)}R sell {second_fraction:.0%}; "
                "remainder trend; trail by close - 1.5*ATR14"
            ),
            time_exit_rule=(
                f"exit after {stop.time_stop_days} trading days "
                "without +1R and below entry"
            ),
        )

    @staticmethod
    def _resistance_above_entry(
        candidate: Candidate,
        entry_price: Decimal,
    ) -> Decimal | None:
        raw_resistance = candidate.features.get("high_60d_raw")
        if raw_resistance is None or not math.isfinite(raw_resistance):
            return None
        resistance = Decimal(str(raw_resistance))
        if resistance <= entry_price:
            return None
        return resistance

    @staticmethod
    def _target(
        candidate: Candidate,
        entry_price: Decimal,
        r_value: Decimal,
        multiple: float,
        fraction: float,
        resistance: Decimal | None,
    ) -> TakeProfitTarget:
        raw_price = entry_price + Decimal(str(multiple)) * r_value
        if resistance is not None:
            raw_price = min(raw_price, resistance)
        target_price = price(raw_price)
        if target_price <= entry_price:
            raise _reject(
                candidate,
                f"target {target_price} must be above entry {entry_price}",
            )
        r_multiple = float((target_price - entry_price) / r_value)
        if not math.isfinite(r_multiple) or r_multiple <= 0:
            raise _reject(
                candidate,
                f"target {target_price} has invalid R multiple {r_multiple}",
            )
        return TakeProfitTarget(
            r_multiple=r_multiple,
            price=target_price,
            fraction=fraction,
        )

    def _plan_b(
        self,
        candidate: Candidate,
        entry_price: Decimal,
        r_value: Decimal,
        stop: StopPlan,
        config: StrategyConfig,
    ) -> TakeProfitPlan:
        take_profit = config.template_b.take_profit
        second_fraction = _fraction(
            candidate, "B second fraction", take_profit.second_fraction
        )
        r1, r2 = take_profit.r1, take_profit.r2
        targets = [
            self._target(
                candidate, entry_price, r_value, r2, second_fraction, resistance=None
            )
        ]
        return TakeProfitPlan(
            targets=targets,
            trailing_rule=(
                f"+{_format_r(r1)}R move stop to breakeven; "
                f"+{_format_r(r2)}R sell {second_fraction:.0%}; "
                "remainder trail by MA5/1.5*ATR14"
            ),
            time_exit_rule=(
                f"exit after {stop.time_stop_days} trading days "
                "without a new high; exit immediately on ebb stage"
            ),
        )
