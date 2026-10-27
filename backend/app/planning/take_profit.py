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

_TWO = Decimal("2")
_THREE = Decimal("3")
_RESISTANCE_MIN_R = Decimal("1.5")


def _reject(candidate: Candidate, detail: str) -> PoorRiskRewardError:
    return PoorRiskRewardError(
        f"{candidate.ts_code} ({candidate.template.value}): {detail}"
    )


def _fraction(candidate: Candidate, name: str, value: float) -> float:
    if not math.isfinite(value) or not 0 < value <= 1:
        raise _reject(candidate, f"{name} must be in (0, 1], got {value}")
    return value


class TakeProfitPlanner:
    """Build take-profit targets and exit rules from the entry and stop plans.

    ``EntryPlan.entry_price`` (the zone midpoint when a full zone exists) and
    ``StopPlan.r_value`` are authoritative and must be finite and positive.
    Target prices are computed in :class:`~decimal.Decimal` as
    ``entry + multiple * R`` and rounded half-up exactly once by ``price()``.

    Template A sells ``first_fraction`` at +2R and ``second_fraction`` at +3R,
    leaving the remainder for the trailing rule.  A known ``high_60d_raw``
    resistance above entry must sit at least 1.5R away; resistance at or below
    entry, or non-finite resistance, is ignored.

    Template B sells ``second_fraction`` at +2R and trails the remainder by
    MA5/ATR with no fixed percentage target.  The time-exit text reflects
    ``StopPlan.time_stop_days`` so configuration changes flow through.
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

        if candidate.template == Template.A:
            return self._plan_a(candidate, entry_price, r_value, stop, config)
        return self._plan_b(candidate, entry_price, r_value, stop, config)

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
        self._check_resistance(candidate, entry_price, r_value)
        targets = [
            TakeProfitTarget(
                r_multiple=2.0,
                price=price(entry_price + _TWO * r_value),
                fraction=first_fraction,
            ),
            TakeProfitTarget(
                r_multiple=3.0,
                price=price(entry_price + _THREE * r_value),
                fraction=second_fraction,
            ),
        ]
        return TakeProfitPlan(
            targets=targets,
            trailing_rule=(
                "+1R move stop to breakeven; +2R move stop to +1R; "
                "trail by close - 1.5*ATR14"
            ),
            time_exit_rule=(
                f"exit after {stop.time_stop_days} trading days "
                "without +1R and below entry"
            ),
        )

    @staticmethod
    def _check_resistance(
        candidate: Candidate,
        entry_price: Decimal,
        r_value: Decimal,
    ) -> None:
        raw_resistance = candidate.features.get("high_60d_raw")
        if raw_resistance is None or not math.isfinite(raw_resistance):
            return
        resistance = Decimal(str(raw_resistance))
        distance = resistance - entry_price
        if 0 < distance < _RESISTANCE_MIN_R * r_value:
            raise _reject(
                candidate,
                f"nearest resistance {resistance} is {distance} above entry, "
                f"closer than 1.5R ({_RESISTANCE_MIN_R * r_value})",
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
        targets = [
            TakeProfitTarget(
                r_multiple=2.0,
                price=price(entry_price + _TWO * r_value),
                fraction=second_fraction,
            )
        ]
        return TakeProfitPlan(
            targets=targets,
            trailing_rule=(
                "+1R move stop to breakeven; +2R sell half; "
                "trail remainder by MA5/1.5*ATR14"
            ),
            time_exit_rule=(
                f"exit after {stop.time_stop_days} trading days "
                "without a new high; exit immediately on ebb stage"
            ),
        )
