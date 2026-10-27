from __future__ import annotations

import math
from decimal import Decimal

from app.core.config import StrategyConfig
from app.domain.enums import Template
from app.domain.models import Candidate, EntryPlan, StopPlan
from app.domain.money import price
from app.planning.errors import StopTooWideError

_ONE = Decimal("1")
_ATR_FALLBACK_MULTIPLE = Decimal("1.5")
_B_ATR_BUFFER = Decimal("0.5")

_A_TRAILING_RULES = [
    "move to breakeven at +1R",
    "move to +1R at +2R",
    "trail by close - 1.5*ATR14",
]
_B_TRAILING_RULES = [
    "move to breakeven at +1R",
    "move to +1R at +2R",
    "trail by close - 1.5*ATR14 and MA5",
]

_SUPPORT_FIELDS = {
    Template.A: ("low_10d_raw", "ma20_raw"),
    Template.B: ("low_5d_raw", "ma5_raw", "ma10_raw"),
}


def _reject(candidate: Candidate, detail: str) -> StopTooWideError:
    return StopTooWideError(
        f"{candidate.ts_code} ({candidate.template.value}): {detail}"
    )


class StopPlanner:
    """Build the initial structural stop and risk unit for a candidate.

    ``EntryPlan.entry_price`` (the zone midpoint when a full zone exists, else
    the trigger) is authoritative for risk sizing.

    The structure stop is the highest candidate support strictly below the
    entry price minus ``atr_buffer * ATR14``.  With no support below entry, or
    when that structure stop is not below entry, the planner falls back to
    ``entry - 1.5 * ATR14``.  Template A rejects a stop distance above
    ``stop_max_pct``; Template B clamps it to the max instead.  Both widen a
    stop tighter than ``stop_min_pct``.  ``stop_distance_pct`` and ``r_value``
    are derived from the final, cent-rounded stop.

    All support, buffer, stop, and R arithmetic is done in
    :class:`~decimal.Decimal` before ``price()`` so half-up rounding happens
    exactly once on decimal values.
    """

    def plan(
        self,
        candidate: Candidate,
        entry: EntryPlan,
        features: dict[str, float],
        config: StrategyConfig,
    ) -> StopPlan:
        entry_price = entry.entry_price
        if not entry_price.is_finite() or entry_price <= 0:
            raise _reject(
                candidate,
                f"entry price must be finite and positive, got {entry_price}",
            )

        atr_raw = features.get("atr14_raw")
        if atr_raw is None:
            raise _reject(candidate, "ATR unavailable for stop")
        if not math.isfinite(atr_raw) or atr_raw <= 0:
            raise _reject(
                candidate,
                f"ATR must be finite and positive, got {atr_raw}",
            )
        atr = Decimal(str(atr_raw))

        if candidate.template == Template.A:
            buffer = Decimal(str(config.template_a.atr_buffer))
            time_stop_days = config.template_a.time_stop_days
            stop_min = Decimal(str(config.template_a.stop_min_pct))
            stop_max = Decimal(str(config.template_a.stop_max_pct))
            trailing_rules = list(_A_TRAILING_RULES)
        else:
            buffer = _B_ATR_BUFFER
            time_stop_days = config.template_b.time_stop_days
            stop_min = Decimal(str(config.template_b.stop_min_pct))
            stop_max = Decimal(str(config.template_b.stop_max_pct))
            trailing_rules = list(_B_TRAILING_RULES)

        if not stop_min.is_finite() or not stop_max.is_finite() or stop_min > stop_max:
            raise _reject(
                candidate,
                f"invalid stop bounds: min {stop_min}, max {stop_max}",
            )

        supports: list[Decimal] = []
        for field in _SUPPORT_FIELDS[candidate.template]:
            raw = features.get(field)
            if raw is None or not math.isfinite(raw):
                continue
            value = Decimal(str(raw))
            if value < entry_price:
                supports.append(value)
        if supports:
            support = max(supports)
            structure_stop = support - buffer * atr
            structure_ref = f"support={support:.2f}"
        else:
            structure_stop = entry_price - _ATR_FALLBACK_MULTIPLE * atr
            structure_ref = "ATR fallback"
        if structure_stop >= entry_price:
            structure_stop = entry_price - _ATR_FALLBACK_MULTIPLE * atr
            structure_ref = "ATR fallback"

        distance = (entry_price - structure_stop) / entry_price
        if distance > stop_max:
            if candidate.template == Template.A:
                raise _reject(
                    candidate,
                    f"A stop distance {distance:.2%} exceeds {stop_max:.2%}",
                )
            structure_stop = entry_price * (_ONE - stop_max)
            structure_ref += "; clamped to max stop"
        elif distance < stop_min:
            structure_stop = entry_price * (_ONE - stop_min)
            structure_ref += "; widened to min stop"

        stop = price(structure_stop)
        if stop <= 0:
            raise _reject(
                candidate,
                f"final stop must be positive, got {stop}",
            )
        r_value = price(entry_price - stop)
        if r_value <= 0:
            raise _reject(
                candidate,
                f"non-positive R for entry {entry_price} and stop {stop}",
            )
        return StopPlan(
            initial_stop=stop,
            stop_distance_pct=float(r_value / entry_price),
            atr14=float(atr),
            structure_ref=structure_ref,
            time_stop_days=time_stop_days,
            trailing_rules=trailing_rules,
            r_value=r_value,
        )
