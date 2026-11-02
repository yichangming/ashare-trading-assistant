"""Unified A-F entry/stop/take-profit construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from app.core.calendar import TradingCalendar
from app.core.full_config import FullSystemConfig
from app.domain.enums import EntryType, Strategy, SubMode
from app.domain.models import (
    Candidate,
    EntryPlan,
    ExitRules,
    StopPlan,
    TakeProfitPlan,
    TakeProfitTarget,
)
from app.domain.money import price
from app.planning.errors import (
    EntryInvalidError,
    PoorRiskRewardError,
    StopTooWideError,
)


@dataclass
class StrategyPlanParts:
    entry: EntryPlan
    stop: StopPlan
    take_profit: TakeProfitPlan
    confirmation_price: Decimal
    max_acceptable_price: Decimal
    execution_low: Decimal
    execution_high: Decimal
    exit_rules: ExitRules


def _validity_days(candidate: Candidate, config: FullSystemConfig) -> int:
    strategy = candidate.strategy or Strategy.A
    if candidate.sub_mode == SubMode.B2:
        return 1
    if strategy == Strategy.A:
        return config.strategies.a.valid_days
    if strategy == Strategy.B:
        return 1
    if strategy == Strategy.C:
        if candidate.sub_mode == SubMode.C2:
            return config.strategies.c.c2_valid_days
        return config.strategies.c.c1_valid_days
    if strategy == Strategy.D:
        return 1
    if strategy == Strategy.E:
        return config.strategies.e.valid_days
    if strategy == Strategy.F:
        mode = str(candidate.evidence.get("price_mode") or "c2")
        return (
            config.strategies.c.c2_valid_days
            if mode == "c2"
            else config.strategies.c.c1_valid_days
        )
    return 2


def _stop_limits(candidate: Candidate, config: FullSystemConfig) -> tuple[float, float, int]:
    strategy = candidate.strategy or Strategy.A
    if candidate.sub_mode == SubMode.B2:
        b = config.strategies.b
        return b.b2_stop_min, b.b2_stop_max, b.time_stop_days
    if strategy == Strategy.A:
        a = config.strategies.a
        return a.stop_min, a.stop_max, a.time_stop_days
    if strategy == Strategy.B:
        b = config.strategies.b
        return b.stop_min, b.stop_max, b.time_stop_days
    if strategy == Strategy.C:
        c = config.strategies.c
        return c.stop_min, c.stop_max, c.time_stop_days
    if strategy == Strategy.D:
        d = config.strategies.d
        return d.stop_min, d.stop_max, d.time_stop_days
    if strategy == Strategy.E:
        e = config.strategies.e
        return e.stop_min, e.stop_max, e.time_stop_days
    c = config.strategies.c
    return c.stop_min, c.stop_max, c.time_stop_days


def _stop_reference(candidate: Candidate, entry: float, atr: float) -> tuple[float, str]:
    evidence = candidate.evidence
    strategy = candidate.strategy or Strategy.A
    if strategy == Strategy.A:
        value = evidence.get("pullback_low")
        if value is None:
            value = candidate.features.get("ma20_raw")
        return float(value), "pullback_low"
    if candidate.sub_mode == SubMode.B2:
        value = evidence.get("frozen_low")
        return float(value), "frozen_low"
    if strategy == Strategy.B:
        value = evidence.get("board_low")
        return float(value), "board_low"
    if strategy == Strategy.C:
        value = (
            evidence.get("pullback_low")
            if candidate.sub_mode == SubMode.C2
            else None
        ) or evidence.get("box_upper")
        return float(value), (
            "c2_pullback_low" if candidate.sub_mode == SubMode.C2 else "platform_upper"
        )
    if strategy == Strategy.D:
        value = evidence.get("observation_low")
        return float(value), "observation_low"
    if strategy == Strategy.E:
        value = evidence.get("range_lower")
        return float(value), "range_lower"
    value = evidence.get("box_upper")
    if value is None:
        value = evidence.get("range_lower")
    return float(value), "event_structure"


def _trailing_rules(strategy: Strategy) -> tuple[str, str]:
    if strategy == Strategy.A:
        return (
            "move to breakeven at +1R; move to +1R at +2R; trail by close - 1.5*ATR14",
            "exit after 10 trading days without +1R and below entry",
        )
    if strategy == Strategy.B:
        return (
            "move to breakeven at +1R; move to +1R at +2R; trail by MA5/1.5*ATR",
            "exit after 3 trading days without a new high; exit on ebb stage",
        )
    if strategy == Strategy.C:
        return (
            "move to breakeven at +1R; move to +1R at +2R; trail by close - 1.5*ATR14",
            "exit after 10 trading days without +1R and below entry",
        )
    if strategy == Strategy.D:
        return (
            "move to breakeven at +1R; move to +1R at +2R; trail by close - 1.5*ATR14",
            "exit after 5 trading days without a new high",
        )
    if strategy == Strategy.E:
        return (
            "move to breakeven at +1R; trail by range midpoint / close - 1.5*ATR",
            "exit after 10 trading days or on range breakdown",
        )
    return (
        "move to breakeven at +1R; move to +1R at +2R; trail by close - 1.5*ATR14",
        "exit after 5 event trading days if the event is negated or expires",
    )


def _exit_rules(
    strategy: Strategy, time_stop_days: int, config: FullSystemConfig
) -> ExitRules:
    if strategy == Strategy.B:
        return ExitRules(
            trailing="breakeven_1r_2r_ma5_atr",
            time_exit="no_new_high",
            time_stop_days=time_stop_days,
        )
    if strategy == Strategy.D:
        return ExitRules(
            trailing="breakeven_1r_2r_ma5_atr",
            time_exit="no_new_high",
            time_stop_days=time_stop_days,
        )
    if strategy == Strategy.E:
        return ExitRules(
            trailing="breakeven_1r_2r_range_atr",
            time_exit="below_entry",
            time_stop_days=time_stop_days,
        )
    if strategy == Strategy.F:
        return ExitRules(
            trailing="breakeven_1r_2r_atr",
            time_exit="event_expiry",
            time_stop_days=time_stop_days,
            event_expiry_days=config.strategies.f.event_window_days,
        )
    return ExitRules(
        trailing="breakeven_1r_2r_atr",
        time_exit="below_entry",
        time_stop_days=time_stop_days,
    )


class StrategyPlanner:
    def plan_parts(
        self,
        candidate: Candidate,
        calendar: TradingCalendar,
        config: FullSystemConfig,
    ) -> StrategyPlanParts:
        strategy = candidate.strategy or Strategy.A
        row = candidate.features
        entry_raw = row.get("close")
        atr = row.get("atr14_raw") or row.get("atr14")
        if entry_raw is None or atr is None or float(atr) <= 0:
            raise EntryInvalidError("entry price or ATR unavailable")
        entry_price = float(entry_raw)
        atr_value = float(atr)
        max_acceptable = entry_price + min(
            entry_price * 0.01, atr_value * 0.25
        )
        stop_ref, structure_ref = _stop_reference(candidate, entry_price, atr_value)
        stop_raw = stop_ref - 0.5 * atr_value
        if stop_raw >= entry_price:
            stop_raw = entry_price - 1.5 * atr_value
            structure_ref = "ATR fallback"
        stop_min, stop_max, time_stop = _stop_limits(candidate, config)
        distance = (entry_price - stop_raw) / entry_price
        if distance > stop_max:
            raise StopTooWideError(
                f"{strategy.value} stop distance {distance:.2%} exceeds {stop_max:.2%}"
            )
        if distance < stop_min:
            # Keep the structural stop; tighten only via size, never by
            # raising the stop above the structure.
            structure_ref += "; below min stop, structure kept"
        stop_price = price(stop_raw)
        r_value = price(entry_price - float(stop_price))
        if r_value <= 0:
            raise StopTooWideError("non-positive R")
        validity_days = _validity_days(candidate, config)
        expires_on = (
            candidate.trade_date
            if candidate.sub_mode == SubMode.B2
            else self._expiry(calendar, candidate.trade_date, validity_days)
        )
        entry_type = {
            Strategy.A: EntryType.PULLBACK,
            Strategy.B: EntryType.CONFIRMATION,
            Strategy.C: EntryType.BREAKOUT,
            Strategy.D: EntryType.RECLAIM,
            Strategy.E: EntryType.RECLAIM,
            Strategy.F: EntryType.BREAKOUT,
        }[strategy]
        entry = EntryPlan(
            type=entry_type,
            trigger_price=price(entry_price),
            entry_zone_low=None,
            entry_zone_high=None,
            validity_days=validity_days,
            validity_minutes=10 if candidate.sub_mode == SubMode.B2 else None,
            max_chase_pct=0.01,
            cancel_conditions=[
                "close below initial stop",
                "structure invalidated or data stale",
                "expired",
            ],
            expires_on=expires_on,
        )
        stop = StopPlan(
            initial_stop=stop_price,
            stop_distance_pct=float(r_value) / entry_price,
            atr14=atr_value,
            structure_ref=structure_ref,
            time_stop_days=time_stop,
            trailing_rules=[_trailing_rules(strategy)[0]],
            r_value=r_value,
        )
        take_profit = self._take_profit(
            candidate, entry_price, float(r_value), config, strategy
        )
        return StrategyPlanParts(
            entry=entry,
            stop=stop,
            take_profit=take_profit,
            confirmation_price=price(entry_price),
            max_acceptable_price=price(max_acceptable),
            execution_low=price(entry_price),
            execution_high=price(max_acceptable),
            exit_rules=_exit_rules(strategy, time_stop, config),
        )

    def _expiry(self, calendar: TradingCalendar, as_of: date, days: int) -> date:
        try:
            return calendar.add_trading_days(as_of, days)
        except ValueError as exc:
            raise EntryInvalidError(f"cannot schedule expiry: {exc}") from exc

    def _take_profit(
        self,
        candidate: Candidate,
        entry: float,
        r_value: float,
        config: FullSystemConfig,
        strategy: Strategy,
    ) -> TakeProfitPlan:
        trailing, time_exit = _trailing_rules(strategy)
        if strategy == Strategy.A:
            a = config.strategies.a
            resistance = candidate.features.get("high_60d_raw")
            target1 = entry + a.r2 * r_value
            target2 = entry + a.r3 * r_value
            if resistance is not None and resistance > entry:
                if resistance - entry < a.min_rr * r_value:
                    raise PoorRiskRewardError(
                        "nearest resistance is closer than 1.5R"
                    )
                # Only the second target is capped by resistance; the first
                # tranche stays at +2R per the A rule.
                target2 = min(target2, float(resistance))
            return TakeProfitPlan(
                targets=[
                    TakeProfitTarget(
                        r_multiple=max(0.01, (target1 - entry) / r_value),
                        price=price(target1),
                        fraction=a.first_fraction,
                    ),
                    TakeProfitTarget(
                        r_multiple=max(0.01, (target2 - entry) / r_value),
                        price=price(target2),
                        fraction=a.second_fraction,
                    ),
                ],
                trailing_rule=trailing,
                time_exit_rule=time_exit,
            )
        if strategy in (Strategy.B, Strategy.D):
            fraction = 0.5
            return TakeProfitPlan(
                targets=[
                    TakeProfitTarget(
                        r_multiple=2.0,
                        price=price(entry + 2 * r_value),
                        fraction=fraction,
                    )
                ],
                trailing_rule=trailing,
                time_exit_rule=time_exit,
            )
        if strategy == Strategy.E:
            upper = candidate.evidence.get("range_upper")
            if upper is None or float(upper) <= entry:
                raise PoorRiskRewardError("range upper target unavailable")
            fraction = min(0.5, 1 - 0.001)
            return TakeProfitPlan(
                targets=[
                    TakeProfitTarget(
                        r_multiple=max(0.01, (float(upper) - entry) / r_value),
                        price=price(float(upper)),
                        fraction=fraction,
                    )
                ],
                trailing_rule=trailing,
                time_exit_rule=time_exit,
            )
        # C and F use the A/C two-tranche structure; only the second target is
        # capped by the nearest reliable resistance.
        fraction = 1 / 3
        second_target = entry + 3 * r_value
        resistance = candidate.features.get("high_60d_raw")
        if resistance is not None and float(resistance) > entry:
            second_target = min(second_target, float(resistance))
        return TakeProfitPlan(
            targets=[
                TakeProfitTarget(
                    r_multiple=2.0,
                    price=price(entry + 2 * r_value),
                    fraction=fraction,
                ),
                TakeProfitTarget(
                    r_multiple=max(0.01, (second_target - entry) / r_value),
                    price=price(second_target),
                    fraction=fraction,
                ),
            ],
            trailing_rule=trailing,
            time_exit_rule=time_exit,
        )
