"""Conservative daily OHLC simulator for plan execution.

The simulator shares :mod:`app.portfolio.exit_rules` with the live position
review.  It deliberately does not invent an intrabar path: same-day stop and
target hits resolve to the stop, gaps use the open, and shares bought today
cannot be sold today (T+1).  Orders that did not fill remain pending until
their stated expiry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from math import sqrt
from statistics import fmean, pstdev
from zoneinfo import ZoneInfo

import polars as pl

from app.core.calendar import TradingCalendar
from app.core.full_config import FullSystemConfig
from app.domain.enums import (
    ExitSession,
    PlanStatus,
    Strategy,
    SubMode,
    Template,
)
from app.domain.models import (
    AccountState,
    ExitRules,
    PendingExposure,
    PositionState,
    TargetState,
    TradePlan,
)
from app.domain.money import price
from app.portfolio.exit_rules import (
    ExitObservation,
    ExitPolicy,
    evaluate_exit,
    proposed_trailing_stop,
    target_states_from_plan,
)
from app.portfolio.fees import calculate_fees

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _event_day(event: dict) -> date | None:
    value = (
        event.get("first_seen_at")
        or event.get("ann_date")
        or event.get("published_at")
    )
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _number(row: dict, key: str, default: float | None = None) -> float | None:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _suspended(row: dict) -> bool:
    if bool(row.get("is_suspended")):
        return True
    return (_number(row, "vol", 0.0) or 0.0) <= 0


def _limit_prices(row: dict, open_: float) -> tuple[float | None, float | None]:
    pre_close = _number(row, "pre_close", open_) or open_
    up_limit = _number(row, "up_limit")
    down_limit = _number(row, "down_limit")
    return (
        up_limit if up_limit is not None else pre_close * 1.10,
        down_limit if down_limit is not None else pre_close * 0.90,
    )


def _default_exit_rules(strategy: Strategy | None, config: FullSystemConfig) -> ExitRules:
    if strategy == Strategy.B:
        return ExitRules(
            trailing="breakeven_1r_2r_ma5_atr",
            time_exit="no_new_high",
            time_stop_days=config.strategies.b.time_stop_days,
        )
    if strategy == Strategy.D:
        return ExitRules(
            trailing="breakeven_1r_2r_ma5_atr",
            time_exit="no_new_high",
            time_stop_days=config.strategies.d.time_stop_days,
        )
    if strategy == Strategy.E:
        return ExitRules(
            trailing="breakeven_1r_2r_range_atr",
            time_exit="below_entry",
            time_stop_days=config.strategies.e.time_stop_days,
        )
    if strategy == Strategy.F:
        return ExitRules(
            trailing="breakeven_1r_2r_atr",
            time_exit="event_expiry",
            time_stop_days=config.strategies.f.event_window_days,
            event_expiry_days=config.strategies.f.event_window_days,
        )
    return ExitRules(
        trailing="breakeven_1r_2r_atr",
        time_exit="below_entry",
        time_stop_days=config.strategies.a.time_stop_days,
    )


@dataclass
class SimPosition:
    ts_code: str
    sector: str
    strategy: Strategy | None
    sub_mode: str | None
    shares: int
    initial_shares: int
    entry_price: float
    initial_stop: float
    current_stop: float
    opened_on: date
    atr14: float
    highest: float
    targets: list[TargetState]
    target_index: int = 0
    time_stop_days: int = 10
    trailing_rule: str = ""
    days_held: int = 0
    current_price: float = 0.0
    add_on_count: int = 0
    initial_r: float = 0.0
    entry_fee_total: float = 0.0
    entry_fee_remaining: float = 0.0
    ma5: float | None = None
    range_midpoint: float | None = None
    exit_rules: ExitRules = field(default_factory=ExitRules)
    event_expiry: date | None = None
    event_invalid: bool = False
    event_data_available: bool = True
    event_id: str | None = None
    last_new_high_on: date | None = None
    pending_stop: float | None = None
    pending_stop_source_day: date | None = None
    last_exit_reason: str | None = None
    primary_plan_id: str = ""


@dataclass
class PendingOrder:
    plan: TradePlan
    remaining: int
    created_on: date
    filled_shares: int = 0
    state: str = "pending"
    # pending / partial / filled / expired / invalidated / cancelled /
    # cancel_unconfirmed
    expired_on: date | None = None


@dataclass
class SimTrade:
    ts_code: str
    strategy: str
    sub_mode: str | None
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    fees: float
    r_multiple: float
    pnl: float
    exit_reason: str


@dataclass
class Simulator:
    config: FullSystemConfig
    cash: float
    initial_equity: float
    peak_equity: float
    positions: dict[str, SimPosition] = field(default_factory=dict)
    pending: list[PendingOrder] = field(default_factory=list)
    trades: list[SimTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    planned_shares: int = 0
    filled_shares: int = 0
    invested_ratios: list[float] = field(default_factory=list)
    calendar: TradingCalendar | None = None
    events_by_code: dict[str, list[dict]] | None = None
    day_index: dict[date, int] = field(default_factory=dict)
    recent_bars: dict[str, list[dict]] = field(default_factory=dict)
    expired_orders: list[dict] = field(default_factory=list)
    cancelled_orders: list[dict] = field(default_factory=list)
    invalidated_orders: list[dict] = field(default_factory=list)
    cancel_unconfirmed_orders: list[dict] = field(default_factory=list)
    entry_fees_paid: float = 0.0
    exit_fees_paid: float = 0.0

    def position_value(self) -> float:
        return sum(
            (position.current_price or position.entry_price) * position.shares
            for position in self.positions.values()
        )

    def equity(self) -> float:
        return self.cash + self.position_value()

    def reserved_cash(self, *, exclude: PendingOrder | None = None) -> float:
        total = 0.0
        for order in self.pending:
            if order is exclude or order.remaining <= 0:
                continue
            if order.state not in ("pending", "partial", "cancel_unconfirmed"):
                continue
            fill = float(
                order.plan.confirmation_price or order.plan.entry_plan.entry_price
            )
            gross = fill * order.remaining
            fees = calculate_fees(
                side="buy",
                price=fill,
                shares=order.remaining,
                costs=self.config.costs,
            ).total
            total += gross + float(fees)
        return total

    def account_snapshot(self, as_of: date) -> AccountState:
        positions = [
            PositionState(
                ts_code=position.ts_code,
                sector=position.sector,
                shares=position.shares,
                available_shares=0 if position.opened_on >= as_of else position.shares,
                cost_price=price(position.entry_price),
                current_stop=price(max(0.0, position.current_stop)),
                current_price=price(
                    getattr(position, "current_price", position.entry_price)
                ),
                template=Template.B
                if position.strategy == Strategy.B
                else Template.A,
                strategy=position.strategy,
                sub_mode=SubMode(position.sub_mode) if position.sub_mode else None,
                primary_strategy=position.strategy,
                initial_r=Decimal(str(position.initial_r))
                if position.initial_r > 0
                else None,
                initial_shares=position.initial_shares,
                add_on_count=position.add_on_count,
                targets=list(position.targets),
            )
            for position in self.positions.values()
        ]
        pending = [
            PendingExposure(
                ts_code=order.plan.candidate.ts_code,
                sector=order.plan.candidate.sector,
                entry_price=order.plan.entry_plan.entry_price,
                stop_price=order.plan.stop_plan.initial_stop,
                shares=order.remaining,
                template=Template.B
                if order.plan.strategy == Strategy.B
                else Template.A,
                expires_on=order.plan.expires_on,
                strategy=order.plan.strategy,
                sub_mode=order.plan.sub_mode,
            )
            for order in self.pending
            if order.remaining > 0
            and order.state in ("pending", "partial", "cancel_unconfirmed")
        ]
        equity = self.equity()
        return AccountState(
            equity=Decimal(str(max(equity, 1.0))),
            cash=Decimal(str(max(self.cash, 0.0))),
            peak_equity=Decimal(str(max(self.peak_equity, equity, 1.0))),
            positions=positions,
            pending_plans=pending,
        )

    def process_day(self, day: date, bars: pl.DataFrame) -> list[SimTrade]:
        if bars.is_empty():
            self.missing_data.append(day.isoformat())
            return []
        by_code = {row["ts_code"]: row for row in bars.iter_rows(named=True)}
        if day not in self.day_index:
            self.day_index[day] = len(self.day_index)
        for code, row in by_code.items():
            history = self.recent_bars.setdefault(code, [])
            history.append(
                {
                    "high": float(row.get("high") or 0),
                    "low": float(row.get("low") or 0),
                    "close": float(row.get("close") or 0),
                }
            )
            if len(history) > 20:
                del history[:-20]
        new_trades: list[SimTrade] = []
        # Exits first; positions bought today are protected by T+1.
        for code in list(self.positions):
            position = self.positions[code]
            bar_row = by_code.get(code)
            if bar_row is None:
                self.missing_data.append(
                    f"{day.isoformat()}:{code}: no bar (suspended/delisted?)"
                )
                continue
            high = float(bar_row["high"])
            close = float(bar_row["close"])
            position.current_price = close
            # A protection line calculated at a prior close takes effect only
            # on the next processed trading day; it is never applied to the
            # bar that produced it.
            if (
                position.pending_stop is not None
                and position.pending_stop_source_day is not None
                and day > position.pending_stop_source_day
            ):
                position.current_stop = max(
                    position.current_stop, position.pending_stop
                )
                position.pending_stop = None
                position.pending_stop_source_day = None
            if high > position.highest + 1e-9:
                position.highest = high
                position.last_new_high_on = day
            self._update_trailing_inputs(position, code)
            self._update_event_state(position, day)
            if position.opened_on >= day:
                continue
            if _suspended(bar_row):
                continue
            position.days_held += 1
            trades = self._check_exit(position, day, bar_row)
            if trades:
                new_trades.extend(trades)
            else:
                self._update_stop(position, high, close, day=day)

        # Entries from previously approved plans.  An unfilled order that has
        # not expired stays pending; it is not deleted just because today's
        # bar did not reach the entry.
        remaining_orders: list[PendingOrder] = []
        for order in self.pending:
            if order.state in ("filled", "cancelled", "expired", "invalidated"):
                continue
            if order.remaining <= 0:
                order.state = "filled"
                continue
            if day > order.plan.expires_on:
                order.state = "expired"
                order.expired_on = day
                self.expired_orders.append(
                    {
                        "plan_id": order.plan.id,
                        "ts_code": order.plan.candidate.ts_code,
                        "expired_on": day.isoformat(),
                        "remaining": order.remaining,
                    }
                )
                continue
            if order.created_on >= day:
                remaining_orders.append(order)
                continue
            bar_row = by_code.get(order.plan.candidate.ts_code)
            if bar_row is None:
                remaining_orders.append(order)
                continue
            filled = self._try_entry(order, day, bar_row)
            if filled and order.remaining <= 0:
                order.state = "filled"
                continue
            if order.remaining > 0:
                order.state = "partial"
                remaining_orders.append(order)
        self.pending = remaining_orders
        equity = self.equity()
        self.peak_equity = max(self.peak_equity, equity)
        self.equity_curve.append(
            {
                "trade_date": day.isoformat(),
                "equity": equity,
                "cash": self.cash,
                "reserved_cash": self.reserved_cash(),
            }
        )
        self.invested_ratios.append(
            max(0.0, (equity - self.cash) / equity) if equity > 0 else 0.0
        )
        return new_trades

    def cancel_plan(self, plan_id: str, day: date, reason: str = "cancelled") -> bool:
        kept: list[PendingOrder] = []
        cancelled = False
        for order in self.pending:
            if order.plan.id == plan_id and order.state in ("pending", "partial"):
                order.state = "cancelled"
                self.cancelled_orders.append(
                    {
                        "plan_id": plan_id,
                        "ts_code": order.plan.candidate.ts_code,
                        "cancelled_on": day.isoformat(),
                        "reason": reason,
                        "remaining": order.remaining,
                    }
                )
                cancelled = True
                continue
            kept.append(order)
        self.pending = kept
        return cancelled

    def invalidate_plan(
        self, plan_id: str, day: date, reason: str = "structure invalidated"
    ) -> bool:
        return self._terminate_plan(
            plan_id,
            day,
            state="invalidated",
            reason=reason,
            target=self.invalidated_orders,
        )

    def mark_cancel_unconfirmed(
        self, plan_id: str, day: date, reason: str = "broker cancel pending"
    ) -> bool:
        """Mark a cancel request that the broker has not confirmed.

        The order keeps its remaining quantity and cash reservation until a
        later ``confirm_cancel``; it may still fill.
        """
        for order in self.pending:
            if order.plan.id == plan_id and order.state in (
                "pending",
                "partial",
                "cancel_unconfirmed",
            ):
                order.state = "cancel_unconfirmed"
                self.cancel_unconfirmed_orders.append(
                    {
                        "plan_id": plan_id,
                        "ts_code": order.plan.candidate.ts_code,
                        "requested_on": day.isoformat(),
                        "reason": reason,
                        "remaining": order.remaining,
                    }
                )
                return True
        return False

    def confirm_cancel(
        self, plan_id: str, day: date, reason: str = "broker cancel confirmed"
    ) -> bool:
        return self._terminate_plan(
            plan_id,
            day,
            state="cancelled",
            reason=reason,
            target=self.cancelled_orders,
        )

    def _terminate_plan(
        self,
        plan_id: str,
        day: date,
        *,
        state: str,
        reason: str,
        target: list[dict],
    ) -> bool:
        kept: list[PendingOrder] = []
        found = False
        for order in self.pending:
            if order.plan.id == plan_id and order.state in (
                "pending",
                "partial",
                "cancel_unconfirmed",
            ):
                order.state = state
                target.append(
                    {
                        "plan_id": plan_id,
                        "ts_code": order.plan.candidate.ts_code,
                        "date": day.isoformat(),
                        "reason": reason,
                        "remaining": order.remaining,
                    }
                )
                found = True
                continue
            kept.append(order)
        self.pending = kept
        return found

    def _try_entry(self, order: PendingOrder, day: date, row: dict) -> bool:
        plan = order.plan
        entry = float(plan.confirmation_price or plan.entry_plan.entry_price)
        maximum = float(plan.max_acceptable_price or entry * 1.01)
        open_ = float(row["open"])
        high = float(row["high"])
        if _suspended(row):
            return False
        up_limit, _ = _limit_prices(row, open_)
        if up_limit is not None and open_ >= up_limit - 0.001:
            # Conservative: a sealed limit-up cannot be relied on for a fill.
            return False
        if open_ > maximum:
            # Price left the execution band; the order remains pending until
            # expiry rather than being silently deleted.
            return False
        if entry <= open_ <= maximum:
            fill = open_
        elif open_ < entry <= high:
            fill = entry
        else:
            return False
        # Tushare daily ``vol`` is in lots (手); one lot is 100 shares.
        volume_lots = float(row.get("vol") or 0)
        volume_shares = volume_lots * 100
        if volume_shares <= 0:
            return False
        max_shares = int(volume_shares * 0.01 // 100 * 100)
        max_shares = max(100, max_shares)
        shares = min(order.remaining, max_shares)
        if shares < 100:
            return False
        slip = (
            self.config.costs.slippage_b
            if plan.strategy == Strategy.B
            else self.config.costs.slippage_a
        )
        fill_price = price(min(fill * (1 + slip), maximum))
        fees = calculate_fees(
            side="buy",
            price=fill_price,
            shares=shares,
            costs=self.config.costs,
        ).total
        cost = float(fill_price) * shares + float(fees)
        available_cash = self.cash - self.reserved_cash(exclude=order)
        if cost > available_cash + 1e-9:
            return False
        self.cash -= cost
        self.entry_fees_paid += float(fees)
        position = self.positions.get(plan.candidate.ts_code)
        if position is None:
            initial_stop = float(plan.stop_plan.initial_stop)
            initial_r = float(fill_price) - initial_stop
            targets = target_states_from_plan(
                [target.model_dump(mode="json") for target in plan.take_profit_plan.targets],
                total_shares=shares,
                initial_shares=shares,
            )
            exit_rules = plan.exit_rules or _default_exit_rules(
                plan.strategy, self.config
            )
            range_midpoint = self._range_midpoint_for(plan)
            event_expiry = self._event_expiry_for(
                plan, exit_rules, day
            )
            evidence = plan.candidate.evidence or {}
            events_for_code = (
                (self.events_by_code or {}).get(plan.candidate.ts_code, [])
                if self.events_by_code is not None
                else []
            )
            event_id = (
                str(evidence.get("event_id"))
                if evidence.get("event_id") is not None
                else None
            )
            events_available = (
                plan.strategy != Strategy.F
                or (
                    self.events_by_code is not None
                    and (
                        event_id is None
                        or any(
                            str(event.get("event_id")) == event_id
                            for event in events_for_code
                        )
                    )
                )
            )
            if plan.strategy == Strategy.F and not events_available:
                self.missing_data.append(
                    f"{day.isoformat()}:{plan.candidate.ts_code}: "
                    "历史事件覆盖缺失，F 事件否定/撤回无法完整验证"
                )
            position = SimPosition(
                ts_code=plan.candidate.ts_code,
                sector=plan.candidate.sector,
                # Legacy/template-A plans may omit the explicit strategy; the
                # candidate template still identifies them as A.
                strategy=plan.strategy or Strategy.A,
                sub_mode=plan.sub_mode.value if plan.sub_mode else None,
                shares=shares,
                initial_shares=shares,
                entry_price=float(fill_price),
                initial_stop=initial_stop,
                current_stop=initial_stop,
                opened_on=day,
                atr14=float(plan.stop_plan.atr14),
                highest=float(fill_price),
                targets=targets,
                time_stop_days=plan.stop_plan.time_stop_days,
                trailing_rule=plan.take_profit_plan.trailing_rule,
                current_price=float(fill_price),
                add_on_count=0,
                initial_r=initial_r,
                entry_fee_total=float(fees),
                entry_fee_remaining=float(fees),
                ma5=self._ma5_for(plan.candidate.ts_code),
                range_midpoint=range_midpoint,
                exit_rules=exit_rules,
                event_expiry=event_expiry,
                event_data_available=events_available,
                event_id=event_id,
                last_new_high_on=day,
                primary_plan_id=plan.id,
            )
            self.positions[position.ts_code] = position
        else:
            prior = position.shares
            total = prior + shares
            position.entry_price = (
                position.entry_price * prior + float(fill_price) * shares
            ) / total
            position.shares = total
            position.entry_fee_total += float(fees)
            position.entry_fee_remaining += float(fees)
            if float(fill_price) > position.highest + 1e-9:
                position.highest = float(fill_price)
                position.last_new_high_on = day
            is_add_on = bool(
                plan.position_plan.is_add_on
                or (
                    position.primary_plan_id
                    and position.primary_plan_id != plan.id
                )
            )
            if is_add_on:
                position.add_on_count += 1
                additional = target_states_from_plan(
                    [target.model_dump(mode="json") for target in plan.take_profit_plan.targets],
                    total_shares=total,
                    initial_shares=shares,
                )
                merged: dict[float, TargetState] = {
                    float(target.price): target for target in position.targets
                }
                for target in additional:
                    key = float(target.price)
                    if key in merged:
                        merged[key].planned_shares += target.planned_shares
                    else:
                        merged[key] = target
                position.targets = list(merged.values())
            else:
                # The same plan filling in several tranches is still one
                # initial batch, not an add-on.
                position.initial_shares = total
                position.targets = target_states_from_plan(
                    [target.model_dump(mode="json") for target in plan.take_profit_plan.targets],
                    total_shares=total,
                    initial_shares=total,
                )
        order.remaining -= shares
        order.filled_shares += shares
        self.filled_shares += shares
        return True

    def _ma5_for(self, code: str) -> float | None:
        rows = self.recent_bars.get(code) or []
        closes = [row["close"] for row in rows if row["close"] > 0][-5:]
        if len(closes) < 5:
            return None
        return sum(closes) / len(closes)

    def _atr14_for(self, code: str) -> float | None:
        rows = self.recent_bars.get(code) or []
        if len(rows) < 15:
            return None
        ranges: list[float] = []
        for index in range(1, len(rows)):
            high = rows[index]["high"]
            low = rows[index]["low"]
            previous_close = rows[index - 1]["close"]
            ranges.append(
                max(
                    high - low,
                    abs(high - previous_close),
                    abs(low - previous_close),
                )
            )
        window = ranges[-14:]
        if len(window) < 14:
            return None
        return sum(window) / len(window)

    def _update_trailing_inputs(self, position: SimPosition, code: str) -> None:
        ma5 = self._ma5_for(code)
        if ma5 is not None:
            position.ma5 = ma5
        atr = self._atr14_for(code)
        if atr is not None:
            position.atr14 = atr

    @staticmethod
    def _date_value(value) -> date | None:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None

    def _range_midpoint_for(self, plan: TradePlan) -> float | None:
        evidence = plan.candidate.evidence or {}
        upper = evidence.get("range_upper")
        lower = evidence.get("range_lower")
        try:
            if upper is None or lower is None:
                return None
            return (float(upper) + float(lower)) / 2
        except (TypeError, ValueError):
            return None

    def _event_expiry_for(
        self, plan: TradePlan, rules: ExitRules, entry_day: date
    ) -> date | None:
        if plan.strategy != Strategy.F:
            return None
        evidence = plan.candidate.evidence or {}
        base = (
            self._date_value(evidence.get("event_ann_date"))
            or self._date_value(evidence.get("first_seen_at"))
            or self._date_value(plan.signal_time)
            or (
                plan.candidate.data_time.date()
                if plan.candidate.data_time is not None
                else None
            )
            or entry_day
        )
        days = (
            rules.event_expiry_days
            or self.config.strategies.f.event_window_days
        )
        if self.calendar is not None:
            anchor = base
            if not self.calendar.is_trading_day(anchor):
                upcoming = [
                    item for item in self.calendar.dates if item >= anchor
                ]
                if upcoming:
                    anchor = upcoming[0]
            if self.calendar.is_trading_day(anchor):
                try:
                    return self.calendar.add_trading_days(anchor, days)
                except ValueError:
                    pass
        return base + timedelta(days=days)

    def _update_event_state(self, position: SimPosition, day: date) -> None:
        if position.strategy != Strategy.F or not position.event_data_available:
            return
        for event in (self.events_by_code or {}).get(position.ts_code, []):
            event_day = _event_day(event)
            if event_day is None or event_day > day:
                continue
            same_event = (
                position.event_id is not None
                and str(event.get("event_id")) == position.event_id
            )
            if not same_event and event_day < position.opened_on:
                continue
            if event.get("retracted") or (
                event.get("direction") == "negative"
                and event.get("verification_status") == "verified"
            ):
                position.event_invalid = True
                return

    def _days_since_new_high(
        self, position: SimPosition, day: date
    ) -> int | None:
        if position.last_new_high_on is None:
            return None
        current_index = self.day_index.get(day)
        high_index = self.day_index.get(position.last_new_high_on)
        if current_index is None or high_index is None:
            return max(0, (day - position.last_new_high_on).days)
        return max(0, current_index - high_index)

    def _policy(self, position: SimPosition) -> ExitPolicy:
        return ExitPolicy(
            strategy=position.strategy,
            sub_mode=position.sub_mode,
            strategy_version="backtest",
            entry_price=position.entry_price,
            initial_stop=position.initial_stop,
            initial_r=position.initial_r,
            current_stop=position.current_stop,
            highest_price=position.highest,
            atr14=position.atr14,
            ma5=position.ma5,
            range_midpoint=position.range_midpoint,
            targets=tuple(position.targets),
            exit_rules=position.exit_rules,
            event_expiry=position.event_expiry,
            event_invalid=position.event_invalid,
        )

    def _check_exit(
        self,
        position: SimPosition,
        day: date,
        row: dict,
    ) -> list[SimTrade]:
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        _, down_limit = _limit_prices(row, open_)
        if down_limit is not None and open_ <= down_limit + 0.001:
            # Sealed limit-down: do not pretend the position could be sold.
            return []
        policy = self._policy(position)
        observation = ExitObservation(
            open=open_,
            high=high,
            low=low,
            close=close,
            price=close,
            session=ExitSession.DAILY,
            intrabar_stop=True,
            days_held=position.days_held,
            days_since_new_high=self._days_since_new_high(position, day),
            last_new_high_on=position.last_new_high_on,
            data_time=datetime.combine(
                day, time(15, 0), tzinfo=SHANGHAI
            ),
            data_ok=True,
        )
        decision = evaluate_exit(
            policy,
            observation,
            total_shares=position.shares,
            available_shares=position.shares,
        )
        if decision.action.value == "data_anomaly":
            return []
        if decision.action.value == "exit":
            remaining = position.shares
            if position.current_stop > 0 and (
                low <= position.current_stop + 1e-9
            ):
                fill = open_ if open_ < position.current_stop else position.current_stop
                reason = "stop"
            else:
                fill = close
                reason = {
                    "time_stop": "time",
                    "event_invalidated": "event",
                    "event_expired": "event",
                    "targets_complete": "target",
                    "hard_stop": "stop",
                }.get(decision.reason_code, "exit")
            trade = self._exit(position, day, fill, reason, remaining)
            return [trade] if trade is not None else []
        if decision.action.value == "reduce":
            trades: list[SimTrade] = []
            for _index, target_price, shares in decision.target_reductions:
                if shares <= 0 or position.shares <= 0:
                    continue
                shares = min(shares, position.shares)
                fill = max(float(target_price), open_)
                trade = self._partial_exit(
                    position, day, fill, shares, "target"
                )
                if trade is not None:
                    trades.append(trade)
                existing = next(
                    (
                        target
                        for target in position.targets
                        if abs(float(target.price) - float(target_price)) < 1e-9
                    ),
                    None,
                )
                if existing is not None:
                    existing.filled_shares = min(
                        existing.planned_shares,
                        existing.filled_shares + shares,
                    )
                    existing.status = (
                        "done"
                        if existing.filled_shares >= existing.planned_shares
                        else "partial"
                    )
                if position.shares <= 0:
                    break
            if position.shares > 0:
                # The stop proposal is for the next check, after today's
                # target execution; it is never applied retroactively.
                self._update_stop(position, high, close, day=day)
            return trades
        self._update_stop(position, high, close, day=day)
        return []

    def _update_stop(
        self,
        position: SimPosition,
        high: float,
        close: float,
        *,
        day: date | None = None,
    ) -> None:
        proposed = proposed_trailing_stop(
            self._policy(position), high=high, close=close
        )
        if proposed is None:
            return
        candidate = max(position.current_stop, proposed)
        if candidate <= position.current_stop + 1e-9:
            return
        # Effective from the next processed trading day, never on the bar
        # that produced the calculation.
        position.pending_stop = max(
            position.pending_stop or 0.0, candidate
        )
        position.pending_stop_source_day = day or position.opened_on

    def _allocate_entry_fee(self, position: SimPosition, shares: int) -> float:
        if position.shares <= 0:
            return 0.0
        allocated = position.entry_fee_remaining * (shares / position.shares)
        position.entry_fee_remaining = max(
            0.0, position.entry_fee_remaining - allocated
        )
        return allocated

    def _partial_exit(
        self,
        position: SimPosition,
        day: date,
        fill: float,
        shares: int,
        reason: str,
    ) -> SimTrade | None:
        shares = min(shares, position.shares)
        if shares <= 0:
            return None
        slip = (
            self.config.costs.slippage_b
            if position.strategy == Strategy.B
            else self.config.costs.slippage_a
        )
        fill_price = price(fill * (1 - slip))
        exit_fees = calculate_fees(
            side="sell",
            price=fill_price,
            shares=shares,
            costs=self.config.costs,
        ).total
        entry_fee = self._allocate_entry_fee(position, shares)
        self.cash += float(fill_price) * shares - float(exit_fees)
        self.exit_fees_paid += float(exit_fees)
        risk = position.initial_r
        r_multiple = (
            (float(fill_price) - position.entry_price) / risk if risk > 0 else 0.0
        )
        pnl = (
            (float(fill_price) - position.entry_price) * shares
            - float(exit_fees)
            - entry_fee
        )
        trade = SimTrade(
            ts_code=position.ts_code,
            strategy=position.strategy.value if position.strategy else "",
            sub_mode=position.sub_mode,
            entry_date=position.opened_on,
            exit_date=day,
            entry_price=position.entry_price,
            exit_price=float(fill_price),
            shares=shares,
            fees=float(exit_fees) + entry_fee,
            r_multiple=r_multiple,
            pnl=pnl,
            exit_reason=reason,
        )
        self.trades.append(trade)
        position.shares -= shares
        if position.shares <= 0:
            position.last_exit_reason = reason
            self.positions.pop(position.ts_code, None)
        return trade

    def _exit(
        self,
        position: SimPosition,
        day: date,
        fill: float,
        reason: str,
        shares: int | None = None,
    ) -> SimTrade | None:
        return self._partial_exit(
            position, day, fill, shares if shares is not None else position.shares, reason
        )

    def add_plans(self, plans: list[TradePlan], day: date) -> None:
        for plan in plans:
            if plan.status != PlanStatus.PENDING_EXECUTION:
                continue
            self.planned_shares += plan.position_plan.shares
            self.pending.append(
                PendingOrder(
                    plan=plan,
                    remaining=plan.position_plan.shares,
                    created_on=day,
                )
            )

    @staticmethod
    def _group_metrics(trades: list[SimTrade]) -> dict[str, dict]:
        groups: dict[str, dict] = {}
        for trade in trades:
            key = trade.strategy
            item = groups.setdefault(
                key,
                {"trades": 0, "pnl": 0.0, "wins": 0, "avg_r": 0.0},
            )
            item["trades"] += 1
            item["pnl"] += trade.pnl
            item["wins"] += int(trade.pnl > 0)
            item["avg_r"] += trade.r_multiple
        for item in groups.values():
            item["win_rate"] = item["wins"] / item["trades"] if item["trades"] else 0.0
            item["avg_r"] = item["avg_r"] / item["trades"] if item["trades"] else 0.0
        return groups

    def metrics(self) -> dict:
        final_equity = self.equity()
        curve = [point["equity"] for point in self.equity_curve] or [
            self.initial_equity
        ]
        peak = self.initial_equity
        max_drawdown = 0.0
        for value in curve:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = max(max_drawdown, 1 - value / peak)
        returns = [
            curve[index] / curve[index - 1] - 1
            for index in range(1, len(curve))
            if curve[index - 1] > 0
        ]
        years = len(curve) / 252 if curve else 0
        total_return = final_equity / self.initial_equity - 1
        annualized = (
            (final_equity / self.initial_equity) ** (1 / years) - 1
            if years > 0 and final_equity > 0
            else 0.0
        )
        volatility = pstdev(returns) if len(returns) > 1 else 0.0
        sharpe = (
            fmean(returns) / volatility * sqrt(252)
            if volatility > 0 and returns
            else 0.0
        )
        downside = [value for value in returns if value < 0]
        downside_vol = pstdev(downside) if len(downside) > 1 else 0.0
        sortino = (
            fmean(returns) / downside_vol * sqrt(252)
            if downside_vol > 0 and returns
            else 0.0
        )
        calmar = annualized / max_drawdown if max_drawdown > 0 else 0.0
        wins = [trade for trade in self.trades if trade.pnl > 0]
        losses = [trade for trade in self.trades if trade.pnl < 0]
        gross_profit = sum(trade.pnl for trade in wins)
        gross_loss = abs(sum(trade.pnl for trade in losses))
        average_equity = fmean(curve) if curve else self.initial_equity
        traded_value = sum(
            (trade.entry_price + trade.exit_price) * trade.shares
            for trade in self.trades
        )
        holding_days = [
            max(0, (trade.exit_date - trade.entry_date).days)
            for trade in self.trades
        ]
        sub_mode_groups: dict[str, dict] = {}
        for trade in self.trades:
            item = sub_mode_groups.setdefault(
                trade.sub_mode or "unknown",
                {"trades": 0, "pnl": 0.0, "wins": 0},
            )
            item["trades"] += 1
            item["pnl"] += trade.pnl
            item["wins"] += int(trade.pnl > 0)
        for item in sub_mode_groups.values():
            item["win_rate"] = item["wins"] / item["trades"] if item["trades"] else 0.0
        realized = sum(trade.pnl for trade in self.trades)
        unrealized = sum(
            (position.current_price - position.entry_price) * position.shares
            - position.entry_fee_remaining
            for position in self.positions.values()
        )
        return {
            "initial_equity": self.initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "annualized_return": annualized,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0.0,
            "avg_r": (
                fmean(trade.r_multiple for trade in self.trades)
                if self.trades
                else 0.0
            ),
            "expectancy": (
                fmean(trade.pnl for trade in self.trades) if self.trades else 0.0
            ),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "turnover": (
                traded_value / average_equity / years
                if average_equity > 0 and years > 0
                else 0.0
            ),
            "fill_rate": (
                self.filled_shares / self.planned_shares
                if self.planned_shares > 0
                else 0.0
            ),
            "avg_holding_days": fmean(holding_days) if holding_days else 0.0,
            "capital_utilization": (
                fmean(self.invested_ratios) if self.invested_ratios else 0.0
            ),
            "by_strategy": self._group_metrics(self.trades),
            "by_sub_mode": sub_mode_groups,
            "missing_data": self.missing_data,
            "expired_orders": self.expired_orders,
            "cancelled_orders": self.cancelled_orders,
            "invalidated_orders": self.invalidated_orders,
            "cancel_unconfirmed_orders": self.cancel_unconfirmed_orders,
            "entry_fees_paid": self.entry_fees_paid,
            "exit_fees_paid": self.exit_fees_paid,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
        }
