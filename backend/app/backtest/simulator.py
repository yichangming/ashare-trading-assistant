"""Conservative daily OHLC simulator for plan execution.

This module simulates fills and exits with daily bars.  It deliberately does
not invent an intrabar path: same-day stop and target hits resolve to the stop,
gaps use the open, and shares bought today cannot be sold today (T+1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from math import sqrt
from statistics import fmean, pstdev

import polars as pl

from app.core.full_config import FullSystemConfig
from app.domain.enums import PlanStatus, Strategy, SubMode, Template
from app.domain.models import (
    AccountState,
    PendingExposure,
    PositionState,
    TradePlan,
)
from app.domain.money import price
from app.portfolio.fees import calculate_fees


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
    targets: list[tuple[float, float]]
    target_index: int = 0
    time_stop_days: int = 10
    trailing_rule: str = ""
    days_held: int = 0
    current_price: float = 0.0
    add_on_count: int = 0


@dataclass
class PendingOrder:
    plan: TradePlan
    remaining: int
    created_on: date


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

    def position_value(self) -> float:
        return sum(
            (position.current_price or position.entry_price) * position.shares
            for position in self.positions.values()
        )

    def equity(self) -> float:
        return self.cash + self.position_value()

    def account_snapshot(self, as_of: date) -> AccountState:
        positions = [
            PositionState(
                ts_code=position.ts_code,
                sector=position.sector,
                shares=position.shares,
                available_shares=0 if position.opened_on >= as_of else position.shares,
                cost_price=price(position.entry_price),
                current_stop=price(position.current_stop),
                current_price=price(getattr(position, "current_price", position.entry_price)),
                template=Template.B
                if position.strategy == Strategy.B
                else Template.A,
                strategy=position.strategy,
                sub_mode=SubMode(position.sub_mode) if position.sub_mode else None,
                initial_shares=position.initial_shares,
                add_on_count=position.add_on_count,
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
        new_trades: list[SimTrade] = []
        # Exits first; positions bought today are protected by T+1.
        for code in list(self.positions):
            position = self.positions[code]
            row = by_code.get(code)
            if row is None:
                self.missing_data.append(
                    f"{day.isoformat()}:{code}: no bar (suspended/delisted?)"
                )
                continue
            high = float(row["high"])
            close = float(row["close"])
            position.current_price = close
            position.highest = max(position.highest, high)
            if position.opened_on >= day:
                continue
            if _suspended(row):
                continue
            position.days_held += 1
            exit_trade = self._check_exit(position, day, row)
            if exit_trade is not None:
                new_trades.append(exit_trade)
            else:
                self._trail(position, high, close)
        # Entries from previously approved plans.
        remaining_orders: list[PendingOrder] = []
        for order in self.pending:
            if order.created_on >= day:
                remaining_orders.append(order)
                continue
            if day > order.plan.expires_on:
                continue
            row = by_code.get(order.plan.candidate.ts_code)
            if row is None:
                remaining_orders.append(order)
                continue
            filled = self._try_entry(order, day, row)
            if filled and order.remaining > 0:
                remaining_orders.append(order)
        self.pending = remaining_orders
        equity = self.equity()
        self.peak_equity = max(self.peak_equity, equity)
        self.equity_curve.append(
            {"trade_date": day.isoformat(), "equity": equity, "cash": self.cash}
        )
        self.invested_ratios.append(
            max(0.0, (equity - self.cash) / equity) if equity > 0 else 0.0
        )
        return new_trades

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
            order.remaining = 0
            return False
        if entry <= open_ <= maximum:
            fill = open_
        elif open_ < entry <= high:
            fill = entry
        else:
            return False
        volume = float(row.get("vol") or 0)
        if volume <= 0:
            return False
        max_shares = int(volume * 0.01 // 100 * 100)
        max_shares = max(100, max_shares)
        shares = min(order.remaining, max_shares)
        if shares < 100:
            return False
        slip = (
            self.config.costs.slippage_b
            if plan.strategy == Strategy.B
            else self.config.costs.slippage_a
        )
        # Keep the simulated fill inside the plan's execution band; slippage is
        # absorbed at the cap rather than silently exceeding max_acceptable.
        fill_price = price(min(fill * (1 + slip), maximum))
        fees = calculate_fees(
            side="buy",
            price=fill_price,
            shares=shares,
            costs=self.config.costs,
        ).total
        cost = float(fill_price) * shares + float(fees)
        if cost > self.cash:
            order.remaining = 0
            return False
        self.cash -= cost
        position = self.positions.get(plan.candidate.ts_code)
        if position is None:
            position = SimPosition(
                ts_code=plan.candidate.ts_code,
                sector=plan.candidate.sector,
                strategy=plan.strategy,
                sub_mode=plan.sub_mode.value if plan.sub_mode else None,
                shares=shares,
                initial_shares=shares,
                entry_price=float(fill_price),
                initial_stop=float(plan.stop_plan.initial_stop),
                current_stop=float(plan.stop_plan.initial_stop),
                opened_on=day,
                atr14=float(plan.stop_plan.atr14),
                highest=float(fill_price),
                targets=[
                    (float(target.price), float(target.fraction))
                    for target in plan.take_profit_plan.targets
                ],
                time_stop_days=plan.stop_plan.time_stop_days,
                trailing_rule=plan.take_profit_plan.trailing_rule,
                current_price=float(fill_price),
            )
            self.positions[position.ts_code] = position
        else:
            total = position.shares + shares
            position.entry_price = (
                position.entry_price * position.shares + float(fill_price) * shares
            ) / total
            position.shares = total
            position.add_on_count += 1
        order.remaining -= shares
        self.filled_shares += shares
        return True

    def _check_exit(
        self,
        position: SimPosition,
        day: date,
        row: dict,
    ) -> SimTrade | None:
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        _, down_limit = _limit_prices(row, open_)
        if down_limit is not None and open_ <= down_limit + 0.001:
            # Sealed limit-down: do not pretend the position could be sold.
            return None
        stop = position.current_stop
        if low <= stop:
            fill = open_ if open_ < stop else stop
            return self._exit(position, day, fill, "stop")
        # Targets are processed in order; stop has priority on same-day hits.
        while position.target_index < len(position.targets):
            target_price, fraction = position.targets[position.target_index]
            if high < target_price:
                break
            target_shares = max(
                100, int(position.initial_shares * fraction // 100 * 100)
            )
            target_shares = min(target_shares, position.shares)
            if target_shares < 100:
                break
            # A gap through the target fills at the better open price.
            fill = max(target_price, open_)
            self._partial_exit(position, day, fill, target_shares, "target")
            position.target_index += 1
            if position.shares <= 0:
                return self.trades[-1]
        if (
            position.shares > 0
            and position.days_held >= position.time_stop_days
            and close < position.entry_price
        ):
            return self._exit(position, day, close, "time")
        return None

    def _trail(self, position: SimPosition, high: float, close: float) -> None:
        entry = position.entry_price
        risk = entry - position.initial_stop
        if risk <= 0:
            return
        if high >= entry + 2 * risk:
            position.current_stop = max(position.current_stop, entry + risk)
        elif high >= entry + risk:
            position.current_stop = max(position.current_stop, entry)
        trailing = close - 1.5 * position.atr14
        if position.trailing_rule and "trail" in position.trailing_rule:
            position.current_stop = max(position.current_stop, trailing)

    def _partial_exit(
        self,
        position: SimPosition,
        day: date,
        fill: float,
        shares: int,
        reason: str,
    ) -> None:
        slip = (
            self.config.costs.slippage_b
            if position.strategy == Strategy.B
            else self.config.costs.slippage_a
        )
        fill_price = price(fill * (1 - slip))
        fees = calculate_fees(
            side="sell",
            price=fill_price,
            shares=shares,
            costs=self.config.costs,
        ).total
        self.cash += float(fill_price) * shares - float(fees)
        risk = position.entry_price - position.initial_stop
        r_multiple = (
            (float(fill_price) - position.entry_price) / risk if risk > 0 else 0.0
        )
        pnl = (float(fill_price) - position.entry_price) * shares - float(fees)
        trade = SimTrade(
            ts_code=position.ts_code,
            strategy=position.strategy.value if position.strategy else "",
            sub_mode=position.sub_mode,
            entry_date=position.opened_on,
            exit_date=day,
            entry_price=position.entry_price,
            exit_price=float(fill_price),
            shares=shares,
            fees=float(fees),
            r_multiple=r_multiple,
            pnl=pnl,
            exit_reason=reason,
        )
        self.trades.append(trade)
        position.shares -= shares
        if position.shares <= 0:
            self.positions.pop(position.ts_code, None)

    def _exit(
        self,
        position: SimPosition,
        day: date,
        fill: float,
        reason: str,
    ) -> SimTrade:
        self._partial_exit(position, day, fill, position.shares, reason)
        return self.trades[-1]

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
        }
