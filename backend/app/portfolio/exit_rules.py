"""Shared, executable exit rules.

Both the live position review and the backtest simulator call these pure
functions.  A decision is derived from structured fields, never by parsing the
display text stored in a plan.  The functions do not mutate state: callers
apply the returned stop proposal *after* the exit check so a stop that can only
be known at the close is never used against an earlier low of the same day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from math import isfinite
from typing import TypeGuard

from app.domain.enums import ExitSession, PositionAction, Strategy
from app.domain.models import ExitRules, TargetState


@dataclass(frozen=True)
class ExitPolicy:
    strategy: Strategy | None
    sub_mode: str | None = None
    strategy_version: str = ""
    entry_price: float = 0.0
    initial_stop: float | None = None
    initial_r: float | None = None
    current_stop: float | None = None
    highest_price: float = 0.0
    atr14: float | None = None
    ma5: float | None = None
    range_midpoint: float | None = None
    targets: tuple[TargetState, ...] = ()
    exit_rules: ExitRules = field(default_factory=ExitRules)
    event_expiry: date | None = None
    event_invalid: bool = False
    needs_reconciliation: bool = False


@dataclass(frozen=True)
class ExitObservation:
    close: float
    high: float
    low: float
    open: float = 0.0
    price: float | None = None
    session: ExitSession = ExitSession.DAILY
    intrabar_stop: bool = False
    days_held: int = 0
    data_time: datetime | None = None
    data_ok: bool = True
    data_issues: tuple[str, ...] = ()

    @property
    def latest_price(self) -> float:
        if self.price is not None:
            return float(self.price)
        return float(self.close)


@dataclass(frozen=True)
class ExitDecision:
    action: PositionAction
    reason_code: str
    explanation: str
    suggested_shares: int = 0
    restricted_shares: int = 0
    proposed_stop: float | None = None
    target_reductions: tuple[tuple[int, float, int], ...] = ()
    invalidation: str = ""
    data_issues: tuple[str, ...] = ()
    trigger_price: float | None = None

    @property
    def is_exit(self) -> bool:
        return self.action in (
            PositionAction.EXIT,
            PositionAction.REDUCE,
            PositionAction.DATA_ANOMALY,
        )


def _valid(value: float | None) -> TypeGuard[float]:
    return value is not None and isfinite(float(value)) and float(value) > 0


def _lots_down(value: float) -> int:
    return int(max(0.0, value) // 100 * 100)


def _sellable(total: int, available: int, session: ExitSession) -> tuple[int, int]:
    """Return (suggested, restricted) for an exit.

    Intraday exits can only use shares that are already sellable.  Daily
    review suggestions apply to the next session, when all T+1 restrictions
    have cleared, so the full position can be shown as actionable.
    """
    if session == ExitSession.INTRADAY:
        available = max(0, min(available, total))
        return available, total - available
    return total, 0


def proposed_trailing_stop(
    policy: ExitPolicy,
    *,
    high: float,
    close: float,
) -> float | None:
    """Return the protection line that should apply to the next check.

    The caller must not use this value to decide the current bar's exit.  It
    is computed only from the current high/close and the rule snapshot.
    """
    entry = policy.entry_price
    stop = policy.current_stop
    if not _valid(entry) or not _valid(stop):
        return None
    risk = policy.initial_r
    if not _valid(risk):
        initial_stop = policy.initial_stop
        if not _valid(initial_stop):
            return None
        risk = entry - float(initial_stop)
    if risk is None or risk <= 0:
        return None
    current = float(stop)
    new_stop = current
    rules = policy.exit_rules
    trailing = rules.trailing
    atr = policy.atr14 if _valid(policy.atr14) else None
    reached_1r = high >= entry + risk - 1e-9
    reached_2r = high >= entry + 2 * risk - 1e-9

    if trailing == "breakeven_1r_2r_atr":
        if reached_1r:
            new_stop = max(new_stop, entry)
        if reached_2r:
            new_stop = max(new_stop, entry + risk)
            if atr is not None:
                new_stop = max(new_stop, close - 1.5 * atr)
    elif trailing == "breakeven_1r_2r_ma5_atr":
        # B/D: breakeven at +1R; the remaining position trails on MA5 and
        # 1.5 ATR once the +2R tranche has been reached.
        if reached_1r:
            new_stop = max(new_stop, entry)
        if reached_2r:
            if _valid(policy.ma5):
                new_stop = max(new_stop, float(policy.ma5))
            if atr is not None:
                new_stop = max(new_stop, close - 1.5 * atr)
    elif trailing == "breakeven_1r_2r_range_atr":
        if reached_1r:
            new_stop = max(new_stop, entry)
        if reached_2r:
            if _valid(policy.range_midpoint):
                new_stop = max(new_stop, float(policy.range_midpoint))
            if atr is not None:
                new_stop = max(new_stop, close - 1.5 * atr)
    elif trailing == "breakeven_1r_atr":
        if reached_1r:
            new_stop = max(new_stop, entry)
            if atr is not None:
                new_stop = max(new_stop, close - 1.5 * atr)
    return max(current, new_stop)


def evaluate_exit(
    policy: ExitPolicy,
    observation: ExitObservation,
    *,
    total_shares: int,
    available_shares: int,
) -> ExitDecision:
    """Evaluate one position against the shared exit rules.

    Order of precedence: data integrity, event risk, hard stop, time exit,
    take-profit tranches, then hold/trailing proposal.  Add-on eligibility is
    applied by the service after this function, so an exit can never be
    overridden by an add-on suggestion.
    """
    issues = list(observation.data_issues)
    if not observation.data_ok:
        return ExitDecision(
            action=PositionAction.DATA_ANOMALY,
            reason_code="data_unavailable",
            explanation="行情缺失或不可用；不能默认判定为安全持有。",
            data_issues=tuple(issues or ["quote unavailable"]),
        )
    if not _valid(policy.initial_stop) or not _valid(policy.current_stop):
        return ExitDecision(
            action=PositionAction.DATA_ANOMALY,
            reason_code="stop_missing",
            explanation=(
                "保护线缺失或无效，标记为待补全；"
                "不得把买入价当作有效止损。"
            ),
            data_issues=tuple(issues or ["stop missing"]),
        )
    latest = observation.latest_price
    high = observation.high
    low = observation.low
    if not all(_valid(value) for value in (latest, high, low)):
        return ExitDecision(
            action=PositionAction.DATA_ANOMALY,
            reason_code="invalid_price",
            explanation="行情价格无效，暂停给出持有建议。",
            data_issues=tuple(issues or ["invalid price"]),
        )

    total = max(0, int(total_shares))
    stop = float(policy.current_stop)
    sellable, restricted = _sellable(
        total, max(0, int(available_shares)), observation.session
    )

    def _exit(reason: str, explanation: str) -> ExitDecision:
        return ExitDecision(
            action=PositionAction.EXIT,
            reason_code=reason,
            explanation=explanation,
            suggested_shares=sellable,
            restricted_shares=restricted,
            invalidation="保护线/结构已失效，退出优先于任何新增建议。",
            data_issues=tuple(issues),
            trigger_price=latest,
        )

    hard_hit = (
        (observation.intrabar_stop and low <= stop + 1e-9)
        or latest <= stop + 1e-9
    )
    if hard_hit:
        return _exit(
            "hard_stop",
            f"价格 {latest:.2f} 触及保护线 {stop:.2f}；硬风险退出优先。",
        )

    if policy.event_invalid:
        return _exit(
            "event_invalidated",
            "事件被撤回、证伪或转为负面，事件驱动逻辑失效。",
        )
    if (
        policy.event_expiry is not None
        and observation.data_time is not None
        and observation.data_time.date() >= policy.event_expiry
    ):
        return _exit(
            "event_expired",
            "事件窗口已到期且未兑现，按事件退出规则处理。",
        )

    if policy.needs_reconciliation or policy.strategy is None:
        return ExitDecision(
            action=PositionAction.DATA_ANOMALY,
            reason_code="position_incomplete",
            explanation=(
                "持仓缺少策略/保护线/初始风险记录，标记为待补全；"
                "不得把买入价当作有效止损。"
            ),
            data_issues=tuple(issues or ["position needs reconciliation"]),
        )

    rules = policy.exit_rules
    close = observation.close
    if rules.time_exit == "below_entry":
        if observation.days_held >= rules.time_stop_days and close < policy.entry_price:
            return _exit(
                "time_stop",
                f"持有 {observation.days_held} 个交易日仍未达到 +1R 且低于成本。",
            )
    elif rules.time_exit == "no_new_high" and (
        observation.days_held >= rules.time_stop_days
        and policy.highest_price <= policy.entry_price + 1e-9
    ):
        return _exit(
            "time_stop",
            f"持有 {observation.days_held} 个交易日未创新高。",
        )

    # The stop proposal is computed from today's high/close but only included
    # as the next-day protection line; it is never used for today's exit.
    proposed = proposed_trailing_stop(policy, high=high, close=close)

    # Take-profit tranches.  Stop was checked first, so a same-day stop always
    # wins over a target.  A tranche that was already filled is skipped.
    reductions: list[tuple[int, float, int]] = []
    suggested = 0
    for index, target in enumerate(policy.targets):
        if target.status == "done" or target.remaining_shares <= 0:
            continue
        if high < float(target.price) - 1e-9:
            continue
        shares = target.remaining_shares
        if suggested + shares >= total:
            shares = max(0, total - suggested)
        shares = min(shares, max(0, total - suggested))
        if shares <= 0:
            continue
        reductions.append((index, float(target.price), shares))
        suggested += shares
    if suggested > 0:
        suggested = min(suggested, sellable)
        if suggested > 0:
            if suggested >= total:
                return ExitDecision(
                    action=PositionAction.EXIT,
                    reason_code="targets_complete",
                    explanation="所有分批止盈目标均已触及，剩余仓位按退出处理。",
                    suggested_shares=suggested,
                    restricted_shares=max(0, total - suggested),
                    proposed_stop=proposed,
                    target_reductions=tuple(reductions),
                    data_issues=tuple(issues),
                    trigger_price=latest,
                )
            return ExitDecision(
                action=PositionAction.REDUCE,
                reason_code=f"target_{reductions[0][0] + 1}",
                explanation=(
                    "触及分批止盈目标，建议减仓 "
                    f"{suggested} 股；止损仍优先于止盈。"
                ),
                suggested_shares=suggested,
                restricted_shares=max(0, total - sellable),
                proposed_stop=proposed,
                target_reductions=tuple(reductions),
                data_issues=tuple(issues),
                trigger_price=latest,
            )

    return ExitDecision(
        action=PositionAction.HOLD,
        reason_code="hold",
        explanation=(
            "未触发退出条件；"
            + (
                f"下一交易日保护线可上移至 {proposed:.2f}。"
                if proposed is not None and proposed > stop + 1e-9
                else "维持现有保护线。"
            )
        ),
        proposed_stop=proposed,
        data_issues=tuple(issues),
        trigger_price=latest,
    )


def target_states_from_plan(
    targets: list[dict] | list[object],
    total_shares: int,
    initial_shares: int,
) -> list[TargetState]:
    """Build plan tranche state using the initial (not remaining) position.

    Fractions are applied to the planned position size so later sells do not
    cause a target to disappear or be recomputed from the current remainder.
    """
    result: list[TargetState] = []
    for item in targets:
        if isinstance(item, dict):
            price = float(item.get("price") or 0)
            fraction = float(item.get("fraction") or 0.5)
        else:
            price = float(getattr(item, "price", 0))
            fraction = float(getattr(item, "fraction", 0) or 0.5)
        if price <= 0 or fraction <= 0:
            continue
        planned = max(100, _lots_down(float(initial_shares or total_shares) * fraction))
        result.append(
            TargetState(
                price=Decimal(str(price)),
                fraction=fraction,
                planned_shares=min(planned, max(0, total_shares)),
                filled_shares=0,
                status="pending",
            )
        )
    return result
