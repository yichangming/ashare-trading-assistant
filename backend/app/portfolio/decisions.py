"""Unified position action service.

The review job, intraday monitor, web positions page, daily report and the
backtest simulator all consume the exit rules in :mod:`exit_rules`.  This
service adds persistence, market-data freshness and notification state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.calendar import TradingCalendar
from app.core.full_config import FullSystemConfig
from app.data.interfaces import MarketDataProvider
from app.db.models import (
    EventLinkRow,
    EventRow,
    ExecutionRow,
    OrderRow,
    PlanRow,
    PositionRow,
)
from app.db.repositories import (
    NotificationRepository,
    PlanRepository,
    PositionRepository,
)
from app.domain.enums import (
    ExecutionSide,
    ExitSession,
    OrderStatus,
    PlanStatus,
    PositionAction,
    Strategy,
    SubMode,
    ValuationStatus,
)
from app.domain.models import ExitRules, PositionDecision, TargetState
from app.portfolio.exit_rules import (
    ExitObservation,
    ExitPolicy,
    evaluate_exit,
    target_states_from_plan,
)
from app.portfolio.market import (
    SHANGHAI,
    PricePoint,
    market_phase,
    next_session_open,
    points_from_bars,
)


@dataclass
class DecisionSummary:
    checked: int = 0
    holds: int = 0
    add_on_candidates: int = 0
    reduces: int = 0
    exits: int = 0
    data_anomalies: int = 0
    waits: int = 0
    stop_updates: int = 0
    notifications: int = 0


class PositionDecisionService:
    def __init__(
        self,
        session: Session,
        config: FullSystemConfig,
        *,
        account_id: str = "default",
        provider: MarketDataProvider | None = None,
    ):
        self.session = session
        self.config = config
        self.account_id = account_id
        self.provider = provider

    # ---- public API ---------------------------------------------------
    def evaluate_all(
        self,
        as_of: date,
        *,
        points: dict[str, PricePoint] | None = None,
        session_type: ExitSession = ExitSession.DAILY,
        now: datetime | None = None,
        persist: bool = True,
        notify: bool = False,
    ) -> list[PositionDecision]:
        moment = now or datetime.now(UTC)
        if points is None:
            points = self._daily_points(as_of)
        ma5 = self._ma5_map(as_of) if session_type == ExitSession.DAILY else {}
        calendar = self._calendar(as_of)
        decisions: list[PositionDecision] = []
        for position in PositionRepository.list_positions(
            self.session, self.account_id
        ):
            point = points.get(position.ts_code)
            if (
                point is not None
                and point.data_time is not None
                and position.opened_at is not None
                and point.data_time.date() < position.opened_at.date()
            ):
                # Never value a position with a bar from before it existed.
                point = None
            decision = self.evaluate_position(
                position,
                as_of=as_of,
                point=point,
                session_type=session_type,
                now=moment,
                ma5=ma5.get(position.ts_code),
                calendar=calendar,
                persist_state=persist,
            )
            if persist:
                self._persist(
                    position,
                    decision,
                    now=moment,
                )
            if notify:
                self._notify(
                    position,
                    decision,
                    as_of,
                    now=moment,
                    session_type=session_type,
                )
            decisions.append(decision)
        if persist:
            self.session.flush()
        return decisions

    def current_all(
        self,
        as_of: date,
        *,
        points: dict[str, PricePoint] | None = None,
        now: datetime | None = None,
    ) -> list[PositionDecision]:
        """Return the latest valid decision for every position.

        Page/API/SSE views must not recompute a previous-close decision and
        overwrite a newer intraday decision.  Persisted decisions are compared
        with a fresh post-close baseline by data time; stale snapshots are
        explicitly marked instead of being presented as a safe hold.
        """
        moment = now or datetime.now(SHANGHAI)
        if points is None:
            points = self._daily_points(as_of)
        calendar = self._calendar_for(max(as_of, moment.date()))
        ma5 = self._ma5_map(as_of)
        decisions: list[PositionDecision] = []
        for position in PositionRepository.list_positions(
            self.session, self.account_id
        ):
            point = points.get(position.ts_code)
            if (
                point is not None
                and point.data_time is not None
                and position.opened_at is not None
                and point.data_time.date() < position.opened_at.date()
            ):
                point = None
            fresh = self.evaluate_position(
                position,
                as_of=as_of,
                point=point,
                session_type=ExitSession.DAILY,
                now=moment,
                ma5=ma5.get(position.ts_code),
                calendar=calendar,
                persist_state=False,
            )
            persisted = self.latest(position.ts_code)
            chosen = self._choose_current(
                fresh, persisted, now=moment, calendar=calendar
            )
            decisions.append(chosen)
        return decisions

    def _choose_current(
        self,
        fresh: PositionDecision,
        persisted: PositionDecision | None,
        *,
        now: datetime,
        calendar: TradingCalendar | None,
    ) -> PositionDecision:
        chosen = fresh
        if persisted is not None and persisted.data_time is not None:
            fresh_time = self._aware_shanghai(fresh.data_time)
            persisted_time = self._aware_shanghai(persisted.data_time)
            if (
                fresh_time is None
                or persisted_time is None
                or persisted_time > fresh_time
            ):
                chosen = persisted
                if chosen.decision_source == "daily":
                    # Legacy payload: a persisted decision newer than the
                    # daily baseline can only have come from an intraday run.
                    chosen = chosen.model_copy(
                        update={"decision_source": "persisted"}
                    )
        phase = market_phase(now, calendar)
        return self._assess_current(
            chosen, now=now, phase=phase, calendar=calendar
        )

    def _assess_current(
        self,
        chosen: PositionDecision,
        *,
        now: datetime,
        phase: str,
        calendar: TradingCalendar | None,
    ) -> PositionDecision:
        moment = now.astimezone(SHANGHAI) if now.tzinfo else now.replace(
            tzinfo=SHANGHAI
        )
        day = moment.date()
        chosen = chosen.model_copy(update={"market_phase": phase})
        risk_actions = {
            PositionAction.EXIT,
            PositionAction.REDUCE,
            PositionAction.DATA_ANOMALY,
        }
        intraday = chosen.decision_source in ("intraday", "persisted")
        data_day = chosen.data_time.date() if chosen.data_time else None
        reason: str | None = None
        if data_day is None:
            reason = "decision_time_missing"
        elif intraday:
            expiry = self._aware_shanghai(chosen.valid_until)
            if expiry is not None and moment > expiry:
                reason = "intraday_quote_expired"
            elif data_day < day and phase in ("trading", "lunch"):
                reason = "intraday_quote_from_earlier_day"
            elif data_day < day and phase in ("post_close",):
                reason = "intraday_snapshot_before_latest_close"
        else:
            # Daily baseline: it is current as the latest completed close, but
            # during a newer trading session it is not a live decision.
            if phase in ("trading", "lunch") and data_day < day:
                reason = "intraday_decision_missing"
            elif phase == "post_close" and data_day < day:
                expected = self._latest_completed_trading_day(day, calendar)
                if expected is not None and data_day < expected:
                    reason = "post_close_data_missing"
            elif data_day < day and phase in ("holiday", "weekend", "pre_open"):
                # Weekend/holiday/pre-open: the previous close is the latest
                # available market state, not an error.
                expected = self._latest_completed_trading_day(day, calendar)
                if expected is not None and data_day < expected:
                    reason = "daily_data_stale"
        if reason is None:
            return chosen.model_copy(update={"is_current": True, "stale_reason": None})
        if chosen.action in risk_actions:
            explanation = (
                f"{chosen.explanation}（行情状态：{reason}；"
                "底层风险动作仍保留，不以过期行情改为持有。）"
            )
            return chosen.model_copy(
                update={
                    "is_current": False,
                    "stale_reason": reason,
                    "valuation_status": (
                        chosen.valuation_status
                        if chosen.valuation_status.value != "ok"
                        else ValuationStatus.STALE
                    ),
                    "explanation": explanation,
                }
            )
        return chosen.model_copy(
            update={
                "action": PositionAction.DATA_ANOMALY,
                "reason_code": "quote_stale",
                "explanation": (
                    f"最新决策数据时间为 {chosen.data_time}，状态 {reason}；"
                    "不能把过期/缺失行情视为安全持有。"
                ),
                "is_current": False,
                "stale_reason": reason,
                "valuation_status": ValuationStatus.STALE,
                "suggested_shares": 0,
                "covered_shares": 0,
                "confirmed": False,
            }
        )

    @staticmethod
    def _latest_completed_trading_day(
        day: date, calendar: TradingCalendar | None
    ) -> date | None:
        if calendar is not None and calendar.covers(day):
            candidates = [item for item in calendar.dates if item <= day]
            return candidates[-1] if candidates else None
        candidate = day
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
        return candidate

    def evaluate_position(
        self,
        position: PositionRow,
        *,
        as_of: date,
        point: PricePoint | None,
        session_type: ExitSession,
        now: datetime,
        ma5: float | None = None,
        calendar: TradingCalendar | None = None,
        persist_state: bool = True,
    ) -> PositionDecision:
        moment = now if now.tzinfo else now.replace(tzinfo=UTC)
        calendar = calendar or self._calendar_for(
            max(as_of, moment.astimezone(SHANGHAI).date())
        )
        effective_stop = self._effective_stop(
            position, point, moment, persist_state=persist_state
        )
        entry_price = self._entry_price(position)
        prior_high = float(position.highest_price or entry_price or 0)
        daily_bar = (
            session_type == ExitSession.DAILY
            and point is not None
            and point.usable
        )
        tick_high = float(point.high) if daily_bar and point is not None else 0.0
        new_high_today = tick_high > prior_high + 1e-9
        policy_high = max(prior_high, tick_high)
        last_high_on = position.last_new_high_on or (
            position.opened_at.date() if position.opened_at is not None else None
        )
        if new_high_today and point is not None and point.data_time is not None:
            last_high_on = point.data_time.date()
        days_since_high = self._days_since_high(last_high_on, as_of, calendar)
        policy = self._policy(
            position,
            ma5=ma5,
            as_of=as_of,
            calendar=calendar,
            current_stop=effective_stop,
            highest_price=policy_high,
        )
        if point is None:
            observation = ExitObservation(
                close=float(position.current_price or 0),
                high=float(position.current_price or 0),
                low=float(position.current_price or 0),
                session=session_type,
                intrabar_stop=session_type != ExitSession.DAILY,
                data_time=None,
                data_ok=False,
                data_issues=("缺少行情",),
                days_since_new_high=days_since_high,
                last_new_high_on=last_high_on,
            )
            valuation = ValuationStatus.MISSING
        else:
            observation = ExitObservation(
                open=point.open,
                high=point.high,
                low=point.low,
                close=point.close,
                price=point.price,
                session=session_type,
                intrabar_stop=session_type != ExitSession.DAILY,
                days_held=self._days_held(position, as_of, calendar),
                days_since_new_high=days_since_high,
                last_new_high_on=last_high_on,
                data_time=point.data_time,
                data_ok=point.usable,
                data_issues=point.issues,
            )
            valuation = point.status
        raw = evaluate_exit(
            policy,
            observation,
            total_shares=position.shares,
            available_shares=position.available_shares,
        )
        pending_shares, executed_shares, pending_orders = self._execution_totals(
            position
        )
        action = raw.action
        reason_code = raw.reason_code
        explanation = raw.explanation
        suggested = raw.suggested_shares
        covered = 0
        if action == PositionAction.HOLD and self._add_on_eligible(position):
            action = PositionAction.ADD_CONDITIONAL
            reason_code = "conditional_add"
            explanation = (
                "未触发退出；持仓盈利且加仓名额未用完，可等待回踩确认后"
                "按加仓规则评估（退出/减仓仍优先）。"
            )
        if action in (PositionAction.EXIT, PositionAction.REDUCE) and pending_shares > 0:
            # The underlying risk action is preserved.  Pending sell orders
            # only cover part of it; the uncovered remainder remains a real
            # recommendation, and partial fills count from the actual filled
            # quantity rather than the original order size.
            covered = min(pending_shares, suggested)
            suggested = max(0, suggested - covered)
            explanation = (
                f"{explanation} 已有 {pending_shares} 股卖出委托待成交，"
                f"覆盖本次建议 {covered} 股，尚未覆盖 {suggested} 股；"
                "请核对是否需要撤改单或补充卖出。"
            )
        target_states = tuple(policy.targets)
        # "Confirmed" means a real sell order/acknowledged action exists, not
        # merely that the same alert was shown before.
        confirmed = pending_shares > 0
        proposed = raw.proposed_stop
        if (
            persist_state
            and session_type == ExitSession.DAILY
            and point is not None
            and point.usable
            and action not in (PositionAction.EXIT, PositionAction.DATA_ANOMALY)
            and proposed is not None
            and (effective_stop is None or proposed > effective_stop + 1e-9)
            and (
                position.pending_stop is None
                or proposed > float(position.pending_stop) + 1e-9
            )
        ):
            position.pending_stop = Decimal(str(proposed))
            position.pending_stop_data_time = point.data_time
            position.pending_stop_effective_at = next_session_open(
                as_of, calendar
            )
        if (
            persist_state
            and session_type == ExitSession.DAILY
            and point is not None
            and point.usable
        ):
            if new_high_today:
                position.highest_price = Decimal(str(tick_high))
                if point.data_time is not None:
                    position.last_new_high_on = point.data_time.date()
            elif position.last_new_high_on is None and position.opened_at is not None:
                position.last_new_high_on = position.opened_at.date()
        valid_until: datetime | None
        if session_type == ExitSession.INTRADAY:
            valid_until = now + timedelta(
                seconds=self.config.intraday.max_quote_age_seconds
            )
        else:
            valid_until = next_session_open(
                as_of, calendar
            ) if point is not None else None
        pending_stop_value = (
            float(position.pending_stop)
            if position.pending_stop is not None
            else None
        )
        pending_stop_effective_at = position.pending_stop_effective_at
        pending_effective_moment = self._aware_shanghai(
            pending_stop_effective_at
        )
        moment_for_pending = self._aware_shanghai(
            point.data_time if point else None
        ) or self._aware_shanghai(now)
        if (
            pending_stop_value is not None
            and pending_effective_moment is not None
            and moment_for_pending is not None
            and moment_for_pending >= pending_effective_moment
        ):
            pending_stop_value = None
            pending_stop_effective_at = None
        return PositionDecision(
            ts_code=position.ts_code,
            sector=position.sector,
            primary_strategy=policy.strategy,
            sub_mode=SubMode(policy.sub_mode) if policy.sub_mode else None,
            rule_version=policy.strategy_version or self.config.version,
            action=action,
            reason_code=reason_code,
            explanation=explanation,
            trigger_price=(
                Decimal(str(raw.trigger_price))
                if raw.trigger_price is not None
                else (
                    Decimal(str(point.price))
                    if point is not None and point.usable
                    else None
                )
            ),
            trigger_time=point.data_time if point else None,
            data_time=point.data_time if point else None,
            valid_until=valid_until,
            decision_source=(
                "daily" if session_type == ExitSession.DAILY else "intraday"
            ),
            market_phase=market_phase(moment, calendar),
            is_current=valuation == ValuationStatus.OK,
            total_shares=position.shares,
            available_shares=position.available_shares,
            suggested_shares=suggested,
            restricted_shares=raw.restricted_shares,
            covered_shares=covered,
            current_stop=Decimal(str(effective_stop))
            if effective_stop is not None and effective_stop > 0
            else None,
            proposed_stop=(
                Decimal(str(raw.proposed_stop))
                if raw.proposed_stop is not None
                else None
            ),
            pending_stop=(
                Decimal(str(pending_stop_value))
                if pending_stop_value is not None
                else None
            ),
            pending_stop_effective_at=pending_stop_effective_at,
            last_new_high_on=last_high_on,
            days_since_new_high=days_since_high,
            targets=list(target_states),
            invalidation=raw.invalidation,
            confirmed=confirmed,
            executed_shares=executed_shares,
            pending_shares=pending_shares,
            pending_sell_orders=pending_orders,
            valuation_status=valuation,
            needs_reconciliation=policy.needs_reconciliation,
            data_issues=list(raw.data_issues),
        )

    def latest(self, ts_code: str) -> PositionDecision | None:
        position = PositionRepository.get(
            self.session, self.account_id, ts_code
        )
        if position is None or not position.last_decision_json:
            return None
        try:
            return PositionDecision.model_validate(position.last_decision_json)
        except Exception:  # noqa: BLE001 - stale/legacy payload
            return None

    # ---- persistence / notifications ---------------------------------
    def _persist(
        self,
        position: PositionRow,
        decision: PositionDecision,
        *,
        now: datetime,
    ) -> None:
        position.last_decision_json = decision.model_dump(mode="json")
        position.last_decision_at = now
        if decision.data_time is not None:
            position.valuation_at = decision.data_time
        position.valuation_status = decision.valuation_status.value
        if (
            decision.valuation_status == ValuationStatus.OK
            and decision.trigger_price is not None
        ):
            position.current_price = decision.trigger_price

    def _notify(
        self,
        position: PositionRow,
        decision: PositionDecision,
        as_of: date,
        *,
        now: datetime,
        session_type: ExitSession,
    ) -> None:
        previous = position.last_alert_status
        action = decision.action
        risk_actions = {
            PositionAction.EXIT,
            PositionAction.REDUCE,
            PositionAction.DATA_ANOMALY,
        }
        level = None
        code = None
        title = None
        if action in risk_actions:
            level = "critical" if action == PositionAction.EXIT else "warning"
            code = {
                "hard_stop": (
                    "exit_stop"
                    if session_type == ExitSession.DAILY
                    else "hard_stop_breached"
                ),
                "target_1": "take_profit_hit",
                "target_2": "take_profit_hit",
                "targets_complete": "take_profit_hit",
                "time_stop": "time_stop",
                "position_incomplete": "position_data_anomaly",
                "data_unavailable": "intraday_data_degraded",
                "stop_missing": "position_data_anomaly",
                "invalid_price": "position_data_anomaly",
            }.get(decision.reason_code, f"position_{action.value}")
            title = {
                PositionAction.EXIT: f"{position.ts_code} 建议清仓",
                PositionAction.REDUCE: f"{position.ts_code} 建议减仓",
                PositionAction.DATA_ANOMALY: f"{position.ts_code} 数据异常",
            }[action]
        elif previous in {item.value for item in risk_actions}:
            level = "info"
            code = "position_recovered"
            title = f"{position.ts_code} 风险状态恢复"
        if level is not None and code is not None and title is not None:
            _, created = NotificationRepository.add(
                self.session,
                level=level,
                code=code,
                title=title,
                message=f"{decision.explanation}（{decision.reason_code}）",
                dedupe_key=(
                    f"position-decision:{as_of.isoformat()}:{position.ts_code}:"
                    f"{decision.reason_code}"
                ),
                trade_date=as_of,
            )
        elif (
            action in (PositionAction.HOLD, PositionAction.ADD_CONDITIONAL)
            and decision.trigger_price is not None
            and position.current_stop > 0
            and float(position.current_stop)
            < float(decision.trigger_price)
            <= float(position.current_stop) * 1.02
        ):
            # Keep the old "close to protection" awareness without changing
            # the structured action; one notification per symbol per day.
            NotificationRepository.add(
                self.session,
                level="warning",
                code="near_stop",
                title=f"{position.ts_code} 接近保护线",
                message=(
                    f"现价 {decision.trigger_price} 距保护线 "
                    f"{position.current_stop} 不足 2%。"
                ),
                dedupe_key=f"near-stop:{as_of.isoformat()}:{position.ts_code}",
                trade_date=as_of,
            )
        position.last_alert_status = action.value
        position.last_alert_at = now

    # ---- policy construction ------------------------------------------
    def _policy(
        self,
        position: PositionRow,
        *,
        ma5: float | None,
        as_of: date,
        calendar: TradingCalendar | None,
        current_stop: float | None,
        highest_price: float,
    ) -> ExitPolicy:
        plan = (
            PlanRepository.get(self.session, position.plan_id)
            if position.plan_id
            else None
        )
        payload = (plan.plan_json or {}) if plan is not None else {}
        strategy_value = position.primary_strategy or (
            plan.strategy if plan is not None else None
        )
        try:
            strategy = Strategy(strategy_value) if strategy_value else None
        except ValueError:
            strategy = None
        sub_mode_value = position.sub_mode or (
            plan.sub_mode if plan is not None else None
        )
        stop_payload = payload.get("stop_plan") or {}
        entry_payload = payload.get("entry_plan") or {}
        entry_price = float(
            position.cost_price
            or entry_payload.get("trigger_price")
            or stop_payload.get("initial_stop")
            or 0
        )
        initial_stop_value = stop_payload.get("initial_stop")
        if initial_stop_value is not None:
            initial_stop = float(initial_stop_value)
        elif position.current_stop > 0:
            initial_stop = float(position.current_stop)
        else:
            initial_stop = None
        initial_r_value = position.initial_r
        derived_initial_r = None
        if initial_stop is not None and entry_price > initial_stop:
            derived_initial_r = entry_price - initial_stop
        if initial_r_value is None and derived_initial_r is not None:
            # The persisted batch initial R is authoritative.  For legacy rows
            # without it, average cost minus the recorded structural stop is a
            # deterministic reconstruction; migration still marks old rows for
            # human review.
            initial_r_value = Decimal(str(derived_initial_r))
        elif initial_r_value is None and stop_payload.get("r_value") is not None:
            initial_r_value = Decimal(str(stop_payload["r_value"]))
        exit_rules = self._exit_rules(strategy, payload, sub_mode_value)
        targets = self._target_states(position, payload)
        range_midpoint = self._range_midpoint(payload)
        event_expiry = None
        event_invalid = False
        evidence = payload.get("candidate", {}).get("evidence") or {}
        if strategy == Strategy.F:
            days = (
                exit_rules.event_expiry_days
                or self.config.strategies.f.event_window_days
            )
            base = None
            signal_time = payload.get("signal_time")
            if signal_time:
                try:
                    base = datetime.fromisoformat(str(signal_time)).date()
                except ValueError:
                    base = None
            if base is None and position.opened_at is not None:
                base = position.opened_at.date()
            if base is not None:
                if calendar is not None and calendar.is_trading_day(base):
                    try:
                        event_expiry = calendar.add_trading_days(base, days)
                    except ValueError:
                        event_expiry = base + timedelta(days=days)
                else:
                    event_expiry = base + timedelta(days=days)
            negative_events = list(
                self.session.scalars(
                    select(EventRow)
                    .join(EventLinkRow)
                    .where(
                        EventLinkRow.ts_code == position.ts_code,
                        EventRow.verification_status == "verified",
                        EventRow.ann_date.is_not(None),
                        EventRow.ann_date >= as_of
                        - timedelta(
                            days=self.config.strategies.d.event_risk_days
                        ),
                        EventRow.ann_date <= as_of,
                        or_(
                            EventRow.retracted.is_(True),
                            EventRow.direction == "negative",
                        ),
                    )
                )
            )
            event_invalid = bool(
                negative_events
                or evidence.get("event_invalid")
                or evidence.get("event_retracted")
                or evidence.get("direction") == "negative"
            )
        highest = highest_price
        return ExitPolicy(
            strategy=strategy,
            sub_mode=str(sub_mode_value) if sub_mode_value else None,
            strategy_version=(
                position.strategy_version
                or (plan.strategy_version if plan is not None else "")
            ),
            entry_price=entry_price,
            initial_stop=initial_stop,
            initial_r=(
                float(initial_r_value) if initial_r_value is not None else None
            ),
            current_stop=current_stop,
            highest_price=highest,
            atr14=float(stop_payload.get("atr14") or 0) or None,
            ma5=ma5,
            range_midpoint=range_midpoint,
            targets=tuple(targets),
            exit_rules=exit_rules,
            event_expiry=event_expiry,
            event_invalid=event_invalid,
            needs_reconciliation=bool(
                position.needs_reconciliation
                or strategy is None
                or initial_stop is None
                or initial_stop <= 0
                or initial_r_value is None
            ),
        )

    def _exit_rules(
        self, strategy: Strategy | None, payload: dict[str, Any], sub_mode: str | None
    ) -> ExitRules:
        raw = payload.get("exit_rules")
        if isinstance(raw, dict):
            try:
                return ExitRules.model_validate(raw)
            except Exception:  # noqa: BLE001 - legacy plan
                pass
        if strategy == Strategy.B:
            return ExitRules(
                trailing="breakeven_1r_2r_ma5_atr",
                time_exit="no_new_high",
                time_stop_days=self.config.strategies.b.time_stop_days,
            )
        if strategy == Strategy.D:
            return ExitRules(
                trailing="breakeven_1r_2r_ma5_atr",
                time_exit="no_new_high",
                time_stop_days=self.config.strategies.d.time_stop_days,
            )
        if strategy == Strategy.E:
            return ExitRules(
                trailing="breakeven_1r_2r_range_atr",
                time_exit="below_entry",
                time_stop_days=self.config.strategies.e.time_stop_days,
            )
        if strategy == Strategy.F:
            return ExitRules(
                trailing="breakeven_1r_2r_atr",
                time_exit="event_expiry",
                time_stop_days=self.config.strategies.f.event_window_days,
                event_expiry_days=self.config.strategies.f.event_window_days,
            )
        days = (
            self.config.strategies.a.time_stop_days
            if strategy == Strategy.A
            else self.config.strategies.c.time_stop_days
            if strategy == Strategy.C
            else self.config.strategies.a.time_stop_days
        )
        return ExitRules(
            trailing="breakeven_1r_2r_atr",
            time_exit="below_entry",
            time_stop_days=days,
        )

    def _target_states(
        self, position: PositionRow, payload: dict[str, Any]
    ) -> list[TargetState]:
        stored = [
            TargetState.model_validate(item)
            for item in (position.target_state_json or [])
            if isinstance(item, dict)
        ]
        if stored:
            return stored
        targets = (payload.get("take_profit_plan") or {}).get("targets") or []
        return target_states_from_plan(
            targets,
            total_shares=position.shares,
            initial_shares=position.initial_shares or position.shares,
        )

    def _range_midpoint(self, payload: dict[str, Any]) -> float | None:
        evidence = payload.get("candidate", {}).get("evidence") or {}
        upper = evidence.get("range_upper")
        lower = evidence.get("range_lower")
        if upper is None or lower is None:
            return None
        try:
            return (float(upper) + float(lower)) / 2
        except (TypeError, ValueError):
            return None

    # ---- market data ---------------------------------------------------
    def _daily_points(self, as_of: date) -> dict[str, PricePoint]:
        if self.provider is None:
            return {}
        try:
            bars = self.provider.load_bars(as_of, as_of)
        except Exception:  # noqa: BLE001 - caller receives data anomalies
            return {}
        return points_from_bars(bars, as_of)

    def _ma5_map(self, as_of: date) -> dict[str, float]:
        if self.provider is None:
            return {}
        try:
            bars = self.provider.load_bars(as_of - timedelta(days=14), as_of)
        except Exception:  # noqa: BLE001
            return {}
        if bars.is_empty() or "ts_code" not in bars.columns:
            return {}
        result: dict[str, float] = {}
        for code, frame in bars.sort("trade_date").group_by("ts_code"):
            closes = [
                float(value)
                for value in frame["close"].to_list()
                if value is not None
            ]
            if closes:
                window = closes[-5:]
                result[str(code[0] if isinstance(code, tuple) else code)] = sum(
                    window
                ) / len(window)
        return result

    def _calendar(self, as_of: date) -> TradingCalendar | None:
        return self._calendar_for(as_of)

    def _calendar_for(self, day: date) -> TradingCalendar | None:
        if self.provider is None:
            return None
        try:
            frame = self.provider.load_trade_cal(
                day - timedelta(days=400), day + timedelta(days=15)
            )
        except Exception:  # noqa: BLE001
            return None
        calendar = TradingCalendar.from_trade_cal(frame)
        return calendar if calendar.dates or calendar.covered_start else None

    @staticmethod
    def _entry_price(position: PositionRow) -> float:
        return float(position.cost_price or 0)

    @staticmethod
    def _aware_shanghai(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=SHANGHAI)
        return value.astimezone(SHANGHAI)

    def _effective_stop(
        self,
        position: PositionRow,
        point: PricePoint | None,
        now: datetime,
        *,
        persist_state: bool,
    ) -> float | None:
        stop = (
            float(position.current_stop)
            if position.current_stop is not None and position.current_stop > 0
            else None
        )
        pending = position.pending_stop
        effective_at = self._aware_shanghai(
            position.pending_stop_effective_at
        )
        if pending is None or pending <= 0 or effective_at is None:
            return stop
        moment = self._aware_shanghai(point.data_time if point else None) or (
            self._aware_shanghai(now)
        )
        if moment is None or moment < effective_at:
            return stop
        promoted = float(pending)
        if stop is not None and promoted <= stop:
            if persist_state:
                position.pending_stop = None
                position.pending_stop_data_time = None
                position.pending_stop_effective_at = None
            return stop
        if persist_state:
            position.current_stop = Decimal(str(promoted))
            position.pending_stop = None
            position.pending_stop_data_time = None
            position.pending_stop_effective_at = None
        return promoted

    @staticmethod
    def _days_since_high(
        last_high_on: date | None,
        as_of: date,
        calendar: TradingCalendar | None,
    ) -> int | None:
        if last_high_on is None:
            return None
        if as_of <= last_high_on:
            return 0
        if calendar is not None:
            return calendar.count_between(
                last_high_on + timedelta(days=1), as_of
            )
        return max(0, (as_of - last_high_on).days)

    def _days_held(
        self,
        position: PositionRow,
        as_of: date,
        calendar: TradingCalendar | None,
    ) -> int:
        if position.opened_at is None:
            return 0
        opened = position.opened_at.date()
        if calendar is not None:
            return calendar.count_between(opened, as_of)
        return max(0, sum(1 for _ in range((as_of - opened).days + 1)))

    def _execution_totals(
        self, position: PositionRow
    ) -> tuple[int, int, list[dict]]:
        executions = list(
            self.session.scalars(
                select(ExecutionRow).where(
                    ExecutionRow.account_id == self.account_id,
                    ExecutionRow.ts_code == position.ts_code,
                )
            )
        )
        pending = list(
            self.session.scalars(
                select(OrderRow).where(
                    OrderRow.account_id == self.account_id,
                    OrderRow.ts_code == position.ts_code,
                    OrderRow.side == ExecutionSide.SELL.value,
                    OrderRow.status.in_(
                        [
                            OrderStatus.PENDING.value,
                            OrderStatus.PARTIAL.value,
                            OrderStatus.CANCEL_REQUESTED.value,
                        ]
                    ),
                )
            )
        )
        # The original order size is not the remaining exposure.  A partial
        # fill reduces the quantity that is still working at the broker.
        details: list[dict[str, Any]] = [
            {
                "order_id": order.id,
                "ts_code": order.ts_code,
                "price": str(order.price),
                "shares": int(order.shares),
                "filled_shares": int(order.filled_shares or 0),
                "remaining_shares": max(
                    0, int(order.shares) - int(order.filled_shares or 0)
                ),
                "status": order.status,
            }
            for order in pending
        ]
        pending_shares = sum(
            item["remaining_shares"] for item in details
        )
        opened_at = position.opened_at
        executed_shares = sum(
            execution.shares
            for execution in executions
            if execution.side == ExecutionSide.SELL.value
            and (
                opened_at is None
                or execution.executed_at >= opened_at
            )
        )
        return pending_shares, executed_shares, details

    def _add_on_eligible(self, position: PositionRow) -> bool:
        strategy_value = position.primary_strategy
        if not strategy_value:
            return False
        try:
            strategy = Strategy(strategy_value)
        except ValueError:
            return False
        if strategy not in self.config.risk.add_on_strategies:
            return False
        if position.add_on_count >= self.config.risk.add_on_max:
            return False
        if position.current_price <= position.cost_price:
            return False
        if position.current_stop <= 0:
            return False
        return (
            self.session.scalar(
                select(PlanRow).where(
                    PlanRow.account_id == self.account_id,
                    PlanRow.ts_code == position.ts_code,
                    PlanRow.status.in_(
                        (
                            PlanStatus.PENDING_EXECUTION.value,
                            PlanStatus.PENDING_ENTRY.value,
                            PlanStatus.PARTIAL_FILL.value,
                        )
                    ),
                    PlanRow.is_add_on.is_(True),
                )
            )
            is None
        )
