"""B2 intraday confirmation: watchlist -> 5-minute signal -> plan.

The watchlist comes from the most recent *completed* post-close snapshot on
the previous trading day, not necessarily yesterday's calendar date.  The
monitor consumes completed 5-minute bars, uses trading-minute validity across
the lunch break, and never falls back to unfinished daily bars.
"""

from __future__ import annotations

import os
import threading
import time as clock
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Protocol

import polars as pl
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.calendar import TradingCalendar
from app.core.full_config import FullSystemConfig
from app.data.interfaces import MarketDataProvider
from app.db.repositories import (
    AccountRepository,
    NotificationRepository,
    PlanRepository,
    SnapshotRepository,
)
from app.domain.enums import PermissionMode, PlanStatus, Strategy, SubMode
from app.domain.models import (
    Candidate,
    MarketRegime,
    PendingExposure,
    TradePlan,
)
from app.planning.errors import PlanningError
from app.planning.strategy_planner import StrategyPlanner
from app.planning.strategy_sizing import StrategyPositionSizer
from app.risk.strategy_governor import StrategyRiskGovernor
from app.strategies.b2_intraday import B2Signal, detect_b2

SHANGHAI_MINUTE_COLUMNS = {"trade_time", "open", "high", "low", "close", "vol", "amount"}


class MinuteBarSource(Protocol):
    def load(self, ts_code: str, day: date) -> pl.DataFrame: ...


@dataclass
class TushareMinuteSource:
    """``stk_mins`` adapter with a process-wide request gate.

    The endpoint is limited to roughly one request per minute.  The lock and
    timestamp live on the source instance, so different symbols cannot each
    run their own cache timer and jointly exceed the endpoint limit.
    """

    token: str | None = None
    min_interval_seconds: float = 60.0
    _cache: dict[tuple[str, date], tuple[datetime, pl.DataFrame]] = field(
        default_factory=dict, repr=False
    )
    _last_request: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    requests_made: int = field(default=0, repr=False)

    def _fetch(self, ts_code: str, day: date, token: str) -> pl.DataFrame:
        try:
            import tushare as ts

            frame = ts.pro_api(token).stk_mins(
                ts_code=ts_code,
                freq="5min",
                start_date=f"{day.isoformat()} 09:00:00",
                end_date=f"{day.isoformat()} 15:00:00",
            )
        except Exception:  # noqa: BLE001 - optional/limited data source
            return pl.DataFrame()
        return (
            pl.from_pandas(frame)
            if frame is not None and not getattr(frame, "empty", True)
            else pl.DataFrame()
        )

    def load(self, ts_code: str, day: date) -> pl.DataFrame:
        key = (ts_code, day)
        with self._lock:
            now = datetime.now()
            cached = self._cache.get(key)
            if cached is not None and (now - cached[0]).total_seconds() < 300:
                return cached[1]
            wait = self.min_interval_seconds - (
                clock.monotonic() - self._last_request
            )
            if wait > 0:
                clock.sleep(wait)
            self._last_request = clock.monotonic()
            token = self.token or os.environ.get("TUSHARE_TOKEN")
            if not token:
                return pl.DataFrame()
            self.requests_made += 1
            result = self._fetch(ts_code, day, token)
            if not result.is_empty():
                for column in ("open", "high", "low", "close", "vol", "amount"):
                    if column in result.columns:
                        result = result.with_columns(
                            pl.col(column).cast(pl.Float64, strict=False)
                        )
                self._cache[key] = (now, result)
            return result


class B2MonitorResult(BaseModel):
    checked: int = 0
    signals: list[str] = []
    plans: list[str] = []
    rejected: list[str] = []
    degraded: bool = False
    watchlist_date: date | None = None
    monitor_date: date | None = None
    data_issues: list[str] = []
    permission: str = "normal"


@dataclass
class B2Monitor:
    provider: MarketDataProvider
    config: FullSystemConfig
    minute_source: MinuteBarSource | None = None
    planner: StrategyPlanner = field(default_factory=StrategyPlanner)
    sizer: StrategyPositionSizer = field(default_factory=StrategyPositionSizer)
    governor: StrategyRiskGovernor = field(default_factory=StrategyRiskGovernor)

    def _calendar(self, day: date) -> TradingCalendar | None:
        try:
            frame = self.provider.load_trade_cal(
                day - timedelta(days=60), day + timedelta(days=15)
            )
        except Exception:  # noqa: BLE001
            return None
        if frame.is_empty():
            return None
        return TradingCalendar.from_trade_cal(frame)

    def _previous_trading_day(
        self, day: date, calendar: TradingCalendar | None
    ) -> date | None:
        if calendar is not None:
            earlier = [item for item in calendar.dates if item < day]
            return earlier[-1] if earlier else None
        candidate = day - timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
        return candidate

    def _watchlist(
        self,
        session: Session,
        day: date,
        calendar: TradingCalendar | None,
    ) -> tuple[list[dict], date | None, list[str], dict | None]:
        previous = self._previous_trading_day(day, calendar)
        if previous is None:
            return [], None, ["无法确定上一交易日"], None
        snapshots = SnapshotRepository.list_recent(session, "default", limit=20)
        snapshot = next(
            (
                item
                for item in reversed(snapshots)
                if item.trade_date == previous
            ),
            None,
        )
        if snapshot is None:
            return (
                [],
                None,
                [f"缺少上一交易日 {previous.isoformat()} 的盘后观察池"],
                None,
            )
        detail = snapshot.detail_json or {}
        result: list[dict] = []
        for item in detail.get("candidates", []):
            evidence = item.get("evidence") or {}
            if item.get("sub_mode") == SubMode.B2.value and evidence.get(
                "stage"
            ) in ("watchlist", "observing"):
                result.append(item)
        return result, previous, [], detail

    def check(
        self,
        day: date,
        *,
        now: datetime,
        session: Session,
    ) -> B2MonitorResult:
        result = B2MonitorResult(monitor_date=day)
        if self.minute_source is None:
            result.degraded = True
            result.data_issues.append("分钟数据源未配置")
            self._notify_degraded(session, day, "分钟数据源未配置")
            return result
        calendar = self._calendar(day)
        if calendar is not None and not calendar.is_trading_day(day):
            result.data_issues.append("非交易日")
            return result
        PlanRepository.expire_intraday(session, now)
        watchlist, watchlist_date, issues, detail = self._watchlist(
            session, day, calendar
        )
        result.watchlist_date = watchlist_date
        result.data_issues.extend(issues)
        if issues:
            result.degraded = True
            self._notify_degraded(session, day, "；".join(issues))
        if not watchlist:
            return result

        regime_data = (detail or {}).get("regime")
        if not regime_data:
            result.degraded = True
            result.data_issues.append("缺少行情判断快照")
            self._notify_degraded(session, day, "缺少行情判断快照")
            return result
        regime = MarketRegime.model_validate(regime_data)
        permission = regime.strategy_permissions.get(Strategy.B)
        if permission is not None:
            result.permission = permission.mode.value
            if permission.mode == PermissionMode.PAUSED:
                return result
        account = AccountRepository.to_state(session)

        for item in watchlist:
            code = str(item.get("ts_code") or "")
            if not code:
                continue
            bars = self.minute_source.load(code, day)
            if bars.is_empty() or not set(bars.columns) >= SHANGHAI_MINUTE_COLUMNS:
                result.degraded = True
                issue = f"{code} 未取得已完成的 5 分钟数据"
                result.data_issues.append(issue)
                self._notify_degraded(session, day, issue)
                continue
            result.checked += 1
            signal = detect_b2(
                code,
                bars,
                self.config.strategies.b,
                completed_until=now,
            )
            if signal is None or not signal.actionable or now > signal.valid_until:
                continue
            result.signals.append(code)
            account, plan_id = self._build_plan(
                session, day, item, signal, regime, account, result
            )
            if plan_id:
                result.plans.append(plan_id)
        PlanRepository.expire_intraday(session, now)
        session.flush()
        return result

    def _notify_degraded(self, session: Session, day: date, message: str) -> None:
        NotificationRepository.add(
            session,
            level="warning",
            code="b2_data_degraded",
            title="B2 盘中数据不可用",
            message=message,
            dedupe_key=f"b2-degraded:{day.isoformat()}",
            trade_date=day,
        )

    def _build_plan(
        self,
        session: Session,
        day: date,
        item: dict,
        signal: B2Signal,
        regime: MarketRegime,
        account,
        result: B2MonitorResult,
    ):
        dedupe_key = (
            f"b2-signal:{day.isoformat()}:{signal.ts_code}:"
            f"{signal.signal_time.isoformat()}"
        )
        if NotificationRepository.get_by_key(session, dedupe_key) is not None:
            return account, None
        # Restart-safe: the notification may have been purged, so also check
        # the persisted plan signal time.
        for row in PlanRepository.list_for_date(session, day):
            payload = row.plan_json or {}
            if (
                row.ts_code == signal.ts_code
                and payload.get("signal_time")
                == signal.signal_time.isoformat()
            ):
                return account, None
        features = {
            str(key): float(value)
            for key, value in (item.get("features") or {}).items()
            if isinstance(value, (int, float))
        }
        features.update({"close": signal.entry_price, "atr14_raw": signal.atr5})
        evidence = {
            **(item.get("evidence") or {}),
            "stage": "confirmed",
            "frozen_low": signal.frozen_low,
            "frozen_high": signal.frozen_high,
            "vwap": signal.vwap,
            "atr5": signal.atr5,
            "signal_time": signal.signal_time.isoformat(),
            "bar_completed_at": signal.bar_completed_at.isoformat(),
            "vwap_approximate": signal.vwap_approximate,
            "confirmed": True,
        }
        candidate = Candidate.model_validate(
            {
                **item,
                "trade_date": day.isoformat(),
                "strategy": "B",
                "sub_mode": "b2",
                "features": features,
                "evidence": evidence,
            }
        )
        try:
            parts = self.planner.plan_parts(
                candidate,
                calendar_or_raise(self.provider, day),
                self.config,
            )
            position = self.sizer.size(
                candidate,
                parts.entry,
                parts.stop,
                account,
                regime,
                self.config,
            )
        except PlanningError as exc:
            result.rejected.append(f"{signal.ts_code}: {exc}")
            return account, None
        plan = TradePlan(
            candidate=candidate,
            entry_plan=parts.entry,
            stop_plan=parts.stop,
            position_plan=position,
            take_profit_plan=parts.take_profit,
            expires_on=parts.entry.expires_on,
            expires_at=signal.valid_until,
            exit_rules=parts.exit_rules,
            strategy_version=self.config.version,
            data_version=regime.data_version,
            code_version="intraday",
            reasons=candidate.reasons,
            strategy=candidate.strategy,
            sub_mode=SubMode.B2,
            signal_time=signal.signal_time,
            data_time=signal.bar_completed_at,
            confirmation_price=parts.confirmation_price,
            max_acceptable_price=parts.max_acceptable_price,
            execution_low=parts.execution_low,
            execution_high=parts.execution_high,
        )
        approved = self.governor.apply([plan], account, regime, self.config)
        for item_plan in approved:
            PlanRepository.save(session, item_plan, trade_date=day)
            NotificationRepository.add(
                session,
                level="critical"
                if item_plan.status == PlanStatus.PENDING_EXECUTION
                else "info",
                code="b2_signal_ready",
                title=f"{signal.ts_code} B2 盘中确认",
                message=(
                    f"信号 {signal.signal_time:%H:%M} 价格 "
                    f"{signal.entry_price:.2f} 止损 "
                    f"{signal.stop_price:.2f} 状态 {item_plan.status.value}；"
                    f"有效至 {signal.valid_until:%H:%M}"
                ),
                dedupe_key=dedupe_key,
                trade_date=day,
            )
            if item_plan.status == PlanStatus.PENDING_EXECUTION:
                updated = account.with_pending(
                    PendingExposure(
                        ts_code=item_plan.candidate.ts_code,
                        sector=item_plan.candidate.sector,
                        entry_price=item_plan.entry_plan.entry_price,
                        stop_price=item_plan.stop_plan.initial_stop,
                        shares=item_plan.position_plan.shares,
                        template=item_plan.candidate.template,
                        expires_on=item_plan.expires_on,
                        strategy=item_plan.strategy,
                        sub_mode=item_plan.sub_mode,
                    )
                )
                return updated, item_plan.id
            result.rejected.append(
                f"{signal.ts_code}: {item_plan.status.value}"
            )
            return account, None
        return account, None


def calendar_or_raise(
    provider: MarketDataProvider, day: date
) -> TradingCalendar:
    """Calendar used by the intraday planner.

    A 15-day forward window is sufficient for B2's same-day/next-day expiry;
    the daily pipeline performs the stricter long-window check.
    """
    frame = provider.load_trade_cal(day - timedelta(days=60), day + timedelta(days=15))
    return TradingCalendar.from_trade_cal(frame)
