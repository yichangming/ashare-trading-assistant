"""B2 intraday confirmation: watchlist -> 5-minute signal -> plan.

Daily B1/B2 candidates are a watchlist only.  This monitor consumes completed
5-minute bars, runs the B2 state machine, and (when a signal is actionable)
builds a plan through the same planner/sizer/governor used by the daily
pipeline.  Missing minute data is reported, never replaced with daily bars.
"""

from __future__ import annotations

import os
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
from app.domain.models import Candidate, MarketRegime, TradePlan
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
    """``stk_mins`` adapter. The endpoint is rate limited to ~1/min."""

    token: str | None = None
    _cache: dict[tuple[str, date], tuple[datetime, pl.DataFrame]] = field(
        default_factory=dict, repr=False
    )

    def load(self, ts_code: str, day: date) -> pl.DataFrame:
        key = (ts_code, day)
        now = datetime.now()
        cached = self._cache.get(key)
        if cached is not None and (now - cached[0]).total_seconds() < 300:
            return cached[1]
        token = self.token or os.environ.get("TUSHARE_TOKEN")
        if not token:
            return pl.DataFrame()
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
        result = (
            pl.from_pandas(frame)
            if frame is not None and not getattr(frame, "empty", True)
            else pl.DataFrame()
        )
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


@dataclass
class B2Monitor:
    provider: MarketDataProvider
    config: FullSystemConfig
    minute_source: MinuteBarSource | None = None
    planner: StrategyPlanner = field(default_factory=StrategyPlanner)
    sizer: StrategyPositionSizer = field(default_factory=StrategyPositionSizer)
    governor: StrategyRiskGovernor = field(default_factory=StrategyRiskGovernor)

    def _watchlist(self, session: Session, day: date) -> list[dict]:
        snapshots = SnapshotRepository.list_recent(session, "default", limit=1)
        if not snapshots or snapshots[-1].trade_date != day:
            return []
        detail = snapshots[-1].detail_json or {}
        result = []
        for item in detail.get("candidates", []):
            evidence = item.get("evidence") or {}
            if item.get("sub_mode") == SubMode.B2.value and evidence.get(
                "stage"
            ) in ("watchlist", "observing"):
                result.append(item)
        return result

    def check(
        self,
        day: date,
        *,
        now: datetime,
        session: Session,
    ) -> B2MonitorResult:
        result = B2MonitorResult()
        if self.minute_source is None:
            result.degraded = True
            self._notify_degraded(session, day, "分钟数据源未配置")
            return result
        watchlist = self._watchlist(session, day)
        if not watchlist:
            return result

        snapshots = SnapshotRepository.list_recent(session, "default", limit=1)
        detail = (snapshots[-1].detail_json or {}) if snapshots else {}
        regime_data = detail.get("regime")
        if not regime_data:
            result.degraded = True
            self._notify_degraded(session, day, "缺少行情判断快照")
            return result
        regime = MarketRegime.model_validate(regime_data)
        permission = regime.strategy_permissions.get(Strategy.B)
        if permission is not None and permission.mode == PermissionMode.PAUSED:
            return result
        account = AccountRepository.to_state(session)
        try:
            calendar = TradingCalendar.from_trade_cal(
                self.provider.load_trade_cal(day, day + timedelta(days=15))
            )
        except Exception:  # noqa: BLE001
            calendar = TradingCalendar([])

        for item in watchlist:
            code = str(item.get("ts_code") or "")
            if not code:
                continue
            bars = self.minute_source.load(code, day)
            if bars.is_empty() or not set(bars.columns) >= SHANGHAI_MINUTE_COLUMNS:
                result.degraded = True
                self._notify_degraded(
                    session, day, f"{code} 未取得已完成的 5 分钟数据"
                )
                continue
            result.checked += 1
            completed_until = now - timedelta(
                minutes=self.config.strategies.b.b2_bar_minutes
            )
            signal = detect_b2(
                code,
                bars,
                self.config.strategies.b,
                completed_until=completed_until,
            )
            if signal is None or not signal.actionable or now > signal.valid_until:
                continue
            result.signals.append(code)
            self._build_plan(
                session, day, item, signal, regime, account, calendar, result
            )
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
        calendar: TradingCalendar,
        result: B2MonitorResult,
    ) -> None:
        dedupe_key = (
            f"b2-signal:{day.isoformat()}:{signal.ts_code}:"
            f"{signal.signal_time.isoformat()}"
        )
        if NotificationRepository.get_by_key(session, dedupe_key) is not None:
            return
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
            parts = self.planner.plan_parts(candidate, calendar, self.config)
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
            return
        plan = TradePlan(
            candidate=candidate,
            entry_plan=parts.entry,
            stop_plan=parts.stop,
            position_plan=position,
            take_profit_plan=parts.take_profit,
            expires_on=parts.entry.expires_on,
            strategy_version=self.config.version,
            data_version=regime.data_version,
            code_version="intraday",
            reasons=candidate.reasons,
            strategy=candidate.strategy,
            sub_mode=SubMode.B2,
            signal_time=signal.signal_time,
            data_time=signal.signal_time,
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
                level="critical" if item_plan.status == PlanStatus.PENDING_EXECUTION else "info",
                code="b2_signal_ready",
                title=f"{signal.ts_code} B2 盘中确认",
                message=(
                    f"信号 {signal.signal_time:%H:%M} 价格 {signal.entry_price:.2f} "
                    f"止损 {signal.stop_price:.2f} 状态 {item_plan.status.value}"
                ),
                dedupe_key=dedupe_key,
                trade_date=day,
            )
            if item_plan.status == PlanStatus.PENDING_EXECUTION:
                result.plans.append(item_plan.id)
            else:
                result.rejected.append(
                    f"{signal.ts_code}: {item_plan.status.value}"
                )
            break
