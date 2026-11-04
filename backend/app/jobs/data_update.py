"""Automatic trading-day data update before the daily decision job.

Stages are explicit and idempotent: ready local stages are reused, failed or
missing external stages are reported as ``unavailable``/``failed`` instead of
being presented as a successful sync.  Selection stops when critical local
data is not ready; existing-position protection is still run by the daily job.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.calendar import TradingCalendar
from app.core.config import load_app_config
from app.core.full_config import FullSystemConfig
from app.data.assemble import DataAssembler
from app.data.provider import ParquetMarketDataProvider
from app.data.quality import DataQualityGate, resolve_quality_universe
from app.data.store import ParquetStore
from app.data.sync import SyncError, SyncService
from app.data.tushare_client import TushareClient
from app.db.repositories import SettingRepository
from app.events.ingest import EventIngestor, TushareEventSync


class StageResult(BaseModel):
    name: str
    status: str  # succeeded / reused / skipped / unavailable / failed
    detail: str = ""
    data: dict = {}


class DataUpdateResult(BaseModel):
    as_of: date
    status: str
    data_ready: bool
    stages: list[StageResult] = []
    issues: list[str] = []
    event_status: str = "unavailable"
    events_available: bool = False


@dataclass
class DataUpdateService:
    store: ParquetStore
    config: FullSystemConfig
    token: str | None = None
    sync_factory: Callable[[], SyncService] | None = None
    assembler_factory: Callable[[ParquetStore], DataAssembler] = DataAssembler
    event_client_factory: Callable[[str], TushareClient] | None = None
    event_ingestor_factory: Callable[[Session], EventIngestor] = EventIngestor

    def _token(self) -> str | None:
        return self.token or os.environ.get("TUSHARE_TOKEN")

    def _sync(self) -> SyncService:
        if self.sync_factory is not None:
            return self.sync_factory()
        token = self._token()
        if not token:
            raise SyncError("TUSHARE_TOKEN is not configured")
        app_config = load_app_config()
        return SyncService(
            TushareClient(
                token,
                rate_limit_per_min=app_config.data.rate_limit_per_min,
            ),
            self.store,
        )

    def _calendar(self) -> TradingCalendar:
        return TradingCalendar.from_trade_cal(
            self.store.read_reference("trade_cal")
        )

    def _processed_ready(self, as_of: date) -> bool:
        root = self.store.processed_dir / "bars" / f"trade_date={as_of.isoformat()}"
        return (root / "data.parquet").exists()

    def _reference_ready(self, as_of: date) -> bool:
        calendar = self._calendar()
        if not calendar.dates or calendar.covered_end is None:
            return False
        required_end = as_of + timedelta(days=self._forward_days())
        return (
            calendar.covers(as_of)
            and calendar.is_trading_day(as_of)
            and calendar.covered_end >= required_end
            and self.store.reference_path("stock_basic").exists()
        )

    def _quality(self, as_of: date) -> tuple[bool, list[str]]:
        provider = ParquetMarketDataProvider(self.store)
        bars = provider.load_bars(as_of, as_of)
        stocks = provider.load_stock_basic(as_of)
        suspend = provider.load_suspend(as_of, as_of)
        if bars.is_empty() or stocks.is_empty():
            return False, ["处理后日线或股票基础信息为空"]
        expected_codes, allowed_codes, expected_count, warnings = (
            resolve_quality_universe(stocks, suspend)
        )
        if allowed_codes is not None:
            allowed_codes = allowed_codes | set(
                bars["ts_code"].drop_nulls().to_list()
            )
        report = DataQualityGate().check(
            as_of,
            bars,
            expected_count=expected_count,
            expected_codes=expected_codes,
            allowed_codes=allowed_codes,
        )
        return report.passed, [*warnings, *report.failures]

    def _event_sync(
        self, session: Session, as_of: date
    ) -> tuple[str, dict, bool]:
        payload: dict[str, object]
        token = self._token()
        if not token and self.event_client_factory is None:
            payload = {
                "status": "unavailable",
                "as_of": as_of.isoformat(),
                "reason": "TUSHARE_TOKEN is not configured",
                "checked_at": datetime.now().isoformat(),
            }
            SettingRepository.set(session, "event_sync", payload)
            return "unavailable", payload, False
        try:
            client = (
                self.event_client_factory(token or "")
                if self.event_client_factory is not None
                else TushareClient(token)
            )
            ingestor = self.event_ingestor_factory(session)
            sync = TushareEventSync(client, ingestor)
            counts = sync.sync_buybacks(as_of - timedelta(days=7), as_of)
            # Earnings disclosures are refreshed on Mondays to avoid a large
            # full-period pull every day.  A zero row count is a confirmed
            # "no new events" only when the endpoint call itself succeeded.
            if as_of.weekday() == 0:
                quarter_end = _latest_quarter_end(as_of)
                counts.update(sync.sync_earnings(quarter_end))
            status = "ok" if any(counts.values()) else "no_new_events"
            payload = {
                "status": status,
                "as_of": as_of.isoformat(),
                "counts": counts,
                "checked_at": datetime.now().isoformat(),
                "coverage": "structured_endpoints_only",
            }
            SettingRepository.set(session, "event_sync", payload)
            return status, counts, True
        except Exception as exc:  # noqa: BLE001 - status is the deliverable
            payload = {
                "status": "failed",
                "as_of": as_of.isoformat(),
                "error": str(exc),
                "checked_at": datetime.now().isoformat(),
            }
            SettingRepository.set(session, "event_sync", payload)
            return "failed", {"error": str(exc)}, False

    def _forward_days(self) -> int:
        """Calendar days needed for the longest plan validity window.

        The calendar must cover future expiry dates used by plans created on
        ``as_of``; a fixed 45-day window is only a floor.
        """
        strategies = self.config.strategies
        validity = max(
            strategies.a.valid_days,
            strategies.c.c1_valid_days,
            strategies.c.c2_valid_days,
            strategies.e.valid_days,
            strategies.f.event_window_days,
        )
        return max(45, validity * 3 + 20)

    def run(self, as_of: date, session: Session) -> DataUpdateResult:
        stages: list[StageResult] = []
        issues: list[str] = []
        calendar = self._calendar()
        coverage_known = calendar.covers(as_of)
        if coverage_known and not calendar.is_trading_day(as_of):
            stages.append(
                StageResult(
                    name="trading_day",
                    status="skipped",
                    detail=f"{as_of} 不在交易日历开市日，确认休市",
                )
            )
            return DataUpdateResult(
                as_of=as_of,
                status="skipped",
                data_ready=False,
                stages=stages,
                issues=[f"{as_of} 是已确认的休市日"],
            )
        if coverage_known:
            stages.append(StageResult(name="trading_day", status="succeeded"))
        else:
            # A date outside the downloaded calendar range is not evidence of
            # a holiday.  Try to refresh coverage; if that fails, continue as
            # unknown rather than permanently skipping the day.
            stages.append(
                StageResult(
                    name="trading_day",
                    status="unknown",
                    detail=(
                        f"交易日历未覆盖 {as_of}；"
                        "覆盖不足不等于休市，将尝试刷新"
                    ),
                )
            )
            issues.append(f"交易日历未覆盖 {as_of}，休市状态未知")

        reference_ready = self._reference_ready(as_of) and calendar.covers(as_of)
        if reference_ready:
            stages.append(
                StageResult(
                    name="reference",
                    status="reused",
                    detail="交易日历/股票基础信息已就绪",
                )
            )
        else:
            try:
                sync = self._sync()
                counts = sync.sync_reference(
                    as_of - timedelta(days=10),
                    as_of + timedelta(days=self._forward_days()),
                )
                stages.append(
                    StageResult(
                        name="reference",
                        status="succeeded",
                        detail="参考数据/交易日历已刷新",
                        data=counts,
                    )
                )
                reference_ready = self._reference_ready(as_of)
                calendar = self._calendar()
                if calendar.covers(as_of):
                    if not calendar.is_trading_day(as_of):
                        stages[-2] = StageResult(
                            name="trading_day",
                            status="skipped",
                            detail=f"刷新后确认 {as_of} 休市",
                        )
                        return DataUpdateResult(
                            as_of=as_of,
                            status="skipped",
                            data_ready=False,
                            stages=stages,
                            issues=[f"{as_of} 是已确认的休市日"],
                        )
                    stages[-2] = StageResult(
                        name="trading_day", status="succeeded"
                    )
                else:
                    issues.append(
                        f"交易日历刷新后仍不覆盖 {as_of}；按未知继续，不判定为休市"
                    )
            except Exception as exc:  # noqa: BLE001
                stages.append(
                    StageResult(
                        name="reference",
                        status="unavailable"
                        if not self._token()
                        else "failed",
                        detail=str(exc),
                    )
                )
                issues.append(f"参考数据未更新：{exc}")

        manifest_ready = self.store.read_manifest(as_of) is not None
        if manifest_ready:
            stages.append(
                StageResult(
                    name="incremental_sync",
                    status="reused",
                    detail=f"{as_of} 原始数据已存在",
                )
            )
        else:
            try:
                sync = self._sync()
                results = sync.sync_incremental(as_of)
                stages.append(
                    StageResult(
                        name="incremental_sync",
                        status="succeeded",
                        detail=f"同步 {len(results)} 个交易日",
                        data={"days": [item.trade_date.isoformat() for item in results]},
                    )
                )
                manifest_ready = self.store.read_manifest(as_of) is not None
            except Exception as exc:  # noqa: BLE001
                stages.append(
                    StageResult(
                        name="incremental_sync",
                        status="unavailable"
                        if not self._token()
                        else "failed",
                        detail=str(exc),
                    )
                )
                issues.append(f"日线增量同步未完成：{exc}")

        if self._processed_ready(as_of):
            stages.append(
                StageResult(
                    name="assemble",
                    status="reused",
                    detail=f"{as_of} 处理后日线已存在",
                )
            )
        else:
            try:
                assembler = self.assembler_factory(self.store)
                bars = assembler.build_day(as_of)
                stages.append(
                    StageResult(
                        name="assemble",
                        status="succeeded",
                        detail=f"组装 {bars.height} 行",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                stages.append(
                    StageResult(
                        name="assemble",
                        status="failed",
                        detail=str(exc),
                    )
                )
                issues.append(f"清洗组装失败：{exc}")

        quality_ok = False
        if self._processed_ready(as_of):
            quality_ok, quality_issues = self._quality(as_of)
            stages.append(
                StageResult(
                    name="quality",
                    status="succeeded" if quality_ok else "failed",
                    detail="质量门通过" if quality_ok else "；".join(quality_issues),
                )
            )
            if not quality_ok:
                issues.extend(quality_issues)
        else:
            stages.append(
                StageResult(
                    name="quality",
                    status="skipped",
                    detail="处理后日线不存在",
                )
            )

        event_status, event_counts, events_available = self._event_sync(
            session, as_of
        )
        stages.append(
            StageResult(
                name="events",
                status=(
                    "succeeded"
                    if event_status == "ok"
                    else "reused"
                    if event_status == "no_new_events"
                    else "unavailable"
                    if event_status == "unavailable"
                    else "failed"
                ),
                detail=event_status,
                data=event_counts,
            )
        )
        if not events_available:
            issues.append(
                "事件数据未确认覆盖：F 策略暂停，D 策略降低权限；"
                "这不等于确认没有新事件。"
            )

        data_ready = bool(reference_ready and manifest_ready and quality_ok)
        status = "ready" if data_ready else "incomplete"
        if not data_ready:
            issues.append("关键数据未就绪：停止新增信号，仅保留已有持仓保护。")
        return DataUpdateResult(
            as_of=as_of,
            status=status,
            data_ready=data_ready,
            stages=stages,
            issues=issues,
            event_status=event_status,
            events_available=events_available,
        )


def _latest_quarter_end(as_of: date) -> str:
    quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    for month, day in reversed(quarter_ends):
        candidate = date(as_of.year, month, day)
        if candidate <= as_of:
            return candidate.strftime("%Y%m%d")
    return date(as_of.year - 1, 12, 31).strftime("%Y%m%d")
