"""Data/account freshness checks for the scheduler and the health page."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.calendar import TradingCalendar
from app.data.interfaces import MarketDataProvider
from app.db.repositories import AccountRepository, NotificationRepository
from app.db.session import session_scope

SHANGHAI = ZoneInfo("Asia/Shanghai")


class DataHealthReport(BaseModel):
    as_of: date
    expected_trade_date: date | None
    latest_processed: str | None
    latest_raw: str | None
    account_last_synced: datetime | None
    account_last_reconciled: datetime | None = None
    account_last_valuation: datetime | None = None
    account_stale: bool
    ok: bool
    issues: list[str] = []


def _latest_partition(root: Path) -> str | None:
    if not root.exists():
        return None
    names = sorted(
        child.name.split("=", 1)[1]
        for child in root.iterdir()
        if child.is_dir() and child.name.startswith("trade_date=")
    )
    return names[-1] if names else None


def _latest_manifest(store) -> str | None:
    root = Path(store.raw_dir) / "manifests"
    if not root.exists():
        return None
    names = sorted(path.stem for path in root.glob("*.json"))
    return names[-1] if names else None


@dataclass
class DataHealthService:
    provider: MarketDataProvider
    account_id: str = "default"
    account_max_age_days: int = 3

    def _expected_trade_date(self, as_of: date) -> date | None:
        try:
            frame = self.provider.load_trade_cal(as_of - timedelta(days=45), as_of)
        except Exception:  # noqa: BLE001 - health check must not crash the job
            return None
        calendar = TradingCalendar.from_trade_cal(frame)
        candidates = [item for item in calendar.dates if item <= as_of]
        return candidates[-1] if candidates else None

    def check(self, as_of: date, session: Session | None = None) -> DataHealthReport:
        expected = self._expected_trade_date(as_of)
        store = getattr(self.provider, "store", None)
        latest_processed = (
            _latest_partition(Path(store.processed_dir) / "bars") if store else None
        )
        latest_raw = _latest_manifest(store) if store else None

        issues: list[str] = []
        if expected is None:
            issues.append("交易日历不可用")
        elif latest_processed is None:
            issues.append("尚未生成处理后日线")
        else:
            try:
                processed_date = date.fromisoformat(latest_processed)
            except ValueError:
                processed_date = None
            if processed_date is None or processed_date < expected:
                issues.append(
                    f"处理后日线过期：{latest_processed} < {expected.isoformat()}"
                )

        account_last_synced: datetime | None = None
        account_last_reconciled: datetime | None = None
        account_last_valuation: datetime | None = None
        account_stale = True
        if session is not None:
            account = AccountRepository.get(session, self.account_id)
            if account is not None:
                account_last_synced = account.last_synced_at
                account_last_reconciled = account.last_reconciled_at
                account_last_valuation = account.last_valuation_at
                account_stale = bool(account.stale)
                reference = (
                    account_last_reconciled.date()
                    if account_last_reconciled is not None
                    else None
                )
                if account.mode == "real" and reference is None:
                    issues.append("账户从未人工核对（盘后估值不算对账）")
                elif (
                    account.mode == "real"
                    and reference is not None
                    and (as_of - reference).days > self.account_max_age_days
                ):
                    issues.append(
                        f"账户人工核对过期：{reference.isoformat()} "
                        f"> {self.account_max_age_days} 天"
                    )
                if account.valuation_incomplete:
                    issues.append(
                        f"估值不完整：{account.valuation_note or '部分持仓缺少行情'}"
                    )

        return DataHealthReport(
            as_of=as_of,
            expected_trade_date=expected,
            latest_processed=latest_processed,
            latest_raw=latest_raw,
            account_last_synced=account_last_synced,
            account_last_reconciled=account_last_reconciled,
            account_last_valuation=account_last_valuation,
            account_stale=account_stale,
            ok=not issues,
            issues=issues,
        )

    def check_and_notify(self, as_of: date | None = None) -> DataHealthReport:
        day = as_of or datetime.now(SHANGHAI).date()
        with session_scope() as session:
            report = self.check(day, session)
            for index, issue in enumerate(report.issues):
                NotificationRepository.add(
                    session,
                    level="warning",
                    code="data_health",
                    title="数据/账户健康检查",
                    message=issue,
                    dedupe_key=f"data-health:{day.isoformat()}:{index}:{issue}",
                    trade_date=day,
                )
        return report
