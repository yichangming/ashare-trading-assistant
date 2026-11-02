from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from app.core.full_config import load_full_config
from app.db.repositories import (
    AccountRepository,
    JobRepository,
    NotificationRepository,
    PositionRepository,
)
from app.db.session import init_db, reset_engine, session_scope
from app.domain.enums import GlobalRiskMode
from app.domain.models import AccountState
from app.jobs.daily import DailyJobService
from app.jobs.health import DataHealthService
from app.jobs.intraday import IntradayMonitor, is_trading_session
from app.jobs.scheduler import SchedulerService

SHANGHAI = ZoneInfo("Asia/Shanghai")


class _Quote:
    def __init__(self, price: float | None):
        self.price = price

    def get_quote(self, ts_code: str):
        if self.price is None:
            return None
        return self.price, datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI)


class _FakeReports:
    def generate(self, result, **_kwargs):
        return {"markdown": "", "html": "", "markdown_path": "", "html_path": ""}


class _StubRegime(SimpleNamespace):
    def model_dump(self, **_kwargs):
        return {
            "global_risk": self.global_risk.value,
            "degraded_flags": self.degraded_flags,
            "data_version": self.data_version,
        }


class _StubPipeline:
    def run(self, day, account, **kwargs):
        state = AccountState(
            equity=Decimal("1000000"),
            cash=Decimal("1000000"),
            peak_equity=Decimal("1000000"),
        )
        return SimpleNamespace(
            regime=_StubRegime(
                global_risk=GlobalRiskMode.NORMAL,
                degraded_flags=[],
                data_version="dv",
            ),
            account=state,
            plans=[],
            candidates=[],
            observations=[],
            rejected=[],
            by_strategy={},
        )


def test_trading_session_boundaries():
    assert is_trading_session(datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI))
    assert not is_trading_session(datetime(2026, 11, 2, 12, 0, tzinfo=SHANGHAI))
    assert not is_trading_session(datetime(2026, 11, 7, 10, 0, tzinfo=SHANGHAI))


def test_intraday_monitor_raises_hard_stop_without_auto_trade():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            sector="银行",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=10,
            current_price=10.5,
        )
    monitor = IntradayMonitor(_Quote(9.9))
    result = monitor.check(
        date(2026, 11, 2),
        now=datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI),
    )
    assert result["checked"] == 1
    assert result["alerts"] == 1
    with session_scope() as session:
        notifications = NotificationRepository.list_recent(session)
        assert any(item.code == "hard_stop_breached" for item in notifications)


def test_intraday_monitor_reports_data_degradation_without_fake_price():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="000001.SZ",
            sector="银行",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=10,
            current_price=10.5,
        )
    monitor = IntradayMonitor(_Quote(None))
    result = monitor.check(
        date(2026, 11, 2),
        now=datetime(2026, 11, 2, 10, 0, tzinfo=SHANGHAI),
    )
    assert result["degraded"]
    assert result["checked"] == 0
    with session_scope() as session:
        assert any(
            item.code == "intraday_data_degraded"
            for item in NotificationRepository.list_recent(session)
        )


def test_daily_job_is_idempotent():
    reset_engine()
    init_db("sqlite://")
    service = DailyJobService(
        provider=SimpleNamespace(store=None),
        config=load_full_config(),
        pipeline=_StubPipeline(),
        reports=_FakeReports(),
    )
    first = service.run(date(2026, 11, 2))
    assert first.status == "succeeded" and not first.skipped
    second = service.run(date(2026, 11, 2))
    assert second.skipped
    with session_scope() as session:
        jobs = JobRepository.list_recent(session)
        assert jobs[0].status == "succeeded"


class _FailingPipeline:
    def __init__(self):
        self.calls = 0

    def run(self, day, account, **kwargs):
        self.calls += 1
        raise RuntimeError("pipeline exploded")


def test_daily_failure_is_persisted_and_notifies():
    reset_engine()
    init_db("sqlite://")
    pipeline = _FailingPipeline()
    service = DailyJobService(
        provider=SimpleNamespace(store=None),
        config=load_full_config(),
        pipeline=pipeline,
        reports=_FakeReports(),
    )
    with pytest.raises(RuntimeError, match="pipeline exploded"):
        service.run(date(2026, 11, 2))
    with session_scope() as session:
        jobs = JobRepository.list_recent(session)
        assert jobs[0].status == "failed"
        assert "pipeline exploded" in (jobs[0].error or "")
        assert any(
            item.code == "daily_job_failed"
            for item in NotificationRepository.list_recent(session)
        )


def test_running_job_is_single_flight():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        JobRepository.start(
            session,
            job_name="daily",
            idempotency_key="daily:2026-11-02:all",
            trade_date=date(2026, 11, 2),
        )
    pipeline = _FailingPipeline()
    service = DailyJobService(
        provider=SimpleNamespace(store=None),
        config=load_full_config(),
        pipeline=pipeline,
        reports=_FakeReports(),
    )
    outcome = service.run(date(2026, 11, 2))
    assert outcome.skipped
    assert pipeline.calls == 0


def test_restart_recovery_marks_running_jobs_failed():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        job = JobRepository.start(
            session,
            job_name="daily",
            idempotency_key="daily:2026-11-02:all",
            trade_date=date(2026, 11, 2),
        )
        job_id = job.id
    with session_scope() as session:
        assert JobRepository.recover_running(session) == 1
    with session_scope() as session:
        row = session.get(type(job), job_id)
        assert row is not None
        assert row.status == "failed"
        assert "restart" in (row.error or "")
        assert row.retry_count == 1


class _CalendarProvider:
    def __init__(self, days: list[date]):
        self.days = days

    def load_trade_cal(self, start: date, end: date):
        rows = [day for day in self.days if start <= day <= end]
        return pl.DataFrame(
            {
                "cal_date": rows,
                "is_open": [1 for _ in rows],
            }
        )


class _FakeDaily:
    def __init__(self, provider):
        self.provider = provider
        self.calls: list[date] = []
        self.fail = False

    def run(self, day, **kwargs):
        self.calls.append(day)
        if self.fail:
            raise RuntimeError("daily failed")


def test_scheduler_trading_day_gate():
    provider = _CalendarProvider([date(2026, 11, 2)])
    service = SchedulerService(
        daily=_FakeDaily(provider),
        intraday=SimpleNamespace(),
        config=load_full_config(),
    )
    assert service.is_trading_day(date(2026, 11, 2))
    assert not service.is_trading_day(date(2026, 11, 1))


def test_scheduler_schedules_retry_with_backoff():
    config = load_full_config().model_copy(deep=True)
    config.scheduler.retry_count = 2
    config.scheduler.retry_base_seconds = 1
    provider = _CalendarProvider([date(2026, 11, 2)])
    service = SchedulerService(
        daily=_FakeDaily(provider),
        intraday=SimpleNamespace(),
        config=config,
    )
    service._schedule_retry(date(2026, 11, 2), 0)
    jobs = service.scheduler.get_jobs()
    assert any(job.id == "daily-retry:2026-11-02" for job in jobs)
    service.shutdown()


def test_data_health_reports_missing_processed_bars(tmp_path):
    provider = _CalendarProvider([date(2026, 11, 2)])
    provider.store = SimpleNamespace(
        processed_dir=tmp_path / "processed",
        raw_dir=tmp_path / "raw",
    )
    report = DataHealthService(provider).check(date(2026, 11, 2))
    assert not report.ok
    assert any("处理后日线" in issue for issue in report.issues)

    bars = tmp_path / "processed" / "bars" / "trade_date=2026-11-02"
    bars.mkdir(parents=True)
    report = DataHealthService(provider).check(date(2026, 11, 2))
    assert report.ok
    assert report.latest_processed == "2026-11-02"


def test_notifications_order_risk_before_info():
    reset_engine()
    init_db("sqlite://")
    with session_scope() as session:
        NotificationRepository.add(
            session,
            level="info",
            code="info",
            title="info",
            message="info",
            dedupe_key="n-info",
        )
        NotificationRepository.add(
            session,
            level="critical",
            code="risk",
            title="risk",
            message="risk",
            dedupe_key="n-risk",
        )
    with session_scope() as session:
        levels = [
            item.level for item in NotificationRepository.list_recent(session)
        ]
    assert levels[0] == "critical"


def test_log_redaction_hides_secrets():
    from app.core.logging import redact

    text = redact("api failed token=abc123 password: hunter2 ok")
    assert "abc123" not in text
    assert "hunter2" not in text
    assert "***" in text
