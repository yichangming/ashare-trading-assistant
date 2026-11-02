"""Process-local APScheduler wrapper.

The scheduler runs in the server process, independent of any browser page.
A multi-worker deployment must run exactly one scheduler process; the Docker
Compose deployment uses one worker for this reason.

Reliability rules:

* every job has a persisted idempotency key in ``job_runs``;
* a crash leaves a ``running`` row that startup recovery marks failed;
* failed daily runs are retried with exponential backoff, up to
  ``scheduler.retry_count``;
* the post-close job is skipped on non-trading days from ``trade_cal``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.calendar import TradingCalendar
from app.core.full_config import FullSystemConfig
from app.jobs.daily import DailyJobService
from app.jobs.health import DataHealthService
from app.jobs.intraday import IntradayMonitor, is_trading_session

logger = logging.getLogger(__name__)
SHANGHAI = ZoneInfo("Asia/Shanghai")


class SchedulerService:
    def __init__(
        self,
        daily: DailyJobService,
        intraday: IntradayMonitor,
        config: FullSystemConfig,
        health: DataHealthService | None = None,
    ):
        self.daily = daily
        self.intraday = intraday
        self.config = config
        self.health = health or DataHealthService(daily.provider)
        self.scheduler = BackgroundScheduler(timezone=SHANGHAI)

    def start(self) -> None:
        hour, minute = (
            int(part) for part in self.config.scheduler.daily_time.split(":")
        )
        health_hour, health_minute = (
            int(part) for part in self.config.scheduler.data_health_time.split(":")
        )
        self.scheduler.add_job(
            self.run_daily,
            CronTrigger(
                day_of_week="mon-fri",
                hour=hour,
                minute=minute,
                timezone=SHANGHAI,
            ),
            id="daily",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )
        self.scheduler.add_job(
            self.run_data_health,
            CronTrigger(
                day_of_week="mon-fri",
                hour=health_hour,
                minute=health_minute,
                timezone=SHANGHAI,
            ),
            id="data_health",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self.run_intraday,
            IntervalTrigger(
                seconds=self.config.scheduler.intraday_interval_seconds
            ),
            id="intraday",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def is_trading_day(self, day: date) -> bool:
        try:
            frame = self.daily.provider.load_trade_cal(day, day)
        except Exception:  # noqa: BLE001 - fall back to weekday rather than block
            logger.warning("trade calendar unavailable; using weekday check")
            return day.weekday() < 5
        if frame.is_empty():
            return day.weekday() < 5
        calendar = TradingCalendar.from_trade_cal(frame)
        return calendar.is_trading_day(day)

    def run_daily(self, attempt: int = 0, force: bool = False) -> None:
        day = datetime.now(SHANGHAI).date()
        if not self.is_trading_day(day):
            logger.info("skipping daily job on non-trading day %s", day)
            return
        try:
            outcome = self.daily.run(
                day, run_data_update=True, force=force
            )
            if (
                outcome is not None
                and getattr(outcome, "incomplete", False)
                and not getattr(outcome, "plans", [])
                and not getattr(outcome, "skipped", False)
            ):
                logger.warning("daily job incomplete for %s; retrying", day)
                self._schedule_retry(day, attempt)
        except Exception:  # noqa: BLE001
            logger.exception("daily job failed for %s (attempt %s)", day, attempt)
            self._schedule_retry(day, attempt)

    def _schedule_retry(self, day: date, attempt: int) -> None:
        retry_count = self.config.scheduler.retry_count
        if attempt >= retry_count:
            logger.error("daily job exhausted %s retries for %s", retry_count, day)
            return
        delay = self.config.scheduler.retry_base_seconds * (2**attempt)
        when = datetime.now(SHANGHAI) + timedelta(seconds=delay)
        self.scheduler.add_job(
            self.run_daily,
            trigger="date",
            run_date=when,
            args=[attempt + 1, True],
            id=f"daily-retry:{day.isoformat()}",
            replace_existing=True,
            max_instances=1,
        )
        logger.warning("daily job retry %s scheduled for %s", attempt + 1, when)

    def run_data_health(self) -> None:
        try:
            self.health.check_and_notify(datetime.now(SHANGHAI).date())
        except Exception:  # noqa: BLE001
            logger.exception("data health job failed")

    def run_intraday(self) -> None:
        now = datetime.now(SHANGHAI)
        if not is_trading_session(now):
            return
        try:
            self.intraday.check(now.date(), now=now)
        except Exception:  # noqa: BLE001
            logger.exception("intraday monitor failed")
