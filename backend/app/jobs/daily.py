"""Idempotent post-close daily decision job."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import StrategyConfig, load_strategy_config
from app.core.errors import DataNotReadyError, DataQualityError
from app.core.full_config import FullSystemConfig, load_full_config
from app.core.logging import redact
from app.data.interfaces import MarketDataProvider
from app.db.repositories import (
    AccountRepository,
    JobRepository,
    NotificationRepository,
    PlanRepository,
    SnapshotRepository,
)
from app.db.session import session_scope
from app.domain.enums import AccountMode, GlobalRiskMode
from app.events.risk import EventRiskService
from app.jobs.data_update import DataUpdateResult, DataUpdateService
from app.jobs.exits import ExitReviewService
from app.jobs.reports import ReportService
from app.pipeline_v2 import StrategyPipeline, StrategyPipelineResult
from app.portfolio.service import AccountService


class DailyJobOutcome(BaseModel):
    job_id: str
    status: str
    skipped: bool
    incomplete: bool = False
    data_issues: list[str] = []
    data_notes: list[str] = []
    plans: list[str] = []
    report: dict[str, str] = {}
    decisions: list[dict] = []
    data_stages: list[dict] = []


@dataclass
class DailyJobService:
    provider: MarketDataProvider
    config: FullSystemConfig = field(default_factory=load_full_config)
    legacy_config: StrategyConfig = field(default_factory=load_strategy_config)
    pipeline: StrategyPipeline | None = None
    reports: ReportService = field(default_factory=ReportService)
    data_update: DataUpdateService | None = None
    code_version: str = "dev"

    def __post_init__(self) -> None:
        self.pipeline = self.pipeline or StrategyPipeline(
            provider=self.provider,
            config=self.config,
            legacy_config=self.legacy_config,
            code_version=self.code_version,
        )

    def run(
        self,
        as_of: date,
        *,
        account_id: str = "default",
        strategy_mode: str = "all",
        force: bool = False,
        run_data_update: bool = False,
    ) -> DailyJobOutcome:
        with session_scope() as session:
            key = f"daily:{as_of.isoformat()}:{strategy_mode}"
            existing = JobRepository.get_by_key(session, key)
            job = JobRepository.start(
                session,
                job_name="daily",
                idempotency_key=key,
                trade_date=as_of,
            )
            if job.status == "succeeded" and not force:
                return DailyJobOutcome(
                    job_id=job.id, status=job.status, skipped=True
                )
            if existing is not None and existing.status == "running" and not force:
                # Single-flight: another caller (scheduler or API) owns this run.
                return DailyJobOutcome(
                    job_id=job.id, status=job.status, skipped=True
                )
            if force:
                job.status = "running"
                session.flush()
            # Persist the running marker before doing work so a crash leaves a
            # recoverable row and concurrent callers see the single-flight lock.
            session.commit()
            try:
                AccountRepository.ensure(
                    session,
                    account_id,
                    equity=1_000_000,
                    cash=1_000_000,
                    mode=AccountMode.PAPER,
                )
                data_result: DataUpdateResult | None = None
                incomplete: list[str] = []
                data_notes: list[str] = []
                if self.data_update is not None and run_data_update:
                    data_result = self.data_update.run(as_of, session)
                    if not data_result.data_ready:
                        incomplete.extend(data_result.issues)
                    data_notes.extend(
                        issue
                        for issue in data_result.issues
                        if issue not in incomplete
                    )
                PlanRepository.expire_overdue(session, as_of)
                account_service = AccountService(
                    session, self.config, account_id
                )
                # Order matters: corporate actions -> T+1 availability ->
                # reconciliation freshness -> mark-to-market + existing
                # position decisions -> only then new plans.
                account_service.apply_due_corporate_actions(as_of)
                account_service.refresh_available(as_of)
                account_service.mark_stale(as_of)
                EventRiskService(session, self.config, account_id).apply(as_of)
                exit_result = ExitReviewService(
                    self.provider, self.config, account_id
                ).review(as_of, session)
                decisions = exit_result.decisions
                account = AccountRepository.to_state(session, account_id)

                data_version = self._data_version(as_of)
                result: StrategyPipelineResult | None = None
                if data_result is None or data_result.data_ready:
                    try:
                        result = self.pipeline.run(  # type: ignore[union-attr]
                            as_of,
                            account,
                            strategy_mode=strategy_mode,
                            session=session,
                            data_version=data_version,
                            events_available=(
                                data_result.events_available
                                if data_result is not None
                                else None
                            ),
                        )
                    except (DataNotReadyError, DataQualityError) as exc:
                        # Existing-position protection has already run.  Missing
                        # selection data must not roll it back or stop the report.
                        incomplete.append(redact(str(exc)))

                plan_ids: list[str] = []
                if result is not None:
                    for plan in result.plans:
                        PlanRepository.save(
                            session,
                            plan,
                            account_id=account_id,
                            trade_date=as_of,
                        )
                        plan_ids.append(plan.id)
                    self._notify(session, as_of, result)
                else:
                    NotificationRepository.add(
                        session,
                        level="warning",
                        code="daily_data_incomplete",
                        title=f"每日任务数据不完整 {as_of}",
                        message=(
                            "；".join(incomplete)
                            or "关键数据未就绪；已保留持仓保护，停止新增信号。"
                        ),
                        dedupe_key=f"daily-incomplete:{as_of.isoformat()}",
                        trade_date=as_of,
                    )

                state = AccountRepository.to_state(session, account_id)
                risk_mode = (
                    result.regime.global_risk
                    if result is not None
                    else GlobalRiskMode.NORMAL
                )
                SnapshotRepository.save(
                    session,
                    account_id=account_id,
                    trade_date=as_of,
                    state=state,
                    risk_mode=risk_mode,
                    detail={
                        "by_strategy": result.by_strategy if result else {},
                        "regime": (
                            result.regime.model_dump(mode="json")
                            if result is not None
                            else None
                        ),
                        "candidates": (
                            [
                                candidate.model_dump(mode="json")
                                for candidate in result.candidates
                            ]
                            if result is not None
                            else []
                        ),
                        "observations": (
                            [
                                candidate.model_dump(mode="json")
                                for candidate in result.observations
                            ]
                            if result is not None
                            else []
                        ),
                        "rejected": (
                            [
                                item.model_dump(mode="json")
                                for item in result.rejected
                            ]
                            if result is not None
                            else []
                        ),
                        "decisions": [
                            decision.model_dump(mode="json")
                            for decision in decisions
                        ],
                        "incomplete": incomplete,
                        "data_notes": data_notes,
                        "data_update": (
                            data_result.model_dump(mode="json")
                            if data_result is not None
                            else None
                        ),
                    },
                )
                NotificationRepository.purge_older_than(
                    session,
                    datetime.now(UTC)
                    - timedelta(
                        days=self.config.scheduler.notification_retention_days
                    ),
                )
                report = self.reports.generate(
                    result,
                    as_of=as_of,
                    account=state,
                    decisions=decisions,
                    incomplete=incomplete,
                    data_notes=data_notes,
                    position_reviewed=True,
                )
                JobRepository.finish(
                    session,
                    job.id,
                    status="succeeded",
                    data_version=(
                        result.regime.data_version if result else None
                    ),
                    strategy_version=self.config.version,
                )
                job.detail_json = {
                    "incomplete": incomplete,
                    "data_notes": data_notes,
                    "decisions": [
                        decision.model_dump(mode="json")
                        for decision in decisions
                    ],
                    "data_update": (
                        data_result.model_dump(mode="json")
                        if data_result is not None
                        else None
                    ),
                }
                return DailyJobOutcome(
                    job_id=job.id,
                    status="succeeded",
                    skipped=False,
                    incomplete=bool(incomplete),
                    data_issues=incomplete,
                    data_notes=data_notes,
                    plans=plan_ids,
                    report=report,
                    decisions=[
                        decision.model_dump(mode="json")
                        for decision in decisions
                    ],
                    data_stages=(
                        [
                            stage.model_dump(mode="json")
                            for stage in data_result.stages
                        ]
                        if data_result is not None
                        else []
                    ),
                )
            except Exception as exc:
                # Drop partial plans/snapshots but persist the failure and its
                # notification; session_scope would otherwise roll everything
                # back, hiding the failure from the scheduler and UI.
                session.rollback()
                message = redact(str(exc))
                JobRepository.finish(session, job.id, status="failed", error=message)
                NotificationRepository.add(
                    session,
                    level="critical",
                    code="daily_job_failed",
                    title=f"每日任务失败 {as_of}",
                    message=message,
                    dedupe_key=f"daily-failed:{as_of.isoformat()}",
                    trade_date=as_of,
                )
                session.commit()
                raise

    def _data_version(self, as_of: date) -> str | None:
        store = getattr(self.provider, "store", None)
        if store is None:
            return None
        manifest = store.read_manifest(as_of)
        if manifest and manifest.get("data_version"):
            return str(manifest["data_version"])
        return None

    def _notify(
        self, session: Session, as_of: date, result: StrategyPipelineResult
    ) -> None:
        regime = result.regime
        if regime.global_risk.value != "normal":
            NotificationRepository.add(
                session,
                level="risk",
                code="global_risk",
                title=f"全局风险开关：{regime.global_risk.value}",
                message="；".join(regime.risk_reasons) or "市场风险/数据状态触发全局开关",
                dedupe_key=f"global-risk:{as_of.isoformat()}:{regime.global_risk.value}",
                trade_date=as_of,
            )
        for flag in regime.degraded_flags:
            NotificationRepository.add(
                session,
                level="warning",
                code=f"degraded:{flag}",
                title=f"数据降级：{flag}",
                message="数据不完整，相关信号已降档/暂停；请查看运行健康。",
                dedupe_key=f"degraded:{as_of.isoformat()}:{flag}",
                trade_date=as_of,
            )
        for plan in result.plans:
            if plan.status.value == "PENDING_EXECUTION":
                NotificationRepository.add(
                    session,
                    level="info",
                    code="plan_ready",
                    title=f"{plan.candidate.ts_code} 计划待执行",
                    message=(
                        f"{plan.strategy.value if plan.strategy else ''} "
                        f"{plan.candidate.ts_code} 入场 {plan.entry_plan.entry_price} "
                        f"止损 {plan.stop_plan.initial_stop} "
                        f"股数 {plan.position_plan.shares}"
                    ),
                    dedupe_key=f"plan:{plan.id}",
                    trade_date=as_of,
                )
