"""FastAPI application for the local single-user dashboard."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import secrets
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from app.api.auth import AuthConfig
from app.backtest.engine import BacktestEngine
from app.core.config import BACKEND_ROOT, load_app_config
from app.core.full_config import load_full_config
from app.core.logging import configure_logging
from app.data.provider import ParquetMarketDataProvider
from app.data.store import ParquetStore
from app.db.models import BacktestRunRow, PlanRow
from app.db.repositories import (
    AccountRepository,
    AuditRepository,
    BacktestRepository,
    ExecutionRepository,
    JobRepository,
    NotificationRepository,
    PlanRepository,
    SettingRepository,
    SnapshotRepository,
)
from app.db.session import get_db, prepare_database
from app.domain.enums import (
    AccountMode,
    ExecutionSide,
    ExitSession,
    PlanStatus,
)
from app.events.ingest import EventIngestor
from app.jobs.b2 import B2Monitor, TushareMinuteSource
from app.jobs.daily import DailyJobService
from app.jobs.data_update import DataUpdateService
from app.jobs.health import DataHealthService
from app.jobs.intraday import IntradayMonitor, TushareQuoteAdapter
from app.jobs.scheduler import SchedulerService
from app.portfolio.csv_io import TEMPLATES, import_csv
from app.portfolio.decisions import PositionDecisionService
from app.portfolio.service import AccountService

SHANGHAI = ZoneInfo("Asia/Shanghai")
STRATEGY_NAMES = {
    "A": "趋势回踩",
    "B": "涨停承接",
    "C": "平台突破",
    "D": "超跌修复",
    "E": "区间交易",
    "F": "事件驱动",
}
ACTION_LABELS = {
    "hold": "持有",
    "conditional_add": "条件加仓",
    "reduce": "减仓",
    "exit": "清仓",
    "wait_execution": "等待执行",
    "data_anomaly": "数据异常",
}
RISK_ACTIONS = {"exit", "reduce", "data_anomaly"}


def _create_store() -> ParquetStore:
    app_config = load_app_config()
    data = app_config.data
    return ParquetStore(
        BACKEND_ROOT / data.raw_dir,
        BACKEND_ROOT / data.processed_dir,
        BACKEND_ROOT / data.regime_dir,
    )


def _latest_partition(root: Path) -> str | None:
    if not root.exists():
        return None
    dates = sorted(
        child.name.split("=", 1)[1]
        for child in root.iterdir()
        if child.is_dir() and child.name.startswith("trade_date=")
    )
    return dates[-1] if dates else None


def _latest_manifest(store: ParquetStore) -> str | None:
    root = store.raw_dir / "manifests"
    if not root.exists():
        return None
    files = sorted(path.stem for path in root.glob("*.json"))
    return files[-1] if files else None


def _notification_count(session: Session) -> int:
    return len(NotificationRepository.list_recent(session, unacknowledged_only=True, limit=1000))


def _regime_from_snapshot(session: Session) -> dict | None:
    snapshots = SnapshotRepository.list_recent(session, "default", limit=1)
    if not snapshots:
        return None
    return (snapshots[-1].detail_json or {}).get("regime")


def _plan_view(row: PlanRow) -> dict:
    payload = row.plan_json or {}
    entry_plan = payload.get("entry_plan") or {}
    stop_plan = payload.get("stop_plan") or {}
    return {
        "id": row.id,
        "trade_date": row.trade_date.isoformat(),
        "ts_code": row.ts_code,
        "sector": row.sector,
        "strategy": row.strategy,
        "sub_mode": row.sub_mode,
        "status": row.status,
        "confirmation_price": (
            str(row.confirmation_price) if row.confirmation_price is not None else None
        ),
        "max_acceptable_price": (
            str(row.max_acceptable_price)
            if row.max_acceptable_price is not None
            else None
        ),
        "entry": str(entry_plan.get("trigger_price") or entry_plan.get("entry_zone_high") or ""),
        "stop": str(row.stop_price) if row.stop_price is not None else "",
        "shares": row.shares,
        "r_value": str(stop_plan.get("r_value") or ""),
        "reasons": payload.get("reasons") or [],
        "cancel_reason": payload.get("cancel_reason"),
        "expires_on": row.expires_on.isoformat(),
    }


def _candidate_view(item: dict) -> dict:
    return {
        "strategy": item.get("strategy") or "?",
        "sub_mode": item.get("sub_mode"),
        "ts_code": item.get("ts_code"),
        "sector": item.get("sector"),
        "score": float(item.get("score") or 0),
        "stage": (item.get("evidence") or {}).get("stage", "observing"),
        "reasons": item.get("reasons") or [],
        "cancel_conditions": (item.get("evidence") or {}).get("cancel_conditions") or [],
        "data_time": item.get("data_time"),
        "setup_type": item.get("setup_type"),
    }


class _FakeReport:
    def generate(self, result) -> dict[str, str]:
        return {}


def create_app(database_url: str | None = None) -> FastAPI:
    configure_logging()
    config = load_full_config()
    store = _create_store()
    provider = ParquetMarketDataProvider(store)
    auth = AuthConfig.from_env()
    app = FastAPI(title="A 股交易辅助台", docs_url=None, redoc_url=None)
    app.add_middleware(
        SessionMiddleware,
        secret_key=auth.secret_key,
        same_site="lax",
        https_only=auth.https_only,
    )
    templates = Jinja2Templates(
        directory=str(Path(__file__).resolve().parents[1] / "web" / "templates")
    )
    app.mount(
        "/static",
        StaticFiles(
            directory=str(Path(__file__).resolve().parents[1] / "web" / "static")
        ),
        name="static",
    )
    daily = DailyJobService(
        provider=provider,
        config=config,
        data_update=DataUpdateService(store=store, config=config),
    )
    minute_source = TushareMinuteSource()
    intraday = IntradayMonitor(
        TushareQuoteAdapter(),
        b2=B2Monitor(provider, config, minute_source),
        minute_source=minute_source,
        calendar_provider=provider,
        provider=provider,
        config=config,
    )
    scheduler = SchedulerService(daily, intraday, config)
    app.state.daily = daily
    app.state.scheduler = scheduler
    app.state.provider = provider
    app.state.store = store
    app.state.config = config

    prepare_database(database_url)

    def logged_in(request: Request) -> bool:
        return request.session.get("user") == "owner"

    def page_guard(request: Request) -> RedirectResponse | None:
        if not logged_in(request):
            next_path = request.url.path
            return RedirectResponse(f"/login?next={next_path}", status_code=303)
        return None

    def api_guard(request: Request) -> None:
        if not logged_in(request):
            raise HTTPException(status_code=401, detail="登录已失效")

    def account_service(session: Session) -> AccountService:
        return AccountService(session, config)

    def base_context(request: Request, active: str, **kwargs: Any) -> dict:
        with next(get_db()) as session:
            account_state = AccountRepository.to_state(session)
            account = {
                "equity": float(account_state.equity),
                "cash": float(account_state.cash),
                "positions_value": float(account_state.positions_value),
                "heat": account_state.heat,
                "drawdown_pct": account_state.drawdown_pct,
                "mode": account_state.mode.value,
                "risk_config_configured": account_state.risk_config_configured,
                "last_synced_at": account_state.last_synced_at.isoformat()
                if account_state.last_synced_at
                else None,
                "last_valuation_at": account_state.last_valuation_at.isoformat()
                if account_state.last_valuation_at
                else None,
                "last_reconciled_at": account_state.last_reconciled_at.isoformat()
                if account_state.last_reconciled_at
                else None,
                "stale": account_state.stale,
                "valuation_incomplete": account_state.valuation_incomplete,
                "valuation_note": account_state.valuation_note,
                "net_deposits": float(account_state.net_deposits),
                "max_drawdown": account_state.max_drawdown,
            }
            latest_processed = _latest_partition(store.processed_dir / "bars")
            try:
                value_date = (
                    date.fromisoformat(latest_processed)
                    if latest_processed
                    else datetime.now(SHANGHAI).date()
                )
            except ValueError:
                value_date = datetime.now(SHANGHAI).date()
            decisions = PositionDecisionService(
                session, config, provider=provider
            ).evaluate_all(
                value_date,
                session_type=ExitSession.DAILY,
                persist=False,
                notify=False,
            )
            notifications = NotificationRepository.list_recent(
                session, unacknowledged_only=True, limit=20
            )
            notification_count = _notification_count(session)
            demo = bool(request.session.get("demo", False)) or (
                account_state.mode == AccountMode.DEMO
            )
            regime = _regime_from_snapshot(session)
        with next(get_db()) as session:
            latest_plan_date = session.scalar(
                select(PlanRow.trade_date).order_by(PlanRow.trade_date.desc()).limit(1)
            )
            plan_rows = (
                session.scalars(
                    select(PlanRow)
                    .where(PlanRow.trade_date == latest_plan_date)
                    .order_by(PlanRow.created_at.desc())
                ).all()
                if latest_plan_date
                else []
            )
            plans = [_plan_view(row) for row in plan_rows]
            execution_rows = ExecutionRepository.list_for_account(session)
            execution_rows = execution_rows[-100:][::-1]
            executions = [
                {
                    "executed_at": row.executed_at.isoformat(),
                    "ts_code": row.ts_code,
                    "side": row.side,
                    "price": str(row.price),
                    "shares": row.shares,
                    "fees": str(row.fees),
                    "source": row.source,
                    "plan_id": row.plan_id,
                }
                for row in execution_rows
            ]
            data_status = {
                "latest_processed": latest_processed,
                "latest_raw": _latest_manifest(store),
                "stale": account["stale"],
                "valuation_incomplete": account["valuation_incomplete"],
                "quote_freshness": (
                    "unknown"
                    if not decisions
                    else (
                        "partial"
                        if any(
                            item.valuation_status.value != "ok"
                            for item in decisions
                        )
                        else "ok"
                    )
                ),
            }
            pending_count = sum(
                1
                for plan in plans
                if plan["status"] in ("PENDING_EXECUTION", "PENDING_ENTRY")
            )
        permissions = (regime or {}).get("strategy_permissions") or {}
        context = {
            "request": request,
            "active": active,
            "today": datetime.now(SHANGHAI).date().isoformat(),
            "data_time": data_status["latest_processed"],
            "demo": demo,
            "account": account,
            "regime": regime,
            "permissions": permissions,
            "strategy_names": STRATEGY_NAMES,
            "plans": plans,
            "pending_count": pending_count,
            "positions": sorted(
                [
                    {
                        "ts_code": decision.ts_code,
                        "sector": decision.sector,
                        "shares": decision.total_shares,
                        "available_shares": decision.available_shares,
                        "cost_price": (
                            str(state_position.cost_price)
                            if state_position is not None
                            else "0"
                        ),
                        "current_price": str(
                            decision.trigger_price
                            or (
                                state_position.current_price
                                if state_position is not None
                                else 0
                            )
                        ),
                        "current_stop": str(decision.current_stop or 0),
                        "proposed_stop": (
                            str(decision.proposed_stop)
                            if decision.proposed_stop is not None
                            else None
                        ),
                        "strategy": (
                            decision.primary_strategy.value
                            if decision.primary_strategy
                            else None
                        ),
                        "sub_mode": (
                            decision.sub_mode.value
                            if decision.sub_mode
                            else None
                        ),
                        "rule_version": decision.rule_version,
                        "action": decision.action.value,
                        "action_label": ACTION_LABELS.get(
                            decision.action.value, decision.action.value
                        ),
                        "reason_code": decision.reason_code,
                        "explanation": decision.explanation,
                        "suggested_shares": decision.suggested_shares,
                        "restricted_shares": decision.restricted_shares,
                        "executed_shares": decision.executed_shares,
                        "pending_shares": decision.pending_shares,
                        "data_time": (
                            decision.data_time.isoformat()
                            if decision.data_time
                            else None
                        ),
                        "valuation_status": decision.valuation_status.value,
                        "needs_reconciliation": decision.needs_reconciliation,
                        "confirmed": decision.confirmed,
                        "risk": decision.action.value in RISK_ACTIONS,
                        "distance_pct": (
                            (
                                float(decision.trigger_price)
                                - float(decision.current_stop)
                            )
                            / float(decision.trigger_price)
                            if decision.trigger_price
                            and decision.current_stop
                            and float(decision.trigger_price) > 0
                            else None
                        ),
                    }
                    for decision in decisions
                    for state_position in [
                        next(
                            (
                                item
                                for item in account_state.positions
                                if item.ts_code == decision.ts_code
                            ),
                            None,
                        )
                    ]
                ],
                key=lambda item: (
                    0 if item["action"] in RISK_ACTIONS else 1,
                    item["ts_code"],
                ),
            ),
            "executions": executions,
            "notifications": notifications,
            "notification_count": notification_count,
            "data_status": data_status,
            "scheduler_running": bool(getattr(app.state.scheduler.scheduler, "running", False)),
            "demo_portfolio": demo,
        }
        context.update(kwargs)
        return context

    @app.on_event("startup")
    def _startup() -> None:
        with next(get_db()) as session:
            AccountRepository.ensure(
                session,
                equity=1_000_000,
                cash=1_000_000,
                mode=AccountMode.PAPER,
            )
            recovered = JobRepository.recover_running(session)
            if recovered:
                NotificationRepository.add(
                    session,
                    level="warning",
                    code="jobs_recovered",
                    title="服务重启：恢复未完成任务",
                    message=f"{recovered} 个运行中的任务已标记失败，可重新执行。",
                    dedupe_key=f"jobs-recovered:{datetime.now(SHANGHAI).date().isoformat()}",
                )
            session.commit()
        if os.environ.get("ASHARE_ENABLE_SCHEDULER", "0") == "1":
            scheduler.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        scheduler.shutdown()

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, next: str = "/"):
        return templates.TemplateResponse(
            request, "login.html", {"error": None, "next": next}
        )

    @app.post("/login")
    def login_submit(
        request: Request,
        password: str = Form(...),
        next: str = Form("/"),
    ):
        if not auth.check(password):
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "密码不正确", "next": next},
                status_code=401,
            )
        request.session["user"] = "owner"
        target = next if next.startswith("/") and not next.startswith("//") else "/"
        return RedirectResponse(target, status_code=303)

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request):
        if (redirect := page_guard(request)) is not None:
            return redirect
        return templates.TemplateResponse(
            request, "overview.html", base_context(request, "overview")
        )

    @app.get("/opportunities", response_class=HTMLResponse)
    def opportunities(request: Request, strategy: str = ""):
        if (redirect := page_guard(request)) is not None:
            return redirect
        with next(get_db()) as session:
            snapshots = SnapshotRepository.list_recent(session, "default", limit=1)
            detail = (snapshots[-1].detail_json or {}) if snapshots else {}
            candidates = [
                _candidate_view(item) for item in detail.get("candidates", [])
            ] + [_candidate_view(item) for item in detail.get("observations", [])]
        if strategy:
            candidates = [item for item in candidates if item["strategy"] == strategy]
        return templates.TemplateResponse(
            request,
            "opportunities.html",
            base_context(
                request,
                "opportunities",
                candidates=candidates,
                strategy_filter=strategy,
            ),
        )

    @app.get("/positions", response_class=HTMLResponse)
    def positions(request: Request):
        if (redirect := page_guard(request)) is not None:
            return redirect
        return templates.TemplateResponse(
            request, "positions.html", base_context(request, "positions")
        )

    @app.get("/plans", response_class=HTMLResponse)
    def plans_page(request: Request):
        if (redirect := page_guard(request)) is not None:
            return redirect
        return templates.TemplateResponse(
            request, "plans.html", base_context(request, "plans")
        )

    @app.get("/reports", response_class=HTMLResponse)
    def reports_page(request: Request, date_value: str = ""):
        if (redirect := page_guard(request)) is not None:
            return redirect
        root = BACKEND_ROOT / "data" / "reports"
        report_dates = sorted(
            [path.name for path in root.iterdir() if path.is_dir()]
        )[::-1] if root.exists() else []
        selected = date_value or (report_dates[0] if report_dates else "")
        markdown = ""
        if selected:
            path = root / selected / "daily.md"
            if path.exists():
                markdown = path.read_text(encoding="utf-8")
        reports = [{"date": item} for item in report_dates]
        return templates.TemplateResponse(
            request,
            "reports.html",
            base_context(
                request,
                "reports",
                reports=reports,
                selected_date=selected,
                selected_markdown=markdown,
            ),
        )

    @app.get("/reports/{report_date}/markdown")
    def report_markdown(request: Request, report_date: str):
        if (redirect := page_guard(request)) is not None:
            return redirect
        path = BACKEND_ROOT / "data" / "reports" / report_date / "daily.md"
        if not path.exists():
            raise HTTPException(404, "report not found")
        return FileResponse(path, media_type="text/markdown; charset=utf-8")

    @app.get("/reports/{report_date}/html")
    def report_html(request: Request, report_date: str):
        if (redirect := page_guard(request)) is not None:
            return redirect
        path = BACKEND_ROOT / "data" / "reports" / report_date / "daily.html"
        if not path.exists():
            raise HTTPException(404, "report not found")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    def _backtest_view(run: BacktestRunRow) -> dict:
        artifacts = run.artifacts_json or {}
        metrics = run.metrics_json or {}
        return {
            "id": run.id,
            "period_start": run.period_start.isoformat(),
            "period_end": run.period_end.isoformat(),
            "strategy_mode": run.strategy_mode,
            "status": run.status,
            "is_demo": run.is_demo,
            "total_return": metrics.get("total_return"),
            "metrics": metrics,
            "trades": artifacts.get("trades", []),
            "equity": artifacts.get("equity_curve", []),
            "missing_data": artifacts.get("missing_data", []),
            "notes": artifacts.get("notes", []),
        }

    @app.get("/backtest", response_class=HTMLResponse)
    def backtest_page(request: Request, run: str = ""):
        if (redirect := page_guard(request)) is not None:
            return redirect
        with next(get_db()) as session:
            rows = BacktestRepository.list_recent(session)
            runs = [_backtest_view(row) for row in rows]
            selected = next((item for item in runs if item["id"] == run), None)
        return templates.TemplateResponse(
            request,
            "backtest.html",
            base_context(request, "backtest", runs=runs, selected=selected),
        )

    @app.post("/backtest/run")
    def backtest_run(
        background: BackgroundTasks,
        request: Request,
        start: date = Form(...),
        end: date = Form(...),
        strategy_mode: str = Form("all"),
        initial_equity: float = Form(1_000_000),
        demo: str | None = Form(None),
    ):
        if not logged_in(request):
            return RedirectResponse("/login", status_code=303)
        run_id = secrets.token_hex(16)
        with next(get_db()) as session:
            BacktestRepository.save(
                session,
                BacktestRunRow(
                    id=run_id,
                    name="web",
                    strategy_mode=strategy_mode,
                    period_start=start,
                    period_end=end,
                    status="running",
                    is_demo=bool(demo),
                    config_json={"initial_equity": initial_equity},
                ),
            )
            session.commit()
        background.add_task(
            _run_backtest,
            run_id,
            start,
            end,
            strategy_mode,
            initial_equity,
            bool(demo),
        )
        return RedirectResponse(f"/backtest?run={run_id}", status_code=303)

    @app.get("/notifications", response_class=HTMLResponse)
    def notifications_page(request: Request, all: str = ""):
        if (redirect := page_guard(request)) is not None:
            return redirect
        with next(get_db()) as session:
            notifications = NotificationRepository.list_recent(
                session, unacknowledged_only=not bool(all), limit=200
            )
        return templates.TemplateResponse(
            request,
            "notifications.html",
            base_context(request, "notifications", notifications=notifications),
        )

    @app.post("/notifications/{notification_id}/ack")
    def acknowledge_notification(request: Request, notification_id: str):
        if not logged_in(request):
            return RedirectResponse("/login", status_code=303)
        with next(get_db()) as session:
            NotificationRepository.acknowledge(session, notification_id)
            session.commit()
        return RedirectResponse("/notifications", status_code=303)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        if (redirect := page_guard(request)) is not None:
            return redirect
        with next(get_db()) as session:
            account = AccountRepository.get(session)
            health = DataHealthService(provider).check(
                datetime.now(SHANGHAI).date(), session
            )
            event_sync = SettingRepository.get(session, "event_sync", {})
            recent_jobs = JobRepository.list_recent(session, limit=1)
            data_update = (
                ((recent_jobs[0].detail_json or {}).get("data_update"))
                if recent_jobs
                else None
            )
            account_view = {
                "equity": float(account.equity) if account else 0,
                "cash": float(account.cash) if account else 0,
                "mode": account.mode if account else "paper",
                "max_drawdown": account.max_drawdown if account else 0.15,
                "risk_config_configured": bool(account and account.risk_config_configured),
                "last_synced_at": account.last_synced_at.isoformat()
                if account and account.last_synced_at
                else None,
                "last_valuation_at": account.last_valuation_at.isoformat()
                if account and account.last_valuation_at
                else None,
                "last_reconciled_at": account.last_reconciled_at.isoformat()
                if account and account.last_reconciled_at
                else None,
                "valuation_incomplete": bool(
                    account and account.valuation_incomplete
                ),
                "valuation_note": account.valuation_note if account else None,
                "stale": bool(account and account.stale),
            }
        return templates.TemplateResponse(
            request,
            "settings.html",
            base_context(
                request,
                "settings",
                account=account_view,
                health=health,
                event_sync=event_sync,
                data_update=data_update,
            ),
        )

    @app.post("/positions/execute")
    def execute_form(
        request: Request,
        ts_code: str = Form(...),
        side: str = Form(...),
        price: float = Form(...),
        shares: int = Form(...),
        fees: float | None = Form(None),
        executed_at: str = Form(...),
        plan_id: str | None = Form(None),
        external_id: str | None = Form(None),
    ):
        if not logged_in(request):
            return RedirectResponse("/login", status_code=303)
        with next(get_db()) as session:
            service = account_service(session)
            try:
                service.record_execution(
                    ts_code=ts_code,
                    side=ExecutionSide(side),
                    price=price,
                    shares=shares,
                    executed_at=datetime.fromisoformat(executed_at),
                    fees=fees,
                    plan_id=plan_id or None,
                    external_id=external_id or None,
                )
                session.commit()
            except Exception as exc:
                session.rollback()
                return templates.TemplateResponse(
                    request,
                    "positions.html",
                    base_context(request, "positions", error=str(exc)),
                    status_code=400,
                )
        return RedirectResponse("/positions", status_code=303)

    @app.post("/positions/import")
    def import_form(
        request: Request,
        background: BackgroundTasks,
        kind: str = Form(...),
        file: UploadFile = File(...),
    ):
        if not logged_in(request):
            return RedirectResponse("/login", status_code=303)
        target = BACKEND_ROOT / "data" / "imports" / f"{secrets.token_hex(8)}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.file.read())
        with next(get_db()) as session:
            service = account_service(session)
            try:
                result = import_csv(
                    session=session,
                    service=service,
                    config=config,
                    kind=kind,
                    path=target,
                )
                session.commit()
            except Exception as exc:
                session.rollback()
                return templates.TemplateResponse(
                    request,
                    "positions.html",
                    base_context(request, "positions", error=str(exc)),
                    status_code=400,
                )
        return RedirectResponse(
            f"/positions?imported={result.imported}&duplicates={result.duplicates}",
            status_code=303,
        )

    @app.post("/settings/risk")
    def settings_risk(
        request: Request,
        equity: float = Form(...),
        cash: float = Form(...),
        max_drawdown: float = Form(...),
        mode: str = Form("real"),
    ):
        if not logged_in(request):
            return RedirectResponse("/login", status_code=303)
        with next(get_db()) as session:
            service = account_service(session)
            service.configure_risk(
                equity=equity,
                cash=cash,
                max_drawdown=max_drawdown,
                risk_profile={"mode": mode},
            )
            row = AccountRepository.get(session)
            if row:
                row.mode = mode
            AuditRepository.log(
                session,
                actor="user",
                action="settings_risk",
                target="default",
                after={"equity": equity, "max_drawdown": max_drawdown, "mode": mode},
            )
            session.commit()
        return RedirectResponse("/settings", status_code=303)

    @app.post("/settings/demo")
    def settings_demo(request: Request, enabled: str = Form("0")):
        if not logged_in(request):
            return RedirectResponse("/login", status_code=303)
        request.session["demo"] = enabled == "1"
        return RedirectResponse("/settings", status_code=303)

    @app.post("/settings/events/import")
    def settings_events_import(request: Request, file: UploadFile = File(...)):
        if not logged_in(request):
            return RedirectResponse("/login", status_code=303)
        suffix = Path(file.filename or "events.csv").suffix or ".csv"
        target = BACKEND_ROOT / "data" / "imports" / f"events-{secrets.token_hex(8)}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.file.read())
        with next(get_db()) as session:
            try:
                EventIngestor(session).ingest_manual_file(target)
                session.commit()
            except Exception:
                session.rollback()
                raise
        return RedirectResponse("/settings", status_code=303)

    # ---- JSON API -----------------------------------------------------
    @app.get("/api/health")
    def health(request: Request):
        day = datetime.now(SHANGHAI).date()
        try:
            with next(get_db()) as session:
                report = DataHealthService(provider).check(day, session)
            data = report.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - health must answer
            data = {"ok": False, "issues": [f"health check failed: {exc}"]}
        return {
            "status": "ok" if data.get("ok") else "degraded",
            "time": datetime.now(SHANGHAI).isoformat(),
            "scheduler": bool(getattr(app.state.scheduler.scheduler, "running", False)),
            "data": data,
        }

    @app.get("/api/data/status")
    def data_status(request: Request):
        api_guard(request)
        with next(get_db()) as session:
            jobs = JobRepository.list_recent(session, limit=5)
            event_sync = SettingRepository.get(session, "event_sync", {})
            data_update = (
                ((jobs[0].detail_json or {}).get("data_update"))
                if jobs
                else None
            )
        return {
            "latest_processed": _latest_partition(store.processed_dir / "bars"),
            "latest_raw": _latest_manifest(store),
            "event_sync": event_sync,
            "data_update": data_update,
            "jobs": [
                {
                    "id": job.id,
                    "job_name": job.job_name,
                    "status": job.status,
                    "trade_date": job.trade_date.isoformat() if job.trade_date else None,
                    "error": job.error,
                }
                for job in jobs
            ],
        }

    @app.get("/api/regime")
    def api_regime(request: Request):
        api_guard(request)
        with next(get_db()) as session:
            return _regime_from_snapshot(session) or {}

    @app.get("/api/candidates")
    def api_candidates(request: Request):
        api_guard(request)
        with next(get_db()) as session:
            snapshots = SnapshotRepository.list_recent(session, "default", limit=1)
            detail = (snapshots[-1].detail_json or {}) if snapshots else {}
        return {
            "candidates": detail.get("candidates", []),
            "observations": detail.get("observations", []),
        }

    @app.get("/api/plans")
    def api_plans(request: Request, trade_date: date | None = None, status: str | None = None):
        api_guard(request)
        with next(get_db()) as session:
            query = select(PlanRow).order_by(PlanRow.created_at.desc()).limit(200)
            if trade_date:
                query = query.where(PlanRow.trade_date == trade_date)
            if status:
                query = query.where(PlanRow.status == status)
            rows = session.scalars(query).all()
            return [row.plan_json for row in rows]

    @app.get("/api/positions")
    def api_positions(request: Request):
        api_guard(request)
        with next(get_db()) as session:
            state = AccountRepository.to_state(session)
            latest = _latest_partition(store.processed_dir / "bars")
            try:
                value_date = (
                    date.fromisoformat(latest)
                    if latest
                    else datetime.now(SHANGHAI).date()
                )
            except ValueError:
                value_date = datetime.now(SHANGHAI).date()
            decisions = PositionDecisionService(
                session, config, provider=provider
            ).evaluate_all(
                value_date,
                session_type=ExitSession.DAILY,
                persist=False,
                notify=False,
            )
            payload = state.model_dump(mode="json")
            payload["positions_value"] = str(state.positions_value)
            payload["heat"] = state.heat
            payload["drawdown_pct"] = state.drawdown_pct
            payload["decisions"] = [
                decision.model_dump(mode="json") for decision in decisions
            ]
            payload["valuation"] = {
                "incomplete": state.valuation_incomplete,
                "note": state.valuation_note,
                "last_valuation_at": state.last_valuation_at.isoformat()
                if state.last_valuation_at
                else None,
                "last_reconciled_at": state.last_reconciled_at.isoformat()
                if state.last_reconciled_at
                else None,
                "quote_freshness": (
                    "unknown"
                    if not decisions
                    else (
                        "partial"
                        if any(
                            item.valuation_status.value != "ok"
                            for item in decisions
                        )
                        else "ok"
                    )
                ),
            }
            return payload

    @app.get("/api/positions/decisions")
    def api_position_decisions(request: Request):
        api_guard(request)
        with next(get_db()) as session:
            latest = _latest_partition(store.processed_dir / "bars")
            try:
                value_date = (
                    date.fromisoformat(latest)
                    if latest
                    else datetime.now(SHANGHAI).date()
                )
            except ValueError:
                value_date = datetime.now(SHANGHAI).date()
            decisions = PositionDecisionService(
                session, config, provider=provider
            ).evaluate_all(
                value_date,
                session_type=ExitSession.DAILY,
                persist=False,
                notify=False,
            )
            return {
                "data_time": latest,
                "quote_freshness": (
                    "unknown"
                    if not decisions
                    else (
                        "partial"
                        if any(
                            item.valuation_status.value != "ok"
                            for item in decisions
                        )
                        else "ok"
                    )
                ),
                "decisions": [
                    decision.model_dump(mode="json")
                    for decision in decisions
                ],
            }

    @app.get("/api/executions")
    def api_executions(request: Request):
        api_guard(request)
        with next(get_db()) as session:
            return [
                {
                    "id": row.id,
                    "ts_code": row.ts_code,
                    "side": row.side,
                    "price": str(row.price),
                    "shares": row.shares,
                    "fees": str(row.fees),
                    "executed_at": row.executed_at.isoformat(),
                    "plan_id": row.plan_id,
                    "source": row.source,
                }
                for row in ExecutionRepository.list_for_account(session)
            ]

    @app.get("/api/notifications")
    def api_notifications(request: Request, unacknowledged: int = 0, limit: int = 100):
        api_guard(request)
        with next(get_db()) as session:
            items = NotificationRepository.list_recent(
                session, unacknowledged_only=bool(unacknowledged), limit=limit
            )
            count = _notification_count(session)
        return {
            "items": [
                {
                    "id": item.id,
                    "level": item.level,
                    "code": item.code,
                    "title": item.title,
                    "message": item.message,
                    "trade_date": item.trade_date.isoformat() if item.trade_date else None,
                    "acknowledged_at": item.acknowledged_at.isoformat()
                    if item.acknowledged_at
                    else None,
                }
                for item in items
            ],
            "unacknowledged": count,
        }

    @app.post("/api/notifications/{notification_id}/ack")
    def api_ack(request: Request, notification_id: str):
        if not logged_in(request):
            raise HTTPException(401, "login required")
        with next(get_db()) as session:
            NotificationRepository.acknowledge(session, notification_id)
            session.commit()
        return {"ok": True}

    @app.post("/api/plans/{plan_id}/confirm")
    def api_confirm(
        request: Request, plan_id: str, redirect: str | None = Form(None)
    ):
        if not logged_in(request):
            raise HTTPException(401, "login required")
        with next(get_db()) as session:
            plan = PlanRepository.get(session, plan_id)
            if plan is None:
                raise HTTPException(404, "plan not found")
            if plan.status not in ("PENDING_EXECUTION", "PENDING_ENTRY"):
                raise HTTPException(409, "plan is not awaiting confirmation")
            payload = dict(plan.plan_json or {})
            reasons = list(payload.get("reasons") or [])
            if "用户已确认计划" not in reasons:
                reasons.append("用户已确认计划")
            payload["reasons"] = reasons
            plan.plan_json = payload
            AuditRepository.log(
                session,
                actor="user",
                action="confirm_plan",
                target=plan_id,
                after={"status": plan.status, "confirmed": True},
            )
            session.commit()
        if redirect:
            return RedirectResponse(redirect, status_code=303)
        return {"ok": True, "plan_id": plan_id}

    @app.post("/api/plans/{plan_id}/cancel")
    def api_cancel(
        request: Request, plan_id: str, redirect: str | None = Form(None)
    ):
        if not logged_in(request):
            raise HTTPException(401, "login required")
        with next(get_db()) as session:
            plan = PlanRepository.update_status(
                session, plan_id, PlanStatus.CANCELLED, "user cancelled"
            )
            AuditRepository.log(
                session,
                actor="user",
                action="cancel_plan",
                target=plan_id,
                after={"status": plan.status},
            )
            session.commit()
        if redirect:
            return RedirectResponse(redirect, status_code=303)
        return {"ok": True}

    @app.post("/api/jobs/run")
    def api_run_job(
        background: BackgroundTasks,
        request: Request,
        trade_date: date | None = Form(None),
        strategy_mode: str = Form("all"),
        force: bool = Form(False),
        redirect: str | None = Form(None),
    ):
        if not logged_in(request):
            raise HTTPException(401, "login required")
        target = trade_date or datetime.now(SHANGHAI).date()
        background.add_task(
            daily.run,
            target,
            strategy_mode=strategy_mode,
            force=force,
            run_data_update=True,
        )
        if redirect:
            return RedirectResponse(redirect, status_code=303)
        return {"queued": True, "trade_date": target.isoformat()}

    @app.get("/api/events/stream")
    async def events_stream(request: Request):
        if not logged_in(request):
            raise HTTPException(401, "login required")

        def snapshot() -> dict:
            with next(get_db()) as session:
                count = _notification_count(session)
                jobs = JobRepository.list_recent(session, limit=1)
                notifications = NotificationRepository.list_recent(
                    session, unacknowledged_only=True, limit=5
                )
                latest = jobs[0] if jobs else None
                state = AccountRepository.to_state(session)
                latest_processed = _latest_partition(store.processed_dir / "bars")
                try:
                    value_date = (
                        date.fromisoformat(latest_processed)
                        if latest_processed
                        else datetime.now(SHANGHAI).date()
                    )
                except ValueError:
                    value_date = datetime.now(SHANGHAI).date()
                decisions = PositionDecisionService(
                    session, config, provider=provider
                ).evaluate_all(
                    value_date,
                    session_type=ExitSession.DAILY,
                    persist=False,
                    notify=False,
                )
                ordered = sorted(
                    decisions,
                    key=lambda item: (
                        0
                        if item.action.value in RISK_ACTIONS
                        else 1,
                        item.ts_code,
                    ),
                )
                quote_freshness = (
                    "unknown"
                    if not decisions
                    else (
                        "partial"
                        if any(
                            item.valuation_status.value != "ok"
                            for item in decisions
                        )
                        else "ok"
                    )
                )
                state_prices = {
                    item.ts_code: float(item.current_price or 0)
                    for item in state.positions
                }
                positions_value = sum(
                    float(
                        item.trigger_price
                        or state_prices.get(item.ts_code, 0)
                    )
                    * item.total_shares
                    for item in decisions
                )
                cash_value = float(state.cash)
                return {
                    "unacknowledged": count,
                    "equity": round(cash_value + positions_value, 2),
                    "cash": round(cash_value, 2),
                    "positions_value": round(positions_value, 2),
                    "latest_job": latest.status if latest else None,
                    "latest_job_name": latest.job_name if latest else None,
                    "latest_job_date": latest.trade_date.isoformat()
                    if latest and latest.trade_date
                    else None,
                    "latest_job_error": latest.error if latest else None,
                    "data_time": latest_processed,
                    "quote_freshness": quote_freshness,
                    "valuation_incomplete": state.valuation_incomplete,
                    "last_valuation_at": state.last_valuation_at.isoformat()
                    if state.last_valuation_at
                    else None,
                    "last_reconciled_at": state.last_reconciled_at.isoformat()
                    if state.last_reconciled_at
                    else None,
                    "positions": [
                        {
                            "ts_code": item.ts_code,
                            "strategy": (
                                item.primary_strategy.value
                                if item.primary_strategy
                                else None
                            ),
                            "sub_mode": (
                                item.sub_mode.value if item.sub_mode else None
                            ),
                            "shares": item.total_shares,
                            "available_shares": item.available_shares,
                            "current_price": str(
                                item.trigger_price
                                or state_prices.get(item.ts_code, 0.0)
                            ),
                            "current_stop": str(item.current_stop or ""),
                            "proposed_stop": str(item.proposed_stop or ""),
                            "action": item.action.value,
                            "action_label": ACTION_LABELS.get(
                                item.action.value, item.action.value
                            ),
                            "reason_code": item.reason_code,
                            "explanation": item.explanation,
                            "suggested_shares": item.suggested_shares,
                            "pending_shares": item.pending_shares,
                            "executed_shares": item.executed_shares,
                            "data_time": item.data_time.isoformat()
                            if item.data_time
                            else None,
                            "valuation_status": item.valuation_status.value,
                            "risk": item.action.value in RISK_ACTIONS,
                        }
                        for item in ordered
                    ],
                    "notifications": [
                        {
                            "id": item.id,
                            "level": item.level,
                            "title": item.title,
                            "message": item.message,
                        }
                        for item in notifications
                    ],
                }

        async def generator():
            while True:
                payload = await asyncio.to_thread(snapshot)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(config.web.sse_heartbeat_seconds)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/backtests")
    def api_backtests(request: Request):
        api_guard(request)
        with next(get_db()) as session:
            rows = BacktestRepository.list_recent(session)
            return [
                {
                    "id": row.id,
                    "period_start": row.period_start.isoformat(),
                    "period_end": row.period_end.isoformat(),
                    "strategy_mode": row.strategy_mode,
                    "status": row.status,
                    "metrics": row.metrics_json,
                    "is_demo": row.is_demo,
                }
                for row in rows
            ]

    @app.get("/api/backtests/{run_id}")
    def api_backtest(request: Request, run_id: str):
        api_guard(request)
        with next(get_db()) as session:
            row = BacktestRepository.get(session, run_id)
            if row is None:
                raise HTTPException(404, "backtest not found")
            return {
                "id": row.id,
                "status": row.status,
                "metrics": row.metrics_json,
                "artifacts": row.artifacts_json,
                "is_demo": row.is_demo,
            }

    @app.get("/api/csv/template/{filename}")
    def csv_template(request: Request, filename: str):
        api_guard(request)
        kind = filename.removesuffix(".csv")
        if kind not in TEMPLATES:
            raise HTTPException(404, "template not found")
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(TEMPLATES[kind])
        return Response(
            buffer.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app


def _run_backtest(
    run_id: str,
    start: date,
    end: date,
    strategy_mode: str,
    initial_equity: float,
    is_demo: bool,
) -> None:
    config = load_full_config()
    store = _create_store()
    provider = ParquetMarketDataProvider(store)
    with next(get_db()) as session:
        row = BacktestRepository.get(session, run_id)
        if row is None:
            return
    try:
        result = BacktestEngine(provider=provider, config=config).run(
            start,
            end,
            initial_equity=initial_equity,
            strategy_mode=strategy_mode,
            is_demo=is_demo,
        )
        payload = result.model_dump(mode="json")
        metrics = {
            "initial_equity": result.initial_equity,
            "final_equity": result.final_equity,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "calmar": result.calmar,
            "win_rate": result.win_rate,
            "avg_r": result.avg_r,
            "expectancy": result.expectancy,
            "profit_factor": result.profit_factor,
            "turnover": result.turnover,
            "fill_rate": result.fill_rate,
            "avg_holding_days": result.avg_holding_days,
            "capital_utilization": result.capital_utilization,
            "benchmark_return": result.benchmark_return,
            "by_strategy": result.by_strategy,
            "by_sub_mode": result.by_sub_mode,
        }
        artifacts = {
            "trades": payload["trades"],
            "equity_curve": payload["equity_curve"],
            "missing_data": result.missing_data,
            "notes": result.notes,
        }
        with next(get_db()) as session:
            row = BacktestRepository.get(session, run_id)
            if row:
                row.status = "succeeded"
                row.metrics_json = metrics
                row.artifacts_json = artifacts
                session.commit()
    except Exception as exc:  # noqa: BLE001
        with next(get_db()) as session:
            row = BacktestRepository.get(session, run_id)
            if row:
                row.status = "failed"
                row.metrics_json = {"error": str(exc)}
                session.commit()


# Run with ``uvicorn --factory app.api.app:create_app`` so importing this
# module never requires credentials or touches the default database.

