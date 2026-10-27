from __future__ import annotations

import functools
import os
import tempfile
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import ParamSpec, TypeVar
from zoneinfo import ZoneInfo

import polars as pl
import typer
from dotenv import load_dotenv

from app.core.config import BACKEND_ROOT, load_app_config, load_strategy_config
from app.core.dates import parse_date
from app.core.errors import DataNotReadyError, DataQualityError
from app.data.assemble import DataAssembler
from app.data.provider import ParquetMarketDataProvider
from app.data.quality import (
    DataQualityGate,
    QualityReport,
    resolve_quality_universe,
)
from app.data.store import ParquetStore
from app.data.sync import SyncError, SyncService
from app.data.tushare_client import TushareClient
from app.domain.enums import PlanStatus
from app.domain.models import AccountState, PendingExposure, TradePlan
from app.pipeline import TradingPipeline

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_MISSING_SUSPEND_WARNING = (
    "suspend_d unavailable; using count-based coverage"
)

P = ParamSpec("P")
R = TypeVar("R")

app = typer.Typer(
    help="A-share data and dual-template planning CLI",
    rich_markup_mode=None,
)


def _command_errors(func: Callable[P, R]) -> Callable[P, R]:
    """Turn expected data/argument failures into clean CLI errors."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (
            DataNotReadyError,
            DataQualityError,
            SyncError,
            ValueError,
            OSError,
        ) as exc:
            typer.echo(f"ERROR: {exc}", err=True)
            raise typer.Exit(code=1) from None

    return wrapper


def _today() -> date:
    return datetime.now(_SHANGHAI).date()


def _store() -> ParquetStore:
    config = load_app_config()
    data = config.data
    return ParquetStore(
        BACKEND_ROOT / data.raw_dir,
        BACKEND_ROOT / data.processed_dir,
        BACKEND_ROOT / data.regime_dir,
    )


def _load_env() -> None:
    """Load repo-root and backend ``.env`` files without overriding exports."""
    for path in (BACKEND_ROOT.parent / ".env", BACKEND_ROOT / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)


def _sync_service() -> SyncService:
    _load_env()
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise typer.BadParameter("TUSHARE_TOKEN is not set")
    config = load_app_config()
    return SyncService(
        TushareClient(token, rate_limit_per_min=config.data.rate_limit_per_min),
        _store(),
    )


def load_account(path: Path) -> AccountState:
    return AccountState.model_validate_json(path.read_text(encoding="utf-8"))


def save_account(path: Path, account: AccountState) -> None:
    """Write ``account`` to ``path`` atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            account.model_dump_json(indent=2), encoding="utf-8"
        )
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def build_account(
    account_file: Path | None,
    equity: float,
    cash: float,
    peak_equity: float | None,
) -> AccountState:
    """Load persisted state when present, otherwise build it from CLI args.

    ``peak_equity`` defaults to ``equity`` (there is no prior peak without a
    persisted account) and is never allowed below ``equity``.
    """
    if account_file is not None and account_file.exists():
        return load_account(account_file)
    equity_value = Decimal(str(equity))
    cash_value = Decimal(str(cash))
    peak_value = (
        Decimal(str(peak_equity)) if peak_equity is not None else equity_value
    )
    return AccountState(
        equity=equity_value,
        cash=cash_value,
        peak_equity=max(equity_value, peak_value),
    )


def restart_account(account: AccountState) -> AccountState:
    """Manually clear the drawdown stop latch.

    The half-risk latch is intentionally preserved; the next pipeline run
    clears it once drawdown falls below the recovery threshold.  Clearing
    ``trading_halted`` is not permanent: the pipeline re-latches it when
    drawdown is still at or above the configured stop threshold, which is
    the intended risk policy rather than a failed restart.
    """
    return account.model_copy(update={"trading_halted": False})


def merge_pending_plans(
    account: AccountState, plans: list[TradePlan]
) -> AccountState:
    """Add approved same-day plans to ``account.pending_plans``.

    Plans whose status is not ``PENDING_ENTRY`` are ignored, and a code that
    already has a pending exposure is not added twice.  Each exposure carries
    the plan's ``expires_on`` so stale entries can be pruned on later runs.
    """
    seen = {exposure.ts_code for exposure in account.pending_plans}
    merged = account
    for plan in plans:
        if plan.status != PlanStatus.PENDING_ENTRY:
            continue
        code = plan.candidate.ts_code
        if code in seen:
            continue
        merged = merged.with_pending(
            PendingExposure(
                ts_code=code,
                sector=plan.candidate.sector,
                entry_price=plan.entry_plan.entry_price,
                stop_price=plan.stop_plan.initial_stop,
                shares=plan.position_plan.shares,
                template=plan.candidate.template,
                expires_on=plan.expires_on,
            )
        )
        seen.add(code)
    return merged


def prune_expired_pending(
    account: AccountState, as_of: date
) -> tuple[AccountState, int]:
    """Drop pending plans that expired before ``as_of``.

    Entries without an ``expires_on`` are retained; they predate expiry
    tracking or were created without a plan horizon.
    """
    kept = [
        exposure
        for exposure in account.pending_plans
        if exposure.expires_on is None or exposure.expires_on >= as_of
    ]
    removed = len(account.pending_plans) - len(kept)
    if not removed:
        return account, 0
    return account.model_copy(update={"pending_plans": kept}), removed


def _warn_authoritative_account(
    account: AccountState,
    path: Path,
    equity: float,
    cash: float,
    peak_equity: float | None,
    explicit: set[str],
) -> None:
    differing: list[str] = []
    if "equity" in explicit and Decimal(str(equity)) != account.equity:
        differing.append("--equity")
    if "cash" in explicit and Decimal(str(cash)) != account.cash:
        differing.append("--cash")
    if (
        "peak_equity" in explicit
        and peak_equity is not None
        and Decimal(str(peak_equity)) != account.peak_equity
    ):
        differing.append("--peak-equity")
    if differing:
        typer.echo(
            f"WARN: account file {path} is authoritative; ignoring "
            f"{', '.join(differing)}",
            err=True,
        )


def quality_report(store: ParquetStore, trade_date: date) -> QualityReport:
    """Run the suspension-aware bar quality gate for one day.

    The required/allowed universe comes from the shared
    :func:`app.data.quality.resolve_quality_universe` helper, which is also
    used by the pipeline, so the CLI and the real run cannot drift.
    """
    provider = ParquetMarketDataProvider(store)
    bars = provider.load_bars(trade_date, trade_date)
    stocks = provider.load_stock_basic(trade_date)
    suspend = provider.load_suspend(trade_date, trade_date)
    required_codes, allowed_codes, expected_count, warnings = (
        resolve_quality_universe(stocks, suspend)
    )
    if required_codes is not None and not required_codes:
        return QualityReport(
            trade_date=trade_date,
            passed=False,
            coverage=0.0,
            failures=[
                "no required stocks: stock_basic is empty or every listed "
                "stock is fully suspended"
            ],
            failure_codes=["required_universe_empty"],
        )
    report = DataQualityGate().check(
        trade_date,
        bars,
        expected_count=expected_count,
        expected_codes=required_codes,
        allowed_codes=allowed_codes,
    )
    for code in warnings:
        report.warning_codes.append(code)
        report.warnings.append(
            _MISSING_SUSPEND_WARNING if code == "suspend_d_missing" else code
        )
    return report


def _calendar_bounds(store: ParquetStore) -> tuple[date | None, date | None]:
    reference = store.read_reference("trade_cal")
    if reference.is_empty() or "cal_date" not in reference.columns:
        return None, None
    calendar_frame = ParquetMarketDataProvider(store).load_trade_cal(
        date.min, date.max
    )
    if calendar_frame.is_empty():
        return None, None
    cal_dates = calendar_frame["cal_date"].drop_nulls()
    if cal_dates.is_empty():
        return None, None
    minimum = cal_dates.min()
    maximum = cal_dates.max()
    if not isinstance(minimum, date) or not isinstance(maximum, date):
        return None, None
    return minimum, maximum


@app.command()
@_command_errors
def sync(
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
    force: bool = typer.Option(False),
) -> None:
    service = _sync_service()
    results = service.backfill(
        parse_date(start), parse_date(end), force=force
    )
    typer.echo(f"synced {len(results)} trading days")
    for warning in dict.fromkeys(service.reference_degraded):
        typer.echo(f"WARN: {warning}", err=True)
    for result in results:
        for warning in result.degraded:
            typer.echo(
                f"WARN {result.trade_date.isoformat()}: {warning}",
                err=True,
            )


@app.command()
@_command_errors
def calendar(
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Refresh and merge reference data, including the forward trade_cal."""
    start_date = parse_date(start)
    end_date = parse_date(end)
    service = _sync_service()
    counts = service.sync_reference(start_date, end_date)
    rendered = ", ".join(f"{name}={count}" for name, count in counts.items())
    typer.echo(f"reference data refreshed: {rendered}")
    for warning in service.reference_degraded:
        typer.echo(f"WARN: {warning}", err=True)
    cal_min, cal_max = _calendar_bounds(service.store)
    if cal_min is None or cal_min > start_date:
        rendered_min = cal_min.isoformat() if cal_min is not None else "unknown"
        typer.echo(
            f"WARN: trade_cal starts at {rendered_min}, after requested start "
            f"{start_date.isoformat()}; sync/backfill an earlier window",
            err=True,
        )
        raise typer.Exit(code=1)
    if cal_max is None or cal_max < end_date:
        rendered_max = cal_max.isoformat() if cal_max is not None else "unknown"
        typer.echo(
            f"WARN: trade_cal ends at {rendered_max}, before requested end "
            f"{end_date.isoformat()}; sync/backfill a later window",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
@_command_errors
def assemble(day: str = typer.Option(..., help="YYYY-MM-DD")) -> None:
    trade_date = parse_date(day)
    bars = DataAssembler(_store()).build_day(trade_date)
    typer.echo(
        f"assembled {bars.height} rows for {trade_date.isoformat()}"
    )


@app.command()
@_command_errors
def assemble_range(
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Assemble every open day between --start and --end.

    Open days are read from the persisted trade_cal and are never extended
    beyond --end or beyond today (Asia/Shanghai), so future dates are not
    visited.  The calendar must cover both requested boundaries; open days
    without raw daily data are skipped and reported, while critical assembly
    failures are reported and exit non-zero.
    """
    start_date = parse_date(start)
    end_date = parse_date(end)
    if start_date > end_date:
        raise typer.BadParameter("start must be on or before end")
    store = _store()
    reference = store.read_reference("trade_cal")
    if reference.is_empty():
        typer.echo(
            "trade_cal reference is missing; run the calendar command first",
            err=True,
        )
        raise typer.Exit(code=1)
    if not {"cal_date", "is_open"} <= set(reference.columns):
        typer.echo(
            "trade_cal is missing cal_date/is_open; refresh it with the "
            "calendar command",
            err=True,
        )
        raise typer.Exit(code=1)

    today = _today()
    effective_end = min(end_date, today)
    if start_date > effective_end:
        typer.echo(
            f"requested range {start_date.isoformat()}..{end_date.isoformat()} "
            f"is entirely in the future (today is {today.isoformat()}); "
            "nothing to assemble",
            err=True,
        )
        raise typer.Exit(code=1)
    cal_min, cal_max = _calendar_bounds(store)
    if cal_min is None or cal_min > start_date:
        rendered_min = cal_min.isoformat() if cal_min is not None else "unknown"
        typer.echo(
            f"WARN: trade_cal starts at {rendered_min}, after requested start "
            f"{start_date.isoformat()}; refresh it with the calendar command",
            err=True,
        )
        raise typer.Exit(code=1)
    if cal_max is None or cal_max < effective_end:
        rendered_max = cal_max.isoformat() if cal_max is not None else "unknown"
        typer.echo(
            f"WARN: trade_cal ends at {rendered_max}, before requested end "
            f"{effective_end.isoformat()}; refresh it with the calendar command",
            err=True,
        )
        raise typer.Exit(code=1)

    calendar_frame = ParquetMarketDataProvider(store).load_trade_cal(
        start_date, effective_end
    )
    if calendar_frame.is_empty():
        typer.echo(
            f"no open trading days in {start_date}..{end_date}"
        )
        return
    open_dates = sorted(
        set(
            calendar_frame.filter(pl.col("is_open") == 1)["cal_date"]
            .drop_nulls()
            .to_list()
        )
    )
    if not open_dates:
        typer.echo(
            f"no open trading days in {start_date}..{end_date}"
        )
        return

    assembler = DataAssembler(store)
    assembled = 0
    row_count = 0
    skipped: list[date] = []
    failures: list[tuple[date, str]] = []
    for trade_date in open_dates:
        key = trade_date.strftime("%Y%m%d")
        if not store.raw_path("daily", key).exists():
            skipped.append(trade_date)
            continue
        try:
            bars = assembler.build_day(trade_date)
        except Exception as exc:  # keep assembling the remaining days
            failures.append((trade_date, str(exc)))
            continue
        assembled += 1
        row_count += bars.height

    typer.echo(
        f"assembled {assembled}/{len(open_dates)} open days ({row_count} rows)"
    )
    if skipped:
        rendered = ", ".join(day.isoformat() for day in skipped)
        typer.echo(
            f"skipped {len(skipped)} open days without raw daily: {rendered}"
        )
    for trade_date, message in failures:
        typer.echo(f"FAIL {trade_date.isoformat()}: {message}")
    if failures:
        raise typer.Exit(code=1)


@app.command()
@_command_errors
def quality(day: str = typer.Option(..., help="YYYY-MM-DD")) -> None:
    report = quality_report(_store(), parse_date(day))
    typer.echo(f"passed={report.passed} coverage={report.coverage:.2%}")
    for message, code in zip(
        report.warnings, report.warning_codes, strict=False
    ):
        typer.echo(f"WARN [{code}]: {message}")
    for message, code in zip(
        report.failures, report.failure_codes, strict=False
    ):
        typer.echo(f"FAIL [{code}]: {message}")
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
@_command_errors
def run(
    ctx: typer.Context,
    day: str = typer.Option(..., help="YYYY-MM-DD"),
    equity: float = typer.Option(1_000_000.0),
    cash: float = typer.Option(1_000_000.0),
    peak_equity: float | None = typer.Option(None),
    template_mode: str = typer.Option("dual"),
    account_file: str | None = typer.Option(None),
    restart: bool = typer.Option(False),
    init_account: bool = typer.Option(
        False,
        "--init-account",
        help="Create a missing --account-file from the equity/cash options",
    ),
    output: str | None = typer.Option(None),
) -> None:
    """Run the planning pipeline and persist account state.

    The account is loaded from --account-file when it exists, otherwise it
    is created from --equity/--cash/--peak-equity only when --init-account
    is passed; a missing file is an error otherwise.  --restart manually
    clears a latched stop before the run and therefore requires an existing
    account file; the pipeline re-latches trading_halted when drawdown is
    still at or above the configured stop threshold.  Expired pending plans
    are pruned on load, approved pending plans are merged into the returned
    account, and the result is saved back atomically.
    """
    trade_date = parse_date(day)
    account_path = Path(account_file) if account_file else None
    if (
        output
        and account_path is not None
        and Path(output).resolve() == account_path.resolve()
    ):
        raise ValueError(
            "--output and --account-file must be different files"
        )
    if restart and (account_path is None or not account_path.is_file()):
        raise typer.BadParameter(
            "--restart requires an existing --account-file"
        )
    if init_account and account_path is None:
        raise typer.BadParameter("--init-account requires --account-file")
    account_exists = account_path is not None and account_path.is_file()
    if account_path is not None and not account_exists and not init_account:
        typer.echo(
            f"ERROR: account file {account_path} does not exist; pass "
            "--init-account to create it from --equity/--cash/--peak-equity, "
            "or omit --account-file for an ephemeral account",
            err=True,
        )
        raise typer.Exit(code=1)

    account = build_account(account_path, equity, cash, peak_equity)
    if account_exists and account_path is not None:
        explicit = {
            name
            for name in ("equity", "cash", "peak_equity")
            if (
                (source := ctx.get_parameter_source(name)) is not None
                and source.name == "COMMANDLINE"
            )
        }
        _warn_authoritative_account(
            account,
            account_path,
            equity,
            cash,
            peak_equity,
            explicit,
        )
    if restart:
        account = restart_account(account)
    account, pruned = prune_expired_pending(account, trade_date)
    if pruned:
        typer.echo(
            f"pruned {pruned} expired pending plans", err=True
        )

    store = _store()
    manifest = store.read_manifest(trade_date)
    data_version = manifest["data_version"] if manifest is not None else None
    pipeline = TradingPipeline(
        provider=ParquetMarketDataProvider(store),
        config=load_strategy_config(),
    )
    result = pipeline.run(
        as_of=trade_date,
        account=account,
        template_mode=template_mode,
        data_version=data_version,
    )
    if account_path is not None:
        result = result.model_copy(
            update={
                "account": merge_pending_plans(
                    result.account, result.plans
                )
            }
        )
        save_account(account_path, result.account)

    payload = result.model_dump_json(indent=2)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        typer.echo(payload)

    approved = sum(
        1
        for plan in result.plans
        if plan.status == PlanStatus.PENDING_ENTRY
    )
    risk_rejected = sum(
        1 for plan in result.plans if plan.status == PlanStatus.REJECTED
    )
    typer.echo(
        f"stage={result.regime.sentiment_stage.value} "
        f"trend={result.regime.trend_regime.value} "
        f"risk={result.regime.risk_factor:.2f}",
        err=True,
    )
    typer.echo(
        f"candidates={len(result.candidates)} plans={len(result.plans)} "
        f"approved={approved} "
        f"rejected={len(result.rejected) + risk_rejected}",
        err=True,
    )
    if account_path is not None:
        typer.echo(f"account saved to {account_path}", err=True)


@app.command()
@_command_errors
def account(
    account_file: str = typer.Option(..., help="Account JSON path"),
    clear_pending: bool = typer.Option(
        False,
        "--clear-pending",
        help="Remove every pending plan from the account file",
    ),
) -> None:
    """Inspect an account file and optionally clear its pending plans."""
    path = Path(account_file)
    if not path.is_file():
        typer.echo(f"ERROR: account file {path} does not exist", err=True)
        raise typer.Exit(code=1)
    state = load_account(path)
    if clear_pending:
        cleared = len(state.pending_plans)
        state = state.model_copy(update={"pending_plans": []})
        save_account(path, state)
        typer.echo(f"cleared {cleared} pending plans")
    typer.echo(
        f"equity={state.equity} cash={state.cash} "
        f"peak_equity={state.peak_equity}"
    )
    typer.echo(
        f"positions={len(state.positions)} pending={len(state.pending_plans)} "
        f"trading_halted={state.trading_halted} "
        f"drawdown_half_latched={state.drawdown_half_latched}"
    )
    for exposure in state.pending_plans:
        typer.echo(
            f"PENDING {exposure.ts_code} template={exposure.template.value} "
            f"shares={exposure.shares} entry={exposure.entry_price} "
            f"stop={exposure.stop_price} expires_on={exposure.expires_on}"
        )


if __name__ == "__main__":
    app()
