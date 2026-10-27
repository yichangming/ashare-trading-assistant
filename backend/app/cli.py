from __future__ import annotations

import os
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import polars as pl
import typer

from app.core.config import BACKEND_ROOT, load_app_config, load_strategy_config
from app.core.dates import parse_date
from app.data.assemble import DataAssembler
from app.data.provider import ParquetMarketDataProvider
from app.data.quality import DataQualityGate, QualityReport
from app.data.store import ParquetStore
from app.data.sync import SyncService
from app.data.tushare_client import TushareClient
from app.domain.enums import PlanStatus
from app.domain.models import AccountState, PendingExposure, TradePlan
from app.pipeline import TradingPipeline

app = typer.Typer(help="A-share data and dual-template planning CLI")


def _store() -> ParquetStore:
    config = load_app_config()
    data = config.data
    return ParquetStore(
        BACKEND_ROOT / data.raw_dir,
        BACKEND_ROOT / data.processed_dir,
        BACKEND_ROOT / data.regime_dir,
    )


def _sync_service() -> SyncService:
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
    clears it once drawdown falls below the recovery threshold.
    """
    return account.model_copy(update={"trading_halted": False})


def merge_pending_plans(
    account: AccountState, plans: list[TradePlan]
) -> AccountState:
    """Add approved same-day plans to ``account.pending_plans``.

    Plans whose status is not ``PENDING_ENTRY`` are ignored, and a code that
    already has a pending exposure is not added twice.
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
            )
        )
        seen.add(code)
    return merged


def _full_day_suspended_codes(suspend: pl.DataFrame) -> set[str]:
    if suspend.is_empty() or "ts_code" not in suspend.columns:
        return set()
    suspended = suspend.drop_nulls(subset=["ts_code"])
    if "suspend_timing" not in suspended.columns:
        return set(suspended["ts_code"].to_list())
    timing = (
        pl.col("suspend_timing").cast(pl.String, strict=False).str.strip_chars()
    )
    full_day = suspended.filter(
        pl.col("suspend_timing").is_null() | (timing == "")
    )
    return set(full_day["ts_code"].to_list())


def quality_report(store: ParquetStore, trade_date: date) -> QualityReport:
    """Run the suspension-aware bar quality gate for one day.

    A full-day suspension (null/blank ``suspend_timing``) is removed from the
    required-code set, while every listed code remains allowed.  When no
    ``suspend_d`` partition exists at all, fall back to count-based coverage
    against the full listed universe, as the pipeline does.
    """
    provider = ParquetMarketDataProvider(store)
    bars = provider.load_bars(trade_date, trade_date)
    stocks = provider.load_stock_basic(trade_date)
    stock_codes = (
        set(stocks["ts_code"].to_list())
        if not stocks.is_empty() and "ts_code" in stocks.columns
        else set()
    )
    suspend = provider.load_suspend(trade_date, trade_date)
    if suspend is None:
        report = DataQualityGate().check(
            trade_date, bars, expected_count=stocks.height
        )
        report.warnings.append(
            "suspend_d unavailable; using count-based coverage"
        )
        report.warning_codes.append("suspend_d_missing")
        return report
    required_codes = stock_codes - _full_day_suspended_codes(suspend)
    if not required_codes:
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
    return DataQualityGate().check(
        trade_date,
        bars,
        expected_count=len(required_codes),
        expected_codes=required_codes,
        allowed_codes=stock_codes,
    )


@app.command()
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


@app.command()
def calendar(
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Refresh and merge reference data, including the forward trade_cal."""
    service = _sync_service()
    counts = service.sync_reference(parse_date(start), parse_date(end))
    rendered = ", ".join(f"{name}={count}" for name, count in counts.items())
    typer.echo(f"reference data refreshed: {rendered}")
    for warning in service.reference_degraded:
        typer.echo(f"WARN: {warning}", err=True)


@app.command()
def assemble(day: str = typer.Option(..., help="YYYY-MM-DD")) -> None:
    trade_date = parse_date(day)
    bars = DataAssembler(_store()).build_day(trade_date)
    typer.echo(
        f"assembled {bars.height} rows for {trade_date.isoformat()}"
    )


@app.command()
def assemble_range(
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Assemble every open day in ``[start, end]``.

    Open days are read from the persisted ``trade_cal`` and are never
    extended beyond ``end``.  Open days without raw daily data are skipped
    and reported; critical assembly failures are reported and exit non-zero.
    """
    start_date = parse_date(start)
    end_date = parse_date(end)
    if start_date > end_date:
        raise typer.BadParameter("start must be on or before end")
    store = _store()
    if store.read_reference("trade_cal").is_empty():
        typer.echo(
            "trade_cal reference is missing; run the calendar command first",
            err=True,
        )
        raise typer.Exit(code=1)
    calendar_frame = ParquetMarketDataProvider(store).load_trade_cal(
        start_date, end_date
    )
    if calendar_frame.is_empty():
        typer.echo(
            f"no open trading days in {start_date}..{end_date}"
        )
        return
    if not {"cal_date", "is_open"} <= set(calendar_frame.columns):
        typer.echo(
            "trade_cal is missing cal_date/is_open; refresh it with the "
            "calendar command",
            err=True,
        )
        raise typer.Exit(code=1)
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
        if store.read_raw("daily", key).is_empty():
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
def quality(day: str = typer.Option(..., help="YYYY-MM-DD")) -> None:
    report = quality_report(_store(), parse_date(day))
    typer.echo(f"passed={report.passed} coverage={report.coverage:.2%}")
    for message, code in zip(
        report.warnings, report.warning_codes, strict=True
    ):
        typer.echo(f"WARN [{code}]: {message}")
    for message, code in zip(
        report.failures, report.failure_codes, strict=True
    ):
        typer.echo(f"FAIL [{code}]: {message}")
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def run(
    day: str = typer.Option(..., help="YYYY-MM-DD"),
    equity: float = typer.Option(1_000_000.0),
    cash: float = typer.Option(1_000_000.0),
    peak_equity: float | None = typer.Option(None),
    template_mode: str = typer.Option("dual"),
    account_file: str | None = typer.Option(None),
    restart: bool = typer.Option(False),
    output: str | None = typer.Option(None),
) -> None:
    """Run the planning pipeline and persist account state.

    The account is loaded from ``--account-file`` when it exists, otherwise
    built from ``--equity``/``--cash``/``--peak-equity``.  ``--restart``
    manually clears a latched stop before the run and therefore requires an
    existing account file.  Approved pending plans are merged into the
    returned account and saved back atomically.
    """
    trade_date = parse_date(day)
    account_path = Path(account_file) if account_file else None
    if restart and (account_path is None or not account_path.is_file()):
        raise typer.BadParameter(
            "--restart requires an existing --account-file"
        )
    try:
        account = build_account(account_path, equity, cash, peak_equity)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            f"cannot load account state: {exc}"
        ) from exc
    if restart:
        account = restart_account(account)

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


if __name__ == "__main__":
    app()
