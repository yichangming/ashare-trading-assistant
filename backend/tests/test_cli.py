from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

import app.cli as cli_module
from app.cli import (
    app,
    build_account,
    load_account,
    merge_pending_plans,
    prune_expired_pending,
    quality_report,
    restart_account,
    save_account,
)
from app.core.calendar import TradingCalendar
from app.core.config import load_strategy_config
from app.core.errors import DataNotReadyError
from app.data.provider import ParquetMarketDataProvider
from app.data.quality import resolve_quality_universe
from app.data.store import ParquetStore
from app.data.sync import SyncService
from app.data.tushare_client import TushareClient
from app.domain.enums import (
    PlanStatus,
    SentimentStage,
    Template,
    TrendRegime,
)
from app.domain.models import (
    AccountState,
    Candidate,
    MarketRegime,
    PendingExposure,
    RegimeIndicators,
    TemplatePolicy,
    TradePlan,
)
from app.pipeline import TradingPipeline
from tests.fixtures.fake_tushare import FakePro, RangeAwareFakePro

_DAY = date(2024, 1, 2)

_COMMANDS = (
    "sync",
    "calendar",
    "assemble",
    "assemble-range",
    "quality",
    "run",
    "account",
)


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "raw", tmp_path / "processed", tmp_path / "regime")


def _bars(*codes: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "ts_code": code,
                "trade_date": _DAY,
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "pre_close": 10.0,
                "vol": 100_000.0,
                "amount": 1_000_000.0,
                "adj_factor": 1.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
            for code in codes
        ],
        schema_overrides={"trade_date": pl.Date},
    )


def _write_stocks(store: ParquetStore, codes: list[str]) -> None:
    store.write_reference(
        "stock_basic",
        pl.DataFrame(
            {
                "ts_code": codes,
                "name": [f"股票{index}" for index in range(len(codes))],
                "market": ["主板"] * len(codes),
                "list_date": [date(2015, 1, 1)] * len(codes),
                "delist_date": [None] * len(codes),
            },
            schema_overrides={
                "list_date": pl.Date,
                "delist_date": pl.Date,
            },
        ),
    )


def _write_suspend(store: ParquetStore, rows: list[dict]) -> None:
    store.write_raw(
        "suspend_d",
        _DAY.strftime("%Y%m%d"),
        pl.DataFrame(
            rows,
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.String,
                "suspend_type": pl.String,
                "suspend_timing": pl.String,
            },
        ),
    )


def test_cli_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sync" in result.stdout
    assert "run" in result.stdout


@pytest.mark.parametrize("command", _COMMANDS)
def test_cli_registers_workflow_commands(command: str):
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert command in result.stdout


@pytest.mark.parametrize("timing", [None, "", "  "])
def test_quality_excludes_full_day_suspensions(
    store: ParquetStore, timing: str | None
):
    _write_stocks(store, ["000001.SZ", "000002.SZ", "000003.SZ"])
    store.write_processed("bars", _DAY, _bars("000001.SZ", "000002.SZ"))
    _write_suspend(
        store,
        [
            {
                "trade_date": _DAY,
                "ts_code": "000003.SZ",
                "suspend_type": "S",
                "suspend_timing": timing,
            }
        ],
    )

    report = quality_report(store, _DAY)

    assert report.passed, report.failures
    assert report.coverage == 1.0


def test_quality_partial_day_suspension_stays_required(store: ParquetStore):
    _write_stocks(store, ["000001.SZ", "000002.SZ", "000003.SZ"])
    store.write_processed("bars", _DAY, _bars("000001.SZ", "000002.SZ"))
    _write_suspend(
        store,
        [
            {
                "trade_date": _DAY,
                "ts_code": "000003.SZ",
                "suspend_type": "S",
                "suspend_timing": "09:30-10:30",
            }
        ],
    )

    report = quality_report(store, _DAY)

    assert not report.passed
    assert "missing_expected_codes" in report.failure_codes
    assert "000003.SZ" in "; ".join(report.failures)


def test_quality_unknown_extra_code_fails(store: ParquetStore):
    _write_stocks(store, ["000001.SZ", "000002.SZ", "000003.SZ"])
    store.write_processed(
        "bars", _DAY, _bars("000001.SZ", "000002.SZ", "000004.SZ")
    )
    _write_suspend(
        store,
        [
            {
                "trade_date": _DAY,
                "ts_code": "000003.SZ",
                "suspend_type": "S",
                "suspend_timing": None,
            }
        ],
    )

    report = quality_report(store, _DAY)

    assert not report.passed
    assert "unexpected_codes" in report.failure_codes
    assert "000004.SZ" in "; ".join(report.failures)


def test_quality_without_suspend_partition_uses_count_fallback(
    store: ParquetStore,
):
    _write_stocks(store, ["000001.SZ", "000002.SZ"])
    store.write_processed("bars", _DAY, _bars("000001.SZ", "000002.SZ"))

    report = quality_report(store, _DAY)

    assert report.passed, report.failures
    assert report.coverage == 1.0


def test_quality_fallback_flags_unexpected_code_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, store: ParquetStore
):
    _write_stocks(store, ["000001.SZ", "000002.SZ"])
    store.write_processed("bars", _DAY, _bars("000001.SZ", "000004.SZ"))
    monkeypatch.setattr(cli_module, "_store", lambda: store)

    result = CliRunner().invoke(
        cli_module.app, ["quality", "--day", "2024-01-02"]
    )

    assert result.exit_code == 1
    assert "unexpected_codes" in result.output
    assert "000004.SZ" in result.output


def test_quality_empty_bars_fails_cleanly(store: ParquetStore):
    _write_stocks(store, ["000001.SZ"])

    report = quality_report(store, _DAY)

    assert not report.passed
    assert "bars_empty" in report.failure_codes


def test_quality_report_parity_with_shared_universe(store: ParquetStore):
    _write_stocks(store, ["000001.SZ", "000002.SZ", "000003.SZ"])
    store.write_processed("bars", _DAY, _bars("000001.SZ", "000002.SZ"))
    _write_suspend(
        store,
        [
            {
                "trade_date": _DAY,
                "ts_code": "000003.SZ",
                "suspend_type": "S",
                "suspend_timing": None,
            }
        ],
    )
    provider = ParquetMarketDataProvider(store)
    required, allowed, expected_count, warnings = resolve_quality_universe(
        provider.load_stock_basic(_DAY),
        provider.load_suspend(_DAY, _DAY),
    )

    report = quality_report(store, _DAY)

    assert required == {"000001.SZ", "000002.SZ"}
    assert allowed == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert expected_count == 2
    assert warnings == []
    assert report.passed, report.failures


def test_quality_command_exits_one_on_failure(
    monkeypatch: pytest.MonkeyPatch, store: ParquetStore
):
    _write_stocks(store, ["000001.SZ", "000002.SZ"])
    store.write_processed("bars", _DAY, _bars("000001.SZ"))
    monkeypatch.setattr(cli_module, "_store", lambda: store)

    result = CliRunner().invoke(
        cli_module.app, ["quality", "--day", "2024-01-02"]
    )

    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "coverage_below_min" in result.output


def test_quality_command_exits_zero_when_passing(
    monkeypatch: pytest.MonkeyPatch, store: ParquetStore
):
    _write_stocks(store, ["000001.SZ"])
    store.write_processed("bars", _DAY, _bars("000001.SZ"))
    monkeypatch.setattr(cli_module, "_store", lambda: store)

    result = CliRunner().invoke(
        cli_module.app, ["quality", "--day", "2024-01-02"]
    )

    assert result.exit_code == 0, result.output
    assert "passed=True" in result.output


def _account(
    *,
    equity: str = "1000000",
    cash: str = "1000000",
    peak_equity: str = "1000000",
    trading_halted: bool = False,
    drawdown_half_latched: bool = False,
    pending_plans: list[PendingExposure] | None = None,
) -> AccountState:
    return AccountState(
        equity=Decimal(equity),
        cash=Decimal(cash),
        peak_equity=Decimal(peak_equity),
        trading_halted=trading_halted,
        drawdown_half_latched=drawdown_half_latched,
        pending_plans=pending_plans or [],
    )


def _pending(
    code: str = "000003.SZ",
    expires_on: date | None = None,
) -> PendingExposure:
    return PendingExposure(
        ts_code=code,
        sector="银行",
        entry_price=Decimal("10"),
        stop_price=Decimal("9"),
        shares=100,
        template=Template.A,
        expires_on=expires_on,
    )


def test_account_file_round_trip(tmp_path: Path):
    path = tmp_path / "state" / "account.json"
    pending = _pending(expires_on=date(2024, 1, 5))
    account = _account(
        trading_halted=True,
        drawdown_half_latched=True,
        pending_plans=[pending],
    )

    save_account(path, account)
    loaded = load_account(path)

    assert loaded == account
    assert loaded.pending_plans == [pending]
    assert loaded.pending_plans[0].expires_on == date(2024, 1, 5)


def test_restart_account_clears_halt_only():
    account = _account(trading_halted=True, drawdown_half_latched=True)

    restarted = restart_account(account)

    assert restarted.trading_halted is False
    assert restarted.drawdown_half_latched is True
    assert account.trading_halted is True


def test_build_account_uses_args_and_floors_peak_at_equity(tmp_path: Path):
    missing = tmp_path / "missing.json"

    account = build_account(missing, 1_000_000.0, 500_000.0, None)
    assert account.peak_equity == Decimal("1000000")

    lowered = build_account(missing, 1_000_000.0, 500_000.0, 900_000.0)
    assert lowered.peak_equity == Decimal("1000000")

    raised = build_account(missing, 1_000_000.0, 500_000.0, 1_500_000.0)
    assert raised.peak_equity == Decimal("1500000")


def test_build_account_prefers_existing_file(tmp_path: Path):
    path = tmp_path / "account.json"
    saved = _account(equity="1200000", cash="300000", peak_equity="2000000")
    save_account(path, saved)

    loaded = build_account(path, 1_000_000.0, 500_000.0, 900_000.0)

    assert loaded == saved


def _write_raw_day(store: ParquetStore, key: str) -> None:
    store.write_raw(
        "daily",
        key,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [key],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "pre_close": [10.0],
                "vol": [100_000.0],
                "amount": [1_000_000.0],
            }
        ),
    )
    store.write_raw(
        "adj_factor",
        key,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [key],
                "adj_factor": [1.0],
            }
        ),
    )
    store.write_raw(
        "daily_basic",
        key,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [key],
                "turnover_rate": [1.5],
                "circ_mv": [1_000_000.0],
            }
        ),
    )
    store.write_raw(
        "index_daily",
        key,
        pl.DataFrame(
            {
                "ts_code": ["000300.SH"],
                "trade_date": [key],
                "open": [4000.0],
                "high": [4050.0],
                "low": [3980.0],
                "close": [4020.0],
                "vol": [1000.0],
                "amount": [2000.0],
            }
        ),
    )


def _write_trade_cal(store: ParquetStore, days: list[date]) -> None:
    store.write_reference(
        "trade_cal",
        pl.DataFrame(
            {"cal_date": days, "is_open": [1] * len(days)},
            schema_overrides={"cal_date": pl.Date},
        ),
    )


def test_assemble_range_assembles_only_open_days_in_range(
    monkeypatch: pytest.MonkeyPatch, store: ParquetStore
):
    _write_stocks(store, ["000001.SZ"])
    _write_raw_day(store, "20240102")
    _write_raw_day(store, "20240105")  # raw exists but lies outside the range
    _write_trade_cal(
        store,
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
    )
    monkeypatch.setattr(cli_module, "_store", lambda: store)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "assemble-range",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-04",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "assembled 1/3" in result.output
    assert "skipped 2" in result.output
    assert not store.read_processed(
        "bars", date(2024, 1, 2), date(2024, 1, 2)
    ).is_empty()
    assert store.read_processed(
        "bars", date(2024, 1, 3), date(2024, 1, 3)
    ).is_empty()
    assert store.read_processed(
        "bars", date(2024, 1, 5), date(2024, 1, 5)
    ).is_empty()


def test_assemble_range_rejects_malformed_trade_cal(
    monkeypatch: pytest.MonkeyPatch, store: ParquetStore
):
    store.write_reference("trade_cal", pl.DataFrame({"is_open": [1]}))
    monkeypatch.setattr(cli_module, "_store", lambda: store)

    result = CliRunner().invoke(
        cli_module.app,
        ["assemble-range", "--start", "2024-01-02", "--end", "2024-01-03"],
    )

    assert result.exit_code == 1
    assert "cal_date" in result.output
    assert "KeyError" not in result.output


def test_assemble_range_does_not_visit_future_dates(
    monkeypatch: pytest.MonkeyPatch, store: ParquetStore
):
    today = date(2024, 1, 2)
    future = today + timedelta(days=1)
    _write_stocks(store, ["000001.SZ"])
    _write_raw_day(store, future.strftime("%Y%m%d"))
    _write_trade_cal(store, [today, future])
    monkeypatch.setattr(cli_module, "_store", lambda: store)
    monkeypatch.setattr(cli_module, "_today", lambda: today)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "assemble-range",
            "--start",
            today.isoformat(),
            "--end",
            future.isoformat(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "assembled 0/1" in result.output
    assert "skipped 1" in result.output
    assert store.read_processed("bars", future, future).is_empty()


def test_assemble_range_rejects_fully_future_range(
    monkeypatch: pytest.MonkeyPatch, store: ParquetStore
):
    today = date(2024, 1, 2)
    _write_trade_cal(store, [date(2024, 1, 3), date(2024, 1, 4)])
    monkeypatch.setattr(cli_module, "_store", lambda: store)
    monkeypatch.setattr(cli_module, "_today", lambda: today)

    result = CliRunner().invoke(
        cli_module.app,
        ["assemble-range", "--start", "2024-01-03", "--end", "2024-01-04"],
    )

    assert result.exit_code == 1
    assert "entirely in the future" in result.stderr
    assert "2024-01-02" in result.stderr


@pytest.mark.parametrize(
    ("calendar_days", "start", "end", "message"),
    [
        (
            [date(2024, 1, 3), date(2024, 1, 4)],
            "2024-01-02",
            "2024-01-04",
            "after requested start",
        ),
        (
            [date(2024, 1, 2), date(2024, 1, 3)],
            "2024-01-02",
            "2024-01-04",
            "before requested end",
        ),
    ],
)
def test_assemble_range_requires_calendar_coverage(
    monkeypatch: pytest.MonkeyPatch,
    store: ParquetStore,
    calendar_days: list[date],
    start: str,
    end: str,
    message: str,
):
    _write_trade_cal(store, calendar_days)
    monkeypatch.setattr(cli_module, "_store", lambda: store)

    result = CliRunner().invoke(
        cli_module.app, ["assemble-range", "--start", start, "--end", end]
    )

    assert result.exit_code == 1
    assert message in result.stderr


def _calendar() -> TradingCalendar:
    return TradingCalendar([_DAY + timedelta(days=index) for index in range(10)])


def _regime() -> MarketRegime:
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    return MarketRegime(
        trade_date=_DAY,
        trend_regime=TrendRegime.ATTACK,
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={
            Template.A: policy,
            Template.B: TemplatePolicy(
                enabled=True, mode="normal", max_positions=2
            ),
        },
        indicators=RegimeIndicators(
            limit_up_count=80,
            limit_down_count=0,
            max_board_height=5,
            promotion_rate=0.5,
            blow_rate=0.1,
            red_rate=0.6,
            limit_up_premium=0.02,
            breadth=0.6,
            index_close=4000.0,
            index_ma20=3900.0,
            index_ma60=3800.0,
        ),
        data_version="dv_test",
    )


def _candidate() -> Candidate:
    return Candidate(
        ts_code="000001.SZ",
        trade_date=_DAY,
        template=Template.A,
        setup_type="test",
        horizon="mid",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={
            "ma20_raw": 10.0,
            "ma5_raw": 10.5,
            "ma10_raw": 10.0,
            "board_height": 0.0,
            "atr14_raw": 0.5,
            "low_10d_raw": 9.5,
            "low_5d_raw": 9.7,
            "high_60d_raw": 30.0,
            "close": 10.2,
        },
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


class _StubStore:
    def __init__(self, manifest: dict | None):
        self.manifest = manifest

    def read_manifest(self, trade_date: date) -> dict | None:
        return self.manifest


class _FakePipeline:
    instances: list[_FakePipeline] = []

    def __init__(self, provider, config):
        self.config = config
        self.calls: list[dict] = []
        _FakePipeline.instances.append(self)

    def run(
        self,
        as_of: date,
        account: AccountState,
        template_mode: str = "dual",
        data_version: str | None = None,
    ):
        self.calls.append(
            {
                "as_of": as_of,
                "account": account,
                "template_mode": template_mode,
                "data_version": data_version,
            }
        )
        return TradingPipeline(provider=object(), config=self.config).run(
            as_of=as_of,
            account=account,
            template_mode=template_mode,
            regime_override=_regime(),
            candidates_override=[_candidate()],
            calendar_override=_calendar(),
        )


def _approved_plan() -> TradePlan:
    result = TradingPipeline(
        provider=object(), config=load_strategy_config()
    ).run(
        as_of=_DAY,
        account=_account(),
        regime_override=_regime(),
        candidates_override=[_candidate()],
        calendar_override=_calendar(),
    )
    return next(
        plan
        for plan in result.plans
        if plan.status == PlanStatus.PENDING_ENTRY
    )


def test_run_cli_restarts_merges_pending_and_uses_manifest_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        cli_module,
        "_store",
        lambda: _StubStore({"data_version": "dv_manifest"}),
    )
    monkeypatch.setattr(cli_module, "TradingPipeline", _FakePipeline)
    account_path = tmp_path / "account.json"
    save_account(
        account_path,
        _account(
            trading_halted=True,
            drawdown_half_latched=True,
            pending_plans=[_pending("000009.SZ", date(2023, 1, 1))],
        ),
    )
    output_path = tmp_path / "result.json"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            "--day",
            "2024-01-02",
            "--account-file",
            str(account_path),
            "--restart",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    pipeline = _FakePipeline.instances[-1]
    assert pipeline.calls[0]["data_version"] == "dv_manifest"
    assert pipeline.calls[0]["account"].trading_halted is False
    assert "pruned 1 expired pending plans" in result.stderr

    saved = load_account(account_path)
    assert saved.trading_halted is False
    assert [plan.ts_code for plan in saved.pending_plans] == ["000001.SZ"]
    assert saved.pending_plans[0].expires_on is not None

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["as_of"] == "2024-01-02"
    assert [
        pending["ts_code"] for pending in payload["account"]["pending_plans"]
    ] == ["000001.SZ"]


def test_run_restart_requires_an_account_file(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_module, "_store", lambda: _StubStore(None))

    result = CliRunner().invoke(
        cli_module.app, ["run", "--day", "2024-01-02", "--restart"]
    )

    assert result.exit_code != 0
    assert "--restart requires" in result.output


def test_invalid_date_has_clean_error():
    result = CliRunner().invoke(
        cli_module.app, ["assemble", "--day", "not-a-date"]
    )

    assert result.exit_code == 1
    assert "ERROR:" in result.stderr
    assert "Traceback" not in result.output


def test_missing_bars_has_clean_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    monkeypatch.setattr(cli_module, "_store", lambda: store)
    output_path = tmp_path / "result.json"

    result = CliRunner().invoke(
        cli_module.app,
        ["run", "--day", "2024-01-02", "--output", str(output_path)],
    )

    assert result.exit_code == 1
    assert "bars missing for 2024-01-02" in result.stderr
    assert "Traceback" not in result.output
    assert not output_path.exists()


class _ForwardCalendarPipeline:
    def __init__(self, provider, config):
        pass

    def run(self, **kwargs):
        raise DataNotReadyError(
            "trade_cal does not cover a forward window through 2024-02-16"
        )


def test_forward_calendar_error_is_clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_module, "_store", lambda: _StubStore(None))
    monkeypatch.setattr(
        cli_module, "TradingPipeline", _ForwardCalendarPipeline
    )

    result = CliRunner().invoke(
        cli_module.app, ["run", "--day", "2024-01-02"]
    )

    assert result.exit_code == 1
    assert "forward window" in result.stderr
    assert "Traceback" not in result.output


class _InvalidTemplatePipeline:
    def __init__(self, provider, config):
        pass

    def run(self, **kwargs):
        raise ValueError("unknown template_mode: bogus")


def test_invalid_template_mode_is_clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_module, "_store", lambda: _StubStore(None))
    monkeypatch.setattr(
        cli_module, "TradingPipeline", _InvalidTemplatePipeline
    )

    result = CliRunner().invoke(
        cli_module.app,
        ["run", "--day", "2024-01-02", "--template-mode", "bogus"],
    )

    assert result.exit_code == 1
    assert "unknown template_mode" in result.stderr
    assert "Traceback" not in result.output


class _RecordingClient:
    def __init__(self, token, **kwargs):
        self.token = token


def _prepare_dotenv_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    backend = tmp_path / "backend"
    backend.mkdir()
    (tmp_path / ".env").write_text(
        "TUSHARE_TOKEN=dotenv-token\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli_module, "BACKEND_ROOT", backend)
    monkeypatch.setattr(cli_module, "_store", lambda: object())
    monkeypatch.setattr(cli_module, "TushareClient", _RecordingClient)
    monkeypatch.setattr(os, "environ", dict(os.environ))


def test_sync_service_loads_token_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import os

    _prepare_dotenv_test(monkeypatch, tmp_path)
    os.environ.pop("TUSHARE_TOKEN", None)

    service = cli_module._sync_service()

    assert service.client.token == "dotenv-token"


def test_exported_token_wins_over_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import os

    _prepare_dotenv_test(monkeypatch, tmp_path)
    os.environ["TUSHARE_TOKEN"] = "exported-token"

    service = cli_module._sync_service()

    assert service.client.token == "exported-token"


def test_merge_pending_plans_sets_expiry_and_deduplicates():
    plan = _approved_plan()

    merged = merge_pending_plans(_account(), [plan])

    assert len(merged.pending_plans) == 1
    exposure = merged.pending_plans[0]
    assert exposure.ts_code == plan.candidate.ts_code
    assert exposure.sector == plan.candidate.sector
    assert exposure.entry_price == plan.entry_plan.entry_price
    assert exposure.stop_price == plan.stop_plan.initial_stop
    assert exposure.shares == plan.position_plan.shares
    assert exposure.template == plan.candidate.template
    assert exposure.expires_on == plan.expires_on

    again = merge_pending_plans(merged, [plan])
    assert again.pending_plans == merged.pending_plans


def test_prune_expired_pending_plans():
    as_of = date(2024, 1, 10)
    expired = _pending("000001.SZ", date(2024, 1, 9))
    current = _pending("000002.SZ", as_of)
    unknown = _pending("000003.SZ", None)
    account = _account(pending_plans=[expired, current, unknown])

    pruned, count = prune_expired_pending(account, as_of)

    assert count == 1
    assert [item.ts_code for item in pruned.pending_plans] == [
        "000002.SZ",
        "000003.SZ",
    ]
    assert account.pending_plans == [expired, current, unknown]


def test_account_command_clears_pending(tmp_path: Path):
    path = tmp_path / "account.json"
    save_account(
        path,
        _account(pending_plans=[_pending("000001.SZ", date(2024, 2, 1))]),
    )

    result = CliRunner().invoke(
        cli_module.app,
        ["account", "--account-file", str(path), "--clear-pending"],
    )

    assert result.exit_code == 0, result.output
    assert "cleared 1 pending plans" in result.output
    assert "pending=0" in result.output
    assert load_account(path).pending_plans == []


def test_run_requires_init_account_for_missing_file(tmp_path: Path):
    path = tmp_path / "account.json"

    result = CliRunner().invoke(
        cli_module.app,
        ["run", "--day", "2024-01-02", "--account-file", str(path)],
    )

    assert result.exit_code == 1
    assert "does not exist" in result.stderr
    assert "--init-account" in result.stderr
    assert not path.exists()


def test_run_init_account_creates_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(cli_module, "_store", lambda: _StubStore(None))
    monkeypatch.setattr(cli_module, "TradingPipeline", _FakePipeline)
    path = tmp_path / "account.json"
    output_path = tmp_path / "result.json"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            "--day",
            "2024-01-02",
            "--account-file",
            str(path),
            "--init-account",
            "--equity",
            "2500000",
            "--cash",
            "1000000",
            "--peak-equity",
            "2500000",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    saved = load_account(path)
    assert saved.equity == Decimal("2500000")
    assert saved.cash == Decimal("1000000")
    assert saved.peak_equity == Decimal("2500000")
    assert [item.ts_code for item in saved.pending_plans] == ["000001.SZ"]


def test_run_warns_when_account_file_is_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(cli_module, "_store", lambda: _StubStore(None))
    monkeypatch.setattr(cli_module, "TradingPipeline", _FakePipeline)
    path = tmp_path / "account.json"
    save_account(
        path,
        _account(
            equity="2000000", cash="1000000", peak_equity="2000000"
        ),
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            "--day",
            "2024-01-02",
            "--account-file",
            str(path),
            "--equity",
            "1000000",
            "--cash",
            "500000",
            "--peak-equity",
            "1500000",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "authoritative" in result.stderr
    assert load_account(path).equity == Decimal("2000000")


def test_run_rejects_output_equal_to_account_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(cli_module, "_store", lambda: _StubStore(None))
    monkeypatch.setattr(cli_module, "TradingPipeline", _FakePipeline)
    path = tmp_path / "account.json"
    save_account(path, _account())

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            "--day",
            "2024-01-02",
            "--account-file",
            str(path),
            "--output",
            str(path),
        ],
    )

    assert result.exit_code == 1
    assert "must be different" in result.stderr


def _fake_service(store: ParquetStore, pro) -> SyncService:
    return SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)


def test_sync_command_backfills_with_fake_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    service = _fake_service(store, FakePro())
    monkeypatch.setattr(cli_module, "_sync_service", lambda: service)

    result = CliRunner().invoke(
        cli_module.app, ["sync", "--start", "2024-01-01", "--end", "2024-01-03"]
    )

    assert result.exit_code == 0, result.output
    assert "synced 2 trading days" in result.output
    assert store.raw_path("daily", "20240102").exists()
    assert store.raw_path("daily", "20240103").exists()
    assert store.read_manifest(date(2024, 1, 2)) is not None


def test_calendar_command_refreshes_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    service = _fake_service(store, FakePro())
    monkeypatch.setattr(cli_module, "_sync_service", lambda: service)

    result = CliRunner().invoke(
        cli_module.app,
        ["calendar", "--start", "2024-01-01", "--end", "2024-01-05"],
    )

    assert result.exit_code == 0, result.output
    assert "trade_cal=5" in result.output
    assert store.read_reference("trade_cal").height == 5


def test_calendar_command_warns_when_end_not_covered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    service = _fake_service(store, FakePro())
    monkeypatch.setattr(cli_module, "_sync_service", lambda: service)

    result = CliRunner().invoke(
        cli_module.app,
        ["calendar", "--start", "2024-01-01", "--end", "2024-01-06"],
    )

    assert result.exit_code == 1
    assert "before requested end" in result.stderr


def test_fresh_store_e2e_sync_calendar_assemble_quality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    service = _fake_service(store, RangeAwareFakePro())
    monkeypatch.setattr(cli_module, "_sync_service", lambda: service)
    monkeypatch.setattr(cli_module, "_store", lambda: store)

    synced = CliRunner().invoke(
        cli_module.app, ["sync", "--start", "2024-01-01", "--end", "2024-01-03"]
    )
    assert synced.exit_code == 0, synced.output

    refreshed = CliRunner().invoke(
        cli_module.app,
        ["calendar", "--start", "2024-01-01", "--end", "2024-01-10"],
    )
    assert refreshed.exit_code == 0, refreshed.output

    assembled = CliRunner().invoke(
        cli_module.app,
        [
            "assemble-range",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-03",
        ],
    )
    assert assembled.exit_code == 0, assembled.output
    assert "assembled 3/3" in assembled.output

    quality = CliRunner().invoke(
        cli_module.app, ["quality", "--day", "2024-01-02"]
    )
    assert quality.exit_code == 0, quality.output
    assert "passed=True" in quality.output
