from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from app.db.models import Base

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _config(monkeypatch, url: str) -> Config:
    monkeypatch.setenv("ASHARE_DATABASE_URL", url)
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    return config


def test_initial_migration_creates_operational_schema(tmp_path, monkeypatch):
    database = tmp_path / "app.db"
    url = f"sqlite:///{database}"
    config = _config(monkeypatch, url)

    command.upgrade(config, "head")

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()

    assert set(Base.metadata.tables) <= tables
    assert "alembic_version" in tables
    assert "accounts" in tables
    assert "position_lots" in tables
    assert "backtest_runs" in tables

    # Re-running at head is a no-op rather than an error.
    command.upgrade(config, "head")


def test_migration_downgrade_removes_schema(tmp_path, monkeypatch):
    database = tmp_path / "app.db"
    url = f"sqlite:///{database}"
    config = _config(monkeypatch, url)

    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()

    assert not (set(Base.metadata.tables) & tables)


def test_review_migration_adds_columns_and_marks_old_positions(tmp_path, monkeypatch):
    from datetime import datetime

    from sqlalchemy import text

    database = tmp_path / "app.db"
    url = f"sqlite:///{database}"
    config = _config(monkeypatch, url)
    command.upgrade(config, "a3fe5121272f")

    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO accounts (id, name, mode, equity, cash,"
                " peak_equity, initial_equity, max_drawdown,"
                " risk_config_configured, risk_profile_json, trading_halted,"
                " drawdown_half_latched, last_synced_at, stale, created_at,"
                " updated_at) VALUES ('default', 'default', 'paper', 100000,"
                " 90000, 100000, 100000, 0.15, 0, '{}', 0, 0, NULL, 0,"
                " :now, :now)"
            ),
            {"now": datetime(2024, 1, 2)},
        )
        connection.execute(
            text(
                "INSERT INTO positions (id, account_id, ts_code, sector, shares,"
                " available_shares, cost_price, current_stop, current_price,"
                " primary_strategy, plan_id, opened_at, highest_price,"
                " realized_r, status, updated_at) VALUES ('p1', 'default',"
                " '000001.SZ', '银行', 1000, 1000, 10, 9.5, 10, 'A', NULL,"
                " :now, 10, 0, 'open', :now)"
            ),
            {"now": datetime(2024, 1, 2)},
        )
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(url)
    columns = {column["name"] for column in inspect(engine).get_columns("positions")}
    assert {
        "initial_r",
        "initial_shares",
        "add_on_count",
        "target_state_json",
        "needs_reconciliation",
        "valuation_status",
    } <= columns
    account_columns = {
        column["name"] for column in inspect(engine).get_columns("accounts")
    }
    assert {"net_deposits", "last_reconciled_at", "last_valuation_at"} <= account_columns
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT needs_reconciliation, valuation_status"
                " FROM positions WHERE id = 'p1'"
            )
        ).one()
    engine.dispose()
    assert row[0] == 1
    assert row[1] == "unknown"
