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
