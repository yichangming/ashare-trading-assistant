from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BACKEND_ROOT / "data" / "app.db"

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def database_url() -> str:
    url = os.environ.get("ASHARE_DATABASE_URL")
    if url:
        return url
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


def configure_engine(url: str | None = None) -> Engine:
    """Create (and remember) the process-wide engine.

    Tests pass ``sqlite://`` (with StaticPool) or a temporary file URL.
    """
    global _engine, _session_factory
    target = url or database_url()
    kwargs: dict = {"future": True}
    if target.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if target in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    _engine = create_engine(target, **kwargs)
    _session_factory = sessionmaker(
        bind=_engine, expire_on_commit=False, autoflush=False, future=True
    )
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        configure_engine()
    assert _engine is not None
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        configure_engine()
    assert _session_factory is not None
    return _session_factory


def init_db(url: str | None = None, create: bool = True) -> Engine:
    engine = configure_engine(url)
    if create:
        Base.metadata.create_all(engine)
    return engine


def prepare_database(url: str | None = None) -> Engine:
    """Create/upgrade the operational schema.

    In-memory databases used by tests get ``create_all``.  File-backed
    databases go through Alembic; a legacy database created before Alembic was
    wired in is stamped at head once, then upgraded normally.
    """
    target = url or database_url()
    if target in ("sqlite://", "sqlite:///:memory:"):
        return init_db(target)

    from alembic.config import Config
    from sqlalchemy import inspect

    from alembic import command

    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.attributes["database_url"] = target
    existing_engine = create_engine(target)
    try:
        tables = set(inspect(existing_engine).get_table_names())
    finally:
        existing_engine.dispose()
    if tables and "alembic_version" not in tables:
        command.stamp(config, "head")
    else:
        command.upgrade(config, "head")
    return init_db(target, create=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
