"""SQLAlchemy operational models.

Market data stays in Parquet/DuckDB.  This database holds account, plan,
execution, event, notification, job and audit state that must survive a
process restart.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

MONEY = Numeric(20, 4)
PRICE = Numeric(18, 4)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class AccountRow(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), default="default")
    mode: Mapped[str] = mapped_column(String(16), default="paper")
    equity: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    cash: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    peak_equity: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    initial_equity: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    max_drawdown: Mapped[float] = mapped_column(default=0.15)
    risk_config_configured: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    trading_halted: Mapped[bool] = mapped_column(Boolean, default=False)
    drawdown_half_latched: Mapped[bool] = mapped_column(Boolean, default=False)
    net_deposits: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    valuation_incomplete: Mapped[bool] = mapped_column(Boolean, default=False)
    valuation_note: Mapped[str | None] = mapped_column(Text)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_valuation_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_reconciled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    stale: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    positions: Mapped[list[PositionRow]] = relationship(
        back_populates="account", cascade="all, delete-orphan"
    )
    plans: Mapped[list[PlanRow]] = relationship(back_populates="account")


class PositionRow(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("account_id", "ts_code", name="uq_position"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    sector: Mapped[str] = mapped_column(String(80), default="未知")
    shares: Mapped[int] = mapped_column(Integer, default=0)
    available_shares: Mapped[int] = mapped_column(Integer, default=0)
    cost_price: Mapped[Decimal] = mapped_column(PRICE, default=Decimal("0"))
    current_stop: Mapped[Decimal] = mapped_column(PRICE, default=Decimal("0"))
    current_price: Mapped[Decimal] = mapped_column(PRICE, default=Decimal("0"))
    primary_strategy: Mapped[str | None] = mapped_column(String(4))
    sub_mode: Mapped[str | None] = mapped_column(String(24))
    strategy_version: Mapped[str | None] = mapped_column(String(80))
    plan_id: Mapped[str | None] = mapped_column(String(32))
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    highest_price: Mapped[Decimal | None] = mapped_column(PRICE)
    initial_r: Mapped[Decimal | None] = mapped_column(PRICE)
    initial_shares: Mapped[int] = mapped_column(Integer, default=0)
    add_on_count: Mapped[int] = mapped_column(Integer, default=0)
    target_state_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list
    )
    last_decision_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    last_decision_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_alert_status: Mapped[str | None] = mapped_column(String(24))
    last_alert_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    needs_reconciliation: Mapped[bool] = mapped_column(Boolean, default=False)
    valuation_status: Mapped[str] = mapped_column(String(16), default="unknown")
    valuation_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    realized_r: Mapped[float] = mapped_column(default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="open")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    account: Mapped[AccountRow] = relationship(back_populates="positions")
    lots: Mapped[list[PositionLotRow]] = relationship(
        back_populates="position", cascade="all, delete-orphan"
    )


class PositionLotRow(Base):
    __tablename__ = "position_lots"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    position_id: Mapped[str] = mapped_column(
        ForeignKey("positions.id", ondelete="CASCADE"), index=True
    )
    strategy: Mapped[str | None] = mapped_column(String(4))
    sub_mode: Mapped[str | None] = mapped_column(String(24))
    sector: Mapped[str | None] = mapped_column(String(80))
    strategy_version: Mapped[str | None] = mapped_column(String(80))
    shares: Mapped[int] = mapped_column(Integer, default=0)
    cost_price: Mapped[Decimal] = mapped_column(PRICE, default=Decimal("0"))
    initial_r: Mapped[Decimal | None] = mapped_column(PRICE)
    initial_stop: Mapped[Decimal | None] = mapped_column(PRICE)
    entry_fees: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    is_add_on: Mapped[bool] = mapped_column(Boolean, default=False)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    add_index: Mapped[int] = mapped_column(Integer, default=0)
    plan_id: Mapped[str | None] = mapped_column(String(32))

    position: Mapped[PositionRow] = relationship(back_populates="lots")


class PlanRow(Base):
    __tablename__ = "plans"
    __table_args__ = (
        Index("ix_plans_date_status", "trade_date", "status"),
        Index("ix_plans_strategy", "strategy"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    sector: Mapped[str] = mapped_column(String(80), default="未知")
    strategy: Mapped[str] = mapped_column(String(4), default="A")
    sub_mode: Mapped[str | None] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(32), default="DRAFT")
    expires_on: Mapped[date] = mapped_column(Date)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_add_on: Mapped[bool] = mapped_column(Boolean, default=False)
    confirmation_price: Mapped[Decimal | None] = mapped_column(PRICE)
    max_acceptable_price: Mapped[Decimal | None] = mapped_column(PRICE)
    stop_price: Mapped[Decimal | None] = mapped_column(PRICE)
    shares: Mapped[int] = mapped_column(Integer, default=0)
    risk_amount: Mapped[Decimal | None] = mapped_column(MONEY)
    strategy_version: Mapped[str] = mapped_column(String(80), default="")
    data_version: Mapped[str] = mapped_column(String(80), default="")
    code_version: Mapped[str] = mapped_column(String(80), default="")
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    account: Mapped[AccountRow] = relationship(back_populates="plans")
    executions: Mapped[list[ExecutionRow]] = relationship(back_populates="plan")


class ExecutionRow(Base):
    __tablename__ = "executions"
    __table_args__ = (
        UniqueConstraint(
            "account_id", "external_id", name="uq_execution_external"
        ),
        Index("ix_executions_code_time", "ts_code", "executed_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    plan_id: Mapped[str | None] = mapped_column(ForeignKey("plans.id"), index=True)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[Decimal] = mapped_column(PRICE)
    shares: Mapped[int] = mapped_column(Integer)
    fees: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    strategy: Mapped[str | None] = mapped_column(String(4))
    sub_mode: Mapped[str | None] = mapped_column(String(24))
    sector: Mapped[str | None] = mapped_column(String(80))
    strategy_version: Mapped[str | None] = mapped_column(String(80))
    batch_index: Mapped[int | None] = mapped_column(Integer)
    initial_r: Mapped[Decimal | None] = mapped_column(PRICE)
    plan_deviation: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(16), default="manual")
    external_id: Mapped[str | None] = mapped_column(String(120))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    plan: Mapped[PlanRow | None] = relationship(back_populates="executions")


class OrderRow(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"), index=True)
    ts_code: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[Decimal] = mapped_column(PRICE)
    shares: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    broker_order_id: Mapped[str | None] = mapped_column(String(120))
    cancel_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class CashFlowRow(Base):
    __tablename__ = "cash_flows"
    __table_args__ = (
        UniqueConstraint("account_id", "external_id", name="uq_cashflow_external"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    flow_type: Mapped[str] = mapped_column(String(24))
    amount: Mapped[Decimal] = mapped_column(MONEY)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)
    external_id: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CorporateActionRow(Base):
    __tablename__ = "corporate_actions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    ex_date: Mapped[date] = mapped_column(Date)
    action_type: Mapped[str] = mapped_column(String(24))
    cash_per_share: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    share_ratio: Mapped[float] = mapped_column(default=0.0)
    applied: Mapped[bool] = mapped_column(Boolean, default=True)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EventRow(Base):
    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("dedupe_key", name="uq_event_dedupe"),
        Index("ix_events_type_date", "event_type", "ann_date"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(24), index=True)
    sub_type: Mapped[str | None] = mapped_column(String(40))
    ts_code: Mapped[str | None] = mapped_column(String(16), index=True)
    title: Mapped[str] = mapped_column(String(300))
    summary: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(40))
    source_url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ann_date: Mapped[date | None] = mapped_column(Date, index=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    retracted: Mapped[bool] = mapped_column(Boolean, default=False)
    stage: Mapped[str] = mapped_column(String(24), default="unknown")
    certainty: Mapped[str] = mapped_column(String(16), default="unknown")
    direction: Mapped[str] = mapped_column(String(16), default="neutral")
    materiality: Mapped[float] = mapped_column(default=0.0)
    verification_status: Mapped[str] = mapped_column(String(16), default="pending")
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dedupe_key: Mapped[str] = mapped_column(String(200), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    links: Mapped[list[EventLinkRow]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )
    revisions: Mapped[list[EventRevisionRow]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class EventRevisionRow(Base):
    __tablename__ = "event_revisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_id: Mapped[str] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(300))
    summary: Mapped[str | None] = mapped_column(Text)
    direction: Mapped[str] = mapped_column(String(16), default="neutral")
    materiality: Mapped[float] = mapped_column(default=0.0)
    verification_status: Mapped[str] = mapped_column(String(16), default="pending")
    retracted: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    event: Mapped[EventRow] = relationship(back_populates="revisions")


class EventLinkRow(Base):
    __tablename__ = "event_links"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_id: Mapped[str] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), index=True
    )
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    relation: Mapped[str] = mapped_column(String(40), default="subject")
    ambiguous: Mapped[bool] = mapped_column(Boolean, default=False)

    event: Mapped[EventRow] = relationship(back_populates="links")


class NotificationRow(Base):
    __tablename__ = "notifications"
    __table_args__ = (
        UniqueConstraint("dedupe_key", name="uq_notification_dedupe"),
        Index("ix_notifications_level_created", "level", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    level: Mapped[str] = mapped_column(String(16), default="info")
    code: Mapped[str] = mapped_column(String(60))
    title: Mapped[str] = mapped_column(String(300))
    message: Mapped[str] = mapped_column(Text)
    trade_date: Mapped[date | None] = mapped_column(Date, index=True)
    dedupe_key: Mapped[str] = mapped_column(String(200), index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class JobRunRow(Base):
    __tablename__ = "job_runs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_job_idempotency"),
        Index("ix_jobs_date_status", "trade_date", "status"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_name: Mapped[str] = mapped_column(String(60))
    trade_date: Mapped[date | None] = mapped_column(Date, index=True)
    scope: Mapped[str] = mapped_column(String(40), default="full")
    status: Mapped[str] = mapped_column(String(16), default="queued")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    data_version: Mapped[str | None] = mapped_column(String(80))
    strategy_version: Mapped[str | None] = mapped_column(String(80))
    detail_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(200), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLogRow(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    actor: Mapped[str] = mapped_column(String(80), default="system")
    action: Mapped[str] = mapped_column(String(80))
    target: Mapped[str] = mapped_column(String(200))
    before_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PortfolioSnapshotRow(Base):
    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        UniqueConstraint("account_id", "trade_date", name="uq_snapshot_day"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    equity: Mapped[Decimal] = mapped_column(MONEY)
    cash: Mapped[Decimal] = mapped_column(MONEY)
    positions_value: Mapped[Decimal] = mapped_column(MONEY)
    heat: Mapped[float] = mapped_column(default=0.0)
    drawdown: Mapped[float] = mapped_column(default=0.0)
    risk_mode: Mapped[str] = mapped_column(String(16), default="normal")
    detail_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BacktestRunRow(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), default="backtest")
    strategy_mode: Mapped[str] = mapped_column(String(40), default="dual")
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    artifacts_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    data_version: Mapped[str] = mapped_column(String(80), default="")
    code_version: Mapped[str] = mapped_column(String(80), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SettingRow(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value_json: Mapped[dict[str, Any] | list[Any] | str | int | float | bool | None] = (
        mapped_column(JSON)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
