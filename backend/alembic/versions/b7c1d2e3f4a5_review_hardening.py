"""review hardening: valuation, attribution, targets and exit state

Revision ID: b7c1d2e3f4a5
Revises: a3fe5121272f
Create Date: 2026-11-02 09:30:00.000000

Existing rows are marked ``needs_reconciliation`` instead of guessing their
strategy, initial R, or stop from incomplete history.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7c1d2e3f4a5"
down_revision: Union[str, Sequence[str], None] = "a3fe5121272f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add(table: str, column: sa.Column) -> None:
    with op.batch_alter_table(table) as batch_op:
        batch_op.add_column(column)


def upgrade() -> None:
    _add(
        "accounts",
        sa.Column(
            "net_deposits",
            sa.Numeric(precision=20, scale=4),
            nullable=False,
            server_default="0",
        ),
    )
    _add(
        "accounts",
        sa.Column(
            "valuation_incomplete",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    _add("accounts", sa.Column("valuation_note", sa.Text(), nullable=True))
    _add(
        "accounts",
        sa.Column("last_valuation_at", sa.DateTime(timezone=True), nullable=True),
    )
    _add(
        "accounts",
        sa.Column("last_reconciled_at", sa.DateTime(timezone=True), nullable=True),
    )

    _add("positions", sa.Column("sub_mode", sa.String(length=24), nullable=True))
    _add(
        "positions", sa.Column("strategy_version", sa.String(length=80), nullable=True)
    )
    _add(
        "positions",
        sa.Column("initial_r", sa.Numeric(precision=18, scale=4), nullable=True),
    )
    _add(
        "positions",
        sa.Column(
            "initial_shares", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    _add(
        "positions",
        sa.Column(
            "add_on_count", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    _add(
        "positions",
        sa.Column("target_state_json", sa.JSON(), nullable=False, server_default="[]"),
    )
    _add("positions", sa.Column("last_decision_json", sa.JSON(), nullable=True))
    _add(
        "positions",
        sa.Column("last_decision_at", sa.DateTime(timezone=True), nullable=True),
    )
    _add("positions", sa.Column("last_alert_status", sa.String(length=24), nullable=True))
    _add(
        "positions",
        sa.Column("last_alert_at", sa.DateTime(timezone=True), nullable=True),
    )
    _add(
        "positions",
        sa.Column(
            "needs_reconciliation",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    _add(
        "positions",
        sa.Column(
            "valuation_status",
            sa.String(length=16),
            nullable=False,
            server_default="unknown",
        ),
    )
    _add(
        "positions",
        sa.Column("valuation_at", sa.DateTime(timezone=True), nullable=True),
    )

    _add("position_lots", sa.Column("sector", sa.String(length=80), nullable=True))
    _add(
        "position_lots",
        sa.Column("strategy_version", sa.String(length=80), nullable=True),
    )
    _add(
        "position_lots",
        sa.Column(
            "initial_stop", sa.Numeric(precision=18, scale=4), nullable=True
        ),
    )
    _add(
        "position_lots",
        sa.Column(
            "entry_fees",
            sa.Numeric(precision=20, scale=4),
            nullable=False,
            server_default="0",
        ),
    )
    _add(
        "position_lots",
        sa.Column(
            "is_add_on", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )

    _add("plans", sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))
    _add(
        "plans",
        sa.Column("is_add_on", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    _add("executions", sa.Column("strategy", sa.String(length=4), nullable=True))
    _add("executions", sa.Column("sub_mode", sa.String(length=24), nullable=True))
    _add("executions", sa.Column("sector", sa.String(length=80), nullable=True))
    _add(
        "executions", sa.Column("strategy_version", sa.String(length=80), nullable=True)
    )
    _add("executions", sa.Column("batch_index", sa.Integer(), nullable=True))
    _add(
        "executions",
        sa.Column("initial_r", sa.Numeric(precision=18, scale=4), nullable=True),
    )
    _add("executions", sa.Column("plan_deviation", sa.Text(), nullable=True))

    _add(
        "job_runs",
        sa.Column("detail_json", sa.JSON(), nullable=False, server_default="{}"),
    )

    # Existing account/position history cannot be reliably reconstructed.
    op.execute(
        "UPDATE positions SET needs_reconciliation = 1, "
        "valuation_status = 'unknown'"
    )
    op.execute("UPDATE accounts SET valuation_incomplete = 1")


def downgrade() -> None:
    for table, columns in (
        (
            "job_runs",
            ["detail_json"],
        ),
        (
            "executions",
            [
                "plan_deviation",
                "initial_r",
                "batch_index",
                "strategy_version",
                "sector",
                "sub_mode",
                "strategy",
            ],
        ),
        ("plans", ["is_add_on", "expires_at"]),
        (
            "position_lots",
            ["is_add_on", "entry_fees", "initial_stop", "strategy_version", "sector"],
        ),
        (
            "positions",
            [
                "valuation_at",
                "valuation_status",
                "needs_reconciliation",
                "last_alert_at",
                "last_alert_status",
                "last_decision_at",
                "last_decision_json",
                "target_state_json",
                "add_on_count",
                "initial_shares",
                "initial_r",
                "strategy_version",
                "sub_mode",
            ],
        ),
        (
            "accounts",
            [
                "last_reconciled_at",
                "last_valuation_at",
                "valuation_note",
                "valuation_incomplete",
                "net_deposits",
            ],
        ),
    ):
        with op.batch_alter_table(table) as batch_op:
            for column in columns:
                batch_op.drop_column(column)
