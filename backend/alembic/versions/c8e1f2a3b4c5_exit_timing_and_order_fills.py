"""exit timing, pending stops and order fill quantities

Revision ID: c8e1f2a3b4c5
Revises: b7c1d2e3f4a5
Create Date: 2026-11-02 12:30:00.000000

The effective protection line and the next-session protection line are stored
separately so a post-close calculation can never be applied retroactively to
the same day's intraday prices.  Order rows also carry the filled quantity so
remaining exposure is calculated from facts rather than the original order
size.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c8e1f2a3b4c5"
down_revision: Union[str, Sequence[str], None] = "b7c1d2e3f4a5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add(table: str, column: sa.Column) -> None:
    with op.batch_alter_table(table) as batch_op:
        batch_op.add_column(column)


def upgrade() -> None:
    _add(
        "positions",
        sa.Column("pending_stop", sa.Numeric(precision=18, scale=4), nullable=True),
    )
    _add(
        "positions",
        sa.Column(
            "pending_stop_data_time", sa.DateTime(timezone=True), nullable=True
        ),
    )
    _add(
        "positions",
        sa.Column(
            "pending_stop_effective_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    _add("positions", sa.Column("last_new_high_on", sa.Date(), nullable=True))
    _add(
        "orders",
        sa.Column(
            "filled_shares", sa.Integer(), nullable=False, server_default="0"
        ),
    )


def downgrade() -> None:
    with op.batch_alter_table("orders") as batch_op:
        batch_op.drop_column("filled_shares")
    with op.batch_alter_table("positions") as batch_op:
        batch_op.drop_column("last_new_high_on")
        batch_op.drop_column("pending_stop_effective_at")
        batch_op.drop_column("pending_stop_data_time")
        batch_op.drop_column("pending_stop")
