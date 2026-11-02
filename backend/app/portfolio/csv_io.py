"""Standard CSV templates and idempotent imports.

The templates are intentionally broker-neutral; a broker export can be mapped
to these columns rather than assuming any one broker's format.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.full_config import FullSystemConfig
from app.db.models import CorporateActionRow, ExecutionRow, OrderRow, PositionLotRow
from app.db.repositories import (
    AccountRepository,
    AuditRepository,
    PositionRepository,
)
from app.domain.enums import ExecutionSide, OrderStatus
from app.portfolio.service import AccountService, PortfolioError

TEMPLATES: dict[str, list[str]] = {
    "positions": [
        "ts_code",
        "sector",
        "shares",
        "available_shares",
        "cost_price",
        "current_stop",
        "current_price",
        "strategy",
        "sub_mode",
        "plan_id",
        "opened_at",
    ],
    "executions": [
        "executed_at",
        "ts_code",
        "side",
        "price",
        "shares",
        "fees",
        "plan_id",
        "source",
        "external_id",
        "note",
    ],
    "orders": [
        "plan_id",
        "ts_code",
        "side",
        "price",
        "shares",
        "status",
        "broker_order_id",
        "note",
    ],
    "cash_flows": ["occurred_at", "flow_type", "amount", "external_id", "note"],
    "corporate_actions": [
        "ts_code",
        "ex_date",
        "action_type",
        "cash_per_share",
        "share_ratio",
        "note",
    ],
}


@dataclass
class ImportResult:
    imported: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


def write_templates(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for kind, header in TEMPLATES.items():
        path = directory / f"{kind}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
        paths.append(path)
    return paths


def stable_external_id(kind: str, row: dict[str, str]) -> str:
    payload = "|".join(f"{key}={row.get(key, '')}" for key in sorted(row))
    return hashlib.sha256(f"{kind}:{payload}".encode()).hexdigest()[:24]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def import_csv(
    *,
    session: Session,
    service: AccountService,
    config: FullSystemConfig,
    kind: str,
    path: Path | str,
    source: str = "csv",
) -> ImportResult:
    if kind not in TEMPLATES:
        raise ValueError(f"unknown CSV kind: {kind}")
    result = ImportResult()
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(TEMPLATES[kind]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                if kind == "executions":
                    imported = _import_execution(session, service, config, row, source)
                elif kind == "positions":
                    imported = _import_position(session, service, row)
                elif kind == "orders":
                    imported = _import_order(session, service, row)
                elif kind == "cash_flows":
                    imported = _import_cash_flow(session, service, row)
                else:
                    imported = _import_corporate_action(session, service, row)
                if imported:
                    result.imported += 1
                else:
                    result.duplicates += 1
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"line {line_number}: {exc}")
    return result


def _import_execution(
    session: Session,
    service: AccountService,
    config: FullSystemConfig,
    row: dict[str, str],
    source: str,
) -> bool:
    executed_at = _parse_datetime(row.get("executed_at"))
    if executed_at is None:
        raise PortfolioError("invalid executed_at")
    external_id = row.get("external_id") or stable_external_id("executions", row)
    before = session.scalar(
        select(ExecutionRow).where(
            ExecutionRow.account_id == service.account_id,
            ExecutionRow.external_id == external_id,
        )
    )
    if before is not None:
        return False
    service.record_execution(
        ts_code=row["ts_code"],
        side=ExecutionSide(row["side"]),
        price=Decimal(row["price"]),
        shares=int(row["shares"]),
        executed_at=executed_at,
        fees=Decimal(row.get("fees") or "0"),
        plan_id=row.get("plan_id") or None,
        source=source,
        external_id=external_id,
        note=row.get("note") or None,
        enforce_plan=False,
    )
    return True


def _import_position(
    session: Session, service: AccountService, row: dict[str, str]
) -> bool:
    existing = PositionRepository.get(session, service.account_id, row["ts_code"])
    if existing is not None:
        return False
    opened_at = _parse_datetime(row.get("opened_at"))
    position = PositionRepository.upsert(
        session,
        account_id=service.account_id,
        ts_code=row["ts_code"],
        sector=row.get("sector") or "未知",
        shares=int(row["shares"]),
        available_shares=int(row["available_shares"] or row["shares"]),
        cost_price=Decimal(row["cost_price"]),
        current_stop=Decimal(row["current_stop"]),
        current_price=Decimal(row["current_price"]),
        primary_strategy=row.get("strategy") or None,
        sub_mode=row.get("sub_mode") or None,
        plan_id=row.get("plan_id") or None,
        opened_at=opened_at,
        needs_reconciliation=True,
        valuation_status="unknown",
    )
    # Selling reduces lots FIFO; an imported position without a lot would make
    # every later sell fail with a lot-accounting mismatch.
    session.add(
        PositionLotRow(
            id=uuid4().hex,
            position_id=position.id,
            strategy=row.get("strategy") or None,
            sub_mode=row.get("sub_mode") or None,
            shares=int(row["shares"]),
            cost_price=Decimal(row["cost_price"]),
            opened_at=opened_at,
            add_index=0,
            plan_id=row.get("plan_id") or None,
        )
    )
    account = AccountRepository.ensure(session, service.account_id)
    positions = PositionRepository.list_positions(session, service.account_id)
    account.equity = account.cash + sum(
        (position.current_price * position.shares for position in positions),
        Decimal("0"),
    )
    account.peak_equity = max(account.peak_equity, account.equity)
    AuditRepository.log(
        session,
        actor="user",
        action="import_position",
        target=row["ts_code"],
        after={"shares": int(row["shares"])},
    )
    session.flush()
    return True


def _import_order(
    session: Session, service: AccountService, row: dict[str, str]
) -> bool:
    external = row.get("broker_order_id") or stable_external_id("orders", row)
    existing = session.scalar(
        select(OrderRow).where(
            OrderRow.account_id == service.account_id,
            OrderRow.broker_order_id == external,
        )
    )
    if existing is not None:
        return False
    session.add(
        OrderRow(
            id=uuid4().hex,
            account_id=service.account_id,
            plan_id=row["plan_id"],
            ts_code=row["ts_code"],
            side=ExecutionSide(row["side"]).value,
            price=Decimal(row["price"]),
            shares=int(row["shares"]),
            status=OrderStatus(row.get("status") or "pending").value,
            broker_order_id=external,
        )
    )
    session.flush()
    return True


def _import_cash_flow(
    session: Session, service: AccountService, row: dict[str, str]
) -> bool:
    occurred = _parse_datetime(row.get("occurred_at"))
    if occurred is None:
        raise PortfolioError("invalid occurred_at")
    external_id = row.get("external_id") or stable_external_id("cash_flows", row)
    return service.record_cash_flow(
        flow_type=row["flow_type"],
        amount=Decimal(row["amount"]),
        occurred_at=occurred,
        external_id=external_id,
        note=row.get("note") or None,
    )


def _import_corporate_action(
    session: Session, service: AccountService, row: dict[str, str]
) -> bool:
    ex_date = _parse_date(row.get("ex_date"))
    if ex_date is None:
        raise PortfolioError("invalid ex_date")
    existing = session.scalar(
        select(CorporateActionRow).where(
            CorporateActionRow.account_id == service.account_id,
            CorporateActionRow.ts_code == row["ts_code"],
            CorporateActionRow.ex_date == ex_date,
            CorporateActionRow.action_type == row["action_type"],
        )
    )
    if existing is not None:
        return False
    service.apply_corporate_action(
        ts_code=row["ts_code"],
        ex_date=ex_date,
        action_type=row["action_type"],
        cash_per_share=Decimal(row.get("cash_per_share") or "0"),
        share_ratio=float(row.get("share_ratio") or 0),
        note=row.get("note") or None,
        apply_now=ex_date <= date.today(),
    )
    return True
