"""Structured event ingestion and human verification.

Tushare official disclosures are stored as verified structured events.  Manual
contracts/policies default to pending verification; ambiguous beneficiaries
never become tradeable without an explicit verified decision.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
from sqlalchemy.orm import Session

from app.data.tushare_client import TushareClient
from app.db.repositories import EventRepository
from app.domain.enums import EventDirection, EventType

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean(value) for key, value in row.items()}


def _to_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.strptime(value, "%Y%m%d").date()
        except ValueError:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
    return None


def _earnings_direction(row: dict[str, Any]) -> EventDirection:
    kind = str(row.get("type") or "")
    if any(token in kind for token in ("增", "扭亏", "预盈")):
        return EventDirection.POSITIVE
    if any(token in kind for token in ("减", "亏", "预降")):
        return EventDirection.NEGATIVE
    change = row.get("p_change_min")
    if isinstance(change, (int, float)) and not math.isnan(float(change)):
        if change > 0:
            return EventDirection.POSITIVE
        if change < 0:
            return EventDirection.NEGATIVE
    return EventDirection.NEUTRAL


class EventIngestor:
    def __init__(self, session: Session):
        self.session = session

    def ingest_earnings(self, frame: pl.DataFrame, *, sub_type: str, source: str) -> int:
        imported = 0
        for row in frame.iter_rows(named=True):
            ann_date = _to_date(row.get("ann_date"))
            ts_code = row.get("ts_code")
            if not ts_code or ann_date is None:
                continue
            key = (
                f"{source}:{sub_type}:{ts_code}:{row.get('end_date')}:"
                f"{row.get('first_ann_date') or ann_date}"
            )
            direction = _earnings_direction(row)
            materiality = min(
                1.0,
                abs(float(row.get("p_change_min") or 0)) / 100
                if isinstance(row.get("p_change_min"), (int, float))
                else 0.0,
            )
            _, created = EventRepository.upsert(
                self.session,
                event_type=EventType.EARNINGS.value,
                sub_type=sub_type,
                title=f"{ts_code} {sub_type} {row.get('type') or ''}".strip(),
                summary=str(row.get("summary") or row.get("change_reason") or ""),
                source=source,
                dedupe_key=key,
                ts_code=str(ts_code),
                ann_date=ann_date,
                published_at=datetime.combine(
                    ann_date, time(17, 0), tzinfo=SHANGHAI
                ),
                stage="implementation",
                certainty="formal",
                direction=direction.value,
                materiality=materiality,
                verification_status="verified",
                raw_payload=_payload(row),
            )
            imported += int(created)
        return imported

    def ingest_repurchase(self, frame: pl.DataFrame, *, source: str) -> int:
        imported = 0
        for row in frame.iter_rows(named=True):
            ann_date = _to_date(row.get("ann_date"))
            ts_code = row.get("ts_code")
            if not ts_code or ann_date is None:
                continue
            stage = {
                "预案": "plan",
                "实施": "implementation",
                "完成": "completed",
            }.get(str(row.get("proc") or ""), "unknown")
            key = f"{source}:repurchase:{ts_code}:{ann_date}:{row.get('end_date')}"
            _, created = EventRepository.upsert(
                self.session,
                event_type=EventType.BUYBACK.value,
                sub_type="repurchase",
                title=f"{ts_code} 回购 {row.get('proc') or ''}".strip(),
                summary=(
                    f"金额 {row.get('amount')} "
                    f"价格 {row.get('low_limit')}–{row.get('high_limit')}"
                ),
                source=source,
                dedupe_key=key,
                ts_code=str(ts_code),
                ann_date=ann_date,
                published_at=datetime.combine(
                    ann_date, time(17, 0), tzinfo=SHANGHAI
                ),
                stage=stage,
                certainty="formal",
                direction=EventDirection.POSITIVE.value,
                materiality=min(
                    1.0, abs(float(row.get("amount") or 0)) / 1e9
                ),
                verification_status="verified",
                raw_payload=_payload(row),
            )
            imported += int(created)
        return imported

    def ingest_holder_trade(self, frame: pl.DataFrame, *, source: str) -> int:
        imported = 0
        for row in frame.iter_rows(named=True):
            ann_date = _to_date(row.get("ann_date"))
            ts_code = row.get("ts_code")
            if not ts_code or ann_date is None:
                continue
            direction = (
                EventDirection.POSITIVE
                if str(row.get("in_de")) == "IN"
                else EventDirection.NEGATIVE
                if str(row.get("in_de")) == "DE"
                else EventDirection.NEUTRAL
            )
            key = (
                f"{source}:holdertrade:{ts_code}:{ann_date}:"
                f"{row.get('holder_name')}:{row.get('in_de')}:{row.get('change_vol')}"
            )
            _, created = EventRepository.upsert(
                self.session,
                event_type=EventType.BUYBACK.value,
                sub_type="holder_trade",
                title=f"{ts_code} 股东{'增持' if direction == EventDirection.POSITIVE else '减持'}",
                summary=f"{row.get('holder_name')} 变动 {row.get('change_vol')} 股",
                source=source,
                dedupe_key=key,
                ts_code=str(ts_code),
                ann_date=ann_date,
                published_at=datetime.combine(
                    ann_date, time(17, 0), tzinfo=SHANGHAI
                ),
                stage="implementation",
                certainty="formal",
                direction=direction.value,
                materiality=min(
                    1.0, abs(float(row.get("change_ratio") or 0))
                ),
                verification_status="verified",
                raw_payload=_payload(row),
            )
            imported += int(created)
        return imported

    def ingest_manual(self, rows: list[dict[str, Any]], *, source: str = "manual") -> int:
        imported = 0
        for row in rows:
            event_type = str(row.get("event_type") or "")
            if event_type not in {item.value for item in EventType}:
                raise ValueError(f"unknown event_type: {event_type}")
            ts_code = row.get("ts_code")
            ann_date = _to_date(row.get("ann_date") or row.get("published_at"))
            if ann_date is None:
                raise ValueError("manual event requires ann_date")
            verification = str(row.get("verification_status") or "pending")
            key = str(
                row.get("dedupe_key")
                or f"{source}:{event_type}:{ts_code}:{ann_date}:{row.get('title')}"
            )
            _, created = EventRepository.upsert(
                self.session,
                event_type=event_type,
                sub_type=row.get("sub_type"),
                title=str(row.get("title") or event_type),
                summary=row.get("summary"),
                source=source,
                source_url=row.get("source_url"),
                dedupe_key=key,
                ts_code=ts_code,
                ann_date=ann_date,
                published_at=datetime.combine(
                    ann_date, time(17, 0), tzinfo=SHANGHAI
                ),
                stage=str(row.get("stage") or "unknown"),
                certainty=str(row.get("certainty") or "unknown"),
                direction=str(row.get("direction") or "neutral"),
                materiality=float(row.get("materiality") or 0),
                verification_status=verification,
                retracted=bool(row.get("retracted") or False),
                raw_payload=row,
            )
            imported += int(created)
        return imported

    def ingest_manual_file(self, path: Path | str, *, source: str = "manual") -> int:
        path = Path(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("events", [])
        else:
            import csv

            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        return self.ingest_manual(rows, source=source)


class TushareEventSync:
    def __init__(self, client: TushareClient, ingestor: EventIngestor):
        self.client = client
        self.ingestor = ingestor

    def sync_earnings(self, period: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for endpoint, subtype in (
            ("forecast_vip", "forecast"),
            ("express_vip", "express"),
            ("income_vip", "income"),
        ):
            frame = self.client.fetch(endpoint, period=period)
            counts[endpoint] = self.ingestor.ingest_earnings(
                frame, sub_type=subtype, source=endpoint
            )
        return counts

    def sync_buybacks(self, start: date, end: date) -> dict[str, int]:
        start_text, end_text = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        repurchase = self.client.fetch(
            "repurchase", start_date=start_text, end_date=end_text
        )
        holder = self.client.fetch(
            "stk_holdertrade", start_date=start_text, end_date=end_text
        )
        return {
            "repurchase": self.ingestor.ingest_repurchase(
                repurchase, source="repurchase"
            ),
            "stk_holdertrade": self.ingestor.ingest_holder_trade(
                holder, source="stk_holdertrade"
            ),
        }
