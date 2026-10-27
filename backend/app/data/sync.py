from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from app.data.manifest import DataManifest
from app.data.store import ParquetStore
from app.data.tushare_client import TushareClient


class SyncError(RuntimeError):
    pass


@dataclass
class SyncResult:
    trade_date: date
    row_counts: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


def _to_date(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame = frame.with_columns(
                pl.col(column)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias(column)
            )
    return frame


class SyncService:
    def __init__(self, client: TushareClient, store: ParquetStore):
        self.client = client
        self.store = store

    def sync_reference(self, start: date, end: date) -> dict[str, int]:
        counts: dict[str, int] = {}
        trade_cal = self.client.fetch(
            "trade_cal",
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        trade_cal = _to_date(trade_cal, ["cal_date", "pretrade_date"])
        self.store.write_reference("trade_cal", trade_cal)
        counts["trade_cal"] = trade_cal.height

        stock_basic = self.client.fetch(
            "stock_basic",
            exchange="",
            list_status="L,D,P",
            fields=(
                "ts_code,symbol,name,area,industry,market,"
                "list_date,delist_date,is_hs"
            ),
        )
        stock_basic = _to_date(stock_basic, ["list_date", "delist_date"])
        self.store.write_reference("stock_basic", stock_basic)
        counts["stock_basic"] = stock_basic.height

        namechange = self.client.fetch("namechange")
        namechange = _to_date(namechange, ["start_date", "end_date"])
        self.store.write_reference("namechange", namechange)
        counts["namechange"] = namechange.height

        for endpoint in ("index_classify", "index_member"):
            try:
                frame = self.client.fetch(endpoint)
                frame = _to_date(frame, ["in_date", "out_date"])
                self.store.write_reference(endpoint, frame)
                counts[endpoint] = frame.height
            except Exception:
                counts[endpoint] = 0
        return counts

    def sync_day(self, trade_date: date) -> SyncResult:
        key = trade_date.strftime("%Y%m%d")
        result = SyncResult(trade_date=trade_date)
        required = {
            "daily": {"trade_date": key},
            "adj_factor": {"trade_date": key},
            "daily_basic": {"trade_date": key},
            "stk_limit": {"trade_date": key},
            "suspend_d": {"trade_date": key},
            "index_daily": {"trade_date": key},
        }
        for endpoint, params in required.items():
            try:
                frame = self.client.fetch(endpoint, **params)
            except Exception as exc:
                raise SyncError(f"critical endpoint {endpoint} failed: {exc}") from exc
            self.store.write_raw(endpoint, key, frame)
            result.row_counts[endpoint] = frame.height

        try:
            frame = self.client.fetch("limit_list_d", trade_date=key)
            self.store.write_raw("limit_list_d", key, frame)
            result.row_counts["limit_list_d"] = frame.height
        except Exception as exc:
            result.degraded.append(f"limit_list_d unavailable: {exc}")

        self._write_manifest(result)
        return result

    def backfill(
        self, start: date, end: date, force: bool = False
    ) -> list[SyncResult]:
        self.sync_reference(start, end)
        calendar = self.store.read_reference("trade_cal")
        if calendar.is_empty():
            raise SyncError("trade_cal is empty; cannot backfill")
        open_dates = [
            item
            for item in calendar.filter(pl.col("is_open") == 1)["cal_date"].to_list()
            if start <= item <= end
        ]
        results: list[SyncResult] = []
        for trade_date in open_dates:
            key = trade_date.strftime("%Y%m%d")
            if not force and self.store.read_raw("daily", key).height > 0:
                continue
            results.append(self.sync_day(trade_date))
        return results

    def sync_incremental(
        self, as_of: date, re_pull_days: int = 5
    ) -> list[SyncResult]:
        calendar = self.store.read_reference("trade_cal")
        if calendar.is_empty():
            raise SyncError("trade_cal is empty; run sync_reference first")
        open_dates = [
            item
            for item in calendar.filter(pl.col("is_open") == 1)["cal_date"].to_list()
            if item <= as_of
        ]
        selected = open_dates[-re_pull_days:] if re_pull_days > 0 else []
        return [self.sync_day(item) for item in selected]

    def _write_manifest(self, result: SyncResult) -> None:
        checksums: dict[str, str] = {}
        for endpoint in result.row_counts:
            path = (
                self.store.raw_dir
                / endpoint
                / f"{result.trade_date.strftime('%Y%m%d')}.parquet"
            )
            if path.exists():
                checksums[endpoint] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = DataManifest(
            trade_date=result.trade_date,
            row_counts=result.row_counts,
            checksums=checksums,
            degraded=result.degraded,
        )
        payload = manifest.model_dump(mode="json")
        payload["data_version"] = manifest.version()
        output = (
            self.store.raw_dir
            / "manifests"
            / f"{result.trade_date.strftime('%Y%m%d')}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
