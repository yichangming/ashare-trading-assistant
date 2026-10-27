from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta

import polars as pl

from app.data.interfaces import DataSource
from app.data.manifest import DataManifest
from app.data.store import ParquetStore

_NON_EMPTY_ENDPOINTS = frozenset(
    ("daily", "adj_factor", "daily_basic", "stk_limit", "index_daily")
)
_REFERENCE_NAMES = (
    "trade_cal",
    "stock_basic",
    "namechange",
    "index_classify",
    "index_member",
)


class SyncError(RuntimeError):
    pass


@dataclass
class SyncResult:
    trade_date: date
    row_counts: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    reference_degraded: list[str] = field(default_factory=list)


def _to_date(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    for column in columns:
        if column in frame.columns and frame.schema[column] != pl.Date:
            frame = frame.with_columns(
                pl.col(column)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias(column)
            )
    return frame


class SyncService:
    def __init__(self, client: DataSource, store: ParquetStore):
        self.client = client
        self.store = store
        self.reference_degraded: list[str] = []

    def sync_reference(self, start: date, end: date) -> dict[str, int]:
        self.reference_degraded = []
        counts: dict[str, int] = {}

        trade_cal = self._fetch_trade_cal(start, end)
        trade_cal = _to_date(trade_cal, ["cal_date", "pretrade_date"])
        self.store.write_reference("trade_cal", trade_cal)
        counts["trade_cal"] = trade_cal.height

        stock_basic = self._fetch_stock_basic()
        stock_basic = _to_date(stock_basic, ["list_date", "delist_date"])
        self.store.write_reference("stock_basic", stock_basic)
        counts["stock_basic"] = stock_basic.height

        optional = (
            ("namechange", ["start_date", "end_date"]),
            ("index_classify", []),
            ("index_member", ["in_date", "out_date"]),
        )
        for endpoint, date_columns in optional:
            try:
                frame = self.client.fetch(endpoint)
            except Exception as exc:
                self.reference_degraded.append(f"{endpoint} unavailable: {exc}")
                self.store.remove_reference(endpoint)
                counts[endpoint] = 0
                continue
            frame = _to_date(frame, date_columns)
            self.store.write_reference(endpoint, frame)
            counts[endpoint] = frame.height
        return counts

    def sync_day(self, trade_date: date) -> SyncResult:
        key = trade_date.strftime("%Y%m%d")
        self.store.invalidate_manifest(trade_date)
        result = SyncResult(
            trade_date=trade_date,
            reference_degraded=list(self.reference_degraded),
        )
        required = {
            "daily": {"trade_date": key},
            "adj_factor": {"trade_date": key},
            "daily_basic": {"trade_date": key},
            "stk_limit": {"trade_date": key},
            "suspend_d": {"trade_date": key},
            "index_daily": {"trade_date": key},
        }
        frames: dict[str, pl.DataFrame] = {}
        for endpoint, params in required.items():
            try:
                frame = self.client.fetch(endpoint, **params)
            except Exception as exc:
                raise SyncError(
                    f"critical endpoint {endpoint} failed: {exc}"
                ) from exc
            if endpoint in _NON_EMPTY_ENDPOINTS and frame.is_empty():
                raise SyncError(f"critical endpoint {endpoint} returned no rows")
            frames[endpoint] = frame

        for endpoint, frame in frames.items():
            self.store.write_raw(endpoint, key, frame)
            result.row_counts[endpoint] = frame.height

        try:
            limit_list_d = self.client.fetch("limit_list_d", trade_date=key)
        except Exception as exc:
            result.degraded.append(f"limit_list_d unavailable: {exc}")
            self.store.remove_raw("limit_list_d", key)
        else:
            self.store.write_raw("limit_list_d", key, limit_list_d)
            result.row_counts["limit_list_d"] = limit_list_d.height

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
            if not force and self.store.read_manifest(trade_date) is not None:
                continue
            results.append(self.sync_day(trade_date))
        return results

    def sync_incremental(
        self, as_of: date, re_pull_days: int = 5
    ) -> list[SyncResult]:
        if re_pull_days < 0:
            raise ValueError("re_pull_days must be non-negative")
        self.reference_degraded = []
        self._refresh_calendar(as_of)
        calendar = self.store.read_reference("trade_cal")
        if calendar.is_empty():
            raise SyncError("trade_cal is empty; cannot sync incrementally")
        open_dates = [
            item
            for item in calendar.filter(pl.col("is_open") == 1)["cal_date"].to_list()
            if item <= as_of
        ]
        missing_days = [
            trade_date
            for trade_date in open_dates
            if self.store.read_manifest(trade_date) is None
        ]
        recent_days = open_dates[-re_pull_days:] if re_pull_days > 0 else []
        selected = sorted(set(missing_days) | set(recent_days))
        return [self.sync_day(trade_date) for trade_date in selected]

    def _fetch_trade_cal(self, start: date, end: date) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(date(cursor.year, 12, 31), end)
            frame = self.client.fetch(
                "trade_cal",
                exchange="SSE",
                start_date=cursor.strftime("%Y%m%d"),
                end_date=chunk_end.strftime("%Y%m%d"),
            )
            if not frame.is_empty():
                frames.append(frame)
            cursor = date(cursor.year + 1, 1, 1)
        if not frames:
            return pl.DataFrame()
        merged = (
            frames[0]
            if len(frames) == 1
            else pl.concat(frames, how="vertical_relaxed")
        )
        subset = [column for column in ("exchange", "cal_date") if column in merged.columns]
        if subset:
            merged = merged.unique(subset=subset, keep="last")
        if "cal_date" in merged.columns:
            merged = merged.sort("cal_date")
        return merged

    def _fetch_stock_basic(self) -> pl.DataFrame:
        fields = (
            "ts_code,symbol,name,area,industry,market,"
            "list_date,delist_date,is_hs"
        )
        frames = [
            self.client.fetch(
                "stock_basic",
                exchange="",
                list_status=status,
                fields=fields,
            )
            for status in ("L", "D", "P")
        ]
        frames = [frame for frame in frames if not frame.is_empty()]
        if not frames:
            return pl.DataFrame()
        merged = (
            frames[0]
            if len(frames) == 1
            else pl.concat(frames, how="vertical_relaxed")
        )
        if "ts_code" in merged.columns:
            merged = merged.unique(subset=["ts_code"], keep="last")
        return merged

    def _refresh_calendar(self, as_of: date) -> None:
        existing = self.store.read_reference("trade_cal")
        try:
            fresh = self._fetch_trade_cal(as_of - timedelta(days=60), as_of)
        except Exception as exc:
            if existing.is_empty():
                raise SyncError(f"trade_cal refresh failed: {exc}") from exc
            self.reference_degraded.append(f"trade_cal refresh failed: {exc}")
            return

        fresh = _to_date(fresh, ["cal_date", "pretrade_date"])
        if existing.is_empty():
            merged = fresh
        else:
            existing = _to_date(existing, ["cal_date", "pretrade_date"])
            merged = (
                existing
                if fresh.is_empty()
                else pl.concat([existing, fresh], how="vertical_relaxed")
            )
        subset = [column for column in ("exchange", "cal_date") if column in merged.columns]
        if subset:
            merged = merged.unique(subset=subset, keep="last")
        sort_columns = [
            column for column in ("exchange", "cal_date") if column in merged.columns
        ]
        if sort_columns:
            merged = merged.sort(sort_columns)
        self.store.write_reference("trade_cal", merged)

    def _write_manifest(self, result: SyncResult) -> None:
        key = result.trade_date.strftime("%Y%m%d")
        checksums: dict[str, str] = {}
        for endpoint in result.row_counts:
            path = self.store.raw_path(endpoint, key)
            if path.exists():
                checksums[endpoint] = hashlib.sha256(path.read_bytes()).hexdigest()
        reference_checksums: dict[str, str] = {}
        for name in _REFERENCE_NAMES:
            path = self.store.reference_path(name)
            if path.exists():
                reference_checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = DataManifest(
            trade_date=result.trade_date,
            row_counts=result.row_counts,
            checksums=checksums,
            reference_checksums=reference_checksums,
            degraded=result.degraded,
            reference_degraded=result.reference_degraded,
        )
        payload = manifest.model_dump(mode="json")
        payload["data_version"] = manifest.version()
        self.store.write_manifest(result.trade_date, payload)
