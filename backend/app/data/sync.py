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
_MAX_RE_PULL_DAYS = 365 * 10


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


def _merge_trade_cal(existing: pl.DataFrame, fresh: pl.DataFrame) -> pl.DataFrame:
    existing = _to_date(existing, ["cal_date", "pretrade_date"])
    fresh = _to_date(fresh, ["cal_date", "pretrade_date"])
    frames = [frame for frame in (existing, fresh) if not frame.is_empty()]
    if not frames:
        return pl.DataFrame()
    merged = (
        frames[0]
        if len(frames) == 1
        else pl.concat(frames, how="vertical_relaxed")
    )
    subset = [
        column for column in ("exchange", "cal_date") if column in merged.columns
    ]
    if subset:
        merged = merged.unique(subset=subset, keep="last")
    sort_columns = [
        column for column in ("exchange", "cal_date") if column in merged.columns
    ]
    if sort_columns:
        merged = merged.sort(sort_columns)
    return merged


class SyncService:
    def __init__(self, client: DataSource, store: ParquetStore):
        self.client = client
        self.store = store
        self.reference_degraded: list[str] = []

    def sync_reference(
        self, start: date, end: date, strict_calendar: bool = False
    ) -> dict[str, int]:
        if start > end:
            raise ValueError("start must be on or before end")
        self.reference_degraded = []
        existing_calendar = self.store.read_reference("trade_cal")
        fetch_failed = False
        try:
            fresh_calendar = self._fetch_trade_cal(start, end)
        except Exception as exc:
            if strict_calendar or existing_calendar.is_empty():
                raise SyncError(
                    f"critical endpoint trade_cal failed: {exc}"
                ) from exc
            self.reference_degraded.append(f"trade_cal unavailable: {exc}")
            fresh_calendar = pl.DataFrame()
            fetch_failed = True

        if not fetch_failed and fresh_calendar.is_empty():
            if strict_calendar:
                raise SyncError("critical endpoint trade_cal returned no rows")
            self.reference_degraded.append("trade_cal returned no rows")

        calendar = _merge_trade_cal(existing_calendar, fresh_calendar)
        if calendar.is_empty():
            raise SyncError("critical endpoint trade_cal returned no rows")
        if strict_calendar:
            self._check_calendar_coverage(calendar, start, end)

        try:
            stock_basic = self._fetch_stock_basic()
        except Exception as exc:
            raise SyncError(
                f"critical endpoint stock_basic failed: {exc}"
            ) from exc
        if stock_basic.is_empty():
            raise SyncError("critical endpoint stock_basic returned no rows")

        counts: dict[str, int] = {}
        self.store.write_reference("trade_cal", calendar)
        counts["trade_cal"] = calendar.height
        stock_basic = _to_date(stock_basic, ["list_date", "delist_date"])
        self.store.write_reference("stock_basic", stock_basic)
        counts["stock_basic"] = stock_basic.height

        optional: tuple[tuple[str, list[str]], ...] = (
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
        if start > end:
            raise ValueError("start must be on or before end")
        self.sync_reference(start, end, strict_calendar=True)
        calendar = self.store.read_reference("trade_cal")
        if calendar.is_empty():
            raise SyncError("trade_cal is empty; cannot backfill")
        open_dates = [
            trade_date
            for trade_date in self._calendar_open_dates(calendar)
            if start <= trade_date <= end
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
        """Refresh references and sync missing/recent open days up to as_of.

        The calendar refresh starts the day after the last stored open date
        (or ``as_of - re_pull_days`` when earlier), fetched in yearly chunks so
        long gaps are filled rather than truncated. After a large gap this can
        serially sync many missing days; run a full backfill instead when
        bootstrapping a long history.
        """
        if not 0 <= re_pull_days <= _MAX_RE_PULL_DAYS:
            raise ValueError(
                f"re_pull_days must be between 0 and {_MAX_RE_PULL_DAYS}; "
                "use backfill for longer history"
            )
        self.reference_degraded = []
        stored_calendar = self.store.read_reference("trade_cal")
        if stored_calendar.is_empty():
            raise SyncError(
                "trade_cal is empty; run sync_reference/backfill first"
            )

        stored_open_dates = self._calendar_open_dates(stored_calendar)
        last_open_date = max(stored_open_dates) if stored_open_dates else None
        refresh_start = as_of - timedelta(days=re_pull_days)
        if last_open_date is not None:
            refresh_start = min(refresh_start, last_open_date + timedelta(days=1))

        self.sync_reference(refresh_start, as_of)

        calendar = self.store.read_reference("trade_cal")
        if calendar.is_empty():
            raise SyncError(
                "trade_cal is empty; run sync_reference/backfill first"
            )
        open_dates = [
            trade_date
            for trade_date in self._calendar_open_dates(calendar)
            if trade_date <= as_of
        ]
        missing_days = [
            trade_date
            for trade_date in open_dates
            if self.store.read_manifest(trade_date) is None
        ]
        recent_days = open_dates[-re_pull_days:] if re_pull_days > 0 else []
        selected = sorted(set(missing_days) | set(recent_days))
        return [self.sync_day(trade_date) for trade_date in selected]

    def _calendar_open_dates(self, calendar: pl.DataFrame) -> list[date]:
        if "cal_date" not in calendar.columns or "is_open" not in calendar.columns:
            raise SyncError("trade_cal is missing cal_date or is_open")
        calendar = _to_date(calendar, ["cal_date"])
        open_calendar = calendar.filter(pl.col("is_open") == 1)
        if open_calendar["cal_date"].null_count() > 0:
            raise SyncError(
                "trade_cal contains null cal_date values for open days"
            )
        return sorted(set(open_calendar["cal_date"].to_list()))

    def _check_calendar_coverage(
        self, calendar: pl.DataFrame, start: date, end: date
    ) -> None:
        if "cal_date" not in calendar.columns:
            raise SyncError("trade_cal is missing cal_date")
        earliest = calendar["cal_date"].min()
        latest = calendar["cal_date"].max()
        if not isinstance(earliest, date) or earliest > start:
            raise SyncError(
                f"trade_cal does not cover start boundary {start.isoformat()}"
            )
        if not isinstance(latest, date) or latest < end:
            raise SyncError(
                f"trade_cal does not cover end boundary {end.isoformat()}"
            )

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
        subset = [
            column
            for column in ("exchange", "cal_date")
            if column in merged.columns
        ]
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
