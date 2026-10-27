import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from app.data.manifest import DataManifest
from app.data.store import ParquetStore
from app.data.sync import SyncError, SyncService
from app.data.tushare_client import TushareClient
from tests.fixtures.fake_tushare import FakePro, RangeAwareFakePro


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "raw", tmp_path / "processed", tmp_path / "regime")


def test_manifest_version_is_stable():
    first = DataManifest(
        trade_date=date(2024, 1, 2),
        row_counts={"daily": 2},
        checksums={"daily": "abc"},
        created_at=datetime(2024, 1, 2, tzinfo=UTC),
    )
    later = DataManifest(
        trade_date=date(2024, 1, 2),
        row_counts={"daily": 2},
        checksums={"daily": "abc"},
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
    )
    different = DataManifest(
        trade_date=date(2024, 1, 2),
        row_counts={"daily": 3},
        checksums={"daily": "abc"},
        created_at=datetime(2024, 1, 2, tzinfo=UTC),
    )
    assert first.version() == later.version()
    assert first.version() != different.version()
    assert len(first.version()) == 12


def test_sync_reference_normalizes_dates(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    counts = service.sync_reference(date(2024, 1, 1), date(2024, 1, 5))
    assert counts["trade_cal"] == 5
    frame = store.read_reference("trade_cal")
    assert str(frame.schema["cal_date"]) == "Date"
    stock_basic = store.read_reference("stock_basic")
    assert str(stock_basic.schema["list_date"]) == "Date"


def test_sync_day_writes_raw_and_manifest(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    result = service.sync_day(date(2024, 1, 2))
    assert result.row_counts["daily"] == 2
    assert store.read_raw("daily", "20240102").height == 2
    manifest_path = store.raw_dir / "manifests" / "20240102.json"
    assert manifest_path.exists()
    assert "data_version" in manifest_path.read_text(encoding="utf-8")


def test_optional_endpoint_failure_degrades(store: ParquetStore):
    pro = FakePro(fail_endpoints={"limit_list_d"})
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    result = service.sync_day(date(2024, 1, 2))
    assert any("limit_list_d" in item for item in result.degraded)
    assert store.read_raw("daily", "20240102").height == 2


def test_backfill_skips_non_trading_days(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    results = service.backfill(date(2024, 1, 1), date(2024, 1, 3))
    assert [r.trade_date for r in results] == [date(2024, 1, 2), date(2024, 1, 3)]


def test_required_fetch_failure_leaves_no_manifest_and_retry_succeeds(
    store: ParquetStore,
):
    pro = FakePro(fail_endpoints={"stk_limit"})
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)

    with pytest.raises(SyncError, match="stk_limit"):
        service.sync_day(date(2024, 1, 2))
    assert store.read_manifest(date(2024, 1, 2)) is None
    assert not store.raw_path("daily", "20240102").exists()

    pro.fail_endpoints.clear()
    service.sync_day(date(2024, 1, 2))

    assert store.read_manifest(date(2024, 1, 2)) is not None


def test_failed_forced_resync_invalidates_previous_manifest(store: ParquetStore):
    pro = FakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_day(date(2024, 1, 2))
    assert store.read_manifest(date(2024, 1, 2)) is not None

    pro.fail_endpoints.add("adj_factor")
    with pytest.raises(SyncError, match="adj_factor"):
        service.backfill(date(2024, 1, 2), date(2024, 1, 2), force=True)

    assert store.read_manifest(date(2024, 1, 2)) is None


@pytest.mark.parametrize(
    "endpoint",
    ["daily", "adj_factor", "daily_basic", "stk_limit", "index_daily"],
)
def test_empty_required_endpoint_raises(store: ParquetStore, endpoint: str):
    pro = FakePro(empty_endpoints={endpoint})
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)

    with pytest.raises(SyncError, match=f"{endpoint} returned no rows"):
        service.sync_day(date(2024, 1, 2))

    assert store.read_manifest(date(2024, 1, 2)) is None
    assert not store.raw_path("daily", "20240102").exists()


def test_limit_list_d_failure_removes_stale_raw(store: ParquetStore):
    pro = FakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_day(date(2024, 1, 2))
    assert store.raw_path("limit_list_d", "20240102").exists()

    pro.fail_endpoints.add("limit_list_d")
    result = service.sync_day(date(2024, 1, 2))

    assert any("limit_list_d" in item for item in result.degraded)
    assert "limit_list_d" not in result.row_counts
    assert not store.raw_path("limit_list_d", "20240102").exists()
    manifest = store.read_manifest(date(2024, 1, 2))
    assert manifest is not None
    assert any("limit_list_d" in item for item in manifest["degraded"])


def test_reference_fetch_failure_degrades_and_removes_stale_reference(
    store: ParquetStore,
):
    store.write_reference(
        "index_member",
        pl.DataFrame({"index_code": ["stale"], "con_code": ["000001.SZ"]}),
    )
    pro = FakePro(fail_endpoints={"index_member"})
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)

    counts = service.sync_reference(date(2024, 1, 1), date(2024, 1, 5))

    assert counts["index_member"] == 0
    assert not store.reference_path("index_member").exists()
    assert any("index_member" in item for item in service.reference_degraded)

    result = service.sync_day(date(2024, 1, 2))
    assert any("index_member" in item for item in result.reference_degraded)
    manifest = store.read_manifest(date(2024, 1, 2))
    assert manifest is not None
    assert any("index_member" in item for item in manifest["reference_degraded"])


def test_manifest_round_trips_with_reference_checksums(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    service.sync_reference(date(2024, 1, 1), date(2024, 1, 5))
    service.sync_day(date(2024, 1, 2))

    payload = json.loads(
        store.manifest_path(date(2024, 1, 2)).read_text(encoding="utf-8")
    )
    manifest = DataManifest.model_validate(payload)

    assert manifest.row_counts["daily"] == 2
    assert "daily" in manifest.checksums
    assert all(len(value) == 64 for value in manifest.checksums.values())
    assert set(manifest.reference_checksums) == {
        "trade_cal",
        "stock_basic",
        "namechange",
        "index_classify",
        "index_member",
    }
    for name, digest in manifest.reference_checksums.items():
        expected = hashlib.sha256(store.reference_path(name).read_bytes()).hexdigest()
        assert digest == expected
    assert manifest.data_version is not None
    assert len(manifest.data_version) == 12
    assert manifest.data_version == manifest.version()


def test_backfill_retries_day_with_raw_data_but_no_manifest(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    service.sync_reference(date(2024, 1, 2), date(2024, 1, 2))
    store.write_raw(
        "daily",
        "20240102",
        pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240102"]}),
    )
    assert store.read_manifest(date(2024, 1, 2)) is None

    results = service.backfill(date(2024, 1, 2), date(2024, 1, 2))

    assert [r.trade_date for r in results] == [date(2024, 1, 2)]
    assert store.read_manifest(date(2024, 1, 2)) is not None


def test_backfill_skips_days_with_valid_manifest(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    service.sync_day(date(2024, 1, 2))

    results = service.backfill(date(2024, 1, 2), date(2024, 1, 3))

    assert [r.trade_date for r in results] == [date(2024, 1, 3)]


def test_read_manifest_returns_none_for_corrupt_json(store: ParquetStore):
    path = store.manifest_path(date(2024, 1, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert store.read_manifest(date(2024, 1, 2)) is None


def test_incremental_refreshes_stale_calendar_and_syncs_missing_days(
    store: ParquetStore,
):
    store.write_reference(
        "trade_cal",
        pl.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": [date(2024, 1, 1), date(2024, 1, 2)],
                "is_open": [0, 1],
                "pretrade_date": [None, None],
            }
        ),
    )
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    service.sync_day(date(2024, 1, 2))

    results = service.sync_incremental(date(2024, 1, 4), re_pull_days=1)

    assert [r.trade_date for r in results] == [date(2024, 1, 3), date(2024, 1, 4)]
    calendar = store.read_reference("trade_cal")
    assert calendar.height == 5
    assert calendar.schema["cal_date"] == pl.Date
    assert store.read_manifest(date(2024, 1, 3)) is not None
    assert store.read_manifest(date(2024, 1, 4)) is not None


def test_incremental_degrades_when_calendar_refresh_fails(store: ParquetStore):
    store.write_reference(
        "trade_cal",
        pl.DataFrame(
            {
                "exchange": ["SSE"] * 3,
                "cal_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "is_open": [0, 1, 1],
                "pretrade_date": [None, None, None],
            }
        ),
    )
    pro = FakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_day(date(2024, 1, 2))
    service.sync_day(date(2024, 1, 3))

    pro.fail_endpoints.add("trade_cal")
    results = service.sync_incremental(date(2024, 1, 4), re_pull_days=1)

    assert [r.trade_date for r in results] == [date(2024, 1, 3)]
    assert any("trade_cal" in item for item in service.reference_degraded)


def test_incremental_rejects_negative_re_pull_days(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    with pytest.raises(ValueError, match="re_pull_days"):
        service.sync_incremental(date(2024, 1, 4), re_pull_days=-1)


def test_sync_reference_chunks_trade_cal(store: ParquetStore):
    pro = FakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)

    counts = service.sync_reference(date(2018, 1, 1), date(2024, 1, 5))

    boundaries = [
        (kwargs["start_date"], kwargs["end_date"])
        for endpoint, kwargs in pro.calls
        if endpoint == "trade_cal"
    ]
    assert boundaries == [
        ("20180101", "20181231"),
        ("20190101", "20191231"),
        ("20200101", "20201231"),
        ("20210101", "20211231"),
        ("20220101", "20221231"),
        ("20230101", "20231231"),
        ("20240101", "20240105"),
    ]
    assert counts["trade_cal"] == 5


def test_sync_reference_fetches_stock_basic_per_status(store: ParquetStore):
    pro = FakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)

    service.sync_reference(date(2024, 1, 1), date(2024, 1, 5))

    statuses = [
        kwargs["list_status"]
        for endpoint, kwargs in pro.calls
        if endpoint == "stock_basic"
    ]
    assert statuses == ["L", "D", "P"]
    assert store.read_reference("stock_basic").height == 2


def test_sync_reference_preserves_calendar_history(store: ParquetStore):
    service = SyncService(
        TushareClient(pro=RangeAwareFakePro(), sleep=lambda _: None), store
    )
    service.sync_reference(date(2024, 1, 1), date(2024, 1, 31))
    full_dates = set(store.read_reference("trade_cal")["cal_date"].to_list())
    assert len(full_dates) == 31

    counts = service.sync_reference(date(2024, 1, 15), date(2024, 1, 15))

    assert counts["trade_cal"] == 31
    assert set(store.read_reference("trade_cal")["cal_date"].to_list()) == full_dates

    service.backfill(date(2024, 1, 16), date(2024, 1, 16))

    assert set(store.read_reference("trade_cal")["cal_date"].to_list()) == full_dates


def test_incremental_fills_long_calendar_gap(store: ParquetStore):
    service = SyncService(
        TushareClient(pro=RangeAwareFakePro(), sleep=lambda _: None), store
    )
    service.sync_reference(date(2024, 1, 2), date(2024, 1, 2))
    assert set(store.read_reference("trade_cal")["cal_date"].to_list()) == {
        date(2024, 1, 2)
    }

    results = service.sync_incremental(date(2024, 3, 8), re_pull_days=1)

    expected_dates = {
        date(2024, 1, 2) + timedelta(days=offset) for offset in range(67)
    }
    expected_open = sorted(
        trade_date for trade_date in expected_dates if trade_date.weekday() < 5
    )
    calendar_dates = set(store.read_reference("trade_cal")["cal_date"].to_list())
    assert calendar_dates == expected_dates
    assert [result.trade_date for result in results] == expected_open
    for sample in (date(2024, 1, 3), date(2024, 2, 1), date(2024, 3, 1)):
        assert sample in calendar_dates
        assert store.read_manifest(sample) is not None


def test_incremental_refreshes_references_and_persists_degradation(
    store: ParquetStore,
):
    pro = RangeAwareFakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_reference(date(2024, 1, 2), date(2024, 1, 5))
    assert store.reference_path("index_member").exists()

    pro.fail_endpoints.add("index_member")
    results = service.sync_incremental(date(2024, 1, 4), re_pull_days=1)

    assert not store.reference_path("index_member").exists()
    assert any("index_member" in item for item in service.reference_degraded)
    assert results
    assert all(
        any("index_member" in item for item in result.reference_degraded)
        for result in results
    )
    manifest = store.read_manifest(date(2024, 1, 4))
    assert manifest is not None
    assert any("index_member" in item for item in manifest["reference_degraded"])


def test_incremental_requires_existing_calendar(store: ParquetStore):
    pro = FakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)

    with pytest.raises(SyncError, match="run sync_reference/backfill first"):
        service.sync_incremental(date(2024, 1, 4))

    assert pro.calls == []


def test_sync_reference_rejects_empty_stock_basic(store: ParquetStore):
    pro = FakePro(empty_endpoints={"stock_basic"})
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)

    with pytest.raises(SyncError, match="stock_basic returned no rows"):
        service.sync_reference(date(2024, 1, 1), date(2024, 1, 5))

    assert not store.reference_path("stock_basic").exists()
    assert not store.reference_path("trade_cal").exists()


def test_sync_reference_rejects_empty_trade_cal(store: ParquetStore):
    pro = FakePro(empty_endpoints={"trade_cal"})
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)

    with pytest.raises(SyncError, match="trade_cal returned no rows"):
        service.sync_reference(date(2024, 1, 1), date(2024, 1, 5))

    assert not store.reference_path("trade_cal").exists()


def test_read_manifest_rejects_schema_invalid_payload(store: ParquetStore):
    path = store.manifest_path(date(2024, 1, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"nonsense": true}', encoding="utf-8")

    assert store.read_manifest(date(2024, 1, 2)) is None


def test_calendar_with_null_open_date_raises_clear_error(store: ParquetStore):
    store.write_reference(
        "trade_cal",
        pl.DataFrame(
            {
                "exchange": ["SSE"],
                "cal_date": [None],
                "is_open": [1],
                "pretrade_date": [None],
            }
        ),
    )
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)

    with pytest.raises(SyncError, match="null cal_date"):
        service.backfill(date(2024, 1, 2), date(2024, 1, 3))


class TruncatedCalendarFake(RangeAwareFakePro):
    def trade_cal(self, **kwargs):
        kwargs["end_date"] = min(kwargs["end_date"], "20240131")
        return super().trade_cal(**kwargs)


def test_backfill_fails_when_calendar_fetch_fails(store: ParquetStore):
    pro = RangeAwareFakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_reference(date(2024, 1, 1), date(2024, 1, 31))

    pro.fail_endpoints.add("trade_cal")
    with pytest.raises(SyncError, match="permission denied"):
        service.backfill(date(2024, 2, 1), date(2024, 2, 5))


def test_backfill_fails_on_empty_calendar_response(store: ParquetStore):
    pro = RangeAwareFakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_reference(date(2024, 1, 1), date(2024, 1, 31))

    pro.empty_endpoints.add("trade_cal")
    with pytest.raises(SyncError, match="trade_cal returned no rows"):
        service.backfill(date(2024, 2, 1), date(2024, 2, 5))


def test_backfill_fails_when_empty_response_leaves_partial_coverage(
    store: ParquetStore,
):
    pro = RangeAwareFakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_reference(date(2024, 1, 1), date(2024, 1, 31))

    pro.empty_endpoints.add("trade_cal")
    with pytest.raises(SyncError, match="trade_cal returned no rows"):
        service.backfill(date(2024, 1, 15), date(2024, 2, 5))


def test_backfill_fails_when_calendar_does_not_reach_end(store: ParquetStore):
    pro = TruncatedCalendarFake()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_reference(date(2024, 1, 1), date(2024, 1, 31))

    with pytest.raises(SyncError, match="end boundary"):
        service.backfill(date(2024, 1, 15), date(2024, 2, 5))


def test_incremental_remains_resilient_to_empty_calendar(store: ParquetStore):
    pro = FakePro()
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    service.sync_reference(date(2024, 1, 1), date(2024, 1, 5))

    pro.empty_endpoints.add("trade_cal")
    results = service.sync_incremental(date(2024, 1, 4), re_pull_days=1)

    assert [result.trade_date for result in results] == [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]
    assert any(
        "trade_cal returned no rows" in item
        for item in service.reference_degraded
    )
    manifest = store.read_manifest(date(2024, 1, 4))
    assert manifest is not None
    assert any(
        "trade_cal returned no rows" in item
        for item in manifest["reference_degraded"]
    )


def test_read_manifest_requires_data_version_and_matching_date(
    store: ParquetStore,
):
    manifest = DataManifest(
        trade_date=date(2024, 1, 2),
        row_counts={"daily": 2},
        checksums={"daily": "abc"},
    )
    payload = manifest.model_dump(mode="json")
    payload.pop("data_version")
    path = store.manifest_path(date(2024, 1, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert store.read_manifest(date(2024, 1, 2)) is None

    payload["data_version"] = "abcdef123456"
    payload["trade_date"] = "2024-01-03"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.read_manifest(date(2024, 1, 2)) is None

    payload["trade_date"] = "2024-01-02"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.read_manifest(date(2024, 1, 2)) is not None


def test_inverted_ranges_are_rejected(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    with pytest.raises(ValueError, match="start must be on or before end"):
        service.sync_reference(date(2024, 1, 5), date(2024, 1, 1))
    with pytest.raises(ValueError, match="start must be on or before end"):
        service.backfill(date(2024, 1, 5), date(2024, 1, 1))


def test_incremental_rejects_absurd_re_pull_days(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    with pytest.raises(ValueError, match="re_pull_days"):
        service.sync_incremental(date(2024, 1, 4), re_pull_days=10**9)


def test_duplicate_calendar_rows_sync_each_day_once(store: ParquetStore):
    store.write_reference(
        "trade_cal",
        pl.DataFrame(
            {
                "exchange": ["SSE", "SZSE", "SSE"],
                "cal_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3)],
                "is_open": [1, 1, 1],
                "pretrade_date": [None, None, None],
            }
        ),
    )
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)

    results = service.backfill(date(2024, 1, 2), date(2024, 1, 3), force=True)

    assert [result.trade_date for result in results] == [
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
