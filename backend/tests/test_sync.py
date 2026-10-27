from datetime import date
from pathlib import Path

import pytest

from app.data.manifest import DataManifest
from app.data.store import ParquetStore
from app.data.sync import SyncService
from app.data.tushare_client import TushareClient
from tests.fixtures.fake_tushare import FakePro


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "raw", tmp_path / "processed", tmp_path / "regime")


def test_manifest_version_is_stable():
    manifest = DataManifest(
        trade_date=date(2024, 1, 2),
        row_counts={"daily": 2},
        checksums={"daily": "abc"},
    )
    assert manifest.version() == manifest.version()
    assert len(manifest.version()) == 12


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
    results = service.backfill(date(2024, 1, 2), date(2024, 1, 3))
    assert [r.trade_date for r in results] == [date(2024, 1, 2), date(2024, 1, 3)]
