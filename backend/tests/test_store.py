from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.data.provider import ParquetMarketDataProvider
from app.data.store import ParquetStore
from app.data.tushare_client import TushareClient


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "raw", tmp_path / "processed", tmp_path / "regime")


def test_processed_roundtrip(store: ParquetStore):
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "close": [10.0, 20.0],
        }
    )
    store.write_processed("bars", date(2024, 1, 2), df)
    loaded = store.read_processed("bars", date(2024, 1, 2), date(2024, 1, 2))
    assert loaded.sort("ts_code").to_dicts() == df.sort("ts_code").to_dicts()


def test_reference_roundtrip(store: ParquetStore):
    df = pl.DataFrame({"ts_code": ["000001.SZ"], "market": ["主板"]})
    store.write_reference("stock_basic", df)
    loaded = store.read_reference("stock_basic")
    assert loaded.to_dicts() == df.to_dicts()


def test_tushare_client_retries():
    class FlakyPro:
        def __init__(self):
            self.calls = 0

        def daily(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary")
            return pl.DataFrame({"ts_code": ["000001.SZ"]}).to_pandas()

    pro = FlakyPro()
    client = TushareClient(pro=pro, sleep=lambda _: None, max_retries=2)
    result = client.fetch("daily", trade_date="20240102")
    assert pro.calls == 2
    assert result.height == 1


def test_provider_filters_stock_basic(store: ParquetStore):
    store.write_reference(
        "stock_basic",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "list_date": [date(2000, 1, 1), date(2024, 12, 1), date(2000, 1, 1)],
                "delist_date": [None, None, date(2023, 1, 1)],
            }
        ),
    )
    provider = ParquetMarketDataProvider(store)
    loaded = provider.load_stock_basic(date(2024, 1, 2))
    assert loaded["ts_code"].to_list() == ["000001.SZ"]
