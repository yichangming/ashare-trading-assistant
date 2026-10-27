import json
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


def test_provider_normalizes_tushare_string_dates(store: ParquetStore):
    store.write_reference(
        "stock_basic",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "list_date": ["20000407", "20240102", "20000101", "2024-01-01"],
                "delist_date": [None, None, "20230101", None],
            }
        ),
    )
    provider = ParquetMarketDataProvider(store)
    loaded = provider.load_stock_basic(date(2024, 1, 2))
    assert loaded["ts_code"].to_list() == ["000001.SZ", "000002.SZ", "000004.SZ"]
    assert loaded.schema["list_date"] == pl.Date
    assert loaded.schema["delist_date"] == pl.Date


def test_provider_normalizes_trade_cal_string_dates(store: ParquetStore):
    store.write_reference(
        "trade_cal",
        pl.DataFrame(
            {
                "cal_date": ["20240101", "20240102", "20240103"],
                "is_open": [0, 1, 1],
            }
        ),
    )
    provider = ParquetMarketDataProvider(store)
    loaded = provider.load_trade_cal(date(2024, 1, 2), date(2024, 1, 2))
    assert loaded.to_dicts() == [{"cal_date": date(2024, 1, 2), "is_open": 1}]
    assert loaded.schema["cal_date"] == pl.Date


def test_provider_load_suspend_range(store: ParquetStore):
    store.write_raw(
        "suspend_d",
        "20240102",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": ["20240102", "20240102"],
                "suspend_type": ["S", "R"],
            }
        ),
    )
    store.write_raw(
        "suspend_d",
        "20240103",
        pl.DataFrame(
            {
                "ts_code": ["000003.SZ"],
                "trade_date": ["20240103"],
                "suspend_type": ["S"],
            }
        ),
    )
    provider = ParquetMarketDataProvider(store)
    loaded = provider.load_suspend(date(2024, 1, 2), date(2024, 1, 3))
    assert loaded is not None
    assert loaded.sort("ts_code")["ts_code"].to_list() == [
        "000001.SZ",
        "000003.SZ",
    ]
    assert loaded.schema["trade_date"] == pl.Date
    assert (loaded["trade_date"] == date(2024, 1, 2)).sum() == 1


def test_provider_load_suspend_no_partition_returns_none(store: ParquetStore):
    loaded = ParquetMarketDataProvider(store).load_suspend(
        date(2024, 1, 2), date(2024, 1, 2)
    )
    assert loaded is None


def test_provider_load_suspend_present_but_no_s_rows_is_typed_empty(
    store: ParquetStore,
):
    # An R-only partition still proves the endpoint was synced.
    store.write_raw(
        "suspend_d",
        "20240102",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240102"],
                "suspend_type": ["R"],
            }
        ),
    )
    loaded = ParquetMarketDataProvider(store).load_suspend(
        date(2024, 1, 2), date(2024, 1, 2)
    )
    assert loaded is not None
    assert loaded.is_empty()
    assert loaded.schema["trade_date"] == pl.Date
    assert loaded.schema["ts_code"] == pl.String


def test_provider_load_suspend_empty_partition_is_typed_empty(
    store: ParquetStore,
):
    store.write_raw(
        "suspend_d",
        "20240102",
        pl.DataFrame(
            schema={
                "ts_code": pl.String,
                "trade_date": pl.String,
                "suspend_type": pl.String,
            }
        ),
    )
    loaded = ParquetMarketDataProvider(store).load_suspend(
        date(2024, 1, 2), date(2024, 1, 2)
    )
    assert loaded is not None
    assert loaded.is_empty()
    assert loaded.schema["trade_date"] == pl.Date


def test_processed_multi_partition_range(store: ParquetStore):
    store.write_processed(
        "bars",
        date(2024, 1, 2),
        pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2)],
                "ts_code": ["000001.SZ"],
                "close": [10.0],
            }
        ),
    )
    store.write_processed(
        "bars",
        date(2024, 1, 3),
        pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 3)],
                "ts_code": ["000002.SZ"],
                "close": [20.0],
                "limit_pct": [0.1],
            }
        ),
    )
    store.write_processed(
        "bars",
        date(2024, 1, 4),
        pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 4)],
                "ts_code": ["000003.SZ"],
                "close": [30.0],
            }
        ),
    )

    loaded = store.read_processed("bars", date(2024, 1, 2), date(2024, 1, 3))

    assert loaded.height == 2
    assert set(loaded["ts_code"].to_list()) == {"000001.SZ", "000002.SZ"}
    assert set(loaded["trade_date"].to_list()) == {date(2024, 1, 2), date(2024, 1, 3)}
    assert "limit_pct" in loaded.columns
    day_one = loaded.filter(pl.col("ts_code") == "000001.SZ").row(0, named=True)
    day_two = loaded.filter(pl.col("ts_code") == "000002.SZ").row(0, named=True)
    assert day_one["limit_pct"] is None
    assert day_two["limit_pct"] == pytest.approx(0.1)


def test_processed_rejects_partition_date_mismatch(store: ParquetStore):
    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["000001.SZ", "000002.SZ"],
        }
    )
    with pytest.raises(ValueError, match="must match partition date"):
        store.write_processed("bars", date(2024, 1, 2), frame)


def test_processed_requires_date_dtype(store: ParquetStore):
    frame = pl.DataFrame(
        {"trade_date": ["20240102"], "ts_code": ["000001.SZ"]}
    )
    with pytest.raises(ValueError, match="dtype Date"):
        store.write_processed("bars", date(2024, 1, 2), frame)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_retries": 0}, "max_retries must be at least 1"),
        ({"rate_limit_per_min": 0}, "rate_limit_per_min must be at least 1"),
    ],
)
def test_tushare_client_rejects_invalid_config(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        TushareClient(pro=object(), **kwargs)


def test_tushare_client_returns_empty_frame_for_none():
    class EmptyPro:
        def daily(self, **kwargs):
            return None

    client = TushareClient(pro=EmptyPro(), sleep=lambda _: None)
    result = client.fetch("daily", trade_date="20240102")
    assert result.is_empty()


def test_tushare_client_raises_after_retry_exhaustion():
    class BrokenPro:
        def __init__(self):
            self.calls = 0

        def daily(self, **kwargs):
            self.calls += 1
            raise RuntimeError("still broken")

    pro = BrokenPro()
    client = TushareClient(pro=pro, sleep=lambda _: None, max_retries=3)
    with pytest.raises(RuntimeError, match="still broken"):
        client.fetch("daily", trade_date="20240102")
    assert pro.calls == 3


def test_regime_roundtrip(store: ParquetStore):
    path = store.write_regime(
        date(2024, 1, 2),
        {"stage": "start", "as_of": date(2024, 1, 2), "risk_factor": 0.8},
    )
    assert path == store.regime_dir / "trade_date=2024-01-02.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "stage": "start",
        "as_of": "2024-01-02",
        "risk_factor": 0.8,
    }


def test_atomic_write_preserves_existing_file_on_failure(
    store: ParquetStore, monkeypatch: pytest.MonkeyPatch
):
    path = store.write_reference(
        "stock_basic", pl.DataFrame({"ts_code": ["000001.SZ"]})
    )

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("app.data.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.write_reference(
            "stock_basic", pl.DataFrame({"ts_code": ["000002.SZ"]})
        )

    assert pl.read_parquet(path).to_dicts() == [{"ts_code": "000001.SZ"}]
    assert list(path.parent.glob(".stock_basic.parquet.*.tmp")) == []
