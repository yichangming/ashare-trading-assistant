from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.core.calendar import TradingCalendar
from app.core.dates import parse_date
from app.data.assemble import DataAssembler
from app.data.store import ParquetStore


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "raw", tmp_path / "processed", tmp_path / "regime")


def _write_raw(store: ParquetStore, day: str = "20240102") -> None:
    store.write_raw(
        "daily",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.8, 19.5],
                "close": [10.2, 20.2],
                "pre_close": [10.0, 20.0],
                "vol": [100000.0, 200000.0],
                "amount": [1020000.0, 4040000.0],
            }
        ),
    )
    store.write_raw(
        "adj_factor",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "adj_factor": [1.0, 2.0],
            }
        ),
    )
    store.write_raw(
        "daily_basic",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "turnover_rate": [1.5, 2.0],
                "circ_mv": [1000000.0, 2000000.0],
            }
        ),
    )
    store.write_raw(
        "stk_limit",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "up_limit": [11.0, 22.0],
                "down_limit": [9.0, 18.0],
            }
        ),
    )
    store.write_raw(
        "suspend_d",
        day,
        pl.DataFrame({"ts_code": [], "trade_date": [], "suspend_type": []}),
    )
    store.write_raw(
        "index_daily",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000300.SH"],
                "trade_date": [day],
                "open": [4000.0],
                "high": [4050.0],
                "low": [3980.0],
                "close": [4020.0],
                "vol": [1000.0],
                "amount": [2000.0],
            }
        ),
    )
    store.write_reference(
        "stock_basic",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "name": ["平安银行", "万科A"],
                "market": ["主板", "主板"],
                "list_date": [date(2000, 4, 7), date(1991, 1, 29)],
                "delist_date": [None, None],
            }
        ),
    )


def test_parse_date():
    assert parse_date("20240102") == date(2024, 1, 2)
    assert parse_date(date(2024, 1, 2)) == date(2024, 1, 2)


def test_calendar_add_trading_days():
    calendar = TradingCalendar(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    )
    assert calendar.add_trading_days(date(2024, 1, 2), 2) == date(2024, 1, 4)
    assert calendar.count_between(date(2024, 1, 2), date(2024, 1, 3)) == 2


def test_assemble_day(store: ParquetStore):
    _write_raw(store)
    assembler = DataAssembler(store)
    bars = assembler.build_day(date(2024, 1, 2))
    assert bars.height == 2
    assert bars.filter(pl.col("ts_code") == "000002.SZ")["adj_factor"][0] == 2.0
    assert bars["turnover_rate"].to_list() == [0.015, 0.02]
    processed = store.read_processed("bars", date(2024, 1, 2), date(2024, 1, 2))
    assert processed.height == 2
    index_bars = store.read_processed(
        "index_bars", date(2024, 1, 2), date(2024, 1, 2)
    )
    assert index_bars.height == 1


def test_assemble_derives_limits_when_missing(store: ParquetStore):
    _write_raw(store)
    (store.raw_dir / "stk_limit" / "20240102.parquet").unlink()
    assembler = DataAssembler(store)
    bars = assembler.build_day(date(2024, 1, 2))
    assert bars["limit_derived"].to_list() == [True, True]
    assert bars.filter(pl.col("ts_code") == "000001.SZ")["up_limit"][0] == 11.0
