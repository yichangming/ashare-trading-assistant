from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.core.calendar import TradingCalendar
from app.core.dates import parse_date
from app.core.errors import DataNotReadyError
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


def _write_case(
    store: ParquetStore,
    *,
    markets: list[str],
    names: list[str],
    pre_closes: list[float],
    volumes: list[float] | None = None,
    amounts: list[float] | None = None,
    day: str = "20240102",
    namechange: pl.DataFrame | None = None,
    suspend: pl.DataFrame | None = None,
) -> list[str]:
    codes = [f"00000{index}.SZ" for index in range(1, len(markets) + 1)]
    count = len(codes)
    if volumes is None:
        volumes = [100000.0] * count
    if amounts is None:
        amounts = [1000000.0] * count
    store.write_raw(
        "daily",
        day,
        pl.DataFrame(
            {
                "ts_code": codes,
                "trade_date": [day] * count,
                "open": pre_closes,
                "high": pre_closes,
                "low": pre_closes,
                "close": pre_closes,
                "pre_close": pre_closes,
                "vol": volumes,
                "amount": amounts,
            }
        ),
    )
    store.write_raw(
        "adj_factor",
        day,
        pl.DataFrame(
            {
                "ts_code": codes,
                "trade_date": [day] * count,
                "adj_factor": [1.0] * count,
            }
        ),
    )
    store.write_raw(
        "daily_basic",
        day,
        pl.DataFrame(
            {
                "ts_code": codes,
                "trade_date": [day] * count,
                "turnover_rate": [1.0] * count,
                "circ_mv": [100000.0] * count,
            }
        ),
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
                "ts_code": codes,
                "name": names,
                "market": markets,
                "list_date": [date(2000, 1, 1)] * count,
                "delist_date": [None] * count,
            }
        ),
    )
    if namechange is not None:
        store.write_reference("namechange", namechange)
    if suspend is not None:
        store.write_raw("suspend_d", day, suspend)
    return codes


def test_parse_date():
    assert parse_date("20240102") == date(2024, 1, 2)
    assert parse_date(date(2024, 1, 2)) == date(2024, 1, 2)


def test_calendar_add_trading_days():
    calendar = TradingCalendar(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    )
    assert calendar.add_trading_days(date(2024, 1, 2), 2) == date(2024, 1, 4)
    assert calendar.count_between(date(2024, 1, 2), date(2024, 1, 3)) == 2


def test_calendar_add_trading_days_beyond_end_raises():
    calendar = TradingCalendar([date(2024, 1, 2), date(2024, 1, 3)])
    with pytest.raises(ValueError, match="calendar coverage"):
        calendar.add_trading_days(date(2024, 1, 3), 1)


def test_calendar_rejects_negative_count():
    calendar = TradingCalendar([date(2024, 1, 2)])
    with pytest.raises(ValueError, match="non-negative"):
        calendar.add_trading_days(date(2024, 1, 2), -1)


def test_calendar_dedupes_and_sorts_dates():
    calendar = TradingCalendar(
        [date(2024, 1, 3), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    )
    assert calendar.dates == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    assert calendar.add_trading_days(date(2024, 1, 3), 1) == date(2024, 1, 4)


def test_assemble_day(store: ParquetStore):
    _write_raw(store)
    assembler = DataAssembler(store)
    bars = assembler.build_day(date(2024, 1, 2)).sort("ts_code")
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
    bars = assembler.build_day(date(2024, 1, 2)).sort("ts_code")
    assert bars["limit_derived"].to_list() == [True, True]
    assert bars.filter(pl.col("ts_code") == "000001.SZ")["up_limit"][0] == 11.0


def test_assembler_converts_units(store: ParquetStore):
    _write_raw(store)
    bars = DataAssembler(store).build_day(date(2024, 1, 2)).sort("ts_code")
    assert bars["amount"].to_list() == [1020000000.0, 4040000000.0]
    assert bars["turnover_rate"].to_list() == [0.015, 0.02]
    assert bars["circ_mv"].to_list() == [10000000000.0, 20000000000.0]
    index_bars = store.read_processed(
        "index_bars", date(2024, 1, 2), date(2024, 1, 2)
    )
    assert index_bars["amount"].to_list() == [2000000.0]


def test_date_normalization_accepts_date_and_iso_strings(store: ParquetStore):
    _write_raw(store)
    day = "20240102"
    daily = store.read_raw("daily", day).with_columns(
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("trade_date")
    )
    store.write_raw("daily", day, daily)
    adj = store.read_raw("adj_factor", day).with_columns(
        pl.lit("2024-01-02").alias("trade_date")
    )
    store.write_raw("adj_factor", day, adj)
    bars = DataAssembler(store).build_day(date(2024, 1, 2))
    assert bars.height == 2
    assert bars.schema["trade_date"] == pl.Date


@pytest.mark.parametrize(
    ("pre_close", "expected_up", "expected_down"),
    [(1.15, 1.27, 1.04), (0.95, 1.05, 0.86)],
)
def test_derived_limits_round_half_up(
    store: ParquetStore, pre_close: float, expected_up: float, expected_down: float
):
    _write_case(store, markets=["主板"], names=["正常公司"], pre_closes=[pre_close])
    bars = DataAssembler(store).build_day(date(2024, 1, 2))
    assert bars["up_limit"][0] == expected_up
    assert bars["down_limit"][0] == expected_down


def test_derived_limits_follow_board_and_st_rules(store: ParquetStore):
    _write_case(
        store,
        markets=["主板", "主板", "创业板", "科创板", "北交所"],
        names=["正常公司", "ST测试", "正常公司", "正常公司", "正常公司"],
        pre_closes=[10.0, 10.0, 10.0, 10.0, 10.0],
    )
    bars = DataAssembler(store).build_day(date(2024, 1, 2)).sort("ts_code")
    assert bars["up_limit"].to_list() == [11.0, 10.5, 12.0, 12.0, 13.0]
    assert bars["down_limit"].to_list() == [9.0, 9.5, 8.0, 8.0, 7.0]


@pytest.mark.parametrize("historical_name", ["ST测试", "退市测试"])
def test_derived_limits_use_point_in_time_st(
    store: ParquetStore, historical_name: str
):
    namechange = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": [historical_name],
            "start_date": [date(2023, 1, 1)],
            "end_date": [date(2024, 6, 1)],
        }
    )
    _write_case(
        store,
        markets=["主板"],
        names=["正常公司"],
        pre_closes=[10.0],
        namechange=namechange,
    )
    bars = DataAssembler(store).build_day(date(2024, 1, 2))
    assert bars["up_limit"][0] == 10.5
    assert bars["down_limit"][0] == 9.5


def test_suspension_uses_only_s_events(store: ParquetStore):
    suspend = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20240102", "20240102"],
            "suspend_type": ["S", "R"],
        }
    )
    _write_case(
        store,
        markets=["主板", "主板"],
        names=["正常公司", "正常公司"],
        pre_closes=[10.0, 10.0],
        suspend=suspend,
    )
    bars = DataAssembler(store).build_day(date(2024, 1, 2)).sort("ts_code")
    assert bars["is_suspended"].to_list() == [True, False]


def test_suspension_timing_marks_only_full_day_rows(store: ParquetStore):
    suspend = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "trade_date": ["20240102"] * 3,
            "suspend_type": ["S", "S", "S"],
            "suspend_timing": ["09:30-10:30", None, ""],
        }
    )
    _write_case(
        store,
        markets=["主板", "主板", "主板"],
        names=["正常公司", "正常公司", "正常公司"],
        pre_closes=[10.0, 10.0, 10.0],
        suspend=suspend,
    )
    bars = DataAssembler(store).build_day(date(2024, 1, 2)).sort("ts_code")
    # The partial-day name still traded; null/empty timing is a full-day halt.
    assert bars["is_suspended"].to_list() == [False, True, True]


def test_suspension_includes_zero_volume_or_amount(store: ParquetStore):
    _write_case(
        store,
        markets=["主板", "主板", "主板"],
        names=["正常公司", "正常公司", "正常公司"],
        pre_closes=[10.0, 10.0, 10.0],
        volumes=[0.0, 100000.0, 100000.0],
        amounts=[1000000.0, 0.0, 1000000.0],
    )
    bars = DataAssembler(store).build_day(date(2024, 1, 2)).sort("ts_code")
    assert bars["is_suspended"].to_list() == [True, True, False]


def test_missing_adj_factor_row_stays_null(store: ParquetStore):
    _write_raw(store)
    adj = store.read_raw("adj_factor", "20240102").filter(
        pl.col("ts_code") == "000001.SZ"
    )
    store.write_raw("adj_factor", "20240102", adj)
    bars = DataAssembler(store).build_day(date(2024, 1, 2)).sort("ts_code")
    assert bars.filter(pl.col("ts_code") == "000001.SZ")["adj_factor"][0] == 1.0
    assert bars.filter(pl.col("ts_code") == "000002.SZ")["adj_factor"][0] is None


@pytest.mark.parametrize("endpoint", ["adj_factor", "daily_basic", "index_daily"])
@pytest.mark.parametrize("empty_frame", [False, True])
def test_missing_or_empty_critical_raw_input_raises(
    store: ParquetStore, endpoint: str, empty_frame: bool
):
    _write_raw(store)
    if empty_frame:
        store.write_raw(endpoint, "20240102", pl.DataFrame({"ts_code": []}))
    else:
        store.remove_raw(endpoint, "20240102")
    with pytest.raises(DataNotReadyError, match=endpoint):
        DataAssembler(store).build_day(date(2024, 1, 2))
    assert store.read_processed(
        "bars", date(2024, 1, 2), date(2024, 1, 2)
    ).is_empty()


@pytest.mark.parametrize("empty_frame", [False, True])
def test_missing_or_empty_stock_basic_raises(
    store: ParquetStore, empty_frame: bool
):
    _write_raw(store)
    if empty_frame:
        store.write_reference("stock_basic", pl.DataFrame({"ts_code": []}))
    else:
        store.remove_reference("stock_basic")
    with pytest.raises(DataNotReadyError, match="stock_basic"):
        DataAssembler(store).build_day(date(2024, 1, 2))
    assert store.read_processed(
        "bars", date(2024, 1, 2), date(2024, 1, 2)
    ).is_empty()


def test_stock_basic_missing_columns_raises(store: ParquetStore):
    _write_raw(store)
    store.write_reference(
        "stock_basic",
        pl.DataFrame({"ts_code": ["000001.SZ"], "name": ["平安银行"]}),
    )
    with pytest.raises(DataNotReadyError, match="market"):
        DataAssembler(store).build_day(date(2024, 1, 2))


def test_bars_schema_matches_with_and_without_stk_limit(store: ParquetStore):
    _write_raw(store)
    assembler = DataAssembler(store)
    limits = store.read_raw("stk_limit", "20240102")
    store.remove_raw("stk_limit", "20240102")
    derived = assembler.build_day(date(2024, 1, 2))
    store.write_raw("stk_limit", "20240102", limits)
    provided = assembler.build_day(date(2024, 1, 2))

    assert sorted(derived.columns) == sorted(provided.columns)
    assert dict(derived.schema) == dict(provided.schema)
    for frame in (derived, provided):
        assert {"up_limit", "down_limit", "limit_derived"} <= set(frame.columns)
        assert not {"name", "market", "limit_pct"} & set(frame.columns)
