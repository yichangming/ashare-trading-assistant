from datetime import date, timedelta

import polars as pl
import pytest

from app.core.calendar import TradingCalendar
from app.core.config import load_strategy_config
from app.core.errors import DataNotReadyError
from app.data.universe import UniverseBuilder


def test_universe_excludes_st_new_and_suspended():
    config = load_strategy_config()
    features = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "trade_date": [date(2024, 1, 2)] * 4,
            "sector": ["银行", "银行", "银行", "银行"],
            "close": [10.0, 20.0, 30.0, 40.0],
            "amount_20d_avg": [2e8, 2e8, 2e8, 2e8],
            "is_suspended": [False, False, True, False],
            "is_limit_up": [False, False, False, True],
        }
    )
    stocks = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "name": ["平安银行", "ST万科", "浦发银行", "招商银行"],
            "market": ["主板", "主板", "主板", "主板"],
            "industry": ["银行"] * 4,
            "list_date": [date(2000, 1, 1), date(2000, 1, 1), date(2000, 1, 1), date(2024, 1, 1)],
            "delist_date": [None, None, None, None],
        }
    )
    calendar = TradingCalendar(
        [date(2023, 1, 1) + timedelta(days=i) for i in range(400)]
    )
    universe = UniverseBuilder(calendar).build(
        date(2024, 1, 2), features, stocks, pl.DataFrame(), config
    )
    assert universe["ts_code"].to_list() == ["000001.SZ"]


def _calendar() -> TradingCalendar:
    return TradingCalendar(
        [date(2023, 1, 1) + timedelta(days=i) for i in range(400)]
    )


def _weekday_calendar(days: int) -> tuple[TradingCalendar, list[date]]:
    dates: list[date] = []
    current = date(2023, 1, 2)
    while len(dates) < days:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return TradingCalendar(dates), dates


def _features_frame(
    codes: list[str],
    *,
    trade_date: date,
    close: list[float] | None = None,
    amount: list[float] | None = None,
    suspended: list[bool] | None = None,
) -> pl.DataFrame:
    size = len(codes)
    return pl.DataFrame(
        {
            "ts_code": codes,
            "trade_date": [trade_date] * size,
            "close": close if close is not None else [10.0] * size,
            "amount_20d_avg": amount if amount is not None else [2e8] * size,
            "is_suspended": (
                suspended if suspended is not None else [False] * size
            ),
        }
    )


def _stocks_frame(
    codes: list[str],
    *,
    names: list[str] | None = None,
    markets: list[str] | None = None,
    list_dates: list[date] | None = None,
    delist_dates: list[date | None] | None = None,
) -> pl.DataFrame:
    size = len(codes)
    return pl.DataFrame(
        {
            "ts_code": codes,
            "name": names if names is not None else ["正常公司"] * size,
            "market": markets if markets is not None else ["主板"] * size,
            "industry": ["银行"] * size,
            "list_date": (
                list_dates
                if list_dates is not None
                else [date(2000, 1, 1)] * size
            ),
            "delist_date": (
                delist_dates if delist_dates is not None else [None] * size
            ),
        }
    )


def test_universe_uses_point_in_time_st_history():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    codes = ["600401.SH", "600402.SH", "600403.SH"]
    features = pl.DataFrame(
        {
            "ts_code": codes,
            "trade_date": [as_of] * 3,
            "close": [10.0, 10.0, 10.0],
            "amount_20d_avg": [2e8, 2e8, 2e8],
            "is_suspended": [False, False, False],
        }
    )
    stocks = pl.DataFrame(
        {
            "ts_code": codes,
            "name": ["退市海润", "正常公司", "ST某某"],
            "market": ["主板", "主板", "主板"],
            "industry": ["银行", "银行", "银行"],
            "list_date": [date(2000, 1, 1)] * 3,
            "delist_date": [None, None, None],
        }
    )
    namechange = pl.DataFrame(
        {
            "ts_code": ["600401.SH", "600402.SH", "600403.SH"],
            "name": ["海润光伏", "ST某某", "ST某某"],
            "start_date": [date(2022, 1, 1), date(2022, 1, 1), date(2020, 1, 1)],
            "end_date": [None, None, date(2021, 1, 1)],
        }
    )
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, stocks, namechange, config
    )
    # 600401: active historical non-ST interval overrides the current
    # "退市海润" name. 600402: active historical ST interval excludes it.
    # 600403: history exists but no active interval, so it is non-ST and the
    # current name is not consulted.
    assert universe["ts_code"].to_list() == ["600401.SH", "600403.SH"]


def test_universe_falls_back_to_current_name_without_namechange():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    features = pl.DataFrame(
        {
            "ts_code": ["000002.SZ"],
            "trade_date": [as_of],
            "close": [10.0],
            "amount_20d_avg": [2e8],
            "is_suspended": [False],
        }
    )
    stocks = pl.DataFrame(
        {
            "ts_code": ["000002.SZ"],
            "name": ["ST万科"],
            "market": ["主板"],
            "industry": ["银行"],
            "list_date": [date(2000, 1, 1)],
            "delist_date": [None],
        }
    )
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, stocks, pl.DataFrame(), config
    )
    assert universe.is_empty()


def test_universe_board_amount_and_price_filters():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    codes = [
        "600030.SH",
        "600031.SH",
        "600032.SH",
        "600033.SH",
        "600034.SH",
        "600035.SH",
    ]
    features = _features_frame(
        codes,
        trade_date=as_of,
        close=[10.0, 10.0, 10.0, 10.0, 2.99, 3.00],
        amount=[2e8, 2e8, 99_999_999.99, 100_000_000.0, 2e8, 2e8],
    )
    stocks = _stocks_frame(
        codes,
        markets=["创业板", "科创板", "主板", "主板", "主板", "主板"],
    )
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, stocks, pl.DataFrame(), config
    )
    assert universe["ts_code"].to_list() == [
        "600030.SH",
        "600033.SH",
        "600035.SH",
    ]


def test_universe_ignores_namechange_starting_after_as_of():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    features = _features_frame(["600040.SH"], trade_date=as_of)
    stocks = _stocks_frame(["600040.SH"], names=["ST某某"])
    namechange = pl.DataFrame(
        {
            "ts_code": ["600040.SH"],
            "name": ["ST某某"],
            "start_date": [as_of + timedelta(days=1)],
            "end_date": [None],
        }
    )
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, stocks, namechange, config
    )
    assert universe["ts_code"].to_list() == ["600040.SH"]


def test_universe_dedupes_stock_rows():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    features = _features_frame(["600001.SH"], trade_date=as_of)
    stocks = _stocks_frame(["600001.SH"])
    duplicated = pl.concat([stocks, stocks])
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, duplicated, pl.DataFrame(), config
    )
    assert universe.height == 1


def test_universe_empty_features_preserves_schema():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    features = _features_frame(["600001.SH"], trade_date=as_of).head(0)
    stocks = _stocks_frame(["600001.SH"])
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, stocks, pl.DataFrame(), config
    )
    assert universe.is_empty()
    assert universe.columns == [*features.columns, "listing_days", "is_st"]
    assert universe.schema["listing_days"] == pl.Int64
    assert universe.schema["is_st"] == pl.Boolean


def test_universe_filters_features_to_as_of():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    stale = _features_frame(
        ["600001.SH"], trade_date=as_of - timedelta(days=1)
    )
    current = _features_frame(["600001.SH"], trade_date=as_of)
    features = pl.concat([stale, current])
    stocks = _stocks_frame(["600001.SH"])
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, stocks, pl.DataFrame(), config
    )
    assert universe.height == 1
    assert universe["trade_date"].to_list() == [as_of]


def test_universe_respects_delist_date():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    codes = ["600010.SH", "600011.SH", "600012.SH"]
    features = _features_frame(codes, trade_date=as_of)
    stocks = _stocks_frame(
        codes,
        delist_dates=[as_of, as_of + timedelta(days=1), None],
    )
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, stocks, pl.DataFrame(), config
    )
    assert universe["ts_code"].to_list() == ["600011.SH", "600012.SH"]


def test_universe_listing_day_boundary():
    config = load_strategy_config()
    calendar, dates = _weekday_calendar(130)
    as_of = dates[-1]
    codes = ["600020.SH", "600021.SH"]
    features = _features_frame(codes, trade_date=as_of)
    stocks = _stocks_frame(
        codes, list_dates=[dates[-120], dates[-119]]
    )
    universe = UniverseBuilder(calendar).build(
        as_of, features, stocks, pl.DataFrame(), config
    )
    assert universe["ts_code"].to_list() == ["600020.SH"]


def test_universe_requires_calendar_coverage():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    features = _features_frame(["600001.SH"], trade_date=as_of)
    stocks = _stocks_frame(["600001.SH"])
    with pytest.raises(DataNotReadyError, match="calendar"):
        UniverseBuilder(TradingCalendar([as_of])).build(
            as_of, features, stocks, pl.DataFrame(), config
        )
    with pytest.raises(DataNotReadyError, match="calendar"):
        UniverseBuilder(TradingCalendar([])).build(
            as_of, features, stocks, pl.DataFrame(), config
        )


def test_universe_parses_string_namechange_dates():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    features = _features_frame(["600050.SH"], trade_date=as_of)
    stocks = _stocks_frame(["600050.SH"])
    namechange = pl.DataFrame(
        {
            "ts_code": ["600050.SH"],
            "name": ["ST某某"],
            "start_date": ["20220101"],
            "end_date": ["20241231"],
        }
    )
    universe = UniverseBuilder(_calendar()).build(
        as_of, features, stocks, namechange, config
    )
    assert universe.is_empty()


def test_universe_rejects_namechange_missing_end_date():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    features = _features_frame(["600050.SH"], trade_date=as_of)
    stocks = _stocks_frame(["600050.SH"])
    namechange = pl.DataFrame(
        {
            "ts_code": ["600050.SH"],
            "name": ["ST某某"],
            "start_date": [date(2022, 1, 1)],
        }
    )
    with pytest.raises(ValueError, match="end_date"):
        UniverseBuilder(_calendar()).build(
            as_of, features, stocks, namechange, config
        )
