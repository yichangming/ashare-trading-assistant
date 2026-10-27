from datetime import date, timedelta

import polars as pl

from app.core.calendar import TradingCalendar
from app.core.config import load_strategy_config
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
