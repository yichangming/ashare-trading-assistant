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
