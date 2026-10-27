from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.core.config import IndustryConfig, load_strategy_config
from app.data.industry_map import IndustryMap
from app.data.provider import ParquetMarketDataProvider
from app.data.store import ParquetStore
from app.regime.industry import IndustryStrengthService


def _history() -> pl.DataFrame:
    rows = []
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    for trade_date in dates:
        for index in range(5):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"S{index}",
                    "sector": "强",
                    "ret_20d": 0.20,
                    "ret_5d": 0.05,
                    "above_ma20": True,
                    "amount_5d_avg": 1.5e8,
                    "amount_20d_avg": 1.0e8,
                    "is_limit_up": index == 0,
                    "prev_is_limit_up": index < 2,
                    # Fixture correction: with the plan's uniform 9.9 both
                    # prior-limit-up stocks are red and red_rate would be 1.0.
                    # One of the two must not be red for the intended 0.5.
                    "prev_close": 10.1 if index == 1 else 9.9,
                    "close": 10.0,
                }
            )
        for index in range(5):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"W{index}",
                    "sector": "弱",
                    "ret_20d": 0.0,
                    "ret_5d": 0.0,
                    "above_ma20": False,
                    "amount_5d_avg": 1.0e8,
                    "amount_20d_avg": 1.0e8,
                    "is_limit_up": False,
                    "prev_is_limit_up": False,
                    "prev_close": 10.0,
                    "close": 10.0,
                }
            )
    return pl.DataFrame(rows)


def _feature_row(code: str, sector: str | None, **overrides: object) -> dict:
    row: dict = {
        "trade_date": date(2024, 1, 4),
        "ts_code": code,
        "sector": sector,
        "ret_20d": 0.0,
        "ret_5d": 0.0,
        "above_ma20": True,
        "amount_5d_avg": 1.0e8,
        "amount_20d_avg": 1.0e8,
        "is_limit_up": False,
        "prev_is_limit_up": False,
        "prev_close": 10.0,
        "close": 10.0,
    }
    row.update(overrides)
    return row


def _industry_config(**overrides: object) -> IndustryConfig:
    return load_strategy_config().industry.model_copy(update=overrides)


def test_industry_strength_ranks_and_persists():
    config = load_strategy_config()
    config.industry.top_n = 1
    sectors = IndustryStrengthService().compute(
        _history(), date(2024, 1, 4), config.industry
    )
    assert sectors[0].sector == "强"
    assert sectors[0].rank == 1
    assert sectors[0].persistent_days == 3
    assert sectors[1].persistent_days == 0
    assert sectors[0].limit_up_count == 1
    assert sectors[0].red_rate == 0.5


def test_industry_strength_empty_input_returns_empty():
    assert (
        IndustryStrengthService().compute(
            pl.DataFrame(), date(2024, 1, 4), _industry_config()
        )
        == []
    )


def test_industry_strength_handles_empty_industry_mapping():
    config = _industry_config(top_n=3, min_stocks=1)
    features = pl.DataFrame([_feature_row("A", "ignored")]).drop("sector")
    features = features.join(
        IndustryMap({}, "stock_basic", True).to_frame(),
        on="ts_code",
        how="left",
    )
    assert features["sector"].null_count() == features.height
    assert (
        IndustryStrengthService().compute(
            features, date(2024, 1, 4), config
        )
        == []
    )


def test_industry_strength_drops_null_and_unknown_sectors():
    config = _industry_config(top_n=1, min_stocks=1)
    frame = pl.DataFrame(
        [
            _feature_row("A", None, ret_20d=0.10),
            _feature_row("B", "", ret_20d=0.90),
            _feature_row("C", "未知", ret_20d=0.90),
            _feature_row("D", "银行", ret_20d=0.10),
        ]
    )
    sectors = IndustryStrengthService().compute(
        frame, date(2024, 1, 4), config
    )
    assert [strength.sector for strength in sectors] == ["银行"]
    assert sectors[0].rank == 1
    assert sectors[0].score == pytest.approx(100.0)


def test_industry_strength_excludes_suspended_rows():
    config = _industry_config(top_n=3, min_stocks=2)
    prior = date(2024, 1, 3)
    as_of = date(2024, 1, 4)
    rows: list[dict] = []
    for trade_date in (prior, as_of):
        rows.extend(
            [
                _feature_row(
                    "A",
                    "X",
                    trade_date=trade_date,
                    above_ma20=True,
                    amount_5d_avg=1.5e8,
                    is_limit_up=True,
                    prev_is_limit_up=True,
                    prev_close=9.9,
                    close=10.0,
                    is_suspended=False,
                ),
                _feature_row(
                    "B",
                    "X",
                    trade_date=trade_date,
                    above_ma20=False,
                    prev_is_limit_up=True,
                    prev_close=10.0,
                    close=10.0,
                    is_suspended=False,
                ),
            ]
        )
    rows.append(
        _feature_row(
            "C",
            "X",
            trade_date=prior,
            above_ma20=False,
            amount_5d_avg=0.0,
            is_limit_up=True,
            prev_is_limit_up=True,
            prev_close=9.9,
            close=9.0,
            is_suspended=False,
        )
    )
    rows.append(
        _feature_row(
            "C",
            "X",
            trade_date=as_of,
            above_ma20=False,
            amount_5d_avg=0.0,
            is_limit_up=True,
            prev_is_limit_up=True,
            prev_close=9.9,
            close=9.0,
            is_suspended=True,
        )
    )
    sectors = IndustryStrengthService().compute(
        pl.DataFrame(rows), as_of, config
    )
    assert len(sectors) == 1
    strength = sectors[0]
    assert strength.stock_count == 2
    assert strength.breadth == pytest.approx(0.5)
    assert strength.limit_up_count == 1
    assert strength.limit_up_share == pytest.approx(0.5)
    assert strength.red_rate == pytest.approx(0.5)
    assert strength.amount_change == pytest.approx(0.25)


def test_industry_strength_excludes_sectors_below_min_stocks():
    config = _industry_config(top_n=3, min_stocks=3)
    frame = pl.DataFrame(
        [
            *[_feature_row(f"B{index}", "big") for index in range(3)],
            _feature_row("S0", "small"),
            _feature_row("S1", "small"),
        ]
    )
    sectors = IndustryStrengthService().compute(
        frame, date(2024, 1, 4), config
    )
    assert [strength.sector for strength in sectors] == ["big"]
    assert sectors[0].stock_count == 3


def test_industry_strength_drops_sector_days_with_null_metrics():
    config = _industry_config(top_n=1, min_stocks=1)
    frame = pl.DataFrame(
        [
            _feature_row(
                "O", "old", ret_20d=0.10, ret_5d=0.02, above_ma20=True
            ),
            _feature_row(
                "N", "new", ret_20d=None, ret_5d=None, above_ma20=None
            ),
        ]
    )
    sectors = IndustryStrengthService().compute(
        frame, date(2024, 1, 4), config
    )
    assert [strength.sector for strength in sectors] == ["old"]
    assert sectors[0].rank == 1
    assert sectors[0].score == pytest.approx(100.0)


def test_industry_strength_returns_empty_when_as_of_absent():
    config = _industry_config(top_n=3, min_stocks=1)
    frame = pl.DataFrame([_feature_row("A", "银行")])
    assert (
        IndustryStrengthService().compute(frame, date(2024, 1, 1), config)
        == []
    )
    assert (
        IndustryStrengthService().compute(frame, date(2024, 1, 5), config)
        == []
    )


def test_industry_strength_persistence_breaks_at_nonleader_day():
    config = _industry_config(top_n=1, min_stocks=1)
    rows = []
    for trade_date, strong_return in (
        (date(2024, 1, 2), 0.20),
        (date(2024, 1, 3), 0.20),
        (date(2024, 1, 4), 0.00),
        (date(2024, 1, 5), 0.20),
    ):
        rows.append(
            _feature_row(
                "A", "A", trade_date=trade_date, ret_20d=strong_return
            )
        )
        rows.append(
            _feature_row(
                "B", "B", trade_date=trade_date, ret_20d=0.20 - strong_return
            )
        )
    sectors = IndustryStrengthService().compute(
        pl.DataFrame(rows), date(2024, 1, 5), config
    )
    assert [strength.sector for strength in sectors] == ["A", "B"]
    assert sectors[0].persistent_days == 1
    assert sectors[1].persistent_days == 0


def test_industry_strength_handles_ranking_ties_deterministically():
    config = _industry_config(top_n=3, min_stocks=1)
    forward = pl.DataFrame(
        [_feature_row("A", "A"), _feature_row("B", "B")]
    )
    for frame in (forward, forward.reverse()):
        sectors = IndustryStrengthService().compute(
            frame, date(2024, 1, 4), config
        )
        assert [(strength.sector, strength.rank) for strength in sectors] == [
            ("A", 1),
            ("B", 2),
        ]
        assert sectors[0].score == pytest.approx(75.0)
        assert sectors[1].score == pytest.approx(75.0)


def test_industry_strength_respects_as_of():
    config = _industry_config(top_n=1, min_stocks=1)
    rows = []
    for trade_date, a_return, b_return in (
        (date(2024, 1, 2), 0.20, 0.00),
        (date(2024, 1, 3), 0.20, 0.00),
        (date(2024, 1, 4), 0.00, 0.20),
    ):
        rows.append(
            _feature_row("A", "A", trade_date=trade_date, ret_20d=a_return)
        )
        rows.append(
            _feature_row("B", "B", trade_date=trade_date, ret_20d=b_return)
        )
    sectors = IndustryStrengthService().compute(
        pl.DataFrame(rows), date(2024, 1, 3), config
    )
    assert [strength.sector for strength in sectors] == ["A", "B"]
    assert sectors[0].persistent_days == 2


def test_industry_strength_red_rate_guards_prev_close():
    config = _industry_config(top_n=3, min_stocks=3)
    rows = []
    for trade_date in (date(2024, 1, 3), date(2024, 1, 4)):
        rows.extend(
            [
                _feature_row(
                    "A",
                    "X",
                    trade_date=trade_date,
                    prev_is_limit_up=True,
                    prev_close=None,
                    close=10.0,
                ),
                _feature_row(
                    "B",
                    "X",
                    trade_date=trade_date,
                    prev_is_limit_up=True,
                    prev_close=0.0,
                    close=10.0,
                ),
                _feature_row(
                    "C",
                    "X",
                    trade_date=trade_date,
                    prev_is_limit_up=True,
                    prev_close=9.0,
                    close=10.0,
                ),
            ]
        )
    sectors = IndustryStrengthService().compute(
        pl.DataFrame(rows), date(2024, 1, 4), config
    )
    assert sectors[0].red_rate == pytest.approx(1 / 3)


def test_industry_strength_red_rate_resets_after_missing_market_day():
    config = _industry_config(top_n=3, min_stocks=1)
    start = date(2024, 1, 2)
    middle = date(2024, 1, 3)
    as_of = date(2024, 1, 4)
    frame = pl.DataFrame(
        [
            _feature_row(
                "A",
                "X",
                trade_date=start,
                is_limit_up=True,
                prev_is_limit_up=False,
                prev_close=9.9,
                close=10.0,
            ),
            _feature_row("B", "X", trade_date=start),
            _feature_row("B", "X", trade_date=middle),
            _feature_row(
                "A",
                "X",
                trade_date=as_of,
                prev_is_limit_up=True,
                prev_close=9.9,
                close=10.0,
            ),
            _feature_row("B", "X", trade_date=as_of),
        ]
    )
    sectors = IndustryStrengthService().compute(frame, as_of, config)
    assert len(sectors) == 1
    assert sectors[0].stock_count == 2
    assert sectors[0].red_rate == 0.0


def test_industry_strength_amount_change_guards_nonpositive_average():
    config = _industry_config(top_n=3, min_stocks=3)
    frame = pl.DataFrame(
        [
            _feature_row("A", "X", amount_5d_avg=2.0e8, amount_20d_avg=1.0e8),
            _feature_row("B", "X", amount_5d_avg=1.0e8, amount_20d_avg=0.0),
            _feature_row("C", "X", amount_5d_avg=1.0e8, amount_20d_avg=-1.0),
        ]
    )
    sectors = IndustryStrengthService().compute(
        frame, date(2024, 1, 4), config
    )
    assert sectors[0].amount_change == pytest.approx(1 / 3)


def test_industry_strength_validates_required_columns():
    frame = pl.DataFrame(
        {"trade_date": [date(2024, 1, 4)], "ts_code": ["A"]}
    )
    with pytest.raises(
        ValueError, match="features_history missing required columns: sector"
    ):
        IndustryStrengthService().compute(
            frame, date(2024, 1, 4), _industry_config()
        )


def test_industry_map_prefers_sw_membership():
    stocks = pl.DataFrame(
        {"ts_code": ["A", "B"], "industry": ["银行", "地产"]}
    )
    members = pl.DataFrame(
        {
            "con_code": ["A"],
            "index_code": ["801010.SI"],
            "in_date": [date(2000, 1, 1)],
            "out_date": [None],
        }
    )
    classify = pl.DataFrame(
        {"index_code": ["801010.SI"], "industry_name": ["银行"]}
    )
    mapping = IndustryMap.from_sources(
        stocks, members, classify, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "sw"
    assert not mapping.degraded
    assert mapping.mapping["A"] == "银行"
    assert mapping.mapping["B"] == "地产"


def test_industry_map_falls_back_and_respects_dates():
    stocks = pl.DataFrame({"ts_code": ["A"], "industry": ["银行"]})
    members = pl.DataFrame(
        {
            "con_code": ["A"],
            "index_code": ["801010.SI"],
            "in_date": [date(2025, 1, 1)],
            "out_date": [None],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks, members, None, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "stock_basic"
    assert mapping.degraded


def test_industry_map_explicit_stock_basic_source_skips_sw():
    stocks = pl.DataFrame(
        {"ts_code": ["A", "B"], "industry": ["银行", "地产"]}
    )
    members = pl.DataFrame(
        {
            "con_code": ["A", "B"],
            "index_code": ["801010.SI", "801020.SI"],
            "in_date": [date(2000, 1, 1), date(2000, 1, 1)],
            "out_date": [None, None],
        }
    )
    classify = pl.DataFrame(
        {
            "index_code": ["801010.SI", "801020.SI"],
            "industry_name": ["SW银行", "SW地产"],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks,
        members,
        classify,
        as_of=date(2024, 1, 2),
        source="stock_basic",
    )
    assert mapping.source == "stock_basic"
    assert mapping.degraded
    assert mapping.mapping == {"A": "银行", "B": "地产"}


def test_industry_map_respects_membership_dates():
    stocks = pl.DataFrame(
        {
            "ts_code": ["A", "B", "C", "D"],
            "industry": ["银行", "地产", "煤炭", "钢铁"],
        }
    )
    members = pl.DataFrame(
        {
            "con_code": ["A", "B", "C", "D"],
            "index_code": [
                "801010.SI",
                "801020.SI",
                "801030.SI",
                "801040.SI",
            ],
            "in_date": [
                date(2000, 1, 1),
                date(2025, 1, 1),
                date(2024, 1, 2),
                date(2026, 1, 1),
            ],
            "out_date": [date(2023, 12, 31), None, date(2024, 1, 2), None],
        }
    )
    classify = pl.DataFrame(
        {
            "index_code": [
                "801010.SI",
                "801020.SI",
                "801030.SI",
                "801040.SI",
            ],
            "industry_name": ["SW银行", "SW地产", "SW煤炭", "SW钢铁"],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks, members, classify, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "sw"
    assert not mapping.degraded
    assert mapping.mapping["A"] == "银行"
    assert mapping.mapping["B"] == "地产"
    assert mapping.mapping["C"] == "SW煤炭"
    assert mapping.mapping["D"] == "钢铁"


def test_industry_map_parses_string_membership_dates():
    stocks = pl.DataFrame(
        {
            "ts_code": ["A", "B", "C"],
            "industry": ["银行原始", "地产原始", "煤炭原始"],
        }
    )
    members = pl.DataFrame(
        {
            "con_code": ["A", "B", "C"],
            "index_code": ["801010.SI", "801020.SI", "801030.SI"],
            "in_date": ["20000101", "2025-01-01", "2000-01-01"],
            "out_date": [None, None, "2023-12-31"],
        }
    )
    classify = pl.DataFrame(
        {
            "index_code": ["801010.SI", "801020.SI", "801030.SI"],
            "industry_name": ["SW银行", "SW地产", "SW煤炭"],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks, members, classify, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "sw"
    assert mapping.mapping["A"] == "SW银行"
    assert mapping.mapping["B"] == "地产原始"
    assert mapping.mapping["C"] == "煤炭原始"


def test_industry_map_filters_l1_memberships():
    stocks = pl.DataFrame(
        {"ts_code": ["A", "B"], "industry": ["银行原始", "地产原始"]}
    )
    classify = pl.DataFrame(
        {
            "index_code": ["801010.SI", "801011.SI", "801012.SI"],
            "industry_name": ["银行", "银行二级", "地产二级"],
            "level": ["L1", "L2", "L2"],
        }
    )
    members = pl.DataFrame(
        {
            "con_code": ["A", "A", "B"],
            "index_code": ["801010.SI", "801011.SI", "801012.SI"],
            "in_date": [
                date(2000, 1, 1),
                date(2023, 1, 1),
                date(2000, 1, 1),
            ],
            "out_date": [None, None, None],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks, members, classify, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "sw"
    assert not mapping.degraded
    assert mapping.mapping["A"] == "银行"
    assert mapping.mapping["B"] == "地产原始"


def test_industry_map_filters_to_one_classification_source():
    stocks = pl.DataFrame(
        {"ts_code": ["A", "B"], "industry": ["银行原始", "地产原始"]}
    )
    classify = pl.DataFrame(
        {
            "index_code": ["801010.SI", "801020.SI"],
            "industry_name": ["银行2021", "地产旧版"],
            "level": ["L1", "L1"],
            "src": ["SW2021", "SW"],
        }
    )
    members = pl.DataFrame(
        {
            "con_code": ["A", "B"],
            "index_code": ["801010.SI", "801020.SI"],
            "in_date": [date(2000, 1, 1), date(2000, 1, 1)],
            "out_date": [None, None],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks, members, classify, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "sw"
    assert not mapping.degraded
    assert mapping.mapping["A"] == "银行2021"
    assert mapping.mapping["B"] == "地产原始"


def test_industry_map_duplicate_memberships_are_deterministic():
    stocks = pl.DataFrame({"ts_code": ["A"], "industry": ["银行原始"]})
    classify = pl.DataFrame(
        {
            "index_code": ["801010.SI", "801020.SI", "801030.SI"],
            "industry_name": ["银行", "地产", "煤炭"],
            "level": ["L1", "L1", "L1"],
        }
    )
    rows = [
        {
            "con_code": "A",
            "index_code": "801020.SI",
            "in_date": date(2020, 1, 1),
            "out_date": None,
        },
        {
            "con_code": "A",
            "index_code": "801010.SI",
            "in_date": date(2024, 1, 1),
            "out_date": None,
        },
        {
            "con_code": "A",
            "index_code": "801030.SI",
            "in_date": date(2024, 1, 1),
            "out_date": None,
        },
        {
            "con_code": "A",
            "index_code": "801040.SI",
            "in_date": date(2022, 1, 1),
            "out_date": None,
        },
    ]
    for member_rows in (rows, list(reversed(rows))):
        mapping = IndustryMap.from_sources(
            stocks,
            pl.DataFrame(member_rows),
            classify,
            as_of=date(2024, 6, 1),
        )
        assert mapping.source == "sw"
        assert not mapping.degraded
        assert mapping.mapping["A"] == "银行"


def test_industry_map_marks_bad_membership_dates_degraded():
    stocks = pl.DataFrame(
        {
            "ts_code": ["A", "B", "C", "D"],
            "industry": ["银行原始", "地产原始", "煤炭原始", "钢铁原始"],
        }
    )
    classify = pl.DataFrame(
        {
            "index_code": [
                "801010.SI",
                "801020.SI",
                "801030.SI",
                "801040.SI",
            ],
            "industry_name": ["银行", "地产", "煤炭", "钢铁"],
        }
    )
    members = pl.DataFrame(
        {
            "con_code": ["A", "B", "C", "D"],
            "index_code": [
                "801010.SI",
                "801020.SI",
                "801030.SI",
                "801040.SI",
            ],
            "in_date": [
                "20000101",
                "not-a-date",
                None,
                "20000101",
            ],
            "out_date": [None, None, None, "not-a-date"],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks, members, classify, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "sw"
    assert mapping.degraded
    assert mapping.mapping["A"] == "银行"
    assert mapping.mapping["B"] == "地产原始"
    assert mapping.mapping["C"] == "煤炭原始"
    assert mapping.mapping["D"] == "钢铁原始"


def test_industry_map_raises_when_all_membership_dates_unusable():
    stocks = pl.DataFrame({"ts_code": ["A"], "industry": ["银行"]})
    members = pl.DataFrame(
        {
            "con_code": ["A"],
            "index_code": ["801010.SI"],
            "in_date": ["not-a-date"],
            "out_date": [None],
        }
    )
    with pytest.raises(ValueError, match="no usable membership rows"):
        IndustryMap.from_sources(
            stocks, members, None, as_of=date(2024, 1, 2)
        )


@pytest.mark.parametrize(
    "classify",
    [
        None,
        pl.DataFrame(
            {"index_code": ["999999.SI"], "industry_name": ["其他"]}
        ),
        pl.DataFrame({"index_code": ["801010.SI"]}),
        pl.DataFrame(
            {"index_code": ["801010.SI"], "industry_name": [None]}
        ),
    ],
)
def test_industry_map_uses_index_code_when_classify_name_missing(
    classify: pl.DataFrame | None,
):
    stocks = pl.DataFrame({"ts_code": ["A"], "industry": ["银行"]})
    members = pl.DataFrame(
        {
            "con_code": ["A"],
            "index_code": ["801010.SI"],
            "in_date": [date(2000, 1, 1)],
            "out_date": [None],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks, members, classify, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "sw"
    assert not mapping.degraded
    assert mapping.mapping["A"] == "801010.SI"


def test_industry_map_validates_required_columns():
    stocks = pl.DataFrame({"ts_code": ["A"], "industry": ["银行"]})
    members = pl.DataFrame(
        {
            "con_code": ["A"],
            "index_code": ["801010.SI"],
            "in_date": [date(2000, 1, 1)],
            "out_date": [None],
        }
    )
    with pytest.raises(
        ValueError, match="stock_basic missing required columns: ts_code"
    ):
        IndustryMap.from_sources(pl.DataFrame({"industry": ["银行"]}))
    with pytest.raises(
        ValueError, match="index_member missing required columns: index_code"
    ):
        IndustryMap.from_sources(stocks, members.drop("index_code"))
    with pytest.raises(
        ValueError, match="index_member missing required columns: in_date"
    ):
        IndustryMap.from_sources(stocks, members.drop("in_date"))
    with pytest.raises(
        ValueError, match="index_member missing required columns: out_date"
    ):
        IndustryMap.from_sources(stocks, members.drop("out_date"))
    with pytest.raises(
        ValueError,
        match="index_member missing required columns: con_code or ts_code",
    ):
        IndustryMap.from_sources(stocks, members.drop("con_code"))
    with pytest.raises(
        ValueError, match="index_classify missing required columns: index_code"
    ):
        IndustryMap.from_sources(
            stocks, members, pl.DataFrame({"industry_name": ["银行"]})
        )


def test_industry_map_to_frame():
    stocks = pl.DataFrame(
        {"ts_code": ["A", "B"], "industry": ["银行", None]}
    )
    frame = IndustryMap.from_sources(stocks).to_frame()
    assert frame.columns == ["ts_code", "sector"]
    assert frame.sort("ts_code").to_dicts() == [
        {"ts_code": "A", "sector": "银行"},
        {"ts_code": "B", "sector": "未知"},
    ]
    empty = IndustryMap({}, "stock_basic", True).to_frame()
    assert empty.columns == ["ts_code", "sector"]
    assert empty.is_empty()


def test_provider_exposes_index_classify_and_limit_list(tmp_path: Path):
    store = ParquetStore(
        tmp_path / "raw", tmp_path / "processed", tmp_path / "regime"
    )
    classify = pl.DataFrame(
        {"index_code": ["801010.SI"], "industry_name": ["银行"]}
    )
    limit_list = pl.DataFrame(
        {"trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SZ"]}
    )
    store.write_reference("index_classify", classify)
    store.write_processed("limit_list", date(2024, 1, 2), limit_list)
    provider = ParquetMarketDataProvider(store)
    assert provider.load_index_classify().to_dicts() == classify.to_dicts()
    loaded = provider.load_limit_list(date(2024, 1, 2), date(2024, 1, 2))
    assert loaded.to_dicts() == limit_list.to_dicts()
