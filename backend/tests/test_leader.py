from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from app.core.config import load_strategy_config
from app.domain.models import SectorStrength
from app.regime.leader import LeaderScoreService

_REQUIRED_METRICS = (
    "ret_20d",
    "ret_5d",
    "amount_20d_avg",
    "turnover_rate",
    "close_to_high_60d",
    "circ_mv",
    "close",
    "days_since_limit_up",
)


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["A", "B", "C", "D"],
            "trade_date": [date(2024, 1, 2)] * 4,
            "sector": ["银行"] * 4,
            "ret_20d": [0.30, 0.20, 0.10, 0.0],
            "ret_5d": [0.10, 0.05, 0.02, 0.0],
            "amount_20d_avg": [3e8, 1e8, 1e8, 1e8],
            "turnover_rate": [0.10, 0.08, 0.05, 0.03],
            "close_to_high_60d": [0.99, 0.95, 0.90, 0.80],
            "board_height": [2, 1, 0, 0],
            "days_since_limit_up": [0, 0, 10, 20],
            "circ_mv": [1e10, 1e10, 1e10, 1e10],
            "close": [10.0, 10.0, 10.0, 10.0],
        }
    )


def _sector() -> SectorStrength:
    return SectorStrength(
        sector="银行",
        score=90.0,
        rank=1,
        return_20d=0.2,
        return_5d=0.05,
        breadth=0.8,
        amount_change=0.2,
        limit_up_share=0.2,
        limit_up_count=3,
        red_rate=0.6,
        stock_count=10,
        persistent_days=2,
    )


def test_leader_rank_and_followers():
    config = load_strategy_config()
    leaders = LeaderScoreService().compute(_features(), [_sector()], config)
    rows = {row["ts_code"]: row for row in leaders.to_dicts()}
    assert rows["A"]["leader_rank"] == 1
    assert not rows["A"]["is_follower"]
    assert rows["B"]["is_follower"]
    assert rows["D"]["is_follower"]


def _stock_row(code: str, sector: str = "银行", **overrides: object) -> dict:
    row: dict = {
        "ts_code": code,
        "trade_date": date(2024, 1, 2),
        "sector": sector,
        "ret_20d": 0.10,
        "ret_5d": 0.02,
        "amount_20d_avg": 1e8,
        "turnover_rate": 0.05,
        "close_to_high_60d": 0.90,
        "board_height": 0,
        "days_since_limit_up": 20,
        "circ_mv": 1e10,
        "close": 10.0,
    }
    row.update(overrides)
    return row


def _frame(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


def _sector_with(**overrides: object) -> SectorStrength:
    return _sector().model_copy(update=overrides)


def test_output_columns_match_screener_contract():
    leaders = LeaderScoreService().compute(
        _features(), [_sector()], load_strategy_config()
    )
    assert leaders.columns == [
        "ts_code",
        "sector",
        "leader_score",
        "leader_rank",
        "is_follower",
        "amount_20d_avg",
    ]


def test_non_persistent_sector_is_excluded():
    config = load_strategy_config()
    leaders = LeaderScoreService().compute(
        _frame(_stock_row("A")),
        [_sector_with(persistent_days=config.industry.persistence_days - 1)],
        config,
    )
    assert leaders.is_empty()


def test_sector_outside_top_n_is_excluded():
    config = load_strategy_config()
    leaders = LeaderScoreService().compute(
        _frame(_stock_row("A")),
        [_sector_with(rank=config.industry.top_n + 1, persistent_days=5)],
        config,
    )
    assert leaders.is_empty()


def test_follower_by_rank_cutoff():
    config = load_strategy_config()
    rows = [
        _stock_row("A", ret_20d=0.40),
        _stock_row("B", ret_20d=0.30),
        _stock_row("C", ret_20d=0.20),
        _stock_row("D", ret_20d=0.10),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    by_code = {row["ts_code"]: row for row in leaders.to_dicts()}
    assert by_code["D"]["leader_rank"] == config.leader.follower_rank_cutoff + 1
    assert by_code["D"]["is_follower"]
    assert not by_code["C"]["is_follower"]


def test_follower_by_amount_vs_sector_leader():
    config = load_strategy_config()
    rows = [
        _stock_row("A", ret_20d=0.30, amount_20d_avg=4e8),
        _stock_row("B", ret_20d=0.20, amount_20d_avg=1e8),
        _stock_row("C", ret_20d=0.10, amount_20d_avg=3e8),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    by_code = {row["ts_code"]: row for row in leaders.to_dicts()}
    assert by_code["B"]["leader_rank"] <= config.leader.follower_rank_cutoff
    assert by_code["B"]["is_follower"]
    assert not by_code["A"]["is_follower"]
    assert not by_code["C"]["is_follower"]


def test_single_stock_sector_is_leader():
    config = load_strategy_config()
    row = {
        "ts_code": "A",
        "sector": "银行",
        "ret_20d": 0.10,
        "ret_5d": 0.02,
        "amount_20d_avg": 1e8,
        "turnover_rate": 0.05,
        "close_to_high_60d": 0.90,
        "board_height": 2,
        "days_since_limit_up": 0,
        "circ_mv": 1e10,
        "close": 10.0,
    }
    leaders = LeaderScoreService().compute(
        pl.DataFrame([row]), [_sector_with(stock_count=1)], config
    )
    result = leaders.to_dicts()[0]
    assert result["leader_rank"] == 1
    assert result["leader_score"] == pytest.approx(100.0)
    assert result["is_follower"] is False


def test_limit_status_and_size_fit_tiers():
    config = load_strategy_config()
    rows = [
        _stock_row("A", board_height=2, days_since_limit_up=0),
        _stock_row(
            "B", board_height=1, days_since_limit_up=0, circ_mv=8e10, close=10.0
        ),
        _stock_row("C", board_height=0, days_since_limit_up=3, close=2.0),
        _stock_row("D", board_height=0, days_since_limit_up=20, circ_mv=8e10, close=200.0),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    by_code = {row["ts_code"]: row for row in leaders.to_dicts()}
    # Four tied stocks give every percentile metric 2.5/4 = 0.625.
    assert by_code["A"]["leader_score"] == pytest.approx(71.875)
    assert by_code["B"]["leader_score"] == pytest.approx(63.875)
    assert by_code["C"]["leader_score"] == pytest.approx(54.375)
    assert by_code["D"]["leader_score"] == pytest.approx(46.875)


def test_tie_scores_are_ranked_by_ts_code():
    config = load_strategy_config()
    codes = ["A", "B", "C", "D"]
    forward = LeaderScoreService().compute(
        _frame(*[_stock_row(code) for code in codes]), [_sector()], config
    )
    reverse = LeaderScoreService().compute(
        _frame(*[_stock_row(code) for code in reversed(codes)]), [_sector()], config
    )
    expected = {"A": 1, "B": 2, "C": 3, "D": 4}
    forward_ranks = {
        row["ts_code"]: row["leader_rank"] for row in forward.to_dicts()
    }
    reverse_ranks = {
        row["ts_code"]: row["leader_rank"] for row in reverse.to_dicts()
    }
    assert forward["leader_score"].n_unique() == 1
    assert forward_ranks == expected
    assert reverse_ranks == expected


@pytest.mark.parametrize("column", _REQUIRED_METRICS)
def test_rows_with_null_required_metrics_are_dropped(column: str):
    config = load_strategy_config()
    rows = [
        _stock_row("A", board_height=2, days_since_limit_up=0),
        _stock_row("B", **{column: None}),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    assert leaders["ts_code"].to_list() == ["A"]
    result = leaders.to_dicts()[0]
    assert result["leader_score"] is not None
    assert result["leader_rank"] == 1
    assert result["is_follower"] is False


def test_all_null_metrics_return_empty_frame():
    config = load_strategy_config()
    rows = [
        _stock_row("A", ret_20d=None),
        _stock_row("B", amount_20d_avg=None),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    assert leaders.is_empty()


@pytest.mark.parametrize(
    "amounts",
    [
        (0.0, 0.0, 0.0),
        (-1e8, -2e8, -3e8),
    ],
)
def test_non_positive_sector_leader_amount_does_not_mark_followers(
    amounts: tuple[float, float, float],
):
    config = load_strategy_config()
    rows = [
        _stock_row("A", ret_20d=0.30, amount_20d_avg=amounts[0]),
        _stock_row("B", ret_20d=0.20, amount_20d_avg=amounts[1]),
        _stock_row("C", ret_20d=0.10, amount_20d_avg=amounts[2]),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    assert leaders["is_follower"].to_list() == [False, False, False]


def test_empty_features_or_sectors_return_empty_frame():
    config = load_strategy_config()
    service = LeaderScoreService()
    assert service.compute(pl.DataFrame(), [_sector()], config).is_empty()
    assert service.compute(_features(), [], config).is_empty()
