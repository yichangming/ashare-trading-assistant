from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from app.core.config import LeaderWeights, load_strategy_config
from app.domain.models import SectorStrength
from app.regime.leader import LeaderScoreService

_REQUIRED_METRICS = (
    "ret_20d",
    "ret_5d",
    "amount_20d_avg",
    "turnover_rate",
    "close_to_high_60d",
    "board_height",
    "days_since_limit_up",
    "circ_mv",
    "close",
)
_LEADER_WEIGHT_FIELDS = (
    "rs_20d",
    "rs_5d",
    "amount_turnover",
    "high_distance",
    "limit_up_status",
    "size_price_fit",
)
_OUTPUT_COLUMNS = [
    "ts_code",
    "sector",
    "leader_score",
    "leader_rank",
    "is_follower",
    "amount_20d_avg",
]


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


def _config_with_solo_weight(name: str):
    values = {field: 0.0 for field in _LEADER_WEIGHT_FIELDS}
    values[name] = 1.0
    config = load_strategy_config()
    leader = config.leader.model_copy(update={"weights": LeaderWeights(**values)})
    return config.model_copy(update={"leader": leader})


def test_output_columns_match_screener_contract():
    leaders = LeaderScoreService().compute(
        _features(), [_sector()], load_strategy_config()
    )
    assert leaders.columns == _OUTPUT_COLUMNS


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


def test_multi_sector_ranking_is_independent():
    config = load_strategy_config()
    rows = [
        _stock_row("A", sector="银行", ret_20d=0.30, amount_20d_avg=3e8),
        _stock_row("B", sector="银行", ret_20d=0.20),
        _stock_row("C", sector="银行", ret_20d=0.10),
        _stock_row("D", sector="医药", ret_20d=0.03, amount_20d_avg=2e8),
        _stock_row("E", sector="医药", ret_20d=0.02),
        _stock_row("F", sector="医药", ret_20d=0.01),
    ]
    sectors = [
        _sector(),
        _sector_with(sector="医药", rank=2, persistent_days=3),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), sectors, config)
    by_code = {row["ts_code"]: row for row in leaders.to_dicts()}
    assert {by_code[code]["leader_rank"] for code in ("A", "B", "C")} == {1, 2, 3}
    assert {by_code[code]["leader_rank"] for code in ("D", "E", "F")} == {1, 2, 3}
    assert by_code["A"]["leader_rank"] == 1
    assert by_code["D"]["leader_rank"] == 1
    # Both sector leaders have identical within-sector percentiles.
    assert by_code["D"]["leader_score"] == pytest.approx(
        by_code["A"]["leader_score"]
    )


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


def test_follower_amount_baseline_is_rank_one_leader():
    config = load_strategy_config()
    rows = [
        _stock_row(
            "A",
            ret_20d=0.50,
            ret_5d=0.20,
            amount_20d_avg=1e8,
            turnover_rate=0.10,
            close_to_high_60d=0.99,
            board_height=2,
            days_since_limit_up=0,
        ),
        _stock_row("B", ret_20d=0.05, ret_5d=0.01, amount_20d_avg=4e8),
        _stock_row("C", ret_20d=0.04, ret_5d=0.01, amount_20d_avg=3e8),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    by_code = {row["ts_code"]: row for row in leaders.to_dicts()}
    assert by_code["A"]["leader_rank"] == 1
    # Sector maximum is 4e8, but the baseline is rank-1 A's 1e8.
    assert not by_code["A"]["is_follower"]
    assert by_code["A"]["amount_20d_avg"] < by_code["B"]["amount_20d_avg"]


def test_follower_amount_boundary_is_strict():
    config = load_strategy_config()
    rows = [
        _stock_row("A", ret_20d=0.30, amount_20d_avg=4e8),
        _stock_row("B", ret_20d=0.20, amount_20d_avg=2e8),
        _stock_row("C", ret_20d=0.10, amount_20d_avg=1.99e8),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    by_code = {row["ts_code"]: row for row in leaders.to_dicts()}
    assert by_code["A"]["leader_rank"] == 1
    assert by_code["B"]["leader_rank"] <= config.leader.follower_rank_cutoff
    assert by_code["C"]["leader_rank"] <= config.leader.follower_rank_cutoff
    assert not by_code["B"]["is_follower"]
    assert by_code["C"]["is_follower"]


def test_single_stock_sector_is_leader():
    config = load_strategy_config()
    row = {
        "ts_code": "A",
        "trade_date": date(2024, 1, 2),
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


@pytest.mark.parametrize(
    ("board_height", "days_since_limit_up", "circ_mv", "close", "expected"),
    [
        (2, 100, 1e10, 10.0, 100.0),
        (1, 100, 1e10, 10.0, 97.0),
        (0, 5, 1e10, 10.0, 92.5),
        (0, 6, 1e10, 10.0, 85.0),
        (2, 100, 5e9, 5.0, 100.0),
        (2, 100, 5e10, 100.0, 100.0),
        (2, 100, 2.5e9, 3.0, 95.0),
        (2, 100, 1e11, 150.0, 95.0),
        (2, 100, 4.999e9, 10.0, 95.0),
        (2, 100, 1e10, 4.999, 95.0),
        (2, 100, 1e10, 100.001, 95.0),
        (2, 100, 1e10, 2.999, 90.0),
        (2, 100, 1e10, 150.001, 90.0),
        (2, 100, 2.499e9, 10.0, 90.0),
        (2, 100, 1.0001e11, 10.0, 90.0),
    ],
)
def test_limit_and_size_tier_boundaries(
    board_height: int,
    days_since_limit_up: int,
    circ_mv: float,
    close: float,
    expected: float,
):
    config = load_strategy_config()
    row = _stock_row(
        "A",
        board_height=board_height,
        days_since_limit_up=days_since_limit_up,
        circ_mv=circ_mv,
        close=close,
    )
    leaders = LeaderScoreService().compute(
        pl.DataFrame([row]), [_sector_with(stock_count=1)], config
    )
    assert leaders.to_dicts()[0]["leader_score"] == pytest.approx(expected)


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


def _disagreeing_rows() -> list[dict]:
    return [
        _stock_row(
            "A",
            ret_20d=0.50,
            ret_5d=0.10,
            amount_20d_avg=3e8,
            turnover_rate=0.02,
            close_to_high_60d=0.80,
            board_height=2,
            days_since_limit_up=0,
        ),
        _stock_row(
            "B",
            ret_20d=0.40,
            ret_5d=0.20,
            amount_20d_avg=5e8,
            turnover_rate=0.01,
            close_to_high_60d=0.90,
            board_height=1,
            days_since_limit_up=0,
        ),
        _stock_row(
            "C",
            ret_20d=0.30,
            ret_5d=0.30,
            amount_20d_avg=1e8,
            turnover_rate=0.05,
            close_to_high_60d=0.99,
            board_height=0,
            days_since_limit_up=5,
            circ_mv=8e10,
        ),
        _stock_row(
            "D",
            ret_20d=0.20,
            ret_5d=0.40,
            amount_20d_avg=4e8,
            turnover_rate=0.03,
            close_to_high_60d=0.95,
            board_height=0,
            days_since_limit_up=6,
            circ_mv=2e9,
        ),
        _stock_row(
            "E",
            ret_20d=0.10,
            ret_5d=0.50,
            amount_20d_avg=2e8,
            turnover_rate=0.04,
            close_to_high_60d=0.85,
            board_height=3,
            days_since_limit_up=0,
        ),
    ]


def test_percentile_metrics_disagree_and_pin_each_weight():
    config = load_strategy_config()
    rows = _disagreeing_rows()
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    by_code = {row["ts_code"]: row for row in leaders.to_dicts()}
    expected_default = {"A": 66.0, "B": 69.0, "C": 63.5, "D": 48.0, "E": 63.0}
    for code, expected in expected_default.items():
        assert by_code[code]["leader_score"] == pytest.approx(expected)

    expected_solo = {
        "rs_20d": {"A": 100.0, "B": 80.0, "C": 60.0, "D": 40.0, "E": 20.0},
        "rs_5d": {"A": 20.0, "B": 40.0, "C": 60.0, "D": 80.0, "E": 100.0},
        "amount_turnover": {"A": 50.0, "B": 60.0, "C": 60.0, "D": 70.0, "E": 60.0},
        "high_distance": {"A": 20.0, "B": 60.0, "C": 100.0, "D": 80.0, "E": 40.0},
        "limit_up_status": {"A": 100.0, "B": 80.0, "C": 50.0, "D": 0.0, "E": 100.0},
        "size_price_fit": {"A": 100.0, "B": 100.0, "C": 50.0, "D": 0.0, "E": 100.0},
    }
    for weight_name, expected_scores in expected_solo.items():
        solo = LeaderScoreService().compute(
            _frame(*rows), [_sector()], _config_with_solo_weight(weight_name)
        )
        solo_by_code = {row["ts_code"]: row for row in solo.to_dicts()}
        for code, expected in expected_scores.items():
            assert solo_by_code[code]["leader_score"] == pytest.approx(expected)


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


def test_non_finite_metrics_are_dropped():
    config = load_strategy_config()
    rows = [
        _stock_row("A", board_height=2, days_since_limit_up=0),
        _stock_row("B", ret_20d=float("nan")),
        _stock_row("C", amount_20d_avg=float("inf")),
        _stock_row("D", close=float("-inf")),
    ]
    leaders = LeaderScoreService().compute(_frame(*rows), [_sector()], config)
    assert leaders["ts_code"].to_list() == ["A"]


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


def test_multi_date_features_are_rejected():
    features = _frame(
        _stock_row("A"), _stock_row("B", trade_date=date(2024, 1, 3))
    )
    with pytest.raises(ValueError, match="exactly one trade_date"):
        LeaderScoreService().compute(
            features, [_sector()], load_strategy_config()
        )


def test_duplicate_ts_code_is_rejected():
    features = _frame(_stock_row("A"), _stock_row("A", ret_20d=0.2))
    with pytest.raises(ValueError, match="duplicate ts_code"):
        LeaderScoreService().compute(
            features, [_sector()], load_strategy_config()
        )


def test_missing_required_column_raises_value_error():
    features = _features().drop("turnover_rate")
    with pytest.raises(ValueError, match="missing required columns.*turnover_rate"):
        LeaderScoreService().compute(
            features, [_sector()], load_strategy_config()
        )


def test_empty_results_have_output_schema():
    config = load_strategy_config()
    service = LeaderScoreService()
    empties = [
        service.compute(pl.DataFrame(), [_sector()], config),
        service.compute(_features(), [], config),
        service.compute(
            _frame(_stock_row("A")),
            [_sector_with(rank=config.industry.top_n + 1)],
            config,
        ),
        service.compute(
            _frame(_stock_row("A", ret_20d=None)), [_sector()], config
        ),
    ]
    for leaders in empties:
        assert leaders.is_empty()
        assert leaders.columns == _OUTPUT_COLUMNS
        assert leaders.schema["leader_score"] == pl.Float64
        assert leaders.schema["leader_rank"] == pl.UInt32
        assert leaders.schema["is_follower"] == pl.Boolean


def test_empty_features_or_sectors_return_empty_frame():
    config = load_strategy_config()
    service = LeaderScoreService()
    assert service.compute(pl.DataFrame(), [_sector()], config).is_empty()
    assert service.compute(_features(), [], config).is_empty()
