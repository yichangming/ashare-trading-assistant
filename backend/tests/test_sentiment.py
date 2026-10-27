from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from app.core.config import load_strategy_config
from app.regime.sentiment import SentimentService, SentimentStageClassifier


def _bars() -> pl.DataFrame:
    rows = []
    base = date(2024, 1, 1)
    for day in range(4):
        trade_date = base + timedelta(days=day)
        for code, is_limit_up in (
            ("A", day in (2, 3)),
            ("B", day == 3),
            ("C", False),
        ):
            close = 11.0 if is_limit_up else 10.0
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.9,
                    "close": close,
                    "pre_close": 10.0,
                    "up_limit": 11.0,
                    "down_limit": 9.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                    "adj_factor": 1.0,
                }
            )
    return pl.DataFrame(rows)


def _history(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).sort("trade_date")


def _stage_row(trade_date: date, **overrides: float | int) -> dict:
    row: dict = {
        "trade_date": trade_date,
        "limit_up_count": 80,
        "limit_down_count": 0,
        "max_board_height": 5,
        "promotion_rate": 0.5,
        "blow_rate": 0.1,
        "red_rate": 0.6,
        "limit_up_premium": 0.02,
        "blow_rate_ma3": 0.1,
        "max_board_ma3": 5,
        "premium_ma3": 0.02,
    }
    row.update(overrides)
    return row


def test_sentiment_history_contains_indicators():
    history = SentimentService().compute_history(_bars())
    last = history.sort("trade_date").tail(1).row(0, named=True)
    assert last["limit_up_count"] == 2
    assert last["max_board_height"] == 2
    assert "promotion_rate" in last
    assert "blow_rate" in last
    assert "red_rate" in last


def test_stage_classifier_two_day_confirmation():
    config = load_strategy_config()
    history = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "limit_up_count": [80, 80, 80],
            "limit_down_count": [0, 0, 0],
            "max_board_height": [5, 5, 5],
            "promotion_rate": [0.5, 0.5, 0.5],
            "blow_rate": [0.1, 0.1, 0.1],
            "red_rate": [0.6, 0.6, 0.6],
            "limit_up_premium": [0.02, 0.02, 0.02],
            "blow_rate_ma3": [0.1, 0.1, 0.1],
            "max_board_ma3": [5, 5, 5],
            "premium_ma3": [0.02, 0.02, 0.02],
        }
    )
    classified = SentimentStageClassifier().classify_series(history, config)
    stages = classified["sentiment_stage"].to_list()
    assert stages[0] == "start"
    assert stages[1] == "ferment"
    assert stages[2] == "ferment"


def test_red_rate_and_premium_use_exchange_pre_close():
    bars = pl.DataFrame(
        [
            {
                "ts_code": "A",
                "trade_date": date(2024, 1, 2),
                "open": 10.0,
                "high": 11.0,
                "low": 10.0,
                "close": 11.0,
                "pre_close": 10.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
                "vol": 100.0,
                "amount": 1000.0,
                "adj_factor": 1.0,
            },
            {
                "ts_code": "A",
                "trade_date": date(2024, 1, 3),
                # 2-for-1 split: the previous raw close (11.0) is not the
                # correct reference, but the exchange pre_close (5.5) is.
                "open": 5.6,
                "high": 5.8,
                "low": 5.5,
                "close": 5.7,
                "pre_close": 5.5,
                "up_limit": 6.05,
                "down_limit": 4.95,
                "vol": 100.0,
                "amount": 1000.0,
                "adj_factor": 2.0,
            },
        ]
    )
    history = SentimentService().compute_history(bars).sort("trade_date")
    first = history.row(0, named=True)
    second = history.row(1, named=True)
    assert first["red_rate"] == 0.0
    assert first["limit_up_premium"] == 0.0
    assert second["red_rate"] == 1.0
    assert second["limit_up_premium"] == pytest.approx(5.6 / 5.5 - 1)


def test_zero_limit_up_and_suspended_day_has_zero_rates():
    bars = pl.DataFrame(
        [
            {
                "ts_code": code,
                "trade_date": date(2024, 1, 2),
                "open": 10.0,
                "high": 10.0 if suspended else 10.1,
                "low": 10.0 if suspended else 9.9,
                "close": 10.0,
                "pre_close": 10.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
                "vol": 0.0 if suspended else 100.0,
                "amount": 0.0 if suspended else 1000.0,
                "adj_factor": 1.0,
                "is_suspended": suspended,
            }
            for code, suspended in (("A", False), ("B", True))
        ]
    )
    row = SentimentService().compute_history(bars).row(0, named=True)
    assert row["limit_up_count"] == 0
    assert row["limit_down_count"] == 0
    assert row["max_board_height"] == 0
    for column in (
        "promotion_rate",
        "blow_rate",
        "red_rate",
        "limit_up_premium",
    ):
        assert row[column] == 0.0
        assert math.isfinite(row[column])


def test_limit_flags_require_positive_limits():
    bars = pl.DataFrame(
        [
            {
                "ts_code": "A",
                "trade_date": date(2024, 1, 2),
                "open": 10.0,
                "high": 10.5,
                "low": 9.9,
                "close": 10.0,
                "pre_close": 10.0,
                "up_limit": 0.0,
                "down_limit": 0.0,
                "vol": 100.0,
                "amount": 1000.0,
                "adj_factor": 1.0,
            },
            {
                "ts_code": "B",
                "trade_date": date(2024, 1, 2),
                "open": 10.0,
                "high": 11.0,
                "low": 9.9,
                "close": 11.0,
                "pre_close": 10.0,
                "up_limit": None,
                "down_limit": None,
                "vol": 100.0,
                "amount": 1000.0,
                "adj_factor": 1.0,
            },
        ],
        schema_overrides={"up_limit": pl.Float64, "down_limit": pl.Float64},
    )
    row = SentimentService().compute_history(bars).row(0, named=True)
    assert row["limit_up_count"] == 0
    assert row["limit_down_count"] == 0
    assert row["blow_rate"] == 0.0


def test_compute_history_handles_empty_bars():
    assert SentimentService().compute_history(pl.DataFrame()).is_empty()


def test_stage_classifier_holds_stage_when_fewer_than_two_conditions():
    config = load_strategy_config()
    ambiguous = {
        "limit_up_count": 60,
        "max_board_height": 5,
        "blow_rate": 0.5,
        "blow_rate_ma3": 0.5,
        "red_rate": 0.35,
    }
    history = _history(
        [
            _stage_row(date(2024, 1, 1)),
            _stage_row(date(2024, 1, 2)),
            _stage_row(date(2024, 1, 3), **ambiguous),
            _stage_row(date(2024, 1, 4), **ambiguous),
        ]
    )
    classified = SentimentStageClassifier().classify_series(history, config)
    assert classified["sentiment_stage"].to_list() == [
        "start",
        "ferment",
        "ferment",
        "ferment",
    ]


def test_stage_classifier_tie_uses_defensive_priority():
    config = load_strategy_config()
    tie = {
        "limit_up_count": 25,
        "limit_down_count": 35,
        "max_board_height": 5,
        "blow_rate": 0.2,
        "blow_rate_ma3": 0.2,
        "red_rate": 0.5,
    }
    history = _history(
        [
            _stage_row(date(2024, 1, 1), **tie),
            _stage_row(date(2024, 1, 2), **tie),
        ]
    )
    classified = SentimentStageClassifier().classify_series(history, config)
    assert classified["sentiment_stage"].to_list() == ["start", "ebb"]


def test_stage_classifier_starts_at_start():
    config = load_strategy_config()
    history = _history(
        [
            _stage_row(
                date(2024, 1, 1),
                limit_up_count=60,
                max_board_height=5,
                blow_rate=0.5,
                blow_rate_ma3=0.5,
                red_rate=0.35,
            )
        ]
    )
    classified = SentimentStageClassifier().classify_series(history, config)
    assert classified["sentiment_stage"].to_list() == ["start"]


def test_stage_classifier_has_no_lookahead():
    config = load_strategy_config()
    tie = {
        "limit_up_count": 25,
        "limit_down_count": 35,
        "max_board_height": 5,
        "blow_rate": 0.2,
        "blow_rate_ma3": 0.2,
        "red_rate": 0.5,
    }
    rows = [
        _stage_row(date(2024, 1, 1)),
        _stage_row(date(2024, 1, 2)),
        _stage_row(date(2024, 1, 3)),
        _stage_row(date(2024, 1, 4), **tie),
        _stage_row(date(2024, 1, 5), **tie),
        _stage_row(date(2024, 1, 6)),
    ]
    full = SentimentStageClassifier().classify_series(_history(rows), config)
    for cutoff in (date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)):
        truncated = SentimentStageClassifier().classify_series(
            _history([row for row in rows if row["trade_date"] <= cutoff]),
            config,
        )
        expected = full.filter(pl.col("trade_date") <= cutoff)[
            "sentiment_stage"
        ].to_list()
        assert truncated["sentiment_stage"].to_list() == expected


def test_stage_classifier_handles_empty_history():
    classified = SentimentStageClassifier().classify_series(
        pl.DataFrame(), load_strategy_config()
    )
    assert classified.is_empty()
    assert "sentiment_stage" in classified.columns
