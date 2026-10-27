from datetime import date

import polars as pl
import pytest

from app.data.quality import DataQualityGate, QualityReport


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [date(2024, 1, 2)] * 2,
            "open": [10.0, 20.0],
            "high": [10.5, 20.5],
            "low": [9.8, 19.5],
            "close": [10.2, 20.2],
            "pre_close": [10.0, 20.0],
            "vol": [100000.0, 200000.0],
            "amount": [1000000.0, 2000000.0],
            "adj_factor": [1.0, 1.0],
            "up_limit": [11.0, 22.0],
            "down_limit": [9.0, 18.0],
        }
    )


def test_clean_data_passes():
    report = DataQualityGate().check(date(2024, 1, 2), _bars(), expected_count=2)
    assert isinstance(report, QualityReport)
    assert report.passed
    assert report.coverage == 1.0


def test_duplicate_and_ohlc_fail():
    bad = pl.concat([_bars(), _bars().head(1)]).with_columns(
        pl.when(pl.col("ts_code") == "000002.SZ")
        .then(19.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    report = DataQualityGate().check(date(2024, 1, 2), bad, expected_count=2)
    assert not report.passed
    assert any("duplicate" in item for item in report.failures)
    assert any("OHLC" in item for item in report.failures)


def test_missing_coverage_fails():
    report = DataQualityGate().check(date(2024, 1, 2), _bars().head(1), expected_count=3)
    assert not report.passed
    assert any("coverage" in item for item in report.failures)


def test_missing_sentiment_fields_fail():
    report = DataQualityGate().check_sentiment({"limit_up_count": 10})
    assert not report.passed
    assert any("missing sentiment field" in item for item in report.failures)


def test_derived_limit_prices_warn_but_pass():
    bars = _bars().with_columns(pl.Series("limit_derived", [True, False]))
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert report.passed
    assert any("limit prices derived for 1 rows" in item for item in report.warnings)
    assert not any("limit prices derived" in item for item in report.failures)


@pytest.mark.parametrize("column", ["up_limit", "down_limit"])
def test_null_limit_prices_fail(column: str):
    limits = _bars()[column].to_list()
    limits[-1] = None
    bars = _bars().with_columns(pl.Series(column, limits))
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("null limit prices: 1" in item for item in report.failures)
    # Null rows are handled by the explicit null check, not as mismatches.
    assert not any(
        "limit price mismatches" in item
        for item in report.failures + report.warnings
    )


def test_limit_mismatch_beyond_tolerance_fails():
    bars = _bars().with_columns(
        pl.Series("close", [10.2, 23.0]), pl.Series("high", [10.5, 23.5])
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("limit price mismatches: 1/2" in item for item in report.failures)


def test_limit_mismatch_within_tolerance_warns_but_passes():
    count = 1000
    bars = pl.DataFrame(
        {
            "ts_code": [f"{index:06d}.SZ" for index in range(count)],
            "trade_date": [date(2024, 1, 2)] * count,
            "open": [10.0] * count,
            "high": [12.5] + [10.5] * (count - 1),
            "low": [9.8] * count,
            "close": [12.0] + [10.2] * (count - 1),
            "pre_close": [10.0] * count,
            "vol": [100000.0] * count,
            "amount": [1000000.0] * count,
            "adj_factor": [1.0] * count,
            "up_limit": [11.0] * count,
            "down_limit": [9.0] * count,
        }
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=count)
    assert report.passed
    assert any("limit price mismatches: 1" in item for item in report.warnings)


def test_complete_sentiment_passes():
    report = DataQualityGate().check_sentiment(
        {
            "trade_date": date(2024, 1, 2),
            "limit_up_count": 10,
            "limit_down_count": 2,
            "max_board_height": 3,
            "promotion_rate": 0.5,
            "blow_rate": 0.1,
            "red_rate": 0.6,
            "limit_up_premium": 0.02,
        }
    )
    assert report.passed
    assert report.coverage == 1.0
    assert report.failures == []


def test_empty_bars_fail():
    report = DataQualityGate().check(
        date(2024, 1, 2), _bars().head(0), expected_count=2
    )
    assert not report.passed
    assert any("bars is empty" in item for item in report.failures)
    assert report.coverage == 0.0


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("adj_factor", [1.0, None], "invalid adj_factor"),
        ("adj_factor", [1.0, 0.0], "invalid adj_factor"),
        ("vol", [100000.0, -1.0], "negative volume/amount"),
        ("amount", [1000000.0, -1.0], "negative volume/amount"),
    ],
)
def test_invalid_values_fail(column: str, values: list[float | None], message: str):
    bars = _bars().with_columns(pl.Series(column, values))
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any(message in item for item in report.failures)
