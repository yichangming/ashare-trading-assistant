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


def test_missing_required_columns_fail():
    report = DataQualityGate().check(
        date(2024, 1, 2), _bars().drop("adj_factor"), expected_count=2
    )
    assert not report.passed
    assert any(
        "missing required columns: adj_factor" in item for item in report.failures
    )
    assert "missing_required_columns" in report.failure_codes


def test_zero_column_empty_frame_fails_without_crashing():
    report = DataQualityGate().check(
        date(2024, 1, 2), pl.DataFrame(), expected_count=2
    )
    assert not report.passed
    assert report.coverage == 0.0
    assert "bars_empty" in report.failure_codes


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "vol", "amount"])
def test_null_market_data_fails(column: str):
    values = _bars()[column].to_list()
    values[1] = None
    bars = _bars().with_columns(pl.Series(column, values))
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("null OHLCV rows: 1" in item for item in report.failures)
    assert "null_ohlcv" in report.failure_codes


@pytest.mark.parametrize("column", ["adj_factor", "vol", "amount"])
def test_non_finite_market_data_fails(column: str):
    values = _bars()[column].to_list()
    values[1] = float("nan")
    bars = _bars().with_columns(pl.Series(column, values))
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("non-finite OHLCV rows: 1" in item for item in report.failures)
    assert "non_finite_ohlcv" in report.failure_codes


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_prices_fail(column: str, value: float):
    values = _bars()[column].to_list()
    values[0] = value
    bars = _bars().with_columns(pl.Series(column, values))
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("non-positive prices: 1" in item for item in report.failures)
    assert "non_positive_price" in report.failure_codes


def test_zero_volume_and_amount_allowed():
    bars = _bars().with_columns(
        pl.Series("vol", [0.0, 0.0]), pl.Series("amount", [0.0, 0.0])
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert report.passed
    assert "negative_volume_amount" not in report.failure_codes


def test_multiple_trade_dates_fail():
    bars = _bars().with_columns(
        pl.Series("trade_date", [date(2024, 1, 2), date(2024, 1, 3)])
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("multiple trade dates" in item for item in report.failures)
    assert "multiple_trade_dates" in report.failure_codes
    assert report.coverage == 1.0


def test_wrong_trade_date_fails():
    bars = _bars().with_columns(pl.Series("trade_date", [date(2024, 1, 3)] * 2))
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("trade_date mismatch" in item for item in report.failures)
    assert "trade_date_mismatch" in report.failure_codes


def test_duplicate_rows_fail_and_coverage_uses_unique_codes():
    bars = pl.concat([_bars(), _bars().head(1)])
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert "duplicate_rows" in report.failure_codes
    # Raw row height would give 1.5; unique ts_code coverage is 1.0.
    assert report.coverage == 1.0


def test_expected_count_unavailable_fails():
    report = DataQualityGate().check(date(2024, 1, 2), _bars(), expected_count=0)
    assert not report.passed
    assert "expected_universe_unavailable" in report.failure_codes
    assert report.coverage == 0.0


def test_coverage_above_one_fails():
    report = DataQualityGate().check(date(2024, 1, 2), _bars(), expected_count=1)
    assert not report.passed
    assert "coverage_exceeds_one" in report.failure_codes


def test_suspended_rows_exempt_from_null_limit_failure():
    partly_suspended = _bars().with_columns(
        pl.Series("up_limit", [None, None]),
        pl.Series("is_suspended", [True, False]),
    )
    report = DataQualityGate().check(
        date(2024, 1, 2), partly_suspended, expected_count=2
    )
    assert not report.passed
    assert any("null limit prices: 1" in item for item in report.failures)
    assert "null_limit_prices" in report.failure_codes

    fully_suspended = _bars().with_columns(
        pl.Series("up_limit", [None, None]),
        pl.Series("is_suspended", [True, True]),
    )
    assert DataQualityGate().check(
        date(2024, 1, 2), fully_suspended, expected_count=2
    ).passed


def test_partial_limit_columns_fail_and_remaining_column_checked():
    bars = _bars().drop("down_limit").with_columns(
        pl.Series("up_limit", [9.0, None])
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("partial limit columns" in item for item in report.failures)
    assert "partial_limit_columns" in report.failure_codes
    assert any("null limit prices: 1" in item for item in report.failures)
    assert any("limit price mismatches: 1/2" in item for item in report.failures)
    assert "limit_price_mismatch" in report.failure_codes


def test_integer_limit_derived_column_warns():
    bars = _bars().with_columns(
        pl.Series("limit_derived", [1, None], dtype=pl.Int64)
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert report.passed
    assert any("limit prices derived for 1 rows" in item for item in report.warnings)
    assert "limit_prices_derived" in report.warning_codes


def _sentiment() -> dict:
    return {
        "trade_date": date(2024, 1, 2),
        "limit_up_count": 10,
        "limit_down_count": 2,
        "max_board_height": 3,
        "promotion_rate": 0.5,
        "blow_rate": 0.1,
        "red_rate": 0.6,
        "limit_up_premium": 0.02,
    }


def test_sentiment_missing_trade_date_fails():
    snapshot = _sentiment()
    snapshot.pop("trade_date")
    report = DataQualityGate().check_sentiment(snapshot)
    assert not report.passed
    assert any("trade_date" in item for item in report.failures)
    assert "missing_sentiment_field" in report.failure_codes
    assert report.coverage == 1.0


def test_sentiment_invalid_trade_date_fails():
    snapshot = _sentiment()
    snapshot["trade_date"] = "2024-01-02"
    report = DataQualityGate().check_sentiment(snapshot)
    assert not report.passed
    assert any("trade_date" in item for item in report.failures)
    assert "invalid_sentiment_field" in report.failure_codes


def test_sentiment_invalid_values_fail():
    snapshot = _sentiment()
    snapshot.update(
        {
            "limit_up_count": "10",
            "promotion_rate": float("nan"),
            "red_rate": 1.5,
        }
    )
    report = DataQualityGate().check_sentiment(snapshot)
    assert not report.passed
    assert any("limit_up_count" in item for item in report.failures)
    assert any("promotion_rate" in item for item in report.failures)
    assert any("red_rate" in item for item in report.failures)
    assert "invalid_sentiment_field" in report.failure_codes
    assert report.coverage == 4 / 7


def test_sentiment_out_of_range_values_fail():
    snapshot = _sentiment()
    snapshot.update(
        {"limit_down_count": -1, "max_board_height": -2, "blow_rate": 1.2}
    )
    report = DataQualityGate().check_sentiment(snapshot)
    assert not report.passed
    assert any("limit_down_count" in item for item in report.failures)
    assert any("max_board_height" in item for item in report.failures)
    assert any("blow_rate" in item for item in report.failures)
    assert report.coverage == 4 / 7


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"min_coverage": 1.1}, "min_coverage"),
        ({"min_coverage": -0.01}, "min_coverage"),
        ({"min_coverage": float("nan")}, "min_coverage"),
        ({"max_limit_mismatch_pct": 2.0}, "max_limit_mismatch_pct"),
        ({"max_limit_mismatch_pct": -1.0}, "max_limit_mismatch_pct"),
    ],
)
def test_invalid_thresholds_raise(kwargs: dict, field: str):
    with pytest.raises(ValueError, match=field):
        DataQualityGate(**kwargs)


@pytest.mark.parametrize("premium", [-0.03, 1.2])
def test_signed_limit_up_premium_passes(premium: float):
    snapshot = _sentiment()
    snapshot["limit_up_premium"] = premium
    report = DataQualityGate().check_sentiment(snapshot)
    assert report.passed
    assert report.coverage == 1.0


def test_non_finite_limit_up_premium_fails():
    snapshot = _sentiment()
    snapshot["limit_up_premium"] = float("inf")
    report = DataQualityGate().check_sentiment(snapshot)
    assert not report.passed
    assert any("limit_up_premium" in item for item in report.failures)
    assert "invalid_sentiment_field" in report.failure_codes
    assert report.coverage == 6 / 7


def test_fractional_sentiment_count_fails():
    snapshot = _sentiment()
    snapshot["limit_up_count"] = 2.5
    report = DataQualityGate().check_sentiment(snapshot)
    assert not report.passed
    assert any("limit_up_count" in item for item in report.failures)
    assert "invalid_sentiment_field" in report.failure_codes
    assert report.coverage == 6 / 7


def test_integral_float_sentiment_count_passes():
    snapshot = _sentiment()
    snapshot["limit_up_count"] = 3.0
    report = DataQualityGate().check_sentiment(snapshot)
    assert report.passed


@pytest.mark.parametrize("column", ["up_limit", "down_limit"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_limit_prices_fail(column: str, value: float):
    values = _bars()[column].to_list()
    values[1] = value
    bars = _bars().with_columns(pl.Series(column, values))
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("non-finite limit prices: 1" in item for item in report.failures)
    assert "non_finite_limit_prices" in report.failure_codes


def test_suspended_zero_volume_null_price_row_passes():
    bars = _bars().with_columns(
        pl.Series("open", [10.0, None]),
        pl.Series("high", [10.5, None]),
        pl.Series("low", [9.8, None]),
        pl.Series("close", [10.2, None]),
        pl.Series("vol", [100000.0, 0.0]),
        pl.Series("amount", [1000000.0, 0.0]),
        pl.Series("up_limit", [11.0, None]),
        pl.Series("down_limit", [9.0, None]),
        pl.Series("is_suspended", [False, True]),
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert report.passed, report.failures


def test_suspended_zero_price_row_passes():
    bars = _bars().with_columns(
        pl.Series("open", [10.0, 0.0]),
        pl.Series("high", [10.5, 0.0]),
        pl.Series("low", [9.8, 0.0]),
        pl.Series("close", [10.2, 0.0]),
        pl.Series("vol", [100000.0, 0.0]),
        pl.Series("amount", [1000000.0, 0.0]),
        pl.Series("is_suspended", [False, True]),
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert report.passed, report.failures


def test_suspended_rows_exempt_from_non_finite_values_and_mismatch():
    bars = _bars().with_columns(
        pl.Series("close", [10.2, float("nan")]),
        pl.Series("up_limit", [11.0, float("nan")]),
        pl.Series("is_suspended", [False, True]),
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert report.passed, report.failures

    mismatched = _bars().with_columns(
        pl.Series("close", [10.2, 23.0]),
        pl.Series("high", [10.5, 23.5]),
        pl.Series("is_suspended", [False, True]),
    )
    report = DataQualityGate().check(
        date(2024, 1, 2), mismatched, expected_count=2
    )
    assert report.passed, report.failures


def test_suspended_mask_accepts_integer_and_null():
    bars = _bars().with_columns(
        pl.Series("up_limit", [None, None]),
        pl.Series("is_suspended", [1, None], dtype=pl.Int64),
    )
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any("null limit prices: 1" in item for item in report.failures)


def test_missing_both_limit_columns_fail():
    bars = _bars().drop(["up_limit", "down_limit"])
    report = DataQualityGate().check(date(2024, 1, 2), bars, expected_count=2)
    assert not report.passed
    assert any(
        "missing required columns" in item
        and "up_limit" in item
        and "down_limit" in item
        for item in report.failures
    )
    assert "missing_required_columns" in report.failure_codes


def test_expected_codes_clean_match_passes():
    report = DataQualityGate().check(
        date(2024, 1, 2),
        _bars(),
        expected_count=2,
        expected_codes={"000001.SZ", "000002.SZ"},
    )
    assert report.passed
    assert report.failure_codes == []


def test_expected_codes_same_count_swap_fails():
    report = DataQualityGate().check(
        date(2024, 1, 2),
        _bars(),
        expected_count=2,
        expected_codes={"000001.SZ", "000003.SZ"},
    )
    assert not report.passed
    assert any("missing expected codes: 1" in item for item in report.failures)
    assert any("unexpected codes: 1" in item for item in report.failures)
    assert "missing_expected_codes" in report.failure_codes
    assert "unexpected_codes" in report.failure_codes


def test_expected_codes_extra_code_reports_unexpected_not_coverage():
    report = DataQualityGate().check(
        date(2024, 1, 2),
        _bars(),
        expected_count=1,
        expected_codes={"000001.SZ"},
    )
    assert not report.passed
    assert "unexpected_codes" in report.failure_codes
    # Exact expected codes identify the stray code; the generic coverage
    # failure would only duplicate that diagnosis.
    assert "coverage_exceeds_one" not in report.failure_codes
    assert not any("coverage" in item for item in report.failures)
