from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import polars as pl

_KEY_COLUMNS = ("ts_code", "trade_date")
_OHLCV_COLUMNS = ("open", "high", "low", "close", "vol", "amount")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_REQUIRED_COLUMNS = (*_KEY_COLUMNS, *_OHLCV_COLUMNS, "adj_factor")
_FLOAT_COLUMNS = (*_OHLCV_COLUMNS, "adj_factor")
_SENTIMENT_COUNTS = ("limit_up_count", "limit_down_count", "max_board_height")
_SENTIMENT_RATES = ("promotion_rate", "blow_rate", "red_rate")
_SENTIMENT_PREMIUM = "limit_up_premium"
_SENTIMENT_FIELDS = (*_SENTIMENT_COUNTS, *_SENTIMENT_RATES, _SENTIMENT_PREMIUM)


def _validate_fraction(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")
    return float(value)


def _is_float_dtype(dtype: pl.DataType) -> bool:
    return dtype in (pl.Float32, pl.Float64)


@dataclass
class QualityReport:
    trade_date: date
    passed: bool
    coverage: float
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failure_codes: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)


class DataQualityGate:
    def __init__(
        self,
        min_coverage: float = 0.99,
        max_limit_mismatch_pct: float = 0.001,
    ):
        self.min_coverage = _validate_fraction("min_coverage", min_coverage)
        self.max_limit_mismatch_pct = _validate_fraction(
            "max_limit_mismatch_pct", max_limit_mismatch_pct
        )

    def check(
        self,
        trade_date: date,
        bars: pl.DataFrame,
        expected_count: int,
        expected_codes: set[str] | None = None,
    ) -> QualityReport:
        failures: list[str] = []
        warnings: list[str] = []
        failure_codes: list[str] = []
        warning_codes: list[str] = []

        def fail(message: str, code: str) -> None:
            failures.append(message)
            failure_codes.append(code)

        def warn(message: str, code: str) -> None:
            warnings.append(message)
            warning_codes.append(code)

        def report(coverage: float) -> QualityReport:
            return QualityReport(
                trade_date=trade_date,
                passed=not failures,
                coverage=coverage,
                failures=failures,
                warnings=warnings,
                failure_codes=failure_codes,
                warning_codes=warning_codes,
            )

        if bars.is_empty():
            coverage = 0.0
            fail("bars is empty", "bars_empty")
            if expected_count <= 0:
                fail(
                    f"expected universe unavailable: expected_count={expected_count}",
                    "expected_universe_unavailable",
                )
            elif coverage < self.min_coverage:
                fail(
                    f"coverage {coverage:.2%} below {self.min_coverage:.2%}",
                    "coverage_below_min",
                )
            if expected_codes:
                fail(
                    f"missing expected codes: {len(expected_codes)}",
                    "missing_expected_codes",
                )
            return report(coverage)

        missing = [column for column in _REQUIRED_COLUMNS if column not in bars.columns]
        if missing:
            fail(
                f"missing required columns: {', '.join(missing)}",
                "missing_required_columns",
            )
            return report(0.0)

        coverage = (
            bars["ts_code"].n_unique() / expected_count if expected_count > 0 else 0.0
        )
        if expected_count <= 0:
            fail(
                f"expected universe unavailable: expected_count={expected_count}",
                "expected_universe_unavailable",
            )
        else:
            if coverage < self.min_coverage:
                fail(
                    f"coverage {coverage:.2%} below {self.min_coverage:.2%}",
                    "coverage_below_min",
                )
            if coverage > 1.0:
                fail(
                    f"coverage {coverage:.2%} exceeds 100%",
                    "coverage_exceeds_one",
                )

        if expected_codes is not None:
            actual_codes = set(bars["ts_code"].drop_nulls().to_list())
            missing_codes = expected_codes - actual_codes
            extra_codes = actual_codes - expected_codes
            if missing_codes:
                fail(
                    f"missing expected codes: {len(missing_codes)}",
                    "missing_expected_codes",
                )
            if extra_codes:
                fail(
                    f"unexpected codes: {len(extra_codes)}",
                    "unexpected_codes",
                )

        unique_dates = bars["trade_date"].unique()
        if unique_dates.len() > 1:
            rendered = ", ".join(
                str(value) for value in unique_dates.sort().to_list()
            )
            fail(f"multiple trade dates: {rendered}", "multiple_trade_dates")
        elif unique_dates.len() == 1 and unique_dates[0] != trade_date:
            fail(
                f"trade_date mismatch: expected {trade_date}, got {unique_dates[0]}",
                "trade_date_mismatch",
            )

        duplicates = (
            bars.group_by(["ts_code", "trade_date"])
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        if duplicates:
            fail(
                f"duplicate (ts_code, trade_date) rows: {duplicates}",
                "duplicate_rows",
            )

        if "is_suspended" in bars.columns:
            suspended = (
                pl.col("is_suspended")
                .cast(pl.Boolean, strict=False)
                .fill_null(False)
            )
        else:
            suspended = pl.lit(False, dtype=pl.Boolean)

        def active(condition: pl.Expr) -> pl.Expr:
            return condition & ~suspended

        null_keys = bars.filter(
            pl.any_horizontal(
                [pl.col(column).is_null() for column in _KEY_COLUMNS]
            )
        ).height
        if null_keys:
            fail(f"null key rows: {null_keys}", "null_key")

        null_ohlcv = bars.filter(
            active(
                pl.any_horizontal(
                    [pl.col(column).is_null() for column in _OHLCV_COLUMNS]
                )
            )
        ).height
        if null_ohlcv:
            fail(f"null OHLCV rows: {null_ohlcv}", "null_ohlcv")

        float_columns = [
            column
            for column in _FLOAT_COLUMNS
            if _is_float_dtype(bars.schema[column])
        ]
        ohlcv_non_finite = pl.any_horizontal(
            [
                (~pl.col(column).is_finite()).fill_null(False)
                for column in float_columns
            ]
        )
        adj_non_finite = (
            (~pl.col("adj_factor").is_finite()).fill_null(False)
            if "adj_factor" in float_columns
            else pl.lit(False)
        )
        non_finite = bars.filter(
            active(ohlcv_non_finite) | adj_non_finite
        ).height
        if non_finite:
            fail(f"non-finite OHLCV rows: {non_finite}", "non_finite_ohlcv")

        bad_ohlc = bars.filter(
            active(
                (pl.col("low") > pl.col("open"))
                | (pl.col("low") > pl.col("close"))
                | (pl.col("high") < pl.col("open"))
                | (pl.col("high") < pl.col("close"))
                | (pl.col("low") > pl.col("high"))
            )
        ).height
        if bad_ohlc:
            fail(f"OHLC violations: {bad_ohlc}", "ohlc_violation")

        non_positive = bars.filter(
            active(
                pl.any_horizontal(
                    [
                        (pl.col(column) <= 0).fill_null(False)
                        for column in _PRICE_COLUMNS
                    ]
                )
            )
        ).height
        if non_positive:
            fail(f"non-positive prices: {non_positive}", "non_positive_price")

        bad_adj = bars.filter(
            pl.col("adj_factor").is_null() | (pl.col("adj_factor") <= 0)
        ).height
        if bad_adj:
            fail(f"invalid adj_factor rows: {bad_adj}", "invalid_adj_factor")

        negative = bars.filter(
            active((pl.col("vol") < 0) | (pl.col("amount") < 0))
        ).height
        if negative:
            fail(
                f"negative volume/amount rows: {negative}",
                "negative_volume_amount",
            )

        limit_columns = [
            column
            for column in ("up_limit", "down_limit")
            if column in bars.columns
        ]
        if not limit_columns:
            fail(
                "missing required columns: up_limit, down_limit",
                "missing_required_columns",
            )
        elif len(limit_columns) == 1:
            missing_limit = (
                "down_limit" if limit_columns[0] == "up_limit" else "up_limit"
            )
            fail(
                f"partial limit columns: missing {missing_limit}",
                "partial_limit_columns",
            )
        if limit_columns:
            non_finite_limit_columns = [
                column
                for column in limit_columns
                if _is_float_dtype(bars.schema[column])
            ]
            if non_finite_limit_columns:
                non_finite_limits = bars.filter(
                    active(
                        pl.any_horizontal(
                            [
                                (~pl.col(column).is_finite()).fill_null(False)
                                for column in non_finite_limit_columns
                            ]
                        )
                    )
                ).height
                if non_finite_limits:
                    fail(
                        f"non-finite limit prices: {non_finite_limits}",
                        "non_finite_limit_prices",
                    )

            null_limits = active(
                pl.any_horizontal(
                    [pl.col(column).is_null() for column in limit_columns]
                )
            )
            null_limit_rows = bars.filter(null_limits).height
            if null_limit_rows:
                fail(
                    f"null limit prices: {null_limit_rows}",
                    "null_limit_prices",
                )

            mismatch_conditions = []
            if "up_limit" in limit_columns:
                mismatch_conditions.append(
                    pl.col("close") > pl.col("up_limit") + 0.001
                )
            if "down_limit" in limit_columns:
                mismatch_conditions.append(
                    pl.col("close") < pl.col("down_limit") - 0.001
                )
            mismatched = bars.filter(
                active(
                    pl.any_horizontal(mismatch_conditions).fill_null(False)
                )
            ).height
            if (
                bars.height
                and mismatched / bars.height > self.max_limit_mismatch_pct
            ):
                fail(
                    f"limit price mismatches: {mismatched}/{bars.height}",
                    "limit_price_mismatch",
                )
            elif mismatched:
                warn(
                    f"limit price mismatches: {mismatched}",
                    "limit_price_mismatch",
                )

        if "limit_derived" in bars.columns:
            derived = bars.filter(
                pl.col("limit_derived")
                .cast(pl.Boolean, strict=False)
                .fill_null(False)
            ).height
            if derived:
                warn(
                    f"limit prices derived for {derived} rows",
                    "limit_prices_derived",
                )

        return report(coverage)

    def check_sentiment(self, snapshot: dict) -> QualityReport:
        failures: list[str] = []
        failure_codes: list[str] = []

        def fail(message: str, code: str) -> None:
            failures.append(message)
            failure_codes.append(code)

        trade_date = snapshot.get("trade_date")
        if trade_date is None:
            fail("missing sentiment field: trade_date", "missing_sentiment_field")
        elif not isinstance(trade_date, date):
            fail(
                "invalid sentiment field: trade_date (expected date)",
                "invalid_sentiment_field",
            )

        valid_fields = 0
        for field_name in _SENTIMENT_FIELDS:
            value = snapshot.get(field_name)
            if value is None:
                fail(
                    f"missing sentiment field: {field_name}",
                    "missing_sentiment_field",
                )
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                fail(
                    f"invalid sentiment field: {field_name} (expected numeric)",
                    "invalid_sentiment_field",
                )
                continue
            if not math.isfinite(value):
                fail(
                    f"invalid sentiment field: {field_name} "
                    "(expected finite number)",
                    "invalid_sentiment_field",
                )
                continue
            if field_name in _SENTIMENT_COUNTS:
                if value % 1 != 0:
                    fail(
                        f"invalid sentiment field: {field_name} "
                        "(expected integer)",
                        "invalid_sentiment_field",
                    )
                    continue
                if value < 0:
                    fail(
                        f"invalid sentiment field: {field_name} "
                        "(expected non-negative)",
                        "invalid_sentiment_field",
                    )
                    continue
            if field_name in _SENTIMENT_RATES and not 0 <= value <= 1:
                fail(
                    f"invalid sentiment field: {field_name} "
                    "(expected between 0 and 1)",
                    "invalid_sentiment_field",
                )
                continue
            valid_fields += 1

        if not isinstance(trade_date, date):
            trade_date = date.today()
        return QualityReport(
            trade_date=trade_date,
            passed=not failures,
            coverage=valid_fields / len(_SENTIMENT_FIELDS),
            failures=failures,
            failure_codes=failure_codes,
        )
