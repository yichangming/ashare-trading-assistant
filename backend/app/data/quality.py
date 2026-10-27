from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl


@dataclass
class QualityReport:
    trade_date: date
    passed: bool
    coverage: float
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataQualityGate:
    def __init__(
        self,
        min_coverage: float = 0.99,
        max_limit_mismatch_pct: float = 0.001,
    ):
        self.min_coverage = min_coverage
        self.max_limit_mismatch_pct = max_limit_mismatch_pct

    def check(
        self, trade_date: date, bars: pl.DataFrame, expected_count: int
    ) -> QualityReport:
        failures: list[str] = []
        warnings: list[str] = []
        coverage = bars.height / expected_count if expected_count else 0.0

        if bars.is_empty():
            failures.append("bars is empty")
        if coverage < self.min_coverage:
            failures.append(
                f"coverage {coverage:.2%} below {self.min_coverage:.2%}"
            )
        duplicates = (
            bars.group_by(["ts_code", "trade_date"])
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        if duplicates:
            failures.append(f"duplicate (ts_code, trade_date) rows: {duplicates}")
        bad_ohlc = bars.filter(
            (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("high"))
        ).height
        if bad_ohlc:
            failures.append(f"OHLC violations: {bad_ohlc}")
        bad_adj = bars.filter(
            pl.col("adj_factor").is_null() | (pl.col("adj_factor") <= 0)
        ).height
        if bad_adj:
            failures.append(f"invalid adj_factor rows: {bad_adj}")
        negative = bars.filter(
            (pl.col("vol") < 0) | (pl.col("amount") < 0)
        ).height
        if negative:
            failures.append(f"negative volume/amount rows: {negative}")
        if {"up_limit", "down_limit"} <= set(bars.columns):
            null_limits = bars.filter(
                pl.col("up_limit").is_null() | pl.col("down_limit").is_null()
            ).height
            if null_limits:
                failures.append(f"null limit prices: {null_limits}")
            mismatched = bars.filter(
                (
                    (pl.col("close") > pl.col("up_limit") + 0.001)
                    | (pl.col("close") < pl.col("down_limit") - 0.001)
                ).fill_null(False)
            ).height
            if bars.height and mismatched / bars.height > self.max_limit_mismatch_pct:
                failures.append(
                    f"limit price mismatches: {mismatched}/{bars.height}"
                )
            elif mismatched:
                warnings.append(f"limit price mismatches: {mismatched}")
        if "limit_derived" in bars.columns:
            derived = bars.filter(pl.col("limit_derived").fill_null(False)).height
            if derived:
                warnings.append(f"limit prices derived for {derived} rows")
        return QualityReport(
            trade_date=trade_date,
            passed=not failures,
            coverage=coverage,
            failures=failures,
            warnings=warnings,
        )

    def check_sentiment(self, snapshot: dict) -> QualityReport:
        required = [
            "limit_up_count",
            "limit_down_count",
            "max_board_height",
            "promotion_rate",
            "blow_rate",
            "red_rate",
            "limit_up_premium",
        ]
        missing = [key for key in required if snapshot.get(key) is None]
        failures = [f"missing sentiment field: {key}" for key in missing]
        trade_date = snapshot.get("trade_date") or date.today()
        return QualityReport(
            trade_date=trade_date,
            passed=not failures,
            coverage=1.0 if not failures else 0.0,
            failures=failures,
        )
