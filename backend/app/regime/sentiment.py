from __future__ import annotations

from typing import Any

import polars as pl

from app.core.config import StrategyConfig
from app.domain.enums import SentimentStage

_HISTORY_SCHEMA: dict[str, Any] = {
    "trade_date": pl.Date,
    "limit_up_count": pl.UInt32,
    "limit_down_count": pl.UInt32,
    "max_board_height": pl.Int64,
    "promotion_rate": pl.Float64,
    "blow_rate": pl.Float64,
    "red_rate": pl.Float64,
    "limit_up_premium": pl.Float64,
    "blow_rate_ma3": pl.Float64,
    "max_board_ma3": pl.Float64,
    "premium_ma3": pl.Float64,
}
_PP_EPSILON = 1e-9


def _prior_three_day_mean(column: str) -> pl.Expr:
    """Mean of the prior three rows, falling back to the first row's value."""
    return pl.coalesce(
        [
            pl.col(column).rolling_mean(3, min_samples=1).shift(1),
            pl.col(column),
        ]
    )


class SentimentService:
    def compute_history(self, bars: pl.DataFrame) -> pl.DataFrame:
        if bars.is_empty():
            return pl.DataFrame(schema=_HISTORY_SCHEMA)
        frame = bars.sort(["ts_code", "trade_date"])
        if "is_suspended" in frame.columns:
            frame = frame.with_columns(
                (~pl.col("is_suspended").fill_null(False)).alias("_active")
            )
        else:
            frame = frame.with_columns(pl.lit(True).alias("_active"))
        frame = frame.with_columns(
            (
                pl.col("_active")
                & (pl.col("up_limit") > 0)
                & (pl.col("close") >= pl.col("up_limit") - 0.001)
            )
            .fill_null(False)
            .alias("is_limit_up"),
            (
                pl.col("_active")
                & (pl.col("down_limit") > 0)
                & (pl.col("close") <= pl.col("down_limit") + 0.001)
            )
            .fill_null(False)
            .alias("is_limit_down"),
            (
                pl.col("_active")
                & (pl.col("up_limit") > 0)
                & (pl.col("high") >= pl.col("up_limit") - 0.001)
            )
            .fill_null(False)
            .alias("touched_limit_up"),
        )

        # Market date sequence, used to detect stocks missing a market day.
        market_dates = frame.select(
            pl.col("trade_date").unique().sort()
        ).with_columns(
            pl.int_range(0, pl.len()).alias("_market_idx")
        )
        frame = frame.join(market_dates, on="trade_date", how="left").sort(
            ["ts_code", "trade_date"]
        )
        frame = frame.with_columns(
            pl.col("_market_idx")
            .shift(1)
            .over("ts_code")
            .alias("_prev_market_idx")
        ).with_columns(
            (
                pl.col("_prev_market_idx").is_null()
                | (
                    pl.col("_market_idx")
                    != pl.col("_prev_market_idx") + 1
                )
            ).alias("_date_gap")
        )

        # The exchange pre_close is the correct reference on ex-dividend days;
        # the previous raw close is the fallback when it is unavailable.
        previous_close = pl.col("close").shift(1).over("ts_code")
        prev_close = (
            pl.coalesce([pl.col("pre_close"), previous_close])
            if "pre_close" in frame.columns
            else previous_close
        )
        frame = frame.with_columns(
            prev_close.alias("prev_close"),
            pl.when(pl.col("_date_gap"))
            .then(False)
            .otherwise(pl.col("is_limit_up").shift(1).over("ts_code"))
            .fill_null(False)
            .alias("prev_is_limit_up"),
        )
        frame = frame.with_columns(
            pl.when(~pl.col("is_limit_up") | pl.col("_date_gap"))
            .then(1)
            .otherwise(0)
            .cum_sum()
            .over("ts_code")
            .alias("lu_reset")
        ).with_columns(
            pl.col("is_limit_up")
            .cast(pl.Int8)
            .cum_sum()
            .over(["ts_code", "lu_reset"])
            .alias("board_height")
        )
        frame = frame.with_columns(
            pl.when(pl.col("_date_gap"))
            .then(0)
            .otherwise(pl.col("board_height").shift(1).over("ts_code"))
            .fill_null(0)
            .alias("prev_board_height")
        )
        frame = frame.with_columns(
            (
                pl.col("_active")
                & (pl.col("prev_board_height") > 0)
                & (
                    pl.col("board_height")
                    == pl.col("prev_board_height") + 1
                )
            ).alias("promoted"),
            (
                pl.col("_active")
                & pl.col("prev_is_limit_up").fill_null(False)
                & (pl.col("prev_close") > 0)
                & (pl.col("close") > pl.col("prev_close"))
            )
            .fill_null(False)
            .alias("red"),
            pl.when(
                pl.col("_active")
                & pl.col("prev_is_limit_up").fill_null(False)
                & (pl.col("prev_close") > 0)
            )
            .then(pl.col("open") / pl.col("prev_close") - 1)
            .otherwise(None)
            .alias("premium"),
            (
                pl.col("_active")
                & pl.col("touched_limit_up").fill_null(False)
                & ~pl.col("is_limit_up")
            )
            .fill_null(False)
            .alias("blow"),
        )
        promoted = pl.col("promoted").sum()
        board_denominator = (
            (pl.col("prev_board_height") > 0) & pl.col("_active")
        ).sum()
        touched = pl.col("touched_limit_up").sum()
        prev_limit_up = (
            pl.col("prev_is_limit_up").fill_null(False)
            & pl.col("_active")
        )
        grouped = (
            frame.group_by("trade_date")
            .agg(
                pl.col("is_limit_up").sum().alias("limit_up_count"),
                pl.col("is_limit_down").sum().alias("limit_down_count"),
                pl.col("board_height").max().alias("max_board_height"),
                pl.when(board_denominator > 0)
                .then(promoted / board_denominator)
                .otherwise(0.0)
                .alias("promotion_rate"),
                pl.when(touched > 0)
                .then(pl.col("blow").sum() / touched)
                .otherwise(0.0)
                .alias("blow_rate"),
                pl.when(prev_limit_up.sum() > 0)
                .then(pl.col("red").sum() / prev_limit_up.sum())
                .otherwise(0.0)
                .alias("red_rate"),
                pl.col("premium")
                .mean()
                .fill_null(0.0)
                .alias("limit_up_premium"),
            )
            .sort("trade_date")
        )
        return grouped.with_columns(
            _prior_three_day_mean("blow_rate").alias("blow_rate_ma3"),
            _prior_three_day_mean("max_board_height").alias(
                "max_board_ma3"
            ),
            _prior_three_day_mean("limit_up_premium").alias(
                "premium_ma3"
            ),
        )


class SentimentStageClassifier:
    _PRIORITY = (
        SentimentStage.EBB,
        SentimentStage.FREEZE,
        SentimentStage.CLIMAX,
        SentimentStage.FERMENT,
        SentimentStage.START,
    )

    def classify_series(
        self, history: pl.DataFrame, config: StrategyConfig
    ) -> pl.DataFrame:
        if history.is_empty():
            return history.with_columns(
                pl.Series("sentiment_stage", [], dtype=pl.String),
                pl.Series("initial_stage", [], dtype=pl.Boolean),
            )
        sorted_history = history.sort("trade_date")
        current = SentimentStage.START
        pending: SentimentStage | None = None
        pending_count = 0
        initial_stage = True
        stages: list[str] = []
        initial_flags: list[bool] = []
        confirm_days = config.environment.stage_confirm_days
        for row in sorted_history.iter_rows(named=True):
            candidate = self._classify_row(row, config)
            if candidate is None or candidate == current:
                pending = None
                pending_count = 0
            else:
                if candidate == pending:
                    pending_count += 1
                else:
                    pending = candidate
                    pending_count = 1
                if pending_count >= confirm_days:
                    current = candidate
                    pending = None
                    pending_count = 0
                    initial_stage = False
            stages.append(current.value)
            initial_flags.append(initial_stage)
        return sorted_history.with_columns(
            pl.Series("sentiment_stage", stages),
            pl.Series("initial_stage", initial_flags),
        )

    def _classify_row(
        self, row: dict[str, Any], config: StrategyConfig
    ) -> SentimentStage | None:
        thresholds = config.environment.thresholds
        counts = {
            SentimentStage.FREEZE: sum(
                [
                    row["limit_up_count"] < thresholds.freeze.limit_up_max,
                    row["limit_down_count"] > thresholds.freeze.limit_down_min
                    or row["blow_rate"] > thresholds.freeze.blow_rate_min,
                    row["max_board_height"] <= thresholds.freeze.max_board_max,
                    row["red_rate"] < thresholds.freeze.red_rate_max,
                ]
            ),
            SentimentStage.START: sum(
                [
                    thresholds.start.limit_up_min
                    <= row["limit_up_count"]
                    <= thresholds.start.limit_up_max,
                    thresholds.start.max_board_min
                    <= row["max_board_height"]
                    <= thresholds.start.max_board_max,
                    thresholds.start.blow_rate_min
                    <= row["blow_rate"]
                    <= thresholds.start.blow_rate_max,
                    thresholds.start.red_rate_min
                    <= row["red_rate"]
                    <= thresholds.start.red_rate_max,
                ]
            ),
            SentimentStage.FERMENT: sum(
                [
                    row["limit_up_count"] > thresholds.ferment.limit_up_min,
                    row["max_board_height"] >= thresholds.ferment.max_board_min,
                    row["blow_rate"] < thresholds.ferment.blow_rate_max,
                    row["red_rate"] > thresholds.ferment.red_rate_min,
                ]
            ),
            SentimentStage.CLIMAX: sum(
                [
                    row["limit_up_count"] > thresholds.climax.limit_up_min,
                    row["max_board_height"] >= thresholds.climax.max_board_min,
                    row["red_rate"] > thresholds.climax.red_rate_min,
                    (
                        (row["blow_rate"] - row["blow_rate_ma3"]) * 100
                        >= thresholds.climax.blow_rate_rise_pp
                        - _PP_EPSILON
                    )
                    or (
                        (row["premium_ma3"] - row["limit_up_premium"]) * 100
                        >= thresholds.climax.premium_drop_pp - _PP_EPSILON
                    ),
                ]
            ),
            SentimentStage.EBB: sum(
                [
                    row["limit_up_count"] < thresholds.ebb.limit_up_max,
                    row["limit_down_count"] > thresholds.ebb.limit_down_min,
                    row["blow_rate"] > thresholds.ebb.blow_rate_min,
                    row["max_board_height"] < row["max_board_ma3"],
                    row["red_rate"] < thresholds.ebb.red_rate_max,
                ]
            ),
        }
        best = max(counts.values())
        if best < 2:
            return None
        for stage in self._PRIORITY:
            if counts[stage] == best:
                return stage
        return None
