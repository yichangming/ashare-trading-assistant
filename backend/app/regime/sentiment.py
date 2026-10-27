from __future__ import annotations

from typing import Any

import polars as pl

from app.core.config import StrategyConfig
from app.domain.enums import SentimentStage


class SentimentService:
    def compute_history(self, bars: pl.DataFrame) -> pl.DataFrame:
        if bars.is_empty():
            return pl.DataFrame()
        frame = bars.sort(["ts_code", "trade_date"]).with_columns(
            (
                (pl.col("up_limit") > 0)
                & (pl.col("close") >= pl.col("up_limit") - 0.001)
            )
            .fill_null(False)
            .alias("is_limit_up"),
            (
                (pl.col("down_limit") > 0)
                & (pl.col("close") <= pl.col("down_limit") + 0.001)
            )
            .fill_null(False)
            .alias("is_limit_down"),
            (
                (pl.col("up_limit") > 0)
                & (pl.col("high") >= pl.col("up_limit") - 0.001)
            )
            .fill_null(False)
            .alias("touched_limit_up"),
        )
        # The exchange pre_close is the correct reference on ex-dividend days;
        # the previous raw close is only a fallback when it is unavailable.
        prev_close = (
            pl.col("pre_close")
            if "pre_close" in frame.columns
            else pl.col("close").shift(1).over("ts_code")
        )
        frame = frame.with_columns(
            prev_close.alias("prev_close"),
            pl.col("is_limit_up")
            .shift(1)
            .over("ts_code")
            .alias("prev_is_limit_up"),
        )
        frame = frame.with_columns(
            pl.when(pl.col("is_limit_up"))
            .then(0)
            .otherwise(1)
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
            pl.col("board_height")
            .shift(1)
            .over("ts_code")
            .alias("prev_board_height")
        )
        frame = frame.with_columns(
            (
                (pl.col("prev_board_height") > 0)
                & (
                    pl.col("board_height")
                    == pl.col("prev_board_height") + 1
                )
            ).alias("promoted"),
            (
                pl.col("prev_is_limit_up").fill_null(False)
                & (pl.col("close") > pl.col("prev_close"))
            )
            .fill_null(False)
            .alias("red"),
            pl.when(
                pl.col("prev_is_limit_up").fill_null(False)
                & (pl.col("prev_close") > 0)
            )
            .then(pl.col("open") / pl.col("prev_close") - 1)
            .otherwise(None)
            .alias("premium"),
            (
                pl.col("touched_limit_up").fill_null(False)
                & ~pl.col("is_limit_up")
            ).alias("blow"),
        )
        promoted = pl.col("promoted").sum()
        board_denominator = pl.col("prev_board_height").gt(0).sum()
        touched = pl.col("touched_limit_up").sum()
        prev_limit_up = pl.col("prev_is_limit_up").fill_null(False)
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
            pl.col("blow_rate")
            .rolling_mean(3, min_samples=1)
            .alias("blow_rate_ma3"),
            pl.col("max_board_height")
            .rolling_mean(3, min_samples=1)
            .alias("max_board_ma3"),
            pl.col("limit_up_premium")
            .rolling_mean(3, min_samples=1)
            .alias("premium_ma3"),
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
                pl.Series("sentiment_stage", [], dtype=pl.String)
            )
        sorted_history = history.sort("trade_date")
        current = SentimentStage.START
        pending: SentimentStage | None = None
        pending_count = 0
        stages: list[str] = []
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
            stages.append(current.value)
        return sorted_history.with_columns(
            pl.Series("sentiment_stage", stages)
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
                    )
                    or (
                        (row["limit_up_premium"] - row["premium_ma3"]) * 100
                        <= -thresholds.climax.premium_drop_pp
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
