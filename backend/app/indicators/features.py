from __future__ import annotations

from datetime import date

import polars as pl


class FeatureBuilder:
    def __init__(self, atr_window: int = 14):
        self.atr_window = atr_window

    def build(self, bars: pl.DataFrame, as_of: date) -> pl.DataFrame:
        frame = bars.filter(pl.col("trade_date") <= as_of).sort(
            ["ts_code", "trade_date"]
        )
        frame = frame.with_columns(
            pl.int_range(0, pl.len()).over("ts_code").alias("_row_idx")
        )
        frame = frame.with_columns(
            (pl.col("close") * pl.col("adj_factor")).alias("hfq_close"),
            (pl.col("high") * pl.col("adj_factor")).alias("hfq_high"),
            (pl.col("low") * pl.col("adj_factor")).alias("hfq_low"),
            (pl.col("open") * pl.col("adj_factor")).alias("hfq_open"),
        )
        for window in (5, 10, 20, 60):
            frame = frame.with_columns(
                pl.col("hfq_close")
                .rolling_mean(window, min_samples=window)
                .over("ts_code")
                .alias(f"ma{window}")
            )
        for window in (5, 10, 20, 60):
            frame = frame.with_columns(
                (pl.col(f"ma{window}") / pl.col("adj_factor")).alias(
                    f"ma{window}_raw"
                )
            )

        frame = frame.with_columns(
            (pl.col("hfq_high") - pl.col("hfq_low")).alias("tr1"),
            (pl.col("hfq_high") - pl.col("hfq_close").shift(1).over("ts_code"))
            .abs()
            .alias("tr2"),
            (pl.col("hfq_low") - pl.col("hfq_close").shift(1).over("ts_code"))
            .abs()
            .alias("tr3"),
        ).with_columns(
            pl.max_horizontal("tr1", "tr2", "tr3").alias("true_range")
        )
        frame = frame.with_columns(
            pl.col("true_range")
            .rolling_mean(self.atr_window, min_samples=self.atr_window)
            .over("ts_code")
            .alias("atr14")
        ).with_columns(
            (pl.col("atr14") / pl.col("adj_factor")).alias("atr14_raw")
        )

        frame = frame.with_columns(
            (
                pl.col("hfq_close")
                / pl.col("hfq_close").shift(5).over("ts_code")
                - 1
            ).alias("ret_5d"),
            (
                pl.col("hfq_close")
                / pl.col("hfq_close").shift(20).over("ts_code")
                - 1
            ).alias("ret_20d"),
            (
                pl.col("hfq_close")
                / pl.col("hfq_close").shift(60).over("ts_code")
                - 1
            ).alias("ret_60d"),
        )

        frame = frame.with_columns(
            pl.col("vol")
            .rolling_mean(5, min_samples=5)
            .over("ts_code")
            .alias("vol_5d_avg"),
            pl.col("vol")
            .rolling_mean(20, min_samples=20)
            .over("ts_code")
            .alias("vol_20d_avg"),
            pl.col("amount")
            .rolling_mean(5, min_samples=5)
            .over("ts_code")
            .alias("amount_5d_avg"),
            pl.col("amount")
            .rolling_mean(20, min_samples=20)
            .over("ts_code")
            .alias("amount_20d_avg"),
        ).with_columns(
            (pl.col("vol_5d_avg") / pl.col("vol_20d_avg")).alias(
                "vol_ratio_5_20"
            ),
            (pl.col("vol") / pl.col("vol_20d_avg")).alias(
                "vol_ratio_today_20"
            ),
        )

        frame = frame.with_columns(
            pl.col("hfq_high")
            .rolling_max(60, min_samples=60)
            .over("ts_code")
            .alias("high_60d"),
            pl.col("hfq_low")
            .rolling_min(10, min_samples=10)
            .over("ts_code")
            .alias("low_10d"),
            pl.col("hfq_low")
            .rolling_min(5, min_samples=5)
            .over("ts_code")
            .alias("low_5d"),
            pl.col("hfq_high")
            .rolling_max(20, min_samples=20)
            .over("ts_code")
            .alias("high_20d"),
            pl.col("hfq_close")
            .rolling_max(20, min_samples=20)
            .over("ts_code")
            .alias("close_20d_high"),
        ).with_columns(
            (pl.col("hfq_close") / pl.col("high_60d")).alias(
                "close_to_high_60d"
            ),
            (pl.col("low_5d") / pl.col("adj_factor")).alias("low_5d_raw"),
            (pl.col("low_10d") / pl.col("adj_factor")).alias("low_10d_raw"),
            (pl.col("high_60d") / pl.col("adj_factor")).alias(
                "high_60d_raw"
            ),
        )

        frame = frame.with_columns(
            (pl.col("close") >= pl.col("up_limit") - 0.001).alias(
                "is_limit_up"
            ),
            (pl.col("close") <= pl.col("down_limit") + 0.001).alias(
                "is_limit_down"
            ),
            (pl.col("high") >= pl.col("up_limit") - 0.001).alias(
                "touched_limit_up"
            ),
        )
        frame = frame.with_columns(
            pl.col("close").shift(1).over("ts_code").alias("prev_close"),
            pl.col("is_limit_up").shift(1).over("ts_code").alias(
                "prev_is_limit_up"
            ),
            (pl.col("hfq_close") > pl.col("ma20")).alias("above_ma20"),
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
            pl.when(pl.col("is_limit_up"))
            .then(pl.col("_row_idx"))
            .otherwise(None)
            .forward_fill()
            .over("ts_code")
            .alias("last_limit_up_idx")
        ).with_columns(
            (pl.col("_row_idx") - pl.col("last_limit_up_idx"))
            .fill_null(999)
            .alias("days_since_limit_up")
        )
        frame = frame.with_columns(
            pl.when(pl.col("hfq_close") == pl.col("close_20d_high"))
            .then(pl.col("_row_idx"))
            .otherwise(None)
            .forward_fill()
            .over("ts_code")
            .alias("last_high_idx")
        ).with_columns(
            (pl.col("_row_idx") - pl.col("last_high_idx"))
            .fill_null(999)
            .alias("pullback_days")
        )
        body = (pl.col("close") - pl.col("open")).abs()
        lower_shadow = (
            pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")
        )
        day_range = pl.col("high") - pl.col("low")
        frame = frame.with_columns(
            (
                (pl.col("close") > pl.col("open"))
                | (lower_shadow >= body)
                | (pl.col("close") >= pl.col("low") + 0.5 * day_range)
            ).alias("stabilization"),
            (pl.col("high") == pl.col("low")).alias("one_word"),
            (pl.col("ma20") - pl.col("ma20").shift(10).over("ts_code")).alias(
                "ma20_slope"
            ),
        )

        asof = frame.filter(pl.col("trade_date") == as_of)
        if asof.is_empty():
            return asof
        asof = asof.with_columns(
            (pl.col("ret_20d").rank("average") / pl.len()).alias(
                "rs_20d_market"
            )
        )
        return asof
