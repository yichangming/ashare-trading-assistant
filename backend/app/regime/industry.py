from __future__ import annotations

from datetime import date

import polars as pl

from app.core.config import IndustryConfig
from app.domain.models import SectorStrength


class IndustryStrengthService:
    def compute(
        self,
        features_history: pl.DataFrame,
        as_of: date,
        config: IndustryConfig,
    ) -> list[SectorStrength]:
        if features_history.is_empty():
            return []
        frame = features_history.filter(pl.col("trade_date") <= as_of)
        if "is_suspended" in frame.columns:
            frame = frame.filter(~pl.col("is_suspended").fill_null(False))
        if frame.is_empty():
            return []

        amount_20d = pl.col("amount_20d_avg")
        frame = frame.with_columns(
            pl.when(amount_20d > 0)
            .then(pl.col("amount_5d_avg") / amount_20d - 1)
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("amount_change_stock"),
            (
                pl.col("prev_is_limit_up").fill_null(False)
                & (pl.col("prev_close") > 0)
                & (pl.col("close") > pl.col("prev_close"))
            )
            .fill_null(False)
            .alias("red"),
        )
        prev_limit_up = pl.col("prev_is_limit_up").fill_null(False)
        aggregated = (
            frame.group_by(["trade_date", "sector"])
            .agg(
                pl.col("ret_20d").mean().alias("return_20d"),
                pl.col("ret_5d").mean().alias("return_5d"),
                pl.col("above_ma20").mean().alias("breadth"),
                pl.col("amount_change_stock").mean().alias("amount_change"),
                pl.col("is_limit_up")
                .fill_null(False)
                .mean()
                .alias("limit_up_share"),
                pl.col("is_limit_up")
                .fill_null(False)
                .sum()
                .alias("limit_up_count"),
                pl.when(prev_limit_up.sum() > 0)
                .then(pl.col("red").sum() / prev_limit_up.sum())
                .otherwise(0.0)
                .alias("red_rate"),
                pl.col("ts_code").n_unique().alias("stock_count"),
            )
            .filter(pl.col("stock_count") >= config.min_stocks)
        )
        if aggregated.is_empty():
            return []

        weights = config.weights
        scored = aggregated.with_columns(
            (
                pl.col("return_20d").rank("average").over("trade_date")
                / pl.len().over("trade_date")
            ).alias("p20"),
            (
                pl.col("return_5d").rank("average").over("trade_date")
                / pl.len().over("trade_date")
            ).alias("p5"),
            (
                pl.col("breadth").rank("average").over("trade_date")
                / pl.len().over("trade_date")
            ).alias("pb"),
            (
                pl.col("amount_change").rank("average").over("trade_date")
                / pl.len().over("trade_date")
            ).alias("pa"),
            (
                pl.col("limit_up_share").rank("average").over("trade_date")
                / pl.len().over("trade_date")
            ).alias("pl"),
        ).with_columns(
            (
                weights.return_20d * pl.col("p20")
                + weights.return_5d * pl.col("p5")
                + weights.breadth * pl.col("pb")
                + weights.amount_change * pl.col("pa")
                + weights.limit_up_share * pl.col("pl")
            )
            .mul(100)
            .clip(0.0, 100.0)
            .alias("score")
        )
        # Sort before the ordinal rank so ties break deterministically by
        # sector name instead of relying on group-by row order.
        ranked = scored.sort(
            ["trade_date", "score", "sector"],
            descending=[False, True, False],
        ).with_columns(
            pl.col("score")
            .rank("ordinal", descending=True)
            .over("trade_date")
            .alias("daily_rank")
        )

        current = ranked.filter(pl.col("trade_date") == as_of).sort(
            ["score", "sector"], descending=[True, False]
        )
        if current.is_empty():
            return []

        ranks_by_date: dict[date, dict[str, int]] = {}
        for row in ranked.select(
            "trade_date", "sector", "daily_rank"
        ).iter_rows(named=True):
            ranks_by_date.setdefault(row["trade_date"], {})[
                row["sector"]
            ] = int(row["daily_rank"])
        dates = sorted(ranks_by_date)

        results: list[SectorStrength] = []
        for rank, row in enumerate(current.iter_rows(named=True), start=1):
            streak = 0
            for check_date in reversed(dates):
                day_rank = ranks_by_date[check_date].get(row["sector"])
                if day_rank is not None and day_rank <= config.top_n:
                    streak += 1
                else:
                    break
            results.append(
                SectorStrength(
                    sector=row["sector"],
                    score=float(row["score"]),
                    rank=rank,
                    return_20d=float(row["return_20d"]),
                    return_5d=float(row["return_5d"]),
                    breadth=float(row["breadth"]),
                    amount_change=float(row["amount_change"]),
                    limit_up_share=float(row["limit_up_share"]),
                    limit_up_count=int(row["limit_up_count"]),
                    red_rate=float(row["red_rate"]),
                    stock_count=int(row["stock_count"]),
                    persistent_days=streak,
                )
            )
        return results
