from __future__ import annotations

import polars as pl

from app.core.config import StrategyConfig
from app.domain.models import SectorStrength

_REQUIRED_METRICS = (
    "ret_20d",
    "ret_5d",
    "amount_20d_avg",
    "turnover_rate",
    "close_to_high_60d",
    "circ_mv",
    "close",
    "days_since_limit_up",
)


class LeaderScoreService:
    def compute(
        self,
        features: pl.DataFrame,
        sectors: list[SectorStrength],
        config: StrategyConfig,
    ) -> pl.DataFrame:
        if features.is_empty() or not sectors:
            return pl.DataFrame()

        selected = sorted(
            {
                sector.sector
                for sector in sectors
                if sector.rank <= config.industry.top_n
                and sector.persistent_days >= config.industry.persistence_days
            }
        )
        if not selected:
            return pl.DataFrame()

        frame = features.filter(pl.col("sector").is_in(selected))
        frame = frame.drop_nulls(subset=list(_REQUIRED_METRICS))
        if frame.is_empty():
            return pl.DataFrame()

        weights = config.leader.weights
        frame = frame.with_columns(
            (
                pl.col("ret_20d").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("rs_20d_pct"),
            (
                pl.col("ret_5d").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("rs_5d_pct"),
            (
                pl.col("amount_20d_avg").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("amount_pct"),
            (
                pl.col("turnover_rate").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("turnover_pct"),
            (
                pl.col("close_to_high_60d").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("high_pct"),
            pl.when(pl.col("board_height") >= 2)
            .then(100.0)
            .when(pl.col("board_height") == 1)
            .then(80.0)
            .when(pl.col("days_since_limit_up") <= 5)
            .then(50.0)
            .otherwise(0.0)
            .alias("limit_status"),
            pl.when(
                pl.col("circ_mv").is_between(5e9, 5e10)
                & pl.col("close").is_between(5, 100)
            )
            .then(100.0)
            .when(
                pl.col("circ_mv").is_between(2.5e9, 1e11)
                & pl.col("close").is_between(3, 150)
            )
            .then(50.0)
            .otherwise(0.0)
            .alias("size_fit"),
        ).with_columns(
            (
                weights.rs_20d * pl.col("rs_20d_pct")
                + weights.rs_5d * pl.col("rs_5d_pct")
                + weights.amount_turnover
                * (0.5 * pl.col("amount_pct") + 0.5 * pl.col("turnover_pct"))
                + weights.high_distance * pl.col("high_pct")
                + weights.limit_up_status
                * (pl.col("limit_status") / 100)
                + weights.size_price_fit * (pl.col("size_fit") / 100)
            )
            .mul(100)
            .clip(0.0, 100.0)
            .alias("leader_score")
        )

        # Sort before the ordinal rank so tied scores break by ts_code
        # instead of relying on the caller's row order.
        ranked = frame.sort(
            ["sector", "leader_score", "ts_code"],
            descending=[False, True, False],
        ).with_columns(
            pl.col("leader_score")
            .rank("ordinal", descending=True)
            .over("sector")
            .alias("leader_rank"),
            pl.col("amount_20d_avg")
            .max()
            .over("sector")
            .alias("sector_leader_amount"),
        )

        leader_amount = pl.col("sector_leader_amount")
        result = ranked.with_columns(
            (
                (pl.col("leader_rank") > config.leader.follower_rank_cutoff)
                | (
                    (leader_amount > 0)
                    & (
                        pl.col("amount_20d_avg")
                        < config.leader.amount_vs_leader_min * leader_amount
                    )
                )
            )
            .fill_null(False)
            .alias("is_follower")
        )
        return result.select(
            [
                "ts_code",
                "sector",
                "leader_score",
                "leader_rank",
                "is_follower",
                "amount_20d_avg",
            ]
        ).sort(["sector", "leader_rank", "ts_code"])
