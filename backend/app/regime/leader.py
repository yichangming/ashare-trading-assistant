from __future__ import annotations

import polars as pl

from app.core.config import StrategyConfig
from app.domain.models import SectorStrength

_REQUIRED_COLUMNS = (
    "trade_date",
    "ts_code",
    "sector",
    "ret_20d",
    "ret_5d",
    "amount_20d_avg",
    "turnover_rate",
    "close_to_high_60d",
    "board_height",
    "days_since_limit_up",
    "circ_mv",
    "close",
)
_FINITE_METRICS = (
    "ret_20d",
    "ret_5d",
    "amount_20d_avg",
    "turnover_rate",
    "close_to_high_60d",
    "circ_mv",
    "close",
)
_OUTPUT_SCHEMA = {
    "ts_code": pl.String,
    "sector": pl.String,
    "leader_score": pl.Float64,
    "leader_rank": pl.UInt32,
    "is_follower": pl.Boolean,
    "amount_20d_avg": pl.Float64,
}


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(schema=_OUTPUT_SCHEMA)


class LeaderScoreService:
    def compute(
        self,
        features: pl.DataFrame,
        sectors: list[SectorStrength],
        config: StrategyConfig,
    ) -> pl.DataFrame:
        if features.is_empty():
            return _empty_result()
        self._validate_features(features)
        if not sectors:
            return _empty_result()

        selected = sorted(
            {
                sector.sector
                for sector in sectors
                if sector.rank <= config.industry.top_n
                and sector.persistent_days >= config.industry.persistence_days
            }
        )
        if not selected:
            return _empty_result()

        frame = features.filter(pl.col("sector").is_in(selected))
        frame = frame.drop_nulls(subset=list(_REQUIRED_COLUMNS))
        frame = frame.filter(
            pl.all_horizontal(
                [pl.col(column).is_finite() for column in _FINITE_METRICS]
            )
        )
        if frame.is_empty():
            return _empty_result()

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
            # Tier thresholds are fixed design choices from the plan.
            pl.when(pl.col("board_height") >= 2)
            .then(100.0)
            .when(pl.col("board_height") == 1)
            .then(80.0)
            .when(pl.col("days_since_limit_up") <= 5)
            .then(50.0)
            .otherwise(0.0)
            .alias("limit_status"),
            # Preferred size/price band, then a looser acceptable band.
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
            .alias("leader_rank")
        )
        # The follower amount baseline is the rank-1 leader's amount, not
        # the sector maximum; broadcast it back to every sector row.
        ranked = ranked.with_columns(
            pl.when(pl.col("leader_rank") == 1)
            .then(pl.col("amount_20d_avg"))
            .max()
            .over("sector")
            .alias("sector_leader_amount")
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
            pl.col("ts_code"),
            pl.col("sector"),
            pl.col("leader_score"),
            pl.col("leader_rank"),
            pl.col("is_follower"),
            pl.col("amount_20d_avg").cast(pl.Float64),
        ).sort(["sector", "leader_rank", "ts_code"])

    @staticmethod
    def _validate_features(features: pl.DataFrame) -> None:
        missing = [
            column for column in _REQUIRED_COLUMNS if column not in features.columns
        ]
        if missing:
            raise ValueError(
                "features missing required columns: " + ", ".join(missing)
            )
        trade_date_count = features["trade_date"].n_unique()
        if trade_date_count != 1:
            raise ValueError(
                "features must contain exactly one trade_date, got "
                f"{trade_date_count}"
            )
        if features["ts_code"].n_unique() != features.height:
            raise ValueError("features contains duplicate ts_code rows")
