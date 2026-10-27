from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl

from app.core.config import StrategyConfig
from app.domain.enums import Template
from app.domain.models import Candidate, MarketRegime, SectorStrength
from app.screener.registry import register

# Ratios such as 10.2 / 10.0 - 1 land slightly above the configured
# boundary, so allow a negligible tolerance when checking MA proximity.
_MA_PROXIMITY_EPS = 1e-9


class DualTemplateScreener:
    name = "dual_template_v1"
    version = "1.1"

    def __init__(self) -> None:
        register(self)

    def generate(
        self,
        as_of: date,
        regime: MarketRegime,
        features: pl.DataFrame,
        sectors: list[SectorStrength],
        leaders: pl.DataFrame,
        config: StrategyConfig,
    ) -> list[Candidate]:
        # Empty frames are schema-less in practice, so guard them before
        # selecting leader columns or joining.
        if features.is_empty() or leaders.is_empty():
            return []
        leader_columns = leaders.select(
            pl.col("ts_code").cast(pl.String).alias("ts_code"),
            pl.col("leader_score").cast(pl.Float64).alias("leader_score"),
            pl.col("leader_rank").cast(pl.UInt32).alias("leader_rank"),
            pl.col("is_follower").cast(pl.Boolean).alias("is_follower"),
        )
        frame = features.with_columns(
            pl.col("ts_code").cast(pl.String).alias("ts_code")
        ).join(leader_columns, on="ts_code", how="left")
        sector_by_name = {sector.sector: sector for sector in sectors}
        candidates: list[Candidate] = []
        if self._template_active(regime, Template.A):
            candidates.extend(
                self._select_a(as_of, regime, frame, sector_by_name, config)
            )
        if self._template_active(regime, Template.B):
            candidates.extend(
                self._select_b(as_of, regime, frame, sector_by_name, config)
            )
        return candidates

    @staticmethod
    def _template_active(regime: MarketRegime, template: Template) -> bool:
        policy = regime.template_availability[template]
        return policy.enabled and policy.mode != "observe"

    def _select_a(
        self,
        as_of: date,
        regime: MarketRegime,
        frame: pl.DataFrame,
        sector_by_name: dict[str, SectorStrength],
        config: StrategyConfig,
    ) -> list[Candidate]:
        template = config.template_a
        selected = frame.filter(
            pl.col("sector").is_in(
                [
                    name
                    for name, sector in sector_by_name.items()
                    if sector.rank <= config.industry.top_n
                ]
            )
            & (pl.col("leader_rank") <= config.leader.top_n_per_industry)
            & (~pl.col("is_follower").fill_null(False))
            & (pl.col("hfq_close") > pl.col("ma20"))
            & (pl.col("ma20") > pl.col("ma60"))
            & (pl.col("ma20_slope") > 0)
            & (pl.col("rs_20d_market") >= template.rs_20d_market_min)
            & (
                pl.col("close_to_high_60d")
                >= 1 - template.max_distance_from_60d_high
            )
            & pl.col("pullback_days").is_between(
                template.pullback_days_min, template.pullback_days_max
            )
            & (pl.col("vol_ratio_5_20") < template.volume_shrink_ratio)
            & (
                (
                    (pl.col("hfq_close") / pl.col("ma10") - 1).abs()
                    <= 0.015 + _MA_PROXIMITY_EPS
                )
                | (
                    (pl.col("hfq_close") / pl.col("ma20") - 1).abs()
                    <= template.ma_zone_pct + _MA_PROXIMITY_EPS
                )
            )
            & (pl.col("stabilization"))
            & pl.col("circ_mv").is_between(
                template.market_cap_min, template.market_cap_max
            )
            & pl.col("close").is_between(template.min_price, template.max_price)
            & (~pl.col("is_st"))
            & (~pl.col("is_suspended"))
            & (~pl.col("is_limit_up"))
        )
        if selected.is_empty():
            return []
        scored = (
            selected.with_columns(
                ((1 - pl.col("vol_ratio_5_20")).clip(0, 1) * 400)
                .clip(0, 100)
                .alias("volume_score"),
                (
                    (
                        0.03
                        - pl.min_horizontal(
                            (pl.col("hfq_close") / pl.col("ma10") - 1).abs(),
                            (pl.col("hfq_close") / pl.col("ma20") - 1).abs(),
                        )
                    )
                    / 0.03
                    * 100
                )
                .clip(0, 100)
                .alias("ma_score"),
                (
                    100
                    - (
                        (pl.col("pullback_days") - 5.5).abs() / 2.5 * 100
                    ).clip(0, 100)
                ).alias("pullback_score"),
                pl.when(pl.col("stabilization"))
                .then(100.0)
                .otherwise(0.0)
                .alias("stabilization_score"),
            )
            .with_columns(
                (
                    0.4 * pl.col("volume_score")
                    + 0.2 * pl.col("ma_score")
                    + 0.2 * pl.col("pullback_score")
                    + 0.2 * pl.col("stabilization_score")
                ).alias("setup_score")
            )
            .with_columns(
                (
                    template.score.leader * pl.col("leader_score")
                    + template.score.setup * pl.col("setup_score")
                ).alias("total_score")
            )
            .sort(["total_score", "ts_code"], descending=[True, False])
        )

        results: list[Candidate] = []
        per_sector: dict[str, int] = {}
        for row in scored.iter_rows(named=True):
            if per_sector.get(row["sector"], 0) >= template.max_per_industry:
                continue
            if len(results) >= template.max_candidates:
                break
            per_sector[row["sector"]] = per_sector.get(row["sector"], 0) + 1
            results.append(
                self._candidate(
                    as_of=as_of,
                    regime=regime,
                    row=row,
                    template=Template.A,
                    setup_type="mainline_pullback",
                    horizon="mid",
                    score=float(row["total_score"]),
                    reasons=[
                        f"行业 {row['sector']} 强度前 {config.industry.top_n}",
                        f"龙头分第 {int(row['leader_rank'])} 名",
                        f"缩量比 {row['vol_ratio_5_20']:.2f}",
                        f"回调 {int(row['pullback_days'])} 天且贴近均线",
                        "企稳 K 线",
                    ],
                    config=config,
                )
            )
        return results

    def _select_b(
        self,
        as_of: date,
        regime: MarketRegime,
        frame: pl.DataFrame,
        sector_by_name: dict[str, SectorStrength],
        config: StrategyConfig,
    ) -> list[Candidate]:
        template = config.template_b
        if regime.indicators.blow_rate > template.blow_rate_max:
            return []
        if regime.indicators.max_board_height < template.max_board_min:
            return []
        eligible_sectors = [
            name
            for name, sector in sector_by_name.items()
            if sector.rank <= 2
            and (
                sector.limit_up_count >= template.min_limit_up_in_sector
                or sector.red_rate >= template.sector_red_rate_min
            )
        ]
        selected = frame.filter(
            pl.col("sector").is_in(eligible_sectors)
            & (pl.col("leader_rank") <= config.leader.top_n_per_industry)
            & (~pl.col("is_follower").fill_null(False))
            & (pl.col("is_limit_up"))
            & pl.col("board_height").is_between(1, 2)
            & (pl.col("vol_ratio_today_20") >= template.volume_ratio_min)
            & pl.col("turnover_rate").is_between(
                template.turnover_min, template.turnover_max
            )
            & pl.col("circ_mv").is_between(
                template.market_cap_min, template.market_cap_max
            )
            & pl.col("close").is_between(template.price_min, template.price_max)
            & (~pl.col("is_st"))
            & (~pl.col("is_suspended"))
            & (~pl.col("one_word"))
            & (pl.col("board_height") <= template.max_board)
            & (pl.col("ret_20d") <= template.max_20d_gain)
        )
        if selected.is_empty():
            return []
        degraded = any(
            flag in regime.degraded_flags
            for flag in ("limit_list_d", "industry_stock_basic")
        )
        scored = (
            selected.with_columns(
                (pl.col("board_height").clip(1, 2) / 2 * 100).alias(
                    "board_score"
                ),
                (pl.col("vol_ratio_today_20") / 4 * 100)
                .clip(0, 100)
                .alias("volume_score"),
                pl.col("turnover_rate")
                .is_between(template.turnover_min, template.turnover_max)
                .cast(pl.Float64)
                .mul(100)
                .alias("turnover_score"),
            )
            .with_columns(
                (
                    0.4 * pl.col("board_score")
                    + 0.3 * pl.col("volume_score")
                    + 0.2 * pl.col("turnover_score")
                    + 0.1 * 100
                ).alias("setup_score")
            )
            .with_columns(
                (
                    template.score.leader * pl.col("leader_score")
                    + template.score.setup * pl.col("setup_score")
                    - (5.0 if degraded else 0.0)
                ).alias("total_score")
            )
            .sort(["total_score", "ts_code"], descending=[True, False])
        )

        results: list[Candidate] = []
        per_sector: dict[str, int] = {}
        for row in scored.iter_rows(named=True):
            if per_sector.get(row["sector"], 0) >= template.max_per_industry:
                continue
            if len(results) >= template.max_candidates:
                break
            per_sector[row["sector"]] = per_sector.get(row["sector"], 0) + 1
            reasons = [
                f"主线 {row['sector']} 前 2",
                f"龙头分第 {int(row['leader_rank'])} 名",
                f"{int(row['board_height'])} 板，量比 "
                f"{row['vol_ratio_today_20']:.1f}",
                "未一字板",
            ]
            if degraded:
                reasons.append("B 模板数据降级")
            results.append(
                self._candidate(
                    as_of=as_of,
                    regime=regime,
                    row=row,
                    template=Template.B,
                    setup_type="sentiment_leader",
                    horizon="short",
                    score=float(row["total_score"]),
                    reasons=reasons,
                    config=config,
                )
            )
        return results

    def _candidate(
        self,
        as_of: date,
        regime: MarketRegime,
        row: dict,
        template: Template,
        setup_type: str,
        horizon: Literal["short", "mid"],
        score: float,
        reasons: list[str],
        config: StrategyConfig,
    ) -> Candidate:
        keys = [
            "hfq_close",
            "ma5",
            "ma10",
            "ma20",
            "ma60",
            "ma5_raw",
            "ma10_raw",
            "ma20_raw",
            "ma60_raw",
            "ma20_slope",
            "atr14",
            "atr14_raw",
            "low_5d_raw",
            "low_10d_raw",
            "high_60d_raw",
            "close_to_high_60d",
            "pullback_days",
            "vol_ratio_5_20",
            "vol_ratio_today_20",
            "turnover_rate",
            "circ_mv",
            "board_height",
            "days_since_limit_up",
            "one_word",
            "stabilization",
            "above_ma20",
            "ret_20d",
            "rs_20d_market",
            "close",
        ]
        features = {
            key: float(row[key]) for key in keys if row.get(key) is not None
        }
        return Candidate(
            ts_code=row["ts_code"],
            trade_date=as_of,
            template=template,
            setup_type=setup_type,
            horizon=horizon,
            sector=row["sector"],
            leader_rank=int(row["leader_rank"]),
            leader_score=float(row["leader_score"]),
            score=max(0.0, min(100.0, score)),
            reasons=reasons,
            features=features,
            sentiment_stage=regime.sentiment_stage,
            strategy_version=config.version,
            data_version=regime.data_version,
        )
