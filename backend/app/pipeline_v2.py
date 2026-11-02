"""Six-strategy planning pipeline (A-F).

This pipeline reuses the M1 data layer and the existing regime/industry/leader
services, then replaces the legacy A/B screener/planner with the A-F strategy
registry and the unified strategy planner/sizer/governor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import polars as pl
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig, load_strategy_config
from app.core.errors import DataNotReadyError, DataQualityError
from app.core.full_config import FullSystemConfig, load_full_config
from app.data.industry_map import IndustryMap
from app.data.interfaces import MarketDataProvider
from app.data.quality import DataQualityGate, resolve_quality_universe
from app.data.universe import UniverseBuilder
from app.db.repositories import EventRepository
from app.domain.enums import SentimentStage, Strategy
from app.domain.models import (
    AccountState,
    Candidate,
    MarketRegime,
    SectorStrength,
    TradePlan,
)
from app.indicators.features import FeatureBuilder
from app.planning.errors import PlanningError
from app.planning.strategy_planner import StrategyPlanner
from app.planning.strategy_sizing import StrategyPositionSizer
from app.regime.industry import IndustryStrengthService
from app.regime.leader import LeaderScoreService
from app.regime.market import MarketRegimeService
from app.regime.permissions import assess_market
from app.regime.sentiment import SentimentService, SentimentStageClassifier
from app.risk.strategy_governor import StrategyRiskGovernor
from app.strategies.base import StrategyContext
from app.strategies.registry import MultiStrategyScreener


class RejectedCandidate(BaseModel):
    candidate: Candidate
    reason: str


def _target_status(candidate: Candidate) -> Literal["known", "unknown"]:
    if candidate.strategy in (Strategy.A, Strategy.C, Strategy.F):
        resistance = candidate.features.get("high_60d_raw")
        if resistance is None or float(resistance) <= 0:
            return "unknown"
    return "known"


class StrategyPipelineResult(BaseModel):
    as_of: date
    regime: MarketRegime
    sectors: list[SectorStrength]
    candidates: list[Candidate]
    observations: list[Candidate]
    plans: list[TradePlan]
    rejected: list[RejectedCandidate]
    account: AccountState
    by_strategy: dict[str, int]


def _add_flag(flags: list[str], value: str) -> None:
    if value not in flags:
        flags.append(value)


def _index_ready(index_bars: pl.DataFrame, as_of: date, min_bars: int) -> bool:
    if index_bars.is_empty() or not {"trade_date", "close"} <= set(index_bars.columns):
        return False
    if not index_bars.schema["close"].is_numeric():
        return False
    history = index_bars.filter(pl.col("trade_date") <= as_of).sort("trade_date")
    if history.height < min_bars or history["trade_date"].n_unique() < min_bars:
        return False
    as_of_rows = history.filter(pl.col("trade_date") == as_of)
    if as_of_rows.is_empty() or as_of_rows["close"].is_null().all():
        return False
    trailing = history.tail(min_bars)["close"]
    if trailing.null_count() > 0:
        return False
    if trailing.dtype in (pl.Float32, pl.Float64):
        return bool(trailing.is_finite().all())
    return True


@dataclass
class StrategyPipeline:
    provider: MarketDataProvider
    config: FullSystemConfig = field(default_factory=load_full_config)
    legacy_config: StrategyConfig = field(default_factory=load_strategy_config)
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    regime_service: MarketRegimeService = field(default_factory=MarketRegimeService)
    industry_service: IndustryStrengthService = field(
        default_factory=IndustryStrengthService
    )
    leader_service: LeaderScoreService = field(default_factory=LeaderScoreService)
    sentiment_service: SentimentService = field(default_factory=SentimentService)
    stage_classifier: SentimentStageClassifier = field(
        default_factory=SentimentStageClassifier
    )
    screener: MultiStrategyScreener = field(default_factory=MultiStrategyScreener)
    planner: StrategyPlanner = field(default_factory=StrategyPlanner)
    sizer: StrategyPositionSizer = field(default_factory=StrategyPositionSizer)
    governor: StrategyRiskGovernor = field(default_factory=StrategyRiskGovernor)
    code_version: str = "dev"

    def run(
        self,
        as_of: date,
        account: AccountState,
        *,
        strategy_mode: str = "all",
        session: Session | None = None,
        events_by_code: dict[str, list[dict]] | None = None,
        events_available: bool | None = None,
        minute_health: bool = True,
        data_version: str | None = None,
        data_fault: bool = False,
    ) -> StrategyPipelineResult:
        stop_at = min(
            self.config.risk.drawdown_stop_at,
            account.max_drawdown
            if account.risk_config_configured and account.max_drawdown > 0
            else self.config.risk.drawdown_stop_at,
        )
        account = account.with_risk_latches(
            stop_at,
            min(self.config.risk.drawdown_half_at, stop_at * (2 / 3)),
        )
        expiry_days = max(
            self.config.strategies.a.valid_days,
            self.config.strategies.c.c2_valid_days,
            self.config.strategies.e.valid_days,
        )
        forward_days = max(45, expiry_days * 2 + 10)
        forward_end = as_of + timedelta(days=forward_days)
        history_days = max(
            self.config.backtest.min_history_days,
            self.config.universe.min_listing_days * 2 + 60,
        )
        start = as_of - timedelta(days=history_days)
        bars = self.provider.load_bars(start, as_of)
        if bars.is_empty():
            raise DataNotReadyError(f"bars missing for {as_of}")
        guidance = (
            f"trade_cal does not cover a forward window through {forward_end}; "
            "sync/backfill a forward trade_cal window"
        )
        trade_cal = self.provider.load_trade_cal(start, forward_end)
        if trade_cal.is_empty():
            raise DataNotReadyError(f"{guidance}: trade_cal is empty")
        calendar = TradingCalendar.from_trade_cal(trade_cal)
        if not calendar.is_trading_day(as_of):
            raise DataNotReadyError(f"{guidance}: {as_of} is not an open day")
        try:
            calendar.add_trading_days(as_of, expiry_days)
        except ValueError as exc:
            raise DataNotReadyError(f"{guidance}: {exc}") from exc
        stocks = self.provider.load_stock_basic(as_of)
        if stocks.is_empty():
            raise DataNotReadyError(f"stock_basic missing for {as_of}")

        degraded: list[str] = []
        as_of_bars = bars.filter(pl.col("trade_date") == as_of)
        suspend = self.provider.load_suspend(as_of, as_of)
        expected_codes, allowed_codes, expected_count, warnings = (
            resolve_quality_universe(stocks, suspend)
        )
        for warning in warnings:
            _add_flag(degraded, warning)
        if expected_codes is not None and not expected_codes:
            raise DataNotReadyError("all listed stocks are suspended")
        actual_codes = set(as_of_bars["ts_code"].drop_nulls().to_list())
        if allowed_codes is not None:
            extras = actual_codes - allowed_codes
            if extras:
                _add_flag(degraded, "stock_reference_incomplete")
                allowed_codes = allowed_codes | extras
        quality = DataQualityGate().check(
            as_of,
            as_of_bars,
            expected_count=expected_count,
            expected_codes=expected_codes,
            allowed_codes=allowed_codes,
        )
        if not quality.passed:
            raise DataQualityError("; ".join(quality.failures))
        for code in quality.warning_codes:
            _add_flag(degraded, code)

        namechange = self.provider.load_namechange()
        industry_map = IndustryMap.from_sources(
            stocks,
            self.provider.load_index_member(),
            self.provider.load_index_classify(),
            as_of,
            self.legacy_config.industry.source,
        )
        sector_frame = industry_map.to_frame()
        history_count = max(
            5, self.legacy_config.industry.persistence_days + 3
        )
        dates = sorted(
            bars.filter(pl.col("trade_date") <= as_of)["trade_date"]
            .unique()
            .to_list()
        )[-history_count:]
        features_history = pl.concat(
            [self.feature_builder.build(bars, trade_date) for trade_date in dates]
        ).join(sector_frame, on="ts_code", how="left")
        features_asof = features_history.filter(pl.col("trade_date") == as_of)
        universe_builder = UniverseBuilder(calendar)
        tradeable = universe_builder.build_tradeable(
            as_of,
            features_asof,
            stocks,
            namechange,
            self.legacy_config,
            min_listing_days=self.config.universe.min_listing_days,
        )
        breadth = (
            tradeable.filter(pl.col("above_ma20")).height / tradeable.height
            if not tradeable.is_empty()
            else 0.0
        )
        tradeable_codes = tradeable["ts_code"].to_list()
        industry_features = features_history.filter(
            pl.col("ts_code").is_in(tradeable_codes)
        )
        sectors = self.industry_service.compute(
            industry_features, as_of, self.legacy_config.industry
        )
        sentiment_history = self.sentiment_service.compute_history(bars)
        classified = self.stage_classifier.classify_series(
            sentiment_history, self.legacy_config
        )
        stage_row = classified.filter(pl.col("trade_date") == as_of)
        if stage_row.is_empty():
            raise DataNotReadyError(f"sentiment data missing for {as_of}")
        sentiment_row = stage_row.row(0, named=True)
        sentiment_quality = DataQualityGate().check_sentiment(
            {"trade_date": as_of, **sentiment_row}
        )
        if not sentiment_quality.passed:
            raise DataQualityError("; ".join(sentiment_quality.failures))
        stage = SentimentStage(sentiment_row["sentiment_stage"])
        if sentiment_row.get("initial_stage"):
            _add_flag(degraded, "initial_stage")
        if namechange.is_empty():
            _add_flag(degraded, "st_namechange_missing")
        if industry_map.degraded:
            _add_flag(degraded, "industry_stock_basic")
        if self.provider.load_limit_list(as_of, as_of).is_empty():
            _add_flag(degraded, "limit_list_d")

        index_code = self.config.market.index_code
        fallback = self.config.market.fallback_index_code
        min_index_bars = self.config.market.ma_slow
        index_bars = self.provider.load_index_bars(index_code, start, as_of)
        if not _index_ready(index_bars, as_of, min_index_bars):
            index_bars = self.provider.load_index_bars(fallback, start, as_of)
            if not _index_ready(index_bars, as_of, min_index_bars):
                raise DataNotReadyError(
                    f"index bars missing or insufficient for {as_of}"
                )
            _add_flag(degraded, "index_fallback")

        regime = self.regime_service.compute(
            as_of=as_of,
            index_bars=index_bars,
            sentiment_row=sentiment_row,
            stage=stage,
            breadth=breadth,
            sector_ranking=sectors,
            degraded_flags=degraded,
            data_version=data_version or f"dv_{as_of.isoformat()}",
            config=self.legacy_config,
        )
        regime = assess_market(
            regime,
            index_bars=index_bars,
            bars=bars,
            breadth=breadth,
            config=self.config,
            minute_health=minute_health,
            data_fault=data_fault,
        )
        leaders = self.leader_service.compute(
            tradeable, sectors, self.legacy_config
        )
        events = events_by_code
        if events is None and session is not None:
            events = self._load_events(session, as_of)
        if events_available is None:
            events_available = events_by_code is not None or session is not None
        context = StrategyContext(
            as_of=as_of,
            bars=bars,
            features=tradeable,
            sectors=sectors,
            market=regime,
            config=self.config,
            data_version=regime.data_version,
            leaders=leaders,
            events=events or {},
            events_available=events_available,
            account_mode=account.mode,
        )
        selected = self._selected_strategies(strategy_mode)
        bundle = self.screener.generate(
            context, strategies=selected, max_per_industry=2
        )
        candidates = bundle.all_candidates
        observations = [
            candidate
            for candidate in candidates
            if candidate.evidence.get("stage") != "confirmed"
        ]
        confirmed = [
            candidate
            for candidate in candidates
            if candidate.evidence.get("stage") == "confirmed"
        ]
        correlations = self._correlations_map(
            bars, [candidate.ts_code for candidate in confirmed], account, as_of
        )
        plans: list[TradePlan] = []
        rejected: list[RejectedCandidate] = []
        for candidate in confirmed:
            try:
                parts = self.planner.plan_parts(
                    candidate, calendar, self.config
                )
                position = self.sizer.size(
                    candidate,
                    parts.entry,
                    parts.stop,
                    account,
                    regime,
                    self.config,
                    correlations=correlations.get(candidate.ts_code),
                )
                plans.append(
                    TradePlan(
                        candidate=candidate,
                        entry_plan=parts.entry,
                        stop_plan=parts.stop,
                        position_plan=position,
                        take_profit_plan=parts.take_profit,
                        expires_on=parts.entry.expires_on,
                        exit_rules=parts.exit_rules,
                        strategy_version=self.config.version,
                        data_version=regime.data_version,
                        code_version=self.code_version,
                        reasons=candidate.reasons,
                        strategy=candidate.strategy,
                        sub_mode=candidate.sub_mode,
                        signal_time=candidate.data_time,
                        data_time=candidate.data_time,
                        confirmation_price=parts.confirmation_price,
                        max_acceptable_price=parts.max_acceptable_price,
                        execution_low=parts.execution_low,
                        execution_high=parts.execution_high,
                        target_status=_target_status(candidate),
                    )
                )
            except PlanningError as exc:
                rejected.append(
                    RejectedCandidate(candidate=candidate, reason=str(exc))
                )
        plans = self.governor.apply(
            plans,
            account,
            regime,
            self.config,
            correlations=correlations,
        )
        by_strategy = {
            strategy.value: sum(
                1
                for candidate in candidates
                if candidate.strategy == strategy
            )
            for strategy in selected
        }
        return StrategyPipelineResult(
            as_of=as_of,
            regime=regime,
            sectors=sectors,
            candidates=candidates,
            observations=observations,
            plans=plans,
            rejected=rejected,
            account=account,
            by_strategy=by_strategy,
        )

    def _selected_strategies(self, mode: str) -> list[Strategy]:
        if mode == "all":
            return list(Strategy)
        if mode == "dual":
            return [Strategy.A, Strategy.B]
        try:
            return [Strategy(mode.upper())]
        except ValueError as exc:
            raise ValueError(f"unknown strategy mode: {mode}") from exc

    def _load_events(self, session: Session, as_of: date) -> dict[str, list[dict]]:
        rows = EventRepository.list_recent(
            session,
            start=as_of - timedelta(days=30),
            end=as_of,
            verified_only=True,
            limit=1000,
        )
        result: dict[str, list[dict]] = {}
        for row in rows:
            payload = {
                "event_id": row.id,
                "event_type": row.event_type,
                "title": row.title,
                "source": row.source,
                "ann_date": row.ann_date,
                "published_at": row.published_at,
                "first_seen_at": row.first_seen_at,
                "verification_status": row.verification_status,
                "direction": row.direction,
                "materiality": row.materiality,
                "retracted": row.retracted,
                "stage": row.stage,
                "certainty": row.certainty,
                "dedupe_key": row.dedupe_key,
            }
            for link in row.links:
                result.setdefault(link.ts_code, []).append(
                    {**payload, "ambiguous": link.ambiguous}
                )
        return result

    def _correlations_map(
        self,
        bars: pl.DataFrame,
        candidate_codes: list[str],
        account: AccountState,
        as_of: date,
    ) -> dict[str, dict[str, float]]:
        existing = [item.ts_code for item in account.positions] + [
            item.ts_code for item in account.pending_plans
        ]
        partners = list(dict.fromkeys(existing + candidate_codes))
        if len(partners) < 2:
            return {}
        frame = (
            bars.filter(pl.col("trade_date") <= as_of)
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") * pl.col("adj_factor")).alias("_adj_close")
            )
            .with_columns(
                (
                    pl.col("_adj_close")
                    / pl.col("_adj_close").shift(1).over("ts_code")
                    - 1
                ).alias("ret"),
                pl.col("trade_date").shift(1).over("ts_code").alias("_prev_date"),
            )
        )
        market_dates = (
            frame.select(pl.col("trade_date").unique().sort())
            .with_columns(pl.col("trade_date").shift(1).alias("_market_prev"))
        )
        frame = frame.join(market_dates, on="trade_date", how="left").with_columns(
            pl.when(pl.col("_prev_date") == pl.col("_market_prev"))
            .then(pl.col("ret"))
            .otherwise(None)
            .alias("ret")
        )
        result: dict[str, dict[str, float]] = {}
        for candidate_code in candidate_codes:
            candidate = (
                frame.filter(pl.col("ts_code") == candidate_code)
                .select(["trade_date", "ret"])
                .tail(60)
                .rename({"ret": "candidate_ret"})
            )
            inner: dict[str, float] = {}
            for other in partners:
                if other == candidate_code:
                    continue
                other_frame = (
                    frame.filter(pl.col("ts_code") == other)
                    .select(["trade_date", "ret"])
                    .tail(60)
                    .rename({"ret": "other_ret"})
                )
                joined = candidate.join(
                    other_frame, on="trade_date", how="inner"
                ).drop_nulls()
                if joined.height < 20:
                    continue
                value = joined.select(
                    pl.corr("candidate_ret", "other_ret")
                ).item()
                if value is not None and value == value:
                    inner[other] = float(value)
            if inner:
                result[candidate_code] = inner
        return result
