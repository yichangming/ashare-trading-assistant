from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import polars as pl
from pydantic import BaseModel

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig
from app.core.errors import DataNotReadyError, DataQualityError
from app.data.industry_map import IndustryMap
from app.data.interfaces import MarketDataProvider
from app.data.quality import DataQualityGate
from app.data.universe import UniverseBuilder
from app.domain.enums import SentimentStage, Template
from app.domain.models import (
    AccountState,
    Candidate,
    MarketRegime,
    SectorStrength,
    TradePlan,
)
from app.indicators.features import FeatureBuilder
from app.planning.entry import EntryPlanner
from app.planning.errors import PlanningError
from app.planning.sizing import PositionSizer
from app.planning.stop import StopPlanner
from app.planning.take_profit import TakeProfitPlanner
from app.regime.industry import IndustryStrengthService
from app.regime.leader import LeaderScoreService
from app.regime.market import MarketRegimeService
from app.regime.sentiment import SentimentService, SentimentStageClassifier
from app.risk.governor import RiskGovernor
from app.screener.base import ScreenerStrategy
from app.screener.dual_template import DualTemplateScreener

_HISTORY_DAYS = 400
_FORWARD_CALENDAR_DAYS = 45
_FEATURE_HISTORY_DATES = 5
_CORRELATION_LOOKBACK = 60
_CORRELATION_MIN_OBSERVATIONS = 20


def _add_flag(flags: list[str], flag: str) -> None:
    """Append ``flag`` once, preserving first-seen order."""
    if flag not in flags:
        flags.append(flag)


class RejectedCandidate(BaseModel):
    candidate: Candidate
    reason: str


class PipelineResult(BaseModel):
    as_of: date
    regime: MarketRegime
    sectors: list[SectorStrength]
    candidates: list[Candidate]
    plans: list[TradePlan]
    rejected: list[RejectedCandidate]
    account: AccountState


@dataclass
class TradingPipeline:
    """Orchestrate one end-to-end daily planning run.

    The real path loads a 400-calendar-day bar window ending at ``as_of`` plus
    a forward ``trade_cal`` window.  The forward window is validated before
    planning because ``TradingCalendar.add_trading_days`` raises once the
    calendar is exhausted; a store synced only through ``as_of`` must be
    backfilled with future open dates first.

    As-of bars pass through :class:`~app.data.quality.DataQualityGate` before
    any signals are produced.  Quality warning codes are surfaced as regime
    degraded flags alongside missing-namechange, stock-basic-industry,
    empty-limit-list, index-fallback, and initial-sentiment-stage markers.

    Breadth, industry strength, and leader percentiles use the broad tradable
    universe (listed, non-ST, non-suspended) rather than the narrower
    Template-A investment universe; the screener still joins leaders onto the
    Template-A universe.  Each candidate receives a 60-day close-return
    correlation map against existing positions, pending plans, and the other
    candidates, which feeds the position sizer and the final risk governor.

    ``account`` is latched with the configured drawdown thresholds at the
    start of the run; the run never mutates the caller's account and returns
    the latched copy in :class:`PipelineResult` for the caller to persist.
    """

    provider: MarketDataProvider
    config: StrategyConfig
    screener: ScreenerStrategy = field(default_factory=DualTemplateScreener)
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    regime_service: MarketRegimeService = field(
        default_factory=MarketRegimeService
    )
    industry_service: IndustryStrengthService = field(
        default_factory=IndustryStrengthService
    )
    leader_service: LeaderScoreService = field(
        default_factory=LeaderScoreService
    )
    sentiment_service: SentimentService = field(
        default_factory=SentimentService
    )
    stage_classifier: SentimentStageClassifier = field(
        default_factory=SentimentStageClassifier
    )
    entry_planner: EntryPlanner = field(default_factory=EntryPlanner)
    stop_planner: StopPlanner = field(default_factory=StopPlanner)
    sizer: PositionSizer = field(default_factory=PositionSizer)
    take_profit_planner: TakeProfitPlanner = field(
        default_factory=TakeProfitPlanner
    )
    risk_governor: RiskGovernor = field(default_factory=RiskGovernor)
    code_version: str = "dev"

    def run(
        self,
        as_of: date,
        account: AccountState,
        template_mode: str = "dual",
        regime_override: MarketRegime | None = None,
        candidates_override: list[Candidate] | None = None,
        calendar_override: TradingCalendar | None = None,
        data_version: str | None = None,
    ) -> PipelineResult:
        account = account.with_risk_latches(
            self.config.risk.drawdown_stop_at,
            self.config.risk.drawdown_half_at,
        )
        if candidates_override is not None and regime_override is not None:
            calendar = calendar_override or TradingCalendar(
                [as_of + timedelta(days=i) for i in range(15)]
            )
            candidates = self._filter_mode(candidates_override, template_mode)
            plans, rejected = self._build_plans(
                candidates, account, regime_override, calendar, as_of
            )
            return PipelineResult(
                as_of=as_of,
                regime=regime_override,
                sectors=regime_override.sector_ranking,
                candidates=candidates,
                plans=plans,
                rejected=rejected,
                account=account,
            )

        start = as_of - timedelta(days=_HISTORY_DAYS)
        bars = self.provider.load_bars(start, as_of)
        if bars.is_empty():
            raise DataNotReadyError(f"bars missing for {as_of}")
        forward_end = as_of + timedelta(days=_FORWARD_CALENDAR_DAYS)
        trade_cal = self.provider.load_trade_cal(start, forward_end)
        calendar = TradingCalendar.from_trade_cal(trade_cal)
        if not calendar.is_trading_day(as_of):
            raise DataNotReadyError(f"{as_of} is not a trading day in trade_cal")
        expiry_days = max(
            self.config.template_a.entry_valid_days,
            self.config.template_b.entry_valid_days,
        )
        try:
            calendar.add_trading_days(as_of, expiry_days)
        except ValueError as exc:
            raise DataNotReadyError(
                "trade_cal does not cover the entry expiry window through "
                f"{forward_end}; sync/backfill a forward trade_cal window "
                f"covering {as_of} + {expiry_days} trading days: {exc}"
            ) from exc

        stocks = self.provider.load_stock_basic(as_of)
        if stocks.is_empty():
            raise DataNotReadyError(f"stock_basic missing for {as_of}")

        degraded: list[str] = []
        as_of_bars = bars.filter(pl.col("trade_date") == as_of)
        # ``stocks`` is the listed/delisted reference as of ``as_of``: the
        # provider already drops names delisted by then, but a stock suspended
        # for the day legitimately has no bar and still counts in the coverage
        # denominator.  A large suspension cohort can therefore fail coverage;
        # supply an explicit suspension exclusion if that becomes material.
        # As-of bar codes are added so a code absent from the reference (for
        # example a newly listed name) is not reported as unexpected; names
        # missing from the bars still fail ``missing_expected_codes``.
        expected_codes = set(stocks["ts_code"].to_list()) | set(
            as_of_bars["ts_code"].to_list()
        )
        quality = DataQualityGate().check(
            as_of,
            as_of_bars,
            expected_count=len(expected_codes),
            expected_codes=expected_codes,
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
            self.config.industry.source,
        )
        sector_frame = industry_map.to_frame()

        dates = sorted(
            bars.filter(pl.col("trade_date") <= as_of)["trade_date"]
            .unique()
            .to_list()
        )[-_FEATURE_HISTORY_DATES:]
        features_history = pl.concat(
            [
                self.feature_builder.build(bars, trade_date)
                for trade_date in dates
            ]
        ).join(sector_frame, on="ts_code", how="left")
        features_asof = features_history.filter(
            pl.col("trade_date") == as_of
        )
        universe_builder = UniverseBuilder(calendar)
        tradeable = universe_builder.build_tradeable(
            as_of, features_asof, stocks, namechange, self.config
        )
        universe = universe_builder.build(
            as_of, features_asof, stocks, namechange, self.config
        )
        breadth = (
            tradeable.filter(pl.col("above_ma20")).height / tradeable.height
            if not tradeable.is_empty()
            else 0.0
        )
        # Filter the multi-day feature history to as-of tradeable codes so
        # new/ST/suspended names do not contaminate industry aggregates.
        # Prior days inherit today's membership, a slight as-of approximation
        # that is acceptable for a five-day persistence window.
        tradeable_codes = tradeable["ts_code"].to_list()
        industry_features = features_history.filter(
            pl.col("ts_code").is_in(tradeable_codes)
        )
        sectors = self.industry_service.compute(
            industry_features, as_of, self.config.industry
        )

        sentiment_history = self.sentiment_service.compute_history(bars)
        classified = self.stage_classifier.classify_series(
            sentiment_history, self.config
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

        index_code = self.config.environment.index_code
        index_bars = self.provider.load_index_bars(index_code, start, as_of)
        if index_bars.is_empty():
            index_code = self.config.environment.fallback_index_code
            index_bars = self.provider.load_index_bars(
                index_code, start, as_of
            )
            _add_flag(degraded, "index_fallback")
        if index_bars.is_empty():
            raise DataNotReadyError(
                f"index bars missing for {as_of} (primary and fallback)"
            )

        regime = self.regime_service.compute(
            as_of=as_of,
            index_bars=index_bars,
            sentiment_row=sentiment_row,
            stage=stage,
            breadth=breadth,
            sector_ranking=sectors,
            degraded_flags=degraded,
            data_version=data_version or f"dv_{as_of.isoformat()}",
            config=self.config,
        )
        # Leader percentiles are computed over every tradeable name; the
        # screener later joins them onto the narrower Template-A universe.
        leaders = self.leader_service.compute(
            tradeable, sectors, self.config
        )
        candidates = self.screener.generate(
            as_of, regime, universe, sectors, leaders, self.config
        )
        candidates = self._filter_mode(candidates, template_mode)
        correlations = self._correlations_map(
            bars, [candidate.ts_code for candidate in candidates], account, as_of
        )
        plans, rejected = self._build_plans(
            candidates,
            account,
            regime,
            calendar,
            as_of,
            correlations=correlations,
        )
        return PipelineResult(
            as_of=as_of,
            regime=regime,
            sectors=sectors,
            candidates=candidates,
            plans=plans,
            rejected=rejected,
            account=account,
        )

    def _filter_mode(
        self, candidates: list[Candidate], template_mode: str
    ) -> list[Candidate]:
        if template_mode == "A":
            return [c for c in candidates if c.template == Template.A]
        if template_mode == "B":
            return [c for c in candidates if c.template == Template.B]
        if template_mode == "dual":
            return list(candidates)
        raise ValueError(f"unknown template_mode: {template_mode}")

    def _correlations_map(
        self,
        bars: pl.DataFrame,
        candidate_codes: list[str],
        account: AccountState,
        as_of: date,
    ) -> dict[str, dict[str, float]]:
        """Close-return correlations for each candidate vs every partner.

        Returns ``{candidate_code: {other_code: correlation}}`` where partners
        are existing positions, pending plans, and the other candidates.
        Each series uses up to 60 trading days of close returns ending at
        ``as_of``; pairs with fewer than 20 overlapping observations, or a
        non-finite coefficient, are omitted.  A candidate with no computable
        partner correlation is omitted entirely so the governor treats it as
        unavailable.
        """
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
                (
                    pl.col("close")
                    / pl.col("close").shift(1).over("ts_code")
                    - 1
                ).alias("ret")
            )
        )
        result: dict[str, dict[str, float]] = {}
        for candidate_code in candidate_codes:
            candidate = (
                frame.filter(pl.col("ts_code") == candidate_code)
                .select(["trade_date", "ret"])
                .tail(_CORRELATION_LOOKBACK)
                .rename({"ret": "candidate_ret"})
            )
            inner: dict[str, float] = {}
            for other in partners:
                if other == candidate_code:
                    continue
                other_frame = (
                    frame.filter(pl.col("ts_code") == other)
                    .select(["trade_date", "ret"])
                    .tail(_CORRELATION_LOOKBACK)
                    .rename({"ret": "other_ret"})
                )
                joined = candidate.join(
                    other_frame, on="trade_date", how="inner"
                ).drop_nulls()
                if joined.height < _CORRELATION_MIN_OBSERVATIONS:
                    continue
                value = joined.select(
                    pl.corr("candidate_ret", "other_ret")
                ).item()
                if value is None:
                    continue
                correlation = float(value)
                if not math.isfinite(correlation):
                    continue
                inner[other] = correlation
            if inner:
                result[candidate_code] = inner
        return result

    def _build_plans(
        self,
        candidates: list[Candidate],
        account: AccountState,
        regime: MarketRegime,
        calendar: TradingCalendar,
        as_of: date,
        correlations: dict[str, dict[str, float]] | None = None,
    ) -> tuple[list[TradePlan], list[RejectedCandidate]]:
        plans: list[TradePlan] = []
        rejected: list[RejectedCandidate] = []
        for candidate in candidates:
            try:
                entry = self.entry_planner.plan(
                    candidate,
                    candidate.features,
                    regime,
                    calendar,
                    as_of,
                    self.config,
                )
                stop = self.stop_planner.plan(
                    candidate, entry, candidate.features, self.config
                )
                candidate_correlations = None
                if correlations is not None:
                    candidate_correlations = correlations.get(
                        candidate.ts_code
                    )
                position = self.sizer.size(
                    candidate,
                    entry,
                    stop,
                    account,
                    regime,
                    self.config,
                    correlations=candidate_correlations,
                )
                take_profit = self.take_profit_planner.plan(
                    candidate, entry, stop, self.config
                )
                plans.append(
                    TradePlan(
                        candidate=candidate,
                        entry_plan=entry,
                        stop_plan=stop,
                        position_plan=position,
                        take_profit_plan=take_profit,
                        expires_on=entry.expires_on,
                        strategy_version=self.config.version,
                        data_version=regime.data_version,
                        code_version=self.code_version,
                    )
                )
            except PlanningError as exc:
                rejected.append(
                    RejectedCandidate(candidate=candidate, reason=str(exc))
                )
        plans = self.risk_governor.apply(
            plans,
            account,
            regime,
            self.config,
            correlations=correlations,
        )
        return plans, rejected
