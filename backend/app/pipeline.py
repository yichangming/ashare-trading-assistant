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
    """Outcome of one pipeline run.

    ``account`` is the latched input snapshot; it does **not** include the
    same-day approved pending plans produced by this run.  Callers that
    persist state should merge ``plans`` (status ``PENDING_ENTRY``) into
    ``account.pending_plans`` themselves.
    """

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

        expiry_days = max(
            self.config.template_a.entry_valid_days,
            self.config.template_b.entry_valid_days,
        )
        forward_days = max(_FORWARD_CALENDAR_DAYS, expiry_days * 2 + 10)
        forward_end = as_of + timedelta(days=forward_days)
        history_days = max(
            _HISTORY_DAYS,
            self.config.template_a.min_listing_days * 2 + 60,
        )
        start = as_of - timedelta(days=history_days)
        bars = self.provider.load_bars(start, as_of)
        if bars.is_empty():
            raise DataNotReadyError(f"bars missing for {as_of}")
        calendar_guidance = (
            f"trade_cal does not cover a forward window through {forward_end}; "
            "sync/backfill a forward trade_cal window "
            f"covering {as_of} + {expiry_days} trading days"
        )
        trade_cal = self.provider.load_trade_cal(start, forward_end)
        if trade_cal.is_empty():
            raise DataNotReadyError(
                f"{calendar_guidance}: trade_cal is empty"
            )
        calendar = TradingCalendar.from_trade_cal(trade_cal)
        if not calendar.is_trading_day(as_of):
            raise DataNotReadyError(
                f"{calendar_guidance}: {as_of} is not an open trading day"
            )
        min_listing_days = self.config.template_a.min_listing_days
        listing_coverage = (
            calendar.count_between(calendar.dates[0], as_of)
            if calendar.dates
            else 0
        )
        if listing_coverage < min_listing_days:
            raise DataNotReadyError(
                f"trade_cal covers only {listing_coverage} trading days "
                f"before {as_of}; need at least {min_listing_days} for the "
                "listing filter; sync/backfill a trade_cal window covering "
                "the required history"
            )
        try:
            calendar.add_trading_days(as_of, expiry_days)
        except ValueError as exc:
            raise DataNotReadyError(f"{calendar_guidance}: {exc}") from exc

        stocks = self.provider.load_stock_basic(as_of)
        if stocks.is_empty():
            raise DataNotReadyError(f"stock_basic missing for {as_of}")

        degraded: list[str] = []
        as_of_bars = bars.filter(pl.col("trade_date") == as_of)
        # A suspended stock is legitimately absent from Tushare ``daily``.
        # ``load_suspend`` returns ``None`` only when no raw ``suspend_d``
        # partition exists at all; that means the optional endpoint has not
        # been synced, so fall back to count-based coverage against all listed
        # names where the 99% threshold tolerates a small suspended cohort.
        # A typed frame instead means a partition exists: full-day suspensions
        # are excluded from the required set, while every listed code remains
        # allowed so a legitimate bar for a suspended name is not "unexpected".
        suspend = self.provider.load_suspend(as_of, as_of)
        all_stock_codes = set(stocks["ts_code"].to_list())
        if suspend is None:
            _add_flag(degraded, "suspend_d_missing")
            expected_codes: set[str] | None = None
            allowed_codes: set[str] | None = all_stock_codes
            expected_count = len(stocks)
        else:
            if "ts_code" not in suspend.columns:
                # A malformed partition cannot identify suspensions; keep the
                # exact universe at all listed names.
                full_day_suspended: set[str] = set()
            else:
                suspended = suspend.drop_nulls(subset=["ts_code"])
                if "suspend_timing" in suspended.columns:
                    timing = (
                        pl.col("suspend_timing")
                        .cast(pl.String, strict=False)
                        .str.strip_chars()
                    )
                    full_day = suspended.filter(
                        pl.col("suspend_timing").is_null() | (timing == "")
                    )
                else:
                    full_day = suspended
                full_day_suspended = set(full_day["ts_code"].to_list())
            required_codes = all_stock_codes - full_day_suspended
            if not required_codes:
                raise DataNotReadyError("all listed stocks are suspended")
            expected_codes = required_codes
            allowed_codes = all_stock_codes
            expected_count = len(required_codes)
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
            self.config.industry.source,
        )
        sector_frame = industry_map.to_frame()

        history_count = max(
            _FEATURE_HISTORY_DATES,
            self.config.industry.persistence_days + 3,
        )
        dates = sorted(
            bars.filter(pl.col("trade_date") <= as_of)["trade_date"]
            .unique()
            .to_list()
        )[-history_count:]
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

        primary_index = self.config.environment.index_code
        fallback_index = self.config.environment.fallback_index_code
        min_index_bars = self.config.environment.ma_slow
        index_bars = self.provider.load_index_bars(
            primary_index, start, as_of
        )
        if not self._index_ready(index_bars, as_of, min_index_bars):
            index_bars = self.provider.load_index_bars(
                fallback_index, start, as_of
            )
            if not self._index_ready(index_bars, as_of, min_index_bars):
                raise DataNotReadyError(
                    f"index bars missing or insufficient for {as_of}: "
                    f"{primary_index} and {fallback_index} each need at "
                    f"least {min_index_bars} bars including an as-of row; "
                    "sync/backfill index_daily"
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
        correlations, correlation_unavailable = self._correlations_map(
            bars, [candidate.ts_code for candidate in candidates], account, as_of
        )
        plans, rejected = self._build_plans(
            candidates,
            account,
            regime,
            calendar,
            as_of,
            correlations=correlations,
            correlation_unavailable=correlation_unavailable,
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

    @staticmethod
    def _index_ready(
        index_bars: pl.DataFrame, as_of: date, min_bars: int
    ) -> bool:
        """Return whether ``index_bars`` can produce an as-of regime read."""
        if index_bars.is_empty():
            return False
        if not {"trade_date", "close"} <= set(index_bars.columns):
            return False
        history = index_bars.filter(pl.col("trade_date") <= as_of)
        as_of_rows = history.filter(pl.col("trade_date") == as_of)
        if as_of_rows.is_empty() or as_of_rows["close"].is_null().all():
            return False
        return (
            history["trade_date"].n_unique() >= min_bars
            and history["close"].is_not_null().sum() >= min_bars
        )

    @staticmethod
    def _correlation_returns(
        bars: pl.DataFrame, as_of: date
    ) -> pl.DataFrame:
        """Return gap-aware adjusted close returns per code up to ``as_of``.

        Using ``close * adj_factor`` keeps splits and dividends from creating
        spurious moves.  A return is nulled when the code's previous bar is
        not the previous market date (suspension or listing gap), so a
        resumed/split jump is not treated as a one-day return.
        """
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
                pl.col("trade_date")
                .shift(1)
                .over("ts_code")
                .alias("_prev_date"),
            )
        )
        market_dates = (
            frame.select(pl.col("trade_date").unique().sort())
            .with_columns(
                pl.col("trade_date").shift(1).alias("_market_prev")
            )
        )
        frame = frame.join(market_dates, on="trade_date", how="left")
        return frame.with_columns(
            pl.when(pl.col("_prev_date") == pl.col("_market_prev"))
            .then(pl.col("ret"))
            .otherwise(None)
            .alias("ret")
        ).select(["ts_code", "trade_date", "ret"])

    def _correlations_map(
        self,
        bars: pl.DataFrame,
        candidate_codes: list[str],
        account: AccountState,
        as_of: date,
    ) -> tuple[dict[str, dict[str, float]], set[str]]:
        """Adjusted, gap-aware correlations for each candidate vs partners.

        Returns ``({candidate_code: {other_code: correlation}}, unavailable)``
        where partners are existing positions, pending plans, and the other
        candidates.  Each series uses up to 60 trading days of adjusted close
        returns ending at ``as_of``; pairs with fewer than 20 overlapping
        observations, or a non-finite coefficient, are omitted.  A candidate
        with partners but no computable correlation is omitted from the map
        and listed in the unavailable set so callers can label its decisions.
        """
        existing = [item.ts_code for item in account.positions] + [
            item.ts_code for item in account.pending_plans
        ]
        partners = list(dict.fromkeys(existing + candidate_codes))
        result: dict[str, dict[str, float]] = {}
        unavailable: set[str] = set()
        if len(partners) < 2:
            return result, unavailable
        frame = self._correlation_returns(bars, as_of)
        for candidate_code in candidate_codes:
            has_partner = any(
                other != candidate_code for other in partners
            )
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
            elif has_partner:
                unavailable.add(candidate_code)
        return result, unavailable

    def _build_plans(
        self,
        candidates: list[Candidate],
        account: AccountState,
        regime: MarketRegime,
        calendar: TradingCalendar,
        as_of: date,
        correlations: dict[str, dict[str, float]] | None = None,
        correlation_unavailable: set[str] | None = None,
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
            correlation_unavailable=correlation_unavailable,
        )
        return plans, rejected
