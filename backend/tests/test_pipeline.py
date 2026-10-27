from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import polars as pl
import pytest

from app.core.calendar import TradingCalendar
from app.core.config import load_strategy_config
from app.core.errors import DataNotReadyError, DataQualityError
from app.data.universe import UniverseBuilder
from app.domain.enums import (
    PlanStatus,
    RiskOutcome,
    SentimentStage,
    Template,
    TrendRegime,
)
from app.domain.models import (
    AccountState,
    Candidate,
    MarketRegime,
    PendingExposure,
    PositionState,
    RegimeIndicators,
    TemplatePolicy,
)
from app.indicators.features import FeatureBuilder
from app.pipeline import TradingPipeline


class DummyProvider:
    pass


def _calendar() -> TradingCalendar:
    return TradingCalendar(
        [date(2024, 1, 2) + timedelta(days=i) for i in range(10)]
    )


def _regime() -> MarketRegime:
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime=TrendRegime.ATTACK,
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={
            Template.A: policy,
            Template.B: TemplatePolicy(enabled=True, mode="normal", max_positions=2),
        },
        indicators=RegimeIndicators(
            limit_up_count=80,
            limit_down_count=0,
            max_board_height=5,
            promotion_rate=0.5,
            blow_rate=0.1,
            red_rate=0.6,
            limit_up_premium=0.02,
            breadth=0.6,
            index_close=4000.0,
            index_ma20=3900.0,
            index_ma60=3800.0,
        ),
        data_version="dv_test",
    )


def _candidate(template: Template = Template.A) -> Candidate:
    return Candidate(
        ts_code="000001.SZ",
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={
            "ma20_raw": 10.0,
            "ma5_raw": 10.5,
            "ma10_raw": 10.0,
            "board_height": 0.0,
            "atr14_raw": 0.5,
            "low_10d_raw": 9.5,
            "low_5d_raw": 9.7,
            "high_60d_raw": 30.0,
            "close": 10.2,
        },
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


def _account(
    equity: str = "1000000",
    cash: str = "1000000",
    peak_equity: str = "1000000",
    positions: list[PositionState] | None = None,
    pending_plans: list[PendingExposure] | None = None,
) -> AccountState:
    return AccountState(
        equity=Decimal(equity),
        cash=Decimal(cash),
        peak_equity=Decimal(peak_equity),
        positions=positions or [],
        pending_plans=pending_plans or [],
    )


def test_pipeline_with_overrides_produces_plan():
    config = load_strategy_config()
    account = AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal("1000000"),
    )
    pipeline = TradingPipeline(provider=DummyProvider(), config=config)
    result = pipeline.run(
        as_of=date(2024, 1, 2),
        account=account,
        template_mode="dual",
        regime_override=_regime(),
        candidates_override=[_candidate()],
        calendar_override=_calendar(),
    )
    assert result.regime.sentiment_stage == SentimentStage.FERMENT
    assert len(result.plans) == 1
    assert result.plans[0].status.value == "PENDING_ENTRY"
    assert result.plans[0].risk_decision.outcome.value == "approve"


def test_pipeline_template_filter():
    config = load_strategy_config()
    account = AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal("1000000"),
    )
    pipeline = TradingPipeline(provider=DummyProvider(), config=config)
    result = pipeline.run(
        as_of=date(2024, 1, 2),
        account=account,
        template_mode="A",
        regime_override=_regime(),
        candidates_override=[
            _candidate(Template.A),
            _candidate(Template.B),
        ],
        calendar_override=_calendar(),
    )
    assert [plan.candidate.template for plan in result.plans] == [Template.A]


def test_pipeline_halted_account_has_no_approved_plans():
    config = load_strategy_config()
    account = _account(equity="800000", cash="800000")
    pipeline = TradingPipeline(provider=DummyProvider(), config=config)

    result = pipeline.run(
        as_of=date(2024, 1, 2),
        account=account,
        template_mode="dual",
        regime_override=_regime(),
        candidates_override=[_candidate()],
        calendar_override=_calendar(),
    )

    assert result.plans == []
    assert len(result.rejected) == 1
    assert "halted" in result.rejected[0].reason
    assert result.account.trading_halted is True
    assert result.account.drawdown_half_latched is True
    # The caller's account is not mutated; the latched copy is returned.
    assert account.trading_halted is False
    assert account.drawdown_half_latched is False


def test_correlations_include_positions_pending_and_skip_short_history():
    as_of = date(2024, 2, 1)
    dates = _trading_days_ending(as_of, 25)
    closes = [
        10.0 * (1 + 0.01 * ((index % 5) - 2))
        for index in range(len(dates))
    ]
    rows: list[dict] = []
    for code in ("000001.SZ", "000002.SZ", "000003.SZ"):
        for trade_date, close in zip(dates, closes, strict=True):
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "close": close,
                    "adj_factor": 1.0,
                }
            )
    # A short listing history cannot supply the 20 overlapping observations.
    for trade_date, close in zip(dates[-10:], closes[-10:], strict=True):
        rows.append(
            {
                "ts_code": "000004.SZ",
                "trade_date": trade_date,
                "close": close,
                "adj_factor": 1.0,
            }
        )
    bars = pl.DataFrame(rows)
    account = _account(
        positions=[
            PositionState(
                ts_code="000002.SZ",
                sector="银行",
                shares=100,
                available_shares=100,
                cost_price=Decimal("10"),
                current_stop=Decimal("9"),
                current_price=Decimal("10"),
                template=Template.A,
            )
        ],
        pending_plans=[
            PendingExposure(
                ts_code="000003.SZ",
                sector="银行",
                entry_price=Decimal("10"),
                stop_price=Decimal("9"),
                shares=100,
                template=Template.A,
            )
        ],
    )
    pipeline = TradingPipeline(
        provider=DummyProvider(), config=load_strategy_config()
    )

    correlations, unavailable = pipeline._correlations_map(
        bars, ["000001.SZ", "000004.SZ"], account, as_of
    )

    inner = correlations["000001.SZ"]
    assert inner["000002.SZ"] == pytest.approx(1.0)
    assert inner["000003.SZ"] == pytest.approx(1.0)
    assert "000004.SZ" not in inner
    # Candidates with partners but no computable pair are reported separately.
    assert "000004.SZ" not in correlations
    assert unavailable == {"000004.SZ"}


def test_correlations_omit_non_finite_pairs():
    as_of = date(2024, 2, 1)
    dates = _trading_days_ending(as_of, 25)
    varying = [
        10.0 * (1 + 0.01 * ((index % 5) - 2))
        for index in range(len(dates))
    ]
    rows: list[dict] = []
    for code, closes in (
        ("000001.SZ", varying),
        ("000002.SZ", varying),
        # A constant close produces zero-variance returns and NaN correlation.
        ("000003.SZ", [10.0] * len(dates)),
    ):
        for trade_date, close in zip(dates, closes, strict=True):
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "close": close,
                    "adj_factor": 1.0,
                }
            )
    account = _account(
        pending_plans=[
            PendingExposure(
                ts_code="000002.SZ",
                sector="银行",
                entry_price=Decimal("10"),
                stop_price=Decimal("9"),
                shares=100,
                template=Template.A,
            ),
            PendingExposure(
                ts_code="000003.SZ",
                sector="银行",
                entry_price=Decimal("10"),
                stop_price=Decimal("9"),
                shares=100,
                template=Template.A,
            ),
        ],
    )
    pipeline = TradingPipeline(
        provider=DummyProvider(), config=load_strategy_config()
    )

    correlations, unavailable = pipeline._correlations_map(
        pl.DataFrame(rows), ["000001.SZ"], account, as_of
    )

    inner = correlations["000001.SZ"]
    assert inner["000002.SZ"] == pytest.approx(1.0)
    assert "000003.SZ" not in inner
    assert unavailable == set()


# ---------------------------------------------------------------------------
# Tradable universe / breadth
# ---------------------------------------------------------------------------


def _universe_calendar() -> TradingCalendar:
    return TradingCalendar(
        [date(2023, 1, 1) + timedelta(days=i) for i in range(400)]
    )


def _universe_features(
    codes: list[str],
    *,
    as_of: date,
    close: list[float] | None = None,
    amount: list[float] | None = None,
    suspended: list[bool] | None = None,
    above_ma20: list[bool] | None = None,
) -> pl.DataFrame:
    size = len(codes)
    return pl.DataFrame(
        {
            "ts_code": codes,
            "trade_date": [as_of] * size,
            "close": close if close is not None else [10.0] * size,
            "amount_20d_avg": amount if amount is not None else [2e8] * size,
            "is_suspended": (
                suspended if suspended is not None else [False] * size
            ),
            "above_ma20": (
                above_ma20 if above_ma20 is not None else [True] * size
            ),
        }
    )


def _universe_stocks(
    codes: list[str],
    *,
    markets: list[str] | None = None,
    names: list[str] | None = None,
    list_dates: list[date] | None = None,
) -> pl.DataFrame:
    size = len(codes)
    return pl.DataFrame(
        {
            "ts_code": codes,
            "name": names if names is not None else ["正常公司"] * size,
            "market": markets if markets is not None else ["主板"] * size,
            "industry": ["银行"] * size,
            "list_date": (
                list_dates
                if list_dates is not None
                else [date(2000, 1, 1)] * size
            ),
            "delist_date": [None] * size,
        },
        schema_overrides={"delist_date": pl.Date},
    )


def test_build_tradeable_ignores_board_liquidity_and_price_filters():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    codes = ["600030.SH", "688001.SH", "600032.SH", "600033.SH"]
    features = _universe_features(
        codes,
        as_of=as_of,
        close=[10.0, 10.0, 10.0, 2.99],
        amount=[2e8, 2e8, 9.9e7, 2e8],
    )
    stocks = _universe_stocks(
        codes,
        markets=["主板", "科创板", "主板", "主板"],
    )
    builder = UniverseBuilder(_universe_calendar())

    tradeable = builder.build_tradeable(
        as_of, features, stocks, pl.DataFrame(), config
    )
    # 科创板, low-amount, and low-price names are all still tradable.
    assert tradeable["ts_code"].to_list() == codes

    universe = builder.build(as_of, features, stocks, pl.DataFrame(), config)
    assert universe["ts_code"].to_list() == ["600030.SH"]


def test_build_tradeable_applies_listing_st_and_suspension_gates():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    codes = ["600020.SH", "600021.SH", "600022.SH", "600023.SH"]
    features = _universe_features(
        codes,
        as_of=as_of,
        suspended=[False, True, False, False],
    )
    stocks = _universe_stocks(
        codes,
        names=["正常公司", "正常公司", "ST某某", "正常公司"],
        list_dates=[
            date(2000, 1, 1),
            date(2000, 1, 1),
            date(2000, 1, 1),
            as_of - timedelta(days=5),
        ],
    )
    tradeable = UniverseBuilder(_universe_calendar()).build_tradeable(
        as_of, features, stocks, pl.DataFrame(), config
    )
    assert tradeable["ts_code"].to_list() == ["600020.SH"]


def test_breadth_counts_star_board_stock_excluded_from_template_a():
    config = load_strategy_config()
    as_of = date(2024, 1, 2)
    codes = ["600030.SH", "688001.SH"]
    features = _universe_features(
        codes, as_of=as_of, above_ma20=[False, True]
    )
    stocks = _universe_stocks(codes, markets=["主板", "科创板"])
    builder = UniverseBuilder(_universe_calendar())

    tradeable = builder.build_tradeable(
        as_of, features, stocks, pl.DataFrame(), config
    )
    breadth = (
        tradeable.filter(pl.col("above_ma20")).height / tradeable.height
    )
    # The 科创板 name is not in the Template-A universe but still counts.
    assert tradeable.height == 2
    assert breadth == 0.5
    assert builder.build(
        as_of, features, stocks, pl.DataFrame(), config
    )["ts_code"].to_list() == ["600030.SH"]


# ---------------------------------------------------------------------------
# Real-path fake provider
# ---------------------------------------------------------------------------

_AS_OF = date(2024, 6, 3)
_TRADING_DAY_COUNT = 120
_A_CODES = ("600000.SH", "600001.SH")
_ROTATION_CODES = tuple(f"6001{index:02d}.SH" for index in range(80))
_BACKGROUND_CODES = tuple(f"6002{index:02d}.SH" for index in range(18))
_REAL_CODES = (*_A_CODES, *_ROTATION_CODES, *_BACKGROUND_CODES)


def _business_days(start: date, end: date) -> list[date]:
    values: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return values


def _trading_days_ending(end: date, count: int) -> list[date]:
    values: list[date] = []
    current = end
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current)
        current -= timedelta(days=1)
    return sorted(values)


_REAL_DATES = _trading_days_ending(_AS_OF, _TRADING_DAY_COUNT)


def _a_closes() -> list[float]:
    values: list[float] = [9.00] * 60
    for index in range(40):
        values.append(9.00 + 0.25 * index / 39)
    for index in range(10):
        values.append(9.30 + 0.50 * index / 9)
    values.extend([9.96, 9.98, 10.00, 10.02, 10.04])
    values.extend([10.45, 10.40, 10.35, 10.30, 10.32])
    assert len(values) == _TRADING_DAY_COUNT
    return values


def _rotation_closes(stock_index: int) -> list[float]:
    values = [10.0]
    for day in range(1, _TRADING_DAY_COUNT):
        if (stock_index + day) % 2 == 0:
            change = 1.002
        else:
            change = 1.001 if stock_index % 4 in (0, 1) else 0.999
        values.append(values[-1] * change)
    return values


def _background_closes() -> list[float]:
    values = [10.0]
    for _ in range(1, _TRADING_DAY_COUNT):
        values.append(values[-1] * 0.998)
    return values


def _real_bars(
    bad_quality: bool = False,
    suspended_codes: tuple[str, ...] = (),
    missing_daily_codes: tuple[str, ...] = (),
    short_history_codes: tuple[str, ...] = (),
    zero_volume_codes: tuple[str, ...] = (),
    extra_bar_code: str | None = None,
) -> pl.DataFrame:
    rows: list[dict] = []
    a_path = _a_closes()
    background_path = _background_closes()
    for code in _REAL_CODES:
        if code in _A_CODES:
            closes = a_path
            amount, turnover = 1e9, 0.20
        elif code in _ROTATION_CODES:
            closes = _rotation_closes(_ROTATION_CODES.index(code))
            amount, turnover = 1e8, 0.005
        else:
            closes = background_path
            amount, turnover = 5e8, 0.10
        volumes = [
            1e8 if index < _TRADING_DAY_COUNT - 5 else 5e7
            for index in range(_TRADING_DAY_COUNT)
        ]
        previous = closes[0]
        for index, trade_date in enumerate(_REAL_DATES):
            close = closes[index]
            if code in _A_CODES:
                limit_up = index == _TRADING_DAY_COUNT - 6
            elif code in _ROTATION_CODES:
                limit_up = (
                    _ROTATION_CODES.index(code) + index
                ) % 2 == 0
            else:
                limit_up = False
            high = close if limit_up else max(previous, close) * 1.002
            up_limit = close if limit_up else close * 1.05
            low = min(previous, close) * 0.998
            if (
                bad_quality
                and trade_date == _AS_OF
                and code == _ROTATION_CODES[40]
            ):
                low = 100.0  # force an OHLC violation on the as-of bar
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "open": previous,
                    "high": high,
                    "low": low,
                    "close": close,
                    "pre_close": previous,
                    "vol": volumes[index],
                    "amount": amount,
                    "adj_factor": 1.0,
                    "turnover_rate": turnover,
                    "circ_mv": 1e10,
                    "up_limit": up_limit,
                    "down_limit": close * 0.9,
                    "is_suspended": False,
                    "limit_derived": True,
                }
            )
            previous = close
    omitted_codes = set(suspended_codes) | set(missing_daily_codes)
    if omitted_codes:
        # Tushare ``daily`` omits suspended names for the day.  Missing codes
        # model a reference name absent from daily without a matching suspend
        # row, which the exact expected-code check must catch.
        rows = [
            row
            for row in rows
            if not (
                row["trade_date"] == _AS_OF
                and row["ts_code"] in omitted_codes
            )
        ]
    if short_history_codes:
        # Keep only the last ten bars so return overlap falls below the
        # correlation minimum against full-history partners.
        cutoff = _REAL_DATES[-10]
        rows = [
            row
            for row in rows
            if not (
                row["ts_code"] in short_history_codes
                and row["trade_date"] < cutoff
            )
        ]
    if zero_volume_codes:
        for row in rows:
            if (
                row["trade_date"] == _AS_OF
                and row["ts_code"] in zero_volume_codes
            ):
                row["vol"] = 0.0
                row["amount"] = 0.0
                row["is_suspended"] = True
    if extra_bar_code is not None:
        rows.append(
            {
                "ts_code": extra_bar_code,
                "trade_date": _AS_OF,
                "open": 10.0,
                "high": 10.05,
                "low": 9.95,
                "close": 10.0,
                "pre_close": 10.0,
                "vol": 1e5,
                "amount": 1e8,
                "adj_factor": 1.0,
                "turnover_rate": 0.05,
                "circ_mv": 1e10,
                "up_limit": 11.0,
                "down_limit": 9.0,
                "is_suspended": False,
                "limit_derived": True,
            }
        )
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


class RealPathProvider:
    """Fake provider with enough history for the full pipeline."""

    def __init__(
        self,
        *,
        calendar_end: date | None = None,
        calendar_start: date | None = None,
        degraded: bool = False,
        bad_quality: bool = False,
        suspended_codes: tuple[str, ...] = (),
        suspended_with_bar_codes: tuple[str, ...] = (),
        partial_day_suspended_codes: tuple[str, ...] = (),
        suspend_resume_codes: tuple[str, ...] = (),
        missing_daily_codes: tuple[str, ...] = (),
        short_history_codes: tuple[str, ...] = (),
        zero_volume_codes: tuple[str, ...] = (),
        suspend_available: bool = True,
        extra_bar_code: str | None = None,
        empty_index: bool = False,
        stale_primary_index: bool = False,
        short_primary_index: bool = False,
        short_fallback_index: bool = False,
        empty_trade_cal: bool = False,
    ):
        self.calendar_end = calendar_end or (
            _AS_OF + timedelta(days=45)
        )
        self.calendar_start = calendar_start
        self.degraded = degraded
        self.suspended_codes = suspended_codes
        self.suspended_with_bar_codes = suspended_with_bar_codes
        self.partial_day_suspended_codes = partial_day_suspended_codes
        self.suspend_resume_codes = suspend_resume_codes
        self.suspend_available = suspend_available
        self.empty_index = empty_index
        self.stale_primary_index = stale_primary_index
        self.short_primary_index = short_primary_index
        self.short_fallback_index = short_fallback_index
        self.empty_trade_cal = empty_trade_cal
        self.trade_cal_calls: list[tuple[date, date]] = []
        self.bars_calls: list[tuple[date, date]] = []
        self.bars = _real_bars(
            bad_quality=bad_quality,
            suspended_codes=suspended_codes,
            missing_daily_codes=missing_daily_codes,
            short_history_codes=short_history_codes,
            zero_volume_codes=zero_volume_codes,
            extra_bar_code=extra_bar_code,
        )

    def load_bars(self, start: date, end: date) -> pl.DataFrame:
        self.bars_calls.append((start, end))
        return self.bars.filter(
            (pl.col("trade_date") >= start)
            & (pl.col("trade_date") <= end)
        )

    def load_suspend(self, start: date, end: date) -> pl.DataFrame | None:
        schema = {
            "trade_date": pl.Date,
            "ts_code": pl.String,
            "suspend_type": pl.String,
            "suspend_timing": pl.String,
        }
        if not self.suspend_available or not (start <= _AS_OF <= end):
            return None
        rows = [
            {
                "trade_date": _AS_OF,
                "ts_code": code,
                "suspend_type": "S",
                "suspend_timing": None,
            }
            for code in (
                *self.suspended_codes,
                *self.suspended_with_bar_codes,
            )
        ] + [
            {
                "trade_date": _AS_OF,
                "ts_code": code,
                "suspend_type": "S",
                "suspend_timing": "09:30-10:30",
            }
            for code in self.partial_day_suspended_codes
        ] + [
            {
                "trade_date": _AS_OF,
                "ts_code": code,
                "suspend_type": "R",
                "suspend_timing": None,
            }
            for code in self.suspend_resume_codes
        ]
        return pl.DataFrame(rows, schema=schema).filter(
            pl.col("suspend_type") == "S"
        )

    def load_index_bars(
        self, index_code: str, start: date, end: date
    ) -> pl.DataFrame:
        if self.empty_index:
            return pl.DataFrame()
        if index_code == "000985.CSI":
            if not (self.stale_primary_index or self.short_primary_index):
                return pl.DataFrame()
            if self.stale_primary_index:
                end = end - timedelta(days=1)
            dates = _business_days(start, end)
            if self.short_primary_index:
                dates = dates[-10:]
            return pl.DataFrame(
                {
                    "ts_code": [index_code] * len(dates),
                    "trade_date": dates,
                    "close": [
                        3000.0 + index for index in range(len(dates))
                    ],
                }
            )
        if index_code == "000300.SH":
            dates = _business_days(start, end)
            if self.short_fallback_index:
                dates = dates[-10:]
            return pl.DataFrame(
                {
                    "ts_code": [index_code] * len(dates),
                    "trade_date": dates,
                    "close": [
                        3000.0 + index for index in range(len(dates))
                    ],
                }
            )
        return pl.DataFrame()

    def load_stock_basic(self, as_of: date) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ts_code": list(_REAL_CODES),
                "name": [f"银行{index:03d}" for index in range(len(_REAL_CODES))],
                "market": ["主板"] * len(_REAL_CODES),
                "industry": ["银行"] * len(_REAL_CODES),
                "list_date": [date(2015, 1, 1)] * len(_REAL_CODES),
                "delist_date": [None] * len(_REAL_CODES),
            },
            schema_overrides={"delist_date": pl.Date},
        )

    def load_namechange(self) -> pl.DataFrame:
        if self.degraded:
            return pl.DataFrame()
        return pl.DataFrame(
            {
                "ts_code": list(_REAL_CODES),
                "name": [f"银行{index:03d}" for index in range(len(_REAL_CODES))],
                "start_date": [date(2015, 1, 1)] * len(_REAL_CODES),
                "end_date": [None] * len(_REAL_CODES),
            },
            schema_overrides={"end_date": pl.Date},
        )

    def load_trade_cal(self, start: date, end: date) -> pl.DataFrame:
        self.trade_cal_calls.append((start, end))
        if self.empty_trade_cal:
            return pl.DataFrame()
        limit = min(end, self.calendar_end)
        effective_start = (
            max(start, self.calendar_start)
            if self.calendar_start is not None
            else start
        )
        dates = _business_days(effective_start, limit)
        return pl.DataFrame(
            {"cal_date": dates, "is_open": [1] * len(dates)}
        )

    def load_index_member(self) -> pl.DataFrame:
        if self.degraded:
            return pl.DataFrame()
        return pl.DataFrame(
            {
                "index_code": ["801780.SI"] * len(_REAL_CODES),
                "con_code": list(_REAL_CODES),
                "in_date": [date(2015, 1, 1)] * len(_REAL_CODES),
                "out_date": [None] * len(_REAL_CODES),
            },
            schema_overrides={"out_date": pl.Date},
        )

    def load_index_classify(self) -> pl.DataFrame:
        if self.degraded:
            return pl.DataFrame()
        return pl.DataFrame(
            {
                "index_code": ["801780.SI"],
                "industry_name": ["银行"],
                "level": ["L1"],
                "src": ["SW2021"],
            }
        )

    def load_limit_list(self, start: date, end: date) -> pl.DataFrame:
        if self.degraded:
            return pl.DataFrame()
        return pl.DataFrame(
            {
                "ts_code": ["600101.SH"],
                "trade_date": [_AS_OF],
            }
        )


def _run_real_path(provider: RealPathProvider):
    return TradingPipeline(
        provider=provider, config=load_strategy_config()
    ).run(as_of=_AS_OF, account=_account())


def test_pipeline_real_path_produces_plans_and_correlation_rejection():
    result = _run_real_path(RealPathProvider())

    assert result.as_of == _AS_OF
    assert result.regime.sentiment_stage == SentimentStage.START
    assert result.sectors
    assert result.sectors[0].sector == "银行"
    assert [candidate.ts_code for candidate in result.candidates] == [
        "600000.SH",
        "600001.SH",
    ]
    assert result.rejected == []
    assert all(plan.expires_on > _AS_OF for plan in result.plans)

    approved = [
        plan for plan in result.plans
        if plan.status == PlanStatus.PENDING_ENTRY
    ]
    rejected = [
        plan for plan in result.plans if plan.status == PlanStatus.REJECTED
    ]
    assert len(approved) == 1
    assert len(rejected) == 1
    assert approved[0].risk_decision.outcome == RiskOutcome.APPROVE
    assert rejected[0].risk_decision.outcome == RiskOutcome.REJECT
    assert "correlation above threshold" in rejected[0].risk_decision.reasons[0]

    # The bar quality report ran and its warning became a degraded flag.
    assert "limit_prices_derived" in result.regime.degraded_flags
    # The primary index is unavailable, so the fallback supplied the bars.
    assert "index_fallback" in result.regime.degraded_flags
    assert result.account.trading_halted is False


def test_pipeline_real_path_reports_degraded_flags_without_duplicates():
    result = _run_real_path(RealPathProvider(degraded=True))

    flags = result.regime.degraded_flags
    for flag in (
        "st_namechange_missing",
        "industry_stock_basic",
        "limit_list_d",
        "index_fallback",
        "initial_stage",
        "limit_prices_derived",
    ):
        assert flag in flags
    assert len(flags) == len(set(flags))
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


def test_pipeline_raises_on_bar_quality_failure():
    provider = RealPathProvider(bad_quality=True)
    pipeline = TradingPipeline(
        provider=provider, config=load_strategy_config()
    )
    with pytest.raises(DataQualityError, match="OHLC"):
        pipeline.run(as_of=_AS_OF, account=_account())


def test_pipeline_requires_forward_trade_cal_window():
    provider = RealPathProvider(calendar_end=_AS_OF)
    pipeline = TradingPipeline(
        provider=provider, config=load_strategy_config()
    )
    with pytest.raises(DataNotReadyError, match="forward trade_cal"):
        pipeline.run(as_of=_AS_OF, account=_account())


def test_pipeline_requests_forward_trade_cal_window():
    provider = RealPathProvider()
    _run_real_path(provider)
    assert provider.trade_cal_calls == [
        (_AS_OF - timedelta(days=400), _AS_OF + timedelta(days=45))
    ]


def test_pipeline_requires_open_trade_cal_for_as_of():
    guidance = "sync/backfill a forward trade_cal window"
    empty_calendar = TradingPipeline(
        provider=RealPathProvider(empty_trade_cal=True),
        config=load_strategy_config(),
    )
    with pytest.raises(DataNotReadyError, match=guidance):
        empty_calendar.run(as_of=_AS_OF, account=_account())

    # Saturday is not an open trading day in the fake calendar.
    non_trading_day = TradingPipeline(
        provider=RealPathProvider(), config=load_strategy_config()
    )
    with pytest.raises(DataNotReadyError, match=guidance):
        non_trading_day.run(
            as_of=_AS_OF + timedelta(days=5), account=_account()
        )


def test_pipeline_raises_when_both_index_sources_are_empty():
    with pytest.raises(DataNotReadyError, match="index bars missing"):
        _run_real_path(RealPathProvider(empty_index=True))


def test_pipeline_suspended_reference_code_absent_from_daily_passes():
    suspended = _BACKGROUND_CODES[0]
    result = _run_real_path(
        RealPathProvider(suspended_codes=(suspended,))
    )
    assert [candidate.ts_code for candidate in result.candidates] == [
        "600000.SH",
        "600001.SH",
    ]
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


def test_pipeline_present_empty_suspend_partition_uses_exact_codes():
    # A partition exists but has no S rows, so a reference code missing from
    # daily must fail the exact-code check.
    provider = RealPathProvider(
        missing_daily_codes=(_BACKGROUND_CODES[0],)
    )
    with pytest.raises(DataQualityError, match="missing expected codes"):
        _run_real_path(provider)


def test_pipeline_present_empty_suspend_partition_same_count_swap_fails():
    provider = RealPathProvider(
        missing_daily_codes=(_BACKGROUND_CODES[0],),
        extra_bar_code="699999.SH",
    )
    with pytest.raises(DataQualityError) as excinfo:
        _run_real_path(provider)
    message = str(excinfo.value)
    assert "missing expected codes" in message
    assert "unexpected codes" in message


def test_pipeline_pure_extra_bar_fails_unexpected_codes_on_zero_suspension_day():
    provider = RealPathProvider(extra_bar_code="699999.SH")
    with pytest.raises(DataQualityError) as excinfo:
        _run_real_path(provider)
    message = str(excinfo.value)
    assert "unexpected codes" in message
    assert "coverage" not in message


def test_pipeline_resume_only_suspend_partition_uses_exact_codes():
    # An R-only partition is a present-but-empty suspend day: no suspensions.
    result = _run_real_path(
        RealPathProvider(suspend_resume_codes=(_BACKGROUND_CODES[0],))
    )
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)

    with pytest.raises(DataQualityError, match="missing expected codes"):
        _run_real_path(
            RealPathProvider(
                suspend_resume_codes=(_BACKGROUND_CODES[0],),
                missing_daily_codes=(_BACKGROUND_CODES[1],),
            )
        )


def test_pipeline_no_suspend_partition_falls_back_to_count_coverage():
    provider = RealPathProvider(
        missing_daily_codes=(_BACKGROUND_CODES[0],),
        suspend_available=False,
    )
    result = _run_real_path(provider)
    assert [candidate.ts_code for candidate in result.candidates] == [
        "600000.SH",
        "600001.SH",
    ]
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)
    assert "suspend_d_missing" in result.regime.degraded_flags


def test_pipeline_full_day_suspended_code_with_bar_passes():
    # A full-day suspended name is allowed (not required) to have a bar.
    provider = RealPathProvider(
        suspended_with_bar_codes=(_BACKGROUND_CODES[0],)
    )
    result = _run_real_path(provider)
    assert [candidate.ts_code for candidate in result.candidates] == [
        "600000.SH",
        "600001.SH",
    ]
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


def test_pipeline_partial_day_suspended_code_with_bar_passes():
    # A partial-day suspension still trades, so its bar is required.
    provider = RealPathProvider(
        partial_day_suspended_codes=(_BACKGROUND_CODES[0],)
    )
    result = _run_real_path(provider)
    assert [candidate.ts_code for candidate in result.candidates] == [
        "600000.SH",
        "600001.SH",
    ]
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


def test_pipeline_zero_volume_suspended_bar_passes():
    # A zero-volume bar is still present for a required code, so it passes.
    provider = RealPathProvider(
        zero_volume_codes=(_BACKGROUND_CODES[0],)
    )
    result = _run_real_path(provider)
    assert [candidate.ts_code for candidate in result.candidates] == [
        "600000.SH",
        "600001.SH",
    ]
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


def test_pipeline_all_stocks_suspended_raises():
    provider = RealPathProvider(suspended_codes=_REAL_CODES)
    with pytest.raises(DataNotReadyError, match="all listed stocks are suspended"):
        _run_real_path(provider)


class SingleShortHistoryScreener:
    name = "single_short_history"
    version = "1"

    def generate(self, as_of, regime, features, sectors, leaders, config):
        return [
            Candidate(
                ts_code=_A_CODES[0],
                trade_date=as_of,
                template=Template.A,
                setup_type="test",
                horizon="mid",
                sector="银行",
                leader_rank=1,
                leader_score=90.0,
                score=90.0,
                reasons=["test"],
                features={
                    "ma20_raw": 10.0,
                    "ma5_raw": 10.5,
                    "ma10_raw": 10.0,
                    "board_height": 0.0,
                    "atr14_raw": 0.5,
                    "low_10d_raw": 9.5,
                    "low_5d_raw": 9.7,
                    "high_60d_raw": 30.0,
                    "close": 10.2,
                },
                sentiment_stage=regime.sentiment_stage,
                strategy_version=config.version,
                data_version=regime.data_version,
            )
        ]


def test_pipeline_marks_correlation_unavailable_for_short_history_candidate():
    provider = RealPathProvider(short_history_codes=(_A_CODES[0],))
    account = _account(
        positions=[
            PositionState(
                ts_code=_BACKGROUND_CODES[0],
                sector="银行",
                shares=100,
                available_shares=100,
                cost_price=Decimal("10"),
                current_stop=Decimal("9"),
                current_price=Decimal("10"),
                template=Template.A,
            )
        ]
    )
    result = TradingPipeline(
        provider=provider,
        config=load_strategy_config(),
        screener=SingleShortHistoryScreener(),
    ).run(as_of=_AS_OF, account=account)

    assert len(result.plans) == 1
    plan = result.plans[0]
    assert plan.status == PlanStatus.PENDING_ENTRY
    assert "correlation unavailable" in plan.risk_decision.reasons


def test_correlation_returns_use_adjusted_close_and_drop_gaps():
    as_of = date(2024, 2, 1)
    dates = _trading_days_ending(as_of, 25)
    a_closes = [
        10.0 * (1 + 0.01 * ((index % 5) - 2))
        for index in range(len(dates))
    ]
    rows: list[dict] = []
    for index, trade_date in enumerate(dates):
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "close": a_closes[index],
                "adj_factor": 1.0,
            }
        )
    for index, trade_date in enumerate(dates):
        if index == 10:
            continue  # suspension gap
        adj_factor = 2.0 if index >= 12 else 1.0  # 2:1 split at index 12
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": trade_date,
                "close": a_closes[index] / adj_factor,
                "adj_factor": adj_factor,
            }
        )

    returns = TradingPipeline._correlation_returns(
        pl.DataFrame(rows), as_of
    )
    second = returns.filter(pl.col("ts_code") == "000002.SZ").sort(
        "trade_date"
    )
    gap_return = second.filter(
        pl.col("trade_date") == dates[11]
    )["ret"][0]
    assert gap_return is None
    split_return = second.filter(
        pl.col("trade_date") == dates[12]
    )["ret"][0]
    assert split_return == pytest.approx(
        a_closes[12] / a_closes[11] - 1
    )


def test_pipeline_falls_back_on_stale_primary_index():
    result = _run_real_path(RealPathProvider(stale_primary_index=True))
    assert "index_fallback" in result.regime.degraded_flags
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


def test_pipeline_falls_back_on_short_primary_index():
    result = _run_real_path(RealPathProvider(short_primary_index=True))
    assert "index_fallback" in result.regime.degraded_flags
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


def test_pipeline_raises_when_index_history_insufficient():
    provider = RealPathProvider(
        short_primary_index=True, short_fallback_index=True
    )
    with pytest.raises(
        DataNotReadyError, match="index bars missing or insufficient"
    ):
        _run_real_path(provider)


def test_pipeline_requested_windows_follow_config():
    config = load_strategy_config()
    template_a = config.template_a.model_copy(
        update={"entry_valid_days": 30, "min_listing_days": 250}
    )
    config = config.model_copy(update={"template_a": template_a})
    provider = RealPathProvider()

    result = TradingPipeline(
        provider=provider, config=config
    ).run(as_of=_AS_OF, account=_account())

    assert provider.bars_calls == [
        (_AS_OF - timedelta(days=560), _AS_OF)
    ]
    assert provider.trade_cal_calls == [
        (_AS_OF - timedelta(days=560), _AS_OF + timedelta(days=70))
    ]
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


class RecordingFeatureBuilder:
    def __init__(self) -> None:
        self.calls: list[date] = []
        self._inner = FeatureBuilder()

    def build(self, bars: pl.DataFrame, trade_date: date) -> pl.DataFrame:
        self.calls.append(trade_date)
        return self._inner.build(bars, trade_date)


def test_pipeline_feature_history_follows_persistence_config():
    config = load_strategy_config()
    industry = config.industry.model_copy(update={"persistence_days": 4})
    config = config.model_copy(update={"industry": industry})
    builder = RecordingFeatureBuilder()

    result = TradingPipeline(
        provider=RealPathProvider(),
        config=config,
        feature_builder=builder,
    ).run(as_of=_AS_OF, account=_account())

    assert len(builder.calls) == 7  # max(5, persistence_days + 3)
    assert any(plan.status == PlanStatus.PENDING_ENTRY for plan in result.plans)


def test_pipeline_requires_calendar_history_for_listing_filter():
    provider = RealPathProvider(
        calendar_start=_AS_OF - timedelta(days=100)
    )
    with pytest.raises(DataNotReadyError, match="listing filter"):
        _run_real_path(provider)
