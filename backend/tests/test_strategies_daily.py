from __future__ import annotations

from datetime import timedelta

import polars as pl

from app.core.full_config import load_full_config
from app.domain.enums import Strategy, SubMode
from app.strategies import a_trend, b_limit_up, c_platform, d_oversold, e_range, f_event
from app.strategies.registry import MultiStrategyScreener
from tests.strategy_helpers import make_bars, make_context


def _c1_bars():
    closes = [10.2, 10.3, 10.1, 10.4, 10.2] * 5
    highs = [10.5 if index % 5 == 0 else 10.35 for index in range(25)]
    lows = [10.0 if index % 5 == 2 else 10.15 for index in range(25)]
    opens = [10.2] * 25
    volumes = [20_000_000.0] * 25
    return make_bars(
        closes + [10.8],
        opens=opens + [10.7],
        highs=highs + [10.85],
        lows=lows + [10.7],
        volumes=volumes + [40_000_000.0],
    )


def test_c1_platform_breakout():
    closes = [10.2, 10.3, 10.1, 10.4, 10.2] * 5
    highs = [10.5 if index % 5 == 0 else 10.35 for index in range(25)]
    lows = [10.0 if index % 5 == 2 else 10.15 for index in range(25)]
    opens = [10.2] * 25
    volumes = [20_000_000.0] * 25
    bars = make_bars(
        closes + [10.8],
        opens=opens + [10.7],
        highs=highs + [10.85],
        lows=lows + [10.7],
        volumes=volumes + [40_000_000.0],
    )
    context = make_context(bars)
    candidates = c_platform.generate(context)
    assert any(
        candidate.strategy == Strategy.C and candidate.sub_mode == SubMode.C1
        for candidate in candidates
    )


def test_c2_platform_pullback_confirmation():
    closes = [10.2, 10.3, 10.1, 10.4, 10.2] * 5
    highs = [10.5 if index % 5 == 0 else 10.35 for index in range(25)]
    lows = [10.0 if index % 5 == 2 else 10.15 for index in range(25)]
    opens = [10.2] * 25
    volumes = [20_000_000.0] * 25
    closes += [10.8, 10.55, 10.50, 10.60, 10.90]
    highs += [10.85, 10.60, 10.55, 10.70, 10.95]
    lows += [10.70, 10.45, 10.42, 10.50, 10.80]
    opens += [10.70, 10.50, 10.45, 10.55, 10.70]
    volumes += [40_000_000.0] + [15_000_000.0] * 4
    bars = make_bars(
        closes, opens=opens, highs=highs, lows=lows, volumes=volumes
    )
    context = make_context(bars)
    candidates = c_platform.generate(context)
    assert any(
        candidate.strategy == Strategy.C
        and candidate.sub_mode == SubMode.C2
        and candidate.evidence.get("stage") == "confirmed"
        for candidate in candidates
    )


def test_e_range_entry():
    closes = [10.5] * 35
    highs = [11.0 if index % 5 == 0 else 10.6 for index in range(35)]
    lows = [10.0 if index % 5 == 2 else 10.4 for index in range(35)]
    opens = [10.45] * 35
    volumes = [20_000_000.0] * 35
    closes[-1] = 10.10
    highs[-1] = 10.18
    lows[-1] = 10.05
    bars = make_bars(
        closes + [10.19],
        opens=opens + [10.15],
        highs=highs + [10.20],
        lows=lows + [10.12],
        volumes=volumes + [20_000_000.0],
    )
    context = make_context(bars)
    candidates = e_range.generate(context)
    assert any(
        candidate.strategy == Strategy.E and candidate.sub_mode == SubMode.E_RANGE
        for candidate in candidates
    )


def test_b1_limit_up_absorption():
    closes = [10.0] * 30
    highs = [10.1] * 30
    lows = [9.9] * 30
    opens = [10.0] * 30
    closes[28], highs[28], lows[28], opens[28] = 11.0, 11.0, 10.5, 10.6
    closes[29], highs[29], lows[29], opens[29] = 11.1, 11.2, 10.8, 10.9
    bars = make_bars(
        closes + [11.3],
        opens=opens + [11.1],
        highs=highs + [11.35],
        lows=lows + [11.0],
        volumes=[20_000_000.0] * 31,
    )
    context = make_context(bars)
    candidates = b_limit_up.generate(context)
    assert any(
        candidate.strategy == Strategy.B
        and candidate.sub_mode == SubMode.B1
        and candidate.evidence.get("stage") == "confirmed"
        for candidate in candidates
    )
    assert any(
        candidate.sub_mode == SubMode.B2
        and candidate.evidence.get("stage") == "watchlist"
        for candidate in candidates
    )


def test_d_oversold_repair():
    closes = [10.0 + index * (4.0 / 55) for index in range(56)]
    closes += [13.8, 13.2, 12.5, 11.5, 10.8]
    closes += [10.5, 10.6, 10.8, 11.1]
    highs = [close * 1.005 for close in closes]
    lows = [close * 0.995 for close in closes]
    lows[-5] = 10.2
    lows[-4] = 10.3
    lows[-3] = 10.4
    lows[-2] = 10.5
    lows[-1] = 10.6
    highs[-4] = 10.65
    highs[-3] = 10.75
    bars = make_bars(closes, highs=highs, lows=lows, volumes=[20_000_000.0] * len(closes))
    context = make_context(bars)
    candidates = d_oversold.generate(context)
    assert any(
        candidate.strategy == Strategy.D
        and candidate.sub_mode == SubMode.D_DOWNTREND
        for candidate in candidates
    )


def test_a_trend_pullback_confirmation():
    n = 75
    closes = [10 + index * (3.0 / (n - 6)) for index in range(n - 5)]
    closes += [12.9, 12.7, 12.5, 12.4, 12.3, 12.8]
    highs = [close * 1.004 for close in closes]
    lows = [close * 0.996 for close in closes]
    lows[-2] = 12.55
    highs[-2] = 12.60
    lows[-3] = 12.45
    highs[-3] = 12.55
    volumes = [10_000_000.0] * len(closes)
    volumes[-5:] = [5_000_000.0] * 5
    bars = make_bars(closes, highs=highs, lows=lows, volumes=volumes)
    context = make_context(bars)
    candidates = a_trend.generate(context)
    assert any(
        candidate.strategy == Strategy.A
        and candidate.sub_mode == SubMode.A_SHALLOW
        and candidate.evidence.get("stage") == "confirmed"
        for candidate in candidates
    )


def test_f_event_uses_c_price_confirmation():
    bars = _c1_bars()
    context = make_context(
        bars,
        events={
            "000001.SZ": [
                {
                    "event_id": "evt-1",
                    "event_type": "contract",
                    "title": "重大合同",
                    "source": "manual",
                    "ann_date": bars["trade_date"][-1],
                    "verification_status": "verified",
                    "direction": "positive",
                    "materiality": 0.8,
                    "retracted": False,
                    "stage": "formal",
                    "certainty": "formal",
                }
            ]
        },
    )
    config = load_full_config().model_copy(deep=True)
    config.strategies.f.price_mode = "c1"
    context.config = config
    candidates = f_event.generate(context)
    assert any(
        candidate.strategy == Strategy.F
        and candidate.sub_mode == SubMode.F_CONTRACT
        and candidate.evidence.get("event_id") == "evt-1"
        for candidate in candidates
    )


def test_multi_strategy_screener_keeps_families_separate():
    closes = [10.0] * 30
    bars = make_bars(closes)
    context = make_context(bars)
    bundle = MultiStrategyScreener().generate(context)
    assert set(bundle.by_strategy) == set(Strategy)
    assert bundle.paused == {}


def test_a_trend_rejects_downtrend_with_adjusted_prices():
    # A constant adj_factor makes hfq_close look large versus ma_raw; the
    # trend filter must compare raw close with raw moving averages.
    closes = [20.0 - index * 0.1 for index in range(80)]
    bars = make_bars(closes).with_columns(pl.lit(2.0).alias("adj_factor"))
    context = make_context(bars)
    assert not [
        candidate for candidate in a_trend.generate(context)
        if candidate.strategy == Strategy.A
    ]


def test_f_event_generates_independent_submodes_per_event_type():
    bars = _c1_bars()
    day = bars["trade_date"][-1]
    events = [
        {
            "event_id": f"evt-{event_type}",
            "event_type": event_type,
            "title": event_type,
            "source": "manual",
            "ann_date": day,
            "verification_status": "verified",
            "direction": "positive",
            "materiality": 0.8,
            "retracted": False,
            "stage": "formal",
            "certainty": "formal",
        }
        for event_type in ("earnings", "buyback", "contract", "policy")
    ]
    context = make_context(bars, events={"000001.SZ": events})
    config = load_full_config().model_copy(deep=True)
    config.strategies.f.price_mode = "c1"
    context.config = config
    sub_modes = {
        candidate.sub_mode
        for candidate in f_event.generate(context)
        if candidate.strategy == Strategy.F
    }
    assert sub_modes == {
        SubMode.F_EARNINGS,
        SubMode.F_BUYBACK,
        SubMode.F_CONTRACT,
        SubMode.F_POLICY,
    }


def test_a_and_c_are_invariant_to_future_bars():
    a_closes = [10 + index * (3.0 / 69) for index in range(70)]
    a_closes += [12.9, 12.7, 12.5, 12.4, 12.3, 12.8]
    a_bars = make_bars(a_closes)
    c_bars = _c1_bars()
    for module, bars in ((a_trend, a_bars), (c_platform, c_bars)):
        as_of = bars["trade_date"][-3]
        full = make_context(bars, as_of=as_of)
        truncated = make_context(
            bars.filter(pl.col("trade_date") <= as_of), as_of=as_of
        )
        assert [
            candidate.model_dump() for candidate in module.generate(full)
        ] == [
            candidate.model_dump() for candidate in module.generate(truncated)
        ]


def _d_bars():
    closes = [10.0 + index * (4.0 / 55) for index in range(56)]
    closes += [13.8, 13.2, 12.5, 11.5, 10.8]
    closes += [10.5, 10.6, 10.8, 11.1]
    highs = [close * 1.005 for close in closes]
    lows = [close * 0.995 for close in closes]
    lows[-5:] = [10.2, 10.3, 10.4, 10.5, 10.6]
    highs[-4] = 10.65
    highs[-3] = 10.75
    return make_bars(
        closes, highs=highs, lows=lows, volumes=[20_000_000.0] * len(closes)
    )


def test_d_negative_event_only_blocks_within_risk_window():
    bars = _d_bars()
    as_of = bars["trade_date"][-1]
    recent = {
        "event_id": "neg-recent",
        "event_type": "contract",
        "ann_date": as_of,
        "verification_status": "verified",
        "direction": "negative",
        "title": "negative",
        "materiality": 0.9,
    }
    stale = {
        **recent,
        "event_id": "neg-stale",
        "ann_date": as_of - timedelta(days=60),
    }
    recent_context = make_context(bars, events={"000001.SZ": [recent]})
    stale_context = make_context(bars, events={"000001.SZ": [stale]})
    assert not [
        item for item in d_oversold.generate(recent_context)
        if item.strategy == Strategy.D
    ]
    assert any(
        item.strategy == Strategy.D for item in d_oversold.generate(stale_context)
    )
