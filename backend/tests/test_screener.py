from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from app.core.config import StrategyConfig, load_strategy_config
from app.domain.enums import SentimentStage, Template, TrendRegime
from app.domain.models import (
    Candidate,
    MarketRegime,
    RegimeIndicators,
    SectorStrength,
    TemplatePolicy,
)
from app.screener.dual_template import DualTemplateScreener

_AS_OF = date(2024, 1, 2)

_COMMON: dict[str, object] = {
    "trade_date": _AS_OF,
    "sector": "银行",
    "hfq_close": 10.2,
    "ma5": 10.0,
    "ma10": 9.9,
    "ma20": 10.0,
    "ma60": 9.0,
    "ma20_slope": 0.2,
    "rs_20d_market": 0.9,
    "close_to_high_60d": 0.95,
    "pullback_days": 5,
    "vol_ratio_5_20": 0.7,
    "stabilization": True,
    "circ_mv": 1e10,
    "close": 10.2,
    "is_st": False,
    "is_suspended": False,
    "is_limit_up": False,
    "atr14": 0.4,
    "atr14_raw": 0.4,
    "low_5d_raw": 9.6,
    "low_10d_raw": 9.4,
    "high_60d_raw": 10.7,
    "turnover_rate": 0.05,
    "board_height": 0,
    "days_since_limit_up": 20,
    "one_word": False,
    "ret_20d": 0.20,
    "vol_ratio_today_20": 1.0,
    "above_ma20": True,
}

_B_OVERRIDES: dict[str, object] = {
    "hfq_close": 10.0,
    "close": 10.0,
    "is_limit_up": True,
    "board_height": 1,
    "days_since_limit_up": 0,
    "vol_ratio_today_20": 3.0,
    "turnover_rate": 0.10,
    "ret_20d": 0.30,
}


def _features() -> pl.DataFrame:
    common = dict(_COMMON)
    a = {"ts_code": "000001.SZ", **common}
    b = {
        **common,
        "ts_code": "000002.SZ",
        "hfq_close": 10.0,
        "close": 10.0,
        "is_limit_up": True,
        "board_height": 1,
        "days_since_limit_up": 0,
        "vol_ratio_today_20": 3.0,
        "turnover_rate": 0.10,
        "ret_20d": 0.30,
    }
    follower = {
        **common,
        "ts_code": "000003.SZ",
    }
    return pl.DataFrame([a, b, follower])


def _leaders() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "sector": ["银行"] * 3,
            "leader_score": [90.0, 85.0, 70.0],
            "leader_rank": [1, 2, 3],
            "is_follower": [False, False, True],
            "amount_20d_avg": [3e8, 2e8, 1e8],
        }
    )


def _sector() -> SectorStrength:
    return SectorStrength(
        sector="银行",
        score=90.0,
        rank=1,
        return_20d=0.2,
        return_5d=0.05,
        breadth=0.8,
        amount_change=0.2,
        limit_up_share=0.3,
        limit_up_count=3,
        red_rate=0.6,
        stock_count=10,
        persistent_days=2,
    )


def _indicators(**overrides: object) -> RegimeIndicators:
    values: dict[str, object] = {
        "limit_up_count": 80,
        "limit_down_count": 0,
        "max_board_height": 5,
        "promotion_rate": 0.5,
        "blow_rate": 0.1,
        "red_rate": 0.6,
        "limit_up_premium": 0.02,
        "breadth": 0.6,
        "index_close": 4000.0,
        "index_ma20": 3900.0,
        "index_ma60": 3800.0,
    }
    values.update(overrides)
    return RegimeIndicators.model_validate(values)


def _regime(
    *,
    policy_a: TemplatePolicy | None = None,
    policy_b: TemplatePolicy | None = None,
    indicators: RegimeIndicators | None = None,
    degraded_flags: list[str] | None = None,
) -> MarketRegime:
    a = (
        policy_a
        if policy_a is not None
        else TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    )
    b = (
        policy_b
        if policy_b is not None
        else TemplatePolicy(enabled=True, mode="normal", max_positions=2)
    )
    return MarketRegime(
        trade_date=_AS_OF,
        trend_regime=TrendRegime.ATTACK,
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={Template.A: a, Template.B: b},
        indicators=_indicators() if indicators is None else indicators,
        degraded_flags=[] if degraded_flags is None else degraded_flags,
        data_version="dv_test",
    )


def _config(
    *,
    template_a: dict[str, object] | None = None,
    template_b: dict[str, object] | None = None,
) -> StrategyConfig:
    config = load_strategy_config()
    updates: dict[str, object] = {}
    if template_a is not None:
        updates["template_a"] = config.template_a.model_copy(update=template_a)
    if template_b is not None:
        updates["template_b"] = config.template_b.model_copy(update=template_b)
    return config.model_copy(update=updates) if updates else config


def _run(
    *,
    regime: MarketRegime | None = None,
    features: pl.DataFrame | None = None,
    sectors: list[SectorStrength] | None = None,
    leaders: pl.DataFrame | None = None,
    config: StrategyConfig | None = None,
) -> list[Candidate]:
    return DualTemplateScreener().generate(
        as_of=_AS_OF,
        regime=_regime() if regime is None else regime,
        features=_features() if features is None else features,
        sectors=[_sector()] if sectors is None else sectors,
        leaders=_leaders() if leaders is None else leaders,
        config=load_strategy_config() if config is None else config,
    )


def _single_feature(**overrides: object) -> pl.DataFrame:
    return pl.DataFrame([{"ts_code": "000001.SZ", **_COMMON, **overrides}])


def _b_feature(**overrides: object) -> pl.DataFrame:
    return _single_feature(**{**_B_OVERRIDES, **overrides})


def _leader(
    ts_code: str,
    *,
    sector: str = "银行",
    score: float = 90.0,
    rank: int = 1,
    follower: bool = False,
    amount_20d_avg: float = 1e8,
) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "sector": sector,
        "leader_score": score,
        "leader_rank": rank,
        "is_follower": follower,
        "amount_20d_avg": amount_20d_avg,
    }


def test_dual_template_selects_a_and_b():
    config = load_strategy_config()
    candidates = DualTemplateScreener().generate(
        as_of=date(2024, 1, 2),
        regime=_regime(),
        features=_features(),
        sectors=[_sector()],
        leaders=_leaders(),
        config=config,
    )
    assert {candidate.template for candidate in candidates} == {
        Template.A,
        Template.B,
    }
    assert all(candidate.reasons for candidate in candidates)


def test_candidate_contract_and_selected_codes():
    config = load_strategy_config()
    regime = _regime()
    candidates = _run(config=config, regime=regime)
    assert candidates
    for candidate in candidates:
        assert candidate.reasons
        assert 0.0 <= candidate.score <= 100.0
        assert candidate.leader_rank >= 1
        assert candidate.features
        assert all(isinstance(value, float) for value in candidate.features.values())
        assert candidate.strategy_version == config.version
        assert candidate.data_version == regime.data_version
        assert candidate.trade_date == _AS_OF

    by_template = {candidate.template: candidate for candidate in candidates}
    a = by_template[Template.A]
    assert a.ts_code == "000001.SZ"
    assert a.features["hfq_close"] == pytest.approx(10.2)
    b = by_template[Template.B]
    assert b.ts_code == "000002.SZ"


def test_empty_inputs_return_no_candidates():
    screener = DualTemplateScreener()
    config = load_strategy_config()
    regime = _regime()
    cases = [
        (pl.DataFrame(), _leaders()),
        (_features(), pl.DataFrame()),
        (_features(), pl.DataFrame(schema={"ts_code": pl.String})),
    ]
    for features, leaders in cases:
        assert (
            screener.generate(
                as_of=_AS_OF,
                regime=regime,
                features=features,
                sectors=[_sector()],
                leaders=leaders,
                config=config,
            )
            == []
        )
    assert _run(sectors=[]) == []


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(
            TemplatePolicy(enabled=False, mode="normal", max_positions=0),
            id="disabled",
        ),
        pytest.param(
            TemplatePolicy(enabled=True, mode="observe", max_positions=3),
            id="observe",
        ),
    ],
)
def test_template_a_skipped_when_disabled_or_observing(policy: TemplatePolicy):
    candidates = _run(regime=_regime(policy_a=policy))
    assert not any(candidate.template == Template.A for candidate in candidates)
    assert any(candidate.template == Template.B for candidate in candidates)


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(
            TemplatePolicy(enabled=False, mode="normal", max_positions=0),
            id="disabled",
        ),
        pytest.param(
            TemplatePolicy(enabled=True, mode="observe", max_positions=2),
            id="observe",
        ),
    ],
)
def test_template_b_skipped_when_disabled_or_observing(policy: TemplatePolicy):
    candidates = _run(regime=_regime(policy_b=policy))
    assert not any(candidate.template == Template.B for candidate in candidates)
    assert any(candidate.template == Template.A for candidate in candidates)


@pytest.mark.parametrize(
    ("feature_overrides", "leader_overrides"),
    [
        pytest.param({}, {"follower": True}, id="follower"),
        pytest.param({"is_st": True}, {}, id="st"),
        pytest.param({"is_suspended": True}, {}, id="suspended"),
        pytest.param(
            {"is_limit_up": True, "board_height": 1}, {}, id="limit_up"
        ),
        pytest.param({"close_to_high_60d": 0.80}, {}, id="far_from_60d_high"),
        pytest.param(
            {"hfq_close": 11.0, "close": 11.0, "ma10": 10.0, "ma20": 10.0},
            {},
            id="extended_above_ma",
        ),
    ],
)
def test_template_a_excludes_ineligible_rows(
    feature_overrides: dict[str, object],
    leader_overrides: dict[str, object],
):
    features = _single_feature(**feature_overrides)
    leaders = pl.DataFrame([_leader("000001.SZ", **leader_overrides)])
    candidates = _run(features=features, leaders=leaders)
    assert not any(candidate.template == Template.A for candidate in candidates)


@pytest.mark.parametrize(
    ("feature_overrides", "leader_overrides"),
    [
        pytest.param({}, {"follower": True}, id="follower"),
        pytest.param({"is_st": True}, {}, id="st"),
        pytest.param({"is_suspended": True}, {}, id="suspended"),
        pytest.param({"one_word": True}, {}, id="one_word"),
        pytest.param({"board_height": 3}, {}, id="board_above_entry_range"),
        pytest.param({"board_height": 5}, {}, id="board_above_max"),
        pytest.param({"ret_20d": 0.81}, {}, id="overextended_20d_gain"),
    ],
)
def test_template_b_excludes_ineligible_rows(
    feature_overrides: dict[str, object],
    leader_overrides: dict[str, object],
):
    features = _b_feature(**feature_overrides)
    leaders = pl.DataFrame([_leader("000001.SZ", **leader_overrides)])
    candidates = _run(features=features, leaders=leaders)
    assert not any(candidate.template == Template.B for candidate in candidates)


@pytest.mark.parametrize(
    "indicators",
    [
        pytest.param(_indicators(blow_rate=0.36), id="blow_rate_too_high"),
        pytest.param(_indicators(max_board_height=2), id="max_board_too_low"),
    ],
)
def test_template_b_regime_gates(indicators: RegimeIndicators):
    candidates = _run(
        regime=_regime(indicators=indicators),
        features=_b_feature(),
        leaders=pl.DataFrame([_leader("000001.SZ")]),
    )
    assert not any(candidate.template == Template.B for candidate in candidates)


def test_template_a_per_industry_cap():
    config = _config(template_a={"max_per_industry": 1})
    features = pl.DataFrame(
        [
            {"ts_code": "000001.SZ", **_COMMON},
            {"ts_code": "000002.SZ", **_COMMON},
        ]
    )
    leaders = pl.DataFrame(
        [
            _leader("000001.SZ", score=90.0, rank=1),
            _leader("000002.SZ", score=80.0, rank=2),
        ]
    )
    candidates = [
        candidate
        for candidate in _run(features=features, leaders=leaders, config=config)
        if candidate.template == Template.A
    ]
    assert [candidate.ts_code for candidate in candidates] == ["000001.SZ"]


def test_template_a_global_candidate_cap():
    config = _config(template_a={"max_candidates": 1})
    other_sector = _sector().model_copy(
        update={"sector": "医药", "rank": 2, "score": 80.0}
    )
    features = pl.DataFrame(
        [
            {"ts_code": "000001.SZ", **_COMMON},
            {"ts_code": "000002.SZ", **_COMMON, "sector": "医药"},
        ]
    )
    leaders = pl.DataFrame(
        [
            _leader("000001.SZ", score=90.0, rank=1),
            _leader("000002.SZ", sector="医药", score=80.0, rank=1),
        ]
    )
    candidates = [
        candidate
        for candidate in _run(
            features=features,
            sectors=[_sector(), other_sector],
            leaders=leaders,
            config=config,
        )
        if candidate.template == Template.A
    ]
    assert [candidate.ts_code for candidate in candidates] == ["000001.SZ"]


def test_template_b_per_industry_cap():
    config = _config(template_b={"max_per_industry": 1})
    features = pl.DataFrame(
        [
            {"ts_code": "000001.SZ", **_COMMON, **_B_OVERRIDES},
            {"ts_code": "000002.SZ", **_COMMON, **_B_OVERRIDES},
        ]
    )
    leaders = pl.DataFrame(
        [
            _leader("000001.SZ", score=90.0, rank=1),
            _leader("000002.SZ", score=80.0, rank=2),
        ]
    )
    candidates = [
        candidate
        for candidate in _run(features=features, leaders=leaders, config=config)
        if candidate.template == Template.B
    ]
    assert [candidate.ts_code for candidate in candidates] == ["000001.SZ"]


def test_template_b_global_candidate_cap():
    config = _config(template_b={"max_candidates": 1})
    other_sector = _sector().model_copy(
        update={"sector": "医药", "rank": 2, "score": 80.0}
    )
    features = pl.DataFrame(
        [
            {"ts_code": "000001.SZ", **_COMMON, **_B_OVERRIDES},
            {
                "ts_code": "000002.SZ",
                **_COMMON,
                **_B_OVERRIDES,
                "sector": "医药",
            },
        ]
    )
    leaders = pl.DataFrame(
        [
            _leader("000001.SZ", score=90.0, rank=1),
            _leader("000002.SZ", sector="医药", score=80.0, rank=1),
        ]
    )
    candidates = [
        candidate
        for candidate in _run(
            features=features,
            sectors=[_sector(), other_sector],
            leaders=leaders,
            config=config,
        )
        if candidate.template == Template.B
    ]
    assert [candidate.ts_code for candidate in candidates] == ["000001.SZ"]


@pytest.mark.parametrize("flag", ["limit_list_d", "industry_stock_basic"])
def test_template_b_degraded_flag_lowers_score_and_adds_reason(flag: str):
    features = _b_feature()
    leaders = pl.DataFrame([_leader("000001.SZ", score=85.0, rank=2)])
    base = [
        candidate
        for candidate in _run(features=features, leaders=leaders)
        if candidate.template == Template.B
    ]
    degraded = [
        candidate
        for candidate in _run(
            regime=_regime(degraded_flags=[flag]),
            features=features,
            leaders=leaders,
        )
        if candidate.template == Template.B
    ]
    assert len(base) == len(degraded) == 1
    assert degraded[0].score < base[0].score
    assert degraded[0].score == pytest.approx(base[0].score - 5.0)
    assert "B 模板数据降级" in degraded[0].reasons
    assert "B 模板数据降级" not in base[0].reasons


def _tied_a_codes(order: list[str]) -> list[str]:
    features = pl.DataFrame([{"ts_code": code, **_COMMON} for code in order])
    leaders = pl.DataFrame(
        [
            _leader(code, score=90.0, rank=index + 1)
            for index, code in enumerate(order)
        ]
    )
    return [
        candidate.ts_code
        for candidate in _run(features=features, leaders=leaders)
        if candidate.template == Template.A
    ]


def _tied_b_codes(order: list[str]) -> list[str]:
    features = pl.DataFrame(
        [{"ts_code": code, **_COMMON, **_B_OVERRIDES} for code in order]
    )
    leaders = pl.DataFrame(
        [
            _leader(code, score=90.0, rank=index + 1)
            for index, code in enumerate(order)
        ]
    )
    return [
        candidate.ts_code
        for candidate in _run(features=features, leaders=leaders)
        if candidate.template == Template.B
    ]


def test_template_a_tie_breaks_by_ts_code():
    assert _tied_a_codes(["000001.SZ", "000002.SZ"]) == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert _tied_a_codes(["000002.SZ", "000001.SZ"]) == [
        "000001.SZ",
        "000002.SZ",
    ]


def test_template_b_tie_breaks_by_ts_code():
    assert _tied_b_codes(["000001.SZ", "000002.SZ"]) == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert _tied_b_codes(["000002.SZ", "000001.SZ"]) == [
        "000001.SZ",
        "000002.SZ",
    ]
