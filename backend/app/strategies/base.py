from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Protocol
from zoneinfo import ZoneInfo

import polars as pl

from app.core.full_config import FullSystemConfig, PermissionRuleConfig
from app.domain.enums import (
    AccountMode,
    PermissionMode,
    Strategy,
    SubMode,
    Template,
)
from app.domain.models import Candidate, MarketRegime, SectorStrength

SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass
class StrategyContext:
    as_of: date
    bars: pl.DataFrame
    features: pl.DataFrame
    sectors: list[SectorStrength]
    market: MarketRegime
    config: FullSystemConfig
    data_version: str
    stocks: pl.DataFrame | None = None
    leaders: pl.DataFrame | None = None
    events: dict[str, list[dict]] = field(default_factory=dict)
    events_available: bool = True
    minute_bars: dict[str, pl.DataFrame] = field(default_factory=dict)
    account_mode: AccountMode = AccountMode.PAPER

    def __post_init__(self) -> None:
        history = self.bars.filter(pl.col("trade_date") <= self.as_of).sort(
            ["ts_code", "trade_date"]
        )
        self._history: dict[str, pl.DataFrame] = {}
        if not history.is_empty():
            for key, frame in history.partition_by("ts_code", as_dict=True).items():
                code = key[0] if isinstance(key, tuple) else key
                self._history[str(code)] = frame
        self._feature_rows = {
            str(row["ts_code"]): row
            for row in self.features.iter_rows(named=True)
        }
        self._sector_by_name = {sector.sector: sector for sector in self.sectors}
        self._leader_rows = (
            {str(row["ts_code"]): row for row in self.leaders.iter_rows(named=True)}
            if self.leaders is not None and not self.leaders.is_empty()
            else {}
        )

    @property
    def codes(self) -> list[str]:
        return list(self._feature_rows)

    def feature_row(self, code: str) -> dict | None:
        return self._feature_rows.get(code)

    def history(self, code: str) -> pl.DataFrame | None:
        return self._history.get(code)

    def sector(self, name: str) -> SectorStrength | None:
        return self._sector_by_name.get(name)

    def leader_row(self, code: str) -> dict | None:
        return self._leader_rows.get(code)

    def events_for(self, code: str) -> list[dict]:
        return self.events.get(code, [])

    def permission(self, strategy: Strategy) -> tuple[PermissionMode, float, list[str]]:
        perm = self.market.strategy_permissions.get(strategy)
        if perm is not None:
            return perm.mode, perm.risk_factor, list(perm.reasons)
        rule = self.rule(strategy)
        if not rule.enabled:
            return PermissionMode.PAUSED, 0.0, ["strategy disabled"]
        if self.market.global_risk.value in ("reduce", "clear"):
            return PermissionMode.PAUSED, 0.0, ["global risk switch"]
        return PermissionMode.NORMAL, rule.risk_factor_normal, []

    def rule(self, strategy: Strategy) -> PermissionRuleConfig:
        return self.config.market.permissions[strategy]

    def data_time(self) -> datetime:
        return datetime.combine(self.as_of, time(15, 0), tzinfo=SHANGHAI)

    def event_data_incomplete(self) -> bool:
        return not self.events_available

    def trading_days_since(self, day: date) -> int:
        if "trade_date" not in self.bars.columns:
            return 10**6
        dates = sorted(
            value
            for value in self.bars["trade_date"].unique().to_list()
            if value is not None and value <= self.as_of
        )
        return sum(1 for value in dates if value > day)


class StrategyProtocol(Protocol):
    strategy: Strategy

    def generate(self, context: StrategyContext) -> list[Candidate]: ...


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def float_features(row: dict, extras: dict | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, bool):
            values[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and value is not None:
            values[key] = float(value)
    if extras:
        for key, value in extras.items():
            if isinstance(value, bool):
                values[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)) and value is not None:
                values[key] = float(value)
    return values


def make_candidate(
    context: StrategyContext,
    *,
    ts_code: str,
    strategy: Strategy,
    sub_mode: SubMode,
    setup_type: str,
    horizon: str,
    score: float,
    reasons: list[str],
    evidence: dict | None = None,
    feature_extras: dict | None = None,
) -> Candidate | None:
    row = context.feature_row(ts_code)
    if row is None:
        return None
    leader = context.leader_row(ts_code) or {}
    leader_score = float(leader.get("leader_score") or 0.0)
    leader_rank = int(leader.get("leader_rank") or 1)
    mode, _risk_factor, permission_reasons = context.permission(strategy)
    evidence_payload = dict(evidence or {})
    evidence_payload.setdefault("permission_reasons", "; ".join(permission_reasons))
    if not leader:
        evidence_payload["leader_unavailable"] = True
    sector = str(row.get("sector") or "未知")
    return Candidate(
        ts_code=ts_code,
        trade_date=context.as_of,
        template=Template.B if strategy == Strategy.B else Template.A,
        setup_type=setup_type,
        horizon=horizon,  # type: ignore[arg-type]
        sector=sector,
        leader_rank=max(1, leader_rank),
        leader_score=max(0.0, min(100.0, leader_score)),
        score=clamp_score(score),
        reasons=reasons or [setup_type],
        features=float_features(row, feature_extras),
        sentiment_stage=context.market.sentiment_stage,
        strategy_version=context.config.version,
        data_version=context.data_version,
        strategy=strategy,
        sub_mode=sub_mode,
        permission=mode,
        evidence=evidence_payload,
        data_time=context.data_time(),
    )


def is_limit_up_at(history: pl.DataFrame, index: int) -> bool:
    if index < 0 or index >= history.height:
        return False
    row = history.row(index, named=True)
    if "is_limit_up" in row and row["is_limit_up"] is not None:
        return bool(row["is_limit_up"])
    close = row.get("close")
    up_limit = row.get("up_limit")
    return (
        close is not None
        and up_limit is not None
        and float(close) >= float(up_limit) - 0.001
    )


def board_height_at(history: pl.DataFrame, index: int) -> int:
    height = 0
    cursor = index
    while cursor >= 0 and is_limit_up_at(history, cursor):
        height += 1
        cursor -= 1
    return height


def prior_high(history: pl.DataFrame, index: int) -> float | None:
    if index - 1 < 0:
        return None
    value = history.row(index - 1, named=True).get("high")
    return None if value is None else float(value)


def top_third(row: dict) -> bool:
    high = row.get("high")
    low = row.get("low")
    close = row.get("close")
    if high is None or low is None or close is None:
        return False
    span = float(high) - float(low)
    if span <= 0:
        return False
    return float(close) >= float(low) + (2 / 3) * span


def board_allowed(context: StrategyContext, row: dict) -> bool:
    boards = context.config.universe.allowed_boards
    market = row.get("market")
    return market is None or str(market) in boards


def base_pool(
    context: StrategyContext,
    row: dict,
    *,
    strategy: Strategy,
    min_amount: float,
    cap_min: float,
    cap_max: float,
    price_min: float,
    price_max: float,
    exclude_limit_up: bool = False,
) -> bool:
    if not board_allowed(context, row):
        return False
    if row.get("is_st") or row.get("is_suspended"):
        return False
    close = row.get("close")
    amount = row.get("amount_20d_avg")
    cap = row.get("circ_mv")
    if close is None or amount is None or cap is None:
        return False
    if not (price_min <= float(close) <= price_max):
        return False
    if float(amount) < min_amount:
        return False
    if not (cap_min <= float(cap) <= cap_max):
        return False
    if exclude_limit_up and bool(row.get("is_limit_up")):
        return False
    mode, _factor, _reasons = context.permission(strategy)
    return mode != PermissionMode.PAUSED


def sentiment_ok(context: StrategyContext, strategy: Strategy) -> bool:
    mode, _factor, _reasons = context.permission(strategy)
    return mode != PermissionMode.PAUSED
