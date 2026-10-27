from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig
from app.domain.enums import EntryType, Template
from app.domain.models import Candidate, EntryPlan, MarketRegime
from app.domain.money import price
from app.planning.errors import EntryInvalidError


def _invalid(candidate: Candidate, detail: str) -> EntryInvalidError:
    return EntryInvalidError(
        f"{candidate.ts_code} ({candidate.template.value}): {detail}"
    )


class EntryPlanner:
    """Build deterministic T+1 entry plans from signal-day close features.

    ``features`` carries the raw-price fields emitted by the screener
    (``ma20_raw``, ``ma5_raw``, ``ma10_raw``, ``close``) plus
    ``board_height``; hfq indicator columns are not used because orders are
    placed in raw prices.  For Template B the screener guarantees limit-up,
    non-one-word names with a board height of 1 or 2 (``is_limit_up``,
    ``~one_word``); the planner still re-validates board height so arbitrary
    callers cannot bypass that invariant.

    ``as_of`` is the signal day and must equal ``candidate.trade_date``.  Plans
    are produced after that close, so the earliest execution is T+1.

    When a full entry zone exists, ``EntryPlan.entry_price`` is its midpoint
    and is authoritative for risk, stop, and sizing calculations;
    ``trigger_price`` only describes the order trigger price.

    A missing ``close`` skips the Template A over-extension guard.  This
    fallback exists only for callers that omit the field (the mandated plan
    test does); screener-produced candidates always include ``close``.

    All zone and trigger arithmetic is done in :class:`~decimal.Decimal` before
    ``price()`` so prices round half-up exactly once, on decimal values.
    """

    def plan(
        self,
        candidate: Candidate,
        features: dict[str, float],
        regime: MarketRegime,
        calendar: TradingCalendar,
        as_of: date,
        config: StrategyConfig,
    ) -> EntryPlan:
        # ``regime`` stays in the contract for the pipeline, but entry prices
        # depend only on the signal-day features and template configuration.
        if candidate.template == Template.A:
            return self._plan_a(candidate, features, calendar, as_of, config)
        return self._plan_b(candidate, features, calendar, as_of, config)

    def _plan_a(
        self,
        candidate: Candidate,
        features: dict[str, float],
        calendar: TradingCalendar,
        as_of: date,
        config: StrategyConfig,
    ) -> EntryPlan:
        template = config.template_a
        center = features.get("ma20_raw")
        if center is None or not math.isfinite(center) or center <= 0:
            raise _invalid(candidate, "MA20 unavailable for A entry")
        center_decimal = Decimal(str(center))
        reference = features.get("close")
        if reference is not None:
            if not math.isfinite(reference) or reference <= 0:
                raise _invalid(
                    candidate,
                    f"A entry close must be finite and positive, got {reference}",
                )
            extension_cap = center_decimal * (
                Decimal("1") + Decimal(str(template.max_ma20_extension))
            )
            if Decimal(str(reference)) > extension_cap:
                raise _invalid(
                    candidate,
                    f"A entry close {reference} exceeds MA20 extension cap "
                    f"{extension_cap:.2f}",
                )
        valid_days = template.entry_valid_days
        if valid_days <= 0:
            raise _invalid(
                candidate,
                f"A entry validity must be positive, got {valid_days}",
            )
        zone_pct = Decimal(str(template.ma_zone_pct))
        low = price(center_decimal * (Decimal("1") - zone_pct))
        high = price(center_decimal * (Decimal("1") + zone_pct))
        if low <= 0 or low > high:
            raise _invalid(candidate, "A entry zone is invalid")
        expires_on = self._expiry(calendar, as_of, valid_days, candidate)
        return EntryPlan(
            type=EntryType.PULLBACK,
            trigger_price=price(center_decimal),
            entry_zone_low=low,
            entry_zone_high=high,
            validity_days=valid_days,
            max_chase_pct=template.max_gap_pct,
            cancel_conditions=[
                "close below entry zone low",
                "close below MA20",
                "gap below zone low by more than 3%",
            ],
            expires_on=expires_on,
        )

    def _plan_b(
        self,
        candidate: Candidate,
        features: dict[str, float],
        calendar: TradingCalendar,
        as_of: date,
        config: StrategyConfig,
    ) -> EntryPlan:
        template = config.template_b
        ma5 = features.get("ma5_raw")
        ma10 = features.get("ma10_raw")
        if (
            ma5 is None
            or ma10 is None
            or not math.isfinite(ma5)
            or not math.isfinite(ma10)
        ):
            raise _invalid(candidate, "MA5/MA10 unavailable for B entry")
        if ma5 <= 0 or ma10 <= 0:
            raise _invalid(candidate, "MA5/MA10 must be positive for B entry")
        ma5_decimal = Decimal(str(ma5))
        ma10_decimal = Decimal(str(ma10))
        board_height_value = features.get("board_height")
        if board_height_value is None or not math.isfinite(board_height_value):
            raise _invalid(candidate, "board height unavailable for B entry")
        board_height = int(board_height_value)
        if board_height_value != board_height or board_height not in (1, 2):
            raise _invalid(
                candidate,
                f"board height must be 1 or 2 for B entry, "
                f"got {board_height_value}",
            )
        valid_days = template.entry_valid_days
        if valid_days <= 0:
            raise _invalid(
                candidate,
                f"B entry validity must be positive, got {valid_days}",
            )
        if board_height == 1:
            low = price(ma10_decimal * Decimal("0.98"))
            high = price(ma5_decimal * Decimal("1.02"))
            if low <= 0 or high <= 0 or low >= high:
                raise _invalid(candidate, "B divergence zone is invalid")
            entry_type = EntryType.DIVERGENCE_LOW
            trigger = None
            zone_low, zone_high = low, high
        else:
            trigger = price(ma5_decimal)
            zone_low = price(ma5_decimal * Decimal("0.99"))
            zone_high = price(ma5_decimal * Decimal("1.01"))
            if trigger <= 0 or zone_low <= 0 or zone_low > zone_high:
                raise _invalid(candidate, "B confirmation zone is invalid")
            entry_type = EntryType.CONFIRMATION
        expires_on = self._expiry(calendar, as_of, valid_days, candidate)
        return EntryPlan(
            type=entry_type,
            trigger_price=trigger,
            entry_zone_low=zone_low,
            entry_zone_high=zone_high,
            validity_days=valid_days,
            max_chase_pct=template.max_gap_pct,
            cancel_conditions=[
                "close below stop",
                "close below MA5/MA10 without recovery",
                "sentiment stage turns ebb",
            ],
            expires_on=expires_on,
        )

    @staticmethod
    def _expiry(
        calendar: TradingCalendar,
        as_of: date,
        valid_days: int,
        candidate: Candidate,
    ) -> date:
        try:
            return calendar.add_trading_days(as_of, valid_days)
        except ValueError as exc:
            raise _invalid(
                candidate,
                f"cannot schedule entry expiry: {exc}",
            ) from exc
