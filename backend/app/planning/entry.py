from __future__ import annotations

import math
from datetime import date

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig
from app.domain.enums import EntryType, Template
from app.domain.models import Candidate, EntryPlan, MarketRegime
from app.domain.money import price
from app.planning.errors import EntryInvalidError


class EntryPlanner:
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
            return self._plan_a(features, calendar, as_of, config)
        return self._plan_b(features, calendar, as_of, config)

    def _plan_a(
        self,
        features: dict[str, float],
        calendar: TradingCalendar,
        as_of: date,
        config: StrategyConfig,
    ) -> EntryPlan:
        template = config.template_a
        center = features.get("ma20_raw")
        if center is None or not math.isfinite(center) or center <= 0:
            raise EntryInvalidError("MA20 unavailable for A entry")
        reference = features.get("close")
        # The design rejects A entries when the reference price is more than
        # max_ma20_extension above MA20.  The symmetric zone midpoint is MA20
        # itself, so the guard applies to the signal-day close behind the plan.
        extension_cap = center * (1 + template.max_ma20_extension)
        if reference is not None and (
            not math.isfinite(reference) or reference > extension_cap
        ):
            raise EntryInvalidError(
                f"A entry reference {reference} exceeds MA20 extension cap "
                f"{extension_cap:.2f}"
            )
        valid_days = template.entry_valid_days
        if valid_days <= 0:
            raise EntryInvalidError(
                f"A entry validity must be positive, got {valid_days}"
            )
        low = price(center * (1 - template.ma_zone_pct))
        high = price(center * (1 + template.ma_zone_pct))
        if low <= 0 or low > high:
            raise EntryInvalidError("A entry zone is invalid")
        expires_on = self._expiry(calendar, as_of, valid_days, "A")
        return EntryPlan(
            type=EntryType.PULLBACK,
            trigger_price=price(center),
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
            raise EntryInvalidError("MA5/MA10 unavailable for B entry")
        if ma5 <= 0 or ma10 <= 0:
            raise EntryInvalidError("MA5/MA10 must be positive for B entry")
        board_height_value = features.get("board_height")
        if (
            board_height_value is None
            or not math.isfinite(board_height_value)
            or board_height_value <= 0
        ):
            raise EntryInvalidError("board height unavailable for B entry")
        board_height = int(board_height_value)
        valid_days = template.entry_valid_days
        if valid_days <= 0:
            raise EntryInvalidError(
                f"B entry validity must be positive, got {valid_days}"
            )
        if board_height <= 1:
            low = price(ma10 * 0.98)
            high = price(ma5 * 1.02)
            if low <= 0 or high <= 0 or low >= high:
                raise EntryInvalidError("B divergence zone is invalid")
            entry_type = EntryType.DIVERGENCE_LOW
            trigger = None
            zone_low, zone_high = low, high
        else:
            trigger = price(ma5)
            zone_low = price(ma5 * 0.99)
            zone_high = price(ma5 * 1.01)
            if trigger <= 0 or zone_low <= 0 or zone_low > zone_high:
                raise EntryInvalidError("B confirmation zone is invalid")
            entry_type = EntryType.CONFIRMATION
        expires_on = self._expiry(calendar, as_of, valid_days, "B")
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
        template_name: str,
    ) -> date:
        try:
            return calendar.add_trading_days(as_of, valid_days)
        except ValueError as exc:
            raise EntryInvalidError(
                f"cannot schedule {template_name} entry expiry: {exc}"
            ) from exc
