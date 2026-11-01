from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.core.full_config import CostsConfig
from app.domain.money import money


@dataclass(frozen=True)
class FeeBreakdown:
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    total: Decimal


def calculate_fees(
    *,
    side: str,
    price: float | Decimal,
    shares: int,
    costs: CostsConfig,
) -> FeeBreakdown:
    gross = Decimal(str(price)) * int(shares)
    commission = max(
        gross * Decimal(str(costs.commission_rate)),
        Decimal(str(costs.commission_min)),
    )
    stamp = (
        gross * Decimal(str(costs.stamp_duty_rate))
        if side == "sell"
        else Decimal("0")
    )
    transfer = gross * Decimal(str(costs.transfer_fee_rate))
    return FeeBreakdown(
        commission=money(commission),
        stamp_duty=money(stamp),
        transfer_fee=money(transfer),
        total=money(commission + stamp + transfer),
    )
