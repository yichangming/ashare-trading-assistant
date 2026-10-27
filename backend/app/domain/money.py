from decimal import ROUND_HALF_UP, Decimal

PRICE_QUANT = Decimal("0.01")
MONEY_QUANT = Decimal("0.01")


def price(value: float | int | str | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(PRICE_QUANT, rounding=ROUND_HALF_UP)


def money(value: float | int | str | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
