from __future__ import annotations

from app.screener.base import ScreenerStrategy

_REGISTRY: dict[str, ScreenerStrategy] = {}


def register(strategy: ScreenerStrategy) -> None:
    _REGISTRY[strategy.name] = strategy


def get_strategy(name: str) -> ScreenerStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"screener strategy not registered: {name}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)
