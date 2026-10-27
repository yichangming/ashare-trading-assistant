from __future__ import annotations

import pytest

from app.screener.dual_template import DualTemplateScreener
from app.screener.registry import get_strategy, list_strategies, register


def test_registry_register_and_get():
    strategy = DualTemplateScreener()
    register(strategy)
    assert get_strategy(strategy.name) is strategy
    assert strategy.name in list_strategies()


def test_registry_unknown_name_raises_key_error():
    with pytest.raises(KeyError, match="not_registered"):
        get_strategy("not_registered")


def test_list_strategies_is_sorted_and_unique():
    register(DualTemplateScreener())
    names = list_strategies()
    assert names == sorted(names)
    assert len(names) == len(set(names))
