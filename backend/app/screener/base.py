from __future__ import annotations

from datetime import date
from typing import Protocol

import polars as pl

from app.core.config import StrategyConfig
from app.domain.models import Candidate, MarketRegime, SectorStrength


class ScreenerStrategy(Protocol):
    name: str
    version: str

    def generate(
        self,
        as_of: date,
        regime: MarketRegime,
        features: pl.DataFrame,
        sectors: list[SectorStrength],
        leaders: pl.DataFrame,
        config: StrategyConfig,
    ) -> list[Candidate]: ...
