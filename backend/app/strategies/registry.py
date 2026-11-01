from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.enums import PermissionMode, Strategy
from app.domain.models import Candidate
from app.strategies import a_trend, b_limit_up, c_platform, d_oversold, e_range, f_event
from app.strategies.base import StrategyContext


@dataclass(frozen=True)
class StrategyDefinition:
    strategy: Strategy
    module: object
    priority: int


STRATEGIES: dict[Strategy, StrategyDefinition] = {
    Strategy.A: StrategyDefinition(Strategy.A, a_trend, 10),
    Strategy.C: StrategyDefinition(Strategy.C, c_platform, 20),
    Strategy.E: StrategyDefinition(Strategy.E, e_range, 30),
    Strategy.F: StrategyDefinition(Strategy.F, f_event, 40),
    Strategy.D: StrategyDefinition(Strategy.D, d_oversold, 50),
    Strategy.B: StrategyDefinition(Strategy.B, b_limit_up, 60),
}


@dataclass
class ScreenerBundle:
    by_strategy: dict[Strategy, list[Candidate]] = field(default_factory=dict)
    paused: dict[Strategy, list[str]] = field(default_factory=dict)

    @property
    def all_candidates(self) -> list[Candidate]:
        result: list[Candidate] = []
        for strategy in sorted(self.by_strategy, key=lambda item: STRATEGIES[item].priority):
            result.extend(self.by_strategy[strategy])
        return result


class MultiStrategyScreener:
    """Run every enabled strategy independently.

    Scores are intentionally kept within each strategy; they must not be
    compared across strategy families or treated as win probabilities.
    """

    def generate(
        self,
        context: StrategyContext,
        *,
        strategies: list[Strategy] | None = None,
        max_per_industry: int = 2,
    ) -> ScreenerBundle:
        selected = strategies or list(STRATEGIES)
        bundle = ScreenerBundle()
        for strategy in selected:
            definition = STRATEGIES.get(strategy)
            if definition is None:
                continue
            mode, _risk, reasons = context.permission(strategy)
            if mode == PermissionMode.PAUSED:
                bundle.paused[strategy] = reasons or ["paused"]
                bundle.by_strategy[strategy] = []
                continue
            generated = definition.module.generate(context)  # type: ignore[attr-defined]
            filtered = [
                candidate
                for candidate in generated
                if candidate.permission != PermissionMode.PAUSED
            ]
            # Cap per industry within a strategy; keep score order and ts_code
            # tie-break so runs are reproducible.
            filtered.sort(key=lambda item: (-item.score, item.ts_code))
            per_industry: dict[str, int] = {}
            kept: list[Candidate] = []
            for candidate in filtered:
                count = per_industry.get(candidate.sector, 0)
                if count >= max_per_industry:
                    continue
                per_industry[candidate.sector] = count + 1
                kept.append(candidate)
            bundle.by_strategy[strategy] = kept
        return bundle
