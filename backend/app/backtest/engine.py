"""Daily replay engine sharing the live strategy/planning pipeline.

The pipeline is intentionally invoked once per trading day so the backtest
decision path is the same Python code as the live path.  This is accurate but
not cheap for multi-year ranges; the web job runs it in the background and
reports progress.  B2 intraday is not simulated here because daily bars cannot
reconstruct 5-minute structure; use the minute replay for that sub-mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel

from app.backtest.simulator import Simulator
from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig, load_strategy_config
from app.core.errors import DataQualityError
from app.core.full_config import FullSystemConfig, load_full_config
from app.data.interfaces import MarketDataProvider
from app.pipeline_v2 import StrategyPipeline


class BacktestTrade(BaseModel):
    ts_code: str
    strategy: str
    sub_mode: str | None = None
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    fees: float
    r_multiple: float
    pnl: float
    exit_reason: str


class EquityPoint(BaseModel):
    trade_date: date
    equity: float
    cash: float


class BacktestResult(BaseModel):
    start: date
    end: date
    initial_equity: float
    final_equity: float
    total_return: float
    annualized_return: float = 0.0
    max_drawdown: float
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    win_rate: float
    avg_r: float
    expectancy: float = 0.0
    profit_factor: float = 0.0
    turnover: float = 0.0
    fill_rate: float = 0.0
    avg_holding_days: float = 0.0
    capital_utilization: float = 0.0
    benchmark_return: float | None = None
    trades: list[BacktestTrade]
    equity_curve: list[EquityPoint]
    by_strategy: dict
    by_sub_mode: dict = {}
    missing_data: list[str]
    is_demo: bool
    notes: list[str] = []


@dataclass
class BacktestEngine:
    provider: MarketDataProvider
    config: FullSystemConfig = field(default_factory=load_full_config)
    legacy_config: StrategyConfig = field(default_factory=load_strategy_config)
    pipeline: StrategyPipeline | None = None
    code_version: str = "dev"

    def __post_init__(self) -> None:
        self.pipeline = self.pipeline or StrategyPipeline(
            provider=self.provider,
            config=self.config,
            legacy_config=self.legacy_config,
            code_version=self.code_version,
        )

    def run(
        self,
        start: date,
        end: date,
        *,
        initial_equity: float | None = None,
        strategy_mode: str = "all",
        events_by_code: dict[str, list[dict]] | None = None,
        max_days: int | None = None,
        is_demo: bool = False,
    ) -> BacktestResult:
        equity = initial_equity or self.config.backtest.initial_equity
        calendar = TradingCalendar.from_trade_cal(
            self.provider.load_trade_cal(start, end)
        )
        open_dates = [
            item for item in calendar.dates if start <= item <= end
        ]
        if max_days is not None:
            open_dates = open_dates[:max_days]
        simulator = Simulator(self.config, equity, equity, equity)
        notes: list[str] = [
            "daily OHLC replay; same-day stop and target resolves to stop",
            "unfilled orders remain pending until expires_on",
            "entry and exit fees are included in trade PnL and reconciliation",
            "B2 intraday is not simulated from daily bars",
            "corporate-action-adjusted PnL and delisting lifecycle are limited "
            "when the historical data does not cover them; performance is not "
            "presented as complete in those cases",
        ]
        for day in open_dates:
            bars = self.provider.load_bars(day, day)
            simulator.process_day(day, bars)
            account = simulator.account_snapshot(day)
            try:
                result = self.pipeline.run(  # type: ignore[union-attr]
                    day,
                    account,
                    strategy_mode=strategy_mode,
                    events_by_code=events_by_code,
                    minute_health=False,
                    data_version=f"bt_{day.isoformat()}",
                )
            except DataQualityError as exc:
                simulator.missing_data.append(f"{day.isoformat()}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                simulator.missing_data.append(f"{day.isoformat()}: {exc}")
                continue
            simulator.add_plans(result.plans, day)
        metrics = simulator.metrics()
        return BacktestResult(
            start=start,
            end=end,
            initial_equity=metrics["initial_equity"],
            final_equity=metrics["final_equity"],
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            max_drawdown=metrics["max_drawdown"],
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            calmar=metrics["calmar"],
            win_rate=metrics["win_rate"],
            avg_r=metrics["avg_r"],
            expectancy=metrics["expectancy"],
            profit_factor=metrics["profit_factor"],
            turnover=metrics["turnover"],
            fill_rate=metrics["fill_rate"],
            avg_holding_days=metrics["avg_holding_days"],
            capital_utilization=metrics["capital_utilization"],
            benchmark_return=self._benchmark_return(start, end),
            trades=[
                BacktestTrade(
                    ts_code=trade.ts_code,
                    strategy=trade.strategy,
                    sub_mode=trade.sub_mode,
                    entry_date=trade.entry_date,
                    exit_date=trade.exit_date,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    shares=trade.shares,
                    fees=trade.fees,
                    r_multiple=trade.r_multiple,
                    pnl=trade.pnl,
                    exit_reason=trade.exit_reason,
                )
                for trade in simulator.trades
            ],
            equity_curve=[
                EquityPoint(
                    trade_date=date.fromisoformat(point["trade_date"]),
                    equity=point["equity"],
                    cash=point["cash"],
                )
                for point in simulator.equity_curve
            ],
            by_strategy=metrics["by_strategy"],
            by_sub_mode=metrics["by_sub_mode"],
            missing_data=metrics["missing_data"],
            is_demo=is_demo,
            notes=notes,
        )

    def _benchmark_return(self, start: date, end: date) -> float | None:
        for code in ("000985.CSI", "000300.SH"):
            try:
                frame = self.provider.load_index_bars(code, start, end)
            except Exception:  # noqa: BLE001 - benchmark is optional
                continue
            if frame.is_empty() or "close" not in frame.columns:
                continue
            closes = [
                float(value)
                for value in frame.sort("trade_date")["close"].to_list()
                if value is not None
            ]
            if len(closes) >= 2 and closes[0] > 0:
                return closes[-1] / closes[0] - 1
        return None

    def walk_forward(
        self,
        start: date,
        end: date,
        *,
        folds: int = 3,
        **kwargs,
    ) -> list[dict]:
        """Run disjoint forward windows and return per-fold metrics.

        The strategy parameters are fixed, so this is a stability check rather
        than parameter optimisation; each fold is still an out-of-sample
        period relative to the preceding fold.
        """
        if folds < 1:
            raise ValueError("folds must be >= 1")
        calendar = TradingCalendar.from_trade_cal(
            self.provider.load_trade_cal(start, end)
        )
        dates = [item for item in calendar.dates if start <= item <= end]
        if not dates:
            return []
        size = max(1, len(dates) // folds)
        results: list[dict] = []
        for index in range(folds):
            fold_start = dates[index * size]
            fold_end = (
                dates[(index + 1) * size - 1]
                if index < folds - 1
                else dates[-1]
            )
            result = self.run(fold_start, fold_end, **kwargs)
            payload = result.model_dump(mode="json")
            results.append(
                {
                    "start": fold_start.isoformat(),
                    "end": fold_end.isoformat(),
                    "final_equity": result.final_equity,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "sharpe": result.sharpe,
                    "trades": payload["trades"],
                }
            )
        return results

    def ablation(
        self,
        start: date,
        end: date,
        *,
        strategies: tuple[str, ...] = ("A", "B", "C", "D", "E", "F"),
        **kwargs,
    ) -> dict[str, dict]:
        """Compare the combined run with each single-strategy run."""
        combined = self.run(start, end, strategy_mode="all", **kwargs)
        result = {
            "all": {
                "total_return": combined.total_return,
                "max_drawdown": combined.max_drawdown,
                "trades": len(combined.trades),
            }
        }
        for strategy in strategies:
            single = self.run(start, end, strategy_mode=strategy, **kwargs)
            result[strategy] = {
                "total_return": single.total_return,
                "max_drawdown": single.max_drawdown,
                "trades": len(single.trades),
            }
        return result

    def cost_sensitivity(
        self,
        start: date,
        end: date,
        *,
        multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
        **kwargs,
    ) -> list[dict]:
        """Re-run with scaled commission/slippage to expose cost fragility."""
        base = self.config
        results: list[dict] = []
        for multiplier in multipliers:
            costs = base.costs.model_copy(
                update={
                    "commission_rate": base.costs.commission_rate * multiplier,
                    "stamp_duty_rate": base.costs.stamp_duty_rate * multiplier,
                    "transfer_fee_rate": base.costs.transfer_fee_rate * multiplier,
                    "slippage_a": base.costs.slippage_a * multiplier,
                    "slippage_b": base.costs.slippage_b * multiplier,
                }
            )
            config = base.model_copy(update={"costs": costs})
            engine = BacktestEngine(
                provider=self.provider,
                config=config,
                legacy_config=self.legacy_config,
                code_version=self.code_version,
            )
            result = engine.run(start, end, **kwargs)
            results.append(
                {
                    "multiplier": multiplier,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "trades": len(result.trades),
                }
            )
        return results
