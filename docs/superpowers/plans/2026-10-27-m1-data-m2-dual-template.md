# M1 数据底座 + M2 双模板选股管线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python backend that syncs A-share data from Tushare into Parquet/DuckDB, validates it, computes market regime/sentiment/industry/leader context, and runs the five-module dual-template (A/B) planning pipeline from a CLI.

**Architecture:** Layered backend: data adapters → domain models → regime/sector/leader services → screener → entry/stop/sizing/take-profit planners → risk governor → CLI. All decision logic is deterministic pure functions with injected context; Tushare and Parquet are behind protocols so tests run offline.

**Tech Stack:** Python 3.11, uv, Pydantic v2, Polars, DuckDB, PyArrow, Tushare, Typer, pytest, Hypothesis, ruff.

**Scope:** M1 data foundation + M2 environment/five modules/dual templates. Not included: backtest engine (M3), scheduler/reporting (M4), web dashboard (M5), broker adapters (M6). The CLI is the acceptance surface for this plan.

**Milestone boundary:** Tasks 0–6 produce a working data CLI (`sync`, `assemble`, `quality`). Tasks 7–18 produce a working planning CLI (`regime`, `sectors`, `run`) and end-to-end tests.

---

## File Structure

```text
stock/
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── __init__.py
│   │   ├── cli.py                      # Task 18
│   │   ├── pipeline.py                 # Task 17
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── calendar.py             # Task 5
│   │   │   ├── config.py               # Task 2
│   │   │   ├── dates.py                # Task 5
│   │   │   └── errors.py               # Task 5
│   │   ├── domain/
│   │   │   ├── __init__.py
│   │   │   ├── enums.py                # Task 1
│   │   │   ├── money.py                # Task 1
│   │   │   └── models.py               # Task 1
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py           # Task 3
│   │   │   ├── tushare_client.py       # Task 3
│   │   │   ├── store.py                # Task 3
│   │   │   ├── provider.py             # Task 3
│   │   │   ├── manifest.py             # Task 4
│   │   │   ├── sync.py                 # Task 4
│   │   │   ├── assemble.py             # Task 5
│   │   │   ├── quality.py              # Task 6
│   │   │   ├── industry_map.py         # Task 9
│   │   │   └── universe.py             # Task 7
│   │   ├── indicators/
│   │   │   ├── __init__.py
│   │   │   └── features.py             # Task 7
│   │   ├── regime/
│   │   │   ├── __init__.py
│   │   │   ├── sentiment.py            # Task 8
│   │   │   ├── market.py               # Task 8
│   │   │   ├── industry.py             # Task 9
│   │   │   └── leader.py               # Task 10
│   │   ├── screener/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # Task 11
│   │   │   ├── registry.py             # Task 11
│   │   │   └── dual_template.py        # Task 11
│   │   ├── planning/
│   │   │   ├── __init__.py
│   │   │   ├── errors.py               # Task 12
│   │   │   ├── entry.py                # Task 12
│   │   │   ├── stop.py                 # Task 13
│   │   │   ├── sizing.py               # Task 14
│   │   │   └── take_profit.py          # Task 15
│   │   └── risk/
│   │       ├── __init__.py
│   │       └── governor.py             # Task 16
│   ├── configs/
│   │   ├── app.yaml                    # Task 2
│   │   └── strategies/dual_template_v1.yaml  # Task 2
│   └── tests/
│       ├── conftest.py
│       ├── fixtures/
│       │   ├── fake_tushare.py
│       │   └── synthetic.py
│       ├── test_domain.py
│       ├── test_config.py
│       ├── test_store.py
│       ├── test_sync.py
│       ├── test_assemble.py
│       ├── test_quality.py
│       ├── test_features.py
│       ├── test_universe.py
│       ├── test_sentiment.py
│       ├── test_market_regime.py
│       ├── test_industry.py
│       ├── test_leader.py
│       ├── test_screener.py
│       ├── test_entry.py
│       ├── test_stop.py
│       ├── test_sizing.py
│       ├── test_take_profit.py
│       ├── test_risk.py
│       ├── test_registry.py
│       └── test_pipeline.py
├── configs/                            # optional symlink/copy for root-level runs
├── data/                               # gitignored
├── docs/
└── README.md
```

**Conventions**

- All commands run from `backend/`.
- Use `uv run ...` for Python; never rely on a globally installed package.
- Tests are fully offline. Inject fake Tushare clients/providers; use `tmp_path`.
- Prices and money use `Decimal`; features/indicators use `float`.
- Trading dates are `datetime.date`; data source strings are `YYYYMMDD`.
- Commit after every task with the exact command shown.

---

### Task 0: Backend scaffold and tooling

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/core/__init__.py`, `backend/app/domain/__init__.py`, `backend/app/data/__init__.py`, `backend/app/indicators/__init__.py`, `backend/app/regime/__init__.py`, `backend/app/screener/__init__.py`, `backend/app/planning/__init__.py`, `backend/app/risk/__init__.py`
- Create: `backend/tests/test_smoke.py`
- Create: `.gitignore`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_smoke.py`:

```python
import app


def test_version():
    assert app.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend
uv run pytest tests/test_smoke.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app'` or missing `__version__`.

- [ ] **Step 3: Create `pyproject.toml`**

`backend/pyproject.toml`:

```toml
[project]
name = "ashare-trading-assistant"
version = "0.1.0"
description = "A-share personal trading assistant: data foundation and dual-template planning pipeline"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7,<3",
  "pydantic-settings>=2.3,<3",
  "polars>=0.20,<2",
  "pyarrow>=16,<20",
  "duckdb>=1.0,<2",
  "tushare>=1.4,<2",
  "httpx>=0.27,<1",
  "pyyaml>=6,<7",
  "typer>=0.12,<1",
  "rich>=13,<14",
]

[dependency-groups]
dev = [
  "pytest>=8,<9",
  "pytest-cov>=5,<7",
  "hypothesis>=6,<7",
  "ruff>=0.5,<1",
  "mypy>=1.10,<2",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 4: Create package files and `.gitignore`**

`backend/app/__init__.py`:

```python
"""A-share trading assistant backend."""

__version__ = "0.1.0"
```

All other package `__init__.py` files are empty.

Repository root `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.mypy_cache/
data/
reports/
.env
*.db
```

- [ ] **Step 5: Sync dependencies and run the test**

Run:

```bash
cd backend
uv sync --dev
uv run pytest tests/test_smoke.py -v
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/app backend/tests/test_smoke.py .gitignore
git commit -m "chore: scaffold backend with uv and pytest"
```

---

### Task 1: Domain enums, money helpers, and Pydantic models

**Files:**
- Create: `backend/app/domain/enums.py`
- Create: `backend/app/domain/money.py`
- Create: `backend/app/domain/models.py`
- Test: `backend/tests/test_domain.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_domain.py`:

```python
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.enums import PlanStatus, SentimentStage, Template, TrendRegime
from app.domain.models import (
    AccountState,
    Candidate,
    EntryPlan,
    MarketRegime,
    PendingExposure,
    PositionState,
    RegimeIndicators,
    TemplatePolicy,
)
from app.domain.money import money, price


def test_price_rounds_half_up():
    assert price("10.005") == Decimal("10.01")
    assert price(9.994) == Decimal("9.99")
    assert money("1234.567") == Decimal("1234.57")


def test_candidate_requires_reasons():
    with pytest.raises(ValidationError):
        Candidate(
            ts_code="000001.SZ",
            trade_date=date(2024, 1, 2),
            template=Template.A,
            setup_type="mainline_pullback",
            horizon="mid",
            sector="银行",
            leader_rank=1,
            leader_score=90.0,
            score=88.0,
            reasons=[],
            features={"close": 10.0},
            sentiment_stage=SentimentStage.FERMENT,
            strategy_version="sv_test",
            data_version="dv_test",
        )


def test_market_regime_requires_both_templates():
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    with pytest.raises(ValidationError):
        MarketRegime(
            trade_date=date(2024, 1, 2),
            trend_regime=TrendRegime.ATTACK,
            sentiment_stage=SentimentStage.FERMENT,
            risk_factor=1.0,
            template_availability={Template.A: policy},
            indicators=RegimeIndicators(
                limit_up_count=80,
                limit_down_count=0,
                max_board_height=5,
                promotion_rate=0.5,
                blow_rate=0.1,
                red_rate=0.6,
                limit_up_premium=0.02,
                breadth=0.6,
                index_close=4000.0,
                index_ma20=3900.0,
                index_ma60=3800.0,
            ),
            data_version="dv_test",
        )


def test_entry_plan_zone_midpoint():
    plan = EntryPlan(
        type="pullback",
        entry_zone_low=Decimal("9.80"),
        entry_zone_high=Decimal("10.20"),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=["close < MA20"],
        expires_on=date(2024, 1, 5),
    )
    assert plan.entry_price == Decimal("10.00")


def test_account_heat_includes_pending():
    account = AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("500000"),
        peak_equity=Decimal("1000000"),
        positions=[
            PositionState(
                ts_code="000001.SZ",
                sector="银行",
                shares=10000,
                available_shares=10000,
                cost_price=Decimal("10.00"),
                current_stop=Decimal("9.00"),
                current_price=Decimal("11.00"),
                template=Template.A,
            )
        ],
        pending_plans=[
            PendingExposure(
                ts_code="000002.SZ",
                sector="银行",
                entry_price=Decimal("20.00"),
                stop_price=Decimal("18.00"),
                shares=5000,
                template=Template.A,
            )
        ],
    )
    assert account.heat == pytest.approx((2.0 * 10000 + 2.0 * 5000) / 1000000)


def test_plan_status_defaults():
    assert PlanStatus.DRAFT.value == "DRAFT"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
uv run pytest tests/test_domain.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.domain.enums'`.

- [ ] **Step 3: Implement enums and money helpers**

`backend/app/domain/enums.py`:

```python
from enum import Enum


class Template(str, Enum):
    A = "A"
    B = "B"


class TrendRegime(str, Enum):
    ATTACK = "attack"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"


class SentimentStage(str, Enum):
    FREEZE = "freeze"
    START = "start"
    FERMENT = "ferment"
    CLIMAX = "climax"
    EBB = "ebb"


class PlanStatus(str, Enum):
    DRAFT = "DRAFT"
    REJECTED = "REJECTED"
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class EntryType(str, Enum):
    PULLBACK = "pullback"
    DIVERGENCE_LOW = "divergence_low"
    CONFIRMATION = "confirmation"


class RiskOutcome(str, Enum):
    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"
```

`backend/app/domain/money.py`:

```python
from decimal import Decimal, ROUND_HALF_UP

PRICE_QUANT = Decimal("0.01")
MONEY_QUANT = Decimal("0.01")


def price(value: float | int | str | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(PRICE_QUANT, rounding=ROUND_HALF_UP)


def money(value: float | int | str | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
```

- [ ] **Step 4: Implement domain models**

`backend/app/domain/models.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import (
    EntryType,
    PlanStatus,
    RiskOutcome,
    SentimentStage,
    Template,
    TrendRegime,
)
from app.domain.money import price


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemplatePolicy(DomainModel):
    enabled: bool
    mode: Literal["normal", "pullback_only", "observe"] = "normal"
    max_positions: int = Field(default=0, ge=0)


class RegimeIndicators(DomainModel):
    limit_up_count: int = Field(ge=0)
    limit_down_count: int = Field(ge=0)
    max_board_height: int = Field(ge=0)
    promotion_rate: float = Field(ge=0, le=1)
    blow_rate: float = Field(ge=0, le=1)
    red_rate: float = Field(ge=0, le=1)
    limit_up_premium: float
    breadth: float = Field(ge=0, le=1)
    index_close: float = Field(gt=0)
    index_ma20: float = Field(gt=0)
    index_ma60: float = Field(gt=0)


class SectorStrength(DomainModel):
    sector: str
    score: float = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    return_20d: float
    return_5d: float
    breadth: float = Field(ge=0, le=1)
    amount_change: float
    limit_up_share: float = Field(ge=0, le=1)
    limit_up_count: int = Field(ge=0)
    red_rate: float = Field(ge=0, le=1)
    stock_count: int = Field(ge=0)
    persistent_days: int = Field(ge=0)


class MarketRegime(DomainModel):
    trade_date: date
    trend_regime: TrendRegime
    sentiment_stage: SentimentStage
    risk_factor: float = Field(ge=0, le=1)
    template_availability: dict[Template, TemplatePolicy]
    indicators: RegimeIndicators
    sector_ranking: list[SectorStrength] = Field(default_factory=list)
    degraded_flags: list[str] = Field(default_factory=list)
    data_version: str

    @model_validator(mode="after")
    def _both_templates(self) -> "MarketRegime":
        if set(self.template_availability) != {Template.A, Template.B}:
            raise ValueError("template_availability must contain A and B")
        return self


class Candidate(DomainModel):
    ts_code: str
    trade_date: date
    template: Template
    setup_type: str
    horizon: Literal["short", "mid"]
    sector: str
    leader_rank: int = Field(ge=1)
    leader_score: float = Field(ge=0, le=100)
    score: float = Field(ge=0, le=100)
    reasons: list[str] = Field(min_length=1)
    features: dict[str, float]
    sentiment_stage: SentimentStage
    strategy_version: str
    data_version: str


class EntryPlan(DomainModel):
    type: EntryType
    trigger_price: Decimal | None = None
    entry_zone_low: Decimal | None = None
    entry_zone_high: Decimal | None = None
    validity_days: int = Field(gt=0)
    max_chase_pct: float = Field(ge=0)
    cancel_conditions: list[str] = Field(default_factory=list)
    expires_on: date

    @model_validator(mode="after")
    def _prices_present(self) -> "EntryPlan":
        if self.trigger_price is None and (
            self.entry_zone_low is None or self.entry_zone_high is None
        ):
            raise ValueError("trigger_price or a full entry zone is required")
        if self.entry_zone_low is not None and self.entry_zone_high is not None:
            if self.entry_zone_low > self.entry_zone_high:
                raise ValueError("entry_zone_low must be <= entry_zone_high")
        return self

    @property
    def entry_price(self) -> Decimal:
        if self.entry_zone_low is not None and self.entry_zone_high is not None:
            return price((self.entry_zone_low + self.entry_zone_high) / 2)
        assert self.trigger_price is not None
        return self.trigger_price


class StopPlan(DomainModel):
    initial_stop: Decimal
    stop_distance_pct: float = Field(gt=0)
    atr14: float = Field(gt=0)
    structure_ref: str
    time_stop_days: int = Field(gt=0)
    trailing_rules: list[str]
    r_value: Decimal = Field(gt=0)


class PositionPlan(DomainModel):
    template: Template
    risk_pct: float = Field(gt=0, le=1)
    risk_amount: Decimal = Field(ge=0)
    shares: int = Field(ge=0)
    position_value: Decimal = Field(ge=0)
    position_pct: float = Field(ge=0, le=1)
    heat_before: float = Field(ge=0)
    heat_after: float = Field(ge=0)
    constraints_checked: list[str] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list)


class TakeProfitTarget(DomainModel):
    r_multiple: float = Field(gt=0)
    price: Decimal = Field(gt=0)
    fraction: float = Field(gt=0, le=1)


class TakeProfitPlan(DomainModel):
    targets: list[TakeProfitTarget] = Field(min_length=1)
    trailing_rule: str
    time_exit_rule: str


class RiskDecision(DomainModel):
    outcome: RiskOutcome
    reasons: list[str] = Field(min_length=1)
    adjusted_shares: int | None = Field(default=None, ge=0)


class PendingExposure(DomainModel):
    ts_code: str
    sector: str
    entry_price: Decimal = Field(gt=0)
    stop_price: Decimal = Field(gt=0)
    shares: int = Field(ge=0)
    template: Template


class PositionState(DomainModel):
    ts_code: str
    sector: str
    shares: int = Field(ge=0)
    available_shares: int = Field(ge=0)
    cost_price: Decimal = Field(gt=0)
    current_stop: Decimal = Field(gt=0)
    current_price: Decimal = Field(gt=0)
    template: Template


class AccountState(DomainModel):
    equity: Decimal = Field(gt=0)
    cash: Decimal = Field(ge=0)
    peak_equity: Decimal = Field(gt=0)
    positions: list[PositionState] = Field(default_factory=list)
    pending_plans: list[PendingExposure] = Field(default_factory=list)

    @property
    def drawdown_pct(self) -> float:
        return max(0.0, 1.0 - float(self.equity / self.peak_equity))

    @property
    def positions_value(self) -> Decimal:
        return sum(
            (position.current_price * position.shares for position in self.positions),
            Decimal("0"),
        )

    @property
    def heat(self) -> float:
        equity = float(self.equity)
        if equity <= 0:
            return 0.0
        open_risk = sum(
            max(0.0, float(position.current_price - position.current_stop))
            * position.shares
            for position in self.positions
        )
        pending_risk = sum(
            max(0.0, float(exposure.entry_price - exposure.stop_price))
            * exposure.shares
            for exposure in self.pending_plans
        )
        return (open_risk + pending_risk) / equity

    def with_pending(self, exposure: PendingExposure) -> "AccountState":
        return self.model_copy(
            update={"pending_plans": [*self.pending_plans, exposure]}
        )


class TradePlan(DomainModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    candidate: Candidate
    entry_plan: EntryPlan
    stop_plan: StopPlan
    position_plan: PositionPlan
    take_profit_plan: TakeProfitPlan
    risk_decision: RiskDecision | None = None
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_on: date
    strategy_version: str
    data_version: str
    code_version: str
    reasons: list[str] = Field(default_factory=list)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd backend
uv run pytest tests/test_domain.py -v
```

Expected: `6 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/domain backend/tests/test_domain.py
git commit -m "feat: add domain models and money helpers"
```

---

### Task 2: Configuration models and dual-template YAML

**Files:**
- Create: `backend/app/core/config.py`
- Create: `backend/configs/app.yaml`
- Create: `backend/configs/strategies/dual_template_v1.yaml`
- Test: `backend/tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from app.core.config import load_app_config, load_strategy_config


def test_load_strategy_config():
    config = load_strategy_config()
    assert config.name == "dual_template_v1"
    assert config.template_a.risk_pct == 0.025
    assert config.template_b.risk_pct == 0.02
    assert config.switching.ferment.b.max_positions == 2
    assert config.risk.max_positions == 3
    assert config.environment.stage_confirm_days == 2


def test_load_app_config():
    config = load_app_config()
    assert config.data.provider == "tushare"
    assert config.strategy_path.endswith("dual_template_v1.yaml")


def test_invalid_industry_weights_rejected(tmp_path):
    base = load_strategy_config()
    payload = base.model_dump()
    payload["industry"]["weights"]["return_20d"] = 0.9
    path = tmp_path / "bad.yaml"
    import yaml

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_strategy_config(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
uv run pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.config'`.

- [ ] **Step 3: Implement config models**

`backend/app/core/config.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FreezeThresholds(StrictModel):
    limit_up_max: int
    limit_down_min: int
    blow_rate_min: float
    max_board_max: int
    red_rate_max: float


class StartThresholds(StrictModel):
    limit_up_min: int
    limit_up_max: int
    max_board_min: int
    max_board_max: int
    blow_rate_min: float
    blow_rate_max: float
    red_rate_min: float
    red_rate_max: float


class FermentThresholds(StrictModel):
    limit_up_min: int
    max_board_min: int
    blow_rate_max: float
    red_rate_min: float


class ClimaxThresholds(StrictModel):
    limit_up_min: int
    max_board_min: int
    red_rate_min: float
    blow_rate_rise_pp: float
    premium_drop_pp: float


class EbbThresholds(StrictModel):
    limit_up_max: int
    limit_down_min: int
    blow_rate_min: float
    red_rate_max: float


class EnvironmentThresholds(StrictModel):
    freeze: FreezeThresholds
    start: StartThresholds
    ferment: FermentThresholds
    climax: ClimaxThresholds
    ebb: EbbThresholds


class EnvironmentConfig(StrictModel):
    index_code: str
    fallback_index_code: str
    ma_fast: int = 20
    ma_slow: int = 60
    breadth_attack: float = 0.55
    breadth_defense: float = 0.35
    stage_confirm_days: int = 2
    thresholds: EnvironmentThresholds


class IndustryWeights(StrictModel):
    return_20d: float
    return_5d: float
    breadth: float
    amount_change: float
    limit_up_share: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> "IndustryWeights":
        total = (
            self.return_20d
            + self.return_5d
            + self.breadth
            + self.amount_change
            + self.limit_up_share
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"industry weights must sum to 1.0, got {total}")
        return self


class IndustryConfig(StrictModel):
    top_n: int = 3
    persistence_days: int = 2
    min_stocks: int = 5
    source: Literal["auto", "sw", "stock_basic"] = "auto"
    weights: IndustryWeights


class LeaderWeights(StrictModel):
    rs_20d: float
    rs_5d: float
    amount_turnover: float
    high_distance: float
    limit_up_status: float
    size_price_fit: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> "LeaderWeights":
        total = (
            self.rs_20d
            + self.rs_5d
            + self.amount_turnover
            + self.high_distance
            + self.limit_up_status
            + self.size_price_fit
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"leader weights must sum to 1.0, got {total}")
        return self


class LeaderConfig(StrictModel):
    top_n_per_industry: int = 2
    follower_rank_cutoff: int = 3
    amount_vs_leader_min: float = 0.5
    weights: LeaderWeights


class ScoreWeights(StrictModel):
    leader: float
    setup: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> "ScoreWeights":
        if abs(self.leader + self.setup - 1.0) > 1e-6:
            raise ValueError("score weights must sum to 1.0")
        return self


class TakeProfitAConfig(StrictModel):
    r1: float = 1.0
    r2: float = 2.0
    r3: float = 3.0
    first_fraction: float = 1 / 3
    second_fraction: float = 1 / 3


class TakeProfitBConfig(StrictModel):
    r1: float = 1.0
    r2: float = 2.0
    second_fraction: float = 0.5


class TemplateAConfig(StrictModel):
    max_candidates: int = 15
    max_per_industry: int = 2
    allowed_boards: list[str] = ["主板", "创业板"]
    universe_min_price: float = 3.0
    min_listing_days: int = 120
    min_avg_amount: float = 100_000_000
    min_price: float = 5.0
    max_price: float = 100.0
    market_cap_min: float = 5_000_000_000
    market_cap_max: float = 50_000_000_000
    max_distance_from_60d_high: float = 0.10
    rs_20d_market_min: float = 0.70
    pullback_days_min: int = 3
    pullback_days_max: int = 8
    volume_shrink_ratio: float = 1.0
    ma_zone_pct: float = 0.02
    max_ma20_extension: float = 0.08
    entry_valid_days: int = 3
    max_gap_pct: float = 0.03
    stop_min_pct: float = 0.03
    stop_max_pct: float = 0.08
    atr_buffer: float = 0.5
    time_stop_days: int = 10
    risk_pct: float = 0.025
    take_profit: TakeProfitAConfig
    score: ScoreWeights


class TemplateBConfig(StrictModel):
    max_candidates: int = 8
    max_per_industry: int = 2
    min_limit_up_in_sector: int = 3
    sector_red_rate_min: float = 0.50
    blow_rate_max: float = 0.35
    max_board_min: int = 3
    volume_ratio_min: float = 2.0
    turnover_min: float = 0.05
    turnover_max: float = 0.20
    market_cap_min: float = 5_000_000_000
    market_cap_max: float = 20_000_000_000
    price_min: float = 5.0
    price_max: float = 50.0
    max_board: int = 4
    max_20d_gain: float = 0.80
    entry_valid_days: int = 2
    max_gap_pct: float = 0.03
    stop_min_pct: float = 0.03
    stop_max_pct: float = 0.05
    time_stop_days: int = 3
    risk_pct: float = 0.02
    take_profit: TakeProfitBConfig
    score: ScoreWeights


class RiskConfig(StrictModel):
    max_positions: int = 3
    max_b_positions: int = 2
    max_single_position_pct: float = 0.40
    min_cash_pct: float = 0.10
    max_heat: float = 0.075
    max_industry_positions: int = 2
    correlation_threshold: float = 0.80
    drawdown_half_at: float = 0.10
    drawdown_stop_at: float = 0.15
    round_trip_cost_pct: float = 0.003
    commission_rate: float = 0.00025
    commission_min: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_rate: float = 0.001
    b_slippage_rate: float = 0.002


class TemplatePolicyConfig(StrictModel):
    enabled: bool
    mode: Literal["normal", "pullback_only", "observe"] = "normal"
    max_positions: int = 0


class StagePolicyConfig(StrictModel):
    a: TemplatePolicyConfig
    b: TemplatePolicyConfig
    risk_factor: float


class SwitchingConfig(StrictModel):
    freeze: StagePolicyConfig
    start: StagePolicyConfig
    ferment: StagePolicyConfig
    climax: StagePolicyConfig
    ebb: StagePolicyConfig


class StrategyConfig(StrictModel):
    name: str
    version: str
    environment: EnvironmentConfig
    industry: IndustryConfig
    leader: LeaderConfig
    template_a: TemplateAConfig
    template_b: TemplateBConfig
    switching: SwitchingConfig
    risk: RiskConfig


class DataConfig(StrictModel):
    provider: str = "tushare"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    regime_dir: str = "data/regime"
    history_start: str = "2018-01-01"
    rate_limit_per_min: int = 180


class AppConfig(StrictModel):
    data: DataConfig
    strategy_path: str = "configs/strategies/dual_template_v1.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    return payload


def load_app_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else BACKEND_ROOT / "configs" / "app.yaml"
    return AppConfig.model_validate(_load_yaml(config_path))


def load_strategy_config(path: Path | str | None = None) -> StrategyConfig:
    config_path = (
        Path(path)
        if path is not None
        else BACKEND_ROOT / "configs" / "strategies" / "dual_template_v1.yaml"
    )
    return StrategyConfig.model_validate(_load_yaml(config_path))
```

- [ ] **Step 4: Create the YAML files**

`backend/configs/app.yaml`:

```yaml
data:
  provider: tushare
  raw_dir: data/raw
  processed_dir: data/processed
  regime_dir: data/regime
  history_start: "2018-01-01"
  rate_limit_per_min: 180
strategy_path: configs/strategies/dual_template_v1.yaml
```

`backend/configs/strategies/dual_template_v1.yaml`:

```yaml
name: dual_template_v1
version: "1.1"
environment:
  index_code: "000985.CSI"
  fallback_index_code: "000300.SH"
  ma_fast: 20
  ma_slow: 60
  breadth_attack: 0.55
  breadth_defense: 0.35
  stage_confirm_days: 2
  thresholds:
    freeze:
      limit_up_max: 30
      limit_down_min: 20
      blow_rate_min: 0.40
      max_board_max: 2
      red_rate_max: 0.30
    start:
      limit_up_min: 30
      limit_up_max: 60
      max_board_min: 3
      max_board_max: 4
      blow_rate_min: 0.20
      blow_rate_max: 0.35
      red_rate_min: 0.40
      red_rate_max: 0.55
    ferment:
      limit_up_min: 60
      max_board_min: 5
      blow_rate_max: 0.25
      red_rate_min: 0.55
    climax:
      limit_up_min: 90
      max_board_min: 6
      red_rate_min: 0.70
      blow_rate_rise_pp: 5.0
      premium_drop_pp: 1.0
    ebb:
      limit_up_max: 40
      limit_down_min: 30
      blow_rate_min: 0.45
      red_rate_max: 0.30
industry:
  top_n: 3
  persistence_days: 2
  min_stocks: 5
  source: auto
  weights:
    return_20d: 0.30
    return_5d: 0.20
    breadth: 0.20
    amount_change: 0.15
    limit_up_share: 0.15
leader:
  top_n_per_industry: 2
  follower_rank_cutoff: 3
  amount_vs_leader_min: 0.5
  weights:
    rs_20d: 0.25
    rs_5d: 0.15
    amount_turnover: 0.20
    high_distance: 0.15
    limit_up_status: 0.15
    size_price_fit: 0.10
template_a:
  max_candidates: 15
  max_per_industry: 2
  allowed_boards: ["主板", "创业板"]
  universe_min_price: 3.0
  min_listing_days: 120
  min_avg_amount: 100000000
  min_price: 5.0
  max_price: 100.0
  market_cap_min: 5000000000
  market_cap_max: 50000000000
  max_distance_from_60d_high: 0.10
  rs_20d_market_min: 0.70
  pullback_days_min: 3
  pullback_days_max: 8
  volume_shrink_ratio: 1.0
  ma_zone_pct: 0.02
  max_ma20_extension: 0.08
  entry_valid_days: 3
  max_gap_pct: 0.03
  stop_min_pct: 0.03
  stop_max_pct: 0.08
  atr_buffer: 0.5
  time_stop_days: 10
  risk_pct: 0.025
  take_profit:
    r1: 1.0
    r2: 2.0
    r3: 3.0
    first_fraction: 0.333333
    second_fraction: 0.333333
  score:
    leader: 0.70
    setup: 0.30
template_b:
  max_candidates: 8
  max_per_industry: 2
  min_limit_up_in_sector: 3
  sector_red_rate_min: 0.50
  blow_rate_max: 0.35
  max_board_min: 3
  volume_ratio_min: 2.0
  turnover_min: 0.05
  turnover_max: 0.20
  market_cap_min: 5000000000
  market_cap_max: 20000000000
  price_min: 5.0
  price_max: 50.0
  max_board: 4
  max_20d_gain: 0.80
  entry_valid_days: 2
  max_gap_pct: 0.03
  stop_min_pct: 0.03
  stop_max_pct: 0.05
  time_stop_days: 3
  risk_pct: 0.02
  take_profit:
    r1: 1.0
    r2: 2.0
    second_fraction: 0.5
  score:
    leader: 0.70
    setup: 0.30
switching:
  freeze:
    a: {enabled: false, mode: observe, max_positions: 0}
    b: {enabled: false, mode: observe, max_positions: 0}
    risk_factor: 0.0
  start:
    a: {enabled: true, mode: normal, max_positions: 3}
    b: {enabled: true, mode: normal, max_positions: 1}
    risk_factor: 0.5
  ferment:
    a: {enabled: true, mode: normal, max_positions: 3}
    b: {enabled: true, mode: normal, max_positions: 2}
    risk_factor: 1.0
  climax:
    a: {enabled: true, mode: pullback_only, max_positions: 3}
    b: {enabled: false, mode: observe, max_positions: 0}
    risk_factor: 0.5
  ebb:
    a: {enabled: false, mode: observe, max_positions: 0}
    b: {enabled: false, mode: observe, max_positions: 0}
    risk_factor: 0.0
risk:
  max_positions: 3
  max_b_positions: 2
  max_single_position_pct: 0.40
  min_cash_pct: 0.10
  max_heat: 0.075
  max_industry_positions: 2
  correlation_threshold: 0.80
  drawdown_half_at: 0.10
  drawdown_stop_at: 0.15
  round_trip_cost_pct: 0.003
  commission_rate: 0.00025
  commission_min: 5.0
  stamp_duty_rate: 0.0005
  transfer_fee_rate: 0.00001
  slippage_rate: 0.001
  b_slippage_rate: 0.002
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd backend
uv run pytest tests/test_config.py -v
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/core/config.py backend/configs backend/tests/test_config.py
git commit -m "feat: add strict strategy configuration"
```

---

### Task 3: Parquet store, Tushare client, and data provider

**Files:**
- Create: `backend/app/data/interfaces.py`
- Create: `backend/app/data/tushare_client.py`
- Create: `backend/app/data/store.py`
- Create: `backend/app/data/provider.py`
- Test: `backend/tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_store.py`:

```python
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.data.provider import ParquetMarketDataProvider
from app.data.store import ParquetStore
from app.data.tushare_client import TushareClient


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "raw", tmp_path / "processed", tmp_path / "regime")


def test_processed_roundtrip(store: ParquetStore):
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "close": [10.0, 20.0],
        }
    )
    store.write_processed("bars", date(2024, 1, 2), df)
    loaded = store.read_processed("bars", date(2024, 1, 2), date(2024, 1, 2))
    assert loaded.sort("ts_code").to_dicts() == df.sort("ts_code").to_dicts()


def test_reference_roundtrip(store: ParquetStore):
    df = pl.DataFrame({"ts_code": ["000001.SZ"], "market": ["主板"]})
    store.write_reference("stock_basic", df)
    loaded = store.read_reference("stock_basic")
    assert loaded.to_dicts() == df.to_dicts()


def test_tushare_client_retries():
    class FlakyPro:
        def __init__(self):
            self.calls = 0

        def daily(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary")
            return pl.DataFrame({"ts_code": ["000001.SZ"]}).to_pandas()

    pro = FlakyPro()
    client = TushareClient(pro=pro, sleep=lambda _: None, max_retries=2)
    result = client.fetch("daily", trade_date="20240102")
    assert pro.calls == 2
    assert result.height == 1


def test_provider_filters_stock_basic(store: ParquetStore):
    store.write_reference(
        "stock_basic",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "list_date": [date(2000, 1, 1), date(2024, 12, 1), date(2000, 1, 1)],
                "delist_date": [None, None, date(2023, 1, 1)],
            }
        ),
    )
    provider = ParquetMarketDataProvider(store)
    loaded = provider.load_stock_basic(date(2024, 1, 2))
    assert loaded["ts_code"].to_list() == ["000001.SZ"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_store.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement the store**

`backend/app/data/store.py`:

```python
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

_SAFE = re.compile(r"[^a-zA-Z0-9_\-]")


def _safe(name: str) -> str:
    return _SAFE.sub("_", name)


class ParquetStore:
    def __init__(
        self,
        raw_dir: Path | str,
        processed_dir: Path | str,
        regime_dir: Path | str,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.regime_dir = Path(regime_dir)
        for directory in (self.raw_dir, self.processed_dir, self.regime_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def write_raw(self, endpoint: str, key: str, df: pl.DataFrame) -> Path:
        path = self.raw_dir / _safe(endpoint) / f"{_safe(key)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def read_raw(self, endpoint: str, key: str) -> pl.DataFrame:
        path = self.raw_dir / _safe(endpoint) / f"{_safe(key)}.parquet"
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def write_reference(self, name: str, df: pl.DataFrame) -> Path:
        path = self.raw_dir / "reference" / f"{_safe(name)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def read_reference(self, name: str) -> pl.DataFrame:
        path = self.raw_dir / "reference" / f"{_safe(name)}.parquet"
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def write_processed(self, dataset: str, trade_date: date, df: pl.DataFrame) -> Path:
        path = (
            self.processed_dir
            / _safe(dataset)
            / f"trade_date={trade_date.isoformat()}"
            / "data.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def read_processed(self, dataset: str, start: date, end: date) -> pl.DataFrame:
        root = self.processed_dir / _safe(dataset)
        if not any(root.glob("trade_date=*/data.parquet")):
            return pl.DataFrame()
        pattern = str(root / "trade_date=*" / "data.parquet").replace("'", "''")
        connection = duckdb.connect()
        try:
            return connection.execute(
                f"SELECT * FROM read_parquet('{pattern}') "
                "WHERE trade_date BETWEEN ? AND ?",
                [start, end],
            ).pl()
        finally:
            connection.close()

    def write_regime(self, trade_date: date, payload: dict) -> Path:
        path = self.regime_dir / f"trade_date={trade_date.isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
        return path
```

- [ ] **Step 4: Implement the Tushare client and provider**

`backend/app/data/interfaces.py`:

```python
from datetime import date
from typing import Protocol

import polars as pl


class DataSource(Protocol):
    def fetch(self, endpoint: str, **params) -> pl.DataFrame: ...


class MarketDataProvider(Protocol):
    def load_bars(self, start: date, end: date) -> pl.DataFrame: ...
    def load_index_bars(
        self, index_code: str, start: date, end: date
    ) -> pl.DataFrame: ...
    def load_stock_basic(self, as_of: date) -> pl.DataFrame: ...
    def load_namechange(self) -> pl.DataFrame: ...
    def load_trade_cal(self, start: date, end: date) -> pl.DataFrame: ...
    def load_index_member(self) -> pl.DataFrame: ...
```

`backend/app/data/tushare_client.py`:

```python
from __future__ import annotations

import time

import polars as pl


class TushareClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        pro=None,
        rate_limit_per_min: int = 180,
        max_retries: int = 3,
        sleep=time.sleep,
    ):
        if pro is None:
            import tushare as ts

            if not token:
                raise ValueError("Tushare token is required when pro is not injected")
            pro = ts.pro_api(token)
        self._pro = pro
        self._min_interval = 60.0 / max(rate_limit_per_min, 1)
        self._last_call = 0.0
        self._max_retries = max_retries
        self._sleep = sleep

    def fetch(self, endpoint: str, **params) -> pl.DataFrame:
        method = getattr(self._pro, endpoint)
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                frame = method(**params)
                if frame is None:
                    return pl.DataFrame()
                if isinstance(frame, pl.DataFrame):
                    return frame
                return pl.from_pandas(frame)
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                self._sleep(2**attempt)
        raise RuntimeError("unreachable")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            self._sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()
```

`backend/app/data/provider.py`:

```python
from __future__ import annotations

from datetime import date

import polars as pl

from app.data.store import ParquetStore


class ParquetMarketDataProvider:
    def __init__(self, store: ParquetStore):
        self.store = store

    def load_bars(self, start: date, end: date) -> pl.DataFrame:
        return self.store.read_processed("bars", start, end)

    def load_index_bars(
        self, index_code: str, start: date, end: date
    ) -> pl.DataFrame:
        frame = self.store.read_processed("index_bars", start, end)
        if frame.is_empty():
            return frame
        return frame.filter(pl.col("ts_code") == index_code)

    def load_stock_basic(self, as_of: date) -> pl.DataFrame:
        frame = self.store.read_reference("stock_basic")
        if frame.is_empty():
            return frame
        list_date = pl.col("list_date").cast(pl.Date, strict=False)
        delist_date = pl.col("delist_date").cast(pl.Date, strict=False)
        return frame.filter(
            (list_date <= as_of)
            & (delist_date.is_null() | (delist_date > as_of))
        )

    def load_namechange(self) -> pl.DataFrame:
        return self.store.read_reference("namechange")

    def load_trade_cal(self, start: date, end: date) -> pl.DataFrame:
        frame = self.store.read_reference("trade_cal")
        if frame.is_empty():
            return frame
        cal_date = pl.col("cal_date").cast(pl.Date, strict=False)
        return frame.filter((cal_date >= start) & (cal_date <= end))

    def load_index_member(self) -> pl.DataFrame:
        return self.store.read_reference("index_member")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_store.py -v
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/data backend/tests/test_store.py
git commit -m "feat: add parquet store, tushare client, and data provider"
```

---

### Task 4: Tushare sync, manifests, and incremental backfill

**Files:**
- Create: `backend/app/data/manifest.py`
- Create: `backend/app/data/sync.py`
- Test: `backend/tests/fixtures/fake_tushare.py`
- Test: `backend/tests/test_sync.py`

- [ ] **Step 1: Write the fake Tushare source**

`backend/tests/fixtures/fake_tushare.py`:

```python
from __future__ import annotations

import polars as pl


class FakePro:
    def __init__(self, fail_endpoints: set[str] | None = None):
        self.fail_endpoints = fail_endpoints or set()
        self.calls: list[tuple[str, dict]] = []

    def _maybe_fail(self, endpoint: str) -> None:
        if endpoint in self.fail_endpoints:
            raise RuntimeError(f"{endpoint} permission denied")

    def trade_cal(self, **kwargs):
        self.calls.append(("trade_cal", kwargs))
        return pl.DataFrame(
            {
                "exchange": ["SSE"] * 5,
                "cal_date": ["20240101", "20240102", "20240103", "20240104", "20240105"],
                "is_open": [0, 1, 1, 1, 1],
                "pretrade_date": ["20231229", "20231229", "20240102", "20240103", "20240104"],
            }
        ).to_pandas()

    def stock_basic(self, **kwargs):
        self.calls.append(("stock_basic", kwargs))
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "symbol": ["000001", "000002"],
                "name": ["平安银行", "万科A"],
                "area": ["深圳", "深圳"],
                "industry": ["银行", "房地产"],
                "market": ["主板", "主板"],
                "list_date": ["20000407", "19910129"],
                "delist_date": [None, None],
                "is_hs": ["S", "S"],
            }
        ).to_pandas()

    def namechange(self, **kwargs):
        self.calls.append(("namechange", kwargs))
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["平安银行"],
                "start_date": ["20000407"],
                "end_date": [None],
            }
        ).to_pandas()

    def _daily_frame(self, **kwargs):
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [kwargs.get("trade_date", "20240102")] * 2,
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.8, 19.5],
                "close": [10.2, 20.2],
                "pre_close": [10.0, 20.0],
                "change": [0.2, 0.2],
                "pct_chg": [2.0, 1.0],
                "vol": [100000.0, 200000.0],
                "amount": [1020000.0, 4040000.0],
            }
        ).to_pandas()

    def daily(self, **kwargs):
        self.calls.append(("daily", kwargs))
        self._maybe_fail("daily")
        return self._daily_frame(**kwargs)

    def adj_factor(self, **kwargs):
        self.calls.append(("adj_factor", kwargs))
        self._maybe_fail("adj_factor")
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [kwargs.get("trade_date", "20240102")] * 2,
                "adj_factor": [1.0, 2.0],
            }
        ).to_pandas()

    def daily_basic(self, **kwargs):
        self.calls.append(("daily_basic", kwargs))
        self._maybe_fail("daily_basic")
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [kwargs.get("trade_date", "20240102")] * 2,
                "turnover_rate": [1.5, 2.0],
                "circ_mv": [1000000.0, 2000000.0],
            }
        ).to_pandas()

    def stk_limit(self, **kwargs):
        self.calls.append(("stk_limit", kwargs))
        self._maybe_fail("stk_limit")
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [kwargs.get("trade_date", "20240102")] * 2,
                "up_limit": [11.0, 22.0],
                "down_limit": [9.0, 18.0],
            }
        ).to_pandas()

    def suspend_d(self, **kwargs):
        self.calls.append(("suspend_d", kwargs))
        self._maybe_fail("suspend_d")
        return pl.DataFrame(
            {"ts_code": [], "trade_date": [], "suspend_type": []}
        ).to_pandas()

    def index_daily(self, **kwargs):
        self.calls.append(("index_daily", kwargs))
        return pl.DataFrame(
            {
                "ts_code": ["000300.SH"],
                "trade_date": [kwargs.get("trade_date", "20240102")],
                "open": [4000.0],
                "high": [4050.0],
                "low": [3980.0],
                "close": [4020.0],
                "vol": [1000000.0],
                "amount": [2000000.0],
            }
        ).to_pandas()

    def limit_list_d(self, **kwargs):
        self.calls.append(("limit_list_d", kwargs))
        self._maybe_fail("limit_list_d")
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [kwargs.get("trade_date", "20240102")],
                "first_time": ["09:35:00"],
                "last_time": ["14:50:00"],
                "open_times": [1],
                "limit_amount": [1000000.0],
            }
        ).to_pandas()

    def index_classify(self, **kwargs):
        self.calls.append(("index_classify", kwargs))
        return pl.DataFrame(
            {"index_code": ["801010.SI"], "industry_name": ["银行"]}
        ).to_pandas()

    def index_member(self, **kwargs):
        self.calls.append(("index_member", kwargs))
        return pl.DataFrame(
            {
                "index_code": ["801010.SI"],
                "con_code": ["000001.SZ"],
                "in_date": ["20000407"],
                "out_date": [None],
                "is_new": ["Y"],
            }
        ).to_pandas()
```

- [ ] **Step 2: Write the failing sync tests**

`backend/tests/test_sync.py`:

```python
from datetime import date
from pathlib import Path

import pytest

from app.data.manifest import DataManifest
from app.data.store import ParquetStore
from app.data.sync import SyncService
from app.data.tushare_client import TushareClient
from tests.fixtures.fake_tushare import FakePro


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "raw", tmp_path / "processed", tmp_path / "regime")


def test_manifest_version_is_stable():
    manifest = DataManifest(
        trade_date=date(2024, 1, 2),
        row_counts={"daily": 2},
        checksums={"daily": "abc"},
    )
    assert manifest.version() == manifest.version()
    assert len(manifest.version()) == 12


def test_sync_reference_normalizes_dates(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    counts = service.sync_reference(date(2024, 1, 1), date(2024, 1, 5))
    assert counts["trade_cal"] == 5
    frame = store.read_reference("trade_cal")
    assert str(frame.schema["cal_date"]) == "Date"
    stock_basic = store.read_reference("stock_basic")
    assert str(stock_basic.schema["list_date"]) == "Date"


def test_sync_day_writes_raw_and_manifest(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    result = service.sync_day(date(2024, 1, 2))
    assert result.row_counts["daily"] == 2
    assert store.read_raw("daily", "20240102").height == 2
    manifest_path = store.raw_dir / "manifests" / "20240102.json"
    assert manifest_path.exists()
    assert "data_version" in manifest_path.read_text(encoding="utf-8")


def test_optional_endpoint_failure_degrades(store: ParquetStore):
    pro = FakePro(fail_endpoints={"limit_list_d"})
    service = SyncService(TushareClient(pro=pro, sleep=lambda _: None), store)
    result = service.sync_day(date(2024, 1, 2))
    assert any("limit_list_d" in item for item in result.degraded)
    assert store.read_raw("daily", "20240102").height == 2


def test_backfill_skips_non_trading_days(store: ParquetStore):
    service = SyncService(TushareClient(pro=FakePro(), sleep=lambda _: None), store)
    results = service.backfill(date(2024, 1, 2), date(2024, 1, 3))
    assert [r.trade_date for r in results] == [date(2024, 1, 2), date(2024, 1, 3)]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_sync.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement manifest and sync**

`backend/app/data/manifest.py`:

```python
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class DataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_date: date
    row_counts: dict[str, int]
    checksums: dict[str, str]
    degraded: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def version(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at"})
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]
```

`backend/app/data/sync.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from app.data.manifest import DataManifest
from app.data.store import ParquetStore
from app.data.tushare_client import TushareClient


class SyncError(RuntimeError):
    pass


@dataclass
class SyncResult:
    trade_date: date
    row_counts: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


def _to_date(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame = frame.with_columns(
                pl.col(column)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias(column)
            )
    return frame


class SyncService:
    def __init__(self, client: TushareClient, store: ParquetStore):
        self.client = client
        self.store = store

    def sync_reference(self, start: date, end: date) -> dict[str, int]:
        counts: dict[str, int] = {}
        trade_cal = self.client.fetch(
            "trade_cal",
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        trade_cal = _to_date(trade_cal, ["cal_date", "pretrade_date"])
        self.store.write_reference("trade_cal", trade_cal)
        counts["trade_cal"] = trade_cal.height

        stock_basic = self.client.fetch(
            "stock_basic",
            exchange="",
            list_status="L,D,P",
            fields=(
                "ts_code,symbol,name,area,industry,market,"
                "list_date,delist_date,is_hs"
            ),
        )
        stock_basic = _to_date(stock_basic, ["list_date", "delist_date"])
        self.store.write_reference("stock_basic", stock_basic)
        counts["stock_basic"] = stock_basic.height

        namechange = self.client.fetch("namechange")
        namechange = _to_date(namechange, ["start_date", "end_date"])
        self.store.write_reference("namechange", namechange)
        counts["namechange"] = namechange.height

        for endpoint in ("index_classify", "index_member"):
            try:
                frame = self.client.fetch(endpoint)
                frame = _to_date(frame, ["in_date", "out_date"])
                self.store.write_reference(endpoint, frame)
                counts[endpoint] = frame.height
            except Exception:
                counts[endpoint] = 0
        return counts

    def sync_day(self, trade_date: date) -> SyncResult:
        key = trade_date.strftime("%Y%m%d")
        result = SyncResult(trade_date=trade_date)
        required = {
            "daily": {"trade_date": key},
            "adj_factor": {"trade_date": key},
            "daily_basic": {"trade_date": key},
            "stk_limit": {"trade_date": key},
            "suspend_d": {"trade_date": key},
            "index_daily": {"trade_date": key},
        }
        for endpoint, params in required.items():
            try:
                frame = self.client.fetch(endpoint, **params)
            except Exception as exc:
                raise SyncError(f"critical endpoint {endpoint} failed: {exc}") from exc
            self.store.write_raw(endpoint, key, frame)
            result.row_counts[endpoint] = frame.height

        try:
            frame = self.client.fetch("limit_list_d", trade_date=key)
            self.store.write_raw("limit_list_d", key, frame)
            result.row_counts["limit_list_d"] = frame.height
        except Exception as exc:
            result.degraded.append(f"limit_list_d unavailable: {exc}")

        self._write_manifest(result)
        return result

    def backfill(
        self, start: date, end: date, force: bool = False
    ) -> list[SyncResult]:
        self.sync_reference(start, end)
        calendar = self.store.read_reference("trade_cal")
        if calendar.is_empty():
            raise SyncError("trade_cal is empty; cannot backfill")
        open_dates = [
            item
            for item in calendar.filter(pl.col("is_open") == 1)["cal_date"].to_list()
            if start <= item <= end
        ]
        results: list[SyncResult] = []
        for trade_date in open_dates:
            key = trade_date.strftime("%Y%m%d")
            if not force and self.store.read_raw("daily", key).height > 0:
                continue
            results.append(self.sync_day(trade_date))
        return results

    def sync_incremental(
        self, as_of: date, re_pull_days: int = 5
    ) -> list[SyncResult]:
        calendar = self.store.read_reference("trade_cal")
        if calendar.is_empty():
            raise SyncError("trade_cal is empty; run sync_reference first")
        open_dates = [
            item
            for item in calendar.filter(pl.col("is_open") == 1)["cal_date"].to_list()
            if item <= as_of
        ]
        selected = open_dates[-re_pull_days:] if re_pull_days > 0 else []
        return [self.sync_day(item) for item in selected]

    def _write_manifest(self, result: SyncResult) -> None:
        checksums: dict[str, str] = {}
        for endpoint in result.row_counts:
            path = (
                self.store.raw_dir
                / endpoint
                / f"{result.trade_date.strftime('%Y%m%d')}.parquet"
            )
            if path.exists():
                checksums[endpoint] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = DataManifest(
            trade_date=result.trade_date,
            row_counts=result.row_counts,
            checksums=checksums,
            degraded=result.degraded,
        )
        payload = manifest.model_dump(mode="json")
        payload["data_version"] = manifest.version()
        output = (
            self.store.raw_dir
            / "manifests"
            / f"{result.trade_date.strftime('%Y%m%d')}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_sync.py -v
```

Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/data/manifest.py backend/app/data/sync.py backend/tests/fixtures/fake_tushare.py backend/tests/test_sync.py
git commit -m "feat: sync tushare data with manifests and degradation tracking"
```

---

### Task 5: Trading calendar and processed bars assembler

**Files:**
- Create: `backend/app/core/dates.py`
- Create: `backend/app/core/calendar.py`
- Create: `backend/app/core/errors.py`
- Create: `backend/app/data/assemble.py`
- Test: `backend/tests/test_assemble.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_assemble.py`:

```python
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.core.calendar import TradingCalendar
from app.core.dates import parse_date
from app.data.assemble import DataAssembler
from app.data.store import ParquetStore


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "raw", tmp_path / "processed", tmp_path / "regime")


def _write_raw(store: ParquetStore, day: str = "20240102") -> None:
    store.write_raw(
        "daily",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.8, 19.5],
                "close": [10.2, 20.2],
                "pre_close": [10.0, 20.0],
                "vol": [100000.0, 200000.0],
                "amount": [1020000.0, 4040000.0],
            }
        ),
    )
    store.write_raw(
        "adj_factor",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "adj_factor": [1.0, 2.0],
            }
        ),
    )
    store.write_raw(
        "daily_basic",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "turnover_rate": [1.5, 2.0],
                "circ_mv": [1000000.0, 2000000.0],
            }
        ),
    )
    store.write_raw(
        "stk_limit",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "up_limit": [11.0, 22.0],
                "down_limit": [9.0, 18.0],
            }
        ),
    )
    store.write_raw(
        "suspend_d",
        day,
        pl.DataFrame({"ts_code": [], "trade_date": [], "suspend_type": []}),
    )
    store.write_raw(
        "index_daily",
        day,
        pl.DataFrame(
            {
                "ts_code": ["000300.SH"],
                "trade_date": [day],
                "open": [4000.0],
                "high": [4050.0],
                "low": [3980.0],
                "close": [4020.0],
                "vol": [1000.0],
                "amount": [2000.0],
            }
        ),
    )
    store.write_reference(
        "stock_basic",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "name": ["平安银行", "万科A"],
                "market": ["主板", "主板"],
                "list_date": [date(2000, 4, 7), date(1991, 1, 29)],
                "delist_date": [None, None],
            }
        ),
    )


def test_parse_date():
    assert parse_date("20240102") == date(2024, 1, 2)
    assert parse_date(date(2024, 1, 2)) == date(2024, 1, 2)


def test_calendar_add_trading_days():
    calendar = TradingCalendar(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    )
    assert calendar.add_trading_days(date(2024, 1, 2), 2) == date(2024, 1, 4)
    assert calendar.count_between(date(2024, 1, 2), date(2024, 1, 3)) == 2


def test_assemble_day(store: ParquetStore):
    _write_raw(store)
    assembler = DataAssembler(store)
    bars = assembler.build_day(date(2024, 1, 2))
    assert bars.height == 2
    assert bars.filter(pl.col("ts_code") == "000002.SZ")["adj_factor"][0] == 2.0
    assert bars["turnover_rate"].to_list() == [0.015, 0.02]
    processed = store.read_processed("bars", date(2024, 1, 2), date(2024, 1, 2))
    assert processed.height == 2
    index_bars = store.read_processed(
        "index_bars", date(2024, 1, 2), date(2024, 1, 2)
    )
    assert index_bars.height == 1


def test_assemble_derives_limits_when_missing(store: ParquetStore):
    _write_raw(store)
    (store.raw_dir / "stk_limit" / "20240102.parquet").unlink()
    assembler = DataAssembler(store)
    bars = assembler.build_day(date(2024, 1, 2))
    assert bars["limit_derived"].to_list() == [True, True]
    assert bars.filter(pl.col("ts_code") == "000001.SZ")["up_limit"][0] == 11.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_assemble.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement dates, calendar, and errors**

`backend/app/core/dates.py`:

```python
from __future__ import annotations

from datetime import date, datetime


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    return date.fromisoformat(text)
```

`backend/app/core/errors.py`:

```python
class DataNotReadyError(RuntimeError):
    pass


class DataQualityError(RuntimeError):
    pass
```

`backend/app/core/calendar.py`:

```python
from __future__ import annotations

from bisect import bisect_left
from datetime import date, timedelta

import polars as pl

from app.core.dates import parse_date


class TradingCalendar:
    def __init__(self, dates: list[date]):
        self.dates = sorted(dates)
        self._index = {item: i for i, item in enumerate(self.dates)}

    def is_trading_day(self, day: date) -> bool:
        return day in self._index

    def add_trading_days(self, day: date, count: int) -> date:
        if day not in self._index:
            raise ValueError(f"{day} is not a trading day")
        target = min(self._index[day] + count, len(self.dates) - 1)
        return self.dates[target]

    def count_between(self, start: date, end: date) -> int:
        if start > end:
            return 0
        left = bisect_left(self.dates, start)
        right = bisect_left(self.dates, end + timedelta(days=1))
        return max(0, right - left)

    @classmethod
    def from_trade_cal(cls, frame: pl.DataFrame) -> "TradingCalendar":
        if frame.is_empty():
            return cls([])
        dates = [
            parse_date(item)
            for item in frame.filter(pl.col("is_open") == 1)["cal_date"].to_list()
        ]
        return cls(dates)
```

- [ ] **Step 4: Implement the assembler**

`backend/app/data/assemble.py`:

```python
from __future__ import annotations

from datetime import date

import polars as pl

from app.core.errors import DataNotReadyError
from app.data.store import ParquetStore


def _normalize_daily(frame: pl.DataFrame) -> pl.DataFrame:
    numeric = ["open", "high", "low", "close", "pre_close", "vol", "amount"]
    return frame.with_columns(
        [
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y%m%d", strict=False)
            .alias("trade_date"),
            *[pl.col(column).cast(pl.Float64) for column in numeric],
            (pl.col("amount").cast(pl.Float64) * 1000).alias("amount"),
        ]
    )


def _normalize_keyed(
    frame: pl.DataFrame, numeric: list[str], date_columns: list[str] | None = None
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    expressions = [
        pl.col("trade_date")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("trade_date"),
        *[pl.col(column).cast(pl.Float64) for column in numeric if column in frame.columns],
    ]
    for column in date_columns or []:
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias(column)
            )
    return frame.with_columns(expressions)


def _derive_limits(bars: pl.DataFrame, stocks: pl.DataFrame) -> pl.DataFrame:
    market = pl.col("market").fill_null("主板")
    is_st = pl.col("name").str.contains("ST").fill_null(False)
    base_limit_pct = (
        pl.when(market == "创业板")
        .then(0.20)
        .when(market == "科创板")
        .then(0.20)
        .when(market == "北交所")
        .then(0.30)
        .otherwise(0.10)
    )
    limit_pct = (
        pl.when(is_st & (market == "主板"))
        .then(0.05)
        .otherwise(base_limit_pct)
    )
    joined = bars.join(
        stocks.select(["ts_code", "name", "market"]),
        on="ts_code",
        how="left",
    )
    return joined.with_columns(
        limit_pct.alias("limit_pct"),
        (pl.col("pre_close") * (1 + limit_pct)).round(2).alias("up_limit"),
        (pl.col("pre_close") * (1 - limit_pct)).round(2).alias("down_limit"),
        pl.lit(True).alias("limit_derived"),
    )


class DataAssembler:
    def __init__(self, store: ParquetStore):
        self.store = store

    def build_day(self, trade_date: date) -> pl.DataFrame:
        key = trade_date.strftime("%Y%m%d")
        daily = self.store.read_raw("daily", key)
        if daily.is_empty():
            raise DataNotReadyError(f"daily data missing for {trade_date}")

        bars = _normalize_daily(daily)
        adj = self.store.read_raw("adj_factor", key)
        if not adj.is_empty():
            adj = _normalize_keyed(adj, ["adj_factor"])
            bars = bars.join(adj, on=["ts_code", "trade_date"], how="left")
        else:
            bars = bars.with_columns(pl.lit(1.0).alias("adj_factor"))
        bars = bars.with_columns(pl.col("adj_factor").fill_null(1.0))

        basic = self.store.read_raw("daily_basic", key)
        if not basic.is_empty():
            basic = _normalize_keyed(basic, ["turnover_rate", "circ_mv"])
            basic = basic.with_columns(
                (pl.col("turnover_rate") / 100).alias("turnover_rate"),
                (pl.col("circ_mv") * 10000).alias("circ_mv"),
            )
            bars = bars.join(
                basic.select(["ts_code", "trade_date", "turnover_rate", "circ_mv"]),
                on=["ts_code", "trade_date"],
                how="left",
            )
        else:
            bars = bars.with_columns(
                pl.lit(None).cast(pl.Float64).alias("turnover_rate"),
                pl.lit(None).cast(pl.Float64).alias("circ_mv"),
            )

        limits = self.store.read_raw("stk_limit", key)
        if not limits.is_empty():
            limits = _normalize_keyed(limits, ["up_limit", "down_limit"])
            bars = bars.join(
                limits.select(["ts_code", "trade_date", "up_limit", "down_limit"]),
                on=["ts_code", "trade_date"],
                how="left",
            ).with_columns(pl.lit(False).alias("limit_derived"))
        else:
            stocks = self.store.read_reference("stock_basic")
            bars = _derive_limits(bars, stocks)

        suspend = self.store.read_raw("suspend_d", key)
        if not suspend.is_empty():
            suspended_codes = suspend["ts_code"].unique().to_list()
            bars = bars.with_columns(
                pl.col("ts_code").is_in(suspended_codes).alias("is_suspended")
            )
        else:
            bars = bars.with_columns((pl.col("vol") == 0).alias("is_suspended"))

        self.store.write_processed("bars", trade_date, bars)

        index_raw = self.store.read_raw("index_daily", key)
        if not index_raw.is_empty():
            index_bars = _normalize_keyed(index_raw, ["open", "high", "low", "close", "vol", "amount"])
            self.store.write_processed("index_bars", trade_date, index_bars)

        limit_list = self.store.read_raw("limit_list_d", key)
        if not limit_list.is_empty():
            normalized = limit_list.with_columns(
                pl.col("trade_date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias("trade_date")
            )
            self.store.write_processed("limit_list", trade_date, normalized)
        return bars
```

**Unit conventions:** Tushare `daily.amount` is 千元, `daily_basic.turnover_rate` is percent, and `daily_basic.circ_mv` is 万元. The assembler converts them to 元 and decimal fractions, so downstream config values (1 亿元, 5%–20%, 50–200 亿元) are consistent.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_assemble.py -v
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/core backend/app/data/assemble.py backend/tests/test_assemble.py
git commit -m "feat: assemble processed bars and trading calendar"
```

---

### Task 6: Data quality gate

**Files:**
- Create: `backend/app/data/quality.py`
- Test: `backend/tests/test_quality.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_quality.py`:

```python
from datetime import date

import polars as pl

from app.data.quality import DataQualityGate, QualityReport


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [date(2024, 1, 2)] * 2,
            "open": [10.0, 20.0],
            "high": [10.5, 20.5],
            "low": [9.8, 19.5],
            "close": [10.2, 20.2],
            "pre_close": [10.0, 20.0],
            "vol": [100000.0, 200000.0],
            "amount": [1000000.0, 2000000.0],
            "adj_factor": [1.0, 1.0],
            "up_limit": [11.0, 22.0],
            "down_limit": [9.0, 18.0],
        }
    )


def test_clean_data_passes():
    report = DataQualityGate().check(date(2024, 1, 2), _bars(), expected_count=2)
    assert isinstance(report, QualityReport)
    assert report.passed
    assert report.coverage == 1.0


def test_duplicate_and_ohlc_fail():
    bad = pl.concat([_bars(), _bars().head(1)]).with_columns(
        pl.when(pl.col("ts_code") == "000002.SZ")
        .then(19.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    report = DataQualityGate().check(date(2024, 1, 2), bad, expected_count=2)
    assert not report.passed
    assert any("duplicate" in item for item in report.failures)
    assert any("OHLC" in item for item in report.failures)


def test_missing_coverage_fails():
    report = DataQualityGate().check(date(2024, 1, 2), _bars().head(1), expected_count=3)
    assert not report.passed
    assert any("coverage" in item for item in report.failures)


def test_missing_sentiment_fields_fail():
    report = DataQualityGate().check_sentiment({"limit_up_count": 10})
    assert not report.passed
    assert any("missing sentiment field" in item for item in report.failures)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_quality.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the quality gate**

`backend/app/data/quality.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl


@dataclass
class QualityReport:
    trade_date: date
    passed: bool
    coverage: float
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataQualityGate:
    def __init__(
        self,
        min_coverage: float = 0.99,
        max_limit_mismatch_pct: float = 0.001,
    ):
        self.min_coverage = min_coverage
        self.max_limit_mismatch_pct = max_limit_mismatch_pct

    def check(
        self, trade_date: date, bars: pl.DataFrame, expected_count: int
    ) -> QualityReport:
        failures: list[str] = []
        warnings: list[str] = []
        coverage = bars.height / expected_count if expected_count else 0.0

        if bars.is_empty():
            failures.append("bars is empty")
        if coverage < self.min_coverage:
            failures.append(
                f"coverage {coverage:.2%} below {self.min_coverage:.2%}"
            )
        duplicates = (
            bars.group_by(["ts_code", "trade_date"])
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        if duplicates:
            failures.append(f"duplicate (ts_code, trade_date) rows: {duplicates}")
        bad_ohlc = bars.filter(
            (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("high"))
        ).height
        if bad_ohlc:
            failures.append(f"OHLC violations: {bad_ohlc}")
        bad_adj = bars.filter(
            pl.col("adj_factor").is_null() | (pl.col("adj_factor") <= 0)
        ).height
        if bad_adj:
            failures.append(f"invalid adj_factor rows: {bad_adj}")
        negative = bars.filter(
            (pl.col("vol") < 0) | (pl.col("amount") < 0)
        ).height
        if negative:
            failures.append(f"negative volume/amount rows: {negative}")
        if {"up_limit", "down_limit"} <= set(bars.columns):
            mismatched = bars.filter(
                (pl.col("close") > pl.col("up_limit") + 0.001)
                | (pl.col("close") < pl.col("down_limit") - 0.001)
            ).height
            if bars.height and mismatched / bars.height > self.max_limit_mismatch_pct:
                failures.append(
                    f"limit price mismatches: {mismatched}/{bars.height}"
                )
            elif mismatched:
                warnings.append(f"limit price mismatches: {mismatched}")
        return QualityReport(
            trade_date=trade_date,
            passed=not failures,
            coverage=coverage,
            failures=failures,
            warnings=warnings,
        )

    def check_sentiment(self, snapshot: dict) -> QualityReport:
        required = [
            "limit_up_count",
            "limit_down_count",
            "max_board_height",
            "promotion_rate",
            "blow_rate",
            "red_rate",
            "limit_up_premium",
        ]
        missing = [key for key in required if snapshot.get(key) is None]
        failures = [f"missing sentiment field: {key}" for key in missing]
        trade_date = snapshot.get("trade_date") or date.today()
        return QualityReport(
            trade_date=trade_date,
            passed=not failures,
            coverage=1.0 if not failures else 0.0,
            failures=failures,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_quality.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/data/quality.py backend/tests/test_quality.py
git commit -m "feat: add data quality gate"
```

---

### Task 7: Universe builder and feature computation

**Files:**
- Create: `backend/app/data/universe.py`
- Create: `backend/app/indicators/features.py`
- Test: `backend/tests/test_features.py`
- Test: `backend/tests/test_universe.py`

- [ ] **Step 1: Write the failing feature tests**

`backend/tests/test_features.py`:

```python
from datetime import date, timedelta

import polars as pl
import pytest

from app.indicators.features import FeatureBuilder


def _bars(days: int = 80) -> pl.DataFrame:
    rows = []
    start = date(2023, 10, 1)
    for index in range(days):
        trade_date = start + timedelta(days=index)
        close = 10.0 + index * 0.1
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": close - 0.05,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "pre_close": close - 0.1,
                "vol": 100000.0 + index * 1000,
                "amount": 1000000.0 + index * 10000,
                "adj_factor": 1.0,
                "up_limit": close * 1.1,
                "down_limit": close * 0.9,
                "turnover_rate": 2.0,
                "circ_mv": 10000000000.0,
                "is_suspended": False,
            }
        )
    return pl.DataFrame(rows)


def test_build_features_on_asof():
    bars = _bars()
    as_of = bars["trade_date"][-1]
    features = FeatureBuilder().build(bars, as_of)
    assert features.height == 1
    row = features.row(0, named=True)
    assert row["ma20"] == pytest.approx(
        bars["close"].tail(20).mean()
    )
    assert row["ma60"] == pytest.approx(
        bars["close"].tail(60).mean()
    )
    assert row["ret_20d"] == pytest.approx(
        bars["close"][-1] / bars["close"][-21] - 1
    )
    assert row["close_to_high_60d"] == pytest.approx(
        bars["close"][-1] / bars["high"].tail(60).max()
    )
    assert row["board_height"] == 0
    assert row["rs_20d_market"] == pytest.approx(1.0)


def test_lookahead_truncation_is_invariant():
    bars = _bars()
    as_of = bars["trade_date"][-1]
    full = FeatureBuilder().build(bars, as_of)
    truncated = FeatureBuilder().build(
        bars.filter(pl.col("trade_date") <= as_of), as_of
    )
    assert full.to_dicts() == truncated.to_dicts()
```

- [ ] **Step 2: Write the failing universe tests**

`backend/tests/test_universe.py`:

```python
from datetime import date, timedelta

import polars as pl

from app.core.calendar import TradingCalendar
from app.core.config import load_strategy_config
from app.data.universe import UniverseBuilder


def test_universe_excludes_st_new_and_suspended():
    config = load_strategy_config()
    features = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "trade_date": [date(2024, 1, 2)] * 4,
            "sector": ["银行", "银行", "银行", "银行"],
            "close": [10.0, 20.0, 30.0, 40.0],
            "amount_20d_avg": [2e8, 2e8, 2e8, 2e8],
            "is_suspended": [False, False, True, False],
            "is_limit_up": [False, False, False, True],
        }
    )
    stocks = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "name": ["平安银行", "ST万科", "浦发银行", "招商银行"],
            "market": ["主板", "主板", "主板", "主板"],
            "industry": ["银行"] * 4,
            "list_date": [date(2000, 1, 1), date(2000, 1, 1), date(2000, 1, 1), date(2024, 1, 1)],
            "delist_date": [None, None, None, None],
        }
    )
    calendar = TradingCalendar(
        [date(2023, 1, 1) + timedelta(days=i) for i in range(400)]
    )
    universe = UniverseBuilder(calendar).build(
        date(2024, 1, 2), features, stocks, pl.DataFrame(), config
    )
    assert universe["ts_code"].to_list() == ["000001.SZ"]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_features.py tests/test_universe.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement the feature builder**

`backend/app/indicators/features.py`:

```python
from __future__ import annotations

from datetime import date

import polars as pl


class FeatureBuilder:
    def __init__(self, atr_window: int = 14):
        self.atr_window = atr_window

    def build(self, bars: pl.DataFrame, as_of: date) -> pl.DataFrame:
        frame = bars.filter(pl.col("trade_date") <= as_of).sort(
            ["ts_code", "trade_date"]
        )
        frame = frame.with_columns(
            (pl.col("close") * pl.col("adj_factor")).alias("hfq_close"),
            (pl.col("high") * pl.col("adj_factor")).alias("hfq_high"),
            (pl.col("low") * pl.col("adj_factor")).alias("hfq_low"),
            (pl.col("open") * pl.col("adj_factor")).alias("hfq_open"),
        )
        for window in (5, 10, 20, 60):
            frame = frame.with_columns(
                pl.col("hfq_close")
                .rolling_mean(window, min_periods=window)
                .over("ts_code")
                .alias(f"ma{window}")
            )
        for window in (5, 10, 20, 60):
            frame = frame.with_columns(
                (pl.col(f"ma{window}") / pl.col("adj_factor")).alias(
                    f"ma{window}_raw"
                )
            )

        frame = frame.with_columns(
            (pl.col("hfq_high") - pl.col("hfq_low")).alias("tr1"),
            (pl.col("hfq_high") - pl.col("hfq_close").shift(1).over("ts_code"))
            .abs()
            .alias("tr2"),
            (pl.col("hfq_low") - pl.col("hfq_close").shift(1).over("ts_code"))
            .abs()
            .alias("tr3"),
        ).with_columns(
            pl.max_horizontal("tr1", "tr2", "tr3").alias("true_range")
        )
        frame = frame.with_columns(
            pl.col("true_range")
            .rolling_mean(self.atr_window, min_periods=self.atr_window)
            .over("ts_code")
            .alias("atr14")
        ).with_columns(
            (pl.col("atr14") / pl.col("adj_factor")).alias("atr14_raw")
        )

        frame = frame.with_columns(
            (
                pl.col("hfq_close")
                / pl.col("hfq_close").shift(5).over("ts_code")
                - 1
            ).alias("ret_5d"),
            (
                pl.col("hfq_close")
                / pl.col("hfq_close").shift(20).over("ts_code")
                - 1
            ).alias("ret_20d"),
            (
                pl.col("hfq_close")
                / pl.col("hfq_close").shift(60).over("ts_code")
                - 1
            ).alias("ret_60d"),
        )

        frame = frame.with_columns(
            pl.col("vol")
            .rolling_mean(5, min_periods=5)
            .over("ts_code")
            .alias("vol_5d_avg"),
            pl.col("vol")
            .rolling_mean(20, min_periods=20)
            .over("ts_code")
            .alias("vol_20d_avg"),
            pl.col("amount")
            .rolling_mean(5, min_periods=5)
            .over("ts_code")
            .alias("amount_5d_avg"),
            pl.col("amount")
            .rolling_mean(20, min_periods=20)
            .over("ts_code")
            .alias("amount_20d_avg"),
        ).with_columns(
            (pl.col("vol_5d_avg") / pl.col("vol_20d_avg")).alias(
                "vol_ratio_5_20"
            ),
            (pl.col("vol") / pl.col("vol_20d_avg")).alias(
                "vol_ratio_today_20"
            ),
        )

        frame = frame.with_columns(
            pl.col("hfq_high")
            .rolling_max(60, min_periods=60)
            .over("ts_code")
            .alias("high_60d"),
            pl.col("hfq_low")
            .rolling_min(10, min_periods=10)
            .over("ts_code")
            .alias("low_10d"),
            pl.col("hfq_low")
            .rolling_min(5, min_periods=5)
            .over("ts_code")
            .alias("low_5d"),
            pl.col("hfq_high")
            .rolling_max(20, min_periods=20)
            .over("ts_code")
            .alias("high_20d"),
            pl.col("hfq_close")
            .rolling_max(20, min_periods=20)
            .over("ts_code")
            .alias("close_20d_high"),
        ).with_columns(
            (pl.col("hfq_close") / pl.col("high_60d")).alias(
                "close_to_high_60d"
            ),
            (pl.col("low_5d") / pl.col("adj_factor")).alias("low_5d_raw"),
            (pl.col("low_10d") / pl.col("adj_factor")).alias("low_10d_raw"),
            (pl.col("high_60d") / pl.col("adj_factor")).alias(
                "high_60d_raw"
            ),
        )

        frame = frame.with_columns(
            (pl.col("close") >= pl.col("up_limit") - 0.001).alias(
                "is_limit_up"
            ),
            (pl.col("close") <= pl.col("down_limit") + 0.001).alias(
                "is_limit_down"
            ),
            (pl.col("high") >= pl.col("up_limit") - 0.001).alias(
                "touched_limit_up"
            ),
        )
        frame = frame.with_columns(
            pl.col("close").shift(1).over("ts_code").alias("prev_close"),
            pl.col("is_limit_up").shift(1).over("ts_code").alias(
                "prev_is_limit_up"
            ),
            (pl.col("hfq_close") > pl.col("ma20")).alias("above_ma20"),
        )
        frame = frame.with_columns(
            pl.when(pl.col("is_limit_up"))
            .then(0)
            .otherwise(1)
            .cum_sum()
            .over("ts_code")
            .alias("lu_reset")
        ).with_columns(
            pl.col("is_limit_up")
            .cast(pl.Int8)
            .cum_sum()
            .over(["ts_code", "lu_reset"])
            .alias("board_height")
        )
        frame = frame.with_columns(
            pl.when(pl.col("is_limit_up"))
            .then(pl.col("trade_date"))
            .otherwise(None)
            .forward_fill()
            .over("ts_code")
            .alias("last_limit_up_date")
        ).with_columns(
            (pl.col("trade_date") - pl.col("last_limit_up_date"))
            .dt.total_days()
            .fill_null(999)
            .alias("days_since_limit_up")
        )
        frame = frame.with_columns(
            pl.when(pl.col("hfq_close") == pl.col("close_20d_high"))
            .then(pl.col("trade_date"))
            .otherwise(None)
            .forward_fill()
            .over("ts_code")
            .alias("last_high_date")
        ).with_columns(
            (pl.col("trade_date") - pl.col("last_high_date"))
            .dt.total_days()
            .fill_null(999)
            .alias("pullback_days")
        )
        body = (pl.col("close") - pl.col("open")).abs()
        lower_shadow = (
            pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")
        )
        day_range = pl.col("high") - pl.col("low")
        frame = frame.with_columns(
            (
                (pl.col("close") > pl.col("open"))
                | (lower_shadow >= body)
                | (pl.col("close") >= pl.col("low") + 0.5 * day_range)
            ).alias("stabilization"),
            (pl.col("high") == pl.col("low")).alias("one_word"),
            (pl.col("ma20") - pl.col("ma20").shift(10).over("ts_code")).alias(
                "ma20_slope"
            ),
        )

        asof = frame.filter(pl.col("trade_date") == as_of)
        if asof.is_empty():
            return asof
        asof = asof.with_columns(
            (pl.col("ret_20d").rank("average") / pl.len()).alias(
                "rs_20d_market"
            )
        )
        return asof
```

- [ ] **Step 5: Implement the universe builder**

`backend/app/data/universe.py`:

```python
from __future__ import annotations

from datetime import date

import polars as pl

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig


def _is_st_at(
    namechange: pl.DataFrame, ts_code: str, as_of: date
) -> bool:
    if namechange.is_empty():
        return False
    rows = namechange.filter(pl.col("ts_code") == ts_code)
    for row in rows.iter_rows(named=True):
        start = row.get("start_date")
        end = row.get("end_date")
        if start is None:
            continue
        if start <= as_of and (end is None or end >= as_of):
            name = str(row.get("name", ""))
            if "ST" in name or "退" in name:
                return True
    return False


class UniverseBuilder:
    def __init__(self, calendar: TradingCalendar):
        self.calendar = calendar

    def build(
        self,
        as_of: date,
        features: pl.DataFrame,
        stocks: pl.DataFrame,
        namechange: pl.DataFrame,
        config: StrategyConfig,
    ) -> pl.DataFrame:
        if features.is_empty() or stocks.is_empty():
            return pl.DataFrame()
        merged = features.join(
            stocks.select(
                [
                    "ts_code",
                    "name",
                    "market",
                    "industry",
                    "list_date",
                    "delist_date",
                ]
            ),
            on="ts_code",
            how="left",
        )
        listing_days = []
        is_st = []
        for row in merged.iter_rows(named=True):
            list_date = row.get("list_date")
            listing_days.append(
                self.calendar.count_between(list_date, as_of)
                if list_date is not None
                else 0
            )
            name = str(row.get("name", ""))
            name_is_st = "ST" in name or "退" in name
            is_st.append(
                _is_st_at(namechange, row["ts_code"], as_of) or name_is_st
            )
        merged = merged.with_columns(
            pl.Series("listing_days", listing_days, dtype=pl.Int64),
            pl.Series("is_st", is_st, dtype=pl.Boolean),
        )
        template = config.template_a
        return merged.filter(
            pl.col("market").is_in(template.allowed_boards)
            & (pl.col("listing_days") >= template.min_listing_days)
            & (~pl.col("is_st"))
            & (~pl.col("is_suspended"))
            & (pl.col("amount_20d_avg") >= template.min_avg_amount)
            & (pl.col("close") >= template.universe_min_price)
        )
```

**Point-in-time note:** `_is_st_at` uses historical `namechange` intervals. When `namechange` is empty, the builder falls back to the current name; the pipeline must add a `st_namechange_missing` degraded flag in that case.

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_features.py tests/test_universe.py -v
```

Expected: `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add backend/app/indicators backend/app/data/universe.py backend/tests/test_features.py backend/tests/test_universe.py
git commit -m "feat: compute features and point-in-time universe"
```

---

### Task 8: Sentiment metrics and market regime

**Files:**
- Create: `backend/app/regime/sentiment.py`
- Create: `backend/app/regime/market.py`
- Test: `backend/tests/test_sentiment.py`
- Test: `backend/tests/test_market_regime.py`

- [ ] **Step 1: Write the failing sentiment tests**

`backend/tests/test_sentiment.py`:

```python
from datetime import date, timedelta

import polars as pl

from app.core.config import load_strategy_config
from app.regime.sentiment import SentimentService, SentimentStageClassifier


def _bars() -> pl.DataFrame:
    rows = []
    base = date(2024, 1, 1)
    for day in range(4):
        trade_date = base + timedelta(days=day)
        for code, is_limit_up in (
            ("A", day in (2, 3)),
            ("B", day == 3),
            ("C", False),
        ):
            close = 11.0 if is_limit_up else 10.0
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.9,
                    "close": close,
                    "pre_close": 10.0,
                    "up_limit": 11.0,
                    "down_limit": 9.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                    "adj_factor": 1.0,
                }
            )
    return pl.DataFrame(rows)


def test_sentiment_history_contains_indicators():
    history = SentimentService().compute_history(_bars())
    last = history.sort("trade_date").tail(1).row(0, named=True)
    assert last["limit_up_count"] == 2
    assert last["max_board_height"] == 2
    assert "promotion_rate" in last
    assert "blow_rate" in last
    assert "red_rate" in last


def test_stage_classifier_two_day_confirmation():
    config = load_strategy_config()
    history = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "limit_up_count": [80, 80, 80],
            "limit_down_count": [0, 0, 0],
            "max_board_height": [5, 5, 5],
            "promotion_rate": [0.5, 0.5, 0.5],
            "blow_rate": [0.1, 0.1, 0.1],
            "red_rate": [0.6, 0.6, 0.6],
            "limit_up_premium": [0.02, 0.02, 0.02],
            "blow_rate_ma3": [0.1, 0.1, 0.1],
            "max_board_ma3": [5, 5, 5],
            "premium_ma3": [0.02, 0.02, 0.02],
        }
    )
    classified = SentimentStageClassifier().classify_series(history, config)
    stages = classified["sentiment_stage"].to_list()
    assert stages[0] == "start"
    assert stages[1] == "ferment"
    assert stages[2] == "ferment"
```

- [ ] **Step 2: Write the failing market regime tests**

`backend/tests/test_market_regime.py`:

```python
from datetime import date, timedelta

import polars as pl

from app.core.config import load_strategy_config
from app.domain.enums import SentimentStage, Template, TrendRegime
from app.regime.market import MarketRegimeService


def test_defense_overrides_sentiment():
    config = load_strategy_config()
    start = date(2024, 1, 1)
    indices = pl.DataFrame(
        {
            "trade_date": [start + timedelta(days=i) for i in range(61)],
            "close": [100.0 - i for i in range(61)],
        }
    )
    sentiment_row = {
        "limit_up_count": 80,
        "limit_down_count": 0,
        "max_board_height": 5,
        "promotion_rate": 0.5,
        "blow_rate": 0.1,
        "red_rate": 0.6,
        "limit_up_premium": 0.02,
    }
    regime = MarketRegimeService().compute(
        as_of=date(2024, 3, 1),
        index_bars=indices,
        sentiment_row=sentiment_row,
        stage=SentimentStage.FERMENT,
        breadth=0.60,
        sector_ranking=[],
        degraded_flags=[],
        data_version="dv_test",
        config=config,
    )
    assert regime.trend_regime == TrendRegime.DEFENSIVE
    assert regime.risk_factor == 0.0
    assert not regime.template_availability[Template.A].enabled
    assert not regime.template_availability[Template.B].enabled
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_sentiment.py tests/test_market_regime.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement sentiment services**

`backend/app/regime/sentiment.py`:

```python
from __future__ import annotations

import polars as pl

from app.core.config import StrategyConfig
from app.domain.enums import SentimentStage


class SentimentService:
    def compute_history(self, bars: pl.DataFrame) -> pl.DataFrame:
        if bars.is_empty():
            return pl.DataFrame()
        frame = bars.sort(["ts_code", "trade_date"]).with_columns(
            (pl.col("close") >= pl.col("up_limit") - 0.001).alias(
                "is_limit_up"
            ),
            (pl.col("close") <= pl.col("down_limit") + 0.001).alias(
                "is_limit_down"
            ),
            (pl.col("high") >= pl.col("up_limit") - 0.001).alias(
                "touched_limit_up"
            ),
        )
        frame = frame.with_columns(
            pl.col("close").shift(1).over("ts_code").alias("prev_close"),
            pl.col("is_limit_up").shift(1).over("ts_code").alias(
                "prev_is_limit_up"
            ),
        )
        frame = frame.with_columns(
            pl.when(pl.col("is_limit_up"))
            .then(0)
            .otherwise(1)
            .cum_sum()
            .over("ts_code")
            .alias("lu_reset")
        ).with_columns(
            pl.col("is_limit_up")
            .cast(pl.Int8)
            .cum_sum()
            .over(["ts_code", "lu_reset"])
            .alias("board_height")
        )
        frame = frame.with_columns(
            pl.col("board_height").shift(1).over("ts_code").alias(
                "prev_board_height"
            )
        )
        frame = frame.with_columns(
            (
                (pl.col("prev_board_height") > 0)
                & (pl.col("board_height") == pl.col("prev_board_height") + 1)
            ).alias("promoted"),
            (
                pl.col("prev_is_limit_up").fill_null(False)
                & (pl.col("close") > pl.col("prev_close"))
            ).alias("red"),
            pl.when(
                pl.col("prev_is_limit_up").fill_null(False)
                & pl.col("prev_close").is_not_null()
            )
            .then(pl.col("open") / pl.col("prev_close") - 1)
            .otherwise(None)
            .alias("premium"),
            (
                pl.col("touched_limit_up").fill_null(False)
                & ~pl.col("is_limit_up")
            ).alias("blow"),
        )
        grouped = frame.group_by("trade_date").agg(
            pl.col("is_limit_up").sum().alias("limit_up_count"),
            pl.col("is_limit_down").sum().alias("limit_down_count"),
            pl.col("board_height").max().alias("max_board_height"),
            (
                pl.col("promoted").sum()
                / pl.col("prev_board_height").gt(0).sum()
            ).fill_null(0.0).alias("promotion_rate"),
            (
                pl.col("blow").sum() / pl.col("touched_limit_up").sum()
            ).fill_null(0.0).alias("blow_rate"),
            (
                pl.col("red").sum() / pl.col("prev_is_limit_up").sum()
            ).fill_null(0.0).alias("red_rate"),
            pl.col("premium").mean().fill_null(0.0).alias(
                "limit_up_premium"
            ),
        ).sort("trade_date")
        grouped = grouped.with_columns(
            pl.col("blow_rate").rolling_mean(3, min_periods=1).alias(
                "blow_rate_ma3"
            ),
            pl.col("max_board_height")
            .rolling_mean(3, min_periods=1)
            .alias("max_board_ma3"),
            pl.col("limit_up_premium")
            .rolling_mean(3, min_periods=1)
            .alias("premium_ma3"),
        )
        return grouped


class SentimentStageClassifier:
    def classify_series(
        self, history: pl.DataFrame, config: StrategyConfig
    ) -> pl.DataFrame:
        if history.is_empty():
            return history.with_columns(
                pl.lit(SentimentStage.START.value).alias("sentiment_stage")
            )
        current = SentimentStage.START
        pending: SentimentStage | None = None
        pending_count = 0
        stages: list[str] = []
        confirm_days = config.environment.stage_confirm_days
        for row in history.sort("trade_date").iter_rows(named=True):
            candidate = self._classify_row(row, config)
            if candidate is None:
                pending = None
                pending_count = 0
            elif candidate == current:
                pending = None
                pending_count = 0
            elif candidate == pending:
                pending_count += 1
                if pending_count >= confirm_days:
                    current = candidate
                    pending = None
                    pending_count = 0
            else:
                pending = candidate
                pending_count = 1
            stages.append(current.value)
        return history.sort("trade_date").with_columns(
            pl.Series("sentiment_stage", stages)
        )

    def _classify_row(
        self, row: dict, config: StrategyConfig
    ) -> SentimentStage | None:
        thresholds = config.environment.thresholds
        counts = {
            SentimentStage.FREEZE: sum(
                [
                    row["limit_up_count"] < thresholds.freeze.limit_up_max,
                    row["limit_down_count"] > thresholds.freeze.limit_down_min
                    or row["blow_rate"] > thresholds.freeze.blow_rate_min,
                    row["max_board_height"] <= thresholds.freeze.max_board_max,
                    row["red_rate"] < thresholds.freeze.red_rate_max,
                ]
            ),
            SentimentStage.START: sum(
                [
                    thresholds.start.limit_up_min
                    <= row["limit_up_count"]
                    <= thresholds.start.limit_up_max,
                    thresholds.start.max_board_min
                    <= row["max_board_height"]
                    <= thresholds.start.max_board_max,
                    thresholds.start.blow_rate_min
                    <= row["blow_rate"]
                    <= thresholds.start.blow_rate_max,
                    thresholds.start.red_rate_min
                    <= row["red_rate"]
                    <= thresholds.start.red_rate_max,
                ]
            ),
            SentimentStage.FERMENT: sum(
                [
                    row["limit_up_count"] > thresholds.ferment.limit_up_min,
                    row["max_board_height"] >= thresholds.ferment.max_board_min,
                    row["blow_rate"] < thresholds.ferment.blow_rate_max,
                    row["red_rate"] > thresholds.ferment.red_rate_min,
                ]
            ),
            SentimentStage.CLIMAX: sum(
                [
                    row["limit_up_count"] > thresholds.climax.limit_up_min,
                    row["max_board_height"] >= thresholds.climax.max_board_min,
                    row["red_rate"] > thresholds.climax.red_rate_min,
                    (
                        (row["blow_rate"] - row["blow_rate_ma3"]) * 100
                        >= thresholds.climax.blow_rate_rise_pp
                    )
                    or (
                        (row["limit_up_premium"] - row["premium_ma3"]) * 100
                        <= -thresholds.climax.premium_drop_pp
                    ),
                ]
            ),
            SentimentStage.EBB: sum(
                [
                    row["limit_up_count"] < thresholds.ebb.limit_up_max,
                    row["limit_down_count"] > thresholds.ebb.limit_down_min,
                    row["blow_rate"] > thresholds.ebb.blow_rate_min,
                    row["max_board_height"] < row["max_board_ma3"],
                    row["red_rate"] < thresholds.ebb.red_rate_max,
                ]
            ),
        }
        priority = [
            SentimentStage.EBB,
            SentimentStage.FREEZE,
            SentimentStage.CLIMAX,
            SentimentStage.FERMENT,
            SentimentStage.START,
        ]
        best = max(counts.values())
        if best < 2:
            return None
        for stage in priority:
            if counts[stage] == best:
                return stage
        return None
```

- [ ] **Step 5: Implement the market regime service**

`backend/app/regime/market.py`:

```python
from __future__ import annotations

from datetime import date

import polars as pl

from app.core.config import StrategyConfig
from app.domain.enums import SentimentStage, Template, TrendRegime
from app.domain.models import (
    MarketRegime,
    RegimeIndicators,
    SectorStrength,
    TemplatePolicy,
)


class MarketRegimeService:
    def compute(
        self,
        as_of: date,
        index_bars: pl.DataFrame,
        sentiment_row: dict,
        stage: SentimentStage,
        breadth: float,
        sector_ranking: list[SectorStrength],
        degraded_flags: list[str],
        data_version: str,
        config: StrategyConfig,
    ) -> MarketRegime:
        prepared = index_bars.sort("trade_date").with_columns(
            pl.col("close")
            .rolling_mean(config.environment.ma_fast, min_periods=config.environment.ma_fast)
            .alias("ma_fast"),
            pl.col("close")
            .rolling_mean(config.environment.ma_slow, min_periods=config.environment.ma_slow)
            .alias("ma_slow"),
        )
        row = prepared.filter(pl.col("trade_date") == as_of).tail(1)
        if row.is_empty():
            raise ValueError(f"index data missing for {as_of}")
        index_close = float(row["close"][0])
        ma_fast = float(row["ma_fast"][0])
        ma_slow = float(row["ma_slow"][0])

        if index_close < ma_slow or breadth < config.environment.breadth_defense:
            trend = TrendRegime.DEFENSIVE
            trend_cap = 0.0
        elif index_close > ma_fast and breadth >= config.environment.breadth_attack:
            trend = TrendRegime.ATTACK
            trend_cap = 1.0
        else:
            trend = TrendRegime.NEUTRAL
            trend_cap = 0.5

        stage_policy = getattr(config.switching, stage.value)
        risk_factor = min(stage_policy.risk_factor, trend_cap)
        policies = self._resolve_policies(stage, trend, config)

        indicators = RegimeIndicators(
            limit_up_count=int(sentiment_row["limit_up_count"]),
            limit_down_count=int(sentiment_row["limit_down_count"]),
            max_board_height=int(sentiment_row["max_board_height"]),
            promotion_rate=float(sentiment_row["promotion_rate"]),
            blow_rate=float(sentiment_row["blow_rate"]),
            red_rate=float(sentiment_row["red_rate"]),
            limit_up_premium=float(sentiment_row["limit_up_premium"]),
            breadth=breadth,
            index_close=index_close,
            index_ma20=ma_fast,
            index_ma60=ma_slow,
        )
        return MarketRegime(
            trade_date=as_of,
            trend_regime=trend,
            sentiment_stage=stage,
            risk_factor=risk_factor,
            template_availability=policies,
            indicators=indicators,
            sector_ranking=sector_ranking,
            degraded_flags=degraded_flags,
            data_version=data_version,
        )

    def _resolve_policies(
        self, stage: SentimentStage, trend: TrendRegime, config: StrategyConfig
    ) -> dict[Template, TemplatePolicy]:
        if trend == TrendRegime.DEFENSIVE:
            return {
                Template.A: TemplatePolicy(enabled=False, mode="observe", max_positions=0),
                Template.B: TemplatePolicy(enabled=False, mode="observe", max_positions=0),
            }
        stage_policy = getattr(config.switching, stage.value)
        a = TemplatePolicy(
            enabled=stage_policy.a.enabled,
            mode=stage_policy.a.mode,
            max_positions=stage_policy.a.max_positions,
        )
        max_b = stage_policy.b.max_positions
        if stage == SentimentStage.FERMENT and trend == TrendRegime.NEUTRAL:
            max_b = min(max_b, 1)
        b = TemplatePolicy(
            enabled=stage_policy.b.enabled,
            mode=stage_policy.b.mode,
            max_positions=max_b,
        )
        if stage == SentimentStage.CLIMAX:
            a = TemplatePolicy(enabled=True, mode="pullback_only", max_positions=3)
            b = TemplatePolicy(enabled=False, mode="observe", max_positions=0)
        return {Template.A: a, Template.B: b}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_sentiment.py tests/test_market_regime.py -v
```

Expected: `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add backend/app/regime backend/tests/test_sentiment.py backend/tests/test_market_regime.py
git commit -m "feat: add sentiment cycle and market regime"
```

---

### Task 9: Industry map and industry strength

**Files:**
- Create: `backend/app/data/industry_map.py`
- Create: `backend/app/regime/industry.py`
- Modify: `backend/app/data/interfaces.py` (add `load_index_classify`, `load_limit_list`)
- Modify: `backend/app/data/provider.py` (implement the two methods)
- Test: `backend/tests/test_industry.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_industry.py`:

```python
from datetime import date

import polars as pl

from app.core.config import load_strategy_config
from app.data.industry_map import IndustryMap
from app.regime.industry import IndustryStrengthService


def _history() -> pl.DataFrame:
    rows = []
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    for trade_date in dates:
        for index in range(5):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"S{index}",
                    "sector": "强",
                    "ret_20d": 0.20,
                    "ret_5d": 0.05,
                    "above_ma20": True,
                    "amount_5d_avg": 1.5e8,
                    "amount_20d_avg": 1.0e8,
                    "is_limit_up": index == 0,
                    "prev_is_limit_up": index < 2,
                    "prev_close": 9.9,
                    "close": 10.0,
                }
            )
        for index in range(5):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"W{index}",
                    "sector": "弱",
                    "ret_20d": 0.0,
                    "ret_5d": 0.0,
                    "above_ma20": False,
                    "amount_5d_avg": 1.0e8,
                    "amount_20d_avg": 1.0e8,
                    "is_limit_up": False,
                    "prev_is_limit_up": False,
                    "prev_close": 10.0,
                    "close": 10.0,
                }
            )
    return pl.DataFrame(rows)


def test_industry_strength_ranks_and_persists():
    config = load_strategy_config()
    config.industry.top_n = 1
    sectors = IndustryStrengthService().compute(
        _history(), date(2024, 1, 4), config.industry
    )
    assert sectors[0].sector == "强"
    assert sectors[0].rank == 1
    assert sectors[0].persistent_days == 3
    assert sectors[1].persistent_days == 0
    assert sectors[0].limit_up_count == 1
    assert sectors[0].red_rate == 0.5


def test_industry_map_prefers_sw_membership():
    stocks = pl.DataFrame(
        {"ts_code": ["A", "B"], "industry": ["银行", "地产"]}
    )
    members = pl.DataFrame(
        {
            "con_code": ["A"],
            "index_code": ["801010.SI"],
            "in_date": [date(2000, 1, 1)],
            "out_date": [None],
        }
    )
    classify = pl.DataFrame(
        {"index_code": ["801010.SI"], "industry_name": ["银行"]}
    )
    mapping = IndustryMap.from_sources(
        stocks, members, classify, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "sw"
    assert not mapping.degraded
    assert mapping.mapping["A"] == "银行"
    assert mapping.mapping["B"] == "地产"


def test_industry_map_falls_back_and_respects_dates():
    stocks = pl.DataFrame({"ts_code": ["A"], "industry": ["银行"]})
    members = pl.DataFrame(
        {
            "con_code": ["A"],
            "index_code": ["801010.SI"],
            "in_date": [date(2025, 1, 1)],
            "out_date": [None],
        }
    )
    mapping = IndustryMap.from_sources(
        stocks, members, None, as_of=date(2024, 1, 2)
    )
    assert mapping.source == "stock_basic"
    assert mapping.degraded
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_industry.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 3: Extend the provider protocol**

Add to `backend/app/data/interfaces.py`:

```python
    def load_index_classify(self) -> pl.DataFrame: ...
    def load_limit_list(self, start: date, end: date) -> pl.DataFrame: ...
```

Add to `ParquetMarketDataProvider` in `backend/app/data/provider.py`:

```python
    def load_index_classify(self) -> pl.DataFrame:
        return self.store.read_reference("index_classify")

    def load_limit_list(self, start: date, end: date) -> pl.DataFrame:
        return self.store.read_processed("limit_list", start, end)
```

- [ ] **Step 4: Implement the industry map**

`backend/app/data/industry_map.py`:

```python
from __future__ import annotations

from datetime import date

import polars as pl


class IndustryMap:
    def __init__(self, mapping: dict[str, str], source: str, degraded: bool):
        self.mapping = mapping
        self.source = source
        self.degraded = degraded

    def to_frame(self) -> pl.DataFrame:
        if not self.mapping:
            return pl.DataFrame({"ts_code": [], "sector": []})
        return pl.DataFrame(
            {
                "ts_code": list(self.mapping.keys()),
                "sector": list(self.mapping.values()),
            }
        )

    @classmethod
    def from_sources(
        cls,
        stock_basic: pl.DataFrame,
        index_member: pl.DataFrame | None = None,
        index_classify: pl.DataFrame | None = None,
        as_of: date | None = None,
        source: str = "auto",
    ) -> "IndustryMap":
        if (
            source in ("auto", "sw")
            and index_member is not None
            and not index_member.is_empty()
        ):
            names: dict[str, str] = {}
            if index_classify is not None and not index_classify.is_empty():
                for row in index_classify.iter_rows(named=True):
                    index_code = row.get("index_code")
                    names[index_code] = row.get("industry_name") or index_code
            mapping: dict[str, str] = {}
            for row in index_member.iter_rows(named=True):
                code = row.get("con_code") or row.get("ts_code")
                index_code = row.get("index_code")
                in_date = row.get("in_date")
                out_date = row.get("out_date")
                if code is None or index_code is None:
                    continue
                if as_of is not None and in_date is not None and in_date > as_of:
                    continue
                if as_of is not None and out_date is not None and out_date < as_of:
                    continue
                mapping[code] = names.get(index_code, index_code)
            if mapping:
                if not stock_basic.is_empty():
                    for row in stock_basic.iter_rows(named=True):
                        mapping.setdefault(
                            row["ts_code"], row.get("industry") or "未知"
                        )
                return cls(mapping, "sw", False)

        mapping = {}
        if not stock_basic.is_empty():
            for row in stock_basic.iter_rows(named=True):
                mapping[row["ts_code"]] = row.get("industry") or "未知"
        return cls(mapping, "stock_basic", True)
```

- [ ] **Step 5: Implement industry strength**

`backend/app/regime/industry.py`:

```python
from __future__ import annotations

from datetime import date

import polars as pl

from app.core.config import IndustryConfig
from app.domain.models import SectorStrength


class IndustryStrengthService:
    def compute(
        self,
        features_history: pl.DataFrame,
        as_of: date,
        config: IndustryConfig,
    ) -> list[SectorStrength]:
        if features_history.is_empty():
            return []
        frame = features_history.filter(pl.col("trade_date") <= as_of)
        if frame.is_empty():
            return []
        frame = frame.with_columns(
            (
                pl.col("amount_5d_avg") / pl.col("amount_20d_avg") - 1
            ).alias("amount_change_stock"),
            (
                pl.col("prev_is_limit_up").fill_null(False)
                & (pl.col("close") > pl.col("prev_close"))
            ).alias("red"),
        )
        aggregated = (
            frame.group_by(["trade_date", "sector"])
            .agg(
                pl.col("ret_20d").mean().alias("return_20d"),
                pl.col("ret_5d").mean().alias("return_5d"),
                pl.col("above_ma20").mean().alias("breadth"),
                pl.col("amount_change_stock").mean().alias("amount_change"),
                pl.col("is_limit_up").mean().alias("limit_up_share"),
                pl.col("is_limit_up").sum().alias("limit_up_count"),
                (
                    pl.col("red").sum() / pl.col("prev_is_limit_up").sum()
                )
                .fill_null(0.0)
                .alias("red_rate"),
                pl.col("ts_code").n_unique().alias("stock_count"),
            )
            .filter(pl.col("stock_count") >= config.min_stocks)
        )
        if aggregated.is_empty():
            return []

        weights = config.weights
        ranked = (
            aggregated.with_columns(
                (
                    pl.col("return_20d").rank("average").over("trade_date")
                    / pl.len().over("trade_date")
                ).alias("p20"),
                (
                    pl.col("return_5d").rank("average").over("trade_date")
                    / pl.len().over("trade_date")
                ).alias("p5"),
                (
                    pl.col("breadth").rank("average").over("trade_date")
                    / pl.len().over("trade_date")
                ).alias("pb"),
                (
                    pl.col("amount_change").rank("average").over("trade_date")
                    / pl.len().over("trade_date")
                ).alias("pa"),
                (
                    pl.col("limit_up_share")
                    .rank("average")
                    .over("trade_date")
                    / pl.len().over("trade_date")
                ).alias("pl"),
            )
            .with_columns(
                (
                    weights.return_20d * pl.col("p20")
                    + weights.return_5d * pl.col("p5")
                    + weights.breadth * pl.col("pb")
                    + weights.amount_change * pl.col("pa")
                    + weights.limit_up_share * pl.col("pl")
                )
                .mul(100)
                .alias("score")
            )
            .with_columns(
                pl.col("score")
                .rank("ordinal", descending=True)
                .over("trade_date")
                .alias("daily_rank")
            )
        )

        current = ranked.filter(pl.col("trade_date") == as_of).sort(
            "score", descending=True
        )
        if current.is_empty():
            return []

        dates = ranked["trade_date"].unique().sort().to_list()
        results: list[SectorStrength] = []
        for rank, row in enumerate(current.iter_rows(named=True), start=1):
            streak = 0
            for offset in range(len(dates)):
                check_date = dates[-(offset + 1)]
                match = ranked.filter(
                    (pl.col("trade_date") == check_date)
                    & (pl.col("sector") == row["sector"])
                )
                if (
                    not match.is_empty()
                    and int(match["daily_rank"][0]) <= config.top_n
                ):
                    streak += 1
                else:
                    break
            results.append(
                SectorStrength(
                    sector=row["sector"],
                    score=float(row["score"]),
                    rank=rank,
                    return_20d=float(row["return_20d"]),
                    return_5d=float(row["return_5d"]),
                    breadth=float(row["breadth"]),
                    amount_change=float(row["amount_change"]),
                    limit_up_share=float(row["limit_up_share"]),
                    limit_up_count=int(row["limit_up_count"]),
                    red_rate=float(row["red_rate"]),
                    stock_count=int(row["stock_count"]),
                    persistent_days=streak,
                )
            )
        return results
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_industry.py -v
```

Expected: `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add backend/app/data/industry_map.py backend/app/data/interfaces.py backend/app/data/provider.py backend/app/regime/industry.py backend/tests/test_industry.py
git commit -m "feat: add industry mapping and strength ranking"
```

---

### Task 10: Leader score and follower exclusion

**Files:**
- Create: `backend/app/regime/leader.py`
- Test: `backend/tests/test_leader.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_leader.py`:

```python
from datetime import date

import polars as pl

from app.core.config import load_strategy_config
from app.domain.models import SectorStrength
from app.regime.leader import LeaderScoreService


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["A", "B", "C", "D"],
            "trade_date": [date(2024, 1, 2)] * 4,
            "sector": ["银行"] * 4,
            "ret_20d": [0.30, 0.20, 0.10, 0.0],
            "ret_5d": [0.10, 0.05, 0.02, 0.0],
            "amount_20d_avg": [3e8, 1e8, 1e8, 1e8],
            "turnover_rate": [0.10, 0.08, 0.05, 0.03],
            "close_to_high_60d": [0.99, 0.95, 0.90, 0.80],
            "board_height": [2, 1, 0, 0],
            "days_since_limit_up": [0, 0, 10, 20],
            "circ_mv": [1e10, 1e10, 1e10, 1e10],
            "close": [10.0, 10.0, 10.0, 10.0],
        }
    )


def _sector() -> SectorStrength:
    return SectorStrength(
        sector="银行",
        score=90.0,
        rank=1,
        return_20d=0.2,
        return_5d=0.05,
        breadth=0.8,
        amount_change=0.2,
        limit_up_share=0.2,
        limit_up_count=3,
        red_rate=0.6,
        stock_count=10,
        persistent_days=2,
    )


def test_leader_rank_and_followers():
    config = load_strategy_config()
    leaders = LeaderScoreService().compute(_features(), [_sector()], config)
    rows = {row["ts_code"]: row for row in leaders.to_dicts()}
    assert rows["A"]["leader_rank"] == 1
    assert not rows["A"]["is_follower"]
    assert rows["B"]["is_follower"]
    assert rows["D"]["is_follower"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
uv run pytest tests/test_leader.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the leader service**

`backend/app/regime/leader.py`:

```python
from __future__ import annotations

import polars as pl

from app.core.config import StrategyConfig


class LeaderScoreService:
    def compute(
        self,
        features: pl.DataFrame,
        sectors: list,
        config: StrategyConfig,
    ) -> pl.DataFrame:
        if features.is_empty() or not sectors:
            return pl.DataFrame()
        selected = {
            sector.sector
            for sector in sectors
            if sector.rank <= config.industry.top_n
        }
        frame = features.filter(pl.col("sector").is_in(selected))
        if frame.is_empty():
            return pl.DataFrame()
        weights = config.leader.weights
        frame = frame.with_columns(
            (
                pl.col("ret_20d").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("rs_20d_pct"),
            (
                pl.col("ret_5d").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("rs_5d_pct"),
            (
                pl.col("amount_20d_avg").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("amount_pct"),
            (
                pl.col("turnover_rate").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("turnover_pct"),
            (
                pl.col("close_to_high_60d").rank("average").over("sector")
                / pl.len().over("sector")
            ).alias("high_pct"),
            pl.when(pl.col("board_height") >= 2)
            .then(100.0)
            .when(pl.col("board_height") == 1)
            .then(80.0)
            .when(pl.col("days_since_limit_up") <= 5)
            .then(50.0)
            .otherwise(0.0)
            .alias("limit_status"),
            pl.when(
                pl.col("circ_mv").is_between(5e9, 5e10)
                & pl.col("close").is_between(5, 100)
            )
            .then(100.0)
            .when(
                pl.col("circ_mv").is_between(2.5e9, 1e11)
                & pl.col("close").is_between(3, 150)
            )
            .then(50.0)
            .otherwise(0.0)
            .alias("size_fit"),
        ).with_columns(
            (
                weights.rs_20d * pl.col("rs_20d_pct")
                + weights.rs_5d * pl.col("rs_5d_pct")
                + weights.amount_turnover
                * (0.5 * pl.col("amount_pct") + 0.5 * pl.col("turnover_pct"))
                + weights.high_distance * pl.col("high_pct")
                + weights.limit_up_status
                * (pl.col("limit_status") / 100)
                + weights.size_price_fit * (pl.col("size_fit") / 100)
            )
            .mul(100)
            .alias("leader_score")
        ).with_columns(
            pl.col("leader_score")
            .rank("ordinal", descending=True)
            .over("sector")
            .alias("leader_rank"),
            pl.col("amount_20d_avg")
            .max()
            .over("sector")
            .alias("sector_leader_amount"),
        ).with_columns(
            (
                (pl.col("leader_rank") > config.leader.follower_rank_cutoff)
                | (
                    pl.col("amount_20d_avg")
                    < config.leader.amount_vs_leader_min
                    * pl.col("sector_leader_amount")
                )
            ).alias("is_follower")
        )
        return frame.select(
            [
                "ts_code",
                "sector",
                "leader_score",
                "leader_rank",
                "is_follower",
                "amount_20d_avg",
            ]
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend
uv run pytest tests/test_leader.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/regime/leader.py backend/tests/test_leader.py
git commit -m "feat: add leader scoring and follower exclusion"
```

---

### Task 11: Screener protocol, registry, and dual-template screener

**Files:**
- Create: `backend/app/screener/base.py`
- Create: `backend/app/screener/registry.py`
- Create: `backend/app/screener/dual_template.py`
- Test: `backend/tests/test_screener.py`
- Test: `backend/tests/test_registry.py`

- [ ] **Step 1: Write the failing screener test**

`backend/tests/test_screener.py`:

```python
from datetime import date

import polars as pl

from app.core.config import load_strategy_config
from app.domain.enums import SentimentStage, Template, TrendRegime
from app.domain.models import MarketRegime, RegimeIndicators, SectorStrength, TemplatePolicy
from app.screener.dual_template import DualTemplateScreener


def _features() -> pl.DataFrame:
    common = {
        "trade_date": date(2024, 1, 2),
        "sector": "银行",
        "hfq_close": 10.2,
        "ma5": 10.0,
        "ma10": 9.9,
        "ma20": 10.0,
        "ma60": 9.0,
        "ma20_slope": 0.2,
        "rs_20d_market": 0.9,
        "close_to_high_60d": 0.95,
        "pullback_days": 5,
        "vol_ratio_5_20": 0.7,
        "stabilization": True,
        "circ_mv": 1e10,
        "close": 10.2,
        "is_st": False,
        "is_suspended": False,
        "is_limit_up": False,
        "atr14": 0.4,
        "atr14_raw": 0.4,
        "low_5d_raw": 9.6,
        "low_10d_raw": 9.4,
        "high_60d_raw": 10.7,
        "turnover_rate": 0.05,
        "board_height": 0,
        "days_since_limit_up": 20,
        "one_word": False,
        "ret_20d": 0.20,
        "vol_ratio_today_20": 1.0,
        "above_ma20": True,
    }
    a = {"ts_code": "000001.SZ", **common}
    b = {
        **common,
        "ts_code": "000002.SZ",
        "hfq_close": 10.0,
        "close": 10.0,
        "is_limit_up": True,
        "board_height": 1,
        "days_since_limit_up": 0,
        "vol_ratio_today_20": 3.0,
        "turnover_rate": 0.10,
        "ret_20d": 0.30,
    }
    follower = {
        **common,
        "ts_code": "000003.SZ",
    }
    return pl.DataFrame([a, b, follower])


def _leaders() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "sector": ["银行"] * 3,
            "leader_score": [90.0, 85.0, 70.0],
            "leader_rank": [1, 2, 3],
            "is_follower": [False, False, True],
            "amount_20d_avg": [3e8, 2e8, 1e8],
        }
    )


def _sector() -> SectorStrength:
    return SectorStrength(
        sector="银行",
        score=90.0,
        rank=1,
        return_20d=0.2,
        return_5d=0.05,
        breadth=0.8,
        amount_change=0.2,
        limit_up_share=0.3,
        limit_up_count=3,
        red_rate=0.6,
        stock_count=10,
        persistent_days=2,
    )


def _regime() -> MarketRegime:
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime=TrendRegime.ATTACK,
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={
            Template.A: policy,
            Template.B: TemplatePolicy(enabled=True, mode="normal", max_positions=2),
        },
        indicators=RegimeIndicators(
            limit_up_count=80,
            limit_down_count=0,
            max_board_height=5,
            promotion_rate=0.5,
            blow_rate=0.1,
            red_rate=0.6,
            limit_up_premium=0.02,
            breadth=0.6,
            index_close=4000.0,
            index_ma20=3900.0,
            index_ma60=3800.0,
        ),
        data_version="dv_test",
    )


def test_dual_template_selects_a_and_b():
    config = load_strategy_config()
    candidates = DualTemplateScreener().generate(
        as_of=date(2024, 1, 2),
        regime=_regime(),
        features=_features(),
        sectors=[_sector()],
        leaders=_leaders(),
        config=config,
    )
    assert {candidate.template for candidate in candidates} == {
        Template.A,
        Template.B,
    }
    assert all(candidate.reasons for candidate in candidates)
```

`backend/tests/test_registry.py`:

```python
from app.screener.registry import get_strategy, list_strategies, register
from app.screener.dual_template import DualTemplateScreener


def test_registry_register_and_get():
    strategy = DualTemplateScreener()
    register(strategy)
    assert get_strategy(strategy.name) is strategy
    assert strategy.name in list_strategies()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_screener.py tests/test_registry.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement the protocol and registry**

`backend/app/screener/base.py`:

```python
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
```

`backend/app/screener/registry.py`:

```python
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
```

- [ ] **Step 4: Implement the dual-template screener**

`backend/app/screener/dual_template.py`:

```python
from __future__ import annotations

from datetime import date

import polars as pl

from app.core.config import StrategyConfig
from app.domain.enums import SentimentStage, Template
from app.domain.models import Candidate, MarketRegime, SectorStrength
from app.screener.registry import register


class DualTemplateScreener:
    name = "dual_template_v1"
    version = "1.1"

    def __init__(self) -> None:
        register(self)

    def generate(
        self,
        as_of: date,
        regime: MarketRegime,
        features: pl.DataFrame,
        sectors: list[SectorStrength],
        leaders: pl.DataFrame,
        config: StrategyConfig,
    ) -> list[Candidate]:
        if features.is_empty() or leaders.is_empty():
            return []
        frame = features.join(
            leaders.select(
                ["ts_code", "leader_score", "leader_rank", "is_follower"]
            ),
            on="ts_code",
            how="left",
        )
        sector_by_name = {sector.sector: sector for sector in sectors}
        candidates: list[Candidate] = []
        if regime.template_availability[Template.A].enabled:
            candidates.extend(
                self._select_a(
                    as_of, regime, frame, sector_by_name, config
                )
            )
        if regime.template_availability[Template.B].enabled:
            candidates.extend(
                self._select_b(
                    as_of, regime, frame, sector_by_name, config
                )
            )
        return candidates

    def _select_a(
        self,
        as_of: date,
        regime: MarketRegime,
        frame: pl.DataFrame,
        sector_by_name: dict[str, SectorStrength],
        config: StrategyConfig,
    ) -> list[Candidate]:
        template = config.template_a
        selected = frame.filter(
            pl.col("sector").is_in(
                [
                    name
                    for name, sector in sector_by_name.items()
                    if sector.rank <= config.industry.top_n
                ]
            )
            & (pl.col("leader_rank") <= config.leader.top_n_per_industry)
            & (~pl.col("is_follower").fill_null(False))
            & (pl.col("hfq_close") > pl.col("ma20"))
            & (pl.col("ma20") > pl.col("ma60"))
            & (pl.col("ma20_slope") > 0)
            & (pl.col("rs_20d_market") >= template.rs_20d_market_min)
            & (pl.col("close_to_high_60d") >= 1 - template.max_distance_from_60d_high)
            & pl.col("pullback_days").is_between(
                template.pullback_days_min, template.pullback_days_max
            )
            & (pl.col("vol_ratio_5_20") < template.volume_shrink_ratio)
            & (
                (
                    (pl.col("hfq_close") / pl.col("ma10") - 1).abs()
                    <= 0.015
                )
                | (
                    (pl.col("hfq_close") / pl.col("ma20") - 1).abs()
                    <= template.ma_zone_pct
                )
            )
            & (pl.col("stabilization"))
            & pl.col("circ_mv").is_between(
                template.market_cap_min, template.market_cap_max
            )
            & pl.col("close").is_between(template.min_price, template.max_price)
            & (~pl.col("is_st"))
            & (~pl.col("is_suspended"))
            & (~pl.col("is_limit_up"))
        )
        if selected.is_empty():
            return []
        scored = selected.with_columns(
            (
                (1 - pl.col("vol_ratio_5_20")).clip(0, 1) * 400
            ).clip(0, 100).alias("volume_score"),
            (
                (
                    0.03
                    - pl.min_horizontal(
                        (pl.col("hfq_close") / pl.col("ma10") - 1).abs(),
                        (pl.col("hfq_close") / pl.col("ma20") - 1).abs(),
                    )
                )
                / 0.03
                * 100
            ).clip(0, 100).alias("ma_score"),
            (
                100
                - (
                    (pl.col("pullback_days") - 5.5).abs() / 2.5 * 100
                ).clip(0, 100)
            ).alias("pullback_score"),
            pl.when(pl.col("stabilization")).then(100.0).otherwise(0.0).alias(
                "stabilization_score"
            ),
        ).with_columns(
            (
                0.4 * pl.col("volume_score")
                + 0.2 * pl.col("ma_score")
                + 0.2 * pl.col("pullback_score")
                + 0.2 * pl.col("stabilization_score")
            ).alias("setup_score")
        ).with_columns(
            (
                template.score.leader * pl.col("leader_score")
                + template.score.setup * pl.col("setup_score")
            ).alias("total_score")
        ).sort(["total_score", "ts_code"], descending=[True, False])

        results: list[Candidate] = []
        per_sector: dict[str, int] = {}
        for row in scored.iter_rows(named=True):
            if per_sector.get(row["sector"], 0) >= template.max_per_industry:
                continue
            if len(results) >= template.max_candidates:
                break
            per_sector[row["sector"]] = per_sector.get(row["sector"], 0) + 1
            results.append(
                self._candidate(
                    as_of=as_of,
                    regime=regime,
                    row=row,
                    template=Template.A,
                    setup_type="mainline_pullback",
                    horizon="mid",
                    score=float(row["total_score"]),
                    reasons=[
                        f"行业 {row['sector']} 强度前 {config.industry.top_n}",
                        f"龙头分第 {int(row['leader_rank'])} 名",
                        f"缩量比 {row['vol_ratio_5_20']:.2f}",
                        f"回调 {int(row['pullback_days'])} 天且贴近均线",
                        "企稳 K 线",
                    ],
                    config=config,
                )
            )
        return results

    def _select_b(
        self,
        as_of: date,
        regime: MarketRegime,
        frame: pl.DataFrame,
        sector_by_name: dict[str, SectorStrength],
        config: StrategyConfig,
    ) -> list[Candidate]:
        template = config.template_b
        if regime.indicators.blow_rate > template.blow_rate_max:
            return []
        if regime.indicators.max_board_height < template.max_board_min:
            return []
        eligible_sectors = [
            name
            for name, sector in sector_by_name.items()
            if sector.rank <= 2
            and (
                sector.limit_up_count >= template.min_limit_up_in_sector
                or sector.red_rate >= template.sector_red_rate_min
            )
        ]
        selected = frame.filter(
            pl.col("sector").is_in(eligible_sectors)
            & (pl.col("leader_rank") <= config.leader.top_n_per_industry)
            & (~pl.col("is_follower").fill_null(False))
            & (pl.col("is_limit_up"))
            & pl.col("board_height").is_between(1, 2)
            & (pl.col("vol_ratio_today_20") >= template.volume_ratio_min)
            & pl.col("turnover_rate").is_between(
                template.turnover_min, template.turnover_max
            )
            & pl.col("circ_mv").is_between(
                template.market_cap_min, template.market_cap_max
            )
            & pl.col("close").is_between(template.price_min, template.price_max)
            & (~pl.col("is_st"))
            & (~pl.col("is_suspended"))
            & (~pl.col("one_word"))
            & (pl.col("board_height") <= template.max_board)
            & (pl.col("ret_20d") <= template.max_20d_gain)
        )
        if selected.is_empty():
            return []
        degraded = any(
            flag in regime.degraded_flags
            for flag in ("limit_list_d", "industry_stock_basic")
        )
        scored = selected.with_columns(
            (pl.col("board_height").clip(1, 2) / 2 * 100).alias(
                "board_score"
            ),
            (pl.col("vol_ratio_today_20") / 4 * 100).clip(0, 100).alias(
                "volume_score"
            ),
            pl.col("turnover_rate")
            .is_between(template.turnover_min, template.turnover_max)
            .cast(pl.Float64)
            .mul(100)
            .alias("turnover_score"),
        ).with_columns(
            (
                0.4 * pl.col("board_score")
                + 0.3 * pl.col("volume_score")
                + 0.2 * pl.col("turnover_score")
                + 0.1 * 100
            ).alias("setup_score")
        ).with_columns(
            (
                template.score.leader * pl.col("leader_score")
                + template.score.setup * pl.col("setup_score")
                - (5.0 if degraded else 0.0)
            ).alias("total_score")
        ).sort(["total_score", "ts_code"], descending=[True, False])

        results: list[Candidate] = []
        per_sector: dict[str, int] = {}
        for row in scored.iter_rows(named=True):
            if per_sector.get(row["sector"], 0) >= template.max_per_industry:
                continue
            if len(results) >= template.max_candidates:
                break
            per_sector[row["sector"]] = per_sector.get(row["sector"], 0) + 1
            reasons = [
                f"主线 {row['sector']} 前 2",
                f"龙头分第 {int(row['leader_rank'])} 名",
                f"{int(row['board_height'])} 板，量比 {row['vol_ratio_today_20']:.1f}",
                "未一字板",
            ]
            if degraded:
                reasons.append("B 模板数据降级")
            results.append(
                self._candidate(
                    as_of=as_of,
                    regime=regime,
                    row=row,
                    template=Template.B,
                    setup_type="sentiment_leader",
                    horizon="short",
                    score=float(row["total_score"]),
                    reasons=reasons,
                    config=config,
                )
            )
        return results

    def _candidate(
        self,
        as_of: date,
        regime: MarketRegime,
        row: dict,
        template: Template,
        setup_type: str,
        horizon: str,
        score: float,
        reasons: list[str],
        config: StrategyConfig,
    ) -> Candidate:
        keys = [
            "hfq_close",
            "ma5",
            "ma10",
            "ma20",
            "ma60",
            "ma5_raw",
            "ma10_raw",
            "ma20_raw",
            "ma60_raw",
            "ma20_slope",
            "atr14",
            "atr14_raw",
            "low_5d_raw",
            "low_10d_raw",
            "high_60d_raw",
            "close_to_high_60d",
            "pullback_days",
            "vol_ratio_5_20",
            "vol_ratio_today_20",
            "turnover_rate",
            "circ_mv",
            "board_height",
            "days_since_limit_up",
            "one_word",
            "stabilization",
            "above_ma20",
            "ret_20d",
            "rs_20d_market",
            "close",
        ]
        features = {
            key: float(row[key])
            for key in keys
            if row.get(key) is not None
        }
        return Candidate(
            ts_code=row["ts_code"],
            trade_date=as_of,
            template=template,
            setup_type=setup_type,
            horizon=horizon,
            sector=row["sector"],
            leader_rank=int(row["leader_rank"]),
            leader_score=float(row["leader_score"]),
            score=max(0.0, min(100.0, score)),
            reasons=reasons,
            features=features,
            sentiment_stage=regime.sentiment_stage,
            strategy_version=config.version,
            data_version=regime.data_version,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_screener.py tests/test_registry.py -v
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/screener backend/tests/test_screener.py backend/tests/test_registry.py
git commit -m "feat: add dual-template screener and registry"
```

---

### Task 12: Entry planner

**Files:**
- Create: `backend/app/planning/errors.py`
- Create: `backend/app/planning/entry.py`
- Test: `backend/tests/test_entry.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_entry.py`:

```python
from datetime import date

import pytest

from app.core.calendar import TradingCalendar
from app.core.config import load_strategy_config
from app.domain.enums import EntryType, SentimentStage, Template
from app.domain.models import Candidate, MarketRegime, RegimeIndicators, TemplatePolicy
from app.planning.entry import EntryPlanner
from app.planning.errors import EntryInvalidError


def _calendar() -> TradingCalendar:
    return TradingCalendar(
        [date(2024, 1, d) for d in (2, 3, 4, 5, 6, 7)]
    )


def _regime() -> MarketRegime:
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime="attack",
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={
            Template.A: policy,
            Template.B: policy,
        },
        indicators=RegimeIndicators(
            limit_up_count=80,
            limit_down_count=0,
            max_board_height=5,
            promotion_rate=0.5,
            blow_rate=0.1,
            red_rate=0.6,
            limit_up_premium=0.02,
            breadth=0.6,
            index_close=4000.0,
            index_ma20=3900.0,
            index_ma60=3800.0,
        ),
        data_version="dv_test",
    )


def _candidate(template: Template, features: dict) -> Candidate:
    return Candidate(
        ts_code="000001.SZ",
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={key: float(value) for key, value in features.items()},
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


def test_a_pullback_zone():
    config = load_strategy_config()
    candidate = _candidate(
        Template.A, {"ma20_raw": 10.0, "ma10_raw": 9.9}
    )
    plan = EntryPlanner().plan(
        candidate, candidate.features, _regime(), _calendar(),
        date(2024, 1, 2), config,
    )
    assert plan.type == EntryType.PULLBACK
    assert float(plan.entry_zone_low) == 9.80
    assert float(plan.entry_zone_high) == 10.20
    assert plan.expires_on == date(2024, 1, 5)


def test_b_divergence_and_confirmation():
    config = load_strategy_config()
    base = {
        "ma5_raw": 10.5,
        "ma10_raw": 10.0,
        "board_height": 1,
        "days_since_limit_up": 0,
    }
    candidate = _candidate(Template.B, base)
    plan = EntryPlanner().plan(
        candidate, candidate.features, _regime(), _calendar(),
        date(2024, 1, 2), config,
    )
    assert plan.type == EntryType.DIVERGENCE_LOW

    candidate2 = _candidate(
        Template.B,
        {**base, "board_height": 2},
    )
    plan2 = EntryPlanner().plan(
        candidate2, candidate2.features, _regime(), _calendar(),
        date(2024, 1, 2), config,
    )
    assert plan2.type == EntryType.CONFIRMATION


def test_b_invalid_ma_order():
    config = load_strategy_config()
    candidate = _candidate(
        Template.B,
        {
            "ma5_raw": 10.0,
            "ma10_raw": 10.5,
            "board_height": 1,
            "days_since_limit_up": 0,
        },
    )
    with pytest.raises(EntryInvalidError):
        EntryPlanner().plan(
            candidate, candidate.features, _regime(), _calendar(),
            date(2024, 1, 2), config,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_entry.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement planning errors and entry planner**

`backend/app/planning/errors.py`:

```python
class PlanningError(Exception):
    """Base class for deterministic planning rejections."""


class EntryInvalidError(PlanningError):
    pass


class StopTooWideError(PlanningError):
    pass


class PositionRejectedError(PlanningError):
    pass


class PoorRiskRewardError(PlanningError):
    pass
```

`backend/app/planning/entry.py`:

```python
from __future__ import annotations

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
        if center is None or center <= 0:
            raise EntryInvalidError("MA20 unavailable for A entry")
        low = price(center * (1 - template.ma_zone_pct))
        high = price(center * (1 + template.ma_zone_pct))
        expires_on = calendar.add_trading_days(as_of, template.entry_valid_days)
        return EntryPlan(
            type=EntryType.PULLBACK,
            trigger_price=price(center),
            entry_zone_low=low,
            entry_zone_high=high,
            validity_days=template.entry_valid_days,
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
        board_height = int(features.get("board_height", 0))
        if ma5 is None or ma10 is None:
            raise EntryInvalidError("MA5/MA10 unavailable for B entry")
        if board_height <= 1:
            low = price(ma10 * 0.98)
            high = price(ma5 * 1.02)
            if low >= high:
                raise EntryInvalidError("B divergence zone is invalid")
            entry_type = EntryType.DIVERGENCE_LOW
            trigger = None
            zone_low, zone_high = low, high
        else:
            trigger = price(ma5)
            zone_low = price(ma5 * 0.99)
            zone_high = price(ma5 * 1.01)
            entry_type = EntryType.CONFIRMATION
        expires_on = calendar.add_trading_days(as_of, template.entry_valid_days)
        return EntryPlan(
            type=entry_type,
            trigger_price=trigger,
            entry_zone_low=zone_low,
            entry_zone_high=zone_high,
            validity_days=template.entry_valid_days,
            max_chase_pct=template.max_gap_pct,
            cancel_conditions=[
                "close below stop",
                "close below MA5/MA10 without recovery",
                "sentiment stage turns ebb",
            ],
            expires_on=expires_on,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_entry.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/planning/errors.py backend/app/planning/entry.py backend/tests/test_entry.py
git commit -m "feat: add template-aware entry planner"
```

---

### Task 13: Stop planner

**Files:**
- Create: `backend/app/planning/stop.py`
- Test: `backend/tests/test_stop.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_stop.py`:

```python
from datetime import date

import pytest

from app.core.config import load_strategy_config
from app.domain.enums import EntryType, SentimentStage, Template
from app.domain.models import Candidate, EntryPlan
from app.domain.money import price
from app.planning.errors import StopTooWideError
from app.planning.stop import StopPlanner


def _entry(entry_price: float = 10.0) -> EntryPlan:
    return EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=price(entry_price),
        entry_zone_low=price(entry_price * 0.98),
        entry_zone_high=price(entry_price * 1.02),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=[],
        expires_on=date(2024, 1, 5),
    )


def _candidate(template: Template, features: dict) -> Candidate:
    return Candidate(
        ts_code="000001.SZ",
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={key: float(value) for key, value in features.items()},
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


def test_a_structure_stop():
    config = load_strategy_config()
    candidate = _candidate(
        Template.A,
        {
            "low_10d_raw": 9.5,
            "ma20_raw": 10.5,
            "atr14_raw": 0.5,
        },
    )
    stop = StopPlanner().plan(candidate, _entry(), candidate.features, config)
    assert float(stop.initial_stop) == 9.25
    assert stop.stop_distance_pct == pytest.approx(0.075)
    assert float(stop.r_value) == 0.75


def test_a_too_wide_rejected():
    config = load_strategy_config()
    candidate = _candidate(
        Template.A,
        {
            "low_10d_raw": 8.0,
            "ma20_raw": 10.5,
            "atr14_raw": 0.5,
        },
    )
    with pytest.raises(StopTooWideError):
        StopPlanner().plan(candidate, _entry(), candidate.features, config)


def test_b_clamps_stop_distance():
    config = load_strategy_config()
    wide = _candidate(
        Template.B,
        {
            "low_5d_raw": 9.0,
            "ma5_raw": 9.0,
            "ma10_raw": 8.9,
            "atr14_raw": 0.3,
        },
    )
    stop = StopPlanner().plan(wide, _entry(), wide.features, config)
    assert float(stop.initial_stop) == 9.50

    tight = _candidate(
        Template.B,
        {
            "low_5d_raw": 9.95,
            "ma5_raw": 9.98,
            "ma10_raw": 9.96,
            "atr14_raw": 0.05,
        },
    )
    stop2 = StopPlanner().plan(tight, _entry(), tight.features, config)
    assert float(stop2.initial_stop) == 9.70
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_stop.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the stop planner**

`backend/app/planning/stop.py`:

```python
from __future__ import annotations

from app.core.config import StrategyConfig
from app.domain.enums import Template
from app.domain.models import Candidate, EntryPlan, StopPlan
from app.domain.money import price
from app.planning.errors import StopTooWideError


class StopPlanner:
    def plan(
        self,
        candidate: Candidate,
        entry: EntryPlan,
        features: dict[str, float],
        config: StrategyConfig,
    ) -> StopPlan:
        entry_price = float(entry.entry_price)
        atr = features.get("atr14_raw")
        if atr is None or atr <= 0:
            raise StopTooWideError("ATR unavailable for stop")
        if candidate.template == Template.A:
            support_candidates = [
                features.get("low_10d_raw"),
                features.get("ma20_raw"),
            ]
            stop_min, stop_max = (
                config.template_a.stop_min_pct,
                config.template_a.stop_max_pct,
            )
            buffer = config.template_a.atr_buffer
            time_stop = config.template_a.time_stop_days
            trailing = [
                "move to breakeven at +1R",
                "move to +1R at +2R",
                "trail by close - 1.5*ATR14",
            ]
            template_name = "A"
        else:
            support_candidates = [
                features.get("low_5d_raw"),
                features.get("ma5_raw"),
                features.get("ma10_raw"),
            ]
            stop_min, stop_max = (
                config.template_b.stop_min_pct,
                config.template_b.stop_max_pct,
            )
            buffer = 0.5
            time_stop = config.template_b.time_stop_days
            trailing = [
                "move to breakeven at +1R",
                "move to +1R at +2R",
                "trail by close - 1.5*ATR14 and MA5",
            ]
            template_name = "B"

        supports = [
            float(value)
            for value in support_candidates
            if value is not None and float(value) < entry_price
        ]
        if supports:
            support = max(supports)
            structure_stop = support - buffer * atr
            structure_ref = f"support={support:.2f}"
        else:
            structure_stop = entry_price - 1.5 * atr
            structure_ref = "ATR fallback"
        if structure_stop >= entry_price:
            structure_stop = entry_price - 1.5 * atr
            structure_ref = "ATR fallback"

        distance = (entry_price - structure_stop) / entry_price
        if template_name == "A" and distance > stop_max:
            raise StopTooWideError(
                f"A stop distance {distance:.2%} exceeds {stop_max:.2%}"
            )
        if distance > stop_max:
            structure_stop = entry_price * (1 - stop_max)
            structure_ref += "; clamped to max stop"
        elif distance < stop_min:
            structure_stop = entry_price * (1 - stop_min)
            structure_ref += "; widened to min stop"

        stop = price(structure_stop)
        r_value = price(entry_price - float(stop))
        if float(r_value) <= 0:
            raise StopTooWideError("non-positive R")
        return StopPlan(
            initial_stop=stop,
            stop_distance_pct=float(r_value) / entry_price,
            atr14=atr,
            structure_ref=structure_ref,
            time_stop_days=time_stop,
            trailing_rules=trailing,
            r_value=r_value,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_stop.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/planning/stop.py backend/tests/test_stop.py
git commit -m "feat: add template-aware stop planner"
```

---

### Task 14: Position sizer

**Files:**
- Create: `backend/app/planning/sizing.py`
- Test: `backend/tests/test_sizing.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_sizing.py`:

```python
from datetime import date
from decimal import Decimal

import pytest

from app.core.config import load_strategy_config
from app.domain.enums import EntryType, SentimentStage, Template
from app.domain.models import (
    AccountState,
    Candidate,
    EntryPlan,
    MarketRegime,
    PositionState,
    RegimeIndicators,
    StopPlan,
    TemplatePolicy,
)
from app.planning.errors import PositionRejectedError
from app.planning.sizing import PositionSizer
from app.domain.money import money, price


def _account(
    equity: float = 1_000_000,
    cash: float = 1_000_000,
    peak: float | None = None,
    positions: list | None = None,
) -> AccountState:
    return AccountState(
        equity=Decimal(str(equity)),
        cash=Decimal(str(cash)),
        peak_equity=Decimal(str(peak or equity)),
        positions=positions or [],
    )


def _regime() -> MarketRegime:
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime="attack",
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={Template.A: policy, Template.B: policy},
        indicators=RegimeIndicators(
            limit_up_count=80,
            limit_down_count=0,
            max_board_height=5,
            promotion_rate=0.5,
            blow_rate=0.1,
            red_rate=0.6,
            limit_up_premium=0.02,
            breadth=0.6,
            index_close=4000.0,
            index_ma20=3900.0,
            index_ma60=3800.0,
        ),
        data_version="dv_test",
    )


def _candidate(
    template: Template = Template.A, sector: str = "银行"
) -> Candidate:
    return Candidate(
        ts_code="000001.SZ",
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector=sector,
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={"close": 10.0},
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


def _entry(price_value: float) -> EntryPlan:
    return EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=price(price_value),
        entry_zone_low=price(price_value * 0.98),
        entry_zone_high=price(price_value * 1.02),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=[],
        expires_on=date(2024, 1, 5),
    )


def _stop(entry_price: float, stop_price: float) -> StopPlan:
    return StopPlan(
        initial_stop=price(stop_price),
        stop_distance_pct=(entry_price - stop_price) / entry_price,
        atr14=0.5,
        structure_ref="test",
        time_stop_days=10,
        trailing_rules=[],
        r_value=price(entry_price - stop_price),
    )


def test_base_sizing():
    config = load_strategy_config()
    plan = PositionSizer().size(
        _candidate(), _entry(20.0), _stop(20.0, 18.6), _account(), _regime(), config
    )
    assert plan.shares == 17100
    assert float(plan.position_value) == pytest.approx(342000.0)
    assert plan.position_pct == pytest.approx(0.342)
    assert plan.heat_after == pytest.approx(0.02394, rel=1e-4)


def test_single_position_cap():
    config = load_strategy_config()
    plan = PositionSizer().size(
        _candidate(), _entry(10.0), _stop(10.0, 9.5), _account(), _regime(), config
    )
    assert plan.shares == 40000
    assert plan.position_pct == pytest.approx(0.40)


def test_heat_cap_reduces_size():
    config = load_strategy_config()
    position = PositionState(
        ts_code="000009.SZ",
        sector="地产",
        shares=50000,
        available_shares=50000,
        cost_price=Decimal("10.0"),
        current_stop=Decimal("9.6"),
        current_price=Decimal("11.0"),
        template=Template.A,
    )
    account = _account(positions=[position])
    plan = PositionSizer().size(
        _candidate(), _entry(10.0), _stop(10.0, 9.5), account, _regime(), config
    )
    assert plan.shares == 10000
    assert plan.heat_after == pytest.approx(0.075)


def test_drawdown_and_max_positions_reject():
    config = load_strategy_config()
    with pytest.raises(PositionRejectedError):
        PositionSizer().size(
            _candidate(),
            _entry(20.0),
            _stop(20.0, 18.6),
            _account(equity=1_000_000, peak=1_200_000),
            _regime(),
            config,
        )
    positions = [
        PositionState(
            ts_code=f"00000{i}.SZ",
            sector=f"行业{i}",
            shares=100,
            available_shares=100,
            cost_price=Decimal("10.0"),
            current_stop=Decimal("9.0"),
            current_price=Decimal("10.0"),
            template=Template.A,
        )
        for i in range(3)
    ]
    with pytest.raises(PositionRejectedError):
        PositionSizer().size(
            _candidate(),
            _entry(20.0),
            _stop(20.0, 18.6),
            _account(positions=positions),
            _regime(),
            config,
        )


def test_correlation_above_threshold_rejects():
    config = load_strategy_config()
    position = PositionState(
        ts_code="000009.SZ",
        sector="地产",
        shares=100,
        available_shares=100,
        cost_price=Decimal("10.0"),
        current_stop=Decimal("9.0"),
        current_price=Decimal("10.0"),
        template=Template.A,
    )
    with pytest.raises(PositionRejectedError):
        PositionSizer().size(
            _candidate(),
            _entry(10.0),
            _stop(10.0, 9.5),
            _account(positions=[position]),
            _regime(),
            config,
            correlations={"000009.SZ": 0.90},
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_sizing.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the position sizer**

`backend/app/planning/sizing.py`:

```python
from __future__ import annotations

from app.core.config import StrategyConfig
from app.domain.enums import Template
from app.domain.models import (
    AccountState,
    Candidate,
    EntryPlan,
    MarketRegime,
    PositionPlan,
    StopPlan,
)
from app.domain.money import money
from app.planning.errors import PositionRejectedError


class PositionSizer:
    def size(
        self,
        candidate: Candidate,
        entry: EntryPlan,
        stop: StopPlan,
        account: AccountState,
        regime: MarketRegime,
        config: StrategyConfig,
        correlations: dict[str, float] | None = None,
    ) -> PositionPlan:
        equity = float(account.equity)
        drawdown = account.drawdown_pct
        if drawdown >= config.risk.drawdown_stop_at:
            raise PositionRejectedError("drawdown circuit breaker active")
        drawdown_factor = (
            0.5 if drawdown >= config.risk.drawdown_half_at else 1.0
        )
        risk_pct = (
            config.template_a.risk_pct
            if candidate.template == Template.A
            else config.template_b.risk_pct
        )
        risk_amount = equity * risk_pct * regime.risk_factor * drawdown_factor
        if risk_amount <= 0:
            raise PositionRejectedError("risk factor is zero")

        entry_price = float(entry.entry_price)
        stop_price = float(stop.initial_stop)
        if entry_price <= stop_price:
            raise PositionRejectedError("entry price must exceed stop")
        per_share_risk = (
            entry_price - stop_price
            + entry_price * config.risk.round_trip_cost_pct
        )
        shares = int(risk_amount / per_share_risk // 100 * 100)
        original = shares
        adjustments: list[str] = []
        checks = ["risk_budget"]

        if (
            len(account.positions) + len(account.pending_plans)
            >= config.risk.max_positions
        ):
            raise PositionRejectedError("max positions reached")
        checks.append("max_positions")

        b_count = sum(
            1 for item in account.positions if item.template == Template.B
        ) + sum(
            1
            for item in account.pending_plans
            if item.template == Template.B
        )
        if (
            candidate.template == Template.B
            and b_count >= config.risk.max_b_positions
        ):
            raise PositionRejectedError("max B positions reached")
        checks.append("max_b_positions")

        industry_count = sum(
            1 for item in account.positions if item.sector == candidate.sector
        ) + sum(
            1
            for item in account.pending_plans
            if item.sector == candidate.sector
        )
        if industry_count >= config.risk.max_industry_positions:
            raise PositionRejectedError("max industry positions reached")
        checks.append("industry_concentration")

        if any(
            item.ts_code == candidate.ts_code for item in account.positions
        ) or any(
            item.ts_code == candidate.ts_code
            for item in account.pending_plans
        ):
            raise PositionRejectedError("duplicate position or plan")
        checks.append("duplicate")

        if correlations:
            existing = {
                item.ts_code for item in account.positions
            } | {
                item.ts_code for item in account.pending_plans
            }
            for code, correlation in correlations.items():
                if (
                    code in existing
                    and correlation > config.risk.correlation_threshold
                ):
                    raise PositionRejectedError(
                        "correlation above threshold"
                    )
        checks.append("correlation")

        max_value = equity * config.risk.max_single_position_pct
        shares = min(shares, int(max_value / entry_price // 100 * 100))
        checks.append("single_position_cap")

        max_cash_spend = (
            float(account.cash) - equity * config.risk.min_cash_pct
        )
        if max_cash_spend <= 0:
            raise PositionRejectedError("cash reserve would be breached")
        shares = min(
            shares, int(max_cash_spend / entry_price // 100 * 100)
        )
        checks.append("cash_reserve")

        max_exposure = equity * (1 - config.risk.min_cash_pct)
        shares = min(
            shares,
            int(
                (max_exposure - float(account.positions_value))
                / entry_price
                // 100
                * 100
            ),
        )
        checks.append("total_exposure")

        max_new_risk = equity * config.risk.max_heat - account.heat * equity
        if max_new_risk <= 0:
            raise PositionRejectedError("portfolio heat limit reached")
        shares = min(
            shares,
            int(
                max_new_risk
                / (entry_price - stop_price)
                // 100
                * 100
            ),
        )
        checks.append("portfolio_heat")

        if shares != original:
            adjustments.append(f"shares reduced from {original} to {shares}")
        if shares < 100:
            raise PositionRejectedError("position below one lot")

        position_value = money(entry_price * shares)
        heat_after = (
            account.heat
            + (entry_price - stop_price) * shares / equity
        )
        return PositionPlan(
            template=candidate.template,
            risk_pct=risk_pct,
            risk_amount=money(risk_amount),
            shares=shares,
            position_value=position_value,
            position_pct=float(position_value) / equity,
            heat_before=account.heat,
            heat_after=heat_after,
            constraints_checked=checks,
            adjustments=adjustments,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_sizing.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/planning/sizing.py backend/tests/test_sizing.py
git commit -m "feat: add risk-based position sizer"
```

---

### Task 15: Take-profit planner

**Files:**
- Create: `backend/app/planning/take_profit.py`
- Test: `backend/tests/test_take_profit.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_take_profit.py`:

```python
from datetime import date

import pytest

from app.core.config import load_strategy_config
from app.domain.enums import EntryType, SentimentStage, Template
from app.domain.models import Candidate, EntryPlan, StopPlan
from app.domain.money import price
from app.planning.errors import PoorRiskRewardError
from app.planning.take_profit import TakeProfitPlanner


def _candidate(template: Template, resistance: float | None) -> Candidate:
    features = {"close": 20.0}
    if resistance is not None:
        features["high_60d_raw"] = resistance
    return Candidate(
        ts_code="000001.SZ",
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features=features,
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


def _entry(price_value: float) -> EntryPlan:
    return EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=price(price_value),
        entry_zone_low=price(price_value * 0.98),
        entry_zone_high=price(price_value * 1.02),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=[],
        expires_on=date(2024, 1, 5),
    )


def _stop(entry_price: float, stop_price: float) -> StopPlan:
    return StopPlan(
        initial_stop=price(stop_price),
        stop_distance_pct=(entry_price - stop_price) / entry_price,
        atr14=0.5,
        structure_ref="test",
        time_stop_days=10,
        trailing_rules=[],
        r_value=price(entry_price - stop_price),
    )


def test_a_targets():
    config = load_strategy_config()
    plan = TakeProfitPlanner().plan(
        _candidate(Template.A, resistance=30.0),
        _entry(20.0),
        _stop(20.0, 18.6),
        config,
    )
    assert [float(target.price) for target in plan.targets] == [22.80, 24.20]
    assert plan.targets[0].fraction == pytest.approx(1 / 3, rel=1e-3)


def test_a_poor_risk_reward_rejected():
    config = load_strategy_config()
    with pytest.raises(PoorRiskRewardError):
        TakeProfitPlanner().plan(
            _candidate(Template.A, resistance=21.0),
            _entry(20.0),
            _stop(20.0, 18.6),
            config,
        )


def test_b_target():
    config = load_strategy_config()
    plan = TakeProfitPlanner().plan(
        _candidate(Template.B, resistance=None),
        _entry(10.0),
        _stop(10.0, 9.5),
        config,
    )
    assert float(plan.targets[0].price) == 11.00
    assert plan.targets[0].fraction == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_take_profit.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the take-profit planner**

`backend/app/planning/take_profit.py`:

```python
from __future__ import annotations

from app.core.config import StrategyConfig
from app.domain.enums import Template
from app.domain.models import (
    Candidate,
    EntryPlan,
    StopPlan,
    TakeProfitPlan,
    TakeProfitTarget,
)
from app.domain.money import price
from app.planning.errors import PoorRiskRewardError


class TakeProfitPlanner:
    def plan(
        self,
        candidate: Candidate,
        entry: EntryPlan,
        stop: StopPlan,
        config: StrategyConfig,
    ) -> TakeProfitPlan:
        entry_price = float(entry.entry_price)
        r_value = float(stop.r_value)
        if r_value <= 0:
            raise PoorRiskRewardError("R must be positive")

        if candidate.template == Template.A:
            resistance = candidate.features.get("high_60d_raw")
            if (
                resistance is not None
                and resistance > entry_price
                and resistance - entry_price < 1.5 * r_value
            ):
                raise PoorRiskRewardError(
                    "nearest resistance is closer than 1.5R"
                )
            tp = config.template_a.take_profit
            targets = [
                TakeProfitTarget(
                    r_multiple=2.0,
                    price=price(entry_price + 2 * r_value),
                    fraction=tp.first_fraction,
                ),
                TakeProfitTarget(
                    r_multiple=3.0,
                    price=price(entry_price + 3 * r_value),
                    fraction=tp.second_fraction,
                ),
            ]
            return TakeProfitPlan(
                targets=targets,
                trailing_rule=(
                    "+1R move stop to breakeven; +2R move stop to +1R; "
                    "trail by close - 1.5*ATR14"
                ),
                time_exit_rule=(
                    "exit after 10 trading days without +1R and below entry"
                ),
            )

        tp = config.template_b.take_profit
        targets = [
            TakeProfitTarget(
                r_multiple=2.0,
                price=price(entry_price + 2 * r_value),
                fraction=tp.second_fraction,
            )
        ]
        return TakeProfitPlan(
            targets=targets,
            trailing_rule=(
                "+1R move stop to breakeven; +2R sell half; "
                "trail remainder by MA5/1.5*ATR14"
            ),
            time_exit_rule=(
                "exit after 3 trading days without a new high; "
                "exit immediately on ebb stage"
            ),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_take_profit.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/planning/take_profit.py backend/tests/test_take_profit.py
git commit -m "feat: add take-profit planner"
```

---

### Task 16: Risk governor

**Files:**
- Create: `backend/app/risk/governor.py`
- Test: `backend/tests/test_risk.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_risk.py`:

```python
from datetime import date
from decimal import Decimal

from app.core.config import load_strategy_config
from app.domain.enums import (
    EntryType,
    PlanStatus,
    RiskOutcome,
    SentimentStage,
    Template,
    TrendRegime,
)
from app.domain.models import (
    AccountState,
    Candidate,
    EntryPlan,
    MarketRegime,
    PositionPlan,
    PositionState,
    RegimeIndicators,
    StopPlan,
    TakeProfitPlan,
    TakeProfitTarget,
    TemplatePolicy,
    TradePlan,
)
from app.domain.money import money, price


def _regime() -> MarketRegime:
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime=TrendRegime.ATTACK,
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={
            Template.A: policy,
            Template.B: TemplatePolicy(enabled=True, mode="normal", max_positions=2),
        },
        indicators=RegimeIndicators(
            limit_up_count=80,
            limit_down_count=0,
            max_board_height=5,
            promotion_rate=0.5,
            blow_rate=0.1,
            red_rate=0.6,
            limit_up_premium=0.02,
            breadth=0.6,
            index_close=4000.0,
            index_ma20=3900.0,
            index_ma60=3800.0,
        ),
        data_version="dv_test",
    )


def _account(positions: list | None = None, peak: float = 1_000_000) -> AccountState:
    return AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal(str(peak)),
        positions=positions or [],
    )


def _plan(
    code: str,
    sector: str,
    template: Template = Template.A,
    score: float = 90.0,
    shares: int = 10_000,
    data_version: str = "dv_test",
) -> TradePlan:
    candidate = Candidate(
        ts_code=code,
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector=sector,
        leader_rank=1,
        leader_score=90.0,
        score=score,
        reasons=["test"],
        features={"close": 10.0},
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version=data_version,
    )
    entry = EntryPlan(
        type=EntryType.PULLBACK,
        trigger_price=price(10.0),
        entry_zone_low=price(9.8),
        entry_zone_high=price(10.2),
        validity_days=3,
        max_chase_pct=0.03,
        cancel_conditions=[],
        expires_on=date(2024, 1, 5),
    )
    stop = StopPlan(
        initial_stop=price(9.0),
        stop_distance_pct=0.10,
        atr14=0.5,
        structure_ref="test",
        time_stop_days=10,
        trailing_rules=[],
        r_value=price(1.0),
    )
    position = PositionPlan(
        template=template,
        risk_pct=0.025,
        risk_amount=money(25_000),
        shares=shares,
        position_value=money(10.0 * shares),
        position_pct=10.0 * shares / 1_000_000,
        heat_before=0.0,
        heat_after=shares / 1_000_000,
        constraints_checked=[],
        adjustments=[],
    )
    take_profit = TakeProfitPlan(
        targets=[
            TakeProfitTarget(r_multiple=2.0, price=price(12.0), fraction=0.5)
        ],
        trailing_rule="test",
        time_exit_rule="test",
    )
    return TradePlan(
        candidate=candidate,
        entry_plan=entry,
        stop_plan=stop,
        position_plan=position,
        take_profit_plan=take_profit,
        expires_on=date(2024, 1, 5),
        strategy_version="sv_test",
        data_version=data_version,
        code_version="test",
    )


def test_capacity_and_priority():
    from app.risk.governor import RiskGovernor

    config = load_strategy_config()
    plans = [
        _plan(f"00000{i}.SZ", f"行业{i}", score=90 - i)
        for i in range(4)
    ]
    result = RiskGovernor().apply(plans, _account(), _regime(), config)
    assert sum(1 for plan in result if plan.status == PlanStatus.PENDING_ENTRY) == 3
    assert sum(1 for plan in result if plan.status == PlanStatus.REJECTED) == 1


def test_b_max_and_disabled_template():
    from app.risk.governor import RiskGovernor

    config = load_strategy_config()
    b_plans = [
        _plan(f"00010{i}.SZ", f"行业{i}", template=Template.B)
        for i in range(3)
    ]
    result = RiskGovernor().apply(b_plans, _account(), _regime(), config)
    assert sum(1 for plan in result if plan.status == PlanStatus.PENDING_ENTRY) == 2

    regime = _regime().model_copy(
        update={
            "template_availability": {
                Template.A: TemplatePolicy(
                    enabled=False, mode="observe", max_positions=0
                ),
                Template.B: TemplatePolicy(
                    enabled=False, mode="observe", max_positions=0
                ),
            }
        }
    )
    result = RiskGovernor().apply(b_plans[:1], _account(), regime, config)
    assert result[0].status == PlanStatus.REJECTED
    assert result[0].risk_decision.outcome == RiskOutcome.REJECT


def test_heat_reduces_and_version_mismatch_rejects():
    from app.risk.governor import RiskGovernor

    config = load_strategy_config()
    position = PositionState(
        ts_code="000009.SZ",
        sector="地产",
        shares=50000,
        available_shares=50000,
        cost_price=Decimal("10.0"),
        current_stop=Decimal("9.6"),
        current_price=Decimal("11.0"),
        template=Template.A,
    )
    plan = _plan("000001.SZ", "银行", shares=10_000)
    result = RiskGovernor().apply([plan], _account(positions=[position]), _regime(), config)
    assert result[0].status == PlanStatus.PENDING_ENTRY
    assert result[0].risk_decision.outcome == RiskOutcome.REDUCE
    assert result[0].position_plan.shares == 5000

    bad = _plan("000002.SZ", "银行", data_version="old")
    result = RiskGovernor().apply([bad], _account(), _regime(), config)
    assert result[0].status == PlanStatus.REJECTED


def test_exposure_cap_reduces_total_value():
    from app.risk.governor import RiskGovernor

    config = load_strategy_config()
    plans = [
        _plan(f"00020{i}.SZ", f"行业{i}", shares=40_000)
        for i in range(3)
    ]
    result = RiskGovernor().apply(plans, _account(), _regime(), config)
    approved = [
        plan
        for plan in result
        if plan.status == PlanStatus.PENDING_ENTRY
    ]
    total = sum(
        float(plan.position_plan.position_value) for plan in approved
    )
    assert total <= 900_000
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_risk.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the risk governor**

`backend/app/risk/governor.py`:

```python
from __future__ import annotations

from app.core.config import StrategyConfig
from app.domain.enums import PlanStatus, RiskOutcome, SentimentStage, Template
from app.domain.models import (
    AccountState,
    MarketRegime,
    PendingExposure,
    RiskDecision,
    TradePlan,
)
from app.domain.money import money


class RiskGovernor:
    def apply(
        self,
        plans: list[TradePlan],
        account: AccountState,
        regime: MarketRegime,
        config: StrategyConfig,
    ) -> list[TradePlan]:
        if account.drawdown_pct >= config.risk.drawdown_stop_at:
            return [
                self._reject(plan, "drawdown circuit breaker active")
                for plan in plans
            ]

        ordered = sorted(plans, key=lambda plan: self._priority(plan, regime))
        working = account
        approved_codes: set[str] = set()
        output: list[TradePlan] = []

        for plan in ordered:
            policy = regime.template_availability[plan.candidate.template]
            if not policy.enabled:
                output.append(self._reject(plan, "template disabled"))
                continue
            if plan.data_version != regime.data_version:
                output.append(self._reject(plan, "data version mismatch"))
                continue
            if plan.candidate.ts_code in approved_codes:
                output.append(self._reject(plan, "duplicate plan"))
                continue
            if (
                len(working.positions) + len(working.pending_plans)
                >= config.risk.max_positions
            ):
                output.append(self._reject(plan, "max positions reached"))
                continue
            b_count = sum(
                1
                for item in working.positions
                if item.template == Template.B
            ) + sum(
                1
                for item in working.pending_plans
                if item.template == Template.B
            )
            if (
                plan.candidate.template == Template.B
                and b_count >= config.risk.max_b_positions
            ):
                output.append(self._reject(plan, "max B positions reached"))
                continue
            industry_count = sum(
                1
                for item in working.positions
                if item.sector == plan.candidate.sector
            ) + sum(
                1
                for item in working.pending_plans
                if item.sector == plan.candidate.sector
            )
            if industry_count >= config.risk.max_industry_positions:
                output.append(
                    self._reject(plan, "max industry positions reached")
                )
                continue

            entry_price = float(plan.entry_plan.entry_price)
            stop_price = float(plan.stop_plan.initial_stop)
            equity = float(working.equity)
            max_new_risk = equity * config.risk.max_heat - working.heat * equity
            if max_new_risk <= 0:
                output.append(
                    self._reject(plan, "portfolio heat limit reached")
                )
                continue
            max_shares = int(
                max_new_risk / (entry_price - stop_price) // 100 * 100
            )
            shares = min(plan.position_plan.shares, max_shares)
            current_value = float(working.positions_value) + sum(
                float(item.entry_price * item.shares)
                for item in working.pending_plans
            )
            pending_value = sum(
                float(item.entry_price * item.shares)
                for item in working.pending_plans
            )
            max_value = min(
                equity * config.risk.max_single_position_pct,
                equity * (1 - config.risk.min_cash_pct) - current_value,
                float(working.cash)
                - equity * config.risk.min_cash_pct
                - pending_value,
            )
            if max_value <= 0:
                output.append(
                    self._reject(plan, "cash or exposure limit reached")
                )
                continue
            shares = min(shares, int(max_value / entry_price // 100 * 100))
            if shares < 100:
                output.append(
                    self._reject(
                        plan, "position below one lot after caps"
                    )
                )
                continue

            if shares < plan.position_plan.shares:
                decision = RiskDecision(
                    outcome=RiskOutcome.REDUCE,
                    reasons=["heat cap reduced size"],
                    adjusted_shares=shares,
                )
            else:
                decision = RiskDecision(
                    outcome=RiskOutcome.APPROVE,
                    reasons=["all checks passed"],
                )
            value = money(entry_price * shares)
            position = plan.position_plan.model_copy(
                update={
                    "shares": shares,
                    "position_value": value,
                    "position_pct": float(value) / equity,
                    "heat_after": working.heat
                    + (entry_price - stop_price) * shares / equity,
                }
            )
            approved = plan.model_copy(
                update={
                    "position_plan": position,
                    "risk_decision": decision,
                    "status": PlanStatus.PENDING_ENTRY,
                }
            )
            output.append(approved)
            approved_codes.add(plan.candidate.ts_code)
            working = working.with_pending(
                PendingExposure(
                    ts_code=plan.candidate.ts_code,
                    sector=plan.candidate.sector,
                    entry_price=plan.entry_plan.entry_price,
                    stop_price=plan.stop_plan.initial_stop,
                    shares=shares,
                    template=plan.candidate.template,
                )
            )
        return output

    def _priority(self, plan: TradePlan, regime: MarketRegime) -> tuple:
        score = -plan.candidate.score
        if (
            regime.sentiment_stage == SentimentStage.START
            and plan.candidate.template == Template.B
        ):
            return (0, score)
        return (1, score)

    def _reject(self, plan: TradePlan, reason: str) -> TradePlan:
        return plan.model_copy(
            update={
                "status": PlanStatus.REJECTED,
                "risk_decision": RiskDecision(
                    outcome=RiskOutcome.REJECT, reasons=[reason]
                ),
            }
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_risk.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/risk/governor.py backend/tests/test_risk.py
git commit -m "feat: add portfolio risk governor"
```

---

### Task 17: Pipeline orchestrator

**Files:**
- Create: `backend/app/pipeline.py`
- Test: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_pipeline.py`:

```python
from datetime import date, timedelta
from decimal import Decimal

from app.core.calendar import TradingCalendar
from app.core.config import load_strategy_config
from app.domain.enums import SentimentStage, Template, TrendRegime
from app.domain.models import (
    AccountState,
    Candidate,
    MarketRegime,
    RegimeIndicators,
    TemplatePolicy,
)
from app.pipeline import TradingPipeline


class DummyProvider:
    pass


def _calendar() -> TradingCalendar:
    return TradingCalendar(
        [date(2024, 1, 2) + timedelta(days=i) for i in range(10)]
    )


def _regime() -> MarketRegime:
    policy = TemplatePolicy(enabled=True, mode="normal", max_positions=3)
    return MarketRegime(
        trade_date=date(2024, 1, 2),
        trend_regime=TrendRegime.ATTACK,
        sentiment_stage=SentimentStage.FERMENT,
        risk_factor=1.0,
        template_availability={
            Template.A: policy,
            Template.B: TemplatePolicy(enabled=True, mode="normal", max_positions=2),
        },
        indicators=RegimeIndicators(
            limit_up_count=80,
            limit_down_count=0,
            max_board_height=5,
            promotion_rate=0.5,
            blow_rate=0.1,
            red_rate=0.6,
            limit_up_premium=0.02,
            breadth=0.6,
            index_close=4000.0,
            index_ma20=3900.0,
            index_ma60=3800.0,
        ),
        data_version="dv_test",
    )


def _candidate(template: Template = Template.A) -> Candidate:
    return Candidate(
        ts_code="000001.SZ",
        trade_date=date(2024, 1, 2),
        template=template,
        setup_type="test",
        horizon="mid" if template == Template.A else "short",
        sector="银行",
        leader_rank=1,
        leader_score=90.0,
        score=90.0,
        reasons=["test"],
        features={
            "ma20_raw": 10.0,
            "ma5_raw": 10.5,
            "ma10_raw": 10.0,
            "board_height": 0.0,
            "atr14_raw": 0.5,
            "low_10d_raw": 9.5,
            "low_5d_raw": 9.7,
            "high_60d_raw": 30.0,
            "close": 10.2,
        },
        sentiment_stage=SentimentStage.FERMENT,
        strategy_version="sv_test",
        data_version="dv_test",
    )


def test_pipeline_with_overrides_produces_plan():
    config = load_strategy_config()
    account = AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal("1000000"),
    )
    pipeline = TradingPipeline(provider=DummyProvider(), config=config)
    result = pipeline.run(
        as_of=date(2024, 1, 2),
        account=account,
        template_mode="dual",
        regime_override=_regime(),
        candidates_override=[_candidate()],
        calendar_override=_calendar(),
    )
    assert result.regime.sentiment_stage == SentimentStage.FERMENT
    assert len(result.plans) == 1
    assert result.plans[0].status.value == "PENDING_ENTRY"
    assert result.plans[0].risk_decision.outcome.value == "approve"


def test_pipeline_template_filter():
    config = load_strategy_config()
    account = AccountState(
        equity=Decimal("1000000"),
        cash=Decimal("1000000"),
        peak_equity=Decimal("1000000"),
    )
    pipeline = TradingPipeline(provider=DummyProvider(), config=config)
    result = pipeline.run(
        as_of=date(2024, 1, 2),
        account=account,
        template_mode="A",
        regime_override=_regime(),
        candidates_override=[
            _candidate(Template.A),
            _candidate(Template.B),
        ],
        calendar_override=_calendar(),
    )
    assert [plan.candidate.template for plan in result.plans] == [Template.A]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/test_pipeline.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the pipeline**

`backend/app/pipeline.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl
from pydantic import BaseModel

from app.core.calendar import TradingCalendar
from app.core.config import StrategyConfig
from app.core.errors import DataNotReadyError, DataQualityError
from app.data.industry_map import IndustryMap
from app.data.interfaces import MarketDataProvider
from app.data.quality import DataQualityGate
from app.data.universe import UniverseBuilder
from app.domain.enums import SentimentStage, Template
from app.domain.models import (
    AccountState,
    Candidate,
    MarketRegime,
    SectorStrength,
    TradePlan,
)
from app.indicators.features import FeatureBuilder
from app.planning.entry import EntryPlanner
from app.planning.errors import PlanningError
from app.planning.sizing import PositionSizer
from app.planning.stop import StopPlanner
from app.planning.take_profit import TakeProfitPlanner
from app.regime.industry import IndustryStrengthService
from app.regime.leader import LeaderScoreService
from app.regime.market import MarketRegimeService
from app.regime.sentiment import SentimentService, SentimentStageClassifier
from app.risk.governor import RiskGovernor
from app.screener.base import ScreenerStrategy
from app.screener.dual_template import DualTemplateScreener


class RejectedCandidate(BaseModel):
    candidate: Candidate
    reason: str


class PipelineResult(BaseModel):
    as_of: date
    regime: MarketRegime
    sectors: list[SectorStrength]
    candidates: list[Candidate]
    plans: list[TradePlan]
    rejected: list[RejectedCandidate]


@dataclass
class TradingPipeline:
    provider: MarketDataProvider
    config: StrategyConfig
    screener: ScreenerStrategy | None = None
    feature_builder: FeatureBuilder | None = None
    regime_service: MarketRegimeService | None = None
    industry_service: IndustryStrengthService | None = None
    leader_service: LeaderScoreService | None = None
    sentiment_service: SentimentService | None = None
    stage_classifier: SentimentStageClassifier | None = None
    entry_planner: EntryPlanner | None = None
    stop_planner: StopPlanner | None = None
    sizer: PositionSizer | None = None
    take_profit_planner: TakeProfitPlanner | None = None
    risk_governor: RiskGovernor | None = None
    code_version: str = "dev"

    def __post_init__(self) -> None:
        self.screener = self.screener or DualTemplateScreener()
        self.feature_builder = self.feature_builder or FeatureBuilder()
        self.regime_service = self.regime_service or MarketRegimeService()
        self.industry_service = (
            self.industry_service or IndustryStrengthService()
        )
        self.leader_service = self.leader_service or LeaderScoreService()
        self.sentiment_service = self.sentiment_service or SentimentService()
        self.stage_classifier = (
            self.stage_classifier or SentimentStageClassifier()
        )
        self.entry_planner = self.entry_planner or EntryPlanner()
        self.stop_planner = self.stop_planner or StopPlanner()
        self.sizer = self.sizer or PositionSizer()
        self.take_profit_planner = (
            self.take_profit_planner or TakeProfitPlanner()
        )
        self.risk_governor = self.risk_governor or RiskGovernor()

    def run(
        self,
        as_of: date,
        account: AccountState,
        template_mode: str = "dual",
        regime_override: MarketRegime | None = None,
        candidates_override: list[Candidate] | None = None,
        calendar_override: TradingCalendar | None = None,
        data_version: str | None = None,
    ) -> PipelineResult:
        if candidates_override is not None and regime_override is not None:
            calendar = calendar_override or TradingCalendar(
                [as_of + timedelta(days=i) for i in range(15)]
            )
            candidates = self._filter_mode(candidates_override, template_mode)
            plans, rejected = self._build_plans(
                candidates, account, regime_override, calendar, as_of
            )
            return PipelineResult(
                as_of=as_of,
                regime=regime_override,
                sectors=regime_override.sector_ranking,
                candidates=candidates,
                plans=plans,
                rejected=rejected,
            )

        start = as_of - timedelta(days=400)
        bars = self.provider.load_bars(start, as_of)
        if bars.is_empty():
            raise DataNotReadyError(f"bars missing for {as_of}")
        trade_cal = self.provider.load_trade_cal(start, as_of)
        calendar = TradingCalendar.from_trade_cal(trade_cal)
        stocks = self.provider.load_stock_basic(as_of)
        namechange = self.provider.load_namechange()
        industry_map = IndustryMap.from_sources(
            stocks,
            self.provider.load_index_member(),
            self.provider.load_index_classify(),
            as_of,
            self.config.industry.source,
        )
        sector_frame = industry_map.to_frame()

        dates = sorted(
            bars.filter(pl.col("trade_date") <= as_of)["trade_date"]
            .unique()
            .to_list()
        )[-5:]
        features_history = pl.concat(
            [
                self.feature_builder.build(bars, trade_date)
                for trade_date in dates
            ]
        ).join(sector_frame, on="ts_code", how="left")
        features_asof = features_history.filter(
            pl.col("trade_date") == as_of
        )
        universe = UniverseBuilder(calendar).build(
            as_of, features_asof, stocks, namechange, self.config
        )
        breadth = (
            universe.filter(pl.col("above_ma20")).height / universe.height
            if not universe.is_empty()
            else 0.0
        )
        sectors = self.industry_service.compute(
            features_history, as_of, self.config.industry
        )

        sentiment_history = self.sentiment_service.compute_history(bars)
        classified = self.stage_classifier.classify_series(
            sentiment_history, self.config
        )
        stage_row = classified.filter(pl.col("trade_date") == as_of)
        if stage_row.is_empty():
            raise DataNotReadyError(f"sentiment data missing for {as_of}")
        sentiment_row = stage_row.row(0, named=True)
        sentiment_quality = DataQualityGate().check_sentiment(
            {"trade_date": as_of, **sentiment_row}
        )
        if not sentiment_quality.passed:
            raise DataQualityError("; ".join(sentiment_quality.failures))
        stage = SentimentStage(sentiment_row["sentiment_stage"])

        index_code = self.config.environment.index_code
        index_bars = self.provider.load_index_bars(index_code, start, as_of)
        degraded: list[str] = []
        if index_bars.is_empty():
            index_code = self.config.environment.fallback_index_code
            index_bars = self.provider.load_index_bars(index_code, start, as_of)
            degraded.append("index_fallback")
        if namechange.is_empty():
            degraded.append("st_namechange_missing")
        if industry_map.degraded:
            degraded.append("industry_stock_basic")
        if self.provider.load_limit_list(as_of, as_of).is_empty():
            degraded.append("limit_list_d")

        regime = self.regime_service.compute(
            as_of=as_of,
            index_bars=index_bars,
            sentiment_row=sentiment_row,
            stage=stage,
            breadth=breadth,
            sector_ranking=sectors,
            degraded_flags=degraded,
            data_version=data_version or f"dv_{as_of.isoformat()}",
            config=self.config,
        )
        leaders = self.leader_service.compute(universe, sectors, self.config)
        candidates = self.screener.generate(
            as_of, regime, universe, sectors, leaders, self.config
        )
        candidates = self._filter_mode(candidates, template_mode)
        plans, rejected = self._build_plans(
            candidates, account, regime, calendar, as_of, bars=bars
        )
        return PipelineResult(
            as_of=as_of,
            regime=regime,
            sectors=sectors,
            candidates=candidates,
            plans=plans,
            rejected=rejected,
        )

    def _filter_mode(
        self, candidates: list[Candidate], template_mode: str
    ) -> list[Candidate]:
        if template_mode == "A":
            return [c for c in candidates if c.template == Template.A]
        if template_mode == "B":
            return [c for c in candidates if c.template == Template.B]
        if template_mode == "dual":
            return list(candidates)
        raise ValueError(f"unknown template_mode: {template_mode}")

    def _correlations(
        self,
        bars: pl.DataFrame,
        candidate_code: str,
        existing_codes: list[str],
        as_of: date,
    ) -> dict[str, float]:
        if not existing_codes:
            return {}
        frame = (
            bars.filter(pl.col("trade_date") <= as_of)
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (
                    pl.col("close")
                    / pl.col("close").shift(1).over("ts_code")
                    - 1
                ).alias("ret")
            )
        )
        candidate = (
            frame.filter(pl.col("ts_code") == candidate_code)
            .select(["trade_date", "ret"])
            .tail(60)
            .rename({"ret": "candidate_ret"})
        )
        result: dict[str, float] = {}
        for code in existing_codes:
            other = (
                frame.filter(pl.col("ts_code") == code)
                .select(["trade_date", "ret"])
                .tail(60)
                .rename({"ret": "other_ret"})
            )
            joined = candidate.join(
                other, on="trade_date", how="inner"
            ).drop_nulls()
            if joined.height < 20:
                continue
            value = joined.select(
                pl.corr("candidate_ret", "other_ret")
            ).item()
            if value is not None:
                result[code] = float(value)
        return result

    def _build_plans(
        self,
        candidates: list[Candidate],
        account: AccountState,
        regime: MarketRegime,
        calendar: TradingCalendar,
        as_of: date,
        bars: pl.DataFrame | None = None,
    ) -> tuple[list[TradePlan], list[RejectedCandidate]]:
        plans: list[TradePlan] = []
        rejected: list[RejectedCandidate] = []
        for candidate in candidates:
            try:
                entry = self.entry_planner.plan(
                    candidate,
                    candidate.features,
                    regime,
                    calendar,
                    as_of,
                    self.config,
                )
                stop = self.stop_planner.plan(
                    candidate, entry, candidate.features, self.config
                )
                correlations = None
                if bars is not None:
                    correlations = self._correlations(
                        bars,
                        candidate.ts_code,
                        [item.ts_code for item in account.positions],
                        as_of,
                    )
                position = self.sizer.size(
                    candidate,
                    entry,
                    stop,
                    account,
                    regime,
                    self.config,
                    correlations=correlations,
                )
                take_profit = self.take_profit_planner.plan(
                    candidate, entry, stop, self.config
                )
                plans.append(
                    TradePlan(
                        candidate=candidate,
                        entry_plan=entry,
                        stop_plan=stop,
                        position_plan=position,
                        take_profit_plan=take_profit,
                        expires_on=entry.expires_on,
                        strategy_version=self.config.version,
                        data_version=regime.data_version,
                        code_version=self.code_version,
                    )
                )
            except PlanningError as exc:
                rejected.append(
                    RejectedCandidate(candidate=candidate, reason=str(exc))
                )
        plans = self.risk_governor.apply(plans, account, regime, self.config)
        return plans, rejected
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/test_pipeline.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: orchestrates regime, screener, planners, and risk"
```

---

### Task 18: CLI commands

**Files:**
- Create: `backend/app/cli.py`
- Test: `backend/tests/test_cli.py`

- [ ] **Step 1: Write the failing smoke test**

`backend/tests/test_cli.py`:

```python
from typer.testing import CliRunner

from app.cli import app


def test_cli_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sync" in result.stdout
    assert "run" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
uv run pytest tests/test_cli.py -v
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the CLI**

`backend/app/cli.py`:

```python
from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from pathlib import Path

import typer

from app.core.config import BACKEND_ROOT, load_app_config, load_strategy_config
from app.core.dates import parse_date
from app.data.assemble import DataAssembler
from app.data.provider import ParquetMarketDataProvider
from app.data.quality import DataQualityGate
from app.data.store import ParquetStore
from app.data.sync import SyncService
from app.data.tushare_client import TushareClient
from app.domain.models import AccountState
from app.pipeline import TradingPipeline

app = typer.Typer(help="A-share data and dual-template planning CLI")


def _store() -> ParquetStore:
    config = load_app_config()
    data = config.data
    return ParquetStore(
        BACKEND_ROOT / data.raw_dir,
        BACKEND_ROOT / data.processed_dir,
        BACKEND_ROOT / data.regime_dir,
    )


@app.command()
def sync(
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
    force: bool = typer.Option(False),
) -> None:
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise typer.BadParameter("TUSHARE_TOKEN is not set")
    config = load_app_config()
    service = SyncService(
        TushareClient(
            token, rate_limit_per_min=config.data.rate_limit_per_min
        ),
        _store(),
    )
    results = service.backfill(parse_date(start), parse_date(end), force=force)
    typer.echo(f"synced {len(results)} trading days")


@app.command()
def assemble(day: str = typer.Option(..., help="YYYY-MM-DD")) -> None:
    bars = DataAssembler(_store()).build_day(parse_date(day))
    typer.echo(f"assembled {bars.height} rows for {day}")


@app.command()
def quality(day: str = typer.Option(..., help="YYYY-MM-DD")) -> None:
    trade_date = parse_date(day)
    store = _store()
    bars = store.read_processed("bars", trade_date, trade_date)
    provider = ParquetMarketDataProvider(store)
    stocks = provider.load_stock_basic(trade_date)
    report = DataQualityGate().check(trade_date, bars, stocks.height)
    typer.echo(f"passed={report.passed} coverage={report.coverage:.2%}")
    for failure in report.failures:
        typer.echo(f"FAIL: {failure}")
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def run(
    day: str = typer.Option(..., help="YYYY-MM-DD"),
    equity: float = typer.Option(1_000_000.0),
    cash: float = typer.Option(1_000_000.0),
    template_mode: str = typer.Option("dual"),
    output: str | None = typer.Option(None),
) -> None:
    trade_date = parse_date(day)
    config = load_strategy_config()
    pipeline = TradingPipeline(
        provider=ParquetMarketDataProvider(_store()), config=config
    )
    account = AccountState(
        equity=Decimal(str(equity)),
        cash=Decimal(str(cash)),
        peak_equity=Decimal(str(equity)),
    )
    result = pipeline.run(
        as_of=trade_date, account=account, template_mode=template_mode
    )
    payload = result.model_dump_json(indent=2)
    if output:
        Path(output).write_text(payload, encoding="utf-8")
    else:
        typer.echo(payload)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend
uv run pytest tests/test_cli.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/cli.py backend/tests/test_cli.py
git commit -m "feat: add sync, assemble, quality, and run CLI commands"
```

---

### Task 19: Documentation and M1/M2 acceptance

**Files:**
- Create: `README.md`
- Create: `.env.example`
- Test: run the full suite and ruff.

- [ ] **Step 1: Write the README**

`README.md`:

```markdown
# A-Share Trading Assistant

Personal A-share decision-support backend. M1/M2 implement:

- Tushare sync to Parquet + DuckDB
- Data quality gates
- Point-in-time universe and feature computation
- Market sentiment cycle, industry strength, leader score
- Dual-template screener (A: mainline pullback, B: sentiment leader)
- Entry, stop, position sizing, take-profit, and portfolio risk governor
- CLI: `sync`, `assemble`, `quality`, `run`

## Setup

```bash
cd backend
uv sync --dev
cp ../.env.example .env   # or export TUSHARE_TOKEN
```

## Commands

```bash
# historical backfill
uv run python -m app.cli sync --start 2018-01-01 --end 2024-12-31

# assemble one day
uv run python -m app.cli assemble --day 2024-01-02

# quality gate
uv run python -m app.cli quality --day 2024-01-02

# run planning (after data exists)
uv run python -m app.cli run --day 2024-01-02 --equity 1000000 --cash 1000000
```

## Tests

```bash
cd backend
uv run pytest
uv run ruff check app tests
```

## Safety

- No automatic order placement.
- v1 planning only; execution is manual.
- Backtest engine, scheduler, web dashboard, and broker adapters are M3–M6.
```

`.env.example`:

```text
TUSHARE_TOKEN=
```

- [ ] **Step 2: Run the full test suite**

```bash
cd backend
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Run ruff**

```bash
cd backend
uv run ruff check app tests
```

Expected: no errors. Fix any lint issues before committing.

- [ ] **Step 4: Commit**

```bash
git add README.md .env.example
git commit -m "docs: document M1/M2 data and planning workflow"
```

---

## Self-Review Notes

- **Spec coverage:** M1 data sync/store/quality/universe/features and M2 regime/industry/leader/screener/entry/stop/sizing/take-profit/risk/CLI are covered. Backtest, scheduler, dashboard, and broker adapters are explicitly out of scope and belong to M3–M6.
- **No placeholders:** Each task contains concrete files, code, commands, and expected results.
- **Type consistency:** `MarketRegime`, `Candidate`, `EntryPlan`, `StopPlan`, `PositionPlan`, `TakeProfitPlan`, and `TradePlan` are defined in Task 1 and reused unchanged. `ParquetStore`, `ParquetMarketDataProvider`, and `ScreenerStrategy` signatures are consistent across tasks.
- **M2 simplifications (documented, not hidden):**
  - B-template first-board ordering uses leader score + board height + volume; exact `limit_list_d.first_time` ordering is deferred to M3.
  - Correlation is checked against existing positions; pairwise correlation among same-day approved plans is deferred to M3.
  - B-template intraday triggers use daily approximations; minute-level execution belongs to M3.
- **Known follow-ups for M3:** run the same pipeline over historical dates, build the conservative fill engine, add walk-forward/ablation metrics, and close the three M2 simplifications above. Do not add them to M1/M2 to keep this plan executable.

