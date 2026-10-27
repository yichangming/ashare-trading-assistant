# A-Share Trading Assistant

Personal A-share decision-support backend. It syncs market data, builds a
point-in-time feature store, classifies the market environment, screens for
two setup types, and emits a risk-reviewed trade plan for manual execution.

This repository implements **M1 (data foundation)** and **M2 (dual-template
planning)** only.

| Milestone | Status | Scope |
| --- | --- | --- |
| M1 | Implemented | Tushare sync to Parquet, manifests, DuckDB queries, quality gates, point-in-time universe, features |
| M2 | Implemented | Sentiment/regime, industry strength, leader score, dual-template screener, entry/stop/sizing/take-profit, portfolio risk governor, pipeline, CLI |
| M3 | Out of scope | Backtest engine, versioned reference snapshots, historical replay |
| M4 | Out of scope | Scheduler and reports |
| M5 | Out of scope | Web dashboard |
| M6 | Out of scope | Broker adapters and automated execution |

**This system never places orders.** v1 is planning only; every entry, stop,
and exit is executed manually. It is personal research software, not
investment advice.

## Requirements and setup

- Python 3.11+ (`pyproject.toml` allows 3.11 through 3.13; the checked-in
  `.python-version` is 3.12).
- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- A Tushare Pro token with permission for the endpoints listed in
  [Data prerequisites](#data-prerequisites). Some permissions may be
  degraded or absent; the CLI reports degraded endpoints as warnings, and
  critical endpoints fail the command.

```bash
cd backend
uv sync --dev

# Copy the template and fill in the token. Tushare commands load either location.
cp ../.env.example .env          # backend/.env
# or, from the repository root:
# cp .env.example .env           # repository-root .env
```

Set the token in `.env`:

```text
TUSHARE_TOKEN=your_tushare_pro_token
```

Alternatively export it in the shell:

```bash
export TUSHARE_TOKEN=your_tushare_pro_token
```

For the Tushare commands, the CLI loads the repository-root `.env` and then
`backend/.env` with `override=False`, so an exported `TUSHARE_TOKEN` always
wins, and the repository-root file wins over `backend/.env` when both define
the variable. Only `sync` and `calendar` need the token; `assemble`,
`assemble-range`, `quality`, `run`, and `account` work offline against the
local store.

Default configuration:

- `backend/configs/app.yaml` chooses data directories (`data/raw`,
  `data/processed`, `data/regime`, all relative to `backend/`), the Tushare
  provider, and the API rate limit. It also defines `data.history_start`,
  but that value is currently unused: `sync` and `calendar` require explicit
  `--start`/`--end` arguments.
- `backend/configs/strategies/dual_template_v1.yaml` contains the version
  1.1 strategy thresholds, weights, entry/stop/sizing parameters, and risk
  limits. It is validated strictly at load time.
- All generated data under `backend/data/` is gitignored.

## Fresh-store workflow

Run these steps in order. The pipeline fails closed on missing critical
inputs (empty bars, uncovered calendar, insufficient index history, a failed
quality gate), but a bars history shorter than ~400 days is not itself an
error: features can be mostly null and the run can legitimately produce no
candidates. Assemble enough history before relying on the output.

The example uses a run day of 2024-12-31 and a forward calendar end of
2025-02-14 (45 calendar days later). Replace those example dates with your
target dates. The run day must be an open trading day.

```bash
cd backend

# 1. Raw daily data plus references. Requires TUSHARE_TOKEN.
#    Critical endpoints fail the command; optional endpoints are reported.
uv run python -m app.cli sync --start 2018-01-01 --end 2024-12-31

# 2. Merge reference data, including a forward trade_cal.
#    This does not sync daily bars. The end must be at least
#    RUN_DAY + the forward window required by the pipeline; +45 calendar
#    days or more is recommended.
uv run python -m app.cli calendar --start 2018-01-01 --end 2025-02-14

# 3. Build processed bars/index_bars for the whole period. The pipeline
#    loads ~400 calendar days of history ending at the run day, so the
#    processed store must contain enough history before RUN_DAY.
uv run python -m app.cli assemble-range --start 2018-01-01 --end 2024-12-31

# 4. Inspect the quality gate for the run day (suspension-aware).
uv run python -m app.cli quality --day 2024-12-31

# 5. First planning run. With --init-account the account file is created
#    only after a successful run.
uv run python -m app.cli run \
  --day 2024-12-31 \
  --equity 1000000 \
  --cash 1000000 \
  --account-file data/account.json \
  --init-account
```

For a subsequent trading day, sync and assemble the new day, keep a
forward-extended calendar, and reuse the account file:

```bash
uv run python -m app.cli sync --start 2025-01-02 --end 2025-01-02
uv run python -m app.cli calendar --start 2018-01-01 --end 2025-02-16
uv run python -m app.cli assemble-range --start 2025-01-02 --end 2025-01-02
uv run python -m app.cli quality --day 2025-01-02
uv run python -m app.cli run --day 2025-01-02 --account-file data/account.json
```

Inspect or clear the persisted pending plans:

```bash
uv run python -m app.cli account --account-file data/account.json
uv run python -m app.cli account --account-file data/account.json --clear-pending
```

### Step details and failure modes

- **`sync`** backfills every open day in `--start..--end` from the persisted
  `trade_cal`, after a strict reference refresh of `trade_cal`,
  `stock_basic`, and optional references. Days with an existing manifest are
  skipped unless `--force` is passed. A `trade_cal` or `stock_basic` fetch
  failure aborts the command. Per-day critical endpoints (`daily`,
  `adj_factor`, `daily_basic`, `stk_limit`, `index_daily`, plus `suspend_d`
  fetch errors) fail the command; `limit_list_d` and optional references are
  reported as warnings. The command prints the number of synced trading
  days.
- **`calendar`** refreshes and merges reference data for the requested
  range, including `trade_cal`, without syncing daily bars. It uses a
  non-strict calendar refresh: if the `trade_cal` fetch fails but a
  persisted calendar already exists, the failure is reported as a warning
  and the merge continues. Coverage is then checked inclusively against the
  persisted calendar rows; if the calendar does not span both `--start` and
  `--end`, the command exits non-zero with guidance to backfill an earlier
  or later window. Use a range with a little margin, because requested
  boundaries may be non-trading dates.
- **`assemble-range`** reads open days from the persisted `trade_cal` and
  assembles `bars` and `index_bars` for each one. It never visits dates after
  `--end` or after today (Asia/Shanghai), skips open days with no raw
  `daily` partition, reports skipped days, and exits non-zero if any
  assembly fails. It also exits non-zero when the calendar does not span
  `--start` through the effective end (the earlier of `--end` and today), or
  when the entire requested range is in the future. `assemble --day` is the
  single-day equivalent and fails if that day's raw daily data is missing.
- **`quality`** loads the processed bars for `--day` plus `stock_basic` and
  the raw `suspend_d` partition, prints `passed`, coverage, warning codes,
  and failure codes, and exits non-zero when the gate fails. It does not
  fix data; re-sync or repair the source and re-assemble.
- **`run`** requires the run day's processed bars, processed index bars,
  reference data, and a calendar that covers both history and the forward
  entry-expiry window. Empty or missing bars, an empty `trade_cal` or
  `stock_basic`, an uncovered calendar, insufficient index history, or a
  failed quality/sentiment gate raises a clean error rather than producing a
  partial plan. A merely short (non-empty) bars history can instead yield no
  candidates. Missing optional references such as `namechange` degrade the
  run with flags instead of aborting it.

## What `run` does

`run` executes one end-to-end planning pass in this order:

1. Load or initialize the account state, prune expired pending plans, and
   latch drawdown state at the configured 10% half-risk and 15% stop levels.
2. Run the suspension-aware quality gate on the as-of bars.
3. Build point-in-time features and the tradable universe.
4. Compute industry strength.
5. Compute the sentiment stage.
6. Compute the market regime from the sentiment stage, industry ranking,
   breadth, and index trend.
7. Compute per-industry leader scores.
8. Run the dual-template screener (A: mainline pullback, B: sentiment
   leader), then filter by `--template-mode`.
9. Build entry, initial stop, fixed-risk position sizing, and take-profit
   plans for each candidate.
10. Run the portfolio risk governor, which enforces position slots,
    per-template caps, the single-position cap
    (`max_single_position_pct`), industry concentration, cash reserve,
    total exposure, portfolio heat, duplicate-plan checks, drawdown halt,
    and correlation limits.
11. Emit a JSON result and, when `--account-file` is supplied, persist the
    updated account state atomically.

Approved plans have status `PENDING_ENTRY`. Plans rejected by the risk
governor have status `REJECTED` with a `risk_decision` reason; candidates
rejected earlier while building entry/stop/sizing/take-profit plans appear
in the separate `rejected` list. Nothing is transmitted anywhere. Execution
is manual.

Output and persistence behavior:

- Without `--output`, the full JSON result (`as_of`, `regime`, `sectors`,
  `candidates`, `plans`, `rejected`, and `account`) is written to stdout.
  With `--output PATH`, it is written to that file and a summary still goes
  to stderr. `--output` and `--account-file` must be different paths. When
  an account file is used, the emitted `account` includes the newly
  approved pending plans that were persisted.
- With no `--account-file`, the account is ephemeral: it is built from
  `--equity`/`--cash`/`--peak-equity` for this run only and is not saved.
- If `--account-file` exists, that file is authoritative. Passing
  `--equity`, `--cash`, or `--peak-equity` that differ from the stored values
  prints a warning and the CLI values are ignored. Use
  `--init-account` only when the file is missing. It builds the account from
  the provided values (peak equity defaults to equity and is never below
  equity), but the file is written only after a successful pipeline run; a
  failed run leaves no account file.
- Approved same-day plans are merged into the account's pending plans,
  de-duplicated by `ts_code`, and carry their `expires_on` date. Pending
  plans with `expires_on < run day` are pruned on later runs. Legacy pending
  entries with no `expires_on` are retained; clear them manually with
  `account --clear-pending`.
- `--restart` requires an existing account file and clears the
  `trading_halted` drawdown latch before the run. It is not a permanent
  override: if current drawdown is still at or above the 15% stop threshold,
  the pipeline immediately re-latches trading and rejects every plan. This
  is intentional and requires manual review. The 10% half-risk latch is
  preserved by `--restart`; it clears on a later run once drawdown falls
  below 5%.
- `--template-mode` accepts the case-sensitive values `A`, `B`, or `dual`
  (default). It filters screener output; it does not override regime policy
  or risk limits.

Example with a subset of options:

```bash
uv run python -m app.cli run \
  --day 2024-12-31 \
  --account-file data/account.json \
  --template-mode A \
  --output reports/2024-12-31.json
```

## Command reference

All commands are invoked from `backend/` as
`uv run python -m app.cli <command> [options]`. Run
`uv run python -m app.cli <command> --help` for the same information in the
terminal.

| Command | Flags | Purpose | Token |
| --- | --- | --- | --- |
| `sync` | `--start YYYY-MM-DD` (required), `--end YYYY-MM-DD` (required), `--force/--no-force` | Backfill raw daily data and refresh references; skip existing days unless forced | Yes |
| `calendar` | `--start YYYY-MM-DD` (required), `--end YYYY-MM-DD` (required) | Merge reference data and a forward `trade_cal`; verify inclusive calendar coverage | Yes |
| `assemble` | `--day YYYY-MM-DD` (required) | Build processed `bars`/`index_bars` for one day from raw partitions | No |
| `assemble-range` | `--start YYYY-MM-DD` (required), `--end YYYY-MM-DD` (required) | Build processed `bars`/`index_bars` for every open day in range | No |
| `quality` | `--day YYYY-MM-DD` (required) | Run the suspension-aware quality gate and print warnings/failures | No |
| `run` | `--day YYYY-MM-DD` (required), `--equity FLOAT` (default 1000000), `--cash FLOAT` (default 1000000), `--peak-equity FLOAT`, `--template-mode A\|B\|dual` (default `dual`), `--account-file PATH`, `--restart/--no-restart` (default no-restart), `--init-account`, `--output PATH` | Run the planning pipeline, print/write JSON, optionally persist account state | No |
| `account` | `--account-file PATH` (required), `--clear-pending` | Inspect account equity/positions/pending/latches, or clear all pending plans | No |

## Data prerequisites

`run` does not assemble or sync anything on demand. The local store must
already satisfy all of the following.

### Raw per-day data written by `sync`

| Endpoint | Required | Use |
| --- | --- | --- |
| `daily` | Yes, non-empty | Raw OHLCV, `pre_close`, `amount` |
| `adj_factor` | Yes, non-empty | Adjusted (hfq) feature calculations |
| `daily_basic` | Yes, non-empty | `turnover_rate`, `circ_mv` |
| `stk_limit` | Yes, non-empty | Exchange up/down limits |
| `suspend_d` | Fetched; may be empty | Suspension-aware quality gate and feature exclusion |
| `index_daily` | Yes, non-empty | Market regime index bars |
| `limit_list_d` | Optional | Degraded flag and B-template data-quality context |

Raw partitions are stored at
`backend/data/raw/<endpoint>/<YYYYMMDD>.parquet`. Each synced day writes a
manifest at `backend/data/raw/manifests/<YYYYMMDD>.json` with row counts,
checksums, degraded-endpoint flags, and the day's `data_version`.

### Reference data written by `sync` and `calendar`

| Reference | Required | Use |
| --- | --- | --- |
| `trade_cal` | Yes | Open-day detection, listing-age count, entry expiry scheduling, forward window validation |
| `stock_basic` | Yes | Listed universe, board, listing/delisting dates, ST fallback, industry fallback |
| `namechange` | Optional | Historical ST detection |
| `index_classify` | Optional | SW industry names and level filtering |
| `index_member` | Optional | Point-in-time SW industry membership (latest `in_date` wins) |

When SW membership is unavailable, industry falls back to
`stock_basic.industry` and the run is marked degraded. A missing
`index_classify` while membership exists still uses membership, but sector
names fall back to the index code.

Fetch failures for optional references (`namechange`, `index_classify`,
`index_member`) are reported as warnings and also remove any previously
stored copy of that reference, so a run does not mix stale and current
reference data. The absence is surfaced as a degraded flag where relevant.

### Processed data written by `assemble` / `assemble-range`

| Dataset | Contents |
| --- | --- |
| `bars` | OHLCV, `adj_factor`, `turnover_rate`, `circ_mv`, `up_limit`/`down_limit`, `limit_derived`, `is_suspended` |
| `index_bars` | Index OHLCV used by the trend regime |
| `limit_list` | Per-day limit-list rows when raw `limit_list_d` has rows |

Processed partitions live under
`backend/data/processed/<dataset>/trade_date=YYYY-MM-DD/data.parquet` and
are read through DuckDB with hive partitioning.

### Calendar, history, and index requirements

- The run day must be an open day in `trade_cal`. A closed day fails with
  "not an open trading day".
- `run` loads bars and index bars for a history window of
  `max(400 calendar days, 2 * template_a.min_listing_days + 60)`, ending at
  the run day. With the default `min_listing_days: 120`, that is 400
  calendar days (roughly 270 trading days). Assemble at least that much
  history before the first run, and continue assembling each new day.
- The calendar must contain at least `template_a.min_listing_days` (120)
  trading days across the loaded history window through the run day for the
  listing-age filter.
- The forward calendar window is
  `max(45 calendar days, 2 * max(A entry_valid_days, B entry_valid_days) + 10)`.
  With defaults that is `run_day + 45 calendar days`. At minimum the
  calendar must contain enough forward trading days for the longer template
  entry expiry (3 trading days by default); use the full 45-day window to
  stay well clear of the boundary.
- `index_bars` must contain at least `environment.ma_slow` (60) bars
  including an as-of row. The configured primary index is `000985.CSI`; if
  it is missing or short, the pipeline tries `000300.SH` and flags
  `index_fallback`. Both failing raises `DataNotReadyError`.
- The quality gate consults `suspend_d` for the as-of day; the assembler
  also uses each day's partition to set `is_suspended`. When the as-of
  partition is missing entirely, the gate uses count-based coverage against
  the full listed universe and emits the `suspend_d_missing`/"count-based
  coverage" warning.

### Quality gate semantics

- Minimum bar coverage is 99% of the required universe.
- When raw `suspend_d` exists for the day, **required** codes are listed
  stocks minus full-day `S` suspensions (`suspend_type == "S"` with a null
  or blank `suspend_timing`); **allowed** codes remain every listed stock.
  Full-day suspended names therefore do not lower coverage, and a
  legitimate bar for one is not treated as unexpected. Partial-day
  suspensions remain required.
- When `suspend_d` is missing, required codes are unknown; coverage is
  measured by count against all listed stocks and the missing-endpoint
  warning is surfaced.
- The gate also fails on empty bars, missing/duplicate keys, null or
  non-finite OHLCV/adj-factor values, invalid OHLC or non-positive prices,
  missing/null limit prices, unexpected codes, and excessive limit-price
  mismatches. Small mismatch counts warn; more than 0.1% of rows fail.
- If the assembler had to derive limits because raw `stk_limit` was absent,
  the `limit_prices_derived` warning is emitted. Under a normal `sync`,
  `stk_limit` is a required non-empty endpoint.
- The sentiment stage snapshot is validated separately (required fields,
  finite values, integer counts, rates in `[0, 1]`) as part of `run`.

## Tests and checks

```bash
cd backend
uv run pytest                 # 761 tests pass
uv run ruff check app tests   # no errors
uv run mypy app               # 2 expected missing-stub errors
```

`mypy app` currently reports exactly two known errors because third-party
stub packages are not installed: `Library stubs not installed for "yaml"`
in `app/core/config.py` and missing `tushare` stubs in
`app/data/tushare_client.py`. These are dependency-typing gaps, not type
errors in this codebase.

A quick CLI smoke check:

```bash
uv run python -m app.cli --help
```

The help output must list `sync`, `calendar`, `assemble`,
`assemble-range`, `quality`, `run`, and `account`.

## Known limitations and follow-ups

- **M3-M6 are out of scope:** there is no backtest engine, scheduler,
  reporting layer, web dashboard, or broker adapter. `run` plans one day
  from data already on disk; it does not replay history or fill orders.
- **Reference snapshots are overwritten.** `sync`/`calendar` replace
  `backend/data/raw/reference/*.parquet` in place. Manifests record reference
  checksums for the day they were written, but old reference bytes are not
  retrievable. Versioned reference snapshots are a M3 follow-up.
- **Real Tushare contracts need token verification.** `trade_cal` is fetched
  in yearly chunks and `stock_basic` is fetched once per list status
  (`L`, `D`, `P`). The client calls each endpoint once per request and does
  not implement row/page pagination. The exact pagination behavior,
  `namechange` fields, and `index_member`/`index_classify` contracts should
  be verified against a live token before relying on them for production
  history.
- **A-template overhead resistance is a proxy.** The take-profit resistance
  gate uses `high_60d_raw`, a rolling 60-day high that includes the current
  bar. It is not a true swing-high detector; proper swing-high resistance
  detection is a follow-up.
- **Industry history uses as-of membership.** Multi-day industry strength
  computes prior days with the as-of tradeable membership. This is a slight
  approximation when stocks are reclassified or enter/leave the tradable
  set; a fully point-in-time membership history is a follow-up.
- **`data_version` can fall back.** It is the day manifest hash when a valid
  manifest exists; otherwise the pipeline uses `dv_<YYYY-MM-DD>`. Processed
  data assembled without a manifest therefore has a date-only version.
- **Plan identity is not byte-reproducible.** Plan IDs use UUID4 and
  `created_at` uses the wall clock, so two identical runs may serialize
  different bytes even though the decision inputs match.
- **The implementation plan document is stale.** The file under
  `docs/superpowers/plans/` predates several hardening fixes (calendar
  coverage, account state, quality fallback, and CLI error handling). The
  code and tests are the source of truth; do not copy the plan's older
  `assemble --day`-only workflow for a fresh store.
- **M2 simplifications remain.** B-template first-board ordering uses leader
  score, board height, volume, and turnover rather than exact
  `limit_list_d.first_time` ordering. B-template intraday triggers are daily
  approximations.
  Correlation uses up to 60 trading days of adjusted, gap-aware returns.
  Individual pairs with fewer than 20 overlapping observations are skipped;
  if a candidate has partners but none of its pairs reaches 20 observations,
  the candidate is labelled "correlation unavailable". Such candidates are
  approved with that note rather than blocked.
- **Account state is not reconciled automatically.** Equity, cash,
  positions, and peaks are whatever the account JSON says. The pipeline does
  not pull broker balances or update positions from market data; you must
  maintain that state yourself and use `--restart` with manual review.
- **Regime snapshots are not persisted by the CLI.** `run` includes the
  regime in its JSON output; `backend/data/regime/` is reserved for M4
  reporting. Use `--output` if you want to retain a run.

## Layout and architecture

```text
backend/
  app/
    core/        config loading, trading calendar, date parsing, errors
    data/        Tushare client, sync, Parquet store, assembler, quality,
                 point-in-time universe, industry mapping
    domain/      Pydantic models, enums, decimal money helpers
    indicators/  FeatureBuilder (adjusted OHLC, MAs, ATR, returns, volume,
                 board height, pullback, 60-day high)
    regime/      sentiment cycle, market regime, industry strength,
                 leader score
    screener/    dual-template A/B screener and registry
    planning/    entry, stop, position sizing, take-profit
    risk/        portfolio risk governor
    pipeline.py  end-to-end orchestration and prerequisite checks
    cli.py       Typer CLI
  configs/
    app.yaml
    strategies/dual_template_v1.yaml
  tests/         761 pytest tests
```

Data flow:

```text
Tushare
  -> raw Parquet partitions + references + per-day manifests
  -> assemble -> processed bars / index_bars / limit_list (DuckDB queries)
  -> features + point-in-time universe
  -> sentiment/regime + industry strength + leader score
  -> dual-template screener
  -> entry/stop/sizing/take-profit
  -> portfolio risk governor
  -> JSON plans (PENDING_ENTRY / REJECTED) + optional account persistence
```

Storage layout under `backend/data/` (gitignored):

```text
raw/<endpoint>/<YYYYMMDD>.parquet
raw/reference/<name>.parquet
raw/manifests/<YYYYMMDD>.json
processed/bars/trade_date=YYYY-MM-DD/data.parquet
processed/index_bars/trade_date=YYYY-MM-DD/data.parquet
processed/limit_list/trade_date=YYYY-MM-DD/data.parquet
regime/                                  # reserved for M4
```

## Safety and compliance

- Personal research and decision support only; not investment advice and
  not a solicitation to trade.
- No automatic order placement, no broker connectivity, and no execution
  engine. All plans are manual.
- Tushare data is subject to Tushare Pro's terms, permissions, and rate
  limits. Keep `TUSHARE_TOKEN` out of version control; `.env` files are
  gitignored.
- Data can be stale, incomplete, suspended, or wrong. The quality gate and
  degraded flags reduce risk but do not eliminate it. Validate any plan
  independently and size positions conservatively.
- Backtesting and performance evaluation are not implemented; no claim of
  profitability should be inferred from planning output.

## M1/M2 acceptance checklist

- [x] Raw Tushare sync of daily bars, adjustment factors, daily basics,
  limits, suspensions, index bars, and references, with per-day manifests.
- [x] Parquet storage with DuckDB querying and atomic writes.
- [x] Suspension-aware data quality gate; CLI `quality` fails closed.
- [x] Point-in-time universe with listing-age, ST, suspension, board,
  liquidity, and price filters.
- [x] Adjusted-price features including MAs, ATR, returns, volume ratios,
  60-day high, board height, pullback days, and limit flags.
- [x] Sentiment stage classification, trend regime, risk factor, industry
  strength, and leader score.
- [x] Dual-template screener: A mainline pullback and B sentiment leader.
- [x] Entry, initial stop, fixed-risk sizing, take-profit, and portfolio
  risk governor with drawdown latches.
- [x] CLI workflow: `sync`, `calendar`, `assemble`, `assemble-range`,
  `quality`, `run`, `account`, including account persistence and
  pending-plan pruning.
- [x] Verification: 761 tests pass, ruff reports no errors, mypy reports
  only the two documented missing-stub errors, and `--help` lists all seven
  commands.
- [ ] M3-M6 capabilities (backtest, scheduler/reports, dashboard, broker
  adapters) are intentionally not implemented.

Acceptance above is at the code and test level. Live Tushare permission and
endpoint-contract validation remains a follow-up because the test suite uses
an injected fake Tushare client.
