# NFL Prediction Platform

A reproducible, testable NFL prediction system: data ingestion → feature engineering →
quantitative models (independent + market-aware) → walk-forward backtesting → live
predictions → LLM research agent → decision engine → public website.

**Start here:** [CLAUDE.md](CLAUDE.md) for project principles and phase discipline, then
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for current status and system design.

## Status

Phase 0 (project constitution), Phase 1 (historical data foundation), Phase 2 (feature
engine), Phase 3 (baseline prediction models), Phase 4 (sealed walk-forward backtest),
Phase 5 (market benchmark and edge analysis), Phase 6 (prospective LLM research agent),
Phase 7 (decision engine and verified pick ledger), and Phase 8A (live NFL prediction and
market pipeline) are done. nflverse data for 2010-2025 has been ingested, validated,
normalized, and turned into a leakage-safe pregame feature set (114 registered features,
8,726 team-game rows, 4,363 game rows); a baseline model family (naive, Elo, Ridge
margin/total, logistic win-probability, LightGBM) was fit on 2010-2022 and validated on
2023; that same frozen model family was walk-forward backtested against the
previously-sealed 2024-2025 seasons (570 games); those same frozen predictions were then
benchmarked against historical sportsbook lines; a prospective research-agent layer was
built to surface current pregame context the structured data misses; a deterministic
decision engine with an immutable published-pick ledger was built on top of all of it; and a
live orchestrator now runs those same frozen models and that same unchanged decision engine
against the real, current NFL schedule — see
[docs/PHASE1_DATA_REPORT.md](docs/PHASE1_DATA_REPORT.md),
[docs/PHASE2_FEATURE_REPORT.md](docs/PHASE2_FEATURE_REPORT.md),
[docs/PHASE3_MODEL_REPORT.md](docs/PHASE3_MODEL_REPORT.md),
[docs/PHASE4_BACKTEST_REPORT.md](docs/PHASE4_BACKTEST_REPORT.md),
[docs/PHASE5_MARKET_REPORT.md](docs/PHASE5_MARKET_REPORT.md),
[docs/PHASE6_RESEARCH_AGENT_REPORT.md](docs/PHASE6_RESEARCH_AGENT_REPORT.md),
[docs/PHASE7_DECISION_ENGINE_REPORT.md](docs/PHASE7_DECISION_ENGINE_REPORT.md), and
[docs/PHASE8A_LIVE_PIPELINE_REPORT.md](docs/PHASE8A_LIVE_PIPELINE_REPORT.md). Elo remains
the strongest independent margin and win-probability model, but **the market beats every
independent candidate on every target, and no statistically distinguishable betting edge
was found** - cover rates hover at or below break-even and every ROI confidence interval
straddles zero. Real per-game odds ingestion and real Anthropic research invocation (with a
web-search tool, since a plain LLM call cannot browse the web) now feed the decision packet
with actual market/research values instead of provider-availability placeholders — proven to
reach `QUALIFIED_BET` for both spread and moneyline through the completely unmodified
decision engine, and both live-validated for real after an emergency correction fixed an
Odds-API credit-exhaustion bug (a request pattern that once burned ~500 credits in one run,
now 2 credits per slate refresh under a budget guard) and an Anthropic INVALID_JSON bug
(structured tool-call extraction instead of free-text parsing). The Odds API account
currently has zero usage credits remaining (an external account fact, not a code gap); a
real single-game Anthropic research call succeeded end-to-end. A follow-up correction added
real usage/cost recording (against a fetched, configured Claude Sonnet 5 pricing table -
never hard-coded), configurable per-game/per-run cost limits with a real pre-flight refusal,
and prompt caching for the static half of the research prompt - live-validated with a
~21.7% cost reduction on one real call ($0.246 → $0.192). A further correction fixed a real
production `TypeError` (a bare string where the research schema expected a nested object,
found via a real live run) with explicit type-guarded parsing that rejects malformed shapes
outright rather than coercing them, and fixed a real cost-accounting bug where a downstream
failure after a real, billed LLM call was reported as zero calls made - both proven via an
offline replay against real stored research and model data, no new live calls. A final
correction closed a provenance gap that same replay exposed - persisted live odds had no way
to be attributed back to a canonical game_id - with a new additive event/game mapping index
and end-to-end offline reconstruction proof; a live replay with freshly-fetched real market
data specifically could not be completed because the configured Odds API account has zero
usage credits remaining (a real external account fact, confirmed via one controlled,
authorized refresh attempt - not a code gap). Phase 8B then built a read-only FastAPI
service and a Next.js website on top of all of it - verified end-to-end in a real browser
against real DEN@KC data (the real market refresh, real Elo/Ridge/LightGBM margins, real
`VETO_CONSIDERATION` research), and it caught two more real bugs while doing so: a "latest
research" lookup sorting by `run_id` string instead of each run's real timestamp, and a test
leaking fake research artifacts into real `data/research/`. **No betting picks have
been published, no model probability has ever been changed, and the website is not
deployed** - no ad network, payments, accounts, or subscriptions. See the phase table in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#status) (including notes on how Phase 5 diverged
from the original brief's phase-5 description, and how Phases 8/9 were renumbered into
8A/8B).

## Ingesting data

```bash
python -m nfl_predict.data.ingest --dataset teams
python -m nfl_predict.data.ingest --all --seasons 2010-2025
python -m nfl_predict.data.coverage --seasons 2010-2025
```

Requires the `data` extra (`pip install -e ".[dev,data]"`) on a 3.11–3.13 interpreter (see
the Python version note below). See
[docs/PHASE1_DATA_REPORT.md](docs/PHASE1_DATA_REPORT.md) for what's actually been verified
and [CLAUDE.md](CLAUDE.md) for the full CLI contract.

## Building features

```bash
python -m nfl_predict.features.build --seasons 2010-2025
python -m nfl_predict.features.coverage --seasons 2010-2025
```

See [docs/PHASE2_FEATURE_REPORT.md](docs/PHASE2_FEATURE_REPORT.md) for what's registered,
[docs/FEATURE_DICTIONARY.md](docs/FEATURE_DICTIONARY.md) for the feature families, and
`config/features.yaml` (generated by `scripts/generate_feature_registry.py`) for the full
per-feature definitions.

## Training baseline models

```bash
python -m nfl_predict.models.train
```

Trains naive/Elo/Ridge/logistic/LightGBM baselines on 2010-2022, validates on 2023, and
writes model artifacts + reports under `data/models/` and `data/reports/`. Requires the
`modeling` extra (`pip install -e ".[dev,data,modeling]"`). **Never reads 2024-2025** — see
[docs/PHASE3_MODEL_REPORT.md](docs/PHASE3_MODEL_REPORT.md) and
`src/nfl_predict/models/split.py` for the sealed-holdout guard.

## Running the Phase 4 backtest

```bash
python -m nfl_predict.backtesting.run_freeze     # pre-holdout freeze - must run first
python -m nfl_predict.backtesting.run_backtest   # unseals 2024-2025, walk-forward, ledger
python -m nfl_predict.backtesting.run_report     # scoring, comparisons, CIs, error analysis
```

`run_freeze` writes and hashes `data/backtests/phase4_holdout_freeze.json` — every later
step verifies that hash and refuses to run if it's missing or has been altered (see
`src/nfl_predict/backtesting/holdout_guard.py`). See
[docs/PHASE4_BACKTEST_REPORT.md](docs/PHASE4_BACKTEST_REPORT.md) for the full results.

## Running the Phase 5 market benchmark

```bash
python -m nfl_predict.market.run_market_ingest   # builds market_snapshot for 2010-2025
python -m nfl_predict.market.run_market_report    # benchmark, ATS, moneyline edge, CLV, etc.
```

Reads the Phase 4 prediction ledger and historical sportsbook lines (bundled in nflverse's
`schedules` dataset, never used by the independent model); writes only to `data/market/` and
`data/reports/phase5_results.json`. The independent models are never retrained or altered.
A live odds provider is available (`src/nfl_predict/market/odds_provider.py`) but requires
`NFL_ODDS_API_KEY` in `.env` and has no wired-up HTTP calls yet - see
[docs/PHASE5_MARKET_REPORT.md](docs/PHASE5_MARKET_REPORT.md) for full results.

## Running the Phase 6 research agent

```bash
python scripts/run_phase6_pilot.py   # the real 3-game prospective pilot (2026 Week 1)
```

Builds a structured input packet per game (frozen Elo continued live; Ridge/LightGBM and
the market are marked `UNAVAILABLE` rather than fabricated - no live feature pipeline or
odds provider exists yet), runs it through the research pipeline, and writes immutable
records to `data/research/season=/week=/<game_id>/run_id=/` plus an append-only prospective
ledger entry. No automated LLM provider is configured in this environment
(`NFL_RESEARCH_LLM_API_KEY` unset) - the pilot used a manual provider explicitly labeled as
such; see [docs/PHASE6_RESEARCH_AGENT_REPORT.md](docs/PHASE6_RESEARCH_AGENT_REPORT.md).
Research never overrides a model number - `nfl_predict.research.schemas.ResearchFindings`
has no field that could carry one.

## Running the Phase 7 decision engine

```bash
python scripts/run_phase7_pilot.py   # runs decide() against the real Phase 6 pilot packets
```

Builds a `DecisionInputPacket` from each Phase 6 research run (never modifying it), applies
the versioned `config/decision_rules.yaml` v1 rules, and appends every decision to
`data/decision/decision_log/`. No LLM call happens inside the engine. Publishing a
`QUALIFIED_BET`/`LEAN` decision as an official pick (`nfl_predict.decision.pick_ledger`) is
a separate, deliberate step this script does not take - see
[docs/PHASE7_DECISION_ENGINE_REPORT.md](docs/PHASE7_DECISION_ENGINE_REPORT.md) for the full
architecture, rule rationale, and pilot results (all six decisions were `NO_BET`).

## Running the Phase 8A live pipeline

```bash
python -m nfl_predict.live.run                        # dry-run by default - fetches/computes, writes nothing
python -m nfl_predict.live.run --no-dry-run            # writes real predictions + decisions
python -m nfl_predict.live.run --week 2                # a specific week
python -m nfl_predict.live.run --game 2026_01_DEN_KC   # a specific game
python -m nfl_predict.live.run --research-all          # research every target game (real, billed Anthropic calls) instead of the top --max-research-calls
python -m nfl_predict.live.run --show-decisions        # add a per-game/market_type decision + reason-code breakdown to the output
```

Reuses Phase 4's frozen model-fitting functions and Phase 7's unchanged `decide()` against
the real current NFL schedule. When `NFL_ODDS_API_KEY`/`NFL_RESEARCH_LLM_API_KEY` are set,
this makes REAL external calls (fetching real sportsbook odds; invoking Anthropic's Messages
API with a web-search tool for up to `--max-research-calls` games, default 3) and builds
real `MarketPoint`/`ResearchPoint` values from them — not just a provider-availability check.
Without those credentials, status is reported honestly (`ODDS_PROVIDER_UNAVAILABLE` /
`RESEARCH_PROVIDER_UNAVAILABLE` / `INJURY_PROVIDER_UNAVAILABLE`) and decisions correctly fall
back to `NO_BET`/`MISSING_LIVE_DATA` — never fabricated. Never auto-publishes an official
Best Bet — see [docs/PHASE8A_LIVE_PIPELINE_REPORT.md](docs/PHASE8A_LIVE_PIPELINE_REPORT.md)
for the full architecture, the real per-game market/research wiring, a proof that
`QUALIFIED_BET` is genuinely reachable through the unmodified decision engine, a real Week 1
2026 pilot (14 games, 28 decisions, all `NO_BET` because no credential is configured in this
sandbox), and known limitations.

## Running the Phase 8B website locally

```bash
# Terminal 1 - the read-only FastAPI service
pip install -e ".[dev,api]"
uvicorn nfl_predict.api.main:app --reload --port 8000

# Terminal 2 - the Next.js website
cd website
npm install
npm run dev -- --port 3100   # or `npm run dev` if port 3000 is free on your machine
```

The website (`website/`) never imports Python, never opens a database, and never calls
Anthropic or The Odds API - it only fetches JSON from the FastAPI service
(`website/.env.local`'s `NEXT_PUBLIC_API_BASE_URL`, default `http://localhost:8000`). Seven
read-only endpoints (`/api/health`, `/api/nfl/slate/current`, `/api/nfl/games/{game_id}`,
`/api/nfl/best-bets`, `/api/nfl/performance`, `/api/nfl/model-status`,
`/api/nfl/decisions/{game_id}`) and six site routes (home, `/nfl/picks`,
`/nfl/games/[gameId]`, `/nfl/performance`, `/methodology`, `/about`) - see
[docs/PHASE8B_WEBSITE_REPORT.md](docs/PHASE8B_WEBSITE_REPORT.md) for the full architecture,
the real DEN@KC data verified end-to-end in a browser, and two more real bugs this phase
found and fixed while building against the actual accumulated `data/` directory. Backend
tests: `pytest tests/api`. Frontend tests: `cd website && npm test`. Not deployed - no
production database, hosting, or ad network configured yet.

## Development setup

Requires Python **>=3.11, <3.14** (3.12 recommended) — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#python-version-policy) for why, and note that a
newer interpreter installed on your machine is not automatically supported.

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env
pytest
```

Installing the `data` extra (`pip install -e ".[dev,data]"`, needed starting Phase 1) pulls
in `nflreadpy`/Polars, which currently has no Python 3.14 wheels — use a 3.11–3.13
interpreter for that extra.

## Repository layout

See [CLAUDE.md](CLAUDE.md#repository-map).
