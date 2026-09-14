# CLAUDE.md — Project Constitution

This file is authoritative for how work on this repository is done. It is read by every
Claude Code session before doing anything else in this project. If code and this file
disagree, that is a bug — fix the code or fix this file, do not silently pick one.

## What this project is

A reproducible, testable NFL prediction system. The end state is a public website, but the
website is the *last* thing built, not the first. The thing that matters is the quantitative
prediction pipeline: ingest data → build features → predict → backtest → (later) research →
decide → publish.

## Non-negotiable principles

1. **The quantitative model is the only source of numerical predictions.** Predicted scores,
   win probabilities, fair spreads, fair totals, cover probabilities, model edge, and model
   performance numbers come from code paths in `src/nfl_predict/models` and
   `src/nfl_predict/backtesting`, computed from versioned data. An LLM (including Claude,
   including you, right now) may explain, summarize, contextualize, and research around
   these numbers. An LLM may never invent, adjust, "round for narrative," or override them.
   See [docs/MODEL_SPEC.md](docs/MODEL_SPEC.md) and [docs/DECISION_ENGINE.md](docs/DECISION_ENGINE.md).

2. **Independent model vs. market-aware model are separate and never merged into one
   feature set.** The independent model must not take sportsbook lines, odds, or
   market-derived features as predictive inputs. The market-aware model may. Code, config,
   and stored predictions must always be able to say which one produced a given number.
   See [docs/MODEL_SPEC.md](docs/MODEL_SPEC.md).

3. **No look-ahead bias, ever.** A backtested prediction for game G at timestamp T may only
   use data that would have actually been available at T. This applies to box scores,
   injury reports, odds, weather, roster/depth-chart data, and anything else. See
   [docs/BACKTESTING_RULES.md](docs/BACKTESTING_RULES.md) — read it before touching any
   feature-engineering or backtesting code.

4. **Every published prediction is reproducible.** Given a stored prediction, it must be
   possible to reconstruct exactly what data, feature values, model version, and code
   version produced it. Predictions are immutable snapshots once written; they are never
   edited in place, only superseded. This extends all the way back to raw source data: every
   ingested dataset carries a provenance manifest (source, retrieval timestamp, content
   hash, row count — full list in `docs/ARCHITECTURE.md#raw-data-provenance`) and raw files
   are never overwritten, so a prediction can be traced back to the exact upstream snapshot
   that produced it.

5. **The LLM research agent runs after the quantitative model, and never before it.** Its
   job is to flag information the structured data might be missing, not to produce its own
   competing prediction. It must separate FACTS / INTERPRETATION / EXTERNAL OPINION and
   cite sources with timestamps. See [docs/RESEARCH_AGENT.md](docs/RESEARCH_AGENT.md).

6. **Decisions (BET / LEAN / NO BET / VETO) are never forced.** Zero qualifying bets in a
   week is a valid, expected output. See [docs/DECISION_ENGINE.md](docs/DECISION_ENGINE.md).

7. **No placeholder or mocked functionality presented as production-ready.** If something
   isn't built yet, say so — don't fake it with a stub that returns plausible-looking
   numbers.

8. **Secrets never get committed.** Credentials and API keys live in environment variables,
   loaded via `.env` (gitignored) locally. `.env.example` documents the variable names only.

## Phase discipline

This project is built in phases (0 through 10), defined in the original project brief kept
in session history and mirrored in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#phases).
**Do not implement multiple phases at once, and do not skip ahead.** At the start of any new
phase of work:

1. Inspect the existing repository state — don't assume, check.
2. Read the relevant docs in `docs/`.
3. State what this phase needs to accomplish and any dependencies/assumptions.
4. Implement it.
5. Run tests.
6. Run a real (not fabricated) representative example.
7. Update the docs that changed.
8. Summarize what works and what's still missing.

Current phase status is tracked in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#status).

## Repository map

```
CLAUDE.md                  this file
docs/                       living specs — read before changing related code
  ARCHITECTURE.md           system architecture, phase status, data flow
  MODEL_SPEC.md              model families, targets, versioning, independent vs market-aware
  FEATURE_DICTIONARY.md      one entry per feature, required fields, leakage notes
  DATA_SOURCES.md            registry of every external data source and its trust/availability
  BACKTESTING_RULES.md       leakage prevention, walk-forward methodology, metrics
  RESEARCH_AGENT.md          what the LLM research step may and may not do
  DECISION_ENGINE.md         how model + market + research become BET/LEAN/NO BET/VETO
  WEBSITE_SPEC.md            public site pages and how they're generated from the DB
config/
  features.yaml              declarative feature registry (machine-readable mirror of FEATURE_DICTIONARY)
  sources.yaml               declarative data-source registry (machine-readable mirror of DATA_SOURCES)
  decision_thresholds.yaml   numeric thresholds consumed by the decision engine
prompts/                     versioned prompt templates for the LLM research/decision-support steps
src/nfl_predict/             the Python package — all pipeline code lives here
  config.py                  loads config/*.yaml + environment variables
  logging_conf.py            structured logging setup, shared by every module
  data/                      ingestion + storage (Phase 1, done — see docs/PHASE1_DATA_REPORT.md)
  features/                  feature engineering (Phase 2, done — see docs/PHASE2_FEATURE_REPORT.md)
  models/                    model training/inference (Phase 3, done — see docs/PHASE3_MODEL_REPORT.md)
  backtesting/               walk-forward evaluation (Phase 4, done — see docs/PHASE4_BACKTEST_REPORT.md)
  market/                    sportsbook market data + market-benchmark analysis (Phase 5+) —
                              structurally separate from models/features per principle 2;
                              see docs/PHASE5_MARKET_REPORT.md
  research/                  LLM research agent (Phase 6, done — see docs/PHASE6_RESEARCH_AGENT_REPORT.md)
  decision/                  decision engine (Phase 7, done — see docs/PHASE7_DECISION_ENGINE_REPORT.md)
  live/                      live orchestration + live provider integrations (Phase 8A, done
                              — see docs/PHASE8A_LIVE_PIPELINE_REPORT.md); entry point is
                              `python -m nfl_predict.live.run`, superseding the original
                              brief's separate `pipeline/`/`run_nfl.py` (Phase 9) slot
  api/                       FastAPI read API — the ONLY path the website uses to reach
                              storage (Phase 8B, done — see docs/PHASE8B_WEBSITE_REPORT.md);
                              see also docs/ARCHITECTURE.md#website-api-boundary
  content/                   LLM-written game-preview articles (Phase 8B follow-up) — never
                              a source of numbers, mirrors research/'s immutable-storage
                              pattern; see src/nfl_predict/content/prediction_writer.py
  storage/                   shared BlobStore interface (Phase 10) behind research/content/
                              decision/live/market's persisted artifacts — local filesystem
                              or hosted Postgres, selected by NFL_STORAGE_BACKEND; see
                              docs/ARCHITECTURE.md#storage
tests/                       pytest suite, mirrors src/nfl_predict structure
website/                     Next.js/TypeScript/React/Tailwind public website (Phase 8B,
                              done — see docs/PHASE8B_WEBSITE_REPORT.md); talks to the
                              backend exclusively through src/nfl_predict/api (FastAPI, JSON
                              over HTTP) — never a direct database connection, never a Python
                              import; not deployed
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#website-api-boundary) and
[docs/WEBSITE_SPEC.md](docs/WEBSITE_SPEC.md) for the website/API boundary rules.

## Engineering conventions

- **Language/runtime for the pipeline:** Python, pinned to a conservative, verified range —
  not simply whatever interpreter happens to be on a given machine. See
  `docs/ARCHITECTURE.md#python-version-policy` for the current range and rationale (as of
  this writing: `>=3.11,<3.14`, 3.12 recommended). Package managed via `pyproject.toml`,
  installed in editable mode (`pip install -e .[dev]`).
- **Data storage today:** SQLite/Parquet under `data/` (gitignored — data is large and
  regenerable, not source). All storage access goes through a narrow interface in
  `src/nfl_predict/data` so swapping in PostgreSQL later doesn't ripple through the codebase.
  Never hand-roll SQL strings outside that layer.
- **Config vs. secrets:** Non-secret, versioned settings live in `config/*.yaml` and are
  loaded through `src/nfl_predict/config.py`. Secrets (API keys, DB URLs with credentials)
  live only in environment variables / `.env`, never in YAML, never in code.
- **Logging:** use `src/nfl_predict/logging_conf.get_logger(__name__)`, not bare `print` or
  the root logger. Conventions are documented in `docs/ARCHITECTURE.md#logging-conventions`.
- **Testing:** pytest, under `tests/`, mirroring the `src/nfl_predict` package layout. A test
  for functionality that doesn't exist yet is not written "in advance" — write tests for
  what's actually implemented in the current phase.
- **Model/feature versioning:** every model artifact and every feature-engineering run is
  tagged with a version identifier that gets stored alongside predictions. Defined in
  `docs/MODEL_SPEC.md`.

## Current status

Phase 0 (project constitution), Phase 1 (historical data foundation), Phase 2 (feature
engine), Phase 3 (baseline prediction models), Phase 4 (sealed walk-forward backtest),
Phase 5 (market benchmark and edge analysis), Phase 6 (prospective LLM research agent),
Phase 7 (decision engine and verified pick ledger), Phase 8A (live NFL prediction and
market pipeline), and Phase 8B (FastAPI read API + Next.js website) are done. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#status) for the authoritative, up-to-date phase
table (including notes on how Phase 5 diverged from the original brief's phase-5
description, and how Phases 8/9 were renumbered into 8A/8B),
[docs/PHASE1_DATA_REPORT.md](docs/PHASE1_DATA_REPORT.md) for the historical data foundation,
[docs/PHASE2_FEATURE_REPORT.md](docs/PHASE2_FEATURE_REPORT.md) for the pregame feature set,
[docs/PHASE3_MODEL_REPORT.md](docs/PHASE3_MODEL_REPORT.md) for the baseline model comparison,
[docs/PHASE4_BACKTEST_REPORT.md](docs/PHASE4_BACKTEST_REPORT.md) for the frozen models'
walk-forward results, [docs/PHASE5_MARKET_REPORT.md](docs/PHASE5_MARKET_REPORT.md) for how
those same frozen predictions compare against historical sportsbook markets (no
statistically distinguishable edge found), [docs/PHASE6_RESEARCH_AGENT_REPORT.md](docs/PHASE6_RESEARCH_AGENT_REPORT.md)
for the prospective research-agent architecture and its real 3-game pilot,
[docs/PHASE7_DECISION_ENGINE_REPORT.md](docs/PHASE7_DECISION_ENGINE_REPORT.md) for the
deterministic decision engine, the immutable published-pick ledger, and a real 6-decision
pilot (all NO_BET — no live odds provider was configured then), and
[docs/PHASE8A_LIVE_PIPELINE_REPORT.md](docs/PHASE8A_LIVE_PIPELINE_REPORT.md) for the live
orchestrator (`python -m nfl_predict.live.run`), real per-game odds ingestion and real
Anthropic research invocation (with a web-search tool, since a plain API call cannot browse
the web) feeding actual `MarketPoint`/`ResearchPoint` values — proven to reach
`QUALIFIED_BET` for both spread and moneyline through the completely unmodified decision
engine, and an emergency correction that fixed a real Odds-API credit-exhaustion bug and a
real Anthropic INVALID_JSON bug, both live-validated afterward: the Odds API guard correctly
handled a genuine account-level `OUT_OF_USAGE_CREDITS` response with no retry, and a real
single-game Anthropic research call succeeded end-to-end (3 web searches, hash-verified
findings, `ResearchPoint.available=True`). A third correction then added real cost controls
(usage recording against a fetched Claude Sonnet 5 pricing table, configurable per-game/
per-run limits with a real pre-flight refusal, and prompt caching for the static half of the
research prompt only) before any full-slate run was attempted - live-validated with a real
~21.7% cost reduction, and it also caught and fixed a real incident where the test suite
itself made a genuine billed Anthropic call once real credentials appeared in the
environment. A fourth correction fixed a real production `TypeError` (a bare string where
the research schema expected a nested source object) via explicit type-guarded parsing that
rejects malformed shapes outright rather than coercing them, plus a real cost-accounting bug
where a downstream failure after a real, billed LLM call was reported as zero calls made -
both proven via an offline replay against real stored research/model data. A fifth
correction then closed a provenance gap that same replay exposed: persisted live odds were
keyed only by an opaque provider event id, with no way to attribute a stored snapshot back to
a canonical `game_id` - fixed with a new additive `market/event_game_mapping.py` index and
explicit matched/unmatched/ambiguous odds-event matching, proven end-to-end offline
(`tests/live/test_offline_reconstruction.py`); a live DEN@KC replay with freshly-fetched real
market data specifically could not be completed because the configured Odds API account has
zero usage credits remaining (confirmed via one controlled, authorized refresh attempt) - a
real external account fact, not a code gap. 561 tests passing, zero live network calls during
the test suite. See [docs/PHASE8A_LIVE_PIPELINE_REPORT.md](docs/PHASE8A_LIVE_PIPELINE_REPORT.md)
for the full detail on all five corrections. The independent football models remain frozen
and unchanged through Phase 8A — see `src/nfl_predict/models/split.py` and
`src/nfl_predict/backtesting/holdout_guard.py`. The research agent never overrides a model
number (`src/nfl_predict/research/schemas.py` has no field for one), and the decision engine
never invokes an LLM (`src/nfl_predict/decision/engine.py` is pure deterministic code) — live
packets feed the exact same unchanged `decide()` function Phase 7 built. Phase 8B then built
the read-only FastAPI service (`src/nfl_predict/api`) and the Next.js website (`website/`)
on top of all of it - see [docs/PHASE8B_WEBSITE_REPORT.md](docs/PHASE8B_WEBSITE_REPORT.md).
Building it against the real, accumulated `data/` directory (not just fixtures) caught two
more real bugs: `research`'s "latest run" lookup was sorting by `run_id` string instead of
each run's real timestamp (a manually-named pilot run_id sorted after real later ones), and
a test (`test_secrets_never_persisted.py`) was leaking fake research artifacts into real
`data/research/` on every full suite run - both fixed and regression-tested. Phase 8B was
followed by a visual redesign/rebrand (Edge Metric Sports) and, ahead of Phase 10, a real
storage-backend swap: every module under `data_dir` now runs transparently against either
local SQLite/files (unchanged default) or hosted Postgres, selected by `NFL_STORAGE_BACKEND`
- see [docs/ARCHITECTURE.md#storage](docs/ARCHITECTURE.md#storage). Verified against a
genuine embedded Postgres instance (via the optional `postgres` extra), not mocked. 684
tests total (623 backend + 61 frontend), zero live network calls in either suite. No ad
network, payments, accounts, or subscriptions; nothing is deployed yet - Phase 10
(provisioning real hosting and going live) is in progress.
