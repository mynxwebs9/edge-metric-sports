# Architecture

## Purpose

This document describes the system's shape: how data flows from raw sources to a published
prediction, what runs where, what storage looks like today vs. in production, and the
logging/config conventions every module follows. It is updated whenever a phase changes the
architecture — it should always describe the system as it actually exists, not an aspiration.

## Status

<!-- Update this table at the end of every phase. Do not mark a phase Done until it has a
     working implementation, tests, and docs updated. -->

| Phase | Name                          | Status      | Notes |
|-------|-------------------------------|-------------|-------|
| 0     | Project constitution          | Done        | Repo scaffold, docs, config format, logging/testing conventions, Python version policy, nflreadpy loader choice, raw-data provenance contract, predictive-uncertainty contract, website/LLM API boundary — see the Phase 0 correction passes for the full list |
| 1     | Historical data foundation    | Done        | nflverse/nflreadpy ingestion (12 datasets), immutable hashed raw snapshots + provenance manifests, normalized teams/games (SQLite) + play-by-play (partitioned Parquet), validation (FATAL/ERROR/WARNING/INFO), idempotency verified, 2010-2025 backfilled and coverage-reported — see docs/PHASE1_DATA_REPORT.md |
| 2     | Feature engine                | Done        | 114 registered pregame features (104 predictive + 10 sample-size), team-game and game-level Parquet tables, leakage-safe rolling/season/prev-season windows, leakage-safe QB continuity, 118 tests passing (33 Phase 2: 18 leakage + 12 quality + 3 QB-leakage), manual audit of 6 real games, 2010-2025 backfilled — see docs/PHASE2_FEATURE_REPORT.md |
| 3     | Baseline prediction models    | Done        | Naive/Elo/Ridge margin+total/logistic win-prob/LightGBM baselines, 6-tier feature ablation (A-F) + CORE/FULL comparison, leakage-safe temporal split (dev 2010-2022, val 2023, sealed holdout 2024-2025 guarded by `SealedHoldoutError`), 156 tests passing (38 new), preliminary uncertainty diagnostics, no betting/ROI/market comparison — see docs/PHASE3_MODEL_REPORT.md |
| 4     | Backtesting                   | Done        | Sealed walk-forward backtest on 2024-2025 (previously untouched): pre-holdout freeze manifest (hash `a2c595c7...53ed3`) with Elo re-tuning (k=40, interior to grid) and a total-model bias investigation (era-related, not corrected), 44-week walk-forward retraining, immutable prediction ledger (6,838 predictions/570 games), baseline comparison + bootstrap CIs, calibration, uncertainty-estimate validation against Phase 3's frozen sigma, football error analysis, model-selection recommendation (Elo remains best on margin+win-prob; total models add no value over naive), 189 tests passing (30 new), no betting/ROI/market comparison — see docs/PHASE4_BACKTEST_REPORT.md |
| 5     | Market benchmark & edge analysis | Done     | Historical market_snapshot layer (2010-2025, 4,363 games, zero nulls) built from raw schedules data; no-vig odds math; fair-line derivation from frozen Phase 3/4 model outputs + uncertainty; market benchmarked as a predictor (beats every independent candidate on margin/win-prob/total); ATS/moneyline edge analysis with price-aware ROI + bootstrap CIs (no statistically distinguishable edge found); incremental-information test (market+model vs market-only, no meaningful improvement); CLV math implemented but not computable on single-snapshot historical data; provider-neutral live odds interface + append-only live snapshot storage (both unconfigured/untested against live data); 280 tests passing (91 new); independent models unchanged; original brief's "live prediction pipeline" phase-5 scope remains open, see note below — see docs/PHASE5_MARKET_REPORT.md |
| 6     | LLM research agent            | Done        | Prospective-first research-agent architecture: structured input packets (frozen Elo continued live, Ridge/LightGBM/market marked UNAVAILABLE rather than fabricated), 5-way fact/opinion separation, 4-tier source hierarchy, 0-4 materiality rubric, closed research-classification enum, historical-leakage guard (blocks ordinary web research on already-played games without archived-source authorization), immutable append-only storage + prospective ledger (outcomes joined separately, never mutating pregame records), provider-neutral LLM/injury/weather/news interfaces (none live-configured), cost tracking, explicit failure modes, 3-game real prospective pilot researched via WebSearch and frozen ~2 days before kickoff, 344 tests passing (64 new) — model numbers verified immutable throughout, no historical hindsight research created — see docs/PHASE6_RESEARCH_AGENT_REPORT.md |
| 7     | Decision engine & pick ledger  | Done        | Deterministic 5-category decision engine (NO_BET/WATCH/LEAN/QUALIFIED_BET/VETO, default NO_BET, no LLM call inside decide()); versioned conservative rules (config/decision_rules.yaml, all PROSPECTIVE_VALIDATION, thresholds traced to Phase 5's pre-declared buckets or outcome-blind operational reasoning, never fit to results); event-sourced immutable published-pick ledger (line/price/category never alterable post-publication); ALL_MODEL_PREDICTIONS/LEANS/BEST_BETS tracked strictly separately; deterministic spread+moneyline settlement (no totals); price-aware units (price required at publish time, hypothetical -110 kept separate); predefined-window streak/headline engine (no arbitrary date ranges); shadow-decision infrastructure; 6-decision pilot run against the real Phase 6 packets — all 6 correctly NO_BET (MISSING_LIVE_DATA), zero QUALIFIED_BET forced; 405 tests passing (61 new) — see docs/PHASE7_DECISION_ENGINE_REPORT.md |
| 8A    | Live NFL prediction & market pipeline | Done | `python -m nfl_predict.live.run` orchestrator (schedule → live features → live Elo/Ridge/LightGBM/logistic, reusing Phase 4's frozen fitting functions unchanged → REAL per-game odds ingestion (odds_ingestion.py: fetch → persist append-only snapshots → no-vig/median consensus → MarketPoint) → REAL per-game research invocation (research_live.py: trigger-priority game selection under a cost cap → actually calls Anthropic's run_research(), with a server-side web-search tool wired in → parse/evaluate/persist → ResearchPoint) → injury provider status (never a decision-rule input under v1) → Phase 7's unchanged decide() → public ALL_MODEL_PREDICTIONS storage → decision log); a dedicated integration test proves a fully-populated realistic packet reaches QUALIFIED_BET for both spread and moneyline through the completely unmodified decision engine — the earlier "structurally unreachable" state was a data-plumbing gap, not a property of the rules; an emergency correction then fixed two real production bugs found during live validation — an Odds API request pattern that burned ~500 credits in one run (fixed: one batched featured-odds fetch per slate refresh at 2 credits, under a configurable NFL_ODDS_MAX_CREDITS_PER_RUN guard, never retried on a quota error) and an Anthropic INVALID_JSON failure from parsing concatenated free-text blocks (fixed: a structured submit_research_findings tool call, pause_turn continuation handling, real HTTP-error-body capture) — both now live-validated for real: the Odds API guard correctly handled a genuine OUT_OF_USAGE_CREDITS response with no retry, and a real single-game Anthropic research call succeeded end-to-end (3 web searches, hash-verified findings, ResearchPoint.available=True); live weather via Open-Meteo (credential-free, live-tested for real); structured InjuryProvider production-gating distinct from the LLM research provider and never a gate under rule set v1; publication states DRAFT/READY/PUBLISHED/SUPERSEDED with hash-verified, superseding-never-mutates storage; results ingestion + current-season ALL_MODEL_PREDICTIONS performance tracking (no backfilled fake Best Bets record); component-failure isolation, reason-code-count reporting, and a machine-readable automation report; a third correction then added real Anthropic cost controls before any full-slate run — real usage recording (cache/web-search tokens, a dollar cost computed ONLY from config/llm_pricing.yaml, real fetched Claude Sonnet 5 rates, never hard-coded), configurable per-game/per-run limits with a real pre-flight cost-budget refusal, prompt caching (the static research instructions/source-hierarchy/tool-schema half of the prompt, never the game-specific half), and a "Research:" summary section — live-validated with one real call showing a ~21.7% cost reduction ($0.246 → $0.192) on a single sample; this correction also caught and fixed a real incident where the test suite itself made a genuine billed Anthropic call once real credentials appeared in the environment, via a new autouse `conftest.py` fixture; a fourth correction then fixed a real production TypeError (a bare string where the research schema expected a nested source object, uncaught by the parser's exception handling) via explicit `_require_dict`/`_require_list` type guards that convert any such mismatch into the same typed `ResearchOutputParseError` every other malformed-output case already produces, plus a real cost/usage-accounting bug (a downstream failure after a real, billed LLM call was silently reported as zero calls made) fixed by making `run_research_for_game()` always return a result dict carrying real usage/cost data instead of ever raising uncaught, and by distinguishing `research_games_selected`/`research_llm_calls_attempted`/`_succeeded`/`_failed`/`research_artifacts_persisted` in the automation report; both fixes validated via an offline replay against real stored research/model data (no new live calls); a fifth correction then closed a provenance gap that same offline replay exposed — persisted live-odds snapshots were keyed only by an opaque `provider_event_id` with no way to attribute a stored snapshot back to a canonical `game_id`/`season`/`week`, forcing that replay to use a synthetic MarketPoint — fixed with a new additive, append-only `market/event_game_mapping.py` index (provider_event_id → canonical game_id, with auditable home/away normalization, idempotent re-recording, and conflict detection on a genuine remap), explicit matched/unmatched/ambiguous odds-event matching (`live/odds_mapping.match_odds_event_to_game`, never silently resolving an ambiguous multi-game match), consensus/decision provenance fields threaded through `MarketConsensus`/`MarketPoint`/`DecisionRecord` (a real `DecisionInputPacket.content_hash()` replacing the old `"n/a"` placeholder), and end-to-end offline reconstruction proven in `tests/live/test_offline_reconstruction.py`; a live DEN@KC replay with freshly-fetched real market data specifically could not be completed because the configured Odds API account has zero usage credits remaining, confirmed via one controlled, authorized refresh attempt that itself failed with the same real `OUT_OF_USAGE_CREDITS` condition Correction 2 already documented — a real external account fact, not a code gap, and no data was fabricated to route around it; 561 tests passing (158 new total), zero live network calls during the test suite — see docs/PHASE8A_LIVE_PIPELINE_REPORT.md. Supersedes this table's earlier, not-yet-started "Phase 9 Automation" slot — see note below. |
| 8B    | Website                       | Done        | A read-only FastAPI service (`src/nfl_predict/api`) — seven endpoints (health, current slate, game detail, best-bets, performance, model-status, decisions), consumer-safe response schemas, a closed reason-code/decision/classification translation table, and (re-confirming Phase 8A Correction 5's lesson) two reconstruction paths that are never conflated: "current" market data (latest fetch batch only) vs. "as a specific persisted decision saw it" (the exact referenced batch, integrity-checked against its recorded snapshot_reference, never blended or substituted) — plus a Next.js 16/TypeScript/React/Tailwind site (`website/`) with 6 routes (home, picks, game detail, performance, methodology, about) that reads only through that API; building it against the real, accumulated `data/` directory caught and fixed two more real bugs (a "latest research run" lookup that sorted by `run_id` string instead of each run's real timestamp, and a test that was leaking fake research artifacts into real `data/research/` on every suite run); 581 backend + 40 frontend tests passing, zero live network calls in either suite; verified in a real browser against real DEN@KC data (Correction 5's real market refresh, real Elo/Ridge/LightGBM margins, real VETO_CONSIDERATION research classification), light/dark and mobile checked; no ad network, payments, accounts, or subscriptions; not deployed — see docs/PHASE8B_WEBSITE_REPORT.md |
| 10    | Production                    | In progress | Storage-swap prerequisite done: every module under `data_dir` (research, content, decisions, the pick ledger, public predictions, live odds snapshots, the event/game mapping index, teams/games) now runs transparently against either backend, selected by `NFL_STORAGE_BACKEND` (`sqlite`, unchanged default, or `postgres`) - see `#storage` below for the design and `src/nfl_predict/storage/blob_store.py`/`src/nfl_predict/data/db.py` for the implementation. Verified two ways: the full existing test suite (597 tests) re-passes unchanged against the local/sqlite backend (zero behavior regression), and a NEW real-Postgres integration suite (9 tests, `tests/storage/test_postgres_blob_store.py` + `tests/data/test_postgres_connection.py`) passes against a genuine embedded Postgres instance (the `pgserver` package, no Docker/system install needed) that `tests/conftest.py` starts automatically for the test session when the `postgres` extra is installed - not mocked, not skipped. Still open before this phase is Done: actually provisioning a hosted Postgres instance and deploying the FastAPI/Next.js processes (Vercel + a container host, per the user's stated plan) - no cloud accounts/credentials exist yet, so that step has not been attempted. |

**Note on phase 5:** the original project brief's phase 5 was "live prediction pipeline."
The user's actual, detailed Phase 5 instructions (see `docs/PHASE5_MARKET_REPORT.md`) were
"market benchmark and edge analysis" instead — this table reflects what was actually
specified and built. The live-prediction-pipeline work this left open was picked back up,
scoped, and completed as Phase 8A (see below).

**Note on phases 8/9:** the original brief's Phase 8 was "Website" and Phase 9 was
"Automation (`run_nfl.py`)." In practice, the automation/live-pipeline work needed to exist
*before* the website could show anything real, so it was pulled forward, explicitly scoped
by the user as "Phase 8A — Live NFL Prediction and Market Pipeline," and completed first.
The website is now Phase 8B. There is no separate Phase 9 — its scope (a single idempotent
orchestrator with dry-run/week/game/mode flags) is exactly what Phase 8A's
`python -m nfl_predict.live.run` already is.

## Phases

The full phase definitions (what each phase must accomplish, in order) live in the original
project brief. In short:

0. Project constitution — docs, repo structure, config/logging/testing conventions. No modeling.
1. Historical data foundation — ingest nflverse-compatible historical data, validate it.
2. Feature engine — pregame team/game features, documented in the feature dictionary.
3. Baseline models — naive, Elo, linear/logistic, gradient-boosted; predict scores/margins/totals/probabilities.
4. Backtesting — walk-forward, no look-ahead, calibration + error metrics + ROI vs. baselines.
5. Market benchmark and edge analysis — historical sportsbook lines benchmarked against the
   frozen independent models (never merged into their training/selection); ATS/moneyline
   edge, CLV, and incremental-information testing. (Supersedes the original brief's "Live
   prediction pipeline" phase 5 description for this project - see docs/PHASE5_MARKET_REPORT.md.
   Real-time/live prediction-pipeline work, using the provider-neutral odds interface this
   phase built, remains open for a future phase.)
6. LLM research agent — post-model research to flag missing context; facts/interpretation/opinion separated.
7. Decision engine — combine model + market + research into BET/LEAN/NO BET/VETO, never forced.
8A. Live NFL prediction & market pipeline — single idempotent orchestrator
    (`python -m nfl_predict.live.run`, supporting dry-run/week/game/research-only/odds-only/
    models-only/decisions-only/publish flags) that closes Phase 7's live-data gaps: live
    schedule/feature generation, live odds/injury/weather/research provider integration
    (credential-gated, never fabricated), and feeding real live packets into the unchanged
    Phase 7 decision engine. (Supersedes the original brief's separate Phase 9 "Automation"
    step — see the phase-8/9 note above.)
8B. Website — Next.js/TypeScript/React/Tailwind, generated from structured DB data only.
10. Production — hosting, scheduling, monitoring, backups, SEO/analytics/ads.

Phases are sequential. A phase is not started until the previous phase has a working,
tested implementation.

## High-level data flow (target end state)

```
raw sources (nflverse, odds, injuries, weather, ...)
        │  ingestion (src/nfl_predict/data) — each fetch recorded with a provenance manifest
        ▼
   raw store (season/week/game granularity, source-tagged, timestamped, immutable snapshots)
        │  feature engine (src/nfl_predict/features)
        ▼
   pregame feature table (per team, per game, "as-of" timestamped)
        │  models (src/nfl_predict/models)
        ├──────────────► independent model  ──► predicted score/margin/total/prob + predictive
        │                                        distribution / calibrated uncertainty
        └──────────────► market-aware model ──► predicted score/margin/total/prob + predictive
                                                 distribution / calibrated uncertainty
        │  backtesting (src/nfl_predict/backtesting) — offline, walk-forward only
        ▼
   research agent (src/nfl_predict/research) — LLM, post-hoc, facts/interpretation/opinion
        ▼
   decision engine (src/nfl_predict/decision) — BET/LEAN/NO BET/VETO
        ▼
   immutable prediction snapshot (DB) — model outputs, uncertainty, decision, research findings
        │  content generation — LLM prose (prompts/prediction_writer.md), run by the pipeline
        │  offline and persisted with its own prompt version + timestamp; never generated at
        │  website request time (see docs/WEBSITE_SPEC.md)
        ▼
   persisted website content (DB)
        │  orchestration (src/nfl_predict/pipeline, run_nfl.py — Phase 9) triggers the whole
        │  sequence above; it does not change what any stage does
        ▼
   Python FastAPI read API (src/nfl_predict/api — Phase 8) — the ONLY consumer of
   src/nfl_predict/data repository interfaces on the serving path; exposes stable JSON
   endpoints (predictions, games, teams, performance, research, persisted content)
        │  HTTP / JSON — the only channel the website is allowed to use
        ▼
   public website (Next.js, Phase 8) — reads exclusively via the FastAPI JSON endpoints;
   never imports Python code, never opens SQLite/PostgreSQL directly, never computes numbers
   or calls an LLM at request time
```

## Storage

**Local/dev (default, `NFL_STORAGE_BACKEND=sqlite`):** SQLite for relational/tabular state
(teams, games, ingestion manifests, validation issues) and Parquet for bulk columnar data
(play-by-play, feature tables) under a gitignored `data/` directory. Chosen because it
removes the operational overhead of running a database server during modeling/backtesting
iteration, while still forcing real row/column/type discipline (unlike ad hoc CSVs/pickles).
Raw ingested files live under `data/raw/` as immutable, never-overwritten snapshots, each
with its own provenance manifest — see `#raw-data-provenance` below. Phase 8A's live
pipeline artifacts (public ALL_MODEL_PREDICTIONS, timestamped odds snapshots, decisions, the
pick ledger, research/content runs, the event/game mapping index) follow the same
immutable/hash-verified/append-only conventions, all under `data/` today.

**Hosted (Phase 10, `NFL_STORAGE_BACKEND=postgres`):** the same logical storage, backed by a
single Postgres database instead of local files — needed because a deployed FastAPI process
(Vercel/Render/etc.) has no durable local disk. Implemented mechanically, not as a rewrite,
in two pieces:

- **Everything that was a JSON/Parquet blob at a deterministic path or an append-only log**
  (research runs, content previews, the decision log, the published-pick ledger, public
  model predictions, live odds snapshots, the event/game mapping index) now goes through one
  shared `BlobStore` interface (`src/nfl_predict/storage/blob_store.py`):
  `read_bytes`/`write_bytes`/`append_bytes`/`exists`/`list_keys`, keyed by the same
  `/`-separated path strings these modules already used. `LocalFilesystemBlobStore` is the
  unchanged local-file behavior; `PostgresBlobStore` stores the identical keys/bytes as rows
  in one `blobs` table. Every calling module (research/content/decision/live/market storage)
  was refactored onto this interface with its on-disk layout, immutability guarantees, and
  hash-verification behavior otherwise unchanged — proven by the full pre-existing test
  suite re-passing unmodified against the local backend.
- **Relational state (teams, games, ingestion manifests)** stays in `src/nfl_predict/data/db.py`
  and `src/nfl_predict/data/repositories.py`. `get_connection()` returns either a real
  `sqlite3.Connection` or a `_PostgresConnectionAdapter` that speaks the same `?`/`:name`
  placeholder style and `row["column"]` access as `sqlite3.Row`, so every existing
  repository/`schedule_provider` call site is unchanged. The one non-portable piece —
  SQLite's `INSERT OR REPLACE` upsert syntax — is translated to Postgres `ON CONFLICT ... DO
  UPDATE` via an explicit, whitespace-normalized lookup table of the project's actual known
  statements (`_INSERT_OR_REPLACE_TRANSLATIONS`), not a general SQL parser: an unrecognized
  "INSERT OR REPLACE" fails loudly rather than being silently mis-translated.
- A config flag (`NFL_STORAGE_BACKEND=sqlite|postgres`, `NFL_DATABASE_URL` for the connection
  string — see `config.py` and `.env.example`) selects the backend; the rest of the codebase
  is agnostic. The `postgres` extra (`psycopg[binary]`, plus `pgserver` for real local
  testing) is optional — a plain `sqlite` install has no Postgres dependency at all.
- Verified against a genuine Postgres instance, not mocked: `tests/storage/test_postgres_blob_store.py`
  and `tests/data/test_postgres_connection.py` (9 tests) run automatically whenever the
  `postgres` extra is installed — `tests/conftest.py` starts a real embedded Postgres (via
  `pgserver`, no Docker/system install needed) for the test session.

## Website API boundary

The Next.js website never talks to a database directly and never imports Python code. The
only channel between the Python pipeline/storage layer and the website is a **Python
FastAPI read API**:

```
Python pipeline/storage (src/nfl_predict/data repository interfaces)
        ▼
Python FastAPI read API (src/nfl_predict/api)
        ▼
JSON (HTTP)
        ▼
Next.js website
```

Rules that follow from this, fixed now even though the API isn't implemented until Phase 8:

- **`src/nfl_predict/api` is the only code that is allowed to call into
  `src/nfl_predict/data` repository interfaces on behalf of the website.** Feature/model/
  backtesting/decision code also calls those repository interfaces directly (they're part of
  the same Python process/pipeline), but the *website* reaches them exclusively through this
  API — never through a direct import, never through a Next.js API route that opens SQLite
  or PostgreSQL itself.
- **The API exposes stable, read-only JSON endpoints**, at minimum covering: predictions,
  games, teams, model performance, research findings, and persisted written content (the
  prose described in `docs/WEBSITE_SPEC.md`). It does not expose a generic query interface
  or accept writes from the website — the website is a read-only consumer, per
  `docs/WEBSITE_SPEC.md`'s "the website never computes numbers" rule.
- **This is a real process/network boundary, not just a code-organization convention.** The
  website process cannot reach the database even by accident, because it has no driver, no
  connection string, and no Python import path to it — only an HTTP client pointed at the
  API's base URL.
- **This makes the Phase 10 storage swap (SQLite → PostgreSQL) fully transparent to the
  website.** The FastAPI layer sits behind the same `src/nfl_predict/data` repository
  interfaces described under "Storage" above; swapping `STORAGE_BACKEND` changes what's
  behind the API, not the API's contract or any website code.
- The API's own versioning/stability guarantees (e.g. how a breaking response-shape change
  is rolled out) are a Phase 8 implementation decision, not fixed here — this document fixes
  the *boundary*, not the API's design.

## Python version policy

The supported range is **Python >=3.11, <3.14**, enforced by `requires-python` in
`pyproject.toml`. This is a deliberate, verified choice, not a default:

- **Floor (3.11):** pandas 3.0 requires Python >=3.11; scikit-learn 1.9 requires >=3.11.
  Going lower would mean pinning older releases of the modeling stack for no benefit.
- **Ceiling (<3.14):** as of this writing, Polars — a direct dependency of `nflreadpy` (the
  project's nflverse loader, see `docs/DATA_SOURCES.md`) — has published wheels only through
  cp313. There is no Polars wheel for 3.14 yet, so the `data` extra cannot actually be
  installed on 3.14 regardless of what any other library supports.
- **Recommended pinned version for local development and CI: Python 3.12.** It sits solidly
  inside the supported range with mature wheel coverage across the full stack this project
  needs (numpy, pandas, polars, pyarrow, scikit-learn, lightgbm, xgboost, nflreadpy), one
  release behind the newest, avoiding first-release packaging gaps.
- **This policy is not "whatever Python happens to be installed."** A development machine
  having a newer interpreter available is not a reason to raise the ceiling — the ceiling
  moves only after the full dependency chain has verified wheels for the new version.

**Known gap, flagged for action before Phase 1:** the current development machine has only
Python 3.14.6 installed (confirmed via `py -0p`), which is outside the supported range. The
Phase 0 base install (pyyaml, python-dotenv, pytest) works fine under 3.14 and was verified
there, but attempting to install the `data` extra (nflreadpy/polars) on this machine will
fail until a 3.11–3.13 interpreter (3.12 recommended) is installed and a new virtualenv is
built from it. This is a prerequisite for Phase 1, not a Phase 0 blocker — Phase 0 has no
runtime dependency on the `data` extra.

## Tabular data representation boundary

`nflreadpy` returns Polars DataFrames. This project does not force Polars — or pandas — on
the entire pipeline; the two representations are scoped to different layers:

- **Ingestion (`src/nfl_predict/data`, Phase 1+)** may use Polars directly — it's what
  `nflreadpy` hands back, and re-wrapping every call in a pandas conversion for no reason
  would be pure overhead.
- **The raw store boundary** is where Polars stops being visible to the rest of the
  pipeline. Ingestion code writes raw data out as Parquet (Polars writes Parquet natively) or
  converts to pandas/pyarrow (`.to_pandas()` / `.to_arrow()`) before handing data to storage
  repository interfaces. Repository interfaces in `src/nfl_predict/data` accept and return
  plain pandas/pyarrow/stdlib types — never a Polars-specific type — so no caller outside the
  ingestion module needs to import `polars`.
- **Feature engineering (Phase 2+) and modeling (Phase 3+)** read from the raw/feature store
  via pandas/numpy, as already specified in `docs/ARCHITECTURE.md`'s engineering conventions
  and `docs/MODEL_SPEC.md`. They have no dependency on Polars at all.
- Net effect: `polars` is a dependency of the `data` extra only, isolated to the ingestion
  boundary — it does not appear in the `modeling` extra or in feature/model/backtesting code.

## Raw data provenance

Reproducibility (see `CLAUDE.md`, principle 4) requires being able to reconstruct not just
*that* a prediction used certain feature values, but *which exact upstream data snapshot*
those feature values were computed from. This is a Phase 1 contract: ingestion code must
never silently overwrite a previously fetched raw file, and every fetch must be accompanied
by a provenance record.

**Every externally downloaded dataset must be captured with a provenance manifest recording,
at minimum:**

| Field | Meaning |
|-------|---------|
| `source_name` | Matches an entry in `config/sources.yaml` / `docs/DATA_SOURCES.md`. |
| `dataset_name` | The specific dataset within that source (e.g. `play_by_play`, `schedules`). |
| `requested_range` | The season/date range that was requested (e.g. `seasons=[2018..2023]`). |
| `source_identifier` | The source URL, release tag, or API endpoint identifier used. |
| `source_release_version` | The upstream release/version tag, where the source exposes one (nflverse data is released per-season/per-file with identifiable release tags on GitHub; not every source will have this — record `null` explicitly rather than omitting the field). |
| `retrieved_at` | UTC timestamp of when this fetch actually ran. |
| `local_raw_path` | Path to the immutable local raw file this manifest describes. |
| `content_sha256` | SHA-256 (or comparable) hash of the raw file's bytes. |
| `row_count` | Row count of the fetched dataset. |
| `schema_fingerprint` | A practical fingerprint of the schema actually received — at minimum the sorted column name/dtype pairs, hashed or stored as a list; enough to detect an upstream schema change between snapshots. |

**Rules that follow from this:**

- Raw files are immutable once written. A re-fetch of the same logical dataset writes a new
  file (e.g. path-namespaced by `retrieved_at` or by `content_sha256`) alongside the old one,
  with its own manifest — it never overwrites the previous snapshot.
- If the same logical dataset changes upstream between two fetches, the two manifests'
  `content_sha256` (and likely `row_count`/`schema_fingerprint`) will differ, which is exactly
  the signal that distinguishes the versions — no separate versioning scheme is needed beyond
  keeping both snapshots and both manifests.
- Feature engineering and model runs must be able to trace back, for any feature value they
  used, to the manifest (and therefore the exact raw snapshot) it was computed from. The
  concrete mechanism (e.g. a manifest table keyed by `local_raw_path`, referenced by feature
  computation runs) is a Phase 1/2 implementation decision — this section defines the
  contract the implementation must satisfy, not the storage schema itself.
- See `docs/DATA_SOURCES.md` for how this attaches to the source registry, and
  `docs/BACKTESTING_RULES.md` for how snapshot provenance interacts with the no-look-ahead
  rule.

Implemented in Phase 1 — see `src/nfl_predict/data/provenance.py` (manifest schema, SHA-256,
schema fingerprint) and `src/nfl_predict/data/raw_store.py` (immutable snapshot layout,
duplicate-by-hash detection). Real findings from actually running it against nflverse are in
`docs/PHASE1_DATA_REPORT.md`.

## Configuration

Two distinct kinds of configuration, never mixed:

- **Versioned, non-secret config** — `config/*.yaml` (feature registry, source registry,
  decision thresholds). Checked into git. Loaded via `src/nfl_predict/config.py`.
- **Secrets / environment-specific values** — environment variables, loaded from a local
  `.env` (gitignored) via `python-dotenv` in development, real environment variables in
  production. Documented (names only, no values) in `.env.example`. Includes API keys,
  database connection strings, and any credential.

`src/nfl_predict/config.py` is the single place that reads both and hands callers typed,
already-validated config objects. No other module reads `os.environ` or opens a YAML file
directly.

## Logging conventions

- Every module gets its logger via `logging_conf.get_logger(__name__)` — never `print()`
  for anything other than a CLI's final human-facing output, never the bare root logger.
- Log records are structured: a human-readable message plus `extra=` key/value context
  (e.g. `game_id`, `season`, `week`, `model_version`, `source`). This keeps logs greppable
  and machine-parseable without forcing full JSON logging before it's needed.
- Levels:
  - `DEBUG` — verbose, developer-facing detail (raw request params, intermediate values).
  - `INFO` — normal pipeline progress ("ingested 272 games for season=2023").
  - `WARNING` — recoverable data-quality issues (missing field, filled with default,
    fell back to a secondary source).
  - `ERROR` — an operation failed and did not produce its expected output.
  - `CRITICAL` — reserved for failures that should halt a scheduled run (e.g. Phase 9
    orchestrator aborting before publish).
- Any data-quality check that would silently change a number (imputation, dropped rows,
  fallback source) **must** log a `WARNING` with enough context to find the affected
  row/game later. Silent data repair is a leakage/trust risk.
- Log format and destination (stdout vs. file vs. structured sink) are configured once in
  `logging_conf.py`; modules never call `logging.basicConfig()` themselves.

## Testing structure

- `tests/` mirrors `src/nfl_predict/` package-for-package.
- Unit tests (`tests/unit/`) test individual functions/classes with no network access and
  no dependency on real external data — use small fixtures.
- Integration tests (added starting Phase 1, once there's something to integrate) will live
  under `tests/integration/` and are allowed to touch a real (test) database or cached
  sample data, but never live external APIs in CI.
- No test may depend on wall-clock "today" for correctness (backtesting leakage tests in
  particular must pin explicit as-of timestamps).
- A test is written for behavior that exists. Do not pre-write tests (or code) for a future
  phase's functionality.
