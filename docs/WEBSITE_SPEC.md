# Website Spec

## Purpose

Specifies the public website's scope and its non-negotiable relationship to the prediction
system underneath it. Not built yet (Phase 8) — this is the contract implementation must
satisfy when that phase starts. Do not create a `website/` (or similar) directory before
Phase 8 begins.

## Stack

Next.js, TypeScript, React, Tailwind, per the project's technical direction. Pages must
remain indexable by search engines (server-rendered/static-generated, not client-only
rendering for primary content).

## Core rule: the website never computes numbers, and never touches storage directly

Every predicted score, probability, spread, total, edge, or performance statistic shown on
the site is read from the structured database — written by the Python pipeline
(`src/nfl_predict`) — not computed, adjusted, or invented by any code or LLM call in the
website layer. If a number needs a new derivation, that derivation is added to the Python
side and stored, not calculated in a Next.js API route or component.

**The website reaches that data exclusively through a Python FastAPI read API
(`src/nfl_predict/api`), never directly.** Full rationale and diagram in
`docs/ARCHITECTURE.md#website-api-boundary`; the rule as it applies here:

- Next.js contains no SQLite or PostgreSQL driver, no connection string, and no import of
  any `src/nfl_predict` Python module. It only makes HTTP requests to the FastAPI service.
- A Next.js API route or server component that queried the database directly — even
  read-only, even "just this once" — would violate this boundary as much as one that
  computed a number itself. The fix is always the same: expose or extend a FastAPI endpoint,
  never reach past it.
- This is fixed as an architecture decision now (Phase 0/1), even though neither the FastAPI
  service nor the website exists yet (both land in Phase 8) — so no Phase 8 implementation
  choice can reopen it.

LLM-generated prose (e.g. a written game preview that references numbers it's handed,
methodology explanations) never produces or restates a number in a way that could drift
from the stored value — any prose that cites a number cites the exact stored value, not a
paraphrased approximation.

## The website never calls an LLM at request time

This is stricter than "the LLM never invents numbers": **the public website does not make
live LLM calls when an ordinary visitor loads a page, at all** — not even for prose. LLM
content generation (`prompts/prediction_writer.md`) runs as a step in the prediction
pipeline, offline, using the already-finalized prediction snapshot as input, and the
resulting prose is persisted — versioned by prompt version and timestamped — in the same
database the numeric predictions live in. Next.js pages read that persisted prose the same
way they read every other field: from the database, never by invoking an LLM inside a page
component or API route.

This is a real architectural boundary, not a style preference, because it's what delivers:

- **Reproducibility** — what a visitor saw on a given date is exactly reconstructable, same
  as a numeric prediction; an LLM call made fresh per page load would not be.
- **Predictable cost** — content generation cost scales with games predicted, not with page
  views.
- **Fast page loads** — pages render from stored data, not from a live model round-trip.
- **Stable SEO content** — a page's text doesn't change between one crawl and the next
  unless the pipeline actually republished something.
- **Auditability** — exactly what users saw is the thing that's stored, not a
  regenerate-on-demand approximation of it.

Where this step lives in code (part of the pipeline package, invoked as part of publishing)
is decided in Phase 8, per "Phase boundary" below — it is Python pipeline code either way,
never website-layer code.

## Planned pages

- **Home** — current week overview, headline picks if any exist (never forced, per
  `docs/DECISION_ENGINE.md`).
- **Current NFL Week** — all games for the active week with model outputs and decisions.
- **Individual Game Pages** — full detail for one game: both model tracks' predictions,
  market comparison, decision + reasoning, research agent findings (with sources), and
  historical head-to-head context if useful.
- **Best Bets** — filtered view of BET/LEAN decisions only; can legitimately be empty.
- **Expert Picks** (`/nfl/expert`, built) — a human's own picks with their own record on
  the same page, served by `/api/nfl/expert-picks`. The one deliberate exception to "never
  manually authored": the picks and their short notes are hand-entered, but the record,
  units, and grading are still computed by code from the immutable ledger (see
  `docs/DECISION_ENGINE.md`'s `EXPERT_PICKS`), and it is visibly separate from the model's
  Best Bets.
- **Schedule** — full season schedule.
- **Teams** — per-team pages (roster/coaching context, season performance).
- **Model Performance** — backtested and live track record, using the same metrics defined
  in `docs/BACKTESTING_RULES.md` (calibration, MAE, ROI, sample sizes, confidence
  intervals) — not a cherry-picked win/loss record.
- **Methodology** — plain-language explanation of the system (independent vs. market-aware
  models, backtesting approach, decision engine, research agent's role and limits).

## Generation

- Game pages and model-performance pages are generated from structured database records
  (predictions, backtests, decisions) — never manually authored per game.
- Every prediction displayed shows its **timestamp** and **model version** (per
  `docs/MODEL_SPEC.md`), so a user can tell how fresh it is and which model produced it.
- Historical performance pages update only when the pipeline records completed-game
  results (Phase 9 step "update historical performance when completed games exist") — the
  website itself never marks a game complete or edits a result.

## Phase boundary: what Phase 8 builds vs. what Phase 9/10 add

Phases are sequential (`CLAUDE.md`#phase-discipline) — Phase 8 runs before Phase 9 and
Phase 10 exist, so Phase 8 cannot be designed to wait on artifacts those later phases
produce. Each phase's job is scoped so that's never necessary:

- **Phase 8 builds and tests the FastAPI read API and the website against whatever storage
  already exists at that point**: SQLite/Parquet (per `docs/ARCHITECTURE.md#storage`),
  accessed by `src/nfl_predict/api` only through the `src/nfl_predict/data` repository
  interfaces — the same interfaces the rest of the pipeline already uses. The website's
  *only* data-fetching approach is HTTP calls to that API — this is not a Phase 8 decision
  to make, it's fixed by `docs/ARCHITECTURE.md#website-api-boundary`. What Phase 8 does
  decide: the API's concrete endpoint shapes, based on what `src/nfl_predict/data` actually
  exposes by then. Phase 8 is not "done" until both the API and the site are working,
  tested, and running against real (if pre-production) data.
- **Phase 9 (automation) orchestrates the already-working pipeline, API, and website** — it
  adds `run_nfl.py` steps that call ingestion → features → models → research → decision →
  content generation → website content regeneration, on a schedule or on demand, and
  ensures the API/website are serving current data after a run. It does not change how the
  website reaches data; it automates *when* the upstream steps run.
- **Phase 10 (production) swaps the storage backend to PostgreSQL** behind the same
  repository-interface boundary (`STORAGE_BACKEND=postgres`, per
  `docs/ARCHITECTURE.md#storage`), and adds hosting/scheduling/monitoring. Because the
  website only ever talks to the FastAPI layer, and the FastAPI layer only ever talks
  through the repository-interface boundary, this swap changes backend configuration behind
  `src/nfl_predict/api` — it touches neither the API's contract nor any website code.

The test of whether Phase 8 was built correctly: Phase 10 productionizing the database and
hosting should require no rewrite of website page/component logic or API endpoint contracts,
only configuration.

## Open items for Phase 8

- Exact FastAPI endpoint shapes/routes for predictions, games, teams, performance, research,
  and persisted content — decided when Phase 8 starts, based on what
  `src/nfl_predict/data` exposes by then. The requirement that these are the website's only
  data path is not open — see `docs/ARCHITECTURE.md#website-api-boundary`.
- SEO/structured-data (e.g. sports-event schema.org markup) — deferred to Phase 10 alongside
  analytics, but page structure in Phase 8 should not preclude adding it later.
- Where the content-generation step (`prompts/prediction_writer.md`) lives in code — most
  likely `src/nfl_predict/pipeline` or a small dedicated module; it writes to storage
  directly (it's pipeline code, not website code) and is unrelated to the API boundary,
  which only governs how the *website* reads data. See "The website never calls an LLM at
  request time" above.
