# Phase 8B: FastAPI read API + Next.js website

## Objective

Build the consumer-facing NFL prediction website and the read-only API boundary it depends
on, per `docs/WEBSITE_SPEC.md` and `docs/ARCHITECTURE.md#website-api-boundary`: a Python
FastAPI service that is the *only* way the website reaches persisted prediction/market/
research/decision data, and a Next.js site that never imports Python, never opens a
database, and never calls Anthropic or The Odds API. Free, ad-supported - no subscriptions,
payments, or accounts.

## FastAPI read API (`src/nfl_predict/api`)

- `reconstruction.py` - the only module allowed to call into `src/nfl_predict/data`
  repository interfaces and the Phase 6/7/8A storage modules on the website's behalf. Every
  function returns already-persisted data, never a fresh computation, with one narrow
  exception documented in its own docstring (`model_market_disagreement_points` on the game
  detail endpoint - a same-formula reuse of an already-computed comparison, not a new
  prediction).
- `consumer_language.py` - a closed, exhaustively-keyed translation table
  (`Decision`/`ReasonCode`/`ResearchClassification`/`FailureStatus` values → consumer-facing
  text). Raises `KeyError` on an unmapped code rather than leaking raw enum text, so a newly
  added internal code is a loud test failure, not a silent UI leak.
- `schemas.py` - versioned (`schema_version: "1"`) Pydantic response models. No field ever
  carries a filesystem path or a secret.
- `main.py` - seven endpoints, all read-only:

  | Endpoint | Purpose |
  |---|---|
  | `GET /api/health` | liveness check |
  | `GET /api/nfl/slate/current` | current week's games with model/market/decision/research blocks |
  | `GET /api/nfl/games/{game_id}` | full matchup detail, both market types |
  | `GET /api/nfl/best-bets` | official, currently-open Best Bets (empty state, never forced) |
  | `GET /api/nfl/performance` | verified record from the immutable pick ledger |
  | `GET /api/nfl/model-status` | rule version, model ids, provider availability, data freshness |
  | `GET /api/nfl/decisions/{game_id}` | the exact persisted decision artifacts for a game |

### The Correction-5 lesson, applied again at the API layer

Building `latest_market_point()`/`market_point_for_decision_record()` re-confirmed the
Phase 8A Correction 5 finding: a `provider_event_id` accumulates every historical fetch ever
made for that real-world game. `latest_market_point()` (current slate/matchup pages) uses
only the newest batch; `market_point_for_decision_record()` (the `/decisions` endpoint, and
any future historical reconstruction) uses *only* the exact batch matching a specific
persisted `DecisionRecord`'s own `market_snapshot_timestamp`, and refuses (returns `None`)
if the recomputed `snapshot_reference` doesn't match what was recorded at decision time -
never blended, never "latest" substituted for "referenced." `tests/api/test_reconstruction.py`
and `tests/api/test_routes.py` both prove this with a synthetic older/newer-batch scenario
where blending would silently produce the wrong line.

### Two real bugs found and fixed while building this against real data

Both were caught by testing this phase's own reconstruction logic against the real, messy
`data/` directory accumulated over the whole project, not synthetic fixtures alone:

1. **`latest_research_summary` sorted by `run_id` string, not by each run's real
   `research_timestamp`.** Most run_ids are ISO timestamps and sort correctly on their own,
   but a manually-named run_id (`run_id="pilot_20260911"`, from the Phase 6 pilot) sorts
   lexicographically *after* any `"2026-09-12T..."` id (`"p" > "2"` in ASCII) despite being
   chronologically almost a day older. This served a stale Sep-11 research summary as
   "latest" over more than 30 real, later Sep-12 runs. Fixed by reading each run's own
   `research_timestamp` and taking the real max - regression-tested in
   `tests/api/test_reconstruction.py::test_latest_research_summary_sorts_by_real_timestamp_not_by_run_id_string`.
2. **`tests/live/test_secrets_never_persisted.py` leaked fake research artifacts into the
   real `data/research/` directory on every full test-suite run.** That test deliberately
   sets a fake (but present) `NFL_RESEARCH_LLM_API_KEY` and mocks `urllib.request.urlopen`
   to return a fake 401 - real enough to reach `run_research_for_game()`'s `LLM_FAILURE`
   path and call `write_research_run()`, but the test never redirected
   `nfl_predict.research.storage`'s `get_settings()` to a temp directory the way it already
   did for `live_snapshot_store`. Six leaked `failed_run` artifacts (findings sharing the
   fake mock's exact `HTTP 401: {}` empty-body fingerprint) were found and deleted from the
   real `data/research/season=2026/week=1/2026_01_DEN_KC/` directory; the test itself is now
   fixed to patch `nfl_predict.research.storage.get_settings` too.

### Tests

20 new backend tests, no live network calls (`tests/api/test_reconstruction.py`,
`tests/api/test_routes.py`, `tests/api/conftest.py`), covering: exact vs. latest market
reconstruction (item 6's core requirement), the research-summary sort fix, model-prediction
supersession handling, per-market-type decision lookup, Best Bets empty/non-empty states,
performance with an empty ledger (zeros, not fabricated numbers), model-status provider
gating, and 404 handling. **581/581 backend tests passing** (561 carried over from Phase 8A
+ 20 new), zero live network calls anywhere in the suite.

## Next.js website (`website/`)

Next.js 16 (App Router, Turbopack), TypeScript, React 19, Tailwind CSS v4. Every page is an
`async` Server Component that calls `src/lib/api.ts` - the *only* place in this app that
performs a `fetch()` - which talks exclusively to the FastAPI service's JSON endpoints. No
file in `website/` imports a Python module, opens a database connection, or references an
Anthropic/Odds API key; `src/lib/types.ts` mirrors `nfl_predict.api.schemas` field-for-field
so a response shape change surfaces as a TypeScript error, not a silent runtime mismatch.

### Routes

| Route | Purpose |
|---|---|
| `/` | homepage - performance headline, Best Bets teaser, current slate, model-record/methodology teasers |
| `/nfl/picks` | full Best Bets page with the explicit empty state |
| `/nfl/games/[gameId]` | individual matchup - Model Prediction and Betting Decision are visually separate sections, both spread and moneyline shown |
| `/nfl/performance` | verified record + predefined streak windows |
| `/methodology` | static, plain-language explanation |
| `/about` | static about page |

Each game page has a unique, server-rendered `<title>`/meta description/canonical URL via
`generateMetadata`, built from real API data (team names, week) - never a generated
keyword-stuffed template.

### Design

Tailwind CSS variables in `globals.css` define a light/dark-aware neutral palette with one
restrained accent color and five decision colors (`qualified`/`lean`/`watch`/`veto`/`no-bet`).
No flashing UI, no gradients, no fake urgency copy. Ad placements
(`src/components/AdSlot.tsx`) are reserved, labeled, dashed-border placeholders - no ad
network is wired up yet, per this phase's explicit scope.

### Consumer language

Every internal enum value the API returns (`ReasonCode`, `Decision`, `ResearchClassification`)
already arrives translated (`{"code": "...", "label": "..."}`) - the frontend never
re-implements this mapping, so there is exactly one source of truth for what a code means to
a reader (`consumer_language.py`). `ReasonCodeList`/`DecisionBadge` render only the `label`.

### Tests

40 new frontend tests (Vitest + React Testing Library, no live network call - all API calls
mocked), covering: pure formatting helpers, `DecisionBadge` for every decision value plus the
pending state, `ReasonCodeList` empty state and real-label rendering (never raw enum text),
`GameCard` real-data rendering and unavailable-market/model states, `PerformanceHeadline`'s
three states (real headline / settled-but-no-headline / no history), the homepage (real slate
rendering, Best Bets empty state, and resilience when the API is unreachable), and the game
detail page (the Model-Prediction-vs-Betting-Decision separation, translated reason codes,
and `notFound()` on a real 404 from the API). `npm run build` (production build, including a
full TypeScript check) and `npm run lint` both pass clean.

## Real data verification (not fixture-only)

Both services were run locally against this project's actual accumulated `data/` directory
(the FastAPI service on `:8000`, the Next.js dev server on `:3100` - port 3000 is OS-reserved
on this machine) and checked in a real browser:

- The homepage renders the real current Week 1 slate, including games with `game_status:
  "final"` (correctly showing `model.available: false`/`market.available: false` since
  Phase 8A's live pipeline never ran for those historical games) alongside upcoming games.
- The `2026_01_DEN_KC` card/matchup page shows the REAL Correction-5 market refresh (KC -2.5,
  -142 moneyline, 56% no-vig, 9 real sportsbooks, snapshot timestamp matching the actual
  refresh), the REAL Elo/Ridge/LightGBM margins from the last live model run, and the REAL
  research classification (`VETO_CONSIDERATION`, "Significant risk flagged") from the last
  successful Anthropic validation call - not a single synthetic value anywhere on the page.
- The decision cards honestly show `NO_BET` / "Waiting for complete market/research data"
  for this game, because the persisted `DecisionRecord` predates the market/research
  refresh - the site never claims a fresher decision than what was actually computed.
- `/nfl/picks` and `/nfl/performance` both render their genuine empty states, because no
  Best Bet has ever been published in this project (Phases 7/8A both note the same fact) -
  nothing was faked to make the page look more populated.
- Dark mode (`prefers-color-scheme`) and a 375px mobile viewport were both checked visually;
  no console errors on any page.

## Known limitations

- `/api/nfl/decisions/{game_id}` and `/api/nfl/games/{game_id}` resolve `game_id` against
  only the *current* season's schedule - a past-season game_id 404s. Multi-season game
  lookup is a reasonable Phase 9/10 extension, not needed yet since only one season's data
  exists.
- `/api/nfl/model-status`'s `last_*_at` freshness fields scan only the current week's games,
  not full history - an intentional, inexpensive scope matching what a status page needs.
- No ad network, payments, accounts, or subscriptions - explicitly out of scope for this
  phase.
- Not deployed; no production PostgreSQL; no Vercel configuration. Both services run only
  locally, per this phase's explicit stop condition.

## Follow-up: generated game-preview articles (`src/nfl_predict/content`)

A short, offline-generated "why" article per game, added after Phase 8B shipped, per
`docs/WEBSITE_SPEC.md`'s already-anticipated "content generation" step
(`prompts/prediction_writer.md`, previously unimplemented - superseded by the real, wired-up
`prompts/prediction_writer_v1.md`, since a v1 has to exist before it can be frozen).

- **Never generated at request time** - `nfl_predict.content.prediction_writer.write_prediction_preview()`
  is a pipeline step, not a website code path. It gathers the exact same real, persisted data
  `nfl_predict.api.reconstruction` already assembles for `/api/nfl/games/{game_id}` (no second,
  competing way of reading the same state), fills the versioned prompt template, and calls a
  plain Anthropic Messages request - no tools, no web search, the cheapest possible real call
  shape, since the article only narrates numbers already computed.
- **Same discipline as research**: a real, computable pre-flight cost-floor guard
  (`NFL_CONTENT_MAX_COST_PER_GAME`, default $0.05, reusing `research.cost_tracking`'s generic
  guard functions - they were never research-specific despite the module name), immutable
  per-run storage (`data/content/previews/...`, hash-verified, mirroring `research.storage`'s
  pattern) with an explicit `FailedPreviewRun` for every failure mode
  (`PROVIDER_UNAVAILABLE`/`COST_BUDGET_EXCEEDED`/`EMPTY_RESPONSE`/`LLM_FAILURE`) - never a
  silently-missing article.
- **Hard content constraint**: the prompt forbids introducing any number not already in the
  gathered context, and forbids transforming a number with the model's own arithmetic (no
  unit conversions, no implied-odds math) - the same "an LLM never invents or adjusts a
  number" rule this whole project enforces everywhere else.
- **API**: `GET /api/nfl/games/{game_id}` gained a `preview` block
  (`available`/`text`/`generated_at`/`prompt_version`/`model_provider`/`model_name`), sourced
  via `reconstruction.latest_preview()` - which, from the start, picks the latest run by its
  real `generated_at` timestamp, not by sorting `run_id` strings (the exact bug already found
  and fixed for `latest_research_summary`).
- **Website**: a new "Game Preview" section on the matchup page (`GamePreview.tsx`), labeled
  "AI-assisted analysis" for transparency, rendering nothing at all when no article has been
  generated yet (verified visually - no layout gap, no broken state).
- **Tests**: 12 new backend (11 in `tests/content/`, 1 route-level integration test in
  `tests/api/` proving a real generated article flows through `/api/nfl/games/{game_id}`)
  plus 3 new frontend component tests - 593 backend + 43 frontend total, zero live network
  calls (the Anthropic provider's pre-flight cost guard and forced-failure test doubles
  exercise every failure path without ever reaching the network).
- **Not yet generated for real**: no live call has been made for this step yet - it was built
  and verified entirely against the fixture provider and the pre-flight-only paths on the
  real Anthropic provider, per this session's standing rule that a live, billed call requires
  its own explicit go-ahead.

## Running locally

```bash
# Terminal 1 - the read API
pip install -e ".[dev,api]"
uvicorn nfl_predict.api.main:app --reload --port 8000

# Terminal 2 - the website
cd website
npm install
npm run dev -- --port 3100   # or `npm run dev` if port 3000 is free on your machine
```

Set `website/.env.local`'s `NEXT_PUBLIC_API_BASE_URL` if the API isn't on
`http://localhost:8000`. Backend tests: `pytest tests/api`. Frontend tests: `cd website && npm test`.
