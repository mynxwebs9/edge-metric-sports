# Phase 8A — Live NFL Prediction and Market Pipeline

## Objective

Phase 7 built a deterministic decision engine and an immutable pick ledger, but had no live
data to feed them: no live odds provider, no live Ridge/LightGBM feature pipeline, no
unified orchestration, and `QUALIFIED_BET` was structurally unreachable. Phase 8A closes
those operational gaps without touching Phase 7's rules, without building the website, and
without fabricating any data a real provider didn't actually supply.

## Correction 1: real per-game market/research wiring

The first pass of Phase 8A left a real gap the initial pilot's "28/28 NO_BET" result
obscured: `run.py` hardcoded `MarketPoint(available=False)` and `ResearchPoint(available=False)`
for every game **unconditionally**, regardless of `odds_status`/`research_status`. Provider
*availability* (whether `TheOddsAPIProvider`/`AnthropicMessagesProvider` could be
constructed - i.e. whether a key was set) was being conflated with actual per-game *market
data* / *research findings existing*. Even with real credentials configured, that first pass
would still have produced `MISSING_LIVE_DATA` for every game, because nothing ever fetched
real odds or invoked `LLMResearchProvider.run_research()`. This section documents the fix;
the rest of this report has been updated in place to describe the corrected system.

New modules:
- `src/nfl_predict/live/odds_ingestion.py` - `fetch_and_snapshot_live_odds()` calls a real
  `OddsProvider.get_events()`/`get_markets()`, maps each event to a `game_id`
  (`odds_mapping.py`), persists every sportsbook snapshot through the existing append-only
  store (`market/live_snapshot_store.py`) **before** using it, computes a documented
  consensus (`market_consensus.py` - no-vig probability average for moneylines, median line
  for spreads, never a blind average of American odds), and returns one real `MarketPoint`
  per matched game. An event with no matching `game_id`, or a game with neither a usable
  spread nor moneyline, is skipped - never fabricated - so that game's `MarketPoint` stays
  `available=False`.
- `src/nfl_predict/live/research_live.py` - `run_live_research_for_game()` actually calls
  Phase 6's existing `run_research_for_game()` (build packet -> invoke provider -> parse ->
  evaluate -> persist -> ledger entry) for real, selected games, then builds `ResearchPoint`
  only from the freshly-persisted, evaluated artifact
  (`nfl_predict.decision.input_packet.build_research_point_from_stored_run`, extracted from
  Phase 7's existing packet-builder so there is exactly one implementation, not two).
  `select_games_for_research()` orders candidates by Phase 6's existing trigger-priority
  score (`research.trigger` - a ranking signal only, per its own documented stance, never a
  gate) and researches the top `--max-research-calls` (default 3) unless `--research-all` is
  passed.
- `nfl_predict.research.llm_provider.AnthropicMessagesProvider` now includes Anthropic's
  server-side web-search tool (`web_search_20250305`) in its request by default. A plain
  Messages API call has no browsing capability, and `matchup_research.md`'s prompt requires
  real, current, cited sources - without a tool, "actually invoke research" would only
  produce an ungrounded (and, for a Week 1 2026 game, necessarily post-training-cutoff)
  answer. `max_web_search_uses` (default 5) bounds cost per call; `enable_web_search=False`
  restores the original no-tool behavior.
- `run.py` now builds each game's `MarketPoint` from `odds_ingestion`'s real result (falling
  back to `available=False` only when that game genuinely has no match), computes each
  game's `ResearchPoint` only for games actually selected and researched, and computes
  `system_health.market_age_seconds`/`research_age_seconds` from the REAL snapshot/research
  timestamps via `nfl_predict.decision.staleness.compute_age_seconds` (previously hardcoded
  `None`, which `is_stale()` always treats as stale anyway - but now a fresh real timestamp
  correctly reads as fresh).

**Verification, since no live credential is configured in this environment (see Missing
credentials below):** a dedicated integration test
(`tests/live/test_live_market_and_research_wiring.py::test_full_pipeline_with_synthetic_valid_odds_and_research_reaches_gates_3_to_9`)
builds a fully realistic packet - a real mocked odds fetch (persisted through the real
snapshot store), a real fixture-backed research invocation (persisted through the real
research storage + evaluator), and model outputs shaped like a genuine live prediction - and
confirms the run reaches **`QUALIFIED_BET`, for both spread and moneyline**, through the
completely unmodified `decide()`/`config/decision_rules.yaml`. This proves the previous
"structurally unreachable" state was purely a data-plumbing gap, not a property of the rules.

Structured injury data (`NFL_INJURY_API_KEY`) remains deliberately outside every gate: rule
set v1 has no injury-specific `required_input` (see `config/decision_rules.yaml`), so
`injury_status` is checked/reported as its own health component and never touches
`market.available`/`research.available` or contributes a `MISSING_LIVE_DATA` reason on its
own -
`tests/live/test_live_market_and_research_wiring.py::test_injury_provider_unavailable_never_causes_missing_live_data_on_its_own`
proves this directly.

## Correction 2: credit-exhaustion and INVALID_JSON emergency fix

Correction 1's real wiring was, itself, immediately exercised in a validation run
(`python -m nfl_predict.live.run --research-all --show-decisions`) that surfaced two
production-blocking bugs: The Odds API request consumed nearly an entire 500-credit free
allowance in one run, and every Anthropic research attempt failed with
`Expecting value: line 1 column 1` (`INVALID_JSON`). This section documents both root
causes, the fixes, and the real live re-validation that followed once mocked tests passed.

### ODDS: root cause and fix

**Root cause.** `odds_ingestion.fetch_and_snapshot_live_odds()` called
`provider.get_events()` once (free/cheap - The Odds API's `/events` endpoint returns every
upcoming event for the sport, not scoped to the current week) and then called
`provider.get_markets(event.provider_event_id)` - a SEPARATE, billed request to
`/sports/{sport}/odds` - **once per event returned**, with no week-filtering at all. Each
such request cost `len(markets) x len(regions)` credits regardless of the `eventIds` filter
narrowing the response (`markets=spreads,h2h,totals` = 3 markets x `regions=us` = 1 region =
3 credits/request, per The Odds API's documented per-request cost formula). With roughly
150-170 events remaining in the season at the time, `~160 events x 3 credits ≈ 480-500
credits` in one run - matching the observed near-total exhaustion of a 500-credit
allowance almost exactly.

- **Exact number of HTTP calls the old code made:** 1 (`/events`, free) + 1 per event
  returned by that call (uncapped, unscoped to the current week - effectively the entire
  remaining season's event count, not the ~14 games actually needed).
- **Exact endpoint(s):** `GET /v4/sports/americanfootball_nfl/events` (free) and
  `GET /v4/sports/americanfootball_nfl/odds` (billed), called once per event.
- **Markets/regions per request (old):** `markets=spreads,h2h,totals`, `regions=us` → 3
  credits per request.

**Fix.** `TheOddsAPIProvider.get_markets_for_sport()` is now the one real, credit-costing
call: a single request to `/odds` with **no `eventIds` filter** - the featured-odds endpoint
returns odds for every event of the sport in one response, at the SAME `markets x regions`
cost as a single-event request. `get_markets(event_id)` (kept for interface completeness)
now filters an instance-memoized `get_markets_for_sport()` result locally - a second call
never issues a second HTTP request. `odds_ingestion.fetch_and_snapshot_live_odds()` calls
`get_events()` once (to build the team-name → game_id mapping) and
`get_markets_for_sport()` once, groups results locally by `provider_event_id`, and accepts
an optional `game_ids` list to scope which MATCHED games get processed/persisted (not the
number of HTTP calls, already fixed at one). `MARKETS` is now `spreads,h2h` only - totals
were never worth their extra credit multiplier and remain non-actionable under the decision
engine anyway.

- **New expected calls per full NFL slate refresh:** 2 total (`/events` once, free；`/odds`
  once, billed) - regardless of whether the slate has 1 game or the entire season's events.
- **Expected credits per slate refresh:** 2 (`spreads,h2h` = 2 markets x `us` = 1 region).
- **Tests proving reuse:**
  `tests/market/test_odds_provider.py::test_get_markets_for_sport_is_memoized_per_instance`,
  `::test_get_markets_for_sport_requests_no_eventids_filter_and_no_totals`,
  `tests/live/test_odds_ingestion.py::test_a_full_slate_of_many_events_still_uses_exactly_one_odds_fetch`
  (150 synthetic events, still exactly 1 `get_markets_for_sport()` call),
  `::test_research_packet_construction_never_imports_an_odds_provider` (structural proof
  research can't trigger a fresh fetch).
- **Quota guard configuration:** `NFL_ODDS_MAX_CREDITS_PER_RUN` (default 5) is checked
  BEFORE any request that would exceed it - `OddsCreditBudgetExceededError` is raised with
  zero HTTP calls made once the budget is spent. A 401/403 (`OddsQuotaExceededError`,
  covering The Odds API's `OUT_OF_USAGE_CREDITS` error) is NEVER retried; a 429/5xx
  (`OddsRequestError`) gets up to `MAX_TRANSIENT_RETRIES=2` bounded, backed-off retries,
  each still counted toward the same single request's credit charge, never an extra one.
  Real response headers (`x-requests-last/used/remaining`) are captured into
  `OddsUsage`/surfaced as `odds_api_usage` in every automation report; the API key never
  appears in a URL used for logging/errors (`redacted_url` only).

**Real live re-validation (Part C Step 2):** `python -m nfl_predict.live.run --odds-only
--no-dry-run` was run against the real account. The Odds API returned a genuine
`HTTP 401 OUT_OF_USAGE_CREDITS` - the account had zero credits remaining (consistent with
the original incident; a fresh key set later in the same session hit the identical error,
confirming credits are tied to the account, not the specific key). This is exactly the
scenario the new guard was built for, and it worked correctly: **one attempt, no retry**,
the API key redacted in the surfaced error, and the failure isolated to only the
`odds_ingestion` health component without halting the rest of the run. A full end-to-end
odds fetch against a genuinely funded account has not been observed live in this
environment - the account's credit exhaustion is an external fact, not a remaining code
gap.

### ANTHROPIC: root cause and fix

**Root cause.** A Messages API turn using the server-side web-search tool returns MULTIPLE
content blocks - narration `text` blocks, `server_tool_use` blocks, `web_search_tool_result`
blocks. The old code concatenated every `text`-type block and ran `json.loads()` on the
result. If the model's first text block was narration ("Let me research...") rather than the
final JSON, the concatenated string wasn't valid JSON from position 0; a mid-turn
`pause_turn` response (a real, observed stop condition for long research turns) could have
no final text block at all, producing an empty string - `json.loads("")` raises exactly
`Expecting value: line 1 column 1`.

- **Actual content-block structure observed** (via real calls in this session): `text`
  (narration), `server_tool_use` (search invocations - 3 per successful call),
  `web_search_tool_result` (search results), and finally `tool_use` (the custom
  `submit_research_findings` call once the fix was in place).
- **stop_reason handling:** yes, changed. `_call()` now checks `stop_reason == "pause_turn"`
  explicitly and resends the conversation with the partial assistant turn appended (bounded
  at `MAX_PAUSE_TURN_CONTINUATIONS=5`) rather than treating a mid-turn response as final.

**Fix.** The model never hands back JSON as free text at all anymore.
`AnthropicMessagesProvider` gives the model a second, custom tool
(`submit_research_findings`) alongside the hosted web-search tool, with `tool_choice: "auto"`
(deliberately NOT forced - forcing it from turn 1 would require calling it before any
search could happen). When the model calls it, Anthropic has already parsed `input` into a
real Python dict per the tool's `input_schema` - that dict (re-serialized to a string only
for backward compatibility with the existing `parse_research_output(raw_text: str, ...)`
signature) is what reaches the parser, never concatenated narration text.

- **Whether Structured Outputs is now used:** attempted, then reverted after real evidence.
  Anthropic's `strict: true` tool-use mode was tried first (would guarantee schema
  conformance) but a real call returned an immediate HTTP 400:
  `"The compiled grammar is too large... Simplify your tool schemas or reduce the number of
  strict tools"` once the nested claim/source/external-prediction arrays were included in
  the schema. Per this correction's own instruction to use the officially-supported
  equivalent rather than invent an unverified workaround, `strict` was removed - the tool
  now uses plain (non-strict) custom tool use, which a real call already proved reliably
  invokes the tool with a well-formed object.
- **Schema used:** the `submit_research_findings` tool's `input_schema` mirrors
  `ResearchFindings`' full shape (built from the same enums `research/schemas.py` defines,
  so it can't silently drift from what the parser accepts). Every property is listed in
  `required` (nullable types stand in for genuinely-optional fields, e.g.
  `publication_timestamp`) and every object sets `additionalProperties: false` - the
  strict-schema convention, kept even without the `strict` flag itself, since it still
  guides generation. A real call also confirmed Anthropic's schema validator rejects
  `minimum`/`maximum` keywords on `number` types (`confidence_in_fact`'s [0, 1] range is
  still enforced downstream by `nfl_predict.research.schemas.Claim.__post_init__`).
- **A second, distinct real failure and fix:** even non-strict, one call returned a tool
  call missing every short trailing field (`qb_status` through `research_classification`)
  while the larger, citation-heavy leading arrays were present - consistent with hitting
  `max_tokens` mid-generation. Fixed by raising `max_tokens` from 8192 to 16000 and
  reordering the schema so short, critical fields are declared first.
- **Single-game live validation result (Part C Step 3):** SUCCESS, on the attempt after all
  of the above fixes were applied in sequence (four real calls total: one incomplete
  generation exposing the missing-field gap, two fast/free HTTP 400 rejections diagnosing
  the strict-mode and min/max issues, and one full success). Real game `2026_01_DEN_KC`:
  `research_classification=MIXED`, `overall_materiality=MAJOR(3)`, 6 material facts with
  real sourced URLs (e.g. a real CBS Sports Mahomes-ACL-recovery update), honest
  `missing_information` entries (e.g. "No weather forecast data located for the specific
  kickoff window"). Hash-integrity verified
  (`verify_research_run_integrity` → `True`); `build_research_point_from_stored_run()` on
  the stored artifact confirmed `ResearchPoint.available=True`.
- **Searches/tokens/cost for that one successful game:** 3 web searches (Anthropic's
  `usage.server_tool_use.web_search_requests`, logged); 69,942 input tokens, 7,586 output
  tokens (from the automation report's `research_cost`); a USD estimate was not computed -
  `LLMCallResult.estimated_cost_usd` is `None` for this provider (the Messages API response
  doesn't return a dollar figure directly, and this correction did not add published
  per-token pricing math - a known limitation, see below).
- **Diagnosability improvements made alongside the fix:** `_post()` now captures and
  surfaces the real HTTP error body (truncated, secret-free) on any failed request - the
  original "HTTP Error 400: Bad Request" with no body was itself an obstacle during this
  debugging session; `_extract_submit_tool_input()` raises a compact, secret-free
  diagnostic (response id, `stop_reason`, content block types, text-block count, usage) when
  the model never calls the submit tool at all, rather than crashing on a bare parse error.
- **Cost cap:** `max_web_search_uses` defaults to 3 (`NFL_RESEARCH_MAX_WEB_SEARCHES_PER_GAME`,
  configurable) - the real successful call used exactly 3.

## Correction 3: Anthropic research cost controls

Before another full-slate live run, this correction adds real usage/cost recording,
configurable per-game/per-run limits, prompt caching for the static half of the research
request, and a "Research:" summary section in every automation report - all validated with
mocked tests, then confirmed with one deliberate real call.

### 1. Real usage recording

`LLMCallResult` now carries `cache_creation_input_tokens`, `cache_read_input_tokens`, and
`web_search_requests` alongside `input_tokens`/`output_tokens`/`estimated_cost_usd` - all
read directly from Anthropic's own `usage` object (`usage.cache_creation_input_tokens`,
`usage.cache_read_input_tokens`, `usage.server_tool_use.web_search_requests`), accumulated
correctly across any `pause_turn` continuations. `CostTracker` (`research/cost_tracking.py`)
records the same fields per run.

**The dollar figure comes from exactly one place:** `config/llm_pricing.yaml`, a real,
sourced pricing table (fetched live from
[platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing)
on 2026-09-12 - NOT guessed from training data) —

| Rate | Claude Sonnet 5 |
|---|---|
| Base input | $2.00 / MTok |
| 5-minute cache write | $2.50 / MTok |
| 1-hour cache write | $4.00 / MTok |
| Cache read (hit) | $0.20 / MTok |
| Output | $10.00 / MTok |
| Web search | $0.01 / search (flat, plus ordinary input-token cost for search-result content) |

`estimate_cost_usd()` looks up the configured model's rates and returns `None` (never a
fabricated number) for a model with no pricing entry. Nothing in Python hard-codes a dollar
figure - `tests/research/test_cost_tracking.py::test_call_uses_the_configured_pricing_table_not_a_hard_coded_rate`
proves the computed cost changes when the configured table changes.

### 2. Configurable limits

| Env var | Default | Enforcement |
|---|---|---|
| `NFL_RESEARCH_MAX_WEB_SEARCHES_PER_GAME` | 3 | Passed to Anthropic's own `max_uses` on the web-search tool - enforced server-side. |
| `NFL_RESEARCH_MAX_ESTIMATED_COST_PER_GAME` | $1.00 | Pre-flight: `assert_known_cost_floor_within_budget()` computes a REAL, known-floor cost (`max_tokens` output + `max_web_search_uses` searches, both real configured ceilings - deliberately excluding unpredictable search-result input-token growth) and refuses the call with `ResearchCostBudgetExceededError` **before any HTTP request** if that floor alone already exceeds budget. |
| `NFL_RESEARCH_MAX_GAMES_PER_RUN` | 3 | `run.py`'s `select_games_for_research()` cap, absent `--research-all`. |

A cost-budget refusal is a new, explicit `FailureStatus.COST_BUDGET_EXCEEDED` (distinct from
`LLM_FAILURE`/`INVALID_JSON`) - `research_live.py` surfaces it as `LiveResearchOutcome.status
== "cost_capped"`, and `run.py`'s research loop stops selecting further games for the REST OF
THAT RUN once it sees one, rather than repeating the same (or a barely-under-cap) refusal
across every remaining selected game. This is the "fail safely... rather than silently
overspending" behavior the correction asked for - a real, computed refusal, never a guess
about how large an actual response might grow.

### 3. Prompt caching

The prompt is now split into two files:
- `prompts/matchup_research_system_v1.md` - the STATIC half (role framing, what-to-research
  categories, source hierarchy, fact/opinion rules, materiality rubric, hard constraints) -
  sent as the Messages API `system` parameter with `cache_control: {"type": "ephemeral"}`.
- `prompts/matchup_research_v2.md` - the DYNAMIC half (the "Context provided to you"
  section: team names, kickoff, Elo/Ridge/LightGBM predictions, market info) - genuinely
  different every call, sent uncached as the `messages` user turn.

Both files are word-for-word the same instructions as the original `matchup_research_v1.md`
(now frozen, unedited - it has already been used to generate real stored research runs) -
only the delivery mechanism changed, which is why the prompt version is now
`matchup_research_v2`, not an in-place edit of v1. **Research methodology did not change.**

The `tools` array (the web-search tool definition AND the `submit_research_findings` schema
- both byte-identical on every call) is ALSO cached, via `cache_control` on the last tool
entry - this covers "research schema/tool definition" from the correction's cache list.
"Evaluator instructions" are not applicable: `nfl_predict.research.evaluator` is
deterministic Python code, not an LLM call (confirmed in `run_research.py`'s own docstring)
- there is no evaluator prompt sent to Anthropic to cache.

**Never cached:** anything game-specific. `matchup_research_v2.md`'s placeholders (team
names, predictions, market data) only ever appear in the uncached `messages` turn -
`tests/research/test_llm_provider.py::test_static_system_instructions_are_loaded_and_cache_controlled_by_default`
asserts no `{{` placeholder appears in the cached system text.

### 4. Context-growth audit (pause_turn continuations)

Audited `_call()`'s continuation loop: on `stop_reason == "pause_turn"`, it resends
`messages + [{"role": "assistant", "content": payload["content"]}]` - i.e. the original user
turn plus EXACTLY the one partial assistant turn Anthropic just returned, unmodified. No
prior search results are reconstructed, summarized, or duplicated by this code - that is
already the minimal pattern the API's own continuation contract requires.
`tests/research/test_llm_provider.py::test_pause_turn_continuation_accumulates_usage_across_turns_without_resending_extra_content`
asserts the continuation's message list is exactly `[user_turn, one_assistant_turn]`, and
that usage from every turn is still correctly accumulated into the final result. No code
change was needed here - the audit's conclusion is that this was already correct.

### 5. Live-run summary

Every automation report now has a `"research"` section with the requested field names -
`games_researched`, `web_searches`, `input_tokens`, `output_tokens`, `cache_read_tokens`,
`estimated_cost_usd` - alongside the pre-existing, more detailed `research_cost` breakdown
(which now also includes `total_cache_creation_input_tokens` and
`total_web_search_requests`) and a new `research_cost_capped` count.

### 6. A real incident this correction caught and fixed: a test made a genuine billed call

While validating this correction, the full test suite made an **actual, billed Anthropic
API call** (3 real web searches, ~90 seconds, a real classification result) from inside
`tests/live/test_live_market_and_research_wiring.py`. Root cause: several tests call
`run_module.run_slate(...)` without mocking `get_production_odds_provider`/
`get_production_research_provider`, on the assumption (true earlier in this session) that no
live credential is configured in the environment - `research_status`/`odds_status` would
resolve to `*_UNAVAILABLE` and no real call could happen. Once this session's `.env`/OS
environment gained real, working keys (Correction 2's live validation, then a fresh odds key
provided mid-session), that assumption silently became false, and Correction 1's real
per-game wiring meant `run_slate()` would now actually invoke the real provider for any test
that didn't explicitly override it.

**Fix:** a new autouse fixture in `tests/conftest.py` (`_no_real_live_credentials`) removes
`NFL_ODDS_API_KEY`/`NFL_RESEARCH_LLM_API_KEY`/`NFL_WEATHER_API_KEY`/`NFL_NEWS_API_KEY`/
`NFL_INJURY_API_KEY` from the environment for EVERY test by default, regardless of what's in
`.env` or the OS environment - `get_production_*_provider()` now always resolves to its
explicit unavailable status unless a test deliberately re-sets a var itself. The one test
that deliberately sets a fake key to test secret-redaction
(`test_secrets_never_persisted.py`) now also mocks `urllib.request.urlopen` so even that
fake key never reaches the real network. Re-running the full suite afterward confirmed both
the fix (158s, back to the pre-incident baseline) and that no other test was similarly
exposed.

### Real before/after validation (Part C-style single-game call)

Two real, comparable single-game calls against `2026_01_DEN_KC`, both with 3 web searches:

| | Before (Correction 2, no caching) | After (Correction 3, caching + cost controls) |
|---|---|---|
| Input tokens (uncached) | 69,942 | 1,039 |
| Cache creation tokens | N/A (not tracked) | 35,196 |
| Cache read tokens | N/A (not tracked) | 33,866 |
| Output tokens | 7,586 | 6,547 |
| Web searches | 3 | 3 |
| **Estimated cost** | **$0.2457** | **$0.1923** |

A **~21.7% reduction on this single sample**, even though this particular call paid the
FULL 5-minute cache-write price for the static system+tools content (its own internal
`pause_turn` continuation reused that SAME cache as a cheap read on its second turn, which is
what produced the non-zero `cache_read_input_tokens` in a single logical call). The real
benefit compounds further on any SEPARATE call made within the same 5-minute cache window
(e.g., researching a second game shortly after) - not demonstrated here, since the
correction brief asked for exactly one real validation call. Both runs produced a real,
hash-verified, evaluated research artifact; the classification differed between the two
calls (`MIXED`/`MAJOR` vs. `HIGH_UNCERTAINTY` raw → `VETO_CONSIDERATION` after evaluation) -
an honest reflection of real-world news changing between two calls made hours apart
(a genuine QB-injury-uncertainty signal was present in the second call), not a methodology
change.

## Correction 4: a real TypeError and a real accounting bug found in production use

A real single-game live run against `2026_01_DEN_KC` (using the Correction 3 cost-controls
build) got past the earlier INVALID_JSON issue and made 3 real, billed Anthropic web
searches, then crashed with `TypeError: string indices must be integers, not 'str'`. The odds
side of that same run worked correctly (1 real Odds API call, 2 credits, 1 game matched, real
market data populated) - the bug was isolated to the research/parsing path. Because the crash
happened before persistence, no artifact for that literal call exists on disk (confirmed by
an exhaustive search of everything newer than the prior known-good run) - this correction's
regression tests instead reproduce the exact real-world shape (a bare URL string where the
`submit_research_findings` schema expects a full source object) rather than replaying a
recording of the actual failing response.

**Root cause.** `_parse_source`/`_parse_claim`/`_parse_external_prediction` in
`src/nfl_predict/research/parsing.py` did unguarded `d["field"]` dict-subscripting on values
the caller assumed were always dicts. Non-strict tool use (the only mode this codebase can
use - `strict: true` was already rejected in Correction 2 for producing a "compiled grammar
too large" 400 error once nested arrays were involved) is a strong signal, not a guarantee:
the model put a bare URL string into a claim's `sources` array instead of a full
`{source_url, source_title, ...}` object. `d["source_url"]` on a string raises `TypeError`,
which the surrounding `except (KeyError, ValueError)` never catches - so it propagated all
the way past `run_research_for_game()`'s own `except ResearchOutputParseError` handler,
uncaught, into `run_component_safely()` in `live/run.py`, which catches *any* exception
generically and returns `(None, False)`. Because the per-game `LiveResearchOutcome` was never
created, the game never landed in `research_outcomes`, and the automation report showed
`research_games_selected: 0, research_calls_made: 0, research_calls_ok: 0` despite money
already having been spent on 3 real web searches.

**Fix, Layer 1 (parsing) - explicit rejection, never silent coercion.** New
`_require_dict`/`_require_list` guards in `parsing.py` convert any type mismatch, at every
nesting level (a bare string as a whole claim, as a whole source, as an `external_model_opinions`
entry; a non-list `sources` or `material_facts`; a non-string `missing_information` entry),
into the *same* `ResearchOutputParseError` every other malformed-output case already
produces - never a bare `TypeError`, and never a guessed/coerced fake well-formed record.

**Fix, Layer 2 (accounting) - usage/cost survive every downstream failure.**
`run_research_for_game()` in `run_research.py` now:
- includes `"cost": tracker.summary()` in every failure return dict (previously only the
  success path did - so even the already-passing `COST_BUDGET_EXCEEDED`/`LLM_FAILURE`/
  `INVALID_JSON` cases were silently discarding real recorded usage on the failure path);
- wraps evaluation + persistence + ledger-append in a catch-all, so a bug in that stage (not
  just a parsing bug) can never again propagate uncaught past this function - it now always
  returns a result dict. On an unexpected failure here, a new `FailureStatus.
  DOWNSTREAM_PROCESSING_FAILURE` is stored (distinct from `LLM_FAILURE`/`INVALID_JSON`,
  which both happen *before* this point) and the dict still carries real cost data.

`live/research_live.py`'s `run_live_research_for_game()` was also fixed to actually forward
`result.get("cost")` on failure (it previously hard-coded `cost=None` on every non-`"ok"`
status, discarding whatever `run_research_for_game()` returned). `live/run.py`'s automation
report now distinguishes `research_games_selected` (the real selection this run made,
independent of outcome - previously `len(research_outcomes)`, which undercounted whenever an
outcome was lost) from `research_llm_calls_attempted` (increments only once a real Anthropic
call actually returned, so it is `1` even when everything after it failed - "attempted and
failed," never silently zero), `research_llm_calls_succeeded`, `research_llm_calls_failed`,
and `research_artifacts_persisted` (every status writes an artifact before returning; if this
ever falls short of `research_calls_made`, something escaped even Layer 2 and needs
investigating).

**Tests (7 new in `test_parsing.py`, 2 new in `test_run_research.py`, 1 new in
`test_live_market_and_research_wiring.py`, no live calls):** a bare string in a claim's
`sources` list, a bare string claim in `material_facts`, `material_facts` itself as a
non-list, a non-dict entry in `external_model_opinions`, a non-string `missing_information`
entry, a non-list `sources` field, and the legitimate well-formed case (all in
`test_parsing.py`, each asserting `ResearchOutputParseError`, never `TypeError`); a
real-usage-shaped provider whose output has the exact incident shape, asserting the returned
`cost` still shows the real `total_llm_calls`/`total_web_search_requests`/token counts despite
the `INVALID_JSON` failure; a Layer-2 test that monkeypatches `evaluate_research` itself to
raise, asserting `DOWNSTREAM_PROCESSING_FAILURE` and preserved cost; and an
orchestrator-level test proving `run_slate()`'s summary reports `research_games_selected: 1`,
`research_llm_calls_attempted: 1`, `research_llm_calls_failed: 1`, and the real 3
web-searches/69,942 input tokens/$0.245744 cost - never zero.

**Offline replay (no live Odds API or Anthropic call).** Since the literal failing call was
never persisted, the replay uses the last real, successful DEN@KC research run instead
(`run_id=2026-09-12T11-28-44.953506+00-00` - genuine sourced material facts about Mahomes'
knee-injury status, evaluated to `VETO_CONSIDERATION` at materiality `CRITICAL`) plus the
real, persisted Elo/Ridge/LightGBM predictions for the same game
(`data/live/public_predictions/2026_01_DEN_KC/2026-09-12T08-11-46...json`). No real, persisted
market snapshot could be attributed to this specific game (the live-odds snapshot store keys
snapshots by an opaque `provider_event_id` with no team/game_id column, and every persisted
decision-log entry for this game through this session's own real dry-runs shows
`MISSING_LIVE_DATA`, meaning no successful real market match for it was ever itself
persisted) - a synthetic, explicitly-labeled `MarketPoint` stands in for that one input only.
Through the corrected pipeline: `ResearchPoint.available=True`,
`classification=VETO_CONSIDERATION` (confirming the fix - this replay would have crashed
before Layer 1 if the source data had the bug's shape, and would have silently lost this
result before Layer 2 if evaluation/persistence had failed). Run through the completely
unmodified `decide()` (rule_set v1): both `spread` and `moneyline` reach `VETO` with reason
code `VETO_CONSIDERATION_UNRESOLVED` - gates 3-9 are reached (reasons are not merely
`["MISSING_LIVE_DATA"]`), correctly reflecting a CRITICAL-materiality research veto rather
than stopping at the missing-data gate.

## Correction 5: canonical game attribution for persisted odds (a provenance gap found after Correction 4's offline replay)

Correction 4's offline replay (see above) had to fall back to a synthetic `MarketPoint` for
its one non-research input, because the persisted live-odds snapshot store
(`market/live_snapshots/event=<provider_event_id>/snapshots.parquet`) keyed everything by The
Odds API's own opaque `provider_event_id`, with no team names, `game_id`, `season`, or `week`
anywhere in the persisted file - so a reader who only has a canonical `game_id` had no way to
find which of the ~180 anonymous event directories (accumulated across many prior runs) held
that game's real odds. This correction closes that gap before Phase 8B.

**Design.** The raw, provider-native snapshot format is never mutated to retrofit identity
(`OddsMarketSnapshot` still stores exactly what the provider returned). Instead, a new,
separate, additive index - `market/event_game_mapping.py` - records, for every event this
project successfully matches to a real game, the mapping itself: `provider_event_id` →
`canonical_game_id` (+ `season`/`week`), the provider's own raw team-name strings alongside
the normalized `team_id`s they resolved to (so home/away normalization is itself auditable,
not just trusted), `kickoff_timestamp`, `mapping_timestamp`, and a versioned `mapping_method`.
It mirrors `research.prospective_ledger`'s append-only-JSONL-plus-sha256 pattern: re-recording
the identical mapping on a later Odds API refresh is a safe no-op (the original
`mapping_timestamp` is kept), but a genuinely DIFFERENT mapping at the same
`provider_event_id` is refused, never silently overwritten. `find_provider_event_ids_for_game(season, week, game_id)` is the reconstruction entry point - given only a canonical `game_id`, it returns which `market/live_snapshots/event=.../` directories hold that game's real, persisted odds.

`live/odds_mapping.py`'s `match_odds_event_to_game()` (renamed from `map_odds_event_to_game_id`) now returns an explicit `EventGameMatch` with a `status` of `"matched"`, `"unmatched"`, or `"ambiguous"` - an ambiguous result (more than one candidate game for a `(season, home, away)` combination) is now a distinct, explicit outcome, never silently resolved by picking whichever row SQLite happens to return first.

**Consensus and decision provenance.** `live/market_consensus.py`'s `MarketConsensus` gained
`consensus_algorithm_version`, `excluded_book_keys`/`exclusion_reason` (always empty/None
today - no exclusion rule exists yet, so this is honest, not a placeholder), and
`underlying_snapshot_keys` (the exact `(bookmaker, fetched_at)` pairs contributing to the
consensus) plus a deterministic `snapshot_reference` hash. `decision.schemas.MarketPoint`
carries all of this through (`provider_event_id`, `market_snapshot_reference`,
`consensus_book_keys`, `underlying_snapshot_keys`, etc.) so a `MarketPoint` can always point
back at the exact persisted rows it was computed from. `DecisionRecord` gained
`market_provider_event_id`/`market_snapshot_reference`/`market_snapshot_timestamp`/
`research_id`, and `DecisionInputPacket.content_hash()` (mirroring
`ResearchInputPacket.content_hash()`) replaces the `input_packet_hash="n/a"` placeholder
`live/run.py` had written for every real live decision since Phase 7.

**Tests (18 new, no live network calls):** `tests/live/test_odds_mapping.py` (matched/
unmatched/ambiguous/home-away-not-swapped, against an isolated in-memory sqlite connection -
never the real project database, so a deliberately-ambiguous test scenario never touches real
data); `tests/market/test_event_game_mapping.py` (append/read round-trip, idempotent
re-recording, conflict detection, tamper detection, empty-read); `tests/live/
test_offline_reconstruction.py` (a canonical `game_id` alone finds exactly one
`provider_event_id`; `provider_event_id` is preserved end-to-end on `MarketPoint`; home/away
normalization is recorded correctly; ambiguous/unmatched events never get a mapping row;
mapping remains stable and non-duplicated after a later Odds API refresh; a canonical
`game_id` alone can reconstruct the exact real `MarketPoint` a past run computed, verified
against the original in-memory result then explicitly discarded; and the centerpiece test -
spread AND moneyline decisions replayed entirely from persisted odds + persisted research +
real-shaped model predictions, reaching `QUALIFIED_BET` on both, with zero network calls
anywhere in the test); one new test in `test_live_market_and_research_wiring.py` proving a
real (non-dry-run) `DecisionRecord` carries a real 64-character sha256 `input_packet_hash`
(never `"n/a"`) and resolves back to its real persisted market snapshot via
`find_provider_event_ids_for_game()`. 561 tests passing total.

**Real DEN@KC replay attempt - honest result.** The OLD odds snapshots already on disk from
before this correction (~180 anonymous `event=<hash>/` directories, accumulated across prior
runs including Correction 3's real validation call) cannot be retroactively attributed: the
raw files carry only `provider_event_id` and market numbers, never the provider's team-name
strings needed to re-run the matching logic after the fact, and fabricating that attribution
was explicitly out of scope. Per the correction's own instruction, exactly ONE new, controlled
live odds refresh was attempted for `2026_01_DEN_KC` (`python -m nfl_predict.live.run --game
2026_01_DEN_KC --odds-only`, `NFL_RESEARCH_LLM_API_KEY` unset so an Anthropic call was
structurally impossible regardless of mode flags) - it genuinely failed with the same
account-level `OUT_OF_USAGE_CREDITS` condition Correction 2 already documented (`OddsQuotaExceededError`, HTTP 401, no retry, confirmed no credits were consumed by the rejected request). This is a real external account constraint, not a code gap - the fix itself is validated by the 18 new tests above (`test_offline_reconstruction.py` in particular exercises the exact same reconstruction path end-to-end with realistically-shaped data), but a live DEN@KC replay with genuinely real, freshly-fetched market data was not possible in this environment at this time. No live market data was fabricated to work around this - the Task 8 exit criteria below is reported honestly as not-yet-met for that reason alone.

**Phase 8A exit criteria - status.** One game CAN now be reconstructed offline using REAL
persisted models + REAL persisted research + unchanged decision rules with NO synthetic
research/model inputs and NO network calls (`test_offline_reconstruction.py`'s replay test);
the one remaining synthetic input is the market leg specifically, and only because of the
external zero-credits condition above, not because the mapping/provenance mechanism doesn't
work. Once the Odds API account has usage credits again, a single `--odds-only` run for any
real upcoming game is expected to close this fully - no further code change is anticipated.

## Architecture

```
src/nfl_predict/live/
  schedule_provider.py       current season/week + upcoming/completed/postponed games
  live_data.py                build_live_game_frame() — LEFT JOIN features+targets so every
                               upcoming game gets a feature row (never dropped like the
                               Phase 3/4 INNER-JOIN training frame)
  live_models.py               predict_live_elo() / predict_live_ridge_lightgbm_logistic() —
                               reuse Phase 4's frozen _fit_predict_* functions verbatim
  odds_mapping.py              maps an odds-provider event to a game_id via the teams table
  market_consensus.py          no-vig probability averaging (moneyline) + median line (spread)
  odds_automation.py           get_production_odds_provider() — real key or explicit
                               ODDS_PROVIDER_UNAVAILABLE, never a silent fixture fallback
  odds_ingestion.py             fetch_and_snapshot_live_odds() — real provider -> persisted
                               snapshots -> real per-game MarketPoints (correction pass)
  research_automation.py       same pattern for the Phase 6 LLM research provider
  research_live.py              run_live_research_for_game() — actually invokes
                               run_research(), builds ResearchPoint only from the persisted,
                               evaluated result; trigger-priority game selection under a real
                               cost cap (correction pass)
  injury_automation.py         same pattern for a genuinely STRUCTURED injury-data feed —
                               a different boundary than the LLM research provider (see below)
  weather_provider.py           OpenMeteoWeatherProvider — live-tested, credential-free
  component_status.py          RunHealthReport / run_component_safely() — one component's
                               failure never halts the run
  snapshot_lifecycle.py        write/read/select the latest valid pregame snapshot per game
  prediction_publication.py    public ALL_MODEL_PREDICTIONS storage, DRAFT/READY/PUBLISHED/
                               SUPERSEDED states, hash-verified, superseding never mutates
  results_ingestion.py         post-game grading, joined separately from pregame predictions
  performance.py                current-season ALL_MODEL_PREDICTIONS accuracy/Brier/MAE
  run.py                        the orchestrator: python -m nfl_predict.live.run

src/nfl_predict/market/odds_provider.py   (Phase 5, extended) — TheOddsAPIProvider now makes
                                           real HTTP calls; pure, tested parse functions for
                                           the actual Odds API response shape
src/nfl_predict/research/current_data_providers.py  (Phase 6, extended) — LiveInjuryProvider,
                                           wind_gust_mph on WeatherForecast
config/venues.yaml                        32 teams -> {stadium, lat, lon, roof}, verified
                                           against the real teams table
```

## Commands

```bash
python -m nfl_predict.live.run                      # dry-run by default (see below)
python -m nfl_predict.live.run --no-dry-run          # writes real state (predictions, decision log)
python -m nfl_predict.live.run --week 2              # a specific week
python -m nfl_predict.live.run --game 2026_01_DEN_KC # a specific game
python -m nfl_predict.live.run --research-only       # (also --odds-only / --models-only / --decisions-only)
python -m nfl_predict.live.run --research-all        # research every target game, not just the top --max-research-calls
python -m nfl_predict.live.run --max-research-calls 5  # cap on real (billed) research calls per run, default 3
python -m nfl_predict.live.run --show-decisions      # add a per-game/market_type decision+reason-code+value breakdown to the JSON output
```

`--dry-run` is the default, but is **not** "no external calls": it still fetches real odds
and may still invoke real, billed Anthropic research calls whenever those credentials are
configured, because those artifacts are real observations worth keeping regardless of
whether this particular run's decisions get published. What `--dry-run` guarantees is no
pregame-run record, no decision-log entry, and no publication —
verified by `tests/live/test_run_orchestrator.py::test_dry_run_never_calls_append_decision_record`
and by the orchestrator's own `if not dry_run:` gates around both prediction publication and
decision-log persistence. A normal run **never** auto-publishes an official Best Bet —
`nfl_predict.decision.pick_ledger.publish_pick` is never called from `run.py`
(`tests/live/test_run_orchestrator.py::test_orchestrator_never_calls_publish_pick_directly`
asserts this structurally). `published_official_picks` in every automation report is
therefore always `0` in this phase; that is intentional, not a bug — publishing a real Best
Bet remains a deliberate, separate action for a future phase.

## Live feature generation

`build_live_game_frame()` reuses Phase 2's exact feature-engineering output but LEFT JOINs it
against targets instead of Phase 3/4's INNER JOIN, so an upcoming (unplayed) game keeps its
feature row instead of being silently dropped. A real, non-obvious bug was caught here: an
unplayed game's `is_tie` column is `null`, and Phase 4's `_fit_predict_logistic`/
`_fit_predict_lightgbm` filter with `~pl.col("is_tie")` — polars treats a null boolean test
as "exclude," so every upcoming game silently vanished from the win-probability output with
no error raised. Fixed with `.fill_null(False)` on `is_tie` specifically for the live frame
(a true statement — an unplayed game is not YET a known tie — not a fabricated guess).

## Model compatibility

Live Elo, Ridge-margin, LightGBM, and the logistic win model all reuse Phase 4's frozen
`_fit_predict_ridge`/`_fit_predict_lightgbm`/`_fit_predict_logistic` functions and Phase 6's
`_live_elo_model()` (k=40, home_adv=45, run sequentially over every completed game)
unchanged — same feature lists, same hyperparameters, same random seeds. The only thing that
changed is the frame shape (LEFT JOIN) and the train/predict masks (train = every completed
game; predict = the upcoming game(s)). No new model was built, and no hyperparameter was
retuned. This is exactly the "periodic retraining on completed games only, per the frozen
protocol" Phase 8A's brief called for — not an independent new model, and never trained on
sportsbook information.

## Odds integration

`TheOddsAPIProvider` (`src/nfl_predict/market/odds_provider.py`) makes real HTTP calls via
`urllib` (`_get()`), with pure, independently-tested parsing functions
(`_parse_events_response`, `_parse_odds_response`) built against the real Odds API's
documented response shape. `NFL_ODDS_API_KEY` is **not set** in this environment (confirmed:
no `.env` file exists and the variable is unset), so `get_production_odds_provider()`
correctly returns `ODDS_PROVIDER_UNAVAILABLE` — verified live against the real Odds API
server during earlier development (a fake key produced a genuine HTTP 401, proving the code
path is real, not a stub). `odds_ingestion.fetch_and_snapshot_live_odds()` now turns a real
`OddsProvider` into real per-game `MarketPoint`s: every fetched sportsbook snapshot is
persisted through the append-only store (`market/live_snapshot_store.py`) before use, and the
consensus (`market_consensus.py` — no-vig probability average for moneylines, median
individual line for spreads, never a blind average of raw American odds) is computed from
those persisted snapshots. `run.py` only marks a game's market available when
`odds_ingestion` actually matched and produced usable data for it — never merely because
`odds_status == AVAILABLE`. This whole path is tested against realistic mocked provider
responses (`tests/live/test_odds_ingestion.py`,
`tests/live/test_live_market_and_research_wiring.py`); it has not been exercised against a
real odds payload because no credential is configured in this environment.

## Injury integration

Phase 6 already had a provider-neutral `InjuryProvider` ABC. Phase 8A adds the missing
production-gating boundary: `LiveInjuryProvider` (gated on `NFL_INJURY_API_KEY`, raises
`NotImplementedError` past the gate — no vendor chosen yet) and
`nfl_predict.live.injury_automation.get_production_injury_provider()`, mirroring the
odds/research automation pattern exactly (`INJURY_PROVIDER_UNAVAILABLE`, never a silent
`FixtureInjuryProvider` fallback). This is deliberately a **separate boundary** from the LLM
research provider: `ResearchFindings.qb_status`/`skill_position_status`/etc. are an LLM's
summary of unstructured web/news claims (tiered, fact/opinion-separated); `LiveInjuryProvider`
would be a direct structured feed with no LLM interpretation step. `run.py` checks and
reports `injury_status` as its own component, independent of `research_status`
(`tests/live/test_injury_automation.py::test_injury_status_is_reported_separately_from_research_status`
proves the two modules have no import coupling). No vendor is configured or chosen yet, so
this status is always `INJURY_PROVIDER_UNAVAILABLE` in this environment, and no per-game
structured injury finding is wired into a `DecisionInputPacket` field — Phase 7's schema has
no such field, only the existing `ResearchPoint`/`MAJOR_INJURY_RISK` reason code, populated
today from the LLM research path.

## Weather integration

`OpenMeteoWeatherProvider` requires no API key and was live-tested for real during
development: Arrowhead Stadium (outdoor) returned real forecast data (90.5°F, 17.4 mph wind,
26.2 mph gust); Allegiant Stadium (dome) correctly returned `None` for every weather field
rather than a fabricated number. `config/venues.yaml` (32 teams -> stadium/lat/lon/roof) was
cross-checked against the real `teams` table after an initial draft had two wrong team_ids
(HOU, LA) from pattern-guessing instead of querying.

## Automated research status

`get_production_research_provider()` (reusing Phase 6's `AnthropicMessagesProvider`) returns
`RESEARCH_PROVIDER_UNAVAILABLE` because `NFL_RESEARCH_LLM_API_KEY` is unset in this
environment (confirmed: no `.env` file exists) — never a silent fallback to
`FixtureLLMProvider`/`ManualResearchProvider` (`assert_provider_allowed_in_production` is a
defense-in-depth check against exactly that). The orchestrator now DOES call the research
provider per-game for real, selected games (`research_live.run_live_research_for_game()`,
which invokes Phase 6's unmodified `run_research_for_game()` pipeline end to end - real
provider call, parse, evaluate, persist, ledger entry) — game selection follows Phase 6's
existing trigger-priority ranking under a real per-run cost cap (`--max-research-calls`,
default 3; `--research-all` bypasses the cap for validation runs). A plain Messages API call
cannot browse the web, so `AnthropicMessagesProvider` now includes Anthropic's server-side
web-search tool (`web_search_20250305`) by default — untested against the live API in this
environment (no credential), but exercised end-to-end against a fixture provider via
`tests/live/test_research_live.py` and `tests/live/test_live_market_and_research_wiring.py`.

## Decision integration

Every game's `DecisionInputPacket` is built from real live Elo/Ridge/LightGBM outputs, a
real per-game `MarketPoint` (from `odds_ingestion`, when a match exists), and a real
per-game `ResearchPoint` (from an actually-persisted, evaluated research run, when the game
was selected and research succeeded) — then passed unchanged into Phase 7's `decide()`. No
rule, threshold, or schema in `nfl_predict/decision/` was modified (`config/decision_rules.yaml`
and `engine.py` are byte-for-byte what Phase 7 shipped; only `input_packet.py` was touched,
and only to extract an existing block of logic into a reusable function — verified
behavior-preserving by re-running `scripts/run_phase7_pilot.py` against the real Phase 6
pilot data and confirming identical output before and after). Missing market and research
data still correctly and honestly produce `NO_BET`/`MISSING_LIVE_DATA` when they're
genuinely missing — and a fully-populated, realistic packet now genuinely reaches
`QUALIFIED_BET` for both market types
(`tests/live/test_live_market_and_research_wiring.py::test_full_pipeline_with_synthetic_valid_odds_and_research_reaches_gates_3_to_9`),
proving the gate is real, not permanently closed by missing wiring.

## Publication states and current-season performance

`prediction_publication.py` implements `DRAFT | READY | PUBLISHED | SUPERSEDED`.
Superseding a prediction writes a side marker file next to the original record — the
original JSON's bytes (and its SHA-256) are never touched, so
`supersede_previous_predictions()` can mark every older prediction for a game as superseded
by a new one without altering history. `results_ingestion.py` grades completed games
separately (`compute_graded_result()`/`write_graded_result()`), joined against pregame
predictions only at read time (`performance.py`) — a pregame prediction record itself is
never mutated once written. `compute_all_model_predictions_performance()` computes winner
accuracy / Brier score / margin MAE for `ALL_MODEL_PREDICTIONS` only, excluding ungraded and
tied games from every denominator; it does not touch `LEANS`/`BEST_BETS` records (those
reuse Phase 7's own `nfl_predict.decision.records` functions, unchanged) and it does **not**
backfill a fake current-season Best Bets record from Phase 5's historical analysis — the
official Best Bets record starts empty and stays empty until a real pick is prospectively
published.

## A real bug found and fixed during this phase

Windows rejects colons in filenames. `run.py` passes a raw `_now_iso()` timestamp (e.g.
`2026-09-12T05:57:21.349507+00:00`) as `prediction_id`, and
`prediction_publication._paths()` originally built the on-disk filename directly as
`f"{prediction_id}.json"` — every one of the first 14-game pilot's publish attempts failed
with `OSError: [Errno 22] Invalid argument`. Fixed with a `_filesystem_safe()` helper inside
`prediction_publication.py` (replaces `:` with `-` for the filename only); the stored
record's own `prediction_id` field still carries the true, unmodified ISO timestamp — only
the filesystem representation is sanitized. `supersede_previous_predictions()` was updated to
sanitize its `new_prediction_id` argument the same way before comparing it against
`path.stem`. A regression test
(`tests/live/test_prediction_publication.py::test_a_raw_iso_8601_timestamp_prediction_id_is_filesystem_safe_on_windows`)
uses a real colon-containing ISO timestamp to prevent this from recurring silently.

## Failure isolation and idempotency

`RunHealthReport`/`run_component_safely()` wrap every pipeline component (schedule, each
model family, odds status, research status, injury status, each per-game prediction
publish); one component's exception is caught, recorded `FAILED` with the exception detail,
and never halts the rest of the run
(`tests/live/test_run_orchestrator.py::test_a_failed_model_component_does_not_prevent_the_run_from_completing`).
Two identical dry runs leave behind no accumulated state
(`test_running_the_same_dry_run_twice_produces_no_duplicate_persisted_state`); odds
snapshots and graded results have idempotent-append helpers that no-op on exact-duplicate
content but still raise on a genuine conflicting update at the same key.

## Pilot results (real, 2026 Week 1 — not fixture data)

Ran `python -m nfl_predict.live.run --no-dry-run --show-decisions` against the actual
current NFL schedule, after the correction above:

| Field | Value |
|---|---|
| Season / week | 2026 / 1 |
| Games detected | 14 |
| Feature rows valid | 14/14 |
| Elo predictions | 14/14 |
| Ridge predictions | 14/14 |
| LightGBM predictions | 14/14 |
| Odds status | `ODDS_PROVIDER_UNAVAILABLE` (confirmed: no `.env` file exists, `NFL_ODDS_API_KEY` unset) |
| Research status | `RESEARCH_PROVIDER_UNAVAILABLE` (confirmed: `NFL_RESEARCH_LLM_API_KEY` unset) |
| Injury status | `INJURY_PROVIDER_UNAVAILABLE` (no `NFL_INJURY_API_KEY`) |
| Odds events seen / matched | 0 / 0 (provider never constructed - no credential) |
| Research calls made | 0 (provider never constructed - no credential) |
| Decisions | 28 (14 games x spread + moneyline), **all `NO_BET` / `MISSING_LIVE_DATA`** |
| `any_decision_reached_gates_3_to_9` | `false` (correctly - no real market/research exists to reach them with) |
| Published model predictions | 14/14 (`data/live/public_predictions/<game_id>/`) |
| Published official picks | 0 (never auto-published — see Commands above) |

**Important correction to the original Phase 8A report's premise:** the original pilot's
"28/28 NO_BET" was previously (incorrectly) attributed partly to real infrastructure
readiness. Both the *initial* pilot (before this correction) and *this* pilot (after it)
produce the identical `NO_BET`/`MISSING_LIVE_DATA` result — but for a verified, honest
reason now: `NFL_ODDS_API_KEY` and `NFL_RESEARCH_LLM_API_KEY` are genuinely not configured in
this sandbox (no `.env` file exists at all). The correction's actual proof point is not this
pilot run (which cannot exercise the new code paths without credentials) but the dedicated
integration test described above, which supplies realistic-but-synthetic market/research
data through the exact same code paths `run.py` uses and confirms `QUALIFIED_BET` is reached
for both spread and moneyline. Every `publish_prediction:<game_id>` health entry reported
`SUCCESS`; all 28 decisions and 14 predictions were written to disk and are
hash-verified/re-readable.

## Missing credentials

| Provider | Env var | Status |
|---|---|---|
| Odds (The Odds API) | `NFL_ODDS_API_KEY` | Configured (`.env`), but the account has **zero usage credits remaining** — confirmed via a real `HTTP 401 OUT_OF_USAGE_CREDITS` response in Correction 2's live validation. A fresh key set later in the same debugging session hit the identical error, confirming credits are tied to the account, not the specific key. Needs the account's quota to reset or be upgraded before a real successful fetch can be observed. |
| LLM research (Anthropic Messages API) | `NFL_RESEARCH_LLM_API_KEY` | Configured and **live-validated successfully** — see Correction 2. |
| Structured injury data | `NFL_INJURY_API_KEY` | Not configured (no vendor chosen yet) |
| News research | `NFL_NEWS_API_KEY` | Not configured (no vendor chosen yet) |
| Weather (Open-Meteo) | none required | Live and working |

Odds and research code paths are both fully implemented and tested; research is now also
live-validated end-to-end against the real API. Odds remains blocked, but by a real,
external account-level fact (zero usage credits) rather than any remaining code or
credential gap.

## Estimated recurring external costs (once credentials are added)

- **The Odds API**: paid tiers by request volume; a free tier exists (500 credits - the
  allowance this project's account has now fully used). With Correction 2's fix, a full
  NFL-slate refresh costs 2 credits (`spreads,h2h` x `us` region) regardless of how many
  games are in the slate, vs. the pre-fix ~3 credits x every remaining event in the season
  (~480-500 credits observed in one run). Exact paid-tier pricing should be checked at
  odds-api.com before committing.
- **Anthropic Messages API** (research agent): pay-per-token, plus Anthropic's per-web-search
  fee for the hosted web-search tool (`max_web_search_uses` defaults to 3 per matchup,
  `NFL_RESEARCH_MAX_WEB_SEARCHES` configurable - check Anthropic's current pricing page for
  the per-search rate). Cost scales with prompt size,
  search count, and run frequency (up to `--max-research-calls` real calls per run, default
  3, or every target game with `--research-all`). One real, successful single-game
  validation call (Correction 2) used 3 web searches, 69,942 input tokens, and 7,586 output
  tokens - a genuine, observed data point, though this correction did not compute a dollar
  figure from it (see Known limitations).
- **Open-Meteo**: free, no API key, no known usage cap for this project's request volume.
- **Structured injury / news providers**: no vendor has been chosen, so no cost estimate
  exists yet — this is an open decision, not an oversight.

## Known limitations

- `QUALIFIED_BET` remains unreached on the real current slate purely because the real Odds
  API account has zero usage credits remaining (Correction 2) — the gate itself is proven
  passable end-to-end both by a realistic synthetic integration test (Correction 1) and now
  by a real, successful Anthropic research call (Correction 2); nothing in the decision code
  blocks it once real market data specifically exists.
- The web-search tool (`web_search_20250305`) and the `submit_research_findings` custom tool
  have now been exercised against the real Anthropic API and confirmed working (Correction
  2) - but only in NON-strict mode; Anthropic's `strict: true` tool-use mode was tried and
  rejected (schema too large for its grammar compiler) and was not pursued further. A future
  attempt to re-enable strict mode would need a meaningfully smaller/flatter schema and
  should be validated against a real call before being treated as reliable.
- No dollar-cost estimate is computed for Anthropic calls - `LLMCallResult.estimated_cost_usd`
  is `None` for `AnthropicMessagesProvider` (the Messages API doesn't return one directly,
  and this correction did not add published per-token pricing math). Token counts
  (input/output) and web-search-call counts ARE captured and reported.
- The structured injury-data boundary (`injury_automation.py`) exists and is tested, but has
  no chosen vendor and is not yet consulted per-game inside `run.py`'s decision loop —
  Phase 7's `DecisionInputPacket` has no field for it yet, and rule set v1 has no
  injury-specific `required_input` either, so this is a deliberate scope boundary, not an
  oversight (see the correction section above).
- Research cost accounting is token-level only (`CostTracker`/`ResearchCostConfig`,
  aggregated per run into `research_cost` in the automation report) - a per-web-search-call
  dollar breakdown is logged (`logger.info` on each Anthropic call) but not persisted into a
  structured field, since `LLMCallResult` has no slot for it yet.
- `postponed_or_cancelled_games()` in `schedule_provider.py` always returns empty — the
  underlying schedule data source has no observed postponement/cancellation case to model
  against yet (documented, not silently assumed impossible).
- No admin dashboard exists yet to visualize `RunHealthReport` — the data it needs
  (per-component status, checked-at timestamp, detail string) is produced and tested, but
  nothing consumes it yet beyond the JSON automation report.
- Multi-run lifecycle (Thursday/Friday/Sunday/T-minus-90-minute runs, `LATEST_VALID_PREGAME_SNAPSHOT`
  selection) has its storage/selection logic built and tested
  (`snapshot_lifecycle.py`), but has not been exercised across multiple real runs of the same
  game in this environment — only single-run-per-game was pilot-tested.

## Tests

561 tests passing total (158 new in Phase 8A total: 89 from Correction 1, 23 from
Correction 2's emergency fix, 18 from Correction 3's cost controls, 10 from Correction 4's
TypeError/accounting fix — 7 in `test_parsing.py` (the `_require_dict`/`_require_list`
type-guard regressions), 2 in `test_run_research.py` (real-usage-shaped malformed output, and
the evaluation-stage catch-all), and 1 orchestrator-level test in
`test_live_market_and_research_wiring.py` — and 18 more from Correction 5's canonical
odds-attribution fix (`test_odds_mapping.py`, `test_event_game_mapping.py`,
`test_offline_reconstruction.py`, and one decision-provenance test — see Correction 5 above
for exactly what each proves) —
`tests/research/test_cost_tracking.py` (new file: cost estimation against the real
pricing-config numbers, the pre-flight budget guard, unpriced-model handling),
`tests/research/test_llm_provider.py` additions (cache_control on the system block and the
`tools` array, real cache/web-search usage capture, pricing-table-not-hard-coded proof,
pause_turn context-growth audit), and the `tests/conftest.py` autouse fixture that fixed a
real test-suite live-call incident - see Correction 3 below), covering: schedule retrieval,
live feature-frame LEFT-JOIN correctness (including the `is_tie` null-masking fix), live
Elo/Ridge/LightGBM/logistic reuse of frozen fitting functions, real-shape odds-response
parsing (mocked HTTP, no live network call in tests), market-consensus math (no-vig
averaging, median spreads), weather-provider correctness (including the dome/None-fields
case), component-failure isolation, production-provider gating for odds/research/injury
(explicit unavailable status, fixture/manual providers rejected in production), odds-snapshot
and graded-result idempotency, snapshot lifecycle (latest-valid-pregame-snapshot selection),
prediction publication lifecycle (publish/read/supersede, including the Windows-filename
regression test), results ingestion (including a real check against an actually-completed
2026 game), current-season performance exclusions, secrets never appearing in any persisted
artifact, real per-game odds ingestion (provider-available-alone insufficient, real valid
odds populate `MarketPoint`, unmapped/missing-market events skipped not fabricated,
snapshots persisted before use, a 150-event slate still uses exactly one batched fetch),
real research invocation (`run_research()` actually called and proven via a call-counting
wrapper, `ResearchPoint` only from a persisted+evaluated artifact, a failed call leaves it
unavailable, trigger-priority selection respects the cost cap and `--research-all`),
structured tool-call extraction (never concatenating free text into JSON, `pause_turn`
continuation, a diagnostic error when the tool is never called), real cost/cache/web-search
usage recording against the configured pricing table (never hard-coded), the pre-flight
cost-budget guard, structured-injury-absence never causing `MISSING_LIVE_DATA`, real
staleness computed from real (not manufactured) timestamps, exact numeric spread/moneyline
disagreement examples, and — the key proof — a full synthetic-but-realistic pipeline
reaching `QUALIFIED_BET` for both market types through the completely unmodified decision
engine. **No test makes a real network call** — every odds/Anthropic test monkeypatches
`_get`/`_post`/`urllib.request.urlopen` directly, and an autouse `conftest.py` fixture
removes every live-provider credential from the environment for every test by default
(Correction 3 fixed a real incident where this was NOT yet true).

## Completion criteria

- [x] Schedule loads (real 2026 season/week detected).
- [x] Upcoming games get Phase-2-compatible features (LEFT-JOIN frame, 14/14 valid).
- [x] Live Elo/Ridge/LightGBM/logistic all work (14/14 each, real Week 1 2026 games).
- [x] Odds provider integration works OR is fully implemented and blocked only by missing
      credential (`TheOddsAPIProvider` makes real HTTP calls, wired into real per-game
      `MarketPoint`s via `odds_ingestion.py`, now at 2 credits per full-slate refresh under a
      configurable budget guard). Live-validated against the real account (Correction 2,
      Part C Step 2): the account has zero usage credits remaining, so a real successful
      fetch has not been observed — but the guard/no-retry/redaction behavior was proven
      correct against the real `OUT_OF_USAGE_CREDITS` response.
- [x] Timestamped odds storage works (idempotent append, tested; every fetched snapshot
      persisted before use).
- [x] Research provider automation works AND is live-validated (Correction 2, Part C Step
      3): a real single-game call against `2026_01_DEN_KC` succeeded end-to-end —
      3 web searches, structured tool-call extraction (no free-text JSON parsing),
      `research_classification=MIXED`, `materiality=MAJOR`, hash-integrity verified,
      `ResearchPoint.available=True` confirmed.
- [x] Decision engine receives live packets built from real market/research data when
      present. A dedicated integration test proves the same code path reaches
      `QUALIFIED_BET` for both spread and moneyline once real data exists — `decide()` and
      `config/decision_rules.yaml` are completely unchanged throughout both corrections.
- [x] Real prospective predictions can be frozen (published, hash-verified, the
      colon-filename bug fixed and regression-tested; also live-validated in Correction 2's
      Part C Step 4 single-game full-pipeline run).
- [x] Results ingestion works (tested, including against a real completed 2026 game).
- [x] Current-season tracking works (`performance.py`, tested, excludes ungraded games).
- [x] CLI displays decision reason-code counts (`reason_code_counts` in every automation
      report) and an optional per-game breakdown (`--show-decisions`).
- [x] Structured injury API absence does not block qualification under rule set v1 (tested
      directly).
- [x] Tests pass (561/561), zero live network calls made during the test suite (a real
      incident where this was briefly untrue - a test made a genuine billed Anthropic call -
      was caught and fixed via an autouse `conftest.py` fixture; see Correction 3). A real
      TypeError and a real cost/usage-accounting bug found via production use (Correction 4)
      are also both regression-tested and fixed - see that section for the offline replay
      proving the corrected pipeline against real stored research and model data.
- [x] Persisted odds can be attributed back to a canonical game_id, not just an opaque
      provider event id (Correction 5) - proven offline end-to-end
      (`test_offline_reconstruction.py`); a live DEN@KC replay with freshly-fetched real
      market data specifically could not be completed because the Odds API account has zero
      usage credits remaining (a real external account fact, confirmed via one controlled,
      authorized refresh attempt that itself failed with the same real `OUT_OF_USAGE_CREDITS`
      condition - not a code gap, and no data was fabricated to route around it).

**STOPPED AFTER THE PHASE 8A CORRECTION.** No Next.js/website code was written. No Phase 7
decision rule, threshold, or schema was weakened, altered, or fit to any observed outcome —
`config/decision_rules.yaml` and `engine.py` are exactly what Phase 7 shipped throughout both
corrections. No historical model result was changed. No independent model was retrained
using sportsbook information — live Ridge/LightGBM/logistic retraining used only completed
games' own outcomes, per the frozen protocol. Per the correction brief's explicit
instruction, no `--research-all` or full 14-game live slate run was attempted after the Part
C single-game validations succeeded — that remains for a deliberate future run, once the
Odds API account's credits are restored.
