# Data Sources

## Purpose

Registry of every external data source this project consumes or plans to consume. Human-
readable companion to `config/sources.yaml` — same rule as the feature dictionary: they must
agree, and code reads the YAML, people read this.

nflverse is integrated as of Phase 1 (see docs/PHASE1_DATA_REPORT.md for what was actually
ingested and validated). Other sources listed below (odds, injuries, weather) remain
**candidates** — identified for later phases (market-aware modeling in Phase 3+, live
injury/weather in Phase 5) but not yet integrated; Phase 1 deliberately did not ingest
current injury data or live sportsbook odds.

## Required fields per source

| Field                  | Meaning |
|-------------------------|---------|
| `name`                  | Stable identifier. |
| `description`           | What it provides. |
| `access_method`         | How it's fetched (package, REST API, bulk file download, scrape) and where the code that fetches it lives once implemented. |
| `auth_required`         | Whether it needs an API key/credential, and the env var name (never the value) that holds it. |
| `update_frequency`      | How often it changes / should be refetched (e.g. weekly during season, once per game, static historical dump). |
| `historical_coverage`   | Earliest season/date it covers, and any known gaps. |
| `trust_level`           | How authoritative it is for its domain (e.g. official/league-derived vs. third-party aggregation vs. scraped/unofficial). |
| `used_for`              | Which pipeline stage(s) consume it (ingestion table, feature category, research agent, etc.). |
| `known_issues`          | Anything about the source that has bitten (or could bite) data quality — inconsistent team abbreviations across seasons, timezone ambiguity, retroactive corrections, etc. |
| `raw_storage_convention` | Where immutable raw snapshots fetched from this source are written, e.g. `data/raw/<source_name>/<dataset_name>/<retrieved_at_utc>/`. Every file under that path is accompanied by a provenance manifest — see "Provenance requirements" below. |

## Provenance requirements

Every fetch from a source registered here must produce a provenance manifest — the full
required field list and the rules around immutability/versioning are defined once, in
`docs/ARCHITECTURE.md#raw-data-provenance`, not duplicated here. In short: source name,
dataset name, requested range, source identifier/release version, retrieval timestamp, the
local immutable file path, a content hash, row count, and a schema fingerprint. Implemented
for nflverse in Phase 1 (`src/nfl_predict/data/provenance.py`); any future source added here
follows the same contract.

## Sources

### nflverse (primary historical data source)

- **description:** Community-maintained, research-grade NFL data — play-by-play, schedules,
  rosters, snap counts, injuries, some historical betting lines — distributed as versioned
  CSV/Parquet releases and via the `nflreadr` (R) / `nflreadpy` (Python) packages.
- **access_method:** `nflreadpy` (Python; **not** `nfl_data_py`, which is archived/deprecated
  upstream and must not be used) planned for `src/nfl_predict/data/ingestion`, or direct
  download of the versioned release files if tighter control over caching is needed.
  `nflreadpy` returns Polars DataFrames — see
  `docs/ARCHITECTURE.md#tabular-data-representation-boundary` for where that stops being
  visible to the rest of the pipeline. Every fetch through this loader must produce a
  provenance manifest per `docs/ARCHITECTURE.md#raw-data-provenance` (see also "Provenance
  requirements" below).
- **auth_required:** No.
- **update_frequency:** Updated in-season, typically within a day or two of games.
- **historical_coverage:** Play-by-play back to 1999 in most nflverse releases; other tables
  (rosters, snap counts, injuries) have shorter/patchier coverage that must be confirmed
  empirically during Phase 1 ingestion, not assumed.
- **trust_level:** `third_party_aggregated` — nflverse is a community-maintained aggregator,
  not an official league data provider, and must not be labeled `official` in
  `config/sources.yaml`. Much of the on-field data it redistributes (play-by-play,
  schedules, scores) is *derived from* authoritative/official sources, which is why it's
  trustworthy for on-field facts; but nflverse's own role is aggregation and cleaning, and
  some derived tables (depth charts, some injury data) are further aggregated from
  third-party reporting with no official backing at all — see the depth_charts 2025 finding
  in `docs/PHASE1_DATA_REPORT.md` for a concrete example of this distinction mattering.
- **used_for:** Games/schedules table, play-by-play, feature engineering inputs, baseline
  historical market lines where included.
- **known_issues:** Team abbreviations have changed over time (relocations/rebrands — e.g.
  franchise moves) and must be normalized to a stable team ID before joining across seasons.
  `nflreadpy` is pre-1.0 (0.1.x as of this writing) — pinned conservatively
  (`nflreadpy>=0.1.5,<0.2.0` in `pyproject.toml`) since a pre-1.0 package can make breaking
  changes across minor versions; the pin must be revisited deliberately, not bumped
  casually, when it needs to move.

### Sportsbook odds — historical (resolved Phase 5: nflverse `schedules`)

- **description:** Historical betting lines (`spread_line`, `total_line`, `home_moneyline`,
  `away_moneyline`, `home_spread_odds`, `away_spread_odds`, `over_odds`, `under_odds`) bundled
  in the same raw `schedules` snapshots Phase 1 already ingests for game metadata - never
  read by any Phase 2/3/4 football-feature code (see
  `src/nfl_predict/features/registry.py`'s `MARKET_FIELD_DENYLIST`), first read in Phase 5.
- **access_method:** Already-ingested raw `schedules` Parquet snapshots
  (`nflreadpy.load_schedules()`) - no new fetch. `src/nfl_predict/market/snapshot_store.py`.
- **auth_required:** No.
- **update_frequency:** Static per season, same as `schedules` itself.
- **historical_coverage:** Verified 2010-2025, 4,363 games, **zero nulls** in every market
  column.
- **trust_level:** `third_party_aggregated`, same caveat as `schedules` generally. Field
  *meaning* is documented (verified directly against `nflreadr::dictionary_schedules`:
  `spread_line` positive = home favored); snapshot *timing* is NOT documented upstream -
  exactly one row per game, no sportsbook attribution, never labeled "closing" (see
  `docs/PHASE5_MARKET_REPORT.md`). CLV cannot be computed from this source.
- **used_for:** Phase 5's market benchmark, ATS/moneyline edge analysis, fair-line
  derivation. NOT used by the independent model (enforced by the same denylist as always).
- **known_issues:** No book attribution ("the market" here is whatever single consensus
  nflverse aggregates, undocumented which); no open/closing distinction; sign convention is
  the OPPOSITE of this project's own `docs/MODEL_SPEC.md` cover-probability convention
  (`home_spread_traditional = -spread_line` - the one conversion point,
  `nfl_predict.market.odds_math.nflverse_spread_to_traditional_home_spread`).

### Sportsbook odds — live (resolved Phase 5: provider chosen, not yet wired up)

- **description:** Real-time/near-real-time betting lines for live operation (a future
  phase) - opening/movement/closing snapshots, unlike the single historical row above.
- **access_method:** **The Odds API**, chosen as the first live provider (see
  `src/nfl_predict/market/odds_provider.py`'s `TheOddsAPIProvider`) behind a
  provider-neutral `OddsProvider` interface so a second provider can be added later without
  touching callers.
- **auth_required:** Yes — `NFL_ODDS_API_KEY` environment variable (`.env.example`). Not
  currently set in this environment; `TheOddsAPIProvider` raises
  `OddsProviderNotConfiguredError` rather than fabricating data when it's unset, and its
  HTTP methods are not yet implemented (no live prediction pipeline exists yet to need them).
- **update_frequency:** TBD once live operation exists (multiple times per week as lines
  move, per the original Phase 5 brief).
- **historical_coverage:** N/A (live-only).
- **trust_level:** TBD once live-tested — cannot be assessed without real API access.
- **used_for:** Future live prediction pipeline / decision engine. Storage is append-only
  and timestamped (`src/nfl_predict/market/live_snapshot_store.py`) so opening/closing
  movement and CLV become computable once real data accumulates - tested only against
  synthetic fixtures so far.
- **known_issues:** Line shopping means "the market" isn't a single number across books;
  which book(s) or consensus source is canonical for live operation is still undecided.

### News research (Phase 6: interface built, vendor still TBD)

- **description:** General current-news research (beat reporters, injury-adjacent
  storylines, coaching/personnel commentary) for the Phase 6 research agent's structured
  findings.
- **access_method:** `nfl_predict.research.current_data_providers.NewsResearchProvider` —
  provider-neutral ABC with a `FixtureNewsResearchProvider` for tests. No live vendor has
  been chosen yet; `LiveNewsResearchProvider` is gated on `NFL_NEWS_API_KEY` and raises
  `NotImplementedError` past that gate (no HTTP wiring yet). In Phase 6's real pilot,
  `WebSearch`/`WebFetch` (this session's own tools) stood in for a live provider — real,
  current, but not going through this structured interface.
- **auth_required:** Yes, once a vendor is chosen — `NFL_NEWS_API_KEY` (`.env.example`).
- **update_frequency:** Weekly during season, with same-week corrections common.
- **historical_coverage:** N/A — Phase 6 is prospective-only by design (see
  `docs/RESEARCH_AGENT.md`'s historical-leakage guard); this source is never used to
  reconstruct a past game's pregame state.
- **trust_level:** Governed by Phase 6's own source-tier hierarchy
  (`docs/PHASE6_RESEARCH_AGENT_REPORT.md`) — official reports are Tier 1, reputable
  reporters Tier 2, analytics publications Tier 3, community sources (Reddit etc.) Tier 4
  and never treated as verified fact without corroboration.
- **used_for:** Research agent findings (Phase 6) only — never a feature to the independent
  model.
- **known_issues:** As-of-timestamp correctness is critical — a claim is revised multiple
  times in a week; the historical-leakage guard exists specifically because ordinary current
  search cannot reconstruct what was knowable at a past timestamp.

### Structured injury data (Phase 6 interface, Phase 8A production gating: vendor still TBD)

- **description:** Official/structured injury designations (e.g. OUT/DOUBTFUL/QUESTIONABLE
  by team and player) from a direct feed — deliberately a *different boundary* than the News
  research source above. The LLM research agent's `qb_status`/`skill_position_status`/etc.
  fields are its own summary of unstructured web/news claims (tiered, fact/opinion-separated
  per `nfl_predict.research.schemas`); this source is a structured feed with no LLM
  interpretation step. The two are read and reported separately and are never merged into
  one field or one status.
- **access_method:** `nfl_predict.research.current_data_providers.InjuryProvider` —
  provider-neutral ABC with a `FixtureInjuryProvider` for tests. `LiveInjuryProvider` is
  gated on `NFL_INJURY_API_KEY` and raises `NotImplementedError` past that gate (no vendor
  chosen, no HTTP wiring yet). `nfl_predict.live.injury_automation.get_production_injury_provider()`
  is the production entry point (Phase 8A) — mirrors
  `nfl_predict.live.research_automation`'s and `nfl_predict.live.odds_automation`'s pattern:
  returns an explicit `INJURY_PROVIDER_UNAVAILABLE` status, never a silent fixture fallback,
  when the key is unset. `nfl_predict.live.run.run_slate` checks and reports this status
  (`injury_provider` component, `injury_status` in the automation report) independently of
  `research_status`.
- **auth_required:** Yes, once a vendor is chosen — `NFL_INJURY_API_KEY` (`.env.example`).
- **update_frequency:** Multiple times per week; official reports are typically finalized
  close to kickoff (e.g. Friday for early-week designations, in-game-day updates for
  gameday decisions).
- **historical_coverage:** N/A — prospective-only, same reasoning as News research above.
- **trust_level:** TBD until a live vendor is chosen and tested; conceptually Tier 1
  (official) by construction, since this boundary exists specifically to carry structured,
  non-LLM-summarized official data.
- **used_for:** Future research-agent/decision-pipeline context (Phase 8A only stands up the
  production-gating boundary and health-status reporting; no per-game structured injury
  finding is wired into a decision packet field yet — Phase 7's `DecisionInputPacket` has no
  injury-specific field, only the existing `ResearchPoint`/reason codes such as
  `MAJOR_INJURY_RISK`, which today are populated from the LLM research path).
- **known_issues:** Same as-of-timestamp sensitivity as News research; additionally, no
  vendor has been evaluated for structured-feed coverage/latency/cost.

### Weather (Phase 6: interface built, vendor still TBD)

- **description:** Game-time weather (temperature, wind, precipitation) for outdoor/retractable-roof stadiums.
- **access_method:** `nfl_predict.research.current_data_providers.WeatherProvider` — same
  ABC/fixture/live-gated pattern as the injury/news providers above.
  `LiveWeatherProvider` is gated on `NFL_WEATHER_API_KEY`, not wired to a real vendor yet.
- **auth_required:** Yes, once a vendor is chosen — `NFL_WEATHER_API_KEY` (`.env.example`).
- **update_frequency:** Live: forecast updates as game time approaches.
- **historical_coverage:** N/A — same prospective-only reasoning as above.
- **trust_level:** TBD until a live vendor is chosen and tested; needs a documented
  stadium→outdoor/dome mapping to know when weather is even relevant.
- **used_for:** Research agent findings (Phase 6). Not currently a feature-engineering
  input (Phase 2's feature set has no weather features yet).
- **known_issues:** Dome/retractable-roof stadiums need explicit handling — "no weather
  effect" is a real value, not missing data.

### LLM research provider (Phase 6: interface built, not live-tested)

- **description:** The LLM that performs Phase 6's matchup research.
- **access_method:** `nfl_predict.research.llm_provider.LLMResearchProvider` ABC.
  `AnthropicMessagesProvider` builds a real Messages API request, including Anthropic's
  server-side web-search tool (`web_search_20250305`, on by default, capped at
  `max_web_search_uses`) since a plain Messages call has no browsing capability and
  `matchup_research.md`'s prompt requires real, current, cited sources (Phase 8A correction
  pass); `FixtureLLMProvider` reads canned JSON for tests. Phase 6's real pilot used neither
  — it used `ManualResearchProvider` (this session acting as the research agent directly via
  its own `WebSearch`/`WebFetch` tools), explicitly labeled as such in every stored artifact.
  Phase 8A's `nfl_predict.live.research_live.run_live_research_for_game` is the production
  entry point that actually invokes this provider per selected game.
- **auth_required:** Yes — `NFL_RESEARCH_LLM_API_KEY` (`.env.example`), deliberately
  separate from any ambient `ANTHROPIC_API_KEY` a host environment might set. Not
  configured in this environment (no `.env` file exists); `AnthropicMessagesProvider` has
  never made a live call, including its new web-search tool wiring — this should be the
  first thing re-verified once a real key is added.
- **update_frequency:** N/A.
- **historical_coverage:** N/A.
- **trust_level:** N/A until live-tested.
- **used_for:** Research agent findings only — never a feature to the independent model,
  never a source of a numeric prediction (`docs/RESEARCH_AGENT.md`).
- **known_issues:** No live key configured; cost/token tracking
  (`nfl_predict.research.cost_tracking`) is implemented but has no real usage data yet.

## Process for adding a source

1. Add the entry to `config/sources.yaml` with every required field filled in.
2. Add the matching entry here.
3. Only after both exist does Phase 1 ingestion code for that source get written.
