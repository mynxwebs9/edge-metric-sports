# Phase 6 — Prospective LLM Research Agent

## Objective

Phase 5 found the sportsbook market outperforms every independent model and that no stable
betting edge exists. Phase 6 does **not** try to invent one. Its question is narrower: can
a disciplined research process identify *current pregame context* — injuries, OL/personnel
changes, coaching notes, weather, logistics — that the structured quantitative data doesn't
capture, without ever adjusting a model number or recommending a bet? The research agent
acts as a **RESEARCH ANALYST / RISK DETECTOR**, never a second prediction model.

## Architecture

```
src/nfl_predict/research/
  schemas.py                 SourceTier, ClaimCategory, MaterialityLevel, ResearchClassification,
                              FailureStatus, SourceRecord, Claim, ExternalPrediction,
                              ResearchFindings, FailedResearchRun, EvaluationResult
  input_packet.py             ResearchInputPacket - historical (Phase 4/5 ledger+market) and
                              live (frozen Elo continued forward) builders
  historical_guard.py         HistoricalResearchLeakageRisk / InvalidPregameTimestampError
  llm_provider.py             LLMResearchProvider ABC, AnthropicMessagesProvider, FixtureLLMProvider
  manual_provider.py           ManualResearchProvider - used for this phase's real pilot (see below)
  current_data_providers.py   InjuryProvider / WeatherProvider / NewsResearchProvider + fixtures
  trigger.py                   configurable research-priority scoring (never gates research)
  cost_tracking.py             CostTracker + ResearchDepthExceededError
  parsing.py                   raw LLM text -> ResearchFindings, or ResearchOutputParseError
  evaluator.py                 deterministic re-derivation: dedup, downgrade, materiality, classification
  storage.py                   immutable data/research/season=/week=/<game_id>/run_id=/ records
  prospective_ledger.py        append-only frozen-pregame ledger, outcomes joined separately
  run_research.py              orchestrator wiring all of the above together
prompts/
  matchup_research.md (v1)     the research-agent prompt
  research_evaluator.md (v1)   written/versioned; NOT currently invoked (see below)
  research_summary.md (v1)     written/versioned; NOT currently invoked (future human-facing summary)
```

### Design note: the evaluator is deterministic code, not a second LLM call

Step 15's evaluator jobs — verify internal consistency, downgrade weak sources,
deduplicate evidence, assess materiality, produce the final classification — are exactly
the kind of reliable, auditable, rule-based work `docs/DECISION_ENGINE.md` already insists
on for anything downstream of an LLM. `nfl_predict.research.evaluator` implements them as
plain Python functions operating on the parsed `ResearchFindings`, not a second prompt
call. `prompts/research_evaluator.md` is still written and versioned (satisfying Step 13's
requirement to store and version it), kept for a possible future LLM-assisted enhancement,
but the current pipeline never calls it.

## Prompt versions

- `matchup_research_v1` — the research-agent prompt, rewritten this phase from a Phase 0
  placeholder (which used a simpler 3-way FACTS/INTERPRETATION/EXTERNAL-OPINION split) to
  match this phase's 5-way fact/opinion split, source-tier hierarchy, materiality rubric,
  and structured JSON output schema.
- `research_evaluator_v1` — written, versioned, not currently invoked (see above).
- `research_summary_v1` — written, versioned, not currently invoked (a future
  human-readable summary step, never a new research or numeric step).

Every stored research run records its `prompt_version`, `model_provider`, `model_name`,
and `input_packet_hash` — verified in `tests/research/test_run_research.py`.

## Source hierarchy

`SourceTier`: `TIER_1_OFFICIAL` (official team/league reports, injury reports, direct
coach/player press-conference reporting) > `TIER_2_REPUTABLE_REPORTER` (beat/national
reporters, credible news orgs) > `TIER_3_ANALYTICS_PUBLICATION` (analytics/betting
publications) > `TIER_4_COMMUNITY` (Reddit and similar). A `VERIFIED_FACT` claim sourced
*only* from Tier 4, with no independent corroboration, is automatically downgraded to
`REPORTED_NOT_CONFIRMED` by the evaluator (`tests/research/test_evaluator.py`).

## Fact/opinion separation

Every claim carries exactly one `ClaimCategory`: `VERIFIED_FACT`, `REPORTED_NOT_CONFIRMED`,
`ANALYST_OPINION`, `COMMUNITY_SENTIMENT`, `MODEL_ANALYTICS_OPINION`. A `VERIFIED_FACT`
cannot be constructed without at least one source (`Claim.__post_init__` raises otherwise)
— unsupported factual assertions are structurally impossible, not just discouraged.

## Materiality rubric

`MaterialityLevel`: `0` NOISE / `1` MINOR / `2` MODERATE / `3` MAJOR / `4` CRITICAL (e.g.
genuine starting-QB uncertainty). `overall_materiality` is the max across a research run's
facts; `CRITICAL` materiality anywhere deterministically forces the final classification to
`VETO_CONSIDERATION`, regardless of what the raw research output self-reported.

## Research classification (closed enum, never free text)

`SUPPORTS_MODEL | SUPPORTS_MARKET | MIXED | NO_MATERIAL_NEW_INFORMATION | HIGH_UNCERTAINTY |
VETO_CONSIDERATION`. These summarize the research's relationship to existing
quantitative/market information — they never create or cancel a bet.

## No numerical overrides

`ResearchFindings` (and every nested type) has **no field** that could carry a spread, a
probability, or a margin. `tests/research/test_run_research.py`'s
`test_llm_output_numeric_fields_are_ignored_and_never_reach_the_ledger` proves this
directly: a hostile provider that returns `"predicted_margin": 999.0` alongside valid
findings produces a stored ledger entry whose `elo_predicted_margin` is still exactly the
original frozen value — the injected number is silently dropped by the parser, which only
ever reads the fields the schema defines.

## Historical leakage protection

`historical_guard.assert_research_may_proceed` raises `HistoricalResearchLeakageRisk` if a
game's kickoff has already passed relative to "now" and no `ArchivedSourceAuthorization`
(proof that every source was independently verified pre-kickoff-contamination-free) is
supplied — and raises `InvalidPregameTimestampError` if the research record's own claimed
timestamp is after kickoff at all, authorization or not. `run_research_for_game` is a
**prospective-only** orchestrator (its claimed research timestamp is always "now") — no
supposedly-historical 2024-2025 research was generated this phase, and the architecture
structurally cannot do so without an explicit, human-supplied authorization object that was
never constructed.

## Prospective ledger and immutable storage

`data/research/season=<S>/week=<W>/<game_id>/run_id=<run_id>/{input,findings,evaluation,manifest}.json`
— `write_research_run` raises `ResearchRunAlreadyExistsError` on a duplicate `run_id`, so
Thursday/Friday/Sunday-morning research for the same game are always separate, preserved
runs. `data/research/prospective_ledger/season=<S>/week=<W>/entries.jsonl` is append-only
and hash-verified on every read; `join_outcomes` returns a **new** list with outcomes
merged in — it never mutates the stored ledger.

## Live current-data provider interfaces

`InjuryProvider` / `WeatherProvider` / `NewsResearchProvider` (`current_data_providers.py`)
are each a small ABC with a `Fixture*Provider` for tests and a `Live*Provider` gated on an
env-var API key (`NFL_WEATHER_API_KEY`, `NFL_NEWS_API_KEY`) — no vendor has been chosen for
either yet, so the live adapters raise `NotImplementedError` past the credential gate. The
LLM provider (`AnthropicMessagesProvider`) is real request-construction code gated on
`NFL_RESEARCH_LLM_API_KEY`, not live-tested in this environment (no key configured).

## Cost control

`CostTracker` records LLM-call token/cost totals and search-query/source counts;
`ResearchDepthExceededError` raises once `ResearchCostConfig`'s `max_search_queries`
(default 6) or `max_source_count` (default 20) would be exceeded — an automated research
loop cannot silently run away.

## Failure modes

`FailureStatus`: `NO_WEB_ACCESS | SOURCE_TIMEOUT | LLM_FAILURE | INVALID_JSON |
INSUFFICIENT_SOURCES | CONFLICTING_REPORTS`. A failed run is stored as an explicit
`FailedResearchRun` — never coerced into `NO_MATERIAL_NEW_INFORMATION` — and produces **no**
prospective-ledger entry (`tests/research/test_remaining_proofs.py`).

## The real prospective pilot

Three real, current, upcoming Week 1 (2026 season) games were researched on 2026-09-11,
roughly two days before their Sunday/Monday kickoffs — genuinely prospective, not a
backtest. Research method: real `WebSearch` queries run in this session (not an automated
`AnthropicMessagesProvider` call — no LLM API key is configured in this environment). The
findings were authored directly from those real search results, then run through the
**actual** pipeline (`ManualResearchProvider` → `run_research_for_game` → parsing →
evaluation → storage → prospective ledger), exactly the code path an automated provider
would use — see `scripts/run_phase6_pilot.py` for the exact JSON stored per game and every
source URL.

| Game | Kickoff | Elo margin (home) | Elo home win prob | Current market (web-sourced, informational) | Classification | Materiality |
|---|---|---|---|---|---|---|
| DEN @ **KC** | 2026-09-14 MNF | −3.24 (favors DEN) | 0.391 | KC −2.5→−3, ML −148/+124, total 42.5 | SUPPORTS_MARKET | MAJOR |
| BUF @ **HOU** | 2026-09-13 | +2.02 (favors HOU) | 0.565 | BUF −1.5, ML −120/+100, total 44.5 | MIXED | MINOR |
| CHI @ **CAR** | 2026-09-13 | −1.51 (favors CHI) | 0.447 | CHI −2.5 | MIXED | MAJOR |

Elo's margin is the frozen Phase 4 config (k=40, home_adv=45) continued sequentially
forward through every completed game — **never refit or retrained**, per this phase's
explicit instruction. Ridge/LightGBM predictions and a live market snapshot are
`UNAVAILABLE` in every packet (see "Known limitations" below) — the "current market" column
above was found via `WebSearch`, recorded as research context only, never as a structured
`market_snapshot` field, since no live odds provider is configured.

**Material findings**, one per game (full detail, sources, and materiality reasoning in
each `findings.json`):

- **DEN @ KC**: Elo's large disagreement with the market (Elo favors Denver by ~3.2, the
  market favors Kansas City by ~2.5-3) is NOT obviously explained by anything found — Kansas
  City looks reasonably healthy (Mahomes had full practice participation returning from his
  ACL tear; only a swing tackle and a depth safety are out) while Denver, though fully
  healthy, has no specific new positive information. Classified `SUPPORTS_MARKET`: current
  facts support the market's read better than Elo's team-strength prior, which cannot see
  this week's practice reports at all.
- **BUF @ HOU**: a real but moderate (~3.5 point) Elo/market disagreement (Elo favors
  Houston, the market favors Buffalo) with no clear explanation in the research — both
  teams are healthy at the positions that matter (confirmed starting QBs, only backup-level
  injuries). Classified `MIXED`.
- **CHI @ CAR**: Elo and the market already agree in direction (both favor Chicago, by
  1.5 vs. 2.5 points) — a small gap, not flagged as a priority disagreement. The most
  material fact found is Chicago's starting LT being out on the road, real OL context that
  doesn't clearly resolve toward either side. Classified `MIXED`.

None of these classifications created, cancelled, or adjusted anything — Elo's stored
margin/probability for all three games is exactly what `input_packet.py` computed before
any research began, verified byte-for-byte via each run's `manifest.json` hashes.

## Known limitations

- **No live market snapshot**: Phase 5's live odds provider (`TheOddsAPIProvider`) has no
  wired HTTP implementation and no configured key — every pilot packet's `market.available`
  is `False`. The market context shown in this report's table came from `WebSearch`, not
  the structured pipeline, and is clearly labeled as such.
- **Ridge/LightGBM unavailable live**: both require Phase 2's feature-engineering pipeline
  to have run against the current season's play-by-play, which does not exist yet for any
  season beyond the sealed 2024-2025 holdout. Only Elo — which needs no feature engineering
  by design — could be run live for this pilot.
- **No automated LLM provider was live-tested**: `AnthropicMessagesProvider`'s HTTP request
  construction is real code, gated on `NFL_RESEARCH_LLM_API_KEY`, but that key is not set in
  this environment. The pilot used `ManualResearchProvider` instead, explicitly labeled as
  such (`provider_name="manual-claude-code-session"`) in every stored artifact.
- **The evaluator is rule-based, not LLM-assisted** (see design note above) — a genuinely
  subtle internal-consistency issue between two claims might not be caught by the current
  simple text-based deduplication/downgrade rules.
- **Sample size**: three pilot games is enough to prove the pipeline works end-to-end, not
  enough to say anything about whether the research layer stratifies model trustworthiness
  (Step 18's real question) — that requires many more prospective games over time.
- **No historical (2024-2025) research was attempted**, by design — the historical-leakage
  guard makes this a deliberate architectural choice, not a missing feature.

## Tests

64 new tests (`tests/research/`), covering all 18 required proofs: historical leakage
blocking, pregame-timestamp enforcement, quantitative-value immutability, numeric-override
rejection, unsupported-claim flagging, source-tier downgrading, fact/opinion separation,
duplicate detection, run versioning, append-only multi-run storage, outcome-join
non-mutation, secret-free artifacts, invalid-JSON safe failure, insufficient-source
failure, classification-enum enforcement, prompt-version recording, hash reproducibility,
and timestamp-precedes-kickoff. One real bug was caught and fixed while building this:
`storage.py`'s original `Path.write_text` + hash-from-string pattern would have silently
broken tamper detection on Windows (CRLF translation changes on-disk bytes after the hash
is computed) — fixed to hash the exact bytes written (`write_bytes`), the same pattern
`nfl_predict.backtesting.ledger` and `nfl_predict.market.snapshot_store` already used
correctly.
