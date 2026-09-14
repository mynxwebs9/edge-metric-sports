# Phase 7 — Decision Engine and Verified Pick Ledger

## Objective

Phase 5 found no demonstrated stable betting edge. Phase 6's research layer has only three
pilot games and hasn't demonstrated it improves anything. Phase 7 does **not** try to close
that gap with a profitable-looking system. It builds: (1) a conservative, deterministic
decision engine; (2) a mechanism for prospectively freezing selections; (3) a verified,
append-only record system; (4) the infrastructure to later display truthful streak/record
claims — **only once the published ledger genuinely earns them.**

## Architecture

```
src/nfl_predict/decision/
  schemas.py         Decision (5-value enum), ReasonCode (closed enum), ModelAgreementDescriptor,
                      DecisionInputPacket, DecisionRecord
  rules_config.py     loads config/decision_rules.yaml into typed, versioned Rule objects
  model_agreement.py   Step 8: qualitative agreement/dispersion descriptor across available models
  staleness.py         Step 6: age computation + staleness checks (unknown age = stale, never fresh)
  engine.py            Step 1/23: the deterministic decide() function
  input_packet.py      Step 5: builds DecisionInputPacket from Phase 6's stored research runs
  pick_ledger.py        Step 11/17: event-sourced, append-only official pick ledger
  settlement.py         Step 16: deterministic spread/moneyline settlement (no totals yet)
  records.py            Step 12/15: price-aware records, strictly separate per category
  streaks.py             Step 13/14: predefined-window streak/headline engine
  shadow.py              Step 19: shadow decisions, isolated from the official ledger
  decision_log.py        append-only audit log of every decision (not just published picks)
config/
  decision_rules.yaml    Step 3: versioned, documented decision rules (superseding the
                          Phase 0 decision_thresholds.yaml placeholder)
```

## Decision categories

`NO_BET | WATCH | LEAN | QUALIFIED_BET | VETO` — `docs/DECISION_ENGINE.md` updated from its
earlier 4-category placeholder to match. Default is `NO_BET`. `decide()` takes a
`DecisionInputPacket` and a `RuleSet`, and returns `(Decision, tuple[ReasonCode, ...])`,
deterministically — no LLM call happens inside it; Phase 6's research output enters only as
already-parsed structured fields (`ResearchPoint`).

## Decision tree (rule precedence, all versioned in `config/decision_rules.yaml`)

1. `research.classification == VETO_CONSIDERATION` → **VETO**, unconditionally.
2. Market or research unavailable → **NO_BET** (`MISSING_LIVE_DATA`).
3. Market or research stale (>24h / >72h) → **WATCH** (`STALE_MARKET`/`STALE_RESEARCH`).
4. `research.classification == HIGH_UNCERTAINTY` → **WATCH**.
5. Model-vs-market disagreement below the rule's minimum (3.0 spread points / 3% moneyline
   probability) → **NO_BET** (`DISAGREEMENT_BELOW_MINIMUM`).
6. Disagreement clears the minimum, but model coverage is incomplete (not all of
   Elo/Ridge/LightGBM available and agreeing) → **LEAN** (`INSUFFICIENT_MODEL_COVERAGE`).
7. Coverage complete, but models disagree with EACH OTHER by more than 3.0 points → **LEAN**
   (`MODEL_DISAGREEMENT_HIGH`).
8. Coverage complete, models agree, but `research.classification == SUPPORTS_MARKET` →
   **LEAN** (`RESEARCH_SUPPORTS_MARKET`).
9. Otherwise → **QUALIFIED_BET**, always carrying `NO_DEMONSTRATED_EDGE` as a permanent,
   structural caveat in its reason codes.

## Do not derive thresholds from 2024-2025 (or the Phase 6 pilot)

Every numeric threshold in `config/decision_rules.yaml` traces to one of two sources, never
to "which Phase 5 bucket/pilot game looked best":

1. **Phase 5's own PRE-DECLARED bucket boundaries** (0-1/1-2/2-3/3-4/4+ points;
   <1%/1-2%/2-3%/3-5%/5%+ probability) — fixed before Phase 5 saw any betting result. The
   3.0-point / 3% thresholds land on those boundaries for "meaningfully large," motivated by
   Phase 5's bootstrap CIs never excluding zero below that scale — not by which bucket won.
2. **General, outcome-blind operational reasoning** — a 24-hour market-staleness cap, a
   72-hour research-staleness cap, a 3.0-point model-dispersion cap, requiring all 3 models
   to agree for full qualification. None of these were tuned against any observed win rate.

Every rule's `status` is `PROSPECTIVE_VALIDATION`, never `ACTIVE` — this project does not
claim these thresholds are validated or profitable.

## Model agreement

`compute_model_agreement` reports `models_considered`, `all_agree_on_direction` (`None` if
fewer than 2 models are available — a missing model is never silently counted as agreeing,
per Step 21), and `margin_dispersion`. It never claims agreement implies profitability.

## Immutable, event-sourced pick ledger

`pick_ledger.py` stores every state change as its own appended event
(`PUBLISHED`/`SETTLED`/`VOIDED`) rather than a mutable row — the `PUBLISHED` event's line,
price, selection, category, and every snapshot field are never touched again. Settling or
voiding a pick appends a new event; there is no delete function, no reclassify function, no
promote/demote function anywhere in the module (verified directly in
`tests/decision/test_pick_ledger.py`). `void_pick` only accepts one of four enumerated
operational reasons (`GAME_CANCELLED`, `SPORTSBOOK_MARKET_VOIDED`,
`CORRUPTED_INPUT_DETECTED_PRE_EVENT`, `DUPLICATE_PUBLICATION`) — "the model was wrong" is
not among them.

## Official pick categories (Step 10)

`ALL_MODEL_PREDICTIONS | LEANS | BEST_BETS` — fixed on `PublishedPick.category` at
publication and never changed afterward. `records.py`'s `compute_category_record` computes
each category's win/loss/push/units/ROI **strictly separately** — never combined, never
retroactively reclassified.

## Settlement (Step 16)

`settlement.py` supports `spread` and `moneyline` only — **no totals**, per Phase 4/5's
finding of insufficient independent totals signal. Spread settlement reuses
`nfl_predict.market.odds_math.grade_ats` (the same sign convention and push logic Phase 5
already validated); moneyline settlement treats a tied game as a push, the standard
real-world sportsbook rule.

## Price-aware records (Step 15)

`PublishedPick.price` is a **required** field — a pick is never published without a real
captured price, so the official record never needs to silently assume `-110`.
`compute_hypothetical_assumed_price_record` exists as a clearly separate, explicitly-labeled
function for reference-only "what if every price were -110" analysis; it is never blended
with the official, price-aware record.

## Streak / headline engine (Step 13/14)

Only six predefined windows exist — `last_5 | last_10 | last_20 | last_30 |
season_to_date | current_streak` — `compute_window` has no `start_date`/`end_date`
parameter anywhere, so a cherry-picked range like "17-3 since August 29" cannot be produced.
A headline is only ever generated for the `BEST_BETS` category, and only once at least 5
settled picks exist in the window (`MIN_SETTLED_FOR_HEADLINE = 5`). Pushes are counted but
never break a win/loss streak.

**Fixture demonstration (synthetic test data — not real settled picks; illustrates the
mechanism only, zero connection to actual performance):**

| Window | Record | Headline |
|---|---|---|
| current_streak | 4-0 | *(none — below the 5-pick minimum, correctly withheld)* |
| last_5 | 4-1 | "4-1 Last 5 Best Bets (+2.6 units)" |
| last_10 | 8-2 | "8-2 Last 10 Best Bets (+5.3 units)" |

## Shadow decisions (Step 19)

`shadow.py` stores experimental-rule-version decisions in a completely separate directory
tree (`data/decision/shadow_decisions/<rule_set_version>.jsonl`), keyed by rule set version
— structurally unreachable from `pick_ledger.read_current_picks()`.
`compare_shadow_to_official` is read-only and creates no picks; nothing in this codebase
promotes a shadow rule to official status automatically.

## The Phase 7 pilot

Ran the v1 decision engine against all three Phase 6 pilot packets (`2026_01_DEN_KC`,
`2026_01_BUF_HOU`, `2026_01_CHI_CAR`), for both `spread` and `moneyline` market types (6
decisions total), using `nfl_predict.decision.input_packet.build_decision_packet_from_research_run`
against the exact, unmodified Phase 6 research runs. Result:

| Game | Spread | Moneyline |
|---|---|---|
| DEN @ KC | NO_BET (MISSING_LIVE_DATA) | NO_BET (MISSING_LIVE_DATA) |
| BUF @ HOU | NO_BET (MISSING_LIVE_DATA) | NO_BET (MISSING_LIVE_DATA) |
| CHI @ CAR | NO_BET (MISSING_LIVE_DATA) | NO_BET (MISSING_LIVE_DATA) |

**Zero QUALIFIED_BET, zero LEAN, zero WATCH, zero VETO — all six are NO_BET.** This is the
honest, unforced consequence of Phase 5/6's own documented live-data gap: no live odds
provider is configured, so `MarketPoint.available` is `False` for every pilot packet, and
`missing_market_or_research_blocks_qualified_bet`-equivalent logic (the second rule checked)
returns `NO_BET` before any disagreement/agreement logic even runs. All six decisions were
persisted to `data/decision/decision_log/season=2026/week=1/decisions.jsonl` before kickoff
(kickoff 2026-09-13/14, decision timestamp 2026-09-11) — genuinely prospective, hash-verified,
and never touching the Phase 6 research packets.

## Live-data limitations

Same as Phase 6 reported: no live odds provider is configured
(`NFL_ODDS_API_KEY` unset — `nfl_predict.market.odds_provider.TheOddsAPIProvider` has no
wired HTTP calls), and Ridge/LightGBM have no live feature-engineering pipeline for any
season beyond the sealed 2024-2025 holdout. The engine handles both gracefully: missing
market data blocks qualification entirely (`MISSING_LIVE_DATA`), and missing Ridge/LightGBM
is never silently counted as model agreement (`require_full_model_coverage_for_qualified_bet`
explicitly requires `n_models_available == 3`).

## Known weaknesses

- **No per-game model uncertainty**: Phase 3/4 established only a single global margin
  residual_std, not a situation-specific interval — `ModelPoint.uncertainty_note` carries
  descriptive text, not a number a rule can threshold on. `max_model_uncertainty` (mentioned
  in the original `docs/DECISION_ENGINE.md` placeholder) is not implemented as a numeric
  rule for this reason - documented here as deferred, not silently dropped.
- **QUALIFIED_BET is currently unreachable** given the live-data gap (no live market, and
  full 3-model coverage is never available) — an intentional, honest consequence of
  conservative rules meeting real infrastructure limits, not a bug.
- **The streak/headline engine has never processed a real settled pick** — the
  demonstration above is 100% synthetic fixture data, explicitly labeled.
- **Shadow decisions have no experimental rule set actually running yet** — the
  infrastructure exists and is tested, but no second rule version has been authored.

## Tests

61 new tests (`tests/decision/`) covering all 30 required proofs: default NO_BET, no
minimum-pick-count requirement, pick immutability (deletion/line/price), category
immutability (no reclassify/promote/demote), spread/moneyline settlement, price-aware units
with no silent `-110` assumption, predefined-window streak correctness (current/last-N/
season), headline cherry-picking prevention, research cannot alter model numbers,
determinism, staleness gating, VETO_CONSIDERATION hard-blocking, missing-data safe failure,
shadow-decision isolation, rule-version immutability of prior decisions, append-only
corrections, outcome/snapshot non-mutation, settled-only record denominators, and push
handling in streaks.
