# Decision Engine

## Purpose

Specifies how the independent model, market, model uncertainty/agreement, and research-agent
findings combine into a decision. **Implemented in Phase 7** — see
`src/nfl_predict/decision/` and `docs/PHASE7_DECISION_ENGINE_REPORT.md` for what was
actually built, including a real 6-decision pilot run against Phase 6's prospective pilot
packets.

## Decision categories

**Five** possible outputs per game/market (`moneyline`/`spread` only for now — no `total`,
since Phase 4/5 found insufficient independent totals signal; a single game can have
different decisions per market):

- **QUALIFIED_BET** — passes every frozen qualification rule; may enter the official
  published-pick `BEST_BETS` ledger.
- **LEAN** — a directional preference exists (real model-vs-market disagreement), but
  evidence is insufficient for publication as an official Best Bet (incomplete model
  coverage, high model-vs-model dispersion, or research supporting the market).
- **WATCH** — potentially interesting, but blocked on stale market/research data, or
  research classified `HIGH_UNCERTAINTY`.
- **NO_BET** — the default. No sufficient evidence to justify a selection (disagreement
  below the qualification minimum, or required data missing entirely).
- **VETO** — would otherwise qualify or look interesting, but research flagged
  `VETO_CONSIDERATION` (unresolved material risk) — unconditionally overrides everything
  else.

**The number of QUALIFIED_BET/LEAN decisions in a given week is never forced to a target.**
Zero qualifying games in a week is a valid, expected, reportable output — and, as of Phase
7's real pilot, the actual current outcome, given the live-data gaps documented below. The
engine does not "find something to bet" — it evaluates and reports.

## Inputs (`DecisionInputPacket` — `src/nfl_predict/decision/schemas.py`)

- Elo/Ridge/LightGBM predictions (each explicitly `available`/`unavailable`, never
  fabricated when a live feature pipeline doesn't exist for a model).
- A `ModelAgreementDescriptor` — direction agreement and margin dispersion across whichever
  models are actually available; a missing model is never counted as agreeing.
- Current market snapshot (spread, prices, moneyline, no-vig probability, snapshot
  timestamp) — `available=False` when no live odds provider is configured.
- Research classification, materiality, unresolved risks, and research timestamp (Phase 6).
- System health: missing-data list, market/research age in seconds.

No decision may use information timestamped after its own `decision_timestamp` —
enforced explicitly in `nfl_predict.decision.input_packet.build_decision_packet_from_research_run`.

## Thresholds (rules, not bare numbers)

Every numeric threshold lives in `config/decision_rules.yaml`, each with `rule_id`,
`version`, `description`, `required_inputs`, `threshold`, `rationale`, `provenance`, and
`status` (`EXPERIMENTAL | PROSPECTIVE_VALIDATION | ACTIVE | RETIRED`). Phase 7's initial
rule set (`v1`) is entirely `PROSPECTIVE_VALIDATION` — every threshold traces to Phase 5's
own pre-declared (outcome-blind) bucket boundaries or to general operational reasoning,
**never** to which Phase 5 bucket or Phase 6 pilot game happened to look best. See
`docs/PHASE7_DECISION_ENGINE_REPORT.md`'s "Do not derive thresholds from 2024-2025" section.

## Hard boundaries

- The decision engine reads model outputs and research findings; it never asks an LLM to
  freehand a decision or a number, and no LLM call happens inside `decide()` at all — Phase
  6's research runs strictly before Phase 7's decision. Given the same packet and rule set,
  `decide()` always returns the same classification and reason codes.
- A VETO (and every other decision) records structured `ReasonCode`s — never a bare
  classification with no explanation attached.
- Published picks are stored in an event-sourced, append-only ledger
  (`nfl_predict.decision.pick_ledger`) — a pick's line/price/selection/category, once
  published, are never altered; settlement and void events are separate, appended records
  that never rewrite the original.

## Resolved in Phase 7

- **Decision categories**: `NO_BET | WATCH | LEAN | QUALIFIED_BET | VETO` (this doc's
  earlier `BET/LEAN/NO BET/VETO` 4-category placeholder is superseded).
- **Rule versioning**: `config/decision_rules.yaml`, superseding the null-valued
  `config/decision_thresholds.yaml` placeholder (still present, documented as superseded).
- **Official pick categories**: `ALL_MODEL_PREDICTIONS | LEANS | BEST_BETS`, tracked and
  reported strictly separately, never combined or retroactively reclassified.
- **`EXPERT_PICKS`** (added later): a human's own spread/moneyline picks, entered via
  `python -m nfl_predict.decision.expert_picks`. NOT a decision-engine output - no model,
  market, or research gate applies, and it never feeds or is fed by any other category.
  It shares the ledger, `settle_all_pending_picks()` and price-aware record math, and the
  code enforces what makes such a record credible: `published_at` is always the real time
  of publication (never caller-supplied), a pick is refused once its game has kicked off or
  isn't `scheduled`, and there is one immutable pick per game per market.
- **`EXPERT_PARLAYS`**: a human's multi-leg parlay (`python -m nfl_predict.decision.expert_parlays`),
  its own record, never blended with `EXPERT_PICKS` or the model's categories. Legs may be a
  moneyline, a spread, or a player prop (a whitelisted countable stat over/under a line); each
  leg's game must be pre-kickoff and legs are frozen into the ledger at publish time. The
  record's units use the parlay's own American price as the sportsbook offered it (a
  same-game parlay is not simply the legs multiplied). Grading (`parlay_settlement.py`): a
  parlay loses the moment any leg loses, wins only if every leg wins, stays pending until
  every needed box score is in the ingested `player_stats` snapshot (a stale snapshot never
  grades a leg), and anything ambiguous - a pushed leg, or a player with no stat row - is
  surfaced as `needs_review` and left unsettled, never guessed. Settling needs
  `python -m nfl_predict.data.ingest --dataset player_stats --seasons <season>` alongside the
  schedules ingest.
- **Voiding**: a pick voided before kickoff (only the ledger's restricted reasons) is never
  hidden - the Expert Picks page lists it with the reason, and it counts in no record.
- **Settlement**: deterministic, spread and moneyline only (no totals).
- **Streak/headline engine**: predefined windows only
  (`last_5/10/20/30`/`season_to_date`/`current_streak`) — no arbitrary date ranges.
- **Shadow decisions**: an experimental rule version can be evaluated prospectively without
  ever entering the official ledger.

## Open items for a future phase

- Exact ACTIVE thresholds — still deferred; every Phase 7 rule remains
  `PROSPECTIVE_VALIDATION` until genuinely validated on data that did not inform it.
- Per-game model uncertainty (vs. Phase 3/4's single global residual_std) — no numeric
  `max_model_uncertainty` rule exists yet for this reason.
- A live odds provider and a live Ridge/LightGBM feature pipeline now both exist (Phase 8A —
  see `docs/PHASE8A_LIVE_PIPELINE_REPORT.md`), but no live odds/research credential is
  configured in this environment, so `QUALIFIED_BET` remains not practically reachable — an
  honest consequence of missing credentials, not missing code.
