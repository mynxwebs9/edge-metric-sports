# LLM Research Agent

## Purpose

Specifies what the Phase 6 LLM research step is for, what it may and may not do, and how its
output is structured and stored. **Implemented in Phase 6** — see
`src/nfl_predict/research/` and `docs/PHASE6_RESEARCH_AGENT_REPORT.md` for what was actually
built, including a real 3-game prospective pilot. This document is kept as the durable
contract; where the actual schema is more detailed than what's summarized below (source
tiers, a 5-way fact/opinion split, a 0-4 materiality rubric, a closed classification enum),
`src/nfl_predict/research/schemas.py` is authoritative.

## What the research agent is for

The research agent runs **after** the quantitative model has already produced its
independent and market-aware predictions for a game. Its central question is:

> Is there credible current information that our structured quantitative data is missing —
> especially information that might explain a disagreement between our model and the market?

It is not a second predictor. It does not produce a score, a probability, a spread, or any
numeric quantity that could be confused with a model output. It produces research findings
that a human (or the decision engine, per `docs/DECISION_ENGINE.md`) can weigh.

## Research scope

- Injury context and QB status specifically.
- Offensive-line availability.
- Defensive personnel changes.
- Coach comments and local beat reporting.
- Expected snap-count limitations.
- Roster changes.
- Weather context (beyond the raw forecast the feature pipeline already has).
- Scheme matchup notes.
- Reputable analytical models' outputs (as external reference points, not inputs to be
  copied).
- Reputable analyst opinions.

## Required output structure

Every research agent run for a game produces a structured record (`ResearchFindings` —
`src/nfl_predict/research/schemas.py`) separating claims into exactly one of five
categories, each carrying a `SourceTier` (see below), a `materiality_level` (0 NOISE to 4
CRITICAL), and a `confidence_in_fact`:

1. **VERIFIED_FACT** — verifiable, sourced statements ("Team X's starting LT was listed as
   Questionable on Friday's injury report" with a citation). No inference. Cannot be
   constructed without at least one source.
2. **REPORTED_NOT_CONFIRMED** — a report that hasn't reached the bar of a verified fact.
3. **ANALYST_OPINION** — a named analyst's/reporter's interpretation, attributed to them,
   never adopted as the agent's own claim.
4. **COMMUNITY_SENTIMENT** — aggregate public/fan sentiment, never elevated to fact.
5. **MODEL_ANALYTICS_OPINION** — another analytics model's/publication's output, collected
   for reference, never adopted as the agent's own claim.

Every source cited carries: URL, title, publisher, a source tier (`TIER_1_OFFICIAL` down to
`TIER_4_COMMUNITY` — see `docs/PHASE6_RESEARCH_AGENT_REPORT.md`'s source hierarchy), a
retrieval timestamp, and the publication timestamp of the underlying information when it
can be established (`null`, never invented, otherwise). A `VERIFIED_FACT` sourced only from
Tier 4 with no independent corroboration is automatically downgraded to
`REPORTED_NOT_CONFIRMED` by the evaluator.

## Hard boundaries

- **The research agent never overrides, adjusts, or "corrects" the quantitative model's
  numbers.** It cannot output a modified spread, a modified win probability, or anything
  that looks like a replacement number. If the agent finds something important, that is
  input to a human or to the decision engine (Phase 7) — the model's stored prediction is
  unchanged and the disagreement is exactly the kind of signal the decision engine is
  designed to weigh.
- **The research agent does not run before or during model inference**, and its output is
  never fed back into the model as a feature for the same prediction cycle (see
  `docs/MODEL_SPEC.md`, "What the model may never take as input"). This keeps the
  quantitative track auditable independent of any LLM behavior/prompt changes.
- **External consensus does not automatically move the prediction.** "Everyone else has this
  team favored" is EXTERNAL OPINION, logged as such — it is not, by itself, a reason to
  change the stored model output.
- Every research run is stored with its full output, sources, and timestamp — not
  summarized-then-discarded. It must be reconstructable later, same as a prediction.

## Prompts

The actual prompt templates live under `prompts/` (`matchup_research.md`,
`research_evaluator.md`, `research_summary.md` — plus `prediction_writer.md`, a separate,
Phase 8/9-scoped prompt for website content generation, not part of the research agent) and
are versioned like code (`matchup_research_v1`, etc. — see
`docs/PHASE6_RESEARCH_AGENT_REPORT.md`). Every stored research run records its exact
`prompt_version`. The evaluator prompt is written and versioned but not currently invoked —
`src/nfl_predict/research/evaluator.py` implements Step 15's evaluation jobs as
deterministic code instead, for the same reliability reasons `docs/DECISION_ENGINE.md`
insists on deterministic code downstream of any LLM step.

## Historical vs. prospective research

The research agent is **prospective-first**: `src/nfl_predict/research/historical_guard.py`
blocks ordinary current-web research from being used to generate supposedly-pregame
research for a game whose kickoff has already passed, unless the caller supplies proof
that every source used existed, pre-kickoff-contamination-free, before that timestamp. No
such proof has ever been constructed in this codebase — no historical (2024-2025) research
records exist, by design, not by omission.
