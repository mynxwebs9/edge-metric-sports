"""Phase 8B: assembles consumer-safe data from already-persisted Phase 6/7/8A artifacts.

This is the ONLY layer allowed to read `src/nfl_predict/data` repository interfaces and the
Phase 6/7/8A storage modules on behalf of the website (`docs/ARCHITECTURE.md#website-api-boundary`)
- FastAPI route handlers in `main.py` call these functions and shape the result into response
schemas; they never touch storage directly. Nothing here computes a number that isn't already
in a persisted artifact - see `docs/WEBSITE_SPEC.md`'s "the website never computes numbers"
rule, which this layer inherits even though it's still Python (the FastAPI process), because
its OUTPUT is what the website ultimately displays.

**The Phase 8A snapshot-blending lesson, restated for this layer:** `provider_event_id` is
stable across time for the same real-world game, so `live_snapshot_store.read_live_snapshots`
returns EVERY historical fetch ever persisted for it, not just one. Two different needs read
that history two different ways, and conflating them is the exact bug a real production
incident already found:

- `latest_market_point()` - for "what's the market saying RIGHT NOW" (current slate, live
  matchup page) - uses only the most recent fetch batch, never a cross-time blend.
- `market_point_for_decision_record()` - for "what did THIS SPECIFIC persisted decision
  actually see" (the /decisions endpoint, or any historical reconstruction) - uses ONLY the
  exact batch whose `fetched_at` matches that decision record's own `market_snapshot_timestamp`,
  and verifies the recomputed `snapshot_reference` matches what was recorded at decision time.
  Never "latest," never "everything," never a freshly-computed consensus that includes rows
  the original decision never saw.
"""

from __future__ import annotations

from nfl_predict.content.storage import list_preview_runs, read_preview_run
from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.repositories import TeamsRepository
from nfl_predict.decision.decision_log import read_decision_records
from nfl_predict.decision.pick_ledger import read_current_picks
from nfl_predict.decision.rules_config import load_rule_set
from nfl_predict.decision.schemas import MarketPoint
from nfl_predict.live.market_consensus import compute_market_consensus
from nfl_predict.live.prediction_publication import list_predictions_for_game, read_model_prediction
from nfl_predict.live.schedule_provider import get_current_season, get_schedule
from nfl_predict.market.event_game_mapping import find_provider_event_ids_for_game
from nfl_predict.market.live_snapshot_store import read_live_snapshots
from nfl_predict.market.odds_provider import OddsMarketSnapshot
from nfl_predict.research.storage import list_research_runs, read_research_run


def team_lookup() -> dict[str, dict]:
    """`team_id -> {abbr, name, nickname}` - real, ingested team data, never hard-coded."""
    conn = get_connection()
    init_schema(conn)
    try:
        rows = TeamsRepository(conn).get_all()
        return {r["team_id"]: {"abbr": r["canonical_abbr"], "name": r["name"], "nickname": r["nickname"]} for r in rows}
    finally:
        conn.close()


def current_season_and_schedule(week: int | None = None):
    season = get_current_season()
    return season, get_schedule(season, week)


def _market_point_from_snapshots(snapshots: list[OddsMarketSnapshot], provider_event_id: str) -> MarketPoint | None:
    consensus = compute_market_consensus(snapshots)
    if consensus is None:
        return None
    representative = next((s for s in snapshots if s.home_moneyline is not None and s.away_moneyline is not None), snapshots[0])
    return MarketPoint(
        available=True, source=f"reconstructed from market/live_snapshots/event={provider_event_id}",
        home_spread_traditional=consensus.consensus_home_spread,
        home_spread_price=representative.home_spread_price, away_spread_price=representative.away_spread_price,
        home_moneyline=representative.home_moneyline, away_moneyline=representative.away_moneyline,
        no_vig_home_win_probability=consensus.consensus_home_no_vig_probability,
        snapshot_timestamp=snapshots[0].fetched_at,
        provider_event_id=provider_event_id,
        consensus_algorithm_version=consensus.consensus_algorithm_version,
        consensus_book_keys=consensus.book_keys,
        consensus_excluded_book_keys=consensus.excluded_book_keys,
        consensus_exclusion_reason=consensus.exclusion_reason,
        underlying_snapshot_keys=consensus.underlying_snapshot_keys,
        market_snapshot_reference=consensus.snapshot_reference,
    )


def latest_market_point(season: int, week: int, game_id: str) -> MarketPoint | None:
    """The CURRENT market snapshot for a game - the most recent fetch batch only. Returns
    `None` (never a fabricated MarketPoint) if no odds have ever been matched to this game."""
    provider_event_ids = find_provider_event_ids_for_game(season, week, game_id)
    if not provider_event_ids:
        return None
    provider_event_id = provider_event_ids[0]

    persisted = read_live_snapshots(provider_event_id)
    if persisted.height == 0:
        return None
    latest_fetched_at = max(persisted["fetched_at"].to_list())
    batch = persisted.filter(persisted["fetched_at"] == latest_fetched_at)
    snapshots = [OddsMarketSnapshot(**row) for row in batch.to_dicts()]
    return _market_point_from_snapshots(snapshots, provider_event_id)


def market_point_for_decision_record(record: dict) -> MarketPoint | None:
    """Reconstructs the EXACT market data a specific persisted DecisionRecord was computed
    from - never the latest snapshot, never a blend of every snapshot ever fetched for that
    provider_event_id. Returns `None` if the record predates the Phase 8A provenance
    correction (no `market_provider_event_id` was recorded) or its referenced batch can no
    longer be found - never silently substitutes a different batch."""
    provider_event_id = record.get("market_provider_event_id")
    snapshot_timestamp = record.get("market_snapshot_timestamp")
    if not provider_event_id or not snapshot_timestamp:
        return None

    persisted = read_live_snapshots(provider_event_id)
    if persisted.height == 0:
        return None
    batch = persisted.filter(persisted["fetched_at"] == snapshot_timestamp)
    if batch.height == 0:
        return None
    snapshots = [OddsMarketSnapshot(**row) for row in batch.to_dicts()]
    market = _market_point_from_snapshots(snapshots, provider_event_id)
    if market is not None and record.get("market_snapshot_reference") and market.market_snapshot_reference != record["market_snapshot_reference"]:
        # The recomputed reference must match what was recorded at decision time - a
        # mismatch means the referenced batch is not reproducible as originally computed,
        # which must surface as "unavailable," never a silently different market.
        return None
    return market


def _summarize_stored_research(stored: dict, run_id: str) -> dict:
    """Consumer-safe: classification/materiality/a short evaluator note - never raw prompt
    text or source URLs."""
    manifest = stored["manifest"]
    findings = stored["findings"]
    if manifest["findings_kind"] != "findings":
        return {
            "available": False, "run_id": run_id,
            "failure_status": findings.get("failure_status"),
            "research_timestamp": findings.get("research_timestamp"),
        }

    evaluation = stored.get("evaluation")
    classification = evaluation["final_classification"] if evaluation else findings["research_classification"]
    materiality = evaluation["overall_materiality"] if evaluation else None
    summary = evaluation["notes"] if evaluation else None
    return {
        "available": True, "run_id": run_id, "research_id": findings["research_id"],
        "classification": classification, "materiality_level": materiality, "summary": summary,
        "research_timestamp": findings["research_timestamp"],
        "unresolved_risks": tuple(findings.get("missing_information", ())),
    }


def latest_research_summary(season: int, week: int, game_id: str) -> dict | None:
    """The most recent research run for a game, by each run's own real `research_timestamp`
    - NEVER by sorting `run_id` strings. Most run_ids are ISO timestamps and sort correctly
    on their own, but a manually-named run_id (e.g. a pilot run, `run_id="pilot_20260911"`)
    sorts lexicographically after any `"2026-09-12T..."` id despite being chronologically
    much older - a real bug this exact scenario caught while building this endpoint against
    real persisted data (a stale Sep-11 pilot run was being served as "latest" over several
    real Sep-12 runs). `None` if no research has ever run for this game."""
    run_ids = list_research_runs(season, week, game_id)
    if not run_ids:
        return None
    stored_by_run_id = {run_id: read_research_run(season, week, game_id, run_id) for run_id in run_ids}
    latest_run_id = max(stored_by_run_id, key=lambda rid: stored_by_run_id[rid]["findings"]["research_timestamp"])
    return _summarize_stored_research(stored_by_run_id[latest_run_id], latest_run_id)


def research_summary_for_decision_record(season: int, week: int, record: dict) -> dict | None:
    """The EXACT research artifact a specific persisted DecisionRecord referenced (its own
    `research_id`), not just "whatever the latest research run happens to be" - the same
    exact-artifact discipline as `market_point_for_decision_record`. `season`/`week` come
    from the caller's own request context (`DecisionRecord` has no season/week field)."""
    research_id = record.get("research_id")
    game_id = record["game_id"]
    if not research_id or not research_id.startswith(f"{game_id}_"):
        return None
    run_id = research_id[len(game_id) + 1:]  # research_id is "<game_id>_<run_id>" (see run_research.py)
    stored = read_research_run(season, week, game_id, run_id)
    return _summarize_stored_research(stored, run_id)


def latest_model_prediction(game_id: str) -> dict | None:
    """The most recent, non-superseded public model prediction for a game. Falls back to the
    most recent superseded one only if every record for this game has been superseded (still
    real, still the most complete information available, just labeled accordingly)."""
    prediction_ids = list_predictions_for_game(game_id)
    if not prediction_ids:
        return None
    for prediction_id in reversed(prediction_ids):
        record = read_model_prediction(game_id, prediction_id)
        if record.get("state") != "SUPERSEDED":
            return record
    return read_model_prediction(game_id, prediction_ids[-1])


def latest_decisions_for_game(season: int, week: int, game_id: str) -> dict[str, dict | None]:
    """`{"spread": <DecisionRecord dict or None>, "moneyline": <...>}` - the most recent
    persisted decision of each market type for this game/week, or `None` if no decision has
    ever been recorded for that market type (never fabricated as NO_BET)."""
    records = read_decision_records(season, week)
    matches = [r for r in records if r["game_id"] == game_id]
    result: dict[str, dict | None] = {}
    for market_type in ("spread", "moneyline"):
        candidates = sorted((r for r in matches if r["market_type"] == market_type), key=lambda r: r["decision_timestamp"])
        result[market_type] = candidates[-1] if candidates else None
    return result


def latest_preview(season: int, week: int, game_id: str) -> dict | None:
    """The most recent generated game-preview article, by each run's own real
    `generated_at` - not by sorting `run_id` strings (the same real bug already found and
    fixed for `latest_research_summary`; guarded against here from the start). `None` if no
    preview has ever been generated for this game."""
    run_ids = list_preview_runs(season, week, game_id)
    if not run_ids:
        return None
    stored_by_run_id = {run_id: read_preview_run(season, week, game_id, run_id) for run_id in run_ids}
    latest_run_id = max(stored_by_run_id, key=lambda rid: stored_by_run_id[rid]["preview"]["generated_at"])
    stored = stored_by_run_id[latest_run_id]
    manifest, preview = stored["manifest"], stored["preview"]
    if manifest["kind"] != "preview":
        return {"available": False, "run_id": latest_run_id, "failure_status": preview.get("failure_status")}
    return {
        "available": True, "run_id": latest_run_id, "text": preview["text"],
        "generated_at": preview["generated_at"], "prompt_version": preview["prompt_version"],
        "model_provider": preview["model_provider"], "model_name": preview["model_name"],
    }


def current_best_bets() -> list[dict]:
    """Currently-open (`PUBLISHED`, not yet settled/voided) Best Bets from the official,
    immutable pick ledger - never derived from model confidence, historical hindsight, or
    any logic outside `pick_ledger.py`. Genuinely empty (`[]`) until a real pick is
    published, which has not happened as of Phase 8B (see docs/PHASE7_DECISION_ENGINE_REPORT.md
    and docs/PHASE8A_LIVE_PIPELINE_REPORT.md)."""
    picks = read_current_picks()
    return [p for p in picks if p["category"] == "BEST_BETS" and p["status"] == "PUBLISHED"]


def published_best_bet_market_types(game_id: str) -> set[str]:
    """Which market_types (subset of {"spread", "moneyline"}) currently have a real,
    PUBLISHED Best Bet for this game - distinct from which market_types merely reached
    `QUALIFIED_BET` in the decision engine (`decide()` computes those independently;
    publishing is a separate, deliberate step that doesn't have to follow every
    qualification - see live/run.py's module docstring)."""
    return {p["market_type"] for p in current_best_bets() if p["game_id"] == game_id}


def system_pick_for_game(game_id: str) -> dict | None:
    """The real, published `ALL_MODEL_PREDICTIONS` pick for this game - the model's own
    straight-up selection, at the real market price, published for EVERY game regardless of
    whether the decision engine ever reaches `QUALIFIED_BET` (see
    `nfl_predict.decision.pick_publishing`). This is deliberately a different thing from a
    Best Bet: every game gets one of these once its model/market data exists; only some
    games separately qualify for (and get published as) an actual Best Bet - see
    `published_best_bet_market_types` for that distinct, much narrower set. Returns `None`,
    never fabricated, if nothing has been published yet for this game (e.g. before the first
    real pipeline run of the week) - shown by VOIDED picks are excluded (a void means this
    specific pick was invalidated, not a real system pick to display)."""
    picks = read_current_picks()
    for p in picks:
        if p["game_id"] == game_id and p["category"] == "ALL_MODEL_PREDICTIONS" and p["status"] != "VOID":
            return p
    return None


def rule_set_version() -> str:
    return load_rule_set().rule_set_version
