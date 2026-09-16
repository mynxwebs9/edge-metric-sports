"""Phase 10 follow-up: finds every published pick - in EITHER `PickCategory` - whose game
has gone final and hasn't been settled yet, computes its real WIN/LOSS/PUSH from the real
final score, and records it. Ties together two pieces that already existed separately:
`nfl_predict.decision.settlement` (deterministic spread/moneyline grading, Phase 7 - pure
computation, no I/O) and `nfl_predict.decision.pick_ledger.settle_pick` (persists a given
result to the immutable ledger) - nothing here computes a number that wasn't already real.

Never distinguishes BEST_BETS from ALL_MODEL_PREDICTIONS in how it settles a pick - a pick's
`category` is a permanent, immutable field recorded at publish time; this module just grades
whichever picks exist against the real score, and the two categories' records stay exactly as
separate afterward as they were before (see `pick_publishing.py` for why they're separate).
"""

from __future__ import annotations

from datetime import datetime, timezone

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.decision.pick_ledger import PickStatus, read_current_picks
from nfl_predict.decision.pick_ledger import settle_pick as record_settlement
from nfl_predict.decision.settlement import UnsupportedMarketTypeError
from nfl_predict.decision.settlement import settle_pick as compute_settlement
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)


def settle_all_pending_picks(now: str | None = None) -> dict:
    """Returns real, never-fabricated counts: `settled` (list of {pick_id, category, result}),
    `still_pending` (game not final yet), `skipped_unsupported` (a market_type this project
    doesn't grade, e.g. a future totals pick - recorded, never silently dropped)."""
    now_iso = now or datetime.now(timezone.utc).isoformat()
    settled, still_pending, skipped_unsupported = [], [], []

    conn = get_connection()
    init_schema(conn)
    try:
        for pick in read_current_picks():
            if pick["status"] != PickStatus.PUBLISHED.value:
                continue  # already settled or voided - never re-graded

            game = conn.execute(
                "SELECT home_score, away_score, game_status FROM games WHERE game_id = ?",
                (pick["game_id"],),
            ).fetchone()
            if game is None or game["game_status"] != "final" or game["home_score"] is None or game["away_score"] is None:
                still_pending.append(pick["pick_id"])
                continue

            home_score, away_score = game["home_score"], game["away_score"]
            actual_home_margin = home_score - away_score
            actual_home_win = None if home_score == away_score else home_score > away_score

            try:
                result = compute_settlement(
                    market_type=pick["market_type"], selection=pick["selection"], line=pick["line"],
                    actual_home_margin=actual_home_margin, actual_home_win=actual_home_win,
                )
            except UnsupportedMarketTypeError as e:
                logger.warning("Skipping settlement for pick_id=%s: %s", pick["pick_id"], e)
                skipped_unsupported.append(pick["pick_id"])
                continue

            record_settlement(
                pick_id=pick["pick_id"], settlement=result, settled_at=now_iso,
                result_source=f"nflverse real final score: home {home_score} - away {away_score} (game_id={pick['game_id']})",
            )
            settled.append({"pick_id": pick["pick_id"], "category": pick["category"], "result": result.value})
    finally:
        conn.close()

    logger.info(
        "Settlement pass: %d settled, %d still pending, %d skipped (unsupported market type)",
        len(settled), len(still_pending), len(skipped_unsupported),
    )
    return {"settled": settled, "still_pending": still_pending, "skipped_unsupported": skipped_unsupported}
