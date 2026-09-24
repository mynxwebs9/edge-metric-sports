"""Generates the public game-preview article for every upcoming game in a week - the
repeatable command for what `write_prediction_preview()` does for one game.

    python -m nfl_predict.content.generate_previews --week 3 [--game 2026_03_ATL_GB] [--force]

Each game is a real, billed Anthropic call (cheap: no tools, ~2k max output tokens).

Run it AFTER the week's decisions are recorded and its Best Bets are published, never
before: the article states each market's real publication status, so an article written
too early can describe a game as "no bet" that later gets published as one.

By default a game that already has a successful preview is skipped, so re-running after a
partial failure never double-bills - and never lets a failed retry become the "latest" run
that hides a good article on the website. `--force` regenerates anyway.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from nfl_predict.api import reconstruction as recon
from nfl_predict.content.prediction_writer import (
    DEFAULT_PROMPT_PATH,
    AnthropicContentWriterProvider,
    ContentWriterProvider,
    write_prediction_preview,
)
from nfl_predict.live.schedule_provider import get_current_season, get_schedule, upcoming_games
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)


def generate_previews_for_week(
    week: int, provider: ContentWriterProvider, prompt_template_text: str, game_id: str | None = None,
    force: bool = False, now: str | None = None,
) -> dict:
    """Returns real counts and total estimated cost: `generated`, `skipped_existing`, `failed`
    (each a list of game_ids) and `estimated_cost_usd`."""
    season = get_current_season()
    games = [g for g in upcoming_games(get_schedule(season, week)) if game_id is None or g.game_id == game_id]
    now_iso = now or datetime.now(timezone.utc).isoformat()
    run_id = now_iso.replace(":", "-")

    generated, skipped_existing, failed, cost = [], [], [], 0.0
    for game in games:
        existing = recon.latest_preview(season, game.week, game.game_id)
        if not force and existing is not None and existing.get("available"):
            skipped_existing.append(game.game_id)
            continue
        result = write_prediction_preview(
            season, game.week, game.game_id, game.home_team_id, game.away_team_id, game.kickoff_timestamp,
            provider, prompt_template_text, run_id, now=now_iso,
        )
        if result["status"] == "ok":
            generated.append(game.game_id)
            cost += (result["cost"] or {}).get("estimated_cost_usd") or 0.0
        else:
            failed.append({"game_id": game.game_id, "failure_status": result["failure_status"]})

    logger.info("Previews for week %s: %d generated, %d already had one, %d failed", week, len(generated), len(skipped_existing), len(failed))
    return {"generated": generated, "skipped_existing": skipped_existing, "failed": failed, "estimated_cost_usd": round(cost, 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nfl_predict.content.generate_previews", description=__doc__.split("\n\n")[0])
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--game", default=None, dest="game_id", help="Only this game_id")
    parser.add_argument("--force", action="store_true", help="Regenerate even if a good preview already exists")
    args = parser.parse_args(argv)

    result = generate_previews_for_week(
        args.week, AnthropicContentWriterProvider(), DEFAULT_PROMPT_PATH.read_text(encoding="utf-8"),
        game_id=args.game_id, force=args.force,
    )
    print(f"generated={len(result['generated'])} skipped_existing={len(result['skipped_existing'])} "
          f"failed={len(result['failed'])} estimated_cost_usd={result['estimated_cost_usd']}")
    for f in result["failed"]:
        print(f"  FAILED {f['game_id']}: {f['failure_status']}", file=sys.stderr)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
