"""Phase 6 Step 1/19: blocks ordinary current-web research from being used to fabricate
supposedly-pregame historical research records.

A research run's *claimed* point-in-time is `research_timestamp` (what moment the research
represents - "as of Thursday before kickoff"). Separately, `now` is the actual wall-clock
time this code is running. Two distinct risks are guarded against:

1. **A record cannot claim to be "pregame" if its own claimed timestamp is after kickoff** -
   that's not pregame research by definition, authorized sources or not.
2. **If the game's real kickoff has already passed relative to `now`**, ordinary current-web
   search/tools would surface postgame information (final scores, updated injury histories,
   retrospective analysis, revised depth charts) even when asked a pregame-framed question.
   This is blocked UNLESS the caller supplies proof (`ArchivedSourceAuthorization`) that every
   source used is independently verified to have existed, pre-kickoff-contamination-free,
   before the relevant timestamp. This guard never fabricates or auto-generates that proof -
   it only refuses to proceed without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from nfl_predict.kickoff_time import resolve_kickoff_to_utc


class HistoricalResearchLeakageRisk(Exception):
    """Raised when a research run would use ordinary current-web search on a game whose
    kickoff has already passed, without explicit archived-source authorization."""


class InvalidPregameTimestampError(Exception):
    """Raised when a research record's own claimed timestamp is after the kickoff it
    claims to precede - such a record cannot be labeled pregame research at all."""


@dataclass(frozen=True)
class ArchivedSourceAuthorization:
    """Proof that a specific historical research run's sources are safe to use. Never
    constructed automatically by this codebase - a human (or an explicitly separate,
    independently-audited process) must supply real, checkable evidence for every source."""

    source_urls: tuple[str, ...]
    proof_of_pre_kickoff_existence: str  # e.g. "Wayback Machine snapshot dated <ts>, content verified pre-kickoff and free of postgame references"
    authorized_by: str
    authorized_at: str


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def assert_research_may_proceed(
    kickoff_timestamp: str | None,
    research_timestamp: str,
    now: str | None = None,
    archived_source_authorization: ArchivedSourceAuthorization | None = None,
) -> None:
    """Raises `InvalidPregameTimestampError` or `HistoricalResearchLeakageRisk` if this
    research run is not safe to proceed as pregame research. Returns None (no exception) if
    it may proceed. `kickoff_timestamp=None` (kickoff genuinely unknown) skips the
    kickoff-relative checks - the caller's other validation still applies."""
    now_dt = _parse(now) if now else datetime.now(timezone.utc)
    research_dt = _parse(research_timestamp)

    if research_dt > now_dt:
        raise InvalidPregameTimestampError(
            f"research_timestamp {research_timestamp} is in the future relative to now "
            f"({now_dt.isoformat()}) - a research record cannot represent a moment that "
            "has not happened yet."
        )

    if kickoff_timestamp is None:
        return

    # kickoff_timestamp comes straight from the schedule's `kickoff_time_naive` column,
    # which nflverse documents as Eastern Time with no offset given (see
    # nfl_predict.data.games) - resolve it to a real UTC instant before comparing against
    # `now_dt`/`research_dt`, which already are genuine UTC. Treating the naive Eastern
    # string as if it were already UTC (the bug this replaced) made the system think a game
    # had kicked off up to 5 hours before it actually had.
    kickoff_dt = _parse(resolve_kickoff_to_utc(kickoff_timestamp))

    if research_dt > kickoff_dt:
        raise InvalidPregameTimestampError(
            f"research_timestamp {research_timestamp} is after kickoff {kickoff_timestamp} - "
            "this cannot be labeled pregame research regardless of source authorization."
        )

    game_already_happened = kickoff_dt <= now_dt
    if game_already_happened and archived_source_authorization is None:
        raise HistoricalResearchLeakageRisk(
            f"Kickoff ({kickoff_timestamp}) has already passed relative to now "
            f"({now_dt.isoformat()}) - ordinary current-web research would surface postgame "
            "information. Supply an ArchivedSourceAuthorization proving every source existed, "
            "pre-kickoff-contamination-free, before proceeding. This has NOT been done "
            "automatically and must never be fabricated."
        )
