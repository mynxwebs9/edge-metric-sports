"""Resolves the schedule's ambiguous `kickoff_time_naive` (nflverse's `gametime` field,
documented by the source as Eastern Time with no UTC offset given - see
`nfl_predict.data.games`) into a genuine, unambiguous UTC timestamp.

This is deliberately NOT applied at ingestion (`data/games.py` stores the raw, ambiguous
value verbatim, exactly as documented) or anywhere in the modeling/feature/backtesting
pipeline, which only ever needs kickoff times relative to EACH OTHER (chronological
ordering) - never against real wall-clock time, so the ambiguity is harmless there and
touching it would ripple into the frozen, extensively-validated model protocol for no
benefit.

It IS required anywhere a kickoff time is compared against real `datetime.now()` (has this
game actually started yet?) or shown to a real person (what time does kickoff happen for
them?) - both need a real, correctly-resolved instant, not an ambiguous local clock reading.
Using `zoneinfo` (not a fixed UTC offset) because the Eastern/UTC gap is 4 hours during EDT
(roughly March-November) and 5 during EST - a fixed offset would be wrong for close to half
the calendar.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_NFLVERSE_KICKOFF_TZ = ZoneInfo("America/New_York")


def resolve_kickoff_to_utc(kickoff_time_naive: str | None) -> str | None:
    """Returns a genuine UTC ISO-8601 string, or None if `kickoff_time_naive` is None
    (kickoff genuinely unknown - never fabricated). Idempotent: an input that already
    carries a UTC offset/timezone is converted to UTC and returned as-is in meaning, so
    calling this on an already-resolved value is always safe."""
    if kickoff_time_naive is None:
        return None
    dt = datetime.fromisoformat(kickoff_time_naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_NFLVERSE_KICKOFF_TZ)
    return dt.astimezone(timezone.utc).isoformat()
