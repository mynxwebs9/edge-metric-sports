"""Phase 7 Step 6: staleness checks - a prediction can be mathematically valid but
operationally stale. Unknown age is always treated as stale, never assumed fresh."""

from __future__ import annotations

from datetime import datetime, timezone


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def compute_age_seconds(timestamp: str | None, now: str) -> float | None:
    if timestamp is None:
        return None
    return (_parse(now) - _parse(timestamp)).total_seconds()


def is_stale(age_seconds: float | None, max_age_hours: float) -> bool:
    if age_seconds is None:
        return True
    return age_seconds > max_age_hours * 3600
