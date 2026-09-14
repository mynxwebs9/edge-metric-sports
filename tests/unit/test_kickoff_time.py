"""Real production bug (found via a user directly checking a live game's actual kickoff
time against what the site showed): nflverse's `gametime` is Eastern Time with no offset
given, but several consumers treated the naive combined string as if it were already UTC -
a 4-5 hour error that, among other things, made the system think a game had already kicked
off hours before it actually had."""

from __future__ import annotations

from nfl_predict.kickoff_time import resolve_kickoff_to_utc


def test_none_stays_none():
    assert resolve_kickoff_to_utc(None) is None


def test_edt_kickoff_resolves_to_the_real_utc_instant():
    # 2026-09-14 is in EDT (UTC-4). 8:15 PM Eastern is genuinely 00:15 UTC the next day -
    # this is exactly the DEN@KC case: the site was showing "8:15 PM UTC" (wrong) instead
    # of the real 00:15 UTC / 5:15 PM Pacific kickoff.
    assert resolve_kickoff_to_utc("2026-09-14T20:15:00") == "2026-09-15T00:15:00+00:00"


def test_est_kickoff_resolves_with_the_five_hour_winter_offset():
    # 2026-01-04 is in EST (UTC-5), not EDT - a fixed 4-hour offset would be wrong here.
    assert resolve_kickoff_to_utc("2026-01-04T13:00:00") == "2026-01-04T18:00:00+00:00"


def test_an_already_tz_aware_input_is_converted_to_utc_not_double_shifted():
    assert resolve_kickoff_to_utc("2026-09-14T00:15:00+00:00") == "2026-09-14T00:15:00+00:00"
