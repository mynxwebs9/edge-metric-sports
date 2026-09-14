"""Phase 6 Step 24 proofs #1-2, #18: historical leakage is blocked, research cannot be
created after kickoff and labeled pregame, research timestamps precede kickoff."""

from __future__ import annotations

import pytest

from nfl_predict.research.historical_guard import (
    ArchivedSourceAuthorization,
    HistoricalResearchLeakageRisk,
    InvalidPregameTimestampError,
    assert_research_may_proceed,
)

NOW = "2026-09-11T12:00:00+00:00"


def test_genuinely_prospective_research_is_allowed_without_authorization():
    assert_research_may_proceed(
        kickoff_timestamp="2026-09-14T17:00:00+00:00",
        research_timestamp="2026-09-11T10:00:00+00:00",
        now=NOW,
    )  # does not raise


def test_a_past_kickoff_without_authorization_raises_leakage_risk():
    with pytest.raises(HistoricalResearchLeakageRisk):
        assert_research_may_proceed(
            kickoff_timestamp="2024-09-08T17:00:00+00:00",
            research_timestamp="2024-09-05T10:00:00+00:00",
            now=NOW,
        )


def test_a_past_kickoff_with_authorization_is_allowed():
    auth = ArchivedSourceAuthorization(
        source_urls=("https://web.archive.org/web/20240905000000/https://example.com/injury-report",),
        proof_of_pre_kickoff_existence="Wayback Machine capture dated 2024-09-05, verified pre-kickoff, no postgame references present",
        authorized_by="test-suite",
        authorized_at=NOW,
    )
    assert_research_may_proceed(
        kickoff_timestamp="2024-09-08T17:00:00+00:00",
        research_timestamp="2024-09-05T10:00:00+00:00",
        now=NOW,
        archived_source_authorization=auth,
    )  # does not raise


def test_a_research_timestamp_after_kickoff_is_rejected_even_with_authorization():
    auth = ArchivedSourceAuthorization(
        source_urls=("https://example.com",), proof_of_pre_kickoff_existence="n/a",
        authorized_by="test-suite", authorized_at=NOW,
    )
    with pytest.raises(InvalidPregameTimestampError):
        assert_research_may_proceed(
            kickoff_timestamp="2024-09-08T17:00:00+00:00",
            research_timestamp="2024-09-09T10:00:00+00:00",  # after kickoff
            now=NOW,
            archived_source_authorization=auth,
        )


def test_a_research_timestamp_in_the_future_relative_to_now_is_rejected():
    with pytest.raises(InvalidPregameTimestampError):
        assert_research_may_proceed(
            kickoff_timestamp="2026-09-20T17:00:00+00:00",
            research_timestamp="2026-09-15T10:00:00+00:00",  # after "now"
            now=NOW,
        )


def test_unknown_kickoff_skips_the_kickoff_relative_checks():
    # Cannot judge historicity without a kickoff, but the future-relative-to-now check
    # still applies.
    assert_research_may_proceed(kickoff_timestamp=None, research_timestamp="2026-09-10T10:00:00+00:00", now=NOW)
    with pytest.raises(InvalidPregameTimestampError):
        assert_research_may_proceed(kickoff_timestamp=None, research_timestamp="2026-09-20T10:00:00+00:00", now=NOW)


def test_kickoff_exactly_at_research_timestamp_is_allowed_boundary_case():
    # research_timestamp == kickoff_timestamp == now: isolates the "is research_timestamp
    # after kickoff" boundary (it is not - equal is allowed) from the future-relative-to-now
    # check, by supplying authorization for the separate already-happened branch.
    auth = ArchivedSourceAuthorization(
        source_urls=("https://example.com",), proof_of_pre_kickoff_existence="n/a",
        authorized_by="test-suite", authorized_at=NOW,
    )
    assert_research_may_proceed(
        kickoff_timestamp=NOW, research_timestamp=NOW, now=NOW, archived_source_authorization=auth,
    )  # research AT kickoff instant - boundary, still "pregame" (not after)


def test_kickoff_exactly_at_now_without_authorization_requires_authorization():
    # Kickoff == now: the game is JUST starting/has just started - treated as "already
    # happened" (<=), so ordinary web research is not safe without authorization.
    with pytest.raises(HistoricalResearchLeakageRisk):
        assert_research_may_proceed(kickoff_timestamp=NOW, research_timestamp="2026-09-10T10:00:00+00:00", now=NOW)
