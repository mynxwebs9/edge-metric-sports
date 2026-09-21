"""Expert picks share the model's immutable ledger, settlement pass and record math, but a
human's record is only credible if the rules that keep it honest are enforced in code:
real publish time, nothing after kickoff, nothing editable, nothing blended into another
category. These tests prove each of those, plus the spread-line convention conversion (a
quoted "DEN +3" must be stored as the HOME spread of -3.0, or it would settle backwards)."""

from __future__ import annotations

import pytest

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.decision.expert_picks import ExpertPickError, build_expert_pick, publish_expert_pick
from nfl_predict.decision.pick_ledger import PickCategory, read_current_picks
from nfl_predict.decision.pick_settlement import settle_all_pending_picks
from nfl_predict.decision.records import compute_category_record
from nfl_predict.live.odds_ingestion import fetch_and_snapshot_live_odds
from nfl_predict.market.odds_provider import OddsEvent, OddsMarketSnapshot, OddsProvider

GAME_ID = "2026_02_JAX_DEN"
SEASON, WEEK = 2026, 2
BEFORE_KICKOFF = "2026-09-16T12:00:00+00:00"  # kickoff is 2026-09-20 16:05 ET = 20:05Z
AFTER_KICKOFF = "2026-09-20T21:00:00+00:00"


@pytest.fixture
def seeded_game(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    conn.executemany(
        "INSERT INTO teams (team_id, canonical_abbr, name, nickname) VALUES (?, ?, ?, ?)",
        [("2250", "JAX", "Jacksonville Jaguars", "Jaguars"), ("1400", "DEN", "Denver Broncos", "Broncos")],
    )
    conn.execute(
        "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
        "home_team_id, away_team_id, home_team_abbr, away_team_abbr, game_status, normalized_at) "
        "VALUES (?, ?, 'REG', ?, '2026-09-20', '2026-09-20T16:05:00', '2250', '1400', 'JAX', 'DEN', 'scheduled', '2026-09-16T00:00:00')",
        (GAME_ID, SEASON, WEEK),
    )
    conn.commit()
    conn.close()
    return isolated_data_dir


def _finish_game(home_score: int, away_score: int):
    conn = get_connection()
    conn.execute(
        "UPDATE games SET home_score = ?, away_score = ?, game_status = 'final' WHERE game_id = ?",
        (home_score, away_score, GAME_ID),
    )
    conn.commit()
    conn.close()


def _publish_real_market():
    events = [OddsEvent(provider_event_id="evt_jaxden", game_id=None, home_team="Jacksonville Jaguars", away_team="Denver Broncos", commence_time="2026-09-20T20:05:00Z")]
    snap = OddsMarketSnapshot(
        provider_event_id="evt_jaxden", fetched_at="2026-09-16T09:00:00+00:00", bookmaker="book_a",
        home_spread_traditional=2.5, away_spread_traditional=-2.5, home_spread_price=-112, away_spread_price=-108,
        home_moneyline=130, away_moneyline=-150, total_line=44.5, over_price=-110, under_price=-110,
    )

    class _FakeOdds(OddsProvider):
        def get_events(self):
            return events

        def get_markets_for_sport(self):
            return [snap]

    fetch_and_snapshot_live_odds(_FakeOdds(), season=SEASON)


def test_an_away_team_spread_pick_is_stored_in_the_home_spread_convention(seeded_game):
    """DEN +3 (away, getting 3) means the HOME spread is -3.0 - storing +3 would make the
    settlement grade the pick backwards."""
    pick = publish_expert_pick(GAME_ID, "DEN", "spread", line=3.0, price=-110, now=BEFORE_KICKOFF)
    assert pick.category == PickCategory.EXPERT_PICKS.value
    assert pick.selection == "away"
    assert pick.line == -3.0
    assert pick.price == -110
    assert pick.sportsbook_or_source == "Expert-supplied line and price"
    assert pick.published_at == BEFORE_KICKOFF
    assert pick.kickoff_at == "2026-09-20T20:05:00+00:00"


def test_a_home_team_spread_pick_keeps_the_quoted_line_as_is(seeded_game):
    pick = publish_expert_pick(GAME_ID, "jax", "spread", line=2.5, price=-105, now=BEFORE_KICKOFF)
    assert pick.selection == "home"
    assert pick.line == 2.5


def test_a_moneyline_pick_with_no_price_records_the_real_consensus_price_and_says_so(seeded_game):
    _publish_real_market()
    pick = publish_expert_pick(GAME_ID, "DEN", "moneyline", now=BEFORE_KICKOFF)
    assert pick.selection == "away"
    assert pick.price == -150
    assert pick.line is None
    assert pick.sportsbook_or_source.startswith("Consensus of")
    assert pick.market_snapshot_id is not None


def test_a_spread_pick_with_no_line_or_price_records_the_consensus_market(seeded_game):
    _publish_real_market()
    pick = publish_expert_pick(GAME_ID, "DEN", "spread", now=BEFORE_KICKOFF)
    assert pick.selection == "away"
    assert pick.line == 2.5  # home spread as published
    assert pick.price == -108  # the away side's own spread price
    assert pick.sportsbook_or_source.startswith("Consensus of")


def test_no_market_and_no_supplied_numbers_is_refused_not_guessed(seeded_game):
    with pytest.raises(ExpertPickError, match="No current market"):
        publish_expert_pick(GAME_ID, "DEN", "moneyline", now=BEFORE_KICKOFF)


def test_a_pick_after_kickoff_is_refused(seeded_game):
    with pytest.raises(ExpertPickError, match="kicked off"):
        publish_expert_pick(GAME_ID, "DEN", "moneyline", price=-150, now=AFTER_KICKOFF)
    assert read_current_picks() == []


def test_a_pick_on_an_already_final_game_is_refused(seeded_game):
    _finish_game(24, 17)
    with pytest.raises(ExpertPickError, match="not scheduled"):
        publish_expert_pick(GAME_ID, "DEN", "moneyline", price=-150, now=BEFORE_KICKOFF)


@pytest.mark.parametrize("kwargs,match", [
    ({"game_id": "2026_02_NOPE_NOPE", "team": "DEN", "market_type": "moneyline", "price": -150}, "not in the 2026 schedule"),
    ({"game_id": GAME_ID, "team": "KC", "market_type": "moneyline", "price": -150}, "not in"),
    ({"game_id": GAME_ID, "team": "DEN", "market_type": "total", "price": -110}, "not supported"),
    ({"game_id": GAME_ID, "team": "DEN", "market_type": "moneyline", "price": -150, "line": 3.0}, "no line"),
    ({"game_id": GAME_ID, "team": "DEN", "market_type": "moneyline", "price": -50}, "not valid American odds"),
    ({"game_id": GAME_ID, "team": "DEN", "market_type": "moneyline", "price": -150, "note": "x" * 601}, "keep it under"),
])
def test_bad_input_is_refused_with_a_clear_reason(seeded_game, kwargs, match):
    with pytest.raises(ExpertPickError, match=match):
        publish_expert_pick(now=BEFORE_KICKOFF, **kwargs)
    assert read_current_picks() == []


def test_a_published_pick_can_never_be_replaced(seeded_game):
    publish_expert_pick(GAME_ID, "DEN", "moneyline", price=-150, now=BEFORE_KICKOFF)
    with pytest.raises(ExpertPickError, match="already published"):
        publish_expert_pick(GAME_ID, "JAX", "moneyline", price=130, now=BEFORE_KICKOFF)  # the opposite side, later
    picks = read_current_picks()
    assert len(picks) == 1
    assert picks[0]["selection"] == "away"


def test_spread_and_moneyline_are_separate_picks_on_the_same_game(seeded_game):
    publish_expert_pick(GAME_ID, "DEN", "spread", line=3.0, price=-110, now=BEFORE_KICKOFF)
    publish_expert_pick(GAME_ID, "DEN", "moneyline", price=-150, now=BEFORE_KICKOFF)
    assert {p["pick_id"] for p in read_current_picks()} == {f"{GAME_ID}_expert_spread", f"{GAME_ID}_expert_moneyline"}


def test_dry_run_build_writes_nothing(seeded_game):
    pick = build_expert_pick(GAME_ID, "DEN", "moneyline", price=-150, now=BEFORE_KICKOFF)
    assert pick.pick_id == f"{GAME_ID}_expert_moneyline"
    assert read_current_picks() == []


def test_the_note_is_stored_with_the_pick(seeded_game):
    publish_expert_pick(GAME_ID, "DEN", "moneyline", price=-150, note="  Broncos front seven travels well.  ", now=BEFORE_KICKOFF)
    assert read_current_picks()[0]["note"] == "Broncos front seven travels well."


def test_an_expert_spread_pick_settles_through_the_shared_pass_and_stays_its_own_record(seeded_game):
    """DEN +3, JAX wins by 1 -> DEN covers. Goes through the exact same
    settle_all_pending_picks() the model's picks use, and lands ONLY in EXPERT_PICKS."""
    publish_expert_pick(GAME_ID, "DEN", "spread", line=3.0, price=-110, now=BEFORE_KICKOFF)
    _finish_game(home_score=20, away_score=19)

    result = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    assert result["settled"] == [{"pick_id": f"{GAME_ID}_expert_spread", "category": "EXPERT_PICKS", "result": "WIN"}]

    picks = read_current_picks()
    expert = compute_category_record(picks, "EXPERT_PICKS")
    assert (expert.n_settled, expert.wins, expert.losses) == (1, 1, 0)
    assert expert.total_units == pytest.approx(100 / 110)
    assert compute_category_record(picks, "BEST_BETS").n_settled == 0
    assert compute_category_record(picks, "ALL_MODEL_PREDICTIONS").n_settled == 0


def test_an_expert_spread_pick_that_does_not_cover_settles_as_a_loss(seeded_game):
    publish_expert_pick(GAME_ID, "DEN", "spread", line=3.0, price=-110, now=BEFORE_KICKOFF)
    _finish_game(home_score=24, away_score=17)  # JAX by 7, DEN +3 loses

    result = settle_all_pending_picks(now="2026-09-21T00:00:00+00:00")
    assert result["settled"][0]["result"] == "LOSS"
