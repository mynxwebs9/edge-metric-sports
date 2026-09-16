// Shared, realistically-shaped fixtures matching the real FastAPI response shapes exactly -
// never invented fields the backend doesn't actually produce.
import type { GameCard, GameDetail, PickOut } from "@/lib/types";

export function makePick(overrides: Partial<PickOut> = {}): PickOut {
  return {
    pick_id: "pick_2026_01_DEN_KC_spread_2026-09-14T04:43:04.421709+00:00",
    game_id: "2026_01_DEN_KC",
    published_at: "2026-09-14T04:54:49.791940+00:00",
    kickoff_at: "2026-09-14T20:15:00",
    market_type: "spread",
    selection: "away",
    selection_team: { team_id: "1400", abbr: "DEN", name: "Denver Broncos", nickname: "Broncos" },
    home_team: { team_id: "2310", abbr: "KC", name: "Kansas City Chiefs", nickname: "Chiefs" },
    away_team: { team_id: "1400", abbr: "DEN", name: "Denver Broncos", nickname: "Broncos" },
    line: -2.0, price: -110,
    sportsbook_or_source: "Consensus of 9 sportsbooks (median_spread_mean_no_vig_v1): draftkings, fanduel",
    status: "PUBLISHED",
    ...overrides,
  };
}

export function makeGameCard(overrides: Partial<GameCard> = {}): GameCard {
  return {
    game_id: "2026_01_DEN_KC",
    season: 2026,
    week: 1,
    away_team: { team_id: "1400", abbr: "DEN", name: "Denver Broncos", nickname: "Broncos" },
    home_team: { team_id: "2310", abbr: "KC", name: "Kansas City Chiefs", nickname: "Chiefs" },
    kickoff_timestamp: "2026-09-14T20:15:00",
    game_status: "scheduled",
    model: {
      available: true, predicted_winner: "away", elo_predicted_margin: -3.2, elo_home_win_probability: 0.39,
      ridge_predicted_margin: -3.4, lightgbm_predicted_margin: -0.9, all_agree_on_direction: true,
      margin_dispersion: 2.5, generated_at: "2026-09-12T08:00:00+00:00",
    },
    market: {
      available: true, home_spread_traditional: -2.5, home_moneyline: -142, away_moneyline: 120,
      no_vig_home_win_probability: 0.56, snapshot_timestamp: "2026-09-12T20:08:00+00:00",
      consensus_algorithm_version: "median_spread_mean_no_vig_v1", sportsbooks: ["draftkings", "fanduel"],
    },
    decision: {
      available: true, market_type: "spread", decision: "VETO", decision_label: "Veto",
      reason_codes: [{ code: "VETO_CONSIDERATION_UNRESOLVED", label: "Significant unresolved pregame risk" }],
      decision_timestamp: "2026-09-12T09:00:00+00:00", is_published_best_bet: false,
    },
    system_pick: {
      available: true, selection: "away", selection_team: { team_id: "1400", abbr: "DEN", name: "Denver Broncos", nickname: "Broncos" },
      price: 120, market_type: "moneyline", status: "PUBLISHED", settlement: null, is_also_best_bet: false,
    },
    research: {
      available: true, classification: "VETO_CONSIDERATION", classification_label: "Significant risk flagged",
      materiality_level: 4, summary: "Real, sourced QB injury uncertainty.", research_timestamp: "2026-09-12T11:28:00+00:00",
      unresolved_risks: [], failure_status: null, failure_label: null,
    },
    ...overrides,
  };
}

export function makeGameDetail(overrides: Partial<GameDetail> = {}): GameDetail {
  const card = makeGameCard();
  return {
    schema_version: "1", game_id: card.game_id, season: card.season, week: card.week,
    away_team: card.away_team, home_team: card.home_team, kickoff_timestamp: card.kickoff_timestamp,
    game_status: card.game_status, model: card.model, market: card.market,
    model_market_disagreement_points: -5.7, system_pick: card.system_pick, spread_decision: card.decision,
    moneyline_decision: { available: false, market_type: null, decision: null, decision_label: null, reason_codes: [], decision_timestamp: null, is_published_best_bet: false },
    research: card.research,
    preview: {
      available: true, text: "Denver Broncos at Kansas City Chiefs: the model leans DEN, but the market has KC favored, and real QB-injury uncertainty is flagged as unresolved.",
      generated_at: "2026-09-12T12:00:00+00:00", prompt_version: "prediction_writer_v1", model_provider: "anthropic", model_name: "claude-sonnet-5",
    },
    ...overrides,
  };
}
