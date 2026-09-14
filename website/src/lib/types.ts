// Mirrors src/nfl_predict/api/schemas.py exactly - one Python schema, one TS type each.
// Never redefine a field's meaning here; if the API response shape changes, this file
// changes to match it, never the other way around.

export interface ReasonCodeOut {
  code: string;
  label: string;
}

export interface TeamOut {
  team_id: string;
  abbr: string;
  name: string;
  nickname: string | null;
}

export interface ModelBlock {
  available: boolean;
  predicted_winner: string | null;
  elo_predicted_margin: number | null;
  elo_home_win_probability: number | null;
  ridge_predicted_margin: number | null;
  lightgbm_predicted_margin: number | null;
  all_agree_on_direction: boolean | null;
  margin_dispersion: number | null;
  generated_at: string | null;
}

export interface MarketBlock {
  available: boolean;
  home_spread_traditional: number | null;
  home_moneyline: number | null;
  away_moneyline: number | null;
  no_vig_home_win_probability: number | null;
  snapshot_timestamp: string | null;
  consensus_algorithm_version: string | null;
  sportsbooks: string[];
}

export interface ResearchBlock {
  available: boolean;
  classification: string | null;
  classification_label: string | null;
  materiality_level: number | null;
  summary: string | null;
  research_timestamp: string | null;
  unresolved_risks: string[];
  failure_status: string | null;
  failure_label: string | null;
}

export interface DecisionBlock {
  available: boolean;
  market_type: string | null;
  decision: string | null;
  decision_label: string | null;
  reason_codes: ReasonCodeOut[];
  decision_timestamp: string | null;
  is_published_best_bet: boolean;
}

export interface GameCard {
  game_id: string;
  season: number;
  week: number;
  away_team: TeamOut;
  home_team: TeamOut;
  kickoff_timestamp: string | null;
  game_status: string;
  model: ModelBlock;
  market: MarketBlock;
  decision: DecisionBlock;
  research: ResearchBlock;
}

export interface SlateResponse {
  schema_version: string;
  season: number;
  week: number | null;
  last_updated: string | null;
  games: GameCard[];
}

export interface ScheduleWeeksResponse {
  schema_version: string;
  season: number;
  weeks: number[];
  current_week: number | null;
}

export interface PreviewBlock {
  available: boolean;
  text: string | null;
  generated_at: string | null;
  prompt_version: string | null;
  model_provider: string | null;
  model_name: string | null;
}

export interface GameDetail {
  schema_version: string;
  game_id: string;
  season: number;
  week: number;
  away_team: TeamOut;
  home_team: TeamOut;
  kickoff_timestamp: string | null;
  game_status: string;
  model: ModelBlock;
  market: MarketBlock;
  model_market_disagreement_points: number | null;
  spread_decision: DecisionBlock;
  moneyline_decision: DecisionBlock;
  research: ResearchBlock;
  preview: PreviewBlock;
}

export interface PickOut {
  pick_id: string;
  game_id: string;
  published_at: string;
  kickoff_at: string | null;
  market_type: string;
  selection: string; // "home" | "away" - internal settlement convention, use selection_team for display
  selection_team: TeamOut | null;
  home_team: TeamOut | null;
  away_team: TeamOut | null;
  line: number | null;
  price: number;
  sportsbook_or_source: string;
  status: string;
}

export interface BestBetsResponse {
  schema_version: string;
  picks: PickOut[];
  message: string | null;
}

export interface CategoryRecordOut {
  category: string;
  n_settled: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  total_units: number | null;
  roi_per_bet: number | null;
  average_price: number | null;
}

export interface StreakOut {
  window: string;
  n: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  total_units: number | null;
  headline: string | null;
}

export interface PerformanceResponse {
  schema_version: string;
  best_bets_record: CategoryRecordOut;
  all_model_predictions_record: CategoryRecordOut;
  streaks: StreakOut[];
  headline: string | null;
  has_settled_history: boolean;
}

export interface ModelStatusResponse {
  schema_version: string;
  rule_set_version: string;
  model_ids: string[];
  odds_provider_status: string;
  research_provider_status: string;
  last_model_prediction_at: string | null;
  last_research_run_at: string | null;
  last_decision_at: string | null;
}

export interface DecisionDetail {
  available: boolean;
  market_type: string | null;
  decision: string | null;
  decision_label: string | null;
  reason_codes: ReasonCodeOut[];
  decision_timestamp: string | null;
  decision_rule_version: string | null;
  market: MarketBlock | null;
  research: ResearchBlock | null;
  is_published_best_bet: boolean;
}

export interface DecisionsResponse {
  schema_version: string;
  game_id: string;
  spread: DecisionDetail;
  moneyline: DecisionDetail;
}
