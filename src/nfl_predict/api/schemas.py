"""Phase 8B: stable, versioned JSON response schemas for the read API. Every field here is
either read directly from a persisted artifact or a documented, deterministic translation of
one (see `consumer_language.py`) - nothing is computed fresh. No field ever carries an
internal filesystem path, an API key, or raw LLM prompt text.
"""

from __future__ import annotations

from pydantic import BaseModel

API_SCHEMA_VERSION = "1"


class HealthResponse(BaseModel):
    status: str
    service: str
    schema_version: str


class ReasonCodeOut(BaseModel):
    code: str
    label: str


class TeamOut(BaseModel):
    team_id: str
    abbr: str
    name: str
    nickname: str | None = None


class ModelBlock(BaseModel):
    """`elo`/`ridge`/`lightgbm` are the three frozen model families (`docs/MODEL_SPEC.md`) -
    a `None` value means that model's prediction is genuinely unavailable, never fabricated.
    `all_agree`/`dispersion` are the only real, persisted uncertainty signal this system
    produces (cross-model agreement) - there is no separate confidence interval to show."""

    available: bool
    predicted_winner: str | None = None
    elo_predicted_margin: float | None = None
    elo_home_win_probability: float | None = None
    ridge_predicted_margin: float | None = None
    lightgbm_predicted_margin: float | None = None
    all_agree_on_direction: bool | None = None
    margin_dispersion: float | None = None
    generated_at: str | None = None


class MarketBlock(BaseModel):
    available: bool
    home_spread_traditional: float | None = None
    home_moneyline: int | None = None
    away_moneyline: int | None = None
    no_vig_home_win_probability: float | None = None
    snapshot_timestamp: str | None = None
    consensus_algorithm_version: str | None = None
    sportsbooks: tuple[str, ...] = ()


class ResearchBlock(BaseModel):
    available: bool
    classification: str | None = None
    classification_label: str | None = None
    materiality_level: int | None = None
    summary: str | None = None
    research_timestamp: str | None = None
    unresolved_risks: tuple[str, ...] = ()
    failure_status: str | None = None
    failure_label: str | None = None


class DecisionBlock(BaseModel):
    available: bool
    market_type: str | None = None
    decision: str | None = None
    decision_label: str | None = None
    reason_codes: tuple[ReasonCodeOut, ...] = ()
    decision_timestamp: str | None = None
    # True only when this exact market_type has a real, currently-PUBLISHED entry in the
    # official Best Bets ledger - independent of `decision`, since QUALIFIED_BET and
    # "actually published" are deliberately separate (see reconstruction.published_best_bet_market_types).
    is_published_best_bet: bool = False


class SystemPickBlock(BaseModel):
    """The model's own straight-up pick for this game (`PickCategory.ALL_MODEL_PREDICTIONS`)
    - a real, priced, settleable moneyline pick published for EVERY game with real model/
    market data, entirely independent of whether this game separately qualifies for (and
    gets published as) an actual Best Bet - see `decision` for that distinct, much narrower
    status. `available=False` only means nothing has been published for this game yet
    (e.g. before this week's first real pipeline run), never a fabricated pick."""

    available: bool
    selection: str | None = None  # "home" | "away"
    selection_team: TeamOut | None = None
    price: int | None = None
    market_type: str | None = None
    status: str | None = None  # "PUBLISHED" | "SETTLED"
    settlement: str | None = None  # "WIN" | "LOSS" | "PUSH", or None if not yet settled
    is_also_best_bet: bool = False  # True when THIS market_type separately has a real, published Best Bet for this game - the two are still two distinct ledger rows, never merged


class GameCard(BaseModel):
    """One game as it appears on the current slate / homepage - the moneyline decision is
    shown as the card's headline System decision, matching `system_pick` (always a
    moneyline pick) so the two never visually disagree; the full matchup page shows both
    spread and moneyline explicitly (see GameDetail)."""

    game_id: str
    season: int
    week: int
    away_team: TeamOut
    home_team: TeamOut
    kickoff_timestamp: str | None
    game_status: str
    model: ModelBlock
    market: MarketBlock
    decision: DecisionBlock
    research: ResearchBlock
    system_pick: SystemPickBlock


class SlateResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    season: int
    week: int | None
    last_updated: str | None
    games: tuple[GameCard, ...]


class ScheduleWeeksResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    season: int
    weeks: tuple[int, ...]
    current_week: int | None


class GameIdOut(BaseModel):
    game_id: str
    week: int


class GameIdsResponse(BaseModel):
    """Deliberately cheap: only `schedule.games`'s own fields (already fetched by a single
    `get_schedule()` call) - no per-game model/market/research/decision reconstruction. Built
    for the website's sitemap, which needs every real game's URL but none of its data."""

    schema_version: str = API_SCHEMA_VERSION
    season: int
    game_ids: tuple[GameIdOut, ...]


class PreviewBlock(BaseModel):
    """The offline-generated "why" article for a game - see
    `nfl_predict.content.prediction_writer`. Never generated at request time; `available`
    is `False` both when nothing has been generated yet and when the last attempt failed."""

    available: bool
    text: str | None = None
    generated_at: str | None = None
    prompt_version: str | None = None
    model_provider: str | None = None
    model_name: str | None = None


class GameDetail(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    game_id: str
    season: int
    week: int
    away_team: TeamOut
    home_team: TeamOut
    kickoff_timestamp: str | None
    game_status: str
    model: ModelBlock
    market: MarketBlock
    model_market_disagreement_points: float | None = None
    system_pick: SystemPickBlock
    spread_decision: DecisionBlock
    moneyline_decision: DecisionBlock
    research: ResearchBlock
    preview: PreviewBlock


class PickOut(BaseModel):
    pick_id: str
    game_id: str
    published_at: str
    kickoff_at: str | None
    market_type: str
    selection: str  # "home" | "away" - internal settlement convention, see selection_team for display
    selection_team: TeamOut | None = None
    home_team: TeamOut | None = None
    away_team: TeamOut | None = None
    line: float | None
    price: int
    sportsbook_or_source: str
    status: str
    settlement: str | None = None  # "WIN" | "LOSS" | "PUSH" once settled
    note: str | None = None  # optional short write-up (expert picks)
    void_reason: str | None = None  # set only when status == "VOID"


class BestBetsResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    picks: tuple[PickOut, ...]
    message: str | None = None


class CategoryRecordOut(BaseModel):
    category: str
    n_settled: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None
    total_units: float | None
    roi_per_bet: float | None
    average_price: float | None


class StreakOut(BaseModel):
    window: str
    n: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None
    total_units: float | None
    headline: str | None


class ParlayLegOut(BaseModel):
    description: str
    matchup: str
    price: int | None = None
    result: str | None = None  # "WIN" | "LOSS" | "PUSH" | "NOT_GRADED" once the parlay is settled
    detail: str | None = None  # e.g. "248 passing yards" or the final score


class ParlayOut(BaseModel):
    pick_id: str
    published_at: str
    kickoff_at: str | None
    price: int  # the parlay's own American odds, as the sportsbook offered them
    status: str
    settlement: str | None = None
    note: str | None = None
    sportsbook_or_source: str
    void_reason: str | None = None
    legs: tuple[ParlayLegOut, ...]


class ExpertPicksResponse(BaseModel):
    """A human's own picks and parlays and their own records - separate ledger categories
    from the model's Best Bets and All Model Predictions, never blended into either."""

    schema_version: str = API_SCHEMA_VERSION
    record: CategoryRecordOut
    parlay_record: CategoryRecordOut
    streaks: tuple[StreakOut, ...]
    open_picks: tuple[PickOut, ...]
    settled_picks: tuple[PickOut, ...]
    open_parlays: tuple[ParlayOut, ...]
    settled_parlays: tuple[ParlayOut, ...]
    # Picks/parlays voided before kickoff stay visible - a published pick never vanishes quietly.
    voided_picks: tuple[PickOut, ...] = ()
    voided_parlays: tuple[ParlayOut, ...] = ()


class PerformanceResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    best_bets_record: CategoryRecordOut
    all_model_predictions_record: CategoryRecordOut
    streaks: tuple[StreakOut, ...]
    headline: str | None = None
    has_settled_history: bool


class ModelStatusResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    rule_set_version: str
    model_ids: tuple[str, ...]
    odds_provider_status: str
    research_provider_status: str
    last_model_prediction_at: str | None
    last_research_run_at: str | None
    last_decision_at: str | None


class DecisionDetail(BaseModel):
    available: bool
    market_type: str | None = None
    decision: str | None = None
    decision_label: str | None = None
    reason_codes: tuple[ReasonCodeOut, ...] = ()
    decision_timestamp: str | None = None
    decision_rule_version: str | None = None
    market: MarketBlock | None = None
    research: ResearchBlock | None = None
    is_published_best_bet: bool = False


class DecisionsResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    game_id: str
    spread: DecisionDetail
    moneyline: DecisionDetail
