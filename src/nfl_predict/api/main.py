"""Phase 8B: the FastAPI read API - the ONLY channel the Next.js website is allowed to use
to reach prediction/market/research/decision data (`docs/ARCHITECTURE.md#website-api-boundary`).
Read-only: no endpoint here accepts a write. Every response is built from already-persisted
artifacts via `reconstruction.py` - nothing here computes a model prediction, calls Anthropic,
or calls The Odds API.

Run locally: `uvicorn nfl_predict.api.main:app --reload --port 8000` (see README.md).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nfl_predict.api import consumer_language as lang
from nfl_predict.api import reconstruction as recon
from nfl_predict.api.schemas import (
    API_SCHEMA_VERSION,
    BestBetsResponse,
    CategoryRecordOut,
    DecisionBlock,
    DecisionDetail,
    DecisionsResponse,
    GameCard,
    GameDetail,
    ExpertPicksResponse,
    GameIdOut,
    GameIdsResponse,
    HealthResponse,
    MarketBlock,
    ModelBlock,
    ModelStatusResponse,
    ParlayLegOut,
    ParlayOut,
    PerformanceResponse,
    PickOut,
    PreviewBlock,
    ReasonCodeOut,
    ResearchBlock,
    ScheduleWeeksResponse,
    SlateResponse,
    StreakOut,
    SystemPickBlock,
    TeamOut,
)
from nfl_predict.decision.records import compute_category_record
from nfl_predict.decision.schemas import MarketPoint
from nfl_predict.decision.streaks import compute_all_predefined_windows
from nfl_predict.live.odds_automation import get_production_odds_provider
from nfl_predict.live.research_automation import get_production_research_provider

app = FastAPI(title="NFL Predict API", version=API_SCHEMA_VERSION)

app.add_middleware(
    CORSMiddleware,
    # Local dev only (see module docstring - Phase 8B does not deploy). Port 3000 is the
    # Next.js default; 3100 is used locally on this machine because the OS reserves 3000.
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3100", "http://127.0.0.1:3100"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _team_out(team_id: str | None, teams: dict[str, dict], fallback_abbr: str) -> TeamOut:
    info = teams.get(team_id or "")
    if info is None:
        return TeamOut(team_id=team_id or "unknown", abbr=fallback_abbr, name=fallback_abbr, nickname=None)
    return TeamOut(team_id=team_id, abbr=info["abbr"], name=info["name"], nickname=info["nickname"])


def _reason_codes_out(codes: list[str]) -> tuple[ReasonCodeOut, ...]:
    return tuple(ReasonCodeOut(code=c, label=lang.translate_reason_code(c)) for c in codes)


def _model_block(prediction: dict | None) -> ModelBlock:
    if prediction is None:
        return ModelBlock(available=False)
    return ModelBlock(
        available=True, predicted_winner=prediction.get("predicted_winner"),
        elo_predicted_margin=prediction.get("elo_predicted_margin"),
        elo_home_win_probability=prediction.get("elo_home_win_probability"),
        ridge_predicted_margin=prediction.get("ridge_predicted_margin"),
        lightgbm_predicted_margin=prediction.get("lightgbm_predicted_margin"),
        all_agree_on_direction=prediction.get("model_agreement_all_agree"),
        margin_dispersion=prediction.get("model_agreement_dispersion"),
        generated_at=prediction.get("generated_at"),
    )


def _market_block(market: MarketPoint | None) -> MarketBlock:
    if market is None or not market.available:
        return MarketBlock(available=False)
    return MarketBlock(
        available=True, home_spread_traditional=market.home_spread_traditional,
        home_moneyline=market.home_moneyline, away_moneyline=market.away_moneyline,
        no_vig_home_win_probability=market.no_vig_home_win_probability,
        snapshot_timestamp=market.snapshot_timestamp,
        consensus_algorithm_version=market.consensus_algorithm_version,
        sportsbooks=market.consensus_book_keys,
    )


def _research_block(research: dict | None) -> ResearchBlock:
    if research is None:
        return ResearchBlock(available=False)
    if not research.get("available"):
        failure_status = research.get("failure_status")
        return ResearchBlock(
            available=False, failure_status=failure_status,
            failure_label=lang.translate_research_failure_status(failure_status) if failure_status else None,
            research_timestamp=research.get("research_timestamp"),
        )
    classification = research["classification"]
    return ResearchBlock(
        available=True, classification=classification,
        classification_label=lang.translate_research_classification(classification),
        materiality_level=research.get("materiality_level"), summary=research.get("summary"),
        research_timestamp=research.get("research_timestamp"),
        unresolved_risks=tuple(research.get("unresolved_risks", ())),
    )


def _decision_block(record: dict | None, published_market_types: set[str] = frozenset()) -> DecisionBlock:
    if record is None:
        return DecisionBlock(available=False)
    return DecisionBlock(
        available=True, market_type=record["market_type"], decision=record["decision"],
        decision_label=lang.translate_decision(record["decision"]),
        reason_codes=_reason_codes_out(record["reason_codes"]),
        decision_timestamp=record["decision_timestamp"],
        is_published_best_bet=record["market_type"] in published_market_types,
    )


def _system_pick_block(pick: dict | None, home_team: TeamOut, away_team: TeamOut, published_market_types: set[str] = frozenset()) -> SystemPickBlock:
    if pick is None:
        return SystemPickBlock(available=False)
    selection_team = home_team if pick["selection"] == "home" else away_team
    return SystemPickBlock(
        available=True, selection=pick["selection"], selection_team=selection_team,
        price=pick["price"], market_type=pick["market_type"], status=pick["status"],
        settlement=pick.get("settlement"), is_also_best_bet=pick["market_type"] in published_market_types,
    )


def _preview_block(preview: dict | None) -> PreviewBlock:
    if preview is None or not preview.get("available"):
        return PreviewBlock(available=False)
    return PreviewBlock(
        available=True, text=preview["text"], generated_at=preview["generated_at"],
        prompt_version=preview["prompt_version"], model_provider=preview["model_provider"], model_name=preview["model_name"],
    )


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="nfl_predict-api", schema_version=API_SCHEMA_VERSION)


def _build_slate_for_week(season: int, week: int | None, schedule) -> SlateResponse:
    games = [g for g in schedule.games if g.week == week] if week is not None else []
    teams = recon.team_lookup()

    cards = []
    for g in games:
        model = recon.latest_model_prediction(g.game_id)
        market = recon.latest_market_point(season, g.week, g.game_id)
        decisions = recon.latest_decisions_for_game(season, g.week, g.game_id)
        research = recon.latest_research_summary(season, g.week, g.game_id)
        published_market_types = recon.published_best_bet_market_types(g.game_id)
        system_pick = recon.system_pick_for_game(g.game_id)
        away_team = _team_out(g.away_team_id, teams, g.away_team_abbr)
        home_team = _team_out(g.home_team_id, teams, g.home_team_abbr)
        cards.append(GameCard(
            game_id=g.game_id, season=g.season, week=g.week,
            away_team=away_team, home_team=home_team,
            kickoff_timestamp=g.kickoff_timestamp, game_status=g.game_status,
            model=_model_block(model), market=_market_block(market),
            decision=_decision_block(decisions["moneyline"], published_market_types), research=_research_block(research),
            system_pick=_system_pick_block(system_pick, home_team, away_team, published_market_types),
        ))

    return SlateResponse(season=season, week=week, last_updated=_now_iso(), games=tuple(cards))


@app.get("/api/nfl/slate/current", response_model=SlateResponse)
def current_slate() -> SlateResponse:
    season, schedule = recon.current_season_and_schedule()
    return _build_slate_for_week(season, schedule.current_week, schedule)


@app.get("/api/nfl/schedule", response_model=ScheduleWeeksResponse)
def schedule_weeks() -> ScheduleWeeksResponse:
    season, schedule = recon.current_season_and_schedule()
    weeks = sorted({g.week for g in schedule.games})
    return ScheduleWeeksResponse(season=season, weeks=tuple(weeks), current_week=schedule.current_week)


@app.get("/api/nfl/game-ids", response_model=GameIdsResponse)
def game_ids() -> GameIdsResponse:
    season, schedule = recon.current_season_and_schedule()
    ids = tuple(GameIdOut(game_id=g.game_id, week=g.week) for g in schedule.games)
    return GameIdsResponse(season=season, game_ids=ids)


@app.get("/api/nfl/schedule/{week}", response_model=SlateResponse)
def schedule_for_week(week: int) -> SlateResponse:
    season, schedule = recon.current_season_and_schedule()
    if week not in {g.week for g in schedule.games}:
        raise HTTPException(status_code=404, detail=f"No week {week} in the current season's schedule.")
    return _build_slate_for_week(season, week, schedule)


@app.get("/api/nfl/games/{game_id}", response_model=GameDetail)
def game_detail(game_id: str) -> GameDetail:
    season, schedule = recon.current_season_and_schedule()
    game = next((g for g in schedule.games if g.game_id == game_id), None)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found in the current season's schedule.")

    teams = recon.team_lookup()
    model = recon.latest_model_prediction(game_id)
    market = recon.latest_market_point(season, game.week, game_id)
    decisions = recon.latest_decisions_for_game(season, game.week, game_id)
    research = recon.latest_research_summary(season, game.week, game_id)
    preview = recon.latest_preview(season, game.week, game_id)
    published_market_types = recon.published_best_bet_market_types(game_id)
    system_pick = recon.system_pick_for_game(game_id)

    disagreement = None
    if model is not None and model.get("elo_predicted_margin") is not None and market is not None and market.available and market.home_spread_traditional is not None:
        # Same convention `live/run.py`'s trigger-priority calculation uses - a display-only
        # comparison of two already-persisted numbers, never a new model output.
        disagreement = model["elo_predicted_margin"] - (-market.home_spread_traditional)

    away_team = _team_out(game.away_team_id, teams, game.away_team_abbr)
    home_team = _team_out(game.home_team_id, teams, game.home_team_abbr)

    return GameDetail(
        game_id=game_id, season=game.season, week=game.week,
        away_team=away_team, home_team=home_team,
        kickoff_timestamp=game.kickoff_timestamp, game_status=game.game_status,
        model=_model_block(model), market=_market_block(market),
        model_market_disagreement_points=disagreement,
        system_pick=_system_pick_block(system_pick, home_team, away_team, published_market_types),
        spread_decision=_decision_block(decisions["spread"], published_market_types),
        moneyline_decision=_decision_block(decisions["moneyline"], published_market_types),
        research=_research_block(research),
        preview=_preview_block(preview),
    )


@app.get("/api/nfl/best-bets", response_model=BestBetsResponse)
def best_bets() -> BestBetsResponse:
    picks = recon.current_best_bets()
    if not picks:
        return BestBetsResponse(picks=(), message="No plays currently meet our qualification criteria.")

    _, schedule = recon.current_season_and_schedule()
    games_by_id = {g.game_id: g for g in schedule.games}
    teams = recon.team_lookup()
    return BestBetsResponse(picks=tuple(_pick_out(p, games_by_id, teams) for p in picks))


def _pick_out(p: dict, games_by_id: dict, teams: dict) -> PickOut:
    game = games_by_id.get(p["game_id"])
    home_team = _team_out(game.home_team_id, teams, game.home_team_abbr) if game else None
    away_team = _team_out(game.away_team_id, teams, game.away_team_abbr) if game else None
    selection_team = home_team if p["selection"] == "home" else away_team
    return PickOut(
        pick_id=p["pick_id"], game_id=p["game_id"], published_at=p["published_at"],
        kickoff_at=p["kickoff_at"], market_type=p["market_type"], selection=p["selection"],
        selection_team=selection_team, home_team=home_team, away_team=away_team,
        line=p["line"], price=p["price"], sportsbook_or_source=p["sportsbook_or_source"], status=p["status"],
        settlement=p.get("settlement"), note=p.get("note"), void_reason=p.get("void_reason"),
    )


def _parlay_out(p: dict) -> ParlayOut:
    leg_results = p.get("leg_results") or []
    legs = tuple(
        ParlayLegOut(
            description=leg["description"], matchup=leg["matchup"], price=leg.get("price"),
            result=leg_results[i]["result"] if i < len(leg_results) else None,
            detail=leg_results[i]["detail"] if i < len(leg_results) else None,
        )
        for i, leg in enumerate(p["legs"])
    )
    return ParlayOut(
        pick_id=p["pick_id"], published_at=p["published_at"], kickoff_at=p["kickoff_at"], price=p["price"],
        status=p["status"], settlement=p.get("settlement"), note=p.get("note"),
        sportsbook_or_source=p["sportsbook_or_source"], void_reason=p.get("void_reason"), legs=legs,
    )


def _streak_outs(windows: dict) -> tuple[StreakOut, ...]:
    return tuple(
        StreakOut(window=s.window, n=s.n, wins=s.wins, losses=s.losses, pushes=s.pushes,
                  win_rate=s.win_rate, total_units=s.total_units, headline=s.headline if s.headline_eligible else None)
        for s in windows.values()
    )


def _category_record_out(picks: list[dict], category: str) -> CategoryRecordOut:
    r = compute_category_record(picks, category)
    return CategoryRecordOut(
        category=r.category, n_settled=r.n_settled, wins=r.wins, losses=r.losses, pushes=r.pushes,
        win_rate=r.win_rate, total_units=r.total_units, roi_per_bet=r.roi_per_bet, average_price=r.average_price,
    )


@app.get("/api/nfl/expert-picks", response_model=ExpertPicksResponse)
def expert_picks() -> ExpertPicksResponse:
    from nfl_predict.decision.pick_ledger import read_current_picks

    picks = read_current_picks()
    singles = [p for p in picks if p["category"] == "EXPERT_PICKS"]
    parlays = [p for p in picks if p["category"] == "EXPERT_PARLAYS"]

    season, schedule = recon.current_season_and_schedule()
    games_by_id = {g.game_id: g for g in schedule.games}
    teams = recon.team_lookup()

    def by_status(items: list[dict], status: str) -> list[dict]:
        return [p for p in items if p["status"] == status]

    def soonest_kickoff(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda p: p["kickoff_at"] or "")

    def newest_settled(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda p: p["settled_at"], reverse=True)[:100]

    return ExpertPicksResponse(
        record=_category_record_out(picks, "EXPERT_PICKS"),
        parlay_record=_category_record_out(picks, "EXPERT_PARLAYS"),
        streaks=_streak_outs(compute_all_predefined_windows(picks, "EXPERT_PICKS", season=season)),
        open_picks=tuple(_pick_out(p, games_by_id, teams) for p in soonest_kickoff(by_status(singles, "PUBLISHED"))),
        settled_picks=tuple(_pick_out(p, games_by_id, teams) for p in newest_settled(by_status(singles, "SETTLED"))),
        open_parlays=tuple(_parlay_out(p) for p in soonest_kickoff(by_status(parlays, "PUBLISHED"))),
        settled_parlays=tuple(_parlay_out(p) for p in newest_settled(by_status(parlays, "SETTLED"))),
        voided_picks=tuple(_pick_out(p, games_by_id, teams) for p in by_status(singles, "VOID")),
        voided_parlays=tuple(_parlay_out(p) for p in by_status(parlays, "VOID")),
    )


@app.get("/api/nfl/performance", response_model=PerformanceResponse)
def performance() -> PerformanceResponse:
    from nfl_predict.decision.pick_ledger import read_current_picks

    picks = read_current_picks()
    best_bets_record = _category_record_out(picks, "BEST_BETS")
    all_model_predictions_record = _category_record_out(picks, "ALL_MODEL_PREDICTIONS")

    season = recon.current_season_and_schedule()[0]
    windows = compute_all_predefined_windows(picks, "BEST_BETS", season=season)
    streak_outs = _streak_outs(windows)

    headline = None
    if windows.get("last_10") and windows["last_10"].headline_eligible:
        headline = windows["last_10"].headline
    elif windows.get("current_streak") and windows["current_streak"].headline_eligible:
        headline = windows["current_streak"].headline

    return PerformanceResponse(
        best_bets_record=best_bets_record, all_model_predictions_record=all_model_predictions_record,
        streaks=streak_outs, headline=headline, has_settled_history=best_bets_record.n_settled > 0,
    )


@app.get("/api/nfl/model-status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    season, schedule = recon.current_season_and_schedule()
    week = schedule.current_week
    games = [g for g in schedule.games if g.week == week] if week is not None else []

    last_model_at = last_research_at = last_decision_at = None
    for g in games:
        prediction = recon.latest_model_prediction(g.game_id)
        if prediction and prediction.get("generated_at"):
            last_model_at = max(filter(None, [last_model_at, prediction["generated_at"]]))
        research = recon.latest_research_summary(season, g.week, g.game_id)
        if research and research.get("research_timestamp"):
            last_research_at = max(filter(None, [last_research_at, research["research_timestamp"]]))
        decisions = recon.latest_decisions_for_game(season, g.week, g.game_id)
        for rec in decisions.values():
            if rec and rec.get("decision_timestamp"):
                last_decision_at = max(filter(None, [last_decision_at, rec["decision_timestamp"]]))

    return ModelStatusResponse(
        rule_set_version=recon.rule_set_version(),
        model_ids=("elo_v2", "ridge_margin_E_v1", "lightgbm_F_v1"),
        odds_provider_status=get_production_odds_provider().status.value,
        research_provider_status=get_production_research_provider().status.value,
        last_model_prediction_at=last_model_at, last_research_run_at=last_research_at, last_decision_at=last_decision_at,
    )


@app.get("/api/nfl/decisions/{game_id}", response_model=DecisionsResponse)
def decisions_for_game(game_id: str) -> DecisionsResponse:
    season, schedule = recon.current_season_and_schedule()
    game = next((g for g in schedule.games if g.game_id == game_id), None)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found in the current season's schedule.")

    records = recon.latest_decisions_for_game(season, game.week, game_id)
    published_market_types = recon.published_best_bet_market_types(game_id)

    def _detail(record: dict | None) -> DecisionDetail:
        if record is None:
            return DecisionDetail(available=False)
        market = recon.market_point_for_decision_record(record)
        research = recon.research_summary_for_decision_record(season, game.week, record)
        return DecisionDetail(
            available=True, market_type=record["market_type"], decision=record["decision"],
            decision_label=lang.translate_decision(record["decision"]),
            reason_codes=_reason_codes_out(record["reason_codes"]),
            decision_timestamp=record["decision_timestamp"], decision_rule_version=record["decision_rule_version"],
            market=_market_block(market), research=_research_block(research),
            is_published_best_bet=record["market_type"] in published_market_types,
        )

    return DecisionsResponse(game_id=game_id, spread=_detail(records["spread"]), moneyline=_detail(records["moneyline"]))
