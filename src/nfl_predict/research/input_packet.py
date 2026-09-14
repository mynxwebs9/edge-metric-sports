"""Phase 6 Step 3: the structured, immutable research input packet.

Built ONLY from data the quantitative/market layers have already produced - this module
never invents a prediction or a market number. Two sources exist:

- **Historical (2024-2025)**: read directly from the already-persisted, hash-verified Phase 4
  ledger (`nfl_predict.backtesting.ledger`) and Phase 5 `market_snapshot`
  (`nfl_predict.market.snapshot_store`) - both frozen, both real.
- **Live/current (any season beyond the sealed holdout)**: Elo's frozen config
  (k_factor/home_field_advantage, never refit) is run forward, sequentially, over every
  completed game up to now (exactly the mechanism `EloModel.run_sequential` already uses -
  this is operating the frozen model, not retraining it) to predict an upcoming, unplayed
  game via `EloModel.predict_pregame`. Ridge/LightGBM require Phase 2's feature-engineering
  pipeline to have run against the current season's play-by-play, which does not exist for
  any season beyond the sealed holdout - their `ModelPrediction.available` is explicitly
  `False` rather than fabricating a number. The market section is `available=False` unless
  a live/manually-recorded snapshot is supplied - Phase 5's live odds provider has no wired
  HTTP calls yet (see `nfl_predict.market.odds_provider`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.logging_conf import get_logger
from nfl_predict.models.elo import EloConfig, EloModel, sort_games_chronologically
from nfl_predict.models.targets import build_targets

from nfl_predict.backtesting.holdout_guard import assert_freeze_complete
from nfl_predict.backtesting.ledger import read_prediction_ledger
from nfl_predict.market.snapshot_store import read_market_snapshot

logger = get_logger(__name__)

# Frozen in Phase 4 (docs/PHASE4_BACKTEST_REPORT.md): the Elo->margin linear transform,
# fit ONCE on development-period (2010-2022) Elo history only. Never refit here.
FROZEN_ELO_MARGIN_SLOPE = 0.0427
FROZEN_ELO_MARGIN_INTERCEPT = 0.0655


@dataclass(frozen=True)
class ModelPrediction:
    model_id: str
    available: bool
    predicted_margin: float | None = None
    home_win_probability: float | None = None
    uncertainty_note: str | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class MarketContext:
    available: bool
    home_spread_traditional: float | None = None
    home_spread_price: int | None = None
    away_spread_price: int | None = None
    home_moneyline: int | None = None
    away_moneyline: int | None = None
    total_line: float | None = None
    no_vig_home_win_probability: float | None = None
    snapshot_timestamp: str | None = None
    snapshot_type: str | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class ResearchInputPacket:
    game_id: str
    season: int
    week: int
    season_type: str
    home_team_id: str
    away_team_id: str
    kickoff_timestamp: str | None
    packet_generated_at: str
    elo: ModelPrediction
    ridge: ModelPrediction
    lightgbm: ModelPrediction
    market: MarketContext
    model_market_disagreement_points: float | None
    known_qb_continuity_note: str | None
    known_personnel_continuity_note: str | None
    known_injury_summary: str | None

    def content_hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_historical_input_packet(game_id: str, run_id: str = "phase4_v1") -> ResearchInputPacket:
    """For a game already scored in the Phase 4 ledger (2024-2025 only) - every field is
    real, frozen data, nothing recomputed."""
    ledger = read_prediction_ledger(run_id)
    game_rows = ledger.filter(pl.col("game_id") == game_id)
    if game_rows.height == 0:
        raise ValueError(f"game_id={game_id!r} not found in ledger run_id={run_id!r}")

    row0 = game_rows.row(0, named=True)
    season, week, season_type = row0["season"], row0["week"], row0["season_type"]

    def _model_pred(model_id: str, target: str) -> float | None:
        sub = game_rows.filter((pl.col("model_id") == model_id) & (pl.col("target") == target))
        return float(sub["predicted_value"][0]) if sub.height else None

    elo = ModelPrediction(
        model_id="elo_v2", available=True,
        predicted_margin=_model_pred("elo_v2", "home_margin"),
        home_win_probability=_model_pred("elo_v2", "home_win"),
        uncertainty_note="See docs/PHASE4_BACKTEST_REPORT.md Step 8 for validated margin residual_std.",
    )
    ridge = ModelPrediction(
        model_id="ridge_margin_E_v1", available=True,
        predicted_margin=_model_pred("ridge_margin_E_v1", "home_margin"),
    )
    lightgbm = ModelPrediction(
        model_id="lightgbm_F_v1", available=True,
        predicted_margin=_model_pred("lightgbm_F_v1", "home_margin"),
        home_win_probability=_model_pred("lightgbm_F_v1", "home_win"),
    )

    market_frame = read_market_snapshot([season])
    market_row = market_frame.filter(pl.col("game_id") == game_id)
    if market_row.height:
        m = market_row.row(0, named=True)
        market = MarketContext(
            available=True, home_spread_traditional=m["home_spread_traditional"],
            home_spread_price=m["home_spread_price"], away_spread_price=m["away_spread_price"],
            home_moneyline=m["home_moneyline"], away_moneyline=m["away_moneyline"], total_line=m["total_line"],
            snapshot_timestamp=m["snapshot_timestamp"], snapshot_type=m["snapshot_type"],
        )
        disagreement = elo.predicted_margin - (-m["home_spread_traditional"]) if elo.predicted_margin is not None else None
    else:
        market = MarketContext(available=False, unavailable_reason="No market_snapshot row for this game_id/season.")
        disagreement = None

    conn = get_connection()
    init_schema(conn)
    game_row = conn.execute("SELECT home_team_id, away_team_id, kickoff_time_naive FROM games WHERE game_id = ?", (game_id,)).fetchone()
    conn.close()

    return ResearchInputPacket(
        game_id=game_id, season=season, week=week, season_type=season_type,
        home_team_id=game_row["home_team_id"] if game_row else "", away_team_id=game_row["away_team_id"] if game_row else "",
        kickoff_timestamp=game_row["kickoff_time_naive"] if game_row else None,
        packet_generated_at=datetime.now(timezone.utc).isoformat(),
        elo=elo, ridge=ridge, lightgbm=lightgbm, market=market,
        model_market_disagreement_points=disagreement,
        known_qb_continuity_note=None, known_personnel_continuity_note=None, known_injury_summary=None,
    )


def _live_elo_model() -> EloModel:
    """Runs the frozen Elo config sequentially over every completed game currently in the
    `games` table (2010 through whatever has actually been played) - this is OPERATING the
    frozen model forward in time, never retraining/refitting it. k_factor/home_field_advantage
    come from the same verified Phase 4 freeze manifest Phase 4/5 used."""
    manifest = assert_freeze_complete()
    elo_entry = next(c for c in manifest["candidate_models"] if c["model_id"] == "elo_v2")
    config = EloConfig(
        k_factor=elo_entry["hyperparameters"]["k_factor"],
        home_field_advantage=elo_entry["hyperparameters"]["home_field_advantage"],
        season_regression_fraction=elo_entry["hyperparameters"]["season_regression_fraction"],
        initial_rating=elo_entry["hyperparameters"]["initial_rating"],
    )
    conn = get_connection()
    init_schema(conn)
    seasons = [r[0] for r in conn.execute("SELECT DISTINCT season FROM games").fetchall()]
    targets = build_targets(conn, seasons)
    conn.close()

    sorted_targets = sort_games_chronologically(targets)
    model = EloModel(config)
    model.run_sequential(sorted_targets)
    return model


def build_live_input_packet(game_id: str) -> ResearchInputPacket:
    """For an upcoming, unplayed game (any season beyond the sealed holdout). Elo is real
    (frozen config, run forward over real completed games); Ridge/LightGBM and the market
    are marked unavailable with an explicit reason rather than fabricated - see module
    docstring."""
    conn = get_connection()
    init_schema(conn)
    game_row = conn.execute(
        "SELECT season, week, season_type, home_team_id, away_team_id, kickoff_time_naive, game_status FROM games WHERE game_id = ?",
        (game_id,),
    ).fetchone()
    conn.close()
    if game_row is None:
        raise ValueError(f"game_id={game_id!r} not found in the games table.")

    elo_model = _live_elo_model()
    pred = elo_model.predict_pregame(game_row["home_team_id"], game_row["away_team_id"], game_row["season"])
    predicted_margin = FROZEN_ELO_MARGIN_SLOPE * pred["elo_diff_pre"] + FROZEN_ELO_MARGIN_INTERCEPT
    elo = ModelPrediction(
        model_id="elo_v2", available=True, predicted_margin=predicted_margin,
        home_win_probability=pred["expected_home_win_prob"],
        uncertainty_note="Live continuation of the frozen Phase 4 Elo model - margin transform frozen from Phase 4 development data, never refit.",
    )
    ridge = ModelPrediction(model_id="ridge_margin_E_v1", available=False, unavailable_reason="No Phase 2 feature-engineering pipeline has been run against this season's play-by-play - see docs/ARCHITECTURE.md's open live-prediction-pipeline item.")
    lightgbm = ModelPrediction(model_id="lightgbm_F_v1", available=False, unavailable_reason=ridge.unavailable_reason)
    market = MarketContext(available=False, unavailable_reason="No live odds provider is configured (nfl_predict.market.odds_provider.TheOddsAPIProvider requires NFL_ODDS_API_KEY) - see docs/PHASE6_RESEARCH_AGENT_REPORT.md.")

    return ResearchInputPacket(
        game_id=game_id, season=game_row["season"], week=game_row["week"], season_type=game_row["season_type"],
        home_team_id=game_row["home_team_id"], away_team_id=game_row["away_team_id"],
        kickoff_timestamp=game_row["kickoff_time_naive"],
        packet_generated_at=datetime.now(timezone.utc).isoformat(),
        elo=elo, ridge=ridge, lightgbm=lightgbm, market=market,
        model_market_disagreement_points=None,
        known_qb_continuity_note=None, known_personnel_continuity_note=None, known_injury_summary=None,
    )
