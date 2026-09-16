import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import GameCard from "@/components/GameCard";
import { makeGameCard } from "./fixtures";

describe("GameCard", () => {
  it("renders real team abbreviations, never hard-coded placeholder teams", () => {
    render(<GameCard game={makeGameCard()} />);
    expect(screen.getByText("DEN")).toBeInTheDocument();
    expect(screen.getByText("KC")).toBeInTheDocument();
  });

  it("renders the market spread with the home team's abbreviation", () => {
    render(<GameCard game={makeGameCard()} />);
    expect(screen.getByText("KC -2.5")).toBeInTheDocument();
  });

  it("renders 'No line yet' when the market is unavailable, never a fabricated spread", () => {
    const game = makeGameCard({
      market: { available: false, home_spread_traditional: null, home_moneyline: null, away_moneyline: null, no_vig_home_win_probability: null, snapshot_timestamp: null, consensus_algorithm_version: null, sportsbooks: [] },
    });
    render(<GameCard game={game} />);
    expect(screen.getByText("No line yet")).toBeInTheDocument();
  });

  it("renders 'Not available' for model block when the model hasn't run yet", () => {
    const game = makeGameCard({
      model: { available: false, predicted_winner: null, elo_predicted_margin: null, elo_home_win_probability: null, ridge_predicted_margin: null, lightgbm_predicted_margin: null, all_agree_on_direction: null, margin_dispersion: null, generated_at: null },
    });
    render(<GameCard game={game} />);
    expect(screen.getByText("Not available")).toBeInTheDocument();
  });

  it("links to the individual matchup page", () => {
    render(<GameCard game={makeGameCard()} />);
    const link = screen.getByRole("link", { name: /view matchup/i });
    expect(link).toHaveAttribute("href", "/nfl/games/2026_01_DEN_KC");
  });

  it("always shows a directional pick, even when the decision engine says no bet", () => {
    // The default fixture's decision is VETO (no official bet), but every game still gets
    // a real, priced system pick - the whole point of ALL_MODEL_PREDICTIONS.
    render(<GameCard game={makeGameCard()} />);
    expect(screen.getByText("DEN +120")).toBeInTheDocument();
  });

  it("shows a Best Bet badge only when this pick is also a real published Best Bet", () => {
    const noBadge = makeGameCard();
    render(<GameCard game={noBadge} />);
    expect(screen.queryByText("Best Bet")).not.toBeInTheDocument();
  });

  it("shows the Best Bet badge when the system pick is also an official published pick", () => {
    const game = makeGameCard({
      system_pick: { available: true, selection: "away", selection_team: { team_id: "1400", abbr: "DEN", name: "Denver Broncos", nickname: "Broncos" }, price: 120, market_type: "moneyline", status: "PUBLISHED", settlement: null, is_also_best_bet: true },
    });
    render(<GameCard game={game} />);
    expect(screen.getByText("Best Bet")).toBeInTheDocument();
  });

  it("shows 'Not available' for the pick when nothing has been published for this game yet", () => {
    const game = makeGameCard({
      system_pick: { available: false, selection: null, selection_team: null, price: null, market_type: null, status: null, settlement: null, is_also_best_bet: false },
    });
    render(<GameCard game={game} />);
    const notAvailable = screen.getAllByText("Not available");
    expect(notAvailable.length).toBeGreaterThan(0);
  });
});
