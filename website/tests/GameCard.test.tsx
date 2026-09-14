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

  it("shows the System decision badge", () => {
    render(<GameCard game={makeGameCard()} />);
    expect(screen.getByText("Veto")).toBeInTheDocument();
  });
});
