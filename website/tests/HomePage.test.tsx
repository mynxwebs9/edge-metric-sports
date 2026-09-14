import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { makeGameCard, makePick } from "./fixtures";
import type { BestBetsResponse, PerformanceResponse, SlateResponse } from "@/lib/types";

const { getCurrentSlate, getBestBets, getPerformance } = vi.hoisted(() => ({
  getCurrentSlate: vi.fn(),
  getBestBets: vi.fn(),
  getPerformance: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ getCurrentSlate, getBestBets, getPerformance }));

const EMPTY_BEST_BETS: BestBetsResponse = { schema_version: "1", picks: [], message: "No plays currently meet our qualification criteria." };
const EMPTY_PERFORMANCE: PerformanceResponse = {
  schema_version: "1",
  best_bets_record: { category: "BEST_BETS", n_settled: 0, wins: 0, losses: 0, pushes: 0, win_rate: null, total_units: null, roi_per_bet: null, average_price: null },
  all_model_predictions_record: { category: "ALL_MODEL_PREDICTIONS", n_settled: 0, wins: 0, losses: 0, pushes: 0, win_rate: null, total_units: null, roi_per_bet: null, average_price: null },
  streaks: [], headline: null, has_settled_history: false,
};

describe("HomePage", () => {
  it("renders a real game card from the current slate", async () => {
    const slate: SlateResponse = { schema_version: "1", season: 2026, week: 1, last_updated: "2026-09-12T20:00:00+00:00", games: [makeGameCard()] };
    getCurrentSlate.mockResolvedValue(slate);
    getBestBets.mockResolvedValue(EMPTY_BEST_BETS);
    getPerformance.mockResolvedValue(EMPTY_PERFORMANCE);

    const HomePage = (await import("@/app/page")).default;
    render(await HomePage());

    expect(screen.getByText(/2026 · Week 1/)).toBeInTheDocument();
    expect(screen.getAllByText("DEN").length).toBeGreaterThan(0);
  });

  it("shows the explicit empty-state message when no Best Bets qualify", async () => {
    getCurrentSlate.mockResolvedValue({ schema_version: "1", season: 2026, week: 1, last_updated: null, games: [] });
    getBestBets.mockResolvedValue(EMPTY_BEST_BETS);
    getPerformance.mockResolvedValue(EMPTY_PERFORMANCE);

    const HomePage = (await import("@/app/page")).default;
    render(await HomePage());

    expect(screen.getByText("No plays currently meet our qualification criteria.")).toBeInTheDocument();
  });

  it("renders a real published pick's team name, never the raw 'home'/'away' enum text", async () => {
    getCurrentSlate.mockResolvedValue({ schema_version: "1", season: 2026, week: 1, last_updated: null, games: [] });
    getBestBets.mockResolvedValue({ schema_version: "1", picks: [makePick()], message: null });
    getPerformance.mockResolvedValue(EMPTY_PERFORMANCE);

    const HomePage = (await import("@/app/page")).default;
    render(await HomePage());

    expect(screen.getByText("DEN +2.0")).toBeInTheDocument();
    expect(screen.queryByText(/^away$/)).not.toBeInTheDocument();
  });

  it("stays up even when the API is unreachable, without crashing the page", async () => {
    getCurrentSlate.mockRejectedValue(new Error("network error"));
    getBestBets.mockRejectedValue(new Error("network error"));
    getPerformance.mockRejectedValue(new Error("network error"));

    const HomePage = (await import("@/app/page")).default;
    render(await HomePage());

    expect(screen.getByText("NFL Predictions")).toBeInTheDocument();
    expect(screen.getByText(/performance data is temporarily unavailable/i)).toBeInTheDocument();
  });
});
