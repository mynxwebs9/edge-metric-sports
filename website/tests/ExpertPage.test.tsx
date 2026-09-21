import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { makePick } from "./fixtures";
import type { CategoryRecordOut, ExpertPicksResponse } from "@/lib/types";

const { getExpertPicks } = vi.hoisted(() => ({ getExpertPicks: vi.fn() }));
vi.mock("@/lib/api", () => ({ getExpertPicks }));

const EMPTY_RECORD: CategoryRecordOut = {
  category: "EXPERT_PICKS", n_settled: 0, wins: 0, losses: 0, pushes: 0, win_rate: null, total_units: null, roi_per_bet: null, average_price: null,
};

function response(overrides: Partial<ExpertPicksResponse> = {}): ExpertPicksResponse {
  return { schema_version: "1", record: EMPTY_RECORD, streaks: [], open_picks: [], settled_picks: [], ...overrides };
}

describe("ExpertPicksPage", () => {
  it("shows an honest empty state, never placeholder picks, before any pick exists", async () => {
    getExpertPicks.mockResolvedValue(response());
    const ExpertPicksPage = (await import("@/app/nfl/expert/page")).default;

    render(await ExpertPicksPage());

    expect(screen.getByText("No open expert picks right now.")).toBeInTheDocument();
    expect(screen.getAllByText(/No settled results yet|No expert picks have settled yet/).length).toBeGreaterThan(0);
    expect(screen.queryByText("Results")).not.toBeInTheDocument();
  });

  it("renders an open pick with its team, line, and the expert's note", async () => {
    getExpertPicks.mockResolvedValue(response({
      open_picks: [makePick({ pick_id: "p_open", note: "Denver's front seven travels well." })],
    }));
    const ExpertPicksPage = (await import("@/app/nfl/expert/page")).default;

    render(await ExpertPicksPage());

    expect(screen.getByText("DEN +2.0")).toBeInTheDocument();
    expect(screen.getByText("Denver's front seven travels well.")).toBeInTheDocument();
    expect(screen.queryByText("WIN")).not.toBeInTheDocument();
  });

  it("shows settled picks with their result, and the record on the same page", async () => {
    getExpertPicks.mockResolvedValue(response({
      record: { ...EMPTY_RECORD, n_settled: 3, wins: 2, losses: 1, win_rate: 2 / 3, total_units: 0.9, roi_per_bet: 0.3, average_price: -110 },
      streaks: [{ window: "season_to_date", n: 3, wins: 2, losses: 1, pushes: 0, win_rate: 2 / 3, total_units: 0.9, headline: null }],
      settled_picks: [
        makePick({ pick_id: "p_win", settlement: "WIN", status: "SETTLED" }),
        makePick({ pick_id: "p_loss", settlement: "LOSS", status: "SETTLED" }),
      ],
    }));
    const ExpertPicksPage = (await import("@/app/nfl/expert/page")).default;

    render(await ExpertPicksPage());

    expect(screen.getByText("Results")).toBeInTheDocument();
    expect(screen.getByText("WIN")).toBeInTheDocument();
    expect(screen.getByText("LOSS")).toBeInTheDocument();
    expect(screen.getAllByText("2-1")).toHaveLength(2); // the record card and the season-to-date row
    expect(screen.getByText("Season to date")).toBeInTheDocument();
  });

  it("says so, rather than crashing, when the API is unavailable", async () => {
    getExpertPicks.mockRejectedValue(new Error("down"));
    const ExpertPicksPage = (await import("@/app/nfl/expert/page")).default;

    render(await ExpertPicksPage());

    expect(screen.getByText("Expert picks are temporarily unavailable.")).toBeInTheDocument();
  });
});
