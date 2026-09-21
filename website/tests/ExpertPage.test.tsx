import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { makePick } from "./fixtures";
import type { CategoryRecordOut, ExpertPicksResponse, ParlayOut } from "@/lib/types";

const { getExpertPicks } = vi.hoisted(() => ({ getExpertPicks: vi.fn() }));
vi.mock("@/lib/api", () => ({ getExpertPicks }));

const EMPTY_RECORD: CategoryRecordOut = {
  category: "EXPERT_PICKS", n_settled: 0, wins: 0, losses: 0, pushes: 0, win_rate: null, total_units: null, roi_per_bet: null, average_price: null,
};

function response(overrides: Partial<ExpertPicksResponse> = {}): ExpertPicksResponse {
  return {
    schema_version: "1", record: EMPTY_RECORD, parlay_record: { ...EMPTY_RECORD, category: "EXPERT_PARLAYS" }, streaks: [],
    open_picks: [], settled_picks: [], open_parlays: [], settled_parlays: [], voided_picks: [], voided_parlays: [],
    ...overrides,
  };
}

function makeParlay(overrides: Partial<ParlayOut> = {}): ParlayOut {
  return {
    pick_id: "expert_parlay_abc", published_at: "2026-09-21T06:43:16+00:00", kickoff_at: "2026-09-22T00:15:00+00:00",
    price: 180, status: "PUBLISHED", settlement: null, note: null, sportsbook_or_source: "Expert-supplied parlay price", void_reason: null,
    legs: [
      { description: "LA moneyline", matchup: "NYG @ LA", price: -305, result: null, detail: null },
      { description: "Matthew Stafford 210+ passing yards", matchup: "NYG @ LA", price: -233, result: null, detail: null },
      { description: "Cam Skattebo 40+ rushing yards", matchup: "NYG @ LA", price: -242, result: null, detail: null },
    ],
    ...overrides,
  };
}

async function renderPage() {
  const ExpertPicksPage = (await import("@/app/nfl/expert/page")).default;
  render(await ExpertPicksPage());
}

describe("ExpertPicksPage", () => {
  it("shows an honest empty state, never placeholder picks, before any pick exists", async () => {
    getExpertPicks.mockResolvedValue(response());
    await renderPage();

    expect(screen.getByText("No open parlay right now.")).toBeInTheDocument();
    expect(screen.getByText("No open expert picks right now.")).toBeInTheDocument();
    expect(screen.getAllByText("No settled results yet.")).toHaveLength(2); // both records
    expect(screen.queryByText("Results")).not.toBeInTheDocument();
    expect(screen.queryByText("Voided Before Kickoff")).not.toBeInTheDocument();
  });

  it("puts the expert's name on the page", async () => {
    getExpertPicks.mockResolvedValue(response());
    await renderPage();
    expect(screen.getByText("By Mario Quiterio")).toBeInTheDocument();
  });

  it("renders an open parlay with every leg, each leg's price, and the parlay's own price", async () => {
    getExpertPicks.mockResolvedValue(response({ open_parlays: [makeParlay({ note: "Rams roll tonight." })] }));
    await renderPage();

    expect(screen.getByText("3-Leg Parlay")).toBeInTheDocument();
    expect(screen.getByText("+180")).toBeInTheDocument();
    expect(screen.getByText("LA moneyline")).toBeInTheDocument();
    expect(screen.getByText("Matthew Stafford 210+ passing yards")).toBeInTheDocument();
    expect(screen.getByText("Cam Skattebo 40+ rushing yards")).toBeInTheDocument();
    expect(screen.getByText(/NYG @ LA · -305/)).toBeInTheDocument();
    expect(screen.getByText("Rams roll tonight.")).toBeInTheDocument();
    expect(screen.queryByText("WIN")).not.toBeInTheDocument();
    expect(screen.queryByText("No open parlay right now.")).not.toBeInTheDocument();
  });

  it("shows how every leg of a settled parlay was graded, including a leg never graded because an earlier one lost", async () => {
    const settled = makeParlay({
      status: "SETTLED", settlement: "LOSS",
      legs: [
        { description: "LA moneyline", matchup: "NYG @ LA", price: -305, result: "WIN", detail: "NYG 10 - LA 27" },
        { description: "Cam Skattebo 40+ rushing yards", matchup: "NYG @ LA", price: -242, result: "LOSS", detail: "39 rushing yards" },
        { description: "Other leg", matchup: "LA @ DEN", price: -110, result: "NOT_GRADED", detail: "game not final yet" },
      ],
    });
    getExpertPicks.mockResolvedValue(response({
      settled_parlays: [settled],
      parlay_record: { ...EMPTY_RECORD, category: "EXPERT_PARLAYS", n_settled: 1, wins: 0, losses: 1, win_rate: 0, total_units: -1, roi_per_bet: -1, average_price: 180 },
    }));
    await renderPage();

    expect(screen.getByText("Results")).toBeInTheDocument();
    expect(screen.getByText("39 rushing yards", { exact: false })).toBeInTheDocument();
    expect(screen.getByText("WIN")).toBeInTheDocument();
    expect(screen.getAllByText("LOSS").length).toBe(2); // the leg and the parlay
    expect(screen.getByText("Not graded")).toBeInTheDocument();
    expect(screen.getByText("0-1")).toBeInTheDocument(); // the parlay record card
  });

  it("renders an open single pick with its team, line, and the expert's note", async () => {
    getExpertPicks.mockResolvedValue(response({
      open_picks: [makePick({ pick_id: "p_open", note: "Denver's front seven travels well." })],
    }));
    await renderPage();

    expect(screen.getByText("DEN +2.0")).toBeInTheDocument();
    expect(screen.getByText("Denver's front seven travels well.")).toBeInTheDocument();
    expect(screen.queryByText("WIN")).not.toBeInTheDocument();
  });

  it("shows settled single picks with their result, plus the windows table once any single pick has settled", async () => {
    getExpertPicks.mockResolvedValue(response({
      record: { ...EMPTY_RECORD, n_settled: 3, wins: 2, losses: 1, win_rate: 2 / 3, total_units: 0.9, roi_per_bet: 0.3, average_price: -110 },
      streaks: [{ window: "season_to_date", n: 3, wins: 2, losses: 1, pushes: 0, win_rate: 2 / 3, total_units: 0.9, headline: null }],
      settled_picks: [
        makePick({ pick_id: "p_win", settlement: "WIN", status: "SETTLED" }),
        makePick({ pick_id: "p_loss", settlement: "LOSS", status: "SETTLED" }),
      ],
    }));
    await renderPage();

    expect(screen.getByText("Results")).toBeInTheDocument();
    expect(screen.getByText("WIN")).toBeInTheDocument();
    expect(screen.getByText("LOSS")).toBeInTheDocument();
    expect(screen.getAllByText("2-1")).toHaveLength(2); // the record card and the season-to-date row
    expect(screen.getByText("Season to date")).toBeInTheDocument();
  });

  it("lists a voided pick and parlay with the reason instead of silently dropping them", async () => {
    getExpertPicks.mockResolvedValue(response({
      voided_picks: [makePick({ pick_id: "p_void", status: "VOID", void_reason: "CORRUPTED_INPUT_DETECTED_PRE_EVENT" })],
      voided_parlays: [makeParlay({ status: "VOID", void_reason: "DUPLICATE_PUBLICATION" })],
    }));
    await renderPage();

    expect(screen.getByText("Voided Before Kickoff")).toBeInTheDocument();
    expect(screen.getByText(/DEN \+2\.0 \(DEN @ KC\) - Entered in error and corrected before kickoff/)).toBeInTheDocument();
    expect(screen.getByText(/3-leg parlay \(\+180\).*Duplicate entry/)).toBeInTheDocument();
    expect(screen.getByText("No open parlay right now.")).toBeInTheDocument(); // voided is never "open"
  });

  it("says so, rather than crashing, when the API is unavailable", async () => {
    getExpertPicks.mockRejectedValue(new Error("down"));
    await renderPage();
    expect(screen.getByText("Expert picks are temporarily unavailable.")).toBeInTheDocument();
  });
});
