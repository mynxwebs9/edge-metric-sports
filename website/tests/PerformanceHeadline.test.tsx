import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import PerformanceHeadline from "@/components/PerformanceHeadline";
import type { PerformanceResponse } from "@/lib/types";

function makePerformance(overrides: Partial<PerformanceResponse> = {}): PerformanceResponse {
  return {
    schema_version: "1",
    best_bets_record: { category: "BEST_BETS", n_settled: 0, wins: 0, losses: 0, pushes: 0, win_rate: null, total_units: null, roi_per_bet: null, average_price: null },
    all_model_predictions_record: { category: "ALL_MODEL_PREDICTIONS", n_settled: 0, wins: 0, losses: 0, pushes: 0, win_rate: null, total_units: null, roi_per_bet: null, average_price: null },
    streaks: [], headline: null, has_settled_history: false,
    ...overrides,
  };
}

describe("PerformanceHeadline", () => {
  it("shows a neutral message when there is no settled history and no headline", () => {
    render(<PerformanceHeadline performance={makePerformance()} />);
    expect(screen.getByText(/no settled best bets yet/i)).toBeInTheDocument();
  });

  it("shows a different neutral message when history exists but no headline qualifies", () => {
    render(<PerformanceHeadline performance={makePerformance({ has_settled_history: true })} />);
    expect(screen.getByText(/verified performance tracking is underway/i)).toBeInTheDocument();
  });

  it("renders a real headline when the ledger legitimately produced one", () => {
    render(<PerformanceHeadline performance={makePerformance({ headline: "8-2 Last 10 Best Bets (+5.3 units)", has_settled_history: true })} />);
    expect(screen.getByText(/8-2 LAST 10 BEST BETS/)).toBeInTheDocument();
  });

  it("never fabricates a headline when the API didn't provide one", () => {
    render(<PerformanceHeadline performance={makePerformance({ headline: null })} />);
    expect(screen.queryByText(/🔥/)).not.toBeInTheDocument();
  });
});
