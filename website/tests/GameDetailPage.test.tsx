import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { makeGameDetail } from "./fixtures";

class FakeApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const { getGameDetail } = vi.hoisted(() => ({ getGameDetail: vi.fn() }));
const { notFound } = vi.hoisted(() => ({ notFound: vi.fn(() => { throw new Error("NEXT_NOT_FOUND"); }) }));

vi.mock("@/lib/api", () => ({ getGameDetail, ApiError: FakeApiError }));
vi.mock("next/navigation", () => ({ notFound }));

describe("GameDetailPage", () => {
  it("distinguishes MODEL PREDICTION from BETTING DECISION as separate sections", async () => {
    getGameDetail.mockResolvedValue(makeGameDetail());
    const GameDetailPage = (await import("@/app/nfl/games/[gameId]/page")).default;

    render(await GameDetailPage({ params: Promise.resolve({ gameId: "2026_01_DEN_KC" }), searchParams: Promise.resolve({}) }));

    expect(screen.getByText("Model Prediction")).toBeInTheDocument();
    expect(screen.getByText("Betting Decision")).toBeInTheDocument();
    expect(screen.getByText(/not automatically a bet/i)).toBeInTheDocument();
    expect(screen.getByText("Game Preview")).toBeInTheDocument();
  });

  it("renders both spread and moneyline decision cards with translated reason codes", async () => {
    getGameDetail.mockResolvedValue(makeGameDetail());
    const GameDetailPage = (await import("@/app/nfl/games/[gameId]/page")).default;

    render(await GameDetailPage({ params: Promise.resolve({ gameId: "2026_01_DEN_KC" }), searchParams: Promise.resolve({}) }));

    expect(screen.getByRole("heading", { name: "Market" })).toBeInTheDocument();
    expect(screen.getAllByText("Spread").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Moneyline").length).toBeGreaterThan(0);
    expect(screen.getByText("Significant unresolved pregame risk")).toBeInTheDocument();
    expect(screen.queryByText("VETO_CONSIDERATION_UNRESOLVED")).not.toBeInTheDocument();
  });

  it("shows an unavailable message for a market type with no decision yet", async () => {
    getGameDetail.mockResolvedValue(makeGameDetail());
    const GameDetailPage = (await import("@/app/nfl/games/[gameId]/page")).default;

    render(await GameDetailPage({ params: Promise.resolve({ gameId: "2026_01_DEN_KC" }), searchParams: Promise.resolve({}) }));

    expect(screen.getByText("No decision has been made for this market yet.")).toBeInTheDocument();
  });

  it("distinguishes a published Best Bet from a market that merely qualified - the real screenshot-caught bug", async () => {
    const game = makeGameDetail({
      spread_decision: {
        available: true, market_type: "spread", decision: "QUALIFIED_BET", decision_label: "Qualified Bet",
        reason_codes: [{ code: "MODEL_MARKET_DISAGREEMENT", label: "Model and market disagree meaningfully" }],
        decision_timestamp: "2026-09-14T04:43:04+00:00", is_published_best_bet: false,
      },
      moneyline_decision: {
        available: true, market_type: "moneyline", decision: "QUALIFIED_BET", decision_label: "Qualified Bet",
        reason_codes: [{ code: "MODEL_MARKET_DISAGREEMENT", label: "Model and market disagree meaningfully" }],
        decision_timestamp: "2026-09-14T04:43:04+00:00", is_published_best_bet: true,
      },
    });
    getGameDetail.mockResolvedValue(game);
    const GameDetailPage = (await import("@/app/nfl/games/[gameId]/page")).default;

    render(await GameDetailPage({ params: Promise.resolve({ gameId: "2026_01_DEN_KC" }), searchParams: Promise.resolve({}) }));

    // Both cards say "Qualified Bet" (never "Best Bet" for a mere decision-engine result)...
    expect(screen.getAllByText("Qualified Bet").length).toBe(2);
    // ...but only ONE shows the actual published indicator.
    expect(screen.getByText("✓ Published Best Bet")).toBeInTheDocument();
    expect(screen.getByText(/cleared our automated criteria, but wasn't published/i)).toBeInTheDocument();
  });

  it("calls notFound() for a 404 from the API instead of rendering a broken page", async () => {
    getGameDetail.mockRejectedValue(new FakeApiError(404, "not found"));
    const GameDetailPage = (await import("@/app/nfl/games/[gameId]/page")).default;

    await expect(
      GameDetailPage({ params: Promise.resolve({ gameId: "not_a_real_game" }), searchParams: Promise.resolve({}) })
    ).rejects.toThrow("NEXT_NOT_FOUND");
    expect(notFound).toHaveBeenCalled();
  });
});
