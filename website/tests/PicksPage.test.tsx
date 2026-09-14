import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { makePick } from "./fixtures";

const { getBestBets } = vi.hoisted(() => ({ getBestBets: vi.fn() }));
vi.mock("@/lib/api", () => ({ getBestBets }));

describe("PicksPage", () => {
  it("renders a real pick with a team name, never the raw 'home'/'away' enum text", async () => {
    getBestBets.mockResolvedValue({ schema_version: "1", picks: [makePick()], message: null });
    const PicksPage = (await import("@/app/nfl/picks/page")).default;

    render(await PicksPage());

    expect(screen.getByText("DEN +2.0")).toBeInTheDocument();
    expect(screen.getByText(/DEN @ KC/)).toBeInTheDocument();
    expect(screen.queryByText(/^away$/)).not.toBeInTheDocument();
  });

  it("still shows the explicit empty state when there are no picks", async () => {
    getBestBets.mockResolvedValue({ schema_version: "1", picks: [], message: "No plays currently meet our qualification criteria." });
    const PicksPage = (await import("@/app/nfl/picks/page")).default;

    render(await PicksPage());

    expect(screen.getByText("No plays currently meet our qualification criteria.")).toBeInTheDocument();
  });
});
