import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { makeGameCard } from "./fixtures";
import type { ScheduleWeeksResponse, SlateResponse } from "@/lib/types";

const { getScheduleWeeks, getScheduleForWeek } = vi.hoisted(() => ({
  getScheduleWeeks: vi.fn(),
  getScheduleForWeek: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ getScheduleWeeks, getScheduleForWeek }));

const WEEKS: ScheduleWeeksResponse = { schema_version: "1", season: 2026, weeks: [1, 2, 3], current_week: 1 };

describe("SchedulePage", () => {
  it("defaults to the current week when no ?week= is given", async () => {
    getScheduleWeeks.mockResolvedValue(WEEKS);
    const slate: SlateResponse = { schema_version: "1", season: 2026, week: 1, last_updated: null, games: [makeGameCard()] };
    getScheduleForWeek.mockResolvedValue(slate);

    const SchedulePage = (await import("@/app/nfl/schedule/page")).default;
    render(await SchedulePage({ params: Promise.resolve({}), searchParams: Promise.resolve({}) }));

    expect(getScheduleForWeek).toHaveBeenCalledWith(1);
    expect(screen.getAllByText("DEN").length).toBeGreaterThan(0);
  });

  it("renders the requested week from ?week=", async () => {
    getScheduleWeeks.mockResolvedValue(WEEKS);
    const slate: SlateResponse = { schema_version: "1", season: 2026, week: 2, last_updated: null, games: [] };
    getScheduleForWeek.mockResolvedValue(slate);

    const SchedulePage = (await import("@/app/nfl/schedule/page")).default;
    render(await SchedulePage({ params: Promise.resolve({}), searchParams: Promise.resolve({ week: "2" }) }));

    expect(getScheduleForWeek).toHaveBeenCalledWith(2);
  });

  it("ignores an out-of-range week param and falls back to the current week", async () => {
    getScheduleWeeks.mockResolvedValue(WEEKS);
    getScheduleForWeek.mockResolvedValue({ schema_version: "1", season: 2026, week: 1, last_updated: null, games: [] });

    const SchedulePage = (await import("@/app/nfl/schedule/page")).default;
    render(await SchedulePage({ params: Promise.resolve({}), searchParams: Promise.resolve({ week: "999" }) }));

    expect(getScheduleForWeek).toHaveBeenCalledWith(1);
  });

  it("shows an explicit empty state for a week with no games, never a blank silent page", async () => {
    getScheduleWeeks.mockResolvedValue(WEEKS);
    getScheduleForWeek.mockResolvedValue({ schema_version: "1", season: 2026, week: 1, last_updated: null, games: [] });

    const SchedulePage = (await import("@/app/nfl/schedule/page")).default;
    render(await SchedulePage({ params: Promise.resolve({}), searchParams: Promise.resolve({}) }));

    expect(screen.getByText("No games found for this week.")).toBeInTheDocument();
  });

  it("stays up when the schedule API is unreachable", async () => {
    getScheduleWeeks.mockRejectedValue(new Error("network error"));

    const SchedulePage = (await import("@/app/nfl/schedule/page")).default;
    render(await SchedulePage({ params: Promise.resolve({}), searchParams: Promise.resolve({}) }));

    expect(screen.getByText("Schedule data is temporarily unavailable.")).toBeInTheDocument();
  });
});
