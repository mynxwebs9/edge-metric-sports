import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import WeekTabs from "@/components/WeekTabs";

describe("WeekTabs", () => {
  it("renders a tab for every real week", () => {
    render(<WeekTabs weeks={[1, 2, 3]} activeWeek={1} />);
    expect(screen.getByText("Week 1")).toBeInTheDocument();
    expect(screen.getByText("Week 2")).toBeInTheDocument();
    expect(screen.getByText("Week 3")).toBeInTheDocument();
  });

  it("links each tab to the correct week query param", () => {
    render(<WeekTabs weeks={[1, 2]} activeWeek={1} />);
    expect(screen.getByText("Week 2").closest("a")).toHaveAttribute("href", "/nfl/schedule?week=2");
  });
});
