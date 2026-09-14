import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import DecisionBadge from "@/components/DecisionBadge";

describe("DecisionBadge", () => {
  it("shows Pending when no decision is available", () => {
    render(<DecisionBadge decision={null} label={null} />);
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });

  it.each([
    ["QUALIFIED_BET", "Qualified Bet"],
    ["LEAN", "Lean"],
    ["WATCH", "Watch"],
    ["VETO", "Veto"],
    ["NO_BET", "No Bet"],
  ])("renders the consumer-facing label for %s", (decision, label) => {
    render(<DecisionBadge decision={decision} label={label} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("never renders the raw internal enum text as the visible label", () => {
    render(<DecisionBadge decision="QUALIFIED_BET" label="Qualified Bet" />);
    expect(screen.queryByText("QUALIFIED_BET")).not.toBeInTheDocument();
  });
});
