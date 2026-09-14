import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ReasonCodeList from "@/components/ReasonCodeList";

describe("ReasonCodeList", () => {
  it("shows a friendly empty state with zero reason codes", () => {
    render(<ReasonCodeList codes={[]} />);
    expect(screen.getByText(/no reasons recorded/i)).toBeInTheDocument();
  });

  it("renders each reason code's consumer-facing label, never the raw code", () => {
    render(
      <ReasonCodeList
        codes={[
          { code: "MISSING_LIVE_DATA", label: "Waiting for complete market/research data" },
          { code: "VETO_CONSIDERATION_UNRESOLVED", label: "Significant unresolved pregame risk" },
        ]}
      />
    );
    expect(screen.getByText("Waiting for complete market/research data")).toBeInTheDocument();
    expect(screen.getByText("Significant unresolved pregame risk")).toBeInTheDocument();
    expect(screen.queryByText("MISSING_LIVE_DATA")).not.toBeInTheDocument();
    expect(screen.queryByText("VETO_CONSIDERATION_UNRESOLVED")).not.toBeInTheDocument();
  });
});
