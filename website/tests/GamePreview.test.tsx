import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import GamePreview from "@/components/GamePreview";
import type { PreviewBlock } from "@/lib/types";

function makePreview(overrides: Partial<PreviewBlock> = {}): PreviewBlock {
  return {
    available: true, text: "A short, real preview of the matchup.",
    generated_at: "2026-09-12T12:00:00+00:00", prompt_version: "prediction_writer_v1",
    model_provider: "anthropic", model_name: "claude-sonnet-5",
    ...overrides,
  };
}

describe("GamePreview", () => {
  it("renders the article text and an AI-assisted disclosure label", () => {
    render(<GamePreview preview={makePreview()} />);
    expect(screen.getByText("A short, real preview of the matchup.")).toBeInTheDocument();
    expect(screen.getByText("AI-assisted analysis")).toBeInTheDocument();
  });

  it("renders nothing when no preview has been generated yet", () => {
    const { container } = render(<GamePreview preview={makePreview({ available: false, text: null })} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when available is true but text is somehow missing, never a blank article", () => {
    const { container } = render(<GamePreview preview={makePreview({ text: null })} />);
    expect(container).toBeEmptyDOMElement();
  });
});
