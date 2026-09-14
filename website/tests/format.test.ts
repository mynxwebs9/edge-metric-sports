import { describe, expect, it } from "vitest";
import {
  formatMargin,
  formatMoneyline,
  formatProbabilityPct,
  formatSpread,
  formatUnits,
} from "@/lib/format";

describe("formatMargin", () => {
  it("shows a null value as an em dash", () => {
    expect(formatMargin(null)).toBe("—");
  });
  it("adds a + sign for a positive margin", () => {
    expect(formatMargin(3.2)).toBe("+3.2");
  });
  it("keeps the sign for a negative margin", () => {
    expect(formatMargin(-3.2)).toBe("-3.2");
  });
});

describe("formatSpread", () => {
  it("shows 'No line' for a null value", () => {
    expect(formatSpread(null)).toBe("No line");
  });
  it("prefixes the team abbreviation when given", () => {
    expect(formatSpread(-2.5, "KC")).toBe("KC -2.5");
  });
  it("adds a + sign for a positive spread", () => {
    expect(formatSpread(2.5, "DEN")).toBe("DEN +2.5");
  });
});

describe("formatMoneyline", () => {
  it("shows a null value as an em dash", () => {
    expect(formatMoneyline(null)).toBe("—");
  });
  it("adds a + sign for a positive (underdog) price", () => {
    expect(formatMoneyline(120)).toBe("+120");
  });
  it("keeps a negative (favorite) price as-is", () => {
    expect(formatMoneyline(-142)).toBe("-142");
  });
});

describe("formatProbabilityPct", () => {
  it("shows a null value as an em dash", () => {
    expect(formatProbabilityPct(null)).toBe("—");
  });
  it("rounds to the nearest whole percent", () => {
    expect(formatProbabilityPct(0.5608)).toBe("56%");
  });
});

describe("formatUnits", () => {
  it("shows a null value as an em dash", () => {
    expect(formatUnits(null)).toBe("—");
  });
  it("adds a + sign for positive units", () => {
    expect(formatUnits(5.3)).toBe("+5.3u");
  });
  it("keeps the sign for negative units", () => {
    expect(formatUnits(-2.1)).toBe("-2.1u");
  });
});
