import { describe, expect, it } from "vitest";
import { matchupText, selectionLineText } from "@/lib/picks";
import { makePick } from "./fixtures";

describe("selectionLineText", () => {
  it("never renders the raw internal 'home'/'away' string - a real regression this caught", () => {
    const text = selectionLineText(makePick());
    expect(text).not.toBe("away");
    expect(text).not.toBe("home");
  });

  it("shows the selected team's abbreviation for a spread pick, sign-flipped from the stored home-sign line", () => {
    // line is stored home-sign (-2.0 = home favored by 2); selection is "away", so the
    // away team's own line must be the positive mirror: DEN +2.0.
    expect(selectionLineText(makePick())).toBe("DEN +2.0");
  });

  it("shows the home team's own sign directly when the selection is home", () => {
    const pick = makePick({ selection: "home", selection_team: { team_id: "2310", abbr: "KC", name: "Kansas City Chiefs", nickname: "Chiefs" } });
    expect(selectionLineText(pick)).toBe("KC -2.0");
  });

  it("shows a moneyline pick as '<team> ML <price>'", () => {
    const pick = makePick({ market_type: "moneyline", line: null, price: 114 });
    expect(selectionLineText(pick)).toBe("DEN ML +114");
  });

  it("falls back to the raw selection string only if team info is truly unavailable", () => {
    const pick = makePick({ selection_team: null });
    expect(selectionLineText(pick)).toBe("away +2.0");
  });
});

describe("matchupText", () => {
  it("renders 'AWAY @ HOME' from real team abbreviations", () => {
    expect(matchupText(makePick())).toBe("DEN @ KC");
  });

  it("falls back to the raw game_id if team info is unavailable", () => {
    const pick = makePick({ home_team: null, away_team: null });
    expect(matchupText(pick)).toBe("2026_01_DEN_KC");
  });
});
