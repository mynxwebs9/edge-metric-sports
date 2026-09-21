import { afterEach, describe, expect, it, vi } from "vitest";
import { getExpertPicks } from "@/lib/api";

afterEach(() => vi.unstubAllGlobals());

const EMPTY_RECORD = { category: "EXPERT_PICKS", n_settled: 0, wins: 0, losses: 0, pushes: 0, win_rate: null, total_units: null, roi_per_bet: null, average_price: null };

describe("getExpertPicks", () => {
  it("tolerates the previous API shape during a deploy, defaulting newer fields to honest empties", async () => {
    // What Render served before parlays existed - no parlay_record/open_parlays/voided_* keys.
    const oldShape = { schema_version: "1", record: EMPTY_RECORD, streaks: [], open_picks: [], settled_picks: [] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => oldShape }));

    const data = await getExpertPicks();

    expect(data.open_parlays).toEqual([]);
    expect(data.voided_picks).toEqual([]);
    expect(data.parlay_record.n_settled).toBe(0);
    expect(data.parlay_record.category).toBe("EXPERT_PARLAYS");
  });

  it("passes the real values through untouched when the API has them", async () => {
    const full = {
      schema_version: "1", record: EMPTY_RECORD, parlay_record: { ...EMPTY_RECORD, category: "EXPERT_PARLAYS", n_settled: 2, wins: 1, losses: 1 },
      streaks: [], open_picks: [], settled_picks: [], open_parlays: [{ pick_id: "p" }], settled_parlays: [], voided_picks: [], voided_parlays: [],
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => full }));

    const data = await getExpertPicks();

    expect(data.parlay_record.wins).toBe(1);
    expect(data.open_parlays).toHaveLength(1);
  });
});
