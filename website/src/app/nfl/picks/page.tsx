import type { Metadata } from "next";
import Link from "next/link";
import { getBestBets } from "@/lib/api";
import AdSlot from "@/components/AdSlot";
import { formatMoneyline, formatTimestamp } from "@/lib/format";
import { matchupText, selectionLineText } from "@/lib/picks";

const TITLE = "NFL Best Bets";
const DESCRIPTION = "Our currently qualified NFL picks, derived only from the official prospective pick ledger - never forced.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION, url: "https://edgemetricsports.com/nfl/picks", type: "website" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

export default async function PicksPage() {
  let bestBets;
  try {
    bestBets = await getBestBets();
  } catch {
    bestBets = null;
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <h1 className="mb-2 text-4xl font-black tracking-tight">Best Bets</h1>
      <p className="mb-8 text-muted">
        Every pick here cleared our full decision pipeline - model, market, and research all had to agree there
        was a real edge. We never force a pick just to have something to show.
      </p>

      <AdSlot label="picks-page-top" className="mb-8 h-24" />

      {!bestBets ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center text-muted">
          Picks data is temporarily unavailable.
        </div>
      ) : bestBets.picks.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center">
          <p className="mb-1 font-semibold">{bestBets.message}</p>
          <p className="text-sm text-muted">Zero qualifying bets in a given week is a normal, expected outcome for this system.</p>
        </div>
      ) : (
        <ul className="space-y-3">
          {bestBets.picks.map((pick) => (
            <li key={pick.pick_id} className="rounded-xl border border-border bg-surface p-5">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-bold">{selectionLineText(pick)}</span>
                {pick.market_type === "spread" && <span className="text-sm text-muted">{formatMoneyline(pick.price)}</span>}
              </div>
              <p className="mb-1 text-xs text-muted">
                {matchupText(pick)} ·{" "}
                <Link href={`/nfl/games/${pick.game_id}`} className="font-medium text-accent hover:underline">
                  View matchup
                </Link>
              </p>
              <p className="text-xs text-muted">
                {pick.sportsbook_or_source} · Published {formatTimestamp(pick.published_at)}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
