import type { Metadata } from "next";
import { getBestBets } from "@/lib/api";
import AdSlot from "@/components/AdSlot";
import PickListItem from "@/components/PickListItem";

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
            <PickListItem key={pick.pick_id} pick={pick} />
          ))}
        </ul>
      )}
    </div>
  );
}
