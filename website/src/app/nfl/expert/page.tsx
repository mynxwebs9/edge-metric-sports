import type { Metadata } from "next";
import { getExpertPicks } from "@/lib/api";
import AdSlot from "@/components/AdSlot";
import PickListItem from "@/components/PickListItem";
import RecordCard from "@/components/RecordCard";
import WindowsCard from "@/components/WindowsCard";

const TITLE = "NFL Expert Picks";
const DESCRIPTION = "Hand-made NFL picks from our expert, published before kickoff and graded against the real final score - a verified record, win or lose.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION, url: "https://edgemetricsports.com/nfl/expert", type: "website" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

export default async function ExpertPicksPage() {
  let expert;
  try {
    expert = await getExpertPicks();
  } catch {
    expert = null;
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <h1 className="mb-2 text-4xl font-black tracking-tight">Expert Picks</h1>
      <p className="mb-8 text-muted">
        These are hand-made picks from a person, not the model. Each one is published before kickoff, locked into
        the same append-only ledger as everything else on this site, and graded against the real final score - it
        can never be edited or deleted. This record is completely separate from the model&apos;s Best Bets.
      </p>

      <AdSlot label="expert-page-top" className="mb-8 h-24" />

      {!expert ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center text-muted">
          Expert picks are temporarily unavailable.
        </div>
      ) : (
        <>
          <div className="mb-8 grid gap-4 md:grid-cols-2">
            <RecordCard title="Expert Record" record={expert.record} />
            <WindowsCard
              title="Expert Record by Window"
              streaks={expert.streaks}
              hasHistory={expert.record.n_settled > 0}
              emptyMessage="No expert picks have settled yet - windows will populate as picks are graded."
            />
          </div>

          <section className="mb-8">
            <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted">Open Picks</h2>
            {expert.open_picks.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center">
                <p className="mb-1 font-semibold">No open expert picks right now.</p>
                <p className="text-sm text-muted">Picks are added before each game and appear here until the game is graded.</p>
              </div>
            ) : (
              <ul className="space-y-3">
                {expert.open_picks.map((pick) => (
                  <PickListItem key={pick.pick_id} pick={pick} />
                ))}
              </ul>
            )}
          </section>

          {expert.settled_picks.length > 0 && (
            <section className="mb-8">
              <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted">Results</h2>
              <ul className="space-y-3">
                {expert.settled_picks.map((pick) => (
                  <PickListItem key={pick.pick_id} pick={pick} />
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}
