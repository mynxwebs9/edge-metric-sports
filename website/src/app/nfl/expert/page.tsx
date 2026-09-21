import type { Metadata } from "next";
import { getExpertPicks } from "@/lib/api";
import AdSlot from "@/components/AdSlot";
import ParlayCard from "@/components/ParlayCard";
import PickListItem from "@/components/PickListItem";
import RecordCard from "@/components/RecordCard";
import WindowsCard from "@/components/WindowsCard";
import { formatMoneyline } from "@/lib/format";
import { matchupText, selectionLineText, voidReasonLabel } from "@/lib/picks";

const TITLE = "NFL Expert Picks";
const EXPERT_NAME = "Mario Quiterio";
const DESCRIPTION = `Hand-made NFL picks and parlays from ${EXPERT_NAME}, published before kickoff and graded against the real final score - a verified record, win or lose.`;

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION, url: "https://edgemetricsports.com/nfl/expert", type: "website" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

function SectionHeading({ children }: { children: string }) {
  return <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted">{children}</h2>;
}

export default async function ExpertPicksPage() {
  let expert;
  try {
    expert = await getExpertPicks();
  } catch {
    expert = null;
  }

  const hasVoided = expert !== null && (expert.voided_picks.length > 0 || expert.voided_parlays.length > 0);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <h1 className="mb-1 text-4xl font-black tracking-tight">Expert Picks</h1>
      <p className="mb-3 text-sm font-semibold uppercase tracking-wider text-accent">By {EXPERT_NAME}</p>
      <p className="mb-8 text-muted">
        These are hand-made picks from {EXPERT_NAME}, not the model. Each one is published before kickoff, locked into
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
            <RecordCard title="Parlay Record" record={expert.parlay_record} />
            <RecordCard title="Single Picks Record" record={expert.record} />
          </div>

          {expert.record.n_settled > 0 && (
            <div className="mb-8">
              <WindowsCard
                title="Single Picks by Window"
                streaks={expert.streaks}
                hasHistory
                emptyMessage=""
              />
            </div>
          )}

          <section className="mb-8">
            <SectionHeading>Open Parlays</SectionHeading>
            {expert.open_parlays.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center">
                <p className="mb-1 font-semibold">No open parlay right now.</p>
                <p className="text-sm text-muted">A parlay appears here before kickoff and stays until every leg is graded.</p>
              </div>
            ) : (
              <ul className="space-y-3">
                {expert.open_parlays.map((parlay) => (
                  <ParlayCard key={parlay.pick_id} parlay={parlay} />
                ))}
              </ul>
            )}
          </section>

          <section className="mb-8">
            <SectionHeading>Open Picks</SectionHeading>
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

          {(expert.settled_parlays.length > 0 || expert.settled_picks.length > 0) && (
            <section className="mb-8">
              <SectionHeading>Results</SectionHeading>
              <ul className="space-y-3">
                {expert.settled_parlays.map((parlay) => (
                  <ParlayCard key={parlay.pick_id} parlay={parlay} />
                ))}
                {expert.settled_picks.map((pick) => (
                  <PickListItem key={pick.pick_id} pick={pick} />
                ))}
              </ul>
            </section>
          )}

          {hasVoided && (
            <section className="mb-8">
              <SectionHeading>Voided Before Kickoff</SectionHeading>
              <p className="mb-3 text-xs text-muted">
                A published pick is never quietly removed. If one is voided, it stays listed here with the reason. Voided
                picks are not counted in any record.
              </p>
              <ul className="space-y-2 text-sm text-muted">
                {expert.voided_parlays.map((parlay) => (
                  <li key={parlay.pick_id}>
                    {parlay.legs.length}-leg parlay ({formatMoneyline(parlay.price)}): {parlay.legs.map((l) => l.description).join(", ")}
                    {" - "}
                    {voidReasonLabel(parlay.void_reason)}
                  </li>
                ))}
                {expert.voided_picks.map((pick) => (
                  <li key={pick.pick_id}>
                    {selectionLineText(pick)} ({matchupText(pick)}) - {voidReasonLabel(pick.void_reason)}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}
