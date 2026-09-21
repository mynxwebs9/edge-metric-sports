import type { Metadata } from "next";
import { getPerformance } from "@/lib/api";
import AdSlot from "@/components/AdSlot";
import RecordCard from "@/components/RecordCard";
import WindowsCard from "@/components/WindowsCard";

const TITLE = "NFL Prediction Model Record";
const DESCRIPTION = "Our verified, price-aware Best Bets track record - straight from the immutable official pick ledger, never cherry-picked.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION, url: "https://edgemetricsports.com/nfl/performance", type: "website" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

export default async function PerformancePage() {
  let performance;
  try {
    performance = await getPerformance();
  } catch {
    performance = null;
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <h1 className="mb-2 text-4xl font-black tracking-tight">Verified Performance</h1>
      <p className="mb-8 text-muted">
        Every number below comes straight from our immutable, append-only pick ledger. A published pick is never
        edited, deleted, or quietly reclassified after the fact.
      </p>

      <AdSlot label="performance-page-top" className="mb-8 h-24" />

      {!performance ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center text-muted">
          Performance data is temporarily unavailable.
        </div>
      ) : (
        <>
          <div className="mb-8 grid gap-4 sm:grid-cols-2">
            <RecordCard title="Best Bets" record={performance.best_bets_record} />
            <RecordCard title="All Model Predictions" record={performance.all_model_predictions_record} />
          </div>

          <WindowsCard
            title="Best Bets by Window"
            streaks={performance.streaks}
            hasHistory={performance.has_settled_history}
            emptyMessage="No Best Bets have settled yet - streak windows will populate as picks are graded."
          />
        </>
      )}
    </div>
  );
}
