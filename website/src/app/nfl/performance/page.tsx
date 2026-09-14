import type { Metadata } from "next";
import { getPerformance } from "@/lib/api";
import AdSlot from "@/components/AdSlot";
import { formatProbabilityPct, formatUnits } from "@/lib/format";
import type { CategoryRecordOut, StreakOut } from "@/lib/types";

export const metadata: Metadata = {
  title: "NFL Prediction Model Record",
  description: "Our verified, price-aware Best Bets track record - straight from the immutable official pick ledger, never cherry-picked.",
};

const WINDOW_LABELS: Record<string, string> = {
  last_5: "Last 5",
  last_10: "Last 10",
  last_20: "Last 20",
  last_30: "Last 30",
  season_to_date: "Season to date",
  current_streak: "Current streak",
};

function RecordCard({ title, record }: { title: string; record: CategoryRecordOut }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-6 shadow-sm">
      <h3 className="mb-4 text-lg font-black tracking-tight">{title}</h3>
      {record.n_settled === 0 ? (
        <p className="text-sm text-muted">No settled results yet.</p>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <div className="text-xs text-muted">Record</div>
            <div className="font-mono text-xl font-bold">
              {record.wins}-{record.losses}
              {record.pushes > 0 && `-${record.pushes}`}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted">Win rate</div>
            <div className="font-mono text-xl font-bold">{formatProbabilityPct(record.win_rate)}</div>
          </div>
          <div>
            <div className="text-xs text-muted">Units</div>
            <div className="font-mono text-xl font-bold">{formatUnits(record.total_units)}</div>
          </div>
          <div>
            <div className="text-xs text-muted">ROI / bet</div>
            <div className="font-mono text-xl font-bold">{record.roi_per_bet !== null ? formatUnits(record.roi_per_bet) : "—"}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function StreakRow({ streak }: { streak: StreakOut }) {
  return (
    <tr className="border-b border-border last:border-0">
      <td className="py-2 pr-4 text-sm font-medium">{WINDOW_LABELS[streak.window] ?? streak.window}</td>
      <td className="py-2 pr-4 font-mono text-sm">{streak.n === 0 ? "—" : `${streak.wins}-${streak.losses}${streak.pushes ? `-${streak.pushes}` : ""}`}</td>
      <td className="py-2 pr-4 font-mono text-sm">{formatProbabilityPct(streak.win_rate)}</td>
      <td className="py-2 font-mono text-sm">{streak.total_units !== null ? formatUnits(streak.total_units) : "—"}</td>
    </tr>
  );
}

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

          <div className="rounded-xl border border-border bg-surface p-6 shadow-sm">
            <h3 className="mb-4 text-lg font-black tracking-tight">Best Bets by Window</h3>
            {!performance.has_settled_history ? (
              <p className="text-sm text-muted">No Best Bets have settled yet - streak windows will populate as picks are graded.</p>
            ) : (
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted">
                    <th className="pb-2 font-semibold">Window</th>
                    <th className="pb-2 font-semibold">Record</th>
                    <th className="pb-2 font-semibold">Win rate</th>
                    <th className="pb-2 font-semibold">Units</th>
                  </tr>
                </thead>
                <tbody>
                  {performance.streaks.map((s) => (
                    <StreakRow key={s.window} streak={s} />
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}
