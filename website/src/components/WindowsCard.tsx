import { formatProbabilityPct, formatUnits } from "@/lib/format";
import type { StreakOut } from "@/lib/types";

const WINDOW_LABELS: Record<string, string> = {
  last_5: "Last 5",
  last_10: "Last 10",
  last_20: "Last 20",
  last_30: "Last 30",
  season_to_date: "Season to date",
  current_streak: "Current streak",
};

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

export default function WindowsCard({
  title, streaks, hasHistory, emptyMessage,
}: { title: string; streaks: StreakOut[]; hasHistory: boolean; emptyMessage: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-6 shadow-sm">
      <h3 className="mb-4 text-lg font-black tracking-tight">{title}</h3>
      {!hasHistory ? (
        <p className="text-sm text-muted">{emptyMessage}</p>
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
            {streaks.map((s) => (
              <StreakRow key={s.window} streak={s} />
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
