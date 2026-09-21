import { formatProbabilityPct, formatUnits } from "@/lib/format";
import type { CategoryRecordOut } from "@/lib/types";

export default function RecordCard({ title, record }: { title: string; record: CategoryRecordOut }) {
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
