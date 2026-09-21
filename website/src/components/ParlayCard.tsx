import { formatMoneyline, formatTimestamp } from "@/lib/format";
import type { ParlayOut } from "@/lib/types";

const RESULT_STYLES: Record<string, string> = {
  WIN: "bg-qualified/15 text-qualified",
  LOSS: "bg-veto/15 text-veto",
  PUSH: "bg-no-bet/15 text-no-bet",
  NOT_GRADED: "bg-surface-muted text-muted",
};

function ResultBadge({ result }: { result: string }) {
  return (
    <span
      className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${
        RESULT_STYLES[result] ?? "bg-surface-muted text-muted"
      }`}
    >
      {result === "NOT_GRADED" ? "Not graded" : result}
    </span>
  );
}

export default function ParlayCard({ parlay }: { parlay: ParlayOut }) {
  return (
    <li className="rounded-xl border border-border bg-surface p-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="font-bold">{parlay.legs.length}-Leg Parlay</span>
        <span className="flex items-center gap-2">
          <span className="font-mono text-sm font-bold">{formatMoneyline(parlay.price)}</span>
          {parlay.settlement && <ResultBadge result={parlay.settlement} />}
        </span>
      </div>

      <ul className="mb-3 space-y-2">
        {parlay.legs.map((leg, i) => (
          <li key={i} className="flex items-start justify-between gap-3 border-l-2 border-border pl-3">
            <div>
              <div className="text-sm font-semibold">{leg.description}</div>
              <div className="text-xs text-muted">
                {leg.matchup}
                {leg.price !== null && ` · ${formatMoneyline(leg.price)}`}
                {leg.detail && ` · ${leg.detail}`}
              </div>
            </div>
            {leg.result && <ResultBadge result={leg.result} />}
          </li>
        ))}
      </ul>

      {parlay.note && <p className="mb-2 text-sm">{parlay.note}</p>}
      <p className="text-xs text-muted">
        {parlay.sportsbook_or_source} · Published {formatTimestamp(parlay.published_at)}
      </p>
    </li>
  );
}
