import Link from "next/link";
import { formatMoneyline, formatTimestamp } from "@/lib/format";
import { matchupText, selectionLineText } from "@/lib/picks";
import type { PickOut } from "@/lib/types";

const RESULT_STYLES: Record<string, string> = {
  WIN: "bg-qualified/15 text-qualified",
  LOSS: "bg-veto/15 text-veto",
  PUSH: "bg-no-bet/15 text-no-bet",
};

export default function PickListItem({ pick }: { pick: PickOut }) {
  return (
    <li className="rounded-xl border border-border bg-surface p-5">
      <div className="mb-1 flex items-center justify-between gap-3">
        <span className="font-bold">{selectionLineText(pick)}</span>
        <span className="flex items-center gap-2">
          {pick.market_type === "spread" && <span className="text-sm text-muted">{formatMoneyline(pick.price)}</span>}
          {pick.settlement && (
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${
                RESULT_STYLES[pick.settlement] ?? "bg-surface-muted text-muted"
              }`}
            >
              {pick.settlement}
            </span>
          )}
        </span>
      </div>
      <p className="mb-1 text-xs text-muted">
        {matchupText(pick)} ·{" "}
        <Link href={`/nfl/games/${pick.game_id}`} className="font-medium text-accent hover:underline">
          View matchup
        </Link>
      </p>
      {pick.note && <p className="my-2 text-sm">{pick.note}</p>}
      <p className="text-xs text-muted">
        {pick.sportsbook_or_source} · Published {formatTimestamp(pick.published_at)}
      </p>
    </li>
  );
}
