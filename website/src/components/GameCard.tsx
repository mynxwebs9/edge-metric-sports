import Link from "next/link";
import type { GameCard as GameCardType } from "@/lib/types";
import { formatKickoff, formatProbabilityPct, formatSpread } from "@/lib/format";
import DecisionBadge from "./DecisionBadge";

function projectedMarginText(game: GameCardType): string | null {
  const margin = game.model.elo_predicted_margin;
  if (margin === null) return null;
  const favored = margin >= 0 ? game.home_team.abbr : game.away_team.abbr;
  return `${favored} by ${Math.abs(margin).toFixed(1)}`;
}

function homeWinProbText(game: GameCardType): string | null {
  if (game.model.elo_home_win_probability === null) return null;
  return `${game.home_team.abbr} ${formatProbabilityPct(game.model.elo_home_win_probability)}`;
}

export default function GameCard({ game }: { game: GameCardType }) {
  const projected = projectedMarginText(game);
  const winProb = homeWinProbText(game);

  return (
    <div className="flex flex-col rounded-xl border border-border bg-surface p-5 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono text-xs font-medium uppercase tracking-wide text-muted">{formatKickoff(game.kickoff_timestamp)}</span>
        {game.game_status === "final" && (
          <span className="rounded bg-surface-muted px-2 py-0.5 text-xs font-semibold text-muted">Final</span>
        )}
      </div>

      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="text-center">
          <div className="text-xl font-black tracking-tight">{game.away_team.abbr}</div>
          <div className="text-xs text-muted">{game.away_team.nickname ?? game.away_team.name}</div>
        </div>
        <div className="text-sm font-medium text-muted">@</div>
        <div className="text-center">
          <div className="text-xl font-black tracking-tight">{game.home_team.abbr}</div>
          <div className="text-xs text-muted">{game.home_team.nickname ?? game.home_team.name}</div>
        </div>
      </div>

      <dl className="mb-4 grid grid-cols-2 gap-y-2 text-sm">
        <dt className="text-muted">Model</dt>
        <dd className="text-right font-mono font-semibold">{winProb ?? "Not available"}</dd>
        <dt className="text-muted">Projected</dt>
        <dd className="text-right font-mono font-semibold">{projected ?? "—"}</dd>
        <dt className="text-muted">Market</dt>
        <dd className="text-right font-mono font-semibold">
          {game.market.available ? formatSpread(game.market.home_spread_traditional, game.home_team.abbr) : "No line yet"}
        </dd>
      </dl>

      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-muted">System</span>
        <DecisionBadge decision={game.decision.decision} label={game.decision.decision_label} />
      </div>

      {game.research.available && game.research.summary && (
        <p className="mb-4 line-clamp-2 text-xs text-muted">{game.research.summary}</p>
      )}

      <Link
        href={`/nfl/games/${game.game_id}`}
        className="mt-auto inline-flex items-center gap-1 text-sm font-semibold text-accent hover:underline"
      >
        View Matchup <span aria-hidden>→</span>
      </Link>
    </div>
  );
}
