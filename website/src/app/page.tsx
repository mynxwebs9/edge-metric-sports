import Link from "next/link";
import { getBestBets, getCurrentSlate, getPerformance } from "@/lib/api";
import GameCard from "@/components/GameCard";
import PerformanceHeadline from "@/components/PerformanceHeadline";
import AdSlot from "@/components/AdSlot";
import { formatTimestamp, formatUnits } from "@/lib/format";
import { matchupText, selectionLineText } from "@/lib/picks";
import type { BestBetsResponse, PerformanceResponse, SlateResponse } from "@/lib/types";

async function safeFetch<T>(fn: () => Promise<T>): Promise<T | null> {
  try {
    return await fn();
  } catch {
    return null;
  }
}

export default async function HomePage() {
  const [slate, bestBets, performance] = await Promise.all([
    safeFetch<SlateResponse>(getCurrentSlate),
    safeFetch<BestBetsResponse>(getBestBets),
    safeFetch<PerformanceResponse>(getPerformance),
  ]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <section className="mb-10 text-center">
        <p className="mb-3 text-xs font-bold uppercase tracking-[0.2em] text-accent">Data-Driven NFL Picks</p>
        <h1 className="mb-3 text-4xl font-black tracking-tight sm:text-5xl">NFL Predictions</h1>
        <p className="mb-4 font-mono text-sm text-muted">
          {slate ? `${slate.season} · Week ${slate.week ?? "—"}` : "Current slate"}
          {slate?.last_updated && <> · Updated {formatTimestamp(slate.last_updated)}</>}
        </p>
        {performance ? (
          <PerformanceHeadline performance={performance} />
        ) : (
          <p className="text-sm text-muted">Performance data is temporarily unavailable.</p>
        )}
      </section>

      <AdSlot label="homepage-hero-below" className="mb-10 h-24" />

      <section className="mb-12">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-2xl font-black tracking-tight">Best Bets</h2>
          <Link href="/nfl/picks" className="text-sm font-semibold text-accent hover:underline">
            See all →
          </Link>
        </div>
        {bestBets && bestBets.picks.length > 0 ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {bestBets.picks.slice(0, 3).map((pick) => (
              <div key={pick.pick_id} className="rounded-xl border border-border bg-surface p-4 shadow-sm">
                <p className="font-mono text-sm font-bold">{selectionLineText(pick)}</p>
                <p className="text-xs text-muted">{matchupText(pick)}</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-border bg-surface-muted p-6 text-center text-sm text-muted">
            {bestBets?.message ?? "No plays currently meet our qualification criteria."}
          </div>
        )}
      </section>

      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-black tracking-tight">Current NFL Slate</h2>
        {slate && slate.games.length > 0 ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {slate.games.map((game) => (
              <GameCard key={game.game_id} game={game} />
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-border bg-surface-muted p-6 text-center text-sm text-muted">
            No games in the current slate right now.
          </div>
        )}
      </section>

      <AdSlot label="homepage-between-sections" className="mb-12 h-24" />

      <section className="mb-12 grid gap-4 sm:grid-cols-2">
        <Link href="/nfl/performance" className="rounded-xl border border-border bg-surface p-6 shadow-sm transition-colors hover:border-accent">
          <h3 className="mb-1 text-lg font-black tracking-tight">Model Record</h3>
          <p className="font-mono text-sm text-muted">
            {performance?.has_settled_history
              ? `${performance.best_bets_record.wins}-${performance.best_bets_record.losses} Best Bets, ${formatUnits(performance.best_bets_record.total_units)}`
              : "No settled picks yet — see the full, verified methodology."}
          </p>
        </Link>
        <Link href="/methodology" className="rounded-xl border border-border bg-surface p-6 shadow-sm transition-colors hover:border-accent">
          <h3 className="mb-1 text-lg font-black tracking-tight">How It Works</h3>
          <p className="text-sm text-muted">Independent models, market comparison, research, and a deterministic decision engine.</p>
        </Link>
      </section>
    </div>
  );
}
