import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { ApiError, getGameDetail } from "@/lib/api";
import DecisionBadge from "@/components/DecisionBadge";
import ReasonCodeList from "@/components/ReasonCodeList";
import AdSlot from "@/components/AdSlot";
import GamePreview from "@/components/GamePreview";
import {
  formatKickoff,
  formatMargin,
  formatMoneyline,
  formatProbabilityPct,
  formatSpread,
  formatTimestamp,
} from "@/lib/format";
import type { GameDetail } from "@/lib/types";

async function loadGame(gameId: string): Promise<GameDetail> {
  try {
    return await getGameDetail(gameId);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      notFound();
    }
    throw err;
  }
}

export async function generateMetadata({ params }: PageProps<"/nfl/games/[gameId]">): Promise<Metadata> {
  const { gameId } = await params;
  try {
    const game = await getGameDetail(gameId);
    const title = `${game.away_team.name} vs ${game.home_team.name} Prediction`;
    return {
      title,
      description: `Model prediction, market line, and betting decision for ${game.away_team.name} at ${game.home_team.name} - Week ${game.week}.`,
      alternates: { canonical: `/nfl/games/${gameId}` },
    };
  } catch {
    return { title: "NFL Game Prediction" };
  }
}

function DisagreementNote({ points }: { points: number | null }) {
  if (points === null) return null;
  const magnitude = Math.abs(points);
  if (magnitude < 0.5) return <p className="text-sm text-muted">The model and market are closely aligned.</p>;
  return (
    <p className="text-sm text-muted">
      The model and market disagree by about <span className="font-semibold text-foreground">{magnitude.toFixed(1)} points</span>.
    </p>
  );
}

export default async function GameDetailPage({ params }: PageProps<"/nfl/games/[gameId]">) {
  const { gameId } = await params;
  const game = await loadGame(gameId);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <p className="mb-1 font-mono text-sm font-medium text-muted">{formatKickoff(game.kickoff_timestamp)}</p>
      <h1 className="mb-6 text-3xl font-black tracking-tight sm:text-4xl">
        {game.away_team.name} <span className="text-muted">at</span> {game.home_team.name}
      </h1>

      <section className="mb-8 rounded-xl border border-border bg-surface p-5 shadow-sm">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-xs font-bold uppercase tracking-wider text-muted">System Pick</h2>
          {game.system_pick.is_also_best_bet && (
            <span className="rounded-full bg-qualified/15 px-2.5 py-1 text-xs font-bold uppercase tracking-wide text-qualified">
              ✓ Best Bet
            </span>
          )}
        </div>
        {game.system_pick.available ? (
          <p className="font-mono text-2xl font-black tracking-tight">
            {game.system_pick.selection_team?.abbr} {formatMoneyline(game.system_pick.price)}
          </p>
        ) : (
          <p className="text-sm text-muted">Not available for this game yet.</p>
        )}
        <p className="mt-1 text-xs text-muted">
          The model&apos;s own straight-up pick, at the real market price - published for every game. Not every
          pick clears our bar to become an official Best Bet.
        </p>
      </section>

      <AdSlot label="game-page-top" className="mb-8 h-24" />

      <GamePreview preview={game.preview} />

      <section className="mb-8">
        <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted">Model Prediction</h2>
        <div className="rounded-xl border border-border bg-surface p-5 shadow-sm">
          {game.model.available ? (
            <>
              <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label={`${game.home_team.abbr} win prob.`} value={formatProbabilityPct(game.model.elo_home_win_probability)} />
                <Stat label="Elo margin" value={formatMargin(game.model.elo_predicted_margin)} />
                <Stat label="Ridge margin" value={formatMargin(game.model.ridge_predicted_margin)} />
                <Stat label="LightGBM margin" value={formatMargin(game.model.lightgbm_predicted_margin)} />
              </div>
              <p className="text-sm text-muted">
                {game.model.all_agree_on_direction === true && "All available models agree on the direction of this game."}
                {game.model.all_agree_on_direction === false && "Our models disagree with each other on the direction of this game."}
                {game.model.all_agree_on_direction === null && "Not enough models are available yet to compare agreement."}
              </p>
            </>
          ) : (
            <p className="text-sm text-muted">Model predictions aren&apos;t available for this game yet.</p>
          )}
        </div>
      </section>

      <section className="mb-8">
        <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted">Market</h2>
        <div className="rounded-xl border border-border bg-surface p-5 shadow-sm">
          {game.market.available ? (
            <>
              <div className="mb-3 grid grid-cols-2 gap-4 sm:grid-cols-3">
                <Stat label="Spread" value={formatSpread(game.market.home_spread_traditional, game.home_team.abbr)} />
                <Stat label={`${game.home_team.abbr} moneyline`} value={formatMoneyline(game.market.home_moneyline)} />
                <Stat label="No-vig home prob." value={formatProbabilityPct(game.market.no_vig_home_win_probability)} />
              </div>
              <p className="mb-2 text-xs text-muted">
                Consensus of {game.market.sportsbooks.length} sportsbook{game.market.sportsbooks.length === 1 ? "" : "s"} · Updated {formatTimestamp(game.market.snapshot_timestamp)}
              </p>
              <DisagreementNote points={game.model_market_disagreement_points} />
            </>
          ) : (
            <p className="text-sm text-muted">No market line is available for this game yet.</p>
          )}
        </div>
      </section>

      <section className="mb-8">
        <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted">Research</h2>
        <div className="rounded-xl border border-border bg-surface p-5 shadow-sm">
          {game.research.available ? (
            <>
              <p className="mb-2 font-semibold">{game.research.classification_label}</p>
              {game.research.summary && <p className="mb-3 text-sm text-muted">{game.research.summary}</p>}
              {game.research.unresolved_risks.length > 0 && (
                <>
                  <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">Still unresolved</p>
                  <ul className="mb-2 list-inside list-disc space-y-1 text-sm text-muted">
                    {game.research.unresolved_risks.map((risk) => (
                      <li key={risk}>{risk}</li>
                    ))}
                  </ul>
                </>
              )}
              <p className="text-xs text-muted">As of {formatTimestamp(game.research.research_timestamp)}</p>
            </>
          ) : (
            <p className="text-sm text-muted">{game.research.failure_label ?? "No research is available for this game yet."}</p>
          )}
        </div>
      </section>

      <AdSlot label="game-page-sidebar" className="mb-8 h-24" />

      <section className="mb-8">
        <h2 className="mb-1 text-xs font-bold uppercase tracking-wider text-muted">Betting Decision</h2>
        <p className="mb-3 text-xs text-muted">
          A predicted winner is not automatically a bet - our decision engine separately weighs the model, the
          market, and current research before qualifying anything.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <DecisionCard title="Spread" decision={game.spread_decision} />
          <DecisionCard title="Moneyline" decision={game.moneyline_decision} />
        </div>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted">{label}</div>
      <div className="font-mono text-lg font-bold">{value}</div>
    </div>
  );
}

function DecisionCard({ title, decision }: { title: string; decision: GameDetail["spread_decision"] }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-5 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-semibold">{title}</span>
        <DecisionBadge decision={decision.decision} label={decision.decision_label} />
      </div>
      {decision.available ? (
        <>
          {decision.is_published_best_bet && (
            <div className="mb-3 inline-flex items-center gap-1.5 rounded-full bg-qualified/15 px-2.5 py-1 text-xs font-semibold text-qualified">
              ✓ Published Best Bet
            </div>
          )}
          <ReasonCodeList codes={decision.reason_codes} />
          <p className="mt-3 text-xs text-muted">As of {formatTimestamp(decision.decision_timestamp)}</p>
          {decision.decision === "QUALIFIED_BET" && !decision.is_published_best_bet && (
            <p className="mt-1 text-xs text-muted">Cleared our automated criteria, but wasn&apos;t published as an official pick for this game.</p>
          )}
        </>
      ) : (
        <p className="text-sm text-muted">No decision has been made for this market yet.</p>
      )}
    </div>
  );
}
