import type { Metadata } from "next";
import { getScheduleForWeek, getScheduleWeeks } from "@/lib/api";
import GameCard from "@/components/GameCard";
import WeekTabs from "@/components/WeekTabs";
import AdSlot from "@/components/AdSlot";

export const metadata: Metadata = {
  title: "NFL Schedule",
  description: "Every NFL game this season, week by week - model, market, and decision data for each matchup.",
};

function parseWeek(raw: string | string[] | undefined, weeks: number[], fallback: number | null): number | null {
  const n = Array.isArray(raw) ? Number(raw[0]) : Number(raw);
  return Number.isInteger(n) && weeks.includes(n) ? n : fallback;
}

export default async function SchedulePage({ searchParams }: PageProps<"/nfl/schedule">) {
  const params = await searchParams;

  let weeksInfo;
  try {
    weeksInfo = await getScheduleWeeks();
  } catch {
    weeksInfo = null;
  }

  if (!weeksInfo) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <h1 className="mb-2 text-4xl font-black tracking-tight">NFL Schedule</h1>
        <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center text-muted">
          Schedule data is temporarily unavailable.
        </div>
      </div>
    );
  }

  const activeWeek = parseWeek(params.week, weeksInfo.weeks, weeksInfo.current_week ?? weeksInfo.weeks[0] ?? null);

  let slate;
  try {
    slate = activeWeek !== null ? await getScheduleForWeek(activeWeek) : null;
  } catch {
    slate = null;
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <h1 className="mb-2 text-4xl font-black tracking-tight">NFL Schedule</h1>
      <p className="mb-6 text-muted">Season {weeksInfo.season} - every game stays reachable here, whether it&apos;s upcoming or long final.</p>

      <WeekTabs weeks={weeksInfo.weeks} activeWeek={activeWeek ?? weeksInfo.weeks[0]} />

      <AdSlot label="schedule-page-top" className="mb-8 h-24" />

      {!slate ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center text-muted">
          Week data is temporarily unavailable.
        </div>
      ) : slate.games.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-muted p-8 text-center text-muted">
          No games found for this week.
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {slate.games.map((game) => (
            <GameCard key={game.game_id} game={game} />
          ))}
        </div>
      )}
    </div>
  );
}
