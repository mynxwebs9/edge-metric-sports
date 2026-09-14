import Link from "next/link";

export default function WeekTabs({ weeks, activeWeek }: { weeks: number[]; activeWeek: number }) {
  return (
    <div className="mb-6 flex gap-1.5 overflow-x-auto pb-2">
      {weeks.map((week) => (
        <Link
          key={week}
          href={`/nfl/schedule?week=${week}`}
          className={
            week === activeWeek
              ? "shrink-0 rounded-full bg-accent px-3.5 py-1.5 text-sm font-semibold text-accent-foreground"
              : "shrink-0 rounded-full border border-border bg-surface px-3.5 py-1.5 text-sm font-medium text-muted hover:border-accent hover:text-foreground"
          }
        >
          Week {week}
        </Link>
      ))}
    </div>
  );
}
