// Small, pure display-formatting helpers - never a place where a number gets recomputed or
// adjusted, only formatted for display. The value shown always traces back to exactly what
// the API returned.

export function formatMargin(value: number | null): string {
  if (value === null) return "—";
  const rounded = Math.abs(value) < 0.05 ? 0 : value;
  return rounded > 0 ? `+${rounded.toFixed(1)}` : rounded.toFixed(1);
}

export function formatSpread(value: number | null, teamAbbr?: string): string {
  if (value === null) return "No line";
  const sign = value > 0 ? "+" : "";
  const text = `${sign}${value.toFixed(1)}`;
  return teamAbbr ? `${teamAbbr} ${text}` : text;
}

export function formatMoneyline(value: number | null): string {
  if (value === null) return "—";
  return value > 0 ? `+${value}` : `${value}`;
}

export function formatProbabilityPct(value: number | null): string {
  if (value === null) return "—";
  return `${Math.round(value * 100)}%`;
}

export function formatUnits(value: number | null): string {
  if (value === null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}u`;
}

// Every timestamp the API returns is genuine UTC (see nfl_predict.kickoff_time for how
// kickoff specifically gets resolved from the schedule's ambiguous source data) - display
// is fixed to US Pacific here, not the viewer's or server's local zone, so the site reads
// consistently for its owner regardless of who's looking or where this renders.
const DISPLAY_TIME_ZONE = "America/Los_Angeles";

export function formatKickoff(iso: string | null): string {
  if (!iso) return "Kickoff TBD";
  const date = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  if (Number.isNaN(date.getTime())) return "Kickoff TBD";
  return new Intl.DateTimeFormat("en-US", {
    weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
    timeZoneName: "short", timeZone: DISPLAY_TIME_ZONE,
  }).format(date);
}

export function formatTimestamp(iso: string | null): string {
  if (!iso) return "Not yet available";
  const date = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  if (Number.isNaN(date.getTime())) return "Not yet available";
  return new Intl.DateTimeFormat("en-US", {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
    timeZoneName: "short", timeZone: DISPLAY_TIME_ZONE,
  }).format(date);
}

export function winnerLabel(predictedWinner: string | null, homeAbbr: string, awayAbbr: string): string | null {
  if (predictedWinner === "home") return homeAbbr;
  if (predictedWinner === "away") return awayAbbr;
  return null;
}
