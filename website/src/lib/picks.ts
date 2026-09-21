import { formatMoneyline, formatSpread } from "./format";
import type { PickOut } from "./types";

export function selectionLineText(pick: PickOut): string {
  const teamAbbr = pick.selection_team?.abbr ?? pick.selection;
  if (pick.market_type === "spread") {
    // `line` is stored in the HOME team's traditional sign (settlement convention) -
    // flip it for an "away" selection so the displayed number matches the selected side.
    const displayLine = pick.line === null ? null : pick.selection === "home" ? pick.line : -pick.line;
    return formatSpread(displayLine, teamAbbr);
  }
  return `${teamAbbr} ML ${formatMoneyline(pick.price)}`;
}

export function matchupText(pick: PickOut): string {
  if (!pick.home_team || !pick.away_team) return pick.game_id;
  return `${pick.away_team.abbr} @ ${pick.home_team.abbr}`;
}

const VOID_REASON_LABELS: Record<string, string> = {
  CORRUPTED_INPUT_DETECTED_PRE_EVENT: "Entered in error and corrected before kickoff",
  DUPLICATE_PUBLICATION: "Duplicate entry",
  GAME_CANCELLED: "Game cancelled",
  SPORTSBOOK_MARKET_VOIDED: "Sportsbook voided the market",
};

export function voidReasonLabel(reason: string | null): string {
  return (reason && VOID_REASON_LABELS[reason]) || "Voided before kickoff";
}
