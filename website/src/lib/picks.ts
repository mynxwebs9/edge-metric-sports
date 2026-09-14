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
