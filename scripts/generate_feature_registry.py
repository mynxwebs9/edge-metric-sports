"""Generates config/features.yaml from the authoritative metric definitions in
src/nfl_predict/features/team_game.py (WINDOWED_METRICS, SEASON_ONLY_METRICS,
SPECIAL_TEAMS_METRICS) plus the smaller hand-defined QB/personnel/situational/opponent/
prev-season/sample-size entries below.

This is the SOURCE OF TRUTH generator, not a one-time throwaway: if a metric definition in
team_game.py ever changes, re-run this script rather than hand-editing config/features.yaml,
so the registry can never silently drift from what the code actually computes. Run:

    python scripts/generate_feature_registry.py

Identity/bookkeeping columns (game_id, season, week, team_id, opponent_id, venue,
kickoff_time_naive, as_of_timestamp, game_date, season_type) and `is_home` are deliberately
NOT registered here - see docs/FEATURE_DICTIONARY.md's "Identity & context columns"
section. `is_home` is structurally definitional (it's what makes a row the home-side or
away-side view of a game) even though it also functions as the home-field-advantage signal;
registering it with a formula/rolling-window/leakage-notes treatment it doesn't need would
be exactly the "identity column treated as a predictive feature" anti-pattern the Phase 2
brief warns against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nfl_predict.features.team_game import (  # noqa: E402
    SEASON_ONLY_METRICS,
    SPECIAL_TEAMS_METRICS,
    WINDOWED_METRICS,
)

WINDOWS = [3, 5, 8]

# ---------------------------------------------------------------------------
# Per-base-metric metadata used to expand WINDOWED_METRICS / SEASON_ONLY_METRICS /
# SPECIAL_TEAMS_METRICS into full registry entries. Keyed by MetricSpec.output_name.
# ---------------------------------------------------------------------------

METRIC_META: dict[str, dict] = {
    # --- Windowed core efficiency (offense) ---
    "off_epa_pp": dict(category="offense", unit="epa_per_play", garbage_time_filtered=False,
        description="Offensive expected points added (EPA) per play, averaged over offensive plays (pass or designed run attempts; excludes kickoffs, punts, field goals, extra points, qb_kneel, and qb_spike).",
        formula="sum(epa) / count(plays) where play_type in (pass, run)"),
    "off_success_rate": dict(category="offense", unit="rate_0_1", garbage_time_filtered=False,
        description="Share of offensive plays classified as a 'success' by nflverse's down-and-distance success rule (not simply EPA > 0).",
        formula="sum(success) / count(plays) where play_type in (pass, run)"),
    "off_pass_epa_dropback": dict(category="passing", unit="epa_per_dropback", garbage_time_filtered=False,
        description="Offensive EPA per dropback (pass attempts, sacks, and scrambles - qb_dropback == 1).",
        formula="sum(epa) / count(dropbacks) where qb_dropback == 1"),
    "off_pass_success_rate": dict(category="passing", unit="rate_0_1", garbage_time_filtered=False,
        description="Success rate on dropbacks (pass attempts, sacks, and scrambles).",
        formula="sum(success) / count(dropbacks) where qb_dropback == 1"),
    "off_rush_epa": dict(category="rushing", unit="epa_per_play", garbage_time_filtered=False,
        description="Offensive EPA per DESIGNED rush attempt. QB scrambles are excluded - they are attributed to the quarterback context features instead (qb_scramble_epa_season), not to the run game, since a scramble originates from a passing down. See docs/PHASE2_FEATURE_REPORT.md#offensive-features.",
        formula="sum(epa) / count(rush_attempt==1 & qb_scramble==0)"),
    "off_rush_success_rate": dict(category="rushing", unit="rate_0_1", garbage_time_filtered=False,
        description="Success rate on designed rush attempts (scrambles excluded - see off_rush_epa).",
        formula="sum(success) / count(rush_attempt==1 & qb_scramble==0)"),
    "off_epa_pp_comp": dict(category="offense", unit="epa_per_play", garbage_time_filtered=True,
        description="COMPETITIVE-PLAY variant of off_epa_pp: identical definition, but restricted to plays where the pre-play home win probability was between garbage_time_wp_threshold and 1 - garbage_time_wp_threshold (see config/feature_engine.yaml). Excludes plays from lopsided ('garbage time') game states.",
        formula="sum(epa) / count(plays) where play_type in (pass, run) AND home_wp in [threshold, 1-threshold] (pre-play)"),
    "off_success_rate_comp": dict(category="offense", unit="rate_0_1", garbage_time_filtered=True,
        description="COMPETITIVE-PLAY variant of off_success_rate - see off_epa_pp_comp for the exact filter.",
        formula="sum(success) / count(plays) where play_type in (pass, run) AND home_wp in [threshold, 1-threshold] (pre-play)"),
    # --- Windowed core efficiency (defense) ---
    "def_epa_pp_allowed": dict(category="defense", unit="epa_per_play", garbage_time_filtered=False,
        description="Defensive EPA allowed per play (opponent's offensive plays against this team's defense; same play-type filter as off_epa_pp).",
        formula="sum(epa) / count(plays) where play_type in (pass, run), grouped by defending team"),
    "def_success_rate_allowed": dict(category="defense", unit="rate_0_1", garbage_time_filtered=False,
        description="Success rate allowed per defensive snap (opponent's plays).",
        formula="sum(success) / count(plays) where play_type in (pass, run), grouped by defending team"),
    "def_pass_epa_allowed": dict(category="defense", unit="epa_per_dropback", garbage_time_filtered=False,
        description="EPA allowed per opponent dropback (pass defense).",
        formula="sum(epa) / count(dropbacks), grouped by defending team, where qb_dropback == 1"),
    "def_pass_success_rate_allowed": dict(category="defense", unit="rate_0_1", garbage_time_filtered=False,
        description="Success rate allowed per opponent dropback (pass defense).",
        formula="sum(success) / count(dropbacks), grouped by defending team, where qb_dropback == 1"),
    "def_rush_epa_allowed": dict(category="defense", unit="epa_per_play", garbage_time_filtered=False,
        description="EPA allowed per opponent DESIGNED rush attempt (rush defense; opponent scrambles excluded, same convention as off_rush_epa).",
        formula="sum(epa) / count(rush_attempt==1 & qb_scramble==0), grouped by defending team"),
    "def_rush_success_rate_allowed": dict(category="defense", unit="rate_0_1", garbage_time_filtered=False,
        description="Success rate allowed per opponent designed rush attempt (rush defense).",
        formula="sum(success) / count(rush_attempt==1 & qb_scramble==0), grouped by defending team"),

    # --- Season-only: explosive plays ---
    "off_explosive_pass_rate": dict(category="explosive_plays", unit="rate_0_1", garbage_time_filtered=False,
        description="Share of true pass attempts (excludes scrambles) that were a COMPLETED pass gaining >= 20 yards (explosive_pass_yards_gained in config/feature_engine.yaml). A conventional, explicitly documented threshold - nflverse provides no canonical 'explosive' flag.",
        formula="count(pass_attempt==1 & complete_pass==1 & yards_gained>=20) / count(pass_attempt==1)"),
    "off_explosive_rush_rate": dict(category="explosive_plays", unit="rate_0_1", garbage_time_filtered=False,
        description="Share of designed rush attempts gaining >= 10 yards (explosive_rush_yards_gained). Conventional, documented threshold; scrambles excluded (see off_rush_epa).",
        formula="count(rush_attempt==1 & qb_scramble==0 & yards_gained>=10) / count(rush_attempt==1 & qb_scramble==0)"),
    "def_explosive_pass_rate_allowed": dict(category="explosive_plays", unit="rate_0_1", garbage_time_filtered=False,
        description="Explosive-completion rate allowed (defense) - same threshold as off_explosive_pass_rate.",
        formula="count(pass_attempt==1 & complete_pass==1 & yards_gained>=20) / count(pass_attempt==1), grouped by defending team"),
    "def_explosive_rush_rate_allowed": dict(category="explosive_plays", unit="rate_0_1", garbage_time_filtered=False,
        description="Explosive-rush rate allowed (defense) - same threshold as off_explosive_rush_rate.",
        formula="count(rush_attempt==1 & qb_scramble==0 & yards_gained>=10) / count(rush_attempt==1 & qb_scramble==0), grouped by defending team"),

    # --- Season-only: pressure/sacks ---
    "off_sack_rate_allowed": dict(category="pressure_sacks", unit="rate_0_1", garbage_time_filtered=False,
        description="Sacks allowed per true pass attempt. In nflverse's schema, `pass_attempt==1` already covers completions, incompletions, interceptions, AND sacks (a known nflverse-specific convention, distinct from official box-score 'pass attempts' - see docs/PHASE1_DATA_REPORT.md); this is the correct denominator for sack rate, not qb_dropback (which also includes scrambles, on which a sack cannot occur by definition).",
        formula="count(sack==1) / count(pass_attempt==1)"),
    "off_qb_hit_rate_allowed": dict(category="pressure_sacks", unit="rate_0_1", garbage_time_filtered=False,
        description="QB hits allowed per true pass attempt. A proxy for pressure allowed - nflverse does not provide a true pressure-rate or pass-block-win-rate statistic, and none is invented here.",
        formula="count(pass_attempt==1 & qb_hit==1) / count(pass_attempt==1)"),
    "def_sack_rate_generated": dict(category="pressure_sacks", unit="rate_0_1", garbage_time_filtered=False,
        description="Sacks generated per opponent true pass attempt (defense).",
        formula="count(sack==1) / count(pass_attempt==1), grouped by defending team"),
    "def_qb_hit_rate_generated": dict(category="pressure_sacks", unit="rate_0_1", garbage_time_filtered=False,
        description="QB hits generated per opponent true pass attempt (defense pressure proxy).",
        formula="count(pass_attempt==1 & qb_hit==1) / count(pass_attempt==1), grouped by defending team"),

    # --- Season-only: situational ---
    "off_early_down_epa": dict(category="offense", unit="epa_per_play", garbage_time_filtered=False,
        description="Offensive EPA per play on 1st and 2nd down (early-down efficiency).",
        formula="sum(epa) / count(plays) where play_type in (pass, run) AND down in (1,2)"),
    "off_early_down_success_rate": dict(category="offense", unit="rate_0_1", garbage_time_filtered=False,
        description="Offensive success rate on 1st and 2nd down.",
        formula="sum(success) / count(plays) where play_type in (pass, run) AND down in (1,2)"),
    "off_third_down_conv_rate": dict(category="offense", unit="rate_0_1", garbage_time_filtered=False,
        description="Third-down conversion rate. Treat with caution - third-down samples per game are small (typically 10-15 attempts) and highly variable early in a season; this is NOT presented as an inherently stable predictive measure merely because it's calculable. See docs/PHASE2_FEATURE_REPORT.md's situational-stats caveat.",
        formula="count(third_down_converted==1) / count(down==3)"),
    "off_redzone_td_rate": dict(category="offense", unit="rate_0_1", garbage_time_filtered=False,
        description="Share of red-zone drive trips (any play in the drive with yardline_100 <= 20, per config/feature_engine.yaml) ending in a touchdown. Drive-level, not play-level. Small samples early in a season - same caution as off_third_down_conv_rate.",
        formula="count(drives with fixed_drive_result=='Touchdown' among those reaching yardline_100<=20) / count(drives reaching yardline_100<=20)"),
    "def_early_down_epa_allowed": dict(category="defense", unit="epa_per_play", garbage_time_filtered=False,
        description="EPA allowed per play on opponent 1st/2nd down (early-down defense).",
        formula="sum(epa) / count(plays) where play_type in (pass, run) AND down in (1,2), grouped by defending team"),
    "def_early_down_success_rate_allowed": dict(category="defense", unit="rate_0_1", garbage_time_filtered=False,
        description="Success rate allowed on opponent 1st/2nd down.",
        formula="sum(success) / count(plays) where play_type in (pass, run) AND down in (1,2), grouped by defending team"),
    "def_third_down_allowed": dict(category="defense", unit="rate_0_1", garbage_time_filtered=False,
        description="Opponent third-down conversion rate allowed. Same small-sample caution as off_third_down_conv_rate.",
        formula="count(third_down_converted==1) / count(down==3), grouped by defending team"),
    "def_redzone_td_rate_allowed": dict(category="defense", unit="rate_0_1", garbage_time_filtered=False,
        description="Opponent red-zone-trip touchdown rate allowed. Same small-sample caution as off_redzone_td_rate.",
        formula="count(opponent drives ending in TD among red-zone trips) / count(opponent red-zone trips)"),

    # --- Season-only: yards-based ---
    "off_yards_per_dropback": dict(category="passing", unit="yards_per_dropback", garbage_time_filtered=False,
        description="Raw yards gained per dropback (includes sack yardage losses, since yards_gained reflects the actual play result).",
        formula="sum(yards_gained) / count(dropbacks) where qb_dropback == 1"),
    "off_yards_per_rush": dict(category="rushing", unit="yards_per_play", garbage_time_filtered=False,
        description="Raw yards gained per designed rush attempt (scrambles excluded).",
        formula="sum(yards_gained) / count(rush_attempt==1 & qb_scramble==0)"),

    # --- Season-only: turnovers ---
    "off_int_rate": dict(category="turnovers", unit="rate_0_1", garbage_time_filtered=False,
        description="Interceptions thrown per true pass attempt (offense giveaway - process/opportunity view, since it's rate-per-attempt rather than a raw count).",
        formula="count(interception==1) / count(pass_attempt==1)"),
    "off_fumble_lost_rate": dict(category="turnovers", unit="rate_0_1", garbage_time_filtered=False,
        description="Fumbles lost per offensive play (offense giveaway).",
        formula="count(fumble_lost==1) / count(plays) where play_type in (pass, run)"),
    "def_int_rate_generated": dict(category="turnovers", unit="rate_0_1", garbage_time_filtered=False,
        description="Interceptions generated per opponent true pass attempt (defense takeaway).",
        formula="count(interception==1) / count(pass_attempt==1), grouped by defending team"),
    "def_fumble_forced_rate": dict(category="turnovers", unit="rate_0_1", garbage_time_filtered=False,
        description="Opponent fumbles lost (i.e. recovered by this team's defense) per opponent offensive play (defense takeaway).",
        formula="count(fumble_lost==1) / count(plays) where play_type in (pass, run), grouped by defending team"),

    # --- Season-only: CPOE ---
    "off_cpoe": dict(category="passing", unit="percentage_points", garbage_time_filtered=False,
        description="Completion percentage over expected (CPOE), averaged over pass attempts where nflverse's model produced a value (sacks/scrambles/spikes are naturally excluded since CPOE is undefined for them).",
        formula="sum(cpoe) / count(cpoe is not null)"),

    # --- Special teams ---
    "fg_pct": dict(category="special_teams", unit="rate_0_1", garbage_time_filtered=False,
        description="Field goal make percentage, season-to-date.",
        formula="count(field_goal_attempt==1 & field_goal_result=='made') / count(field_goal_attempt==1)"),
    "punt_yards_avg": dict(category="special_teams", unit="yards_per_punt", garbage_time_filtered=False,
        description="Average GROSS punt distance (kick_distance), season-to-date. NOT net of return yardage - true net punting (accounting for touchbacks, fair catches, out-of-bounds, and return yards) would need reconciling several edge cases this project isn't leaning on yet; documented as a known limitation rather than silently approximated. Return performance is deferred entirely.",
        formula="sum(kick_distance) / count(kick_distance is not null) where play_type == 'punt'"),
}

# category -> (opponent_adjusted, earliest_reliable_season, extra_known_limitations)
CATEGORY_DEFAULTS = {
    "offense": (False, 1999, None),
    "passing": (False, 1999, None),
    "rushing": (False, 1999, None),
    "defense": (False, 1999, None),
    "explosive_plays": (False, 1999, None),
    "pressure_sacks": (False, 1999, None),
    "turnovers": (False, 1999, None),
    "special_teams": (False, 1999, None),
}


def _window_suffix_meta(base_output_name: str, window_label: str) -> dict:
    if window_label == "season":
        return dict(
            rolling_window="season-to-date (resets each season; excludes the current game)",
            leakage_extra="Season-to-date: uses only this team's games strictly before the current one WITHIN THE SAME SEASON; resets to empty at the first game of each season.",
        )
    n = window_label.rstrip("g")
    return dict(
        rolling_window=f"trailing {n} games (does not reset at season boundaries; a Week 1 game's trailing window may include games from the end of the prior season)",
        leakage_extra=f"Trailing {n}-game window: uses this team's {n} most recent completed games strictly before the current one, computed as a rolled sum of the window's raw event counts/EPA divided by the rolled play count (not an average of per-game rates), so games with more plays are weighted correctly.",
    )


def _min_observations_for(window_label: str) -> int:
    return 1  # config/feature_engine.yaml: min_observations_rolling == 1 for both tiers


def make_windowed_entries() -> list[dict]:
    entries = []
    for spec in WINDOWED_METRICS:
        meta = METRIC_META[spec.output_name]
        for window_label in [f"{w}g" for w in WINDOWS] + ["season"]:
            wmeta = _window_suffix_meta(spec.output_name, window_label)
            name = f"{spec.output_name}_{window_label}"
            entries.append(dict(
                name=name,
                category=meta["category"],
                description=meta["description"],
                source_datasets=["pbp"],
                formula=meta["formula"],
                unit=meta["unit"],
                rolling_window=wmeta["rolling_window"],
                min_observations=_min_observations_for(window_label),
                pregame_availability=(
                    "Computable once the minimum observation count of prior qualifying games/plays "
                    "is met; null otherwise (see missing_value_policy)."
                ),
                missing_value_policy=(
                    "Null (never zero) if the team has fewer than min_observations qualifying prior "
                    f"games in this window, OR if the window's underlying play count is zero. Sample "
                    f"size is visible via the shared {'n_games_trailing_' + window_label.rstrip('g') if window_label != 'season' else 'games_played_current_season'} column."
                ),
                opponent_adjusted=False,
                garbage_time_filtered=meta["garbage_time_filtered"],
                earliest_reliable_season=1999,
                known_limitations=(
                    "Early in a season (or early in a team's trailing window), sample size is small; "
                    "does not imply the underlying rate is unstable, but confidence should scale with "
                    "the sample-size column. " + ("Restricted to competitive game states only - see garbage_time_wp_threshold in config/feature_engine.yaml." if meta["garbage_time_filtered"] else "")
                ).strip(),
                leakage_notes=wmeta["leakage_extra"] + " Never includes the current game's own plays.",
                model_tracks=["independent"],
            ))
    return entries


def make_season_only_entries(metrics, source_datasets=("pbp",)) -> list[dict]:
    entries = []
    for spec in metrics:
        meta = METRIC_META[spec.output_name]
        name = f"{spec.output_name}_season"
        entries.append(dict(
            name=name,
            category=meta["category"],
            description=meta["description"],
            source_datasets=list(source_datasets),
            formula=meta["formula"],
            unit=meta["unit"],
            rolling_window="season-to-date (resets each season; excludes the current game)",
            min_observations=1,
            pregame_availability="Computable once at least one qualifying prior game/play exists this season; null otherwise.",
            missing_value_policy="Null (never zero) if no qualifying prior plays exist this season yet (e.g. Week 1, or a team with zero red-zone trips so far).",
            opponent_adjusted=False,
            garbage_time_filtered=meta["garbage_time_filtered"],
            earliest_reliable_season=1999,
            known_limitations=(
                "Small-sample situational stat (third down / red zone / turnovers can be single-digit "
                "counts early in a season) - do not treat as a stable rate without checking the "
                "underlying sample size." if meta["category"] in ("offense", "defense", "turnovers") else
                "Season-to-date only; no trailing-window variant is provided for this metric (see docs/PHASE2_FEATURE_REPORT.md's Feature windows rationale)."
            ),
            leakage_notes=(
                "Season-to-date: uses only this team's games strictly before the current one within "
                "the same season; resets at season boundaries; never includes the current game's own plays."
            ),
            model_tracks=["independent"],
        ))
    return entries


# ---------------------------------------------------------------------------
# Hand-defined entries: QB, personnel, situational, opponent, previous-season,
# sample-size/support.
# ---------------------------------------------------------------------------

QB_ENTRIES = [
    dict(name="qb_primary_id", category="quarterback",
         description="The nflverse GSIS player_id of the quarterback identified as PRIMARY (most dropbacks) in this team's PREVIOUS completed game - not this game. Provided for audit/interpretability and as a categorical join key; not itself a numeric rate. See docs/PHASE2_FEATURE_REPORT.md's QB leakage-safety section.",
         source_datasets=["pbp"], formula="passer_player_id (or rusher_player_id on a scramble) with the most qb_dropback==1 plays for this team in the PRIOR game",
         unit="categorical_id", rolling_window="none (single prior game)", min_observations=1,
         pregame_availability="Known once the team has played at least one prior game this season or a prior season; null for a team's very first game in the dataset.",
         missing_value_policy="Null if the team has no prior game (its first game in the entire 2010-2025 window).",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Identifies who played the most, not necessarily the officially designated 'starter' (rare in-game benching could differ from who took the most first-half/most-total snaps, though this is uncommon).",
         leakage_notes="CRITICAL: derived from the PREVIOUS game's own play-by-play, which is safe because that game is fully completed before the target game's kickoff. NEVER derived from the target game's own starter identity - see docs/PHASE2_FEATURE_REPORT.md#qb-features-and-the-leakage-rule.",
         model_tracks=["independent"]),
    dict(name="qb_consecutive_starts", category="quarterback",
         description="Number of consecutive prior games (ending with the team's most recent completed game, within the current season) for which the SAME quarterback was primary. 0 if this is the team's first game of the season or the primary QB just changed.",
         source_datasets=["pbp"], formula="running streak count of qb_primary_id staying constant across consecutive prior games this season",
         unit="count", rolling_window="none (running streak, season-scoped)", min_observations=1,
         pregame_availability="0 if no prior game this season; otherwise the streak as of the most recent prior game.",
         missing_value_policy="0 (not null) when there is no prior game this season - a real, meaningful zero (no continuity established yet), unlike a rate metric's missingness.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Resets to 0 at the start of every season, even if the same QB started every game of the prior season - continuity is intentionally not carried across the off-season boundary.",
         leakage_notes="Built forward from completed prior games only; never looks at the target game's own QB.",
         model_tracks=["independent"]),
    dict(name="qb_starts_season", category="sample_size_support",
         description="Number of this team's games so far this season (before the current one) where qb_primary_id was the primary QB. The sample-size denominator context for every other qb_*_season metric.",
         source_datasets=["pbp"], formula="count of the primary QB's own prior primary-QB games this season",
         unit="count", rolling_window="season-to-date", min_observations=0,
         pregame_availability="0 if the primary QB has no prior starts this season (e.g. just became primary).",
         missing_value_policy="0, not null - a real count, including zero.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified", leakage_notes="Counts only prior games; never the current one.",
         model_tracks=["independent"]),
    dict(name="qb_epa_dropback_season", category="quarterback",
         description="Season-to-date EPA per dropback for THIS SPECIFIC QUARTERBACK (qb_primary_id), computed only from his own games as primary QB for this team this season - not the team's overall off_pass_epa_dropback_season, which includes any backup's snaps too.",
         source_datasets=["pbp"], formula="sum(his pass_epa_sum across his own prior primary-QB games) / sum(his dropback_n across those games)",
         unit="epa_per_dropback", rolling_window="season-to-date, scoped to this QB's own starts", min_observations=1,
         pregame_availability="Null until the primary QB has at least 1 prior game as primary this season.",
         missing_value_policy="Null (not zero) if qb_starts_season == 0.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="A backup making his first start has no data here (null) even if the team's own off_pass_epa_dropback_season has plenty of history from the previous starter - this is intentional, not a bug.",
         leakage_notes="Scoped to the identified QB's own PRIOR games only; never his performance in the target game.",
         model_tracks=["independent"]),
    dict(name="qb_success_rate_season", category="quarterback",
         description="Season-to-date dropback success rate for the primary QB specifically (see qb_epa_dropback_season).",
         source_datasets=["pbp"], formula="sum(his pass_success_sum) / sum(his dropback_n) across his own prior primary-QB games this season",
         unit="rate_0_1", rolling_window="season-to-date, scoped to this QB's own starts", min_observations=1,
         pregame_availability="Null until qb_starts_season >= 1.", missing_value_policy="Null if qb_starts_season == 0.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Same new-starter caveat as qb_epa_dropback_season.",
         leakage_notes="Same as qb_epa_dropback_season.", model_tracks=["independent"]),
    dict(name="qb_sack_rate_season", category="quarterback",
         description="Season-to-date sack rate taken by the primary QB specifically, per his own true pass attempts.",
         source_datasets=["pbp"], formula="sum(his sack_n) / sum(his pass_attempt_n) across his own prior primary-QB games this season",
         unit="rate_0_1", rolling_window="season-to-date, scoped to this QB's own starts", min_observations=1,
         pregame_availability="Null until qb_starts_season >= 1.", missing_value_policy="Null if qb_starts_season == 0 or his pass_attempt_n sum is 0.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Conflates QB-specific tendencies (e.g. holding the ball longer) with O-line pass protection - not decomposed in Phase 2.",
         leakage_notes="Same as qb_epa_dropback_season.", model_tracks=["independent"]),
    dict(name="qb_int_rate_season", category="quarterback",
         description="Season-to-date interception rate thrown by the primary QB specifically, per his own true pass attempts.",
         source_datasets=["pbp"], formula="sum(his int_n) / sum(his pass_attempt_n) across his own prior primary-QB games this season",
         unit="rate_0_1", rolling_window="season-to-date, scoped to this QB's own starts", min_observations=1,
         pregame_availability="Null until qb_starts_season >= 1.", missing_value_policy="Null if qb_starts_season == 0.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Interceptions are a noisy outcome even for a full season, let alone the partial-season samples typical early on - see docs/PHASE2_FEATURE_REPORT.md's turnovers caveat.",
         leakage_notes="Same as qb_epa_dropback_season.", model_tracks=["independent"]),
    dict(name="qb_cpoe_season", category="quarterback",
         description="Season-to-date completion percentage over expected for the primary QB specifically.",
         source_datasets=["pbp"], formula="sum(his cpoe_sum) / sum(his cpoe_n) across his own prior primary-QB games this season",
         unit="percentage_points", rolling_window="season-to-date, scoped to this QB's own starts", min_observations=1,
         pregame_availability="Null until qb_starts_season >= 1.", missing_value_policy="Null if qb_starts_season == 0 or he has no plays with a defined CPOE yet.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified beyond the general new-starter null behavior.",
         leakage_notes="Same as qb_epa_dropback_season.", model_tracks=["independent"]),
    dict(name="qb_scramble_epa_season", category="quarterback",
         description="Season-to-date EPA generated specifically on scramble plays by the primary QB (his rushing-while-passing-down contribution), normalized per his total dropbacks (not per scramble) so it reflects overall value added via scrambling rather than scramble efficiency alone.",
         source_datasets=["pbp"], formula="sum(his scramble_epa_sum) / sum(his dropback_n) across his own prior primary-QB games this season",
         unit="epa_per_dropback", rolling_window="season-to-date, scoped to this QB's own starts", min_observations=1,
         pregame_availability="Null until qb_starts_season >= 1.", missing_value_policy="Null if qb_starts_season == 0; a QB who never scrambles will show a value near 0, not null, once he has starts (correctly reflecting a real, small/zero scramble contribution).",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Scramble volume is low for most QBs; this can be a near-zero, low-signal value for pocket passers.",
         leakage_notes="Same as qb_epa_dropback_season.", model_tracks=["independent"]),
]

PERSONNEL_ENTRIES = [
    dict(name=f"{side}_snap_continuity_pct", category="personnel_continuity",
         description=f"Share of the team's {label} snaps in its MOST RECENT completed game that went to players who ALSO played {label} snaps (any amount) in the game before that. A roster-stability signal heading into the next game - never involves the target game's own personnel.",
         source_datasets=["snap_counts"],
         formula=f"sum({side}_snaps in game N-1 for players present in game N-2) / sum({side}_snaps in game N-1)",
         unit="rate_0_1", rolling_window="prior two completed games only (not a multi-game rolling window)",
         min_observations=2,
         pregame_availability="Requires the team to have at least 2 prior completed games with snap_counts coverage (2013+ - see docs/PHASE1_DATA_REPORT.md); null before that.",
         missing_value_policy="Null if fewer than 2 prior games have snap_counts data (includes all of 2010-2012, and a team's first 2 games of 2013).",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=2013,
         known_limitations="snap_counts identifies players by pfr_player_id (Pro-Football-Reference), a different ID namespace than play-by-play's GSIS ids - continuity is computed self-referentially within snap_counts alone, never cross-joined to pbp player ids, precisely to avoid needing an unbuilt ID crosswalk. Position-specific (e.g. offensive-line-only) continuity is deferred - see docs/PHASE2_FEATURE_REPORT.md.",
         leakage_notes="Uses only the team's own two most recent COMPLETED prior games; never the target game's roster.",
         model_tracks=["independent"])
    for side, label in [("off", "offensive"), ("def", "defensive")]
]

SITUATIONAL_ENTRIES = [
    dict(name="days_rest", category="rest_situational",
         description="Days between this team's previous game and this one.",
         source_datasets=["schedules"], formula="this game's kickoff date minus the team's previous game's kickoff date, in days",
         unit="days", rolling_window="none", min_observations=1,
         pregame_availability="Known as soon as the schedule is published - long before kickoff.",
         missing_value_policy="Null if there is no previous game (team's first game in the dataset) or the gap exceeds 30 days (an off-season gap is not 'rest' in any football sense).",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified", leakage_notes="Pure schedule metadata; not derived from any play data.",
         model_tracks=["independent"]),
    dict(name="opponent_days_rest", category="rest_situational",
         description="Days between the OPPONENT's previous game and this one (same definition as days_rest, evaluated for the opponent).",
         source_datasets=["schedules"], formula="same as days_rest, looked up for the opponent's own row for this same game_id",
         unit="days", rolling_window="none", min_observations=1,
         pregame_availability="Known as soon as the schedule is published.",
         missing_value_policy="Null under the same conditions as days_rest.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified", leakage_notes="Same-game lookup of public schedule information, not a cross-game leak.",
         model_tracks=["independent"]),
    dict(name="rest_diff", category="rest_situational",
         description="days_rest minus opponent_days_rest - positive means this team has a rest advantage.",
         source_datasets=["schedules"], formula="days_rest - opponent_days_rest",
         unit="days", rolling_window="none", min_observations=1,
         pregame_availability="Known once both teams' days_rest are known.",
         missing_value_policy="Null if either days_rest or opponent_days_rest is null.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified", leakage_notes="Derived from two already-safe values.",
         model_tracks=["independent"]),
    dict(name="is_neutral_site", category="rest_situational",
         description="True if the game is played at a neutral site (not either team's home stadium) - e.g. international games.",
         source_datasets=["schedules"], formula="raw schedules snapshot's location column == 'Neutral'",
         unit="boolean", rolling_window="none", min_observations=1,
         pregame_availability="Known as soon as the schedule is published.",
         missing_value_policy="Defaults to False if the raw schedules snapshot lacks a location column for that season (should not occur in the 2010-2025 window; documented as a defensive default, not an expected case).",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified", leakage_notes="Pure schedule metadata.",
         model_tracks=["independent"]),
    dict(name="is_divisional_game", category="rest_situational",
         description="True if the team and opponent are in the same division (per the normalized teams table's division field).",
         source_datasets=["schedules", "teams"], formula="team's division == opponent's division",
         unit="boolean", rolling_window="none", min_observations=1,
         pregame_availability="Known as soon as the schedule is published.",
         missing_value_policy="False if either team's division cannot be resolved (should not occur for any team in the 2010-2025 window).",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Uses each team's CURRENT division mapping, not any historical realignment - not a concern for 2010-2025 (no divisional realignment occurred in this window).",
         leakage_notes="Pure schedule/team metadata.", model_tracks=["independent"]),
    dict(name="short_week", category="rest_situational",
         description="True if days_rest <= short_week_max_rest_days (config/feature_engine.yaml; 6 as of this writing) - e.g. a Thursday game following a Sunday game.",
         source_datasets=["schedules"], formula="days_rest <= short_week_max_rest_days",
         unit="boolean", rolling_window="none", min_observations=1,
         pregame_availability="Known as soon as the schedule is published.",
         missing_value_policy="False when days_rest is null (treated as 'not proven short', not as True or missing).",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified", leakage_notes="Derived from days_rest, already safe.",
         model_tracks=["independent"]),
    dict(name="post_bye", category="rest_situational",
         description="True if days_rest >= post_bye_min_rest_days (config/feature_engine.yaml; 12 as of this writing) - a conservative floor meant to catch bye-week returns.",
         source_datasets=["schedules"], formula="days_rest >= post_bye_min_rest_days",
         unit="boolean", rolling_window="none", min_observations=1,
         pregame_availability="Known as soon as the schedule is published.",
         missing_value_policy="False when days_rest is null.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="A conservative rest-day floor, not a direct 'was this team on a bye' flag from the schedule - could theoretically also catch a non-bye long-gap edge case (e.g. around a postponed game), though none is known to occur in 2010-2025.",
         leakage_notes="Derived from days_rest, already safe.", model_tracks=["independent"]),
]

OPPONENT_ENTRIES = [
    dict(name=f"opp_{side}_epa_faced_season", category="opponent_quality",
         description=f"Average of each opponent's OWN {label} pregame season-to-date rating ({side}_epa_pp{'_allowed' if side=='def' else ''}_season) at the time this team played them, averaged across all of this team's meetings so far this season. A simple 'strength of schedule faced' signal - deliberately NOT a learned opponent-adjustment model (see docs/PHASE2_FEATURE_REPORT.md#opponent-context).",
         source_datasets=["pbp"],
         formula=f"mean over this team's prior meetings this season of (that opponent's own {side}_epa_pp{'_allowed' if side=='def' else ''}_season value AS OF that specific meeting)",
         unit="epa_per_play", rolling_window="season-to-date", min_observations=1,
         pregame_availability="Null until this team has played at least 1 game this season.",
         missing_value_policy="Null if no prior meetings this season, or if none of those opponents had their own rating available at the time (extremely rare - only a team's own first-ever game would lack it).",
         opponent_adjusted=True, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="A simple average of opponent quality faced, not a true strength-of-schedule adjustment (e.g. no recursive convergence, no accounting for how many games each opponent's own rating was based on) - Phase 3 may build a more rigorous opponent-adjustment model if warranted.",
         leakage_notes="Each opponent's rating used here is itself already leakage-safe (their own season-to-date value AS OF that specific meeting, never including games after it); averaging already-safe values across this team's own prior meetings introduces no new leakage.",
         model_tracks=["independent"])
    for side, label in [("off", "offensive"), ("def", "defensive")]
]

PREV_SEASON_ENTRIES = [
    dict(name=name, category="previous_season", description=desc, source_datasets=["pbp"],
         formula=formula, unit=unit, rolling_window="full previous season (every game, no exclusion)",
         min_observations=1,
         pregame_availability="Known for any team that played in the immediately preceding season; null for a team's first season in the 2010-2025 window (2010 itself) or after not existing the prior season.",
         missing_value_policy="Null (see has_prev_season_data) if the team has no data for the immediately preceding season.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=2011,
         known_limitations="A full-season average, computed once and held constant across every game of the following season - never blended with in-season current-year values using any weighting; Phase 3 decides how (or whether) to combine them.",
         leakage_notes="Uses ONLY the complete prior season's games (season S-1), attached to every game of season S; never includes any game from season S itself.",
         model_tracks=["independent"])
    for name, desc, formula, unit in [
        ("prev_season_off_epa_pp", "Full previous-season offensive EPA/play (every game of season S-1, not excluding any).", "sum(off_epa_sum over all of season S-1) / sum(off_plays_n over all of season S-1)", "epa_per_play"),
        ("prev_season_off_success_rate", "Full previous-season offensive success rate.", "sum(off_success_sum over all of season S-1) / sum(off_plays_n over all of season S-1)", "rate_0_1"),
        ("prev_season_def_epa_pp_allowed", "Full previous-season defensive EPA allowed per play.", "sum(def_epa_sum over all of season S-1) / sum(def_plays_n over all of season S-1)", "epa_per_play"),
        ("prev_season_def_success_rate_allowed", "Full previous-season defensive success rate allowed.", "sum(def_success_sum over all of season S-1) / sum(def_plays_n over all of season S-1)", "rate_0_1"),
    ]
]

SAMPLE_SIZE_ENTRIES = [
    dict(name="games_played_current_season", category="sample_size_support",
         description="Number of this team's games played so far THIS season, before the current one (0 in Week 1, barring bye-affected byes/cancellations).",
         source_datasets=["schedules"], formula="count of this team's games this season with sort_ts strictly before the current game",
         unit="count", rolling_window="season-to-date", min_observations=0,
         pregame_availability="Always known.", missing_value_policy="0, not null - a real count.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified", leakage_notes="Counts only prior games.", model_tracks=["independent"]),
    *[dict(name=f"n_games_trailing_{w}", category="sample_size_support",
         description=f"Number of prior games actually available for the trailing-{w}-game window (0 to {w}; less than {w} early in a team's history).",
         source_datasets=["schedules"], formula=f"min({w}, count of this team's completed prior games)",
         unit="count", rolling_window=f"trailing {w} games", min_observations=0,
         pregame_availability="Always known.", missing_value_policy="0, not null.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Shared across every metric computed over the same trailing window - see docs/PHASE2_FEATURE_REPORT.md's sample-size design rationale for why this is one column, not one per metric.",
         leakage_notes="Counts only prior games.", model_tracks=["independent"])
      for w in WINDOWS],
    dict(name="redzone_trips_n", category="sample_size_support",
         description="Number of red-zone drive trips THIS team had in the CURRENT game (not rolled - sample-size context for off_redzone_td_rate_season's denominator across games).",
         source_datasets=["pbp"], formula="count of drives reaching yardline_100<=20 for this team's offense, this game",
         unit="count", rolling_window="none (single game)", min_observations=0,
         pregame_availability="This is a POST-GAME count for a completed game, used only as historical sample-size context for rolling other teams' rates - never used as a pregame feature for the game it describes.",
         missing_value_policy="Null if the game's play-by-play has no red-zone drives recorded (rare but possible in a defensive struggle).",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Describes THIS game's own red-zone trip count, not a rolled/trailing value - context for interpreting off_redzone_td_rate_season's reliability, not a pregame predictor on its own.",
         leakage_notes="Not used as a feature for the game it describes; only ever consumed as historical input to LATER games' off_redzone_td_rate_season.",
         model_tracks=["independent"]),
    dict(name="redzone_trips_n_allowed", category="sample_size_support",
         description="Same as redzone_trips_n, from the defensive side (opponent's red-zone trips against this team).",
         source_datasets=["pbp"], formula="count of drives reaching yardline_100<=20 for the opponent's offense, this game",
         unit="count", rolling_window="none (single game)", min_observations=0,
         pregame_availability="Post-game count; same usage note as redzone_trips_n.",
         missing_value_policy="Null if no red-zone drives against this team this game.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Same as redzone_trips_n.", leakage_notes="Same as redzone_trips_n.",
         model_tracks=["independent"]),
    dict(name="fg_attempt_n", category="sample_size_support",
         description="Number of field goal attempts THIS team had in the CURRENT game - sample-size context for fg_pct_season.",
         source_datasets=["pbp"], formula="count(field_goal_attempt==1) this game", unit="count",
         rolling_window="none (single game)", min_observations=0,
         pregame_availability="Post-game count; same usage note as redzone_trips_n.",
         missing_value_policy="Null if the team attempted no field goals this game.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Same framing as redzone_trips_n - describes this game, consumed only as historical context for later games.",
         leakage_notes="Same as redzone_trips_n.", model_tracks=["independent"]),
    dict(name="punt_n", category="sample_size_support",
         description="Number of punts THIS team had in the CURRENT game - sample-size context for punt_yards_avg_season.",
         source_datasets=["pbp"], formula="count of punt plays with a recorded kick_distance, this game",
         unit="count", rolling_window="none (single game)", min_observations=0,
         pregame_availability="Post-game count; same usage note as redzone_trips_n.",
         missing_value_policy="Null if the team punted zero times this game.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="Same framing as redzone_trips_n.", leakage_notes="Same as redzone_trips_n.",
         model_tracks=["independent"]),
    dict(name="has_prev_season_data", category="sample_size_support",
         description="True if prev_season_* fields are populated for this team-game (i.e. the team has data for the immediately preceding season).",
         source_datasets=["pbp"], formula="prev_season_off_epa_pp is not null",
         unit="boolean", rolling_window="none", min_observations=0,
         pregame_availability="Always known.", missing_value_policy="False, not null - a real indicator, including the false case.",
         opponent_adjusted=False, garbage_time_filtered=False, earliest_reliable_season=1999,
         known_limitations="none identified", leakage_notes="Derived from already-safe prev_season_* values.",
         model_tracks=["independent"]),
]


def apply_snap_count_earliest_season(entries: list[dict]) -> None:
    for e in entries:
        if "snap_counts" in e["source_datasets"]:
            e["earliest_reliable_season"] = 2013


def main() -> None:
    entries: list[dict] = []
    entries += make_windowed_entries()
    entries += make_season_only_entries(SEASON_ONLY_METRICS)
    entries += make_season_only_entries(SPECIAL_TEAMS_METRICS, source_datasets=("pbp",))
    entries += QB_ENTRIES
    entries += PERSONNEL_ENTRIES
    entries += SITUATIONAL_ENTRIES
    entries += OPPONENT_ENTRIES
    entries += PREV_SEASON_ENTRIES
    entries += SAMPLE_SIZE_ENTRIES
    apply_snap_count_earliest_season(entries)

    names = [e["name"] for e in entries]
    assert len(names) == len(set(names)), f"duplicate names generated: {[n for n in names if names.count(n) > 1]}"

    header = (
        "# Declarative feature registry - machine-readable mirror of docs/FEATURE_DICTIONARY.md.\n"
        "#\n"
        "# GENERATED by scripts/generate_feature_registry.py from the authoritative metric\n"
        "# definitions in src/nfl_predict/features/team_game.py. Do not hand-edit entries whose\n"
        "# formula/window is generated from a MetricSpec - change team_game.py and re-run the\n"
        "# generator instead, or this file will drift from what the code actually computes.\n"
        "# Hand-maintained sections (QB/personnel/situational/opponent/prev-season/sample-size)\n"
        "# live in the generator script itself, not here, for the same reason.\n"
        "#\n"
        "# Identity/bookkeeping columns (game_id, season, season_type, week, game_date,\n"
        "# kickoff_time_naive, as_of_timestamp, team_id, opponent_id, venue) and `is_home` are\n"
        "# deliberately NOT registered here - see docs/FEATURE_DICTIONARY.md's \"Identity &\n"
        "# context columns\" section.\n"
        "#\n"
        "# The game-level feature table (src/nfl_predict/features/game_table.py) is not\n"
        "# separately registered either: home_<name>/away_<name> are just this team-game\n"
        "# feature evaluated for each side of the game, and diff_<name> = home_<name> -\n"
        "# away_<name>, a mechanical transform of two already-registered values - not a new\n"
        "# feature definition of its own.\n"
        "\n"
    )

    path = PROJECT_ROOT / "config" / "features.yaml"
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump({"features": entries}, f, sort_keys=False, default_flow_style=False, width=100)

    print(f"Wrote {len(entries)} feature entries to {path}")


if __name__ == "__main__":
    main()
