# Feature Dictionary

## Purpose

This is the human-readable companion to `config/features.yaml`. Every feature that ever
feeds a model has an entry in both places, and they must agree — `config/features.yaml` is
what code reads (and what `src/nfl_predict/features/registry.py` validates against the
engine's actual output columns); this file is what a person reads to understand *why* a
feature exists, how it's grouped, and what its risks are. If they diverge, that's a bug.

**Status: Phase 2 is complete.** 114 features are registered, covering the team-game
pregame feature table produced by `src/nfl_predict/features/team_game.py`. Full
per-feature detail (formula, missing-value policy, leakage notes, etc.) lives in
`config/features.yaml`; this document organizes that same set by football concept and
explains the handful of engine-wide conventions (ALL-PLAY vs. COMPETITIVE-PLAY, rolling
windows, sample-size columns) that apply across many features rather than repeating them
114 times.

`config/features.yaml` is **generated**, not hand-maintained — see
`scripts/generate_feature_registry.py`. If a feature's definition changes, that script (and
the `MetricSpec` lists in `src/nfl_predict/features/team_game.py` it reads from) is what
changes; re-run it rather than hand-editing the YAML.

## Required fields per feature

Every entry in `config/features.yaml` has:

| Field                     | Meaning |
|----------------------------|---------|
| `name`                     | Unique, stable identifier (snake_case). |
| `category`                 | One of the football-concept groupings below. |
| `description`              | Plain-English definition. |
| `source_datasets`          | Which ingested dataset(s) it's computed from (`pbp`, `schedules`, `teams`, `snap_counts`). |
| `formula`                  | Exact computation — precise enough that two independent implementations would agree. |
| `unit`                     | e.g. `epa_per_play`, `rate_0_1`, `days`, `boolean`, `count`, `categorical_id`. |
| `rolling_window`           | `none`, `season-to-date`, `trailing N games`, `full previous season`, or (for personnel continuity) `prior two completed games only`. |
| `min_observations`         | Minimum prior games/plays required before the value is populated instead of null. |
| `pregame_availability`     | When the value is actually knowable relative to kickoff. |
| `missing_value_policy`     | What happens when it can't be computed — never a silent zero unless zero is the correct value. |
| `opponent_adjusted`        | `true`/`false`. Only `opp_off_epa_faced_season`/`opp_def_epa_faced_season` are `true` — see "Opponent quality," below; nothing else in Phase 2 adjusts for opponent strength. |
| `garbage_time_filtered`    | `true`/`false` — whether it's a COMPETITIVE-PLAY variant. |
| `earliest_reliable_season` | 1999 for anything derived purely from play-by-play; 2013 for anything needing `snap_counts`; 2011 for previous-season features (2010 is the first season in this project's window, so no team has a "previous season" value until 2011). |
| `known_limitations`        | Real caveats — small samples, proxy metrics, ID-namespace issues, etc. |
| `leakage_notes`            | Exactly which prior games/plays the value is allowed to use, and why that's safe. |
| `model_tracks`             | Always `[independent]` for every Phase 2 feature — see "Market data exclusion," below. |

## Identity & context columns (not registered as features)

These columns exist in the team-game and game feature tables but are **not** in
`config/features.yaml`, because they're structural bookkeeping, not computed football
variables with a formula/window/leakage story of their own:

`game_id`, `season`, `season_type`, `week`, `game_date`, `kickoff_time_naive`,
`as_of_timestamp`, `team_id`, `opponent_id`, `venue`.

**`is_home`** is a deliberate special case: it's structurally definitional (it's what makes
a row the home-side or away-side view of a game) *and* it functions as the home-field
signal a model reads directly. It is grouped with the identity columns in code
(`team_game.IDENTITY_COLUMNS`) precisely because — unlike every registered feature — it has
no formula, no rolling window, and no leakage risk to document; registering it with that
apparatus would be exactly the "identity column treated as a predictive feature" anti-pattern
this phase was warned against. A model consuming the feature table should read `is_home`
directly from the identity block.

## Sample-size / support columns

10 columns exist purely to tell a downstream consumer how much data backs a rate feature —
they are registered (category `sample_size_support`) but are metadata, not football
performance metrics themselves: `games_played_current_season`, `n_games_trailing_3`,
`n_games_trailing_5`, `n_games_trailing_8`, `qb_starts_season`, `redzone_trips_n`,
`redzone_trips_n_allowed`, `fg_attempt_n`, `punt_n`, `has_prev_season_data`.

They're deliberately **shared counters**, not one-per-metric: every rate metric computed
over the same trailing window (or the same red-zone/FG/punt sample) draws on the identical
underlying set of prior games, so a per-metric copy of the same number would be pure
duplication. The exceptions (`qb_starts_season`, `redzone_trips_n(_allowed)`, `fg_attempt_n`,
`punt_n`) are scoped to the one metric family that specifically needs that denominator
visible.

## ALL-PLAY vs. COMPETITIVE-PLAY

Garbage time is real: a blowout's fourth quarter doesn't represent either team's typical
effort or scheme intent, and lets a team's raw EPA/play look better or worse than it should.
Rather than picking one filtered view and hiding the difference, Phase 2 provides **both**,
clearly named, for the four metrics where garbage time is most likely to distort perception
of a team's core offensive identity (`off_epa_pp`, `off_success_rate`, and their `_comp`
counterparts, each across all four windows — 16 columns total):

- **ALL-PLAY** (no suffix beyond the window, e.g. `off_epa_pp_5g`): every offensive play
  (`play_type` in `pass`/`run`), regardless of game state.
- **COMPETITIVE-PLAY** (`_comp` in the name, e.g. `off_epa_pp_comp_5g`): restricted to plays
  where the **pre-play** home win probability (`home_wp`, before that play happened) was
  between `garbage_time_wp_threshold` and `1 - garbage_time_wp_threshold`
  (`config/feature_engine.yaml`; `0.05` as of this writing — i.e. both teams still had at
  least a 5% win probability). Using *pre-play* win probability, never post-play or the
  final score, is what keeps this leakage-safe: the filter reflects what the game state
  looked like the moment before the play, not hindsight about how the game ended.

`qb_kneel` and `qb_spike` plays are excluded from **both** variants, not because of garbage
time, but because the `pass`/`run` play-type filter used everywhere in this engine simply
never includes them — they're clock-management plays, not representative offensive
snaps, in either view.

Every other metric family (passing/rushing splits, explosive rates, third down, red zone,
sacks, turnovers, special teams) is ALL-PLAY only, per `garbage_time_filtered: false` in the
registry — extending the competitive-play treatment to all of them was judged not worth the
~85 additional columns it would cost relative to the value, particularly since situational
plays (3rd down, red zone) rarely occur in genuine garbage time anyway.

## Rolling windows

Two kinds, used consistently across every windowed feature:

- **Trailing N games** (`_3g`/`_5g`/`_8g`): the team's N most recent *completed* games
  strictly before the current one. Does **not** reset at season boundaries — a Week 1
  game's trailing window can include games from the end of the prior season, which is a
  deliberate design choice (a team's most recent 5 games are still its most recent 5 games
  across an off-season gap), not an oversight. Computed as a rolled **sum** of the metric's
  numerator and denominator across the window, then divided once — never an average of
  per-game rates — so a game with 75 plays is weighted correctly relative to one with 55.
- **Season-to-date** (`_season`): every game this team has played so far *this season*,
  strictly before the current one. Resets to empty at the first game of every season.

**Previous-season** features (`prev_season_*`) are a third, distinct thing: the *entire*
prior season's totals (every game, none excluded), attached as a constant to every game of
the following season. This is deliberately not the same number as the prior season's final
`_season` value, which — following the same "strictly before the current game" rule as
everything else — excludes that season's own last game.

Per the Phase 2 brief, current-season and previous-season values are never blended with an
arbitrary weighting scheme; they're kept as separate columns so Phase 3 can learn (or not)
how to combine them.

## Market data exclusion

Every registered feature is `model_tracks: [independent]`. No feature in this registry is
computed from `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`,
`home_spread_odds`, `away_spread_odds`, `under_odds`, or `over_odds` — those fields exist in
the raw `schedules` snapshots (Phase 1) but were never read by any Phase 2 module. See
`src/nfl_predict/features/registry.py`'s `MARKET_FIELD_DENYLIST` and the automated leakage
test that asserts none of them can enter the team-game or game feature tables
(`tests/features/test_leakage.py`).

## Feature families

### Offense (20 features)

Overall offensive efficiency — EPA and success rate per play, early-down efficiency,
third-down conversion, red-zone touchdown rate. See "ALL-PLAY vs. COMPETITIVE-PLAY" above
for the `_comp` variants.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `off_early_down_epa_season` | epa_per_play | season-to-date | 1999 |
| `off_early_down_success_rate_season` | rate_0_1 | season-to-date | 1999 |
| `off_epa_pp_3g` | epa_per_play | trailing 3 games | 1999 |
| `off_epa_pp_5g` | epa_per_play | trailing 5 games | 1999 |
| `off_epa_pp_8g` | epa_per_play | trailing 8 games | 1999 |
| `off_epa_pp_comp_3g` | epa_per_play | trailing 3 games | 1999 |
| `off_epa_pp_comp_5g` | epa_per_play | trailing 5 games | 1999 |
| `off_epa_pp_comp_8g` | epa_per_play | trailing 8 games | 1999 |
| `off_epa_pp_comp_season` | epa_per_play | season-to-date | 1999 |
| `off_epa_pp_season` | epa_per_play | season-to-date | 1999 |
| `off_redzone_td_rate_season` | rate_0_1 | season-to-date | 1999 |
| `off_success_rate_3g` | rate_0_1 | trailing 3 games | 1999 |
| `off_success_rate_5g` | rate_0_1 | trailing 5 games | 1999 |
| `off_success_rate_8g` | rate_0_1 | trailing 8 games | 1999 |
| `off_success_rate_comp_3g` | rate_0_1 | trailing 3 games | 1999 |
| `off_success_rate_comp_5g` | rate_0_1 | trailing 5 games | 1999 |
| `off_success_rate_comp_8g` | rate_0_1 | trailing 8 games | 1999 |
| `off_success_rate_comp_season` | rate_0_1 | season-to-date | 1999 |
| `off_success_rate_season` | rate_0_1 | season-to-date | 1999 |
| `off_third_down_conv_rate_season` | rate_0_1 | season-to-date | 1999 |

### Passing (10 features)

Dropback-scoped passing efficiency (`qb_dropback == 1`: true pass attempts, sacks, and
scrambles) plus CPOE and raw yards/dropback.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `off_cpoe_season` | percentage_points | season-to-date | 1999 |
| `off_pass_epa_dropback_3g` | epa_per_dropback | trailing 3 games | 1999 |
| `off_pass_epa_dropback_5g` | epa_per_dropback | trailing 5 games | 1999 |
| `off_pass_epa_dropback_8g` | epa_per_dropback | trailing 8 games | 1999 |
| `off_pass_epa_dropback_season` | epa_per_dropback | season-to-date | 1999 |
| `off_pass_success_rate_3g` | rate_0_1 | trailing 3 games | 1999 |
| `off_pass_success_rate_5g` | rate_0_1 | trailing 5 games | 1999 |
| `off_pass_success_rate_8g` | rate_0_1 | trailing 8 games | 1999 |
| `off_pass_success_rate_season` | rate_0_1 | season-to-date | 1999 |
| `off_yards_per_dropback_season` | yards_per_dropback | season-to-date | 1999 |

### Rushing (9 features)

DESIGNED rush attempts only (`rush_attempt == 1 & qb_scramble == 0`). QB scrambles are
booked to the passing/QB side, not here — see `docs/PHASE2_FEATURE_REPORT.md`'s
"Offensive features" section for the reasoning.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `off_rush_epa_3g` | epa_per_play | trailing 3 games | 1999 |
| `off_rush_epa_5g` | epa_per_play | trailing 5 games | 1999 |
| `off_rush_epa_8g` | epa_per_play | trailing 8 games | 1999 |
| `off_rush_epa_season` | epa_per_play | season-to-date | 1999 |
| `off_rush_success_rate_3g` | rate_0_1 | trailing 3 games | 1999 |
| `off_rush_success_rate_5g` | rate_0_1 | trailing 5 games | 1999 |
| `off_rush_success_rate_8g` | rate_0_1 | trailing 8 games | 1999 |
| `off_rush_success_rate_season` | rate_0_1 | season-to-date | 1999 |
| `off_yards_per_rush_season` | yards_per_play | season-to-date | 1999 |

### Defense (28 features)

Exact defensive counterparts of the offense/passing/rushing families above — same formulas,
grouped by the *defending* team instead of the *possessing* team. Naming convention:
offense features never end in "_allowed"; defense features always do (or "_generated" for
sacks/turnovers/pressure, since those are things a defense actively produces).

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `def_early_down_epa_allowed_season` | epa_per_play | season-to-date | 1999 |
| `def_early_down_success_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |
| `def_epa_pp_allowed_3g` | epa_per_play | trailing 3 games | 1999 |
| `def_epa_pp_allowed_5g` | epa_per_play | trailing 5 games | 1999 |
| `def_epa_pp_allowed_8g` | epa_per_play | trailing 8 games | 1999 |
| `def_epa_pp_allowed_season` | epa_per_play | season-to-date | 1999 |
| `def_pass_epa_allowed_3g` | epa_per_dropback | trailing 3 games | 1999 |
| `def_pass_epa_allowed_5g` | epa_per_dropback | trailing 5 games | 1999 |
| `def_pass_epa_allowed_8g` | epa_per_dropback | trailing 8 games | 1999 |
| `def_pass_epa_allowed_season` | epa_per_dropback | season-to-date | 1999 |
| `def_pass_success_rate_allowed_3g` | rate_0_1 | trailing 3 games | 1999 |
| `def_pass_success_rate_allowed_5g` | rate_0_1 | trailing 5 games | 1999 |
| `def_pass_success_rate_allowed_8g` | rate_0_1 | trailing 8 games | 1999 |
| `def_pass_success_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |
| `def_redzone_td_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |
| `def_rush_epa_allowed_3g` | epa_per_play | trailing 3 games | 1999 |
| `def_rush_epa_allowed_5g` | epa_per_play | trailing 5 games | 1999 |
| `def_rush_epa_allowed_8g` | epa_per_play | trailing 8 games | 1999 |
| `def_rush_epa_allowed_season` | epa_per_play | season-to-date | 1999 |
| `def_rush_success_rate_allowed_3g` | rate_0_1 | trailing 3 games | 1999 |
| `def_rush_success_rate_allowed_5g` | rate_0_1 | trailing 5 games | 1999 |
| `def_rush_success_rate_allowed_8g` | rate_0_1 | trailing 8 games | 1999 |
| `def_rush_success_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |
| `def_success_rate_allowed_3g` | rate_0_1 | trailing 3 games | 1999 |
| `def_success_rate_allowed_5g` | rate_0_1 | trailing 5 games | 1999 |
| `def_success_rate_allowed_8g` | rate_0_1 | trailing 8 games | 1999 |
| `def_success_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |
| `def_third_down_allowed_season` | rate_0_1 | season-to-date | 1999 |

### Quarterback (8 features)

**Read `docs/PHASE2_FEATURE_REPORT.md`'s "QB features and the leakage rule" section before
using any of these.** Every one of them describes the quarterback identified as primary in
this team's *previous* completed game — never this game's own starter — and his own
performance in his own prior primary-QB starts this season, never the team's aggregate
(which would include any backup's snaps).

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `qb_consecutive_starts` | count | none | 1999 |
| `qb_cpoe_season` | percentage_points | season-to-date | 1999 |
| `qb_epa_dropback_season` | epa_per_dropback | season-to-date | 1999 |
| `qb_int_rate_season` | rate_0_1 | season-to-date | 1999 |
| `qb_primary_id` | categorical_id | none | 1999 |
| `qb_sack_rate_season` | rate_0_1 | season-to-date | 1999 |
| `qb_scramble_epa_season` | epa_per_dropback | season-to-date | 1999 |
| `qb_success_rate_season` | rate_0_1 | season-to-date | 1999 |

### Explosive plays (4 features)

Conventional, explicitly documented thresholds (20+ yards on a completed pass, 10+ yards on
a designed rush — `config/feature_engine.yaml`), since nflverse provides no canonical
"explosive play" flag.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `def_explosive_pass_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |
| `def_explosive_rush_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |
| `off_explosive_pass_rate_season` | rate_0_1 | season-to-date | 1999 |
| `off_explosive_rush_rate_season` | rate_0_1 | season-to-date | 1999 |

### Pressure / sacks (4 features)

Proxy metrics, documented as such: nflverse has no true pressure-rate, pass-block-win-rate,
or pass-rush-win-rate statistic, and none is invented here. Sack rate and QB-hit rate are
what's actually measurable.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `def_qb_hit_rate_generated_season` | rate_0_1 | season-to-date | 1999 |
| `def_sack_rate_generated_season` | rate_0_1 | season-to-date | 1999 |
| `off_qb_hit_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |
| `off_sack_rate_allowed_season` | rate_0_1 | season-to-date | 1999 |

### Turnovers (4 features)

Process-style rates (interceptions/fumbles-lost per relevant play), deliberately not
presented as if turnovers are perfectly stable or predictable — see
`docs/PHASE2_FEATURE_REPORT.md`'s turnovers caveat.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `def_fumble_forced_rate_season` | rate_0_1 | season-to-date | 1999 |
| `def_int_rate_generated_season` | rate_0_1 | season-to-date | 1999 |
| `off_fumble_lost_rate_season` | rate_0_1 | season-to-date | 1999 |
| `off_int_rate_season` | rate_0_1 | season-to-date | 1999 |

### Special teams (2 features)

Deliberately restrained per the Phase 2 brief ("do not overbuild this area in V1"). Punt
yardage is **gross**, not net of return yardage — see `docs/PHASE2_FEATURE_REPORT.md` for
why net punting was deferred. Return performance is deferred entirely.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `fg_pct_season` | rate_0_1 | season-to-date | 1999 |
| `punt_yards_avg_season` | yards_per_punt | season-to-date | 1999 |

### Personnel continuity (2 features)

Self-referential within `snap_counts` (2013+ only) — compares a team's most recent two
completed games to each other, never the target game's own roster. See
`docs/PHASE2_FEATURE_REPORT.md` for why this avoids needing a player-ID crosswalk between
`snap_counts`' PFR ids and play-by-play's GSIS ids.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `def_snap_continuity_pct` | rate_0_1 | prior two completed games only | 2013 |
| `off_snap_continuity_pct` | rate_0_1 | prior two completed games only | 2013 |

### Opponent quality (2 features)

The only `opponent_adjusted: true` features in this registry — and deliberately a simple
average, not a learned adjustment model. See `docs/PHASE2_FEATURE_REPORT.md`'s "Opponent
context" section for why a more rigorous strength-of-schedule model is explicitly deferred
to Phase 3.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `opp_def_epa_faced_season` | epa_per_play | season-to-date | 1999 |
| `opp_off_epa_faced_season` | epa_per_play | season-to-date | 1999 |

### Previous season (4 features)

See "Rolling windows" above for how this differs from a season's own final `_season` value.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `prev_season_def_epa_pp_allowed` | epa_per_play | full previous season | 2011 |
| `prev_season_def_success_rate_allowed` | rate_0_1 | full previous season | 2011 |
| `prev_season_off_epa_pp` | epa_per_play | full previous season | 2011 |
| `prev_season_off_success_rate` | rate_0_1 | full previous season | 2011 |

### Rest / situational (7 features)

Pure schedule metadata — known the moment the schedule is published, long before kickoff.
No leakage risk beyond the general "don't read the target game's outcome" rule.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `days_rest` | days | none | 1999 |
| `is_divisional_game` | boolean | none | 1999 |
| `is_neutral_site` | boolean | none | 1999 |
| `opponent_days_rest` | days | none | 1999 |
| `post_bye` | boolean | none | 1999 |
| `rest_diff` | days | none | 1999 |
| `short_week` | boolean | none | 1999 |

### Sample-size / support (10 features)

See "Sample-size / support columns" above.

| Name | Unit | Window | Earliest season |
|---|---|---|---|
| `fg_attempt_n` | count | none | 1999 |
| `games_played_current_season` | count | season-to-date | 1999 |
| `has_prev_season_data` | boolean | none | 1999 |
| `n_games_trailing_3` | count | trailing 3 games | 1999 |
| `n_games_trailing_5` | count | trailing 5 games | 1999 |
| `n_games_trailing_8` | count | trailing 8 games | 1999 |
| `punt_n` | count | none | 1999 |
| `qb_starts_season` | count | season-to-date | 1999 |
| `redzone_trips_n` | count | none | 1999 |
| `redzone_trips_n_allowed` | count | none | 1999 |

## Process for adding a feature (Phase 3+)

1. Confirm the underlying data exists in `docs/DATA_SOURCES.md` with known coverage.
2. Add or extend the relevant list in `src/nfl_predict/features/team_game.py` (or a new
   module, for a genuinely new family) with a real, tested implementation.
3. Re-run `python scripts/generate_feature_registry.py` (extending its metadata tables
   first, for a new metric family) so `config/features.yaml` reflects the new column(s).
4. Add the matching category/table entry to this file.
5. Add a unit test that checks the leakage guard explicitly (e.g. a test asserting the
   feature value for game G does not change if a later game's data is mutated) — see
   `tests/features/test_leakage.py` for the existing pattern.
6. Only after (1)-(5) does a feature become eligible for use in a model's input set.
