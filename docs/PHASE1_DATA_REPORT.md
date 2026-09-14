# Phase 1 Data Report

## Purpose

Phase 1's goal was not to build predictive features — it was to prove the pipeline can
download, snapshot, validate, normalize, store, and reproduce the historical information
later phases will consume. This document records what was actually found by running real
ingestion against nflverse (via `nflreadpy`) for the 2010–2025 window, not what the
library's documentation claims. Machine-readable detail (one row per dataset x season) is in
[`data/reports/phase1_coverage.csv`](../data/reports/phase1_coverage.csv) (gitignored —
regenerate with `python -m nfl_predict.data.coverage --seasons 2010-2025`).

## Coverage report terminology: `fetched` vs `usable`

The coverage CSV reports two distinct booleans per dataset×season, not one ambiguous
"available" flag:

- **`fetched`** — an upstream snapshot was successfully retrieved and preserved (a
  provenance manifest exists). Says nothing about data quality.
- **`usable`** — `fetched`, AND it passes the FATAL validation gate, AND it has at least one
  row. A snapshot that is fetched but has zero rows, or fetched but FATAL, is **not** usable.

Concretely: an out-of-range season (nflreadpy rejected the request) is `fetched=False,
usable=False`. `depth_charts` 2025 (finding #4 below) is `fetched=True, usable=False` — the
snapshot exists and is preserved, but it's structurally unusable. `snap_counts` 2012
(finding #2 below) is `fetched=True, usable=False` — the request succeeded and returned a
real, valid, zero-row result, but a snapshot with no rows provides no analytical
observations, so it is not usable regardless of its (currently zero) validation-issue count.
This is a deliberate coverage-policy decision, not something inferred from validation levels
alone — see `src/nfl_predict/data/coverage.py`'s module docstring for the exact rule.

## Historical window

2010–2025 inclusive (`DEFAULT_HISTORICAL_SEASONS` in
[`src/nfl_predict/data/season_range.py`](../src/nfl_predict/data/season_range.py) — a
config default, not a hardcoded assumption; any range can be passed via `--seasons`). 2026
was deliberately excluded, per the phase brief.

## Datasets and what was actually confirmed

### Core (Phase 1 "at minimum" list) — all confirmed for the full 2010–2025 window

| Dataset | nflreadpy loader | Seasons confirmed | Total rows | Snapshots |
|---|---|---|---|---|
| teams | `load_teams()` | n/a (season-less) | 36 (32 franchises + 4 relocation aliases) | 1 |
| schedules | `load_schedules()` | 2010–2025, all 16 | 4,363 games | 16 |
| pbp | `load_pbp()` | 2010–2025, all 16 | 770,337 plays | 16 |
| player_stats | `load_player_stats()` | 2010–2025, all 16 | 287,187 rows | 16 |
| rosters | `load_rosters()` | 2010–2025, all 16 | 43,856 rows | 16 |

Zero FATAL/ERROR/WARNING validation issues across every one of these snapshots.

### Optional (Phase 1 "also investigate availability" list)

| Dataset | Seasons confirmed | Total rows | Snapshots | Notes |
|---|---|---|---|---|
| rosters_weekly | 2010–2025, all 16 | 656,941 | 16 | Clean. |
| snap_counts | 2013–2025 (13 seasons with data) | 324,611 | 14 (incl. a real 0-row 2012 snapshot) | Documented by nflreadpy as "since 2012," but **2012 itself returns 0 rows** — verified by actually fetching it, not assumed. Effective usable coverage starts 2013. 2010–2011 are out of nflreadpy's supported range and were recorded as `unavailable`, not fetched. |
| participation | 2016–2025, all 10 | 478,989 | 10 | Clean, except a real schema change: 2023–2025 have 6 more columns than 2016–2022 (see "Findings" below). Has no `season`/`week` columns of its own — keyed by `nflverse_game_id` + `play_id`; joining to attach season/week is Phase 2's job, not done here. |
| depth_charts | 2010–2024 confirmed usable; **2025 is NOT usable as historical data** | 1,106,729 (includes the anomalous 2025 snapshot) | 16 | See "Findings" — 2025 returns a structurally different dataset. |
| nextgen_passing | 2016–2025, all 10 | 5,933 | 10 | Clean. |
| nextgen_receiving | 2016–2025, all 10 | 14,731 | 10 | Clean. |
| nextgen_rushing | 2016–2025, all 10 | 6,059 | 10 | Clean. |

**Not ingested in Phase 1** (explicitly out of scope per the phase brief): current injury
data, live sportsbook odds APIs. Historical betting lines that happen to be bundled inside
`schedules` (`spread_line`, `total_line`, `away_moneyline`, `home_moneyline`, etc.) were
captured as part of that raw snapshot since they're part of the same file, but are **not**
used anywhere in Phase 1 and are **not** part of the normalized `games` table's schema —
see "Football data vs. market data" below for why that separation is deliberate and
permanent, not a gap to be casually closed.

## Findings worth carrying into later phases

1. **nflreadpy validates season ranges itself, and rejects an entire season list if any one
   season in it is out of range** (e.g. `load_snap_counts(seasons=[2010, 2013])` raises for
   the whole call, not just 2010). The ingestion CLI always fetches one season at a time
   (`src/nfl_predict/data/nflverse_loader.py`) specifically because of this — never a batch
   list — both to isolate failures per season and because it naturally partitions raw
   storage by season.
2. **A season inside the documented range can still return zero rows.** `load_snap_counts(seasons=2012)`
   is documented as valid ("available since 2012") but returns an empty frame; 2013 is the
   first season with real rows. This is recorded as a real, successful, empty snapshot —
   not skipped and not treated the same as an out-of-range error: `fetched=True` in the
   coverage report. It is `usable=False`, though — see "Coverage report terminology" above.
3. **`participation` gained 6 columns starting in 2023** (verified via the schema-fingerprint/
   column-set comparison the ingestion CLI runs season-over-season). Recorded as a WARNING,
   not a FATAL — the dataset is still usable, but a Phase 2 feature relying on those 6
   columns will not be computable before 2023.
4. **`depth_charts` for 2025 is a different data product, not a historical snapshot.**
   2010–2024 all return a consistent schema (`season`, `week`, `gsis_id`, `club_code`,
   `depth_position`, etc., ~32–38k rows/season). 2025 instead returns 554,215 rows with a
   completely different schema (`dt`, `espn_id`, `pos_abb`, `pos_rank`, ... — no `season` or
   `week` columns at all, timestamped entries like `2026-03-14T07:32:09Z`). This is
   consistent with nflverse's live/current depth-chart feed (ESPN-sourced) being returned
   instead of a season-archived file once a season is no longer mid-processing in the
   historical pipeline. **Correctly caught**: `validate_critical_columns` flagged this FATAL
   (`missing required columns ['season', 'week']`), which is why this project's validation
   exists — `fetched=True, usable=False` in the coverage report. depth_charts is not
   promoted to any normalized structure in Phase 1 regardless (see "Raw vs. normalized"
   below), but a Phase 2 feature that wants 2025 depth-chart data needs to either find a
   different source for that season or write a dedicated adapter for this schema — it
   cannot naively reuse the 2010–2024 parsing logic.
5. **Team identity**: nflverse's own `load_teams()` output already encodes a stable numeric
   franchise `team_id` that correctly groups every historical abbreviation — confirmed for
   all three real relocations in the modern era: Rams (STL → LA → LAR, team_id `2510`),
   Chargers (SD → LAC, team_id `4400`), Raiders (OAK → LV, team_id `2520`). This project
   uses that upstream `team_id` directly (`src/nfl_predict/data/teams.py`) rather than
   inventing a new identity scheme — 32 current franchises, 36 total known abbreviation
   aliases as of this ingestion.
6. **Game identity**: nflverse's `game_id` (format `{season}_{week:02d}_{away_team}_{home_team}`,
   e.g. `2025_01_DAL_PHI`) is used verbatim as this project's internal game identifier — it's
   deterministic and built from immutable inputs (season, week, team abbreviations), never
   from a mutable field like the final score. `play-by-play` game_ids matched normalized
   `games` game_ids with **zero** orphans and **zero** missing-pbp warnings across every
   season checked (all core-dataset seasons, 2010–2025).
7. **Regular+postseason game counts by season, as actually observed**: 267 for 2010–2019,
   269 for 2020, 285 for 2021/2023/2024/2025, 284 for 2022. The 2022 count is one below the
   17-game-season norm, consistent with the well-documented cancellation of the Week 17
   2022 Bills–Bengals game. These counts were not assumed — they came from the row counts of
   the actual normalized `games` table.
8. **`gametime`/timezone**: `schedules` publishes a local kickoff time string with no UTC
   offset. `src/nfl_predict/data/games.py` stores it as a naive (timezone-unaware)
   `kickoff_time_naive` field and documents this as a known limitation — resolving it to a
   real timezone-aware timestamp is unresolved and deferred, not silently guessed.
9. **`source_release_identifier` is `null` in every manifest.** nflverse-data publishes
   rolling GitHub release tags that are updated in place; `nflreadpy`'s public API doesn't
   expose a resolvable per-fetch release/commit identifier. This was a known, accepted
   limitation from the Phase 0 provenance contract, not a Phase 1 surprise — content hash +
   retrieval timestamp are what distinguish snapshots over time instead.

## Validation summary

Across the entire 2010–2025 backfill (177 dataset×season rows in the coverage report):

- **1 FATAL**: `depth_charts` 2025 (finding #4 above) — correctly blocked from being treated
  as usable historical data.
- **4 WARNING**: `participation` 2023 schema-columns-added (finding #3), plus 3 warnings on
  the same `depth_charts` 2025 snapshot (schema-columns-added, schema-columns-removed,
  suspicious-row-count) — all downstream symptoms of finding #4, not independent issues.
- **0 ERROR** anywhere.
- Every core dataset (teams, schedules, pbp, player_stats, rosters) is issue-free across the
  entire window.

A real bug was found and fixed during this backfill: the schema-change check initially
compared a season's columns against whichever snapshot happened to be most recently written
to disk, rather than the chronologically preceding season — harmless for datasets whose
schema never changed, but it mis-attributed `participation`'s 2023 schema change to the
wrong seasons (2016–2022) before the fix. Fixed in `src/nfl_predict/data/ingest.py` to look
up the specific prior-season manifest from the incrementally-populated database instead of
"whatever's on disk," and the already-collected validation rows were corrected in place
(re-derived from stored manifests, no re-fetch needed). Covered going forward by the
existing `test_validate_schema_change_*` unit tests plus this documented incident.

## Idempotency, demonstrated

`schedules` for 2025 was ingested twice. The second run: detected the byte-identical content
via SHA-256, recorded a new manifest (so the fetch attempt itself is never lost) with
`duplicate_of_retrieval_id` pointing at the first, did **not** write a duplicate Parquet
file, and re-ran normalization — the `games` table still has exactly 285 rows for season
2025 afterward (verified directly against the database), not 570. See
`tests/integration/test_ingestion_integration.py` for the same behavior under test.

## Raw vs. normalized

- **Raw** (immutable, one Parquet + one `manifest.json` per fetch, never overwritten):
  `data/raw/nflverse/<dataset_name>/<retrieval_id>/`. All 12 datasets above have raw
  snapshots preserved here regardless of whether they were promoted further.
- **Normalized** (Phase 1 scope — only where the phase brief specifically asked for a
  normalized table):
  - `teams` / `team_abbr_aliases` tables in SQLite (`data/nfl_predict.sqlite`).
  - `games` table in SQLite, keyed by nflverse's `game_id`.
  - `play_by_play` Parquet, partitioned by season, under
    `data/normalized/play_by_play/season=<year>/data.parquet` — kept out of SQLite
    deliberately (770k rows across 372 raw columns; SQLite holds relational metadata, not
    bulk analytical data, per `docs/ARCHITECTURE.md#storage`).
- The other 7 datasets (player_stats, rosters, rosters_weekly, snap_counts, participation,
  depth_charts, nextgen_*) are snapshotted and validated but **not** promoted to a dedicated
  normalized table in Phase 1 — the phase brief's explicit "normalized table" requirement
  only named games/teams/play-by-play. Building normalized tables for the rest is a Phase 2
  decision, made once it's clear which features actually need them.

## Football data vs. market data

`schedules` raw snapshots contain both game/result fields and historical betting-market
fields (`spread_line`, `total_line`, `home_moneyline`, `away_moneyline`,
`away_spread_odds`, `home_spread_odds`, `under_odds`, `over_odds`) in the same file, because
that's how nflverse publishes it. Phase 1's normalization deliberately splits these apart:

- The normalized `games` table (`src/nfl_predict/data/games.py`, `data/nfl_predict.sqlite`)
  contains **only** game identity/schedule/result fields. It has no market-line columns at
  all, by construction — `normalize_games()`'s output schema simply doesn't include them.
- The market-line fields exist **only** inside the immutable raw `schedules` snapshots. They
  were captured as a side effect of preserving the full raw file, not deliberately
  extracted, and nothing in Phase 1 reads them.

This is intentional, not an oversight, and it must stay this way going forward per
`docs/MODEL_SPEC.md`'s independent/market-aware split (`CLAUDE.md` principle 2): the
independent model must never take sportsbook-derived data as a feature. If `games` had
market-line columns sitting right next to game identity, it would be far too easy for a
Phase 2/3 feature-engineering pass to accidentally join them into the independent model's
inputs. Keeping market data physically absent from the normalized football-data table is a
stronger guardrail than a naming convention or a code-review reminder.

**When Phase 2/3 actually need historical market lines** (for the market-aware track or for
backtesting ATS/CLV performance), that is a deliberate decision that must create a
**separate market-data representation** — e.g. a `historical_market_lines` table keyed by
`game_id`, extracted explicitly from the raw `schedules` snapshots, tagged as market data in
`config/features.yaml`'s `model_tracks: [market_aware]` (never `[independent]`), and
documented in `docs/DATA_SOURCES.md`/`docs/MODEL_SPEC.md` when it happens. It is not created
in Phase 1.

## What Phase 2 can safely use right now

- `teams` / `team_abbr_aliases` (SQLite) — safe, complete, stable team identity for
  2010–2025.
- `games` (SQLite) — safe, complete, 4,363 games across 2010–2025. Contains game
  identity/schedule/result information only (`game_id`, season/week/date, teams, scores,
  venue, status) — **no betting-market fields**. Historical `spread_line`, `total_line`,
  `home_moneyline`, `away_moneyline`, etc. remain in the immutable raw `schedules` snapshots
  only (`data/raw/nflverse/schedules/<retrieval_id>/data.parquet`) and were deliberately
  **not** promoted into this table — see "Football data vs. market data" below.
- `play_by_play` (Parquet, per-season) — safe, complete, 2010–2025, zero linkage issues
  against `games`.
- `player_stats`, `rosters`, `rosters_weekly` (raw Parquet snapshots only) — safe to build
  features from once Phase 2 adds normalization for them; schemas were validated and are
  clean across the whole window.
- `snap_counts` — safe from **2013** onward only; do not build a feature assuming 2010–2012
  coverage.
- `participation` — safe from **2016** onward; a feature using any of the 6 columns added in
  2023 is only computable from **2023** onward, not before.
- `nextgen_passing` / `nextgen_receiving` / `nextgen_rushing` — safe from **2016** onward.
- `depth_charts` — safe for **2010–2024**. Do **not** use the 2025 snapshot as-is; either
  exclude 2025 from any depth-chart-derived feature until this is resolved, or (a later,
  deliberate decision) write a dedicated adapter for the different 2025 schema.

## Reproducing this report

```bash
python -m nfl_predict.data.ingest --dataset teams
python -m nfl_predict.data.ingest --all --seasons 2010-2025
python -m nfl_predict.data.coverage --seasons 2010-2025
```

Re-running these commands is safe: unchanged upstream data is detected by content hash and
does not duplicate raw files or normalized rows (see "Idempotency" above).
