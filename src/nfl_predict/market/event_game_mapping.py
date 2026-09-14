"""Phase 8A provenance correction: an immutable, additive index mapping a live odds
provider's opaque `provider_event_id` to this project's canonical game identity.

The raw, provider-native snapshot store (`market.live_snapshot_store`) is never mutated to
retrofit this identity - `OddsMarketSnapshot` stays exactly what the provider returned,
keyed only by `provider_event_id`. This module is a SEPARATE, additive index recording, for
every event this project successfully matched to a game, the mapping itself: which
`canonical_game_id` (season, week) it resolved to, the provider's own raw team-name strings
alongside the normalized `team_id`s they resolved to (so home/away normalization is itself
auditable, not just trusted), and when/how the mapping was established. A reader who only has
a canonical `game_id` can use `find_provider_event_ids_for_game()` to find which
`market/live_snapshots/event=<provider_event_id>/` directories hold that game's real,
persisted odds - satisfying exactly the reconstruction gap the Phase 8A offline replay hit
(no way to attribute an anonymous snapshot file back to a specific game_id).

Mirrors `research.prospective_ledger`'s append-only-JSONL-plus-sha256 pattern. A mapping is
keyed by `provider_event_id` within its (season, week) directory: re-recording the exact
same mapping (the odds provider is re-polled and re-matches to the same game) is a safe,
idempotent no-op; a genuinely DIFFERENT mapping at the same `provider_event_id` (which should
never legitimately happen - a provider's event id should always resolve to the same real
game) is refused, never silently overwritten, so any such conflict surfaces immediately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from nfl_predict.storage.blob_store import get_blob_store

MAPPING_DIRNAME = "market/event_game_mapping"

# The fields that must be identical for two recordings of "the same" mapping to be treated
# as a safe re-observation rather than a conflicting remap. `mapping_timestamp` is
# deliberately excluded - it legitimately differs every time the mapping is re-confirmed by
# a later Odds API refresh, and the FIRST-recorded timestamp is the more meaningful one to
# keep (see `append_event_game_mapping_idempotent`).
_STABLE_IDENTITY_FIELDS = (
    "provider_event_id", "canonical_game_id", "season", "week",
    "provider_home_team", "provider_away_team", "home_team_id", "away_team_id",
    "kickoff_timestamp", "mapping_method", "status",
)


class EventGameMappingConflictError(Exception):
    """Raised when `provider_event_id` already has a DIFFERENT persisted mapping - this
    index is additive and a genuine conflicting remap is refused, never silently
    overwritten."""


@dataclass(frozen=True)
class EventGameMapping:
    provider_event_id: str
    canonical_game_id: str
    season: int
    week: int
    provider_home_team: str  # the provider's own raw team-name string, e.g. "Kansas City Chiefs"
    provider_away_team: str
    home_team_id: str  # this project's normalized team_id the provider_home_team resolved to
    away_team_id: str
    kickoff_timestamp: str | None
    mapping_timestamp: str  # when this mapping was actually established - the real market snapshot's own fetched_at, never a fabricated "now"
    mapping_method: str  # e.g. "exact_team_name_match_v1" - versioned so a future matching-algorithm change is distinguishable
    status: str = "matched"  # only "matched" mappings are ever persisted here - unmatched/ambiguous events produce no market data to attribute, so there is nothing to index


def _mapping_keys(season: int, week: int) -> tuple[str, str]:
    prefix = f"{MAPPING_DIRNAME}/season={season}/week={week}"
    return f"{prefix}/mappings.jsonl", f"{prefix}/mappings.sha256"


def append_event_game_mapping_idempotent(mapping: EventGameMapping) -> bool:
    """Returns `True` if a new line was appended, `False` if an identical mapping already
    existed (safe no-op). Raises `EventGameMappingConflictError` if a DIFFERENT mapping
    already exists for this `provider_event_id`."""
    store = get_blob_store()
    data_key, hash_key = _mapping_keys(mapping.season, mapping.week)

    new_dict = asdict(mapping)
    if store.exists(data_key):
        for line in store.read_bytes(data_key).decode("utf-8").splitlines():
            row = json.loads(line)
            if row["provider_event_id"] != mapping.provider_event_id:
                continue
            if all(row.get(k) == new_dict[k] for k in _STABLE_IDENTITY_FIELDS):
                return False  # identical mapping already recorded - no-op, existing mapping_timestamp is kept
            raise EventGameMappingConflictError(
                f"provider_event_id={mapping.provider_event_id!r} already maps to "
                f"canonical_game_id={row['canonical_game_id']!r}, but a DIFFERENT mapping "
                f"(canonical_game_id={mapping.canonical_game_id!r}) was just computed for the "
                "same provider_event_id - this index is additive and refuses a genuine "
                "conflicting remap rather than silently overwriting it."
            )

    store.append_bytes(data_key, (json.dumps(new_dict, sort_keys=True, default=str) + "\n").encode("utf-8"))
    digest = hashlib.sha256(store.read_bytes(data_key)).hexdigest()
    store.write_bytes(hash_key, digest.encode("utf-8"), exist_ok=True)
    return True


def read_event_game_mappings(season: int, week: int) -> list[dict]:
    store = get_blob_store()
    data_key, hash_key = _mapping_keys(season, week)
    if not store.exists(data_key):
        return []
    content = store.read_bytes(data_key)
    if store.exists(hash_key):
        actual = hashlib.sha256(content).hexdigest()
        recorded = store.read_bytes(hash_key).decode("utf-8").strip()
        if actual != recorded:
            raise ValueError(
                f"Event/game mapping index for season={season} week={week} does not match "
                "its recorded hash - it was modified after being written."
            )
    return [json.loads(line) for line in content.decode("utf-8").splitlines()]


def find_provider_event_ids_for_game(season: int, week: int, game_id: str) -> list[str]:
    """The reconstruction entry point: given only a canonical game_id, find which
    `market/live_snapshots/event=<provider_event_id>/` directories hold its real, persisted
    odds - without needing a new Odds API call or any provider-side lookup."""
    return [m["provider_event_id"] for m in read_event_game_mappings(season, week) if m["canonical_game_id"] == game_id]
