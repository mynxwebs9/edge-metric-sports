"""Phase 7 Step 19: shadow decisions.

An experimental decision-rule version can evaluate games prospectively WITHOUT ever
appearing in the public/official pick ledger (`pick_ledger.py`) - stored in a completely
separate directory tree, keyed by its own `rule_set_version`, so a shadow decision can never
accidentally leak into `pick_ledger.read_current_picks()`'s output. Comparing a shadow rule
version against the active/published one is a read-only, later, human-initiated analysis -
this module never automatically promotes a shadow rule to official status based on which
performed better.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from nfl_predict.config import get_settings

SHADOW_DIRNAME = "decision/shadow_decisions"


@dataclass(frozen=True)
class ShadowDecisionRecord:
    shadow_decision_id: str
    game_id: str
    market_type: str
    rule_set_version: str  # the EXPERIMENTAL rule set being evaluated
    decision: str
    reason_codes: tuple[str, ...]
    decision_timestamp: str
    input_packet_hash: str


def _shadow_path(rule_set_version: str) -> Path:
    return get_settings().data_dir / SHADOW_DIRNAME / f"{rule_set_version}.jsonl"


def record_shadow_decision(record: ShadowDecisionRecord) -> None:
    path = _shadow_path(record.rule_set_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), sort_keys=True, default=str) + "\n")


def read_shadow_decisions(rule_set_version: str) -> list[dict]:
    path = _shadow_path(rule_set_version)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compare_shadow_to_official(shadow_decisions: list[dict], official_picks: list[dict]) -> dict:
    """Read-only comparison for later human analysis - never an automatic swap. Matches by
    (game_id, market_type)."""
    official_by_key = {(p["game_id"], p["market_type"]): p for p in official_picks}
    comparisons = []
    for shadow in shadow_decisions:
        key = (shadow["game_id"], shadow["market_type"])
        official = official_by_key.get(key)
        comparisons.append({
            "game_id": shadow["game_id"], "market_type": shadow["market_type"],
            "shadow_decision": shadow["decision"],
            "official_pick_exists": official is not None,
            "official_status": official["status"] if official else None,
        })
    return {
        "n_shadow_decisions": len(shadow_decisions),
        "n_matched_to_official_picks": sum(1 for c in comparisons if c["official_pick_exists"]),
        "comparisons": comparisons,
    }
