"""Phase 7: an append-only, hash-verified log of every `DecisionRecord` the engine ever
produced - separate from the official pick ledger (`pick_ledger.py`). Only a QUALIFIED_BET
(or, per a future policy, a LEAN) decision might go on to become a published pick; NO_BET /
WATCH / VETO decisions are still recorded here for a full audit trail but never enter the
public ledger.

Phase 10: storage moved from raw `pathlib.Path` I/O to the pluggable `BlobStore` (see
`nfl_predict.storage.blob_store`); the append-once-per-record-plus-sidecar-hash behavior is
unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from nfl_predict.decision.schemas import DecisionRecord
from nfl_predict.storage.blob_store import get_blob_store

DECISION_LOG_DIRNAME = "decision/decision_log"


def _keys(season: int, week: int) -> tuple[str, str]:
    prefix = f"{DECISION_LOG_DIRNAME}/season={season}/week={week}"
    return f"{prefix}/decisions.jsonl", f"{prefix}/decisions.sha256"


def append_decision_record(record: DecisionRecord, season: int, week: int) -> None:
    data_key, hash_key = _keys(season, week)
    store = get_blob_store()
    store.append_bytes(data_key, (json.dumps(asdict(record), sort_keys=True, default=str) + "\n").encode("utf-8"))
    digest = hashlib.sha256(store.read_bytes(data_key)).hexdigest()
    store.write_bytes(hash_key, digest.encode("utf-8"), exist_ok=True)


def read_decision_records(season: int, week: int) -> list[dict]:
    data_key, hash_key = _keys(season, week)
    store = get_blob_store()
    if not store.exists(data_key):
        return []
    content = store.read_bytes(data_key)
    if store.exists(hash_key):
        actual = hashlib.sha256(content).hexdigest()
        recorded = store.read_bytes(hash_key).decode("utf-8").strip()
        if actual != recorded:
            raise ValueError(f"Decision log for season={season} week={week} does not match its recorded hash - it was modified after being written.")
    return [json.loads(line) for line in content.decode("utf-8").splitlines()]
