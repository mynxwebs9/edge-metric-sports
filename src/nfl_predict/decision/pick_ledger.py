"""Phase 7 Step 11/17: the immutable, append-only official published-pick ledger.

Event-sourced, not a mutable row-per-pick table: `publish_pick` appends a `PUBLISHED` event
carrying every immutable field (line, price, selection, market_type, snapshots); later state
changes (`settle_pick`, `void_pick`) append their OWN event, referencing `pick_id`, and never
touch the original `PUBLISHED` event's bytes. `read_current_picks` folds all events for a
pick_id into its current view. This is what makes "a published pick may never be deleted"
and "a published pick may never have its original line/price changed" both literally true
at the storage layer, not just a policy nobody enforces.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

from nfl_predict.storage.blob_store import get_blob_store

LEDGER_DIRNAME = "decision/pick_ledger"


class PickStatus(str, Enum):
    PUBLISHED = "PUBLISHED"
    VOID = "VOID"
    SETTLED = "SETTLED"


class SettlementResult(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"
    VOID = "VOID"


class PickCategory(str, Enum):
    """Step 10: distinct public categories that must never be conflated."""

    ALL_MODEL_PREDICTIONS = "ALL_MODEL_PREDICTIONS"
    LEANS = "LEANS"
    BEST_BETS = "BEST_BETS"


VALID_VOID_REASONS = frozenset({
    "GAME_CANCELLED", "SPORTSBOOK_MARKET_VOIDED", "CORRUPTED_INPUT_DETECTED_PRE_EVENT", "DUPLICATE_PUBLICATION",
})


class PickAlreadyExistsError(Exception):
    """Raised when a pick_id has already been published - picks are immutable, never
    overwritten or republished under the same id."""


class PickNotFoundError(Exception):
    pass


class InvalidPickStateTransitionError(Exception):
    """Raised when settling/voiding a pick that is not currently PUBLISHED (e.g. settling
    an already-settled or already-voided pick)."""


class InvalidVoidReasonError(Exception):
    """Raised for a void_reason outside `VALID_VOID_REASONS` - a losing pick can never be
    voided merely because the model/data turned out to be wrong (Step 17)."""


@dataclass(frozen=True)
class PublishedPick:
    pick_id: str
    game_id: str
    published_at: str
    kickoff_at: str | None
    decision_id: str
    rule_version: str
    category: str  # PickCategory value
    market_type: str  # "spread" | "moneyline" - Step 16, no totals yet
    selection: str
    line: float | None
    price: int  # REQUIRED - a pick is never published without a real, captured price (Step 15)
    sportsbook_or_source: str
    market_snapshot_id: str | None
    model_prediction_snapshot: dict
    research_snapshot_id: str | None
    validation_status: str  # "PROSPECTIVE" for every Phase 7 pick (Step 18)


def _ledger_keys() -> tuple[str, str]:
    return f"{LEDGER_DIRNAME}/events.jsonl", f"{LEDGER_DIRNAME}/events.sha256"


def _append_event(event: dict) -> None:
    data_key, hash_key = _ledger_keys()
    store = get_blob_store()
    store.append_bytes(data_key, (json.dumps(event, sort_keys=True, default=str) + "\n").encode("utf-8"))
    digest = hashlib.sha256(store.read_bytes(data_key)).hexdigest()
    store.write_bytes(hash_key, digest.encode("utf-8"), exist_ok=True)


def _read_events() -> list[dict]:
    data_key, hash_key = _ledger_keys()
    store = get_blob_store()
    if not store.exists(data_key):
        return []
    content = store.read_bytes(data_key)
    if store.exists(hash_key):
        actual = hashlib.sha256(content).hexdigest()
        recorded = store.read_bytes(hash_key).decode("utf-8").strip()
        if actual != recorded:
            raise ValueError("Pick ledger does not match its recorded hash - it was modified after being written.")
    return [json.loads(line) for line in content.decode("utf-8").splitlines()]


def publish_pick(pick: PublishedPick) -> None:
    for event in _read_events():
        if event["pick_id"] == pick.pick_id and event["event_type"] == "PUBLISHED":
            raise PickAlreadyExistsError(f"pick_id={pick.pick_id!r} already published - picks are never republished under the same id.")
    _append_event({"event_type": "PUBLISHED", "pick_id": pick.pick_id, **asdict(pick)})


def _current_status(pick_id: str, events: list[dict]) -> str:
    status = None
    for e in events:
        if e["pick_id"] != pick_id:
            continue
        if e["event_type"] == "PUBLISHED":
            status = PickStatus.PUBLISHED.value
        elif e["event_type"] == "SETTLED":
            status = PickStatus.SETTLED.value
        elif e["event_type"] == "VOIDED":
            status = PickStatus.VOID.value
    return status


def settle_pick(pick_id: str, settlement: SettlementResult, settled_at: str, result_source: str) -> None:
    events = _read_events()
    if not any(e["pick_id"] == pick_id and e["event_type"] == "PUBLISHED" for e in events):
        raise PickNotFoundError(f"No published pick with pick_id={pick_id!r}")
    if _current_status(pick_id, events) != PickStatus.PUBLISHED.value:
        raise InvalidPickStateTransitionError(f"pick_id={pick_id!r} is not currently PUBLISHED - cannot settle a pick that is already settled or voided.")
    _append_event({
        "event_type": "SETTLED", "pick_id": pick_id, "settlement": settlement.value,
        "settled_at": settled_at, "result_source": result_source,
    })


def void_pick(pick_id: str, void_reason: str, void_timestamp: str, authorized_by: str) -> None:
    if void_reason not in VALID_VOID_REASONS:
        raise InvalidVoidReasonError(f"void_reason={void_reason!r} is not an allowed reason - must be one of {sorted(VALID_VOID_REASONS)}")
    events = _read_events()
    if not any(e["pick_id"] == pick_id and e["event_type"] == "PUBLISHED" for e in events):
        raise PickNotFoundError(f"No published pick with pick_id={pick_id!r}")
    if _current_status(pick_id, events) != PickStatus.PUBLISHED.value:
        raise InvalidPickStateTransitionError(f"pick_id={pick_id!r} is not currently PUBLISHED - cannot void a pick that is already settled or voided.")
    _append_event({
        "event_type": "VOIDED", "pick_id": pick_id, "void_reason": void_reason,
        "void_timestamp": void_timestamp, "void_authorized_by": authorized_by,
    })


def read_current_picks() -> list[dict]:
    """Folds every event into each pick_id's current view - the PUBLISHED event's fields
    (line, price, selection, ...) are always exactly what was originally published; only
    `status`/`settlement`/`settled_at`/`result_source`/`void_*` reflect later events."""
    events = _read_events()
    picks: dict[str, dict] = {}
    order: list[str] = []
    for e in events:
        pick_id = e["pick_id"]
        if e["event_type"] == "PUBLISHED":
            picks[pick_id] = {k: v for k, v in e.items() if k != "event_type"}
            picks[pick_id]["status"] = PickStatus.PUBLISHED.value
            picks[pick_id].setdefault("settlement", None)
            picks[pick_id].setdefault("settled_at", None)
            picks[pick_id].setdefault("result_source", None)
            picks[pick_id].setdefault("void_reason", None)
            picks[pick_id].setdefault("void_timestamp", None)
            picks[pick_id].setdefault("void_authorized_by", None)
            order.append(pick_id)
        elif e["event_type"] == "SETTLED":
            picks[pick_id]["status"] = PickStatus.SETTLED.value
            picks[pick_id]["settlement"] = e["settlement"]
            picks[pick_id]["settled_at"] = e["settled_at"]
            picks[pick_id]["result_source"] = e["result_source"]
        elif e["event_type"] == "VOIDED":
            picks[pick_id]["status"] = PickStatus.VOID.value
            picks[pick_id]["void_reason"] = e["void_reason"]
            picks[pick_id]["void_timestamp"] = e["void_timestamp"]
            picks[pick_id]["void_authorized_by"] = e["void_authorized_by"]
    return [picks[pid] for pid in order]
