"""Phase 8A Steps 18-19: public-safe "ALL_MODEL_PREDICTIONS" storage - every upcoming game
gets one of these regardless of whether the Phase 7 decision engine calls it a Best Bet.
Distinct publication states (`DRAFT | READY | PUBLISHED | SUPERSEDED`); a newer snapshot may
mark an older one `SUPERSEDED`, but the original record is NEVER edited in place - a
superseding relationship is recorded as a separate marker file next to the original, so the
original JSON's bytes (and its hash) never change; `read_model_prediction` derives the
current display state by combining the original record with that marker.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

from nfl_predict.storage.blob_store import get_blob_store

PUBLIC_PREDICTIONS_DIRNAME = "live/public_predictions"


class PublicationState(str, Enum):
    DRAFT = "DRAFT"
    READY = "READY"
    PUBLISHED = "PUBLISHED"
    SUPERSEDED = "SUPERSEDED"


class PublicPredictionAlreadyExistsError(Exception):
    pass


@dataclass(frozen=True)
class PublicModelPrediction:
    prediction_id: str
    game_id: str
    season: int
    week: int
    generated_at: str
    elo_home_win_probability: float | None
    elo_predicted_margin: float | None
    ridge_predicted_margin: float | None
    lightgbm_predicted_margin: float | None
    model_agreement_all_agree: bool | None
    model_agreement_dispersion: float | None
    predicted_winner: str | None  # "home" | "away" | None


def _filesystem_safe(prediction_id: str) -> str:
    """`prediction_id` is often a raw ISO-8601 timestamp (e.g. from `_now_iso()`), which
    contains colons - illegal in Windows filenames. The true, colon-containing value is
    still what gets stored inside the record itself (see `publish_model_prediction`); only
    the on-disk filename goes through this. Idempotent, so round-tripping an id already
    read back from a filename (via `.stem`) is a no-op."""
    return prediction_id.replace(":", "-")


def _keys(game_id: str, prediction_id: str) -> tuple[str, str]:
    directory = f"{PUBLIC_PREDICTIONS_DIRNAME}/{game_id}"
    safe_id = _filesystem_safe(prediction_id)
    return f"{directory}/{safe_id}.json", f"{directory}/{safe_id}.sha256"


def _marker_key(game_id: str, prediction_id: str) -> str:
    return f"{PUBLIC_PREDICTIONS_DIRNAME}/{game_id}/{_filesystem_safe(prediction_id)}.superseded_by"


def publish_model_prediction(prediction: PublicModelPrediction, state: PublicationState) -> str:
    store = get_blob_store()
    data_key, hash_key = _keys(prediction.game_id, prediction.prediction_id)
    if store.exists(data_key):
        raise PublicPredictionAlreadyExistsError(f"prediction_id={prediction.prediction_id!r} for game_id={prediction.game_id!r} already exists - immutable once written.")
    payload = {**asdict(prediction), "state": state.value}
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    store.write_bytes(data_key, encoded)
    store.write_bytes(hash_key, hashlib.sha256(encoded).hexdigest().encode("utf-8"))
    return data_key


def supersede_previous_predictions(game_id: str, new_prediction_id: str) -> list[str]:
    """Marks every OTHER prediction record for this game as superseded by
    `new_prediction_id`, via a side marker file - never touches the old record's own JSON.
    Returns the list of prediction_ids just marked."""
    store = get_blob_store()
    directory_prefix = f"{PUBLIC_PREDICTIONS_DIRNAME}/{game_id}/"
    new_safe_id = _filesystem_safe(new_prediction_id)
    superseded = []
    for key in store.list_keys(directory_prefix):
        if not key.endswith(".json"):
            continue
        prediction_id = key[len(directory_prefix):-len(".json")]
        if prediction_id == new_safe_id:
            continue
        marker_key = f"{directory_prefix}{prediction_id}.superseded_by"
        if not store.exists(marker_key):
            store.write_bytes(marker_key, new_prediction_id.encode("utf-8"))
            superseded.append(prediction_id)
    return sorted(superseded)


def read_model_prediction(game_id: str, prediction_id: str) -> dict:
    store = get_blob_store()
    data_key, hash_key = _keys(game_id, prediction_id)
    content = store.read_bytes(data_key)
    if store.exists(hash_key):
        actual = hashlib.sha256(content).hexdigest()
        recorded = store.read_bytes(hash_key).decode("utf-8").strip()
        if actual != recorded:
            raise ValueError(f"Prediction {prediction_id!r} for {game_id!r} does not match its recorded hash - modified after being written.")
    record = json.loads(content.decode("utf-8"))
    marker_key = _marker_key(game_id, prediction_id)
    if store.exists(marker_key):
        record = {**record, "state": PublicationState.SUPERSEDED.value, "superseded_by": store.read_bytes(marker_key).decode("utf-8").strip()}
    return record


def list_predictions_for_game(game_id: str) -> list[str]:
    directory_prefix = f"{PUBLIC_PREDICTIONS_DIRNAME}/{game_id}/"
    return sorted(
        key[len(directory_prefix):-len(".json")]
        for key in get_blob_store().list_keys(directory_prefix)
        if key.endswith(".json")
    )
