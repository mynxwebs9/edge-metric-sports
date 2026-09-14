"""Phase 8B follow-up: immutable storage for generated game-preview articles.

    content/previews/season=<S>/week=<W>/<game_id>/run_id=<run_id>/
      context.json     the exact gathered context the article was written from
      preview.json      GamePreview, or a FailedPreviewRun - see manifest's "kind"
      manifest.json

Mirrors `research.storage`'s pattern exactly (never overwritten - a new run_id is required to
regenerate an article for a game) so the same reproducibility guarantee applies: a persisted
article is always traceable to the exact context and prompt version that produced it.

Phase 10: storage moved from raw `pathlib.Path` I/O to the pluggable `BlobStore` (see
`nfl_predict.storage.blob_store`), mirroring the same change in `research.storage`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from nfl_predict.storage.blob_store import get_blob_store

CONTENT_DIRNAME = "content/previews"


class PreviewRunAlreadyExistsError(Exception):
    """Raised when a preview run_id already exists - runs are immutable and never
    overwritten; use a new run_id to regenerate an article for this game."""


def _run_prefix(season: int, week: int, game_id: str, run_id: str) -> str:
    return f"{CONTENT_DIRNAME}/season={season}/week={week}/{game_id}/run_id={run_id}/"


def _to_jsonable(obj: Any) -> Any:
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def _encode_json(obj: Any) -> bytes:
    return json.dumps(_to_jsonable(obj), indent=2, sort_keys=True, default=str).encode("utf-8")


def write_preview_run(
    season: int, week: int, game_id: str, run_id: str, context: dict, preview_or_failure: Any, kind: str,
) -> str:
    """`kind`: "preview" or "failed_run" - recorded in the manifest so a reader knows which
    schema `preview.json` follows without guessing from its shape. Returns the run's key
    prefix."""
    store = get_blob_store()
    prefix = _run_prefix(season, week, game_id, run_id)
    if store.exists(prefix + "manifest.json"):
        raise PreviewRunAlreadyExistsError(
            f"Preview run already exists at {prefix} - runs are immutable and never "
            "overwritten. Use a new run_id to regenerate this game's article."
        )

    context_encoded = _encode_json(context)
    preview_encoded = _encode_json(preview_or_failure)
    store.write_bytes(prefix + "context.json", context_encoded)
    store.write_bytes(prefix + "preview.json", preview_encoded)
    context_hash = hashlib.sha256(context_encoded).hexdigest()
    preview_hash = hashlib.sha256(preview_encoded).hexdigest()

    manifest = {
        "season": season, "week": week, "game_id": game_id, "run_id": run_id,
        "kind": kind, "context_hash": context_hash, "preview_hash": preview_hash,
    }
    store.write_bytes(prefix + "manifest.json", _encode_json(manifest))
    return prefix


def list_preview_runs(season: int, week: int, game_id: str) -> list[str]:
    game_prefix = f"{CONTENT_DIRNAME}/season={season}/week={week}/{game_id}/"
    run_ids = set()
    for key in get_blob_store().list_keys(game_prefix):
        remainder = key[len(game_prefix):]
        if remainder.startswith("run_id="):
            run_ids.add(remainder.split("/", 1)[0][len("run_id="):])
    return sorted(run_ids)


def read_preview_run(season: int, week: int, game_id: str, run_id: str) -> dict:
    store = get_blob_store()
    prefix = _run_prefix(season, week, game_id, run_id)
    if not store.exists(prefix + "manifest.json"):
        raise FileNotFoundError(f"No preview run at {prefix}")
    result = {}
    for name in ("context", "preview", "manifest"):
        key = f"{prefix}{name}.json"
        if store.exists(key):
            result[name] = json.loads(store.read_bytes(key).decode("utf-8"))
    return result


def verify_preview_run_integrity(season: int, week: int, game_id: str, run_id: str) -> bool:
    store = get_blob_store()
    prefix = _run_prefix(season, week, game_id, run_id)
    manifest = json.loads(store.read_bytes(prefix + "manifest.json").decode("utf-8"))
    for key, filename in (("context_hash", "context.json"), ("preview_hash", "preview.json")):
        actual = hashlib.sha256(store.read_bytes(prefix + filename)).hexdigest()
        if actual != manifest[key]:
            raise ValueError(f"{filename} in preview run {run_id} does not match its recorded hash - it was modified after being written.")
    return True
