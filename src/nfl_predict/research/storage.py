"""Phase 6 Step 16: immutable research storage.

    research/season=<S>/week=<W>/<game_id>/run_id=<run_id>/
      input.json
      findings.json       (ResearchFindings, or a FailedResearchRun - see manifest's "kind")
      evaluation.json     (EvaluationResult - absent if evaluation hasn't run yet)
      manifest.json

Never overwrites: `write_research_run` raises `ResearchRunAlreadyExistsError` if the target
run_id already exists, so multiple runs for one game (Thursday research, Friday research,
Sunday-morning research) are always preserved as separate runs, never merged or replaced -
this is what STEP 25/24-proof-#10 depends on.

Phase 10: storage moved from raw `pathlib.Path` I/O to the pluggable `BlobStore` (see
`nfl_predict.storage.blob_store`) so this same logic works unchanged against a hosted
Postgres backend, not just the local filesystem. Behavior and on-disk layout under the local
backend are unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from nfl_predict.storage.blob_store import get_blob_store

RESEARCH_DIRNAME = "research"


class ResearchRunAlreadyExistsError(Exception):
    """Raised when a research run_id already exists - runs are immutable and
    never overwritten; use a new run_id for a new run."""


def _run_prefix(season: int, week: int, game_id: str, run_id: str) -> str:
    return f"{RESEARCH_DIRNAME}/season={season}/week={week}/{game_id}/run_id={run_id}/"


def _game_prefix(season: int, week: int, game_id: str) -> str:
    return f"{RESEARCH_DIRNAME}/season={season}/week={week}/{game_id}/"


def _to_jsonable(obj: Any) -> Any:
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def _encode_json(obj: Any) -> bytes:
    # json.dumps + encode("utf-8"), not any text-mode write: guarantees the hashed bytes and
    # the stored bytes are identical, matching Phase 4/5's SHA-256 tamper-evidence pattern.
    return json.dumps(_to_jsonable(obj), indent=2, sort_keys=True, default=str).encode("utf-8")


def write_research_run(
    season: int, week: int, game_id: str, run_id: str, input_packet: Any,
    findings_or_failure: Any, findings_kind: str, evaluation: Any | None = None,
) -> str:
    """`findings_kind`: "findings" or "failed_run" - recorded in the manifest so a reader
    knows which schema `findings.json` follows without guessing from its shape. Returns the
    run's key prefix."""
    store = get_blob_store()
    prefix = _run_prefix(season, week, game_id, run_id)
    if store.exists(prefix + "manifest.json"):
        raise ResearchRunAlreadyExistsError(
            f"Research run already exists at {prefix} - runs are immutable and never "
            "overwritten. Use a new run_id for a new research run on this game."
        )

    input_encoded = _encode_json(input_packet)
    findings_encoded = _encode_json(findings_or_failure)
    store.write_bytes(prefix + "input.json", input_encoded)
    store.write_bytes(prefix + "findings.json", findings_encoded)
    input_hash = hashlib.sha256(input_encoded).hexdigest()
    findings_hash = hashlib.sha256(findings_encoded).hexdigest()

    evaluation_hash = None
    if evaluation is not None:
        evaluation_encoded = _encode_json(evaluation)
        store.write_bytes(prefix + "evaluation.json", evaluation_encoded)
        evaluation_hash = hashlib.sha256(evaluation_encoded).hexdigest()

    manifest = {
        "season": season, "week": week, "game_id": game_id, "run_id": run_id,
        "findings_kind": findings_kind,
        "input_hash": input_hash, "findings_hash": findings_hash, "evaluation_hash": evaluation_hash,
    }
    store.write_bytes(prefix + "manifest.json", _encode_json(manifest))
    return prefix


def list_research_runs(season: int, week: int, game_id: str) -> list[str]:
    game_prefix = _game_prefix(season, week, game_id)
    run_ids = set()
    for key in get_blob_store().list_keys(game_prefix):
        remainder = key[len(game_prefix):]
        if remainder.startswith("run_id="):
            run_ids.add(remainder.split("/", 1)[0][len("run_id="):])
    return sorted(run_ids)


def read_research_run(season: int, week: int, game_id: str, run_id: str) -> dict:
    store = get_blob_store()
    prefix = _run_prefix(season, week, game_id, run_id)
    if not store.exists(prefix + "manifest.json"):
        raise FileNotFoundError(f"No research run at {prefix}")
    result = {}
    for name in ("input", "findings", "evaluation", "manifest"):
        key = f"{prefix}{name}.json"
        if store.exists(key):
            result[name] = json.loads(store.read_bytes(key).decode("utf-8"))
    return result


def verify_research_run_integrity(season: int, week: int, game_id: str, run_id: str) -> bool:
    """Re-hashes every stored file and compares against the manifest's recorded hashes -
    the same tamper-evidence pattern as Phase 4/5's freeze manifest and prediction ledger."""
    store = get_blob_store()
    prefix = _run_prefix(season, week, game_id, run_id)
    manifest = json.loads(store.read_bytes(prefix + "manifest.json").decode("utf-8"))

    for key, filename in (("input_hash", "input.json"), ("findings_hash", "findings.json")):
        actual = hashlib.sha256(store.read_bytes(prefix + filename)).hexdigest()
        if actual != manifest[key]:
            raise ValueError(f"{filename} in run {run_id} does not match its recorded hash - it was modified after being written.")

    if manifest.get("evaluation_hash") is not None:
        actual = hashlib.sha256(store.read_bytes(prefix + "evaluation.json")).hexdigest()
        if actual != manifest["evaluation_hash"]:
            raise ValueError(f"evaluation.json in run {run_id} does not match its recorded hash - it was modified after being written.")

    return True
