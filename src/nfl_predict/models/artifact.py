"""Model artifact persistence with full provenance.

Storage layout:
    data/models/<model_version>/<family>/<model_id>.joblib   (the fitted model object)
    data/models/<model_version>/<family>/<model_id>.json     (the manifest below)

`git_revision` is read via the same helper Phase 2 uses for feature-build manifests
(`nfl_predict.features.provenance.get_git_revision`) - if there are no commits, it is
`null`, never fabricated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from nfl_predict.config import get_settings
from nfl_predict.features.provenance import compute_feature_registry_config_hash, get_git_revision

MODELS_DIRNAME = "models"


@dataclass(frozen=True)
class ModelArtifactManifest:
    model_id: str
    model_type: str
    model_version: str
    feature_version: str | None
    feature_names: list[str]
    feature_config_hash: str | None
    training_seasons: list[int]
    validation_season: int | None
    training_row_count: int
    creation_timestamp: str
    git_revision: str | None
    hyperparameters: dict[str, Any]
    preprocessing: str
    target: str
    metrics: dict[str, Any]
    artifact_hash: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


def _artifact_dir(model_version: str, family: str) -> Path:
    return get_settings().data_dir / MODELS_DIRNAME / model_version / family


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_model_artifact(
    model_object: Any,
    *,
    model_id: str,
    model_type: str,
    model_version: str,
    family: str,
    feature_version: str | None,
    feature_names: list[str],
    training_seasons: list[int],
    validation_season: int | None,
    training_row_count: int,
    hyperparameters: dict[str, Any],
    preprocessing: str,
    target: str,
    metrics: dict[str, Any],
    project_root: Path,
    extra: dict[str, Any] | None = None,
) -> ModelArtifactManifest:
    directory = _artifact_dir(model_version, family)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{model_id}.joblib"
    joblib.dump(model_object, model_path)
    artifact_hash = _hash_file(model_path)

    feature_config_hash = None
    try:
        feature_config_hash = compute_feature_registry_config_hash(project_root)
    except FileNotFoundError:
        pass

    manifest = ModelArtifactManifest(
        model_id=model_id, model_type=model_type, model_version=model_version,
        feature_version=feature_version, feature_names=feature_names,
        feature_config_hash=feature_config_hash, training_seasons=training_seasons,
        validation_season=validation_season, training_row_count=training_row_count,
        creation_timestamp=datetime.now(timezone.utc).isoformat(),
        git_revision=get_git_revision(project_root),
        hyperparameters=hyperparameters, preprocessing=preprocessing, target=target,
        metrics=metrics, artifact_hash=artifact_hash, extra=extra or {},
    )
    (directory / f"{model_id}.json").write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def load_model_artifact(model_version: str, family: str, model_id: str) -> tuple[Any, ModelArtifactManifest]:
    directory = _artifact_dir(model_version, family)
    model_object = joblib.load(directory / f"{model_id}.joblib")
    manifest_dict = json.loads((directory / f"{model_id}.json").read_text(encoding="utf-8"))
    return model_object, ModelArtifactManifest(**manifest_dict)
