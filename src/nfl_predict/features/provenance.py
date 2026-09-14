"""Provenance manifests for a feature-build run.

Mirrors src/nfl_predict/data/provenance.py's pattern (same reasoning: a stored feature
dataset must be traceable back to exactly what produced it). Changing a feature's
definition must produce a new `feature_version` (see config/feature_engine.yaml), never
silently rewrite historical meaning under the same version - this manifest is what makes
that checkable after the fact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nfl_predict.data.provenance import compute_schema_fingerprint
from nfl_predict.data.raw_store import list_manifests


@dataclass(frozen=True)
class FeatureBuildManifest:
    feature_version: str
    table: str  # "team_game" | "game"
    seasons: list[int]
    generated_at: str  # UTC ISO 8601
    row_count: int
    column_count: int
    schema_fingerprint: str
    feature_registry_config_hash: str
    source_raw_snapshot_retrieval_ids: list[str]
    normalized_data_version: str
    git_revision: str | None
    extra: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def compute_feature_registry_config_hash(project_root: Path) -> str:
    """Hash of the two config files that fully determine feature *definitions*
    (config/features.yaml documents them; config/feature_engine.yaml sets the thresholds
    the code actually reads) - changing either changes this hash, which is recorded on
    every build so a stored feature dataset can be tied back to the exact config that
    produced it."""
    hasher = hashlib.sha256()
    for name in ("features.yaml", "feature_engine.yaml"):
        path = project_root / "config" / name
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def get_git_revision(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def collect_source_raw_snapshot_ids(seasons: list[int], datasets: list[str]) -> list[str]:
    """Every non-duplicate raw-snapshot retrieval_id for `datasets` covering `seasons` -
    what this feature build actually read from. Duplicate snapshots are excluded since they
    point at the same physical file as their canonical counterpart."""
    ids: list[str] = []
    for dataset in datasets:
        for m in list_manifests("nflverse", dataset):
            if m.duplicate_of_retrieval_id is not None:
                continue
            if m.requested_seasons and not set(m.requested_seasons) & set(seasons):
                continue
            ids.append(m.retrieval_id)
    return sorted(ids)


def build_feature_manifest(
    df_columns: list[str], df_dtypes: dict[str, str], row_count: int,
    table: str, seasons: list[int], feature_version: str, project_root: Path,
) -> FeatureBuildManifest:
    return FeatureBuildManifest(
        feature_version=feature_version,
        table=table,
        seasons=seasons,
        generated_at=datetime.now(timezone.utc).isoformat(),
        row_count=row_count,
        column_count=len(df_columns),
        schema_fingerprint=compute_schema_fingerprint(df_columns, df_dtypes),
        feature_registry_config_hash=compute_feature_registry_config_hash(project_root),
        source_raw_snapshot_retrieval_ids=collect_source_raw_snapshot_ids(
            seasons, ["schedules", "pbp", "teams", "snap_counts"]
        ),
        normalized_data_version="games-v1_pbp-v1",
        git_revision=get_git_revision(project_root),
    )
