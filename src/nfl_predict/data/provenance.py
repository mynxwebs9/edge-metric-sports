"""Provenance manifests for raw ingested snapshots.

Implements the Phase 1 contract defined in docs/ARCHITECTURE.md#raw-data-provenance: every
raw file fetched from an external source is accompanied by a manifest recording enough
information to know exactly what was fetched, when, and whether a later fetch produced
identical or different bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProvenanceManifest:
    """One manifest per raw snapshot. See docs/ARCHITECTURE.md#raw-data-provenance for the
    field-by-field rationale; this dataclass is the code-level mirror of that contract.
    """

    source_name: str
    dataset_name: str
    requested_seasons: list[int]
    loader_function: str
    loader_package_version: str
    retrieved_at: str  # UTC ISO 8601, e.g. "2026-09-10T23:12:45.123456+00:00"
    source_identifier: str
    # nflverse-data publishes rolling GitHub release tags that are updated in place, not
    # per-fetch immutable versions; nflreadpy's public API does not expose a resolvable
    # release/commit identifier for a given fetch. Record null rather than inventing one —
    # see docs/DATA_SOURCES.md's "Provenance requirements" section for the full explanation.
    source_release_identifier: str | None
    raw_file_path: str
    content_sha256: str
    row_count: int
    column_count: int
    column_names: list[str]
    column_dtypes: dict[str, str]
    schema_fingerprint: str
    retrieval_id: str
    # Set when this fetch produced content identical (by hash) to an earlier snapshot for
    # the same dataset — the fetch is still recorded, but the raw file is not duplicated on
    # disk; raw_file_path points at the original. See "Idempotency" in
    # docs/ARCHITECTURE.md#raw-data-provenance and the Phase 1 completion report.
    duplicate_of_retrieval_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ProvenanceManifest:
        return cls(**json.loads(text))


def compute_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks so large Parquet files don't need to fit
    in memory twice."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_schema_fingerprint(column_names: list[str], column_dtypes: dict[str, str]) -> str:
    """A stable hash of a dataset's (name, dtype) pairs, sorted by column name so column
    reordering alone doesn't change the fingerprint — only an actual schema change does."""
    pairs = sorted((name, column_dtypes.get(name, "")) for name in column_names)
    canonical = json.dumps(pairs, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(manifest: ProvenanceManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


def read_manifest(path: Path) -> ProvenanceManifest:
    return ProvenanceManifest.from_json(path.read_text(encoding="utf-8"))
