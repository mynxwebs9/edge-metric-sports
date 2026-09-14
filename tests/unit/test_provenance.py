import hashlib
from pathlib import Path

from nfl_predict.data.provenance import (
    ProvenanceManifest,
    compute_schema_fingerprint,
    compute_sha256,
    read_manifest,
    write_manifest,
)


def test_compute_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"some deterministic content")

    assert compute_sha256(path) == hashlib.sha256(b"some deterministic content").hexdigest()


def test_compute_sha256_differs_for_different_content(tmp_path):
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"content A")
    path_b.write_bytes(b"content B")

    assert compute_sha256(path_a) != compute_sha256(path_b)


def test_schema_fingerprint_is_order_independent():
    fp1 = compute_schema_fingerprint(["a", "b"], {"a": "Int64", "b": "Utf8"})
    fp2 = compute_schema_fingerprint(["b", "a"], {"a": "Int64", "b": "Utf8"})
    assert fp1 == fp2


def test_schema_fingerprint_changes_with_dtype():
    fp1 = compute_schema_fingerprint(["a"], {"a": "Int64"})
    fp2 = compute_schema_fingerprint(["a"], {"a": "Float64"})
    assert fp1 != fp2


def test_schema_fingerprint_changes_with_column_set():
    fp1 = compute_schema_fingerprint(["a", "b"], {"a": "Int64", "b": "Utf8"})
    fp2 = compute_schema_fingerprint(["a"], {"a": "Int64"})
    assert fp1 != fp2


def _sample_manifest(**overrides) -> ProvenanceManifest:
    base = dict(
        source_name="nflverse",
        dataset_name="schedules",
        requested_seasons=[2025],
        loader_function="load_schedules",
        loader_package_version="0.1.5",
        retrieved_at="2026-09-10T23:00:00+00:00",
        source_identifier="nflverse-data GitHub release, via nflreadpy.load_schedules()",
        source_release_identifier=None,
        raw_file_path="/tmp/data.parquet",
        content_sha256="abc123",
        row_count=285,
        column_count=46,
        column_names=["game_id", "season"],
        column_dtypes={"game_id": "Utf8", "season": "Int32"},
        schema_fingerprint="fp",
        retrieval_id="2025_20260910T230000000000Z",
    )
    base.update(overrides)
    return ProvenanceManifest(**base)


def test_manifest_roundtrip_via_json(tmp_path: Path):
    manifest = _sample_manifest()
    path = tmp_path / "manifest.json"

    write_manifest(manifest, path)
    loaded = read_manifest(path)

    assert loaded == manifest


def test_manifest_records_duplicate_of_when_set():
    manifest = _sample_manifest(duplicate_of_retrieval_id="2025_20260101T000000000000Z")
    assert manifest.duplicate_of_retrieval_id == "2025_20260101T000000000000Z"


def test_manifest_defaults_duplicate_of_to_none():
    manifest = _sample_manifest()
    assert manifest.duplicate_of_retrieval_id is None
