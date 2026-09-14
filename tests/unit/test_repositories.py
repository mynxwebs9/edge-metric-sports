from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.provenance import ProvenanceManifest
from nfl_predict.data.repositories import ManifestsRepository


def _manifest(retrieval_id: str, requested_seasons: list[int], retrieved_at: str, column_names: list[str], duplicate_of: str | None = None) -> ProvenanceManifest:
    return ProvenanceManifest(
        source_name="nflverse",
        dataset_name="schedules",
        requested_seasons=requested_seasons,
        loader_function="load_schedules",
        loader_package_version="0.1.5",
        retrieved_at=retrieved_at,
        source_identifier="test",
        source_release_identifier=None,
        raw_file_path=f"/tmp/{retrieval_id}.parquet",
        content_sha256=f"hash-{retrieval_id}",
        row_count=10,
        column_count=len(column_names),
        column_names=column_names,
        column_dtypes={c: "Utf8" for c in column_names},
        schema_fingerprint=f"fp-{retrieval_id}",
        retrieval_id=retrieval_id,
        duplicate_of_retrieval_id=duplicate_of,
    )


def test_get_latest_canonical_manifest_prefers_most_recent_retrieval(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    repo = ManifestsRepository(conn)

    # Simulates nflverse correcting historical data for 2019 and the season being
    # re-fetched later - two real, non-duplicate snapshots for the same season.
    old = _manifest("2019_old", [2019], "2026-01-01T00:00:00+00:00", ["a", "b"])
    new = _manifest("2019_new", [2019], "2026-06-01T00:00:00+00:00", ["a", "b", "c"])
    # Inserted deliberately out of chronological order to prove the method isn't relying on
    # insertion/file-listing order.
    repo.record(new)
    repo.record(old)

    result = repo.get_latest_canonical_manifest("nflverse", "schedules", 2019)

    assert result["retrieval_id"] == "2019_new"
    assert result["column_names"] == '["a", "b", "c"]'
    conn.close()


def test_get_latest_canonical_manifest_ignores_duplicates(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    repo = ManifestsRepository(conn)

    canonical = _manifest("2020_canonical", [2020], "2026-01-01T00:00:00+00:00", ["a", "b"])
    later_duplicate = _manifest(
        "2020_dup", [2020], "2026-06-01T00:00:00+00:00", ["a", "b"], duplicate_of="2020_canonical"
    )
    repo.record(canonical)
    repo.record(later_duplicate)

    result = repo.get_latest_canonical_manifest("nflverse", "schedules", 2020)

    # The duplicate has a later retrieved_at but must not be selected - it isn't canonical.
    assert result["retrieval_id"] == "2020_canonical"
    conn.close()


def test_get_latest_canonical_manifest_returns_none_when_absent(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    repo = ManifestsRepository(conn)

    assert repo.get_latest_canonical_manifest("nflverse", "schedules", 1999) is None
    conn.close()


def test_get_latest_canonical_manifest_does_not_match_multi_season_requests(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    repo = ManifestsRepository(conn)

    multi = _manifest("multi", [2019, 2020], "2026-01-01T00:00:00+00:00", ["a"])
    repo.record(multi)

    # A manifest that batched multiple seasons together is not "the" canonical single-season
    # snapshot for 2019 - this project only ever fetches one season at a time in practice,
    # but the method's contract should not silently misattribute a multi-season manifest.
    assert repo.get_latest_canonical_manifest("nflverse", "schedules", 2019) is None
    conn.close()
