"""Isolates every storage location the content step touches to `tmp_path` - mirrors
`tests/api/conftest.py`'s `api_data_dir` fixture, extended with `content.storage`.
"""

from __future__ import annotations

import pytest

from nfl_predict.data.db import get_connection, init_schema

STORAGE_MODULES = (
    "nfl_predict.data.db",
    # Phase 10: market.live_snapshot_store, market.event_game_mapping, research.storage,
    # decision.decision_log, decision.pick_ledger, live.prediction_publication, and
    # content.storage all read data_dir indirectly through nfl_predict.storage.blob_store now
    # - patching that one module's get_settings covers all of them.
    "nfl_predict.storage.blob_store",
    "nfl_predict.research.prospective_ledger",
)


@pytest.fixture
def content_data_dir(tmp_path, monkeypatch):
    for module in STORAGE_MODULES:
        monkeypatch.setattr(
            f"{module}.get_settings",
            lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})(),
        )

    conn = get_connection()
    init_schema(conn)
    conn.executemany(
        "INSERT INTO teams (team_id, canonical_abbr, name, nickname) VALUES (?, ?, ?, ?)",
        [("2310", "KC", "Kansas City Chiefs", "Chiefs"), ("1400", "DEN", "Denver Broncos", "Broncos")],
    )
    conn.execute(
        "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
        "home_team_id, away_team_id, home_team_abbr, away_team_abbr, game_status, normalized_at) "
        "VALUES ('2026_01_DEN_KC', 2026, 'REG', 1, '2026-09-14', '2026-09-14T20:15:00', "
        "'2310', '1400', 'KC', 'DEN', 'scheduled', '2026-09-14T00:00:00')"
    )
    conn.commit()
    conn.close()
    return tmp_path
