"""Phase 8A Step 28 proofs #14, #22, #24: research runs before the deterministic decision,
dry run cannot publish, and re-running identical inputs is idempotent (no duplicate
decision-log entries when run in dry-run mode; append-only decision log otherwise)."""

from __future__ import annotations

import inspect

from nfl_predict.live import run as run_module


def test_dry_run_never_calls_append_decision_record(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_module, "append_decision_record", lambda *a, **k: calls.append((a, k)))
    result = run_module.run_slate(game_id="2026_01_DEN_KC", dry_run=True)
    assert calls == []
    assert result["dry_run"] is True


def test_orchestrator_never_calls_publish_pick_directly():
    """Structural proof: `nfl_predict.live.run` never imports or calls
    `nfl_predict.decision.pick_ledger.publish_pick` - official-pick publication is a
    deliberately separate, not-yet-wired step (Step 2/27: a normal run must not
    auto-publish)."""
    source = inspect.getsource(run_module)
    assert "publish_pick(" not in source


def test_research_provider_status_is_resolved_before_any_decision_is_computed():
    """Structural proof: within `run_slate`, the research-provider resolution happens
    strictly before the decisions loop that calls `decide()` - checked by source order,
    since research is deliberately a separate upstream step (Phase 6 before Phase 7)."""
    source = inspect.getsource(run_module.run_slate)
    research_idx = source.index("research_status = ResearchProviderStatus")
    decide_idx = source.index("decide(packet, rule_set, market_type)")
    assert research_idx < decide_idx


def test_a_failed_model_component_does_not_prevent_the_run_from_completing(monkeypatch):
    monkeypatch.setattr(run_module, "predict_live_elo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("elo boom")))
    result = run_module.run_slate(game_id="2026_01_DEN_KC", dry_run=True)
    assert result["health"]["elo_model"]["state"] == "FAILED"
    assert "decision_counts" in result  # the run still completed and reached decisions


def test_running_the_same_dry_run_twice_produces_no_duplicate_persisted_state(tmp_path, monkeypatch):
    """Idempotency: two identical dry-run invocations must not leave behind any
    accumulated on-disk state (dry runs write nothing) - the decision log stays empty both
    times."""
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    from nfl_predict.decision.decision_log import read_decision_records

    run_module.run_slate(game_id="2026_01_DEN_KC", dry_run=True)
    run_module.run_slate(game_id="2026_01_DEN_KC", dry_run=True)
    assert read_decision_records(2026, 1) == []
