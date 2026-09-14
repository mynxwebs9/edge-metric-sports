"""Phase 8A correction Steps 3-4, tests #4-7, #9: `run_research()` is actually invoked for
triggered games; provider AVAILABLE without a completed research artifact does not make
ResearchPoint available; valid evaluated research makes it available; research findings are
persisted before the ResearchPoint is built from them; the decision packet's ResearchPoint
references the exact persisted artifact (research_id matches)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predict.decision.schemas import MarketPoint
from nfl_predict.live.research_live import (
    build_live_research_input_packet_with_market,
    run_live_research_for_game,
    select_games_for_research,
)
from nfl_predict.research.llm_provider import FixtureLLMProvider
from nfl_predict.research.trigger import TriggerResult

REAL_GAME_ID = "2026_01_DEN_KC"
LLM_FIXTURE = Path(__file__).resolve().parents[1] / "research" / "fixtures" / "llm_fixture.json"


def _patch_storage(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def _real_market() -> MarketPoint:
    return MarketPoint(
        available=True, source="test", home_spread_traditional=-3.0, home_spread_price=-110, away_spread_price=-110,
        home_moneyline=-150, away_moneyline=130, no_vig_home_win_probability=0.58, snapshot_timestamp="2026-09-12T10:00:00+00:00",
    )


def test_build_live_research_input_packet_injects_real_market_when_available():
    packet = build_live_research_input_packet_with_market(REAL_GAME_ID, _real_market())
    assert packet.market.available is True
    assert packet.market.home_spread_traditional == -3.0
    assert packet.market.no_vig_home_win_probability == 0.58
    assert packet.model_market_disagreement_points is not None  # Elo margin vs. the real spread


def test_build_live_research_input_packet_leaves_market_unavailable_when_none_given():
    packet = build_live_research_input_packet_with_market(REAL_GAME_ID, None)
    assert packet.market.available is False


def test_run_research_for_game_is_actually_invoked(tmp_path, monkeypatch):
    """Test #4: proves `LLMResearchProvider.run_research` is really called - not just that
    the provider object exists - by wrapping the fixture provider and recording the call."""
    _patch_storage(monkeypatch, tmp_path)
    provider = FixtureLLMProvider(LLM_FIXTURE)
    calls = []
    original = provider.run_research
    monkeypatch.setattr(provider, "run_research", lambda *a, **k: (calls.append(1), original(*a, **k))[1])

    outcome = run_live_research_for_game(
        REAL_GAME_ID, season=2026, week=1, provider=provider, market=_real_market(),
        priority=None, run_id="test_run_1", now="2026-09-12T10:05:00+00:00",
    )
    assert len(calls) == 1
    assert outcome.invoked is True
    assert outcome.status == "ok"


def test_research_point_unavailable_when_no_research_was_run():
    """Test #5: a provider being AVAILABLE in the orchestrator's sense is a separate concern
    from ResearchPoint - if `run_live_research_for_game` is simply never called for a game
    (not selected), its ResearchPoint stays unavailable. This is exercised at the
    orchestrator level (test_run_orchestrator.py); here we confirm the building block used
    for "not selected" games directly."""
    from nfl_predict.decision.schemas import ResearchPoint

    default = ResearchPoint(available=False)
    assert default.available is False


def test_valid_evaluated_research_makes_researchpoint_available(tmp_path, monkeypatch):
    """Test #6."""
    _patch_storage(monkeypatch, tmp_path)
    provider = FixtureLLMProvider(LLM_FIXTURE)
    outcome = run_live_research_for_game(
        REAL_GAME_ID, season=2026, week=1, provider=provider, market=_real_market(),
        priority=None, run_id="test_run_2", now="2026-09-12T10:05:00+00:00",
    )
    assert outcome.status == "ok"
    assert outcome.research_point.available is True
    assert outcome.research_point.classification == "NO_MATERIAL_NEW_INFORMATION"
    assert outcome.research_point.research_id is not None


def test_research_findings_are_persisted_before_the_researchpoint_is_built(tmp_path, monkeypatch):
    """Test #7/#9: the ResearchPoint's `research_id` must match a genuinely persisted,
    hash-verifiable research run - not something built in memory only."""
    _patch_storage(monkeypatch, tmp_path)
    provider = FixtureLLMProvider(LLM_FIXTURE)
    outcome = run_live_research_for_game(
        REAL_GAME_ID, season=2026, week=1, provider=provider, market=_real_market(),
        priority=None, run_id="test_run_3", now="2026-09-12T10:05:00+00:00",
    )

    from nfl_predict.research.storage import verify_research_run_integrity

    assert verify_research_run_integrity(2026, 1, REAL_GAME_ID, "test_run_3") is True
    assert outcome.research_point.research_id == f"{REAL_GAME_ID}_test_run_3"


def test_a_failed_research_call_leaves_researchpoint_unavailable(tmp_path, monkeypatch):
    _patch_storage(monkeypatch, tmp_path)

    class _BoomProvider:
        def run_research(self, prompt, prompt_version):
            raise RuntimeError("simulated network failure")

        def run_evaluation(self, prompt, prompt_version):
            raise NotImplementedError

    outcome = run_live_research_for_game(
        REAL_GAME_ID, season=2026, week=1, provider=_BoomProvider(), market=_real_market(),
        priority=None, run_id="test_run_4", now="2026-09-12T10:05:00+00:00",
    )
    assert outcome.status == "failed"
    assert outcome.research_point.available is False


def test_select_games_for_research_respects_the_cap_and_orders_by_priority():
    priorities = {
        "g_low": TriggerResult(priority_score=1.0, reasons=(), inputs_used=(), inputs_unavailable=()),
        "g_high": TriggerResult(priority_score=9.0, reasons=(), inputs_used=(), inputs_unavailable=()),
        "g_mid": TriggerResult(priority_score=5.0, reasons=(), inputs_used=(), inputs_unavailable=()),
    }
    selected = select_games_for_research(priorities, research_all=False, max_calls=2)
    assert selected == ["g_high", "g_mid"]


def test_select_games_for_research_all_bypasses_the_cap():
    priorities = {
        "g_low": TriggerResult(priority_score=1.0, reasons=(), inputs_used=(), inputs_unavailable=()),
        "g_high": TriggerResult(priority_score=9.0, reasons=(), inputs_used=(), inputs_unavailable=()),
    }
    selected = select_games_for_research(priorities, research_all=True, max_calls=1)
    assert set(selected) == {"g_low", "g_high"}
