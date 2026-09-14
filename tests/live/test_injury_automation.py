"""Phase 8A Step 11/28: missing structured-injury-data key produces an explicit unavailable
status - never a silent fallback to FixtureInjuryProvider - and this status is tracked
separately from the (also-unconfigured) LLM research provider status."""

from __future__ import annotations

from nfl_predict.live.injury_automation import InjuryProviderStatus, get_production_injury_provider


def test_missing_injury_key_produces_explicit_unavailable_status(monkeypatch):
    monkeypatch.delenv("NFL_INJURY_API_KEY", raising=False)
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    result = get_production_injury_provider()
    assert result.status == InjuryProviderStatus.INJURY_PROVIDER_UNAVAILABLE
    assert result.provider is None
    get_settings.cache_clear()


def test_configured_injury_key_produces_available_status(monkeypatch):
    monkeypatch.setenv("NFL_INJURY_API_KEY", "test-key-not-a-real-secret")
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    result = get_production_injury_provider()
    assert result.status == InjuryProviderStatus.AVAILABLE
    assert result.provider is not None
    get_settings.cache_clear()


def test_injury_status_is_reported_separately_from_research_status():
    """Structural proof of the boundary: injury_automation's own code (not its docstring)
    never imports from research_automation (the LLM/web research provider), and vice versa -
    they are checked and reported as fully independent components."""
    import ast
    import inspect

    from nfl_predict.live import injury_automation, research_automation

    def _imported_module_names(module) -> set[str]:
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
            elif isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
        return names

    assert not any("research_automation" in name for name in _imported_module_names(injury_automation))
    assert not any("injury_automation" in name for name in _imported_module_names(research_automation))
