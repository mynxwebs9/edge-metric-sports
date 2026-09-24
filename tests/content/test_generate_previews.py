"""The weekly preview command: one article per upcoming game, never double-billing a game
that already has a good article, and never letting a failed retry hide a good one. No live
network call - a counting fake provider stands in for Anthropic."""

from __future__ import annotations

from pathlib import Path

from nfl_predict.content.generate_previews import generate_previews_for_week
from nfl_predict.content.prediction_writer import (
    ContentWriteResult,
    ContentWriterProvider,
    ContentWriterResponseError,
)
from nfl_predict.api import reconstruction as recon

GAME_ID = "2026_01_DEN_KC"
SEASON, WEEK = 2026, 1
PROMPT_TEMPLATE = (Path(__file__).resolve().parents[2] / "prompts" / "prediction_writer_v2.md").read_text(encoding="utf-8")


class _CountingProvider(ContentWriterProvider):
    def __init__(self, cost: float | None = 0.03):
        self.calls = 0
        self._cost = cost

    def write(self, prompt: str) -> ContentWriteResult:
        self.calls += 1
        return ContentWriteResult("fake", "fake-model", "A real-looking preview article.", 100, 50, self._cost)


class _FailingProvider(ContentWriterProvider):
    def write(self, prompt: str) -> ContentWriteResult:
        raise ContentWriterResponseError("No text content in response: stop_reason='max_tokens'")


def test_generates_one_preview_per_upcoming_game_and_reports_real_cost(content_data_dir):
    provider = _CountingProvider()
    result = generate_previews_for_week(WEEK, provider, PROMPT_TEMPLATE, now="2026-09-13T12:00:00+00:00")

    assert result["generated"] == [GAME_ID]
    assert result["skipped_existing"] == [] and result["failed"] == []
    assert result["estimated_cost_usd"] == 0.03
    assert provider.calls == 1
    assert recon.latest_preview(SEASON, WEEK, GAME_ID)["text"] == "A real-looking preview article."


def test_a_game_that_already_has_a_good_preview_is_skipped_not_rebilled(content_data_dir):
    generate_previews_for_week(WEEK, _CountingProvider(), PROMPT_TEMPLATE, now="2026-09-13T12:00:00+00:00")

    second = _CountingProvider()
    result = generate_previews_for_week(WEEK, second, PROMPT_TEMPLATE, now="2026-09-13T13:00:00+00:00")

    assert result["generated"] == [] and result["skipped_existing"] == [GAME_ID]
    assert second.calls == 0


def test_force_regenerates_even_when_a_good_preview_exists(content_data_dir):
    generate_previews_for_week(WEEK, _CountingProvider(), PROMPT_TEMPLATE, now="2026-09-13T12:00:00+00:00")

    forced = _CountingProvider()
    result = generate_previews_for_week(WEEK, forced, PROMPT_TEMPLATE, force=True, now="2026-09-13T13:00:00+00:00")

    assert result["generated"] == [GAME_ID] and forced.calls == 1


def test_a_failure_is_reported_with_its_reason_and_leaves_nothing_looking_published(content_data_dir):
    result = generate_previews_for_week(WEEK, _FailingProvider(), PROMPT_TEMPLATE, now="2026-09-13T12:00:00+00:00")

    assert result["generated"] == []
    assert result["failed"] == [{"game_id": GAME_ID, "failure_status": "EMPTY_RESPONSE"}]
    assert recon.latest_preview(SEASON, WEEK, GAME_ID)["available"] is False


def test_a_failed_rerun_is_retried_on_the_next_pass_instead_of_being_skipped(content_data_dir):
    generate_previews_for_week(WEEK, _FailingProvider(), PROMPT_TEMPLATE, now="2026-09-13T12:00:00+00:00")

    retry = _CountingProvider()
    result = generate_previews_for_week(WEEK, retry, PROMPT_TEMPLATE, now="2026-09-13T13:00:00+00:00")

    assert result["generated"] == [GAME_ID] and retry.calls == 1


def test_only_the_requested_game_is_generated(content_data_dir):
    provider = _CountingProvider()
    result = generate_previews_for_week(WEEK, provider, PROMPT_TEMPLATE, game_id="2026_01_SOME_OTHER", now="2026-09-13T12:00:00+00:00")
    assert result["generated"] == [] and provider.calls == 0
