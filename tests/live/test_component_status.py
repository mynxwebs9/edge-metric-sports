"""Phase 8A Step 28 proof #23: one provider's failure is isolated and does not corrupt the
rest of a run."""

from __future__ import annotations

from nfl_predict.live.component_status import ComponentState, RunHealthReport, run_component_safely


def test_a_failing_component_is_recorded_as_failed_not_raised():
    report = RunHealthReport()

    def boom():
        raise RuntimeError("weather API down")

    result, ok = run_component_safely(report, "weather", "2026-09-11T20:00:00+00:00", boom)
    assert result is None
    assert ok is False
    assert report.statuses[0].state == ComponentState.FAILED
    assert "weather API down" in report.statuses[0].detail


def test_a_failure_in_one_component_does_not_prevent_recording_others():
    report = RunHealthReport()
    run_component_safely(report, "weather", "t", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    result, ok = run_component_safely(report, "odds", "t", lambda: {"spread": -3.0})
    assert ok is True
    assert result == {"spread": -3.0}
    assert len(report.statuses) == 2
    assert report.overall_ok() is False  # weather failed
    assert report.statuses[1].state == ComponentState.SUCCESS


def test_all_success_means_overall_ok():
    report = RunHealthReport()
    run_component_safely(report, "schedule", "t", lambda: [1, 2, 3])
    run_component_safely(report, "odds", "t", lambda: {"a": 1})
    assert report.overall_ok() is True
