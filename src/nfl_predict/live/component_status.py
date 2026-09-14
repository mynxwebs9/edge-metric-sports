"""Phase 8A Step 25/26: per-component status reporting, so one provider's failure never
corrupts an entire pipeline run - each component's outcome is recorded independently, and
the decision engine (already deterministic, Phase 7) is what decides whether a missing
component blocks qualification, not this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ComponentState(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ComponentStatus:
    component: str
    state: ComponentState
    detail: str
    checked_at: str


@dataclass
class RunHealthReport:
    """Accumulates one `ComponentStatus` per component for a single pipeline run - never
    raises on a component failure itself; callers decide whether to continue."""

    statuses: list[ComponentStatus] = field(default_factory=list)

    def record(self, component: str, state: ComponentState, detail: str, checked_at: str) -> None:
        self.statuses.append(ComponentStatus(component=component, state=state, detail=detail, checked_at=checked_at))

    def overall_ok(self) -> bool:
        return all(s.state != ComponentState.FAILED for s in self.statuses)

    def as_dict(self) -> dict:
        return {s.component: {"state": s.state.value, "detail": s.detail, "checked_at": s.checked_at} for s in self.statuses}


def run_component_safely(report: RunHealthReport, component: str, checked_at: str, func, *args, **kwargs):
    """Runs `func(*args, **kwargs)`, records SUCCESS/FAILED on `report`, and returns
    `(result, ok)` - `result` is None on failure. A raised exception from `func` is caught
    HERE ONLY (never silently swallowed elsewhere) so one component's failure cannot take
    down the rest of the run."""
    try:
        result = func(*args, **kwargs)
        report.record(component, ComponentState.SUCCESS, "ok", checked_at)
        return result, True
    except Exception as e:  # noqa: BLE001 - intentional: isolate ANY component failure
        report.record(component, ComponentState.FAILED, f"{type(e).__name__}: {e}", checked_at)
        return None, False
