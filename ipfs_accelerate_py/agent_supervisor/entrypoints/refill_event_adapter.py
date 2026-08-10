"""Production event adapters for residual evaluation (ASE3-021).

Adapters normalize scheduler, validation, review, merge, Doctor, retry,
drift, low-water, and open-goal signals into a single observation surface.
They never authorize append or completion themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

from .refill_controller import RefillObservation

PRODUCTION_EVENT_ADAPTER_MANIFEST: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-refill-event-adapter@1"
)

_EVENT_KINDS: Final[tuple[str, ...]] = (
    "scheduler_low_water",
    "scheduler_drained_open_goal",
    "validation_rejected",
    "review_rejected",
    "merge_rejected",
    "doctor_finding",
    "retry_exhausted",
    "actionable_drift",
    "rollout_threshold_missed",
    "stale_evidence",
    "branch_only_completion",
)


@dataclass(frozen=True)
class ProductionRefillEventAdapter:
    """Compose production events into a :class:`RefillObservation`."""

    schema: str = PRODUCTION_EVENT_ADAPTER_MANIFEST

    def supported_event_kinds(self) -> tuple[str, ...]:
        return _EVENT_KINDS

    def to_observation(
        self,
        *,
        plan_root_cid: str,
        revision: int,
        events: Sequence[Mapping[str, Any]] = (),
        ready_tasks: int = 0,
        active_tasks: int = 0,
        open_goals: int = 0,
    ) -> RefillObservation:
        kinds = {str(item.get("kind") or "") for item in events}
        unknown = sorted(k for k in kinds if k and k not in _EVENT_KINDS)
        if unknown:
            raise ValueError(f"unsupported production refill events: {unknown}")
        return RefillObservation(
            plan_root_cid=plan_root_cid,
            revision=revision,
            ready_tasks=ready_tasks,
            active_tasks=active_tasks,
            open_goals=open_goals,
            validation_rejected="validation_rejected" in kinds,
            review_rejected="review_rejected" in kinds,
            merge_rejected="merge_rejected" in kinds,
            stale_evidence="stale_evidence" in kinds,
            branch_only_completion="branch_only_completion" in kinds,
            actionable_drift="actionable_drift" in kinds or "doctor_finding" in kinds,
            retry_exhausted_with_refinement="retry_exhausted" in kinds,
            rollout_threshold_missed="rollout_threshold_missed" in kinds,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "event_kinds": list(_EVENT_KINDS),
            "authorizes_append": False,
            "authorizes_completion": False,
        }


__all__ = [
    "PRODUCTION_EVENT_ADAPTER_MANIFEST",
    "ProductionRefillEventAdapter",
]
