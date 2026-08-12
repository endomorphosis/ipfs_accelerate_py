"""Final fail-closed revalidation immediately before every external effect."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json


class LaunchGuardError(RuntimeError):
    pass


class StaleLaunchPlanError(LaunchGuardError):
    pass


@dataclass(frozen=True)
class EffectBoundarySnapshot:
    run_id: str
    run_revision: int
    target_tree_cid: str
    scope_cid: str
    authority_cid: str
    policy_cid: str
    provider_id: str
    task_source_cid: str
    lease_id: str
    fencing_generation: int
    plan_cid: str
    effect_kind: str

    def __post_init__(self) -> None:
        missing = [name for name, value in asdict(self).items() if value in ("", None, 0)]
        if self.run_revision < 1 or self.fencing_generation < 1 or missing:
            raise LaunchGuardError("incomplete effect boundary snapshot: " + ", ".join(missing))

    @property
    def content_id(self) -> str:
        return cid_for_dag_json({"schema": "ipfs_accelerate_py/agent-supervisor/effect-boundary@1", **asdict(self)})


@dataclass(frozen=True)
class LaunchRevalidationReceipt:
    plan_cid: str
    snapshot_cid: str
    accepted: bool
    reason_codes: tuple[str, ...]


class LaunchPlanGuard:
    """Compares every mutable authority field, not just a PID or projection."""
    FIELDS = tuple(EffectBoundarySnapshot.__dataclass_fields__)

    def revalidate(self, planned: EffectBoundarySnapshot, current: EffectBoundarySnapshot) -> LaunchRevalidationReceipt:
        if not isinstance(planned, EffectBoundarySnapshot) or not isinstance(current, EffectBoundarySnapshot):
            raise LaunchGuardError("launch revalidation requires complete snapshots")
        stale = tuple("stale_" + field for field in self.FIELDS if getattr(planned, field) != getattr(current, field))
        receipt = LaunchRevalidationReceipt(planned.plan_cid, current.content_id, not stale, stale or ("revalidated",))
        if stale:
            raise StaleLaunchPlanError("launch plan changed before effect: " + ", ".join(stale))
        return receipt

    validate_before_effect = revalidate

    def execute(self, planned: EffectBoundarySnapshot, current: Callable[[], EffectBoundarySnapshot], effect: Callable[[], object]) -> object:
        self.revalidate(planned, current())
        return effect()


# ASE3-020 production alias
CompleteLaunchPlanGuard = LaunchPlanGuard


__all__ = ["CompleteLaunchPlanGuard", "EffectBoundarySnapshot", "LaunchGuardError", "LaunchPlanGuard", "LaunchRevalidationReceipt", "StaleLaunchPlanError"]
