"""Opt-in native repository context through the existing create-plan service.

This structural pilot has no behavioral facts, model calls, signed evidence
admission or worker authority. Native source and complete policy roots are
observed at service boundaries; static roots cannot replace those observations.
Preparation/capture is explicit and never runs during a preview or cache read.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Any

from ..prompt.plan_create_service import (
    PlanCreateMaterials, PlanCreateMode, PlanCreateService,
    freeze_plan_create_input_snapshot,
)
from .adaptive_planner import FrozenPlanningGoal
from .obligation_graph_compiler import TypedIntent
from .plan_revision_contracts import PlanAuthorityRoots, PlanCreateRequest
from .structural_codebase_context import structural_codebase_context

SCHEMA = "supervisor-repository-plan-preview@1"
PROFILE = "repository-live-root-observation@1"
_RESERVED_CONTEXT = frozenset({"request_cid", "input_snapshot_cid", "mode",
    "bounds_digest", "scope_paths", "structural_codebase_context_cid", "structural_codebase"})
_RESERVED_EXTRA = frozenset({"structural_codebase_context_cid", "structural_codebase"})


class RepositoryPlanPreviewError(ValueError):
    """A request cannot use this structural-only live-owner profile."""


@dataclass(frozen=True, slots=True)
class RepositoryPlanPreviewOwner:
    """Ephemeral native owner and selected head; never serialized as materials."""

    index: Any
    repository: Path
    expected_head: Any
    scheduler: Any = None
    parent_lease: Any = None
    cancel_event: Any = None
    timeout_seconds: float = 30.0
    memory_mb: int = 64

    def __post_init__(self):
        from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseHead
        from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
        if (type(self.index) is not RepositoryCodebaseIndex
                or type(self.index.catalog) is not CodebaseCatalog
                or self.index.catalog.store is not self.index.ingestor.store
                or self.index.catalog.artifacts is not self.index.artifacts
                or type(self.expected_head) is not CodebaseHead):
            raise RepositoryPlanPreviewError("exact native codebase owner and selected head required")
        repository = Path(self.repository)
        if (not repository.is_absolute() or repository.resolve(strict=True) != repository
                or repository.is_symlink() or not repository.is_dir()):
            raise RepositoryPlanPreviewError("canonical existing repository directory required")
        if (type(self.timeout_seconds) not in {int, float}
                or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 90
                or type(self.memory_mb) is not int or not 32 <= self.memory_mb <= 4096):
            raise RepositoryPlanPreviewError("bounded exact preview time and memory required")
        object.__setattr__(self, "repository", repository)


def preview_repository_plan(*, owner: RepositoryPlanPreviewOwner,
        request: PlanCreateRequest, materials: PlanCreateMaterials,
        policy_observer: Callable[[PlanCreateRequest], PlanAuthorityRoots],
        service_factory: Callable[..., PlanCreateService] = PlanCreateService) -> dict[str, Any]:
    """Consume fresh structural metadata in a private model-off service preview.

    ``policy_observer`` and ``service_factory`` are trusted application wiring,
    not serialized request fields. The native owner always observes source.
    Policy, source/head, complete materials and deadline must remain unchanged.
    The returned preview is proposal-tier even if a generic stage says admitted.
    """
    if (type(owner) is not RepositoryPlanPreviewOwner or type(request) is not PlanCreateRequest
            or type(materials) is not PlanCreateMaterials or not callable(policy_observer)
            or not callable(service_factory)):
        raise RepositoryPlanPreviewError("typed owner, request, materials and live policy wiring required")
    if (request.repository_root != str(owner.repository)
            or request.repository_id != owner.expected_head.repository_id
            or request.roots.repository_root_cid != owner.expected_head.snapshot_cid
            or request.roots.dirty_worktree_root != owner.expected_head.snapshot_cid):
        raise RepositoryPlanPreviewError("request repository roots differ from the selected native head")
    if (materials.current_roots is not None or materials.current_facts
            or materials.obligation_graph is not None or materials.evidence_bundle is not None
            or materials.admission_materials is not None or materials.model_provider is not None
            or materials.evidence_adapters or materials.evidence_queries or materials.scan is not None
            or materials.parallel_request is not None or materials.parallel_tasks is not None):
        raise RepositoryPlanPreviewError("structural preview cannot accept facts, static roots or injected stages")
    if (type(materials.intent) is not TypedIntent or type(materials.frozen_goal) is not FrozenPlanningGoal
            or materials.intent.current_root_id != owner.expected_head.snapshot_cid
            or materials.frozen_goal.repository_tree_id != owner.expected_head.snapshot_cid
            or materials.frozen_goal.policy.require_proof
            or set(materials.candidate_context) & _RESERVED_CONTEXT
            or set(materials.extra) & _RESERVED_EXTRA):
        raise RepositoryPlanPreviewError("explicit source-rooted model-off proposal materials required")
    if (materials.extra.get("root_observation_profile", PROFILE) != PROFILE
            or not materials.task_candidates or len(materials.task_candidates) > 16
            or len(materials.intent.desired_predicates) > 16):
        raise RepositoryPlanPreviewError("closed structural profile and bounded complete task population required")
    deadline = time.monotonic() + min(owner.timeout_seconds, request.budget.max_latency_ms / 1000)

    def remaining():
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
            LeaseCancelledError, LeaseTimeoutError,
        )
        if owner.cancel_event is not None and owner.cancel_event.is_set():
            raise LeaseCancelledError("repository preview cancelled")
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise LeaseTimeoutError("repository preview deadline exceeded")
        return duration

    def context_controls():
        return {"repository_id": owner.expected_head.repository_id,
            "expected_head": owner.expected_head, "scheduler": owner.scheduler,
            "parent_lease": owner.parent_lease, "cancel_event": owner.cancel_event,
            "timeout_seconds": remaining(), "memory_mb": owner.memory_mb}

    with structural_codebase_context(owner.index, owner.repository, **context_controls()) as context:
        if request.roots.program_root != context.semantic_state_cid:
            raise RepositoryPlanPreviewError("program root differs from native structural semantic state")
        record = context.to_dict()

        def observe_roots(typed_request):
            if typed_request != request:
                raise RepositoryPlanPreviewError("service changed the exact repository request")
            with structural_codebase_context(owner.index, owner.repository, **context_controls()) as current:
                if current != context:
                    raise RepositoryPlanPreviewError("native structural context changed during preview")
                roots = policy_observer(request)
                if type(roots) is not PlanAuthorityRoots:
                    raise RepositoryPlanPreviewError("complete independently observed policy roots required")
                roots.require_current(request.roots)
                remaining()
            return roots

        extra = {**dict(materials.extra), "structural_codebase_context_cid": context.cid,
            "structural_codebase": record, "root_observation_profile": PROFILE}
        bound = replace(materials, scan={"scan_cid": context.cid, "structural_codebase": record},
            candidate_context={**dict(materials.candidate_context),
                "structural_codebase_context_cid": context.cid, "structural_codebase": record}, extra=extra)
        snapshot = freeze_plan_create_input_snapshot(request, materials=bound)
        service = service_factory(root_observer=observe_roots, require_live_root_observation=True)
        if (not isinstance(service, PlanCreateService) or service.root_observer is not observe_roots
                or service.require_live_root_observation is not True or service.receipt_store is not None):
            raise RepositoryPlanPreviewError("private strict-root create-plan service required")
        preview = service.preview_create(request, mode=PlanCreateMode.DETERMINISTIC, materials=bound)
        if preview.input_snapshot_cid != snapshot.snapshot_cid:
            raise RepositoryPlanPreviewError("service consumed a different frozen repository input")
        remaining()
    remaining()
    return {"schema": SCHEMA, "preview": preview.to_dict(), "input_snapshot": snapshot.to_dict(),
        "structural_context": record, "structural_context_cid": context.cid,
        "observed_facts_supplied": 0, "model_calls": 0,
        "source_semantics_verified": False, "proof_authority": False,
        "production_admitted": False, "worker_launched": False,
        "execution_authority": False, "completion_authority": False}


__all__ = ["SCHEMA", "PROFILE", "RepositoryPlanPreviewError", "RepositoryPlanPreviewOwner",
    "preview_repository_plan"]
