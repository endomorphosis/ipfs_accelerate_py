"""V3 durable authority and final-effect-boundary regression tests."""
from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.launch_guard import (
    EffectBoundarySnapshot,
    LaunchPlanGuard,
    StaleLaunchPlanError,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry_backend import (
    DuckDBRunRegistryBackend,
    DurableRunHead,
    RunRevisionCAS,
    RunRevisionConflictError,
)


def _head(revision: int = 1, handle: str = "handle-1") -> DurableRunHead:
    return DurableRunHead("run-1", revision, handle, "admitted", "unknown", updated_at_ms=revision)


def _snapshot() -> EffectBoundarySnapshot:
    return EffectBoundarySnapshot("run-1", 1, "tree", "scope", "authority", "policy", "provider", "tasks", "lease", 1, "plan", "start")


def test_revision_cas_and_effect_recovery_are_single_winner(tmp_path) -> None:
    backend = DuckDBRunRegistryBackend(tmp_path)
    base = backend.create(_head())
    winner = backend.compare_and_swap(RunRevisionCAS("run-1", 1, base.handle_cid), _head(2, "handle-2"))
    with pytest.raises(RunRevisionConflictError):
        backend.compare_and_swap(RunRevisionCAS("run-1", 1, base.handle_cid), _head(2, "conflict"))
    assert backend.reconstruct("run-1") == winner
    backend.record_intent(run_id="run-1", effect_key="start:1", intent_cid="intent")
    assert backend.continuation_for(run_id="run-1", effect_key="start:1") == "perform_effect"
    backend.record_effect(run_id="run-1", effect_key="start:1", effect_cid="birth")
    assert backend.continuation_for(run_id="run-1", effect_key="start:1") == "record_receipt"
    backend.record_receipt(run_id="run-1", effect_key="start:1", receipt_cid="receipt")
    assert backend.continuation_for(run_id="run-1", effect_key="start:1") == "already_complete"


def test_effect_guard_rejects_any_stale_plan_field_before_effect() -> None:
    plan = _snapshot()
    assert LaunchPlanGuard().revalidate(plan, plan).accepted
    with pytest.raises(StaleLaunchPlanError):
        LaunchPlanGuard().revalidate(plan, replace(plan, fencing_generation=2))


def test_process_adoption_requires_exact_birth_identity_and_health(tmp_path) -> None:
    backend = DuckDBRunRegistryBackend(tmp_path)
    running = DurableRunHead("run-2", 1, "handle", "running", "healthy", "process", "birth:42")
    backend.create(running)
    assert backend.adopt_healthy_matching_process(run_id="run-2", process_cid="process", process_birth_identity="birth:42", healthy=True) == running
    with pytest.raises(Exception):
        backend.adopt_healthy_matching_process(run_id="run-2", process_cid="process", process_birth_identity="birth:other", healthy=True)
