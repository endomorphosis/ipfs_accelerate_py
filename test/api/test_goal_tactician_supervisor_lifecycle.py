"""Lifecycle evidence for GoalTacticianSupervisorLifecycle@1 (FVT-G051 / FVT-031).

Proves the acceptance subset for fenced, durable tactician supervisor execution:

* transitions for end-goal, proof graph, candidate, verification,
  counterexample, closure, and completion are content-addressed;
* stale workers / receipts cannot close or mutate a plan;
* cancellation / timeout / backpressure are durable control signals;
* completion requires all selected graph leaves and counterexamples to have
  adequate fresh receipts bound to the current tree and fencing epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof.goal_tactician_lifecycle import (
    GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE,
    GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_SCHEMA,
    LIFECYCLE_CACHE_KEY_SCHEMA,
    LIFECYCLE_STATE_SCHEMA,
    ExactLifecycleCacheKey,
    GoalTacticianLifecycleConfig,
    GoalTacticianLifecycleError,
    GoalTacticianSupervisorLifecycle,
    LifecycleControlActiveError,
    LifecycleControlSignal,
    LifecyclePlanStatus,
    LifecycleTransitionKind,
    ReceiptKind,
    ResourcePolicy,
    StaleReceiptError,
    StaleWorkerError,
    WorkerLease,
    claims_authority,
    create_goal_tactician_supervisor_lifecycle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bounds(**overrides: Any) -> dict[str, Any]:
    payload = {
        "wall_time_ms": 30_000,
        "memory_bytes": 256 * 1024 * 1024,
        "max_steps": 64,
    }
    payload.update(overrides)
    return payload


def _open_plan(
    lifecycle: GoalTacticianSupervisorLifecycle,
    **overrides: Any,
) -> Any:
    payload: dict[str, Any] = {
        "tree_id": "tree:repo@abc123",
        "end_goal_id": "goal:lease-safety",
        "proof_graph_id": "graph:lease-safety@1",
        "provider_id": "provider:leanstral",
        "provider_version": "1.2.3",
        "policy_id": "policy:fvt-tactician",
        "bounds": _bounds(),
        "resource_class": "cpu-supervisor",
        "max_retries": 2,
        "selected_leaf_ids": ("leaf:a", "leaf:b"),
        "selected_counterexample_ids": ("cex:1",),
        "toolchain_id": "toolchain:locked@1",
        "end_goal": {
            "end_goal_id": "goal:lease-safety",
            "statement": "leases are fenced",
        },
        "proof_graph": {
            "proof_graph_id": "graph:lease-safety@1",
            "leaf_ids": ["leaf:a", "leaf:b"],
        },
    }
    payload.update(overrides)
    return lifecycle.open_plan(**payload)


@pytest.fixture
def lifecycle(tmp_path: Path) -> GoalTacticianSupervisorLifecycle:
    return create_goal_tactician_supervisor_lifecycle(tmp_path)


# ---------------------------------------------------------------------------
# Interface / cache key / authority hygiene
# ---------------------------------------------------------------------------


def test_interface_identity_and_factory(tmp_path: Path) -> None:
    assert (
        GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE
        == "GoalTacticianSupervisorLifecycle@1"
    )
    assert (
        GoalTacticianSupervisorLifecycle.interface
        == GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE
    )
    assert (
        GoalTacticianSupervisorLifecycle.schema
        == GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_SCHEMA
    )
    instance = create_goal_tactician_supervisor_lifecycle(tmp_path)
    assert isinstance(instance, GoalTacticianSupervisorLifecycle)
    assert instance.config.state_path.parent == tmp_path


def test_exact_cache_key_includes_required_components() -> None:
    key = ExactLifecycleCacheKey(
        tree_id="tree:1",
        end_goal_id="goal:1",
        proof_graph_id="graph:1",
        provider_id="provider:x",
        provider_version="1.0.0",
        policy_id="policy:y",
        bounds=_bounds(),
        resource_class="cpu-supervisor",
        max_retries=3,
        selected_leaf_ids=("leaf:a",),
        selected_counterexample_ids=("cex:1",),
    )
    payload = key.to_dict()
    assert payload["schema"] == LIFECYCLE_CACHE_KEY_SCHEMA
    for field_name in (
        "tree_id",
        "end_goal_id",
        "proof_graph_id",
        "provider_id",
        "provider_version",
        "policy_id",
        "bounds",
        "resource_class",
        "max_retries",
    ):
        assert field_name in payload
    assert key.key_id.startswith("b")
    # Changing tree identity changes the exact key.
    other = ExactLifecycleCacheKey.from_dict(
        {**payload, "tree_id": "tree:2"}
    )
    assert other.key_id != key.key_id


def test_claims_authority_detects_completion_bypass() -> None:
    assert claims_authority({"complete": True})
    assert claims_authority({"nested": {"verified": True}})
    assert not claims_authority({"status": "ok", "note": "candidate"})


def test_resource_policy_bounds() -> None:
    policy = ResourcePolicy(
        resource_class="cpu-supervisor",
        max_concurrent_workers=1,
        wall_time_ms=10_000,
        memory_bytes=64 * 1024 * 1024,
        max_retries=0,
    )
    assert policy.to_dict()["max_retries"] == 0
    with pytest.raises(GoalTacticianLifecycleError):
        ResourcePolicy(resource_class="", max_concurrent_workers=1)


# ---------------------------------------------------------------------------
# Plan open and transition sequence
# ---------------------------------------------------------------------------


def test_open_plan_records_end_goal_and_proof_graph(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    state = _open_plan(lifecycle)
    assert state.status is LifecyclePlanStatus.OPEN
    assert state.to_dict()["schema"] == LIFECYCLE_STATE_SCHEMA
    kinds = [item.kind for item in state.transitions]
    assert LifecycleTransitionKind.END_GOAL in kinds
    assert LifecycleTransitionKind.PROOF_GRAPH in kinds
    assert state.end_goal["end_goal_id"] == "goal:lease-safety"
    assert state.proof_graph["proof_graph_id"] == "graph:lease-safety@1"
    assert state.cache_key.selected_leaf_ids == ("leaf:a", "leaf:b")
    assert state.cache_key.selected_counterexample_ids == ("cex:1",)
    # Content identity is stable for the same material.
    assert state.content_id == state.content_id
    reloaded = GoalTacticianSupervisorLifecycle(
        GoalTacticianLifecycleConfig(state_dir=lifecycle.config.state_dir)
    )
    assert reloaded.authoritative_state().content_id == state.content_id


def test_fenced_transition_sequence_under_lease(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")
    assert lease.active
    assert lease.fencing_token == 1
    assert lease.fencing_epoch == 1

    lifecycle.record_transition(
        LifecycleTransitionKind.CANDIDATE,
        {"candidate_id": "cand:1", "kind": "lemma"},
        lease,
    )
    leaf_a = lifecycle.build_receipt(
        receipt_id="receipt:leaf:a",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:a",
        lease=lease,
    )
    lifecycle.record_transition(
        LifecycleTransitionKind.VERIFICATION,
        {"subject_id": "leaf:a", "verdict": "proved"},
        lease,
        receipt=leaf_a,
    )
    leaf_b = lifecycle.build_receipt(
        receipt_id="receipt:leaf:b",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:b",
        lease=lease,
    )
    lifecycle.record_transition(
        LifecycleTransitionKind.VERIFICATION,
        {"subject_id": "leaf:b", "verdict": "proved"},
        lease,
        receipt=leaf_b,
    )
    cex = lifecycle.build_receipt(
        receipt_id="receipt:cex:1",
        kind=ReceiptKind.COUNTEREXAMPLE,
        subject_id="cex:1",
        lease=lease,
        assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    lifecycle.record_transition(
        LifecycleTransitionKind.COUNTEREXAMPLE,
        {"counterexample_id": "cex:1", "closed": True},
        lease,
        receipt=cex,
    )
    lifecycle.record_transition(
        LifecycleTransitionKind.CLOSURE,
        {"closed": True, "witness_ids": ["cex:1"]},
        lease,
    )

    state = lifecycle.authoritative_state()
    kinds = {item.kind for item in state.transitions}
    assert LifecycleTransitionKind.CANDIDATE in kinds
    assert LifecycleTransitionKind.VERIFICATION in kinds
    assert LifecycleTransitionKind.COUNTEREXAMPLE in kinds
    assert LifecycleTransitionKind.CLOSURE in kinds
    assert len(state.receipts) == 3
    assert all(item.content_id for item in state.transitions)


# ---------------------------------------------------------------------------
# Stale workers cannot mutate
# ---------------------------------------------------------------------------


def test_stale_worker_cannot_mutate_after_successor_lease(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(lifecycle)
    first = lifecycle.acquire_lease("worker-stale")
    second = lifecycle.acquire_lease("worker-fresh")
    assert second.fencing_token > first.fencing_token

    with pytest.raises(StaleWorkerError):
        lifecycle.record_transition(
            LifecycleTransitionKind.CANDIDATE,
            {"candidate_id": "cand:stale"},
            first,
        )

    # Fresh worker can still mutate.
    lifecycle.record_transition(
        LifecycleTransitionKind.CANDIDATE,
        {"candidate_id": "cand:fresh"},
        second,
    )
    state = lifecycle.authoritative_state()
    assert any(
        item.get("candidate_id") == "cand:fresh" for item in state.candidates
    )


def test_stale_worker_cannot_complete(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(
        lifecycle,
        selected_leaf_ids=(),
        selected_counterexample_ids=(),
    )
    first = lifecycle.acquire_lease("worker-a")
    lifecycle.acquire_lease("worker-b")
    with pytest.raises(StaleWorkerError):
        lifecycle.try_complete(first)


def test_expired_lease_is_rejected(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1", lease_seconds=1)
    # Manually craft an expired projection of the same fencing token.
    expired = WorkerLease(
        worker_id=lease.worker_id,
        plan_id=lease.plan_id,
        fencing_token=lease.fencing_token,
        fencing_epoch=lease.fencing_epoch,
        acquired_at_ms=lease.acquired_at_ms,
        expires_at_ms=lease.acquired_at_ms - 1,
        resource_class=lease.resource_class,
        active=True,
    )
    with pytest.raises(StaleWorkerError):
        lifecycle.record_transition(
            LifecycleTransitionKind.CANDIDATE,
            {"candidate_id": "cand:expired"},
            expired,
        )


# ---------------------------------------------------------------------------
# Stale / inadequate receipts cannot close a plan
# ---------------------------------------------------------------------------


def test_stale_tree_receipt_rejected(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")
    from ipfs_accelerate_py.agent_supervisor.proof.goal_tactician_lifecycle import (
        LifecycleReceipt,
    )

    stale = LifecycleReceipt(
        receipt_id="receipt:stale-tree",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:a",
        tree_id="tree:other",
        fencing_epoch=lease.fencing_epoch,
        fencing_token=lease.fencing_token,
        assurance=AssuranceLevel.KERNEL_VERIFIED,
        independently_validated=True,
    )
    with pytest.raises(StaleReceiptError):
        lifecycle.record_transition(
            LifecycleTransitionKind.VERIFICATION,
            {"subject_id": "leaf:a"},
            lease,
            receipt=stale,
        )


def test_inadequate_receipt_rejected_for_assurance(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")
    weak = lifecycle.build_receipt(
        receipt_id="receipt:weak",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:a",
        lease=lease,
        assurance=AssuranceLevel.CANDIDATE,
        independently_validated=True,
    )
    with pytest.raises(StaleReceiptError):
        lifecycle.record_transition(
            LifecycleTransitionKind.VERIFICATION,
            {"subject_id": "leaf:a"},
            lease,
            receipt=weak,
        )


def test_unvalidated_receipt_rejected(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")
    draft = lifecycle.build_receipt(
        receipt_id="receipt:draft",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:a",
        lease=lease,
        independently_validated=False,
    )
    with pytest.raises(StaleReceiptError):
        lifecycle.record_transition(
            LifecycleTransitionKind.VERIFICATION,
            {"subject_id": "leaf:a"},
            lease,
            receipt=draft,
        )


# ---------------------------------------------------------------------------
# Durable control signals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "signal",
    [
        LifecycleControlSignal.CANCELLED,
        LifecycleControlSignal.TIMED_OUT,
        LifecycleControlSignal.BACKPRESSURE,
    ],
)
def test_durable_control_signals_block_mutation(
    lifecycle: GoalTacticianSupervisorLifecycle,
    signal: LifecycleControlSignal,
) -> None:
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")
    state = lifecycle.signal_control(signal, lease, reason_code=signal.value)
    assert state.control_signal is signal
    assert state.status is LifecyclePlanStatus.BLOCKED

    with pytest.raises(LifecycleControlActiveError):
        lifecycle.record_transition(
            LifecycleTransitionKind.CANDIDATE,
            {"candidate_id": "cand:blocked"},
            lease,
        )

    # Signal remains after reloading durable state.
    reloaded = GoalTacticianSupervisorLifecycle(
        GoalTacticianLifecycleConfig(state_dir=lifecycle.config.state_dir)
    )
    assert reloaded.authoritative_state().control_signal is signal
    assert reloaded.authoritative_state().status is LifecyclePlanStatus.BLOCKED


def test_completion_blocked_while_control_active(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(
        lifecycle,
        selected_leaf_ids=(),
        selected_counterexample_ids=(),
    )
    lease = lifecycle.acquire_lease("worker-1")
    lifecycle.signal_control(LifecycleControlSignal.CANCELLED, lease)
    decision = lifecycle.try_complete(lease)
    assert not decision.admitted
    assert "control_cancelled" in decision.reason_codes


# ---------------------------------------------------------------------------
# Completion requires all selected leaves + counterexamples
# ---------------------------------------------------------------------------


def test_completion_requires_all_selected_evidence(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")

    # Only one leaf — incomplete.
    leaf_a = lifecycle.build_receipt(
        receipt_id="receipt:leaf:a",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:a",
        lease=lease,
    )
    lifecycle.record_transition(
        LifecycleTransitionKind.VERIFICATION,
        {"subject_id": "leaf:a"},
        lease,
        receipt=leaf_a,
    )
    decision = lifecycle.try_complete(lease)
    assert not decision.admitted
    assert "leaf:b" in decision.missing_leaf_ids
    assert "cex:1" in decision.missing_counterexample_ids

    leaf_b = lifecycle.build_receipt(
        receipt_id="receipt:leaf:b",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:b",
        lease=lease,
    )
    lifecycle.record_transition(
        LifecycleTransitionKind.VERIFICATION,
        {"subject_id": "leaf:b"},
        lease,
        receipt=leaf_b,
    )
    cex = lifecycle.build_receipt(
        receipt_id="receipt:cex:1",
        kind=ReceiptKind.COUNTEREXAMPLE,
        subject_id="cex:1",
        lease=lease,
    )
    lifecycle.record_transition(
        LifecycleTransitionKind.COUNTEREXAMPLE,
        {"counterexample_id": "cex:1"},
        lease,
        receipt=cex,
    )
    admitted = lifecycle.try_complete(lease)
    assert admitted.admitted
    assert lifecycle.authoritative_state().status is LifecyclePlanStatus.COMPLETED


def test_completed_plan_rejects_further_mutation(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(
        lifecycle,
        selected_leaf_ids=(),
        selected_counterexample_ids=(),
    )
    lease = lifecycle.acquire_lease("worker-1")
    decision = lifecycle.try_complete(lease)
    assert decision.admitted
    # New lease cannot reopen mutation on a completed plan.
    with pytest.raises(GoalTacticianLifecycleError):
        lifecycle.acquire_lease("worker-2")


def test_authority_claims_without_receipt_rejected(
    lifecycle: GoalTacticianSupervisorLifecycle,
) -> None:
    _open_plan(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")
    with pytest.raises(GoalTacticianLifecycleError):
        lifecycle.record_transition(
            LifecycleTransitionKind.CANDIDATE,
            {"candidate_id": "cand:1", "complete": True},
            lease,
        )
