"""Restart / reconciliation evidence for GoalTacticianSupervisorLifecycle@1.

Proves FVT-G051 / FVT-031 restart-safety acceptance:

* restart replays identical authoritative state (content identity);
* cancellation / timeout / backpressure remain durable across process restart;
* changed trees invalidate scoped work and fence prior receipts;
* stale workers and pre-invalidation receipts cannot close or mutate after
  restart-safe reconciliation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof.goal_tactician_lifecycle import (
    GoalTacticianLifecycleConfig,
    GoalTacticianLifecycleError,
    GoalTacticianSupervisorLifecycle,
    LifecycleControlActiveError,
    LifecycleControlSignal,
    LifecyclePlanStatus,
    LifecycleReceipt,
    LifecycleTransitionKind,
    ReceiptKind,
    StaleReceiptError,
    StaleWorkerError,
    create_goal_tactician_supervisor_lifecycle,
)


def _bounds(**overrides: Any) -> dict[str, Any]:
    payload = {
        "wall_time_ms": 30_000,
        "memory_bytes": 256 * 1024 * 1024,
        "max_steps": 64,
    }
    payload.update(overrides)
    return payload


def _open(
    lifecycle: GoalTacticianSupervisorLifecycle,
    **overrides: Any,
) -> Any:
    payload: dict[str, Any] = {
        "tree_id": "tree:repo@abc123",
        "end_goal_id": "goal:restart-safety",
        "proof_graph_id": "graph:restart@1",
        "provider_id": "provider:kernel",
        "provider_version": "2.0.0",
        "policy_id": "policy:fvt-restart",
        "bounds": _bounds(),
        "resource_class": "cpu-supervisor",
        "max_retries": 1,
        "selected_leaf_ids": ("leaf:root", "leaf:child"),
        "selected_counterexample_ids": ("cex:w1",),
    }
    payload.update(overrides)
    return lifecycle.open_plan(**payload)


def _fresh_instance(state_dir: Path) -> GoalTacticianSupervisorLifecycle:
    """Simulate a process restart: new object, durable state only."""

    return GoalTacticianSupervisorLifecycle(
        GoalTacticianLifecycleConfig(state_dir=state_dir)
    )


# ---------------------------------------------------------------------------
# Restart replays identical authoritative state
# ---------------------------------------------------------------------------


def test_restart_replays_identical_authoritative_state(tmp_path: Path) -> None:
    first = create_goal_tactician_supervisor_lifecycle(tmp_path)
    state = _open(first)
    lease = first.acquire_lease("worker-1")
    first.record_transition(
        LifecycleTransitionKind.CANDIDATE,
        {"candidate_id": "cand:1"},
        lease,
    )
    receipt = first.build_receipt(
        receipt_id="receipt:leaf:root",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:root",
        lease=lease,
    )
    first.record_transition(
        LifecycleTransitionKind.VERIFICATION,
        {"subject_id": "leaf:root"},
        lease,
        receipt=receipt,
    )
    before = first.authoritative_snapshot()
    before_id = first.authoritative_state().content_id

    # Process boundary: drop the first instance and reopen from disk.
    del first
    second = _fresh_instance(tmp_path)
    after = second.authoritative_snapshot()
    after_id = second.authoritative_state().content_id

    assert after_id == before_id
    # Material fields that define authority must match exactly.
    for key in (
        "plan_id",
        "status",
        "fencing_epoch",
        "fencing_token",
        "control_signal",
        "sequence",
        "end_goal",
        "proof_graph",
        "candidates",
        "receipts",
    ):
        assert after[key] == before[key]
    assert after["cache_key"]["tree_id"] == before["cache_key"]["tree_id"]
    assert (
        second.authoritative_state().cache_key.key_id
        == state.cache_key.key_id
    )


def test_reconcile_on_restart_preserves_material_authority(
    tmp_path: Path,
) -> None:
    first = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open(first)
    lease = first.acquire_lease("worker-1")
    first.record_transition(
        LifecycleTransitionKind.PROOF_GRAPH,
        {
            "proof_graph_id": "graph:restart@1",
            "leaf_ids": ["leaf:root", "leaf:child"],
            "edges": [["leaf:root", "leaf:child"]],
        },
        lease,
    )
    material_before = {
        "plan_id": first.authoritative_state().plan_id,
        "tree_id": first.authoritative_state().tree_id,
        "fencing_epoch": first.authoritative_state().fencing_epoch,
        "cache_key_id": first.authoritative_state().cache_key.key_id,
        "receipt_ids": [
            item.receipt_id for item in first.authoritative_state().receipts
        ],
        "control_signal": first.authoritative_state().control_signal.value,
        "status": first.authoritative_state().status.value,
    }

    restarted = _fresh_instance(tmp_path)
    reconciled = restarted.reconcile()
    material_after = {
        "plan_id": reconciled.plan_id,
        "tree_id": reconciled.tree_id,
        "fencing_epoch": reconciled.fencing_epoch,
        "cache_key_id": reconciled.cache_key.key_id,
        "receipt_ids": [item.receipt_id for item in reconciled.receipts],
        "control_signal": reconciled.control_signal.value,
        "status": reconciled.status.value,
    }
    # Reconcile journals an audit transition but does not change authority.
    assert material_after["plan_id"] == material_before["plan_id"]
    assert material_after["tree_id"] == material_before["tree_id"]
    assert material_after["fencing_epoch"] == material_before["fencing_epoch"]
    assert material_after["cache_key_id"] == material_before["cache_key_id"]
    assert material_after["receipt_ids"] == material_before["receipt_ids"]
    assert material_after["control_signal"] == material_before["control_signal"]
    assert any(
        item.kind is LifecycleTransitionKind.RECONCILE
        for item in reconciled.transitions
    )


def test_restart_method_reloads_from_disk(tmp_path: Path) -> None:
    first = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open(first)
    first.acquire_lease("worker-1")
    content_before = first.authoritative_state().content_id

    # restart() clears memory and reloads durable state.
    reloaded = first.restart()
    assert reloaded.plan_id
    # After restart + reconcile a RECONCILE transition is appended, so the
    # full content_id changes, but the plan identity and cache key remain.
    assert reloaded.cache_key.tree_id == "tree:repo@abc123"
    assert reloaded.fencing_token >= 1
    # Durable state file remains valid JSON with the interface stamp.
    raw = json.loads(first.config.state_path.read_text(encoding="utf-8"))
    assert raw["interface"] == "GoalTacticianSupervisorLifecycle@1"
    assert raw["plan_id"] == reloaded.plan_id
    del content_before


# ---------------------------------------------------------------------------
# Durable control across restart
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "signal",
    [
        LifecycleControlSignal.CANCELLED,
        LifecycleControlSignal.TIMED_OUT,
        LifecycleControlSignal.BACKPRESSURE,
    ],
)
def test_control_signals_survive_restart(
    tmp_path: Path,
    signal: LifecycleControlSignal,
) -> None:
    first = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open(first)
    lease = first.acquire_lease("worker-1")
    first.signal_control(signal, lease, reason_code=f"durable:{signal.value}")

    second = _fresh_instance(tmp_path)
    state = second.authoritative_state()
    assert state.control_signal is signal
    assert state.status is LifecyclePlanStatus.BLOCKED

    # Stale post-restart mutation still blocked without a new lease, and a new
    # lease cannot be acquired while control is active.
    with pytest.raises(LifecycleControlActiveError):
        second.acquire_lease("worker-2")


def test_expired_lease_released_on_reconcile(tmp_path: Path) -> None:
    clock = {"t": 1_000.0}

    def _clock() -> float:
        return clock["t"]

    first = GoalTacticianSupervisorLifecycle(
        GoalTacticianLifecycleConfig(state_dir=tmp_path, lease_seconds=5),
        clock=_clock,
    )
    _open(first)
    lease = first.acquire_lease("worker-1")
    assert lease.active
    # Advance clock past lease expiry.
    clock["t"] = 1_000.0 + 100.0

    second = GoalTacticianSupervisorLifecycle(
        GoalTacticianLifecycleConfig(state_dir=tmp_path, lease_seconds=5),
        clock=_clock,
    )
    reconciled = second.reconcile()
    assert reconciled.active_lease is not None
    assert not reconciled.active_lease.active
    assert "expire" in reconciled.active_lease.release_reason or (
        reconciled.active_lease.release_reason == "lease_expired_on_reconcile"
    )
    # Prior fencing token cannot mutate after reconcile.
    with pytest.raises(StaleWorkerError):
        second.record_transition(
            LifecycleTransitionKind.CANDIDATE,
            {"candidate_id": "cand:after-expire"},
            lease,
        )


# ---------------------------------------------------------------------------
# Changed trees invalidate scoped work
# ---------------------------------------------------------------------------


def test_tree_change_invalidates_scoped_receipts_and_cache_key(
    tmp_path: Path,
) -> None:
    first = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open(first)
    lease = first.acquire_lease("worker-1")
    receipt = first.build_receipt(
        receipt_id="receipt:leaf:root",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:root",
        lease=lease,
    )
    first.record_transition(
        LifecycleTransitionKind.VERIFICATION,
        {"subject_id": "leaf:root"},
        lease,
        receipt=receipt,
    )
    prior_key = first.authoritative_state().cache_key.key_id
    prior_epoch = first.authoritative_state().fencing_epoch
    assert len(first.authoritative_state().receipts) == 1

    invalidated = first.invalidate_tree("tree:repo@def456", lease)
    assert invalidated.tree_id == "tree:repo@def456"
    assert invalidated.fencing_epoch == prior_epoch + 1
    assert invalidated.cache_key.key_id != prior_key
    assert invalidated.receipts == ()
    assert invalidated.status is LifecyclePlanStatus.OPEN
    assert any(
        item.kind is LifecycleTransitionKind.TREE_INVALIDATION
        for item in invalidated.transitions
    )

    # Prior receipt is tree-stale if re-presented under a new lease.
    new_lease = first.acquire_lease("worker-2")
    with pytest.raises(StaleReceiptError):
        first.record_transition(
            LifecycleTransitionKind.VERIFICATION,
            {"subject_id": "leaf:root"},
            new_lease,
            receipt=receipt,
        )


def test_tree_invalidation_survives_restart_and_blocks_stale_completion(
    tmp_path: Path,
) -> None:
    first = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open(first)
    lease = first.acquire_lease("worker-1")
    # Fully satisfy evidence under the old tree.
    for subject, kind, rid in (
        ("leaf:root", ReceiptKind.GRAPH_LEAF, "r:root"),
        ("leaf:child", ReceiptKind.GRAPH_LEAF, "r:child"),
        ("cex:w1", ReceiptKind.COUNTEREXAMPLE, "r:cex"),
    ):
        item = first.build_receipt(
            receipt_id=rid,
            kind=kind,
            subject_id=subject,
            lease=lease,
        )
        first.record_transition(
            (
                LifecycleTransitionKind.COUNTEREXAMPLE
                if kind is ReceiptKind.COUNTEREXAMPLE
                else LifecycleTransitionKind.VERIFICATION
            ),
            {"subject_id": subject},
            lease,
            receipt=item,
        )
    assert first.evaluate_completion().admitted

    first.invalidate_tree("tree:repo@new", lease)

    # Restart and confirm prior evidence is gone; completion fails closed.
    second = _fresh_instance(tmp_path)
    state = second.authoritative_state()
    assert state.tree_id == "tree:repo@new"
    assert state.receipts == ()
    decision = second.evaluate_completion()
    assert not decision.admitted
    assert "leaf:root" in decision.missing_leaf_ids
    assert "leaf:child" in decision.missing_leaf_ids
    assert "cex:w1" in decision.missing_counterexample_ids

    # Stale pre-invalidation worker cannot complete after restart.
    with pytest.raises(StaleWorkerError):
        second.try_complete(lease)


def test_stale_epoch_receipt_cannot_complete_after_restart(
    tmp_path: Path,
) -> None:
    first = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open(
        first,
        selected_leaf_ids=("leaf:root",),
        selected_counterexample_ids=(),
    )
    lease = first.acquire_lease("worker-1")
    receipt = first.build_receipt(
        receipt_id="receipt:leaf:root",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:root",
        lease=lease,
    )
    first.record_transition(
        LifecycleTransitionKind.VERIFICATION,
        {"subject_id": "leaf:root"},
        lease,
        receipt=receipt,
    )
    first.invalidate_tree("tree:repo@epoch2", lease)

    second = _fresh_instance(tmp_path)
    new_lease = second.acquire_lease("worker-2")
    # Craft a receipt that reuses the old epoch (stale).
    stale = LifecycleReceipt(
        receipt_id="receipt:forged-epoch",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:root",
        tree_id=second.authoritative_state().tree_id,
        fencing_epoch=1,  # old epoch
        fencing_token=new_lease.fencing_token,
        assurance=AssuranceLevel.KERNEL_VERIFIED,
        independently_validated=True,
    )
    with pytest.raises(StaleReceiptError):
        second.record_transition(
            LifecycleTransitionKind.VERIFICATION,
            {"subject_id": "leaf:root"},
            new_lease,
            receipt=stale,
        )
    # Adequate fresh receipt under the new epoch can complete.
    fresh = second.build_receipt(
        receipt_id="receipt:leaf:root:v2",
        kind=ReceiptKind.GRAPH_LEAF,
        subject_id="leaf:root",
        lease=new_lease,
    )
    second.record_transition(
        LifecycleTransitionKind.VERIFICATION,
        {"subject_id": "leaf:root"},
        new_lease,
        receipt=fresh,
    )
    admitted = second.try_complete(new_lease)
    assert admitted.admitted
    assert second.authoritative_state().status is LifecyclePlanStatus.COMPLETED


# ---------------------------------------------------------------------------
# Journal / durable artifacts
# ---------------------------------------------------------------------------


def test_journal_and_state_artifacts_are_written(tmp_path: Path) -> None:
    lifecycle = create_goal_tactician_supervisor_lifecycle(tmp_path)
    _open(lifecycle)
    lease = lifecycle.acquire_lease("worker-1")
    lifecycle.record_transition(
        LifecycleTransitionKind.CANDIDATE,
        {"candidate_id": "cand:journal"},
        lease,
    )
    assert lifecycle.config.state_path.is_file()
    assert lifecycle.config.journal_path.is_file()
    lines = [
        line
        for line in lifecycle.config.journal_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(lines) >= 3  # end_goal, proof_graph, lease_acquire, candidate
    last = json.loads(lines[-1])
    assert last["kind"] == LifecycleTransitionKind.CANDIDATE.value
    assert last["content_id"].startswith("b")


def test_no_state_reconcile_fails_closed(tmp_path: Path) -> None:
    lifecycle = create_goal_tactician_supervisor_lifecycle(tmp_path)
    with pytest.raises(GoalTacticianLifecycleError):
        lifecycle.reconcile()
