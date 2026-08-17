"""ASE3-021 durable production refill saga hardening tests."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_adapters import (
    CurrentTreeResidualEvaluator,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_controller import (
    CompletionAuthorityDecision,
    ProductionRefillRuntime,
    RefillObservation,
    ResidualGap,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_event_adapter import (
    ProductionRefillEventAdapter,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_store import (
    SIGNED_REFILL_POLICY_SCHEMA,
    RefillSagaPhase,
    RefillStore,
    RefillStoreError,
    SignedRefillPolicy,
)


def _gap(scope: str = "scope-a") -> ResidualGap:
    return ResidualGap(
        "goal",
        "evidence",
        scope,
        ("goal", "root"),
        0,
        {
            "priority": "P0",
            "track": "refill",
            "parallel_lane": "lane",
            "resource_class": "cpu",
        },
    )


def _policy(*, activated: bool) -> SignedRefillPolicy:
    return SignedRefillPolicy(
        schema=SIGNED_REFILL_POLICY_SCHEMA,
        policy_cid="sha256:" + ("a" * 64),
        max_epochs=8,
        max_new_work_per_epoch=3,
        max_unchanged_epochs=2,
        activation_authorized=activated,
        signer_identity_did="did:key:test",
    )


def _runtime(tmp_path: Path, *, activated: bool, gaps=(_gap(),)):
    store = RefillStore(tmp_path / "refill-store")
    evaluator = CurrentTreeResidualEvaluator(
        required_tree_id="tree-1",
        residual_fn=lambda _obs: gaps,
        completion_fn=lambda _obs: CompletionAuthorityDecision(False),
    )
    return ProductionRefillRuntime(
        store=store,
        policy=_policy(activated=activated),
        evaluator=evaluator,
        event_adapter=ProductionRefillEventAdapter(),
    ), store


def test_dormant_until_activation_authorization(tmp_path: Path) -> None:
    runtime, store = _runtime(tmp_path, activated=False)
    receipt = runtime.run_once(
        RefillObservation("plan-root", 1, open_goals=1),
        tree_id="tree-1",
        logical_attempt_id="attempt-dormant",
    )
    assert receipt.dormant is True
    assert receipt.disposition == "dormant"
    assert store.load_cursor("attempt-dormant") is None


def test_full_saga_lineage_is_monotonic(tmp_path: Path) -> None:
    runtime, store = _runtime(tmp_path, activated=True)
    receipt = runtime.run_once(
        RefillObservation("plan-root", 4, open_goals=1),
        tree_id="tree-1",
        logical_attempt_id="attempt-full",
        now_ms=1_000,
    )
    assert receipt.dormant is False
    assert receipt.disposition == "refilled"
    assert receipt.phase == RefillSagaPhase.ADOPTED.value
    assert receipt.winner is True
    assert receipt.append_receipt_cid.startswith("sha256:")
    assert receipt.plan_invalidation_cid.startswith("sha256:")
    assert receipt.recompile_cid.startswith("sha256:")
    assert receipt.dispatch_cid.startswith("sha256:")
    assert receipt.gap_identities

    cursor = store.load_cursor("attempt-full")
    assert cursor is not None
    assert cursor.phase == RefillSagaPhase.ADOPTED.value
    assert cursor.predecessor_cid
    assert cursor.append_receipt_cid == receipt.append_receipt_cid


def test_second_process_adopts_same_terminal(tmp_path: Path) -> None:
    runtime, store = _runtime(tmp_path, activated=True)
    first = runtime.run_once(
        RefillObservation("plan-root", 2, open_goals=1),
        tree_id="tree-1",
        logical_attempt_id="attempt-mp",
        now_ms=2_000,
    )
    second = runtime.run_once(
        RefillObservation("plan-root", 2, open_goals=1),
        tree_id="tree-1",
        logical_attempt_id="attempt-mp",
        now_ms=2_500,
    )
    assert first.disposition == "refilled"
    assert second.disposition == "adopted"
    assert second.winner is False
    assert second.append_receipt_cid == first.append_receipt_cid
    assert second.dispatch_cid == first.dispatch_cid


def _worker(root: str, attempt_id: str, q: mp.Queue) -> None:
    store = RefillStore(root)
    evaluator = CurrentTreeResidualEvaluator(
        required_tree_id="tree-1",
        residual_fn=lambda _obs: (_gap("shared"),),
        completion_fn=lambda _obs: CompletionAuthorityDecision(False),
    )
    runtime = ProductionRefillRuntime(
        store=store,
        policy=_policy(activated=True),
        evaluator=evaluator,
    )
    receipt = runtime.run_once(
        RefillObservation("plan-root", 9, open_goals=1),
        tree_id="tree-1",
        logical_attempt_id=attempt_id,
    )
    q.put(
        {
            "disposition": receipt.disposition,
            "append": receipt.append_receipt_cid,
            "dispatch": receipt.dispatch_cid,
            "phase": receipt.phase,
            "winner": receipt.winner,
        }
    )


def test_two_real_processes_one_revision(tmp_path: Path) -> None:
    root = str(tmp_path / "mp-store")
    # Seed winner in parent so children only adopt.
    runtime, _store = _runtime(tmp_path / "seed-unused", activated=True)
    seed_store = RefillStore(root)
    seed_runtime = ProductionRefillRuntime(
        store=seed_store,
        policy=_policy(activated=True),
        evaluator=CurrentTreeResidualEvaluator(
            required_tree_id="tree-1",
            residual_fn=lambda _obs: (_gap("shared"),),
            completion_fn=lambda _obs: CompletionAuthorityDecision(False),
        ),
    )
    seed = seed_runtime.run_once(
        RefillObservation("plan-root", 9, open_goals=1),
        tree_id="tree-1",
        logical_attempt_id="attempt-shared",
    )
    assert seed.disposition == "refilled"

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    workers = [
        ctx.Process(target=_worker, args=(root, "attempt-shared", q))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    results = [q.get(timeout=15) for _ in workers]
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    assert all(item["disposition"] == "adopted" for item in results)
    assert {item["append"] for item in results} == {seed.append_receipt_cid}
    assert {item["dispatch"] for item in results} == {seed.dispatch_cid}


def test_phase_skip_and_stale_fence_fail_closed(tmp_path: Path) -> None:
    store = RefillStore(tmp_path / "store")
    cursor, created, _ = store.begin_or_adopt(
        logical_attempt_id="attempt-fence",
        plan_root_cid="plan",
        tree_id="tree-1",
        epoch=1,
        activation_authorized=True,
        now_ms=10,
    )
    assert created is True
    with pytest.raises(RefillStoreError, match="fence"):
        store.advance(
            "attempt-fence",
            fence_token="wrong",
            next_phase=RefillSagaPhase.APPEND_RESERVED.value,
            tree_id="tree-1",
            now_ms=11,
        )
    with pytest.raises(RefillStoreError, match="cannot advance"):
        store.advance(
            "attempt-fence",
            fence_token=cursor.fence_token,
            next_phase=RefillSagaPhase.APPENDED.value,
            tree_id="tree-1",
            now_ms=12,
        )


def test_event_adapter_does_not_authorize_completion() -> None:
    adapter = ProductionRefillEventAdapter()
    manifest = adapter.manifest()
    assert manifest["authorizes_append"] is False
    assert manifest["authorizes_completion"] is False
    observation = adapter.to_observation(
        plan_root_cid="plan",
        revision=1,
        events=({"kind": "validation_rejected"}, {"kind": "scheduler_low_water"}),
        ready_tasks=0,
        open_goals=2,
    )
    assert observation.validation_rejected is True
    assert observation.open_goals == 2


def test_boolean_callback_cannot_force_append_when_dormant(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path, activated=False)
    # Even with aggressive residual pressure, dormant policy denies effects.
    receipt = runtime.run_once(
        RefillObservation(
            "plan-root",
            1,
            open_goals=10,
            ready_tasks=0,
            validation_rejected=True,
            actionable_drift=True,
        ),
        tree_id="tree-1",
        logical_attempt_id="attempt-no-bool",
    )
    assert receipt.dormant is True
    assert receipt.append_receipt_cid == ""


def test_deadline_expiry_fails_closed(tmp_path: Path) -> None:
    store = RefillStore(tmp_path / "store")
    cursor, created, _ = store.begin_or_adopt(
        logical_attempt_id="attempt-deadline",
        plan_root_cid="plan",
        tree_id="tree-1",
        epoch=1,
        activation_authorized=True,
        phase_budget_ms=5,
        now_ms=100,
    )
    assert created
    with pytest.raises(RefillStoreError, match="deadline"):
        store.advance(
            "attempt-deadline",
            fence_token=cursor.fence_token,
            next_phase=RefillSagaPhase.APPEND_RESERVED.value,
            tree_id="tree-1",
            now_ms=200,
            phase_budget_ms=5,
        )
