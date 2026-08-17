from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import EventCursor
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    append_jsonl_event,
    latest_event_cursor,
    read_jsonl_events,
)
from ipfs_accelerate_py.agent_supervisor.runtime.learning_checkpoint import (
    CAMPAIGN_DURABILITY_REQUIREMENT_ID,
    LEARNING_CHECKPOINT_BINDING_FIELDS,
    IncompatibleResumeError,
    LearningCheckpointBinding,
    PromotionMutationError,
    StaleFenceError,
)
from ipfs_accelerate_py.agent_supervisor.rescue.learning_recovery import (
    LearningCheckpointAdapter,
)
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    BOUNDED_RECOVERY_REQUIREMENT_ID,
    FaultInjector,
    RecoveryCheckpoint,
    RecoveryCheckpointStore,
    RecoveryDisposition,
    RecoveryFault,
    RecoveryIntegrityError,
    RecoveryPolicy,
    RepairReceipt,
    SupervisorRecovery,
    verify_repair_receipt,
)


def _checkpoint(
    root: Path,
    *,
    generation: int = 1,
    evidence: tuple[str, ...] = ("merged-tree-proof",),
) -> tuple[Path, RecoveryCheckpoint]:
    event_log = root / "events.jsonl"
    append_jsonl_event(event_log, "task_changed", {"task_id": "ASI-118"})
    cursor = latest_event_cursor(event_log)
    manager = SupervisorRecovery(root / "recovery")
    checkpoint = manager.checkpoint(
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=generation,
        state={"phase": "validation", "attempt": 2},
        cursor=cursor,
        accepted_merged_tree_evidence=evidence,
    )
    return event_log, checkpoint


def test_fault_population_is_closed_and_requirement_is_explicit() -> None:
    assert BOUNDED_RECOVERY_REQUIREMENT_ID
    assert {item.value for item in RecoveryFault} == {
        "process_crash",
        "kill_escalation",
        "partial_event_write",
        "partial_checkpoint_write",
        "stale_lease",
        "corrupt_cache",
        "duplicate_event",
        "provider_loss",
        "disk_full",
        "slow_disk",
        "interrupted_validation",
        "interrupted_merge",
        "restart_during_refill",
    }


def test_partial_event_tail_is_quarantined_from_checkpoint(
    tmp_path: Path,
) -> None:
    event_log, checkpoint = _checkpoint(tmp_path)
    with event_log.open("ab") as stream:
        stream.write(b'{"type":"partial"')

    manager = SupervisorRecovery(tmp_path / "recovery")
    receipt = manager.recover(
        incident_id="partial-event-1",
        fault=RecoveryFault.PARTIAL_EVENT_WRITE,
        repository_id="repository:current",
        tree_id="tree:merged",
        event_log_path=event_log,
        verify=lambda restored: restored.checkpoint_id == checkpoint.checkpoint_id,
    )

    assert receipt.disposition is RecoveryDisposition.RECOVERED
    assert receipt.checkpoint_id == checkpoint.checkpoint_id
    assert receipt.event_cursor == checkpoint.cursor
    assert receipt.preserved_evidence_ids == ("merged-tree-proof",)
    assert receipt.evidence_claim_ids == (BOUNDED_RECOVERY_REQUIREMENT_ID,)
    assert len(receipt.quarantined_paths) == 1
    assert Path(receipt.quarantined_paths[0]).read_bytes() == b'{"type":"partial"'
    assert [event["type"] for event in read_jsonl_events(event_log)] == ["task_changed"]


def test_corrupt_latest_checkpoint_falls_back_to_last_valid(
    tmp_path: Path,
) -> None:
    event_log, first = _checkpoint(tmp_path)
    manager = SupervisorRecovery(tmp_path / "recovery")
    second = manager.checkpoint(
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=2,
        state={"phase": "merge"},
        cursor=latest_event_cursor(event_log),
    )
    latest_path = manager.checkpoints._checkpoint_path(second)
    latest_path.write_bytes(b'{"partial":')

    restored = manager.checkpoints.load_last_valid()

    assert restored == first
    assert not latest_path.exists()
    assert list(manager.checkpoints.quarantine.glob("*.invalid-*"))


def test_retry_exhaustion_quarantines_and_receipt_is_idempotent(
    tmp_path: Path,
) -> None:
    _event_log, checkpoint = _checkpoint(tmp_path)
    suspect = tmp_path / "partial-state.json"
    suspect.write_text("partial", encoding="utf-8")
    manager = SupervisorRecovery(
        tmp_path / "recovery",
        policy=RecoveryPolicy(max_attempts=2),
    )
    attempts: list[int] = []

    receipt = manager.recover(
        incident_id="provider-loss-1",
        fault=RecoveryFault.PROVIDER_LOSS,
        repository_id="repository:current",
        tree_id="tree:merged",
        repair=lambda _checkpoint, attempt: attempts.append(attempt) or False,
        quarantine_paths=(suspect,),
    )

    assert attempts == [1, 2]
    assert receipt.disposition is RecoveryDisposition.QUARANTINED
    assert receipt.attempts == 2
    assert receipt.checkpoint_id == checkpoint.checkpoint_id
    assert not suspect.exists()
    assert Path(receipt.quarantined_paths[0]).exists()
    assert receipt.evidence_claim_ids == ()
    assert (
        manager.recover(
            incident_id="provider-loss-1",
            fault=RecoveryFault.PROVIDER_LOSS,
            repository_id="repository:current",
            tree_id="tree:merged",
        ).receipt_id
        == receipt.receipt_id
    )


def test_stale_actor_is_fenced_without_losing_merged_tree_evidence(
    tmp_path: Path,
) -> None:
    _event_log, _checkpoint_value = _checkpoint(
        tmp_path, evidence=("merge-receipt", "validation-receipt")
    )
    manager = SupervisorRecovery(tmp_path / "recovery")

    receipt = manager.recover(
        incident_id="stale-lease-1",
        fault=RecoveryFault.STALE_LEASE,
        repository_id="repository:current",
        tree_id="tree:merged",
        current_fencing_token=8,
        observed_fencing_token=7,
    )

    assert receipt.stale_actor_fenced
    assert "fence_stale_actor" in receipt.actions
    assert receipt.preserved_evidence_ids == (
        "merge-receipt",
        "validation-receipt",
    )


def test_receipt_identity_detects_tampering_and_tree_staleness(
    tmp_path: Path,
) -> None:
    _event_log, _checkpoint_value = _checkpoint(tmp_path)
    manager = SupervisorRecovery(tmp_path / "recovery")
    receipt = manager.recover(
        incident_id="crash-1",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id="repository:current",
        tree_id="tree:merged",
    )
    assert (
        verify_repair_receipt(
            receipt,
            repository_id="repository:current",
            tree_id="tree:merged",
        )
        is receipt
    )
    tampered = receipt.to_dict()
    tampered["reason_code"] = "invented_success"
    with pytest.raises(RecoveryIntegrityError):
        RepairReceipt.from_dict(tampered)
    with pytest.raises(RecoveryIntegrityError):
        verify_repair_receipt(receipt, tree_id="tree:new")


def test_fault_injector_is_explicit_and_finite() -> None:
    injector = FaultInjector()
    injector.arm("write", OSError("disk full"), times=2)
    with pytest.raises(OSError, match="disk full"):
        injector.inject("write")
    with pytest.raises(OSError, match="disk full"):
        injector.inject("write")
    assert injector.inject("write") is False


def test_missing_checkpoint_fails_closed_with_exact_receipt(
    tmp_path: Path,
) -> None:
    manager = SupervisorRecovery(tmp_path / "recovery")
    receipt = manager.recover(
        incident_id="missing-checkpoint-1",
        fault=RecoveryFault.RESTART_DURING_REFILL,
        repository_id="repository:current",
        tree_id="tree:merged",
    )
    persisted = json.loads(manager._receipt_path(receipt.incident_id).read_text(encoding="utf-8"))
    assert receipt.disposition is RecoveryDisposition.FAILED_CLOSED
    assert receipt.reason_code == "no_valid_checkpoint"
    assert RepairReceipt.from_dict(persisted) == receipt


def test_checkpoint_store_prunes_history_to_policy_bound(
    tmp_path: Path,
) -> None:
    event_log = tmp_path / "events.jsonl"
    append_jsonl_event(event_log, "started", {})
    cursor = latest_event_cursor(event_log)
    store = RecoveryCheckpointStore(
        tmp_path / "recovery",
        policy=RecoveryPolicy(max_checkpoints=2),
    )
    for generation in range(1, 5):
        store.save(
            RecoveryCheckpoint(
                repository_id="repository:current",
                tree_id="tree:merged",
                generation=generation,
                state={"generation": generation},
                cursor=cursor,
            )
        )
    assert len(list(store.checkpoints.glob("*.json"))) == 2
    assert store.load_last_valid().generation == 4  # type: ignore[union-attr]


def test_concurrent_incident_recovery_collapses_to_one_repair(
    tmp_path: Path,
) -> None:
    _event_log, _checkpoint_value = _checkpoint(tmp_path)
    manager = SupervisorRecovery(tmp_path / "recovery")
    repair_count = 0
    count_lock = threading.Lock()

    def repair(_checkpoint: RecoveryCheckpoint, _attempt: int) -> bool:
        nonlocal repair_count
        with count_lock:
            repair_count += 1
        return True

    def recover() -> RepairReceipt:
        return manager.recover(
            incident_id="concurrent-crash-1",
            fault=RecoveryFault.PROCESS_CRASH,
            repository_id="repository:current",
            tree_id="tree:merged",
            repair=repair,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _index: recover(), range(4)))

    assert repair_count == 1
    assert len({receipt.receipt_id for receipt in receipts}) == 1


def _learning_binding(**overrides: object) -> LearningCheckpointBinding:
    payload = {
        "architecture_id": "arch:v1",
        "weights_id": "weights:0",
        "optimizer_id": "opt:adam",
        "scheduler_id": "sched:cosine",
        "tokenizer_id": "tok:v1",
        "vocab_id": "vocab:v1",
        "cursor_id": "cursor:0",
        "corpus_id": "corpus:v1",
        "split_id": "split:v1",
        "curriculum_id": "curr:v1",
        "loss_id": "loss:ce",
        "random_id": "rng:0",
        "env_id": "env:v1",
        "code_id": "code:v1",
        "compiler_id": "compiler:v1",
        "cursor_step": 0,
    }
    payload.update(overrides)
    return LearningCheckpointBinding.from_dict(payload)


def test_learning_checkpoint_binds_required_identities(tmp_path: Path) -> None:
    adapter = LearningCheckpointAdapter(tmp_path / "recovery")
    binding = _learning_binding()
    cursor = EventCursor.initial("stream:learning", snapshot_id="tree:merged")
    checkpoint = adapter.save(
        binding,
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=1,
        cursor=cursor,
        fence=1,
    )
    stored = LearningCheckpointBinding.from_dict(checkpoint.state["binding"])
    assert CAMPAIGN_DURABILITY_REQUIREMENT_ID
    assert tuple(stored.to_dict())  # identity is complete
    for name in LEARNING_CHECKPOINT_BINDING_FIELDS:
        assert getattr(stored, name)
    assert checkpoint.state["promotion_authority"] is False
    assert stored.lineage_id == binding.lineage_id


def test_incompatible_resume_fails_closed(tmp_path: Path) -> None:
    adapter = LearningCheckpointAdapter(tmp_path / "recovery")
    cursor = EventCursor.initial("stream:learning", snapshot_id="tree:merged")
    adapter.save(
        _learning_binding(),
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=1,
        cursor=cursor,
        fence=1,
    )
    with pytest.raises(IncompatibleResumeError, match="tokenizer_id"):
        adapter.resume(
            _learning_binding(tokenizer_id="tok:other"),
            repository_id="repository:current",
            tree_id="tree:merged",
        )
    advanced = _learning_binding(
        weights_id="weights:1",
        cursor_id="cursor:8",
        random_id="rng:8",
        cursor_step=8,
    )
    adapter.save(
        advanced,
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=2,
        cursor=cursor,
        fence=2,
    )
    resumed = adapter.resume(
        advanced,
        repository_id="repository:current",
        tree_id="tree:merged",
        fence=2,
    )
    assert resumed.compatible
    assert resumed.promotion_authority is False
    with pytest.raises(IncompatibleResumeError, match="cursor_step moved backwards"):
        adapter.save(
            _learning_binding(cursor_step=0),
            repository_id="repository:current",
            tree_id="tree:merged",
            generation=3,
            cursor=cursor,
            fence=3,
        )


def test_overwrite_without_current_fence_fails_closed(tmp_path: Path) -> None:
    adapter = LearningCheckpointAdapter(tmp_path / "recovery")
    cursor = EventCursor.initial("stream:learning", snapshot_id="tree:merged")
    adapter.save(
        _learning_binding(),
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=1,
        cursor=cursor,
        fence=4,
    )
    with pytest.raises(StaleFenceError, match="stale fence 3"):
        adapter.save(
            _learning_binding(
                weights_id="weights:1",
                cursor_id="cursor:1",
                random_id="rng:1",
                cursor_step=1,
            ),
            repository_id="repository:current",
            tree_id="tree:merged",
            generation=2,
            cursor=cursor,
            fence=3,
        )


def test_learning_crash_restarts_exactly_once(tmp_path: Path) -> None:
    adapter = LearningCheckpointAdapter(tmp_path / "recovery")
    cursor = EventCursor.initial("stream:learning", snapshot_id="tree:merged")
    adapter.save(
        _learning_binding(),
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=1,
        cursor=cursor,
        fence=1,
    )
    first = adapter.recover_crash(
        incident_id="learning-crash-1",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id="repository:current",
        tree_id="tree:merged",
        requested=_learning_binding(),
    )
    second = adapter.recover_crash(
        incident_id="learning-crash-1",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id="repository:current",
        tree_id="tree:merged",
        requested=_learning_binding(),
    )
    restarted = LearningCheckpointAdapter(tmp_path / "recovery")
    third = restarted.recover_crash(
        incident_id="learning-crash-1",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id="repository:current",
        tree_id="tree:merged",
        requested=_learning_binding(),
    )
    assert first.restart_performed is True
    assert first.restart_count == 1
    assert second.restart_performed is False
    assert second.restart_count == 1
    assert third.restart_performed is False
    assert third.restart_count == 1
    assert first.repair is not None and second.repair is not None
    assert first.repair.receipt_id == second.repair.receipt_id == third.repair.receipt_id
    assert first.promotion_authority is False


def test_learning_checkpoint_rejects_promotion_authority(tmp_path: Path) -> None:
    adapter = LearningCheckpointAdapter(tmp_path / "recovery")
    cursor = EventCursor.initial("stream:learning", snapshot_id="tree:merged")
    with pytest.raises(PromotionMutationError):
        adapter.save(
            _learning_binding(),
            repository_id="repository:current",
            tree_id="tree:merged",
            generation=1,
            cursor=cursor,
            fence=1,
            extra={"promotion_pointer": "current"},
        )
    with pytest.raises(PromotionMutationError):
        adapter.save(
            _learning_binding(),
            repository_id="repository:current",
            tree_id="tree:merged",
            generation=1,
            cursor=cursor,
            fence=1,
            extra={"promotion_authority": True},
        )
