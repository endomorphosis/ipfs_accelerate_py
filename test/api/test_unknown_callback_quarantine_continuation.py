"""Unknown-callback quarantine continuation must not freeze independent work."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_RETRYING_RECEIPT_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.unknown_callback_quarantine_continuation import (
    DEAD_ADMITTED_OPERATION,
    DUAL_IDENTITY_REPORT_REASON,
    DUAL_PENDING_MERGE_IDENTITY_REASON,
    OPERATION,
    SCHEMA,
    SUCCESSOR_REASON,
    admits_owner_continuation,
    continuation_receipt,
    dual_identity_observation,
    is_dual_pending_merge_identity_block,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
    _unknown_callback_quarantine_receipt,
)


def _identity_receipt(**extra: object) -> dict[str, object]:
    receipt = {
        "attempt_id": "attempt:unknown-continuation",
        "claim_id": "claim:unknown-continuation",
        "lease_id": "lease:unknown-continuation",
        "owner_session_id": "session:unknown-continuation",
        "attempt_number": 1,
        "fencing_token": 2,
        "fence_epoch": 3,
    }
    receipt.update(extra)
    return receipt


def _quarantine_task(*, operation: str, **extra: object) -> SimpleNamespace:
    return SimpleNamespace(
        task_cid="task:cid:001",
        status="quarantined",
        revision=4,
        body={"completion_receipt": _identity_receipt(operation=operation, **extra)},
    )


def test_continuation_vocabulary_is_admitted_retrying_operation() -> None:
    assert OPERATION in TYPED_RETRYING_RECEIPT_OPERATIONS
    assert OPERATION == "database_unknown_callback_quarantine_continuation"


def test_dual_pending_merge_identity_is_reported_not_continued() -> None:
    task = SimpleNamespace(
        task_cid="task:cid:063",
        status="blocked",
        revision=4,
        body={
            "completion_receipt": {
                "operation": "database_portal_terminal_failure",
                "reason": DUAL_PENDING_MERGE_IDENTITY_REASON,
                "retryable": True,
            }
        },
    )
    assert is_dual_pending_merge_identity_block(task)
    assert not admits_owner_continuation(task)
    assert continuation_receipt(task, expected_revision=4) is None
    observation = dual_identity_observation(task)
    assert observation["reopened"] is False
    assert observation["reason"] == DUAL_IDENTITY_REPORT_REASON
    assert observation["history_rewritten"] is False
    assert observation["completion_authoritative"] is False


def test_continuation_receipt_preserves_unknown_and_admits_successor() -> None:
    task = _quarantine_task(
        operation="database_portal_neutral_failure_quarantine",
        failure_kind="provider_callback_outcome_unknown",
        retry_suppressed=True,
        provider_effect_state="unknown_may_have_started",
    )
    receipt = continuation_receipt(task, expected_revision=4)
    assert receipt is not None
    assert receipt["schema"] == SCHEMA
    assert receipt["operation"] == OPERATION
    assert receipt["preserved_unknown_receipt"] == task.body["completion_receipt"]
    assert receipt["unknown_preserved"] is True
    assert receipt["completion_authoritative"] is False
    assert receipt["history_rewritten"] is False
    assert receipt["successor_attempt_admitted"] is True
    assert receipt["provider_dispatched"] is False
    assert receipt["attempt_number"] == 2
    assert receipt["control_expected_revision"] == 4
    assert receipt["reason"] == SUCCESSOR_REASON


def test_continuation_receipt_rejects_forged_identity() -> None:
    task = _quarantine_task(
        operation="database_portal_neutral_failure_quarantine",
        failure_kind="provider_callback_outcome_unknown",
        retry_suppressed=True,
        attempt_id="",
    )
    assert continuation_receipt(task, expected_revision=4) is None


def test_lost_attempt_cursor_admits_successor_without_provider_dispatch(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path / "lane",
        repo_root=tmp_path,
        provider_fn=lambda attempt: provider_calls.append(attempt.attempt_id)
        or {"status": "ok", "accepted": True},
        post_commit_candidate_recovery_fn=lambda _attempt: {
            "schema": "ipfs_accelerate_py/agent-supervisor/not-a-candidate@1"
        },
        max_task_attempts=3,
    )
    try:
        population = _population(3)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[2]["dependencies"] = ["task:cid:001"]
        daemon.materialize_population(population)
        source = daemon.task_source.get("task:cid:001")
        assert source is not None
        receipt = _unknown_callback_quarantine_receipt()
        receipt.update(_identity_receipt())
        quarantined = daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(source.revision),
            "quarantined",
            receipt=receipt,
        ).task
        outcome = daemon._reopen_unimplemented_unknown_callback_task(quarantined)
        assert outcome is not None
        assert outcome["reopened"] is True
        assert outcome["reason"] == SUCCESSOR_REASON
        assert outcome["provider_dispatched"] is False
        assert outcome["unknown_preserved"] is True
        assert outcome["completion_authoritative"] is False
        updated = daemon.task_source.get("task:cid:001")
        assert updated is not None
        assert updated.status == "retrying"
        continued = updated.body["completion_receipt"]
        assert continued["operation"] == OPERATION
        assert continued["preserved_unknown_receipt"]["retry_suppressed"] is True
        assert continued["preserved_unknown_receipt"]["failure_kind"] == (
            "provider_callback_outcome_unknown"
        )
        history = [
            entry.get("status")
            for entry in daemon.task_source.task_revision_history_projection(
                "task:cid:001"
            ).get("revisions", [])
            if isinstance(entry, dict)
        ]
        assert "quarantined" in history
        assert provider_calls == []
        independent = daemon.claim_next(exclude_task_cids=("task:cid:001",))
        assert independent is not None
        assert independent.task_cid == "task:cid:002"
        dependent = daemon.claim_next(
            exclude_task_cids=("task:cid:001", independent.task_cid)
        )
        assert dependent is None
        still_open = daemon.task_source.get("task:cid:003")
        assert still_open is not None
        assert still_open.status in {"ready", "todo", "pending"}
    finally:
        daemon.close()


def test_dead_admitted_unknown_without_candidate_admits_successor(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path / "lane",
        repo_root=tmp_path,
        post_commit_candidate_recovery_fn=lambda _attempt: {
            "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-callback-no-effect-recovery@1"
        },
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        source = daemon.task_source.get("task:cid:001")
        assert source is not None
        receipt = _identity_receipt(
            operation=DEAD_ADMITTED_OPERATION,
            reason="database_attempt_admitted_owner_dead_provider_outcome_unknown",
            retry_suppressed=True,
        )
        quarantined = daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(source.revision),
            "quarantined",
            receipt=receipt,
        ).task
        outcome = daemon._reopen_unimplemented_unknown_callback_task(quarantined)
        assert outcome is not None
        assert outcome["reopened"] is True
        assert outcome["unknown_preserved"] is True
        updated = daemon.task_source.get("task:cid:001")
        assert updated is not None and updated.status == "retrying"
        assert updated.body["completion_receipt"]["preserved_unknown_receipt"][
            "operation"
        ] == DEAD_ADMITTED_OPERATION
        claimed = daemon.claim_next()
        assert claimed is not None
        assert claimed.task_cid == "task:cid:001"
    finally:
        daemon.close()


def test_dual_identity_block_is_explicit_and_does_not_reopen(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path / "lane",
        repo_root=tmp_path,
        post_commit_candidate_recovery_fn=lambda _attempt: {
            "schema": "ipfs_accelerate_py/agent-supervisor/not-used@1"
        },
    )
    try:
        daemon.materialize_population(_population(2))
        source = daemon.task_source.get("task:cid:001")
        assert source is not None
        blocked = daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(source.revision),
            "blocked",
            receipt={
                "operation": "database_portal_terminal_failure",
                "reason": DUAL_PENDING_MERGE_IDENTITY_REASON,
                "retryable": True,
            },
        ).task
        outcome = daemon._reopen_unimplemented_unknown_callback_task(blocked)
        assert outcome is not None
        assert outcome["reopened"] is False
        assert outcome["reason"] == DUAL_IDENTITY_REPORT_REASON
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status == "blocked"
        assert current.revision == blocked.revision
        assert current.body["completion_receipt"]["reason"] == (
            DUAL_PENDING_MERGE_IDENTITY_REASON
        )
        independent = daemon.claim_next()
        assert independent is not None
        assert independent.task_cid == "task:cid:002"
    finally:
        daemon.close()


def test_without_recovery_adapter_unknown_callback_stays_fail_closed(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path / "lane", repo_root=tmp_path)
    try:
        daemon.materialize_population(_population(1))
        source = daemon.task_source.get("task:cid:001")
        assert source is not None
        receipt = _unknown_callback_quarantine_receipt()
        receipt.update(_identity_receipt())
        quarantined = daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(source.revision),
            "quarantined",
            receipt=receipt,
        ).task
        assert daemon._reopen_unimplemented_unknown_callback_task(quarantined) is None
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status == "quarantined"
        assert current.revision == quarantined.revision
    finally:
        daemon.close()
