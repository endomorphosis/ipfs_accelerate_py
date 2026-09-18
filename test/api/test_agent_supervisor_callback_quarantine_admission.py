"""Settled queue evidence must retain exact canonical quarantine authority."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge

@pytest.mark.parametrize("drift", [None, "revision", "receipt", "status", "queue"])
def test_unknown_callback_ordinary_implementation_source_regression(
    drift: str | None,
) -> None:
    alias = "DOEP-012"
    request_id = "request:ordinary-callback"
    paths = SimpleNamespace(root=Path("/attempt/ordinary"))
    binding = {"task_alias": alias, "task_cid": "task:ordinary"}
    request = SimpleNamespace(request_id=request_id, status="completed")
    projection = SimpleNamespace(paths=paths, binding=binding)
    attempt = SimpleNamespace(
        attempt_id="attempt:ordinary",
        claim_id="claim:ordinary",
        lease_id="lease:ordinary",
        fencing_token=7,
        fence_epoch=3,
    )
    control_receipt = {
        "operation": "database_portal_neutral_failure_quarantine",
        "failure_kind": "provider_callback_outcome_unknown",
        "retry_suppressed": True,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
    }
    record = SimpleNamespace(
        status="quarantined",
        revision=19,
        body={"completion_receipt": control_receipt},
    )
    projection_calls: list[frozenset[str]] = []

    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.merge_queue = SimpleNamespace(
        get=lambda value: request if value == request_id else None
    )
    bridge._verified_event_chain = lambda _paths: [
        {
            "type": "implementation_finished",
            "task_id": alias,
            "merge_result": {"request_id": request_id},
        }
    ]

    def owned_projection(
        current: object,
        *,
        allowed_task_statuses: frozenset[str],
        **_kwargs: object,
    ) -> object | None:
        projection_calls.append(allowed_task_statuses)
        return (
            projection
            if current is request and record.status in allowed_task_statuses
            else None
        )

    bridge._owned_post_merge_recovery_projection = owned_projection
    bridge._record_for_attempt = lambda *_args: record
    bridge.task_source = object()

    def evidence(
        current: object,
        current_projection: object,
        *,
        revalidate_authority: object,
        **_kwargs: object,
    ) -> dict[str, object] | None:
        assert current is request
        assert current_projection is projection
        assert callable(revalidate_authority) and revalidate_authority()
        if drift == "revision":
            record.revision += 1
        elif drift == "receipt":
            record.body["completion_receipt"] = {
                **control_receipt,
                "claim_id": "claim:other",
            }
        elif drift == "status":
            record.status = "completed"
        elif drift == "queue":
            bridge.merge_queue.get = lambda _value: None
        if drift:
            assert revalidate_authority() is False
            return None
        return {"ordinary": True}

    bridge._post_merge_callback_integration_evidence = evidence

    if drift:
        assert bridge._unknown_callback_landed_recovery_evidence(
            attempt=attempt, paths=paths, binding=binding,
        ) is None
        assert set(projection_calls) == {frozenset({"quarantined"})}
        return

    assert bridge._unknown_callback_landed_recovery_evidence(
        attempt=attempt,
        paths=paths,
        binding=binding,
    ) == {"ordinary": True}
    assert projection_calls == [
        frozenset({"quarantined"}),
        frozenset({"quarantined"}),
    ]


@pytest.mark.parametrize(
    "drift",
    [None, "revision", "receipt", "status", "queue", "event", "reason"],
)
def test_unknown_callback_reconciliation_queued_append_recovery(
    drift: str | None,
) -> None:
    alias = "DOEP-044"
    request_id = "1789274367627893025-3431490-e4cb4b10c1ad"
    commit = "25f3e8ce4db3964d27f6e1557dd0835cff186e1f"
    paths = SimpleNamespace(root=Path("/attempt/queued-append"))
    binding = {"task_alias": alias, "task_cid": "task:queued-append"}
    request = SimpleNamespace(
        request_id=request_id,
        status="quarantined",
        failure_reason="merge_queue_reconciliation_append_unverified",
        failure_count=1,
        claim_generation=2,
        commit_sha=commit,
    )
    projection = SimpleNamespace(paths=paths, binding=binding)
    attempt = SimpleNamespace(
        attempt_id="attempt:queued-append",
        claim_id="claim:queued-append",
        lease_id="lease:queued-append",
        fencing_token=7,
        fence_epoch=3,
    )
    control_receipt = {
        "operation": "database_portal_neutral_failure_quarantine",
        "failure_kind": "provider_callback_outcome_unknown",
        "retry_suppressed": True,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
    }
    record = SimpleNamespace(
        status="quarantined",
        revision=8,
        body={"completion_receipt": control_receipt},
    )
    if drift == "reason":
        request.failure_reason = "merge_queue_reconciliation_receipt_conflict"
    queued_event = {
        "type": "worktree_reconciliation_candidate_queued",
        "task_id": alias,
        "event_id": "sha256:" + "a" * 64,
        "canonical_task_cid": "task:queued-append",
        "canonical_task_key": "task/v1/queued-append",
        "implementation_commit": commit,
        "merge_result": {"request_id": request_id},
    }
    reconciled_event = {
        "type": "merge_reconciled",
        "task_id": alias,
        "event_id": "sha256:" + "b" * 64,
        "completion_source_event_id": queued_event["event_id"],
        "merge_result": {"request_id": request_id},
    }
    projection_calls: list[frozenset[str]] = []

    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.merge_queue = SimpleNamespace(
        get=lambda value: request if value == request_id else None
    )
    bridge._verified_event_chain = lambda _paths: [
        queued_event,
        reconciled_event,
    ]
    bridge._exact_callback_reconciliation_for_completion_source = (
        lambda *_args, **_kwargs: drift != "event"
    )

    def owned_projection(
        current: object,
        *,
        allowed_task_statuses: frozenset[str],
        **_kwargs: object,
    ) -> object | None:
        projection_calls.append(allowed_task_statuses)
        return (
            projection
            if current is request and record.status in allowed_task_statuses
            else None
        )

    bridge._owned_post_merge_recovery_projection = owned_projection
    bridge._record_for_attempt = lambda *_args: record
    bridge.task_source = object()

    def evidence(
        current: object,
        current_projection: object,
        *,
        revalidate_authority: object,
        **_kwargs: object,
    ) -> dict[str, object] | None:
        assert current is request
        assert current_projection is projection
        assert callable(revalidate_authority) and revalidate_authority()
        if drift == "revision":
            record.revision += 1
        elif drift == "receipt":
            record.body["completion_receipt"] = {
                **control_receipt,
                "claim_id": "claim:other",
            }
        elif drift == "status":
            record.status = "completed"
        elif drift == "queue":
            request.claim_generation += 1
        if drift in {"revision", "receipt", "status", "queue"}:
            assert revalidate_authority() is False
            return None
        return {"queued_append": True}

    bridge._post_merge_callback_integration_evidence = evidence

    recovered = bridge._unknown_callback_reconciliation_queued_recovery_evidence(
        attempt=attempt,
        paths=paths,
        binding=binding,
    )
    if drift:
        assert recovered is None
        return

    assert recovered == {"queued_append": True}
    assert projection_calls == [
        frozenset({"quarantined"}),
        frozenset({"quarantined"}),
    ]


def test_queued_append_recovery_rejects_implementation_finished_alias() -> None:
    alias = "DOEP-044"
    request_id = "request:queued-append-alias"
    paths = SimpleNamespace(root=Path("/attempt/queued-append-alias"))
    binding = {"task_alias": alias, "task_cid": "task:queued-append-alias"}
    attempt = SimpleNamespace(
        attempt_id="attempt:queued-append-alias",
        claim_id="claim:queued-append-alias",
        lease_id="lease:queued-append-alias",
        fencing_token=1,
        fence_epoch=1,
    )
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.merge_queue = SimpleNamespace(get=lambda _value: object())
    bridge._verified_event_chain = lambda _paths: [
        {
            "type": "implementation_finished",
            "task_id": alias,
            "merge_result": {"request_id": request_id},
        }
    ]

    assert bridge._unknown_callback_reconciliation_queued_recovery_evidence(
        attempt=attempt,
        paths=paths,
        binding=binding,
    ) is None


def test_recover_post_commit_candidate_uses_queued_append_recovery() -> None:
    attempt = SimpleNamespace(attempt_id="attempt:wired")
    paths = SimpleNamespace(root=Path("/attempt/wired"))
    binding = {"task_alias": "DOEP-044"}
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge._recovery_attempt_binding = lambda *_args, **_kwargs: (paths, binding)

    def fail_post_commit(*_args: object, **_kwargs: object) -> dict[str, object]:
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
            DatabasePortalBridgeError,
        )

        raise DatabasePortalBridgeError("exact terminal event suffix")

    bridge._post_commit_candidate_recovery_receipt = fail_post_commit
    bridge._unknown_callback_landed_recovery_evidence = (
        lambda **_kwargs: None
    )
    bridge._unknown_callback_reconciliation_queued_recovery_evidence = (
        lambda **_kwargs: {"wired": True}
    )

    assert bridge.recover_post_commit_candidate(attempt) == {"wired": True}


def test_queued_source_effect_admission_is_not_an_event_name_alias() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    queued = {"type": "worktree_reconciliation_candidate_queued"}
    finished = {"type": "implementation_finished"}
    reconciled = {"type": "merge_reconciled"}
    assert (
        PortalImplementationDaemon._queued_source_merge_reconciliation_effect_admitted(
            reconciled,
            finished,
            alias="DOEP-044",
            task_cid="task:cid",
            task_key="task/v1",
            artifacts_match=True,
        )
        is False
    )
    assert (
        PortalImplementationDaemon._queued_source_merge_reconciliation_effect_admitted(
            reconciled,
            queued,
            alias="DOEP-044",
            task_cid="task:cid",
            task_key="task/v1",
            artifacts_match=False,
        )
        is False
    )


