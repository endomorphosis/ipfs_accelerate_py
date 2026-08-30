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


