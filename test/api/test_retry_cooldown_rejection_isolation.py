"""A rejected queue fence stays intact without crashing every scheduler pass."""

from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
)


def test_newer_cooldown_rejection_does_not_abort_other_task_repairs():
    source = object.__new__(TypedDatabaseTaskSource)
    tasks = []
    for index in (1, 2):
        tasks.append(
            SimpleNamespace(
                task_cid=f"task:{index}",
                task_alias=f"TASK-{index}",
                status="retrying",
                revision=16,
                body={
                    "completion_receipt": {
                        "operation": "database_post_merge_declared_outputs_callback_integration_recovery",
                        "attempt_id": f"attempt:{index}",
                        "claim_id": f"claim:{index}",
                        "lease_id": f"lease:{index}",
                        "owner_session_id": "owner:test",
                        "queue_reason": "retained-callback",
                        "attempt_number": 2,
                        "fencing_token": 2,
                        "fence_epoch": 2,
                        "backoff_ms": 0,
                        "retry_not_before_ms": 42,
                        "control_expected_revision": 15,
                    }
                },
            )
        )
    source.list_tasks = lambda **kwargs: SimpleNamespace(tasks=tasks)
    source._retry_cooldown_row = lambda task_cid: None
    calls = []

    def record(**payload):
        calls.append(payload)
        if payload["task_cid"] == "task:1":
            raise QuackClientError("retry cooldown refuses a newer queue row")
        return SimpleNamespace(changed=True)

    source.record_task_retry_cooldown = record
    results = source.repair_retrying_cooldown_bindings()
    assert [value["task_cid"] for value in calls] == ["task:1", "task:2"]
    assert results[0]["reason"] == "retrying_cooldown_repair_rejected"
    assert results[0]["changed"] is False
    assert results[0]["provider_dispatched"] is False
    assert results[0]["operator_review_required"] is True
    assert results[1]["changed"] is True
    assert tasks[0].revision == 16
    assert tasks[0].body["completion_receipt"]["attempt_number"] == 2
