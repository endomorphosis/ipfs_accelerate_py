"""A missing priority queue match must not abort normal recovery scanning."""

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)


@pytest.mark.parametrize("ordinary_match", [False, True])
def test_empty_priority_lookup_continues_bounded_public_recovery(
    tmp_path, monkeypatch, ordinary_match,
):
    task_cids = tuple(f"task:cid:{index:03d}" for index in range(35))
    request = SimpleNamespace(request_id="request:ordinary", task_id="TASK-001")
    projection = SimpleNamespace(binding={"task_cid": "task:ordinary"})
    calls = []
    cursors = dict.fromkeys((
        "completed_requests", "pending_requests", "processing_requests",
        "quarantined_requests", "priority_task_cids",
    ), "")

    class Queue:
        max_attempts = 3

        def completed_requests(self, **kwargs):
            calls.append(("completed_page", kwargs))
            return (request,) if ordinary_match else ()

        def get(self, request_id):
            assert request_id == request.request_id
            return request

        def pending_requests(self, **kwargs):
            calls.append(("pending_page", kwargs))
            return ()

        def quarantined_requests(self, **kwargs):
            calls.append(("quarantined_page", kwargs))
            return ()

        def processing_requests(self, **kwargs):
            calls.append(("processing_page", kwargs))
            return ()

    class Train:
        def __init__(self, **kwargs):
            pass

        def run_under_consumer_lease(self, callback):
            calls.append(("consumer_lease", None))
            return True, callback()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.merge.merge_train.MergeTrain", Train,
    )
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.repository_root = tmp_path
    bridge.merge_queue = Queue()
    bridge.merge_target_branch = "main"
    bridge._load_post_merge_recovery_cursors = lambda: dict(cursors)
    bridge._save_post_merge_recovery_cursors = lambda value: cursors.update(value)
    priority_pages = []

    def missing_priority(task_ids):
        priority_pages.append(task_ids)
        return ()

    bridge._priority_repaired_completion_requests = missing_priority
    bridge._owned_post_merge_recovery_projection = lambda row, **kwargs: (
        projection if row is request else None
    )
    bridge._preauthorize_post_merge_recovery = lambda *args, **kwargs: (
        calls.append(("preauthorize", None))
    )
    evidence = {"evidence": "ordinary-current-authority"}
    bridge._post_merge_recovery_evidence = lambda *args, **kwargs: evidence
    result = {"recovered": True, "changed": True, "write_count": 1}

    def recover(value):
        assert value is evidence
        calls.append(("recover", None))
        return result

    authority = SimpleNamespace(
        _database_portal_evidence_digest=lambda value: "sha256:" + "a" * 64,
        preauthorize_post_merge_declared_output_recovery=lambda value: {},
        post_merge_completion_recovery_task_cids=lambda: task_cids,
        recover_blocked_post_merge_declared_outputs=recover,
    )
    observed = bridge.recover_post_merge_declared_outputs(authority)
    assert priority_pages == [task_cids[:32]]
    assert cursors["priority_task_cids"] == task_cids[31]
    if ordinary_match:
        assert observed == result
        assert [name for name, _ in calls] == [
            "completed_page", "consumer_lease", "preauthorize", "recover",
        ]
        assert cursors["completed_requests"] == request.request_id
    else:
        assert observed is None
        assert [name for name, _ in calls] == [
            "completed_page", "pending_page", "quarantined_page", "processing_page",
        ]
        # Persisted progress reaches the remaining identities and wraps,
        # even when none has a matching queue request.
        assert bridge.recover_post_merge_declared_outputs(authority) is None
        assert bridge.recover_post_merge_declared_outputs(authority) is None
        assert priority_pages == [task_cids[:32], task_cids[32:], ()]
        assert cursors["priority_task_cids"] == ""
