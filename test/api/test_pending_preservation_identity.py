"""Future Native checkpoint identity does not depend on administrative cache."""
import os
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import pending_task_claim_recovery as legacy
from test.api.test_pending_task_claim_recovery import pending

pytestmark = pytest.mark.parametrize("pending", ["candidate"], indirect=True)

def test_future_native_preservation_carries_identity_without_registered_cache(pending):
    daemon, task, workspace, _ = pending
    if os.environ.get("TEST_ACTUAL_LEGACY_NATIVE") == "1":
        # Compatibility run targets only the historical bridge tests.
        raise AssertionError("future Native test must run against updated source")
    daemon._task_identity_by_display_id.clear()
    request = legacy.review_recovery(daemon, task_id=task.task_id)
    result = legacy.reconcile_recovery(daemon, request=request)
    assert result["reconciled"], result
    event = next(row for row in daemon._iter_events() if row["type"] == "implementation_pending_candidate_preserved")
    identity = daemon._identity_for_task(task)
    assert event["canonical_task_cid"] == identity.canonical_task_cid
    assert event["canonical_task_key"] == identity.canonical_task_key
    assert event["board_namespace"] == identity.board_namespace
