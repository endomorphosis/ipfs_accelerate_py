"""The installed SDK exposes pending recovery without per-campaign helpers."""
from dataclasses import asdict
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import pending_task_claim_recovery as recovery
from test.api.test_pending_task_claim_recovery import pending


def test_default_native_public_facades_release_same_pending_claim_without_new_authority(pending):
    daemon, task, workspace, claim = pending
    assert type(daemon) is native.PortalImplementationDaemon
    assert Path(recovery.__file__).parent == Path(native.__file__).parent
    assert not (daemon.state_path.parent / "pending-task-claim-runtime.json").exists()
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    source = daemon.todo_path.read_bytes()
    request = daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    assert claim.exists() and before == asdict(native.PortalTaskState.load(daemon.state_path))
    result = daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert result["reconciled"] is True and result["blocked"] is False and not claim.exists()
    assert result["counters_preserved"] is True and result["task_status_unchanged"] is True
    assert all(result[key] is False for key in ("completion_authority", "attempt_refunded",
        "provider_callback_settled", "new_attempt_authority"))
    assert before == asdict(native.PortalTaskState.load(daemon.state_path))
    assert daemon.todo_path.read_bytes() == source
    repeated = daemon.reconcile_quiesced_pending_task_claim(request=request)
    assert repeated["receipt_id"] == result["receipt_id"]
    assert repeated["reason"] == "pending_claim_already_released"


def test_default_native_public_review_refuses_dirty_pending_source_without_claim_mutation(pending):
    daemon, task, workspace, claim = pending
    before = claim.read_bytes()
    (workspace / "unadmitted.txt").write_text("Synthetic unadmitted edit\n")
    with pytest.raises(ValueError, match="pending_workspace_not_clean_baseline"):
        daemon.review_quiesced_pending_task_claim(task_id=task.task_id)
    assert claim.read_bytes() == before
