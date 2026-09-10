"""Closed callback diagnostics must accept the lifecycle producer's denials."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorktreeLifecycleStore,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)


def _denied_reuse_event(tmp_path: Path, state: str) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    workspace = tmp_path / "workspace"
    branch = "implementation/lifecycle-proof"
    store = WorktreeLifecycleStore(
        repo_root=repo, store_dir=tmp_path / "lifecycle"
    )
    record = store.begin_preparing(
        task_id="TASK-1", attempt=1, lane_id="lane-owner",
        workspace_path=workspace, branch=branch, merge_target="main",
    )
    if state in {"active", "settling"}:
        record = store.mark_active(
            workspace, lease_id=record.lease_id, expected_fence=record.fence
        )
    if state == "settling":
        record = store.mark_settling(
            workspace, lease_id=record.lease_id, expected_fence=record.fence
        )
    decision = store.evaluate_cleanup(workspace_path=workspace, branch=branch)
    assert decision.allowed is False
    assert decision.record.state.value == state
    # This is the event producer's body plus the normal event-log envelope.
    return {
        "type": "worktree_pool_reuse_fenced",
        "timestamp": "2026-09-10T07:00:00+00:00",
        "stream_id": "event-log:sha256:" + "1" * 64,
        "snapshot_id": "event-log-snapshot:sha256:" + "2" * 64,
        "sequence": 1,
        "previous_event_id": "",
        "event_id": "sha256:" + "3" * 64,
        "worktree_path": str(workspace), "branch": branch, "phase": "preflight",
        **decision.to_dict(),
    }


@pytest.mark.parametrize("state", ["preparing", "active", "settling"])
def test_native_live_owner_denial_is_a_valid_callback_diagnostic(
    tmp_path: Path, state: str,
) -> None:
    event = _denied_reuse_event(tmp_path, state)
    DatabasePortalExecutionBridge._validate_no_provider_event_shape(event)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason", "nonterminal_active_owner_alive"),
        ("reason", "nonterminal_terminal_owner_alive"),
        ("reason", "process_inspection_unavailable"),
        ("disposition", "allow"),
        ("phase", "claimed"),
        ("allowed", True),
        ("attempt_consumed", True),
        ("provider_call_allowed", True),
        ("record.state", "active"),
        ("record.state", "terminal"),
        ("record.workspace_path", "/another/workspace"),
        ("record.branch", "implementation/another-task"),
    ],
)
def test_mismatched_or_effect_allowing_denial_is_rejected(
    tmp_path: Path, field: str, value: object,
) -> None:
    event = copy.deepcopy(_denied_reuse_event(tmp_path, "settling"))
    if field.startswith("record."):
        event["record"][field.partition(".")[2]] = value
    else:
        event[field] = value
    with pytest.raises(DatabasePortalBridgeError, match="pool-reuse event is malformed"):
        DatabasePortalExecutionBridge._validate_no_provider_event_shape(event)
