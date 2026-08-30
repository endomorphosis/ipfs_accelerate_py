from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
    MergeRequest,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
)


class _ExactQueue:
    def __init__(self, request: MergeRequest) -> None:
        self.request = request

    def get(self, request_id: str) -> MergeRequest | None:
        return self.request if request_id == self.request.request_id else None


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    PortalImplementationDaemon,
    PortalTask,
    dict[str, object],
    MergeRequest,
]:
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "tasks.md"
    todo_path.write_text("# tasks\n", encoding="utf-8")
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state" / "task-state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## TEST-",
        implement=False,
    )
    task = PortalTask(
        task_id="TEST-001",
        title="Recover an integrated completion",
        status="todo",
        completion="auto",
        priority="P0",
        track="recovery",
        outputs=["feature.py"],
    )
    task_identity = daemon._identity_for_task(task)
    candidate = "a" * 40
    branch = "implementation/test-001-attempt-1"
    request_id = "request-integrated-authority-quarantine"
    completion_task_cids = {
        task.task_id: task_identity.canonical_task_cid,
    }
    event: dict[str, object] = {
        "type": "implementation_finished",
        "task_id": task.task_id,
        "task_cid": task_identity.canonical_task_cid,
        "canonical_task_cid": task_identity.canonical_task_cid,
        "canonical_task_key": task_identity.canonical_task_key,
        "attempt": 1,
        "timestamp": "2000-01-01T00:00:00+00:00",
        "branch": branch,
        "implementation_commit": candidate,
        "validation_result": {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [{"validation_result_digest": "b" * 64}],
        },
        "merge_result": {
            "attempted": False,
            "merged": False,
            "queued": True,
            "reason": "merge_queued",
            "request_id": request_id,
            "canonical_task_cid": task_identity.canonical_task_cid,
            "canonical_task_key": task_identity.canonical_task_key,
            "completion_task_cids": completion_task_cids,
        },
        "cleanup_result": {"cleaned": True},
    }
    request = MergeRequest(
        request_id=request_id,
        branch_name=branch,
        task_id=task.task_id,
        priority="P0",
        lane_id="lane-0",
        enqueued_at=1.0,
        attempt=1,
        metadata={
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
            "target_binding_schema": MERGE_TARGET_BINDING_SCHEMA,
            "target_repository_id": daemon.merge_target_repository_id,
            "target_branch": "main",
            "todo_path": str(todo_path),
            "implementation_commit": candidate,
            "completion_task_cids": completion_task_cids,
        },
        commit_sha=candidate,
        canonical_task_id=task_identity.canonical_task_cid,
        canonical_task_key=task_identity.canonical_task_key,
        status="quarantined",
        failure_count=1,
        failure_reason=(
            "cross_board_manual_completion_authority_metadata_invalid"
        ),
    )
    daemon.merge_queue = _ExactQueue(request)  # type: ignore[assignment]
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [event],
    )
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "main")
    return daemon, task, event, request


def test_exact_integrated_authority_quarantine_is_nominated_and_stays_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _task, event, _request = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(daemon, "_git_ref_is_ancestor", lambda *_: True)

    candidates = daemon._failed_merge_candidates()

    assert len(candidates) == 1
    receipt = candidates[0][
        "integrated_authority_quarantine_reconciliation"
    ]
    assert receipt["passed"] is True
    assert receipt["implementation_commit"] == event["implementation_commit"]
    fresh, stale = daemon._partition_stale_failed_merge_candidates(candidates)
    assert fresh == candidates
    assert stale == []


@pytest.mark.parametrize(
    "poison",
    ("not_integrated", "wrong_commit", "wrong_target", "ordinary_failure"),
)
def test_authority_quarantine_nomination_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    daemon, _task, _event, request = _fixture(tmp_path, monkeypatch)
    integrated = poison != "not_integrated"
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor",
        lambda *_: integrated,
    )
    if poison == "wrong_commit":
        object.__setattr__(request, "commit_sha", "c" * 40)
    elif poison == "wrong_target":
        request.metadata["target_branch"] = "another-branch"
    elif poison == "ordinary_failure":
        object.__setattr__(request, "failure_reason", "validation_failed")

    assert daemon._failed_merge_candidates() == []


def test_integrated_authority_quarantine_uses_full_reconciliation_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    candidate = str(event["implementation_commit"])
    integration_commit = "d" * 40
    monkeypatch.setattr(daemon, "_git_ref_is_ancestor", lambda *_: True)
    monkeypatch.setattr(
        daemon,
        "_preserve_generated_nested_worktree_directories",
        lambda: {},
    )
    monkeypatch.setattr(
        daemon,
        "_reconciliation_blocking_dirty_paths",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(daemon, "_git_ref_exists", lambda _ref: False)
    monkeypatch.setattr(
        daemon,
        "_merge_submodule_branches_to_main",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        daemon,
        "_resolved_commit_ref",
        lambda _repo, ref: integration_commit if ref == "main" else "",
    )
    monkeypatch.setattr(
        daemon,
        "_immutable_integration_commit",
        lambda *_args, **_kwargs: {
            "passed": True,
            "implementation_commit": candidate,
            "integration_commit": integration_commit,
            "target_branch": "main",
            "reasons": [],
        },
    )
    gate_calls: list[tuple[list[str], str]] = []

    def declared_output_gate(
        tasks: list[PortalTask],
        *,
        repository_ref: str,
    ) -> dict[str, object]:
        gate_calls.append(([item.task_id for item in tasks], repository_ref))
        return {"passed": True, "repository_ref": repository_ref}

    monkeypatch.setattr(
        daemon,
        "_declared_output_tracking_invariant",
        declared_output_gate,
    )
    monkeypatch.setattr(
        daemon,
        "_cleanup_merged_worktree",
        lambda *_args, **_kwargs: {"cleaned": True},
    )
    completions: list[dict[str, object]] = []

    def complete(
        primary: PortalTask,
        members: list[PortalTask],
        bindings: dict[str, str],
        *,
        validation_evidence: dict[str, object] | None = None,
    ) -> dict[str, object]:
        completions.append(
            {
                "primary": primary.task_id,
                "members": [member.task_id for member in members],
                "bindings": bindings,
                "validation_evidence": validation_evidence,
            }
        )
        return {"updated": True, "durable": True}

    monkeypatch.setattr(daemon, "_mark_reconciled_completion_in_todo", complete)
    monkeypatch.setattr(
        daemon,
        "_reconciled_completion_persisted",
        lambda *_args, **_kwargs: {"passed": True},
    )

    result = daemon._reconcile_failed_merges()

    assert result[0]["resolved"] is True
    assert result[0]["reason"] == "implementation_commit_already_merged"
    assert gate_calls == [([task.task_id], integration_commit)]
    assert completions == [
        {
            "primary": task.task_id,
            "members": [task.task_id],
            "bindings": {
                task.task_id: daemon._canonical_ref(task),
            },
            "validation_evidence": event["validation_result"],
        }
    ]
