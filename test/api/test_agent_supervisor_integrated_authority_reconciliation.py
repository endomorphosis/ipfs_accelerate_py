from __future__ import annotations

import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
    MergeRequest,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)


class _ExactQueue:
    def __init__(self, request: MergeRequest) -> None:
        self.request = request

    def get(self, request_id: str) -> MergeRequest | None:
        return self.request if request_id == self.request.request_id else None


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


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
    PortalTaskState().save(daemon.state_path)
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
        "event_id": "event:implementation-finished",
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
            "changed_submodule_paths": [],
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


def _real_provider_forbidden_landed_fixture(
    tmp_path: Path,
    *,
    protected_board: bool,
) -> tuple[PortalImplementationDaemon, PortalTask, str]:
    repo = tmp_path / "real-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    if protected_board:
        todo_path = repo / "todo.md"
    else:
        (repo / ".gitignore").write_text("state/\n", encoding="utf-8")
        todo_path = repo / "state" / "runtime.todo.md"
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    todo_path.write_text(
        """# Tasks

## TEST-001 Recover an integrated completion

- Status: todo
- Completion: auto
- Priority: P0
- Track: recovery
- Outputs: feature.py
- Acceptance: Preserve exact landed evidence.
""",
        encoding="utf-8",
    )
    tracked = ["feature.py"]
    if protected_board:
        tracked.append("todo.md")
    else:
        tracked.append(".gitignore")
    _git(repo, "add", *tracked)
    _git(repo, "commit", "-m", "seed landed recovery")
    initial_commit = _git(repo, "rev-parse", "HEAD")
    state_dir = repo / "state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## TEST-",
        implementation_protected_paths=(
            ("todo.md",) if protected_board else ()
        ),
        implement=False,
    )
    PortalTaskState().save(daemon.state_path)
    [task] = daemon._load_tasks()
    identity = daemon._identity_for_task(task)
    request_id = "request-real-provider-forbidden-recovery"
    completion_task_cids = {
        task.task_id: identity.canonical_task_cid,
    }
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "task_cid": identity.canonical_task_cid,
            "canonical_task_cid": identity.canonical_task_cid,
            "canonical_task_key": identity.canonical_task_key,
            "attempt": 1,
            "branch": "implementation/test-001-attempt-1",
            "implementation_commit": initial_commit,
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
            },
            "merge_result": {
                "attempted": False,
                "merged": False,
                "queued": True,
                "reason": "merge_queued",
                "request_id": request_id,
                "canonical_task_cid": identity.canonical_task_cid,
                "canonical_task_key": identity.canonical_task_key,
                "completion_task_cids": completion_task_cids,
            },
            "cleanup_result": {"cleaned": True},
        },
    )
    request = MergeRequest(
        request_id=request_id,
        branch_name="implementation/test-001-attempt-1",
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
            "implementation_commit": initial_commit,
            "completion_task_cids": completion_task_cids,
            "changed_submodule_paths": [],
        },
        commit_sha=initial_commit,
        canonical_task_id=identity.canonical_task_cid,
        canonical_task_key=identity.canonical_task_key,
        status="quarantined",
        failure_count=1,
        failure_reason=(
            "cross_board_manual_completion_authority_metadata_invalid"
        ),
    )
    daemon.merge_queue = _ExactQueue(request)  # type: ignore[assignment]
    return daemon, task, initial_commit


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
    (
        "not_integrated",
        "wrong_commit",
        "wrong_target",
        "ordinary_failure",
        "slash_wrapped_submodule",
        "non_string_submodule",
        "dot_submodule",
        "duplicate_submodule",
        "empty_extra_completion",
        "non_string_completion",
    ),
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
    elif poison == "slash_wrapped_submodule":
        request.metadata["changed_submodule_paths"] = ["/external/pkg"]
    elif poison == "non_string_submodule":
        request.metadata["changed_submodule_paths"] = [1]
    elif poison == "dot_submodule":
        request.metadata["changed_submodule_paths"] = ["external/../pkg"]
    elif poison == "duplicate_submodule":
        request.metadata["changed_submodule_paths"] = [
            "external/pkg",
            "external/pkg",
        ]
    elif poison == "empty_extra_completion":
        request.metadata["completion_task_cids"] = {
            **request.metadata["completion_task_cids"],
            "TEST-EXTRA": "",
        }
    elif poison == "non_string_completion":
        request.metadata["completion_task_cids"] = {
            "TEST-001": 1,
        }

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


def _expected_identity(
    daemon: PortalImplementationDaemon,
    task: PortalTask,
) -> dict[str, str]:
    identity = daemon._identity_for_task(task)
    return {
        "task_id": task.task_id,
        "canonical_task_key": identity.canonical_task_key,
        "canonical_task_cid": identity.canonical_task_cid,
        "board_namespace": identity.board_namespace,
    }


@pytest.mark.parametrize("protected_board", (False, True))
def test_provider_forbidden_terminal_recovery_uses_real_markdown_transaction(
    tmp_path: Path,
    protected_board: bool,
) -> None:
    daemon, task, initial_commit = (
        _real_provider_forbidden_landed_fixture(
            tmp_path,
            protected_board=protected_board,
        )
    )
    identity = _expected_identity(daemon, task)

    first = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )
    second = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert first["reconciled"] is True
    assert first["completion_receipt_recorded"] is True
    assert second["reason"] == (
        "provider_forbidden_terminal_receipt_already_present"
    )
    assert "- Status: completed" in daemon.todo_path.read_text(
        encoding="utf-8"
    )
    assert not daemon._repo_merge_lock_path().exists()
    lifecycle = daemon._iter_merge_lifecycle_events()
    assert sum(
        event.get("type")
        == "provider_forbidden_landed_completion_prepared"
        for event in lifecycle
    ) == 1
    assert sum(
        event.get("type") == "merge_reconciled" for event in lifecycle
    ) == 1
    assert sum(
        event.get("type") == "task_completed" for event in lifecycle
    ) == 1
    if protected_board:
        assert _git(daemon.repo_root, "rev-parse", "HEAD") != initial_commit
        assert _git(
            daemon.repo_root,
            "status",
            "--porcelain",
            "--",
            "todo.md",
        ) == ""
    else:
        assert _git(daemon.repo_root, "rev-parse", "HEAD") == initial_commit


def test_provider_forbidden_terminal_recovery_rejects_target_race_before_mark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _initial_commit = (
        _real_provider_forbidden_landed_fixture(
            tmp_path,
            protected_board=False,
        )
    )
    original_mark = daemon._mark_reconciled_completion_in_todo

    def advance_then_mark(*args: object, **kwargs: object) -> dict[str, object]:
        (daemon.repo_root / "target-race.txt").write_text(
            "advanced\n",
            encoding="utf-8",
        )
        _git(daemon.repo_root, "add", "target-race.txt")
        _git(daemon.repo_root, "commit", "-m", "advance target")
        return original_mark(*args, **kwargs)

    monkeypatch.setattr(
        daemon,
        "_mark_reconciled_completion_in_todo",
        advance_then_mark,
    )

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=_expected_identity(daemon, task),
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_landed_recovery_persistence_failed"
    )
    assert "- Status: todo" in daemon.todo_path.read_text(encoding="utf-8")
    lifecycle = daemon._iter_merge_lifecycle_events()
    assert sum(
        event.get("type")
        == "provider_forbidden_landed_completion_prepared"
        for event in lifecycle
    ) == 1
    assert not any(
        event.get("type") in {"merge_reconciled", "task_completed"}
        for event in lifecycle
    )


def test_provider_forbidden_terminal_recovery_rejects_dirty_current_source(
    tmp_path: Path,
) -> None:
    daemon, task, _initial_commit = (
        _real_provider_forbidden_landed_fixture(
            tmp_path,
            protected_board=False,
        )
    )
    (daemon.repo_root / "unrelated-user-work.py").write_text(
        "DIRTY = True\n",
        encoding="utf-8",
    )

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=_expected_identity(daemon, task),
    )

    assert result["reconciled"] is False
    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_landed_recovery_source_not_clean_current"
    )
    assert "- Status: todo" in daemon.todo_path.read_text(encoding="utf-8")
    lifecycle = daemon._iter_merge_lifecycle_events()
    assert not any(
        event.get("type")
        in {
            "provider_forbidden_landed_completion_prepared",
            "merge_reconciled",
            "task_completed",
        }
        for event in lifecycle
    )


def test_provider_forbidden_terminal_recovery_repairs_receipt_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    task = replace(task, status="completed")
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    identity = _expected_identity(daemon, task)
    lifecycle: list[dict[str, object]] = []
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: list(lifecycle),
    )
    monkeypatch.setattr(
        daemon,
        "_successfully_merged_task_ids",
        lambda: {task.task_id},
    )

    def record(event_type: str, body: dict[str, object]) -> None:
        lifecycle.append({"type": event_type, **body})

    monkeypatch.setattr(daemon, "_record_event", record)
    monkeypatch.setattr(
        daemon,
        "run_once",
        lambda: pytest.fail("provider-forbidden recovery entered run_once"),
    )

    first = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )
    second = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    receipts = [
        event for event in lifecycle if event.get("type") == "task_completed"
    ]
    assert first["reconciled"] is True
    assert first["completion_receipt_recorded"] is True
    assert second["reconciled"] is True
    assert second["completion_receipt_recorded"] is False
    assert second["reason"] == "provider_forbidden_terminal_receipt_already_present"
    assert receipts == [
        {
            "type": "task_completed",
            **identity,
            "completion_receipt_repair": True,
            "reason": "missing_exact_completion_receipt",
        }
    ]


@pytest.mark.parametrize("poison", ("unadmitted_merge", "active_lane"))
def test_provider_forbidden_terminal_recovery_fails_closed_without_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    task = replace(task, status="completed")
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    identity = _expected_identity(daemon, task)
    lifecycle: list[dict[str, object]] = []
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: list(lifecycle),
    )
    monkeypatch.setattr(
        daemon,
        "_successfully_merged_task_ids",
        lambda: set(),
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda event_type, body: lifecycle.append(
            {"type": event_type, **body}
        ),
    )
    if poison == "active_lane":
        PortalTaskState(
            active_task_id=task.task_id,
            implementation_in_progress=True,
        ).save(daemon.state_path)

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert result["reconciled"] is False
    assert result["blocked"] is True
    assert not any(
        event.get("type") == "task_completed" for event in lifecycle
    )
    if poison == "active_lane":
        assert result["reason"] == (
            "provider_forbidden_terminal_recovery_lane_active"
        )
    else:
        assert result["reason"] == (
            "provider_forbidden_terminal_recovery_merge_not_admitted"
        )


def _configure_provider_forbidden_landed_recovery(
    daemon: PortalImplementationDaemon,
    task: PortalTask,
    event: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    list[PortalTask],
    list[dict[str, object]],
    dict[str, object],
]:
    monkeypatch.setattr(daemon, "_git_ref_is_ancestor", lambda *_: True)
    [candidate] = daemon._failed_merge_candidates()
    current_task = [task]
    lifecycle = [dict(event)]
    target_commit = "d" * 40
    lease = object()
    monkeypatch.setattr(
        daemon,
        "_candidate_workspace_identity",
        lambda _workspace: {
            "verified": True,
            "head": target_commit,
            "tree": "e" * 40,
            "branch": "main",
            "status_clean": True,
            "status_fingerprint": "sha256:" + "0" * 64,
            "status_bytes": 0,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_strict_dirty_worktree_paths",
        lambda _workspace: set(),
    )
    monkeypatch.setattr(daemon, "_load_tasks", lambda: list(current_task))
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: list(lifecycle),
    )
    monkeypatch.setattr(
        daemon,
        "_failed_merge_candidates",
        lambda **_kwargs: [candidate],
    )
    monkeypatch.setattr(
        daemon,
        "_acquire_checkout_mutation_lease",
        lambda **_kwargs: (lease, "acquired", None, 0.0),
    )
    monkeypatch.setattr(
        daemon,
        "_release_checkout_mutation_lease",
        lambda observed: observed is lease,
    )
    monkeypatch.setattr(
        daemon,
        "_resolved_commit_ref",
        lambda _repo, _ref: target_commit,
    )
    monkeypatch.setattr(
        daemon,
        "_immutable_integration_commit",
        lambda *_args, **kwargs: {
            "passed": True,
            "integration_ref": target_commit,
            "integration_commit": target_commit,
            "implementation_commit": kwargs["implementation_commit"],
            "target_branch": kwargs["target_branch"],
            "reasons": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_integrated_changed_submodule_proof",
        lambda **kwargs: {
            "passed": True,
            "reason": "candidate_handoff_integrated",
            "candidate_commit": kwargs["candidate_commit"],
            "target_commit": kwargs["target_commit"],
            "paths": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_declared_output_tracking_invariant",
        lambda _tasks, *, repository_ref: {
            "passed": True,
            "repository_ref": repository_ref,
        },
    )

    def complete(
        _primary: PortalTask,
        _members: list[PortalTask],
        bindings: dict[str, str],
        **_kwargs: object,
    ) -> dict[str, object]:
        current_task[0] = replace(task, status="completed")
        return {
            "updated": True,
            "durable": True,
            "updated_task_ids": [task.task_id],
            "already_completed_task_ids": [],
            "completion_receipts": [
                {
                    "task_id": task.task_id,
                    "canonical_task_cid": bindings[task.task_id],
                    "status": "succeeded",
                }
            ],
        }

    monkeypatch.setattr(daemon, "_mark_reconciled_completion_in_todo", complete)
    monkeypatch.setattr(
        daemon,
        "_reconciled_completion_persisted",
        lambda *_args, **_kwargs: {"passed": True},
    )

    def record(event_type: str, body: dict[str, object]) -> None:
        lifecycle.append(
            {
                "type": event_type,
                "event_id": f"event:recovery:{len(lifecycle)}",
                **body,
            }
        )

    monkeypatch.setattr(daemon, "_record_event", record)
    monkeypatch.setattr(
        daemon,
        "_successfully_merged_task_ids",
        lambda: (
            {task.task_id}
            if any(
                item.get("type") == "merge_reconciled"
                and item.get("resolved") is True
                for item in lifecycle
            )
            else set()
        ),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("provider-forbidden recovery invoked a forbidden effect")

    for method_name in (
        "_reconcile_failed_merges",
        "_merge_branch_to_main",
        "_merge_submodule_branches_to_main",
        "_attempt_auto_clear_reconciliation_dirt",
        "_run_implementation",
        "run_once",
    ):
        monkeypatch.setattr(daemon, method_name, forbidden)
    return current_task, lifecycle, candidate


def test_provider_forbidden_terminal_recovery_reconciles_exact_landed_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    current_task, lifecycle, _candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    identity = _expected_identity(daemon, task)

    first = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )
    second = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert first["reconciled"] is True
    assert first["completion_receipt_recorded"] is True
    assert second["reason"] == (
        "provider_forbidden_terminal_receipt_already_present"
    )
    assert current_task[0].status == "completed"
    assert [item["type"] for item in lifecycle] == [
        "implementation_finished",
        "provider_forbidden_landed_completion_prepared",
        "merge_reconciled",
        "task_completed",
    ]
    preparation = lifecycle[1]
    merge = lifecycle[2]
    assert preparation["provider_policy"] == "forbidden"
    assert preparation["changed_submodule_paths"] == []
    assert merge["provider_forbidden_recovery_key"] == (
        preparation["recovery_key"]
    )
    assert merge["provider_forbidden_preparation_event_id"] == (
        preparation["event_id"]
    )


def test_provider_forbidden_terminal_recovery_resumes_after_completion_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    current_task, lifecycle, _candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    stable_record = daemon._record_event
    failed_once = False

    def crash_after_completion(
        event_type: str,
        body: dict[str, object],
    ) -> None:
        nonlocal failed_once
        if event_type == "merge_reconciled" and not failed_once:
            failed_once = True
            raise OSError("injected post-completion crash")
        stable_record(event_type, body)

    monkeypatch.setattr(daemon, "_record_event", crash_after_completion)
    identity = _expected_identity(daemon, task)

    interrupted = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )
    recovered = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert interrupted["blocked"] is True
    assert interrupted["reason"] == (
        "provider_forbidden_terminal_recovery_candidates_invalid"
    )
    assert current_task[0].status == "completed"
    assert recovered["reconciled"] is True
    assert [item["type"] for item in lifecycle].count(
        "provider_forbidden_landed_completion_prepared"
    ) == 1
    assert [item["type"] for item in lifecycle].count("merge_reconciled") == 1
    assert [item["type"] for item in lifecycle].count("task_completed") == 1


def test_provider_forbidden_terminal_recovery_resumes_after_preparation_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    current_task, lifecycle, _candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    stable_mark = daemon._mark_reconciled_completion_in_todo
    failed_once = False

    def crash_before_completion(*args: object, **kwargs: object) -> object:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("injected post-preparation crash")
        return stable_mark(*args, **kwargs)

    monkeypatch.setattr(
        daemon,
        "_mark_reconciled_completion_in_todo",
        crash_before_completion,
    )
    identity = _expected_identity(daemon, task)

    interrupted = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )
    recovered = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert interrupted["blocked"] is True
    assert interrupted["reason"] == (
        "provider_forbidden_terminal_recovery_candidates_invalid"
    )
    assert recovered["reconciled"] is True
    assert current_task[0].status == "completed"
    assert [item["type"] for item in lifecycle].count(
        "provider_forbidden_landed_completion_prepared"
    ) == 1
    assert [item["type"] for item in lifecycle].count("merge_reconciled") == 1
    assert [item["type"] for item in lifecycle].count("task_completed") == 1


def test_provider_forbidden_terminal_recovery_revalidates_target_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    current_task, lifecycle, _candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    targets = iter(
        (
            "d" * 40,
            "e" * 40,
            "f" * 40,
            "f" * 40,
            "f" * 40,
            "f" * 40,
        )
    )
    monkeypatch.setattr(
        daemon,
        "_resolved_commit_ref",
        lambda _repo, _ref: next(targets),
    )
    identity = _expected_identity(daemon, task)

    changed = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )
    recovered = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert changed["blocked"] is True
    assert changed["reason"] == (
        "provider_forbidden_landed_recovery_target_changed"
    )
    assert current_task[0].status == "completed"
    assert recovered["reconciled"] is True
    assert [item["type"] for item in lifecycle].count(
        "provider_forbidden_landed_completion_prepared"
    ) == 1
    assert [item["type"] for item in lifecycle].count("merge_reconciled") == 1
    assert [item["type"] for item in lifecycle].count("task_completed") == 1


def test_provider_forbidden_terminal_recovery_rejects_tampered_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    _current_task, lifecycle, _candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    stable_record = daemon._record_event

    def stop_before_merge(
        event_type: str,
        body: dict[str, object],
    ) -> None:
        if event_type == "merge_reconciled":
            raise OSError("injected post-completion crash")
        stable_record(event_type, body)

    monkeypatch.setattr(daemon, "_record_event", stop_before_merge)
    identity = _expected_identity(daemon, task)
    interrupted = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )
    preparation = next(
        item
        for item in lifecycle
        if item["type"]
        == "provider_forbidden_landed_completion_prepared"
    )
    preparation["recovery_key"] = "cid:tampered"

    rejected = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert interrupted["blocked"] is True
    assert rejected["blocked"] is True
    assert rejected["reason"] == (
        "provider_forbidden_landed_recovery_preparation_invalid"
    )
    assert not any(item["type"] == "merge_reconciled" for item in lifecycle)
    assert not any(item["type"] == "task_completed" for item in lifecycle)


def test_provider_forbidden_terminal_recovery_rejects_inexact_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    _current_task, lifecycle, candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    candidate["validation_result"] = {
        "attempted": True,
        "passed": False,
    }

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=_expected_identity(daemon, task),
    )

    assert result["reconciled"] is False
    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_landed_recovery_candidate_inexact"
    )
    assert [item["type"] for item in lifecycle] == [
        "implementation_finished"
    ]


def test_provider_forbidden_terminal_recovery_requires_submodule_handoff_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    _current_task, lifecycle, _candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    monkeypatch.setattr(
        daemon,
        "_integrated_changed_submodule_proof",
        lambda **_kwargs: {
            "passed": False,
            "reason": "changed_submodule_gitlink_not_integrated",
        },
    )

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=_expected_identity(daemon, task),
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_landed_recovery_outputs_invalid"
    )
    assert [item["type"] for item in lifecycle] == [
        "implementation_finished"
    ]


def test_provider_forbidden_terminal_recovery_surfaces_release_loss_on_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    _current_task, lifecycle, _candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    monkeypatch.setattr(
        daemon,
        "_reconcile_provider_forbidden_landed_candidate_under_lease",
        lambda *_args, **_kwargs: {
            "resolved": False,
            "reason": "provider_forbidden_landed_recovery_candidate_inexact",
        },
    )
    monkeypatch.setattr(
        daemon,
        "_release_checkout_mutation_lease",
        lambda _lease: False,
    )

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=_expected_identity(daemon, task),
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_landed_recovery_lease_release_lost"
    )
    assert [item["type"] for item in lifecycle] == [
        "implementation_finished",
        "provider_forbidden_landed_completion_prepared",
    ]


def test_provider_forbidden_terminal_recovery_surfaces_release_loss_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, event, _request = _fixture(tmp_path, monkeypatch)
    _current_task, _lifecycle, _candidate = (
        _configure_provider_forbidden_landed_recovery(
            daemon,
            task,
            event,
            monkeypatch,
        )
    )
    lease = object()
    monkeypatch.setattr(
        daemon,
        "_acquire_checkout_mutation_lease",
        lambda **_kwargs: (lease, "acquired", None, 0.0),
    )
    monkeypatch.setattr(
        daemon,
        "_release_checkout_mutation_lease",
        lambda observed: False if observed is lease else True,
    )
    monkeypatch.setattr(
        daemon,
        "_reconcile_provider_forbidden_landed_candidate_under_lease",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected inner failure")
        ),
    )

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=_expected_identity(daemon, task),
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_landed_recovery_lease_release_lost"
    )


def test_provider_forbidden_terminal_recovery_rejects_manual_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    completed = replace(task, status="completed")
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [completed])
    monkeypatch.setattr(daemon, "_iter_merge_lifecycle_events", lambda: [])
    daemon.manual_completion_authority_task_ids = frozenset({task.task_id})

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=_expected_identity(daemon, completed),
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_terminal_recovery_manual_authority_required"
    )


def test_provider_forbidden_terminal_recovery_rejects_invalid_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    daemon.state_path.unlink()

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=_expected_identity(daemon, task),
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_terminal_recovery_state_invalid"
    )


def test_provider_forbidden_terminal_recovery_rejects_mismatched_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    completed = replace(task, status="completed")
    identity = _expected_identity(daemon, completed)
    lifecycle = [
        {
            "type": "task_completed",
            **identity,
            "canonical_task_cid": "cid:stale-revision",
        }
    ]
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [completed])
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: list(lifecycle),
    )
    monkeypatch.setattr(
        daemon,
        "_successfully_merged_task_ids",
        lambda: {task.task_id},
    )

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_terminal_recovery_receipt_binding_mismatch"
    )
    assert len(lifecycle) == 1


def test_provider_forbidden_terminal_recovery_rejects_replacement_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    original_identity = _expected_identity(daemon, task)
    replacement = replace(task, title="Replacement contract")
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [replacement])

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=original_identity,
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_terminal_recovery_task_changed"
    )


def test_provider_forbidden_terminal_recovery_rechecks_revision_at_receipt_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    completed = replace(task, status="completed")
    replacement = replace(completed, title="Replacement at finalization")
    identity = _expected_identity(daemon, completed)
    task_reads = iter(([completed], [replacement]))
    lifecycle: list[dict[str, object]] = []
    monkeypatch.setattr(daemon, "_load_tasks", lambda: next(task_reads))
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: list(lifecycle),
    )
    monkeypatch.setattr(
        daemon,
        "_successfully_merged_task_ids",
        lambda: {task.task_id},
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda event_type, body: lifecycle.append(
            {"type": event_type, **body}
        ),
    )

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_terminal_recovery_final_binding_changed"
    )
    assert not lifecycle


def test_provider_forbidden_terminal_recovery_rechecks_after_merge_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    completed = replace(task, status="completed")
    replacement = replace(completed, title="Replacement after merge proof")
    identity = _expected_identity(daemon, completed)
    current_task = [completed]
    lifecycle: list[dict[str, object]] = []
    merge_proof_calls = 0
    monkeypatch.setattr(daemon, "_load_tasks", lambda: list(current_task))
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: list(lifecycle),
    )

    def merged_then_replace() -> set[str]:
        nonlocal merge_proof_calls
        merge_proof_calls += 1
        result = {task.task_id}
        if merge_proof_calls == 2:
            current_task[0] = replacement
        return result

    monkeypatch.setattr(
        daemon,
        "_successfully_merged_task_ids",
        merged_then_replace,
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda event_type, body: lifecycle.append(
            {"type": event_type, **body}
        ),
    )

    result = daemon.reconcile_provider_forbidden_terminal_result(
        expected_task_identity=identity,
    )

    assert merge_proof_calls == 2
    assert result["blocked"] is True
    assert result["reason"] == (
        "provider_forbidden_terminal_recovery_final_binding_changed"
    )
    assert not lifecycle


@pytest.mark.parametrize(
    "poison",
    (
        "duplicate_receipt",
        "extra_receipt",
        "extra_completed_member",
        "non_string_completed_member",
    ),
)
def test_provider_forbidden_completion_persistence_requires_exact_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    task_cid = daemon._identity_for_task(task).canonical_task_cid
    result: dict[str, object] = {
        "updated": True,
        "updated_task_ids": [task.task_id],
        "already_completed_task_ids": [],
        "completion_receipts": [
            {
                "task_id": task.task_id,
                "canonical_task_cid": task_cid,
                "status": "succeeded",
            }
        ],
    }
    if poison == "duplicate_receipt":
        result["completion_receipts"] = [
            *result["completion_receipts"],
            dict(result["completion_receipts"][0]),
        ]
    elif poison == "extra_receipt":
        result["completion_receipts"] = [
            *result["completion_receipts"],
            {
                "task_id": "TEST-EXTRA",
                "canonical_task_cid": "cid:extra",
                "status": "succeeded",
            },
        ]
    elif poison == "extra_completed_member":
        result["already_completed_task_ids"] = ["TEST-EXTRA"]
    else:
        result["already_completed_task_ids"] = [1]
    monkeypatch.setattr(
        daemon,
        "_reconciled_completion_persisted",
        lambda *_args, **_kwargs: {"passed": True},
    )

    persistence = daemon._provider_forbidden_completion_persisted(
        result,
        {task.task_id: task_cid},
    )

    assert persistence["passed"] is False
    assert persistence["reason"] == (
        "completion_persistence_population_inexact"
    )


def test_provider_forbidden_terminal_recovery_serializes_concurrent_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, _event, _request = _fixture(tmp_path, monkeypatch)
    completed = replace(task, status="completed")
    identity = _expected_identity(daemon, completed)
    lifecycle: list[dict[str, object]] = []
    lifecycle_lock = threading.Lock()
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [completed])

    def events() -> list[dict[str, object]]:
        with lifecycle_lock:
            return list(lifecycle)

    def record(event_type: str, body: dict[str, object]) -> None:
        with lifecycle_lock:
            lifecycle.append({"type": event_type, **body})

    monkeypatch.setattr(daemon, "_iter_merge_lifecycle_events", events)
    monkeypatch.setattr(daemon, "_record_event", record)
    monkeypatch.setattr(
        daemon,
        "_successfully_merged_task_ids",
        lambda: {task.task_id},
    )
    start = threading.Barrier(3)
    results: list[dict[str, object]] = []

    def recover() -> None:
        start.wait()
        results.append(
            daemon.reconcile_provider_forbidden_terminal_result(
                expected_task_identity=identity,
            )
        )

    threads = [threading.Thread(target=recover) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert all(result["reconciled"] is True for result in results)
    assert sum(
        result.get("completion_receipt_recorded") is True
        for result in results
    ) == 1
    assert [item["type"] for item in lifecycle] == ["task_completed"]
