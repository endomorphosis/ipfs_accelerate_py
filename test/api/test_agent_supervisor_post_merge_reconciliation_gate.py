from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import (
    POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
    build_post_merge_validation_evidence,
    verify_post_merge_validation_evidence,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _reconciliation_candidate(
    tmp_path: Path,
    *,
    validation: str,
    already_merged: bool,
) -> tuple[TodoImplementationDaemon, Path, Path, dict[str, object], str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "checkout", "-qb", "main")
    _git(repo, "config", "user.name", "Post-merge Gate Test")
    _git(repo, "config", "user.email", "post-merge-gate@example.invalid")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        "## PMV-101 Validate reconciled integration\n\n"
        "- Status: todo\n"
        "- Completion: manual\n"
        "- Priority: P0\n"
        "- Track: verification\n"
        "- Outputs: artifact.txt\n"
        f"- Validation: {validation}\n",
        encoding="utf-8",
    )
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt", "todo.md")
    _git(repo, "commit", "-qm", "baseline")

    branch = "implementation/pmv-101"
    _git(repo, "checkout", "-qb", branch)
    (repo / "artifact.txt").write_text("landed\n", encoding="utf-8")
    _git(repo, "add", "artifact.txt")
    _git(repo, "commit", "-qm", "PMV-101: add artifact")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    if already_merged:
        _git(repo, "merge", "--no-ff", "--no-edit", branch)

    state_dir = tmp_path / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## PMV-",
        merge_target_branch="main",
        merge_queue_dir=state_dir / "merge-queue",
        worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=False,
        worktree_submodule_paths=(),
    )
    task = daemon._load_tasks()[0]
    event: dict[str, object] = {
        "task_id": task.task_id,
        "task_cid": daemon._identity_for_task(task).canonical_task_cid,
        "attempt": 1,
        "branch": branch,
        "implementation_commit": implementation_commit,
        # This historical pre-merge verdict is deliberately insufficient.
        "validation_result": {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "selection": {"scope": "pre_merge"},
        },
        "merge_result": {
            "attempted": True,
            "merged": False,
            "reason": "merge_retry_requested",
        },
    }
    daemon._failed_merge_candidates = (  # type: ignore[method-assign]
        lambda skip_task_ids=None: [event]
    )
    return daemon, repo, todo_path, event, implementation_commit


def test_normal_reconciliation_validates_immutable_target_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, repo, todo_path, _event, implementation_commit = (
        _reconciliation_candidate(
            tmp_path,
            validation="test -f artifact.txt",
            already_merged=False,
        )
    )
    monkeypatch.setattr(
        daemon,
        "_proof_workflow_options",
        lambda *_args, **_kwargs: pytest.fail(
            "post-merge validation must not dispatch proof/provider work"
        ),
    )

    result = daemon._reconcile_failed_merges()[0]

    target_commit = str(result["integration_commit_proof"]["integration_commit"])
    target_tree_id = (
        f"git-tree:{_git(repo, 'rev-parse', f'{target_commit}^{{tree}}')}"
    )
    evidence = result["post_merge_validation"]
    assert result["resolved"] is True
    assert result["reason"] == "merge_retried"
    assert result["integration_commit_proof"]["passed"] is True
    assert result["post_merge_declared_output_invariant"]["passed"] is True
    assert evidence["schema"] == POST_MERGE_VALIDATION_EVIDENCE_SCHEMA
    assert evidence["passed"] is True
    assert evidence["target_commit"] == target_commit
    assert evidence["validation_result"]["force_uncached"] is True
    assert verify_post_merge_validation_evidence(
        evidence,
        expected_task_id="PMV-101",
        expected_target_commit=target_commit,
        expected_repository_tree_id=target_tree_id,
    ) == (True, ())
    assert _git(
        repo,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        "main",
    ) == ""
    assert "- Status: completed" in todo_path.read_text(encoding="utf-8")


def test_real_merge_queue_terminally_settles_integrated_validation_failure(
    tmp_path: Path,
) -> None:
    daemon, repo, todo_path, _event, implementation_commit = (
        _reconciliation_candidate(
            tmp_path,
            validation="false",
            already_merged=False,
        )
    )
    task = daemon._load_tasks()[0]
    baseline_ref = _git(
        repo,
        "merge-base",
        implementation_commit,
        "main",
    )
    request, _queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/pmv-101",
        implementation_commit=implementation_commit,
        baseline_ref=baseline_ref,
        worktree_path=None,
        task=task,
        attempt=1,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
            "selection": {"scope": "pre_merge"},
        },
    )

    result = daemon._consume_one_merge_candidate()

    assert result is not None
    assert result["status"] == "integrated_pending_validation"
    assert result["merged"] is True
    assert result["integrated"] is True
    assert result["accepted"] is False
    assert result["acceptance_pending"] is True
    assert result["completion_authoritative"] is False
    assert result["integration_terminal"] is True
    assert result["queue_settlement"] == {
        "status": "completed",
        "terminal": True,
    }
    evidence = result["post_merge_validation"]
    assert evidence["schema"] == POST_MERGE_VALIDATION_EVIDENCE_SCHEMA
    assert evidence["attempted"] is True
    assert evidence["passed"] is False
    stored = daemon.merge_queue.get(request.request_id)
    assert stored is not None
    assert stored.status == "completed"
    assert stored.metadata["completion"]["integrated"] is True
    assert stored.metadata["completion"]["accepted"] is False
    assert daemon.merge_queue.pending_count() == 0
    assert daemon._consume_one_merge_candidate() is None
    assert _git(
        repo,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        "main",
    ) == ""
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")


def test_implementation_prompt_warns_that_validation_replays_after_merge(
    tmp_path: Path,
) -> None:
    daemon, _repo, _todo_path, _event, _implementation_commit = (
        _reconciliation_candidate(
            tmp_path,
            validation="test -f artifact.txt",
            already_merged=False,
        )
    )
    task = daemon._load_tasks()[0]

    context = daemon._compile_implementation_context(task, attempt=1)

    rules = context.capsule.authority["generic_prompt_policy"]
    assert any(
        "uncached in a clean checkout of the immutable merged target" in rule
        and "do not assert pre-commit HEAD or dirty-overlay equality" in rule
        for rule in rules
    )


def test_reconciliation_target_cas_rejects_advance_after_validation_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, repo, todo_path, _event, implementation_commit = (
        _reconciliation_candidate(
            tmp_path,
            validation="test -f artifact.txt",
            already_merged=False,
        )
    )
    real_locked_taskboard = implementation_daemon_module.locked_taskboard
    advanced_commit = ""

    @contextmanager
    def advance_target_before_board_mutation(path: Path):
        nonlocal advanced_commit
        if not advanced_commit:
            (repo / "advanced.txt").write_text(
                "advanced after validation\n",
                encoding="utf-8",
            )
            _git(repo, "add", "advanced.txt")
            _git(repo, "commit", "-qm", "advance target before completion")
            advanced_commit = _git(repo, "rev-parse", "HEAD")
        with real_locked_taskboard(path) as taskboard:
            yield taskboard

    monkeypatch.setattr(
        implementation_daemon_module,
        "locked_taskboard",
        advance_target_before_board_mutation,
    )

    result = daemon._reconcile_failed_merges()[0]

    integration_commit = str(
        result["integration_commit_proof"]["integration_commit"]
    )
    assert result["resolved"] is False
    assert result["reason"] == "post_merge_validation_stale"
    assert result["integration_occurred"] is True
    assert result["completion_skipped"] is True
    assert result["post_merge_validation_gate"][
        "publication_cas_passed"
    ] is False
    assert result["todo_update_result"]["updated"] is False
    assert result["todo_update_result"]["reason"] == (
        "manual_completion_authority_target_changed"
    )
    assert result["todo_update_result"]["expected_target_commit"] == (
        integration_commit
    )
    assert result["todo_update_result"]["actual_target_commit"] == (
        advanced_commit
    )
    assert advanced_commit != integration_commit
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")
    assert _git(
        repo,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        "main",
    ) == ""


def test_reconciliation_preserves_raw_manual_authority_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, repo, _todo_path, _event, _implementation_commit = (
        _reconciliation_candidate(
            tmp_path,
            validation="test -f artifact.txt",
            already_merged=True,
        )
    )
    [task] = daemon._load_tasks()
    raw_authority = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "results": [],
        "manual_completion_authority_context_id": "authority:fresh",
        "manual_completion_authority_revalidation": True,
        "manual_completion_authority_force_uncached": True,
        "manual_completion_authority_task_id": task.task_id,
        "manual_completion_authority_task_cid": "task:fresh",
    }
    observed: dict[str, object] = {}

    def exact_validation(selected_task, **kwargs):
        assert selected_task.task_id == task.task_id
        return build_post_merge_validation_evidence(
            task_id=task.task_id,
            target_commit=kwargs["target_commit"],
            repository_tree_id=kwargs["repository_tree_id"],
            validation_result=raw_authority,
        )

    def mark_completion(*_args, **kwargs):
        observed.update(kwargs)
        return {
            "updated": True,
            "updated_task_ids": [task.task_id],
            "completion_receipts": [],
        }

    monkeypatch.setattr(
        daemon,
        "_validate_exact_post_merge_commit",
        exact_validation,
    )
    monkeypatch.setattr(
        daemon,
        "_mark_reconciled_completion_in_todo",
        mark_completion,
    )
    monkeypatch.setattr(
        daemon,
        "_reconciled_completion_persisted",
        lambda *_args, **_kwargs: {"passed": True},
    )

    [result] = daemon._reconcile_failed_merges()

    assert result["resolved"] is True
    assert observed["validation_evidence"] == raw_authority
    assert observed["validation_evidence"] is not result[
        "post_merge_validation"
    ]
    assert observed["expected_target_commit"] == _git(
        repo,
        "rev-parse",
        "main",
    )
    assert "_manual_completion_authority_evidence" not in result[
        "post_merge_validation_gate"
    ]


def test_merge_callback_target_cas_settles_landed_integration_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, repo, todo_path, _event, implementation_commit = (
        _reconciliation_candidate(
            tmp_path,
            validation="test -f artifact.txt",
            already_merged=False,
        )
    )
    task = daemon._load_tasks()[0]
    baseline_ref = _git(
        repo,
        "merge-base",
        implementation_commit,
        "main",
    )
    request, _queued = daemon._enqueue_merge_candidate(
        branch_name="implementation/pmv-101",
        implementation_commit=implementation_commit,
        baseline_ref=baseline_ref,
        worktree_path=None,
        task=task,
        attempt=1,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
            "selection": {"scope": "pre_merge"},
        },
    )
    real_locked_taskboard = implementation_daemon_module.locked_taskboard
    advanced_commit = ""

    @contextmanager
    def advance_target_before_board_mutation(path: Path):
        nonlocal advanced_commit
        if not advanced_commit:
            (repo / "advanced.txt").write_text(
                "advanced after callback validation\n",
                encoding="utf-8",
            )
            _git(repo, "add", "advanced.txt")
            _git(repo, "commit", "-qm", "advance callback target")
            advanced_commit = _git(repo, "rev-parse", "HEAD")
        with real_locked_taskboard(path) as taskboard:
            yield taskboard

    monkeypatch.setattr(
        implementation_daemon_module,
        "locked_taskboard",
        advance_target_before_board_mutation,
    )

    result = daemon._consume_one_merge_candidate()

    assert result is not None
    assert result["status"] == "integrated_pending_validation"
    assert result["integrated"] is True
    assert result["accepted"] is False
    assert result["queue_settlement"] == {
        "status": "completed",
        "terminal": True,
    }
    callback_result = result["merge_result"]
    assert callback_result["reason"] == "post_merge_validation_stale"
    assert callback_result["integration_occurred"] is True
    assert callback_result["completion_skipped"] is True
    assert callback_result["post_merge_validation_gate"][
        "publication_cas_passed"
    ] is False
    assert callback_result["todo_update_result"]["reason"] == (
        "manual_completion_authority_target_changed"
    )
    assert callback_result["todo_update_result"][
        "actual_target_commit"
    ] == advanced_commit
    assert daemon.merge_queue.get(request.request_id).status == "completed"
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")
    assert _git(
        repo,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        "main",
    ) == ""


def test_failed_integrated_candidate_is_terminal_and_fresh_attempt_completes(
    tmp_path: Path,
) -> None:
    daemon, repo, todo_path, event, implementation_commit = (
        _reconciliation_candidate(
            tmp_path,
            validation="grep -qx good artifact.txt",
            already_merged=True,
        )
    )
    daemon._record_event("implementation_finished", event)

    first = daemon._reconcile_failed_merges()[0]

    assert first["resolved"] is False
    assert first["reason"] == "post_merge_validation_failed"
    assert first["integration_occurred"] is True
    assert first["completion_skipped"] is True
    del daemon.__dict__["_failed_merge_candidates"]
    assert daemon._failed_merge_candidates() == []
    assert daemon._unresolved_merge_failures_by_task() == {}
    assert daemon._reconcile_failed_merges() == []
    daemon.implement = False
    projection = daemon.run_once()
    assert projection["active_task_id"] == "PMV-101"
    assert projection["selectable_ready_count"] == 1
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")

    baseline_ref = _git(repo, "rev-parse", "main")
    corrected_branch = "implementation/pmv-101-corrected"
    _git(repo, "checkout", "-qb", corrected_branch, "main")
    (repo / "artifact.txt").write_text("good\n", encoding="utf-8")
    _git(repo, "add", "artifact.txt")
    _git(repo, "commit", "-qm", "PMV-101: correct merged artifact")
    corrected_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    task = daemon._load_tasks()[0]
    request, _queued = daemon._enqueue_merge_candidate(
        branch_name=corrected_branch,
        implementation_commit=corrected_commit,
        baseline_ref=baseline_ref,
        worktree_path=None,
        task=task,
        attempt=2,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
            "selection": {"scope": "pre_merge"},
        },
    )

    corrected = daemon._consume_one_merge_candidate()

    assert corrected is not None
    assert corrected["status"] == "merged"
    assert corrected["integrated"] is True
    assert corrected["accepted"] is True
    assert corrected["acceptance_pending"] is False
    assert corrected["post_merge_validation"]["passed"] is True
    stored = daemon.merge_queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"
    assert stored.metadata["completion"]["acceptance_receipt_id"].startswith(
        "sha256:"
    )
    assert "- Status: completed" in todo_path.read_text(encoding="utf-8")
    assert (repo / "artifact.txt").read_text(encoding="utf-8") == "good\n"
    assert _git(
        repo,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        "main",
    ) == ""


@pytest.mark.parametrize(
    "failure_mode,validation",
    (
        ("failed", "false"),
        (
            "self_invalidating",
            "test -f artifact.txt && rm artifact.txt",
        ),
        ("missing", "test -f artifact.txt"),
    ),
)
def test_already_merged_reconciliation_never_completes_without_fresh_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    validation: str,
) -> None:
    daemon, repo, todo_path, _event, implementation_commit = (
        _reconciliation_candidate(
            tmp_path,
            validation=validation,
            already_merged=True,
        )
    )
    if failure_mode == "missing":
        monkeypatch.setattr(
            daemon,
            "_validate_exact_post_merge_commit",
            lambda *_args, **_kwargs: {},
        )
    monkeypatch.setattr(
        daemon,
        "_mark_reconciled_completion_in_todo",
        lambda *_args, **_kwargs: pytest.fail(
            "failed, stale, or missing evidence must not mutate the board"
        ),
    )

    result = daemon._reconcile_failed_merges()[0]

    assert result["resolved"] is False
    assert result["reason"] == "post_merge_validation_failed"
    assert result["integration_occurred"] is True
    assert result["completion_skipped"] is True
    assert result["post_merge_validation_gate"]["passed"] is False
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")
    assert (repo / "artifact.txt").read_text(encoding="utf-8") == "landed\n"
    assert _git(
        repo,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        "main",
    ) == ""
    if failure_mode == "self_invalidating":
        evidence = result["post_merge_validation"]
        assert evidence["passed"] is False
        assert evidence["stale"] is True
        assert evidence["validation_result"]["reason"] == (
            "post_merge_validation_workspace_dirty_after_execution"
        )
        assert evidence["validation_result"]["force_uncached"] is True
    elif failure_mode == "failed":
        evidence = result["post_merge_validation"]
        assert evidence["attempted"] is True
        assert evidence["passed"] is False
        assert evidence["stale"] is False
    else:
        assert result["post_merge_validation"] == {}
        assert "post_merge_validation_evidence_missing" in result[
            "post_merge_validation_gate"
        ]["reasons"]
