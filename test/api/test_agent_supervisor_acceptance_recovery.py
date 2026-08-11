from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    PortalTaskState,
    TodoImplementationDaemon,
    parse_task_file,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import (
    POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
    verify_post_merge_validation_evidence,
)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _initialize_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")


def test_no_change_acceptance_recovers_original_merged_implementation_binding(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## AUTO-124 Preserve merged implementation identity

- Status: todo
- Completion: manual
- Priority: P0
- Track: runtime
- Outputs: feature.txt
- Validation: test -f feature.txt
- Acceptance: Preserve the original implementation commit on a clean retry.
""",
        encoding="utf-8",
    )
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt", "todo.md")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-124")
    (repo / "feature.txt").write_text("implemented\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "AUTO-124: implement feature")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", "implementation/auto-124")
    merge_commit = _git(repo, "rev-parse", "HEAD")
    merge_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## AUTO-",
        worktree_submodule_paths=[],
    )
    task = parse_task_file(todo_path, task_header_prefix="## AUTO-")[0]
    daemon._register_task_identities([task])
    validation = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "pre-merge-validation-evidence@1"
        ),
        "task_id": task.task_id,
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "selection": {"scope": "pre_merge"},
        "target_commit": implementation_commit,
    }
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "attempt": 1,
            "implementation_commit": implementation_commit,
            "validation_result": validation,
            "merge_result": {"merged": True, "merge_commit": merge_commit},
        },
    )
    daemon.apply_post_merge_authoritative_acceptance(
        task,
        implementation_commit=implementation_commit,
        merge_commit=merge_commit,
        repository_tree_id=f"git-tree:{merge_tree}",
        validation_result=validation,
        model_invocation_observed=True,
    )

    recovered = daemon._recover_no_change_implementation_binding(
        task,
        merge_commit=merge_commit,
        repository_tree_id=f"git-tree:{merge_tree}",
    )

    assert recovered["recovered"] is True
    assert recovered["implementation_commit"] == implementation_commit
    assert recovered["prior_merge_commit"] == merge_commit
    assert recovered["merge_commit"] == merge_commit
    assert recovered["repository_tree_id"] == f"git-tree:{merge_tree}"
    assert recovered["validation_result"] == validation
    acceptance = daemon.apply_post_merge_authoritative_acceptance(
        task,
        implementation_commit=recovered["implementation_commit"],
        merge_commit=recovered["merge_commit"],
        repository_tree_id=recovered["repository_tree_id"],
        validation_result=recovered["validation_result"],
        gate_evidence=recovered["gate_evidence"],
        model_invocation_observed=recovered["model_invocation_observed"],
    )
    assert "implementation_commit_missing" not in acceptance["reason_codes"]
    assert acceptance.get("authoritatively_completed", False) is False
    assert set(acceptance["pending_gates"]) >= {
        "freshness",
        "semantic",
        "provider_review",
    }
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")


def test_no_change_retry_reruns_fresh_post_merge_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## AUTO-124 Revalidate merged implementation

- Status: todo
- Completion: manual
- Priority: P0
- Track: runtime
- Outputs: feature.txt
- Validation: test -f feature.txt
- Acceptance: Bind a fresh validation receipt to the current merge target.
""",
        encoding="utf-8",
    )
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt", "todo.md")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-124")
    (repo / "feature.txt").write_text("implemented\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "AUTO-124: implement feature")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", "implementation/auto-124")
    merge_commit = _git(repo, "rev-parse", "HEAD")
    merge_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    monkeypatch.setenv(
        implementation_daemon_module.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV,
        "1",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## AUTO-",
        implement=True,
        implementation_command="true",
        worktree_submodule_paths=[],
    )
    task = parse_task_file(todo_path, task_header_prefix="## AUTO-")[0]
    daemon._register_task_identities([task])
    historical_validation = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "pre-merge-validation-evidence@1"
        ),
        "task_id": task.task_id,
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "selection": {"scope": "pre_merge"},
        "target_commit": implementation_commit,
    }
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "attempt": 1,
            "implementation_commit": implementation_commit,
            "validation_result": historical_validation,
            "merge_result": {"merged": True, "merge_commit": merge_commit},
        },
    )
    daemon.apply_post_merge_authoritative_acceptance(
        task,
        implementation_commit=implementation_commit,
        merge_commit=merge_commit,
        repository_tree_id=f"git-tree:{merge_tree}",
        validation_result=historical_validation,
        model_invocation_observed=True,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=PortalTaskState(),
        attempt=2,
        started_at="2026-08-03T00:00:00+00:00",
        log_path=state_dir / "implementation.log",
        prompt="Verify the already-landed implementation.",
    )

    acceptance = result["acceptance_result"]
    assert result["implementation_binding_recovery"]["recovered"] is True
    assert acceptance["completion_authoritative"] is False
    assert acceptance["pending_gates"] == ["provider_review"]
    receipt = acceptance["receipt"]
    for gate_kind in ("freshness", "semantic"):
        evidence = receipt["gate_evidence"][gate_kind]
        assert evidence["validation_scope"] == "post_merge"
        assert evidence["passed"] is True
        assert evidence["target_commit"] == merge_commit
        assert evidence["repository_tree_id"] == f"git-tree:{merge_tree}"
        assert evidence["validation_receipt_id"]
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["type"] == "post_merge_validation_finished"
        and event["task_id"] == task.task_id
        and event["passed"] is True
        for event in events
    )


def test_no_change_acceptance_recovery_rejects_tree_and_identity_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    (repo / "feature.txt").write_text("implemented\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "AUTO-125: implementation")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    merge_commit = implementation_commit
    merge_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=[],
    )
    task = PortalTask(
        task_id="AUTO-125",
        title="Reject ambiguous recovery",
        status="todo",
        completion="manual",
        priority="P0",
        track="runtime",
        outputs=("feature.txt",),
        validation=("test -f feature.txt",),
        acceptance="Only the exact canonical task may recover authority.",
    )
    identity = daemon._identity_for_task(task)
    receipt = {
        "task_id": task.task_id,
        "implementation_commit": implementation_commit,
        "merge_commit": merge_commit,
        "repository_tree_id": f"git-tree:{merge_tree}",
        "merged": True,
        "gate_evidence": {},
    }
    monkeypatch.setattr(
        daemon,
        "_iter_events",
        lambda: [
            {
                "type": "implementation_finished",
                "task_id": task.task_id,
                "canonical_task_cid": identity.canonical_task_cid,
                "canonical_task_key": identity.canonical_task_key,
                "implementation_commit": implementation_commit,
                "validation_result": {"passed": True},
            },
            {
                "type": "implementation_finished",
                "task_id": task.task_id,
                "canonical_task_cid": "foreign-canonical-task",
                "canonical_task_key": identity.canonical_task_key,
                "implementation_commit": implementation_commit,
            },
            {
                "type": "implementation_merged_pending_acceptance",
                "task_id": task.task_id,
                "receipt": receipt,
            },
        ],
    )

    ambiguous = daemon._recover_no_change_implementation_binding(
        task,
        merge_commit=merge_commit,
        repository_tree_id=f"git-tree:{merge_tree}",
    )
    wrong_tree = daemon._validated_recovered_implementation_binding(
        task,
        implementation_commit=implementation_commit,
        prior_merge_commit=merge_commit,
        prior_repository_tree_id="git-tree:not-the-merge-tree",
        current_merge_commit=merge_commit,
        current_repository_tree_id=f"git-tree:{merge_tree}",
        validation_result={},
        gate_evidence={},
        model_invocation_observed=True,
        source="test",
        source_id="tree-mismatch",
    )

    assert ambiguous["recovered"] is False
    assert ambiguous["reason"] == "canonical_task_identity_ambiguous"
    assert wrong_tree["recovered"] is False
    assert "recovered_merge_tree_mismatch" in wrong_tree["reason_codes"]


def test_recovery_does_not_promote_persisted_provider_review_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    (repo / "feature.txt").write_text("implemented\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "AUTO-126: implementation")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    merge_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=[],
    )
    task = PortalTask(
        task_id="AUTO-126",
        title="Do not trust replayed review gates",
        status="todo",
        completion="manual",
        priority="P0",
        track="runtime",
        outputs=("feature.txt",),
        validation=("test -f feature.txt",),
        acceptance="Provider review must be independently verified.",
    )
    identity = daemon._identity_for_task(task)
    monkeypatch.setattr(
        daemon,
        "_iter_events",
        lambda: [
            {
                "type": "implementation_finished",
                "task_id": task.task_id,
                "canonical_task_cid": identity.canonical_task_cid,
                "canonical_task_key": identity.canonical_task_key,
                "implementation_commit": implementation_commit,
                "validation_result": {
                    "attempted": True,
                    "passed": True,
                    "selection": {"scope": "pre_merge"},
                },
            },
            {
                "type": "implementation_merged_pending_acceptance",
                "task_id": task.task_id,
                "implementation_commit": implementation_commit,
                "provider_review_gate_evidence": {"satisfied": True},
                "receipt": {
                    "task_id": task.task_id,
                    "implementation_commit": implementation_commit,
                    "merge_commit": implementation_commit,
                    "repository_tree_id": f"git-tree:{merge_tree}",
                    "merged": True,
                    "model_invocation_observed": True,
                    "gate_evidence": {
                        "provider_review": {
                            "satisfied": True,
                            "provider_review_receipt_id": "forged",
                        }
                    },
                },
            },
        ],
    )

    recovered = daemon._recover_no_change_implementation_binding(
        task,
        merge_commit=implementation_commit,
        repository_tree_id=f"git-tree:{merge_tree}",
    )
    acceptance = daemon.apply_post_merge_authoritative_acceptance(
        task,
        implementation_commit=recovered["implementation_commit"],
        merge_commit=recovered["merge_commit"],
        repository_tree_id=recovered["repository_tree_id"],
        validation_result=recovered["validation_result"],
        gate_evidence=recovered["gate_evidence"],
        model_invocation_observed=recovered["model_invocation_observed"],
    )

    assert recovered["recovered"] is True
    assert "provider_review" not in recovered["gate_evidence"]
    assert "provider_review_gate_evidence" not in recovered["gate_evidence"]
    assert "provider_review" in acceptance["pending_gates"]


def test_merge_reconciliation_uses_authoritative_acceptance_funnel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-008 Reconcile through authority

- Status: todo
- Completion: manual
- Priority: P0
- Track: runtime
- Outputs: reconciled.txt
- Validation: test -f reconciled.txt
- Acceptance: Reconciliation retains exact commit-bound evidence.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/accel-008")
    (repo / "reconciled.txt").write_text("landed\n", encoding="utf-8")
    _git(repo, "add", "reconciled.txt")
    _git(repo, "commit", "-m", "ACCEL-008: implementation")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", "implementation/accel-008")
    merge_commit = _git(repo, "rev-parse", "HEAD")
    merge_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        worktree_submodule_paths=[],
    )
    task = parse_task_file(todo_path, task_header_prefix="## ACCEL-")[0]
    identity = daemon._identity_for_task(task)
    validation = {
        "attempted": True,
        "passed": True,
        "selection": {"scope": "pre_merge"},
    }
    event = {
        "task_id": task.task_id,
        "canonical_task_cid": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key,
        "attempt": 2,
        "branch": "",
        "implementation_commit": implementation_commit,
        "validation_result": validation,
        "merge_result": {
            "attempted": True,
            "merged": False,
            "reason": "cleanup_failed",
        },
    }
    monkeypatch.setattr(
        daemon,
        "_failed_merge_candidates",
        lambda skip_task_ids=None: [event],
    )
    monkeypatch.setattr(
        daemon,
        "_mark_task_completed_in_todo",
        lambda *_args, **_kwargs: pytest.fail(
            "reconciliation must not bypass authoritative acceptance"
        ),
    )
    observed: dict[str, object] = {}

    def authoritative_funnel(selected_task, **kwargs):
        observed["task"] = selected_task
        observed.update(kwargs)
        return {
            "authoritatively_completed": False,
            "completion_authoritative": False,
            "reason": "authoritative_completion_not_admitted",
            "pending_gates": ["freshness", "semantic", "provider_review"],
        }

    monkeypatch.setattr(
        daemon,
        "apply_post_merge_authoritative_acceptance",
        authoritative_funnel,
    )

    result = daemon._reconcile_failed_merges()[0]

    assert observed["task"] == task
    assert observed["implementation_commit"] == implementation_commit
    assert observed["merge_commit"] == merge_commit
    assert observed["repository_tree_id"] == f"git-tree:{merge_tree}"
    post_merge_validation = observed["validation_result"]
    assert isinstance(post_merge_validation, dict)
    assert post_merge_validation["schema"] == POST_MERGE_VALIDATION_EVIDENCE_SCHEMA
    assert post_merge_validation["target_commit"] == merge_commit
    assert post_merge_validation["validated_commit"] == merge_commit
    assert post_merge_validation["repository_tree_id"] == f"git-tree:{merge_tree}"
    assert post_merge_validation["validation_scope"] == "post_merge"
    assert post_merge_validation["passed"] is True
    assert post_merge_validation["validation_result"]["force_uncached"] is True
    assert verify_post_merge_validation_evidence(
        post_merge_validation,
        expected_task_id=task.task_id,
        expected_target_commit=merge_commit,
        expected_repository_tree_id=f"git-tree:{merge_tree}",
    ) == (True, ())
    assert result["resolved"] is True
    assert result["completion_authoritative"] is False
    assert result["acceptance_result"]["acceptance_attempted"] is True
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")


def test_merge_reconciliation_without_exact_board_task_stays_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    todo_path = repo / "todo.md"
    todo_path.write_text("# No matching task\n", encoding="utf-8")
    (repo / "landed.txt").write_text("landed\n", encoding="utf-8")
    _git(repo, "add", "landed.txt", "todo.md")
    _git(repo, "commit", "-m", "landed implementation")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=[],
    )
    event = {
        "task_id": "MISSING-001",
        "attempt": 1,
        "branch": "",
        "implementation_commit": implementation_commit,
        "merge_result": {"merged": False, "reason": "cleanup_failed"},
    }
    monkeypatch.setattr(
        daemon,
        "_failed_merge_candidates",
        lambda skip_task_ids=None: [event],
    )
    monkeypatch.setattr(
        daemon,
        "apply_post_merge_authoritative_acceptance",
        lambda *_args, **_kwargs: pytest.fail(
            "acceptance requires exactly one current board task"
        ),
    )

    result = daemon._reconcile_failed_merges()[0]

    assert result["resolved"] is True
    assert result["acceptance_pending"] is True
    assert result["completion_authoritative"] is False
    assert result["acceptance_result"]["acceptance_attempted"] is False
    assert result["acceptance_result"]["reason"] == (
        "reconciliation_task_unresolved"
    )
    assert result["reason"] == "implementation_commit_already_merged"
