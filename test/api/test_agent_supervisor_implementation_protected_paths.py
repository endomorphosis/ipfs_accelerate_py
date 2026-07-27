from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.checkout_lock import (
    BACKLOG_REFINERY_AUTHOR_EMAIL,
    generated_protected_board_commit_subject,
)
from ipfs_accelerate_py.agent_supervisor.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    normalize_implementation_protected_paths,
    parse_args as parse_implementation_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    parse_args as parse_implementation_supervisor_args,
    supervisor_config_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    run_process_group_stream,
)


POLICY_PATH = "implementation_plan/policies/analyzer-approvals.json"


def _daemon(
    tmp_path: Path,
    *,
    protected_paths: tuple[str, ...] = (POLICY_PATH,),
) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        implement=True,
        implementation_command="implementation-command-that-must-not-run",
        implementation_protected_paths=protected_paths,
    )


def _task(
    *,
    outputs: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> PortalTask:
    return PortalTask(
        task_id="EX-001",
        title="Example implementation",
        status="ready",
        completion="manual",
        priority="P1",
        track="quality",
        outputs=list(outputs or []),
        metadata=dict(metadata or {}),
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _protected_git_worktree_daemon(
    tmp_path: Path,
) -> tuple[PortalImplementationDaemon, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    protected = repo / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    _git(repo, "add", POLICY_PATH)
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "initial",
    )

    worktree_root = tmp_path / "worktrees"
    workspace = worktree_root / "lane"
    worktree_root.mkdir()
    _git(repo, "worktree", "add", "-b", "lane", str(workspace), "HEAD")
    daemon = PortalImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command="fake-agent",
        implementation_protected_paths=(POLICY_PATH,),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    return daemon, repo, workspace, protected


def test_normalize_implementation_protected_paths_is_exact_and_fail_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "implementation_plan" / "policies").mkdir(parents=True)
    assert normalize_implementation_protected_paths(
        [f" {POLICY_PATH},docs/control.json", f"./{POLICY_PATH}"],
        repo_root=tmp_path,
    ) == (POLICY_PATH, "docs/control.json")

    for unsafe in (
        "../outside.json",
        "/tmp/outside.json",
        "C:\\outside.json",
        "docs/control/",
        "https://example.invalid/control.json",
    ):
        with pytest.raises(ValueError, match="protected path"):
            normalize_implementation_protected_paths([unsafe], repo_root=tmp_path)

    with pytest.raises(ValueError, match="not directories"):
        normalize_implementation_protected_paths(
            ["implementation_plan/policies"],
            repo_root=tmp_path,
        )
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    protected_symlink = tmp_path / "protected-link.json"
    protected_symlink.symlink_to(target.name)
    with pytest.raises(ValueError, match="must not be symlinks"):
        normalize_implementation_protected_paths(
            [protected_symlink.name],
            repo_root=tmp_path,
        )


def test_supervisor_parser_config_and_managed_command_propagate_protected_paths(
    tmp_path: Path,
) -> None:
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--implementation-protected-path",
            f"{POLICY_PATH},docs/control.json",
            "--implementation-protected-path",
            f"./{POLICY_PATH}",
        ]
    )
    config = supervisor_config_from_args(args, repo_root=tmp_path)

    assert config.implementation_protected_paths == (
        POLICY_PATH,
        "docs/control.json",
    )
    command = PortalImplementationSupervisor(config)._build_daemon_command()
    protected_values = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--implementation-protected-path"
    ]
    assert protected_values == [POLICY_PATH, "docs/control.json"]


def test_daemon_parser_and_runner_apply_default_protected_paths(
    tmp_path: Path,
) -> None:
    parsed = parse_implementation_daemon_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        parsed,
        repo_root=tmp_path,
        default_implementation_protected_paths=(POLICY_PATH,),
    )
    assert daemon.implementation_protected_paths == (POLICY_PATH,)

    explicit = parse_implementation_daemon_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--implementation-protected-path",
            f"{POLICY_PATH},docs/control.json",
        ]
    )
    explicit_daemon, _context = build_portal_implementation_daemon_from_args(
        explicit,
        repo_root=tmp_path,
        default_implementation_protected_paths=("ignored/default.json",),
    )
    assert explicit_daemon.implementation_protected_paths == (
        POLICY_PATH,
        "docs/control.json",
    )


def test_general_implementation_prompt_marks_protected_files_read_only(
    tmp_path: Path,
) -> None:
    prompt = _daemon(tmp_path)._build_implementation_prompt(
        _task(outputs=["src/example.py"]),
        attempt=1,
    )

    assert "Operator-protected repository files" in prompt
    assert f"- {POLICY_PATH}" in prompt
    assert "read-only; overrides every task" in prompt
    assert "Never create, modify, rename, delete, replace, or regenerate" in prompt


@pytest.mark.parametrize(
    ("outputs", "metadata"),
    [
        ([f"`{POLICY_PATH}` (approval baseline)"], {}),
        ([], {"predicted files": f"src/example.py, ./{POLICY_PATH}"}),
        ([], {"predicted outputs": POLICY_PATH}),
    ],
)
def test_daemon_skips_protected_declarations_before_launch(
    tmp_path: Path,
    outputs: list[str],
    metadata: dict[str, str],
) -> None:
    daemon = _daemon(tmp_path)

    def unexpected_provider_probe() -> dict[str, object]:
        raise AssertionError("provider selection must not run for a protected task")

    daemon._active_provider_capacity_backoff = unexpected_provider_probe  # type: ignore[method-assign]
    result = daemon._run_implementation(
        _task(outputs=outputs, metadata=metadata),
        PortalTaskState(),
    )

    assert result["skipped"] is True
    assert result["reason"] == "implementation_protected_path_declared"
    assert result["protected_paths"] == [POLICY_PATH]


@pytest.mark.parametrize(
    ("initial_kind", "mutation", "expected_change"),
    [
        ("file", "content", "content_changed"),
        ("file", "delete", "deleted"),
        ("missing", "create", "created"),
        ("file", "directory", "type_changed"),
        ("symlink", "symlink", "symlink_changed"),
    ],
)
def test_protected_path_identity_detects_every_mutation_class(
    tmp_path: Path,
    initial_kind: str,
    mutation: str,
    expected_change: str,
) -> None:
    protected = tmp_path / "protected.json"
    if initial_kind == "file":
        protected.write_text("before\n", encoding="utf-8")
    elif initial_kind == "symlink":
        (tmp_path / "target-a.json").write_text("a\n", encoding="utf-8")
        (tmp_path / "target-b.json").write_text("b\n", encoding="utf-8")
        protected.symlink_to("target-a.json")

    # Construct without protected-path normalization for the symlink-only
    # identity test; configured symlinks themselves are rejected.
    daemon = _daemon(tmp_path, protected_paths=())
    daemon.implementation_protected_paths = ("protected.json",)
    before = daemon._implementation_protected_path_snapshot(tmp_path)

    if mutation == "content":
        protected.write_text("after\n", encoding="utf-8")
    elif mutation == "delete":
        protected.unlink()
    elif mutation == "create":
        protected.write_text("created\n", encoding="utf-8")
    elif mutation == "directory":
        protected.unlink()
        protected.mkdir()
    elif mutation == "symlink":
        protected.unlink()
        protected.symlink_to("target-b.json")

    violation = daemon._implementation_protected_path_violation(
        task=_task(),
        attempt=1,
        workspace_path=tmp_path,
        before=before,
    )

    assert violation["reason"] == "implementation_protected_path_mutated"
    assert violation["mutations"][0]["change"] == expected_change
    assert "content" not in violation["mutations"][0]["before"]
    assert "content" not in violation["mutations"][0]["after"]


def test_undeclared_shared_checkout_mutation_fails_before_validation_or_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text('{"human_review_asserted": false}\n', encoding="utf-8")
    daemon = _daemon(tmp_path)
    validation_calls: list[str] = []
    completion_calls: list[str] = []

    def agent_runner(*_args, **_kwargs):
        protected.write_text(
            '{"human_review_asserted": true}\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(["fake-agent"], 0)

    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        agent_runner,
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_commands",
        lambda *_args, **_kwargs: validation_calls.append("validation") or {},
    )
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda *_args, **_kwargs: completion_calls.append("completion") or {},
    )

    result = daemon._run_implementation(
        _task(outputs=["src/example.py"]),
        PortalTaskState(),
    )

    assert result["returncode"] == 1
    assert result["reason"] == "implementation_protected_path_mutated"
    assert result["validation_result"]["attempted"] is False
    assert validation_calls == []
    assert completion_calls == []
    assert protected.read_text(encoding="utf-8") == (
        '{"human_review_asserted": true}\n'
    )
    assert result["protected_path_violation"]["shared_checkout_restored"] is False
    incident = json.loads(
        (
            tmp_path
            / "state"
            / "implementation-protected-path-incident.json"
        ).read_text(encoding="utf-8")
    )
    assert incident["requires_operator_clearance"] is True


def test_external_protected_update_preserves_candidate_without_consuming_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, repo, _workspace, protected = _protected_git_worktree_daemon(tmp_path)
    seeded_context_path = (
        repo / "docs" / "architecture" / "untracked-operator-context.md"
    )
    seeded_context_path.parent.mkdir(parents=True)
    seeded_context_path.write_text(
        "operator context that the implementation did not change\n",
        encoding="utf-8",
    )
    state = PortalTaskState()
    queue_outcomes: list[int] = []

    def agent_runner(*_args, **kwargs):
        candidate = Path(kwargs["cwd"]) / "src" / "candidate.py"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("VALUE = 1\n", encoding="utf-8")
        protected.write_text("operator update\n", encoding="utf-8")
        return subprocess.CompletedProcess(["fake-agent"], 0)

    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        agent_runner,
    )
    monkeypatch.setattr(
        daemon,
        "_record_task_queue_outcome",
        lambda _task, returncode, **_kwargs: queue_outcomes.append(returncode),
    )

    result = daemon._run_implementation(
        _task(outputs=["src/candidate.py"]),
        state,
    )

    preservation = result["failed_preservation_result"]
    rescue_branch = preservation["rescue_branch"]
    assert result["returncode"] == 1
    assert result["reason"] == "implementation_protected_path_mutated"
    assert result["deferred"] is True
    assert result["attempt_consumed"] is False
    assert preservation["preserved"] is True
    assert rescue_branch.endswith("-protected-path-interrupted")
    assert _git(repo, "show", f"{rescue_branch}:src/candidate.py") == "VALUE = 1"
    assert preservation["pruned_seeded_context"] == [
        "docs/architecture/untracked-operator-context.md"
    ]
    seeded_in_rescue = subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            (
                f"{rescue_branch}:"
                "docs/architecture/untracked-operator-context.md"
            ),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert seeded_in_rescue.returncode != 0
    assert state.implementation_attempts == {}
    assert state.implementation_attempts_by_cid == {}
    assert queue_outcomes == []


def test_validation_mutation_fails_before_shared_checkout_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    completion_calls: list[str] = []
    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["fake-agent"], 0),
    )
    monkeypatch.setattr(
        daemon,
        "_validate_implementation_patch",
        lambda *_args, **_kwargs: {},
    )

    def validation(*_args, **_kwargs):
        protected.write_text("changed-by-validation\n", encoding="utf-8")
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        }

    monkeypatch.setattr(daemon, "_run_validation_commands", validation)
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda *_args, **_kwargs: completion_calls.append("completion") or {},
    )

    result = daemon._run_implementation(
        _task(outputs=["src/example.py"]),
        PortalTaskState(),
    )

    assert result["returncode"] == 1
    assert result["reason"] == "implementation_protected_path_mutated"
    assert result["validation_result"]["reason"] == (
        "implementation_protected_path_mutated"
    )
    assert completion_calls == []
    assert protected.read_text(encoding="utf-8") == "changed-by-validation\n"


def test_validated_no_change_guard_rejects_disappeared_candidate() -> None:
    guard = PortalImplementationDaemon._validated_no_change_completion_guard(
        baseline_ref="baseline",
        current_head="rescued-candidate",
        expected_branch="implementation/task-attempt-1",
        current_branch="rescue/worktree/task",
        validation_result={
            "selection": {
                "changed_files": [
                    "src/feature.py",
                    "test/test_feature.py",
                ]
            }
        },
    )

    assert guard["allowed"] is False
    assert guard["reasons"] == [
        "validated_diff_disappeared",
        "head_changed_before_commit",
        "branch_changed_before_commit",
    ]


def test_validated_no_change_guard_accepts_exact_unchanged_baseline() -> None:
    guard = PortalImplementationDaemon._validated_no_change_completion_guard(
        baseline_ref="baseline",
        current_head="baseline",
        expected_branch="implementation/task-attempt-1",
        current_branch="implementation/task-attempt-1",
        validation_result={"selection": {"changed_files": []}},
    )

    assert guard["allowed"] is True
    assert guard["reasons"] == []


def test_crash_snapshot_reconciliation_blocks_before_merge_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=tmp_path,
    )
    assert before
    protected.write_text("after\n", encoding="utf-8")
    merge_calls: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_consume_one_merge_candidate",
        lambda: merge_calls.append("merge") or None,
    )

    result = daemon.run_once()

    assert result["blocked"] is True
    assert result["reason"] == "implementation_protected_path_mutated"
    assert merge_calls == []
    assert protected.read_text(encoding="utf-8") == "after\n"


def test_crash_reconciliation_accepts_missing_ephemeral_workspace_when_shared_is_unchanged(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, _protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    _git(repo, "worktree", "remove", "--force", str(workspace))

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result == {
        "blocked": False,
        "reason": "crash_reconciliation_ephemeral_workspace_missing",
        "task_id": task.task_id,
        "attempt": 1,
        "workspace_path": str(workspace),
    }
    assert not daemon._implementation_protected_active_snapshot_path().exists()
    assert not daemon._implementation_protected_incident_path().exists()


def test_crash_reconciliation_rejects_missing_ephemeral_workspace_when_shared_changed(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    _git(repo, "worktree", "remove", "--force", str(workspace))
    protected.write_text("untrusted shared mutation\n", encoding="utf-8")

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is True
    assert result["reason"] == "implementation_protected_path_mutated"
    assert result["incident"]["protected_paths"] == [POLICY_PATH]
    assert daemon._implementation_protected_incident_path().exists()


def test_ephemeral_fence_accepts_concurrent_daemon_owned_completion_commit(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    protected.write_text("completed by another lane\n", encoding="utf-8")
    _git(repo, "add", POLICY_PATH)
    _git(
        repo,
        "-c",
        "user.name=Implementation Daemon",
        "-c",
        "user.email=implementation-daemon@example.invalid",
        "commit",
        "-m",
        "EX-OTHER: mark todo completed",
    )

    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )

    assert violation == {}
    assert not daemon._implementation_protected_incident_path().exists()
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    accepted = [
        event
        for event in events
        if event["type"]
        == "implementation_protected_path_concurrent_update_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["before_head"] != accepted[0]["after_head"]
    assert accepted[0]["protected_paths"] == [POLICY_PATH]


def test_ephemeral_fence_accepts_tagged_generated_board_commit(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    protected.write_text("generated retry repair\n", encoding="utf-8")
    _git(repo, "add", POLICY_PATH)
    _git(
        repo,
        "-c",
        "user.name=Accelerator Backlog Refinery",
        "-c",
        f"user.email={BACKLOG_REFINERY_AUTHOR_EMAIL}",
        "commit",
        "-m",
        generated_protected_board_commit_subject(
            "Agent: record retry-budget guardrail outputs"
        ),
    )

    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )

    assert violation == {}
    assert not daemon._implementation_protected_incident_path().exists()


def test_ephemeral_fence_rejects_untrusted_shared_checkout_commit(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    protected.write_text("untrusted\n", encoding="utf-8")
    _git(repo, "add", POLICY_PATH)
    _git(
        repo,
        "-c",
        "user.name=Untrusted",
        "-c",
        "user.email=untrusted@example.invalid",
        "commit",
        "-m",
        "change policy",
    )

    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )

    assert violation["reason"] == "implementation_protected_path_mutated"
    assert violation["protected_paths"] == [POLICY_PATH]
    assert daemon._implementation_protected_incident_path().exists()


def test_operator_clearance_requires_exact_untrusted_commit_and_writes_receipt(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    protected.write_text("reviewed operator update\n", encoding="utf-8")
    _git(repo, "add", POLICY_PATH)
    _git(
        repo,
        "-c",
        "user.name=Operator",
        "-c",
        "user.email=operator@example.invalid",
        "commit",
        "-m",
        "update protected policy",
    )
    operator_commit = _git(repo, "rev-parse", "HEAD")
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert violation["reason"] == "implementation_protected_path_mutated"

    denied = daemon.clear_implementation_protected_path_incident(
        operator_note="Reviewed concurrent policy update.",
    )
    assert denied["cleared"] is False
    assert denied["reason"] == "operator_commit_approval_mismatch"
    assert denied["missing_approved_commits"] == [operator_commit]
    assert daemon._implementation_protected_incident_path().exists()

    cleared = daemon.clear_implementation_protected_path_incident(
        approved_commits=[operator_commit[:12]],
        operator_note="Reviewed concurrent policy update.",
    )
    assert cleared["cleared"] is True
    assert cleared["approved_commits"] == [operator_commit]
    assert not daemon._implementation_protected_incident_path().exists()
    assert not daemon._implementation_protected_active_snapshot_path().exists()
    receipt = json.loads(
        Path(cleared["receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["schema"] == "implementation-protected-path-clearance-v1"
    assert receipt["operator_note"] == "Reviewed concurrent policy update."
    assert receipt["history"][0]["trusted_generator"] is False


def test_operator_clearance_rejects_workspace_protected_path_mutation(
    tmp_path: Path,
) -> None:
    daemon, _repo, workspace, _protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    (workspace / POLICY_PATH).write_text(
        "implementation mutation\n",
        encoding="utf-8",
    )
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert "workspace" in {
        item["scope"] for item in violation["mutations"]
    }

    result = daemon.clear_implementation_protected_path_incident(
        operator_note="This must remain blocked.",
    )
    assert result["cleared"] is False
    assert result["reason"] == (
        "implementation_workspace_mutation_requires_manual_recovery"
    )
    assert daemon._implementation_protected_incident_path().exists()


def test_supervisor_commits_generated_updates_to_protected_todo_board(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text(
        """# Tasks

## EX-001 Missing dependency

- Status: ready
- Depends on: EX-999
""",
        encoding="utf-8",
    )
    _git(repo, "add", "tasks.todo.md")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(tmp_path / "state"),
            "--task-prefix",
            "EX-",
            "--implementation-protected-path",
            "tasks.todo.md",
        ]
    )
    supervisor = PortalImplementationSupervisor(
        supervisor_config_from_args(args, repo_root=repo)
    )

    findings = supervisor.record_dependency_guardrails()

    assert len(findings) == 1
    assert _git(repo, "status", "--porcelain", "--", "tasks.todo.md") == ""
    assert _git(repo, "log", "-1", "--pretty=%ae") == BACKLOG_REFINERY_AUTHOR_EMAIL
    assert _git(repo, "log", "-1", "--pretty=%s").endswith(
        "[agent-supervisor:generated-protected-board]"
    )


def test_supervisor_blocks_maintenance_while_protected_snapshot_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--implementation-protected-path",
            POLICY_PATH,
        ]
    )
    supervisor = PortalImplementationSupervisor(
        supervisor_config_from_args(args, repo_root=tmp_path)
    )
    active_path = (
        tmp_path
        / "state"
        / "implementation-protected-path-active.json"
    )
    active_path.parent.mkdir(parents=True)
    active_path.write_text('{"schema":"active"}\n', encoding="utf-8")
    monkeypatch.setattr(
        supervisor,
        "detect_stale_worktrees",
        lambda: (_ for _ in ()).throw(
            AssertionError("maintenance must stop at the protected-path guard")
        ),
    )

    result = supervisor.run_once(include_refill=False)

    assert result["maintenance_blocked"] is True
    assert result["reason"] == "implementation_protected_path_attempt_active"


def test_ephemeral_timeout_mutation_is_not_validated_committed_or_enqueued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_protected = tmp_path / POLICY_PATH
    shared_protected.parent.mkdir(parents=True)
    shared_protected.write_text("shared\n", encoding="utf-8")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        implement=True,
        implementation_command="fake-agent",
        implementation_protected_paths=(POLICY_PATH,),
        use_ephemeral_worktree=True,
        worktree_root=tmp_path / "worktrees",
    )
    calls: list[str] = []

    def seed(worktree_path: Path, _branch: str, *, task=None) -> str:
        worktree_protected = worktree_path / POLICY_PATH
        worktree_protected.parent.mkdir(parents=True)
        worktree_protected.write_text("before\n", encoding="utf-8")
        return "baseline"

    def timeout_agent(*_args, **kwargs):
        (Path(kwargs["cwd"]) / POLICY_PATH).write_text(
            "mutated\n",
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(["fake-agent"], timeout=1)

    monkeypatch.setattr(daemon, "_create_seeded_worktree", seed)
    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        timeout_agent,
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_commands",
        lambda *_args, **_kwargs: calls.append("validation") or {},
    )
    monkeypatch.setattr(
        daemon,
        "_commit_worktree_changes",
        lambda *_args, **_kwargs: calls.append("commit") or {},
    )
    monkeypatch.setattr(
        daemon,
        "_enqueue_validated_worktree",
        lambda *_args, **_kwargs: calls.append("enqueue") or {},
    )
    monkeypatch.setattr(
        daemon,
        "_cleanup_merged_worktree",
        lambda *_args, **kwargs: {
            "cleaned": True,
            "reusable": kwargs.get("reusable", True),
        },
    )
    monkeypatch.setattr(
        daemon,
        "_record_task_queue_outcome",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        daemon,
        "_record_failed_attempt_retry_context",
        lambda *_args, **_kwargs: None,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=_task(outputs=["src/example.py"]),
        state=PortalTaskState(),
        attempt=1,
        started_at="2026-07-24T00:00:00+00:00",
        log_path=tmp_path / "state" / "timeout.log",
        prompt="implement",
    )

    assert result["returncode"] == 1
    assert result["reason"] == "implementation_protected_path_mutated"
    assert calls == []
    assert result["cleanup_result"]["reusable"] is False
    assert shared_protected.read_text(encoding="utf-8") == "shared\n"


def test_merge_callback_rejects_candidate_commit_touching_protected_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()

    git("init")
    git("config", "user.name", "Protected Path Test")
    git("config", "user.email", "protected@example.invalid")
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    git("add", POLICY_PATH)
    git("commit", "-m", "baseline")
    baseline = git("rev-parse", "HEAD")
    protected.write_text("after\n", encoding="utf-8")
    git("add", POLICY_PATH)
    git("commit", "-m", "mutate protected policy")
    candidate = git("rev-parse", "HEAD")

    daemon = _daemon(tmp_path)
    monkeypatch.setattr(
        daemon,
        "_rehydrate_merge_request_branch",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("protected candidate must be rejected before merge setup")
        ),
    )
    request = SimpleNamespace(
        metadata={
            "baseline_ref": baseline,
            "implementation_commit": candidate,
            "implementation_protected_paths": [POLICY_PATH],
            "task": {
                "task_id": "EX-001",
                "title": "Unsafe candidate",
                "status": "ready",
                "completion": "manual",
                "priority": "P1",
                "track": "quality",
            },
        },
        task_id="EX-001",
        branch_name="implementation/ex-001",
        commit_sha=candidate,
        attempt=1,
        priority="P1",
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is False
    assert result["reason"] == "merge_candidate_protected_path_changed"
    assert result["protected_paths_changed"] == [POLICY_PATH]


def test_successful_agent_runner_quiesces_daemonized_descendants(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    log_path = tmp_path / "agent.log"
    child_script = "import time; time.sleep(60)"
    parent_script = (
        "import pathlib, subprocess, sys; "
        "child = subprocess.Popen("
        f"[sys.executable, '-c', {child_script!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))"
    )
    child_pid = 0
    try:
        with log_path.open("w", encoding="utf-8") as log_fh:
            result = run_process_group_stream(
                [sys.executable, "-c", parent_script],
                cwd=tmp_path,
                stdout=log_fh,
                timeout_seconds=5,
                termination_grace_seconds=0.1,
            )
        assert result.returncode == 0
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while pid_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not pid_alive(child_pid)
    finally:
        if child_pid and pid_alive(child_pid):
            try:
                subprocess.run(
                    ["kill", "-KILL", str(child_pid)],
                    check=False,
                    capture_output=True,
                )
            except OSError:
                pass
