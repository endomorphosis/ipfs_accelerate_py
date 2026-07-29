from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import checkout_lock as checkout_lock_module
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    serialized_lock_update,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    core as core_module,
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    BACKLOG_REFINERY_AUTHOR_EMAIL,
    generated_protected_board_commit_subject,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
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
    state_path: Path | None = None,
) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=state_path or tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        implement=True,
        implementation_command="implementation-command-that-must-not-run",
        implementation_protected_paths=protected_paths,
    )


def _supervisor(
    tmp_path: Path,
    *,
    state_path: Path | None = None,
) -> PortalImplementationSupervisor:
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
    return PortalImplementationSupervisor(
        supervisor_config_from_args(
            args,
            repo_root=tmp_path,
            state_path=state_path,
        )
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


def test_shared_protected_maintenance_lease_defers_model_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    lease, guard = supervisor._acquire_protected_path_maintenance_lease()
    assert lease is not None
    assert guard["blocked"] is False
    daemon = _daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    monkeypatch.setattr(
        daemon,
        "_build_implementation_prompt",
        lambda *_args, **_kwargs: pytest.fail(
            "maintenance coordination must precede model prompt construction"
        ),
    )

    try:
        result = daemon._run_implementation(task, PortalTaskState())
    finally:
        supervisor._release_protected_path_maintenance_lease(lease)

    assert result["skipped"] is True
    assert result["reason"] == "implementation_protected_path_maintenance_active"
    assert result["backoff_seconds"] == 30
    assert daemon.task_queue.is_cooled_down(daemon._canonical_ref(task)) is True
    assert not daemon._implementation_task_claim_path(
        task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
    ).exists()


def test_shared_protected_maintenance_waits_for_active_task_claim(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    supervisor = _supervisor(tmp_path)
    task = _task(outputs=["src/example.py"])
    task_claim_path = daemon._implementation_task_claim_path(
        task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
    )
    task_claim_metadata = daemon._build_implementation_task_claim_metadata(
        task,
        1,
        "2026-07-29T00:00:00+00:00",
    )
    acquired, _reason, _existing = (
        daemon._try_acquire_implementation_task_claim(
            task_claim_path,
            task_claim_metadata,
        )
    )
    assert acquired is True

    try:
        lease, guard = supervisor._acquire_protected_path_maintenance_lease()
    finally:
        daemon._release_implementation_task_claim(
            task_claim_path,
            task_claim_metadata,
        )

    assert lease is None
    assert guard["reason"] == "shared_implementation_task_claim_active"
    assert guard["active_claims"][0]["task_id"] == task.task_id
    assert not supervisor._protected_path_maintenance_lock_path().exists()


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
        lambda *_args, **_kwargs: {"accepted": True},
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


def test_crash_snapshot_reconciliation_accepts_device_renumbering_only(
    tmp_path: Path,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("unchanged\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    daemon.worktree_root = tmp_path / "worktrees"
    workspace = daemon.worktree_root / "attempt"
    workspace.mkdir(parents=True)
    # Ephemeral workspaces must have a stable Git HEAD for protected-path fences.
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "test@example.com")
    _git(workspace, "config", "user.name", "test")
    workspace_protected = workspace / POLICY_PATH
    workspace_protected.parent.mkdir(parents=True)
    workspace_protected.write_text("unchanged\n", encoding="utf-8")
    _git(workspace, "add", POLICY_PATH)
    _git(workspace, "commit", "-m", "seed protected path")
    daemon._require_implementation_protected_snapshot(
        task=_task(outputs=["src/example.py"]),
        attempt=1,
        workspace_path=workspace,
    )
    active_path = (
        tmp_path
        / "state"
        / "implementation-protected-path-active.json"
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    assert set(active["snapshot"]) == {"shared_checkout", "workspace"}
    for scope in active["snapshot"].values():
        for identity in scope["paths"].values():
            identity["device"] += 1
    active_path.write_text(
        json.dumps(active, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is False
    assert result["reason"] == "crash_reconciliation_device_renumbered"
    assert not active_path.exists()
    assert not (
        tmp_path
        / "state"
        / "implementation-protected-path-incident.json"
    ).exists()


def test_crash_snapshot_reconciliation_rejects_device_and_inode_changes(
    tmp_path: Path,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("unchanged\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    daemon._require_implementation_protected_snapshot(
        task=_task(outputs=["src/example.py"]),
        attempt=1,
        workspace_path=tmp_path,
    )
    active_path = (
        tmp_path
        / "state"
        / "implementation-protected-path-active.json"
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    for scope in active["snapshot"].values():
        for identity in scope["paths"].values():
            identity["device"] += 1
            identity["inode"] += 1
    active_path.write_text(
        json.dumps(active, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is True
    assert result["reason"] == "implementation_protected_path_mutated"
    assert result["incident"]["mutations"][0]["change"] == "identity_changed"


def test_live_protected_path_fence_rejects_device_renumbering(
    tmp_path: Path,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("unchanged\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    before = daemon._implementation_protected_path_snapshot(tmp_path)
    for scope in before.values():
        for identity in scope["paths"].values():
            identity["device"] += 1

    violation = daemon._implementation_protected_path_violation(
        task=_task(),
        attempt=1,
        workspace_path=tmp_path,
        before=before,
    )

    assert violation["reason"] == "implementation_protected_path_mutated"
    assert violation["mutations"][0]["change"] == "identity_changed"


def test_live_protected_path_fence_rejects_same_content_replacement(
    tmp_path: Path,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("unchanged\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    before = daemon._implementation_protected_path_snapshot(tmp_path)
    replacement = protected.with_suffix(".replacement")
    replacement.write_text("unchanged\n", encoding="utf-8")
    os.replace(replacement, protected)

    violation = daemon._implementation_protected_path_violation(
        task=_task(),
        attempt=1,
        workspace_path=tmp_path,
        before=before,
    )

    assert violation["reason"] == "implementation_protected_path_mutated"
    assert violation["mutations"][0]["change"] == "identity_changed"


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


def test_ephemeral_snapshot_rejects_checkout_without_git_identity(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, _protected = _protected_git_worktree_daemon(
        tmp_path
    )
    _git(repo, "worktree", "remove", "--force", str(workspace))
    workspace.mkdir()

    with pytest.raises(
        RuntimeError,
        match="cannot establish protected-path identity",
    ):
        daemon._require_implementation_protected_snapshot(
            task=_task(outputs=["src/example.py"]),
            attempt=1,
            workspace_path=workspace,
        )

    assert not daemon._implementation_protected_active_snapshot_path().exists()
    assert not daemon._implementation_protected_incident_path().exists()
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == (
        "implementation_protected_path_snapshot_failed"
    )
    assert events[-1]["errors"][-1]["identity"]["error"] == (
        "ephemeral workspace has no stable Git HEAD"
    )


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


def test_operator_clearance_accepts_exact_shared_checkout_rollback(
    tmp_path: Path,
) -> None:
    daemon, _repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    protected.write_text("temporary operator update\n", encoding="utf-8")
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert violation["reason"] == "implementation_protected_path_mutated"

    protected.write_text("before\n", encoding="utf-8")
    cleared = daemon.clear_implementation_protected_path_incident(
        operator_note="Restored the protected controller input exactly.",
    )

    assert cleared["cleared"] is True
    assert cleared["reason"] == "operator_confirmed_shared_checkout_rollback"
    assert cleared["shared_checkout_rollback_confirmed"] is True
    receipt = json.loads(
        Path(cleared["receipt_path"]).read_text(encoding="utf-8")
    )
    proof = receipt["shared_checkout_rollback_proof"]
    assert proof["schema"] == "implementation-protected-path-rollback-proof-v1"
    assert proof["restored_paths"][POLICY_PATH]["sha256"]


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


def test_operator_clearance_can_approve_wholly_disposed_ephemeral_workspace(
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
    (workspace / POLICY_PATH).unlink()
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert {
        item["scope"] for item in violation["mutations"]
    } == {"shared_checkout", "workspace"}

    result = daemon.clear_implementation_protected_path_incident(
        approved_commits=[operator_commit],
        operator_note="Reviewed a wholly disposed managed checkout.",
        approve_disposed_ephemeral_workspace=True,
    )

    assert result["cleared"] is True
    assert result["disposed_ephemeral_workspace_approved"] is True
    receipt = json.loads(
        Path(result["receipt_path"]).read_text(encoding="utf-8")
    )
    proof = receipt["disposed_ephemeral_workspace_proof"]
    assert proof["tracked_path_count"] == proof["deleted_path_count"] == 1
    assert proof["protected_deleted_paths"] == [POLICY_PATH]


def test_operator_clearance_accepts_disposed_exact_baseline_mirror(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, _protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = _task(outputs=["src/example.py"])
    before = daemon._implementation_protected_path_snapshot(workspace)
    before["workspace"].pop("git_head")
    before["workspace"]["paths"][POLICY_PATH] = {"state": "missing"}
    daemon._persist_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
        snapshot=before,
    )

    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert {
        item["scope"] for item in violation["mutations"]
    } == {"workspace"}
    assert violation["mutations"][0]["change"] == "created"
    _git(repo, "worktree", "remove", "--force", str(workspace))

    result = daemon.clear_implementation_protected_path_incident(
        operator_note=(
            "Reviewed an invalid checkout which only mirrored the exact "
            "protected baseline."
        ),
        approve_disposed_ephemeral_workspace=True,
    )

    assert result["cleared"] is True
    assert result["reason"] == (
        "operator_approved_mirrored_ephemeral_workspace"
    )
    assert result["mirrored_ephemeral_workspace_approved"] is True
    assert result["disposed_ephemeral_workspace_approved"] is False
    receipt = json.loads(
        Path(result["receipt_path"]).read_text(encoding="utf-8")
    )
    proof = receipt["mirrored_ephemeral_workspace_proof"]
    assert proof["workspace_absent"] is True
    assert proof["workspace_unregistered"] is True
    assert proof["mirrored_protected_paths"] == [POLICY_PATH]


def test_operator_clearance_accepts_disposed_identity_only_recreation(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, _protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    protected = workspace / POLICY_PATH
    replacement = protected.with_name(f"{protected.name}.replacement")
    replacement.write_bytes(protected.read_bytes())
    replacement.chmod(protected.stat().st_mode)
    replacement.replace(protected)
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert {
        item["scope"] for item in violation["mutations"]
    } == {"workspace"}
    assert violation["mutations"][0]["change"] == "identity_changed"
    assert (
        violation["mutations"][0]["before"]["sha256"]
        == violation["mutations"][0]["after"]["sha256"]
    )
    unrelated = repo / "src" / "unrelated.py"
    unrelated.parent.mkdir(exist_ok=True)
    unrelated.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "src/unrelated.py")
    _git(
        repo,
        "-c",
        "user.name=Unrelated",
        "-c",
        "user.email=unrelated@example.invalid",
        "commit",
        "-m",
        "change an unrelated path",
    )
    _git(repo, "worktree", "remove", "--force", str(workspace))

    result = daemon.clear_implementation_protected_path_incident(
        operator_note=(
            "Reviewed an absent checkout whose protected file was recreated "
            "with identical content and metadata."
        ),
        approve_disposed_ephemeral_workspace=True,
    )

    assert result["cleared"] is True, result
    assert result["reason"] == (
        "operator_approved_mirrored_ephemeral_workspace"
    )
    receipt = json.loads(
        Path(result["receipt_path"]).read_text(encoding="utf-8")
    )
    proof = receipt["mirrored_ephemeral_workspace_proof"]
    assert proof["workspace_absent"] is True
    assert proof["workspace_unregistered"] is True
    assert proof["workspace_git_head_missing_at_snapshot"] is False
    assert proof["mutation_changes"] == ["identity_changed"]
    assert proof["mirrored_protected_paths"] == [POLICY_PATH]
    assert receipt["protected_path_history_unchanged"] is True
    assert receipt["history"] == []


def test_operator_clearance_accepts_shared_commit_and_disposed_identity_recreation(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    workspace_protected = workspace / POLICY_PATH
    replacement = workspace_protected.with_name(
        f"{workspace_protected.name}.replacement"
    )
    replacement.write_bytes(workspace_protected.read_bytes())
    replacement.chmod(workspace_protected.stat().st_mode)
    replacement.replace(workspace_protected)
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
    assert {
        item["scope"] for item in violation["mutations"]
    } == {"shared_checkout", "workspace"}
    assert {
        item["change"]
        for item in violation["mutations"]
        if item["scope"] == "workspace"
    } == {"identity_changed"}
    _git(repo, "worktree", "remove", "--force", str(workspace))

    result = daemon.clear_implementation_protected_path_incident(
        approved_commits=[operator_commit],
        operator_note=(
            "Reviewed the operator commit and the absent checkout's "
            "identity-only recreation."
        ),
        approve_disposed_ephemeral_workspace=True,
    )

    assert result["cleared"] is True, result
    assert result["reason"] == (
        "operator_approved_shared_checkout_commits_and_"
        "mirrored_ephemeral_workspace"
    )
    assert result["approved_commits"] == [operator_commit]
    receipt = json.loads(
        Path(result["receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["history"][0]["commit"] == operator_commit
    proof = receipt["mirrored_ephemeral_workspace_proof"]
    assert proof["mutation_changes"] == ["identity_changed"]
    assert proof["mirrored_protected_paths"] == [POLICY_PATH]


def test_disposed_workspace_approval_rejects_selective_protected_deletion(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    retained = repo / "src" / "retained.py"
    retained.parent.mkdir()
    retained.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "src/retained.py")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "add retained source",
    )
    _git(workspace, "merge", "--ff-only", _git(repo, "rev-parse", "HEAD"))
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
    (workspace / POLICY_PATH).unlink()
    daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )

    result = daemon.clear_implementation_protected_path_incident(
        approved_commits=[operator_commit],
        operator_note="Selective deletion must remain blocked.",
        approve_disposed_ephemeral_workspace=True,
    )

    assert result["cleared"] is False
    assert result["reason"] == "disposed_ephemeral_workspace_proof_failed"
    assert daemon._implementation_protected_incident_path().exists()


def test_auto_clears_workspace_only_protected_deletions_when_shared_intact(
    tmp_path: Path,
) -> None:
    """Ephemeral deletions of protected docs must not permanently stall lanes."""

    repo = tmp_path / "repo"
    worktrees = tmp_path / "worktrees"
    workspace = worktrees / "workspace-ephemeral"
    repo.mkdir()
    worktrees.mkdir()
    # Workspace is gone (typical after failed agent cleanup).
    protected = repo / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("authoritative\n", encoding="utf-8")

    daemon = PortalImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=repo,
        worktree_root=worktrees,
        implement=True,
        implementation_command="implementation-command-that-must-not-run",
        implementation_protected_paths=(POLICY_PATH,),
    )
    daemon._latch_implementation_protected_incident(
        {
            "reason": "implementation_protected_path_mutated",
            "task_id": "EX-001",
            "attempt": 1,
            "workspace_path": str(workspace),
            "mutations": [
                {
                    "scope": "workspace",
                    "path": POLICY_PATH,
                    "change": "deleted",
                    "before": {"state": "present"},
                    "after": {"state": "missing"},
                }
            ],
        }
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result.get("cleared") is True
    assert result.get("auto") is True
    assert result.get("blocked") is False
    assert not daemon._implementation_protected_incident_path().exists()
    assert any(
        json.loads(line)["type"]
        == "implementation_protected_path_incident_auto_cleared"
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def test_auto_clear_refuses_shared_checkout_deletions(tmp_path: Path) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    daemon._latch_implementation_protected_incident(
        {
            "reason": "implementation_protected_path_mutated",
            "task_id": "EX-001",
            "attempt": 1,
            "workspace_path": str(tmp_path / "worktrees" / "ws"),
            "mutations": [
                {
                    "scope": "shared_checkout",
                    "path": POLICY_PATH,
                    "change": "deleted",
                }
            ],
        }
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result.get("blocked") is True
    assert result.get("reason") == "implementation_protected_path_incident_latched"
    assert daemon._implementation_protected_incident_path().exists()


def test_auto_clear_refuses_shared_plan_content_changes(tmp_path: Path) -> None:
    """Shared plan/objectives content edits still require operator clearance."""

    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    daemon._latch_implementation_protected_incident(
        {
            "reason": "implementation_protected_path_mutated",
            "task_id": "EX-001",
            "attempt": 1,
            "workspace_path": str(tmp_path / "worktrees" / "ws"),
            "mutations": [
                {
                    "scope": "shared_checkout",
                    "path": POLICY_PATH,
                    "change": "content_changed",
                    "before": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "aa",
                    },
                    "after": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "bb",
                    },
                }
            ],
        }
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result.get("blocked") is True
    assert result.get("reason") == "implementation_protected_path_incident_latched"


def test_auto_clears_shared_todo_board_content_change(tmp_path: Path) -> None:
    """Supervisor-owned board rewrites must not permanently stall lanes."""

    todo_rel = "docs/architecture/example.todo.md"
    worktrees = tmp_path / "worktrees"
    workspace = worktrees / "workspace-ephemeral"
    worktrees.mkdir()
    workspace.mkdir()
    todo = tmp_path / todo_rel
    todo.parent.mkdir(parents=True)
    todo.write_text("# board\n", encoding="utf-8")

    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=worktrees,
        implement=True,
        implementation_command="implementation-command-that-must-not-run",
        implementation_protected_paths=(todo_rel,),
    )
    daemon._latch_implementation_protected_incident(
        {
            "reason": "implementation_protected_path_mutated",
            "task_id": "EX-001",
            "attempt": 2,
            "workspace_path": str(workspace),
            "mutations": [
                {
                    "scope": "shared_checkout",
                    "path": todo_rel,
                    "change": "content_changed",
                    "before": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "old",
                    },
                    "after": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "new",
                    },
                }
            ],
        }
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result.get("cleared") is True
    assert result.get("auto") is True
    assert result.get("reason") == "shared_todo_board_content_change_accepted"
    assert not daemon._implementation_protected_incident_path().exists()


def test_auto_clears_content_preserving_identity_thrash(tmp_path: Path) -> None:
    """Hardlink/nlink thrash with identical content must not stall lanes."""

    worktrees = tmp_path / "worktrees"
    workspace = worktrees / "workspace-ephemeral"
    worktrees.mkdir()
    workspace.mkdir()
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("authoritative\n", encoding="utf-8")

    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=worktrees,
        implement=True,
        implementation_command="implementation-command-that-must-not-run",
        implementation_protected_paths=(POLICY_PATH,),
    )
    daemon._latch_implementation_protected_incident(
        {
            "reason": "implementation_protected_path_mutated",
            "task_id": "EX-002",
            "attempt": 1,
            "workspace_path": str(workspace),
            "mutations": [
                {
                    "scope": "shared_checkout",
                    "path": POLICY_PATH,
                    "change": "identity_changed",
                    "before": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "same-digest",
                        "links": 1,
                    },
                    "after": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "same-digest",
                        "links": 2,
                    },
                }
            ],
        }
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result.get("cleared") is True
    assert result.get("auto") is True
    assert result.get("reason") == "content_preserving_identity_thrash_accepted"
    assert not daemon._implementation_protected_incident_path().exists()


def test_auto_clears_mixed_identity_and_todo_board_thrash(tmp_path: Path) -> None:
    """Live multi-lane pattern: identity thrash on plan + board content rewrite."""

    plan_rel = "docs/architecture/PLAN.md"
    todo_rel = "docs/architecture/board.todo.md"
    worktrees = tmp_path / "worktrees"
    workspace = worktrees / "workspace-ephemeral"
    worktrees.mkdir()
    workspace.mkdir()
    for relative, body in ((plan_rel, "# plan\n"), (todo_rel, "# board\n")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=worktrees,
        implement=True,
        implementation_command="implementation-command-that-must-not-run",
        implementation_protected_paths=(plan_rel, todo_rel),
    )
    daemon._latch_implementation_protected_incident(
        {
            "reason": "implementation_protected_path_mutated",
            "task_id": "EX-003",
            "attempt": 3,
            "workspace_path": str(workspace),
            "mutations": [
                {
                    "scope": "shared_checkout",
                    "path": plan_rel,
                    "change": "identity_changed",
                    "before": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "plan-digest",
                        "links": 1,
                    },
                    "after": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "plan-digest",
                        "links": 2,
                    },
                },
                {
                    "scope": "shared_checkout",
                    "path": todo_rel,
                    "change": "content_changed",
                    "before": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "old-board",
                    },
                    "after": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "new-board",
                    },
                },
                {
                    "scope": "workspace",
                    "path": todo_rel,
                    "change": "content_changed",
                    "before": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "ws-old",
                    },
                    "after": {
                        "state": "present",
                        "kind": "regular_file",
                        "sha256": "ws-new",
                    },
                },
            ],
        }
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result.get("cleared") is True
    assert result.get("auto") is True
    assert result.get("reason") == "protected_path_stall_auto_cleared"
    assert set(result.get("class_codes") or []) == {
        "content_preserving_identity_thrash",
        "shared_todo_board_content_change",
        "workspace_todo_board_content_change",
    }
    assert not daemon._implementation_protected_incident_path().exists()


def test_latched_incident_checkpoint_acknowledges_wake_and_stops_replay(
    tmp_path: Path,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    daemon._latch_implementation_protected_incident(
        {
            "reason": "implementation_protected_path_mutated",
            "task_id": "EX-001",
            "attempt": 1,
            "workspace_path": str(tmp_path),
            "mutations": [
                {
                    "scope": "shared_checkout",
                    "path": POLICY_PATH,
                    "change": "content_changed",
                }
            ],
        }
    )
    wake = {"kind": "policy"}
    acknowledged: list[object] = []
    daemon._pending_runtime_wake_events = [wake]
    daemon._runtime_wake_coordinator = SimpleNamespace(
        acknowledge=acknowledged.append,
    )

    first = daemon.run_once()
    event_count = sum(
        json.loads(line)["type"]
        == "implementation_protected_path_incident_blocked"
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    )
    second = daemon.run_once()

    assert first["blocked"] is True
    assert first["delta_checkpoint"]["changed"] is True
    assert acknowledged == [wake]
    assert second["blocked"] is True
    assert second["unchanged"] is True
    assert event_count == 1
    assert sum(
        json.loads(line)["type"]
        == "implementation_protected_path_incident_blocked"
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ) == 1


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
    supervisor = _supervisor(tmp_path)
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
    assert not (tmp_path / "state" / "implementation.lock").exists()


def test_supervisor_live_daemon_lock_blocks_before_maintenance_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    lock_path = tmp_path / "state" / "implementation.lock"
    lock_path.parent.mkdir(parents=True)
    daemon_metadata = {
        "kind": "implementation",
        "lease_role": "implementation_attempt",
        "pid": os.getpid(),
        "owner_script": Path(sys.argv[0]).name,
        "repo_root": str(tmp_path.resolve()),
        "state_dir": str(lock_path.parent.resolve()),
        "task_id": "EX-001",
        "started_at": "2026-07-25T00:00:00+00:00",
    }
    lock_path.write_text(
        json.dumps(daemon_metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance_under_lease",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("maintenance must not begin while the daemon owns the lease")
        ),
    )
    phases: list[str] = []

    result = supervisor._run_once_with_maintenance(
        phases.append,
        include_refill=False,
    )

    assert result["maintenance_blocked"] is True
    assert result["reason"] == "implementation_protected_path_attempt_active"
    assert result["protected_path_guard"]["lock_owner_pid"] == os.getpid()
    assert result["protected_path_guard"]["lock_owner_task_id"] == "EX-001"
    assert phases == []
    assert json.loads(lock_path.read_text(encoding="utf-8")) == daemon_metadata


def test_supervisor_maintenance_lease_is_visible_and_removed_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    lock_path = tmp_path / "state" / "implementation.lock"
    shared_lock_path = supervisor._protected_path_maintenance_lock_path()
    observed: dict[str, object] = {}

    def maintenance_body(_update_phase, *, include_refill: bool):
        observed.update(json.loads(lock_path.read_text(encoding="utf-8")))
        assert json.loads(shared_lock_path.read_text(encoding="utf-8"))[
            "lease_role"
        ] == "shared_protected_path_maintenance"
        assert include_refill is False
        assert _daemon(tmp_path)._implementation_lock_owner_is_active(observed)
        return {"stuck": False, "completed_count": 0}

    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance_under_lease",
        maintenance_body,
    )

    result = supervisor._run_once_with_maintenance(
        lambda _phase: None,
        include_refill=False,
    )

    assert result == {"stuck": False, "completed_count": 0}
    assert observed["kind"] == "implementation"
    assert observed["lease_role"] == "supervisor_maintenance"
    assert observed["pid"] == os.getpid()
    assert observed["lease_id"]
    assert not lock_path.exists()
    assert not shared_lock_path.exists()


def test_supervisor_maintenance_lease_is_removed_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    lock_path = tmp_path / "state" / "implementation.lock"
    shared_lock_path = supervisor._protected_path_maintenance_lock_path()

    def failing_maintenance(_update_phase, *, include_refill: bool):
        assert include_refill is False
        assert json.loads(lock_path.read_text(encoding="utf-8"))[
            "lease_role"
        ] == "supervisor_maintenance"
        assert shared_lock_path.exists()
        raise RuntimeError("maintenance failed")

    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance_under_lease",
        failing_maintenance,
    )

    with pytest.raises(RuntimeError, match="maintenance failed"):
        supervisor._run_once_with_maintenance(
            lambda _phase: None,
            include_refill=False,
        )

    assert not lock_path.exists()
    assert not shared_lock_path.exists()


def test_supervisor_maintenance_lease_uses_effective_state_path_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective_state_path = tmp_path / "effective-state" / "task-state.json"
    supervisor = _supervisor(tmp_path, state_path=effective_state_path)
    effective_lock_path = effective_state_path.parent / "implementation.lock"
    configured_lock_path = tmp_path / "state" / "implementation.lock"

    def maintenance_body(_update_phase, *, include_refill: bool):
        assert include_refill is False
        metadata = json.loads(effective_lock_path.read_text(encoding="utf-8"))
        assert metadata["state_path"] == str(effective_state_path.resolve())
        assert _daemon(
            tmp_path,
            state_path=effective_state_path,
        )._implementation_lock_owner_is_active(metadata)
        assert not configured_lock_path.exists()
        return {"stuck": False}

    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance_under_lease",
        maintenance_body,
    )

    result = supervisor._run_once_with_maintenance(
        lambda _phase: None,
        include_refill=False,
    )

    assert result == {"stuck": False}
    assert not effective_lock_path.exists()
    assert not configured_lock_path.exists()


def test_supervisor_does_not_unlink_lock_replaced_while_update_is_serialized(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    lock_path = tmp_path / "state" / "implementation.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("{not-json\n", encoding="utf-8")
    replacement = {
        "kind": "implementation",
        "lease_role": "implementation_attempt",
        "pid": os.getpid(),
        "owner_script": "",
        "repo_root": str(tmp_path.resolve()),
        "state_dir": str(lock_path.parent.resolve()),
        "task_id": "EX-REPLACEMENT",
        "started_at": "2026-07-25T00:00:00+00:00",
    }
    completed = threading.Event()
    result: dict[str, object] = {}

    def acquire() -> None:
        result["value"] = supervisor._acquire_implementation_maintenance_lease()
        completed.set()

    with serialized_lock_update(lock_path):
        worker = threading.Thread(target=acquire)
        worker.start()
        assert not completed.wait(timeout=0.05)
        lock_path.unlink()
        lock_path.write_text(
            json.dumps(replacement, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    worker.join(timeout=2)

    assert completed.is_set()
    lease, guard = result["value"]
    assert lease is None
    assert guard["reason"] == "implementation_protected_path_attempt_active"
    assert guard["lock_owner_task_id"] == "EX-REPLACEMENT"
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement


def test_stale_lock_cleanup_preserves_implementation_lease_protocol_files(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    implementation_lock_path = tmp_path / "state" / "implementation.lock"
    update_guard_path = (
        tmp_path / "state" / ".implementation.lock.update.lock"
    )
    generic_lock_path = tmp_path / "state" / "merge-repair.lock"
    implementation_lock_path.parent.mkdir(parents=True)
    for path in (
        implementation_lock_path,
        update_guard_path,
        generic_lock_path,
    ):
        path.write_text("stale\n", encoding="utf-8")
        os.utime(path, (1, 1))

    result = daemon._cleanup_stale_locks(max_age_seconds=1)

    assert implementation_lock_path.exists()
    assert update_guard_path.exists()
    assert not generic_lock_path.exists()
    managed = {
        item["lock_path"]
        for item in result["skipped"]
        if item.get("reason") == "managed_by_implementation_lease_protocol"
    }
    assert managed == {
        str(implementation_lock_path),
        str(update_guard_path),
    }


def test_runtime_lock_owner_accepts_python_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "implementation.lock"
    lock_path.write_text(
        json.dumps(
            {
                "kind": "implementation",
                "pid": os.getpid(),
                "owner_script": "implementation_daemon.py",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "process_args",
        lambda _pid: (
            "python -m ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_daemon"
        ),
    )

    assert supervisor_runtime.runtime_lock_owner_is_alive(lock_path)


def test_runtime_repair_serializes_implementation_lock_replacement(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lock_path = state_dir / "implementation.lock"
    lock_path.write_text("{not-json\n", encoding="utf-8")
    replacement = {
        "kind": "implementation",
        "pid": os.getpid(),
        "owner_script": "",
        "state_dir": str(state_dir.resolve()),
        "task_id": "EX-RUNTIME-REPLACEMENT",
    }
    completed = threading.Event()
    result: dict[str, object] = {}

    def repair() -> None:
        result["value"] = supervisor_runtime.repair_supervisor_runtime(
            state_dir,
            "agent",
        )
        completed.set()

    with serialized_lock_update(lock_path):
        worker = threading.Thread(target=repair)
        worker.start()
        assert not completed.wait(timeout=0.05)
        lock_path.unlink()
        lock_path.write_text(
            json.dumps(replacement, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    worker.join(timeout=2)

    assert completed.is_set()
    assert str(lock_path) not in result["value"]["removed"]
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement


def test_serialized_lock_update_has_windows_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(_fd: int, mode: int, size: int) -> None:
            calls.append((mode, size))

    monkeypatch.setattr(checkout_lock_module, "fcntl", None)
    monkeypatch.setattr(checkout_lock_module, "msvcrt", FakeMsvcrt)
    lock_path = tmp_path / "state" / "implementation.lock"

    # Resolve through the monkeypatched module so this remains isolated even
    # when another test reloads the checkout-lock module during collection.
    with checkout_lock_module.serialized_lock_update(lock_path):
        assert calls == [(FakeMsvcrt.LK_NBLCK, 1)]

    assert calls == [
        (FakeMsvcrt.LK_NBLCK, 1),
        (FakeMsvcrt.LK_UNLCK, 1),
    ]


def test_windows_pid_probe_never_calls_os_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []
    monkeypatch.setattr(core_module.sys, "platform", "win32")
    monkeypatch.setattr(
        core_module,
        "_windows_pid_alive",
        lambda pid: observed.append(pid) or True,
    )
    monkeypatch.setattr(
        core_module.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("Windows PID probes must not call os.kill")
        ),
    )

    assert core_module.pid_alive(1234)
    assert observed == [1234]


def test_windows_process_args_uses_powershell_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(core_module.sys, "platform", "win32")

    def run(command, **_kwargs):
        commands.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="python -m package.implementation_daemon\n",
        )

    monkeypatch.setattr(core_module.subprocess, "run", run)
    assert (
        core_module.process_args(1234)
        == "python -m package.implementation_daemon"
    )
    assert commands[0][0] == "powershell.exe"

    monkeypatch.setattr(
        core_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("powershell unavailable")
        ),
    )
    assert core_module.process_args(1234) == ""


def test_empty_process_command_line_keeps_live_lock_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "implementation.lock"
    lock_path.write_text(
        json.dumps(
            {
                "kind": "implementation",
                "pid": os.getpid(),
                "owner_script": "implementation_daemon.py",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor_runtime, "process_args", lambda _pid: "")

    assert supervisor_runtime.runtime_lock_owner_is_alive(lock_path)


def test_daemon_implementation_lock_publication_failure_cleans_owned_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    lock_path = tmp_path / "state" / "implementation.lock"
    metadata = daemon._build_implementation_lock_metadata(
        _task(),
        1,
        "2026-07-25T00:00:00+00:00",
    )

    def fail_publication(lock_fd: int, _metadata) -> None:
        os.close(lock_fd)
        raise OSError("simulated publication failure")

    monkeypatch.setattr(daemon, "_write_lock_metadata", fail_publication)

    with pytest.raises(OSError, match="simulated publication failure"):
        daemon._try_acquire_implementation_lock(lock_path, metadata)

    assert not lock_path.exists()


def test_daemon_implementation_lock_release_preserves_replacement(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    lock_path = tmp_path / "state" / "implementation.lock"
    metadata = daemon._build_implementation_lock_metadata(
        _task(),
        1,
        "2026-07-25T00:00:00+00:00",
    )
    acquired, reason, existing = daemon._try_acquire_implementation_lock(
        lock_path,
        metadata,
    )
    assert acquired is True
    assert reason == "acquired"
    assert existing is None

    replacement = {
        **metadata,
        "lease_id": "replacement-lease",
        "task_id": "EX-REPLACEMENT",
    }
    with serialized_lock_update(lock_path):
        lock_path.unlink()
        lock_path.write_text(
            json.dumps(replacement, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    assert daemon._release_implementation_lock(lock_path, metadata) is False
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement


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
        worktree_path.mkdir(parents=True)
        _git(worktree_path, "init")
        worktree_protected = worktree_path / POLICY_PATH
        worktree_protected.parent.mkdir(parents=True)
        worktree_protected.write_text("before\n", encoding="utf-8")
        _git(worktree_path, "add", POLICY_PATH)
        _git(
            worktree_path,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "baseline",
        )
        return _git(worktree_path, "rev-parse", "HEAD")

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
            "target_binding_schema": MERGE_TARGET_BINDING_SCHEMA,
            "target_repository_id": daemon.merge_target_repository_id,
            "target_branch": daemon.resolved_merge_target_branch,
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
