from __future__ import annotations

from dataclasses import replace
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
    BACKLOG_REFINERY_AUTHOR_EMAIL,
    CheckoutMaintenanceLease,
    checkout_lock_metadata,
    checkout_mutation_lock_path,
    crash_fence_reconciliation_lock_path,
    durable_input_generation,
    generated_protected_board_commit_subject,
    generations_match,
    serialized_lock_update,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    core as core_module,
    implementation_daemon as implementation_daemon_module,
    implementation_supervisor as implementation_supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    CrashFenceReconciler,
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
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    ProcessBirthIdentity,
    WorkspaceLifecycleState,
)


POLICY_PATH = "implementation_plan/policies/analyzer-approvals.json"


@pytest.fixture(autouse=True)
def _relax_task_execution_metadata_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit fixtures omit reviewed provider roles; clear the supervisor env gate.

    The live implementation daemon sets
    ``IPFS_ACCELERATE_AGENT_REQUIRE_TASK_EXECUTION_METADATA`` for production
    lanes. Protected-path unit tests must remain independent of that host
    policy so lease and fence regressions stay isolated.
    """

    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_REQUIRE_TASK_EXECUTION_METADATA",
        raising=False,
    )


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


def _generated_protected_supervisor(
    tmp_path: Path,
) -> tuple[PortalImplementationSupervisor, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    _git(repo, "add", "tasks.todo.md")
    _git(repo, "commit", "-m", "initial")
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(tmp_path / "state"),
            "--implementation-protected-path",
            "tasks.todo.md",
        ]
    )
    return (
        PortalImplementationSupervisor(
            supervisor_config_from_args(args, repo_root=repo)
        ),
        repo,
        todo_path,
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


def _temporary_shared_merge(
    repo: Path,
    protected: Path,
    *,
    protected_content: str | None = None,
) -> tuple[str, str]:
    """Advance the shared branch through a merge that a caller can roll back."""

    base = _git(repo, "rev-parse", "HEAD")
    shared_branch = _git(repo, "branch", "--show-current")
    _git(repo, "checkout", "-b", "temporary-sibling")
    sibling_source = repo / "src" / "temporary_sibling.py"
    sibling_source.parent.mkdir(parents=True)
    sibling_source.write_text("VALUE = 'temporary'\n", encoding="utf-8")
    _git(repo, "add", "src/temporary_sibling.py")
    if protected_content is not None:
        protected.write_text(protected_content, encoding="utf-8")
        _git(repo, "add", POLICY_PATH)
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "temporary sibling implementation",
    )
    _git(repo, "checkout", shared_branch)
    _git(
        repo,
        "-c",
        "user.name=Implementation Daemon",
        "-c",
        "user.email=implementation-daemon@example.invalid",
        "merge",
        "--no-ff",
        "temporary-sibling",
        "-m",
        "temporary sibling merge",
    )
    return base, _git(repo, "rev-parse", "HEAD")


def _persist_active_attempt_state(
    daemon: PortalImplementationDaemon,
    *,
    task: PortalTask,
    workspace: Path,
    attempt: int = 1,
) -> PortalTaskState:
    identity = daemon._identity_for_task(task)
    state = PortalTaskState()
    state.implementation_in_progress = True
    state.active_task_id = task.task_id
    state.active_task_key = identity.canonical_task_key
    state.active_task_cid = identity.canonical_task_cid
    state.active_task_title = task.title
    state.active_task_track = task.track
    state.active_attempt = attempt
    state.active_phase = "implementing"
    state.active_worktree_path = str(workspace)
    state.active_branch = "lane"
    state.last_implementation_task_id = task.task_id
    state.last_implementation_task_key = identity.canonical_task_key
    state.last_implementation_task_cid = identity.canonical_task_cid
    state.save(daemon.state_path)
    return state


def _quiesced_terminal_task_claim(
    tmp_path: Path,
    *,
    task_status: str = "completed",
    live_claim_owner: bool = False,
) -> tuple[
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    Path,
    dict[str, object],
]:
    """Model the DQP shutdown gap: clean lane, terminal worktree, stale claim."""

    daemon, _repo, workspace, _protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = replace(_task(outputs=["src/example.py"]), status=task_status)
    identity = daemon._identity_for_task(task)
    attempt = 4
    started_at = "2026-08-09T00:15:25+00:00"
    state = PortalTaskState(
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_started_at=started_at,
        last_implementation_worktree_path=str(workspace),
        last_implementation_branch="lane",
        task_identities={task.task_id: identity.to_dict()},
        implementation_attempts={task.task_id: attempt},
        implementation_attempts_by_cid={identity.canonical_task_cid: attempt},
        task_statuses={task.task_id: task_status},
    )
    state.save(daemon.state_path)
    daemon._load_tasks = lambda: [task]  # type: ignore[method-assign]

    dead_owner = ProcessBirthIdentity(
        pid=2**30 - 73,
        start_time_ticks=1,
        boot_id="provably-dead-worktree-owner",
    )
    lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
        attempt=attempt,
        lane_id="terminated-lane",
        workspace_path=workspace,
        branch="lane",
        merge_target="main",
        state_dir=str(daemon.state_path.parent.resolve()),
        owner=dead_owner,
    )
    lifecycle = daemon.worktree_lifecycle.mark_active(
        workspace,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    terminal = daemon.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
        workspace,
        expected_state_dir=str(daemon.state_path.parent.resolve()),
        reason="controlled_restart_dead_owner",
    )
    assert terminal is not None
    assert terminal.state is WorkspaceLifecycleState.TERMINAL

    claim_path = daemon._implementation_task_claim_path(
        task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
    )
    claim = daemon._build_implementation_task_claim_metadata(
        task,
        attempt,
        started_at,
    )
    if live_claim_owner:
        assert claim["pid"] == os.getpid()
    else:
        # Exercise the dead PID binding used by the live DQP claim.
        claim["pid"] = 2**30 - 79
    acquired, _reason, _existing = (
        daemon._try_acquire_implementation_task_claim(claim_path, claim)
    )
    assert acquired is True
    return daemon, task, state, claim_path, claim


def _record_exact_unfinished_attempt_recovery(
    daemon: PortalImplementationDaemon,
    task: PortalTask,
    state: PortalTaskState,
    *,
    include_start: bool = True,
    recovery_branch: str | None = None,
    later_event_type: str = "",
) -> None:
    identity = daemon._identity_for_task(task)
    common = {
        "task_id": task.task_id,
        "canonical_task_key": identity.canonical_task_key,
        "canonical_task_cid": identity.canonical_task_cid,
        "board_namespace": identity.board_namespace,
        "attempt": 4,
    }
    if include_start:
        daemon._record_event(
            "implementation_started",
            {
                **common,
                "timestamp": "2026-08-09T00:15:26+00:00",
                "worktree_path": state.last_implementation_worktree_path,
                "branch": state.last_implementation_branch,
            },
        )
    daemon._record_event(
        "implementation_state_recovered",
        {
            **common,
            "timestamp": "2026-08-09T00:15:27+00:00",
            "reason": "inflight_process_missing",
            "worktree_path": state.last_implementation_worktree_path,
            "branch": (
                state.last_implementation_branch
                if recovery_branch is None
                else recovery_branch
            ),
            "attempt_recovery": {
                "task_id": task.task_id,
                "canonical_task_cid": identity.canonical_task_cid,
                "attempt": 4,
                "previous_display_count": 4,
                "previous_cid_count": 4,
                "released": True,
                "released_to": 3,
                "consumed": False,
            },
            "finished_attempt": False,
        },
    )
    if later_event_type:
        daemon._record_event(
            later_event_type,
            {
                **common,
                "timestamp": "2026-08-09T00:15:28+00:00",
                "worktree_path": state.last_implementation_worktree_path,
                "branch": state.last_implementation_branch,
            },
        )


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


def test_repo_global_maintenance_lease_defers_daemon_without_local_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    lease, guard = supervisor._acquire_protected_path_maintenance_lease()
    assert lease is not None
    assert guard["blocked"] is False
    daemon = _daemon(tmp_path, protected_paths=())
    task = _task(outputs=["src/example.py"])
    monkeypatch.setattr(
        daemon,
        "_build_implementation_prompt",
        lambda *_args, **_kwargs: pytest.fail(
            "the repo-global lease must precede prompt construction"
        ),
    )

    try:
        result = daemon._run_implementation(task, PortalTaskState())
    finally:
        supervisor._release_protected_path_maintenance_lease(lease)

    assert result["skipped"] is True
    assert result["reason"] == "implementation_protected_path_maintenance_active"
    assert daemon.task_queue.is_cooled_down(daemon._canonical_ref(task)) is True


def test_live_shared_maintenance_lease_survives_empty_process_command_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _supervisor(
        tmp_path,
        state_path=tmp_path / "lane-owner" / "task-state.json",
    )
    contender = _supervisor(
        tmp_path,
        state_path=tmp_path / "lane-contender" / "task-state.json",
    )
    lease, guard = owner._acquire_protected_path_maintenance_lease()
    assert lease is not None
    assert guard["blocked"] is False
    lock_path = owner._protected_path_maintenance_lock_path()
    monkeypatch.setattr(
        implementation_supervisor_module,
        "process_command_line",
        lambda _pid: "",
    )

    try:
        contender_lease, contender_guard = (
            contender._acquire_protected_path_maintenance_lease()
        )
        persisted = json.loads(lock_path.read_text(encoding="utf-8"))
    finally:
        owner._release_protected_path_maintenance_lease(lease)

    assert contender_lease is None
    assert contender_guard["reason"] == "protected_path_maintenance_active"
    assert persisted["lease_id"] == lease["lease_id"]


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
    "fence_filename",
    [
        implementation_daemon_module.IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME,
        implementation_daemon_module.IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME,
    ],
)
def test_shared_maintenance_waits_for_orphan_task_claim_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fence_filename: str,
) -> None:
    daemon = _daemon(
        tmp_path,
        state_path=tmp_path / "lane-worker" / "task-state.json",
    )
    supervisor = _supervisor(
        tmp_path,
        state_path=tmp_path / "lane-maintenance" / "task-state.json",
    )
    task = _task(outputs=["src/example.py"])
    claim_path = daemon._implementation_task_claim_path(
        task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
    )
    claim_metadata = daemon._build_implementation_task_claim_metadata(
        task,
        1,
        "2026-07-29T00:00:00+00:00",
    )
    acquired, _reason, _existing = (
        daemon._try_acquire_implementation_task_claim(
            claim_path,
            claim_metadata,
        )
    )
    assert acquired is True
    fence_path = daemon.state_path.parent / fence_filename
    fence_path.parent.mkdir(parents=True, exist_ok=True)
    fence_path.write_text('{"schema":"test-fence"}\n', encoding="utf-8")
    # The attempt finalizer must keep the repo-wide pointer to its durable
    # lane-local safety fence.
    assert daemon._release_implementation_task_claim(
        claim_path,
        claim_metadata,
    )
    assert claim_path.exists()
    monkeypatch.setattr(
        implementation_supervisor_module,
        "process_is_running",
        lambda _pid: False,
    )

    blocked_lease, blocked_guard = (
        supervisor._acquire_protected_path_maintenance_lease()
    )

    assert blocked_lease is None
    assert blocked_guard["reason"] == "shared_implementation_task_claim_active"
    assert blocked_guard["active_claims"][0]["owner_live"] is False
    assert blocked_guard["active_claims"][0]["protected_fence_paths"] == [
        str(fence_path)
    ]
    assert claim_path.exists()
    assert fence_path.exists()

    fence_path.unlink()
    assert daemon._release_implementation_task_claim(
        claim_path,
        claim_metadata,
    )
    assert not claim_path.exists()
    released_lease, released_guard = (
        supervisor._acquire_protected_path_maintenance_lease()
    )
    try:
        assert released_lease is not None
        assert released_guard["blocked"] is False
    finally:
        if released_lease is not None:
            supervisor._release_protected_path_maintenance_lease(
                released_lease
            )


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
        require_no_change_policy_gate=False,
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
        require_no_change_policy_gate=False,
    )

    assert guard["allowed"] is True
    assert guard["reasons"] == []


def test_no_change_policy_gate_is_universal_across_execution_modes() -> None:
    def task(mode: str | None) -> PortalTask:
        metadata = {"Provider role": "deterministic-only"}
        if mode is not None:
            metadata["No-change completion"] = mode
        return PortalTask(
            task_id="AUTO-DETERMINISTIC",
            title="Run a typed local validation",
            status="todo",
            completion="auto",
            priority="P0",
            track="ops",
            metadata=metadata,
        )

    assert PortalImplementationDaemon._no_change_policy_gate_required(
        task("allowed"),
        deterministic_only=True,
    ) is True
    assert PortalImplementationDaemon._no_change_policy_gate_required(
        task(None),
        deterministic_only=True,
    ) is True
    assert PortalImplementationDaemon._no_change_policy_gate_required(
        task("forbidden"),
        deterministic_only=True,
    ) is True
    assert PortalImplementationDaemon._no_change_policy_gate_required(
        task("allowed"),
        deterministic_only=False,
    ) is True


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
    daemon, _repo, workspace, _protected = _protected_git_worktree_daemon(
        tmp_path
    )
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

    assert result["blocked"] is False
    assert result["reason"] == "crash_reconciliation_ephemeral_workspace_missing"
    assert result["task_id"] == task.task_id
    assert result["attempt"] == 1
    assert result["workspace_path"] == str(workspace)
    proof = result.get("reconciliation_proof") or {}
    assert proof.get("scan_outside_lease") is True
    assert proof.get("critical_section_entered") is True
    assert proof.get("lease_hold_bounded") is True
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


def test_quiesced_shutdown_reconciles_fence_before_operator_board_revision(
    tmp_path: Path,
) -> None:
    daemon, _repo, workspace, protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = _task(outputs=["src/example.py"])
    daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    _persist_active_attempt_state(
        daemon,
        task=task,
        workspace=workspace,
    )

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert result["reason"] == "quiesced_active_attempt_reconciled"
    assert (
        result["protected_path_reconciliation"]["reason"]
        == "crash_reconciliation_unchanged"
    )
    assert not daemon._implementation_protected_active_snapshot_path().exists()
    assert not daemon._implementation_protected_incident_path().exists()
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_in_progress is False
    assert state.active_task_id == ""
    assert state.active_task_cid == ""
    assert state.active_attempt == 0
    assert state.active_worktree_path == ""

    protected.write_text("operator board revision after clean stop\n", encoding="utf-8")
    restart = daemon._reconcile_implementation_protected_path_fence()

    assert restart == {
        "blocked": False,
        "reason": "no_active_snapshot",
        "critical_section_entered": False,
        "scan_outside_lease": True,
    }
    assert not daemon._implementation_protected_incident_path().exists()


def test_quiesced_shutdown_terminalizes_exact_dead_lifecycle_owner_and_unblocks_retry(
    tmp_path: Path,
) -> None:
    daemon, _repo, workspace, _protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = _task(outputs=["src/example.py"])
    identity = daemon._identity_for_task(task)
    daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    _persist_active_attempt_state(
        daemon,
        task=task,
        workspace=workspace,
    )
    lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
        attempt=1,
        lane_id="terminated-lane",
        workspace_path=workspace,
        branch="lane",
        merge_target="main",
        state_dir=str(daemon.state_path.parent.resolve()),
        owner=ProcessBirthIdentity(
            pid=2**30 - 19,
            start_time_ticks=1,
            boot_id="provably-dead-owner",
        ),
    )
    lifecycle = daemon.worktree_lifecycle.mark_active(
        workspace,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    assert lifecycle.expires_at > daemon.worktree_lifecycle.clock()

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is True
    assert result["blocked"] is False
    lifecycle_result = result["worktree_lifecycle_reconciliation"]
    assert (
        lifecycle_result["reason"]
        == "worktree_lifecycle_dead_owner_terminalized"
    )
    assert lifecycle_result["terminal_reason"] == (
        "controlled_shutdown_quiesced_owner"
    )
    terminal = daemon.worktree_lifecycle.load_workspace(workspace)
    assert terminal is not None
    assert terminal.state is WorkspaceLifecycleState.TERMINAL
    assert terminal.fence == lifecycle.fence + 1
    assert PortalTaskState.load(daemon.state_path).implementation_in_progress is False

    # The original six-hour lease remains unexpired, but the terminal task
    # index must no longer reject the same retry attempt in a replacement
    # workspace.
    retry = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
        attempt=1,
        lane_id="replacement-lane",
        workspace_path=workspace.parent / "replacement",
        branch="implementation/example-retry",
        merge_target="main",
        state_dir=str(tmp_path / "replacement-state"),
    )
    assert retry.state is WorkspaceLifecycleState.PREPARING


def test_quiesced_shutdown_releases_exact_dead_canonical_claim_from_clean_state(
    tmp_path: Path,
) -> None:
    daemon, task, _state, claim_path, claim = _quiesced_terminal_task_claim(
        tmp_path
    )

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert result["reason"] == "already_quiesced"
    release = result["task_claim_reconciliation"]
    assert release["reconciled"] is True
    assert release["blocked"] is False
    assert release["reason"] == "quiesced_task_claim_released"
    assert release["task_id"] == task.task_id
    assert release["canonical_task_cid"] == daemon._canonical_ref(task)
    assert release["attempt"] == 4
    assert release["claim_lease_id"] == claim["lease_id"]
    assert release["owner_pid"] == claim["pid"]
    assert release["state_dir"] == str(daemon.state_path.parent.resolve())
    assert not claim_path.exists()

    receipt_path = Path(release["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_body = dict(receipt)
    receipt_id = receipt_body.pop("receipt_id")
    assert receipt["schema"] == (
        implementation_daemon_module.IMPLEMENTATION_TASK_CLAIM_RELEASE_SCHEMA
    )
    assert receipt["phase"] == "released"
    assert receipt["operation_id"] == release["operation_id"]
    assert receipt["claim"]["lease_id"] == claim["lease_id"]
    assert receipt["claim"]["claim_id"] == release["claim_id"]
    assert receipt["worktree_lifecycle"]["state"] == "terminal"
    assert receipt_id == implementation_daemon_module.content_identity(
        receipt_body
    )
    assert receipt_id == release["receipt_id"]

    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    [release_event] = [
        event
        for event in events
        if event["type"] == "implementation_task_claim_released"
    ]
    assert release_event["receipt_id"] == receipt_id
    assert release_event["operation_id"] == release["operation_id"]
    assert release_event["canonical_task_cid"] == daemon._canonical_ref(task)
    assert release_event["claim_lease_id"] == claim["lease_id"]

    replay = daemon.reconcile_quiesced_active_attempt()
    assert replay["reconciled"] is True
    assert replay["task_claim_reconciliation"]["reason"] == "no_task_claim"
    assert not claim_path.exists()


def test_quiesced_shutdown_releases_legacy_claim_without_worktree_root(
    tmp_path: Path,
) -> None:
    daemon, _task, _state, claim_path, claim = (
        _quiesced_terminal_task_claim(tmp_path)
    )
    legacy_claim = dict(claim)
    legacy_claim["worktree_root"] = ""
    claim_path.write_text(
        json.dumps(legacy_claim, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is True
    release = result["task_claim_reconciliation"]
    assert release["reason"] == "quiesced_task_claim_released"
    receipt = json.loads(
        Path(release["receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["claim"]["legacy_worktree_root_missing"] is True
    assert not claim_path.exists()


def test_quiesced_claim_release_accepts_exact_unfinished_attempt_recovery(
    tmp_path: Path,
) -> None:
    daemon, task, state, claim_path, _claim = (
        _quiesced_terminal_task_claim(tmp_path)
    )
    canonical_task_cid = daemon._canonical_ref(task)
    state.implementation_attempts[task.task_id] = 3
    state.implementation_attempts_by_cid[canonical_task_cid] = 3
    state.save(daemon.state_path)
    identity = daemon._identity_for_task(task)
    daemon._record_event(
        "implementation_started",
        {
            "task_id": task.task_id,
            "canonical_task_key": identity.canonical_task_key,
            "canonical_task_cid": canonical_task_cid,
            "board_namespace": identity.board_namespace,
            "attempt": 4,
            "worktree_path": state.last_implementation_worktree_path,
            "branch": state.last_implementation_branch,
        },
    )
    daemon._record_event(
        "implementation_state_recovered",
        {
            "task_id": task.task_id,
            "canonical_task_key": identity.canonical_task_key,
            "canonical_task_cid": canonical_task_cid,
            "board_namespace": identity.board_namespace,
            "attempt": 4,
            "reason": "inflight_process_missing",
            "worktree_path": state.last_implementation_worktree_path,
            "branch": state.last_implementation_branch,
            "attempt_recovery": {
                "task_id": task.task_id,
                "canonical_task_cid": canonical_task_cid,
                "attempt": 4,
                "previous_display_count": 4,
                "previous_cid_count": 4,
                "released": True,
                "released_to": 3,
                "consumed": False,
            },
            "finished_attempt": False,
        },
    )

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is True
    release = result["task_claim_reconciliation"]
    assert release["reason"] == "quiesced_task_claim_released"
    evidence = release["released_unfinished_attempt"]
    assert evidence["released_from"] == 4
    assert evidence["released_to"] == 3
    assert evidence["event_id"].startswith("sha256:")
    receipt = json.loads(
        Path(release["receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["released_unfinished_attempt"] == evidence
    assert not claim_path.exists()


def test_quiesced_claim_release_refuses_mismatched_attempt_recovery_event(
    tmp_path: Path,
) -> None:
    daemon, task, state, claim_path, _claim = (
        _quiesced_terminal_task_claim(tmp_path)
    )
    canonical_task_cid = daemon._canonical_ref(task)
    state.implementation_attempts[task.task_id] = 3
    state.implementation_attempts_by_cid[canonical_task_cid] = 3
    state.save(daemon.state_path)
    identity = daemon._identity_for_task(task)
    daemon._record_event(
        "implementation_started",
        {
            "task_id": task.task_id,
            "canonical_task_key": identity.canonical_task_key,
            "canonical_task_cid": canonical_task_cid,
            "board_namespace": identity.board_namespace,
            "attempt": 4,
            "worktree_path": state.last_implementation_worktree_path,
            "branch": state.last_implementation_branch,
        },
    )
    daemon._record_event(
        "implementation_state_recovered",
        {
            "task_id": task.task_id,
            "canonical_task_key": identity.canonical_task_key,
            "canonical_task_cid": canonical_task_cid,
            "board_namespace": identity.board_namespace,
            "attempt": 4,
            "reason": "inflight_process_missing",
            "worktree_path": state.last_implementation_worktree_path,
            "branch": "different-attempt-branch",
            "attempt_recovery": {
                "task_id": task.task_id,
                "canonical_task_cid": canonical_task_cid,
                "attempt": 4,
                "previous_display_count": 4,
                "previous_cid_count": 4,
                "released": True,
                "released_to": 3,
                "consumed": False,
            },
            "finished_attempt": False,
        },
    )

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is False
    release = result["task_claim_reconciliation"]
    assert release["reason"] == "task_claim_identity_mismatch"
    assert claim_path.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "claim_attempt_string",
        "display_attempt_string",
        "canonical_attempt_boolean",
        "display_attempt_missing",
        "display_attempt_not_predecessor",
    ],
)
def test_quiesced_claim_release_requires_exact_raw_attempt_counters(
    tmp_path: Path,
    corruption: str,
) -> None:
    daemon, task, state, claim_path, _claim = (
        _quiesced_terminal_task_claim(tmp_path)
    )
    canonical_task_cid = daemon._canonical_ref(task)
    state.implementation_attempts[task.task_id] = 3
    state.implementation_attempts_by_cid[canonical_task_cid] = 3
    state.save(daemon.state_path)
    _record_exact_unfinished_attempt_recovery(daemon, task, state)

    if corruption == "claim_attempt_string":
        claim_payload = json.loads(claim_path.read_text(encoding="utf-8"))
        claim_payload["attempt"] = "4"
        claim_path.write_text(
            json.dumps(claim_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        state_payload = json.loads(
            daemon.state_path.read_text(encoding="utf-8")
        )
        if corruption == "display_attempt_string":
            state_payload["implementation_attempts"][task.task_id] = "3"
        elif corruption == "canonical_attempt_boolean":
            state_payload["implementation_attempts_by_cid"][
                canonical_task_cid
            ] = True
        elif corruption == "display_attempt_missing":
            del state_payload["implementation_attempts"][task.task_id]
        else:
            state_payload["implementation_attempts"][task.task_id] = 2
        daemon.state_path.write_text(
            json.dumps(state_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    result = daemon._reconcile_quiesced_implementation_task_claim(state)

    assert result["reconciled"] is False
    assert result["reason"] == "task_claim_identity_mismatch"
    assert claim_path.exists()


@pytest.mark.parametrize(
    ("include_start", "later_event_type"),
    [
        (False, ""),
        (True, "implementation_started"),
        (True, "implementation_finished"),
    ],
)
def test_quiesced_claim_release_rejects_ambiguous_recovery_timeline(
    tmp_path: Path,
    include_start: bool,
    later_event_type: str,
) -> None:
    daemon, task, state, claim_path, _claim = (
        _quiesced_terminal_task_claim(tmp_path)
    )
    canonical_task_cid = daemon._canonical_ref(task)
    state.implementation_attempts[task.task_id] = 3
    state.implementation_attempts_by_cid[canonical_task_cid] = 3
    state.save(daemon.state_path)
    _record_exact_unfinished_attempt_recovery(
        daemon,
        task,
        state,
        include_start=include_start,
        later_event_type=later_event_type,
    )

    result = daemon._reconcile_quiesced_implementation_task_claim(state)

    assert result["reconciled"] is False
    assert result["reason"] == "task_claim_identity_mismatch"
    assert claim_path.exists()


def test_supervisor_startup_preflight_releases_quiesced_dead_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _task_value, _state, claim_path, _claim = (
        _quiesced_terminal_task_claim(tmp_path)
    )
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(daemon.todo_path),
            "--state-dir",
            str(daemon.state_path.parent),
            "--implementation-protected-path",
            POLICY_PATH,
        ]
    )
    supervisor = PortalImplementationSupervisor(
        supervisor_config_from_args(
            args,
            repo_root=daemon.repo_root,
            state_path=daemon.state_path,
        )
    )
    monkeypatch.setattr(
        supervisor,
        "_build_worktree_reconciliation_daemon",
        lambda: daemon,
    )
    maintenance_started: list[bool] = []

    def preflight(*, include_refill: bool) -> dict[str, object]:
        assert include_refill is False
        assert not claim_path.exists()
        maintenance_started.append(True)
        return {"stuck": False}

    class StoppedLoop:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(
                status="stopped",
                restart_count=0,
                last_exit_code=None,
                last_recycle_reason="",
                last_run_id="",
                last_log_path="",
            )

    monkeypatch.setattr(supervisor, "ensure_event_log_file", lambda: {})
    monkeypatch.setattr(
        supervisor,
        "ensure_managed_daemon_pid_file",
        lambda: {"repaired": False, "reason": "missing"},
    )
    monkeypatch.setattr(supervisor, "run_once", preflight)
    supervisor.shared_supervisor_loop_class = StoppedLoop

    supervisor._run_forever_loop()

    assert maintenance_started == [True]
    assert not claim_path.exists()
    receipts = list(
        (
            daemon.state_path.parent
            / implementation_daemon_module.IMPLEMENTATION_TASK_CLAIM_RELEASE_RECEIPT_DIRNAME
        ).glob("*.json")
    )
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["phase"] == (
        "released"
    )


def test_worktree_reconciliation_daemon_preserves_lane_shard_identity(
    tmp_path: Path,
) -> None:
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--task-shard-count",
            "4",
            "--task-shard-index",
            "2",
            "--strict-task-sharding",
        ]
    )
    supervisor = PortalImplementationSupervisor(
        supervisor_config_from_args(args, repo_root=tmp_path)
    )

    daemon = supervisor._build_worktree_reconciliation_daemon()

    assert daemon.task_shard_count == 4
    assert daemon.task_shard_index == 2
    assert daemon.strict_task_sharding is True


def test_supervisor_startup_does_not_reconcile_live_managed_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)

    class StoppedLoop:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(
                status="stopped",
                restart_count=0,
                last_exit_code=None,
                last_recycle_reason="",
                last_run_id="",
                last_log_path="",
            )

    monkeypatch.setattr(supervisor, "ensure_event_log_file", lambda: {})
    monkeypatch.setattr(
        supervisor,
        "ensure_managed_daemon_pid_file",
        lambda: {"repaired": False, "reason": "active", "pid": os.getpid()},
    )
    monkeypatch.setattr(
        supervisor,
        "_reconcile_quiesced_task_claim_at_startup",
        lambda: pytest.fail("a live managed daemon must retain its claims"),
    )
    monkeypatch.setattr(
        supervisor,
        "run_once",
        lambda *, include_refill: {"stuck": False},
    )
    supervisor.shared_supervisor_loop_class = StoppedLoop

    supervisor._run_forever_loop()


def test_quiesced_claim_release_refuses_live_owner(
    tmp_path: Path,
) -> None:
    daemon, _task_value, _state, claim_path, _claim = (
        _quiesced_terminal_task_claim(
            tmp_path,
            live_claim_owner=True,
        )
    )

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is False
    assert result["blocked"] is True
    assert result["reason"] == "task_claim_reconciliation_blocked"
    release = result["task_claim_reconciliation"]
    assert release["reason"] == "task_claim_owner_still_active"
    assert release["owner_pid"] == os.getpid()
    assert claim_path.exists()
    assert not (
        daemon.state_path.parent
        / implementation_daemon_module.IMPLEMENTATION_TASK_CLAIM_RELEASE_RECEIPT_DIRNAME
    ).exists()


def test_quiesced_claim_release_refuses_nonterminal_task(
    tmp_path: Path,
) -> None:
    daemon, _task_value, _state, claim_path, _claim = (
        _quiesced_terminal_task_claim(
            tmp_path,
            task_status="ready",
        )
    )

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is False
    assert result["blocked"] is True
    release = result["task_claim_reconciliation"]
    assert release["reason"] == "canonical_task_not_terminal"
    assert release["observed_task_status"] == "todo"
    assert claim_path.exists()


def test_quiesced_claim_release_refuses_noncanonical_lifecycle_bytes(
    tmp_path: Path,
) -> None:
    daemon, _task, state, claim_path, _claim = (
        _quiesced_terminal_task_claim(tmp_path)
    )
    record_path = daemon.worktree_lifecycle.workspace_path_for(
        Path(state.last_implementation_worktree_path)
    )
    record_payload = json.loads(record_path.read_text(encoding="utf-8"))
    record_payload["unexpected_field"] = True
    corrupted_bytes = (
        json.dumps(record_payload, indent=2, sort_keys=True) + "\n"
    )
    record_path.write_text(corrupted_bytes, encoding="utf-8")

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is False
    release = result["task_claim_reconciliation"]
    assert release["reason"] == "task_claim_worktree_lifecycle_malformed"
    assert claim_path.exists()
    assert record_path.read_text(encoding="utf-8") == corrupted_bytes


def test_quiesced_claim_release_exact_cas_preserves_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _task_value, state, claim_path, claim = (
        _quiesced_terminal_task_claim(tmp_path)
    )
    replacement = dict(claim)
    replacement["lease_id"] = "replacement-lease-that-must-survive"
    original_load = daemon._load_exact_json_object
    claim_reads = 0

    def replace_before_compare(path: Path):
        nonlocal claim_reads
        if path == claim_path:
            claim_reads += 1
            if claim_reads == 2:
                claim_path.write_text(
                    json.dumps(replacement, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        return original_load(path)

    monkeypatch.setattr(
        daemon,
        "_load_exact_json_object",
        replace_before_compare,
    )

    release = daemon._reconcile_quiesced_implementation_task_claim(
        state
    )

    assert release["reconciled"] is False
    assert release["blocked"] is True
    assert release["reason"] == "task_claim_compare_and_delete_lost"
    assert json.loads(claim_path.read_text(encoding="utf-8"))["lease_id"] == (
        "replacement-lease-that-must-survive"
    )
    receipt_dir = (
        daemon.state_path.parent
        / implementation_daemon_module.IMPLEMENTATION_TASK_CLAIM_RELEASE_RECEIPT_DIRNAME
    )
    [prepared_path] = list(receipt_dir.glob("*.json"))
    assert json.loads(prepared_path.read_text(encoding="utf-8"))["phase"] == (
        "prepared"
    )


@pytest.mark.parametrize(
    ("ownership_case", "expected_reason"),
    [
        ("live", "worktree_lifecycle_owner_still_active"),
        ("wrong_state", "worktree_lifecycle_state_dir_mismatch"),
        ("unknown_liveness", "worktree_lifecycle_owner_liveness_unknown"),
    ],
)
def test_quiesced_shutdown_keeps_unproven_lifecycle_ownership_blocked(
    tmp_path: Path,
    ownership_case: str,
    expected_reason: str,
) -> None:
    daemon, _repo, workspace, _protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = _task(outputs=["src/example.py"])
    identity = daemon._identity_for_task(task)
    daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    _persist_active_attempt_state(
        daemon,
        task=task,
        workspace=workspace,
    )
    dead_owner = ProcessBirthIdentity(
        pid=2**30 - 23,
        start_time_ticks=1,
        boot_id="provably-dead-owner",
    )
    lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
        attempt=1,
        lane_id=f"{ownership_case}-lane",
        workspace_path=workspace,
        branch="lane",
        merge_target="main",
        state_dir=str(
            tmp_path / "different-state"
            if ownership_case == "wrong_state"
            else daemon.state_path.parent.resolve()
        ),
        owner=None if ownership_case == "live" else dead_owner,
    )
    lifecycle = daemon.worktree_lifecycle.mark_active(
        workspace,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    if ownership_case == "unknown_liveness":
        daemon.worktree_lifecycle.proc_root = tmp_path / "missing-proc"

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is False
    assert result["blocked"] is True
    assert result["reason"] == "worktree_lifecycle_reconciliation_blocked"
    lifecycle_result = result["worktree_lifecycle_reconciliation"]
    assert lifecycle_result["reason"] == expected_reason
    unchanged = daemon.worktree_lifecycle.load_workspace(workspace)
    assert unchanged is not None
    assert unchanged.state is WorkspaceLifecycleState.ACTIVE
    assert unchanged.fence == lifecycle.fence
    assert PortalTaskState.load(daemon.state_path).implementation_in_progress is True
    assert daemon._implementation_protected_active_snapshot_path().exists()
    assert not daemon._implementation_protected_incident_path().exists()


def test_quiesced_shutdown_preserves_real_protected_path_incident(
    tmp_path: Path,
) -> None:
    daemon, _repo, workspace, protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = _task(outputs=["src/example.py"])
    daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    _persist_active_attempt_state(
        daemon,
        task=task,
        workspace=workspace,
    )
    protected.write_text("implementation-time mutation\n", encoding="utf-8")

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is False
    assert result["blocked"] is True
    assert result["reason"] == "protected_path_reconciliation_blocked"
    assert (
        result["protected_path_reconciliation"]["reason"]
        == "implementation_protected_path_mutated"
    )
    assert daemon._implementation_protected_active_snapshot_path().exists()
    assert daemon._implementation_protected_incident_path().exists()
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_in_progress is True
    assert state.active_task_id == task.task_id


def test_quiesced_shutdown_refuses_live_implementation_lock(
    tmp_path: Path,
) -> None:
    daemon, _repo, workspace, _protected = _protected_git_worktree_daemon(
        tmp_path
    )
    task = _task(outputs=["src/example.py"])
    daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    _persist_active_attempt_state(
        daemon,
        task=task,
        workspace=workspace,
    )
    lock_path = daemon._implementation_lock_path()
    lock_path.write_text(
        json.dumps(
            {
                "kind": "implementation",
                "pid": os.getpid(),
                "state_dir": str(daemon.state_path.parent.resolve()),
                "task_id": task.task_id,
                "attempt": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = daemon.reconcile_quiesced_active_attempt()

    assert result["reconciled"] is False
    assert result["blocked"] is True
    assert result["reason"] == "implementation_lock_owner_still_active"
    assert daemon._implementation_protected_active_snapshot_path().exists()
    assert not daemon._implementation_protected_incident_path().exists()
    assert PortalTaskState.load(daemon.state_path).implementation_in_progress


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


def test_ephemeral_fence_accepts_trusted_update_after_temporary_merge_rollback(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    base, before_head = _temporary_shared_merge(repo, protected)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    _git(repo, "reset", "--hard", base)
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
    after_head = _git(repo, "rev-parse", "HEAD")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", before_head, after_head],
        cwd=repo,
        check=False,
    )
    assert ancestry.returncode == 1

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
    assert accepted[0]["before_head"] == before_head
    assert accepted[0]["after_head"] == after_head
    assert accepted[0]["history_kind"] == "diverged_trusted_after_side"
    assert accepted[0]["merge_base"]
    assert accepted[0]["history_protected_paths"] == [POLICY_PATH]


def test_ephemeral_fence_rejects_diverged_before_side_protected_change(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    base, _before_head = _temporary_shared_merge(
        repo,
        protected,
        protected_content="temporary protected change\n",
    )
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    _git(repo, "reset", "--hard", base)
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

    assert violation["reason"] == "implementation_protected_path_mutated"
    assert violation["protected_paths"] == [POLICY_PATH]
    assert daemon._implementation_protected_incident_path().exists()


def test_ephemeral_fence_rejects_diverged_untrusted_after_side_commit(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    base, _before_head = _temporary_shared_merge(repo, protected)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    _git(repo, "reset", "--hard", base)
    protected.write_text("untrusted update\n", encoding="utf-8")
    _git(repo, "add", POLICY_PATH)
    _git(
        repo,
        "-c",
        "user.name=Untrusted",
        "-c",
        "user.email=untrusted@example.invalid",
        "commit",
        "-m",
        "change protected policy",
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


def test_latched_diverged_incident_auto_clears_with_preserved_trusted_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    base, _before_head = _temporary_shared_merge(repo, protected)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    _git(repo, "reset", "--hard", base)
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

    authorizer = daemon._authorized_concurrent_protected_path_update
    monkeypatch.setattr(
        daemon,
        "_authorized_concurrent_protected_path_update",
        lambda **_kwargs: {},
    )
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert violation["reason"] == "implementation_protected_path_mutated"
    assert daemon._implementation_protected_incident_path().exists()
    monkeypatch.setattr(
        daemon,
        "_authorized_concurrent_protected_path_update",
        authorizer,
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is False
    assert result["reason"] == (
        "implementation_protected_path_incident_auto_cleared"
    )
    assert not daemon._implementation_protected_incident_path().exists()
    assert not daemon._implementation_protected_active_snapshot_path().exists()
    receipt_path = Path(result["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == (
        "implementation-protected-path-auto-clearance-v1"
    )
    assert receipt["incident"]["reason"] == (
        "implementation_protected_path_mutated"
    )
    assert receipt["active_snapshot"]["task_id"] == task.task_id
    assert receipt["authorization"]["history_kind"] == (
        "diverged_trusted_after_side"
    )
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == (
        "implementation_protected_path_incident_auto_cleared"
    )


def test_latched_diverged_incident_with_malformed_attempt_stays_latched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    base, _before_head = _temporary_shared_merge(repo, protected)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    _git(repo, "reset", "--hard", base)
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

    monkeypatch.setattr(
        daemon,
        "_authorized_concurrent_protected_path_update",
        lambda **_kwargs: {},
    )
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert violation["reason"] == "implementation_protected_path_mutated"

    incident_path = daemon._implementation_protected_incident_path()
    active_path = daemon._implementation_protected_active_snapshot_path()
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    active = json.loads(active_path.read_text(encoding="utf-8"))
    incident["attempt"] = "malformed"
    active["attempt"] = "malformed"
    incident_path.write_text(
        json.dumps(incident, sort_keys=True),
        encoding="utf-8",
    )
    active_path.write_text(
        json.dumps(active, sort_keys=True),
        encoding="utf-8",
    )
    incident_before = incident_path.read_bytes()
    active_before = active_path.read_bytes()

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is True
    assert result["reason"] == "implementation_protected_path_incident_latched"
    assert incident_path.read_bytes() == incident_before
    assert active_path.read_bytes() == active_before
    assert not list(
        incident_path.parent.glob(
            "implementation-protected-path-auto-clearance-*.json"
        )
    )


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


@pytest.mark.parametrize("trusted", [True, False])
def test_ephemeral_fence_waits_for_checkout_transaction_before_verifying(
    tmp_path: Path,
    trusted: bool,
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
    lock_path = checkout_mutation_lock_path(repo)
    transaction_visible = threading.Event()

    def commit_peer_update() -> None:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
        metadata = checkout_lock_metadata(
            kind="merge",
            repo_root=repo,
            owner_script="",
            extra={
                "operation": "generated_board_update",
                "lease_id": "peer-generated-board-transaction",
            },
        )
        os.write(
            lock_fd,
            json.dumps(metadata, sort_keys=True).encode("utf-8"),
        )
        os.close(lock_fd)
        protected.write_text("peer transaction\n", encoding="utf-8")
        transaction_visible.set()
        time.sleep(0.1)
        _git(repo, "add", POLICY_PATH)
        if trusted:
            author_name = "Accelerator Backlog Refinery"
            author_email = BACKLOG_REFINERY_AUTHOR_EMAIL
            subject = generated_protected_board_commit_subject(
                "Agent: persist serialized generated board"
            )
        else:
            author_name = "Untrusted User"
            author_email = "untrusted@example.invalid"
            subject = "edit protected board"
        _git(
            repo,
            "-c",
            f"user.name={author_name}",
            "-c",
            f"user.email={author_email}",
            "commit",
            "-m",
            subject,
        )
        lock_path.unlink()

    worker = threading.Thread(target=commit_peer_update)
    worker.start()
    assert transaction_visible.wait(timeout=2)
    started = time.monotonic()
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    waited_seconds = time.monotonic() - started
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert waited_seconds >= 0.05
    if trusted:
        assert violation == {}
        assert not daemon._implementation_protected_incident_path().exists()
    else:
        assert violation["reason"] == (
            "implementation_protected_path_mutated"
        )
        assert daemon._implementation_protected_incident_path().exists()


def test_protected_verification_release_preserves_replacement_lock(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    lock_result = (
        daemon._acquire_implementation_protected_verification_lock(
            task_id="EX-001",
            attempt=1,
            workspace_path=tmp_path,
        )
    )
    assert lock_result["acquired"] is True
    lock_path = Path(lock_result["lock_path"])
    replacement = checkout_lock_metadata(
        kind="merge",
        repo_root=tmp_path,
        owner_script="",
        extra={
            "operation": "replacement_transaction",
            "lease_id": "replacement-lease",
        },
    )
    lock_path.unlink()
    lock_path.write_text(
        json.dumps(replacement, sort_keys=True),
        encoding="utf-8",
    )

    assert (
        daemon._release_implementation_protected_verification_lock(
            lock_result
        )
        is False
    )
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement


def test_protected_verification_lock_timeout_defers_without_latching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _repo, workspace, _protected = (
        _protected_git_worktree_daemon(tmp_path)
    )
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    lock_path = Path(
        daemon._acquire_implementation_protected_verification_lock(
            task_id="EX-PEER",
            attempt=1,
            workspace_path=workspace,
        )["lock_path"]
    )
    replacement = checkout_lock_metadata(
        kind="merge",
        repo_root=daemon.repo_root,
        owner_script="",
        extra={
            "operation": "generated_board_update",
            "lease_id": "live-peer-lease",
        },
    )
    lock_path.unlink()
    lock_path.write_text(
        json.dumps(replacement, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "IMPLEMENTATION_PROTECTED_VERIFICATION_LOCK_TIMEOUT_SECONDS",
        0.0,
    )

    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )

    assert violation["reason"] == (
        "implementation_protected_path_verification_lock_timeout"
    )
    assert violation["verification_deferred"] is True
    assert {
        item["scope"] for item in violation["mutations"]
    } == {"shared_checkout"}
    assert not daemon._implementation_protected_incident_path().exists()
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement
    lock_path.unlink()


def test_shared_terminal_verification_deferral_does_not_consume_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("unchanged\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    canonical_task_cid = daemon._canonical_ref(task)
    state = PortalTaskState(
        implementation_attempts={task.task_id: 2},
        implementation_attempts_by_cid={canonical_task_cid: 2},
    )
    queue_outcomes: list[int] = []
    diagnostics: list[str] = []
    deferral = {
        "reason": "implementation_protected_path_verification_lock_timeout",
        "task_id": task.task_id,
        "attempt": 3,
        "workspace_path": str(tmp_path),
        "protected_paths": [POLICY_PATH],
        # Keep this empty so the regression depends on the explicit signal,
        # not the older shared-checkout-only mutation-scope heuristic.
        "mutations": [],
        "verification_deferred": True,
    }

    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["fake-agent"],
            0,
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_violation",
        lambda **_kwargs: dict(deferral),
    )
    monkeypatch.setattr(
        daemon,
        "_record_task_queue_outcome",
        lambda _task, returncode, **_kwargs: queue_outcomes.append(returncode),
    )
    monkeypatch.setattr(
        daemon,
        "_record_failed_attempt_retry_context",
        lambda *_args, **_kwargs: diagnostics.append("retry") or None,
    )

    result = daemon._run_implementation(task, state)

    assert result["returncode"] == 1
    assert result["reason"] == deferral["reason"]
    assert result["deferred"] is True
    assert result["attempt_consumed"] is False
    assert "diagnostic_receipt_id" not in result
    assert queue_outcomes == []
    assert diagnostics == []
    assert state.implementation_attempts == {task.task_id: 2}
    assert state.implementation_attempts_by_cid == {canonical_task_cid: 2}
    persisted = PortalTaskState.load(daemon.state_path)
    assert persisted.implementation_attempts == {task.task_id: 2}
    assert persisted.implementation_attempts_by_cid == {
        canonical_task_cid: 2
    }


def test_ephemeral_verification_lock_deferral_does_not_consume_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _repo, _workspace, _protected = (
        _protected_git_worktree_daemon(tmp_path)
    )
    task = _task(outputs=["src/example.py"])
    canonical_task_cid = daemon._canonical_ref(task)
    state = PortalTaskState(
        implementation_attempts={task.task_id: 2},
        implementation_attempts_by_cid={canonical_task_cid: 2},
    )
    queue_outcomes: list[int] = []
    diagnostics: list[str] = []
    verification_deferred = False

    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["fake-agent"],
            0,
        ),
    )

    def defer_verification(**_kwargs):
        nonlocal verification_deferred
        verification_deferred = True
        return {
            "acquired": False,
            "reason": "lock_exists",
            "lock_path": str(checkout_mutation_lock_path(daemon.repo_root)),
            "waited_seconds": 0.0,
        }

    monkeypatch.setattr(
        daemon,
        "_acquire_implementation_protected_verification_lock",
        defer_verification,
    )
    monkeypatch.setattr(
        daemon,
        "_preserve_interrupted_worktree",
        lambda *_args, **_kwargs: pytest.fail(
            "verification deferral attempted candidate commit or rescue ref"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_cleanup_merged_worktree",
        lambda *_args, **_kwargs: pytest.fail(
            "verification deferral attempted worktree or branch cleanup"
        ),
    )
    original_run_git = daemon._run_git

    def reject_post_verification_git(*args, **kwargs):
        if verification_deferred:
            pytest.fail(
                "verification deferral attempted a shared Git mutation"
            )
        return original_run_git(*args, **kwargs)

    monkeypatch.setattr(daemon, "_run_git", reject_post_verification_git)
    monkeypatch.setattr(
        daemon,
        "_record_task_queue_outcome",
        lambda _task, returncode, **_kwargs: queue_outcomes.append(returncode),
    )
    monkeypatch.setattr(
        daemon,
        "_record_failed_attempt_retry_context",
        lambda *_args, **_kwargs: diagnostics.append("retry") or None,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=state,
        attempt=3,
        started_at="2026-07-29T00:00:00+00:00",
        log_path=tmp_path / "state" / "ephemeral-deferral.log",
        prompt="implement",
    )

    assert result["returncode"] == 1
    assert result["reason"] == (
        "implementation_protected_path_verification_lock_timeout"
    )
    assert result["protected_path_violation"]["verification_deferred"] is True
    assert result["deferred"] is True
    assert result["attempt_consumed"] is False
    assert "diagnostic_receipt_id" not in result
    retained = result["failed_preservation_result"]
    assert retained["retained"] is True
    assert retained["preserved"] is False
    assert retained["reason"] == (
        "verification_deferred_checkout_lease_active"
    )
    assert retained["commit_result"]["committed"] is False
    assert retained["cleanup_result"] == {
        "cleaned": False,
        "reason": "verification_deferred_checkout_lease_active",
        "retained": True,
    }
    retained_path = Path(result["worktree_path"])
    assert retained_path.exists()
    lifecycle = daemon.worktree_lifecycle.load_workspace(retained_path)
    assert lifecycle is not None
    assert lifecycle.is_terminal
    assert lifecycle.terminal_reason == (
        "verification_deferred_checkout_lease_unavailable"
    )
    assert daemon._active_worktree_lifecycle is None
    cleanup_authorization = daemon.worktree_lifecycle.authorize_cleanup(
        workspace_path=retained_path,
        branch=result["branch"],
    )
    assert cleanup_authorization.allowed
    assert cleanup_authorization.reason == "terminal_record"
    retry_lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=canonical_task_cid,
        attempt=3,
        lane_id=daemon._worktree_lifecycle_lane_id(),
        workspace_path=tmp_path / "worktrees" / "retry-attempt-3",
        branch=f"{result['branch']}-retry",
        merge_target="main",
        state_dir=str(daemon.state_path.parent),
    )
    assert retry_lifecycle.state.value == "preparing"
    assert queue_outcomes == []
    assert diagnostics == []
    assert state.implementation_attempts == {task.task_id: 2}
    assert state.implementation_attempts_by_cid == {canonical_task_cid: 2}
    persisted = PortalTaskState.load(daemon.state_path)
    assert persisted.implementation_attempts == {task.task_id: 2}
    assert persisted.implementation_attempts_by_cid == {
        canonical_task_cid: 2
    }


def test_protected_verification_double_snapshot_workspace_race_latches_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _repo, workspace, _protected = (
        _protected_git_worktree_daemon(tmp_path)
    )
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )
    unchanged_after = json.loads(json.dumps(before))
    (workspace / POLICY_PATH).write_text(
        "changed-between-verification-snapshots\n",
        encoding="utf-8",
    )
    changed_after = daemon._implementation_protected_path_snapshot(workspace)
    snapshots = iter((unchanged_after, changed_after))

    monkeypatch.setattr(
        daemon,
        "_acquire_implementation_protected_verification_lock",
        lambda **_kwargs: {
            "acquired": True,
            "reason": "acquired",
            "lock_path": str(checkout_mutation_lock_path(daemon.repo_root)),
            "waited_seconds": 0.0,
        },
    )
    monkeypatch.setattr(
        daemon,
        "_release_implementation_protected_verification_lock",
        lambda _lock_result: True,
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_snapshot",
        lambda _workspace_path: next(snapshots),
    )

    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )

    assert violation["reason"] == "implementation_protected_path_mutated"
    assert violation.get("verification_deferred", False) is False
    assert {
        item["scope"] for item in violation["mutations"]
    } == {"workspace"}
    assert daemon._implementation_protected_incident_path().exists()


def test_reconciliation_accepts_trusted_board_commit_that_lands_after_latch(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    protected.write_text("generated update awaiting persistence\n", encoding="utf-8")
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert violation["reason"] == "implementation_protected_path_mutated"
    assert daemon._implementation_protected_incident_path().exists()

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
            "Agent: persist delayed generated board"
        ),
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["cleared"] is True
    assert result["auto"] is True
    assert result["blocked"] is False
    assert result["reason"] == "trusted_concurrent_protected_path_update"
    assert result["protected_paths"] == [POLICY_PATH]
    assert not daemon._implementation_protected_incident_path().exists()
    assert not daemon._implementation_protected_active_snapshot_path().exists()
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["schema"] == (
        "implementation-protected-path-trusted-concurrent-clearance-v1"
    )
    assert receipt["commits"][0]["author_email"] == BACKLOG_REFINERY_AUTHOR_EMAIL


def test_reconciliation_keeps_late_untrusted_board_commit_latched(
    tmp_path: Path,
) -> None:
    daemon, repo, workspace, protected = _protected_git_worktree_daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    before = daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=workspace,
    )

    protected.write_text("untrusted delayed update\n", encoding="utf-8")
    violation = daemon._implementation_protected_path_violation(
        task=task,
        attempt=1,
        workspace_path=workspace,
        before=before,
    )
    assert violation["reason"] == "implementation_protected_path_mutated"

    _git(repo, "add", POLICY_PATH)
    _git(
        repo,
        "-c",
        "user.name=Untrusted User",
        "-c",
        "user.email=untrusted@example.invalid",
        "commit",
        "-m",
        "edit protected board",
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is True
    assert result["reason"] == "implementation_protected_path_incident_latched"
    assert daemon._implementation_protected_incident_path().exists()
    assert daemon._implementation_protected_active_snapshot_path().exists()


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
    assert not checkout_mutation_lock_path(repo).exists()


@pytest.mark.parametrize("commit_untrusted", [False, True])
def test_generated_board_producer_retains_lease_for_unsafe_protected_output(
    tmp_path: Path,
    commit_untrusted: bool,
) -> None:
    supervisor, repo, todo_path = _generated_protected_supervisor(
        tmp_path
    )

    def unsafe_producer() -> list[str]:
        todo_path.write_text("# Tasks\n\n## EX-002 Unsafe\n", encoding="utf-8")
        if commit_untrusted:
            _git(repo, "add", "tasks.todo.md")
            _git(repo, "commit", "-m", "untrusted generated update")
        return ["EX-002"]

    expected_reason = (
        "protected_generated_history_untrusted"
        if commit_untrusted
        else "protected_generated_outputs_dirty"
    )
    with pytest.raises(RuntimeError, match=expected_reason):
        supervisor._run_generated_board_producer(
            producer="unsafe-test",
            commit_outputs=True,
            callback=unsafe_producer,
        )

    lock_path = checkout_mutation_lock_path(repo)
    assert lock_path.exists()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["lease_id"]
    events = [
        json.loads(line)
        for line in supervisor.config.events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert events[-1]["type"] == "checkout_mutation_lease_retained"
    assert events[-1]["release_guard"]["reason"] == expected_reason

    unrelated_called = False

    def unrelated_producer() -> list[str]:
        nonlocal unrelated_called
        unrelated_called = True
        return []

    with pytest.raises(
        RuntimeError,
        match="checkout_mutation_protected_recovery_required",
    ):
        supervisor._run_generated_board_producer(
            producer="unrelated-test",
            commit_outputs=True,
            callback=unrelated_producer,
        )
    assert unrelated_called is False


def test_generated_board_same_producer_retry_releases_retained_lease(
    tmp_path: Path,
) -> None:
    supervisor, repo, todo_path = _generated_protected_supervisor(tmp_path)

    def dirty_producer() -> list[str]:
        todo_path.write_text("# Tasks\n\n## EX-002 Retry\n", encoding="utf-8")
        return ["dirty"]

    with pytest.raises(
        RuntimeError,
        match="protected_generated_outputs_dirty",
    ):
        supervisor._run_generated_board_producer(
            producer="retry-test",
            commit_outputs=True,
            callback=dirty_producer,
        )

    def trusted_retry() -> list[str]:
        _git(repo, "add", "tasks.todo.md")
        _git(
            repo,
            "-c",
            "user.name=Agent Supervisor",
            "-c",
            f"user.email={BACKLOG_REFINERY_AUTHOR_EMAIL}",
            "commit",
            "-m",
            generated_protected_board_commit_subject("retry generated output"),
        )
        return ["recovered"]

    assert supervisor.config.generated_dirty_repair_enabled is False
    assert supervisor._run_generated_board_producer(
        producer="retry-test",
        commit_outputs=True,
        callback=trusted_retry,
    ) == ["recovered"]
    assert not checkout_mutation_lock_path(repo).exists()
    assert supervisor._current_supervisor_checkout_lease() is None


def test_generated_dirty_repair_recovers_retained_lease_when_disabled(
    tmp_path: Path,
) -> None:
    supervisor, repo, todo_path = _generated_protected_supervisor(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="protected_generated_outputs_dirty",
    ):
        supervisor._run_generated_board_producer(
            producer="repair-test",
            commit_outputs=True,
            callback=lambda: todo_path.write_text(
                "# Tasks\n\n## EX-002 Repair\n",
                encoding="utf-8",
            ),
        )

    assert supervisor.config.generated_dirty_repair_enabled is False
    result = supervisor.repair_generated_dirty_checkouts()

    assert result["committed_count"] == 1
    assert not checkout_mutation_lock_path(repo).exists()
    assert supervisor._current_supervisor_checkout_lease() is None


def test_fresh_generated_dirty_repair_journals_before_callback_and_retains(
    tmp_path: Path,
) -> None:
    supervisor, repo, todo_path = _generated_protected_supervisor(tmp_path)
    todo_path.write_text(
        "# Tasks\n\n## EX-002 Fresh repair\n",
        encoding="utf-8",
    )
    observed_journal: dict[str, object] = {}

    def incomplete_repair() -> list[str]:
        lease = checkout_lock_module.read_checkout_mutation_lease(
            checkout_mutation_lock_path(repo)
        )
        assert lease is not None
        observed_journal.update(lease.metadata)
        return ["still-dirty"]

    with pytest.raises(
        RuntimeError,
        match="protected_generated_outputs_dirty",
    ):
        supervisor._run_generated_board_producer(
            producer="generated-dirty-repair",
            commit_outputs=True,
            operation="generated_dirty_repair",
            callback=incomplete_repair,
        )

    assert observed_journal["protected_recovery_required"] is True
    assert (
        observed_journal["protected_recovery_owner"]
        == "implementation_supervisor"
    )
    assert checkout_mutation_lock_path(repo).exists()
    assert supervisor._retained_generated_checkout_lease() is True


def test_generated_board_callback_exception_survives_guard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, repo, _todo_path = _generated_protected_supervisor(tmp_path)
    real_guard = supervisor._generated_protected_release_guard
    guard_calls = 0

    def fail_guard(snapshot) -> dict[str, object]:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 1:
            return real_guard(snapshot)
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(
        supervisor,
        "_generated_protected_release_guard",
        fail_guard,
    )

    with pytest.raises(ValueError, match="producer sentinel"):
        supervisor._run_generated_board_producer(
            producer="exception-test",
            commit_outputs=True,
            callback=lambda: (_ for _ in ()).throw(
                ValueError("producer sentinel")
            ),
        )

    assert checkout_mutation_lock_path(repo).exists()
    assert supervisor._supervisor_checkout_transaction_depth() == 0
    assert supervisor._retained_generated_checkout_lease() is True


def test_generated_board_replacement_failed_release_is_not_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, repo, _todo_path = _generated_protected_supervisor(tmp_path)
    monkeypatch.setattr(
        supervisor,
        "_release_supervisor_checkout_lease",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(
        RuntimeError,
        match="checkout_mutation_lease_release_failed",
    ):
        supervisor._run_generated_board_producer(
            producer="replacement-test",
            commit_outputs=True,
            callback=lambda: ["unchanged"],
        )

    assert checkout_mutation_lock_path(repo).exists()
    assert supervisor._retained_generated_checkout_lease() is True
    assert supervisor._current_supervisor_checkout_lease() is not None


def test_generated_board_snapshot_exception_releases_without_fake_nesting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, repo, _todo_path = _generated_protected_supervisor(tmp_path)
    callback_called = False

    def fail_snapshot() -> dict[str, object]:
        raise RuntimeError("snapshot exploded")

    def callback() -> list[str]:
        nonlocal callback_called
        callback_called = True
        return []

    monkeypatch.setattr(
        supervisor,
        "_generated_protected_release_guard_snapshot",
        fail_snapshot,
    )

    with pytest.raises(RuntimeError, match="snapshot exploded"):
        supervisor._run_generated_board_producer(
            producer="snapshot-test",
            commit_outputs=True,
            callback=callback,
        )

    assert callback_called is False
    assert not checkout_mutation_lock_path(repo).exists()
    assert supervisor._current_supervisor_checkout_lease() is None
    assert supervisor._supervisor_checkout_transaction_depth() == 0


def test_supervisor_adopts_and_recovers_journal_after_restart(
    tmp_path: Path,
) -> None:
    supervisor, repo, todo_path = _generated_protected_supervisor(tmp_path)
    observed_journal: dict[str, object] = {}

    def interrupted_producer() -> list[str]:
        lease = checkout_lock_module.read_checkout_mutation_lease(
            checkout_mutation_lock_path(repo)
        )
        assert lease is not None
        observed_journal.update(lease.metadata)
        todo_path.write_text(
            "# Tasks\n\n## EX-002 Restart recovery\n",
            encoding="utf-8",
        )
        return ["EX-002"]

    with pytest.raises(
        RuntimeError,
        match="protected_generated_outputs_dirty",
    ):
        supervisor._run_generated_board_producer(
            producer="restart-test",
            commit_outputs=True,
            callback=interrupted_producer,
        )

    assert observed_journal["protected_recovery_required"] is True
    assert (
        observed_journal["protected_recovery_owner"]
        == "implementation_supervisor"
    )
    guard = dict(observed_journal["protected_release_guard"])
    guard_id = guard.pop("guard_id")
    assert checkout_lock_module.content_identity(guard) == guard_id
    intent = dict(observed_journal["protected_recovery_intent"])
    intent_id = intent.pop("intent_id")
    assert checkout_lock_module.content_identity(intent) == intent_id
    assert intent["operation"] == "generated_board_update"
    assert intent["producer"] == "restart-test"
    assert intent["protected_paths"] == ["tasks.todo.md"]

    stale = checkout_lock_module.read_checkout_mutation_lease(
        checkout_mutation_lock_path(repo)
    )
    assert stale is not None
    dead_owner = checkout_lock_module.update_checkout_mutation_lease(
        stale,
        {
            **dict(stale.metadata),
            "pid": 2_147_483_647,
        },
    )
    assert dead_owner is not None

    restarted = PortalImplementationSupervisor(supervisor.config)
    result = restarted._recover_retained_generated_checkout_lease()

    assert result["recovered"] is True
    assert result["retained_lease"] is False
    assert result["adoption"]["adopted"] is True
    assert not checkout_mutation_lock_path(repo).exists()
    assert _git(repo, "status", "--porcelain", "--", "tasks.todo.md") == ""
    assert _git(repo, "log", "-1", "--pretty=%ae") == (
        BACKLOG_REFINERY_AUTHOR_EMAIL
    )


@pytest.mark.parametrize("commit_parent_gitlink", [False, True])
def test_generated_protected_release_guard_covers_submodule_and_gitlink(
    tmp_path: Path,
    commit_parent_gitlink: bool,
) -> None:
    child_source = tmp_path / "child-source"
    child_source.mkdir()
    _git(child_source, "init")
    child_todo = child_source / "tasks.todo.md"
    child_todo.write_text("# Tasks\n", encoding="utf-8")
    _git(child_source, "add", "tasks.todo.md")
    _git(
        child_source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "initial child",
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "deps/child",
    )
    _git(repo, "add", ".gitmodules", "deps/child")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "initial parent",
    )
    todo_path = repo / "deps/child/tasks.todo.md"
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(tmp_path / "state"),
            "--implementation-protected-path",
            "deps/child/tasks.todo.md",
        ]
    )
    supervisor = PortalImplementationSupervisor(
        supervisor_config_from_args(args, repo_root=repo)
    )
    subject = generated_protected_board_commit_subject("submodule update")

    def producer() -> list[str]:
        todo_path.write_text("# Tasks\n\n## EX-002 Child\n", encoding="utf-8")
        _git(todo_path.parent, "add", "tasks.todo.md")
        _git(
            todo_path.parent,
            "-c",
            "user.name=Agent Supervisor",
            "-c",
            f"user.email={BACKLOG_REFINERY_AUTHOR_EMAIL}",
            "commit",
            "-m",
            subject,
        )
        if commit_parent_gitlink:
            _git(repo, "add", "deps/child")
            _git(
                repo,
                "-c",
                "user.name=Agent Supervisor",
                "-c",
                f"user.email={BACKLOG_REFINERY_AUTHOR_EMAIL}",
                "commit",
                "-m",
                subject,
            )
        return ["EX-002"]

    if commit_parent_gitlink:
        assert supervisor._run_generated_board_producer(
            producer="submodule-test",
            commit_outputs=True,
            callback=producer,
        ) == ["EX-002"]
        assert not checkout_mutation_lock_path(repo).exists()
    else:
        with pytest.raises(
            RuntimeError,
            match="protected_generated_outputs_dirty",
        ):
            supervisor._run_generated_board_producer(
                producer="submodule-test",
                commit_outputs=True,
                callback=producer,
            )
        events = [
            json.loads(line)
            for line in supervisor.config.events_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        scopes = events[-1]["release_guard"]["scope_results"]
        assert {Path(scope["git_root"]) for scope in scopes} == {
            repo.resolve(),
            todo_path.parent.resolve(),
        }
        parent_scope = next(
            scope for scope in scopes if Path(scope["git_root"]) == repo.resolve()
        )
        assert parent_scope["reason"] == "protected_generated_outputs_dirty"


def test_maintenance_recovers_retained_checkout_before_other_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    phases: list[str] = []
    monkeypatch.setattr(
        supervisor,
        "_recover_retained_generated_checkout_lease",
        lambda: {
            "attempted": True,
            "recovered": False,
            "retained_lease": True,
        },
    )
    monkeypatch.setattr(
        supervisor,
        "ensure_event_log_file",
        lambda: pytest.fail("maintenance mutated state before recovery"),
    )

    result = supervisor._run_once_with_maintenance_under_lease(
        phases.append,
        include_refill=False,
    )

    assert phases == ["retained_generated_checkout_recovery"]
    assert result["maintenance_blocked"] is True
    assert result["reason"] == "checkout_mutation_protected_recovery_required"


def test_supervisor_commits_resolved_guardrail_retirement_to_protected_board(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text(
        """# Tasks

## EX-001 Ready source

- Status: todo
- Depends on:
- Outputs: src/example.py

## EX-002 Resolve dependency guardrail for EX-001

- Status: todo
- Depends on:
- Outputs: tasks.todo.md
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
            str(repo / "state"),
            "--task-prefix",
            "## EX-",
            "--implementation-protected-path",
            "tasks.todo.md",
        ]
    )
    supervisor = PortalImplementationSupervisor(
        supervisor_config_from_args(args, repo_root=repo)
    )
    supervisor.config.strategy_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor.config.strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["EX-001"],
                "dependency_guardrail_findings": [],
            }
        ),
        encoding="utf-8",
    )

    releases = supervisor.release_completed_guardrail_blocks()

    assert releases == [
        {
            "source_task_id": "EX-001",
            "follow_up_task_id": "EX-002",
            "guardrail_kind": "dependency_guardrail",
            "reason": "resolved_repair_task_retired",
        }
    ]
    assert "- Status: completed" in todo_path.read_text(encoding="utf-8").split(
        "## EX-002", 1
    )[1]
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


def test_supervisor_live_daemon_lock_blocks_objective_refill_before_mutation(
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
        include_refill=True,
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

    def maintenance_body(
        _update_phase,
        *,
        include_refill: bool,
        implementation_maintenance_lease=None,
    ):
        observed.update(json.loads(lock_path.read_text(encoding="utf-8")))
        assert json.loads(shared_lock_path.read_text(encoding="utf-8"))[
            "lease_role"
        ] == "shared_protected_path_maintenance"
        assert include_refill is False
        assert implementation_maintenance_lease == observed
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

    def failing_maintenance(
        _update_phase,
        *,
        include_refill: bool,
        implementation_maintenance_lease=None,
    ):
        assert include_refill is False
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        assert implementation_maintenance_lease == metadata
        assert metadata["lease_role"] == "supervisor_maintenance"
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


def test_supervisor_preflight_does_not_repair_checkout_before_shared_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)

    class StoppedLoop:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(
                status="stopped",
                restart_count=0,
                last_exit_code=None,
                last_recycle_reason="",
                last_run_id="",
                last_log_path="",
            )

    monkeypatch.setattr(
        supervisor,
        "ensure_event_log_file",
        lambda: {"repaired": False},
    )
    monkeypatch.setattr(
        supervisor,
        "ensure_managed_daemon_pid_file",
        lambda: {"adopted": False},
    )
    monkeypatch.setattr(
        supervisor,
        "repair_main_checkout_merge_state",
        lambda: pytest.fail(
            "checkout repair must run only inside run_once's shared lease"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "run_once",
        lambda *, include_refill=True: {
            "stuck": False,
            "include_refill": include_refill,
        },
    )
    supervisor.shared_supervisor_loop_class = StoppedLoop

    supervisor._run_forever_loop()


def test_supervisor_maintenance_lease_uses_effective_state_path_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective_state_path = tmp_path / "effective-state" / "task-state.json"
    supervisor = _supervisor(tmp_path, state_path=effective_state_path)
    effective_lock_path = effective_state_path.parent / "implementation.lock"
    configured_lock_path = tmp_path / "state" / "implementation.lock"

    def maintenance_body(
        _update_phase,
        *,
        include_refill: bool,
        implementation_maintenance_lease=None,
    ):
        assert include_refill is False
        metadata = json.loads(effective_lock_path.read_text(encoding="utf-8"))
        assert implementation_maintenance_lease == metadata
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
    event_log_lock_path = daemon.events_path.with_name(
        f".{daemon.events_path.name}.lock"
    )
    lane_event_log_lock_path = (
        tmp_path / "state" / ".lane_supervisor_events.jsonl.lock"
    )
    generic_lock_path = tmp_path / "state" / "merge-repair.lock"
    implementation_lock_path.parent.mkdir(parents=True)
    for path in (
        implementation_lock_path,
        update_guard_path,
        event_log_lock_path,
        lane_event_log_lock_path,
        generic_lock_path,
    ):
        path.write_text("stale\n", encoding="utf-8")
        os.utime(path, (1, 1))

    result = daemon._cleanup_stale_locks(max_age_seconds=1)

    assert implementation_lock_path.exists()
    assert update_guard_path.exists()
    assert event_log_lock_path.exists()
    assert lane_event_log_lock_path.exists()
    assert generic_lock_path.exists()
    managed = {
        item["lock_path"]
        for item in result["skipped"]
        if item.get("reason") == "managed_by_implementation_lease_protocol"
    }
    assert managed == {
        str(implementation_lock_path),
        str(update_guard_path),
    }
    event_log_managed = {
        item["lock_path"]
        for item in result["skipped"]
        if item.get("reason") == "managed_by_event_log_flock_protocol"
    }
    assert event_log_managed == {
        str(event_log_lock_path),
        str(lane_event_log_lock_path),
    }
    assert {
        item["lock_path"]
        for item in result["skipped"]
        if item.get("reason") == "persistent_state_flock"
    } == {str(generic_lock_path)}


def test_stale_lock_cleanup_preserves_flocks_and_removes_git_transaction_locks(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    state_dir = tmp_path / "state"
    git_dir = tmp_path / ".git"
    git_ref_dir = git_dir / "refs" / "heads"
    state_dir.mkdir(parents=True)
    git_ref_dir.mkdir(parents=True)
    event_lock_paths = (
        state_dir / ".events.jsonl.lock",
        state_dir / ".portal_supervisor_events.jsonl.lock",
    )
    transient_git_lock_paths = (
        git_dir / "index.lock",
        git_ref_dir / "main.lock",
    )
    persistent_git_flock_path = git_dir / "agent-llm-resolver.lock"
    for path in (
        *event_lock_paths,
        *transient_git_lock_paths,
        persistent_git_flock_path,
    ):
        path.write_text("stale\n", encoding="utf-8")
        os.utime(path, (1, 1))

    result = daemon._cleanup_stale_locks(max_age_seconds=1)

    assert all(path.exists() for path in event_lock_paths)
    assert persistent_git_flock_path.exists()
    assert not any(path.exists() for path in transient_git_lock_paths)
    assert {
        item["lock_path"]
        for item in result["skipped"]
        if item.get("reason") == "managed_by_event_log_flock_protocol"
    } == {str(path) for path in event_lock_paths}
    assert {item["lock_path"] for item in result["removed"]} == {
        str(path) for path in transient_git_lock_paths
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


def test_crash_fence_reconciliation_defers_without_clearance_under_maintenance(
    tmp_path: Path,
) -> None:
    """Active maintenance must not produce a false clear of a crash snapshot."""

    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    task = _task(outputs=["src/example.py"])
    daemon._require_implementation_protected_snapshot(
        task=task,
        attempt=1,
        workspace_path=tmp_path,
    )
    active_path = daemon._implementation_protected_active_snapshot_path()
    assert active_path.exists()
    supervisor = _supervisor(tmp_path)
    lease, guard = supervisor._acquire_protected_path_maintenance_lease()
    assert lease is not None
    assert guard["blocked"] is False

    try:
        result = daemon._reconcile_implementation_protected_path_fence()
    finally:
        supervisor._release_protected_path_maintenance_lease(lease)

    assert result["blocked"] is True
    assert result["deferred"] is True
    assert result["reason"] == "crash_reconciliation_deferred_maintenance_active"
    assert result["scan_outside_lease"] is True
    assert result["critical_section_entered"] is False
    assert active_path.exists()
    assert not daemon._implementation_protected_incident_path().exists()


def test_crash_fence_reconciliation_defers_auto_clear_under_maintenance(
    tmp_path: Path,
) -> None:
    """Active maintenance must not auto-clear a latched incident either."""

    worktrees = tmp_path / "worktrees"
    workspace = worktrees / "workspace-ephemeral"
    worktrees.mkdir()
    # Workspace already gone; shared protected path remains intact.
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
    supervisor = _supervisor(tmp_path)
    lease, guard = supervisor._acquire_protected_path_maintenance_lease()
    assert lease is not None

    try:
        result = daemon._reconcile_implementation_protected_path_fence()
    finally:
        supervisor._release_protected_path_maintenance_lease(lease)

    assert result["deferred"] is True
    assert result["reason"] == "crash_reconciliation_deferred_maintenance_active"
    assert daemon._implementation_protected_incident_path().exists()
    assert result.get("cleared") is not True


def test_crash_fence_reconciliation_scans_outside_exclusive_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    lease_held_during_scan: list[bool] = []
    original_snapshot = daemon._implementation_protected_path_snapshot

    def tracking_snapshot(workspace_path: Path):
        recon_lock = crash_fence_reconciliation_lock_path(tmp_path)
        lease_held_during_scan.append(recon_lock.exists())
        return original_snapshot(workspace_path)

    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_snapshot",
        tracking_snapshot,
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is False
    assert result["reason"] == "crash_reconciliation_unchanged"
    proof = result["reconciliation_proof"]
    assert proof["scan_outside_lease"] is True
    assert proof["critical_section_entered"] is True
    assert proof["lease_hold_bounded"] is True
    assert float(proof["critical_section"]["hold_seconds"]) <= float(
        proof["max_hold_seconds"]
    )
    assert lease_held_during_scan
    assert all(held is False for held in lease_held_during_scan)
    assert not daemon._implementation_protected_active_snapshot_path().exists()


def test_crash_fence_reconciliation_revalidates_inputs_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("before\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    daemon._require_implementation_protected_snapshot(
        task=_task(outputs=["src/example.py"]),
        attempt=1,
        workspace_path=tmp_path,
    )
    active_path = daemon._implementation_protected_active_snapshot_path()
    original_apply = CrashFenceReconciler._apply_under_lease

    def mutate_then_apply(self, plan, *, incident_path, active_path):
        # Simulate a concurrent writer changing the durable fence between the
        # outside scan and the exclusive critical section.
        payload = json.loads(active_path.read_text(encoding="utf-8"))
        payload["task_id"] = "MUTATED-AFTER-SCAN"
        active_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return original_apply(
            self,
            plan,
            incident_path=incident_path,
            active_path=active_path,
        )

    monkeypatch.setattr(CrashFenceReconciler, "_apply_under_lease", mutate_then_apply)

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["deferred"] is True
    assert result["reason"] == "crash_reconciliation_input_changed"
    assert active_path.exists()
    assert json.loads(active_path.read_text(encoding="utf-8"))["task_id"] == (
        "MUTATED-AFTER-SCAN"
    )
    assert not daemon._implementation_protected_incident_path().exists()


def test_crash_fence_reconciliation_reclaims_stale_serialization_lease(
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
    lock_path = crash_fence_reconciliation_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "kind": "crash-fence-reconciliation",
                "lease_id": "stale-lease",
                "pid": 0,
                "owner_script": "dead-reconciler.py",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is False
    assert result["reason"] == "crash_reconciliation_unchanged"
    assert result["reconciliation_proof"]["lease_hold_bounded"] is True
    assert not lock_path.exists()
    assert not daemon._implementation_protected_active_snapshot_path().exists()


def test_crash_fence_reconciliation_fails_closed_on_malformed_serialization_lease(
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
    active_path = daemon._implementation_protected_active_snapshot_path()
    lock_path = crash_fence_reconciliation_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{malformed", encoding="utf-8")

    result = daemon._reconcile_implementation_protected_path_fence()

    assert result["blocked"] is True
    assert result["deferred"] is True
    assert result["reason"] == "crash_reconciliation_serialized"
    assert result["lease_reason"] == "checkout_maintenance_lease_malformed"
    assert active_path.exists()
    assert lock_path.exists()


def test_checkout_maintenance_lease_bounds_hold_time(tmp_path: Path) -> None:
    lock_path = tmp_path / "git" / "implementation-protected-path-crash-fence-recon.lock"
    lease = CheckoutMaintenanceLease(
        lock_path,
        metadata={
            "kind": "crash-fence-reconciliation",
            "lease_role": "unit-test",
            "repo_root": str(tmp_path.resolve()),
        },
        max_hold_seconds=0.05,
    )
    with pytest.raises(RuntimeError, match="hold exceeded bound"):
        with lease.exclusive_section() as timing:
            time.sleep(0.08)
            lease.ensure_within_hold_bound()
            timing["should_not_reach"] = True

    assert not lock_path.exists()
    assert lease.hold_seconds is not None
    assert lease.hold_seconds >= 0.05


def test_durable_input_generation_detects_content_change(tmp_path: Path) -> None:
    path = tmp_path / "fence.json"
    path.write_text('{"v":1}\n', encoding="utf-8")
    before = durable_input_generation(path)
    path.write_text('{"v":2}\n', encoding="utf-8")
    after = durable_input_generation(path)
    assert before["state"] == "present"
    assert after["state"] == "present"
    assert not generations_match(before, after)
    missing = durable_input_generation(tmp_path / "absent.json")
    assert missing["state"] == "missing"


def test_crash_fence_reconciler_clears_unchanged_snapshot_with_proof(
    tmp_path: Path,
) -> None:
    protected = tmp_path / POLICY_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("stable\n", encoding="utf-8")
    daemon = _daemon(tmp_path)
    daemon._require_implementation_protected_snapshot(
        task=_task(outputs=["src/example.py"]),
        attempt=2,
        workspace_path=tmp_path,
    )

    result = CrashFenceReconciler(daemon).reconcile()

    assert result["blocked"] is False
    assert result["reason"] == "crash_reconciliation_unchanged"
    assert result["task_id"] == "EX-001"
    assert result["attempt"] == 2
    proof = result["reconciliation_proof"]
    assert proof["scan_outside_lease"] is True
    assert proof["critical_section_entered"] is True
    assert proof["lease_hold_bounded"] is True
    assert proof["critical_section"]["within_bound"] is True
    assert float(proof["critical_section"]["hold_seconds"]) <= float(
        proof["max_hold_seconds"]
    )
    assert not daemon._implementation_protected_active_snapshot_path().exists()
