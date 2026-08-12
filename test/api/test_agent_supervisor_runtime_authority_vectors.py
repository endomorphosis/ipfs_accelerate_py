"""Independent vectors for content and process runtime authorities."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge import leased_lane as leased_lane_module
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import profile_g_cid
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import LeaseCoordinator
from ipfs_accelerate_py.agent_supervisor.merge.leased_lane import (
    _capture_spawned_direct_child_start_time,
    run_leased_lane_result,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_dag_json,
    validate_cid,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    PortalTaskState,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import terminate_pid_tree
from ipfs_accelerate_py.agent_supervisor import worktree_lifecycle as lifecycle_module
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    DuplicateAttemptError,
    OwnershipError,
    ProcessBirthIdentity,
    WorkspaceLifecycleRecord,
    WorktreeLifecycleStore,
    current_process_birth,
    read_process_birth,
)

KNOWN_RUNTIME_AUTHORITY_CID = "baguqeeracqtgb767f7cbbix3olmiskqv5bpva2aqp7darhmzwomawygz5ila"


class _FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _repository(path: Path) -> str:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Runtime Authority Test")
    _git(path, "config", "user.email", "runtime-authority@example.invalid")
    (path / "src").mkdir()
    (path / "src" / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "baseline")
    return _git(path, "rev-parse", "HEAD")


def _daemon(
    repo: Path,
    *,
    worktree_submodule_paths: tuple[str, ...] = (),
) -> TodoImplementationDaemon:
    runtime = repo.parent / f".{repo.name}-runtime"
    return TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=runtime / "state.json",
        strategy_path=runtime / "strategy.json",
        events_path=runtime / "events.jsonl",
        repo_root=repo,
        worktree_root=runtime / "worktrees",
        validation_cache_dir=runtime / "validation-cache",
        merge_queue_dir=runtime / "merge-queue",
        worktree_submodule_paths=worktree_submodule_paths,
        worktree_pool_enabled=False,
    )


def _diff_task(*outputs: str) -> PortalTask:
    return PortalTask(
        task_id="AUTH-DIFF",
        title="candidate whitespace authority",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime-authority",
        outputs=list(outputs),
        validation=["git diff --check"],
    )


def test_profile_g_runtime_authority_matches_real_cidv1_vector() -> None:
    artifact = {"schema": "runtime-authority@1", "fence": 7}

    assert profile_g_cid(artifact) == KNOWN_RUNTIME_AUTHORITY_CID
    assert cid_for_dag_json(artifact) == KNOWN_RUNTIME_AUTHORITY_CID
    assert (
        validate_cid(KNOWN_RUNTIME_AUTHORITY_CID, codecs=("dag-json",))
        == KNOWN_RUNTIME_AUTHORITY_CID
    )


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="process-birth authority vector requires Linux /proc",
)
def test_naturally_exited_direct_child_retains_empty_group_authority() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        birth = read_process_birth(child.pid)
        assert birth is not None
        assert birth.parent_pid == os.getpid()
        child.wait(timeout=3.0)

        assert terminate_pid_tree(
            child.pid,
            grace_seconds=0.0,
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=child.pid,
            expected_root_start_time_ticks=birth.start_time_ticks,
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1.0)


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="zombie birth capture requires Linux /proc",
)
def test_fast_zombie_child_birth_is_captured_before_reap() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 3.0
        state = ""
        while time.monotonic() < deadline:
            raw = Path(f"/proc/{child.pid}/stat").read_text(encoding="utf-8")
            close = raw.rfind(")")
            state = raw[close + 2 :].split()[0]
            if state == "Z":
                break
            time.sleep(0.005)
        assert state == "Z"
        start_time = _capture_spawned_direct_child_start_time(
            child.pid,
            expected_parent_pid=os.getpid(),
        )
        assert start_time is not None and start_time > 0
        child.wait(timeout=3.0)
        assert terminate_pid_tree(
            child.pid,
            grace_seconds=0.0,
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=child.pid,
            expected_root_start_time_ticks=start_time,
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1.0)


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="fast-child leased success requires Linux /proc",
)
def test_fast_child_leased_lane_success_closes_grant_and_releases_capacity(
    tmp_path: Path,
) -> None:
    coordination = tmp_path / "coordination.sqlite3"
    with LeaseCoordinator(coordination) as coordinator:
        task = coordinator.register_bundle(
            {"bundle_key": "authority/fast-success", "tasks": [{"task_id": "FAST"}]}
        )
        grant = coordinator.claim(
            task["task_cid"],
            "did:web:fast-success.example",
        )

    result = run_leased_lane_result(
        coordination_path=coordination,
        grant=grant,
        command=[sys.executable, "-c", "pass"],
        lease_ms=60_000,
        heartbeat_interval=0.01,
    )

    assert result.disposition == "completed"
    assert result.successful is True
    assert result.exit_code == 0
    assert result.child_exit_code == 0
    assert result.receipt_cid
    assert result.lease_released is True
    assert result.reusable is True
    with LeaseCoordinator(coordination) as coordinator:
        assert coordinator.active_lease(task["task_cid"]) is None
        receipts = coordinator.list_receipts(task["task_cid"])
    assert any(
        receipt.get("receipt_cid") == result.receipt_cid
        for receipt in receipts
    )


def test_unproven_lane_process_fence_returns_typed_failure_without_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordination = tmp_path / "coordination.sqlite3"
    with LeaseCoordinator(coordination) as coordinator:
        task = coordinator.register_bundle(
            {"bundle_key": "authority/fence", "tasks": [{"task_id": "FENCE"}]}
        )
        grant = coordinator.claim(task["task_cid"], "did:web:authority.example")
    monkeypatch.setattr(
        leased_lane_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: False,
    )

    result = run_leased_lane_result(
        coordination_path=coordination,
        grant=grant,
        command=[sys.executable, "-c", "pass"],
        lease_ms=60_000,
        heartbeat_interval=0.01,
    )

    assert result.disposition == "failed"
    assert result.lease_released is False
    assert result.reusable is False
    assert result.receipt_cid is None
    assert result.error.startswith("process_fence_unproven:")
    with LeaseCoordinator(coordination) as coordinator:
        assert coordinator.validate(grant).fencing_token == grant.fencing_token
        coordinator.release(grant)


@pytest.mark.skipif(os.name != "posix", reason="advisory lock ordering requires POSIX")
def test_live_expired_owner_and_replacement_opt_out_both_fail_closed(
    tmp_path: Path,
) -> None:
    clock = _FakeClock()
    store = WorktreeLifecycleStore(
        repo_root=tmp_path,
        store_dir=tmp_path / "lifecycle",
        lease_seconds=5.0,
        clock=clock,
    )
    workspace = tmp_path / "workspace"
    record = store.begin_preparing(
        task_id="LIVE",
        canonical_task_cid="cid:live",
        attempt=1,
        lane_id="lane-a",
        workspace_path=workspace,
        branch="implementation/live",
        merge_target="main",
        owner=current_process_birth(),
    )
    clock.advance(6.0)

    assert store.reclaim_stale(workspace) is None
    with pytest.raises(DuplicateAttemptError):
        store.begin_preparing(
            task_id="LIVE",
            canonical_task_cid="cid:live",
            attempt=1,
            lane_id="lane-b",
            workspace_path=tmp_path / "replacement",
            branch="implementation/live-b",
            merge_target="main",
        )
    assert store.load_workspace(workspace) == record

    dead_workspace = tmp_path / "dead-workspace"
    dead = store.begin_preparing(
        task_id="DEAD",
        canonical_task_cid="cid:dead",
        attempt=1,
        lane_id="lane-a",
        workspace_path=dead_workspace,
        branch="implementation/dead",
        merge_target="main",
        owner=ProcessBirthIdentity(pid=2**30 - 101, start_time_ticks=1),
    )
    clock.advance(6.0)
    with pytest.raises(DuplicateAttemptError, match="replacement disabled"):
        store.begin_preparing(
            task_id="DEAD",
            canonical_task_cid="cid:dead",
            attempt=1,
            lane_id="lane-b",
            workspace_path=dead_workspace,
            branch="implementation/dead-b",
            merge_target="main",
            allow_replace_stale=False,
        )
    assert store.load_workspace(dead_workspace) == dead


@pytest.mark.skipif(os.name != "posix", reason="advisory lock ordering requires POSIX")
def test_stale_terminal_writer_cannot_overwrite_new_task_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    store = WorktreeLifecycleStore(
        repo_root=tmp_path,
        store_dir=tmp_path / "lifecycle",
        lease_seconds=5.0,
        clock=clock,
    )
    old_workspace = tmp_path / "old"
    old = store.begin_preparing(
        task_id="CAS",
        canonical_task_cid="cid:cas",
        attempt=1,
        lane_id="lane-a",
        workspace_path=old_workspace,
        branch="implementation/cas-a",
        merge_target="main",
        owner=ProcessBirthIdentity(pid=2**30 - 102, start_time_ticks=1),
    )
    clock.advance(6.0)
    index_path = store.task_index_path_for(
        canonical_task_cid=old.canonical_task_cid,
        task_id=old.task_id,
        attempt=old.attempt,
    )
    terminal_write_reached = threading.Event()
    allow_terminal_write = threading.Event()
    replacement_finished = threading.Event()
    # Flat-stem alias modules re-export symbols; method free names resolve via
    # the canonical merge.worktree_lifecycle globals.
    patch_target = getattr(
        lifecycle_module, "__canonical_module__", lifecycle_module
    )
    original_write = patch_target._atomic_write_json

    def paused_write(path: Path, payload: dict[str, object]) -> None:
        if (
            path == index_path
            and payload.get("workspace_path") == old.workspace_path
            and payload.get("state") == "terminal"
        ):
            terminal_write_reached.set()
            assert allow_terminal_write.wait(timeout=5.0)
        original_write(path, payload)

    monkeypatch.setattr(patch_target, "_atomic_write_json", paused_write)
    errors: list[BaseException] = []

    def reclaim() -> None:
        try:
            assert store.reclaim_stale(old_workspace) is not None
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    replacement: dict[str, WorkspaceLifecycleRecord] = {}

    def acquire_replacement() -> None:
        try:
            replacement["record"] = store.begin_preparing(
                task_id="CAS",
                canonical_task_cid="cid:cas",
                attempt=1,
                lane_id="lane-b",
                workspace_path=tmp_path / "new",
                branch="implementation/cas-b",
                merge_target="main",
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            replacement_finished.set()

    reclaim_thread = threading.Thread(target=reclaim)
    replacement_thread = threading.Thread(target=acquire_replacement)
    reclaim_thread.start()
    assert terminal_write_reached.wait(timeout=5.0)
    replacement_thread.start()
    assert not replacement_finished.wait(timeout=0.1)
    allow_terminal_write.set()
    reclaim_thread.join(timeout=5.0)
    replacement_thread.join(timeout=5.0)

    assert not errors
    current = replacement["record"]
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["workspace_path"] == current.workspace_path
    assert payload["record_id"] == current.record_id
    assert payload["state"] == "preparing"


def test_lost_worktree_fence_stops_validation_commit_and_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    workspace = repo
    captured = daemon.worktree_lifecycle.begin_preparing(
        task_id="FENCE-BOUNDARY",
        canonical_task_cid="cid:fence-boundary",
        attempt=1,
        lane_id="lane-a",
        workspace_path=workspace,
        branch="implementation/fence-boundary",
        merge_target="main",
    )
    daemon._active_worktree_lifecycle = captured
    daemon.worktree_lifecycle.renew_lease(
        workspace,
        lease_id=captured.lease_id,
        expected_fence=captured.fence,
    )
    task = _diff_task("src/alpha.py")

    daemon._active_worktree_lifecycle = captured
    with pytest.raises(OwnershipError, match="before validation"):
        daemon._mark_worktree_lifecycle_settling(workspace)

    daemon._active_worktree_lifecycle = captured
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_mutation",
        lambda *_args, **_kwargs: pytest.fail("commit mutation was reached"),
    )
    with pytest.raises(OwnershipError, match="before commit"):
        daemon._commit_worktree_changes(workspace, task, 1)

    daemon._active_worktree_lifecycle = captured
    monkeypatch.setattr(
        daemon,
        "_reject_protected_merge_candidate",
        lambda **_kwargs: pytest.fail("enqueue mutation was reached"),
    )
    with pytest.raises(OwnershipError, match="before enqueue"):
        daemon._enqueue_validated_worktree(
            state=object(),
            task=task,
            attempt=1,
            branch_name="implementation/fence-boundary",
            baseline_ref="HEAD",
            worktree_path=workspace,
            implementation_commit="0" * 40,
            commit_result={},
            validation_result={"passed": True},
        )


def test_stale_daemon_finalization_never_adopts_replacement_owner(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    stale_daemon = _daemon(repo)
    workspace = repo
    captured_a = stale_daemon.worktree_lifecycle.begin_preparing(
        task_id="OWNER-A",
        canonical_task_cid="cid:owner-a",
        attempt=1,
        lane_id="lane-a",
        workspace_path=workspace,
        branch="implementation/owner-a",
        merge_target="main",
    )
    stale_daemon._active_worktree_lifecycle = captured_a

    terminal_a = stale_daemon.worktree_lifecycle.mark_terminal(
        workspace,
        lease_id=captured_a.lease_id,
        expected_fence=captured_a.fence,
        reason="owner_a_released",
    )
    assert stale_daemon.worktree_lifecycle.compare_and_delete(
        workspace,
        expected_fence=terminal_a.fence,
        lease_id=terminal_a.lease_id,
    )
    replacement_b = stale_daemon.worktree_lifecycle.begin_preparing(
        task_id="OWNER-B",
        canonical_task_cid="cid:owner-b",
        attempt=1,
        lane_id="lane-b",
        workspace_path=workspace,
        branch="implementation/owner-b",
        merge_target="main",
    )

    with pytest.raises(OwnershipError, match="before validation"):
        stale_daemon._require_active_worktree_lifecycle(
            workspace,
            boundary="validation",
        )
    assert stale_daemon._active_worktree_lifecycle == captured_a

    finalize = stale_daemon._finalize_worktree_lifecycle(
        workspace,
        reason="stale_owner_cleanup",
    )

    assert finalize["finalized"] is False
    assert finalize["reason"] == "lifecycle_finalize_race"
    observed = stale_daemon.worktree_lifecycle.load_workspace(workspace)
    assert observed is not None
    assert observed.record_id == replacement_b.record_id
    assert observed.lease_id == replacement_b.lease_id
    assert not observed.is_terminal
    assert stale_daemon._active_worktree_lifecycle == captured_a


def test_ephemeral_provider_dispatch_rechecks_exact_lifecycle_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    task = _diff_task("src/alpha.py")
    state = PortalTaskState()
    provider_calls: list[bool] = []
    replacement: dict[str, object] = {}
    original_record_event = daemon._record_event

    monkeypatch.setattr(
        daemon,
        "_task_uses_typed_local_execution",
        lambda _task: False,
    )
    monkeypatch.setattr(
        daemon,
        "_build_implementation_command",
        lambda *_args, **_kwargs: [sys.executable, "-c", "raise SystemExit(0)"],
    )

    def replace_owner_after_active(event_type, payload):
        result = original_record_event(event_type, payload)
        if event_type != "implementation_started" or replacement:
            return result
        captured = daemon._active_worktree_lifecycle
        assert captured is not None
        terminal = daemon.worktree_lifecycle.mark_terminal(
            captured.workspace_path,
            lease_id=captured.lease_id,
            expected_fence=captured.fence,
            reason="provider_boundary_test_takeover",
        )
        assert daemon.worktree_lifecycle.compare_and_delete(
            terminal.workspace_path,
            expected_fence=terminal.fence,
            lease_id=terminal.lease_id,
        )
        replacement["record"] = daemon.worktree_lifecycle.begin_preparing(
            task_id="PROVIDER-B",
            canonical_task_cid="cid:provider-b",
            attempt=1,
            lane_id="lane-b",
            workspace_path=terminal.workspace_path,
            branch="implementation/provider-b",
            merge_target="main",
        )
        return result

    def provider_must_not_run(*_args, **_kwargs):
        provider_calls.append(True)
        raise AssertionError("provider ran after lifecycle takeover")

    monkeypatch.setattr(daemon, "_record_event", replace_owner_after_active)
    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        provider_must_not_run,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=state,
        attempt=1,
        started_at="2026-08-11T00:00:00+00:00",
        log_path=tmp_path / "provider-boundary.log",
        prompt="provider boundary test",
    )

    assert replacement
    assert provider_calls == []
    assert result["provider_dispatched"] is False
    assert result["returncode"] == 1
    assert result["exception_result"]["exception_type"] == "OwnershipError"
    replacement_record = replacement["record"]
    observed = daemon.worktree_lifecycle.load_workspace(
        replacement_record.workspace_path
    )
    assert observed is not None
    assert observed.record_id == replacement_record.record_id
    assert not observed.is_terminal


def test_provider_environment_window_rechecks_token_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    task = _diff_task("src/alpha.py")
    state = PortalTaskState()
    environment = daemon._implementation_process_environment
    provider_calls: list[bool] = []
    replacement: dict[str, WorkspaceLifecycleRecord] = {}

    monkeypatch.setattr(
        daemon,
        "_task_uses_typed_local_execution",
        lambda _task: False,
    )
    monkeypatch.setattr(
        daemon,
        "_build_implementation_command",
        lambda *_args, **_kwargs: [sys.executable, "-c", "pass"],
    )

    def prepare_environment_then_replace(*args, **kwargs):
        result = environment(*args, **kwargs)
        captured = daemon._active_worktree_lifecycle
        assert captured is not None
        terminal = daemon.worktree_lifecycle.mark_terminal(
            captured.workspace_path,
            lease_id=captured.lease_id,
            expected_fence=captured.fence,
            reason="environment_boundary_takeover",
        )
        assert daemon.worktree_lifecycle.compare_and_delete(
            terminal.workspace_path,
            expected_fence=terminal.fence,
            lease_id=terminal.lease_id,
        )
        replacement["record"] = daemon.worktree_lifecycle.begin_preparing(
            task_id="ENVIRONMENT-B",
            canonical_task_cid="cid:environment-b",
            attempt=1,
            lane_id="lane-b",
            workspace_path=terminal.workspace_path,
            branch="implementation/environment-b",
            merge_target="main",
        )
        return result

    def provider_must_not_run(*_args, **_kwargs):
        provider_calls.append(True)
        raise AssertionError("provider spawned after environment-window takeover")

    monkeypatch.setattr(
        daemon,
        "_implementation_process_environment",
        prepare_environment_then_replace,
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        provider_must_not_run,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=state,
        attempt=1,
        started_at="2026-08-11T00:00:00+00:00",
        log_path=tmp_path / "environment-boundary.log",
        prompt="provider environment boundary",
    )

    assert replacement
    assert provider_calls == []
    assert result["provider_dispatched"] is False
    assert result["returncode"] == 1
    assert result["exception_result"]["exception_type"] == "OwnershipError"
    observed = daemon.worktree_lifecycle.load_workspace(
        replacement["record"].workspace_path
    )
    assert observed is not None
    assert observed.record_id == replacement["record"].record_id
    assert not observed.is_terminal


def test_prepare_window_fence_loss_blocks_deterministic_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    task = _diff_task("src/alpha.py")
    prepare = daemon._prepare_worktree_for_validation
    validator_calls: list[bool] = []
    advanced: list[WorkspaceLifecycleRecord] = []

    monkeypatch.setattr(
        daemon,
        "_task_uses_typed_local_execution",
        lambda _task: True,
    )

    def prepare_then_advance(*args, **kwargs):
        result = prepare(*args, **kwargs)
        captured = daemon._active_worktree_lifecycle
        assert captured is not None
        advanced.append(
            daemon.worktree_lifecycle.renew_lease(
                captured.workspace_path,
                lease_id=captured.lease_id,
                expected_fence=captured.fence,
            )
        )
        return result

    def scheduler_must_not_run(*_args, **_kwargs):
        validator_calls.append(True)
        raise AssertionError("validation scheduler ran after fence loss")

    monkeypatch.setattr(
        daemon,
        "_prepare_worktree_for_validation",
        prepare_then_advance,
    )
    monkeypatch.setattr(
        daemon.validation_scheduler,
        "run",
        scheduler_must_not_run,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=PortalTaskState(),
        attempt=1,
        started_at="2026-08-11T00:00:00+00:00",
        log_path=tmp_path / "deterministic-validation-boundary.log",
        prompt="",
    )

    assert advanced
    assert validator_calls == []
    assert result["returncode"] == 1
    assert result["exception_result"]["exception_type"] == "OwnershipError"


def test_timeout_salvage_prepare_window_blocks_model_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    task = _diff_task("src/alpha.py")
    prepare = daemon._prepare_worktree_for_validation
    validator_calls: list[bool] = []
    advanced: list[WorkspaceLifecycleRecord] = []

    monkeypatch.setattr(
        daemon,
        "_task_uses_typed_local_execution",
        lambda _task: False,
    )
    monkeypatch.setattr(
        daemon,
        "_build_implementation_command",
        lambda *_args, **_kwargs: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("provider", 1.0)
        ),
    )

    def prepare_then_advance(*args, **kwargs):
        result = prepare(*args, **kwargs)
        captured = daemon._active_worktree_lifecycle
        assert captured is not None
        advanced.append(
            daemon.worktree_lifecycle.renew_lease(
                captured.workspace_path,
                lease_id=captured.lease_id,
                expected_fence=captured.fence,
            )
        )
        return result

    def scheduler_must_not_run(*_args, **_kwargs):
        validator_calls.append(True)
        raise AssertionError("timeout validation scheduler ran after fence loss")

    monkeypatch.setattr(
        daemon,
        "_prepare_worktree_for_validation",
        prepare_then_advance,
    )
    monkeypatch.setattr(
        daemon.validation_scheduler,
        "run",
        scheduler_must_not_run,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=PortalTaskState(),
        attempt=1,
        started_at="2026-08-11T00:00:00+00:00",
        log_path=tmp_path / "timeout-validation-boundary.log",
        prompt="timeout",
    )

    assert advanced
    assert validator_calls == []
    assert result["returncode"] == 124
    assert result["timeout_result"]["salvaged"] is False


@pytest.mark.parametrize("proposal_bound", (False, True))
def test_validation_scheduler_dispatch_requires_exact_lifecycle_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal_bound: bool,
) -> None:
    repo = tmp_path / "repo"
    baseline = _repository(repo)
    daemon = _daemon(repo)
    task = _diff_task("src/alpha.py")
    captured = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid="cid:validation-dispatch",
        attempt=1,
        lane_id="lane-a",
        workspace_path=repo,
        branch="implementation/validation-dispatch",
        merge_target="main",
    )
    daemon._active_worktree_lifecycle = captured
    advanced = daemon.worktree_lifecycle.renew_lease(
        captured.workspace_path,
        lease_id=captured.lease_id,
        expected_fence=captured.fence,
    )
    scheduler_calls: list[str] = []

    def scheduler_must_not_run(*_args, **_kwargs):
        scheduler_calls.append("run")
        raise AssertionError("validation scheduler ran with a stale fence")

    monkeypatch.setattr(
        daemon.validation_scheduler,
        "run",
        scheduler_must_not_run,
    )
    monkeypatch.setattr(
        daemon.validation_scheduler,
        "run_validated",
        scheduler_must_not_run,
    )
    proposal_validation = None
    if proposal_bound:
        proposal_validation = SimpleNamespace(
            accepted=True,
            proposal=SimpleNamespace(
                repository_tree_id=baseline,
                changed_paths=("src/alpha.py",),
                candidate_diff=(),
                proposal_id="proposal:validation-dispatch",
            ),
        )

    with pytest.raises(OwnershipError, match="before effect"):
        daemon._run_validation_commands(
            repo,
            task,
            tmp_path / "validation-dispatch.log",
            state=PortalTaskState(),
            proposal_validation=proposal_validation,
            baseline_ref=baseline,
            lifecycle_record=captured,
        )

    assert scheduler_calls == []
    observed = daemon.worktree_lifecycle.load_workspace(repo)
    assert observed is not None
    assert observed.fence == advanced.fence


def test_all_lifecycle_validation_routes_forward_captured_token() -> None:
    routes = {
        "_execute_deterministic_validation_plan": {
            "_run_clean_candidate_validation",
            "_run_validation_commands",
        },
        "_admit_deterministic_validation_materialization": {
            "_run_validation_with_candidate_binding",
        },
        "_run_clean_candidate_validation": {"_run_validation_commands"},
        "_run_validation_with_candidate_binding": {
            "_run_clean_candidate_validation",
            "_run_validation_commands",
        },
        "_apply_implementation_failure_review": {
            "_run_validation_commands",
        },
        "_restore_and_verify_post_validation_candidate": {
            "_run_validation_commands",
        },
    }
    for method_name, callees in routes.items():
        method = getattr(TodoImplementationDaemon, method_name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in callees
        ]
        assert calls, method_name
        assert all(
            any(keyword.arg == "lifecycle_record" for keyword in call.keywords)
            for call in calls
        ), method_name

    lifecycle_entrypoints = {
        "_run_implementation_in_ephemeral_worktree": {
            "_execute_deterministic_validation_plan",
            "_admit_deterministic_validation_materialization",
            "_run_validation_with_candidate_binding",
            "_apply_implementation_failure_review",
            "_restore_and_verify_post_validation_candidate",
        },
        "_run_manual_completion_authority_revalidation": {
            "_execute_deterministic_validation_plan",
        },
    }
    for method_name, callees in lifecycle_entrypoints.items():
        method = getattr(TodoImplementationDaemon, method_name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in callees
        ]
        assert calls, method_name
        assert all(
            any(keyword.arg == "lifecycle_record" for keyword in call.keywords)
            for call in calls
        ), method_name


def test_no_change_completion_requires_exact_cleanup_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    task = _diff_task("src/alpha.py")
    state = PortalTaskState()
    cleanup = daemon._cleanup_merged_worktree
    advanced: list[WorkspaceLifecycleRecord] = []

    monkeypatch.setattr(
        daemon,
        "_task_uses_typed_local_execution",
        lambda _task: False,
    )
    monkeypatch.setattr(
        daemon,
        "_build_implementation_command",
        lambda *_args, **_kwargs: [sys.executable, "-c", "pass"],
    )

    def advance_before_cleanup(*args, **kwargs):
        if not advanced:
            captured = daemon._active_worktree_lifecycle
            assert captured is not None
            advanced.append(
                daemon.worktree_lifecycle.renew_lease(
                    captured.workspace_path,
                    lease_id=captured.lease_id,
                    expected_fence=captured.fence,
                )
            )
        return cleanup(*args, **kwargs)

    monkeypatch.setattr(
        daemon,
        "_cleanup_merged_worktree",
        advance_before_cleanup,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=state,
        attempt=1,
        started_at="2026-08-11T00:00:00+00:00",
        log_path=tmp_path / "no-change-cleanup.log",
        prompt="make no changes",
    )

    assert advanced
    assert result["commit_result"]["reason"] == "no_changes"
    assert result["commit_result"]["no_change_guard"]["allowed"] is True
    assert result["cleanup_result"]["cleaned"] is False
    assert result["cleanup_result"]["lifecycle_finalize"]["finalized"] is False
    assert result["returncode"] == 1
    assert result["validation_result"]["passed"] is False
    assert result["validation_result"]["reason"] == (
        "no_change_lifecycle_finalization_failed"
    )
    assert result["board_completion"]["complete"] is False
    assert "todo_update_result" not in result
    observed = daemon.worktree_lifecycle.load_workspace(
        result["worktree_path"]
    )
    assert observed is not None
    assert observed.fence == advanced[0].fence
    assert not observed.is_terminal


def test_verification_deferral_never_adopts_same_lease_fence_replacement(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    task = _diff_task("src/alpha.py")
    captured_a = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid="cid:retained-a",
        attempt=1,
        lane_id="lane-a",
        workspace_path=repo,
        branch="implementation/retained-a",
        merge_target="main",
    )
    daemon._active_worktree_lifecycle = captured_a
    terminal_a = daemon.worktree_lifecycle.mark_terminal(
        repo,
        lease_id=captured_a.lease_id,
        expected_fence=captured_a.fence,
        reason="owner_a_released",
    )
    assert daemon.worktree_lifecycle.compare_and_delete(
        repo,
        expected_fence=terminal_a.fence,
        lease_id=terminal_a.lease_id,
    )
    replacement_b = daemon.worktree_lifecycle.begin_preparing(
        task_id="RETAINED-B",
        canonical_task_cid="cid:retained-b",
        attempt=1,
        lane_id="lane-b",
        workspace_path=repo,
        branch="implementation/retained-b",
        merge_target="main",
        lease_id=captured_a.lease_id,
    )
    assert replacement_b.fence == captured_a.fence
    assert replacement_b.lease_id == captured_a.lease_id
    assert replacement_b.record_id != captured_a.record_id

    result = daemon._retain_verification_deferred_worktree(
        repo,
        "implementation/retained-a",
        task,
        1,
        {"verification_deferred": True},
    )

    assert result["lifecycle"]["terminal"] is False
    assert result["lifecycle"]["reason"] == (
        "lifecycle_identity_or_fence_changed"
    )
    observed = daemon.worktree_lifecycle.load_workspace(repo)
    assert observed is not None
    assert observed.record_id == replacement_b.record_id
    assert not observed.is_terminal
    assert daemon._active_worktree_lifecycle == captured_a


def test_physical_cleanup_holds_exact_fence_until_worktree_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _repository(repo)
    daemon = _daemon(repo)
    workspace = tmp_path / "cleanup-worktree"
    branch_a = "implementation/cleanup-a"
    _git(repo, "worktree", "add", "-b", branch_a, str(workspace), "HEAD")
    captured_a = daemon.worktree_lifecycle.begin_preparing(
        task_id="CLEANUP-A",
        canonical_task_cid="cid:cleanup-a",
        attempt=1,
        lane_id="lane-a",
        workspace_path=workspace,
        branch=branch_a,
        merge_target="main",
    )
    daemon._active_worktree_lifecycle = captured_a
    replacement_started = threading.Event()
    replacement_finished = threading.Event()
    replacement: dict[str, WorkspaceLifecycleRecord] = {}
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []

    def acquire_replacement() -> None:
        replacement_started.set()
        try:
            replacement["record"] = daemon.worktree_lifecycle.begin_preparing(
                task_id="CLEANUP-B",
                canonical_task_cid="cid:cleanup-b",
                attempt=1,
                lane_id="lane-b",
                workspace_path=workspace,
                branch="implementation/cleanup-b",
                merge_target="main",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            replacement_finished.set()

    def start_replacement_under_cleanup_lock(*_args, **_kwargs):
        if not threads:
            thread = threading.Thread(target=acquire_replacement)
            threads.append(thread)
            thread.start()
            assert replacement_started.wait(timeout=5.0)
            assert not replacement_finished.wait(timeout=0.1)
        return []

    monkeypatch.setattr(
        daemon,
        "_cleanup_worktree_submodules",
        start_replacement_under_cleanup_lock,
    )

    cleanup = daemon._cleanup_merged_worktree(
        workspace,
        branch_a,
        reusable=False,
    )
    assert threads
    threads[0].join(timeout=5.0)

    assert not errors
    assert replacement_finished.is_set()
    assert cleanup["cleaned"] is True
    assert cleanup["removed_worktree"] is True
    assert cleanup["lifecycle_finalize"]["finalized"] is True
    assert not workspace.exists()
    observed = daemon.worktree_lifecycle.load_workspace(workspace)
    assert observed is not None
    assert observed.record_id == replacement["record"].record_id
    assert not observed.is_terminal


def test_isolated_candidate_diff_check_covers_untracked_and_preserves_index(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    baseline = _repository(repo)
    daemon = _daemon(repo)
    task = _diff_task("new.py")
    (repo / "src" / "alpha.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/alpha.py")
    live_index_before = _git(repo, "diff", "--cached", "--binary")
    (repo / "new.py").write_text("VALUE = 3  \n", encoding="utf-8")

    failed = daemon._enforce_baseline_diff_check(
        workspace_path=repo,
        task=task,
        baseline_ref=baseline,
        validation_result={"attempted": True, "passed": True, "results": []},
    )

    assert failed["passed"] is False
    assert failed["reason"] == "candidate_diff_check_failed"
    assert failed["candidate_diff_check"]["materialization"] == (
        "isolated_temporary_index"
    )
    assert "new.py:1: trailing whitespace" in failed["results"][-1]["output"]
    assert _git(repo, "diff", "--cached", "--binary") == live_index_before

    (repo / "new.py").write_text("VALUE = 3\n", encoding="utf-8")
    passed = daemon._enforce_baseline_diff_check(
        workspace_path=repo,
        task=task,
        baseline_ref=baseline,
        validation_result={"attempted": True, "passed": True, "results": []},
    )
    assert passed["passed"] is True
    assert passed["candidate_diff_check"]["passed"] is True
    assert _git(repo, "diff", "--cached", "--binary") == live_index_before


def test_isolated_candidate_diff_check_covers_dirty_child_submodule(
    tmp_path: Path,
) -> None:
    child_source = tmp_path / "child-source"
    _repository(child_source)
    repo = tmp_path / "repo"
    _repository(repo)
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(child_source),
        "deps/child",
    )
    _git(repo, "commit", "-qam", "add child")
    baseline = _git(repo, "rev-parse", "HEAD")
    child = repo / "deps" / "child"
    child_index_before = _git(child, "diff", "--cached", "--binary")
    daemon = _daemon(repo, worktree_submodule_paths=("deps/child",))
    task = _diff_task("deps/child/src/alpha.py")
    (child / "src" / "alpha.py").write_text("VALUE = 4  \n", encoding="utf-8")

    failed = daemon._enforce_baseline_diff_check(
        workspace_path=repo,
        task=task,
        baseline_ref=baseline,
        validation_result={"attempted": True, "passed": True, "results": []},
    )

    assert failed["passed"] is False
    assert "[deps/child] src/alpha.py:1: trailing whitespace" in (
        failed["results"][-1]["output"]
    )
    assert _git(child, "diff", "--cached", "--binary") == child_index_before

    (child / "src" / "alpha.py").write_text("VALUE = 4\n", encoding="utf-8")
    passed = daemon._enforce_baseline_diff_check(
        workspace_path=repo,
        task=task,
        baseline_ref=baseline,
        validation_result={"attempted": True, "passed": True, "results": []},
    )
    assert passed["passed"] is True
    assert _git(child, "diff", "--cached", "--binary") == child_index_before
