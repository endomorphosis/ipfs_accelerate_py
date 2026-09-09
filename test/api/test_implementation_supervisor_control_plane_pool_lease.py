from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    ProcessBirthIdentity,
    WorktreeLifecycleStore,
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    CONTROL_PLANE_RELOAD_STATUS,
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SUPERVISED_CHILD_IDENTITY_PATH_ENV,
    SUPERVISED_CHILD_OWNER_SCOPE_ENV,
    supervised_child_identity_path,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import (
    WORKTREE_POOL_SCHEMA,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seed_active_database_pool_lease(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    state_dir = repo / "state" / "lane-2"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "vrif_lane_2_task_state.json"
    PortalTaskState(
        implementation_attempts={"VRIF-010": 3},
        implementation_attempts_by_cid={"task:vrif-010": 3},
    ).save(state_path)

    attempt_id = "attempt:vrif-010:1"
    attempt_key = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:24]
    attempt_dir = state_dir / "vrif_lane_2_database_portal_attempts" / attempt_key
    attempt_dir.mkdir(parents=True)
    worktree_root = repo / "worktrees"
    workspace = worktree_root / "workspace_vrif_010"
    workspace.mkdir(parents=True)
    branch = "implementation/vrif-010-attempt-1"
    task_cid = "task:vrif-010"
    birth = current_process_birth()

    lifecycle_store = WorktreeLifecycleStore(repo)
    lifecycle = lifecycle_store.begin_preparing(
        task_id="VRIF-010",
        canonical_task_cid=task_cid,
        attempt=1,
        lane_id="lane-2",
        workspace_path=workspace,
        branch=branch,
        merge_target="main",
        state_dir=str(attempt_dir),
        owner=birth,
    )

    lease_token = "vrif-010-lease"
    pool_path = worktree_root / ".pool-state" / f"{lease_token}.json"
    pool = {
        "schema": WORKTREE_POOL_SCHEMA,
        "lease_token": lease_token,
        "path": str(workspace),
        "repo_root": str(repo),
        "repo_common_dir": str(repo / ".git"),
        "cache_key": "vrif",
        "base_commit": "base",
        "dependency_paths": [],
        "state": "leased",
        "lease_pid": os.getpid(),
        "branch": branch,
    }
    _write_json(pool_path, pool)
    lock_path = pool_path.with_suffix(".lock")
    _write_json(lock_path, {"pid": os.getpid(), "created_at_epoch": 1.0})

    binding = {
        "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
        "interface": "DatabasePortalExecutionBridge@1",
        "attempt_id": attempt_id,
        "claim_id": "claim:vrif-010",
        "task_cid": task_cid,
        "canonical_task_key": task_cid,
        "task_contract_digest": "sha256:" + "a" * 64,
        "repository_tree_id": "b" * 40,
        "task_alias": "VRIF-010",
        "goal_cid": "goal:vrif",
        "plan_cid": "plan:vrif",
        "task_revision": 1,
        "fencing_token": 1,
        "fence_epoch": 1,
        "lease_id": "database-lease-vrif-010",
        "task_body_digest": "sha256:body",
        "projection_seed_digest": "sha256:seed",
        "projection_immutable_digest": "sha256:projection",
        "authoritative_task_store": "duckdb",
        "projection_authority": False,
    }
    binding["binding_id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                binding,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
    )
    binding_path = attempt_dir / "database-attempt-binding.json"
    _write_json(binding_path, binding)
    nested_state_path = attempt_dir / "portal-task-state.json"
    PortalTaskState(
        active_task_id="VRIF-010",
        active_task_cid=task_cid,
        active_attempt=1,
        active_phase="validating",
        active_worktree_path=str(workspace),
        active_branch=branch,
        implementation_in_progress=True,
    ).save(nested_state_path)

    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            state_prefix="vrif_lane_2",
            repo_root=repo,
            worktree_root=worktree_root,
            database_program=SimpleNamespace(
                environment=lambda **_kwargs: {},
                endpoint_secret_handle="",
            ),
        )
    )
    child = SimpleNamespace(
        pid=os.getpid(),
        identity_process_birth=birth,
    )
    return {
        "repo": repo,
        "state_path": state_path,
        "supervisor": supervisor,
        "child": child,
        "pool": pool,
        "pool_path": pool_path,
        "lock_path": lock_path,
        "binding_path": binding_path,
        "nested_state_path": nested_state_path,
        "lifecycle_path": lifecycle_store.workspace_path_for(workspace),
        "workspace": workspace,
        "worktree_root": worktree_root,
        "branch": branch,
        "lifecycle": lifecycle,
    }


def test_supervisor_loop_config_binds_managed_child_identity_to_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _seed_active_database_pool_lease(tmp_path)
    supervisor = fixture["supervisor"]
    monkeypatch.setattr(
        supervisor,
        "_build_daemon_command",
        lambda: ["python", "-m", "managed-daemon"],
    )

    loop_config = supervisor.build_supervisor_loop_config()

    identity_path = Path(loop_config.child_env[SUPERVISED_CHILD_IDENTITY_PATH_ENV])
    assert identity_path == supervised_child_identity_path(loop_config.spec.child_pid_path)
    owner_scope = json.loads(loop_config.child_env[SUPERVISED_CHILD_OWNER_SCOPE_ENV])
    assert owner_scope == supervisor._managed_daemon_owner_scope()
    assert owner_scope["repo_root"] == str(fixture["repo"].resolve())
    assert owner_scope["state_dir"] == str(fixture["state_path"].parent.resolve())
    assert owner_scope["state_prefix"] == "vrif_lane_2"


def test_database_pool_lease_accepts_adopted_parent_pid_drift(
    tmp_path: Path,
) -> None:
    fixture = _seed_active_database_pool_lease(tmp_path)
    observed = fixture["child"].identity_process_birth
    adopted_parent = observed.parent_pid + 1
    fixture["child"].identity_process_birth = ProcessBirthIdentity(
        pid=observed.pid,
        start_time_ticks=observed.start_time_ticks,
        boot_id=observed.boot_id,
        parent_pid=adopted_parent,
    )
    lifecycle = json.loads(fixture["lifecycle_path"].read_text(encoding="utf-8"))
    lifecycle["owner"]["parent_pid"] = adopted_parent
    _write_json(fixture["lifecycle_path"], lifecycle)

    activity = fixture["supervisor"]._active_managed_database_pool_lease(fixture["child"])

    assert activity is not None
    assert activity["task_id"] == "VRIF-010"
    assert activity["lease_pid"] == str(os.getpid())


def test_control_plane_reload_defers_for_exact_nested_database_pool_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _seed_active_database_pool_lease(tmp_path)
    supervisor = fixture["supervisor"]
    supervisor._loaded_control_plane_source = {
        "source_id": "loaded-source",
        "repository_revision": "loaded-revision",
    }
    monkeypatch.setattr(
        supervisor,
        "_control_plane_source_snapshot",
        lambda: {
            "source_id": "current-source",
            "repository_revision": "current-revision",
        },
    )
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    original_state = fixture["state_path"].read_bytes()

    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    assert decision.action == "continue"
    assert (
        loop.config.status_extra_fields["control_plane_reload_deferred_reason"]
        == "active_managed_database_worktree_pool_lease"
    )
    assert loop.config.status_extra_fields["control_plane_reload_deferred_task_id"] == "VRIF-010"
    assert loop.config.status_extra_fields["control_plane_reload_attempt_budget_consumed"] is False
    assert (
        loop.config.status_extra_fields["control_plane_reload_provider_invocation_consumed"]
        is False
    )
    assert fixture["state_path"].read_bytes() == original_state

    idle = dict(fixture["pool"])
    idle.update({"state": "idle", "lease_pid": 0, "branch": ""})
    _write_json(fixture["pool_path"], idle)
    fixture["lock_path"].unlink()

    released = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    assert released.action == "stop"
    assert released.status == CONTROL_PLANE_RELOAD_STATUS
    assert released.reason == "control_plane_source_changed"
    assert loop.config.status_extra_fields["control_plane_reload_attempt_budget_consumed"] is False
    assert (
        loop.config.status_extra_fields["control_plane_reload_provider_invocation_consumed"]
        is False
    )
    assert fixture["state_path"].read_bytes() == original_state


@pytest.mark.parametrize("include_observation", [True, False])
def test_watchdog_threads_one_exact_census_into_both_stuck_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_observation: bool,
) -> None:
    fixture = _seed_active_database_pool_lease(tmp_path)
    supervisor = fixture["supervisor"]
    supervisor._last_supervisor_maintenance_at = 0.0
    census = {
        "worker_metrics_available": True,
        "phase": "implementing",
        "required": True,
        "active_worker_count": 1,
        "active_worker_pids": [8765],
        "stalled_without_active_worker": False,
    }
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    if include_observation:
        loop._last_worker_status = census
    expected = census if include_observation else {}
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        supervisor,
        "_refresh_loop_proof_rollout_status",
        lambda _loop: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        lambda: {"control_plane_update_pending": False},
    )

    def fake_is_stuck(_state, *, now_ts, worker_status):
        assert now_ts > 0
        seen["initial"] = worker_status
        return False, ""

    def fake_maintenance(_update_phase, *, worker_status=None):
        seen["maintenance"] = worker_status
        return {
            "stuck": False,
            "main_checkout_repair": {"repaired": False},
        }

    monkeypatch.setattr(supervisor, "is_stuck", fake_is_stuck)
    monkeypatch.setattr(
        supervisor,
        "_begin_supervisor_maintenance_heartbeat",
        lambda *_args, **_kwargs: (
            lambda *_args, **_kwargs: None,
            lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        fake_maintenance,
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    assert decision.action == "continue"
    assert seen["initial"] == expected
    assert seen["maintenance"] == expected
    assert seen["initial"] is seen["maintenance"]


@pytest.mark.parametrize(
    "case",
    [
        "idle",
        "initializing",
        "peer",
        "dead",
        "dead_child",
        "missing_child_birth_identity",
        "mismatched_child_birth_identity",
        "mismatched_lifecycle_stable_identity",
        "malformed_pool",
        "foreign_root",
        "pid_only",
        "malformed_lifecycle",
        "terminal_lifecycle",
        "malformed_binding",
        "malformed_nested_state",
    ],
)
def test_database_pool_lease_never_defers_without_exact_corroboration(
    tmp_path: Path,
    case: str,
) -> None:
    fixture = _seed_active_database_pool_lease(tmp_path)
    pool = dict(fixture["pool"])
    if case == "idle":
        pool.update({"state": "idle", "lease_pid": 0, "branch": ""})
        _write_json(fixture["pool_path"], pool)
        fixture["lock_path"].unlink()
    elif case == "initializing":
        pool["state"] = "initializing"
        _write_json(fixture["pool_path"], pool)
    elif case == "peer":
        pool["lease_pid"] = os.getppid()
        _write_json(fixture["pool_path"], pool)
        _write_json(fixture["lock_path"], {"pid": os.getppid()})
    elif case == "dead":
        pool["lease_pid"] = 2**30 - 1
        _write_json(fixture["pool_path"], pool)
        _write_json(fixture["lock_path"], {"pid": 2**30 - 1})
    elif case == "dead_child":
        fixture["child"].pid = 2**30 - 1
    elif case == "missing_child_birth_identity":
        fixture["child"].identity_process_birth = None
    elif case == "mismatched_child_birth_identity":
        observed = fixture["child"].identity_process_birth
        fixture["child"].identity_process_birth = ProcessBirthIdentity(
            pid=observed.pid,
            start_time_ticks=observed.start_time_ticks + 1,
            boot_id=observed.boot_id,
            parent_pid=observed.parent_pid,
        )
    elif case == "mismatched_lifecycle_stable_identity":
        lifecycle = json.loads(fixture["lifecycle_path"].read_text(encoding="utf-8"))
        lifecycle["owner"]["boot_id"] = "foreign-boot-id"
        _write_json(fixture["lifecycle_path"], lifecycle)
    elif case == "malformed_pool":
        fixture["pool_path"].write_text("{", encoding="utf-8")
    elif case == "foreign_root":
        foreign = fixture["repo"] / "foreign-workspace"
        foreign.mkdir()
        pool["path"] = str(foreign)
        _write_json(fixture["pool_path"], pool)
    elif case == "pid_only":
        fixture["lifecycle_path"].unlink()
    elif case == "malformed_lifecycle":
        fixture["lifecycle_path"].write_text("[]\n", encoding="utf-8")
    elif case == "terminal_lifecycle":
        lifecycle = json.loads(fixture["lifecycle_path"].read_text(encoding="utf-8"))
        lifecycle["state"] = "terminal"
        lifecycle["terminal_reason"] = "finished"
        _write_json(fixture["lifecycle_path"], lifecycle)
    elif case == "malformed_binding":
        fixture["binding_path"].write_text("[]\n", encoding="utf-8")
    elif case == "malformed_nested_state":
        fixture["nested_state_path"].write_text("[]\n", encoding="utf-8")

    assert fixture["supervisor"]._active_managed_database_pool_lease(fixture["child"]) is None


@pytest.mark.parametrize("field,value", [
    ("canonical_task_key", ""),
    ("canonical_task_key", 1),
    ("repository_tree_id", ""),
    ("repository_tree_id", 1),
    ("task_contract_digest", "sha256:bad"),
])
def test_database_pool_lease_rejects_malformed_current_binding_fields(
    tmp_path: Path, field: str, value: Any,
) -> None:
    fixture = _seed_active_database_pool_lease(tmp_path)
    binding = json.loads(fixture["binding_path"].read_text())
    binding[field] = value
    binding.pop("binding_id")
    binding["binding_id"] = "sha256:" + hashlib.sha256(
        json.dumps(binding, ensure_ascii=False, separators=(",", ":"),
                   sort_keys=True, default=str).encode()
    ).hexdigest()
    _write_json(fixture["binding_path"], binding)
    assert fixture["supervisor"]._active_managed_database_pool_lease(
        fixture["child"]
    ) is None


@pytest.mark.parametrize("invalid", [False, True])
def test_database_phase_census_uses_only_custody_proved_nested_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: bool,
) -> None:
    from datetime import datetime, timezone
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_loop

    fixture = _seed_active_database_pool_lease(tmp_path)
    supervisor = fixture["supervisor"]
    state = PortalTaskState.load(fixture["nested_state_path"])
    state.active_phase = "implementing"
    state.active_phase_started_at = datetime.now(timezone.utc).isoformat()
    state.active_phase_detail = "provider_launch_birth"
    state.save(fixture["nested_state_path"])
    nested_before = fixture["nested_state_path"].read_bytes()
    fixture["state_path"].unlink()
    monkeypatch.setattr(supervisor, "_build_daemon_command",
                        lambda: ["python", "-m", "managed-daemon"])
    monkeypatch.setattr(supervisor_loop, "procfs_descendant_processes",
                        lambda _pid: [{"pid": 4321,
                            "cmdline": "/usr/local/bin/grok --model grok-4.6"}])
    if invalid:
        binding = json.loads(fixture["binding_path"].read_text())
        binding["binding_id"] = "sha256:" + "0" * 64
        _write_json(fixture["binding_path"], binding)
    monkeypatch.setattr(supervisor, "_shared_active_worktree_owners",
                        lambda _root: pytest.fail("projection scanned sibling state"))
    loop = supervisor_loop.SupervisorLoop(supervisor.build_supervisor_loop_config())
    observed = loop._observe_worker_status(fixture["child"], {
        "active_phase": "implementing",
        "active_phase_started_at": "2020-01-01T00:00:00+00:00",
        "worktree_no_child_stall_seconds": 1,
    })
    assert observed["phase"] == ("" if invalid else "implementing")
    assert observed["required"] is (not invalid)
    assert observed["worker_metrics_available"] is True
    assert observed["active_worker_count"] == 1
    assert observed["stalled_without_active_worker"] is not True
    if not invalid:
        assert observed["threshold_seconds"] == loop.config.status_static_fields[
            "worktree_no_child_stall_seconds"]
    assert fixture["nested_state_path"].read_bytes() == nested_before
