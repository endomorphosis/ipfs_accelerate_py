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


def _seed_active_database_pool_lease(
    tmp_path: Path,
    *,
    mark_lifecycle_active: bool = True,
) -> dict[str, Any]:
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
    if mark_lifecycle_active:
        lifecycle = lifecycle_store.mark_active(
            workspace,
            lease_id=lifecycle.lease_id,
            expected_fence=lifecycle.fence,
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
        "canonical_task_key": "task/v1/vrif-010",
        "task_alias": "VRIF-010",
        "goal_cid": "goal:vrif",
        "plan_cid": "plan:vrif",
        "task_revision": 1,
        "fencing_token": 1,
        "fence_epoch": 1,
        "lease_id": "database-lease-vrif-010",
        "task_body_digest": "sha256:body",
        "task_contract_digest": "sha256:" + "a" * 64,
        "repository_tree_id": "git-tree:" + "b" * 40,
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
        "attempt_dir": attempt_dir,
        "binding": binding,
        "nested_state_path": nested_state_path,
        "lifecycle_path": lifecycle_store.workspace_path_for(workspace),
        "workspace": workspace,
        "worktree_root": worktree_root,
        "branch": branch,
        "lifecycle": lifecycle,
    }


def _seed_live_unprojected_database_attempt(
    tmp_path: Path,
    *,
    lifecycle_state: str = "preparing",
) -> dict[str, Any]:
    fixture = _seed_active_database_pool_lease(
        tmp_path,
        mark_lifecycle_active=lifecycle_state != "preparing",
    )
    if lifecycle_state == "settling":
        lifecycle = fixture["lifecycle"]
        fixture["lifecycle"] = WorktreeLifecycleStore(
            fixture["repo"]
        ).mark_settling(
            fixture["workspace"],
            lease_id=lifecycle.lease_id,
            expected_fence=lifecycle.fence,
        )
    fixture["pool_path"].unlink()
    fixture["lock_path"].unlink()
    fixture["nested_state_path"].unlink()
    implementation_lock_path = fixture["attempt_dir"] / "implementation.lock"
    _write_json(
        implementation_lock_path,
        {
            "kind": "implementation",
            "pid": os.getpid(),
            "repo_root": str(fixture["repo"].resolve()),
            "state_dir": str(fixture["attempt_dir"].resolve()),
            "task_id": "VRIF-010",
            "canonical_task_cid": "task:vrif-010",
            "canonical_task_key": fixture["binding"]["canonical_task_key"],
            "attempt": 1,
            "lease_id": "implementation-lease-vrif-010",
            "started_at": "2026-08-24T00:00:00+00:00",
        },
    )
    fixture["implementation_lock_path"] = implementation_lock_path
    return fixture


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


@pytest.mark.parametrize("lifecycle_state", ("preparing", "active", "settling"))
def test_control_plane_reload_defers_for_exact_live_database_nonterminal_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_state: str,
) -> None:
    fixture = _seed_live_unprojected_database_attempt(
        tmp_path,
        lifecycle_state=lifecycle_state,
    )
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
        == "active_managed_database_nonterminal_lifecycle_claim"
    )
    assert loop.config.status_extra_fields["control_plane_reload_deferred_task_id"] == "VRIF-010"
    assert loop.config.status_extra_fields["control_plane_reload_attempt_budget_consumed"] is False
    assert (
        loop.config.status_extra_fields["control_plane_reload_provider_invocation_consumed"]
        is False
    )
    assert fixture["state_path"].read_bytes() == original_state

    fixture["implementation_lock_path"].unlink()
    released = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    assert released.action == "stop"
    assert released.status == CONTROL_PLANE_RELOAD_STATUS
    assert released.reason == "control_plane_source_changed"
    assert fixture["state_path"].read_bytes() == original_state


@pytest.mark.parametrize(
    "case",
    [
        "missing_lock",
        "foreign_lock_pid",
        "lock_task_mismatch",
        "tampered_binding",
        "mismatched_child_birth",
        "foreign_repo_root",
        "terminal_lifecycle",
    ],
)
def test_database_nonterminal_claim_never_defers_without_exact_corroboration(
    tmp_path: Path,
    case: str,
) -> None:
    fixture = _seed_live_unprojected_database_attempt(tmp_path)
    if case == "missing_lock":
        fixture["implementation_lock_path"].unlink()
    elif case == "foreign_lock_pid":
        payload = json.loads(
            fixture["implementation_lock_path"].read_text(encoding="utf-8")
        )
        payload["pid"] = os.getppid()
        _write_json(fixture["implementation_lock_path"], payload)
    elif case == "lock_task_mismatch":
        payload = json.loads(
            fixture["implementation_lock_path"].read_text(encoding="utf-8")
        )
        payload["task_id"] = "VRIF-999"
        _write_json(fixture["implementation_lock_path"], payload)
    elif case == "tampered_binding":
        payload = json.loads(fixture["binding_path"].read_text(encoding="utf-8"))
        payload["task_cid"] = "task:foreign"
        _write_json(fixture["binding_path"], payload)
    elif case == "mismatched_child_birth":
        observed = fixture["child"].identity_process_birth
        fixture["child"].identity_process_birth = ProcessBirthIdentity(
            pid=observed.pid,
            start_time_ticks=observed.start_time_ticks + 1,
            boot_id=observed.boot_id,
            parent_pid=observed.parent_pid,
        )
    elif case == "foreign_repo_root":
        payload = json.loads(fixture["lifecycle_path"].read_text(encoding="utf-8"))
        payload["repo_root"] = str(tmp_path)
        _write_json(fixture["lifecycle_path"], payload)
    elif case == "terminal_lifecycle":
        lifecycle = fixture["lifecycle"]
        store = WorktreeLifecycleStore(fixture["repo"])
        store.mark_terminal(
            fixture["workspace"],
            lease_id=lifecycle.lease_id,
            expected_fence=lifecycle.fence,
            reason="finished",
        )

    assert (
        fixture["supervisor"]._active_managed_database_nonterminal_claim(
            fixture["child"]
        )
        is None
    )


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
        "legacy_binding_schema",
        "missing_canonical_task_key",
        "tampered_task_contract",
        "missing_repository_tree",
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
    elif case == "legacy_binding_schema":
        binding = json.loads(
            fixture["binding_path"].read_text(encoding="utf-8")
        )
        binding.pop("canonical_task_key")
        binding.pop("task_contract_digest")
        binding.pop("repository_tree_id")
        body = dict(binding)
        body.pop("binding_id")
        binding["binding_id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        )
        _write_json(fixture["binding_path"], binding)
    elif case == "tampered_task_contract":
        binding = json.loads(
            fixture["binding_path"].read_text(encoding="utf-8")
        )
        binding["task_contract_digest"] = "sha256:not-a-digest"
        body = dict(binding)
        body.pop("binding_id")
        binding["binding_id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        )
        _write_json(fixture["binding_path"], binding)
    elif case == "missing_canonical_task_key":
        binding = json.loads(
            fixture["binding_path"].read_text(encoding="utf-8")
        )
        binding["canonical_task_key"] = ""
        body = dict(binding)
        body.pop("binding_id")
        binding["binding_id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        )
        _write_json(fixture["binding_path"], binding)
    elif case == "missing_repository_tree":
        binding = json.loads(
            fixture["binding_path"].read_text(encoding="utf-8")
        )
        binding["repository_tree_id"] = ""
        body = dict(binding)
        body.pop("binding_id")
        binding["binding_id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        )
        _write_json(fixture["binding_path"], binding)
    elif case == "malformed_nested_state":
        fixture["nested_state_path"].write_text("[]\n", encoding="utf-8")

    assert fixture["supervisor"]._active_managed_database_pool_lease(fixture["child"]) is None
