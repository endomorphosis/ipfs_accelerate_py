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
    _projection_immutable_digest,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTaskState,
    parse_task_text,
    portal_task_identity,
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

PCSM_027_DATABASE_TASK_CID = (
    "baguqeerasd5j7p3geimlxdb62nar2sodoawnsdqfpy3udw2zihhpamcbn4ja"
)
PCSM_027_OBSERVED_PORTAL_TASK_CID = (
    "baguqeera7swojthhh6pfwqqiswos4cjb36pb3rmwlgdw7rxkacu5si2wsi7q"
)
PCSM_027_DATABASE_ATTEMPT_ID = "attempt:7a81388546b64c40a9f01c2fc9479425"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_recommitted_binding(path: Path, payload: dict[str, Any]) -> None:
    body = dict(payload)
    body.pop("binding_id", None)
    payload["binding_id"] = (
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
    _write_json(path, payload)


def _recommit_projection_identity_change(fixture: dict[str, Any]) -> None:
    projection = fixture["projection_path"].read_text(encoding="utf-8")
    changed = projection.replace(
        f"- Allowed Paths: artifacts/{fixture['task_alias']}.json",
        "- Allowed Paths: artifacts/foreign-successor.json",
    )
    assert changed != projection
    fixture["projection_path"].write_text(changed, encoding="utf-8")
    binding = json.loads(fixture["binding_path"].read_text(encoding="utf-8"))
    binding["projection_seed_digest"] = (
        "sha256:" + hashlib.sha256(changed.encode("utf-8")).hexdigest()
    )
    binding["projection_immutable_digest"] = _projection_immutable_digest(changed)
    _write_recommitted_binding(fixture["binding_path"], binding)


def _seed_active_database_pool_lease(
    tmp_path: Path,
    *,
    mark_lifecycle_active: bool = True,
    task_alias: str = "VRIF-010",
    database_task_cid: str = "task:database-vrif-010",
    database_attempt_number: int = 2,
    attempt_id: str = "attempt:vrif-010:database-2",
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
        implementation_attempts={task_alias: 3},
        implementation_attempts_by_cid={database_task_cid: 3},
    ).save(state_path)

    attempt_key = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:24]
    attempt_dir = state_dir / "vrif_lane_2_database_portal_attempts" / attempt_key
    attempt_dir.mkdir(parents=True)
    worktree_root = repo / "worktrees"
    task_slug = task_alias.lower().replace("-", "_")
    workspace = worktree_root / f"workspace_{task_slug}"
    workspace.mkdir(parents=True)
    branch = f"implementation/{task_alias.lower()}-attempt-1"
    birth = current_process_birth()

    canonical_task_key = f"task-key:{task_slug}"
    owner_session_id = f"owner-session:{task_slug}"
    projection = "\n".join(
        (
            "# Database attempt projection (non-authoritative)",
            "",
            f"## {task_alias} Active nested database work",
            "",
            "- Status: ready",
            f"- Database task CID: {database_task_cid}",
            f"- Database attempt ID: {attempt_id}",
            f"- Database claim ID: claim:{task_slug}",
            f"- Database attempt number: {database_attempt_number}",
            f"- Database owner session ID: {owner_session_id}",
            f"- Canonical task key: {canonical_task_key}",
            f"- Canonical task CID: {database_task_cid}",
            "- Projection authority: false",
            f"- Allowed Paths: artifacts/{task_alias}.json",
            "",
        )
    )
    projection_path = attempt_dir / "task-projection.md"
    projection_path.write_text(projection, encoding="utf-8")
    binding = {
        "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
        "interface": "DatabasePortalExecutionBridge@1",
        "attempt_id": attempt_id,
        "claim_id": f"claim:{task_slug}",
        "attempt_number": database_attempt_number,
        "owner_session_id": owner_session_id,
        "task_cid": database_task_cid,
        "canonical_task_key": canonical_task_key,
        "task_alias": task_alias,
        "goal_cid": "goal:vrif",
        "plan_cid": "plan:vrif",
        "task_revision": 1,
        "fencing_token": 1,
        "fence_epoch": 1,
        "lease_id": f"database-lease-{task_slug}",
        "task_body_digest": "sha256:" + hashlib.sha256(b"body").hexdigest(),
        "task_contract_digest": "sha256:"
        + hashlib.sha256(b"contract").hexdigest(),
        "repository_tree_id": "sha256:"
        + hashlib.sha256(b"repository-tree").hexdigest(),
        "projection_seed_digest": "sha256:"
        + hashlib.sha256(projection.encode("utf-8")).hexdigest(),
        "projection_immutable_digest": _projection_immutable_digest(projection),
        "authoritative_task_store": "duckdb",
        "projection_authority": False,
        "landed_completion_recovery_seed_id": "",
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

    projected_tasks = parse_task_text(
        projection,
        path=projection_path,
        task_header_prefix=f"## {task_alias}",
    )
    assert len(projected_tasks) == 1
    portal_identity = portal_task_identity(
        projected_tasks[0],
        todo_path=projection_path.resolve(),
    )
    portal_task_cid = portal_identity.canonical_task_cid
    assert portal_task_cid != database_task_cid

    lifecycle_store = WorktreeLifecycleStore(repo)
    lifecycle = lifecycle_store.begin_preparing(
        task_id=task_alias,
        canonical_task_cid=portal_task_cid,
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

    lease_token = f"{task_slug}-lease"
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

    nested_state_path = attempt_dir / "portal-task-state.json"
    PortalTaskState(
        active_task_id=task_alias,
        active_task_cid=portal_task_cid,
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
        "projection_path": projection_path,
        "attempt_dir": attempt_dir,
        "binding": binding,
        "database_task_cid": database_task_cid,
        "database_attempt_number": database_attempt_number,
        "portal_task_cid": portal_task_cid,
        "portal_task_key": portal_identity.canonical_task_key,
        "task_alias": task_alias,
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
    task_alias: str = "VRIF-010",
    database_task_cid: str = "task:database-vrif-010",
    database_attempt_number: int = 2,
    attempt_id: str = "attempt:vrif-010:database-2",
) -> dict[str, Any]:
    fixture = _seed_active_database_pool_lease(
        tmp_path,
        mark_lifecycle_active=lifecycle_state != "preparing",
        task_alias=task_alias,
        database_task_cid=database_task_cid,
        database_attempt_number=database_attempt_number,
        attempt_id=attempt_id,
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
            "task_id": fixture["task_alias"],
            "canonical_task_cid": fixture["portal_task_cid"],
            "canonical_task_key": fixture["binding"]["canonical_task_key"],
            "attempt": 1,
            "lease_id": f"implementation-lease-{fixture['task_alias'].lower()}",
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


def _seed_stale_predecessor_projection(
    fixture: dict[str, Any],
    *,
    task_id: str = "VRIF-009",
    task_cid: str = "task:vrif-009",
) -> bytes:
    state = PortalTaskState.load(fixture["state_path"])
    state.active_task_id = task_id
    state.active_task_cid = task_cid
    state.active_attempt = 1
    state.active_phase = "implementing"
    state.implementation_in_progress = True
    state.save(fixture["state_path"])
    return fixture["state_path"].read_bytes()


def test_watchdog_maintenance_defers_for_exact_nested_database_pool_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _seed_active_database_pool_lease(tmp_path)
    supervisor = fixture["supervisor"]
    original_state = _seed_stale_predecessor_projection(fixture)
    maintenance_calls: list[object] = []
    monkeypatch.setattr(
        supervisor,
        "is_stuck",
        lambda *_args, **_kwargs: (
            True,
            "no progress on stale predecessor VRIF-009",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda update: maintenance_calls.append(update),
    )
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))

    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    assert decision.action == "continue"
    assert maintenance_calls == []
    assert fixture["state_path"].read_bytes() == original_state


def test_watchdog_defers_at_pcsm_024_to_pcsm_027_two_identity_successor_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _seed_active_database_pool_lease(
        tmp_path,
        task_alias="PCSM-027",
        database_task_cid=PCSM_027_DATABASE_TASK_CID,
        database_attempt_number=2,
        attempt_id=PCSM_027_DATABASE_ATTEMPT_ID,
    )
    original_state = _seed_stale_predecessor_projection(
        fixture,
        task_id="PCSM-024",
        task_cid="",
    )
    supervisor = fixture["supervisor"]
    maintenance_calls: list[object] = []
    monkeypatch.setattr(
        supervisor,
        "is_stuck",
        lambda *_args, **_kwargs: (
            True,
            "no progress on active task PCSM-024",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda update: maintenance_calls.append(update),
    )
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))

    activity = supervisor._active_managed_database_pool_lease(fixture["child"])
    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    # The incident's DuckDB identity D and observed Portal identity L are
    # intentionally different.  This temporary projection derives its own
    # path-bound L through the same production contract.
    assert PCSM_027_OBSERVED_PORTAL_TASK_CID != PCSM_027_DATABASE_TASK_CID
    assert fixture["portal_task_cid"] != PCSM_027_DATABASE_TASK_CID
    assert fixture["binding"]["attempt_number"] == 2
    assert fixture["lifecycle"].attempt == 1
    assert activity is not None
    assert activity["task_id"] == "PCSM-027"
    assert activity["task_cid"] == fixture["portal_task_cid"]
    assert activity["attempt"] == "1"
    assert decision.action == "continue"
    assert maintenance_calls == []
    assert fixture["state_path"].read_bytes() == original_state


@pytest.mark.parametrize("lifecycle_state", ("preparing", "active", "settling"))
def test_watchdog_maintenance_defers_for_exact_live_database_nonterminal_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_state: str,
) -> None:
    fixture = _seed_live_unprojected_database_attempt(
        tmp_path,
        lifecycle_state=lifecycle_state,
    )
    supervisor = fixture["supervisor"]
    original_state = _seed_stale_predecessor_projection(fixture)
    maintenance_calls: list[object] = []
    monkeypatch.setattr(
        supervisor,
        "is_stuck",
        lambda *_args, **_kwargs: (
            True,
            "no progress on stale predecessor VRIF-009",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda update: maintenance_calls.append(update),
    )
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))

    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    assert decision.action == "continue"
    assert maintenance_calls == []
    assert fixture["state_path"].read_bytes() == original_state


def test_watchdog_maintenance_resumes_without_exact_database_corroboration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _seed_live_unprojected_database_attempt(tmp_path)
    supervisor = fixture["supervisor"]
    _seed_stale_predecessor_projection(fixture)
    fixture["implementation_lock_path"].unlink()
    maintenance_calls: list[object] = []
    monkeypatch.setattr(
        supervisor,
        "is_stuck",
        lambda *_args, **_kwargs: (
            True,
            "no progress on stale predecessor VRIF-009",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_begin_supervisor_maintenance_heartbeat",
        lambda *_args, **_kwargs: (
            lambda _phase: None,
            lambda _status, _message="": None,
        ),
    )

    def _maintenance(update: object) -> dict[str, Any]:
        maintenance_calls.append(update)
        return {
            "main_checkout_repair": {"repaired": False},
            "stuck": False,
        }

    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        _maintenance,
    )
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))

    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    assert decision.action == "continue"
    assert len(maintenance_calls) == 1


@pytest.mark.parametrize(
    "case",
    ("missing_projection", "tampered_projection", "recommitted_portal_identity"),
)
def test_watchdog_maintenance_resumes_without_exact_projection_identity_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    fixture = _seed_active_database_pool_lease(tmp_path)
    _seed_stale_predecessor_projection(fixture)
    if case == "missing_projection":
        fixture["projection_path"].unlink()
    elif case == "tampered_projection":
        fixture["projection_path"].write_text(
            fixture["projection_path"].read_text(encoding="utf-8")
            + "\n- Allowed Paths: artifacts/uncommitted-tamper.json\n",
            encoding="utf-8",
        )
    elif case == "recommitted_portal_identity":
        _recommit_projection_identity_change(fixture)
    maintenance_calls: list[object] = []
    supervisor = fixture["supervisor"]
    monkeypatch.setattr(
        supervisor,
        "is_stuck",
        lambda *_args, **_kwargs: (
            True,
            "no progress on stale predecessor VRIF-009",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_begin_supervisor_maintenance_heartbeat",
        lambda *_args, **_kwargs: (
            lambda _phase: None,
            lambda _status, _message="": None,
        ),
    )

    def _maintenance(update: object) -> dict[str, Any]:
        maintenance_calls.append(update)
        return {
            "main_checkout_repair": {"repaired": False},
            "stuck": False,
        }

    monkeypatch.setattr(supervisor, "_run_once_with_maintenance", _maintenance)
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))

    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        fixture["child"],
        {},
    )

    assert decision.action == "continue"
    assert len(maintenance_calls) == 1


@pytest.mark.parametrize(
    "case",
    [
        "missing_lock",
        "foreign_lock_pid",
        "lock_task_mismatch",
        "tampered_binding",
        "missing_projection",
        "recommitted_portal_identity",
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
    elif case == "missing_projection":
        fixture["projection_path"].unlink()
    elif case == "recommitted_portal_identity":
        _recommit_projection_identity_change(fixture)
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
        "missing_projection",
        "malformed_projection",
        "recommitted_portal_identity",
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
    elif case == "missing_projection":
        fixture["projection_path"].unlink()
    elif case == "malformed_projection":
        fixture["projection_path"].write_text("# malformed\n", encoding="utf-8")
    elif case == "recommitted_portal_identity":
        _recommit_projection_identity_change(fixture)
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
