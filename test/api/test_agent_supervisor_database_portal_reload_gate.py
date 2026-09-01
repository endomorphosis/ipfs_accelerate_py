from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    ProcessBirthIdentity,
    WorkspaceLifecycleRecord,
    WorkspaceLifecycleState,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskRecord,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_RETRY_BUDGET_SCHEMA,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    CONTROL_PLANE_RELOAD_DEFERRED_UNSEALED_STATUS,
    CONTROL_PLANE_RELOAD_STATUS,
    DATABASE_PORTAL_RELOAD_PROJECTION_SCHEMA,
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


def _config(tmp_path: Path) -> PortalSupervisorConfig:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    state_dir = repo / "state" / "lane-0"
    state_dir.mkdir(parents=True)
    return PortalSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        state_dir=state_dir,
        state_prefix="pctdd_lane_0",
        task_prefix="## PCTDD-",
        database_program=DatabaseProgramConfig(
            authority_mode="quack",
            task_source_kind="duckdb",
            endpoint_secret_handle="env://TEST_PCTDD_QUACK_TOKEN",
            quack_endpoint="quack:127.0.0.1:4242",
            store_id="state/control.duckdb",
            store_generation="pctdd-logical-g8",
            schema_revision="1",
            failover_policy="fail_closed",
        ),
        database_owner_session_id="pctdd-owner:lane-0",
        repo_root=repo,
        worktree_root=repo / "worktrees",
        merge_target_branch="agent/pctdd",
    )


def _receipt(
    *,
    owner: str = "pctdd-owner:lane-0",
    task_cid: str = "cid:PCTDD-031",
) -> dict[str, object]:
    return {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        "operation": "database_claim",
        "task_cid": task_cid,
        "validation_spec_cid": "sha256:" + "1" * 64,
        "attempts_used": 1,
        "max_task_attempts": 3,
        "retry_exhausted": False,
        "process_instance_id": "process-instance-1",
        "owner_session_id": owner,
        "attempt_id": "attempt-PCTDD-031-1",
        "claim_id": "claim-PCTDD-031-1",
        "lease_id": "database-lease-PCTDD-031-1",
        "attempt_number": 1,
        "fencing_token": 7,
        "fence_epoch": 2,
    }


def _task(
    receipt: dict[str, object],
    *,
    revision: int = 11,
) -> TaskRecord:
    return TaskRecord(
        task_cid=str(receipt["task_cid"]),
        task_alias="PCTDD-031",
        goal_cid="PCTDD-G030",
        ordinal=31,
        status="in_progress",
        revision=revision,
        priority="P1",
        body={
            "title": "fixture-aware scheduling",
            "completion_receipt": receipt,
        },
    )


def _owner_binding(generation: int) -> dict[str, object]:
    return {
        "server_id": "server-1",
        "store_id": "state/control.duckdb",
        "database_uuid": "database-1",
        "schema_revision": 8,
        "schema_fingerprint": "sha256:" + "2" * 64,
        "generation": generation,
        "process_birth_id": f"birth-{generation}",
        "listen_uri": "quack:127.0.0.1:4242",
        "extension_fingerprint": "sha256:" + "3" * 64,
    }


def _idle_projection() -> dict[str, object]:
    return {
        "task_source_revision": 19,
        "task_ids": [],
        "attempts": [],
        "nonterminal_attempt_count": 0,
        "post_provider_recovery_saga_count": 0,
        "activity_detected": False,
        "defer_reload": False,
        "defer_maintenance": False,
        "reason": "database_portal_population_idle",
    }


def _authenticated_watchdog_projection(*, active: bool) -> dict[str, object]:
    projection = {
        "schema": DATABASE_PORTAL_RELOAD_PROJECTION_SCHEMA,
        "applicable": True,
        "authority_available": True,
        "integrity_verified": True,
        "error_type": "",
        "quack_owner": _owner_binding(41),
    }
    projection.update(_idle_projection())
    projection.update(
        {
            "activity_detected": active,
            "defer_reload": active,
            "defer_maintenance": active,
            "reason": (
                "database_portal_claim_or_recovery_saga"
                if active
                else "database_portal_population_idle"
            ),
            "task_ids": ["PCTDD-034"] if active else [],
            "nonterminal_attempt_count": 1 if active else 0,
        }
    )
    return projection


def _inconclusive_watchdog_projection() -> dict[str, object]:
    return {
        "schema": DATABASE_PORTAL_RELOAD_PROJECTION_SCHEMA,
        "applicable": True,
        "authority_available": False,
        "integrity_verified": False,
        "activity_detected": False,
        "defer_reload": True,
        "defer_maintenance": True,
        "reason": "database_portal_projection_inconclusive",
        "error_type": "RuntimeError",
    }


@pytest.mark.parametrize(
    ("claim_appears", "sealed_dispatch"),
    [(False, False), (True, True), (True, False)],
)
def test_source_reload_quiesces_before_post_idle_claim_recheck(
    tmp_path,
    monkeypatch,
    claim_appears,
    sealed_dispatch,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    supervisor.config.plan_bound_dispatch = sealed_dispatch
    order: list[str] = []
    claim_visible = False

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        lambda: {"control_plane_update_pending": True},
    )
    monkeypatch.setattr(supervisor, "_set_loop_status_fields", lambda *_: None)
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )

    def quiesce(*_args, **_kwargs):
        nonlocal claim_visible
        order.append("quiesce")
        claim_visible = claim_appears
        return {"quiesced": True, "supervised_child_alive": False}

    def project():
        order.append("project")
        assert claim_visible is claim_appears
        projection = _idle_projection()
        if claim_visible:
            projection.update(
                {
                    "activity_detected": True,
                    "defer_reload": True,
                    "defer_maintenance": True,
                    "reason": "database_portal_claim_or_recovery_saga",
                }
            )
        return projection

    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        quiesce,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection",
        project,
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        SimpleNamespace(config=SimpleNamespace(status_extra_fields={})),
        SimpleNamespace(pid=987654),
        {},
    )

    assert order == ["quiesce", "project"]
    if claim_appears:
        if sealed_dispatch:
            assert decision.action == "recycle"
            assert decision.reason == "control_plane_reload_deferred"
        else:
            assert decision.action == "stop"
            assert decision.reason == "control_plane_reload_deferred_unsealed"
            assert decision.status == (
                CONTROL_PLANE_RELOAD_DEFERRED_UNSEALED_STATUS
            )
    else:
        assert decision.action == "stop"
        assert decision.status == CONTROL_PLANE_RELOAD_STATUS


@pytest.mark.parametrize(
    ("projection", "projection_reason"),
    [
        (
            _authenticated_watchdog_projection(active=True),
            "database_portal_claim_or_recovery_saga",
        ),
        (
            _inconclusive_watchdog_projection(),
            "database_portal_projection_inconclusive",
        ),
        (
            _authenticated_watchdog_projection(active=False),
            "database_portal_population_idle",
        ),
        (
            {
                "applicable": True,
                "activity_detected": "unknown",
                "reason": "malformed_projection",
            },
            "malformed_projection",
        ),
    ],
    ids=(
        "authenticated-active",
        "inconclusive-fail-closed",
        "authenticated-idle-race-closed",
        "malformed-applicable-fail-closed",
    ),
)
def test_watchdog_defers_before_quiescence_for_database_portal(
    tmp_path,
    monkeypatch,
    projection,
    projection_reason,
):
    config = _config(tmp_path)
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    supervisor._last_supervisor_maintenance_at = 0.0
    events: list[tuple[str, object]] = []
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        lambda: {"control_plane_update_pending": False},
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection",
        lambda: (
            events.append(("projection", None))
            or projection
        ),
    )
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        supervisor,
        "_record_event",
        lambda name, detail: events.append((name, detail)),
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "routine maintenance must not quiesce a configured Quack child"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_begin_supervisor_maintenance_heartbeat",
        lambda *_args, **_kwargs: pytest.fail(
            "deferred maintenance must not start a maintenance heartbeat"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda *_args, **_kwargs: pytest.fail(
            "deferred maintenance must not run"
        ),
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        SimpleNamespace(pid=987654),
        {},
    )

    assert decision.action == "continue"
    assert [name for name, _detail in events] == [
        "projection",
        "supervisor_maintenance_deferred_for_database_portal",
    ]
    assert loop.config.status_extra_fields[
        "supervisor_maintenance_deferred"
    ] is True
    assert loop.config.status_extra_fields[
        "supervisor_maintenance_deferred_reason"
    ] == "database_portal_routine_maintenance_deferred"
    assert loop.config.status_extra_fields[
        "database_portal_projection_reason"
    ] == projection_reason
    assert loop.config.status_extra_fields[
        "database_portal_reload_projection"
    ] == projection
    assert supervisor._last_supervisor_maintenance_at > 0.0


def test_non_quack_watchdog_maintenance_quiesces_child_before_mutating(
    tmp_path,
    monkeypatch,
):
    config = replace(_config(tmp_path), database_program=None)
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    supervisor._last_supervisor_maintenance_at = 0.0
    events: list[str] = []
    finished: list[tuple[str, str]] = []

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        lambda: {"control_plane_update_pending": False},
    )
    monkeypatch.setattr(supervisor, "_set_loop_status_fields", lambda *_: None)
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection",
        lambda: events.append("projection") or {
            "schema": DATABASE_PORTAL_RELOAD_PROJECTION_SCHEMA,
            "applicable": False,
            "reason": "database_portal_authority_not_configured",
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: (
            events.append("quiesce")
            or {"quiesced": True, "supervised_child_alive": False}
        ),
    )

    def maintenance(_update, **kwargs):
        assert events == ["projection", "quiesce"]
        assert kwargs == {"managed_daemon_launch_lock_held": True}
        events.append("maintenance")
        return {
            "stuck": False,
            "maintenance_blocked": False,
            "reason": "",
            "main_checkout_repair": {"repaired": False},
        }

    monkeypatch.setattr(supervisor, "_run_once_with_maintenance", maintenance)
    monkeypatch.setattr(
        supervisor,
        "_begin_supervisor_maintenance_heartbeat",
        lambda *_args, **_kwargs: (
            lambda _phase: None,
            lambda status="completed", error="": finished.append(
                (status, error)
            ),
        ),
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        SimpleNamespace(config=SimpleNamespace(status_extra_fields={})),
        SimpleNamespace(pid=987654),
        {},
    )

    assert events == ["projection", "quiesce", "maintenance"]
    assert decision.action == "recycle"
    assert decision.reason == "supervisor_maintenance_completed_after_quiescence"
    assert finished == [("completed", "")]


@pytest.mark.parametrize(
    ("after_generation", "verified"),
    [(41, True), (42, False)],
)
def test_quack_projection_uses_one_connection_and_detects_generation_churn(
    tmp_path,
    monkeypatch,
    after_generation,
    verified,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    events: list[str] = []
    owner_values = iter((_owner_binding(41), _owner_binding(after_generation)))

    class Connection:
        def __init__(self):
            self._quack_mutation_binding = _owner_binding(41)
            self.closed = False

        def execute(self, statement):
            events.append(statement)
            return self

        def close(self):
            self.closed = True
            events.append("close")

    connection = Connection()

    @contextmanager
    def owner_lock(_path, *, timeout_seconds):
        assert timeout_seconds == 2.0
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    monkeypatch.setattr(
        supervisor,
        "_database_reconciliation_program_environment",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        duckdb_state,
        "quack_owner_mutation_write_lock_path",
        lambda _store: tmp_path / "owner.lock",
    )
    monkeypatch.setattr(duckdb_state, "exclusive_file_lock", owner_lock)
    monkeypatch.setattr(
        duckdb_state,
        "_resolve_quack_token_handle",
        lambda **_kwargs: ("not-published", next(owner_values)),
    )
    open_count = 0

    def open_connection(_uri, *, token):
        nonlocal open_count
        assert token == "not-published"
        open_count += 1
        return connection

    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        open_connection,
    )

    def project(task_source):
        events.append("project")
        with task_source.intent._connection(write=False) as observed:
            assert observed is connection
        return _idle_projection()

    monkeypatch.setattr(
        supervisor,
        "_database_portal_claim_projection",
        project,
    )

    result = supervisor._database_portal_reload_projection()

    assert open_count == 1
    assert events == [
        "lock",
        "BEGIN TRANSACTION",
        "project",
        "COMMIT",
        "close",
        "unlock",
    ]
    assert connection.closed is True
    assert result["integrity_verified"] is verified
    assert result["activity_detected"] is False
    assert result["defer_reload"] is (not verified)
    if verified:
        assert result["quack_owner"]["generation"] == 41
        assert result["quack_owner"]["generation"] != "pctdd-logical-g8"
    else:
        assert result["reason"] == "database_portal_projection_inconclusive"


@pytest.mark.parametrize(
    ("lifecycle_state", "recovery"),
    [
        (WorkspaceLifecycleState.ACTIVE, False),
        (WorkspaceLifecycleState.TERMINAL, True),
    ],
)
def test_exact_attempt_binding_and_lifecycle_defer_reload(
    tmp_path,
    monkeypatch,
    lifecycle_state,
    recovery,
):
    config = _config(tmp_path)
    supervisor = PortalImplementationSupervisor(config)
    receipt = _receipt()
    task = _task(receipt)
    attempt = supervisor._database_claim_attempt(task)

    class Source:
        def get_task(self, task_cid):
            return task if task_cid == task.task_cid else None

    source = Source()
    bridge = DatabasePortalExecutionBridge(
        task_source=source,
        attempt_root=config.state_dir
        / f"{config.state_prefix}_database_portal_attempts",
        portal_factory=lambda *_: None,
    )
    paths, binding = bridge._ensure_attempt_projection(attempt, task)
    identity = bridge._projection_task_identity(paths, binding)
    workspace = Path(config.worktree_root) / "PCTDD-031"
    workspace.mkdir(parents=True)
    state = PortalTaskState(
        active_task_id="PCTDD-031" if not recovery else "",
        active_task_key=identity["canonical_task_key"] if not recovery else "",
        active_task_cid=identity["canonical_task_cid"] if not recovery else "",
        active_attempt=1 if not recovery else 0,
        active_phase="validating" if not recovery else "",
        active_worktree_path=str(workspace) if not recovery else "",
        active_branch="implementation/PCTDD-031/1" if not recovery else "",
        implementation_in_progress=not recovery,
        implementation_attempts={"PCTDD-031": 1},
        implementation_attempts_by_cid={identity["canonical_task_cid"]: 1},
        task_identities={"PCTDD-031": identity},
    )
    state.save(paths.state)
    now = time.time()
    lifecycle = WorkspaceLifecycleRecord(
        task_id="PCTDD-031",
        canonical_task_cid=identity["canonical_task_cid"],
        attempt=1,
        lane_id="lane-0",
        state=lifecycle_state,
        owner=ProcessBirthIdentity(pid=0, start_time_ticks=0),
        lease_id="worktree-lease-1",
        fence=3,
        workspace_path=str(workspace.resolve()),
        branch="implementation/PCTDD-031/1",
        merge_target=config.merge_target_branch,
        created_at=now - 10,
        updated_at=now - 5,
        expires_at=now + 600,
        repo_root=str(config.repo_root.resolve()),
        state_dir=str(paths.root.resolve()),
    )

    class Store:
        def __init__(self, **_kwargs):
            pass

        def load_task_attempt(self, **lookup):
            assert lookup == {
                "canonical_task_cid": identity["canonical_task_cid"],
                "task_id": "PCTDD-031",
                "attempt": 1,
            }
            return lifecycle

        def load_workspace(self, path):
            assert path == lifecycle.workspace_path
            return lifecycle

    from ipfs_accelerate_py.agent_supervisor.merge import worktree_lifecycle

    monkeypatch.setattr(worktree_lifecycle, "WorktreeLifecycleStore", Store)

    projected = supervisor._database_portal_claim_lifecycle(
        source,
        task,
        attempt,
    )

    assert projected["task_revision"] == task.revision
    assert projected["lifecycle_record_id"] == lifecycle.record_id
    assert projected["post_provider_recovery"] is recovery

    stale = _task(receipt, revision=task.revision + 1)

    class StaleSource:
        def get_task(self, task_cid):
            return stale if task_cid == stale.task_cid else None

    with pytest.raises(RuntimeError, match="binding is not current"):
        supervisor._database_portal_claim_lifecycle(
            StaleSource(),
            stale,
            attempt,
        )


@pytest.mark.parametrize("symlink_kind", ["root", "binding"])
def test_exact_attempt_symlink_fails_before_binding_read(
    tmp_path,
    monkeypatch,
    symlink_kind,
):
    config = _config(tmp_path)
    supervisor = PortalImplementationSupervisor(config)
    task = _task(_receipt())
    attempt = supervisor._database_claim_attempt(task)

    class Source:
        def get_task(self, task_cid):
            return task if task_cid == task.task_cid else None

    source = Source()
    bridge = DatabasePortalExecutionBridge(
        task_source=source,
        attempt_root=config.state_dir
        / f"{config.state_prefix}_database_portal_attempts",
        portal_factory=lambda *_: None,
    )
    paths, _binding = bridge._ensure_attempt_projection(attempt, task)
    if symlink_kind == "root":
        escaped = tmp_path / "escaped-attempt-root"
        paths.root.rename(escaped)
        paths.root.symlink_to(escaped, target_is_directory=True)
        expected = "attempt root is not confined"
    else:
        escaped = tmp_path / "escaped-attempt-binding.json"
        paths.binding.rename(escaped)
        paths.binding.symlink_to(escaped)
        expected = "attempt artifact is not a confined regular file"

    monkeypatch.setattr(
        DatabasePortalExecutionBridge,
        "_read_binding",
        staticmethod(
            lambda _path: pytest.fail(
                "unsafe nominated binding must not be opened"
            )
        ),
    )

    with pytest.raises(RuntimeError, match=expected):
        supervisor._database_portal_claim_lifecycle(
            source,
            task,
            attempt,
        )


def test_remote_claim_and_unknown_receipt_defer_without_local_directory_scan(
    tmp_path,
    monkeypatch,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    local = _task(_receipt())
    remote = _task(
        _receipt(
            owner="pctdd-owner:lane-1",
            task_cid="cid:PCTDD-032",
        )
    )

    class Source:
        def list_tasks(self, *, status, cursor, limit):
            assert (status, cursor, limit) == ("in_progress", "", 128)
            return SimpleNamespace(
                tasks=(local, remote),
                revision=9,
                next_cursor="",
            )

        def get_task(self, task_cid):
            return {
                local.task_cid: local,
                remote.task_cid: remote,
            }.get(task_cid)

    monkeypatch.setattr(
        supervisor,
        "_database_portal_claim_lifecycle",
        lambda *_: pytest.fail("remote attempt directory must not be read"),
    )
    with pytest.raises(RuntimeError, match="another supervisor lane"):
        supervisor._database_portal_claim_projection(Source())

    malformed = dict(_receipt())
    malformed["self_selected_policy"] = "accept"
    with pytest.raises(RuntimeError, match="closed record"):
        supervisor._database_claim_attempt(_task(malformed))


def test_worktree_maintenance_holds_repo_lease_and_defers_on_quack_failure(
    tmp_path,
    monkeypatch,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    events: list[str] = []
    lease = SimpleNamespace(lock_path=tmp_path / "repo.lock")
    lease_held = False

    def acquire(*_args):
        nonlocal lease_held
        lease_held = True
        events.append("lease_acquired")
        return lease, "acquired", None

    def release(*_args, **_kwargs):
        nonlocal lease_held
        assert lease_held
        lease_held = False
        events.append("lease_released")
        return True

    monkeypatch.setattr(supervisor, "_acquire_supervisor_checkout_lease", acquire)
    monkeypatch.setattr(supervisor, "_release_supervisor_checkout_lease", release)
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection",
        lambda: {
            "defer_maintenance": True,
            "activity_detected": False,
            "reason": "database_portal_projection_inconclusive",
        },
    )
    for name in (
        "detect_stale_worktrees",
        "repair_stale_active_execution_state",
        "reconcile_backlogged_worktrees",
        "recover_already_merged_reconciliation_candidates",
        "_cleanup_backlogged_worktrees_locked",
    ):
        monkeypatch.setattr(
            supervisor,
            name,
            lambda *_, _name=name, **__: pytest.fail(
                f"{_name} must not mutate on inconclusive Quack state"
            ),
        )

    result = supervisor._run_database_portal_guarded_worktree_maintenance(
        lambda phase: events.append(phase),
        implementation_maintenance_lease=None,
        managed_daemon_launch_lock_held=True,
    )

    assert result["maintenance_blocked"] is True
    assert result["database_portal_reload_projection"]["activity_detected"] is False
    assert events == [
        "lease_acquired",
        "database_portal_reload_projection",
        "lease_released",
    ]
    assert lease_held is False


def test_retained_checkout_is_not_recovered_before_quack_projection(
    tmp_path,
    monkeypatch,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    retained = SimpleNamespace(lock_path=tmp_path / "retained.lock")

    monkeypatch.setattr(
        supervisor,
        "_retained_generated_checkout_lease",
        lambda: True,
    )
    monkeypatch.setattr(
        supervisor,
        "_current_supervisor_checkout_lease",
        lambda: retained,
    )
    import ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor as module

    monkeypatch.setattr(module, "checkout_mutation_lease_state", lambda _lease: "current")
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection",
        lambda: {
            "defer_maintenance": True,
            "activity_detected": False,
            "reason": "database_portal_projection_inconclusive",
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_recover_retained_generated_checkout_lease",
        lambda: pytest.fail("retained checkout must not mutate after Quack failure"),
    )

    result = supervisor._run_database_portal_guarded_worktree_maintenance(
        lambda _phase: None,
        implementation_maintenance_lease=None,
        managed_daemon_launch_lock_held=True,
    )

    assert result["maintenance_blocked"] is True
    assert result["reason"] == "database_portal_projection_inconclusive"
    assert result["database_portal_reload_projection"]["activity_detected"] is False


def test_public_run_once_quiesces_and_records_deferred_not_completed(
    tmp_path,
    monkeypatch,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    events: list[str] = []
    finished: list[tuple[str, str]] = []

    monkeypatch.setattr(
        supervisor,
        "_begin_supervisor_maintenance_heartbeat",
        lambda *_args, **_kwargs: (
            lambda phase: events.append(phase),
            lambda status="completed", error="": finished.append(
                (status, error)
            ),
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        lambda **kwargs: (
            events.append("quiesce")
            or {
                "quiesced": kwargs.get("_launch_lock_held") is True,
            }
        ),
    )

    def maintenance(_update, **kwargs):
        assert events == ["quiesce"]
        assert kwargs == {
            "include_refill": False,
            "managed_daemon_launch_lock_held": True,
        }
        events.append("maintenance")
        return {
            "stuck": False,
            "maintenance_blocked": True,
            "reason": "database_portal_projection_inconclusive",
        }

    monkeypatch.setattr(supervisor, "_run_once_with_maintenance", maintenance)

    result = supervisor.run_once(include_refill=False)

    assert events == ["quiesce", "maintenance"]
    assert result["maintenance_blocked"] is True
    assert finished == [
        ("deferred", "database_portal_projection_inconclusive")
    ]


def test_worktree_maintenance_stops_before_daemon_owned_reconcile(
    tmp_path,
    monkeypatch,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    events: list[str] = []
    lease = SimpleNamespace(lock_path=tmp_path / "repo.lock")
    lease_held = False

    def acquire(*_args):
        nonlocal lease_held
        assert not lease_held
        lease_held = True
        events.append("lease_acquired")
        return lease, "acquired", None

    def release(*_args, **_kwargs):
        nonlocal lease_held
        assert lease_held
        lease_held = False
        events.append("lease_released")
        return True

    def reconcile(*_args, **kwargs):
        if kwargs.get("rescue_dirty_only") is True:
            assert lease_held
            events.append("guarded_rescue")
            return {"attempted": True, "processed_count": 1}
        pytest.fail("daemon reconciliation must run in the later lease-free phase")

    monkeypatch.setattr(supervisor, "_acquire_supervisor_checkout_lease", acquire)
    monkeypatch.setattr(supervisor, "_release_supervisor_checkout_lease", release)
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection",
        _idle_projection,
    )
    monkeypatch.setattr(
        supervisor,
        "detect_stale_worktrees",
        lambda: events.append("detect") or {},
    )
    monkeypatch.setattr(
        supervisor,
        "repair_stale_active_execution_state",
        lambda: events.append("state_repair") or {},
    )
    monkeypatch.setattr(
        supervisor,
        "release_completed_leftover_execution",
        lambda **kwargs: (
            events.append("release_leftover")
            or {"launch_lock_held": kwargs.get("_launch_lock_held")}
        ),
    )
    monkeypatch.setattr(supervisor, "reconcile_backlogged_worktrees", reconcile)
    monkeypatch.setattr(
        supervisor,
        "_cleanup_backlogged_worktrees_locked",
        lambda: events.append("cleanup") or {},
    )
    monkeypatch.setattr(
        supervisor,
        "recover_already_merged_reconciliation_candidates",
        lambda **_kwargs: events.append("replay") or {},
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda: None,
    )
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )

    result = supervisor._run_database_portal_guarded_worktree_maintenance(
        lambda phase: events.append(f"phase:{phase}"),
        implementation_maintenance_lease={"lease_id": "implementation"},
        managed_daemon_launch_lock_held=True,
    )

    assert result["maintenance_blocked"] is False
    assert result["database_portal_daemon_reconciliation_pending"] is True
    assert events.index("guarded_rescue") < events.index("lease_released")
    assert "daemon_reconcile" not in events
    assert "replay" not in events
    assert lease_held is False


def test_daemon_reconciliation_runs_after_implementation_lease_release(
    tmp_path,
    monkeypatch,
):
    config = replace(
        _config(tmp_path),
        implementation_protected_paths=("todo.md",),
    )
    supervisor = PortalImplementationSupervisor(config)
    implementation_lock = config.state_path.parent / "implementation.lock"
    events: list[str] = []

    def supervisor_phase(
        _update,
        *,
        include_refill,
        implementation_maintenance_lease,
        managed_daemon_launch_lock_held,
    ):
        assert include_refill is False
        assert managed_daemon_launch_lock_held is True
        assert implementation_maintenance_lease is not None
        assert implementation_lock.exists()
        events.append("supervisor_phase")
        return {
            "stuck": False,
            "database_portal_daemon_reconciliation_pending": True,
        }

    def reconcile(*_args, **kwargs):
        assert not implementation_lock.exists()
        assert kwargs == {"allow_dirty_rescue": False}
        events.append("daemon_reconcile")
        return {"attempted": True, "reconciled_count": 1}

    def replay(*_args, **kwargs):
        assert not implementation_lock.exists()
        assert kwargs == {}
        events.append("daemon_replay")
        return {"attempted": True, "completed_count": 1}

    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance_under_lease",
        supervisor_phase,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda: None,
    )
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection",
        lambda: events.append("fresh_projection") or _idle_projection(),
    )
    monkeypatch.setattr(supervisor, "reconcile_backlogged_worktrees", reconcile)
    monkeypatch.setattr(
        supervisor,
        "recover_already_merged_reconciliation_candidates",
        replay,
    )

    result = supervisor._run_once_with_maintenance(
        lambda _phase: None,
        include_refill=False,
        managed_daemon_launch_lock_held=True,
    )

    assert events == [
        "supervisor_phase",
        "fresh_projection",
        "daemon_reconcile",
        "daemon_replay",
    ]
    assert result["maintenance_blocked"] is False
    assert result["database_portal_daemon_reconciliation_pending"] is False
    assert result["worktree_reconciliation"]["reconciled_count"] == 1
    assert not implementation_lock.exists()


def test_projection_checkout_lease_spans_later_supervisor_mutations(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    events: list[str] = []

    def assert_outer_lease(label):
        lease = supervisor._current_supervisor_checkout_lease()
        assert lease is not None, label
        assert supervisor._supervisor_checkout_transaction_depth() > 0, label
        events.append(label)

    def projection():
        assert_outer_lease("projection")
        return _idle_projection()

    def guarded(*_args, **_kwargs):
        projection_value = supervisor._database_portal_reload_projection()
        for label in (
            "stale_active_repair",
            "dirty_rescue",
            "cleanup",
        ):
            assert_outer_lease(label)
        return {
            "maintenance_blocked": False,
            "database_portal_reload_projection": projection_value,
            "retained_generated_checkout_recovery": {},
            "stale_worktree_detection": {},
            "stale_active_state_repair": {},
            "completed_leftover_execution": {},
            "worktree_cleanup": {},
            "database_portal_daemon_reconciliation_pending": True,
        }

    def main_repair(*, _checkout_lease_held=False):
        assert _checkout_lease_held is True
        assert_outer_lease("main_repair")
        return {"repaired": False}

    def generated_repair(*_args, **_kwargs):
        assert_outer_lease("generated_repair")
        return {}

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection",
        projection,
    )
    monkeypatch.setattr(
        supervisor,
        "_run_database_portal_guarded_worktree_maintenance",
        guarded,
    )
    monkeypatch.setattr(supervisor, "repair_main_checkout_merge_state", main_repair)
    monkeypatch.setattr(
        supervisor,
        "repair_generated_dirty_checkouts",
        generated_repair,
    )
    monkeypatch.setattr(supervisor, "ensure_event_log_file", lambda: {})
    monkeypatch.setattr(supervisor, "ensure_state_file", lambda: {})
    monkeypatch.setattr(
        supervisor,
        "_implementation_protected_maintenance_guard",
        lambda: {"blocked": False},
    )
    monkeypatch.setattr(supervisor, "ensure_strategy_file", lambda: {})
    monkeypatch.setattr(supervisor, "ensure_todo_board_for_refill", lambda: {})
    monkeypatch.setattr(
        supervisor,
        "migrate_legacy_objective_goal_completion",
        lambda: {},
    )
    monkeypatch.setattr(supervisor, "reconcile_objective_task_janitor", lambda: {})
    monkeypatch.setattr(
        supervisor,
        "record_reconciliation_guardrails",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        supervisor,
        "release_completed_guardrail_blocks",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(supervisor, "is_stuck", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(supervisor, "record_retry_budget_guardrails", lambda: [])
    monkeypatch.setattr(supervisor, "record_dependency_guardrails", lambda: [])
    monkeypatch.setattr(supervisor, "_record_event", lambda *_args, **_kwargs: None)

    result = supervisor._run_once_with_maintenance_under_lease(
        lambda _phase: None,
        include_refill=False,
        managed_daemon_launch_lock_held=True,
    )

    assert events[:4] == [
        "projection",
        "stale_active_repair",
        "dirty_rescue",
        "cleanup",
    ]
    assert "main_repair" in events
    assert events.count("generated_repair") == 2
    assert result["database_portal_daemon_reconciliation_pending"] is True
    assert supervisor._current_supervisor_checkout_lease() is None
