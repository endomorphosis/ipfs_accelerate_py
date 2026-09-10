from __future__ import annotations

import json
import os
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
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SupervisedChildIdentity,
)
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    ProcessBirthIdentity as SupervisedProcessBirthIdentity,
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


def _owner_status_with_replica(generation: int) -> dict[str, object]:
    binding = _owner_binding(generation)
    return {
        **binding,
        "read_replica": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "read-replica-observation@1"
            ),
            "authority": "non_authoritative_read_replica",
            "path": "/sealed/read-replica.duckdb",
            "source_database_path": "/sealed/control.duckdb",
            "server_id": binding["server_id"],
            "database_uuid": binding["database_uuid"],
            "generation": binding["generation"],
            "schema_revision": binding["schema_revision"],
            "schema_fingerprint": "schema-profile:test",
            "storage_schema_fingerprint": binding["schema_fingerprint"],
            "sha256": "sha256:" + ("4" * 64),
            "size_bytes": 4096,
            "refresh_sequence": generation,
            "refreshed_at_ms": 1_700_000_000_000 + generation,
            "live": True,
        },
    }


def _empty_quack_mutation_barrier() -> dict[str, object]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "fenced-provider-outer-mutation-barrier@1"
        ),
        "store_id": "state/control.duckdb",
        "active_request_count": 0,
        "active_processing_count": 0,
        "active_population_digest": (
            "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba8"
            "73c2f11161202b945"
        ),
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


def _changed_control_plane_projection() -> dict[str, object]:
    return {
        "control_plane_source_schema": "control-plane-source@1",
        "control_plane_source_id": "loaded-source",
        "control_plane_current_source_id": "current-source",
        "control_plane_source_tree_id": "loaded-tree",
        "control_plane_current_source_tree_id": "current-tree",
        "control_plane_source_revision": "loaded-revision",
        "control_plane_current_source_revision": "current-revision",
        "control_plane_update_pending": True,
        "control_plane_update_detected_at": "2026-09-01T21:09:42Z",
        "control_plane_reload_deferred": False,
        "control_plane_reload_deferred_reason": "",
        "control_plane_reload_deferred_task_id": "",
    }


def _stable_control_plane_projection() -> dict[str, object]:
    return {
        "control_plane_source_schema": "control-plane-source@1",
        "control_plane_source_id": "same-source",
        "control_plane_current_source_id": "same-source",
        "control_plane_source_tree_id": "same-tree",
        "control_plane_current_source_tree_id": "same-tree",
        "control_plane_source_revision": "same-revision",
        "control_plane_current_source_revision": "same-revision",
        "control_plane_update_pending": False,
        "control_plane_update_detected_at": "",
        "control_plane_reload_deferred": False,
        "control_plane_reload_deferred_reason": "",
        "control_plane_reload_deferred_task_id": "",
    }


@pytest.mark.parametrize(
    ("missing_fields", "expected_reason"),
    [
        (
            (
                "control_plane_current_source_revision",
                "control_plane_current_source_tree_id",
            ),
            "control_plane_current_source_incomplete",
        ),
        (
            (
                "control_plane_source_revision",
                "control_plane_source_tree_id",
            ),
            "control_plane_loaded_source_incomplete",
        ),
    ],
    ids=("current-source", "loaded-source"),
)
def test_source_reload_defers_incomplete_source_identity_before_portal_or_quiescence(
    tmp_path,
    monkeypatch,
    missing_fields,
    expected_reason,
):
    config = _config(tmp_path)
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    status = _changed_control_plane_projection()
    for field in missing_fields:
        status[field] = ""
    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        lambda: status,
    )
    monkeypatch.setattr(
        supervisor,
        "_record_event",
        lambda name, detail: events.append((name, detail)),
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        lambda: pytest.fail(
            "incomplete source must fail before portal lock acquisition"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "incomplete source must not quiesce the managed child"
        ),
    )

    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        SimpleNamespace(pid=987654),
        {},
    )

    assert decision.action == "continue"
    assert loop.config.status_extra_fields[
        "control_plane_reload_deferred_reason"
    ] == expected_reason
    assert loop.config.status_extra_fields[
        "control_plane_source_identity_missing_fields"
    ] == list(missing_fields)
    assert events[0][0] == "supervisor_control_plane_reload_deferred"


@pytest.mark.parametrize(
    ("projection", "expected_reason"),
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
            {
                **_authenticated_watchdog_projection(active=False),
                "unexpected": "not-closed",
            },
            "database_portal_projection_not_admitted",
        ),
        (
            {
                **_authenticated_watchdog_projection(active=False),
                "reason": "unrecognized-idle-claim",
            },
            "database_portal_projection_not_admitted",
        ),
        (
            {
                **_authenticated_watchdog_projection(active=False),
                "quack_owner": {
                    key: value
                    for key, value in _owner_binding(41).items()
                    if key != "process_birth_id"
                },
            },
            "database_portal_projection_not_admitted",
        ),
    ],
    ids=(
        "active",
        "inconclusive",
        "open-field-set",
        "unrecognized-idle-reason",
        "incomplete-owner-binding",
    ),
)
def test_source_reload_defers_unsafe_portal_before_quiescence(
    tmp_path,
    monkeypatch,
    projection,
    expected_reason,
):
    config = _config(tmp_path)
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    order: list[str] = []

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        _changed_control_plane_projection,
    )
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)

    @contextmanager
    def portal_fence():
        order.append("portal_fence_enter")
        try:
            yield config.database_program
        finally:
            order.append("portal_fence_exit")

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _program: order.append("pre_projection") or projection,
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "unsafe portal projection must not quiesce the managed child"
        ),
    )

    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        SimpleNamespace(pid=987654),
        {},
    )

    assert decision.action == "continue"
    assert order == [
        "portal_fence_enter",
        "pre_projection",
        "portal_fence_exit",
    ]
    assert loop.config.status_extra_fields[
        "control_plane_reload_deferred_reason"
    ] == expected_reason
    assert loop.config.status_extra_fields[
        "control_plane_reload_quiescence"
    ]["attempted"] is False


def test_source_reload_holds_portal_mutation_fence_across_idle_quiescence_and_postcheck(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    order: list[str] = []
    projection_count = 0

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        _changed_control_plane_projection,
    )
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )

    @contextmanager
    def portal_fence():
        order.append("portal_fence_enter")
        try:
            yield config.database_program
        finally:
            order.append("portal_fence_exit")

    def project(_program):
        nonlocal projection_count
        projection_count += 1
        order.append(f"projection_{projection_count}")
        return _authenticated_watchdog_projection(active=False)

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        project,
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: (
            order.append("quiesce")
            or {"quiesced": True, "supervised_child_alive": False}
        ),
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        SimpleNamespace(config=SimpleNamespace(status_extra_fields={})),
        SimpleNamespace(pid=987654),
        {},
    )

    assert order == [
        "portal_fence_enter",
        "projection_1",
        "quiesce",
        "projection_2",
        "portal_fence_exit",
    ]
    assert decision.action == "stop"
    assert decision.reason == "control_plane_source_changed"
    assert decision.status == CONTROL_PLANE_RELOAD_STATUS


@pytest.mark.parametrize("sealed_dispatch", [True, False])
def test_source_reload_postcheck_race_never_authorizes_reload(
    tmp_path,
    monkeypatch,
    sealed_dispatch,
):
    config = _config(tmp_path)
    config.plan_bound_dispatch = sealed_dispatch
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    projections = iter(
        (
            _authenticated_watchdog_projection(active=False),
            _authenticated_watchdog_projection(active=True),
        )
    )
    order: list[str] = []

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        _changed_control_plane_projection,
    )
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )

    @contextmanager
    def portal_fence():
        order.append("portal_fence_enter")
        try:
            yield config.database_program
        finally:
            order.append("portal_fence_exit")

    def project(_program):
        order.append("project")
        return next(projections)

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        project,
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: (
            order.append("quiesce") or {"quiesced": True}
        ),
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        SimpleNamespace(config=SimpleNamespace(status_extra_fields={})),
        SimpleNamespace(pid=987654),
        {},
    )

    assert order == [
        "portal_fence_enter",
        "project",
        "quiesce",
        "project",
        "portal_fence_exit",
    ]
    assert decision.reason != "control_plane_source_changed"
    if sealed_dispatch:
        assert decision.action == "recycle"
        assert decision.reason == "control_plane_reload_deferred"
    else:
        assert decision.action == "stop"
        assert decision.reason == "control_plane_reload_deferred_unsealed"
        assert decision.status == CONTROL_PLANE_RELOAD_DEFERRED_UNSEALED_STATUS


def test_source_reload_owner_mutation_lock_contention_defers_without_quiescence(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        _changed_control_plane_projection,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_reconciliation_program_environment",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        duckdb_state,
        "quack_owner_mutation_write_lock_path",
        lambda _store: tmp_path / "contended-owner.lock",
    )

    @contextmanager
    def contended_lock(_path, *, timeout_seconds):
        assert timeout_seconds == 60.0
        raise TimeoutError("owner mutation lock remains active")
        yield

    monkeypatch.setattr(duckdb_state, "exclusive_file_lock", contended_lock)
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "lock contention must not quiesce the managed child"
        ),
    )

    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        SimpleNamespace(pid=987654),
        {},
    )

    assert decision.action == "continue"
    assert loop.config.status_extra_fields[
        "control_plane_reload_deferred_reason"
    ] == "database_portal_projection_inconclusive"
    assert loop.config.status_extra_fields[
        "database_portal_reload_projection"
    ]["error_type"] == "TimeoutError"


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
    @contextmanager
    def portal_fence():
        yield config.database_program

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _program: (
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
    ] == projection_reason
    assert loop.config.status_extra_fields[
        "database_portal_projection_reason"
    ] == projection_reason
    assert loop.config.status_extra_fields[
        "database_portal_reload_projection"
    ] == projection
    assert supervisor._last_supervisor_maintenance_at > 0.0


def test_watchdog_releases_quack_owner_fence_before_idle_maintenance(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    PortalTaskState().save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    supervisor._last_supervisor_maintenance_at = 0.0
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    order: list[str] = []
    finished: list[tuple[str, str]] = []
    idle = _authenticated_watchdog_projection(active=False)

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        lambda: {"control_plane_update_pending": False},
    )

    @contextmanager
    def portal_fence():
        order.append("portal_fence_enter")
        try:
            yield config.database_program
        finally:
            order.append("portal_fence_exit")

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _program: order.append("projection") or idle,
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: (
            order.append("quiesce")
            or {"quiesced": True, "supervised_child_alive": False}
        ),
    )
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )
    monkeypatch.setattr(supervisor, "_record_event", lambda *_args: None)
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

    def maintenance(_update, **kwargs):
        assert order == [
            "portal_fence_enter",
            "projection",
            "quiesce",
            "projection",
            "portal_fence_exit",
        ]
        assert kwargs == {
            "managed_daemon_launch_lock_held": True,
            "database_portal_quiesced_idle": True,
            "database_portal_fenced_program": None,
        }
        order.append("maintenance")
        return {
            "stuck": False,
            "maintenance_blocked": False,
            "reason": "",
            "main_checkout_repair": {"repaired": False},
        }

    monkeypatch.setattr(supervisor, "_run_once_with_maintenance", maintenance)

    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        SimpleNamespace(pid=987654),
        {},
    )

    assert order == [
        "portal_fence_enter",
        "projection",
        "quiesce",
        "projection",
        "portal_fence_exit",
        "maintenance",
    ]
    assert decision.action == "recycle"
    assert decision.reason == "supervisor_maintenance_completed_after_quiescence"
    assert finished == [("completed", "")]


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
        lambda: pytest.fail(
            "non-Quack maintenance no longer performs a portal projection"
        ),
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
        assert events == ["quiesce"]
        assert kwargs == {
            "managed_daemon_launch_lock_held": True,
            "database_portal_fenced_program": None,
            "database_portal_quiesced_idle": False,
        }
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

    assert events == ["quiesce", "maintenance"]
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
    owner_values = iter(
        (
            _owner_binding(41),
            _owner_binding(after_generation),
            _owner_binding(after_generation),
        )
    )

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
        assert timeout_seconds == 60.0
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
        supervisor,
        "_database_portal_mutation_inbox_barrier",
        lambda store_id: {
            "store_id": store_id,
            "active_request_count": 0,
            "active_processing_count": 0,
        },
    )
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
    ("after_generation", "receipt_replica_tamper"),
    ((41, False), (42, False), (41, True)),
)
def test_outer_owner_receipt_uses_existing_fence_and_one_quack_transaction(
    tmp_path,
    monkeypatch,
    after_generation,
    receipt_replica_tamper,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        database_task_source,
        duckdb_state,
        intent_repository,
    )

    events: list[str] = []
    owner_values = iter(
        (
            _owner_status_with_replica(41),
            _owner_status_with_replica(after_generation),
        )
    )
    mutation_barrier = _empty_quack_mutation_barrier()

    class Connection:
        def __init__(self):
            self._quack_mutation_binding = _owner_binding(41)

        def execute(self, statement):
            events.append(statement)
            return self

        def close(self):
            events.append("close")

    connection = Connection()

    @contextmanager
    def owner_lock(_path, *, timeout_seconds):
        assert timeout_seconds == 60.0
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
    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda _uri, *, token: connection,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_mutation_inbox_barrier",
        lambda _store_id: dict(mutation_barrier),
    )

    def read_receipt(_source, **subject):
        events.append("receipt")
        assert subject["expected_store_id"] == "state/control.duckdb"
        replica = dict(subject["controller_replica_observation"])
        if receipt_replica_tamper:
            replica["sha256"] = "sha256:" + ("5" * 64)
        return {
            "authority": {
                "owner_binding": _owner_binding(41),
                "read_replica_observation": replica,
                "mutation_barrier": subject["controller_mutation_barrier"],
            }
        }

    monkeypatch.setattr(
        database_task_source.DatabaseTaskSource,
        "fenced_provider_outer_authority_population_receipt",
        read_receipt,
    )
    monkeypatch.setattr(
        intent_repository,
        "fenced_provider_outer_authority_population_receipt_valid",
        lambda _receipt: True,
    )
    subject = {
        "task_cid": "cid:PCTDD-006",
        "task_alias": "PCTDD-006",
        "task_revision": 28,
        "expected_task_status": "blocked",
        "attempt_id": "attempt:outer",
        "claim_id": "claim:outer",
        "lease_id": "lease:outer",
        "owner_session_id": "owner:lane-0",
        "fencing_token": 6,
        "fence_epoch": 0,
        "expected_store_id": "state/control.duckdb",
        "expected_store_generation": 41,
        "receipt_nonce": "nonce:outer",
        "receipt_epoch": 1,
    }
    if after_generation == 41 and not receipt_replica_tamper:
        result = supervisor._database_portal_fenced_provider_outer_authority_receipt(
            **subject
        )
        assert result["authority"]["owner_binding"]["generation"] == 41
    else:
        with pytest.raises(RuntimeError, match="generation changed"):
            supervisor._database_portal_fenced_provider_outer_authority_receipt(
                **subject
            )
    assert events == [
        "lock",
        "BEGIN TRANSACTION",
        "receipt",
        "COMMIT",
        "close",
        "unlock",
    ]


def test_outer_owner_receipt_and_task_cas_share_fence_and_transaction(
    tmp_path,
    monkeypatch,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    program = supervisor.config.database_program
    assert program is not None
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        database_task_source,
        duckdb_state,
        intent_repository,
    )

    events: list[str] = []
    owner_values = iter(
        (_owner_status_with_replica(41), _owner_status_with_replica(41))
    )
    mutation_barrier = _empty_quack_mutation_barrier()

    class Connection:
        def __init__(self):
            self._quack_mutation_binding = _owner_binding(41)
            self.in_transaction = False

        def execute(self, statement, _parameters=None):
            events.append(statement)
            if statement == "BEGIN TRANSACTION":
                self.in_transaction = True
            elif statement in {"COMMIT", "ROLLBACK"}:
                self.in_transaction = False
            return self

        def close(self):
            events.append("close")

    connection = Connection()

    @contextmanager
    def mutation_fence():
        events.append("lock")
        try:
            yield program
        finally:
            events.append("unlock")

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        mutation_fence,
    )
    monkeypatch.setattr(
        duckdb_state,
        "_resolve_quack_token_handle",
        lambda **_kwargs: ("not-published", next(owner_values)),
    )
    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda _uri, *, token: connection,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_mutation_inbox_barrier",
        lambda _store_id: dict(mutation_barrier),
    )

    def read_receipt(_source, **subject):
        events.append("receipt")
        return {
            "authority": {
                "owner_binding": _owner_binding(41),
                "read_replica_observation": dict(
                    subject["controller_replica_observation"]
                ),
                "mutation_barrier": dict(
                    subject["controller_mutation_barrier"]
                ),
            }
        }

    monkeypatch.setattr(
        database_task_source.DatabaseTaskSource,
        "fenced_provider_outer_authority_population_receipt",
        read_receipt,
    )
    monkeypatch.setattr(
        intent_repository,
        "fenced_provider_outer_authority_population_receipt_valid",
        lambda _receipt: True,
    )

    def apply_cas(_repository, actual_connection, **arguments):
        assert actual_connection is connection
        assert connection.in_transaction is True
        assert arguments["task_cid"] == "cid:PCTDD-006"
        assert arguments["expected_revision"] == 28
        events.append("cas")
        return SimpleNamespace(changed=True, revision=29)

    monkeypatch.setattr(
        intent_repository.IntentRepository,
        "_cas_task_status_on_connection",
        apply_cas,
    )
    subject = {
        "task_cid": "cid:PCTDD-006",
        "task_alias": "PCTDD-006",
        "task_revision": 28,
        "expected_task_status": "blocked",
        "attempt_id": "attempt:outer",
        "claim_id": "claim:outer",
        "lease_id": "lease:outer",
        "owner_session_id": "owner:lane-0",
        "fencing_token": 6,
        "fence_epoch": 0,
        "expected_store_id": "state/control.duckdb",
        "expected_store_generation": 41,
        "receipt_nonce": "nonce:outer",
        "receipt_epoch": 1,
    }

    def callback(receipt, pinned):
        events.append("callback")
        assert receipt["authority"]["owner_binding"]["generation"] == 41
        result = pinned.cas_task_status(
            task_cid="cid:PCTDD-006",
            expected_revision=28,
            new_status="retrying",
        )
        with pytest.raises(RuntimeError, match="one-shot"):
            pinned.cas_task_status(
                task_cid="cid:PCTDD-006",
                expected_revision=29,
                new_status="blocked",
            )
        return {"resulting_revision": result.revision}

    result = (
        supervisor._database_portal_execute_with_fenced_provider_outer_authority_cas(
            subject=subject,
            callback=callback,
        )
    )

    assert result == {"resulting_revision": 29}
    assert events == [
        "lock",
        "BEGIN TRANSACTION",
        "receipt",
        "callback",
        "cas",
        "COMMIT",
        "close",
        "unlock",
    ]


@pytest.mark.parametrize("suffix", ["done", "cancelled"])
def test_outer_owner_barrier_retains_results_beyond_pending_budget(tmp_path, monkeypatch, suffix):
    from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server

    repo = tmp_path / "repo"
    inbox = repo / "state" / "quack-owner" / "mutations"
    inbox.mkdir(parents=True, mode=0o700)
    monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(repo))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "state/control.duckdb")
    monkeypatch.setattr(quack_state_server, "MUTATION_MAX_DIRECTORY_ENTRIES", 2)
    for letter in "abcd":
        (inbox / ("b" + letter * 40 + f".{suffix}.json")).write_text("retained")
    barrier = PortalImplementationSupervisor._database_portal_mutation_inbox_barrier("state/control.duckdb")
    assert barrier["active_request_count"] == barrier["active_processing_count"] == 0
    assert len(list(inbox.iterdir())) == 4
    active = inbox / ("b" + "z" * 40 + ".processing.json")
    active.write_text("{}")
    with pytest.raises(RuntimeError, match="unsettled owner mutation"):
        PortalImplementationSupervisor._database_portal_mutation_inbox_barrier("state/control.duckdb")
    active.unlink()
    unsafe = inbox / ("b" + "z" * 40 + ".done.json")
    unsafe.symlink_to(tmp_path / "absent")
    with pytest.raises(RuntimeError, match="entry is not admitted"):
        PortalImplementationSupervisor._database_portal_mutation_inbox_barrier("state/control.duckdb")


@pytest.mark.parametrize("active_suffix", ["request", "processing"])
def test_outer_owner_receipt_barrier_rejects_unsettled_quack_mutation(
    tmp_path,
    monkeypatch,
    active_suffix,
):
    repo = tmp_path / "repo"
    inbox = repo / "state" / "quack-owner" / "mutations"
    inbox.mkdir(parents=True, mode=0o700)
    inbox.chmod(0o700)
    monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(repo))
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
        "state/control.duckdb",
    )
    active = inbox / ("b" + ("a" * 40) + f".{active_suffix}.json")
    active.write_text("{}", encoding="utf-8")
    active.chmod(0o600)

    with pytest.raises(RuntimeError, match="unsettled owner mutation"):
        PortalImplementationSupervisor._database_portal_mutation_inbox_barrier(
            "state/control.duckdb"
        )

    active.unlink()
    barrier = (
        PortalImplementationSupervisor._database_portal_mutation_inbox_barrier(
            "state/control.duckdb"
        )
    )
    assert barrier["active_request_count"] == 0
    assert barrier["active_processing_count"] == 0


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
    seen: list[str] = []

    class MixedSource:
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

    def local_lifecycle(_source, task, attempt):
        cid = str(getattr(task, "task_cid", "") or "")
        seen.append(cid)
        if cid != local.task_cid:
            raise AssertionError("remote attempt directory must not be read")
        return {
            "task_id": "PCTDD-031",
            "task_cid": cid,
            "task_revision": 11,
            "attempt_id": str(attempt.attempt_id),
            "binding_id": "binding-1",
            "lifecycle_record_id": "lifecycle-1",
            "lifecycle_state": "claimed",
            "workspace_path": str(tmp_path / "worktree"),
            "active_phase": "implementing",
            "post_provider_recovery": False,
        }

    monkeypatch.setattr(
        supervisor,
        "_database_portal_claim_lifecycle",
        local_lifecycle,
    )
    mixed = supervisor._database_portal_claim_projection(MixedSource())
    assert seen == [local.task_cid]
    assert mixed["task_ids"] == ["PCTDD-031"]
    assert mixed["reason"] == "database_portal_claim_or_recovery_saga"
    assert mixed["activity_detected"] is True

    class RemoteOnlySource:
        def list_tasks(self, *, status, cursor, limit):
            return SimpleNamespace(
                tasks=(remote,),
                revision=9,
                next_cursor="",
            )

        def get_task(self, task_cid):
            return remote if task_cid == remote.task_cid else None

    monkeypatch.setattr(
        supervisor,
        "_database_portal_claim_lifecycle",
        lambda *_: pytest.fail("remote attempt directory must not be read"),
    )
    idle = supervisor._database_portal_claim_projection(RemoteOnlySource())
    assert idle["reason"] == "database_portal_population_idle"
    assert idle["activity_detected"] is False
    assert idle["defer_maintenance"] is False
    assert idle["defer_reload"] is False
    assert idle["attempts"] == []
    assert idle["task_ids"] == []

    malformed = dict(_receipt())
    malformed["self_selected_policy"] = "accept"
    with pytest.raises(RuntimeError, match="closed record"):
        supervisor._database_claim_attempt(_task(malformed))


def test_peer_home_shard_in_progress_does_not_fail_close_home_projection(
    tmp_path,
    monkeypatch,
):
    config = replace(
        _config(tmp_path),
        task_shard_count=4,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    supervisor = PortalImplementationSupervisor(config)
    peer = TaskRecord(
        task_cid="cid:PCTDD-034",
        task_alias="PCTDD-034",
        goal_cid="PCTDD-G033",
        ordinal=34,
        status="in_progress",
        revision=50,
        priority="P1",
        body={
            "title": "peer-lane claim",
            "completion_receipt": {
                **_receipt(owner="pctdd-owner:lane-0", task_cid="cid:PCTDD-034"),
                "retained_recovery_admission": {"task_cid": "not-a-chain"},
            },
        },
    )

    class PeerSource:
        def list_tasks(self, *, status, cursor, limit):
            return SimpleNamespace(tasks=(peer,), revision=12, next_cursor="")

        def get_task(self, task_cid):
            raise AssertionError("peer-home-shard claim must not be fetched")

    monkeypatch.setattr(
        supervisor,
        "_database_portal_claim_lifecycle",
        lambda *_: pytest.fail("peer attempt directory must not be read"),
    )
    monkeypatch.setattr(
        supervisor,
        "_database_claim_attempt",
        lambda *_: pytest.fail("peer retained chain must not be validated"),
    )
    idle = supervisor._database_portal_claim_projection(PeerSource())
    assert idle["reason"] == "database_portal_population_idle"
    assert idle["activity_detected"] is False
    assert idle["defer_maintenance"] is False
    assert idle["attempts"] == []


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
        "_database_portal_reload_projection_fenced",
        lambda _program: {
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
        database_portal_fenced_program=supervisor.config.database_program,
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
        "_database_portal_reload_projection_fenced",
        lambda _program: {
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
        database_portal_fenced_program=supervisor.config.database_program,
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

    @contextmanager
    def portal_fence():
        events.append("portal_fence_enter")
        try:
            yield supervisor.config.database_program
        finally:
            events.append("portal_fence_exit")

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _program: (
            events.append("projection")
            or _authenticated_watchdog_projection(active=False)
        ),
    )

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
        assert events == [
            "portal_fence_enter",
            "projection",
            "quiesce",
            "projection",
            "portal_fence_exit",
        ]
        assert kwargs == {
            "include_refill": False,
            "managed_daemon_launch_lock_held": True,
            "database_portal_quiesced_idle": True,
        }
        events.append("maintenance")
        return {
            "stuck": False,
            "maintenance_blocked": True,
            "reason": "database_portal_projection_inconclusive",
        }

    monkeypatch.setattr(supervisor, "_run_once_with_maintenance", maintenance)

    result = supervisor.run_once(include_refill=False)

    assert events == [
        "portal_fence_enter",
        "projection",
        "quiesce",
        "projection",
        "portal_fence_exit",
        "maintenance",
    ]
    assert result["maintenance_blocked"] is True
    assert finished == [
        ("deferred", "database_portal_projection_inconclusive")
    ]


@pytest.mark.parametrize("event_io_failure", [False, True])
def test_public_run_once_preserves_current_retained_startup_blocker(
    tmp_path,
    monkeypatch,
    event_io_failure,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    recorded: list[tuple[str, dict[str, object]]] = []
    finished: list[tuple[str, str]] = []
    blocked = {
        "reconciled": False,
        "blocked": True,
        "reason": "database_portal_retained_reconciliation_failed",
        "reconciliation_complete": False,
        "quiesced": False,
        "safe_to_restart": False,
        "retained_occurrence_reconciliation": {
            "blocked": True,
            "error_type": "DatabaseImplementationConflictError",
        },
    }

    @contextmanager
    def portal_fence():
        yield supervisor.config.database_program

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _program: _authenticated_watchdog_projection(active=False),
    )
    monkeypatch.setattr(
        supervisor,
        "_retained_fenced_provider_program_applicable",
        lambda _program: True,
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        lambda **_kwargs: {"quiesced": True},
    )
    monkeypatch.setattr(
        supervisor,
        "_reconcile_interrupted_database_portal_attempts_bound",
        lambda *_args, **_kwargs: dict(blocked),
    )
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda *_args, **_kwargs: pytest.fail(
            "blocked retained recovery must not enter ordinary maintenance"
        ),
    )
    def record_event(name, payload):
        if event_io_failure:
            raise OSError("injected event-store outage")
        recorded.append((name, dict(payload)))

    monkeypatch.setattr(supervisor, "_record_event", record_event)
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

    result = supervisor.run_once(include_refill=False)

    assert result["reason"] == blocked["reason"]
    assert result["reason"] != "database_portal_owner_mutation_fence_unavailable"
    assert result["retained_startup_reconciliation"] == blocked
    assert recorded == (
        []
        if event_io_failure
        else [("database_portal_retained_startup_blocked", result)]
    )
    assert finished == [("deferred", blocked["reason"])]


def test_retained_startup_allows_launch_when_peer_writer_is_busy() -> None:
    busy = {
        "reconciled": False,
        "blocked": False,
        "reason": "database_portal_retained_peer_writer_busy",
        "reconciliation_complete": False,
        "quiesced": True,
        "safe_to_restart": True,
        "peer_lane": 3,
        "error": "embedded execution store already has an active database writer",
    }
    assert (
        PortalImplementationSupervisor._retained_startup_allows_normal_launch(
            busy
        )
        is True
    )
    blocked = dict(busy)
    blocked["safe_to_restart"] = False
    assert (
        PortalImplementationSupervisor._retained_startup_allows_normal_launch(
            blocked
        )
        is False
    )
    post_cas = {
        "reconciled": False,
        "blocked": True,
        "reason": "database_portal_post_cas_reconciliation_blocked",
        "reconciliation_complete": False,
        "quiesced": False,
        "safe_to_restart": False,
    }
    assert (
        PortalImplementationSupervisor._retained_startup_allows_normal_launch(
            post_cas
        )
        is True
    )


def test_extra_gate_shard_defers_mutation_fence_timeout_until_next_pass(
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace

    config = replace(_config(tmp_path), task_shard_index=2)
    supervisor = PortalImplementationSupervisor(config)
    lock_attempts = {"count": 0}

    @contextmanager
    def exclusive_lock(_path, *, timeout_seconds=60.0):
        lock_attempts["count"] += 1
        if lock_attempts["count"] == 1:
            raise TimeoutError(
                "timed out acquiring DuckDB process lock: write-transaction.lock"
            )
        yield

    monkeypatch.setattr(
        supervisor,
        "_database_reconciliation_program_environment",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state.exclusive_file_lock",
        exclusive_lock,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state.quack_owner_mutation_write_lock_path",
        lambda *_args, **_kwargs: tmp_path / "write-transaction.lock",
    )

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        _DatabasePortalOwnerMutationFenceDeferred,
    )

    with pytest.raises(_DatabasePortalOwnerMutationFenceDeferred) as deferred:
        with supervisor._database_portal_reload_mutation_fence():
            pytest.fail("a timed-out mutation fence must not enter its body")
    assert isinstance(deferred.value.__cause__, TimeoutError)
    assert lock_attempts["count"] == 1

    # A later supervisor pass may retry; one pass never waits in a loop.
    with supervisor._database_portal_reload_mutation_fence() as program:
        assert program is supervisor.config.database_program

    assert lock_attempts["count"] == 2
    assert supervisor._database_portal_extra_gate_shard_must_wait_for_mutation_fence() is True


def test_lane0_mutation_fence_timeout_still_fail_closes(
    tmp_path,
    monkeypatch,
):
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    finished: list[tuple[str, str]] = []

    @contextmanager
    def portal_fence():
        raise TimeoutError(
            "timed out acquiring DuckDB process lock: write-transaction.lock"
        )
        yield supervisor.config.database_program

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
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

    result = supervisor.run_once(include_refill=False)

    assert result["maintenance_blocked"] is True
    assert result["reason"] == "database_portal_owner_mutation_fence_unavailable"
    assert finished == [
        ("deferred", "database_portal_owner_mutation_fence_unavailable")
    ]
    assert supervisor._database_portal_extra_gate_shard_must_wait_for_mutation_fence() is True


def test_public_run_once_does_not_label_peer_writer_busy_as_fence_unavailable(
    tmp_path,
    monkeypatch,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationAuthorityError,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        DATABASE_PORTAL_RETAINED_PEER_WRITER_BUSY_REASON,
    )

    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    recorded: list[tuple[str, dict[str, object]]] = []
    finished: list[tuple[str, str]] = []

    @contextmanager
    def portal_fence():
        yield supervisor.config.database_program

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _program: _authenticated_watchdog_projection(active=False),
    )
    monkeypatch.setattr(
        supervisor,
        "_retained_fenced_provider_program_applicable",
        lambda _program: True,
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        lambda **_kwargs: {"quiesced": True},
    )

    def explode(*_args, **_kwargs):
        raise DatabaseImplementationAuthorityError(
            "embedded execution store already has an active database writer"
        )

    monkeypatch.setattr(
        supervisor,
        "_reconcile_interrupted_database_portal_attempts_bound",
        explode,
    )
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda *_args, **_kwargs: pytest.fail(
            "peer writer busy must not fall through to ordinary maintenance"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_record_event",
        lambda name, payload: recorded.append((name, dict(payload))),
    )
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

    result = supervisor.run_once(include_refill=False)

    assert result["reason"] == DATABASE_PORTAL_RETAINED_PEER_WRITER_BUSY_REASON
    assert result["reason"] != "database_portal_owner_mutation_fence_unavailable"
    assert result.get("maintenance_blocked") is not True
    assert result["safe_to_restart"] is True
    assert recorded == [
        ("database_portal_retained_peer_writer_busy", result)
    ]
    assert finished == [("completed", "")]


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
        "_database_portal_reload_projection_fenced",
        lambda _program: _idle_projection(),
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
        lambda **_kwargs: events.append("cleanup") or {},
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
        database_portal_fenced_program=supervisor.config.database_program,
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
        database_portal_fenced_program,
    ):
        assert include_refill is False
        assert managed_daemon_launch_lock_held is True
        assert implementation_maintenance_lease is not None
        assert database_portal_fenced_program is config.database_program
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
        "_database_portal_reload_projection_fenced",
        lambda _program: events.append("fresh_projection") or _idle_projection(),
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
        database_portal_fenced_program=config.database_program,
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

    def projection(_program):
        assert_outer_lease("projection")
        return _idle_projection()

    def guarded(*_args, **_kwargs):
        projection_value = supervisor._database_portal_reload_projection_fenced(
            config.database_program
        )
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
        "_database_portal_reload_projection_fenced",
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
        database_portal_fenced_program=config.database_program,
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


def test_retained_pins_are_already_rearmed_for_retrying_extra_gate_tasks() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
        DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
    )

    pins = (
        *DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
        DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
    )
    tasks = {
        str(pin["task_cid"]): SimpleNamespace(
            task_cid=str(pin["task_cid"]),
            task_alias=str(pin["task_alias"]),
            status="retrying",
            body={
                "completion_receipt": {
                    "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
                    "forced_block": False,
                }
            },
        )
        for pin in pins
    }
    daemon = SimpleNamespace(task_source=SimpleNamespace(get=lambda cid: tasks.get(str(cid))))
    assert PortalImplementationSupervisor._retained_pins_are_already_rearmed(
        daemon
    ) is True
    tasks[str(pins[0]["task_cid"])].status = "in_progress"
    assert PortalImplementationSupervisor._retained_pins_are_already_rearmed(
        daemon
    ) is True
    tasks[str(pins[0]["task_cid"])].status = "retrying"
    tasks[str(pins[0]["task_cid"])].body["completion_receipt"] = {
        "operation": "database_retry_rearmed",
        "retry_exhausted": False,
    }
    assert PortalImplementationSupervisor._retained_pins_are_already_rearmed(
        daemon
    ) is True
    tasks[str(pins[0]["task_cid"])].status = "blocked"
    tasks[str(pins[0]["task_cid"])].body["completion_receipt"] = {
        "operation": "database_unknown_outcome_blocked",
        "authority_outcome": "unknown",
        "forced_block": True,
    }
    assert PortalImplementationSupervisor._retained_pins_are_already_rearmed(
        daemon
    ) is True
    tasks[str(pins[0]["task_cid"])].body["completion_receipt"][
        "operation"
    ] = "database_retry_exhausted"
    assert PortalImplementationSupervisor._retained_pins_are_already_rearmed(
        daemon
    ) is False


def test_later_epoch_blocked_extra_gate_pin_is_already_rearmed() -> None:
    """034 blocked after a later attempt must not fence home-lane prelaunch."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
        DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
        DatabaseImplementationDaemon,
    )

    pins = (
        *DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
        DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
    )
    tasks = {
        str(pin["task_cid"]): SimpleNamespace(
            task_cid=str(pin["task_cid"]),
            task_alias=str(pin["task_alias"]),
            status="retrying",
            revision=int(pin["blocked_task_revision"]) + 4,
            body={
                "completion_receipt": {
                    "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
                    "forced_block": False,
                    "retained_recovery_admission": {"schema": "stale-one-shot"},
                    "retained_recovery_consumption": {
                        "schema": "stale-one-shot"
                    },
                }
            },
        )
        for pin in pins
    }
    pin_034 = next(pin for pin in pins if pin["task_alias"] == "PCTDD-034")
    tasks[str(pin_034["task_cid"])].status = "blocked"
    tasks[str(pin_034["task_cid"])].body["completion_receipt"] = {
        "operation": "database_portal_projection_incomplete",
        "forced_block": True,
    }
    daemon = SimpleNamespace(
        task_source=SimpleNamespace(get=lambda cid: tasks.get(str(cid)))
    )
    assert PortalImplementationSupervisor._retained_pins_are_already_rearmed(
        daemon
    ) is True
    assert (
        DatabaseImplementationDaemon._retained_occurrence_has_left_sealed_pin(
            tasks[str(pin_034["task_cid"])],
            pin_034,
        )
        is True
    )
    sealed = SimpleNamespace(
        task_cid=str(pin_034["task_cid"]),
        task_alias="PCTDD-034",
        status="blocked",
        revision=int(pin_034["blocked_task_revision"]),
        body={
            "completion_receipt": {
                "operation": "database_unknown_outcome_blocked"
            }
        },
    )
    assert (
        DatabaseImplementationDaemon._retained_occurrence_has_left_sealed_pin(
            sealed,
            pin_034,
        )
        is False
    )
    tasks[str(pin_034["task_cid"])].revision = int(
        pin_034["blocked_task_revision"]
    )
    tasks[str(pin_034["task_cid"])].body["completion_receipt"] = {
        "operation": "database_unknown_outcome_blocked",
        "authority_outcome": "unknown",
        "forced_block": True,
    }
    assert PortalImplementationSupervisor._retained_pins_are_already_rearmed(
        daemon
    ) is True
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_protected_checkout_peer_deferral_is_recognized() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    assert (
        DatabaseImplementationDaemon._exception_is_protected_checkout_peer_deferral(
            RuntimeError("external_protected_checkout_recovery_required")
        )
        is True
    )
    assert (
        DatabaseImplementationDaemon._exception_is_protected_checkout_peer_deferral(
            RuntimeError("protected_recovery_owner_active")
        )
        is True
    )
    assert (
        DatabaseImplementationDaemon._exception_is_protected_checkout_peer_deferral(
            RuntimeError("callback_authority_incomplete_blocked")
        )
        is False
    )



def test_legacy_receipt_core_equality_does_not_override_native_launch_fence() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        _prepared_reconciliation_barrier_core_matches,
    )

    prepared = {
        "stage": "prepared",
        "receipt_id": "sha256:" + "a" * 64,
        "reason": "nested_quiesced",
        "nested_state": {"state_digest": "sha256:" + "b" * 64, "age_seconds": 0.776},
        "backoff_seconds": 20.0,
    }
    barrier = {
        "stage": "commit_barrier",
        "receipt_id": "sha256:" + "c" * 64,
        "prepared_reconciliation_receipt_id": prepared["receipt_id"],
        "reason": "nested_quiesced",
        "nested_state": {"state_digest": "sha256:" + "b" * 64, "age_seconds": 0.776},
        "backoff_seconds": 20.0,
    }
    assert _prepared_reconciliation_barrier_core_matches(prepared, barrier) is True
    mismatched = {**barrier, "backoff_seconds": 30.0}
    assert (
        _prepared_reconciliation_barrier_core_matches(prepared, mismatched)
        is False
    )

    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )
    assert PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_checkout_fence_unavailable",
        }
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        _DatabasePortalOwnerMutationFenceDeferred,
    )

    timed = TimeoutError(
        "timed out acquiring DuckDB process lock: write-transaction.lock"
    )
    assert (
        PortalImplementationSupervisor._database_portal_mutation_fence_acquisition_timed_out(
            timed
        )
        is True
    )
    assert issubclass(
        _DatabasePortalOwnerMutationFenceDeferred, RuntimeError
    )
    assert (
        PortalImplementationSupervisor._extra_gate_portal_state_must_preserve_worker(
            {"active_task_id": "PCTDD-005"}
        )
        is True
    )
    assert (
        PortalImplementationSupervisor._extra_gate_portal_state_must_preserve_worker(
            {"active_task_id": "PCTDD-008"}
        )
        is False
    )


def test_extra_gate_grok_descendants_are_preserved_on_exited_supervisor() -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        _process_argv_is_grok_cli_runner,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    grok_argv = [
        "/usr/bin/python3.12",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "--workspace",
        "/tmp/workspace_b6c6c987ab22",
        "--model",
        "grok-4.6",
    ]
    assert _process_argv_is_grok_cli_runner(grok_argv) is True
    assert (
        _process_argv_is_grok_cli_runner(
            ["/usr/bin/python3.12", "-m", "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"]
        )
        is False
    )
    assert (
        _process_argv_is_grok_cli_runner(
            [
                "/usr/bin/python3.12",
                "-c",
                "protected-path ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
            ]
        )
        is False
    )
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_extra_gate_preserve_uses_daemon_pid_when_supervisor_profile_misses_grok(
    monkeypatch,
) -> None:
    """Daemon is a session leader, so supervisor snapshot misses grok.

    Master restarting-exited never logged preserving extra-gate and
    SIGTERM-killed PCTDD-005 grok 1053569. Extra-gate aliases still
    cannot bypass safe_to_restart=False.
    """

    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        _extra_gate_grok_descendants_must_preserve,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner._pid_tree_has_grok_cli_runner",
        lambda pid: int(pid) == 951233,
    )
    process = SimpleNamespace(_agent_supervisor_lifecycle_profile=None)
    assert (
        _extra_gate_grok_descendants_must_preserve(process, daemon_pid=951233)
        is True
    )
    assert (
        _extra_gate_grok_descendants_must_preserve(process, daemon_pid=1)
        is False
    )
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_restarting_stale_or_exited_preserves_extra_gate_via_supervisor_pid(
    monkeypatch,
) -> None:
    """Stale restart fenced extra-gate grok because only exited-path preserved.

    Master logged restarting exited with zero preserving extra-gate lines
    and SIGTERM-killed PCTDD-005. Daemon pid markers can already be gone;
    scan the supervisor pid tree. Extra-gate aliases still cannot bypass
    ``safe_to_restart=False``.
    """

    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        _restarting_track_must_preserve_extra_gate_grok,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner._pid_tree_has_grok_cli_runner",
        lambda pid: int(pid) == 2668379,
    )
    process = SimpleNamespace(
        pid=2668379,
        _agent_supervisor_lifecycle_profile=None,
    )
    assert (
        _restarting_track_must_preserve_extra_gate_grok(
            process,
            {"daemon_pid": None, "daemon_status": "missing"},
        )
        is True
    )
    assert (
        _restarting_track_must_preserve_extra_gate_grok(
            process,
            {"daemon_pid": 2681548, "stale_daemon_pid": None},
        )
        is False
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner._pid_tree_has_grok_cli_runner",
        lambda pid: int(pid) == 2681548,
    )
    assert (
        _restarting_track_must_preserve_extra_gate_grok(
            process,
            {"daemon_pid": 2681548},
        )
        is True
    )
    _extra_gate_cannot_bypass_safe_to_restart()


def test_source_reload_defers_without_quiesce_when_extra_gate_in_progress(
    tmp_path,
    monkeypatch,
):
    """Watchdog source-change must not kill extra-gate in-progress daemons.

    run_once skip-terminate never ran on this path, so PCTDD-005 was
    quiesced mid-worktree on control_plane_source_changed. Extra-gate
    aliases still cannot bypass safe_to_restart=False.
    """

    config = _config(tmp_path)
    state = PortalTaskState()
    state.active_task_id = "PCTDD-005"
    state.save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    order: list[str] = []

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        _changed_control_plane_projection,
    )
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )

    @contextmanager
    def portal_fence():
        order.append("portal_fence_enter")
        try:
            yield config.database_program
        finally:
            order.append("portal_fence_exit")

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _program: (
            order.append("projection")
            or _authenticated_watchdog_projection(active=False)
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "extra-gate in-progress must not quiesce the managed child"
        ),
    )

    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        SimpleNamespace(pid=987654),
        {},
    )

    assert order == [
        "portal_fence_enter",
        "projection",
        "portal_fence_exit",
    ]
    assert decision.action == "continue"
    assert decision.reason != "control_plane_source_changed"
    assert loop.config.status_extra_fields[
        "control_plane_reload_deferred_reason"
    ] == "extra_gate_in_progress_preserve_worker"
    assert loop.config.status_extra_fields[
        "control_plane_reload_deferred_task_id"
    ] == "PCTDD-005"
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_watchdog_maintenance_preserves_extra_gate_in_progress(
    tmp_path,
    monkeypatch,
):
    """Periodic watchdog maintenance must not kill extra-gate grok.

    Source-change skip never ran on this path, so PCTDD-005 grok was
    quiesced as supervisor_watchdog_maintenance. Extra-gate aliases
    still cannot bypass safe_to_restart=False.
    """

    config = _config(tmp_path)
    state = PortalTaskState()
    state.active_task_id = "PCTDD-005"
    state.save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    supervisor._last_supervisor_maintenance_at = 0.0

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        _stable_control_plane_projection,
    )
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "extra-gate in-progress must not quiesce for watchdog maintenance"
        ),
    )

    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        SimpleNamespace(pid=987654),
        {},
    )

    assert decision.action == "continue"
    assert decision.reason != "control_plane_source_changed"
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_watchdog_maintenance_preserves_live_grok_without_active_task_id(
    tmp_path,
    monkeypatch,
):
    """Empty portal active_task_id must not quiesce a live extra-gate grok.

    After watchdog_startup_grace_seconds the first maintenance pass killed
    PCTDD-005 grok because PortalTaskState.active_task_id was already
    cleared. Extra-gate aliases still cannot bypass safe_to_restart=False.
    """

    config = _config(tmp_path)
    state = PortalTaskState()
    state.active_task_id = ""
    state.save(config.state_path)
    supervisor = PortalImplementationSupervisor(config)
    supervisor._last_supervisor_maintenance_at = 0.0

    monkeypatch.setattr(
        supervisor,
        "_control_plane_status_projection",
        _stable_control_plane_projection,
    )
    monkeypatch.setattr(supervisor, "_record_event", lambda *_: None)
    monkeypatch.setattr(
        supervisor,
        "_active_agent_worker_processes",
        lambda *_args, **_kwargs: (
            [
                {
                    "pid": 3112389,
                    "cmdline": (
                        "python3.12",
                        "-m",
                        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
                    ),
                }
            ]
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_quiesce_supervised_child_for_control_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "live grok must not be quiesced when portal active_task_id is empty"
        ),
    )

    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    decision = supervisor._supervisor_loop_watchdog_decision(
        loop,
        SimpleNamespace(pid=987654),
        {},
    )

    assert decision.action == "continue"
    assert decision.reason != "control_plane_source_changed"
    assert (
        PortalImplementationSupervisor._extra_gate_portal_state_must_preserve_worker(
            {"active_task_id": ""}
        )
        is False
    )
    assert supervisor._live_in_progress_worker_must_preserve(state) is True
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def _write_leftover_managed_daemon_identity(
    supervisor: PortalImplementationSupervisor,
    *,
    pid: int,
) -> None:
    identity = SupervisedChildIdentity(
        process_birth=SupervisedProcessBirthIdentity(
            pid=pid,
            start_time_ticks=39360547,
            boot_id="boot-test",
            parent_pid=17,
        ),
        command=(
            "/usr/bin/python3.12",
            "-P",
            "-m",
            "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
        ),
        owner_scope=supervisor._managed_daemon_owner_scope(),
        created_at="2026-09-09T20:00:00+00:00",
    )
    path = supervisor._managed_daemon_identity_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extra_gate_cannot_bypass_safe_to_restart() -> None:
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_recorded_managed_daemon_pid_falls_back_to_identity(tmp_path) -> None:
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    _write_leftover_managed_daemon_identity(supervisor, pid=2892700)
    assert supervisor._read_managed_daemon_pid() is None
    assert supervisor._recorded_managed_daemon_pid() == 2892700
    _extra_gate_cannot_bypass_safe_to_restart()


def test_live_in_progress_preserves_identity_grok_when_lane_state_missing(
    tmp_path,
    monkeypatch,
) -> None:
    """Restart run_once must see leftover extra-gate grok via identity.

    Lane task-state.json was missing and the pid file was already unlinked
    after the supervisor exited, so preserve missed grok 2932295 under
    leftover daemon 2892700. Extra-gate aliases still cannot bypass
    ``safe_to_restart=False``.
    """

    config = _config(tmp_path)
    supervisor = PortalImplementationSupervisor(config)
    _write_leftover_managed_daemon_identity(supervisor, pid=2892700)
    assert not config.state_path.exists()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.active_codex_exec_workers",
        lambda pid, status=None: (
            [
                {
                    "pid": 2932295,
                    "cmdline": (
                        "python3.12 -m "
                        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"
                    ),
                }
            ]
            if int(pid) == 2892700
            else []
        ),
    )
    assert supervisor._live_in_progress_worker_must_preserve() is True
    _extra_gate_cannot_bypass_safe_to_restart()


def test_public_run_once_does_not_terminate_identity_extra_gate_grok(
    tmp_path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    supervisor = PortalImplementationSupervisor(config)
    _write_leftover_managed_daemon_identity(supervisor, pid=2892700)
    events: list[str] = []
    finished: list[tuple[str, str]] = []

    @contextmanager
    def portal_fence():
        events.append("portal_fence_enter")
        try:
            yield supervisor.config.database_program
        finally:
            events.append("portal_fence_exit")

    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_mutation_fence",
        portal_fence,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_reload_projection_fenced",
        lambda _program: (
            events.append("projection")
            or _authenticated_watchdog_projection(active=False)
        ),
    )
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
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.active_codex_exec_workers",
        lambda pid, status=None: (
            [
                {
                    "pid": 2932295,
                    "cmdline": (
                        "python3.12 -m "
                        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"
                    ),
                }
            ]
            if int(pid) == 2892700
            else []
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        lambda **_kwargs: pytest.fail(
            "leftover extra-gate grok must not be terminated on supervisor restart"
        ),
    )

    def maintenance(_update, **kwargs):
        assert kwargs["database_portal_quiesced_idle"] is False
        events.append("maintenance")
        return {"stuck": False}

    monkeypatch.setattr(supervisor, "_run_once_with_maintenance", maintenance)

    result = supervisor.run_once(include_refill=False)

    assert events == [
        "portal_fence_enter",
        "projection",
        "portal_fence_exit",
        "maintenance",
    ]
    assert result == {"stuck": False}
    _extra_gate_cannot_bypass_safe_to_restart()


def test_terminate_managed_daemon_tree_refuses_identity_extra_gate_grok(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    _write_leftover_managed_daemon_identity(supervisor, pid=2892700)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.active_codex_exec_workers",
        lambda pid, status=None: (
            [
                {
                    "pid": 2932295,
                    "cmdline": (
                        "python3.12 -m "
                        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"
                    ),
                }
            ]
            if int(pid) == 2892700
            else []
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_fence_recorded_managed_daemon",
        lambda **_kwargs: pytest.fail(
            "identity-named extra-gate grok must not be fenced"
        ),
    )

    result = supervisor._terminate_managed_daemon_tree()

    assert result["terminated"] is False
    assert result["quiesced"] is False
    assert result["reason"] == "extra_gate_in_progress_preserve_worker"
    assert result["daemon_fence"]["reason"] == (
        "extra_gate_in_progress_preserve_worker"
    )
    assert supervisor._managed_daemon_identity_path().exists()
    _extra_gate_cannot_bypass_safe_to_restart()


def _write_nested_extra_gate_portal(
    supervisor: PortalImplementationSupervisor,
    *,
    task_id: str = "PCTDD-007",
    runner_pid: int | None = 1816991,
) -> Path:
    root = (
        supervisor.config.state_dir
        / f"{supervisor.config.state_prefix}_database_portal_attempts"
        / "0892ec23fdc05be191ed6d0d"
    )
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "active_task_id": task_id,
        "implementation_in_progress": True,
        "heartbeat_at": "2026-09-10T00:00:37.844083+00:00",
    }
    if runner_pid is not None:
        payload["active_provider_runner"] = {
            "pid": runner_pid,
            "task_id": task_id,
            "schema": "ipfs_accelerate_py.agent_supervisor.ordinary-provider-runner-birth@1",
        }
    path = root / "portal-task-state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_nested_extra_gate_dead_runner_does_not_preserve_worker(
    tmp_path,
    monkeypatch,
) -> None:
    """Stale nested PCTDD-007 must not fence relaunch after grok died.

    Lane-3 terminate returned extra_gate_in_progress_preserve_worker with
    pid=null while runner 1816991 was already gone, so the daemon never
    relaunched to rearm. Extra-gate aliases still cannot bypass
    ``safe_to_restart=False``.
    """

    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    _write_nested_extra_gate_portal(supervisor, runner_pid=1816991)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.process_is_running",
        lambda pid: False,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.active_codex_exec_workers",
        lambda *_args, **_kwargs: [],
    )

    assert supervisor._nested_extra_gate_portal_must_preserve_worker() is False
    assert (
        supervisor._nested_extra_gate_portal_must_preserve_worker(
            require_live_runner=True
        )
        is False
    )
    assert supervisor._live_in_progress_worker_must_preserve() is False
    result = supervisor._terminate_managed_daemon_tree()
    assert result.get("reason") != "extra_gate_in_progress_preserve_worker"
    assert result["daemon_fence"]["reason"] != (
        "extra_gate_in_progress_preserve_worker"
    )
    assert result["daemon_fence"]["safe_to_restart"] is True
    _extra_gate_cannot_bypass_safe_to_restart()


def test_nested_extra_gate_live_runner_still_preserves_worker(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    _write_nested_extra_gate_portal(supervisor, runner_pid=3337834)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.process_is_running",
        lambda pid: int(pid) == 3337834,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.active_codex_exec_workers",
        lambda *_args, **_kwargs: [],
    )

    assert supervisor._nested_extra_gate_portal_must_preserve_worker() is True
    assert (
        supervisor._nested_extra_gate_portal_must_preserve_worker(
            require_live_runner=True
        )
        is True
    )
    result = supervisor._terminate_managed_daemon_tree()
    assert result["reason"] == "extra_gate_in_progress_preserve_worker"
    assert result["terminated"] is False
    _extra_gate_cannot_bypass_safe_to_restart()


def test_nested_extra_gate_mid_worktree_without_runner_still_preserves(
    tmp_path,
) -> None:
    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    _write_nested_extra_gate_portal(supervisor, runner_pid=None)

    assert supervisor._nested_extra_gate_portal_must_preserve_worker() is True
    assert (
        supervisor._nested_extra_gate_portal_must_preserve_worker(
            require_live_runner=True
        )
        is False
    )
    _extra_gate_cannot_bypass_safe_to_restart()


def test_nested_extra_gate_live_runner_survives_newer_idle_attempts(
    tmp_path,
    monkeypatch,
) -> None:
    """Live grok outside the newest-8 window must still preserve.

    Lane-0 accrued 700+ portal attempts; idle 007 claims were newer than
    the PCTDD-005 grok attempt, so the cap-8 scan missed it and recycle
    killed grok. Extra-gate aliases still cannot bypass
    ``safe_to_restart=False``.
    """

    supervisor = PortalImplementationSupervisor(_config(tmp_path))
    live_path = _write_nested_extra_gate_portal(
        supervisor, task_id="PCTDD-005", runner_pid=1592311
    )
    older = live_path.stat().st_mtime - 30
    os.utime(live_path.parent, (older, older))
    os.utime(live_path, (older, older))
    root = live_path.parent.parent
    for index in range(9):
        newer = root / f"{index:024x}"
        newer.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_task_id": "PCTDD-007",
            "implementation_in_progress": False,
            "heartbeat_at": "2026-09-10T02:39:23.989519+00:00",
        }
        (newer / "portal-task-state.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.process_is_running",
        lambda pid: int(pid) == 1592311,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.active_codex_exec_workers",
        lambda *_args, **_kwargs: [],
    )

    assert supervisor._nested_extra_gate_portal_must_preserve_worker() is True
    assert (
        supervisor._nested_extra_gate_portal_must_preserve_worker(
            require_live_runner=True
        )
        is True
    )
    result = supervisor._terminate_managed_daemon_tree()
    assert result["reason"] == "extra_gate_in_progress_preserve_worker"
    assert result["terminated"] is False
    _extra_gate_cannot_bypass_safe_to_restart()


def test_failed_landed_recovery_is_not_a_read_only_diagnostic() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        _read_only_terminal_candidate_quarantine,
    )

    item = {
        "task_cid": "baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza",
        "task_alias": "PCTDD-005",
        "operation": "database_terminal_landed_completion",
        "recovered": False,
        "rearmed": False,
        "blocked": True,
        "reason": "terminal_landed_candidate_recovery_blocked",
        "error_type": "ContractValidationError",
        "error": "canonical proof contracts cannot contain floats",
    }
    assert _read_only_terminal_candidate_quarantine(item) is False
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_shared_no_provider_fence_skips_claimable_extra_gate_rearm() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
        DatabaseImplementationDaemon,
        TASK_SOURCE_QUERY_LIMIT,
    )

    task = SimpleNamespace(
        task_cid="baguqeerali4k6zayrolznqdh23y4xcpnznnowygnnx6vvhsdixztv7peiada",
        task_alias="PCTDD-034",
        status="retrying",
        revision=59,
        body={
            "completion_receipt": {
                "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
                "forced_block": False,
                "retry_exhausted": False,
                "no_provider_rearm_fence": {"saga_id": "sha256:" + "a" * 64},
            }
        },
    )
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.open = lambda: None  # type: ignore[method-assign]
    daemon._task_source = SimpleNamespace(
        list_tasks=lambda limit=TASK_SOURCE_QUERY_LIMIT: SimpleNamespace(
            tasks=[task]
        )
    )
    assert daemon._reconcile_shared_no_provider_rearm_fences() == []


def test_historical_compact_pair_does_not_mismatch_a_new_rearm_claim() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    historical_admission = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-fenced-provider-historical-retained-admission@1",
        "task_cid": "baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza",
        "task_alias": "PCTDD-005",
    }
    historical_consumption = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-fenced-provider-historical-retained-consumption@1",
        "task_cid": "baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza",
        "task_alias": "PCTDD-005",
        "attempt_id": "attempt:80318cb2b8384964a8b784520d9bba70",
        "claim_id": "claim:84143a5851a8491b849327790b4c3e01",
        "lease_id": "lease:db58ff48ac3b497eb60d9f261983622e",
        "owner_session_id": "embedded-store:5a477a1db9402e639fecebb83f5f0873",
        "attempt_number": 7,
        "fencing_token": 7,
        "fence_epoch": 7,
    }
    attempt = SimpleNamespace(
        task_cid="baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza",
        task_alias="PCTDD-005",
        attempt_id="attempt:67925e4ebcdc4b2b9f2d10e07b170e3e",
        claim_id="claim:824c6b1ac06249fb93045f67171c1ddb",
        lease_id="lease:79ab25993a694f45ac27d8fdd2793912",
        owner_session_id="embedded-store:5a477a1db9402e639fecebb83f5f0873",
        attempt_number=8,
        fencing_token=8,
        fence_epoch=8,
        body={
            "control_claim": {"revision": 30},
            "retry_budget": {
                "retained_recovery_admission": historical_admission,
                "retained_recovery_consumption": historical_consumption,
            },
        },
    )
    task = SimpleNamespace(
        task_cid=attempt.task_cid,
        task_alias=attempt.task_alias,
        status="in_progress",
        revision=31,
        body={
            "completion_receipt": {
                "operation": "database_claim",
                "attempt_id": attempt.attempt_id,
                "claim_id": attempt.claim_id,
                "retained_recovery_admission": historical_admission,
                "retained_recovery_consumption": historical_consumption,
            }
        },
    )
    assert (
        DatabaseImplementationDaemon._retained_recovery_pair_binds_attempt(
            attempt,
            admission=historical_admission,
            consumption=historical_consumption,
        )
        is False
    )
    assert (
        DatabaseImplementationDaemon._retained_recovery_pair_is_exact_for_attempt(
            task,
            attempt,
        )
        is True
    )
    reserved = SimpleNamespace(
        task_cid="baguqeerah7muo423u3xf5gi32hazctify2i55cavbdugzzythfqdl4wyif6a",
        task_alias="PCTDD-006",
        attempt_id="attempt:new",
        claim_id="claim:new",
        lease_id="lease:new",
        owner_session_id="embedded-store:5a477a1db9402e639fecebb83f5f0873",
        attempt_number=8,
        fencing_token=8,
        fence_epoch=8,
        body={
            "control_claim": {"revision": 31},
            "retry_budget": {
                "retained_recovery_admission": historical_admission,
                "retained_recovery_consumption": historical_consumption,
            },
        },
    )
    reserved_task = SimpleNamespace(
        body={
            "completion_receipt": {
                "retained_recovery_admission": historical_admission,
                "retained_recovery_consumption": historical_consumption,
            }
        }
    )
    assert (
        DatabaseImplementationDaemon._retained_recovery_pair_is_exact_for_attempt(
            reserved_task,
            reserved,
        )
        is False
    )


def test_unknown_outcome_rearm_receipt_is_not_claim_forbidden(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
        DatabaseImplementationDaemon,
    )

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._automatic_claim_forbidden = lambda _task: False
    task = SimpleNamespace(
        status="retrying",
        task_alias="PCTDD-006",
        task_cid="baguqeerah7muo423u3xf5gi32hazctify2i55cavbdugzzythfqdl4wyif6a",
        revision=33,
        body={
            "completion_receipt": {
                "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
                "forced_block": False,
                "retry_exhausted": False,
            }
        },
    )
    assert daemon._automatic_claim_forbidden_current(task) is False
    task.body["completion_receipt"]["operation"] = "database_retry_rearmed"
    task.body["completion_receipt"]["forced_block"] = None
    assert daemon._automatic_claim_forbidden_current(task) is False
    task.body["completion_receipt"]["forced_block"] = True
    assert daemon._automatic_claim_forbidden_current(task) is True


def test_extra_gate_rearm_with_leftover_no_provider_fence_is_claimable() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
        DatabaseImplementationDaemon,
    )

    daemon = object.__new__(DatabaseImplementationDaemon)
    task = SimpleNamespace(
        status="retrying",
        task_alias="PCTDD-034",
        task_cid="baguqeerali4k6zayrolznqdh23y4xcpnznnowygnnx6vvhsdixztv7peiada",
        revision=59,
        body={
            "completion_receipt": {
                "schema": "ipfs_accelerate_py/agent-supervisor/database-retry-budget@1",
                "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
                "forced_block": False,
                "retry_exhausted": False,
                "unknown_outcome_rearm_count": 1,
                "no_provider_rearm_fence": {
                    "state": "admitted",
                    "saga_id": "sha256:" + "a" * 64,
                },
                "no_provider_rearm_original_block_receipt": {
                    "unknown_outcome_rearm_count": 0
                },
            }
        },
    )
    assert daemon._automatic_claim_forbidden(task) is True
    assert daemon._automatic_claim_forbidden_current(task) is False
    task.body["review_only"] = True
    assert daemon._automatic_claim_forbidden_current(task) is True


def test_blocked_extra_gate_unresolved_saga_remains_fenced() -> None:
    """A reserved task name never makes an unresolved recovery saga audit-only."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
        DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
        DATABASE_RETRY_BUDGET_SCHEMA,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.open = lambda: None  # type: ignore[method-assign]
    daemon._unresolved_database_no_provider_rearm_sagas = lambda: (  # type: ignore[method-assign]
        (
            {
                "task_cid": "baguqeerazst6lunrikvyslwfqzfbqbpwiivb5hxjsdzwvd7jjsqnnfpadwuq",
                "saga_id": "",
                "state": "recovery_required",
            },
        )
    )
    pending = daemon._reconcile_database_no_provider_rearm_sagas()
    assert len(pending) == 1
    assert pending[0]["recovery_required"] is True
    assert pending[0]["rearmed"] is False
    blocked = SimpleNamespace(
        task_cid="baguqeerazst6lunrikvyslwfqzfbqbpwiivb5hxjsdzwvd7jjsqnnfpadwuq",
        task_alias="PCTDD-007",
        status="blocked",
        body={
            "completion_receipt": {
                "no_provider_rearm_fence": {"saga_id": "sha256:" + "b" * 64},
                "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
                "schema": DATABASE_RETRY_BUDGET_SCHEMA,
            }
        },
    )
    daemon._task_source = SimpleNamespace(  # type: ignore[attr-defined]
        list_tasks=lambda limit=32: SimpleNamespace(tasks=[blocked])
    )
    pending = daemon._reconcile_shared_no_provider_rearm_fences()
    assert len(pending) == 1
    assert pending[0]["recovery_required"] is True
    assert pending[0]["rearmed"] is False
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


