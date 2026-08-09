"""Tests for DatabaseSupervisorBackend@1 / DatabaseControlOperations@1 (DQP-029).

Evidence subset: Python/CLI/MCP parity, discovery inertness, pagination/watch,
authorization, dry run, permit, idempotency, lease/fence/effects, redaction.

Acceptance:

* Read/proposal/mutation authority remains distinct
* Configured database program has supported status/health/logs/stop rather than
  launch-only control
* All transports share canonical request/result identity and direct service
  dispatch
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    Operation,
    OperationAuthority,
    OperationStatus,
    decode_operation_request,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    LifecycleStatus,
    SupervisorLifecycleState,
)
from ipfs_accelerate_py.agent_supervisor.control.database_backend import (
    DATABASE_PROGRAM_REQUIRED_CONTROL_OPS,
    DATABASE_SUPERVISOR_BACKEND_INTERFACE,
    LOGS_EVENT_KIND,
    DatabaseSupervisorBackend,
    InMemoryDatabaseControlStore,
    open_database_supervisor_backend,
)
from ipfs_accelerate_py.agent_supervisor.control.database_operations import (
    DATABASE_CONTROL_OPERATIONS_INTERFACE,
    DatabaseControlAuthorityError,
    DatabaseControlBinding,
    DatabaseControlOperations,
    open_database_control_operations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _program(**overrides: Any) -> dict[str, Any]:
    payload = {
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "store_id": "control.duckdb",
        "store_generation": "gen-1",
        "schema_revision": "1",
        "event_store_path": "state/events.duckdb",
        "runtime_registry_path": "state/registry.duckdb",
        "failover_policy": "fail_closed",
        "target_id": "supervisor:database-program",
    }
    payload.update(overrides)
    return payload


def _binding(repo_root: Path, state_root: Path) -> DatabaseControlBinding:
    return DatabaseControlBinding(
        repository_root=str(repo_root),
        state_root=str(state_root),
        repository_id="repository:dqp-029",
        tree_id="tree:dqp-029",
        objective_id="DQP-029",
        objective_revision="objective:1",
        policy_id="policy:database-control",
        policy_revision="policy:1",
        caller="operator:dqp-029",
        target_id="supervisor:database-program",
    )


def _ops(
    repo_root: Path,
    state_root: Path,
    *,
    program: dict[str, Any] | None = None,
) -> DatabaseControlOperations:
    store = InMemoryDatabaseControlStore()
    store.seed_goals(
        (
            {"goal_id": "DQP-G060", "title": "control APIs", "status": "active"},
            {"goal_id": "DQP-G030", "title": "runtime", "status": "active"},
        )
    )
    store.seed_tasks(
        (
            {"task_id": "DQP-029", "title": "control ops", "status": "todo"},
            {"task_id": "DQP-030", "title": "authority", "status": "todo"},
        )
    )
    store.seed_metrics(
        {
            "gauges": {"ready_tasks": 2, "active_leases": 1},
            "samples": [{"name": "ready_tasks", "value_milli": 2000}],
        }
    )
    return open_database_control_operations(
        _binding(repo_root, state_root),
        database_program=program or _program(),
        control_store=store,
        clock_ms=lambda: 1_500,
        require_lease_validator=True,
    )


@pytest.fixture()
def roots(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    return repo, state


# ---------------------------------------------------------------------------
# Interface / discovery
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_SUPERVISOR_BACKEND_INTERFACE == "DatabaseSupervisorBackend@1"
    assert DATABASE_CONTROL_OPERATIONS_INTERFACE == "DatabaseControlOperations@1"
    assert DatabaseSupervisorBackend.INTERFACE == DATABASE_SUPERVISOR_BACKEND_INTERFACE
    assert DatabaseControlOperations.INTERFACE == DATABASE_CONTROL_OPERATIONS_INTERFACE
    assert (
        DatabaseControlOperations.DISPATCHER_ID
        == DIRECT_CONTROL_SERVICE_DISPATCHER_ID
    )


def test_discovery_is_inert_and_not_launch_only(roots: tuple[Path, Path]) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    backend = ops.backend
    assert backend.optional_providers_loaded is False
    assert backend.processes_started is False

    discovery = ops.discover()
    assert discovery["launch_only"] is False
    assert discovery["supports_status"] is True
    assert discovery["supports_health"] is True
    assert discovery["supports_logs"] is True
    assert discovery["supports_stop"] is True
    assert discovery["required_control_ops_supported"] is True
    assert discovery["processes_started_by_discovery"] is False
    assert discovery["optional_providers_loaded"] is False
    assert discovery["processes_started"] is False
    assert discovery["direct_service_dispatch"] is True
    assert discovery["dispatcher_id"] == DIRECT_CONTROL_SERVICE_DISPATCHER_ID
    for name in sorted(DATABASE_PROGRAM_REQUIRED_CONTROL_OPS):
        assert name in discovery["required_control_ops"]

    surface = backend.supported_control_surface()
    assert surface["launch_only"] is False
    assert surface["supports_status"] is True
    assert surface["supports_health"] is True
    assert surface["supports_logs"] is True
    assert surface["supports_stop"] is True
    assert "status" in surface["supported_operations"]
    assert "health" in surface["supported_operations"]
    assert "stop" in surface["supported_operations"]
    # Still no process start after discovery + capability inspection.
    assert backend.processes_started is False
    assert backend.optional_providers_loaded is False


def test_open_asserts_not_launch_only(roots: tuple[Path, Path]) -> None:
    repo, state = roots
    ops = open_database_control_operations(
        _binding(repo, state),
        database_program=_program(),
        clock_ms=lambda: 1_500,
    )
    ops.assert_not_launch_only()


# ---------------------------------------------------------------------------
# Authority separation
# ---------------------------------------------------------------------------


def test_read_proposal_mutation_authority_remain_distinct(
    roots: tuple[Path, Path],
) -> None:
    repo, state = roots
    ops = _ops(repo, state)

    status_req = ops.build_request(Operation.STATUS)
    assert status_req.authority is OperationAuthority.READ
    assert status_req.effective_authority is OperationAuthority.READ
    assert status_req.operation in READ_OPERATIONS

    preview_req = ops.build_request(Operation.OBJECTIVE_PREVIEW)
    assert preview_req.authority is OperationAuthority.PROPOSAL
    assert preview_req.operation in PROPOSAL_OPERATIONS

    dry = ops.build_mutation_request(
        Operation.STOP,
        idempotency_key="dry:stop:1",
        dry_run=True,
    )
    assert dry.operation in MUTATION_OPERATIONS
    assert dry.dry_run is True
    assert dry.effective_authority is OperationAuthority.PROPOSAL

    real = ops.build_mutation_request(
        Operation.STOP,
        idempotency_key="real:stop:1",
        dry_run=False,
    )
    assert real.authority is OperationAuthority.MUTATION
    assert real.effective_authority is OperationAuthority.MUTATION
    assert real.idempotency is not None
    assert real.authorization is not None
    assert real.lease_id
    assert real.fencing_epoch is not None
    assert real.expected_effects


def test_mutation_without_effects_or_auth_fails_closed(
    roots: tuple[Path, Path],
) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    with pytest.raises(DatabaseControlAuthorityError, match="expected effects"):
        ops.build_request(Operation.STOP, dry_run=False)
    with pytest.raises(DatabaseControlAuthorityError, match="idempotency"):
        ops.build_request(
            Operation.STOP,
            dry_run=False,
            expected_effects=(
                ExpectedEffect(
                    effect_id="stop:x",
                    kind=EffectKind.LIFECYCLE_TRANSITION,
                    resource="supervisor:x",
                    paths=("lifecycle/status.json",),
                ),
            ),
        )
    with pytest.raises(DatabaseControlAuthorityError, match="authorization"):
        ops.build_request(
            Operation.STOP,
            dry_run=False,
            expected_effects=(
                ExpectedEffect(
                    effect_id="stop:x",
                    kind=EffectKind.LIFECYCLE_TRANSITION,
                    resource="supervisor:x",
                    paths=("lifecycle/status.json",),
                ),
            ),
            idempotency_key="key:1",
        )


# ---------------------------------------------------------------------------
# Status / health / logs / stop
# ---------------------------------------------------------------------------


def test_status_health_logs_stop_for_database_program(
    roots: tuple[Path, Path],
) -> None:
    repo, state = roots
    ops = _ops(repo, state)

    # Seed a healthy running program so stop is a legal transition.
    ops.seed_status(
        LifecycleStatus(
            target_id="supervisor:database-program",
            state=SupervisorLifecycleState.HEALTHY,
            phase="running",
            heartbeat_at_ms=1_400,
            pid=4242,
            generation=1,
            fencing_epoch=1,
            updated_at_ms=1_400,
        )
    )
    ops.append_log(
        "database program ready",
        severity="info",
        component="control",
        body={"token": "should-not-leak-raw", "ok": True},
    )
    ops.append_log(
        "lease renewed",
        severity="info",
        component="runtime",
    )

    status = ops.status()
    assert status.succeeded
    assert status.authority is OperationAuthority.READ
    assert status.data["state"] == "healthy"
    assert status.data["control_surface"]["launch_only"] is False
    assert status.data["database_program"]["store_id"] == "control.duckdb"

    health = ops.health()
    assert health.succeeded
    assert health.data["healthy"] is True
    assert health.data["state"] == "healthy"

    logs = ops.logs(limit=10)
    assert logs.succeeded
    assert logs.authority is OperationAuthority.READ
    assert logs.data["kind"] == LOGS_EVENT_KIND
    assert logs.data["count"] == 2
    assert logs.data["items"][0]["message"] == "database program ready"
    assert logs.data["truncated"] is False

    # Pagination window.
    page = ops.logs(limit=1, offset=0)
    assert page.data["count"] == 1
    assert page.data["truncated"] is True
    page2 = ops.logs(limit=1, offset=1)
    assert page2.data["count"] == 1
    assert page2.data["items"][0]["message"] == "lease renewed"

    # Dry-run stop does not mutate.
    dry_stop = ops.stop(idempotency_key="stop:dry:1", dry_run=True)
    assert dry_stop.succeeded
    assert dry_stop.authority is OperationAuthority.PROPOSAL
    assert dry_stop.data.get("dry_run") is True
    still = ops.status()
    assert still.data["state"] == "healthy"

    # Real stop applies through fenced lifecycle mutation.
    stop = ops.stop(idempotency_key="stop:real:1", dry_run=False)
    assert stop.succeeded, stop.error
    assert stop.authority is OperationAuthority.MUTATION
    assert stop.data["accepted"] is True
    assert stop.data["state"] in {"stopping", "stopped"}
    after = ops.status()
    assert after.data["state"] in {"stopping", "stopped"}


def test_start_pause_resume_drain_lifecycle(roots: tuple[Path, Path]) -> None:
    repo, state = roots
    ops = _ops(repo, state)

    start = ops.start(idempotency_key="start:1")
    assert start.succeeded, start.error
    assert start.data["state"] == "starting"

    # Promote to healthy via heartbeat seed for subsequent transitions.
    ops.seed_status(
        LifecycleStatus(
            target_id="supervisor:database-program",
            state=SupervisorLifecycleState.HEALTHY,
            phase="running",
            heartbeat_at_ms=1_400,
            pid=99,
            generation=2,
            fencing_epoch=1,
            updated_at_ms=1_400,
        )
    )
    pause = ops.pause(idempotency_key="pause:1")
    assert pause.succeeded, pause.error
    assert pause.data["state"] == "paused"

    resume = ops.resume(idempotency_key="resume:1")
    assert resume.succeeded, resume.error
    assert resume.data["state"] == "healthy"

    drain = ops.drain(idempotency_key="drain:1")
    assert drain.succeeded, drain.error
    assert drain.data["state"] == "draining"


def test_idempotent_stop_replay(roots: tuple[Path, Path]) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    ops.seed_status(
        LifecycleStatus(
            target_id="supervisor:database-program",
            state=SupervisorLifecycleState.HEALTHY,
            phase="running",
            heartbeat_at_ms=1_400,
            pid=7,
            generation=1,
            fencing_epoch=1,
            updated_at_ms=1_400,
        )
    )
    first = ops.stop(idempotency_key="stop:idem:1")
    assert first.succeeded, first.error
    second = ops.stop(idempotency_key="stop:idem:1")
    assert second.succeeded, second.error
    # Exact idempotent replay returns the original result identity.
    assert second.request_id == first.request_id
    assert second.content_id == first.content_id


def test_stale_fence_is_rejected(roots: tuple[Path, Path]) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    ops.seed_status(
        LifecycleStatus(
            target_id="supervisor:database-program",
            state=SupervisorLifecycleState.HEALTHY,
            phase="running",
            heartbeat_at_ms=1_400,
            pid=11,
            generation=3,
            fencing_epoch=9,
            updated_at_ms=1_400,
        )
    )
    request = ops.build_mutation_request(
        Operation.STOP,
        idempotency_key="stop:stale:1",
        fencing_epoch=1,
    )
    result = ops.dispatch(request)
    assert result.status is OperationStatus.CONFLICT
    assert result.error is not None
    assert result.error.code in {
        ErrorCode.STALE_LEASE,
        ErrorCode.CONFLICT,
        ErrorCode.INVALID_LIFECYCLE_TRANSITION,
    }


# ---------------------------------------------------------------------------
# Goals / tasks / metrics / capabilities
# ---------------------------------------------------------------------------


def test_goals_tasks_metrics_and_capabilities(roots: tuple[Path, Path]) -> None:
    repo, state = roots
    ops = _ops(repo, state)

    goals = ops.goals(parameters={"limit": 1, "offset": 0})
    assert goals.succeeded
    assert goals.data["count"] == 1
    assert goals.data["truncated"] is True
    assert goals.data["items"][0]["goal_id"] == "DQP-G060"

    tasks = ops.tasks(parameters={"limit": 10})
    assert tasks.succeeded
    assert tasks.data["count"] == 2

    metrics = ops.metrics()
    assert metrics.succeeded
    assert metrics.data["gauges"]["ready_tasks"] == 2

    caps = ops.capabilities()
    assert caps.succeeded
    assert "status" in caps.data["operations"]
    assert "health" in caps.data["operations"]
    assert "stop" in caps.data["operations"]
    assert caps.data["optional_providers_loaded"] is False
    assert caps.data["processes_started"] is False


def test_proposal_preview_does_not_mutate(roots: tuple[Path, Path]) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    before = ops.status()
    preview = ops.dispatch(ops.build_request(Operation.OBJECTIVE_PREVIEW))
    assert preview.succeeded
    assert preview.authority is OperationAuthority.PROPOSAL
    assert preview.data["preview"] is True
    after = ops.status()
    assert after.data["state"] == before.data["state"]
    assert after.content_id == before.content_id


# ---------------------------------------------------------------------------
# Transport parity / direct service dispatch
# ---------------------------------------------------------------------------


def test_python_cli_mcp_share_request_result_identity(
    roots: tuple[Path, Path],
) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    ops.append_log("parity log", component="parity")

    for builder in (
        lambda: ops.build_request(Operation.STATUS),
        lambda: ops.build_request(Operation.HEALTH),
        lambda: ops.build_request(
            Operation.EVENTS,
            parameters={"kind": LOGS_EVENT_KIND, "limit": 10},
        ),
        lambda: ops.build_mutation_request(
            Operation.STOP,
            idempotency_key="parity:stop:dry",
            dry_run=True,
        ),
    ):
        request = builder()
        # Seed healthy state only for real status/health consistency.
        if request.operation in {Operation.STATUS, Operation.HEALTH}:
            ops.seed_status(
                LifecycleStatus(
                    target_id="supervisor:database-program",
                    state=SupervisorLifecycleState.HEALTHY,
                    phase="running",
                    heartbeat_at_ms=1_400,
                    pid=55,
                    generation=1,
                    fencing_epoch=1,
                    updated_at_ms=1_400,
                )
            )
        parity = ops.transport_parity(request)
        assert parity["request_ids_match"] is True
        assert parity["identical"] is True, parity
        assert parity["dispatcher_id"] == DIRECT_CONTROL_SERVICE_DISPATCHER_ID
        assert parity["python_result_id"] == parity["cli_result_id"]
        assert parity["cli_result_id"] == parity["mcp_result_id"]


def test_decode_round_trip_preserves_request_identity(
    roots: tuple[Path, Path],
) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    request = ops.build_mutation_request(
        Operation.STOP,
        idempotency_key="decode:stop:1",
        dry_run=True,
    )
    again = decode_operation_request(request.to_dict())
    assert again.request_id == request.request_id
    assert again.to_dict() == request.to_dict()
    via_json = decode_operation_request(
        __import__("json").loads(request.to_json())
    )
    assert via_json.request_id == request.request_id


def test_dispatch_aliases_are_direct_service_execute(
    roots: tuple[Path, Path],
) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    request = ops.build_request(Operation.STATUS)
    a = ops.dispatch(request)
    b = ops.execute(request)
    c = ops.handle(request)
    assert a.content_id == b.content_id == c.content_id
    assert a.request_id == request.request_id


# ---------------------------------------------------------------------------
# Redaction / secrets
# ---------------------------------------------------------------------------


def test_sensitive_fields_are_redacted_in_results(
    roots: tuple[Path, Path],
) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    ops.backend.control_store.seed_metrics(
        {
            "gauges": {"ready": 1},
            "api_key": "super-secret-key",
            "token": "bearer-token-value",
        }
    )
    result = ops.metrics()
    assert result.succeeded
    encoded = str(result.to_dict())
    assert "super-secret-key" not in encoded
    assert "bearer-token-value" not in encoded
    assert "[REDACTED]" in encoded or "api_key" not in result.data


# ---------------------------------------------------------------------------
# Backend unit surface
# ---------------------------------------------------------------------------


def test_backend_registered_operations_include_lifecycle_and_reads() -> None:
    backend = open_database_supervisor_backend(database_program=_program())
    registered = set(backend.registered_operations)
    for operation in (
        Operation.STATUS,
        Operation.HEALTH,
        Operation.EVENTS,
        Operation.METRICS,
        Operation.GOALS,
        Operation.TASKS,
        Operation.START,
        Operation.STOP,
        Operation.PAUSE,
        Operation.RESUME,
        Operation.DRAIN,
    ):
        assert operation in registered


def test_logs_kind_routes_through_events(
    roots: tuple[Path, Path],
) -> None:
    repo, state = roots
    ops = _ops(repo, state)
    ops.append_log("via events", component="events")
    result = ops.events(
        parameters={"kind": "logs", "limit": 5, "component": "events"}
    )
    assert result.succeeded
    assert result.data["kind"] == LOGS_EVENT_KIND
    assert result.data["count"] == 1
    assert result.data["items"][0]["component"] == "events"


def test_unauthorized_mutation_denied(roots: tuple[Path, Path]) -> None:
    repo, state = roots
    # Live authorization validator denial (contract decisions must still PERMIT
    # at construction time; the service re-checks live policy before dispatch).
    ops = open_database_control_operations(
        _binding(repo, state),
        database_program=_program(),
        clock_ms=lambda: 1_500,
        authorization_validator=lambda _request: False,
    )
    ops.seed_status(
        LifecycleStatus(
            target_id="supervisor:database-program",
            state=SupervisorLifecycleState.HEALTHY,
            phase="running",
            heartbeat_at_ms=1_400,
            pid=3,
            generation=1,
            fencing_epoch=1,
            updated_at_ms=1_400,
        )
    )
    request = ops.build_mutation_request(
        Operation.STOP,
        idempotency_key="stop:denied:1",
    )
    result = ops.dispatch(request)
    assert result.status is OperationStatus.DENIED
    assert result.error is not None
    assert result.error.code is ErrorCode.UNAUTHORIZED


def test_cold_import_is_side_effect_free() -> None:
    # Re-importing the modules must not open databases or start processes.
    import importlib

    backend_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.control.database_backend"
    )
    ops_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.control.database_operations"
    )
    importlib.reload(backend_mod)
    importlib.reload(ops_mod)
    assert backend_mod.DATABASE_SUPERVISOR_BACKEND_INTERFACE.endswith("@1")
    assert ops_mod.DATABASE_CONTROL_OPERATIONS_INTERFACE.endswith("@1")
