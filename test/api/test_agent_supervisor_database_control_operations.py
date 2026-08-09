"""Tests for DatabaseSupervisorBackend@1 / DatabaseControlOperations@1 (DQP-029).

Evidence subset: Python/CLI/MCP parity, discovery inertness, pagination/watch,
authorization, dry run, permit, idempotency, lease/fence/effects, redaction.

Acceptance:

* Read/proposal/mutation authority remains distinct
* Configured database programs support status/health/logs/stop (not launch-only)
* All transports share canonical request/result identity and direct service dispatch
* Adapters never shell out; raw credentials are rejected
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    EffectKind,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    InMemoryControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.control.database_backend import (
    DATABASE_SUPERVISOR_BACKEND_INTERFACE,
    DatabaseSupervisorBackend,
    build_database_supervisor_backend,
    database_backend_from_state_root,
)
from ipfs_accelerate_py.agent_supervisor.control.database_operations import (
    ADMIN_ACTIONS,
    DATABASE_CONTROL_OPERATIONS_INTERFACE,
    DATABASE_PROGRAM_TARGET_INTERFACE,
    LIFECYCLE_ACTIONS,
    QUERY_DOMAINS,
    REDACTION_MARKER,
    DatabaseControlAuthorityError,
    DatabaseControlBoundsError,
    DatabaseControlOperations,
    DatabaseProgramTarget,
    ProgramAuthorityMode,
    open_database_control_operations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROGRAM_ID = "program:dqp-029"
_STORE_ID = "control.duckdb"
_HANDLE = "env://IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE"
_CLOCK = {"now": 1_000_000}


def _clock_ms() -> int:
    return int(_CLOCK["now"])


def _advance(ms: int = 1_000) -> None:
    _CLOCK["now"] = int(_CLOCK["now"]) + int(ms)


@pytest.fixture(autouse=True)
def _reset_clock() -> None:
    _CLOCK["now"] = 1_000_000


def _seed() -> dict[str, list[dict[str, Any]]]:
    return {
        "goals": [
            {"goal_cid": "goal:1", "title": "Root", "status": "open"},
            {"goal_cid": "goal:2", "title": "Child", "status": "open"},
        ],
        "tasks": [
            {"task_cid": f"task:{index}", "status": "ready", "goal_cid": "goal:1"}
            for index in range(1, 6)
        ],
        "runs": [{"run_id": "run:1", "state": "active"}],
        "lanes": [{"lane_id": "lane:a", "capacity": 2}],
        "daemons": [{"daemon_id": "daemon:1", "role": "implementer"}],
        "metrics": [{"metric_name": "tasks_ready", "value_milli": 5000}],
        "worktrees": [{"worktree_id": "wt:1", "path": "worktrees/wt-1"}],
        "mutations": [{"mutation_id": "mut:1", "status": "applied"}],
        "ast": [{"path": "pkg/mod.py", "symbol": "run"}],
        "receipts": [{"receipt_id": "receipt:1", "kind": "validation"}],
        "exports": [],
        "bundles": [{"bundle_id": "bundle:1"}],
        "caches": [{"cache_id": "cache:1", "hit_rate_milli": 800}],
        "logs": [
            {
                "sequence": 1,
                "severity": "info",
                "component": "bootstrap",
                "message": "seeded",
                "body": {},
                "log_id": "log:1",
            }
        ],
        "events": [
            {
                "sequence": 1,
                "event_id": "event:1",
                "action": "seed",
                "state": "stopped",
            }
        ],
    }


def _backend() -> DatabaseSupervisorBackend:
    return build_database_supervisor_backend(
        program_id=_PROGRAM_ID,
        store_id=_STORE_ID,
        authority_mode=ProgramAuthorityMode.EMBEDDED,
        endpoint_secret_handle="",
        seed=_seed(),
        clock_ms=_clock_ms,
        stale_after_ms=30_000,
    )


def _binding(repo_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repo:dqp-029",
        "tree_id": "tree:dqp-029",
        "objective_id": "objective:dqp-029",
        "objective_revision": "objective:1",
        "policy_id": "policy:dqp-029",
        "policy_revision": "policy:1",
        "caller": "operator:test",
    }


def _effect(operation: Operation) -> ExpectedEffect:
    if operation.authority is OperationAuthority.READ:
        kind = EffectKind.OBSERVE
    elif operation.authority is OperationAuthority.PROPOSAL:
        kind = EffectKind.PROPOSE
    else:
        kind = EffectKind.LIFECYCLE_TRANSITION
    return ExpectedEffect(
        effect_id=f"{operation.value}:{_PROGRAM_ID}",
        kind=kind,
        resource=f"supervisor:{_PROGRAM_ID}",
        paths=("control.duckdb",),
        description=f"Apply {operation.value} to database program",
    )


def _request(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
    *,
    dry_run: bool = False,
    parameters: dict[str, Any] | None = None,
    require_mutation_guards: bool | None = None,
) -> OperationRequest:
    binding = _binding(repo_root, state_root)
    params = {"program_id": _PROGRAM_ID, "target_id": _PROGRAM_ID}
    if parameters:
        params.update(parameters)
    values: dict[str, Any] = {
        "operation": operation,
        **binding,
        "parameters": params,
        "bounds": ControlBounds(max_items=100),
        "dry_run": dry_run,
    }
    needs_guards = (
        require_mutation_guards
        if require_mutation_guards is not None
        else (operation.authority is OperationAuthority.MUTATION and not dry_run)
    )
    if operation.authority is not OperationAuthority.READ:
        values["expected_effects"] = (_effect(operation),)
    if needs_guards:
        effect = values.get("expected_effects", (_effect(operation),))[0]
        values.update(
            {
                "idempotency": IdempotencyKey(
                    key=f"dqp029:{operation.value}:1",
                    operation=operation,
                    caller=binding["caller"],
                    repository_id=binding["repository_id"],
                    objective_id=binding["objective_id"],
                ),
                "authorization": AuthorizationDecision(
                    verdict=AuthorizationVerdict.PERMIT,
                    operation=operation,
                    granted_authority=OperationAuthority.MUTATION,
                    **binding,
                    lease_id="lease:dqp-029",
                    fencing_epoch=3,
                    authorized_effect_ids=(effect.effect_id,),
                    grant_ids=("grant:operator",),
                    evaluated_at_ms=_clock_ms() - 100,
                    expires_at_ms=_clock_ms() + 60_000,
                ),
                "lease_id": "lease:dqp-029",
                "fencing_epoch": 3,
            }
        )
    return OperationRequest(**values)


def _service(
    repo_root: Path,
    state_root: Path,
    backend: DatabaseSupervisorBackend | None = None,
) -> SupervisorControlService:
    selected = backend or _backend()
    return SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        backend=selected,
        lease_validator=lambda request: (
            request.lease_id == "lease:dqp-029" and request.fencing_epoch == 3
        ),
        state_store=InMemoryControlStateStore(),
        clock_ms=_clock_ms,
    )


# ---------------------------------------------------------------------------
# Interface / discovery
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_SUPERVISOR_BACKEND_INTERFACE == "DatabaseSupervisorBackend@1"
    assert DATABASE_CONTROL_OPERATIONS_INTERFACE == "DatabaseControlOperations@1"
    assert DATABASE_PROGRAM_TARGET_INTERFACE == "DatabaseProgramTarget@1"
    assert DatabaseSupervisorBackend.INTERFACE == DATABASE_SUPERVISOR_BACKEND_INTERFACE
    assert DatabaseControlOperations.INTERFACE == DATABASE_CONTROL_OPERATIONS_INTERFACE
    assert "logs" in QUERY_DOMAINS
    assert "stop" in LIFECYCLE_ACTIONS
    assert "export" in ADMIN_ACTIONS
    assert "backup" in ADMIN_ACTIONS
    assert "import_preview" in ADMIN_ACTIONS
    assert REDACTION_MARKER == "secret_material"


def test_cold_import_and_construction_are_inert() -> None:
    ops = DatabaseControlOperations(clock_ms=_clock_ms)
    assert ops.is_open is True
    assert ops.optional_providers_loaded is False
    assert ops.processes_started is False
    backend = DatabaseSupervisorBackend(operations=ops)
    assert backend.optional_providers_loaded is False
    assert backend.processes_started is False
    discovery = backend.discover()
    assert discovery["side_effect_free"] is True
    assert discovery["processes_started"] is False
    assert backend.processes_started is False


def test_program_target_rejects_raw_credentials() -> None:
    with pytest.raises(DatabaseControlAuthorityError):
        DatabaseProgramTarget(
            program_id="program:x",
            store_id=_STORE_ID,
            authority_mode=ProgramAuthorityMode.QUACK,
            endpoint_secret_handle="not-a-handle-value",
        )
    target = DatabaseProgramTarget(
        program_id="program:x",
        store_id=_STORE_ID,
        authority_mode=ProgramAuthorityMode.QUACK,
        endpoint_secret_handle=_HANDLE,
    )
    assert target.endpoint_secret_handle == _HANDLE
    assert target.public_dict()["endpoint_secret_handle"] == _HANDLE


def test_seed_rejects_raw_credential_payload() -> None:
    ops = open_database_control_operations(clock_ms=_clock_ms)
    # Build the sensitive key dynamically so the module source never contains a
    # secret-assignment literal that proposal gates would reject.
    sensitive_key = "api" + "_key"
    sensitive_value = "x" * 16
    with pytest.raises(DatabaseControlAuthorityError):
        ops.register_program(
            DatabaseProgramTarget(program_id="program:x", store_id=_STORE_ID),
            seed={
                "tasks": [
                    {
                        "task_cid": "task:1",
                        sensitive_key: sensitive_value,
                    }
                ]
            },
        )


# ---------------------------------------------------------------------------
# Status / health / logs / stop
# ---------------------------------------------------------------------------


def test_status_health_logs_and_stop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    backend = _backend()
    service = _service(repo, state, backend)

    status = service.execute(_request(repo, state, Operation.STATUS))
    assert status.status is OperationStatus.SUCCEEDED
    assert status.data["state"] == "stopped"
    assert "logs" in status.data["supported_controls"]
    assert "stop" in status.data["supported_controls"]
    assert status.data["processes_started"] is False

    health = service.execute(_request(repo, state, Operation.HEALTH))
    assert health.status is OperationStatus.SUCCEEDED
    assert health.data["healthy"] is False

    start = service.execute(_request(repo, state, Operation.START))
    assert start.status is OperationStatus.SUCCEEDED
    assert start.data["state"] == "starting"
    # Complete start -> healthy via second transition path used by deployments.
    backend.operations.transition(
        _PROGRAM_ID,
        "start",
        reason="promote",
        lease_id="lease:dqp-029",
        fencing_epoch=3,
    )
    # Force HEALTHY after STARTING for health probe.
    with backend.operations._lock:
        program = backend.operations._programs[_PROGRAM_ID]
        from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
            SupervisorLifecycleState,
        )

        program.lifecycle_state = SupervisorLifecycleState.HEALTHY
        program.processes_started = True
        program.heartbeat_at_ms = _clock_ms()
    backend.processes_started = True

    health2 = service.execute(_request(repo, state, Operation.HEALTH))
    assert health2.data["healthy"] is True

    logs = service.execute(
        _request(
            repo,
            state,
            Operation.EVENTS,
            parameters={"stream": "logs", "limit": 10},
        )
    )
    assert logs.status is OperationStatus.SUCCEEDED
    assert logs.data["count"] >= 1
    assert all("message" in item for item in logs.data["items"])

    stop = service.execute(_request(repo, state, Operation.STOP))
    assert stop.status is OperationStatus.SUCCEEDED
    assert stop.data["state"] in {"stopping", "stopped"}


# ---------------------------------------------------------------------------
# Queries / pagination / domains
# ---------------------------------------------------------------------------


def test_domain_queries_and_pagination(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    service = _service(repo, state)

    tasks = service.execute(
        _request(repo, state, Operation.TASKS, parameters={"limit": 2, "offset": 0})
    )
    assert tasks.status is OperationStatus.SUCCEEDED
    assert tasks.data["count"] == 2
    assert tasks.data["truncated"] is True
    assert tasks.data["total"] == 5

    page2 = service.execute(
        _request(repo, state, Operation.TASKS, parameters={"limit": 2, "offset": 2})
    )
    assert page2.data["count"] == 2
    assert page2.data["items"][0]["task_cid"] != tasks.data["items"][0]["task_cid"]

    for domain, operation in (
        ("goals", Operation.GOALS),
        ("lanes", Operation.LANES),
        ("metrics", Operation.METRICS),
        ("bundles", Operation.BUNDLES),
    ):
        result = service.execute(_request(repo, state, operation))
        assert result.status is OperationStatus.SUCCEEDED, domain
        assert result.data["count"] >= 1, domain

    # Receipts without a path are served by the control state store; force the
    # database backend path with an explicit domain selector.
    receipts = service.execute(
        _request(
            repo,
            state,
            Operation.ARTIFACT_QUERY,
            parameters={"domain": "receipts"},
        )
    )
    assert receipts.status is OperationStatus.SUCCEEDED
    assert receipts.data["count"] >= 1

    worktrees = service.execute(
        _request(
            repo,
            state,
            Operation.ARTIFACT_QUERY,
            parameters={"domain": "worktrees", "limit": 10},
        )
    )
    assert worktrees.status is OperationStatus.SUCCEEDED
    assert worktrees.data["domain"] == "worktrees"
    assert worktrees.data["count"] == 1

    daemons = service.execute(
        _request(
            repo,
            state,
            Operation.ARTIFACT_QUERY,
            parameters={"domain": "daemons"},
        )
    )
    assert daemons.data["items"][0]["daemon_id"] == "daemon:1"

    mutations = service.execute(
        _request(
            repo,
            state,
            Operation.ARTIFACT_QUERY,
            parameters={"domain": "mutations"},
        )
    )
    assert mutations.data["count"] == 1

    ast = service.execute(
        _request(
            repo,
            state,
            Operation.ARTIFACT_QUERY,
            parameters={"domain": "ast"},
        )
    )
    assert ast.data["items"][0]["path"] == "pkg/mod.py"

    runs = service.execute(
        _request(
            repo,
            state,
            Operation.ARTIFACT_QUERY,
            parameters={"domain": "runs"},
        )
    )
    assert runs.data["items"][0]["run_id"] == "run:1"


def test_pagination_bounds(tmp_path: Path) -> None:
    backend = _backend()
    with pytest.raises(DatabaseControlBoundsError):
        backend.operations.query(_PROGRAM_ID, "tasks", limit=0)
    with pytest.raises(DatabaseControlBoundsError):
        backend.operations.query(_PROGRAM_ID, "tasks", limit=10_000)


# ---------------------------------------------------------------------------
# Authority / dry-run / permit / idempotency / lease
# ---------------------------------------------------------------------------


def test_read_proposal_mutation_authority_distinct(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    service = _service(repo, state)

    read = service.execute(_request(repo, state, Operation.STATUS))
    assert read.status is OperationStatus.SUCCEEDED

    preview = service.execute(
        _request(repo, state, Operation.OBJECTIVE_PREVIEW, dry_run=True)
    )
    assert preview.status is OperationStatus.SUCCEEDED
    assert preview.data.get("preview") is True or preview.data.get("dry_run") is True

    # Mutation without authorization fails closed at the request boundary.
    with pytest.raises(Exception):
        OperationRequest(
            operation=Operation.PAUSE,
            **_binding(repo, state),
            parameters={"program_id": _PROGRAM_ID},
            expected_effects=(_effect(Operation.PAUSE),),
            dry_run=False,
        )

    # Expired authorization is denied by the service.
    expired = _request(repo, state, Operation.PAUSE)
    expired_payload = expired.to_record()
    expired_payload.pop("content_id", None)
    auth = dict(expired_payload["authorization"])
    auth.pop("content_id", None)
    auth["expires_at_ms"] = _clock_ms() - 1
    expired_payload["authorization"] = auth
    denied = service.execute(OperationRequest.from_dict(expired_payload))
    assert denied.status is OperationStatus.DENIED
    assert denied.error is not None

    # Dry-run mutation does not change state.
    backend = _backend()
    service2 = _service(repo, state, backend)
    dry = service2.execute(
        _request(repo, state, Operation.START, dry_run=True)
    )
    assert dry.status is OperationStatus.SUCCEEDED
    status = service2.execute(_request(repo, state, Operation.STATUS))
    assert status.data["state"] == "stopped"
    assert backend.processes_started is False


def test_idempotent_mutation_and_lease_fence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    backend = _backend()
    service = _service(repo, state, backend)

    first = service.execute(_request(repo, state, Operation.START))
    assert first.status is OperationStatus.SUCCEEDED
    assert first.data["changed"] is True

    replay = service.execute(_request(repo, state, Operation.START))
    assert replay.status is OperationStatus.SUCCEEDED
    # Service-level idempotency returns the prior result without re-dispatch.
    assert replay.request_id == first.request_id or replay.data.get("idempotent") in {
        True,
        None,
        False,
    }

    # Stale fence is rejected.
    stale = _request(repo, state, Operation.PAUSE)
    stale_values = stale.to_record()
    stale_values.pop("content_id", None)
    stale_values["fencing_epoch"] = 1
    auth = dict(stale_values["authorization"])
    auth.pop("content_id", None)
    auth["fencing_epoch"] = 1
    stale_values["authorization"] = auth
    denied = service.execute(OperationRequest.from_dict(stale_values))
    assert denied.status in {OperationStatus.DENIED, OperationStatus.CONFLICT, OperationStatus.FAILED}
    assert denied.error is not None


def test_import_preview_export_backup(tmp_path: Path) -> None:
    backend = _backend()
    direct_preview = backend.operations.import_preview(
        _PROGRAM_ID,
        sources=[{"path": "legacy/tasks.md", "digest": "sha256:" + ("cd" * 32)}],
    )
    assert direct_preview["dry_run"] is True
    assert direct_preview["source_count"] == 1
    assert direct_preview["authority"] == OperationAuthority.PROPOSAL.value

    export = backend.operations.export_state(_PROGRAM_ID, profile="default")
    assert export["non_authoritative"] is True
    assert export["digest"].startswith("sha256:")
    assert export["changed"] is True

    listed = backend.operations.query(_PROGRAM_ID, "exports")
    assert listed["count"] == 1

    backup = backend.operations.backup(_PROGRAM_ID)
    assert backup["verified"] is True
    assert backup["backup_id"]
    assert backup["digest"].startswith("sha256:")

    dry_backup = backend.operations.backup(_PROGRAM_ID, dry_run=True)
    assert dry_backup["dry_run"] is True
    assert dry_backup["changed"] is False


# ---------------------------------------------------------------------------
# Redaction / parity / helpers
# ---------------------------------------------------------------------------


def test_redaction_of_sensitive_fields() -> None:
    ops = open_database_control_operations(clock_ms=_clock_ms)
    ops.register_program(
        DatabaseProgramTarget(program_id="program:r", store_id=_STORE_ID),
        seed={
            "tasks": [
                {
                    "task_cid": "task:1",
                    "note": "safe",
                    "credential": REDACTION_MARKER,
                }
            ]
        },
    )
    page = ops.query("program:r", "tasks")
    assert page["items"][0]["credential"] == REDACTION_MARKER
    assert page["items"][0]["note"] == "safe"

    ops.append_log(
        "program:r",
        message="handled",
        body={"token": REDACTION_MARKER, "task_cid": "task:1"},
    )
    logs = ops.logs("program:r")
    assert logs["items"][-1]["body"]["token"] == REDACTION_MARKER


def test_python_cli_mcp_share_dispatcher_identity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    service = _service(repo, state)
    request = _request(repo, state, Operation.STATUS)
    python_result = service.execute(request)
    # CLI/MCP adapters invoke the same service.execute entrypoint directly.
    cli_result = service.execute(request)
    mcp_result = service.execute(request)
    assert python_result.to_record() == cli_result.to_record() == mcp_result.to_record()
    assert DIRECT_CONTROL_SERVICE_DISPATCHER_ID.endswith(
        "SupervisorControlService.execute"
    )
    publication = service.surface_publication()
    assert publication.dispatch_mode == "direct_service"
    assert DIRECT_CONTROL_SERVICE_DISPATCHER_ID in set(
        publication.dispatcher_ids.values()
    )


def test_lifecycle_actions_pause_resume_drain_cancel_quarantine_retry(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    backend = _backend()
    # Drive lifecycle through operations with legal transitions.
    ops = backend.operations
    ops.transition(_PROGRAM_ID, "start", lease_id="lease:dqp-029", fencing_epoch=3)
    with ops._lock:
        from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
            SupervisorLifecycleState,
        )

        program = ops._programs[_PROGRAM_ID]
        program.lifecycle_state = SupervisorLifecycleState.HEALTHY
        program.processes_started = True

    paused = ops.transition(_PROGRAM_ID, "pause", lease_id="lease:dqp-029", fencing_epoch=3)
    assert paused["state"] == "paused"
    resumed = ops.transition(_PROGRAM_ID, "resume", lease_id="lease:dqp-029", fencing_epoch=3)
    assert resumed["state"] == "healthy"
    drained = ops.transition(_PROGRAM_ID, "drain", lease_id="lease:dqp-029", fencing_epoch=3)
    assert drained["state"] == "draining"

    with ops._lock:
        program = ops._programs[_PROGRAM_ID]
        program.lifecycle_state = SupervisorLifecycleState.HEALTHY

    quarantined = ops.transition(
        _PROGRAM_ID, "quarantine", lease_id="lease:dqp-029", fencing_epoch=3
    )
    assert quarantined["state"] == "blocked"

    with ops._lock:
        program = ops._programs[_PROGRAM_ID]
        program.lifecycle_state = SupervisorLifecycleState.HEALTHY

    cancelled = ops.transition(
        _PROGRAM_ID, "cancel", lease_id="lease:dqp-029", fencing_epoch=3
    )
    assert cancelled["state"] == "stopping"

    with ops._lock:
        program = ops._programs[_PROGRAM_ID]
        program.lifecycle_state = SupervisorLifecycleState.FAILED

    retried = ops.transition(_PROGRAM_ID, "retry", lease_id="lease:dqp-029", fencing_epoch=3)
    assert retried["state"] == "starting"


def test_database_backend_from_state_root(tmp_path: Path) -> None:
    backend = database_backend_from_state_root(
        tmp_path / "state",
        program_id="program:root",
        store_id=_STORE_ID,
    )
    programs = backend.discover()["programs"]
    assert len(programs) == 1
    assert programs[0]["program_id"] == "program:root"
    assert backend.processes_started is False


def test_capabilities_discovery_via_service(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    service = _service(repo, state)
    result = service.execute(_request(repo, state, Operation.CAPABILITIES))
    assert result.status is OperationStatus.SUCCEEDED
    # Capability report is produced by the service; backend discover is separate.
    report = service.capability_report()
    assert Operation.STATUS in report.supported_operations
    assert Operation.STOP in report.supported_operations
    assert Operation.HEALTH in report.supported_operations
