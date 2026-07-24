from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    ControlContractError,
    ControlSurfaceParityCase,
    ControlSurfaceParityEvidence,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    IdempotencyKey,
    READ_OPERATIONS,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
    operation_request_json_schema,
    operation_result_json_schema,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    JsonlControlStateStore,
    RepositorySupervisorBackend,
    StaleLeaseError,
    SupervisorClient,
    SupervisorControlService,
    SupervisorTarget,
)


def _binding(repo_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repo:fixture",
        "tree_id": "tree:abc",
        "objective_id": "ASI-G070",
        "objective_revision": "objective:1",
        "policy_id": "policy:supervisor",
        "policy_revision": "policy:1",
        "caller": "operator:alice",
    }


def _effect(operation: Operation) -> ExpectedEffect:
    return ExpectedEffect(
        effect_id=f"{operation.value}:target",
        kind=(
            EffectKind.EXECUTE_VALIDATION
            if operation is Operation.VALIDATION_REPLAY
            else EffectKind.WRITE_STATE
        ),
        resource="supervisor:fixture",
        paths=("data/agent_supervisor",),
        description=f"Apply {operation.value}",
    )


def _mutation_request(
    repo_root: Path,
    state_root: Path,
    operation: Operation = Operation.PAUSE,
    *,
    key: str = "request:one",
    parameters: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> OperationRequest:
    binding = _binding(repo_root, state_root)
    effect = _effect(operation)
    values: dict[str, Any] = {
        "operation": operation,
        **binding,
        "expected_effects": (effect,),
        "parameters": parameters or {"target_id": "supervisor:fixture"},
        "dry_run": dry_run,
    }
    if not dry_run:
        values.update(
            {
                "idempotency": IdempotencyKey(
                    key=key,
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
                    lease_id="lease:7",
                    fencing_epoch=7,
                    authorized_effect_ids=(effect.effect_id,),
                    grant_ids=("grant:operator",),
                    evaluated_at_ms=1_000,
                    expires_at_ms=2_000,
                ),
                "lease_id": "lease:7",
                "fencing_epoch": 7,
            }
        )
    return OperationRequest(**values)


def _read_request(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
    parameters: dict[str, Any] | None = None,
    *,
    bounds: ControlBounds | None = None,
) -> OperationRequest:
    return OperationRequest(
        operation=operation,
        **_binding(repo_root, state_root),
        parameters=parameters or {},
        bounds=bounds or ControlBounds(),
    )


def _service(
    repo_root: Path,
    state_root: Path,
    *,
    handlers: dict[Operation, Any] | None = None,
    lease_validator: Any = lambda _request: True,
    state_store: InMemoryControlStateStore | None = None,
) -> SupervisorControlService:
    return SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers=handlers,
        lease_validator=lease_validator,
        state_store=state_store or InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )


def _parity_service(
    repo_root: Path,
    state_root: Path,
) -> SupervisorControlService:
    def operation_handler(request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            data={"operation": request.operation.value},
            changed=bool(request.expected_effects),
            applied_effect_ids=tuple(
                effect.effect_id for effect in request.expected_effects
            ),
        )

    return _service(
        repo_root,
        state_root,
        handlers={
            operation: operation_handler
            for operation in Operation
            if operation not in READ_OPERATIONS
        }
        | {Operation.STATUS: operation_handler},
    )


def _parity_cases(
    service: SupervisorControlService,
    repo_root: Path,
    state_root: Path,
) -> tuple[ControlSurfaceParityCase, ...]:
    requests = (
        (
            "read_success",
            _read_request(repo_root, state_root, Operation.STATUS),
        ),
        (
            "proposal_success",
            _mutation_request(
                repo_root,
                state_root,
                Operation.PAUSE,
                dry_run=True,
            ),
        ),
        (
            "stable_failure",
            _read_request(
                repo_root,
                state_root,
                Operation.HEALTH,
                {"health_path": "missing-health.json"},
            ),
        ),
        (
            "mutation_success",
            _mutation_request(
                repo_root,
                state_root,
                Operation.PAUSE,
                key="parity:mutation",
            ),
        ),
    )
    cases = []
    for scenario, request in requests:
        record = service.execute(request).to_record()
        cases.append(
            ControlSurfaceParityCase(
                scenario=scenario,
                request=request,
                python_result=record,
                cli_result=record,
                mcp_result=record,
            )
        )
    return tuple(cases)


def test_capabilities_are_complete_typed_and_side_effect_free(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    handlers = {
        operation: (lambda _request: {})
        for operation in Operation
        if operation not in {
            Operation.CAPABILITIES,
            *tuple(READ_OPERATIONS),
        }
    }
    service = _service(repo_root, state_root, handlers=handlers)

    report = service.capabilities()

    assert report.supported_operations == tuple(
        sorted(Operation, key=lambda item: item.value)
    )
    assert report.processes_started is False
    assert report.optional_providers_loaded is False
    for operation in Operation:
        capability = report.capability_for(operation)
        assert capability is not None
        assert capability.authority is operation.authority
        assert capability.requires_idempotency is operation.mutating
        assert capability.requires_authorization is operation.mutating


def test_read_client_uses_direct_repository_apis_and_bounded_results(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    (repo_root / "objectives.md").write_text(
        "\n".join(
            (
                "# Objectives",
                "## G-1 First objective",
                "- Status: active",
                "- Acceptance: receipt",
                "## G-2 Second objective",
                "- Status: complete",
                "- Acceptance: proof",
            )
        ),
        encoding="utf-8",
    )
    (repo_root / "tasks.todo.md").write_text(
        "\n".join(
            (
                "## ASI-1 First task",
                "- Status: todo",
                "## ASI-2 Second task",
                "- Status: complete",
            )
        ),
        encoding="utf-8",
    )
    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        state_store=InMemoryControlStateStore(),
        max_query_items=2,
        clock_ms=lambda: 1_500,
    )
    client = SupervisorClient(
        service,
        target=SupervisorTarget(**_binding(repo_root, state_root)),
    )

    goals = client.goals(objective_path="objectives.md", limit=1)
    tasks = client.tasks(
        todo_path="tasks.todo.md", task_header_prefix="ASI-", limit=2
    )

    assert goals.succeeded
    assert goals.data["count"] == 1
    assert goals.data["truncated"] is True
    assert goals.data["items"][0]["goal_id"] == "G-1"
    assert tasks.data["count"] == 2
    assert [item["task_id"] for item in tasks.data["items"]] == [
        "ASI-1",
        "ASI-2",
    ]
    assert goals.audit_receipt_id.startswith("sha256:")


def test_mutation_is_authorized_fenced_audited_and_idempotent(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls: list[str] = []
    leases: list[tuple[str, int | None]] = []
    store = InMemoryControlStateStore()

    def pause(request: OperationRequest) -> BackendResponse:
        calls.append(request.request_id)
        return BackendResponse(
            data={"state": "paused"},
            changed=True,
            applied_effect_ids=("pause:target",),
        )

    def validate_lease(request: OperationRequest) -> bool:
        leases.append((request.lease_id, request.fencing_epoch))
        return request.lease_id == "lease:7" and request.fencing_epoch == 7

    service = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: pause},
        lease_validator=validate_lease,
        state_store=store,
    )
    request = _mutation_request(repo_root, state_root)

    first = service.pause(request)
    replay = service.execute(request)

    assert first is replay
    assert first.status is OperationStatus.SUCCEEDED
    assert first.data["state"] == "paused"
    assert first.effects[0].applied is True
    assert first.effects[0].receipt_id == first.audit_receipt_id
    assert first.idempotency_key == "request:one"
    assert calls == [request.request_id]
    assert leases == [("lease:7", 7)]


def test_same_idempotency_key_with_different_payload_conflicts(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: lambda _request: {"state": "paused"}},
    )
    first = _mutation_request(repo_root, state_root)
    changed = _mutation_request(
        repo_root,
        state_root,
        parameters={"target_id": "supervisor:other"},
    )

    assert service.execute(first).succeeded
    conflict = service.execute(changed)

    assert conflict.status is OperationStatus.CONFLICT
    assert conflict.error is not None
    assert conflict.error.code is ErrorCode.IDEMPOTENCY_CONFLICT


def test_default_store_replays_exact_mutation_result_after_restart(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls = 0

    def pause(_request: OperationRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"state": "paused"}

    request = _mutation_request(repo_root, state_root)
    first_service = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: pause},
        state_store=JsonlControlStateStore(),
    )
    first = first_service.execute(request)
    restarted = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: pause},
        state_store=JsonlControlStateStore(),
    )

    replay = restarted.execute(request)

    assert replay == first
    assert replay.result_id == first.result_id
    assert calls == 1


def test_dry_run_never_calls_mutation_or_requires_a_live_lease(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls = 0

    def forbidden_call(_request: OperationRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    service = _service(
        repo_root,
        state_root,
        handlers={Operation.QUARANTINE: forbidden_call},
        lease_validator=lambda _request: (_ for _ in ()).throw(
            AssertionError("dry run checked a live lease")
        ),
    )
    request = _mutation_request(
        repo_root,
        state_root,
        operation=Operation.QUARANTINE,
        dry_run=True,
    )

    result = service.quarantine(request)

    assert result.succeeded
    assert result.authority is OperationAuthority.PROPOSAL
    assert result.preview is not None
    assert result.preview.would_change is True
    assert result.effects == ()
    assert result.preview.expected_effects == request.expected_effects
    assert calls == 0


def test_allowlists_bounds_and_paths_fail_with_stable_errors(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    other_repo = tmp_path / "other"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    other_repo.mkdir()
    state_root.mkdir()
    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        state_store=InMemoryControlStateStore(),
        max_query_items=2,
        clock_ms=lambda: 1_500,
    )

    denied = service.execute(
        _read_request(
            other_repo,
            state_root,
            Operation.STATUS,
            {"status_path": "status.json"},
        )
    )
    bounded = service.execute(
        _read_request(
            repo_root,
            state_root,
            Operation.EVENTS,
            {"events_path": "events.jsonl", "limit": 3},
        )
    )
    escaped = service.execute(
        _read_request(
            repo_root,
            state_root,
            Operation.STATUS,
            {"status_path": "../outside.json"},
        )
    )

    assert denied.status is OperationStatus.DENIED
    assert denied.error and denied.error.code is ErrorCode.FORBIDDEN
    assert bounded.error and bounded.error.code is ErrorCode.BOUNDS_EXCEEDED
    assert escaped.error and escaped.error.code is ErrorCode.PATH_ESCAPE


def test_stale_fence_and_expired_authorization_fail_before_backend(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls = 0

    def backend(_request: OperationRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"state": "paused"}

    service = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: backend},
        lease_validator=lambda _request: (_ for _ in ()).throw(
            StaleLeaseError("fencing epoch has been superseded")
        ),
    )
    stale = service.execute(_mutation_request(repo_root, state_root))

    assert stale.status is OperationStatus.CONFLICT
    assert stale.error and stale.error.code is ErrorCode.STALE_LEASE
    assert calls == 0

    expired_service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={Operation.PAUSE: backend},
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 2_001,
    )
    expired = expired_service.execute(_mutation_request(repo_root, state_root))
    assert expired.status is OperationStatus.DENIED
    assert expired.error and expired.error.code is ErrorCode.UNAUTHORIZED
    assert calls == 0

    denied_service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        authorization_validator=lambda _request: False,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )
    denied = denied_service.execute(
        _read_request(
            repo_root,
            state_root,
            Operation.STATUS,
            {"status_path": "status.json"},
        )
    )
    assert denied.status is OperationStatus.DENIED
    assert denied.error and denied.error.code is ErrorCode.UNAUTHORIZED


@pytest.mark.parametrize(
    ("exception", "code", "status"),
    (
        (FileNotFoundError("missing"), ErrorCode.NOT_FOUND, OperationStatus.NOT_FOUND),
        (TimeoutError("slow"), ErrorCode.TIMED_OUT, OperationStatus.TIMED_OUT),
        (ValueError("bad selector"), ErrorCode.INVALID_REQUEST, OperationStatus.FAILED),
        (RuntimeError("secret backend detail"), ErrorCode.INTERNAL_ERROR, OperationStatus.FAILED),
    ),
)
def test_backend_failures_are_translated_to_stable_typed_errors(
    tmp_path: Path,
    exception: Exception,
    code: ErrorCode,
    status: OperationStatus,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()

    def fail(_request: OperationRequest) -> dict[str, Any]:
        raise exception

    service = _service(
        repo_root,
        state_root,
        handlers={Operation.PLAN: fail},
    )
    result = service.plan(
        _read_request(repo_root, state_root, Operation.PLAN, {"limit": 1})
    )

    assert result.status is status
    assert result.error is not None
    assert result.error.code is code
    if code is ErrorCode.INTERNAL_ERROR:
        assert result.error.message == "control operation failed"


def test_receipt_query_is_bounded_and_does_not_require_raw_state_access(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    store = InMemoryControlStateStore()
    service = _service(repo_root, state_root, state_store=store)

    status_request = _read_request(
        repo_root,
        state_root,
        Operation.STATUS,
        {"status_path": "status.json"},
    )
    # Missing reads are audited too.
    assert service.status(status_request).status is OperationStatus.NOT_FOUND
    receipts = service.receipts(
        _read_request(
            repo_root,
            state_root,
            Operation.RECEIPTS,
            {"limit": 1},
        )
    )

    assert receipts.succeeded
    assert receipts.data["count"] == 1
    assert receipts.data["items"][0]["operation"] == "status"
    assert receipts.data["items"][0]["error_code"] == "not_found"


def test_service_calls_registered_python_handler_without_shell_translation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    seen: list[OperationRequest] = []

    def replay(request: OperationRequest) -> BackendResponse:
        seen.append(request)
        return BackendResponse(
            data={"validation_receipt": "receipt:new"},
            changed=True,
            applied_effect_ids=("validation_replay:target",),
        )

    backend = RepositorySupervisorBackend(
        {Operation.VALIDATION_REPLAY: replay}
    )
    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        backend=backend,
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )
    request = _mutation_request(
        repo_root,
        state_root,
        operation=Operation.VALIDATION_REPLAY,
    )

    result = service.validation_replay(request)

    assert result.succeeded
    assert seen == [request]
    assert result.data["validation_receipt"] == "receipt:new"


def test_shared_wire_schemas_cover_every_operation_and_mutation_guard() -> None:
    request_schema = operation_request_json_schema()
    result_schema = operation_result_json_schema()

    assert set(request_schema["properties"]["operation"]["enum"]) == {
        item.value for item in Operation
    }
    assert set(result_schema["properties"]["operation"]["enum"]) == {
        item.value for item in Operation
    }
    pause_schema = operation_request_json_schema(Operation.PAUSE)
    assert pause_schema["properties"]["operation"]["const"] == "pause"
    assert {
        "expected_effects",
        "idempotency",
        "authorization",
        "lease_id",
        "fencing_epoch",
    }.issubset(pause_schema["allOf"][0]["then"]["required"])


def test_typed_surface_parity_evidence_proves_exact_requirement(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _parity_service(repo_root, state_root)
    cases = _parity_cases(service, repo_root, state_root)
    request = cases[0].request
    assert isinstance(request, OperationRequest)
    with pytest.raises(ControlContractError, match="complete behavior matrix"):
        ControlSurfaceParityEvidence(
            repository_tree=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            policy_revision=request.policy_revision,
            capability_report=service.capability_report(),
            cases=(cases[0],),
        )
    evidence = ControlSurfaceParityEvidence(
        repository_tree=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        policy_revision=request.policy_revision,
        capability_report=service.capability_report().to_record(),
        cases=cases,
    )

    assert evidence.proved_requirement_ids == (
        CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
    )
    assert ControlSurfaceParityEvidence.from_dict(evidence.to_record()) == evidence
    assert evidence.request_schema_id
    assert evidence.result_schema_id


def test_surface_parity_evidence_rejects_behavior_or_schema_drift(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _parity_service(repo_root, state_root)
    request = _read_request(repo_root, state_root, Operation.STATUS)
    record = service.execute(request).to_record()
    drifted = dict(record)
    drifted["data"] = {"state": "degraded"}
    drifted.pop("content_id")

    with pytest.raises(ControlContractError, match="not canonically identical"):
        ControlSurfaceParityCase(
            scenario="read_success",
            request=request,
            python_result=record,
            cli_result=drifted,
            mcp_result=record,
        )

    cases = _parity_cases(service, repo_root, state_root)
    evidence = ControlSurfaceParityEvidence(
        repository_tree=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        policy_revision=request.policy_revision,
        capability_report=service.capability_report(),
        cases=cases,
    ).to_record()
    evidence["request_schema_id"] = "sha256:forged"
    evidence.pop("content_id")
    with pytest.raises(ControlContractError, match="request_schema_id"):
        ControlSurfaceParityEvidence.from_dict(evidence)
