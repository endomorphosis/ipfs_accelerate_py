from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor import control_cli
from ipfs_accelerate_py.agent_supervisor import control_plane as control_plane_module
from ipfs_accelerate_py.agent_supervisor.control_cli import (
    AGENT_CLI_EXIT_CANCELLED,
    AGENT_CLI_EXIT_CONFLICT,
    AGENT_CLI_EXIT_NOT_FOUND,
    AGENT_CLI_EXIT_TIMED_OUT,
    AGENT_CLI_EXIT_UNAVAILABLE,
    COMMAND_OPERATIONS,
    agent_cli_discovery_manifest,
)
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    MUTATION_OPERATIONS,
    OPERATION_CATALOG_V2,
    PROPOSAL_OPERATIONS,
    AuthorizationDecision,
    AuthorizationVerdict,
    CapabilityDegradation,
    ControlContractError,
    ControlDiscoveryManifest,
    ControlSurface,
    EffectKind,
    EventCursor,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationCatalog,
    OperationRequest,
    OperationResult,
    OperationStatus,
    PaginationKind,
    UnsupportedCapabilityError,
    operation_request_json_schema,
    operation_result_json_schema,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    CONTROL_CATALOG_CONFORMANCE_EVIDENCE_SCHEMA,
    CONTROL_CONFORMANCE_V2_REQUIREMENT_ID,
    CONTROL_OPERATION_CONFORMANCE_CASE_SCHEMA,
    CONTROL_OPTIONAL_PROVIDER_MODULE_PREFIXES,
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    BackendCancelledError,
    BackendConflictError,
    BackendNotFoundError,
    BackendResponse,
    BackendTimeoutError,
    ControlCatalogConformanceError,
    ControlCatalogConformanceEvidence,
    ControlOperationConformanceCase,
    ControlSurfacePublication,
    InMemoryControlStateStore,
    OperationUnavailableError,
    SupervisorControlService,
    capture_control_discovery_runtime_state,
    control_operation_behavior_id,
    publish_control_catalog,
    validate_catalog_publication,
    validate_control_surface_publication,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    agent_supervisor_discovery_manifest,
    agent_supervisor_service_resolution_count,
    configure_agent_supervisor_control,
    register_native_agent_supervisor_tools,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    native_agent_supervisor_tools as native_tools,
)


class _RecordingToolManager:
    def __init__(self) -> None:
        self.definitions: list[dict[str, Any]] = []

    def register_tool(self, **definition: Any) -> None:
        self.definitions.append(definition)


@pytest.fixture(autouse=True)
def _reset_mcp_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent_supervisor_control()
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST",
        raising=False,
    )
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_STATE_ALLOWLIST",
        raising=False,
    )
    yield
    configure_agent_supervisor_control()


def _binding(repo_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repository:asi-115",
        "tree_id": "tree:asi-115",
        "objective_id": "ASI-G270",
        "objective_revision": "ASI-G270@ASI-115",
        "policy_id": "policy:asi-115",
        "policy_revision": "policy:asi-115@1",
        "caller": "test:asi-115",
    }


def _selector_value(name: str, binding: Mapping[str, Any]) -> str:
    if name in binding:
        return str(binding[name])
    return {
        "service_id": "service:asi-115",
        "task_id": "ASI-115",
        "bundle_id": "bundle:control",
        "lane_id": "lane:control-conformance",
        "stream_id": "events:asi-115",
        "receipt_id": "receipt:asi-115",
        "cache_namespace": "cache:asi-115",
        "artifact_id": "artifact:asi-115",
        "validation_id": "validation:asi-115",
    }[name]


def _expected_effect(operation: Operation) -> ExpectedEffect:
    kind = (
        EffectKind.EXECUTE_VALIDATION
        if operation is Operation.VALIDATION_REPLAY
        else (
            EffectKind.LIFECYCLE_TRANSITION
            if operation
            in {
                Operation.START,
                Operation.PAUSE,
                Operation.RESUME,
                Operation.DRAIN,
                Operation.STOP,
            }
            else EffectKind.WRITE_STATE
        )
    )
    return ExpectedEffect(
        effect_id=f"{operation.value}:asi-115",
        kind=kind,
        resource=f"agent-supervisor:{operation.value}",
        paths=(f"control/{operation.value}.json",),
        description=f"Apply the ASI-115 {operation.value} fixture",
    )


def _request_for_operation(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
) -> OperationRequest:
    binding = _binding(repo_root, state_root)
    descriptor = OPERATION_CATALOG_V2.operation(operation)
    selectors = {
        name: _selector_value(name, binding)
        for name in descriptor.target_descriptor.required_selectors
    }
    parameters: dict[str, Any] = {
        "target": selectors,
        **selectors,
    }
    if descriptor.pagination.kind is PaginationKind.CURSOR:
        parameters.update(
            {
                "limit": 2,
                "cursor": f"cursor:{operation.value}:0",
            }
        )
    elif descriptor.pagination.kind is PaginationKind.EVENT_CURSOR:
        cursor = EventCursor.initial(
            selectors["stream_id"],
            snapshot_id="snapshot:asi-115",
        )
        parameters.update(
            {
                "limit": 2,
                "event_cursor": cursor.to_token(),
            }
        )

    effects: tuple[ExpectedEffect, ...] = ()
    if operation in PROPOSAL_OPERATIONS:
        effects = (
            ExpectedEffect(
                effect_id=f"{operation.value}:proposal",
                kind=EffectKind.PROPOSE,
                resource=f"objective:{operation.value}",
                paths=("docs/architecture",),
                description=f"Preview {operation.value}",
            ),
        )
    elif operation in MUTATION_OPERATIONS:
        effects = (_expected_effect(operation),)

    values: dict[str, Any] = {
        "operation": operation,
        **binding,
        "parameters": parameters,
        "expected_effects": effects,
    }
    if operation in MUTATION_OPERATIONS:
        effect = effects[0]
        values.update(
            {
                "idempotency": IdempotencyKey(
                    key=f"asi-115:{operation.value}",
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
                    lease_id="lease:asi-115",
                    fencing_epoch=115,
                    authorized_effect_ids=(effect.effect_id,),
                    grant_ids=("grant:asi-115",),
                    evaluated_at_ms=1_000,
                    expires_at_ms=2_000,
                ),
                "lease_id": "lease:asi-115",
                "fencing_epoch": 115,
            }
        )
    return OperationRequest(**values)


def _canonical_backend(
    calls: list[Operation],
) -> Callable[[OperationRequest], BackendResponse]:
    def execute(request: OperationRequest) -> BackendResponse:
        calls.append(request.operation)
        descriptor = OPERATION_CATALOG_V2.operation(request.operation)
        target = {
            name: request.parameters[name]
            for name in descriptor.target_descriptor.required_selectors
        }
        data: dict[str, Any] = {
            "operation": request.operation.value,
            "target": target,
        }
        if descriptor.pagination.kind is PaginationKind.CURSOR:
            data["pagination"] = {
                "items": [{"id": f"{request.operation.value}:1"}],
                "limit": request.parameters["limit"],
                "next_cursor": f"cursor:{request.operation.value}:1",
                "has_more": False,
            }
        elif descriptor.pagination.kind is PaginationKind.EVENT_CURSOR:
            cursor = EventCursor.from_token(
                str(request.parameters["event_cursor"])
            )
            next_cursor = cursor.advance(position=1, event_id="event:1")
            data["pagination"] = {
                "events": [{"sequence": 1, "event_id": "event:1"}],
                "limit": request.parameters["limit"],
                "next_event_cursor": next_cursor.to_token(),
                "has_more": False,
            }
        return BackendResponse(
            data=data,
            changed=bool(request.expected_effects),
            applied_effect_ids=(
                tuple(
                    effect.effect_id for effect in request.expected_effects
                )
                if request.operation in MUTATION_OPERATIONS
                and not request.dry_run
                else ()
            ),
            checks=("catalog_operation", "target", "pagination"),
        )

    return execute


def _service(
    repo_root: Path,
    state_root: Path,
    *,
    handler: Callable[[OperationRequest], Any] | None = None,
) -> tuple[SupervisorControlService, list[Operation]]:
    calls: list[Operation] = []
    selected_handler = handler or _canonical_backend(calls)
    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={
            operation: selected_handler
            for operation in Operation
        },
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )
    return service, calls


def _cli_command(operation: Operation) -> str:
    return next(
        command
        for command, candidate in COMMAND_OPERATIONS.items()
        if candidate is operation
    )


def _invoke_cli(
    capsys: pytest.CaptureFixture[str],
    service: SupervisorControlService,
    request: OperationRequest,
) -> tuple[int, dict[str, Any]]:
    exit_status = cli.main(
        [
            "agent",
            _cli_command(request.operation),
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    return int(exit_status), json.loads(captured.out)


def test_catalog_schemas_and_static_surface_populations_are_exact() -> None:
    catalog_operations = tuple(
        sorted(Operation, key=lambda item: item.value)
    )
    command_operations = tuple(
        sorted(COMMAND_OPERATIONS.values(), key=lambda item: item.value)
    )
    tool_operations = tuple(
        sorted(AGENT_SUPERVISOR_OPERATION_TOOLS, key=lambda item: item.value)
    )
    manifests = (
        ControlDiscoveryManifest(surface=ControlSurface.PYTHON),
        agent_cli_discovery_manifest(),
        agent_supervisor_discovery_manifest(),
    )

    assert OPERATION_CATALOG_V2.operations == catalog_operations
    assert command_operations == catalog_operations
    assert tool_operations == catalog_operations
    assert len(COMMAND_OPERATIONS) == len(Operation)
    assert len(AGENT_SUPERVISOR_OPERATION_TOOLS) == len(Operation)
    assert {
        manifest.schema_population_id for manifest in manifests
    } == {manifests[0].schema_population_id}

    manager = _RecordingToolManager()
    register_native_agent_supervisor_tools(manager)
    definitions = {
        Operation(definition["name"]): definition
        for definition in manager.definitions
    }
    assert tuple(
        sorted(definitions, key=lambda item: item.value)
    ) == catalog_operations

    for descriptor in OPERATION_CATALOG_V2:
        operation = descriptor.operation
        assert descriptor.request_schema_id == content_identity(
            operation_request_json_schema(operation)
        )
        assert descriptor.result_schema_id == content_identity(
            operation_result_json_schema(operation)
        )
        assert descriptor.request_schema_id == (
            manifests[0].request_schema_ids[operation.value]
        )
        assert descriptor.result_schema_id == (
            manifests[0].result_schema_ids[operation.value]
        )
        definition = definitions[operation]
        tool_schema = definition["input_schema"]
        assert content_identity(
            tool_schema["properties"]["request"]
        ) == descriptor.request_schema_id
        assert content_identity(
            tool_schema["x-output-schema"]
        ) == descriptor.result_schema_id
        contract = tool_schema["x-agent-supervisor-contract"]
        assert contract["request_schema_id"] == descriptor.request_schema_id
        assert contract["result_schema_id"] == descriptor.result_schema_id


@pytest.mark.asyncio
async def test_every_catalog_operation_has_exact_python_cli_and_mcp_behavior(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()

    compared: set[Operation] = set()
    cases: list[ControlOperationConformanceCase] = []
    for operation in OPERATION_CATALOG_V2.operations:
        request = _request_for_operation(
            repo_root,
            state_root,
            operation,
        )
        assert OperationRequest.from_json(request.to_json()) == request
        generated_command = control_cli.build_agent_cli_command(request)
        assert generated_command[:3] == (
            "ipfs-accelerate",
            "agent",
            _cli_command(operation),
        )
        assert OperationRequest.from_json(generated_command[4]) == request
        assert generated_command[-1] == "--output-json"
        descriptor = OPERATION_CATALOG_V2.operation(operation)
        assert set(
            descriptor.target_descriptor.required_selectors
        ).issubset(request.parameters)

        python_service, python_calls = _service(repo_root, state_root)
        entry_point = getattr(python_service, operation.value)
        python_result = entry_point(request)
        assert isinstance(python_result, OperationResult)
        python_record = python_result.to_record()

        cli_service, cli_calls = _service(repo_root, state_root)
        exit_status, cli_record = _invoke_cli(
            capsys,
            cli_service,
            request,
        )

        mcp_service, mcp_calls = _service(repo_root, state_root)
        configure_agent_supervisor_control(service=mcp_service)
        mcp_record = await AGENT_SUPERVISOR_OPERATION_TOOLS[operation](
            request=request.to_record()
        )

        assert exit_status == 0, operation.value
        assert cli_record == python_record, operation.value
        assert mcp_record == python_record, operation.value
        for record in (python_record, cli_record, mcp_record):
            result = OperationResult.from_dict(record)
            result.validate_against(request)
            assert result.operation is operation
            assert result.status is OperationStatus.SUCCEEDED
            assert record["content_id"] == python_record["content_id"]

        if descriptor.pagination.kind is not PaginationKind.NONE:
            # The canonical result envelope is shared. Domain backends may
            # retain their established flat page fields while newer handlers
            # use a nested pagination object, so parity does not prescribe a
            # second, transport-specific page shape.
            python_page = python_record["data"].get(
                "pagination",
                python_record["data"],
            )
            cli_page = cli_record["data"].get(
                "pagination",
                cli_record["data"],
            )
            mcp_page = mcp_record["data"].get(
                "pagination",
                mcp_record["data"],
            )
            assert cli_page == python_page
            assert mcp_page == python_page
        if descriptor.pagination.kind is PaginationKind.EVENT_CURSOR:
            token = python_page["next_event_cursor"]
            assert EventCursor.from_token(token).position == 1

        expected_calls = (
            []
            if operation
            in {Operation.CAPABILITIES, Operation.RECEIPTS}
            else [operation]
        )
        assert python_calls == expected_calls
        assert cli_calls == expected_calls
        assert mcp_calls == expected_calls
        if operation in MUTATION_OPERATIONS:
            assert python_result.idempotency_key == request.idempotency_key
            assert python_result.audit_receipt_id
            assert {
                effect.effect_id for effect in python_result.effects
            } == {
                effect.effect_id for effect in request.expected_effects
            }
            assert all(effect.applied for effect in python_result.effects)
        elif operation in PROPOSAL_OPERATIONS:
            assert python_result.preview is not None
            assert not any(effect.applied for effect in python_result.effects)
        else:
            assert python_result.effects == ()
        cases.append(
            ControlOperationConformanceCase(
                scenario=f"{operation.value}:canonical-success",
                request=request,
                python_result=python_record,
                cli_result=cli_record,
                mcp_result=mcp_record,
                cli_exit_status=exit_status,
            )
        )
        compared.add(operation)

    assert compared == set(OPERATION_CATALOG_V2.operations) == set(Operation)
    assert len(cases) == len(Operation) == 26
    assert len({case.operation for case in cases}) == 26
    assert all(
        ControlOperationConformanceCase.from_dict(case.to_record()) == case
        for case in cases
    )

    service, _ = _service(repo_root, state_root)
    manifests = (
        service.discovery_manifest(),
        agent_cli_discovery_manifest(),
        agent_supervisor_discovery_manifest(),
    )
    evidence = validate_catalog_publication(
        OPERATION_CATALOG_V2,
        manifests,
        cases,
    )

    assert isinstance(evidence, ControlCatalogConformanceEvidence)
    assert CONTROL_CONFORMANCE_V2_REQUIREMENT_ID == (
        "107787885166558411314422313513714746721"
    )
    assert evidence.proved_requirement_ids == (
        CONTROL_CONFORMANCE_V2_REQUIREMENT_ID,
    )
    assert evidence.completion_authoritative is False
    assert len(evidence.cases) == len(Operation) == 26
    assert {case.operation for case in evidence.cases} == set(Operation)
    assert {
        manifest.surface for manifest in evidence.manifests
    } == set(ControlSurface)
    assert len(evidence.manifests) == len(ControlSurface) == 3
    assert all(
        case.to_record()["schema"]
        == CONTROL_OPERATION_CONFORMANCE_CASE_SCHEMA
        for case in evidence.cases
    )
    assert (
        evidence.to_record()["schema"]
        == CONTROL_CATALOG_CONFORMANCE_EVIDENCE_SCHEMA
    )
    restored = ControlCatalogConformanceEvidence.from_dict(
        evidence.to_record()
    )
    assert restored == evidence
    assert restored.canonical_bytes() == evidence.canonical_bytes()
    assert publish_control_catalog(
        OPERATION_CATALOG_V2,
        manifests,
        cases,
    ) == evidence


def _direct_conformance_case(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
    *,
    scenario: str | None = None,
) -> ControlOperationConformanceCase:
    request = _request_for_operation(repo_root, state_root, operation)
    service, _ = _service(repo_root, state_root)
    record = service.execute(request).to_record()
    return ControlOperationConformanceCase(
        scenario=scenario or f"{operation.value}:direct-success",
        request=request,
        python_result=record,
        cli_result=record,
        mcp_result=record,
        cli_exit_status=0,
    )


def test_typed_catalog_evidence_rejects_population_result_effect_and_exit_drift(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    cases = tuple(
        _direct_conformance_case(repo_root, state_root, operation)
        for operation in OPERATION_CATALOG_V2.operations
    )
    service, _ = _service(repo_root, state_root)
    manifests = (
        service.discovery_manifest(),
        agent_cli_discovery_manifest(),
        agent_supervisor_discovery_manifest(),
    )

    with pytest.raises(ControlContractError, match="population drift"):
        validate_catalog_publication(
            OPERATION_CATALOG_V2,
            manifests,
            cases[:-1],
        )
    with pytest.raises(ControlContractError, match="unique|population drift"):
        validate_catalog_publication(
            OPERATION_CATALOG_V2,
            manifests,
            cases + (cases[0],),
        )

    status_case = next(
        case for case in cases if case.operation is Operation.STATUS
    )
    status_record = status_case.python_result.to_record()
    inconsistent_result = dict(status_record)
    inconsistent_result["data"] = {"transport": "cli-only-drift"}
    inconsistent_result.pop("content_id")
    with pytest.raises(ControlContractError, match="inconsistent"):
        ControlOperationConformanceCase(
            scenario="status:result-drift",
            request=status_case.request,
            python_result=status_record,
            cli_result=inconsistent_result,
            mcp_result=status_record,
            cli_exit_status=0,
        )

    mutation_case = next(
        case for case in cases if case.operation is Operation.START
    )
    mutation_record = mutation_case.python_result.to_record()
    inconsistent_effects = dict(mutation_record)
    inconsistent_effects["effects"] = []
    inconsistent_effects.pop("content_id")
    with pytest.raises(ControlContractError, match="inconsistent"):
        ControlOperationConformanceCase(
            scenario="start:effect-drift",
            request=mutation_case.request,
            python_result=mutation_record,
            cli_result=mutation_record,
            mcp_result=inconsistent_effects,
            cli_exit_status=0,
        )

    with pytest.raises(ControlContractError, match="exit status"):
        ControlOperationConformanceCase(
            scenario="status:exit-drift",
            request=status_case.request,
            python_result=status_record,
            cli_result=status_record,
            mcp_result=status_record,
            cli_exit_status=AGENT_CLI_EXIT_TIMED_OUT,
        )


_ERROR_CASES = (
    (
        BackendNotFoundError("fixture absent"),
        OperationStatus.NOT_FOUND,
        "not_found",
        AGENT_CLI_EXIT_NOT_FOUND,
        False,
    ),
    (
        BackendConflictError("fixture conflict"),
        OperationStatus.CONFLICT,
        "conflict",
        AGENT_CLI_EXIT_CONFLICT,
        False,
    ),
    (
        OperationUnavailableError("fixture capability unavailable"),
        OperationStatus.UNAVAILABLE,
        "unavailable",
        AGENT_CLI_EXIT_UNAVAILABLE,
        False,
    ),
    (
        BackendTimeoutError("fixture deadline elapsed"),
        OperationStatus.TIMED_OUT,
        "timed_out",
        AGENT_CLI_EXIT_TIMED_OUT,
        True,
    ),
    (
        BackendCancelledError("fixture cancelled"),
        OperationStatus.CANCELLED,
        "cancelled",
        AGENT_CLI_EXIT_CANCELLED,
        False,
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "raised",
        "expected_status",
        "expected_error",
        "expected_exit",
        "retryable",
    ),
    _ERROR_CASES,
    ids=("not-found", "conflict", "unavailable", "timeout", "cancelled"),
)
async def test_error_status_exit_timeout_and_cancellation_are_normalized(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    raised: Exception,
    expected_status: OperationStatus,
    expected_error: str,
    expected_exit: int,
    retryable: bool,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _request_for_operation(
        repo_root,
        state_root,
        Operation.STATUS,
    )

    def fail(_request: OperationRequest) -> None:
        raise type(raised)(str(raised))

    python_service, _ = _service(
        repo_root,
        state_root,
        handler=fail,
    )
    python_record = python_service.execute(request).to_record()

    cli_service, _ = _service(
        repo_root,
        state_root,
        handler=fail,
    )
    exit_status, cli_record = _invoke_cli(capsys, cli_service, request)

    mcp_service, _ = _service(
        repo_root,
        state_root,
        handler=fail,
    )
    configure_agent_supervisor_control(service=mcp_service)
    mcp_record = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS](
        request=request.to_record()
    )

    assert exit_status == expected_exit
    assert cli_record == python_record
    assert mcp_record == python_record
    result = OperationResult.from_dict(python_record)
    result.validate_against(request)
    assert result.status is expected_status
    assert result.error is not None
    assert result.error.code.value == expected_error
    assert result.error.retryable is retryable
    assert result.audit_receipt_id


def test_capability_degradation_is_closed_and_transport_neutral() -> None:
    records: dict[Operation, dict[str, Any]] = {}
    for descriptor in OPERATION_CATALOG_V2:
        operation = descriptor.operation
        try:
            resolution = OPERATION_CATALOG_V2.resolve_backend_capability(
                operation,
                (),
            )
        except UnsupportedCapabilityError:
            assert descriptor.degradation in {
                CapabilityDegradation.FAIL_CLOSED,
                CapabilityDegradation.NOT_APPLICABLE,
            }
        else:
            assert resolution.degraded
            assert not resolution.supported
            assert resolution.degradation is descriptor.degradation
            assert descriptor.degradation in {
                CapabilityDegradation.LOCAL_READ_ONLY,
                CapabilityDegradation.PROPOSAL_ONLY,
            }
            records[operation] = resolution.to_record()

    expected_degraded = {
        descriptor.operation
        for descriptor in OPERATION_CATALOG_V2
        if descriptor.degradation
        in {
            CapabilityDegradation.LOCAL_READ_ONLY,
            CapabilityDegradation.PROPOSAL_ONLY,
        }
    }
    assert set(records) == expected_degraded
    assert all(
        record["backend_capability"]
        == OPERATION_CATALOG_V2.operation(operation).backend_capability
        for operation, record in records.items()
    )


@pytest.mark.asyncio
async def test_mcp_dispatch_never_uses_cli_or_a_child_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service, _ = _service(repo_root, state_root)
    configure_agent_supervisor_control(service=service)
    request = _request_for_operation(
        repo_root,
        state_root,
        Operation.STATUS,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("MCP attempted CLI-string or process dispatch")

    monkeypatch.setattr(cli, "main", forbidden)
    monkeypatch.setattr(control_cli, "run_agent_cli", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "call", forbidden)
    monkeypatch.setattr(subprocess, "check_call", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(os, "system", forbidden)

    record = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS](
        request=request.to_record()
    )

    assert record == service.execute(request).to_record()


def test_package_import_and_mcp_tools_list_are_provider_and_process_free(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "import_probe.py"
    probe.write_text(
        """
import json
import sys

provider_prefixes = (
    "ipfs_datasets_py",
    "ipfs_accelerate_py.agent_supervisor.ipfs_datasets_",
    "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
    "ipfs_accelerate_py.agent_supervisor.formal_verification_provider",
)
started = []

def audit(event, args):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
        started.append(event)
        raise RuntimeError("import or tools/list started a process")

sys.addaudithook(audit)
before = set(sys.modules)

import ipfs_accelerate_py
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    agent_supervisor_discovery_manifest,
    agent_supervisor_service_resolution_count,
    register_native_agent_supervisor_tools,
)

class Manager:
    def __init__(self):
        self.definitions = []
    def register_tool(self, **definition):
        self.definitions.append(definition)

manager = Manager()
resolutions_before = agent_supervisor_service_resolution_count()
first = agent_supervisor_discovery_manifest()
register_native_agent_supervisor_tools(manager)
second = agent_supervisor_discovery_manifest()
resolutions_after = agent_supervisor_service_resolution_count()
loaded = sorted(
    name
    for name in set(sys.modules).difference(before)
    if name.startswith(provider_prefixes)
)
print(json.dumps({
    "loaded": loaded,
    "processes": started,
    "resolutions_before": resolutions_before,
    "resolutions_after": resolutions_after,
    "operation_count": len(first.operations),
    "tool_count": len(manager.definitions),
    "repeatable": first.to_record() == second.to_record(),
}))
""".strip(),
        encoding="utf-8",
    )
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("IPFS_ACCEL_SKIP_CORE", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(repository_root),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    completed = subprocess.run(
        [sys.executable, str(probe)],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    observation = json.loads(completed.stdout)
    assert observation == {
        "loaded": [],
        "processes": [],
        "resolutions_before": 0,
        "resolutions_after": 0,
        "operation_count": len(Operation),
        "tool_count": len(Operation),
        "repeatable": True,
    }


def test_mcp_discovery_observation_changes_no_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    original_import = builtins.__import__

    def observed_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name.startswith(CONTROL_OPTIONAL_PROVIDER_MODULE_PREFIXES):
            imports.append(name)
        return original_import(name, globals, locals, fromlist, level)

    def forbidden_process(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("MCP tools/list started a process")

    monkeypatch.setattr(builtins, "__import__", observed_import)
    monkeypatch.setattr(subprocess, "Popen", forbidden_process)
    resolution_before = agent_supervisor_service_resolution_count()
    before = capture_control_discovery_runtime_state(
        service_resolution_count=resolution_before,
        optional_provider_load_count=0,
        process_start_count=0,
    )

    first = agent_supervisor_discovery_manifest()
    manager = _RecordingToolManager()
    register_native_agent_supervisor_tools(manager)
    second = agent_supervisor_discovery_manifest()

    resolution_after = agent_supervisor_service_resolution_count()
    after = capture_control_discovery_runtime_state(
        service_resolution_count=resolution_after,
        optional_provider_load_count=len(imports),
        process_start_count=0,
    )
    assert first == second
    assert first.schema_population_id == second.schema_population_id
    assert before == after
    assert imports == []
    assert len(manager.definitions) == len(Operation)


def test_current_catalog_rejects_missing_and_duplicate_publication() -> None:
    descriptors = OPERATION_CATALOG_V2.operation_descriptors
    with pytest.raises(ControlContractError, match="exactly cover"):
        OperationCatalog(descriptors[:-1])
    with pytest.raises(ControlContractError, match="duplicate"):
        OperationCatalog(descriptors + (descriptors[-1],))


def test_python_cli_and_mcp_publish_one_validated_catalog() -> None:
    service_publication = getattr(
        control_plane_module,
        "control_service_publication",
    )
    publications = (
        service_publication(),
        control_cli.cli_control_surface_publication(),
        native_tools.mcp_control_surface_publication(),
    )

    assert {
        publication.surface for publication in publications
    } == set(ControlSurface)
    expected_schema_ids = {
        descriptor.operation.value: descriptor.request_schema_id
        for descriptor in OPERATION_CATALOG_V2
    }
    expected_result_ids = {
        descriptor.operation.value: descriptor.result_schema_id
        for descriptor in OPERATION_CATALOG_V2
    }
    expected_behavior_ids = {
        descriptor.operation.value: control_operation_behavior_id(descriptor)
        for descriptor in OPERATION_CATALOG_V2
    }
    for publication in publications:
        assert validate_control_surface_publication(publication) is publication
        assert publication.catalog_id == OPERATION_CATALOG_V2.catalog_id
        assert publication.operations == OPERATION_CATALOG_V2.operations
        assert dict(publication.request_schema_ids) == expected_schema_ids
        assert dict(publication.result_schema_ids) == expected_result_ids
        assert dict(publication.behavior_ids) == expected_behavior_ids
        assert set(publication.dispatcher_ids.values()) == {
            DIRECT_CONTROL_SERVICE_DISPATCHER_ID
        }
        assert publication.dispatch_mode == "direct_service"
        assert publication.provider_free
        assert publication.process_free
        assert ControlSurfacePublication.from_dict(
            publication.to_record()
        ) == publication


def test_catalog_publication_fails_closed_on_population_and_semantic_drift() -> None:
    service_publication = getattr(
        control_plane_module,
        "control_service_publication",
    )
    canonical = service_publication()

    with pytest.raises(
        ControlCatalogConformanceError,
        match="operation population",
    ):
        validate_control_surface_publication(
            replace(canonical, operations=canonical.operations[:-1])
        )

    extra_operation = canonical.to_record()
    extra_operation.pop("content_id")
    extra_operation["operations"] = [
        *extra_operation["operations"],
        "execute_shell",
    ]
    with pytest.raises(
        ControlCatalogConformanceError,
        match="unknown operation",
    ):
        validate_control_surface_publication(extra_operation)

    request_schema_ids = dict(canonical.request_schema_ids)
    request_schema_ids[Operation.STATUS.value] = "sha256:request-drift"
    with pytest.raises(
        ControlCatalogConformanceError,
        match="request_schema_ids drift",
    ):
        validate_control_surface_publication(
            replace(canonical, request_schema_ids=request_schema_ids)
        )

    result_schema_ids = dict(canonical.result_schema_ids)
    result_schema_ids[Operation.STATUS.value] = "sha256:result-drift"
    with pytest.raises(
        ControlCatalogConformanceError,
        match="result_schema_ids drift",
    ):
        validate_control_surface_publication(
            replace(canonical, result_schema_ids=result_schema_ids)
        )

    behavior_ids = dict(canonical.behavior_ids)
    behavior_ids[Operation.STATUS.value] = "sha256:behavior-drift"
    with pytest.raises(
        ControlCatalogConformanceError,
        match="behavior_ids drift",
    ):
        validate_control_surface_publication(
            replace(canonical, behavior_ids=behavior_ids)
        )

    dispatcher_ids = dict(canonical.dispatcher_ids)
    dispatcher_ids[Operation.STATUS.value] = (
        "ipfs_accelerate_py.agent_supervisor.control_cli:run_agent_cli"
    )
    with pytest.raises(
        ControlCatalogConformanceError,
        match="dispatch directly",
    ):
        validate_control_surface_publication(
            replace(canonical, dispatcher_ids=dispatcher_ids)
        )

    with pytest.raises(
        ControlCatalogConformanceError,
        match="direct_service",
    ):
        validate_control_surface_publication(
            replace(canonical, dispatch_mode="cli_string")
        )
    with pytest.raises(
        ControlCatalogConformanceError,
        match="provider-free and process-free",
    ):
        validate_control_surface_publication(
            replace(canonical, provider_free=False)
        )

    missing_contract = canonical.to_record()
    missing_contract.pop("content_id")
    missing_contract.pop("contract_version")
    with pytest.raises(ControlCatalogConformanceError, match="missing fields"):
        validate_control_surface_publication(missing_contract)

    drifted_contract = canonical.to_record()
    drifted_contract.pop("content_id")
    drifted_contract["contract_version"] += 1
    with pytest.raises(ControlCatalogConformanceError, match="unsupported"):
        validate_control_surface_publication(drifted_contract)


@pytest.mark.parametrize(
    "schema_name",
    ("request_schema", "result_schema"),
)
def test_publication_rejects_catalog_schema_drift(
    schema_name: str,
) -> None:
    service_publication = getattr(
        control_plane_module,
        "control_service_publication",
    )
    canonical = service_publication()
    status = OPERATION_CATALOG_V2.operation(Operation.STATUS)
    schema = (
        operation_request_json_schema(Operation.STATUS)
        if schema_name == "request_schema"
        else operation_result_json_schema(Operation.STATUS)
    )
    schema["title"] = f"drifted {schema_name}"
    drifted_status = replace(status, **{schema_name: schema})
    drifted_catalog = OperationCatalog(
        tuple(
            drifted_status
            if descriptor.operation is Operation.STATUS
            else descriptor
            for descriptor in OPERATION_CATALOG_V2
        )
    )

    with pytest.raises(
        ControlCatalogConformanceError,
        match=rf"{schema_name.removesuffix('_schema')} schema drift",
    ):
        validate_control_surface_publication(
            canonical,
            catalog=drifted_catalog,
        )


def test_publication_rejects_catalog_behavior_drift() -> None:
    service_publication = getattr(
        control_plane_module,
        "control_service_publication",
    )
    canonical = service_publication()
    status = OPERATION_CATALOG_V2.operation(Operation.STATUS)
    drifted_status = replace(status, family="behaviorally-drifted-status")
    drifted_catalog = OperationCatalog(
        tuple(
            drifted_status
            if descriptor.operation is Operation.STATUS
            else descriptor
            for descriptor in OPERATION_CATALOG_V2
        )
    )

    with pytest.raises(
        ControlCatalogConformanceError,
        match="behavior drift",
    ):
        validate_control_surface_publication(
            canonical,
            catalog=drifted_catalog,
        )


def test_mcp_discovery_rejects_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = native_tools._tool_input_schema

    def drifted(operation: Operation) -> dict[str, Any]:
        schema = original(operation)
        if operation is Operation.STATUS:
            schema = json.loads(json.dumps(schema))
            schema["properties"]["request"]["title"] = "drifted request"
        return schema

    monkeypatch.setattr(native_tools, "_tool_input_schema", drifted)
    with pytest.raises(ControlContractError, match="request schema drift"):
        agent_supervisor_discovery_manifest()
