from __future__ import annotations

import builtins
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    COMMAND_OPERATIONS,
    agent_cli_discovery_manifest,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID,
    CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
    AuthorityViolationError,
    AuthorizationBindingError,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    ControlDiscoveryObservation,
    ControlDiscoverySafetyEvidence,
    ControlSurface,
    ControlSurfaceParityCase,
    ControlSurfaceParityEvidence,
    EffectKind,
    ExpectedEffect,
    IdempotencyKey,
    MissingIdempotencyError,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationResult,
    PathEscapeError,
    READ_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    CONTROL_REDACTION_MARKER,
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
    capture_control_discovery_runtime_state,
)
from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import (
    HierarchicalToolManager,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    AgentSupervisorMCPConfigurationError,
    agent_supervisor_discovery_manifest,
    agent_supervisor_service_resolution_count,
    configure_agent_supervisor_control,
    register_native_agent_supervisor_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] = []

    def register_tool(self, **definition: Any) -> None:
        self.tools.append(definition)


@pytest.fixture(autouse=True)
def _reset_control_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent_supervisor_control()
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST", raising=False
    )
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_STATE_ALLOWLIST", raising=False)
    yield
    configure_agent_supervisor_control()


def _binding(repo_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repo:fixture",
        "tree_id": "tree:abc",
        "objective_id": "ASI-G103",
        "objective_revision": "objective:1",
        "policy_id": "policy:control",
        "policy_revision": "policy:1",
        "caller": "mcp:test",
    }


def _request(
    repo_root: Path,
    state_root: Path,
    operation: Operation = Operation.STATUS,
) -> OperationRequest:
    return OperationRequest(
        operation=operation,
        **_binding(repo_root, state_root),
        bounds=ControlBounds(max_items=16, max_paths=16, max_effects=16),
    )


def _matrix_requests(
    repo_root: Path,
    state_root: Path,
) -> tuple[tuple[str, OperationRequest], ...]:
    binding = _binding(repo_root, state_root)
    effect = ExpectedEffect(
        effect_id="pause:fixture",
        kind=EffectKind.LIFECYCLE_TRANSITION,
        resource="supervisor:fixture",
        paths=("supervisor.json",),
        description="Pause the fixture supervisor",
    )
    proposal = OperationRequest(
        operation=Operation.PAUSE,
        **binding,
        parameters={
            "target_id": "supervisor:fixture",
            "reason": "parity validation",
            "requested_state": "paused",
        },
        expected_effects=(effect,),
        dry_run=True,
    )
    mutation = OperationRequest(
        operation=Operation.PAUSE,
        **binding,
        parameters={
            "target_id": "supervisor:fixture",
            "reason": "parity validation",
            "requested_state": "paused",
        },
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key="parity:pause:1",
            operation=Operation.PAUSE,
            caller=binding["caller"],
            repository_id=binding["repository_id"],
            objective_id=binding["objective_id"],
        ),
        authorization=AuthorizationDecision(
            verdict=AuthorizationVerdict.PERMIT,
            operation=Operation.PAUSE,
            granted_authority=OperationAuthority.MUTATION,
            **binding,
            lease_id="lease:parity",
            fencing_epoch=3,
            authorized_effect_ids=(effect.effect_id,),
            grant_ids=("grant:parity",),
            evaluated_at_ms=500,
            expires_at_ms=1_500,
        ),
        lease_id="lease:parity",
        fencing_epoch=3,
    )
    return (
        ("independent_read_success", _request(repo_root, state_root)),
        ("independent_proposal_success", proposal),
        (
            "independent_stable_failure",
            OperationRequest(
                operation=Operation.HEALTH,
                **binding,
                parameters={"health_path": "missing-health.json"},
            ),
        ),
        ("independent_mutation_success", mutation),
    )


def _guard_rejection_payload(
    request: OperationRequest, scenario: str
) -> dict[str, Any]:
    payload = request.to_record()
    payload.pop("content_id")
    if scenario == "unauthorized":
        payload.pop("authorization")
    elif scenario == "unscoped_idempotency":
        idempotency = dict(payload["idempotency"])
        idempotency.pop("content_id")
        idempotency["objective_id"] = "objective:outside-request-scope"
        payload["idempotency"] = idempotency
    elif scenario == "unfenced":
        payload.pop("lease_id")
        payload.pop("fencing_epoch")
    elif scenario == "stale_binding":
        payload["tree_id"] = f"{request.tree_id}:stale-request-binding"
    elif scenario == "path_escape":
        parameters = dict(payload["parameters"])
        parameters["target_path"] = "../outside-repository"
        payload["parameters"] = parameters
    elif scenario == "undeclared_effect":
        payload["expected_effects"] = []
    else:
        raise AssertionError(f"unknown guard scenario {scenario}")
    return payload


def _service(
    repo_root: Path,
    state_root: Path,
    *,
    mutation_calls: list[str] | None = None,
) -> SupervisorControlService:
    def operation_handler(request: OperationRequest) -> BackendResponse:
        if (
            mutation_calls is not None
            and request.operation.mutating
            and not request.dry_run
        ):
            mutation_calls.append(request.request_id)
        return BackendResponse(
            data={
                "state": "healthy",
                "phase": "idle",
                "operation": request.operation.value,
            },
            changed=bool(request.expected_effects),
            applied_effect_ids=tuple(
                effect.effect_id for effect in request.expected_effects
            ),
        )

    return SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={
            operation: operation_handler
            for operation in Operation
            if operation not in READ_OPERATIONS
        }
        | {Operation.STATUS: operation_handler},
        state_store=InMemoryControlStateStore(),
        lease_validator=lambda _request: True,
        clock_ms=lambda: 1_000,
    )


def test_registration_covers_every_operation_with_shared_schema() -> None:
    manager = _DummyManager()
    register_native_agent_supervisor_tools(manager)

    assert {item["name"] for item in manager.tools} == {
        operation.value for operation in Operation
    }
    assert len(manager.tools) == len(Operation)
    manifest = agent_supervisor_discovery_manifest()
    for definition in manager.tools:
        operation = Operation(definition["name"])
        assert definition["category"] == "agent_supervisor"
        assert definition["runtime"] == "fastapi"
        request_schema = definition["input_schema"]["properties"]["request"]
        result_schema = definition["input_schema"]["x-output-schema"]
        contract = definition["input_schema"]["x-agent-supervisor-contract"]
        assert request_schema["properties"]["operation"]["const"] == operation.value
        assert result_schema["properties"]["operation"]["const"] == operation.value
        assert contract == {
            "surface": ControlSurface.MCP.value,
            "operation": operation.value,
            "request_schema_id": manifest.request_schema_ids[operation.value],
            "result_schema_id": manifest.result_schema_ids[operation.value],
        }
        assert "request" in definition["input_schema"]["required"]
        assert operation.authority.value in definition["tags"]
        assert {"bounded", "policy-controlled", "redacted"}.issubset(
            definition["tags"]
        )
        if operation.mutating:
            assert {
                "authorization-required",
                "audit-receipt",
                "dry-run",
                "idempotent",
                "lease-fenced",
            }.issubset(definition["tags"])

    with pytest.raises(TypeError):
        AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS] = lambda: None  # type: ignore[index]


def test_discovery_and_registration_do_not_resolve_a_service() -> None:
    calls = 0

    def forbidden_factory(_request: OperationRequest) -> SupervisorControlService:
        nonlocal calls
        calls += 1
        raise AssertionError("discovery resolved the control service")

    configure_agent_supervisor_control(service_factory=forbidden_factory)
    manager = HierarchicalToolManager()
    manager.register_category_loader(
        "agent_supervisor",
        lambda value: register_native_agent_supervisor_tools(value),
    )

    assert "agent_supervisor" in manager.list_categories()
    assert calls == 0
    assert len(manager.list_tools("agent_supervisor")) == len(Operation)
    assert calls == 0
    assert len(AGENT_SUPERVISOR_OPERATION_TOOLS) == len(Operation)


def test_discovery_safety_evidence_uses_observed_python_cli_and_mcp_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    factory_calls = 0
    provider_loads = 0
    process_starts = 0
    original_import = builtins.__import__

    def forbidden_factory(_request: OperationRequest) -> SupervisorControlService:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("discovery resolved the MCP service")

    def observed_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        nonlocal provider_loads
        if name == "ipfs_datasets_py" or name.startswith(
            (
                "ipfs_datasets_py.",
                "ipfs_accelerate_py.agent_supervisor.ipfs_datasets_",
                "ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider",
                "ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider",
            )
        ):
            provider_loads += 1
        return original_import(name, globals, locals, fromlist, level)

    def forbidden_process(*_args: Any, **_kwargs: Any) -> None:
        nonlocal process_starts
        process_starts += 1
        raise AssertionError("discovery started a process")

    monkeypatch.setattr(builtins, "__import__", observed_import)
    monkeypatch.setattr(subprocess, "Popen", forbidden_process)
    configure_agent_supervisor_control(service_factory=forbidden_factory)
    observations: list[ControlDiscoveryObservation] = []

    for surface, discover in (
        (ControlSurface.PYTHON, service.discovery_manifest),
        (ControlSurface.CLI, agent_cli_discovery_manifest),
    ):
        before = capture_control_discovery_runtime_state(
            optional_provider_load_count=provider_loads,
            process_start_count=process_starts,
        )
        first = discover()
        second = discover()
        after = capture_control_discovery_runtime_state(
            optional_provider_load_count=provider_loads,
            process_start_count=process_starts,
        )
        observations.append(
            ControlDiscoveryObservation(
                surface=surface,
                first_manifest=first,
                second_manifest=second,
                before=before,
                after=after,
            )
        )

    resolution_before = agent_supervisor_service_resolution_count()
    before = capture_control_discovery_runtime_state(
        service_resolution_count=resolution_before,
        optional_provider_load_count=provider_loads,
        process_start_count=process_starts,
    )
    first = agent_supervisor_discovery_manifest()
    manager = HierarchicalToolManager()
    manager.register_category_loader(
        "agent_supervisor",
        lambda value: register_native_agent_supervisor_tools(value),
    )
    assert len(manager.list_tools("agent_supervisor")) == len(Operation)
    second = agent_supervisor_discovery_manifest()
    resolution_after = agent_supervisor_service_resolution_count()
    after = capture_control_discovery_runtime_state(
        service_resolution_count=resolution_after,
        optional_provider_load_count=provider_loads,
        process_start_count=process_starts,
    )
    observations.append(
        ControlDiscoveryObservation(
            surface=ControlSurface.MCP,
            first_manifest=first,
            second_manifest=second,
            before=before,
            after=after,
        )
    )
    evidence = ControlDiscoverySafetyEvidence(
        repository_tree="tree:abc",
        objective_id="ASI-G105",
        policy_id="policy:control",
        policy_revision="policy:1",
        capability_report=service.capability_report(),
        observations=tuple(observations),
    )

    assert factory_calls == 0
    assert provider_loads == process_starts == 0
    assert resolution_before == resolution_after
    assert evidence.proved_requirement_ids == (
        CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID,
    )
    population_ids = {
        observation.manifest.schema_population_id
        for observation in evidence.observations
    }
    assert len(population_ids) == 1
    for observation in evidence.observations:
        assert (
            observation.first_manifest.canonical_bytes()
            == observation.second_manifest.canonical_bytes()
        )
        assert observation.manifest.operations == tuple(
            sorted(Operation, key=lambda item: item.value)
        )
    assert ControlDiscoverySafetyEvidence.from_dict(
        evidence.to_record()
    ) == evidence


@pytest.mark.asyncio
async def test_mcp_result_is_exactly_the_python_service_record(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    configure_agent_supervisor_control(service=service)
    request = _request(repo_root, state_root)

    expected = service.execute(request).to_record()
    resolutions_before = agent_supervisor_service_resolution_count()
    actual = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS](
        request=request.to_record()
    )

    assert actual == expected
    assert (
        agent_supervisor_service_resolution_count()
        == resolutions_before + 1
    )
    assert OperationResult.from_dict(actual).result_id == expected["content_id"]


@pytest.mark.asyncio
async def test_hierarchical_dispatch_uses_direct_control_service(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    configure_agent_supervisor_control(service=service)
    request = _request(repo_root, state_root)
    manager = HierarchicalToolManager()
    manager.register_category_loader(
        "agent_supervisor",
        lambda value: register_native_agent_supervisor_tools(value),
    )

    record = await manager.dispatch(
        "agent_supervisor",
        "status",
        {"request": request.to_record()},
    )

    assert record == service.execute(request).to_record()


@pytest.mark.asyncio
async def test_unconfigured_mcp_adapter_fails_closed_without_request_roots(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _request(repo_root, state_root)

    with pytest.raises(
        AgentSupervisorMCPConfigurationError,
        match="server-configured repository and state allowlists",
    ):
        await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS](
            request=request.to_record()
        )


@pytest.mark.asyncio
async def test_named_tool_rejects_a_different_request_operation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    configure_agent_supervisor_control(service=_service(repo_root, state_root))
    request = _request(repo_root, state_root, Operation.HEALTH)
    resolutions_before = agent_supervisor_service_resolution_count()

    with pytest.raises(ValueError, match="does not match"):
        await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS](
            request=request.to_record()
        )
    assert agent_supervisor_service_resolution_count() == resolutions_before


@pytest.mark.asyncio
async def test_every_mcp_operation_has_exact_python_execution_parity(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    configure_agent_supervisor_control(service=service)

    for operation in sorted(Operation, key=lambda item: item.value):
        if operation.mutating:
            effect = ExpectedEffect(
                effect_id=f"{operation.value}:parity",
                kind=EffectKind.LIFECYCLE_TRANSITION,
                resource="supervisor:fixture",
                paths=("supervisor.json",),
                description=f"Preview {operation.value}",
            )
            request = OperationRequest(
                operation=operation,
                **_binding(repo_root, state_root),
                parameters={
                    "target_id": "supervisor:fixture",
                    "reason": "exhaustive MCP parity validation",
                },
                expected_effects=(effect,),
                dry_run=True,
            )
        else:
            request = _request(repo_root, state_root, operation)

        python_record = service.execute(request).to_record()
        mcp_record = await AGENT_SUPERVISOR_OPERATION_TOOLS[operation](
            request=request.to_record()
        )

        assert mcp_record == python_record, operation.value
        assert OperationResult.from_dict(mcp_record).operation is operation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "error_type"),
    (
        ("unauthorized", AuthorizationBindingError),
        ("unscoped_idempotency", MissingIdempotencyError),
        ("unfenced", AuthorizationBindingError),
        ("stale_binding", AuthorizationBindingError),
        ("path_escape", PathEscapeError),
        ("undeclared_effect", AuthorityViolationError),
    ),
)
async def test_mcp_rejects_every_unsafe_real_mutation_before_service_resolution(
    tmp_path: Path,
    scenario: str,
    error_type: type[Exception],
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _matrix_requests(repo_root, state_root)[-1][1]
    payload = _guard_rejection_payload(request, scenario)
    factory_calls = 0

    def forbidden_factory(_request: OperationRequest) -> SupervisorControlService:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("malformed mutation resolved a control service")

    configure_agent_supervisor_control(service_factory=forbidden_factory)
    resolutions_before = agent_supervisor_service_resolution_count()

    with pytest.raises(error_type):
        await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.PAUSE](
            request=payload
        )

    assert agent_supervisor_service_resolution_count() == resolutions_before
    assert factory_calls == 0


@pytest.mark.asyncio
async def test_python_cli_mcp_matrix_emits_typed_parity_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    mutation_calls: list[str] = []
    service = _service(
        repo_root,
        state_root,
        mutation_calls=mutation_calls,
    )
    configure_agent_supervisor_control(service=service)
    cases = []
    exit_codes = []
    for scenario, request in _matrix_requests(repo_root, state_root):
        python_record = service.execute(request).to_record()
        command = next(
            name
            for name, operation in COMMAND_OPERATIONS.items()
            if operation is request.operation
        )
        exit_codes.append(
            cli.main(
                [
                    "agent",
                    command,
                    "--request-json",
                    request.to_json(),
                    "--output-json",
                ],
                agent_control_service=service,
            )
        )
        captured = capsys.readouterr()
        assert captured.err == ""
        cli_record = json.loads(captured.out)
        mcp_record = await AGENT_SUPERVISOR_OPERATION_TOOLS[
            request.operation
        ](request=request.to_record())
        cases.append(
            ControlSurfaceParityCase(
                scenario=scenario,
                request=request,
                python_result=python_record,
                cli_result=cli_record,
                mcp_result=mcp_record,
            )
        )
    request = cases[0].request
    assert isinstance(request, OperationRequest)
    evidence = ControlSurfaceParityEvidence(
        repository_tree=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        policy_revision=request.policy_revision,
        capability_report=service.capability_report(),
        cases=tuple(cases),
    )

    assert exit_codes == [0, 0, 4, 0]
    mutation_request = next(
        case.request
        for case in cases
        if case.scenario == "independent_mutation_success"
    )
    assert mutation_calls == [mutation_request.request_id]
    assert evidence.proved_requirement_ids == (
        CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
    )
    mcp_manifest = agent_supervisor_discovery_manifest()
    assert dict(evidence.request_schema_ids) == dict(
        mcp_manifest.request_schema_ids
    )
    assert dict(evidence.result_schema_ids) == dict(
        mcp_manifest.result_schema_ids
    )
    assert evidence.schema_population_id == mcp_manifest.schema_population_id
    # The independently invoked matrix is the operational witness for G103,
    # not authority to complete its own objective.  Even with producer tasks
    # complete, the goal remains provisional until independent criterion
    # validations, exact coverage, analyzer health, and exhaustive quorum are
    # supplied to the completion gate.
    assert evidence.completion_authoritative is False
    no_independent_completion_proof = (
        evidence.evaluate_objective_completion(tasks_complete=True)
    )
    assert (
        no_independent_completion_proof.state.value
        == "provisionally_complete"
    )
    assert not no_independent_completion_proof.verified
    assert (
        no_independent_completion_proof.gate is not None
        and not no_independent_completion_proof.gate.passed
    )


@pytest.mark.asyncio
async def test_mcp_read_pagination_and_allowlists_are_enforced(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    other_root = tmp_path / "other"
    repo_root.mkdir()
    state_root.mkdir()
    other_root.mkdir()
    events = state_root / "events.jsonl"
    events.write_text(
        "".join(
            json.dumps({"event_id": f"event:{index}"}) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        state_store=InMemoryControlStateStore(),
        max_query_items=2,
        clock_ms=lambda: 1_000,
    )
    configure_agent_supervisor_control(service=service)

    request = OperationRequest(
        operation=Operation.EVENTS,
        **_binding(repo_root, state_root),
        parameters={"events_path": "events.jsonl", "limit": 1, "offset": 1},
        bounds=ControlBounds(max_items=16, max_paths=16, max_effects=16),
    )
    page = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.EVENTS](
        request=request.to_record()
    )

    assert page["status"] == "succeeded"
    assert page["data"] == {
        "count": 1,
        "items": [{"event_id": "event:1"}],
        "limit": 1,
        "offset": 1,
        "truncated": True,
    }

    oversized = replace(
        request,
        parameters={"events_path": "events.jsonl", "limit": 3},
    )
    oversized_result = await AGENT_SUPERVISOR_OPERATION_TOOLS[
        Operation.EVENTS
    ](request=oversized.to_record())
    assert oversized_result["status"] == "failed"
    assert oversized_result["error"]["code"] == "bounds_exceeded"

    outside = OperationRequest(
        operation=Operation.STATUS,
        **_binding(other_root, state_root),
        parameters={"status_path": "status.json"},
    )
    outside_result = await AGENT_SUPERVISOR_OPERATION_TOOLS[
        Operation.STATUS
    ](request=outside.to_record())
    assert outside_result["status"] == "denied"
    assert outside_result["error"]["code"] == "forbidden"


@pytest.mark.asyncio
async def test_mcp_dry_run_is_proposal_only_and_skips_backend_and_lease(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    proposal = _matrix_requests(repo_root, state_root)[1][1]
    backend_calls = 0
    lease_calls = 0

    def forbidden_backend(_request: OperationRequest) -> BackendResponse:
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("dry-run invoked the mutation backend")

    def forbidden_lease(_request: OperationRequest) -> bool:
        nonlocal lease_calls
        lease_calls += 1
        raise AssertionError("dry-run invoked live lease validation")

    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={Operation.PAUSE: forbidden_backend},
        state_store=InMemoryControlStateStore(),
        lease_validator=forbidden_lease,
        clock_ms=lambda: 1_000,
    )
    configure_agent_supervisor_control(service=service)

    record = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.PAUSE](
        request=proposal.to_record()
    )

    assert record["status"] == "succeeded"
    assert record["authority"] == "proposal"
    assert record["data"] == {"dry_run": True, "would_change": True}
    assert record["effects"] == []
    assert record["preview"]["would_change"] is True
    assert record["audit_receipt_id"]
    assert backend_calls == lease_calls == 0


@pytest.mark.asyncio
async def test_mcp_mutation_is_fenced_idempotent_and_audited(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    mutation = _matrix_requests(repo_root, state_root)[-1][1]
    backend_calls = 0
    lease_calls = 0

    def backend(request: OperationRequest) -> BackendResponse:
        nonlocal backend_calls
        backend_calls += 1
        return BackendResponse(
            data={"state": "paused"},
            changed=True,
            applied_effect_ids=tuple(
                effect.effect_id for effect in request.expected_effects
            ),
        )

    def lease(_request: OperationRequest) -> bool:
        nonlocal lease_calls
        lease_calls += 1
        return True

    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={Operation.PAUSE: backend},
        state_store=InMemoryControlStateStore(),
        lease_validator=lease,
        clock_ms=lambda: 1_000,
    )
    configure_agent_supervisor_control(service=service)
    tool = AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.PAUSE]

    first = await tool(request=mutation.to_record())
    replay = await tool(request=mutation.to_record())

    assert replay == first
    assert backend_calls == lease_calls == 1
    assert first["status"] == "succeeded"
    assert first["audit_receipt_id"]
    assert len(first["effects"]) == 1
    assert first["effects"][0]["applied"] is True
    assert first["effects"][0]["receipt_id"] == first["audit_receipt_id"]

    receipt_request = OperationRequest(
        operation=Operation.RECEIPTS,
        **_binding(repo_root, state_root),
        parameters={"limit": 10},
    )
    receipts = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.RECEIPTS](
        request=receipt_request.to_record()
    )
    assert receipts["status"] == "succeeded"
    assert receipts["data"]["count"] == 1
    assert receipts["data"]["items"][0]["operation"] == "pause"
    assert (
        receipts["data"]["items"][0]["receipt_id"]
        == first["audit_receipt_id"]
    )

    conflict = replace(
        mutation,
        parameters={
            **dict(mutation.parameters),
            "reason": "conflicting reuse of the same idempotency key",
        },
    )
    conflict_result = await tool(request=conflict.to_record())
    assert conflict_result["status"] == "conflict"
    assert conflict_result["error"]["code"] == "idempotency_conflict"
    assert backend_calls == lease_calls == 1


@pytest.mark.asyncio
async def test_shared_redaction_preserves_python_cli_mcp_result_parity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    secret = "super-secret-control-token"

    def handler(_request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            data={
                "api-key": secret,
                "nested": {
                    "password": secret,
                    "safe": "visible",
                    "message": f"authorization=Bearer-{secret}",
                },
                "tokens": [{"refresh_token": secret}],
            },
        )

    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={Operation.STATUS: handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )
    configure_agent_supervisor_control(service=service)
    request = _request(repo_root, state_root)

    python_record = service.execute(request).to_record()
    exit_code = cli.main(
        [
            "agent",
            "status",
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    cli_record = json.loads(captured.out)
    mcp_record = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS](
        request=request.to_record()
    )

    assert exit_code == 0
    assert captured.err == ""
    assert python_record == cli_record == mcp_record
    assert secret not in json.dumps(mcp_record, sort_keys=True)
    assert mcp_record["data"]["api-key"] == CONTROL_REDACTION_MARKER
    assert mcp_record["data"]["nested"] == {
        "message": f"authorization={CONTROL_REDACTION_MARKER}",
        "password": CONTROL_REDACTION_MARKER,
        "safe": "visible",
    }
    assert mcp_record["data"]["tokens"] == [
        {"refresh_token": CONTROL_REDACTION_MARKER}
    ]
