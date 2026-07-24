from __future__ import annotations

import builtins
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control_cli import (
    COMMAND_OPERATIONS,
    agent_cli_discovery_manifest,
)
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID,
    CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
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
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationResult,
    READ_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
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


def _service(repo_root: Path, state_root: Path) -> SupervisorControlService:
    def operation_handler(request: OperationRequest) -> BackendResponse:
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
    for definition in manager.tools:
        operation = Operation(definition["name"])
        assert definition["category"] == "agent_supervisor"
        assert definition["runtime"] == "fastapi"
        request_schema = definition["input_schema"]["properties"]["request"]
        result_schema = definition["input_schema"]["x-output-schema"]
        assert request_schema["properties"]["operation"]["const"] == operation.value
        assert result_schema["properties"]["operation"]["const"] == operation.value
        assert "request" in definition["input_schema"]["required"]
        assert operation.authority.value in definition["tags"]


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
                "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
                "ipfs_accelerate_py.agent_supervisor.formal_verification_provider",
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
    actual = await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS](
        request=request.to_record()
    )

    assert actual == expected
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

    with pytest.raises(ValueError, match="does not match"):
        await AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.STATUS](
            request=request.to_record()
        )


@pytest.mark.asyncio
async def test_python_cli_mcp_matrix_emits_typed_parity_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
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
    assert evidence.proved_requirement_ids == (
        CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
    )
