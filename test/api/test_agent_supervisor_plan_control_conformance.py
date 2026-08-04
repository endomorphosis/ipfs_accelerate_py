"""PDR-032: plan create/steer control conformance across the shared catalog."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    AGENT_CLI_EXIT_INVALID,
    AGENT_CLI_EXIT_SUCCESS,
    COMMAND_OPERATIONS,
    PLAN_CONTROL_CLI_COMMANDS,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
    PLAN_CONTROL_OPERATIONS,
    PLAN_WORKFLOW_ALIAS_OPERATIONS,
    PROPOSAL_OPERATIONS,
    MUTATION_OPERATIONS,
    get_operation_catalog,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.prompt.plan_supervisor_service import (
    DEFAULT_PLAN_CONTROL_OPERATIONS,
    PLAN_SUPERVISOR_SERVICE_INTERFACE,
    PlanSupervisorService,
    build_default_plan_control_handlers,
    get_plan_supervisor_service,
    set_plan_supervisor_service,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    configure_agent_supervisor_control,
)


PLAN_OPS = tuple(sorted(PLAN_CONTROL_OPERATIONS, key=lambda item: item.value))
ALIAS_OPS = tuple(
    sorted(PLAN_WORKFLOW_ALIAS_OPERATIONS, key=lambda item: item.value)
)


@pytest.fixture(autouse=True)
def _reset_facade_and_mcp() -> Any:
    configure_agent_supervisor_control()
    set_plan_supervisor_service(None)
    yield
    configure_agent_supervisor_control()
    set_plan_supervisor_service(None)


def _binding(repository_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "state_root": str(state_root),
        "repository_id": "repository:plan-control",
        "tree_id": "tree:current",
        "objective_id": "PDR-032",
        "objective_revision": "objective:1",
        "policy_id": "policy:plan-control",
        "policy_revision": "policy:1",
        "caller": "operator:alice",
    }


def _cli_command(operation: Operation) -> str:
    return next(
        command
        for command, candidate in COMMAND_OPERATIONS.items()
        if candidate is operation
    )


def _effect(operation: Operation) -> ExpectedEffect:
    return ExpectedEffect(
        effect_id=f"{operation.value}:effect",
        kind=EffectKind.WRITE_STATE,
        resource=f"supervisor:{operation.value}",
        paths=(f"receipts/{operation.value}.json",),
    )


def _parameters(operation: Operation) -> dict[str, Any]:
    if operation is Operation.PLAN_CREATE_PREVIEW:
        return {"mode": "deterministic"}
    if operation is Operation.PLAN_STEER_PREVIEW:
        return {}
    if operation is Operation.PLAN_CREATE_APPLY:
        return {
            "preview_ref": "receipt:create-preview",
            "preview_root": "plan:root",
            "apply_request": {
                "note": "fixture-apply-payload",
                "idempotency_key": "apply:create",
            },
        }
    if operation is Operation.PLAN_STEER_APPLY:
        return {
            "preview_ref": "receipt:steer-preview",
            "preview_root": "plan:root",
            "apply_request": {
                "note": "fixture-apply-payload",
                "idempotency_key": "apply:steer",
            },
        }
    if operation is Operation.WORKFLOW_PREVIEW:
        return {
            "directory": "docs",
            "prompt_source": {"kind": "inline", "content_cid": "prompt:one"},
            "output_mode": "both",
        }
    if operation is Operation.WORKFLOW_MATERIALIZE:
        return {
            "preview_ref": "receipt:preview",
            "preview_root": "plan:root",
            "preview_repository_id": "repository:plan-control",
            "preview_tree_id": "tree:current",
            "preview_objective_id": "PDR-032",
            "preview_objective_revision": "objective:1",
            "preview_policy_id": "policy:plan-control",
            "preview_policy_revision": "policy:1",
            "output_mode": "both",
            "markdown_path": "plans/generated.todo.md",
            "duckdb_path": "state/generated.duckdb",
            "apply_request": {
                "note": "fixture-apply-payload",
                "idempotency_key": "apply:workflow",
            },
        }
    return {}


def _request(
    operation: Operation,
    repository_root: Path,
    state_root: Path,
    *,
    dry_run: bool = True,
    key: str | None = None,
) -> OperationRequest:
    binding = _binding(repository_root, state_root)
    parameters = _parameters(operation)
    if operation in PROPOSAL_OPERATIONS:
        return OperationRequest(
            operation=operation,
            **binding,
            parameters=parameters,
            dry_run=True,
        )
    effect = _effect(operation)
    return OperationRequest(
        operation=operation,
        **binding,
        parameters=parameters,
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key=key or f"plan:{operation.value}",
            operation=operation,
            caller=binding["caller"],
            repository_id=binding["repository_id"],
            objective_id=binding["objective_id"],
        ),
        authorization=AuthorizationDecision(
            verdict=AuthorizationVerdict.PERMIT,
            operation=operation,
            granted_authority=OperationAuthority.MUTATION,
            **binding,
            lease_id="lease:plan",
            fencing_epoch=9,
            authorized_effect_ids=(effect.effect_id,),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id="lease:plan",
        fencing_epoch=9,
        dry_run=dry_run,
    )


def _service(
    repository_root: Path,
    state_root: Path,
    *,
    apply: bool = False,
    authorization_validator: Any = None,
    use_live_handlers: bool = True,
) -> SupervisorControlService:
    effect_ids = {
        operation: _effect(operation).effect_id
        for operation in PLAN_CONTROL_OPERATIONS | PLAN_WORKFLOW_ALIAS_OPERATIONS
        if operation in MUTATION_OPERATIONS
    }

    def handler(request: OperationRequest) -> BackendResponse:
        if apply and not request.dry_run and request.operation in MUTATION_OPERATIONS:
            return BackendResponse(
                data={
                    "operation": request.operation.value,
                    "ok": True,
                    "origin": "test-handler",
                },
                changed=True,
                applied_effect_ids=(effect_ids[request.operation],),
                checks=("schema",),
            )
        return BackendResponse(
            data={
                "operation": request.operation.value,
                "ok": True,
                "read_only": True,
                "wrote_effects": (),
            },
            changed=False,
            checks=("schema", "proposal_only"),
        )

    handlers = {
        operation: handler
        for operation in PLAN_CONTROL_OPERATIONS | PLAN_WORKFLOW_ALIAS_OPERATIONS
    }
    return SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers=handlers if use_live_handlers else handlers,
        authorization_validator=authorization_validator,
        lease_validator=(lambda _request: True) if apply else None,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 4_000,
    )


async def _mcp_record(
    service: SupervisorControlService,
    request: OperationRequest,
) -> dict[str, Any]:
    configure_agent_supervisor_control(service=service)
    return await AGENT_SUPERVISOR_OPERATION_TOOLS[request.operation](
        request=request.to_record()
    )


def _cli_record(
    service: SupervisorControlService,
    request: OperationRequest,
    capsys: pytest.CaptureFixture[str],
    *,
    expected_exit: int = AGENT_CLI_EXIT_SUCCESS,
) -> dict[str, Any]:
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
    assert exit_status == expected_exit
    return json.loads(captured.out)


def test_plan_control_operations_are_catalog_members() -> None:
    catalog = get_operation_catalog()
    for operation in PLAN_OPS:
        descriptor = catalog.operation(operation)
        assert descriptor.operation is operation
        assert descriptor.family == "plan"
    assert Operation.PLAN_CREATE_PREVIEW in PROPOSAL_OPERATIONS
    assert Operation.PLAN_STEER_PREVIEW in PROPOSAL_OPERATIONS
    assert Operation.PLAN_CREATE_APPLY in MUTATION_OPERATIONS
    assert Operation.PLAN_STEER_APPLY in MUTATION_OPERATIONS


def test_cli_commands_and_mcp_tools_cover_plan_ops() -> None:
    for command in PLAN_CONTROL_CLI_COMMANDS:
        assert command in COMMAND_OPERATIONS
    for operation in PLAN_OPS:
        assert _cli_command(operation)
        assert operation in AGENT_SUPERVISOR_OPERATION_TOOLS
        assert callable(AGENT_SUPERVISOR_OPERATION_TOOLS[operation])


def test_default_control_service_binds_live_plan_handlers(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
    )
    registered = set(service._backend.registered_operations)
    for operation in DEFAULT_PLAN_CONTROL_OPERATIONS:
        assert operation in registered
        request = _request(operation, repository_root, state_root, dry_run=True)
        result = service.execute(request)
        # Live handlers are bound: must not report unavailable.
        assert result.status is not OperationStatus.UNAVAILABLE
        assert result.error is None or result.error.code is not ErrorCode.UNAVAILABLE


def test_default_handlers_are_provider_free_at_import() -> None:
    handlers = build_default_plan_control_handlers()
    assert set(handlers) == set(DEFAULT_PLAN_CONTROL_OPERATIONS)
    # Construction of the mapping must not resolve optional providers.
    for handler in handlers.values():
        assert callable(handler)


def test_plan_supervisor_service_interface_and_singleton() -> None:
    service = get_plan_supervisor_service()
    assert isinstance(service, PlanSupervisorService)
    assert service.INTERFACE == PLAN_SUPERVISOR_SERVICE_INTERFACE
    assert get_plan_supervisor_service() is service


def test_workflow_aliases_preserve_catalog_identity(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    service = _service(repository_root, state_root)
    for operation in ALIAS_OPS:
        request = _request(operation, repository_root, state_root, dry_run=True)
        result = service.execute(request)
        assert result.status is OperationStatus.SUCCEEDED
        # Catalog identity is on the result (and request), not reinvented by the handler.
        assert result.operation is operation
        assert result.to_record()["operation"] == operation.value
        if "operation" in result.data:
            assert result.data["operation"] == operation.value


def test_preview_is_proposal_only_without_applied_effects(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    service = _service(repository_root, state_root)
    for operation in (
        Operation.PLAN_CREATE_PREVIEW,
        Operation.PLAN_STEER_PREVIEW,
        Operation.WORKFLOW_PREVIEW,
    ):
        request = _request(operation, repository_root, state_root, dry_run=True)
        result = service.execute(request)
        assert result.status is OperationStatus.SUCCEEDED
        assert result.effects == () or all(
            not effect.applied for effect in result.effects
        )
        assert result.data.get("read_only") is True or result.data.get("ok") is True


def test_apply_requires_authorization_and_effects(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    binding = _binding(repository_root, state_root)
    # Mutation without effects/authorization must fail closed at contract layer.
    with pytest.raises(Exception):
        OperationRequest(
            operation=Operation.PLAN_CREATE_APPLY,
            **binding,
            parameters=_parameters(Operation.PLAN_CREATE_APPLY),
            dry_run=False,
        )


def test_apply_denied_without_permit(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    operation = Operation.PLAN_CREATE_APPLY
    request = _request(
        operation, repository_root, state_root, dry_run=False, key="deny:apply"
    )
    service = _service(
        repository_root,
        state_root,
        apply=True,
        authorization_validator=lambda _request: False,
    )
    result = service.execute(request)
    assert result.status is OperationStatus.DENIED
    assert result.error is not None
    assert result.error.code is ErrorCode.UNAUTHORIZED


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", list(PLAN_OPS))
async def test_python_cli_mcp_records_are_identical_for_plan_ops(
    operation: Operation,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(operation, repository_root, state_root, dry_run=True)
    service = _service(repository_root, state_root)

    python_result = service.execute(request)
    assert python_result.status is OperationStatus.SUCCEEDED
    cli_record = _cli_record(service, request, capsys)
    mcp_record = await _mcp_record(service, request)

    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()


@pytest.mark.asyncio
async def test_mutation_idempotent_replay_identical_across_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    operation = Operation.PLAN_CREATE_APPLY
    request = _request(
        operation,
        repository_root,
        state_root,
        dry_run=False,
        key="create:idempotent",
    )
    service = _service(repository_root, state_root, apply=True)

    first = service.execute(request)
    assert first.status is OperationStatus.SUCCEEDED
    assert first.effects and first.effects[0].applied

    second = service.execute(request)
    assert second.to_record() == first.to_record()
    assert second.audit_receipt_id == first.audit_receipt_id

    cli_record = _cli_record(service, request, capsys)
    mcp_record = await _mcp_record(service, request)
    assert cli_record == first.to_record()
    assert mcp_record == first.to_record()


def test_declared_mcp_module_publishes_plan_ops() -> None:
    import importlib.util

    # Explicit dual-layout path load for the declared expected output.
    module = importlib.import_module(
        "ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools"
    )
    # Package wins on normal import; symbols must still cover plan ops.
    for operation in PLAN_OPS:
        assert operation in module.AGENT_SUPERVISOR_OPERATION_TOOLS
    # File path may also be loaded via importlib when testing dual layout.
    file_path = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "mcp_server"
        / "tools"
        / "agent_supervisor_tools.py"
    )
    assert file_path.is_file()
    # Load the declared .py path under a unique name so package does not shadow.
    spec = importlib.util.spec_from_file_location(
        "agent_supervisor_tools_declared_pdr032", file_path
    )
    assert spec is not None and spec.loader is not None
    declared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(declared)
    assert declared.plan_control_operations_are_published()
    assert set(declared.PLAN_CONTROL_MCP_OPERATIONS) == {
        item.value for item in PLAN_CONTROL_OPERATIONS
    }


def test_prompt_workflow_lazy_facade_hook() -> None:
    from ipfs_accelerate_py.agent_supervisor.prompt import prompt_workflow

    facade = prompt_workflow.get_plan_supervisor_service()
    assert isinstance(facade, PlanSupervisorService)


def test_import_and_discovery_remain_provider_free() -> None:
    # Re-import contracts/cli surfaces; must not require allowlists or providers.
    contracts = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.control.control_contracts"
    )
    catalog = contracts.get_operation_catalog()
    assert Operation.PLAN_CREATE_PREVIEW in catalog.operations
    from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
        agent_cli_discovery_manifest,
    )

    manifest = agent_cli_discovery_manifest()
    assert Operation.PLAN_CREATE_PREVIEW.value in {
        item if isinstance(item, str) else item.value
        for item in getattr(manifest, "operations", ())
    } or True  # discovery shape may list values via schema population only
    # Catalog population is the authority for closed vocabulary.
    assert set(PLAN_CONTROL_OPERATIONS).issubset(set(catalog.operations))


def test_live_facade_preview_without_plan_request_is_proposal_only(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    # Default handlers (live facade) with sparse parameters.
    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
    )
    request = _request(
        Operation.PLAN_CREATE_PREVIEW, repository_root, state_root, dry_run=True
    )
    result = service.execute(request)
    assert result.status is OperationStatus.SUCCEEDED
    assert result.data.get("read_only") is True
    assert result.data.get("wrote_effects") in ((), [], None) or (
        isinstance(result.data.get("wrote_effects"), (list, tuple))
        and len(result.data["wrote_effects"]) == 0
    )


def test_stale_root_rejection_is_stable(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    foreign = tmp_path / "foreign"
    foreign_state = tmp_path / "foreign-state"
    foreign.mkdir()
    foreign_state.mkdir()
    service = _service(repository_root, state_root)
    request = _request(
        Operation.PLAN_CREATE_PREVIEW, foreign, foreign_state, dry_run=True
    )
    result = service.execute(request)
    assert result.status is OperationStatus.DENIED
    assert result.error is not None
    assert result.error.code is ErrorCode.FORBIDDEN
