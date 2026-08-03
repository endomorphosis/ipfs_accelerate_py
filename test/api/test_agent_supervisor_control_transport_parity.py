"""PDR-032: Python / CLI / MCP transport parity for plan create/steer operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    AGENT_CLI_EXIT_CONFLICT,
    AGENT_CLI_EXIT_INVALID,
    AGENT_CLI_EXIT_SUCCESS,
    COMMAND_OPERATIONS,
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
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    configure_agent_supervisor_control,
)


PARITY_OPS = tuple(
    sorted(
        PLAN_CONTROL_OPERATIONS | PLAN_WORKFLOW_ALIAS_OPERATIONS,
        key=lambda item: item.value,
    )
)


@pytest.fixture(autouse=True)
def _reset_mcp() -> Any:
    configure_agent_supervisor_control()
    yield
    configure_agent_supervisor_control()


def _binding(repository_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "state_root": str(state_root),
        "repository_id": "repository:transport-parity",
        "tree_id": "tree:parity",
        "objective_id": "PDR-032",
        "objective_revision": "objective:parity",
        "policy_id": "policy:transport-parity",
        "policy_revision": "policy:1",
        "caller": "operator:parity",
    }


def _cli_command(operation: Operation) -> str:
    return next(
        command
        for command, candidate in COMMAND_OPERATIONS.items()
        if candidate is operation
    )


def _effect(operation: Operation) -> ExpectedEffect:
    return ExpectedEffect(
        effect_id=f"{operation.value}:parity",
        kind=EffectKind.WRITE_STATE,
        resource=f"supervisor:{operation.value}",
        paths=(f"receipts/{operation.value}.json",),
    )


def _parameters(operation: Operation, repository_root: Path) -> dict[str, Any]:
    if operation is Operation.PLAN_CREATE_PREVIEW:
        return {"mode": "deterministic"}
    if operation is Operation.PLAN_STEER_PREVIEW:
        return {}
    if operation is Operation.PLAN_CREATE_APPLY:
        return {
            "preview_ref": "receipt:create",
            "preview_root": "plan:root",
            "apply_request": {"idempotency_key": "parity:create"},
        }
    if operation is Operation.PLAN_STEER_APPLY:
        return {
            "preview_ref": "receipt:steer",
            "preview_root": "plan:root",
            "apply_request": {"idempotency_key": "parity:steer"},
        }
    if operation is Operation.WORKFLOW_PREVIEW:
        return {
            "directory": str(repository_root),
            "prompt_source": {"kind": "inline", "content_cid": "prompt:parity"},
            "output_mode": "both",
        }
    return {
        "preview_ref": "receipt:preview",
        "preview_root": "plan:root",
        "preview_repository_id": "repository:transport-parity",
        "preview_tree_id": "tree:parity",
        "preview_objective_id": "PDR-032",
        "preview_objective_revision": "objective:parity",
        "preview_policy_id": "policy:transport-parity",
        "preview_policy_revision": "policy:1",
        "output_mode": "both",
        "markdown_path": "plans/parity.todo.md",
        "duckdb_path": "state/parity.duckdb",
        "apply_request": {"idempotency_key": "parity:workflow"},
    }


def _request(
    operation: Operation,
    repository_root: Path,
    state_root: Path,
    *,
    dry_run: bool = True,
    key: str | None = None,
) -> OperationRequest:
    binding = _binding(repository_root, state_root)
    parameters = _parameters(operation, repository_root)
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
            key=key or f"parity:{operation.value}",
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
            lease_id="lease:parity",
            fencing_epoch=3,
            authorized_effect_ids=(effect.effect_id,),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id="lease:parity",
        fencing_epoch=3,
        dry_run=dry_run,
    )


def _service(
    repository_root: Path,
    state_root: Path,
    *,
    apply: bool = False,
    authorization_validator: Any = None,
) -> SupervisorControlService:
    def handler(request: OperationRequest) -> BackendResponse:
        if apply and not request.dry_run and request.operation in MUTATION_OPERATIONS:
            return BackendResponse(
                data={
                    "operation": request.operation.value,
                    "transport": "shared",
                    "ok": True,
                },
                changed=True,
                applied_effect_ids=(_effect(request.operation).effect_id,),
                checks=("schema", "parity"),
            )
        return BackendResponse(
            data={
                "operation": request.operation.value,
                "transport": "shared",
                "ok": True,
                "read_only": True,
                "wrote_effects": (),
            },
            changed=False,
            checks=("schema", "parity", "proposal_only"),
        )

    handlers = {operation: handler for operation in PARITY_OPS}
    return SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers=handlers,
        authorization_validator=authorization_validator,
        lease_validator=(lambda _request: True) if apply else None,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 5_000,
    )


async def _mcp_record(
    service: SupervisorControlService, request: OperationRequest
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
    assert exit_status == expected_exit, captured.err or captured.out
    return json.loads(captured.out)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", list(PARITY_OPS))
async def test_python_cli_mcp_canonical_records_match(
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
    # Canonical roots and operation identity.
    assert python_result.operation is operation
    assert python_result.repository_id == request.repository_id
    assert python_result.tree_id == request.tree_id

    cli_record = _cli_record(service, request, capsys)
    mcp_record = await _mcp_record(service, request)
    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()


@pytest.mark.asyncio
async def test_cursor_and_error_parity_for_stale_roots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    foreign_repo = tmp_path / "foreign"
    foreign_state = tmp_path / "foreign-state"
    foreign_repo.mkdir()
    foreign_state.mkdir()
    operation = Operation.PLAN_CREATE_PREVIEW
    request = _request(operation, foreign_repo, foreign_state, dry_run=True)
    service = _service(repository_root, state_root)

    python_result = service.execute(request)
    assert python_result.status is OperationStatus.DENIED
    assert python_result.error is not None
    assert python_result.error.code is ErrorCode.FORBIDDEN

    cli_record = _cli_record(
        service, request, capsys, expected_exit=AGENT_CLI_EXIT_INVALID
    )
    mcp_record = await _mcp_record(service, request)
    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()


@pytest.mark.asyncio
async def test_authorization_denial_parity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    operation = Operation.PLAN_STEER_APPLY
    request = _request(
        operation, repository_root, state_root, dry_run=False, key="deny:steer"
    )
    service = _service(
        repository_root,
        state_root,
        apply=True,
        authorization_validator=lambda _request: False,
    )
    python_result = service.execute(request)
    assert python_result.status is OperationStatus.DENIED
    assert python_result.error is not None
    assert python_result.error.code is ErrorCode.UNAUTHORIZED

    cli_record = _cli_record(
        service, request, capsys, expected_exit=AGENT_CLI_EXIT_INVALID
    )
    mcp_record = await _mcp_record(service, request)
    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()


@pytest.mark.asyncio
async def test_idempotency_conflict_parity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    operation = Operation.PLAN_CREATE_APPLY
    first = _request(
        operation,
        repository_root,
        state_root,
        dry_run=False,
        key="shared:idem",
    )
    foreign_binding = _binding(repository_root, state_root)
    foreign_binding["tree_id"] = "tree:other"
    effect = _effect(operation)
    conflicting = OperationRequest(
        operation=operation,
        **foreign_binding,
        parameters={
            **_parameters(operation, repository_root),
            "preview_root": "plan:other",
        },
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key="shared:idem",
            operation=operation,
            caller=foreign_binding["caller"],
            repository_id=foreign_binding["repository_id"],
            objective_id=foreign_binding["objective_id"],
        ),
        authorization=AuthorizationDecision(
            verdict=AuthorizationVerdict.PERMIT,
            operation=operation,
            granted_authority=OperationAuthority.MUTATION,
            **foreign_binding,
            lease_id="lease:parity",
            fencing_epoch=3,
            authorized_effect_ids=(effect.effect_id,),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id="lease:parity",
        fencing_epoch=3,
        dry_run=False,
    )
    service = _service(repository_root, state_root, apply=True)
    applied = service.execute(first)
    assert applied.status is OperationStatus.SUCCEEDED

    python_conflict = service.execute(conflicting)
    assert python_conflict.status is OperationStatus.CONFLICT
    assert python_conflict.error is not None
    assert python_conflict.error.code is ErrorCode.IDEMPOTENCY_CONFLICT

    cli_record = _cli_record(
        service, conflicting, capsys, expected_exit=AGENT_CLI_EXIT_CONFLICT
    )
    mcp_record = await _mcp_record(service, conflicting)
    assert cli_record == python_conflict.to_record()
    assert mcp_record == python_conflict.to_record()


@pytest.mark.asyncio
async def test_workflow_alias_identity_preserved_across_transports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    service = _service(repository_root, state_root)
    for operation in (
        Operation.WORKFLOW_PREVIEW,
        Operation.WORKFLOW_MATERIALIZE,
    ):
        request = _request(operation, repository_root, state_root, dry_run=True)
        python_result = service.execute(request)
        assert python_result.operation is operation
        assert python_result.to_record()["operation"] == operation.value
        cli_record = _cli_record(service, request, capsys)
        mcp_record = await _mcp_record(service, request)
        assert cli_record["operation"] == operation.value
        assert mcp_record["operation"] == operation.value
        assert cli_record == python_result.to_record()
        assert mcp_record == python_result.to_record()
