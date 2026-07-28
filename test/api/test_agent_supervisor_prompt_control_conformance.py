"""ASI-153: Python/CLI/MCP parity for prompt workflow control operations."""

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
    PROMPT_CONTROL_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    MutationRecoveryAction,
    MutationTransactionPhase,
    PartialMutationError,
    SupervisorControlService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    configure_agent_supervisor_control,
)


PROMPT_OPS = tuple(
    sorted(PROMPT_CONTROL_OPERATIONS, key=lambda item: item.value)
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
        "repository_id": "repository:prompt",
        "tree_id": "tree:current",
        "objective_id": "ASI-153",
        "objective_revision": "objective:1",
        "policy_id": "policy:prompt-control",
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
        kind=(
            EffectKind.LIFECYCLE_TRANSITION
            if operation is Operation.RESTART
            else EffectKind.WRITE_STATE
        ),
        resource=f"supervisor:{operation.value}",
        paths=(f"receipts/{operation.value}.json",),
    )


def _parameters(operation: Operation, repository_root: Path) -> dict[str, Any]:
    if operation is Operation.WORKFLOW_PREVIEW:
        return {
            "directory": str(repository_root),
            "prompt_source": {"kind": "inline", "content_cid": "prompt:one"},
            "output_mode": "both",
        }
    if operation is Operation.WORKFLOW_MATERIALIZE:
        return {
            "preview_ref": "receipt:preview",
            "preview_root": "plan:root",
            "preview_repository_id": "repository:prompt",
            "preview_tree_id": "tree:current",
            "preview_objective_id": "ASI-153",
            "preview_objective_revision": "objective:1",
            "preview_policy_id": "policy:prompt-control",
            "preview_policy_revision": "policy:1",
            "output_mode": "both",
            "markdown_path": "plans/generated.todo.md",
            "duckdb_path": "state/generated.duckdb",
        }
    if operation is Operation.RESTART:
        return {
            "target_id": "supervisor:prompt",
            "run_id": "run:old",
            "configuration_root": "configuration:1",
            "expected_revision": 1,
            "deadline_ms": 30_000,
            "health_window_ms": 5_000,
            "reason": "parity restart",
        }
    if operation is Operation.RESCUE_PREVIEW:
        return {
            "incident_cid": "incident:one",
            "incident_root": "incident-root:one",
            "incident_repository_id": "repository:prompt",
            "incident_tree_id": "tree:current",
            "incident_objective_id": "ASI-153",
            "incident_objective_revision": "objective:1",
            "incident_policy_id": "policy:prompt-control",
            "incident_policy_revision": "policy:1",
        }
    return {
        "incident_cid": "incident:one",
        "incident_root": "incident-root:one",
        "incident_repository_id": "repository:prompt",
        "incident_tree_id": "tree:current",
        "incident_objective_id": "ASI-153",
        "incident_objective_revision": "objective:1",
        "incident_policy_id": "policy:prompt-control",
        "incident_policy_revision": "policy:1",
        "rescue_plan_cid": "rescue-plan:one",
        "rescue_plan_root": "rescue-plan-root:one",
        "rescue_plan_incident_cid": "incident:one",
        "rescue_plan_tree_id": "tree:current",
        "action_index": 0,
        "expected_revision": 0,
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
    if operation in {Operation.WORKFLOW_PREVIEW, Operation.RESCUE_PREVIEW}:
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
            lease_id="lease:prompt",
            fencing_epoch=9,
            authorized_effect_ids=(effect.effect_id,),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id="lease:prompt",
        fencing_epoch=9,
        dry_run=dry_run,
    )


def _service(
    repository_root: Path,
    state_root: Path,
    operation: Operation,
    *,
    apply: bool = False,
    authorization_validator: Any = None,
) -> SupervisorControlService:
    effect_id = _effect(operation).effect_id

    def handler(request: OperationRequest) -> BackendResponse:
        if apply and not request.dry_run:
            return BackendResponse(
                data={"operation": operation.value, "ok": True},
                changed=True,
                applied_effect_ids=(effect_id,),
                checks=("schema",),
            )
        return BackendResponse(
            data={"operation": operation.value, "ok": True},
            changed=False,
            checks=("schema",),
        )

    return SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={operation: handler},
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


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", list(PROMPT_OPS))
async def test_python_cli_mcp_records_are_identical(
    operation: Operation,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(operation, repository_root, state_root, dry_run=True)
    service = _service(repository_root, state_root, operation)

    python_result = service.execute(request)
    assert python_result.status is OperationStatus.SUCCEEDED
    cli_record = _cli_record(service, request, capsys)
    mcp_record = await _mcp_record(service, request)

    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()


@pytest.mark.asyncio
async def test_mutation_idempotent_replay_is_identical_across_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    operation = Operation.WORKFLOW_MATERIALIZE
    request = _request(
        operation,
        repository_root,
        state_root,
        dry_run=False,
        key="materialize:idempotent",
    )
    service = _service(
        repository_root, state_root, operation, apply=True
    )

    first = service.execute(request)
    assert first.status is OperationStatus.SUCCEEDED
    assert first.effects and first.effects[0].applied

    # Exact replay reuses the receipt without a second backend application.
    second_python = service.execute(request)
    assert second_python.to_record() == first.to_record()
    assert second_python.audit_receipt_id == first.audit_receipt_id

    cli_record = _cli_record(service, request, capsys)
    mcp_record = await _mcp_record(service, request)
    assert cli_record == first.to_record()
    assert mcp_record == first.to_record()


@pytest.mark.asyncio
async def test_stale_root_rejection_is_identical_across_surfaces(
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
    operation = Operation.WORKFLOW_PREVIEW
    request = _request(operation, foreign_repo, foreign_state, dry_run=True)
    service = _service(repository_root, state_root, operation)

    python_result = service.execute(request)
    assert python_result.status is OperationStatus.DENIED
    assert python_result.error
    assert python_result.error.code is ErrorCode.FORBIDDEN

    cli_record = _cli_record(
        service, request, capsys, expected_exit=AGENT_CLI_EXIT_INVALID
    )
    mcp_record = await _mcp_record(service, request)
    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()


@pytest.mark.asyncio
async def test_authorization_denial_is_identical_across_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    operation = Operation.RESCUE
    request = _request(
        operation,
        repository_root,
        state_root,
        dry_run=False,
        key="rescue:denied",
    )
    service = _service(
        repository_root,
        state_root,
        operation,
        apply=True,
        authorization_validator=lambda _request: False,
    )

    python_result = service.execute(request)
    assert python_result.status is OperationStatus.DENIED
    assert python_result.error
    assert python_result.error.code is ErrorCode.UNAUTHORIZED

    cli_record = _cli_record(
        service, request, capsys, expected_exit=AGENT_CLI_EXIT_INVALID
    )
    mcp_record = await _mcp_record(service, request)
    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()


@pytest.mark.asyncio
async def test_idempotency_conflict_is_identical_across_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    operation = Operation.WORKFLOW_MATERIALIZE
    first = _request(
        operation,
        repository_root,
        state_root,
        dry_run=False,
        key="materialize:shared",
    )
    foreign_binding = _binding(repository_root, state_root)
    foreign_binding["tree_id"] = "tree:other"
    effect = _effect(operation)
    conflicting = OperationRequest(
        operation=operation,
        **foreign_binding,
        parameters={
            **_parameters(operation, repository_root),
            "preview_tree_id": "tree:other",
        },
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key="materialize:shared",
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
            lease_id="lease:prompt",
            fencing_epoch=9,
            authorized_effect_ids=(effect.effect_id,),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id="lease:prompt",
        fencing_epoch=9,
        dry_run=False,
    )
    service = _service(
        repository_root, state_root, operation, apply=True
    )
    applied = service.execute(first)
    assert applied.status is OperationStatus.SUCCEEDED

    conflict = service.execute(conflicting)
    assert conflict.status is OperationStatus.CONFLICT
    assert conflict.error
    assert conflict.error.code is ErrorCode.IDEMPOTENCY_CONFLICT

    cli_record = _cli_record(
        service,
        conflicting,
        capsys,
        expected_exit=AGENT_CLI_EXIT_CONFLICT,
    )
    mcp_record = await _mcp_record(service, conflicting)
    assert cli_record == conflict.to_record()
    assert mcp_record == conflict.to_record()


@pytest.mark.asyncio
async def test_partial_saga_is_identical_across_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Partial multi-step mutation yields the same durable conflict on every surface."""

    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    operation = Operation.WORKFLOW_MATERIALIZE
    write_effect = ExpectedEffect(
        effect_id="workflow_materialize:write",
        kind=EffectKind.WRITE_STATE,
        resource="supervisor:workflow_materialize",
        paths=("plans/generated.todo.md",),
    )
    start_effect = ExpectedEffect(
        effect_id="workflow_materialize:start",
        kind=EffectKind.LIFECYCLE_TRANSITION,
        resource="supervisor:workflow_materialize",
        paths=("receipts/workflow_materialize.json",),
    )
    binding = _binding(repository_root, state_root)
    request = OperationRequest(
        operation=operation,
        **binding,
        parameters=_parameters(operation, repository_root),
        expected_effects=(write_effect, start_effect),
        idempotency=IdempotencyKey(
            key="materialize:partial-saga",
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
            lease_id="lease:prompt",
            fencing_epoch=9,
            authorized_effect_ids=(
                write_effect.effect_id,
                start_effect.effect_id,
            ),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id="lease:prompt",
        fencing_epoch=9,
        dry_run=False,
    )
    calls = 0

    def partial_handler(_request: OperationRequest) -> BackendResponse:
        nonlocal calls
        calls += 1
        raise PartialMutationError(
            "workflow materialize start step failed after write",
            applied_effect_ids=(write_effect.effect_id,),
            recovery=MutationRecoveryAction.COMPENSATE,
        )

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={operation: partial_handler},
        authorization_validator=lambda _request: True,
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 4_000,
    )

    python_result = service.execute(request)
    assert python_result.status is OperationStatus.CONFLICT
    assert python_result.error
    assert python_result.error.code is ErrorCode.CONFLICT
    assert python_result.data["transaction"]["recovery_action"] == "compensate"
    assert python_result.data["transaction"]["phase"] == (
        MutationTransactionPhase.COMPENSATION_REQUIRED.value
    )
    assert python_result.data["transaction"]["applied_effect_ids"] == (
        write_effect.effect_id,
    )
    applied_by_id = {
        effect.effect_id: effect.applied for effect in python_result.effects
    }
    assert applied_by_id == {
        write_effect.effect_id: True,
        start_effect.effect_id: False,
    }

    # Exact replay must reuse the durable partial receipt without re-dispatch.
    replay = service.execute(request)
    assert replay.to_record() == python_result.to_record()
    assert calls == 1

    cli_record = _cli_record(
        service,
        request,
        capsys,
        expected_exit=AGENT_CLI_EXIT_CONFLICT,
    )
    mcp_record = await _mcp_record(service, request)
    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()
    assert calls == 1

