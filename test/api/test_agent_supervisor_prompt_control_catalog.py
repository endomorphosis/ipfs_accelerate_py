from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CONTROL_MUTATION_AUDIT_RECEIPT_SCHEMA,
    CONTROL_PROPOSAL_AUDIT_RECEIPT_SCHEMA,
    AuthorizationBindingError,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlContractError,
    ControlRoot,
    ControlTargetKind,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    IdempotencyKey,
    MUTATION_OPERATIONS,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
    OPERATION_CATALOG_V2,
    PROMPT_CONTROL_OPERATIONS,
    PROPOSAL_OPERATIONS,
    PaginationKind,
    PathEscapeError,
    UnknownOperationError,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    RepositorySupervisorBackend,
    RescueHandler,
    RescuePreviewHandler,
    RestartHandler,
    SupervisorControlService,
    WorkflowMaterializeHandler,
    WorkflowPreviewHandler,
)


NEW_OPERATIONS = frozenset(
    {
        Operation.WORKFLOW_PREVIEW,
        Operation.WORKFLOW_MATERIALIZE,
        Operation.RESTART,
        Operation.RESCUE_PREVIEW,
        Operation.RESCUE,
    }
)


def _binding(repository_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "state_root": str(state_root),
        "repository_id": "repository:prompt",
        "tree_id": "tree:current",
        "objective_id": "ASI-150",
        "objective_revision": "objective:1",
        "policy_id": "policy:prompt-control",
        "policy_revision": "policy:1",
        "caller": "operator:alice",
    }


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


def _parameters(operation: Operation, *, tree_id: str) -> dict[str, Any]:
    if operation is Operation.WORKFLOW_MATERIALIZE:
        return {
            "preview_ref": "receipt:preview",
            "preview_root": "plan:root",
            "preview_repository_id": "repository:prompt",
            "preview_tree_id": tree_id,
            "preview_objective_id": "ASI-150",
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
            "expected_revision": 4,
            "deadline_ms": 30_000,
            "health_window_ms": 5_000,
            "reason": "operator requested restart",
        }
    if operation is Operation.RESCUE:
        return {
            "incident_cid": "incident:one",
            "incident_root": "incident-root:one",
            "incident_repository_id": "repository:prompt",
            "incident_tree_id": tree_id,
            "incident_objective_id": "ASI-150",
            "incident_objective_revision": "objective:1",
            "incident_policy_id": "policy:prompt-control",
            "incident_policy_revision": "policy:1",
            "rescue_plan_cid": "rescue-plan:one",
            "rescue_plan_root": "rescue-plan-root:one",
            "rescue_plan_incident_cid": "incident:one",
            "rescue_plan_tree_id": tree_id,
            "action_index": 0,
            "expected_revision": 0,
        }
    raise AssertionError(f"unsupported mutation fixture {operation.value}")


def _mutation_request(
    repository_root: Path,
    state_root: Path,
    operation: Operation,
    *,
    key: str,
    tree_id: str = "tree:current",
) -> OperationRequest:
    binding = _binding(repository_root, state_root)
    binding["tree_id"] = tree_id
    effect = _effect(operation)
    lease_id = "lease:prompt"
    fencing_epoch = 9
    return OperationRequest(
        operation=operation,
        **binding,
        parameters=_parameters(operation, tree_id=tree_id),
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key=key,
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
            lease_id=lease_id,
            fencing_epoch=fencing_epoch,
            authorized_effect_ids=(effect.effect_id,),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id=lease_id,
        fencing_epoch=fencing_epoch,
    )


def test_prompt_control_catalog_is_closed_complete_and_exact() -> None:
    assert PROMPT_CONTROL_OPERATIONS == NEW_OPERATIONS
    assert frozenset(OPERATION_CATALOG_V2.operations) == frozenset(Operation)
    assert {
        Operation.WORKFLOW_PREVIEW,
        Operation.RESCUE_PREVIEW,
    }.issubset(PROPOSAL_OPERATIONS)
    assert {
        Operation.WORKFLOW_MATERIALIZE,
        Operation.RESTART,
        Operation.RESCUE,
    }.issubset(MUTATION_OPERATIONS)

    expected_targets = {
        Operation.WORKFLOW_PREVIEW: ControlTargetKind.WORKFLOW,
        Operation.WORKFLOW_MATERIALIZE: ControlTargetKind.WORKFLOW,
        Operation.RESTART: ControlTargetKind.SERVICE,
        Operation.RESCUE_PREVIEW: ControlTargetKind.INCIDENT,
        Operation.RESCUE: ControlTargetKind.INCIDENT,
    }
    for operation in NEW_OPERATIONS:
        descriptor = OPERATION_CATALOG_V2.operation(operation)
        assert descriptor.authority is operation.authority
        assert descriptor.target.kind is expected_targets[operation]
        assert descriptor.target.allowed_roots == (
            ControlRoot.REPOSITORY,
            ControlRoot.STATE,
        )
        assert descriptor.pagination.kind is PaginationKind.NONE
        assert descriptor.bounds.max_items <= 512
        assert descriptor.bounds.max_effects <= 64
        assert descriptor.bounds.timeout_ms <= 120_000
        assert descriptor.request_schema["additionalProperties"] is False
        assert (
            descriptor.request_schema["properties"]["parameters"][
                "additionalProperties"
            ]
            is False
        )
        assert descriptor.result_schema["additionalProperties"] is False
        assert descriptor.audit_receipt_schema == (
            CONTROL_MUTATION_AUDIT_RECEIPT_SCHEMA
            if operation.mutating
            else CONTROL_PROPOSAL_AUDIT_RECEIPT_SCHEMA
        )
        assert descriptor.requires_authorization is operation.mutating
        assert descriptor.requires_idempotency is operation.mutating
        assert descriptor.requires_lease is operation.mutating
        assert descriptor.requires_fencing is operation.mutating


def test_unknown_operations_fields_paths_and_transport_overrides_fail_closed(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path / "repo", tmp_path / "state")
    with pytest.raises(UnknownOperationError):
        OperationRequest(operation="shell", **binding)
    with pytest.raises(ControlContractError, match="unsupported fields"):
        OperationRequest.from_dict(
            {
                "operation": Operation.WORKFLOW_PREVIEW.value,
                **binding,
                "unknown": True,
            }
        )
    with pytest.raises(PathEscapeError, match="outside repository_root"):
        OperationRequest(
            operation=Operation.WORKFLOW_PREVIEW,
            **binding,
            parameters={
                "directory": "/untrusted/repository",
                "prompt_source": {
                    "kind": "artifact",
                    "content_cid": "prompt:one",
                },
            },
        )
    with pytest.raises(ControlContractError, match="unsupported fields"):
        OperationRequest(
            operation=Operation.WORKFLOW_PREVIEW,
            **binding,
            parameters={"transport_authority": "mutation"},
        )
    with pytest.raises(ControlContractError, match="unsupported fields"):
        OperationRequest(
            operation=Operation.RESCUE_PREVIEW,
            **binding,
            parameters={
                "incident_cid": "incident:one",
                "incident_tree_id": "tree:current",
                "mcp_repository_root": "/untrusted/repository",
            },
        )


def test_preview_operations_describe_but_cannot_apply_mutation_effects(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = OperationRequest(
        operation=Operation.WORKFLOW_PREVIEW,
        **_binding(repository_root, state_root),
        parameters={
            "directory": str(repository_root),
            "prompt_source": {
                "kind": "inline",
                "content_cid": "prompt:one",
            },
            "output_mode": "both",
        },
        expected_effects=(_effect(Operation.WORKFLOW_MATERIALIZE),),
    )
    calls = 0

    def preview_handler(_request: OperationRequest) -> BackendResponse:
        nonlocal calls
        calls += 1
        return BackendResponse(
            data={"proposal_root": "plan:one"},
            changed=True,
            checks=("schema", "admission"),
        )

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={Operation.WORKFLOW_PREVIEW: preview_handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )
    result = service.workflow_preview(request)

    assert calls == 1
    assert result.status is OperationStatus.SUCCEEDED
    assert result.authority is OperationAuthority.PROPOSAL
    assert result.preview is not None
    assert result.preview.expected_effects == request.expected_effects
    assert not result.effects
    assert result.audit_receipt_id

    def lying_handler(_request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            changed=True,
            applied_effect_ids=(request.expected_effects[0].effect_id,),
        )

    denied = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={Operation.WORKFLOW_PREVIEW: lying_handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    ).execute(request)
    assert denied.status is OperationStatus.FAILED
    assert denied.error and denied.error.code is ErrorCode.INVALID_REQUEST


def test_stale_preview_and_incident_bindings_are_rejected(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    binding = _binding(repository_root, state_root)
    materialize = _parameters(
        Operation.WORKFLOW_MATERIALIZE, tree_id="tree:stale"
    )
    with pytest.raises(AuthorizationBindingError, match="preview tree_id"):
        OperationRequest(
            operation=Operation.WORKFLOW_MATERIALIZE,
            **binding,
            parameters=materialize,
            dry_run=True,
        )

    with pytest.raises(AuthorizationBindingError, match="incident tree_id"):
        OperationRequest(
            operation=Operation.RESCUE_PREVIEW,
            **binding,
            parameters={
                "incident_cid": "incident:stale",
                "incident_root": "incident-root:stale",
                "incident_repository_id": "repository:prompt",
                "incident_tree_id": "tree:stale",
                "incident_objective_id": "ASI-150",
                "incident_objective_revision": "objective:1",
                "incident_policy_id": "policy:prompt-control",
                "incident_policy_revision": "policy:1",
            },
        )
    with pytest.raises(
        AuthorizationBindingError, match="different incident"
    ):
        OperationRequest(
            operation=Operation.RESCUE,
            **binding,
            parameters={
                "incident_cid": "incident:one",
                "incident_root": "incident-root:one",
                "incident_repository_id": "repository:prompt",
                "incident_tree_id": "tree:current",
                "incident_objective_id": "ASI-150",
                "incident_objective_revision": "objective:1",
                "incident_policy_id": "policy:prompt-control",
                "incident_policy_revision": "policy:1",
                "rescue_plan_cid": "rescue-plan:foreign",
                "rescue_plan_root": "rescue-plan-root:foreign",
                "rescue_plan_incident_cid": "incident:other",
                "rescue_plan_tree_id": "tree:current",
            },
            dry_run=True,
        )


def test_mutations_require_authority_and_cross_target_replay_conflicts(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    binding = _binding(repository_root, state_root)
    with pytest.raises(AuthorizationBindingError, match="authorization"):
        OperationRequest(
            operation=Operation.RESTART,
            **binding,
            parameters=_parameters(
                Operation.RESTART, tree_id="tree:current"
            ),
            expected_effects=(_effect(Operation.RESTART),),
            idempotency=IdempotencyKey(
                key="restart:missing-authority",
                operation=Operation.RESTART,
                caller=binding["caller"],
                repository_id=binding["repository_id"],
                objective_id=binding["objective_id"],
            ),
            lease_id="lease:prompt",
            fencing_epoch=9,
        )

    first = _mutation_request(
        repository_root,
        state_root,
        Operation.WORKFLOW_MATERIALIZE,
        key="workflow:shared",
    )
    foreign = _mutation_request(
        repository_root,
        state_root,
        Operation.WORKFLOW_MATERIALIZE,
        key="workflow:shared",
        tree_id="tree:other",
    )
    calls = 0

    def materialize_handler(request: OperationRequest) -> BackendResponse:
        nonlocal calls
        calls += 1
        return BackendResponse(
            data={"preview_ref": request.parameters["preview_ref"]},
            changed=True,
            applied_effect_ids=(request.expected_effects[0].effect_id,),
        )

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={
            Operation.WORKFLOW_MATERIALIZE: materialize_handler,
        },
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )
    applied = service.workflow_materialize(first)
    replay_conflict = service.workflow_materialize(foreign)

    assert applied.status is OperationStatus.SUCCEEDED
    assert applied.audit_receipt_id
    assert applied.effects[0].applied
    assert replay_conflict.status is OperationStatus.CONFLICT
    assert replay_conflict.error
    assert replay_conflict.error.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert calls == 1


def test_default_backend_exposes_interfaces_without_eager_effects() -> None:
    optional_before = {
        name
        for name in sys.modules
        if name.startswith("ipfs_accelerate_py.agent_supervisor.todo_daemon.llm")
    }
    backend = RepositorySupervisorBackend()

    assert not NEW_OPERATIONS.intersection(backend.registered_operations)
    assert not backend.optional_providers_loaded
    assert not backend.processes_started
    assert WorkflowPreviewHandler
    assert WorkflowMaterializeHandler
    assert RestartHandler
    assert RescuePreviewHandler
    assert RescueHandler
    assert {
        name
        for name in sys.modules
        if name.startswith("ipfs_accelerate_py.agent_supervisor.todo_daemon.llm")
    } == optional_before
