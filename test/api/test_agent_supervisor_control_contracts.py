from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context_contracts import (
    ContextBoundsError,
    ContextBudget,
    ContextCapsule,
    ContextContractError,
    ContextReference,
    ContextTier,
)
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    AuthorizationBindingError,
    AuthorizationDecision,
    AuthorizationVerdict,
    AuthorityViolationError,
    CapabilityReport,
    ControlBounds,
    ControlBoundsError,
    ControlContractError,
    DryRunPreview,
    EffectClaim,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    IdempotencyKey,
    LifecycleAction,
    LifecycleCommand,
    MissingIdempotencyError,
    Operation,
    OperationAuthority,
    OperationCapability,
    OperationError,
    OperationRequest,
    OperationResult,
    OperationStatus,
    PathEscapeError,
    UnknownOperationError,
    canonical_control_json_bytes,
    operation_authority,
)


ROOTS = {
    "repository_root": "/srv/repos/example",
    "state_root": "/var/lib/example-supervisor",
}
BINDING = {
    **ROOTS,
    "repository_id": "repo:example",
    "tree_id": "tree:abc",
    "objective_id": "ASI-G070",
    "objective_revision": "sha256:objective",
    "policy_id": "supervisor-policy",
    "policy_revision": "sha256:policy",
    "caller": "operator:alice",
}


def _effect(
    effect_id: str = "pause-daemon",
    *,
    kind: EffectKind = EffectKind.LIFECYCLE_TRANSITION,
) -> ExpectedEffect:
    return ExpectedEffect(
        effect_id=effect_id,
        kind=kind,
        resource="supervisor:primary",
        paths=("ipfs_accelerate_py/agent_supervisor",),
        description="Pause the primary supervisor",
    )


def _authorization(
    operation: Operation = Operation.PAUSE,
    *,
    effect_ids: tuple[str, ...] = ("pause-daemon",),
    caller: str = BINDING["caller"],
    verdict: AuthorizationVerdict = AuthorizationVerdict.PERMIT,
) -> AuthorizationDecision:
    return AuthorizationDecision(
        verdict=verdict,
        operation=operation,
        granted_authority=(
            OperationAuthority.MUTATION
            if verdict is AuthorizationVerdict.PERMIT
            else None
        ),
        **{**BINDING, "caller": caller},
        lease_id="lease:7",
        fencing_epoch=7,
        authorized_effect_ids=effect_ids if verdict is AuthorizationVerdict.PERMIT else (),
        reason_code="" if verdict is AuthorizationVerdict.PERMIT else "policy_denied",
        grant_ids=("grant:operator",) if verdict is AuthorizationVerdict.PERMIT else (),
        evaluated_at_ms=1_000,
        expires_at_ms=2_000,
    )


def _idempotency(
    operation: Operation = Operation.PAUSE,
    *,
    caller: str = BINDING["caller"],
) -> IdempotencyKey:
    return IdempotencyKey(
        key="pause/ASI-G070/7",
        operation=operation,
        caller=caller,
        repository_id=BINDING["repository_id"],
        objective_id=BINDING["objective_id"],
    )


def _mutation_request(**changes: object) -> OperationRequest:
    values: dict[str, object] = {
        "operation": Operation.PAUSE,
        **BINDING,
        "expected_effects": (_effect(),),
        "parameters": {
            "target": "supervisor:primary",
            "path": "ipfs_accelerate_py/agent_supervisor",
        },
        "idempotency": _idempotency(),
        "authorization": _authorization(),
        "lease_id": "lease:7",
        "fencing_epoch": 7,
    }
    values.update(changes)
    return OperationRequest(**values)


def _read_request(**changes: object) -> OperationRequest:
    values: dict[str, object] = {
        "operation": Operation.STATUS,
        **BINDING,
        "parameters": {"limit": 20, "path": "docs/architecture"},
    }
    values.update(changes)
    return OperationRequest(**values)


def _context_capsule(**changes: object) -> ContextCapsule:
    values: dict[str, object] = {
        "repository_id": BINDING["repository_id"],
        "tree_id": BINDING["tree_id"],
        "objective_id": BINDING["objective_id"],
        "objective_revision": BINDING["objective_revision"],
        "policy_id": BINDING["policy_id"],
        "policy_revision": BINDING["policy_revision"],
        "caller": BINDING["caller"],
        "stage": "planning",
        "budget": ContextBudget(),
        "goal": {"outcome": "unified typed control"},
        "authority": {"maximum": "proposal"},
        "scope": {"paths": ["ipfs_accelerate_py/agent_supervisor"]},
        "acceptance": {"criteria": ["schema parity", "bounded reads"]},
        "evidence": (
            ContextReference(
                reference_id="architecture-plan",
                kind="plan-section",
                content_id="bafy-plan",
                repository_id=BINDING["repository_id"],
                tree_id=BINDING["tree_id"],
                path="docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md",
                summary="Shared service requirements",
                token_count=40,
            ),
        ),
        "input_tokens": 100,
    }
    values.update(changes)
    return ContextCapsule(**values)


def test_context_budget_capsule_and_reference_are_canonical_round_trips() -> None:
    capsule = _context_capsule()

    restored = ContextCapsule.from_json(capsule.to_json())

    assert restored == capsule
    assert restored.capsule_id == capsule.capsule_id
    assert restored.to_json() == capsule.to_json()
    assert restored.invariant_core["goal"]["outcome"] == "unified typed control"
    assert restored.evidence[0].referenced_content_id == "bafy-plan"
    assert restored.evidence[0].content_id != "bafy-plan"
    assert json.loads(capsule.to_json())["contract_version"] == 1


def test_context_contracts_are_deeply_immutable_and_identity_stable() -> None:
    capsule = _context_capsule()

    with pytest.raises(FrozenInstanceError):
        capsule.tree_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        capsule.goal["outcome"] = "changed"  # type: ignore[index]

    reordered = _context_capsule(
        goal={"z": 1, "outcome": "unified typed control", "a": 2}
    )
    same_reordered = _context_capsule(
        goal={"a": 2, "outcome": "unified typed control", "z": 1}
    )
    assert reordered.content_id == same_reordered.content_id
    assert reordered.to_json() == same_reordered.to_json()


def test_context_enforces_required_core_count_bytes_depth_and_identity() -> None:
    with pytest.raises(ContextContractError, match="acceptance"):
        _context_capsule(acceptance={})
    with pytest.raises(ContextBoundsError, match="input-token"):
        _context_capsule(
            budget=ContextBudget(max_input_tokens=50),
            input_tokens=51,
        )
    with pytest.raises(ContextBoundsError, match="nesting-depth"):
        _context_capsule(
            budget=ContextBudget(max_depth=2),
            goal={"a": {"b": {"c": "too deep"}}},
        )
    with pytest.raises(ContextBoundsError, match="item-count"):
        _context_capsule(
            budget=ContextBudget(max_items=2),
            goal={"a": 1, "b": 2},
        )

    forged = _context_capsule().to_record()
    forged["tree_id"] = "tree:forged"
    with pytest.raises(ContextContractError, match="identity"):
        ContextCapsule.from_dict(forged)


def test_context_rejects_path_escapes_mixed_trees_and_invalid_truncation() -> None:
    with pytest.raises(ContextContractError, match="repository-relative"):
        ContextReference("escape", "source", path="../../etc/passwd")
    with pytest.raises(ContextContractError, match="tree identity"):
        _context_capsule(
            evidence=(
                ContextReference(
                    "stale",
                    "source",
                    repository_id=BINDING["repository_id"],
                    tree_id="tree:stale",
                ),
            )
        )
    with pytest.raises(ContextContractError, match="truncated"):
        _context_capsule(truncated=True)
    with pytest.raises(ContextContractError, match="expansion tier"):
        _context_capsule(
            expansion_references=(
                ContextReference("wrong-tier", "source"),
            )
        )

    delta = _context_capsule(
        parent_capsule_id="capsule:previous",
        truncated=True,
        omissions=("unchanged source excerpts",),
        expansion_references=(
            ContextReference(
                "source-on-demand",
                "source",
                tier=ContextTier.EXPANSION,
            ),
        ),
    )
    assert delta.is_delta


def test_operation_registry_has_explicit_closed_authority_classes() -> None:
    assert operation_authority("status") is OperationAuthority.READ
    assert operation_authority("plan") is OperationAuthority.PROPOSAL
    assert operation_authority("pause") is OperationAuthority.MUTATION

    with pytest.raises(UnknownOperationError, match="unknown operation"):
        operation_authority("shell_exec")
    with pytest.raises(UnknownOperationError, match="unknown operation"):
        _read_request(operation="arbitrary_plugin_call")


def test_read_request_round_trip_binds_all_semantic_identities() -> None:
    request = _read_request()

    restored = OperationRequest.from_json(request.to_json())

    assert restored == request
    assert restored.request_id == request.request_id
    assert restored.authority is OperationAuthority.READ
    assert restored.effective_authority is OperationAuthority.READ
    assert request.parameters["path"] == "docs/architecture"
    assert canonical_control_json_bytes(request) == request.canonical_bytes()


def test_requests_reject_root_and_nested_parameter_path_escapes() -> None:
    with pytest.raises(PathEscapeError, match="absolute"):
        _read_request(repository_root="relative/repository")
    with pytest.raises(PathEscapeError, match="filesystem root"):
        _read_request(state_root="/")
    with pytest.raises(PathEscapeError, match="repository-relative"):
        _read_request(parameters={"filters": {"target_path": "../../secret"}})
    with pytest.raises(PathEscapeError, match="repository-relative"):
        _read_request(parameters={"paths": ["src", "/etc/passwd"]})
    with pytest.raises(PathEscapeError, match="repository-relative"):
        ExpectedEffect(
            "escape",
            EffectKind.WRITE_REPOSITORY,
            "repository",
            paths=("src/../../etc",),
        )


def test_request_enforces_parameter_count_depth_and_serialized_bytes() -> None:
    with pytest.raises(ControlBoundsError, match="item-count"):
        _read_request(
            bounds=ControlBounds(max_items=3, max_paths=3, max_effects=3),
            parameters={"a": 1, "b": 2, "c": 3},
        )
    with pytest.raises(ControlBoundsError, match="nesting-depth"):
        _read_request(
            bounds=ControlBounds(max_depth=2),
            parameters={"a": {"b": {"c": 1}}},
        )
    with pytest.raises(ControlBoundsError, match="serialized-byte"):
        _read_request(
            bounds=ControlBounds(max_serialized_bytes=1_500),
            parameters={"value": "x" * 800},
        )


def test_mutations_require_idempotency_authorization_lease_and_effects() -> None:
    with pytest.raises(MissingIdempotencyError, match="idempotency"):
        _mutation_request(idempotency=None)
    with pytest.raises(AuthorizationBindingError, match="authorization"):
        _mutation_request(authorization=None)
    with pytest.raises(AuthorizationBindingError, match="lease"):
        _mutation_request(lease_id="")
    with pytest.raises(AuthorityViolationError, match="expected effects"):
        _mutation_request(expected_effects=())


def test_mutation_bindings_fail_closed_on_scope_or_policy_mismatch() -> None:
    with pytest.raises(MissingIdempotencyError, match="scope"):
        _mutation_request(idempotency=_idempotency(caller="operator:mallory"))
    with pytest.raises(AuthorizationBindingError, match="binding"):
        _mutation_request(
            authorization=_authorization(caller="operator:mallory")
        )
    with pytest.raises(AuthorizationBindingError, match="every expected effect"):
        _mutation_request(
            authorization=_authorization(effect_ids=("different-effect",))
        )
    with pytest.raises(AuthorizationBindingError, match="permit"):
        _mutation_request(
            authorization=_authorization(verdict=AuthorizationVerdict.DENY)
        )


def test_authorized_mutation_round_trip_retains_nested_contracts() -> None:
    request = _mutation_request()

    restored = OperationRequest.from_json(request.to_json())

    assert restored == request
    assert restored.idempotency_key == "pause/ASI-G070/7"
    assert restored.authorization is not None
    assert restored.authorization.permitted
    assert restored.authorization.decision_id == request.authorization.decision_id
    assert restored.expected_effects[0].authority is OperationAuthority.MUTATION


def test_dry_run_mutation_is_proposal_only_and_needs_no_mutation_credentials() -> None:
    request = OperationRequest(
        operation=Operation.STOP,
        **BINDING,
        dry_run=True,
        expected_effects=(
            ExpectedEffect(
                "stop-daemon",
                EffectKind.STOP_PROCESS,
                "supervisor:primary",
            ),
        ),
    )
    preview = DryRunPreview(
        request_id=request.request_id,
        operation=request.operation,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        caller=request.caller,
        expected_effects=request.expected_effects,
        checks=("lease would be required",),
        would_change=True,
    )
    result = OperationResult(
        request_id=request.request_id,
        operation=request.operation,
        authority=OperationAuthority.PROPOSAL,
        status=OperationStatus.SUCCEEDED,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        caller=request.caller,
        preview=preview,
    )

    assert request.effective_authority is OperationAuthority.PROPOSAL
    assert preview.authority is OperationAuthority.PROPOSAL
    result.validate_against(request)


def test_result_claims_cannot_exceed_operation_or_result_authority() -> None:
    read = _read_request()
    with pytest.raises(AuthorityViolationError, match="outside"):
        OperationResult(
            request_id=read.request_id,
            operation=read.operation,
            authority=OperationAuthority.MUTATION,
            status=OperationStatus.SUCCEEDED,
            repository_id=read.repository_id,
            tree_id=read.tree_id,
            objective_id=read.objective_id,
            policy_id=read.policy_id,
            caller=read.caller,
        )
    with pytest.raises(AuthorityViolationError, match="outside result"):
        OperationResult(
            request_id=read.request_id,
            operation=Operation.PAUSE,
            authority=OperationAuthority.READ,
            status=OperationStatus.SUCCEEDED,
            repository_id=read.repository_id,
            tree_id=read.tree_id,
            objective_id=read.objective_id,
            policy_id=read.policy_id,
            caller=read.caller,
            effects=(
                EffectClaim(
                    "pause-daemon",
                    EffectKind.LIFECYCLE_TRANSITION,
                    "supervisor:primary",
                ),
            ),
        )


def test_result_validation_rejects_undeclared_or_reshaped_effects() -> None:
    request = _mutation_request()
    undeclared = OperationResult(
        request_id=request.request_id,
        operation=request.operation,
        authority=OperationAuthority.MUTATION,
        status=OperationStatus.SUCCEEDED,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        caller=request.caller,
        effects=(
            EffectClaim(
                "delete-unrelated",
                EffectKind.DELETE_REPOSITORY,
                "repository",
            ),
        ),
        idempotency_key=request.idempotency_key,
    )
    with pytest.raises(AuthorityViolationError, match="not declared"):
        undeclared.validate_against(request)

    reshaped = OperationResult(
        request_id=request.request_id,
        operation=request.operation,
        authority=OperationAuthority.MUTATION,
        status=OperationStatus.SUCCEEDED,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        caller=request.caller,
        effects=(
            EffectClaim(
                "pause-daemon",
                EffectKind.DELETE_REPOSITORY,
                "supervisor:primary",
            ),
        ),
        idempotency_key=request.idempotency_key,
    )
    with pytest.raises(AuthorityViolationError, match="declared shape"):
        reshaped.validate_against(request)


def test_applied_mutation_result_requires_and_binds_audit_receipts() -> None:
    request = _mutation_request()
    with pytest.raises(
        Exception, match="applied effect claim requires an audit receipt"
    ):
        EffectClaim(
            "pause-daemon",
            EffectKind.LIFECYCLE_TRANSITION,
            "supervisor:primary",
            applied=True,
        )

    result = OperationResult(
        request_id=request.request_id,
        operation=request.operation,
        authority=OperationAuthority.MUTATION,
        status=OperationStatus.SUCCEEDED,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        caller=request.caller,
        effects=(
            EffectClaim(
                "pause-daemon",
                EffectKind.LIFECYCLE_TRANSITION,
                "supervisor:primary",
                paths=("ipfs_accelerate_py/agent_supervisor",),
                applied=True,
                receipt_id="audit:effect:1",
            ),
        ),
        idempotency_key=request.idempotency_key,
        audit_receipt_id="audit:operation:1",
    )

    result.validate_against(request)
    assert OperationResult.from_json(result.to_json()) == result


def test_failures_require_typed_errors_and_round_trip() -> None:
    request = _read_request()
    error = OperationError(
        ErrorCode.NOT_FOUND,
        "Requested task was not found",
        field="task_id",
        details={"task_id": "ASI-999"},
    )
    result = OperationResult(
        request_id=request.request_id,
        operation=request.operation,
        authority=OperationAuthority.READ,
        status=OperationStatus.NOT_FOUND,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        caller=request.caller,
        error=error,
    )

    assert OperationResult.from_json(result.to_json()) == result
    assert result.error.code is ErrorCode.NOT_FOUND
    with pytest.raises(Exception, match="require a typed error"):
        OperationResult(
            request_id=request.request_id,
            operation=request.operation,
            authority=OperationAuthority.READ,
            status=OperationStatus.FAILED,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
        )


def test_capability_report_is_deterministic_and_mutations_advertise_guards() -> None:
    status = OperationCapability(
        Operation.STATUS,
        OperationAuthority.READ,
    )
    pause = OperationCapability(
        Operation.PAUSE,
        OperationAuthority.MUTATION,
        supports_dry_run=True,
        requires_idempotency=True,
        requires_authorization=True,
    )
    first = CapabilityReport(
        "supervisor-control",
        "1.0",
        (pause, status),
    )
    second = CapabilityReport(
        "supervisor-control",
        "1.0",
        (status, pause),
    )

    assert first == second
    assert first.content_id == second.content_id
    assert first.supports("pause")
    assert first.capability_for(Operation.STATUS) == status
    assert not first.optional_providers_loaded
    assert not first.processes_started
    assert CapabilityReport.from_json(first.to_json()) == first

    with pytest.raises(Exception, match="must advertise"):
        OperationCapability(
            Operation.STOP,
            OperationAuthority.MUTATION,
        )


def test_lifecycle_commands_are_typed_versioned_and_preview_aware() -> None:
    command = LifecycleCommand(
        LifecycleAction.DRAIN,
        target_id="supervisor:primary",
        reason="Prepare for maintenance",
        requested_state="draining",
        dry_run=True,
    )

    assert command.operation is Operation.DRAIN
    assert command.authority is OperationAuthority.PROPOSAL
    assert LifecycleCommand.from_json(command.to_json()) == command

    forged = command.to_record()
    forged["operation"] = "start"
    with pytest.raises(Exception, match="does not match"):
        LifecycleCommand.from_dict(forged)


def test_unknown_fields_noncanonical_values_and_forged_ids_fail_closed() -> None:
    request = _read_request()
    unknown = request.to_dict()
    unknown["shell"] = "rm -rf /"
    with pytest.raises(Exception, match="unsupported fields"):
        OperationRequest.from_dict(unknown)
    with pytest.raises(Exception, match="unsupported value type float"):
        _read_request(parameters={"threshold": 0.5})

    forged = request.to_record()
    forged["tree_id"] = "tree:forged"
    with pytest.raises(Exception, match="identity"):
        OperationRequest.from_dict(forged)

    with pytest.raises(ControlContractError, match="malformed"):
        OperationRequest.from_json("{")
    with pytest.raises(ContextContractError, match="contain an object"):
        ContextCapsule.from_json("[]")
