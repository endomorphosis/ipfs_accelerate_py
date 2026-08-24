"""Closed-contract and fail-closed tests for federation control dispatch."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.control.service import (
    execute_federation_command,
)
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.control_service import (
    FederationControlAuditReceipt,
    FederationControlAuthorization,
    FederationControlAuthorizationError,
    FederationControlCapability,
    FederationControlCapabilityError,
    FederationControlResponse,
    FederationControlResultError,
    FederationControlService,
    FederationControlServiceError,
    FederationControlStaleError,
    qualified_federation_control_capability,
)
from test.api.causal_federation.test_contracts import EXPIRY, NOW, sample_binding

NOW_EPOCH = 1_893_456_000.0


def command(*, operation: contracts.FederationOperation = contracts.FederationOperation.START,
            dry_run: bool = False) -> contracts.FederationCommand:
    return contracts.FederationCommand(
        record_id="command:test",
        revision=1,
        binding=sample_binding(),
        operation=operation,
        target_id="federation:test",
        expected_generation=1,
        expected_revision=1,
        expected_fencing_epoch=2,
        idempotency_key="idempotency:control-test",
        dry_run=dry_run,
        expected_effects=("effect.federation.lifecycle",),
    )


def authorization(value: contracts.FederationCommand) -> FederationControlAuthorization:
    return FederationControlAuthorization(
        authorization_id="authorization:test",
        command_cid=value.cid,
        caller_did="did:test:operator",
        tenant_id=value.binding.tenant_id,
        operation=value.operation,
        target_id=value.target_id,
        policy_ref=value.binding.policy_ref,
        policy_revision=value.binding.policy_revision,
        control_plane_generation=value.expected_generation,
        fencing_epoch=value.expected_fencing_epoch,
        lease_id="lease:test",
        expires_at=EXPIRY,
        decided_at=NOW,
    )


def response(value: contracts.FederationCommand, auth: FederationControlAuthorization) -> FederationControlResponse:
    outcome = "dry_run" if value.dry_run else "applied"
    result = contracts.FederationCommandResult(
        record_id="command-result:test",
        revision=value.expected_revision,
        binding=value.binding,
        outcome=outcome,
        evidence_refs=(value.cid,),
        recorded_at=NOW,
    )
    return FederationControlResponse(
        result=result,
        audit=FederationControlAuditReceipt(
            audit_id="audit:test",
            command_cid=value.cid,
            authorization_id=auth.authorization_id,
            result_ref=result.cid,
            outcome=outcome,
            control_plane_generation=value.expected_generation,
            fencing_epoch=value.expected_fencing_epoch,
            recorded_at=NOW,
        ),
    )


class Authorizer:
    def __init__(self) -> None:
        self.commands: list[contracts.FederationCommand] = []
        self.override: FederationControlAuthorization | None = None

    def authorize(self, value: contracts.FederationCommand) -> FederationControlAuthorization:
        self.commands.append(value)
        return self.override or authorization(value)


class StateOwner:
    def __init__(self) -> None:
        self.calls: list[tuple[contracts.FederationCommand, FederationControlAuthorization]] = []
        self.override: object | None = None

    def execute_federation_command(
        self, value: contracts.FederationCommand, auth: FederationControlAuthorization
    ) -> FederationControlResponse:
        self.calls.append((value, auth))
        return self.override if self.override is not None else response(value, auth)  # type: ignore[return-value]


def service(*, authorizer_: Authorizer | None = None, owner: StateOwner | None = None) -> tuple[FederationControlService, Authorizer, StateOwner]:
    authorizer_ = authorizer_ or Authorizer()
    owner = owner or StateOwner()
    return (
        FederationControlService(
            authorizer=authorizer_,
            state_owner=owner,
            capability=qualified_federation_control_capability(),
            now=lambda: NOW_EPOCH,
        ),
        authorizer_,
        owner,
    )


def test_service_dispatches_only_a_bound_typed_command_and_audit() -> None:
    value = command()
    control, authorizer_, owner = service()

    observed = execute_federation_command(control, value)

    assert observed.result.outcome == "applied"
    assert observed.audit.command_cid == value.cid
    assert authorizer_.commands == [value]
    assert owner.calls == [(value, authorization(value))]


def test_create_remains_on_authenticated_trigger_gateway() -> None:
    value = command(operation=contracts.FederationOperation.CREATE)
    control, authorizer_, owner = service()

    with pytest.raises(FederationControlAuthorizationError, match="Gateway"):
        control.execute(value)

    assert not authorizer_.commands
    assert not owner.calls


def test_missing_or_unqualified_capability_fails_before_dispatch() -> None:
    authorizer_ = Authorizer()
    owner = StateOwner()
    with pytest.raises(FederationControlCapabilityError):
        FederationControlService(authorizer=authorizer_, state_owner=owner, capability=None)  # type: ignore[arg-type]
    with pytest.raises(FederationControlCapabilityError):
        FederationControlCapability(True, "TypedStateOwnerFederationControl@1", True, False, True, False, False)


def test_capability_is_rechecked_before_each_dispatch() -> None:
    value = command()
    control, authorizer_, owner = service()
    object.__setattr__(control.capability, "quack_transport", False)

    with pytest.raises(FederationControlCapabilityError, match="Quack"):
        control.execute(value)

    assert not authorizer_.commands
    assert not owner.calls


def test_control_authorization_and_audit_are_closed_content_addressed_records() -> None:
    value = command()
    admitted = authorization(value)
    audit = response(value, admitted).audit

    assert FederationControlAuthorization.from_dict(admitted.to_dict()) == admitted
    assert FederationControlAuditReceipt.from_dict(audit.to_dict()) == audit
    assert FederationControlAuthorization.from_dict(admitted.to_dict()).cid == admitted.cid
    assert FederationControlAuditReceipt.from_dict(audit.to_dict()).cid == audit.cid

    authorization_payload = admitted.to_dict()
    authorization_payload["model_policy_override"] = True
    with pytest.raises(contracts.UnknownNormativeFieldError):
        FederationControlAuthorization.from_dict(authorization_payload)

    audit_payload = audit.to_dict()
    audit_payload["model_policy_override"] = True
    with pytest.raises(contracts.UnknownNormativeFieldError):
        FederationControlAuditReceipt.from_dict(audit_payload)


def test_live_commands_require_declared_effects() -> None:
    value = replace(command(), expected_effects=())
    control, authorizer_, owner = service()

    with pytest.raises(FederationControlServiceError, match="expected effects"):
        control.execute(value)

    assert not authorizer_.commands
    assert not owner.calls


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, expected_generation=2),
        lambda value: replace(value, expected_revision=0),
        lambda value: replace(value, target_id="supervisor:test"),
    ],
)
def test_stale_or_cross_target_commands_fail_before_authorization(change: object) -> None:
    value = change(command())  # type: ignore[operator]
    control, authorizer_, owner = service()

    with pytest.raises((FederationControlStaleError, FederationControlServiceError)):
        control.execute(value)

    assert not authorizer_.commands
    assert not owner.calls


@pytest.mark.parametrize(
    "change",
    [
        lambda auth: replace(auth, command_cid="command:other"),
        lambda auth: replace(auth, target_id="federation:other"),
        lambda auth: replace(auth, control_plane_generation=2),
        lambda auth: replace(auth, fencing_epoch=3),
        lambda auth: replace(auth, policy_ref="policy:other"),
    ],
)
def test_authorization_must_bind_command_roots_policy_generation_and_fence(change: object) -> None:
    value = command()
    authorizer_ = Authorizer()
    authorizer_.override = change(authorization(value))  # type: ignore[operator]
    control, _, owner = service(authorizer_=authorizer_)

    with pytest.raises((FederationControlAuthorizationError, FederationControlStaleError)):
        control.execute(value)

    assert not owner.calls


def test_expired_authorization_is_rejected() -> None:
    value = command()
    authorizer_ = Authorizer()
    authorizer_.override = replace(
        authorization(value),
        expires_at="2029-01-01T00:00:00Z",
        decided_at="2028-01-01T00:00:00Z",
    )
    control, _, owner = service(authorizer_=authorizer_)

    with pytest.raises(FederationControlAuthorizationError, match="expired"):
        control.execute(value)

    assert not owner.calls


@pytest.mark.parametrize("dry_run", [False, True])
def test_result_and_audit_outcomes_must_match_dry_run_semantics(dry_run: bool) -> None:
    value = command(dry_run=dry_run)
    control, _, owner = service()
    auth = authorization(value)
    wrong_outcome = "applied" if dry_run else "dry_run"
    invalid = response(value, auth)
    owner.override = replace(
        invalid,
        audit=replace(invalid.audit, outcome=wrong_outcome),
    )

    with pytest.raises(FederationControlResultError):
        control.execute(value)


@pytest.mark.parametrize("outcome", ["failed", "rejected"])
def test_owner_may_return_a_matching_typed_fail_closed_outcome(outcome: str) -> None:
    value = command()
    control, _, owner = service()
    admitted = response(value, authorization(value))
    rejected_result = replace(admitted.result, outcome=outcome)
    owner.override = replace(
        admitted,
        result=rejected_result,
        audit=replace(
            admitted.audit,
            outcome=outcome,
            result_ref=rejected_result.cid,
        ),
    )

    observed = control.execute(value)

    assert observed.result.outcome == outcome
    assert observed.audit.outcome == outcome


@pytest.mark.parametrize(
    "response_change",
    [
        lambda item: replace(item, audit=replace(item.audit, command_cid="command:other")),
        lambda item: replace(item, audit=replace(item.audit, result_ref="result:other")),
        lambda item: replace(item, audit=replace(item.audit, fencing_epoch=3)),
        lambda item: replace(item, audit=replace(item.audit, control_plane_generation=2)),
        lambda item: replace(item, result=replace(item.result, evidence_refs=("evidence:other",))),
    ],
)
def test_owner_response_cannot_lose_command_audit_or_fence(response_change: object) -> None:
    value = command()
    control, _, owner = service()
    owner.override = response_change(response(value, authorization(value)))  # type: ignore[operator]

    with pytest.raises(FederationControlResultError):
        control.execute(value)
