"""Tenant isolation, confirm+auth, body bounds, and redaction tests for messaging."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.action_runtime.adapters.messaging import (
    DEFAULT_MAX_BODY_CHARS,
    InMemoryProviderMessageStore,
    MessagingActionAdapter,
    MessagingActionRegistration,
    MessagingInvocationContext,
    MessagingSandboxPolicy,
    ProviderMessageRecord,
    default_messaging_registrations,
)
from ipfs_accelerate_py.action_runtime.catalog_211ai import (
    build_pilot_catalog,
    logical_action_to_descriptor_id,
)
from ipfs_accelerate_py.action_runtime.contracts import (
    ActionDecision,
    ActionDecisionKind,
    ActionProposal,
    ActionStatus,
    RiskClass,
    content_digest,
)
from ipfs_accelerate_py.action_runtime.policy_pilot import (
    PilotAdmissionContext,
    PilotPolicy,
)


READ_ID = "voice.python.read_provider_messages.v1"
LEAVE_ID = "voice.python.leave_provider_message.v1"


def _proposal(
    logical_action: str,
    *,
    arguments: dict[str, str] | None = None,
    tenant_id: str | None = "tenant-a",
    proposal_id: str = "prop-msg-1",
    **kwargs: object,
) -> ActionProposal:
    mapping = logical_action_to_descriptor_id()
    base = {
        "proposal_id": proposal_id,
        "descriptor_id": mapping[logical_action],
        "logical_action": logical_action,
        "arguments": arguments or {},
        "route": "provider_contact_support",
        "channel": "voice",
        "tenant_id": tenant_id,
        "confidence": 0.99,
        "source": "test",
    }
    base.update(kwargs)
    return ActionProposal(**base)  # type: ignore[arg-type]


def _permit(
    proposal: ActionProposal,
    *,
    kind: ActionDecisionKind = ActionDecisionKind.PERMIT_EXECUTE,
    risk_class: RiskClass = RiskClass.WRITE,
) -> ActionDecision:
    return ActionDecision(
        decision_id="dec-msg-1",
        kind=kind,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="digest-test",
        arguments_digest=proposal.arguments_digest,
        reason="test_permit",
        risk_class=risk_class,
    )


def _adapter(
    store: InMemoryProviderMessageStore | None = None,
    *,
    sandbox: MessagingSandboxPolicy | None = None,
) -> MessagingActionAdapter:
    policy = sandbox or MessagingSandboxPolicy()
    regs = (
        MessagingActionRegistration(
            descriptor_id=READ_ID,
            logical_action="read_provider_messages",
            sandbox=policy,
        ),
        MessagingActionRegistration(
            descriptor_id=LEAVE_ID,
            logical_action="leave_provider_message",
            sandbox=policy,
        ),
    )
    return MessagingActionAdapter(regs, store=store or InMemoryProviderMessageStore())


def _auth_ctx(
    *,
    tenant_id: str = "tenant-a",
    confirmed: bool = True,
    authenticated: bool = True,
) -> MessagingInvocationContext:
    return MessagingInvocationContext(
        confirmed=confirmed,
        authenticated=authenticated,
        session_tenant_id=tenant_id,
    )


def _seed_cross_tenant(store: InMemoryProviderMessageStore) -> None:
    store.seed(
        ProviderMessageRecord(
            message_id="msg-a-1",
            tenant_id="tenant-a",
            provider_id="provider-rose",
            client_id="client-abby",
            channel="in_app",
            subject="Intake reminder",
            body="SECRET body for tenant-a only",
            direction="inbound",
            status="sent",
            created_at_epoch_s=1_700_000_100.0,
        ),
        ProviderMessageRecord(
            message_id="msg-b-1",
            tenant_id="tenant-b",
            provider_id="provider-rose",
            client_id="client-casey",
            channel="sms",
            subject="Other tenant",
            body="LEAK-ME body for tenant-b",
            direction="inbound",
            status="sent",
            created_at_epoch_s=1_700_000_200.0,
        ),
    )


def test_default_registrations_match_pilot_catalog() -> None:
    regs = default_messaging_registrations()
    assert {r.descriptor_id for r in regs} == {READ_ID, LEAVE_ID}
    mapping = logical_action_to_descriptor_id()
    assert mapping["read_provider_messages"] == READ_ID
    assert mapping["leave_provider_message"] == LEAVE_ID


def test_leave_denied_without_permitting_decision() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "leave_provider_message",
        arguments={"provider_id": "provider-rose", "body": "Please call me back."},
    )
    decision = ActionDecision(
        decision_id="dec-deny",
        kind=ActionDecisionKind.CONFIRM,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="d",
        arguments_digest=proposal.arguments_digest,
        reason="confirmation_required",
        risk_class=RiskClass.WRITE,
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=decision,
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.DENIED
    assert "does_not_permit" in (receipt.error or "")
    assert isinstance(adapter.store, InMemoryProviderMessageStore)
    assert adapter.store.list_messages(tenant_id="tenant-a") == ()


def test_leave_requires_confirm_at_adapter_boundary() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "leave_provider_message",
        arguments={"provider_id": "provider-rose", "body": "Please call me back."},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(confirmed=False, authenticated=True),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "confirmation_required" in (receipt.error or "")


def test_leave_requires_auth_at_adapter_boundary() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "leave_provider_message",
        arguments={"provider_id": "provider-rose", "body": "Please call me back."},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(confirmed=True, authenticated=False),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "auth_required" in (receipt.error or "")


def test_leave_succeeds_with_confirm_and_auth_and_redacts_body() -> None:
    store = InMemoryProviderMessageStore()
    adapter = _adapter(store)
    body = "Please leave a note that I need a transportation voucher."
    proposal = _proposal(
        "leave_provider_message",
        arguments={
            "provider_id": "provider-rose",
            "client_id": "client-abby",
            "channel": "in_app",
            "subject": "Voucher follow-up",
            "body": body,
        },
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED, receipt.to_dict()
    assert receipt.adapter == "messaging"
    assert receipt.public_result["ok"] == "true"
    assert receipt.public_result["bodies_redacted"] == "true"
    assert "body" not in receipt.public_result
    assert body not in str(receipt.to_dict())
    assert receipt.public_result["body_digest"] == content_digest(body)
    assert receipt.public_result["provider_id"] == "provider-rose"
    assert receipt.public_result["tenant_id"] == "tenant-a"

    stored = store.list_messages(tenant_id="tenant-a")
    assert len(stored) == 1
    assert stored[0].body == body
    assert stored[0].tenant_id == "tenant-a"


def test_leave_body_length_bounded() -> None:
    adapter = _adapter(sandbox=MessagingSandboxPolicy(max_body_chars=32))
    proposal = _proposal(
        "leave_provider_message",
        arguments={
            "provider_id": "provider-rose",
            "body": "x" * 33,
        },
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "body_exceeds_max_chars" in (receipt.error or "")

    ok_proposal = _proposal(
        "leave_provider_message",
        proposal_id="prop-msg-ok",
        arguments={
            "provider_id": "provider-rose",
            "body": "x" * 32,
        },
    )
    ok_receipt = adapter.invoke(
        proposal=ok_proposal,
        decision=_permit(ok_proposal),
        context=_auth_ctx(),
    )
    assert ok_receipt.status is ActionStatus.SUCCEEDED


def test_leave_rejects_empty_body_and_bad_channel() -> None:
    adapter = _adapter()
    empty = _proposal(
        "leave_provider_message",
        arguments={"provider_id": "provider-rose", "body": "   "},
    )
    receipt = adapter.invoke(
        proposal=empty, decision=_permit(empty), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert "non-empty" in (receipt.error or "")

    bad_channel = _proposal(
        "leave_provider_message",
        proposal_id="prop-ch",
        arguments={
            "provider_id": "provider-rose",
            "body": "hello",
            "channel": "carrier-pigeon",
        },
    )
    receipt2 = adapter.invoke(
        proposal=bad_channel, decision=_permit(bad_channel), context=_auth_ctx()
    )
    assert receipt2.status is ActionStatus.FAILED
    assert "unsupported_channel" in (receipt2.error or "")


def test_default_max_body_chars_is_bounded() -> None:
    assert 1 <= DEFAULT_MAX_BODY_CHARS <= 16_384
    policy = MessagingSandboxPolicy()
    assert policy.max_body_chars == DEFAULT_MAX_BODY_CHARS
    assert policy.redact_bodies_in_receipts is True


def test_read_is_tenant_scoped() -> None:
    store = InMemoryProviderMessageStore()
    _seed_cross_tenant(store)
    adapter = _adapter(store)

    proposal = _proposal("read_provider_messages", arguments={})
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_auth_ctx(tenant_id="tenant-a"),
    )
    assert receipt.status is ActionStatus.SUCCEEDED, receipt.to_dict()
    assert receipt.public_result["message_count"] == "1"
    assert "msg-a-1" in receipt.public_result["message_ids"]
    assert "msg-b-1" not in receipt.public_result["message_ids"]
    assert "LEAK-ME" not in str(receipt.to_dict())
    assert "SECRET body" not in str(receipt.to_dict())
    assert receipt.public_result["bodies_redacted"] == "true"
    assert "body" not in receipt.public_result


def test_read_cannot_select_other_tenant_via_session_mismatch() -> None:
    store = InMemoryProviderMessageStore()
    _seed_cross_tenant(store)
    adapter = _adapter(store)
    proposal = _proposal("read_provider_messages", tenant_id="tenant-a")
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_auth_ctx(tenant_id="tenant-b"),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "tenant_session_mismatch" in (receipt.error or "")


def test_read_requires_confirm_and_auth() -> None:
    store = InMemoryProviderMessageStore()
    _seed_cross_tenant(store)
    adapter = _adapter(store)
    proposal = _proposal("read_provider_messages")
    decision = _permit(
        proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
    )

    unconfirmed = adapter.invoke(
        proposal=proposal,
        decision=decision,
        context=_auth_ctx(confirmed=False, authenticated=True),
    )
    assert unconfirmed.status is ActionStatus.FAILED
    assert "confirmation_required" in (unconfirmed.error or "")

    unauth = adapter.invoke(
        proposal=proposal,
        decision=decision,
        context=_auth_ctx(confirmed=True, authenticated=False),
    )
    assert unauth.status is ActionStatus.FAILED
    assert "auth_required" in (unauth.error or "")


def test_read_filters_by_provider_without_crossing_tenants() -> None:
    store = InMemoryProviderMessageStore()
    _seed_cross_tenant(store)
    store.seed(
        ProviderMessageRecord(
            message_id="msg-a-2",
            tenant_id="tenant-a",
            provider_id="provider-other",
            client_id="client-abby",
            channel="email",
            subject="Other provider",
            body="body-a-2",
            direction="inbound",
            status="sent",
            created_at_epoch_s=1_700_000_300.0,
        )
    )
    adapter = _adapter(store)
    proposal = _proposal(
        "read_provider_messages",
        arguments={"provider_id": "provider-rose"},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    assert receipt.public_result["message_count"] == "1"
    assert receipt.public_result["message_ids"] == "msg-a-1"
    assert receipt.public_result["provider_id"] == "provider-rose"


def test_arguments_digest_mismatch_fails_closed() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "leave_provider_message",
        arguments={"provider_id": "provider-rose", "body": "hello"},
    )
    decision = _permit(proposal)
    # Forge a digest that does not match the proposal arguments.
    decision = ActionDecision(
        decision_id=decision.decision_id,
        kind=decision.kind,
        proposal_id=decision.proposal_id,
        descriptor_id=decision.descriptor_id,
        descriptor_digest=decision.descriptor_digest,
        arguments_digest="0" * 64,
        reason=decision.reason,
        risk_class=decision.risk_class,
    )
    receipt = adapter.invoke(
        proposal=proposal, decision=decision, context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert receipt.error == "arguments_digest_mismatch"


def test_no_registration_fails_closed() -> None:
    adapter = MessagingActionAdapter([])
    proposal = _proposal(
        "leave_provider_message",
        arguments={"provider_id": "provider-rose", "body": "hello"},
    )
    receipt = adapter.invoke(
        proposal=proposal, decision=_permit(proposal), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert receipt.error == "no_messaging_registration"


def test_pilot_policy_leave_requires_confirm_and_auth() -> None:
    """End-to-end with pilot policy: leave needs confirm+auth before permit."""

    catalog = build_pilot_catalog()
    # Wall-clock now so decision TTL is still valid when the adapter runs.
    policy = PilotPolicy(catalog=catalog)
    store = InMemoryProviderMessageStore()
    adapter = MessagingActionAdapter(default_messaging_registrations(), store=store)

    proposal = _proposal(
        "leave_provider_message",
        arguments={"provider_id": "provider-rose", "body": "Need callback today."},
    )

    unconfirmed = policy.decide(proposal, PilotAdmissionContext())
    assert unconfirmed.kind is ActionDecisionKind.CONFIRM
    assert not unconfirmed.permits_execution
    denied = adapter.invoke(proposal=proposal, decision=unconfirmed, context=_auth_ctx())
    assert denied.status is ActionStatus.DENIED

    confirmed_no_auth = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    assert confirmed_no_auth.kind is ActionDecisionKind.DENY
    assert confirmed_no_auth.reason == "auth_required"

    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id="tenant-a",
        ),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_EXECUTE
    assert admitted.permits_execution
    receipt = adapter.invoke(
        proposal=proposal,
        decision=admitted,
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    assert "body" not in receipt.public_result
    assert len(store.list_messages(tenant_id="tenant-a")) == 1


def test_pilot_policy_read_requires_confirm_and_auth() -> None:
    catalog = build_pilot_catalog()
    policy = PilotPolicy(catalog=catalog)
    store = InMemoryProviderMessageStore()
    _seed_cross_tenant(store)
    adapter = MessagingActionAdapter(default_messaging_registrations(), store=store)
    proposal = _proposal("read_provider_messages")

    unconfirmed = policy.decide(proposal, PilotAdmissionContext())
    assert unconfirmed.kind is ActionDecisionKind.CONFIRM

    confirmed_no_auth = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    assert confirmed_no_auth.kind is ActionDecisionKind.DENY
    assert confirmed_no_auth.reason == "auth_required"

    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id="tenant-a",
        ),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_READ
    receipt = adapter.invoke(
        proposal=proposal,
        decision=admitted,
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    assert receipt.public_result["message_count"] == "1"
    assert "LEAK-ME" not in str(receipt.to_dict())


def test_sandbox_policy_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="max_body_chars"):
        MessagingSandboxPolicy(max_body_chars=0)
    with pytest.raises(ValueError, match="max_messages_returned"):
        MessagingSandboxPolicy(max_messages_returned=0)


def test_duplicate_registration_rejected() -> None:
    reg = MessagingActionRegistration(
        descriptor_id=READ_ID,
        logical_action="read_provider_messages",
    )
    with pytest.raises(ValueError, match="duplicate"):
        MessagingActionAdapter([reg, reg])


def test_receipt_public_result_values_are_strings() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "leave_provider_message",
        arguments={"provider_id": "provider-rose", "body": "hello world"},
    )
    receipt = adapter.invoke(
        proposal=proposal, decision=_permit(proposal), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    for key, value in receipt.public_result.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
