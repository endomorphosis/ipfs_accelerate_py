"""Idempotency, grounded service_id, confirm+auth, and no-op tests for service interaction."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.action_runtime.adapters.service_interaction import (
    DEFAULT_MAX_NOTES_CHARS,
    OPEN_ARGUMENT_SLOTS,
    SCHEDULE_ARGUMENT_SLOTS,
    InMemoryServiceInteractionStore,
    ServiceCallbackRecord,
    ServiceDetailRecord,
    ServiceInteractionActionAdapter,
    ServiceInteractionActionRegistration,
    ServiceInteractionInvocationContext,
    ServiceInteractionSandboxPolicy,
    default_service_interaction_registrations,
    grounded_service_tokens,
    proposal_idempotency_digest,
    require_grounded_service_id,
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


OPEN_ID = "voice.python.open_service_detail.v1"
SCHEDULE_ID = "voice.workflow.schedule_service_callback.v1"
GROUNDED_SERVICE = "svc-housing-211-demo"
GROUNDED_EVIDENCE = (f"service_id:{GROUNDED_SERVICE}", "bafyEvidenceCidExample0001")


def _proposal(
    logical_action: str,
    *,
    arguments: dict[str, str] | None = None,
    tenant_id: str | None = "tenant-a",
    proposal_id: str = "prop-svc-1",
    evidence: tuple[str, ...] | None = None,
    **kwargs: object,
) -> ActionProposal:
    mapping = logical_action_to_descriptor_id()
    base = {
        "proposal_id": proposal_id,
        "descriptor_id": mapping[logical_action],
        "logical_action": logical_action,
        "arguments": arguments or {},
        "route": "service_interaction_support",
        "channel": "voice",
        "tenant_id": tenant_id,
        "confidence": 0.99,
        "source": "test",
        "evidence": evidence if evidence is not None else GROUNDED_EVIDENCE,
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
        decision_id="dec-svc-1",
        kind=kind,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="digest-test",
        arguments_digest=proposal.arguments_digest,
        reason="test_permit",
        risk_class=risk_class,
    )


def _adapter(
    store: InMemoryServiceInteractionStore | None = None,
    *,
    sandbox: ServiceInteractionSandboxPolicy | None = None,
) -> ServiceInteractionActionAdapter:
    policy = sandbox or ServiceInteractionSandboxPolicy()
    regs = (
        ServiceInteractionActionRegistration(
            descriptor_id=OPEN_ID,
            logical_action="open_service_detail",
            sandbox=policy,
        ),
        ServiceInteractionActionRegistration(
            descriptor_id=SCHEDULE_ID,
            logical_action="schedule_service_callback",
            sandbox=policy,
        ),
    )
    return ServiceInteractionActionAdapter(
        regs, store=store or InMemoryServiceInteractionStore()
    )


def _auth_ctx(
    *,
    tenant_id: str = "tenant-a",
    confirmed: bool = True,
    authenticated: bool = True,
) -> ServiceInteractionInvocationContext:
    return ServiceInteractionInvocationContext(
        confirmed=confirmed,
        authenticated=authenticated,
        session_tenant_id=tenant_id,
    )


def _confirm_ctx(
    *,
    tenant_id: str = "tenant-a",
    confirmed: bool = True,
) -> ServiceInteractionInvocationContext:
    """Open path: confirm only (auth optional under pilot policy)."""

    return ServiceInteractionInvocationContext(
        confirmed=confirmed,
        authenticated=False,
        session_tenant_id=tenant_id,
    )


def _seed_catalog(store: InMemoryServiceInteractionStore) -> None:
    store.seed_services(
        ServiceDetailRecord(
            service_id=GROUNDED_SERVICE,
            title="Emergency Housing Intake",
            provider_name="Community Shelter Network",
            program_name="211 Housing",
            summary="SECRET eligibility notes — do not leak in receipts by default",
            status="available",
        ),
        ServiceDetailRecord(
            service_id="svc-other-tenant-only",
            title="Other tenant service",
            provider_name="Hidden Provider",
            summary="LEAK-ME summary",
            tenant_id="tenant-b",
            status="available",
        ),
    )


def test_default_registrations_match_pilot_catalog() -> None:
    regs = default_service_interaction_registrations()
    assert {r.descriptor_id for r in regs} == {OPEN_ID, SCHEDULE_ID}
    mapping = logical_action_to_descriptor_id()
    assert mapping["open_service_detail"] == OPEN_ID
    assert mapping["schedule_service_callback"] == SCHEDULE_ID
    catalog = build_pilot_catalog()
    open_desc = catalog.require(OPEN_ID)
    schedule_desc = catalog.require(SCHEDULE_ID)
    assert open_desc.risk_class is RiskClass.READ
    assert schedule_desc.risk_class is RiskClass.WRITE
    assert schedule_desc.metadata.get("auth_required") == "true"
    assert open_desc.metadata.get("auth_required") == "false"


def test_structured_slots_are_closed_sets() -> None:
    assert "service_id" in SCHEDULE_ARGUMENT_SLOTS
    assert "callback_at" in SCHEDULE_ARGUMENT_SLOTS
    assert "body" not in SCHEDULE_ARGUMENT_SLOTS
    assert "url" not in SCHEDULE_ARGUMENT_SLOTS
    assert "service_id" in OPEN_ARGUMENT_SLOTS
    assert "transcript" not in OPEN_ARGUMENT_SLOTS


def test_grounded_service_tokens_and_require() -> None:
    tokens = grounded_service_tokens(
        ("service_id:svc-1", "svc-2", "service:svc-3", "bafyCid")
    )
    assert "svc-1" in tokens
    assert "svc-2" in tokens
    assert "svc-3" in tokens
    assert "bafyCid" in tokens
    assert require_grounded_service_id("svc-1", ("service_id:svc-1",)) == "svc-1"
    with pytest.raises(ValueError, match="service_id_requires_grounded_evidence"):
        require_grounded_service_id("svc-1", ())
    with pytest.raises(ValueError, match="service_id_not_in_grounded_evidence"):
        require_grounded_service_id("svc-missing", ("service_id:svc-1",))


def test_schedule_denied_without_permitting_decision_is_noop() -> None:
    store = InMemoryServiceInteractionStore()
    _seed_catalog(store)
    adapter = _adapter(store)
    proposal = _proposal(
        "schedule_service_callback",
        arguments={"service_id": GROUNDED_SERVICE, "channel": "phone"},
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
    assert store.list_callbacks(tenant_id="tenant-a") == ()


def test_schedule_requires_confirm_at_adapter_boundary_is_noop() -> None:
    store = InMemoryServiceInteractionStore()
    adapter = _adapter(store)
    proposal = _proposal(
        "schedule_service_callback",
        arguments={"service_id": GROUNDED_SERVICE},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(confirmed=False, authenticated=True),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "confirmation_required" in (receipt.error or "")
    assert store.list_callbacks(tenant_id="tenant-a") == ()


def test_schedule_requires_auth_at_adapter_boundary_is_noop() -> None:
    store = InMemoryServiceInteractionStore()
    adapter = _adapter(store)
    proposal = _proposal(
        "schedule_service_callback",
        arguments={"service_id": GROUNDED_SERVICE},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(confirmed=True, authenticated=False),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "auth_required" in (receipt.error or "")
    assert store.list_callbacks(tenant_id="tenant-a") == ()


def test_schedule_requires_grounded_service_id_not_free_text_alone() -> None:
    store = InMemoryServiceInteractionStore()
    adapter = _adapter(store)
    # Free-text service_id with empty evidence must fail closed.
    free_text = _proposal(
        "schedule_service_callback",
        arguments={"service_id": "invented-from-speech"},
        evidence=(),
    )
    receipt = adapter.invoke(
        proposal=free_text,
        decision=_permit(free_text),
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "service_id_requires_grounded_evidence" in (receipt.error or "")
    assert store.list_callbacks(tenant_id="tenant-a") == ()

    # Evidence present but service_id not among grounded tokens.
    mismatch = _proposal(
        "schedule_service_callback",
        proposal_id="prop-svc-mismatch",
        arguments={"service_id": "invented-from-speech"},
        evidence=("service_id:svc-housing-211-demo",),
    )
    receipt2 = adapter.invoke(
        proposal=mismatch,
        decision=_permit(mismatch),
        context=_auth_ctx(),
    )
    assert receipt2.status is ActionStatus.FAILED
    assert "service_id_not_in_grounded_evidence" in (receipt2.error or "")
    assert store.list_callbacks(tenant_id="tenant-a") == ()


def test_schedule_succeeds_with_confirm_auth_and_redacts_notes() -> None:
    store = InMemoryServiceInteractionStore()
    _seed_catalog(store)
    adapter = _adapter(store)
    notes = "Call after 3pm; ask for intake desk; SSN 111-22-3333"
    proposal = _proposal(
        "schedule_service_callback",
        arguments={
            "service_id": GROUNDED_SERVICE,
            "channel": "phone",
            "callback_at": "2026-08-06T15:00:00Z",
            "client_id": "client-abby",
            "notes": notes,
            "contact_preference": "afternoon",
        },
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED, receipt.to_dict()
    assert receipt.adapter == "service_interaction"
    assert receipt.public_result["ok"] == "true"
    assert receipt.public_result["notes_redacted"] == "true"
    assert "notes" not in receipt.public_result
    assert notes not in str(receipt.to_dict())
    assert "SSN" not in str(receipt.to_dict())
    assert receipt.public_result["notes_digest"] == content_digest(notes)
    assert receipt.public_result["service_id"] == GROUNDED_SERVICE
    assert receipt.public_result["tenant_id"] == "tenant-a"
    assert receipt.public_result["callback_id"].startswith("cb-")
    assert receipt.public_result["idempotent_replay"] == "false"
    assert receipt.public_result["proposal_digest"] == proposal_idempotency_digest(
        proposal
    )

    stored = store.list_callbacks(tenant_id="tenant-a")
    assert len(stored) == 1
    assert stored[0].notes == notes
    assert stored[0].service_id == GROUNDED_SERVICE
    assert stored[0].tenant_id == "tenant-a"


def test_schedule_is_idempotent_on_proposal_digest() -> None:
    store = InMemoryServiceInteractionStore()
    adapter = _adapter(store)
    args = {
        "service_id": GROUNDED_SERVICE,
        "channel": "sms",
        "callback_at": "2026-08-07T10:00:00Z",
        "notes": "first request",
    }
    first = _proposal(
        "schedule_service_callback",
        proposal_id="prop-idem-1",
        arguments=args,
    )
    second = _proposal(
        "schedule_service_callback",
        proposal_id="prop-idem-2",  # different proposal_id is fine
        arguments=dict(args),
    )
    # Same logical content → same idempotency digest.
    assert proposal_idempotency_digest(first) == proposal_idempotency_digest(second)

    r1 = adapter.invoke(
        proposal=first, decision=_permit(first), context=_auth_ctx()
    )
    r2 = adapter.invoke(
        proposal=second, decision=_permit(second), context=_auth_ctx()
    )
    assert r1.status is ActionStatus.SUCCEEDED
    assert r2.status is ActionStatus.SUCCEEDED
    assert r1.public_result["callback_id"] == r2.public_result["callback_id"]
    assert r2.public_result["idempotent_replay"] == "true"
    assert r1.public_result["proposal_digest"] == r2.public_result["proposal_digest"]
    assert len(store.list_callbacks(tenant_id="tenant-a")) == 1

    # Different arguments → new digest → new callback.
    third = _proposal(
        "schedule_service_callback",
        proposal_id="prop-idem-3",
        arguments={
            "service_id": GROUNDED_SERVICE,
            "channel": "email",
            "callback_at": "2026-08-08T10:00:00Z",
        },
    )
    r3 = adapter.invoke(
        proposal=third, decision=_permit(third), context=_auth_ctx()
    )
    assert r3.status is ActionStatus.SUCCEEDED
    assert r3.public_result["callback_id"] != r1.public_result["callback_id"]
    assert r3.public_result["idempotent_replay"] == "false"
    assert len(store.list_callbacks(tenant_id="tenant-a")) == 2


def test_schedule_rejects_missing_service_id_and_bad_channel() -> None:
    adapter = _adapter()
    missing = _proposal("schedule_service_callback", arguments={"channel": "phone"})
    receipt = adapter.invoke(
        proposal=missing, decision=_permit(missing), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert "missing required argument slot 'service_id'" in (receipt.error or "")

    bad_channel = _proposal(
        "schedule_service_callback",
        proposal_id="prop-ch",
        arguments={"service_id": GROUNDED_SERVICE, "channel": "carrier-pigeon"},
    )
    receipt2 = adapter.invoke(
        proposal=bad_channel, decision=_permit(bad_channel), context=_auth_ctx()
    )
    assert receipt2.status is ActionStatus.FAILED
    assert "unsupported_channel" in (receipt2.error or "")


def test_schedule_rejects_forbidden_and_unexpected_slots() -> None:
    adapter = _adapter()
    # ``webhook`` is adapter-forbidden but not banned at ActionProposal construction
    # (unlike ``url`` / ``command``, which fail earlier in contracts).
    forbidden = _proposal(
        "schedule_service_callback",
        arguments={
            "service_id": GROUNDED_SERVICE,
            "webhook": "https://evil.example/callback",
        },
    )
    receipt = adapter.invoke(
        proposal=forbidden, decision=_permit(forbidden), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert "forbidden" in (receipt.error or "")

    unexpected = _proposal(
        "schedule_service_callback",
        proposal_id="prop-unexpected",
        arguments={
            "service_id": GROUNDED_SERVICE,
            "free_text_blob": "do not accept",
        },
    )
    receipt2 = adapter.invoke(
        proposal=unexpected, decision=_permit(unexpected), context=_auth_ctx()
    )
    assert receipt2.status is ActionStatus.FAILED
    assert "unexpected arguments" in (receipt2.error or "")


def test_schedule_notes_length_bounded() -> None:
    adapter = _adapter(sandbox=ServiceInteractionSandboxPolicy(max_notes_chars=32))
    proposal = _proposal(
        "schedule_service_callback",
        arguments={"service_id": GROUNDED_SERVICE, "notes": "x" * 33},
    )
    receipt = adapter.invoke(
        proposal=proposal, decision=_permit(proposal), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert "notes_exceeds_max_chars" in (receipt.error or "")


def test_schedule_is_tenant_isolated() -> None:
    store = InMemoryServiceInteractionStore()
    adapter = _adapter(store)
    # Seed a callback for tenant-b under a known digest.
    store.seed_callbacks(
        ServiceCallbackRecord(
            callback_id="cb-other",
            tenant_id="tenant-b",
            service_id=GROUNDED_SERVICE,
            proposal_digest="other-digest",
            channel="phone",
            callback_at="",
            client_id="client-b",
            notes="LEAK-ME callback notes",
            status="scheduled",
            created_at_epoch_s=1_700_000_100.0,
        )
    )
    proposal = _proposal(
        "schedule_service_callback",
        tenant_id="tenant-a",
        arguments={"service_id": GROUNDED_SERVICE, "channel": "phone"},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(tenant_id="tenant-a"),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    a_rows = store.list_callbacks(tenant_id="tenant-a")
    b_rows = store.list_callbacks(tenant_id="tenant-b")
    assert len(a_rows) == 1
    assert a_rows[0].tenant_id == "tenant-a"
    assert all(c.tenant_id == "tenant-b" for c in b_rows)
    assert "LEAK-ME" not in str(receipt.to_dict())


def test_schedule_tenant_session_mismatch_fails_closed() -> None:
    store = InMemoryServiceInteractionStore()
    adapter = _adapter(store)
    proposal = _proposal(
        "schedule_service_callback",
        tenant_id="tenant-a",
        arguments={"service_id": GROUNDED_SERVICE},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(tenant_id="tenant-b"),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "tenant_session_mismatch" in (receipt.error or "")
    assert store.list_callbacks(tenant_id="tenant-a") == ()
    assert store.list_callbacks(tenant_id="tenant-b") == ()


def test_open_returns_redacted_detail_when_grounded() -> None:
    store = InMemoryServiceInteractionStore()
    _seed_catalog(store)
    adapter = _adapter(store)
    proposal = _proposal(
        "open_service_detail",
        arguments={"service_id": GROUNDED_SERVICE},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_confirm_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED, receipt.to_dict()
    assert receipt.public_result["found"] == "true"
    assert receipt.public_result["service_id"] == GROUNDED_SERVICE
    assert receipt.public_result["title"] == "Emergency Housing Intake"
    assert receipt.public_result["summary_redacted"] == "true"
    assert "summary" not in receipt.public_result
    assert "SECRET eligibility" not in str(receipt.to_dict())
    assert "Emergency Housing Intake" in receipt.public_result["redacted_summary"]


def test_open_requires_grounded_service_id() -> None:
    store = InMemoryServiceInteractionStore()
    _seed_catalog(store)
    adapter = _adapter(store)
    proposal = _proposal(
        "open_service_detail",
        arguments={"service_id": "not-grounded"},
        evidence=("service_id:svc-housing-211-demo",),
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_confirm_ctx(),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "service_id_not_in_grounded_evidence" in (receipt.error or "")


def test_open_requires_confirm_but_not_auth() -> None:
    store = InMemoryServiceInteractionStore()
    _seed_catalog(store)
    adapter = _adapter(store)
    proposal = _proposal(
        "open_service_detail",
        arguments={"service_id": GROUNDED_SERVICE},
    )
    decision = _permit(
        proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
    )

    unconfirmed = adapter.invoke(
        proposal=proposal,
        decision=decision,
        context=_confirm_ctx(confirmed=False),
    )
    assert unconfirmed.status is ActionStatus.FAILED
    assert "confirmation_required" in (unconfirmed.error or "")

    confirmed_no_auth = adapter.invoke(
        proposal=proposal,
        decision=decision,
        context=_confirm_ctx(confirmed=True),
    )
    assert confirmed_no_auth.status is ActionStatus.SUCCEEDED


def test_open_not_found_for_other_tenant_catalog_row() -> None:
    store = InMemoryServiceInteractionStore()
    _seed_catalog(store)
    adapter = _adapter(store)
    proposal = _proposal(
        "open_service_detail",
        arguments={"service_id": "svc-other-tenant-only"},
        evidence=("service_id:svc-other-tenant-only",),
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_confirm_ctx(tenant_id="tenant-a"),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    assert receipt.public_result["found"] == "false"
    assert "LEAK-ME" not in str(receipt.to_dict())


def test_arguments_digest_mismatch_fails_closed() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "schedule_service_callback",
        arguments={"service_id": GROUNDED_SERVICE},
    )
    decision = _permit(proposal)
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
    adapter = ServiceInteractionActionAdapter([])
    proposal = _proposal(
        "schedule_service_callback",
        arguments={"service_id": GROUNDED_SERVICE},
    )
    receipt = adapter.invoke(
        proposal=proposal, decision=_permit(proposal), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert receipt.error == "no_service_interaction_registration"


def test_default_bounds_are_sane() -> None:
    assert 1 <= DEFAULT_MAX_NOTES_CHARS <= 16_384
    policy = ServiceInteractionSandboxPolicy()
    assert policy.max_notes_chars == DEFAULT_MAX_NOTES_CHARS
    assert policy.redact_notes_in_receipts is True
    assert policy.require_auth_for_schedule is True
    assert policy.require_confirm_for_schedule is True
    assert policy.require_auth_for_open is False
    assert policy.require_confirm_for_open is True


def test_pilot_policy_schedule_requires_confirm_and_auth() -> None:
    """End-to-end with pilot policy: schedule needs confirm+auth before permit."""

    catalog = build_pilot_catalog()
    policy = PilotPolicy(catalog=catalog)
    store = InMemoryServiceInteractionStore()
    adapter = ServiceInteractionActionAdapter(
        default_service_interaction_registrations(), store=store
    )

    proposal = _proposal(
        "schedule_service_callback",
        arguments={"service_id": GROUNDED_SERVICE, "channel": "phone"},
    )

    unconfirmed = policy.decide(proposal, PilotAdmissionContext())
    assert unconfirmed.kind is ActionDecisionKind.CONFIRM
    assert not unconfirmed.permits_execution
    denied = adapter.invoke(
        proposal=proposal, decision=unconfirmed, context=_auth_ctx()
    )
    assert denied.status is ActionStatus.DENIED
    assert store.list_callbacks(tenant_id="tenant-a") == ()

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
    assert "notes" not in receipt.public_result
    assert len(store.list_callbacks(tenant_id="tenant-a")) == 1

    # Replay through policy+adapter remains idempotent.
    replay = adapter.invoke(
        proposal=proposal,
        decision=admitted,
        context=_auth_ctx(),
    )
    assert replay.status is ActionStatus.SUCCEEDED
    assert replay.public_result["idempotent_replay"] == "true"
    assert len(store.list_callbacks(tenant_id="tenant-a")) == 1


def test_pilot_policy_open_requires_confirm_only() -> None:
    catalog = build_pilot_catalog()
    policy = PilotPolicy(catalog=catalog)
    store = InMemoryServiceInteractionStore()
    _seed_catalog(store)
    adapter = ServiceInteractionActionAdapter(
        default_service_interaction_registrations(), store=store
    )
    proposal = _proposal(
        "open_service_detail",
        arguments={"service_id": GROUNDED_SERVICE},
    )

    unconfirmed = policy.decide(proposal, PilotAdmissionContext())
    assert unconfirmed.kind is ActionDecisionKind.CONFIRM

    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_READ
    receipt = adapter.invoke(
        proposal=proposal,
        decision=admitted,
        context=_confirm_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    assert receipt.public_result["found"] == "true"
