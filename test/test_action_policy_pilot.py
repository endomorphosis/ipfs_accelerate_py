"""Pilot policy matrix tests: default deny, confirm/auth, handoff, safety isolation."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.action_runtime.catalog import ActionCatalog, ActionDescriptor
from ipfs_accelerate_py.action_runtime.catalog_211ai import (
    build_pilot_catalog,
    logical_action_to_descriptor_id,
)
from ipfs_accelerate_py.action_runtime.contracts import (
    ActionDecisionKind,
    ActionProposal,
    RiskClass,
    SideEffectClass,
)
from ipfs_accelerate_py.action_runtime.policy_pilot import (
    POLICY_REVISION,
    PilotAdmissionContext,
    PilotPolicy,
    build_pilot_policy,
    descriptor_requires_auth,
    is_handoff_descriptor,
    is_safety_descriptor,
)


def _proposal(
    logical_action: str,
    *,
    confidence: float = 0.0,
    tenant_id: str | None = "211-ai",
    channel: str | None = "voice",
    descriptor_id: str | None = None,
    **kwargs: object,
) -> ActionProposal:
    mapping = logical_action_to_descriptor_id()
    desc_id = descriptor_id or mapping[logical_action]
    base = {
        "proposal_id": f"prop-{logical_action}",
        "descriptor_id": desc_id,
        "logical_action": logical_action,
        "arguments": {},
        "route": "test_route",
        "channel": channel,
        "tenant_id": tenant_id,
        "confidence": confidence,
        "source": "test",
    }
    base.update(kwargs)
    return ActionProposal(**base)  # type: ignore[arg-type]


@pytest.fixture
def catalog() -> ActionCatalog:
    return build_pilot_catalog()


@pytest.fixture
def policy(catalog: ActionCatalog) -> PilotPolicy:
    return PilotPolicy(catalog=catalog, now=lambda: 1_700_000_000.0)


def test_build_pilot_policy_defaults_to_pilot_catalog() -> None:
    policy = build_pilot_policy()
    assert policy.policy_revision == POLICY_REVISION
    decision = policy.decide(_proposal("open_app_surface"))
    assert decision.kind is ActionDecisionKind.CONFIRM


def test_default_deny_unknown_descriptor(policy: PilotPolicy) -> None:
    proposal = ActionProposal(
        proposal_id="prop-unknown",
        descriptor_id="voice.python.not_registered.v1",
        logical_action="not_registered",
        confidence=0.99,
        channel="voice",
        tenant_id="211-ai",
    )
    decision = policy.decide(proposal)
    assert decision.kind is ActionDecisionKind.DENY
    assert decision.reason == "unknown_descriptor"
    assert not decision.permits_execution


def test_default_deny_logical_action_mismatch(policy: PilotPolicy) -> None:
    mapping = logical_action_to_descriptor_id()
    proposal = ActionProposal(
        proposal_id="prop-mismatch",
        descriptor_id=mapping["open_app_surface"],
        logical_action="create_calendar_reminder",  # does not match descriptor
        confidence=1.0,
        channel="voice",
        tenant_id="211-ai",
    )
    decision = policy.decide(proposal)
    assert decision.kind is ActionDecisionKind.DENY
    assert decision.reason == "logical_action_mismatch"
    assert not decision.permits_execution


def test_default_deny_channel_and_tenant(policy: PilotPolicy) -> None:
    bad_channel = _proposal("open_app_surface", channel="sms-unlisted")
    decision = policy.decide(bad_channel)
    assert decision.kind is ActionDecisionKind.DENY
    assert decision.reason == "channel_not_allowed"

    # Tenant restriction: register a narrow descriptor and deny outsiders.
    narrow = ActionDescriptor(
        descriptor_id="voice.python.narrow_read.v1",
        logical_action="open_app_surface",
        adapter="python",
        risk_class=RiskClass.READ,
        side_effect_class=SideEffectClass.LOCAL_READ,
        requires_confirmation=True,
        allowed_channels=("voice",),
        allowed_tenants=("tenant-a",),
    )
    narrow_policy = PilotPolicy(catalog=ActionCatalog([narrow]), now=lambda: 0.0)
    outsider = ActionProposal(
        proposal_id="prop-tenant",
        descriptor_id=narrow.descriptor_id,
        logical_action="open_app_surface",
        tenant_id="tenant-b",
        channel="voice",
    )
    decision = narrow_policy.decide(outsider)
    assert decision.kind is ActionDecisionKind.DENY
    assert decision.reason == "tenant_not_allowed"


def test_confirm_for_read_then_permit(policy: PilotPolicy) -> None:
    proposal = _proposal("open_app_surface", confidence=0.95)
    pending = policy.decide(proposal)
    assert pending.kind is ActionDecisionKind.CONFIRM
    assert pending.reason == "confirmation_required"
    assert pending.risk_class is RiskClass.READ
    assert not pending.permits_execution

    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_READ
    assert admitted.reason == "read_confirmed"
    assert admitted.permits_execution
    assert admitted.risk_class is RiskClass.READ


def test_read_auth_gated_requires_auth_after_confirm(policy: PilotPolicy) -> None:
    proposal = _proposal("read_provider_messages")
    mapping = logical_action_to_descriptor_id()
    descriptor = policy.catalog.require(mapping["read_provider_messages"])
    assert descriptor_requires_auth(descriptor)

    pending = policy.decide(proposal)
    assert pending.kind is ActionDecisionKind.CONFIRM

    confirmed_only = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    assert confirmed_only.kind is ActionDecisionKind.DENY
    assert confirmed_only.reason == "auth_required"
    assert not confirmed_only.permits_execution

    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id="211-ai",
        ),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_READ
    assert admitted.reason == "read_confirmed_authenticated"
    assert admitted.permits_execution


def test_write_requires_auth_and_confirm(policy: PilotPolicy) -> None:
    proposal = _proposal("create_calendar_reminder", confidence=0.99)

    pending = policy.decide(proposal)
    assert pending.kind is ActionDecisionKind.CONFIRM
    assert pending.reason == "confirmation_required"
    assert not pending.permits_execution

    confirmed_no_auth = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    assert confirmed_no_auth.kind is ActionDecisionKind.DENY
    assert confirmed_no_auth.reason == "auth_required"
    assert confirmed_no_auth.risk_class is RiskClass.WRITE
    assert not confirmed_no_auth.permits_execution

    auth_no_confirm = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=False,
            authenticated=True,
            session_tenant_id="211-ai",
        ),
    )
    assert auth_no_confirm.kind is ActionDecisionKind.CONFIRM
    assert not auth_no_confirm.permits_execution

    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id="211-ai",
        ),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_EXECUTE
    assert admitted.reason == "write_confirmed_authenticated"
    assert admitted.permits_execution
    assert admitted.risk_class is RiskClass.WRITE


def test_write_auth_tenant_mismatch_denies(policy: PilotPolicy) -> None:
    proposal = _proposal("leave_provider_message", tenant_id="211-ai")
    decision = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id="other-tenant",
        ),
    )
    assert decision.kind is ActionDecisionKind.DENY
    assert decision.reason == "auth_required"


def test_write_auth_missing_session_tenant_denies(policy: PilotPolicy) -> None:
    proposal = _proposal("schedule_service_callback", tenant_id="211-ai")
    decision = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id=None,
        ),
    )
    assert decision.kind is ActionDecisionKind.DENY
    assert decision.reason == "auth_required"


def test_handoff_policy_path(policy: PilotPolicy) -> None:
    proposal = _proposal("handoff_live_agent")
    mapping = logical_action_to_descriptor_id()
    descriptor = policy.catalog.require(mapping["handoff_live_agent"])
    assert is_handoff_descriptor(descriptor)

    # Auto request under handoff policy (no confirm required for request creation).
    auto = policy.decide(proposal)
    assert auto.kind is ActionDecisionKind.HANDOFF
    assert auto.reason == "handoff_policy_request"
    assert not auto.permits_execution
    assert auto.risk_class is RiskClass.HUMAN

    confirmed = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True),
    )
    assert confirmed.kind is ActionDecisionKind.HANDOFF
    assert confirmed.reason == "handoff_confirmed_request"
    assert not confirmed.permits_execution


def test_handoff_requires_confirm_when_auto_disabled(catalog: ActionCatalog) -> None:
    policy = PilotPolicy(
        catalog=catalog,
        handoff_auto_request=False,
        now=lambda: 0.0,
    )
    proposal = _proposal("handoff_live_agent")
    pending = policy.decide(proposal)
    assert pending.kind is ActionDecisionKind.CONFIRM
    admitted = policy.decide(proposal, PilotAdmissionContext(confirmed=True))
    assert admitted.kind is ActionDecisionKind.HANDOFF


def test_safety_overlay_forces_escalate_only(policy: PilotPolicy) -> None:
    safety = _proposal("escalate_safety", confidence=0.1)
    mapping = logical_action_to_descriptor_id()
    descriptor = policy.catalog.require(mapping["escalate_safety"])
    assert is_safety_descriptor(descriptor)

    forced = policy.decide(
        safety,
        PilotAdmissionContext(safety_overlay=True),
    )
    assert forced.kind is ActionDecisionKind.HANDOFF
    assert forced.reason == "safety_overlay_force_escalate"
    assert not forced.permits_execution
    assert forced.risk_class is RiskClass.HUMAN

    # Policy-driven path without overlay still admits safety handoff only.
    policy_path = policy.decide(safety)
    assert policy_path.kind is ActionDecisionKind.HANDOFF
    assert policy_path.reason == "safety_policy_handoff"


def test_safety_overlay_cannot_widen_to_arbitrary_descriptors(policy: PilotPolicy) -> None:
    """Safety overlay must not open app/write tools (confused deputy)."""

    overlay = PilotAdmissionContext(safety_overlay=True, confirmed=False)

    for logical in (
        "open_app_surface",
        "create_calendar_reminder",
        "leave_provider_message",
        "schedule_service_callback",
        "read_calendar",
    ):
        decision = policy.decide(_proposal(logical, confidence=1.0), overlay)
        assert decision.kind is not ActionDecisionKind.PERMIT_EXECUTE, logical
        assert decision.kind is not ActionDecisionKind.PERMIT_READ, logical
        # Writes/reads still sit behind confirm (or deny), never auto-permit.
        assert decision.kind in {
            ActionDecisionKind.CONFIRM,
            ActionDecisionKind.DENY,
            ActionDecisionKind.HANDOFF,
        }
        if logical != "escalate_safety":
            assert decision.reason != "safety_overlay_force_escalate", logical

    # Even with overlay + confirm, write still needs auth (overlay does not widen).
    write = policy.decide(
        _proposal("create_calendar_reminder", confidence=1.0),
        PilotAdmissionContext(safety_overlay=True, confirmed=True, authenticated=False),
    )
    assert write.kind is ActionDecisionKind.DENY
    assert write.reason == "auth_required"


def test_confidence_cannot_upgrade_authority(policy: PilotPolicy) -> None:
    high = _proposal("open_app_surface", confidence=1.0)
    low = _proposal("open_app_surface", confidence=0.0)

    high_decision = policy.decide(high)
    low_decision = policy.decide(low)
    assert high_decision.kind is ActionDecisionKind.CONFIRM
    assert low_decision.kind is ActionDecisionKind.CONFIRM
    assert high_decision.reason == low_decision.reason == "confirmation_required"

    write_high = policy.decide(
        _proposal("create_calendar_reminder", confidence=1.0),
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    write_low = policy.decide(
        _proposal("create_calendar_reminder", confidence=0.0),
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    assert write_high.kind is ActionDecisionKind.DENY
    assert write_low.kind is ActionDecisionKind.DENY
    assert write_high.reason == write_low.reason == "auth_required"

    # Confidence also cannot turn a non-safety tool into a safety handoff.
    smuggled = policy.decide(
        _proposal("open_app_surface", confidence=1.0),
        PilotAdmissionContext(safety_overlay=True),
    )
    assert smuggled.kind is ActionDecisionKind.CONFIRM
    assert smuggled.reason == "confirmation_required"


def test_risk_class_cannot_be_widened_by_context(policy: PilotPolicy) -> None:
    """Decision risk_class is catalog-bound even after full admission."""

    read = policy.decide(
        _proposal("read_calendar"),
        PilotAdmissionContext(confirmed=True),
    )
    assert read.risk_class is RiskClass.READ
    assert read.kind is ActionDecisionKind.PERMIT_READ

    write = policy.decide(
        _proposal("create_calendar_reminder"),
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id="211-ai",
            elevated_admin_grant=True,  # must not reclassify write as admin
            safety_overlay=True,  # must not reclassify as human/safety
        ),
    )
    assert write.risk_class is RiskClass.WRITE
    assert write.kind is ActionDecisionKind.PERMIT_EXECUTE


def test_admin_default_deny_without_elevated_grant(catalog: ActionCatalog) -> None:
    admin = ActionDescriptor(
        descriptor_id="voice.python.admin_ops.v1",
        logical_action="admin_ops",
        adapter="python",
        risk_class=RiskClass.ADMIN,
        side_effect_class=SideEffectClass.EXTERNAL_MUTATION,
        requires_confirmation=True,
        allowed_channels=("voice", "test"),
        allowed_tenants=("*",),
    )
    policy = PilotPolicy(catalog=ActionCatalog([admin]), now=lambda: 0.0)
    proposal = ActionProposal(
        proposal_id="prop-admin",
        descriptor_id=admin.descriptor_id,
        logical_action="admin_ops",
        channel="voice",
        tenant_id="211-ai",
        confidence=1.0,
    )
    denied = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id="211-ai",
        ),
    )
    assert denied.kind is ActionDecisionKind.DENY
    assert denied.reason == "admin_default_deny"

    pending = policy.decide(
        proposal,
        PilotAdmissionContext(
            elevated_admin_grant=True,
            confirmed=False,
            authenticated=True,
            session_tenant_id="211-ai",
        ),
    )
    assert pending.kind is ActionDecisionKind.CONFIRM

    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(
            elevated_admin_grant=True,
            confirmed=True,
            authenticated=True,
            session_tenant_id="211-ai",
        ),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_EXECUTE
    assert admitted.risk_class is RiskClass.ADMIN


def test_decision_carries_policy_revision_and_digests(policy: PilotPolicy) -> None:
    proposal = _proposal("open_service_detail")
    decision = policy.decide(proposal, PilotAdmissionContext(confirmed=True))
    assert decision.policy_revision == POLICY_REVISION
    assert decision.proposal_id == proposal.proposal_id
    assert decision.descriptor_id == proposal.descriptor_id
    assert decision.arguments_digest == proposal.arguments_digest
    assert len(decision.descriptor_digest) == 64
    assert decision.expires_at_epoch_s == 1_700_000_000.0 + 300.0


def test_unmapped_human_class_denies(catalog: ActionCatalog) -> None:
    orphan = ActionDescriptor(
        descriptor_id="voice.human.mystery.v1",
        logical_action="mystery_human",
        adapter="human",
        risk_class=RiskClass.HUMAN,
        side_effect_class=SideEffectClass.NETWORK,
        requires_confirmation=True,
        allowed_channels=("voice",),
        allowed_tenants=("*",),
        metadata={"family": "unknown"},
    )
    policy = PilotPolicy(catalog=ActionCatalog([orphan]), now=lambda: 0.0)
    proposal = ActionProposal(
        proposal_id="prop-mystery",
        descriptor_id=orphan.descriptor_id,
        logical_action="mystery_human",
        channel="voice",
    )
    decision = policy.decide(proposal)
    assert decision.kind is ActionDecisionKind.DENY
    assert decision.reason == "human_class_unmapped"
