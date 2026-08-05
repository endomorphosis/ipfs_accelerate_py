"""Surface exposure gates in PilotPolicy (VAS2-011)."""

from __future__ import annotations

from ipfs_accelerate_py.action_runtime.catalog_211ai import (
    build_pilot_catalog,
    logical_action_to_descriptor_id,
)
from ipfs_accelerate_py.action_runtime.contracts import (
    ActionDecisionKind,
    ActionProposal,
)
from ipfs_accelerate_py.action_runtime.policy_pilot import PilotAdmissionContext, PilotPolicy
from ipfs_accelerate_py.action_runtime.surface_exposure import SURFACE_EXPOSURE_CLASS


def _proposal(logical: str, surface_id: str, *, channel: str = "voice") -> ActionProposal:
    mapping = logical_action_to_descriptor_id()
    return ActionProposal(
        proposal_id=f"prop-{logical}-{surface_id}",
        descriptor_id=mapping[logical],
        logical_action=logical,
        arguments={"surface_id": surface_id},
        channel=channel,
        tenant_id="tenant-test",
        confidence=0.99,
    )


def test_never_voice_denied_before_confirm() -> None:
    policy = PilotPolicy(catalog=build_pilot_catalog())
    for sid, klass in SURFACE_EXPOSURE_CLASS.items():
        if klass != "never_voice":
            continue
        decision = policy.decide(
            _proposal("open_app_surface", sid),
            PilotAdmissionContext(confirmed=True, authenticated=True),
        )
        assert decision.kind is ActionDecisionKind.DENY, sid
        assert decision.reason == "surface_never_voice", (sid, decision.reason)


def test_staff_only_denied_on_client_voice() -> None:
    policy = PilotPolicy(catalog=build_pilot_catalog())
    for sid, klass in SURFACE_EXPOSURE_CLASS.items():
        if klass != "staff_only":
            continue
        decision = policy.decide(
            _proposal("open_app_surface", sid),
            PilotAdmissionContext(confirmed=True, authenticated=True),
        )
        assert decision.kind is ActionDecisionKind.DENY, sid
        assert decision.reason == "surface_staff_only", (sid, decision.reason)


def test_voice_navigable_requires_confirm_then_permits_read() -> None:
    policy = PilotPolicy(catalog=build_pilot_catalog())
    prop = _proposal("open_app_surface", "home")
    unconfirmed = policy.decide(prop, PilotAdmissionContext(confirmed=False))
    assert unconfirmed.kind is ActionDecisionKind.CONFIRM
    confirmed = policy.decide(prop, PilotAdmissionContext(confirmed=True))
    assert confirmed.kind is ActionDecisionKind.PERMIT_READ


def test_open_wallet_documents_defaults_uploads_surface_gate() -> None:
    policy = PilotPolicy(catalog=build_pilot_catalog())
    mapping = logical_action_to_descriptor_id()
    prop = ActionProposal(
        proposal_id="prop-wallet-docs",
        descriptor_id=mapping["open_wallet_documents"],
        logical_action="open_wallet_documents",
        arguments={},
        channel="voice",
        tenant_id="tenant-test",
    )
    decision = policy.decide(prop, PilotAdmissionContext(confirmed=True))
    assert decision.kind is ActionDecisionKind.PERMIT_READ
