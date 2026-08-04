"""Pilot fail-closed policy matrix for read/write/human/safety classes.

Extends the baseline default-deny policy with pilot predicates:

* read actions require explicit confirmation before permit;
* write (and auth-gated read) actions require confirmation **and** an
  authenticated tenant session;
* human handoff follows the handoff request path (never auto-claims transfer
  completion);
* safety overlay may force ``escalate_safety`` handoff only and cannot widen
  authority onto arbitrary descriptors;
* retrieval confidence never upgrades a decision.

This module is pure: no I/O, no network, no process spawning.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Mapping

from .catalog import ActionCatalog, ActionDescriptor
from .contracts import (
    ActionDecision,
    ActionDecisionKind,
    ActionProposal,
    RiskClass,
)

POLICY_REVISION: str = "pilot-policy-matrix-v1"
SAFETY_LOGICAL_ACTION: str = "escalate_safety"
HANDOFF_LOGICAL_ACTION: str = "handoff_live_agent"

# Decision kinds that may never be used to smuggle execute authority via grants.
_NON_EXECUTE_KINDS = frozenset(
    {
        ActionDecisionKind.DENY,
        ActionDecisionKind.CLARIFY,
        ActionDecisionKind.CONFIRM,
        ActionDecisionKind.HANDOFF,
    }
)


@dataclass(frozen=True)
class PilotAdmissionContext:
    """Caller-supplied admission facts (authority plane; not retrieval).

    Attributes:
        confirmed: Operator/UI/spoken confirm recorded by the authority plane.
        authenticated: Tenant session passed step-up / auth gate.
        session_tenant_id: Authenticated tenant identity when known.
        safety_overlay: Safety/crisis overlay active for this turn.
        elevated_admin_grant: Explicit elevated grant for admin-class actions.
    """

    confirmed: bool = False
    authenticated: bool = False
    session_tenant_id: str | None = None
    safety_overlay: bool = False
    elevated_admin_grant: bool = False


def _truthy_meta(metadata: Mapping[str, str], key: str, default: str = "false") -> bool:
    raw = str(metadata.get(key, default)).strip().lower()
    return raw in {"1", "true", "yes", "required", "on"}


def descriptor_requires_auth(descriptor: ActionDescriptor) -> bool:
    """Return whether the descriptor requires an authenticated tenant session.

    Write and admin classes always require auth. Read/human may opt in via
    catalog metadata ``auth_required=true``.
    """

    if descriptor.risk_class in {RiskClass.WRITE, RiskClass.ADMIN}:
        return True
    return _truthy_meta(descriptor.metadata, "auth_required", "false")


def is_safety_descriptor(descriptor: ActionDescriptor) -> bool:
    return (
        descriptor.logical_action == SAFETY_LOGICAL_ACTION
        or descriptor.metadata.get("family") == "safety"
    )


def is_handoff_descriptor(descriptor: ActionDescriptor) -> bool:
    return (
        descriptor.logical_action == HANDOFF_LOGICAL_ACTION
        or descriptor.metadata.get("family") == "handoff"
    )


@dataclass
class PilotPolicy:
    """Pilot policy matrix: default deny with class-specific gates.

    ``proposal.confidence`` is intentionally never consulted. Callers must
    pass confirmation, authentication, and safety overlay through
    :class:`PilotAdmissionContext`.
    """

    catalog: ActionCatalog
    policy_revision: str = POLICY_REVISION
    # When True, handoff_live_agent may emit HANDOFF (request creation) without
    # an explicit confirm. Completion of a transfer still requires a receipted
    # adapter path outside this policy.
    handoff_auto_request: bool = True
    # When True (default), escalate_safety is admitted as HANDOFF under the
    # policy-driven path even without an active overlay. Overlay still cannot
    # widen other descriptors.
    safety_policy_auto_handoff: bool = True
    decision_ttl_seconds: float = 300.0
    now: Callable[[], float] = time.time

    def decide(
        self,
        proposal: ActionProposal,
        context: PilotAdmissionContext | None = None,
    ) -> ActionDecision:
        """Evaluate a proposal under the pilot matrix.

        Confidence on the proposal is ignored for authority. Safety overlay
        only affects the reviewed safety descriptor.
        """

        ctx = context or PilotAdmissionContext()
        # Bind confidence so static analysis / reviewers see it is discarded.
        _ = float(proposal.confidence)

        descriptor = self.catalog.get(proposal.descriptor_id)
        if descriptor is None:
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="unknown_descriptor",
                descriptor_digest="unknown",
                risk_class=RiskClass.READ,
            )

        if proposal.logical_action != descriptor.logical_action:
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="logical_action_mismatch",
                descriptor=descriptor,
            )

        if proposal.channel and proposal.channel not in descriptor.allowed_channels:
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="channel_not_allowed",
                descriptor=descriptor,
            )

        if (
            proposal.tenant_id
            and "*" not in descriptor.allowed_tenants
            and proposal.tenant_id not in descriptor.allowed_tenants
        ):
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="tenant_not_allowed",
                descriptor=descriptor,
            )

        # Safety overlay: force escalate path only for the safety descriptor.
        # Never widen to open_app_surface, writes, or other arbitrary tools.
        if ctx.safety_overlay:
            if is_safety_descriptor(descriptor):
                return self._decision(
                    proposal,
                    kind=ActionDecisionKind.HANDOFF,
                    reason="safety_overlay_force_escalate",
                    descriptor=descriptor,
                )
            # Overlay active but proposal targets a non-safety descriptor:
            # fall through to normal class gates (no authority widening).

        # Class-specific matrix.
        if descriptor.risk_class is RiskClass.HUMAN:
            return self._decide_human(proposal, descriptor, ctx)
        if descriptor.risk_class is RiskClass.ADMIN:
            return self._decide_admin(proposal, descriptor, ctx)
        if descriptor.risk_class is RiskClass.WRITE:
            return self._decide_write(proposal, descriptor, ctx)
        # READ (default class for non-mutating pilot surfaces).
        return self._decide_read(proposal, descriptor, ctx)

    def _decide_read(
        self,
        proposal: ActionProposal,
        descriptor: ActionDescriptor,
        ctx: PilotAdmissionContext,
    ) -> ActionDecision:
        needs_auth = descriptor_requires_auth(descriptor)
        # Pilot matrix: read side effects never run from retrieval alone.
        if not ctx.confirmed:
            return self._decision(
                proposal,
                kind=ActionDecisionKind.CONFIRM,
                reason="confirmation_required",
                descriptor=descriptor,
            )
        if needs_auth and not self._auth_satisfied(proposal, ctx):
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="auth_required",
                descriptor=descriptor,
            )
        return self._decision(
            proposal,
            kind=ActionDecisionKind.PERMIT_READ,
            reason="read_confirmed" if not needs_auth else "read_confirmed_authenticated",
            descriptor=descriptor,
        )

    def _decide_write(
        self,
        proposal: ActionProposal,
        descriptor: ActionDescriptor,
        ctx: PilotAdmissionContext,
    ) -> ActionDecision:
        if not ctx.confirmed:
            return self._decision(
                proposal,
                kind=ActionDecisionKind.CONFIRM,
                reason="confirmation_required",
                descriptor=descriptor,
            )
        if not self._auth_satisfied(proposal, ctx):
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="auth_required",
                descriptor=descriptor,
            )
        return self._decision(
            proposal,
            kind=ActionDecisionKind.PERMIT_EXECUTE,
            reason="write_confirmed_authenticated",
            descriptor=descriptor,
        )

    def _decide_admin(
        self,
        proposal: ActionProposal,
        descriptor: ActionDescriptor,
        ctx: PilotAdmissionContext,
    ) -> ActionDecision:
        # Admin is default-deny unless elevated grant + confirm + auth.
        if not ctx.elevated_admin_grant:
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="admin_default_deny",
                descriptor=descriptor,
            )
        if not ctx.confirmed:
            return self._decision(
                proposal,
                kind=ActionDecisionKind.CONFIRM,
                reason="confirmation_required",
                descriptor=descriptor,
            )
        if not self._auth_satisfied(proposal, ctx):
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="auth_required",
                descriptor=descriptor,
            )
        return self._decision(
            proposal,
            kind=ActionDecisionKind.PERMIT_EXECUTE,
            reason="admin_elevated_confirmed_authenticated",
            descriptor=descriptor,
        )

    def _decide_human(
        self,
        proposal: ActionProposal,
        descriptor: ActionDescriptor,
        ctx: PilotAdmissionContext,
    ) -> ActionDecision:
        if is_safety_descriptor(descriptor):
            if ctx.safety_overlay:
                return self._decision(
                    proposal,
                    kind=ActionDecisionKind.HANDOFF,
                    reason="safety_overlay_force_escalate",
                    descriptor=descriptor,
                )
            if self.safety_policy_auto_handoff:
                return self._decision(
                    proposal,
                    kind=ActionDecisionKind.HANDOFF,
                    reason="safety_policy_handoff",
                    descriptor=descriptor,
                )
            if not ctx.confirmed:
                return self._decision(
                    proposal,
                    kind=ActionDecisionKind.CONFIRM,
                    reason="confirmation_required",
                    descriptor=descriptor,
                )
            return self._decision(
                proposal,
                kind=ActionDecisionKind.HANDOFF,
                reason="safety_confirmed_handoff",
                descriptor=descriptor,
            )

        if is_handoff_descriptor(descriptor):
            # Handoff policy: may auto-admit *request creation*; never claims
            # transfer success (that is adapter/receipt territory).
            if ctx.confirmed or self.handoff_auto_request:
                reason = (
                    "handoff_confirmed_request"
                    if ctx.confirmed
                    else "handoff_policy_request"
                )
                return self._decision(
                    proposal,
                    kind=ActionDecisionKind.HANDOFF,
                    reason=reason,
                    descriptor=descriptor,
                )
            return self._decision(
                proposal,
                kind=ActionDecisionKind.CONFIRM,
                reason="confirmation_required",
                descriptor=descriptor,
            )

        # Unknown human-class descriptor: fail closed.
        return self._decision(
            proposal,
            kind=ActionDecisionKind.DENY,
            reason="human_class_unmapped",
            descriptor=descriptor,
        )

    def _auth_satisfied(
        self,
        proposal: ActionProposal,
        ctx: PilotAdmissionContext,
    ) -> bool:
        if not ctx.authenticated:
            return False
        # When both sides declare a tenant, they must match (confused-deputy).
        if (
            proposal.tenant_id
            and ctx.session_tenant_id
            and proposal.tenant_id != ctx.session_tenant_id
        ):
            return False
        # Authenticated session must carry a tenant when the proposal is scoped.
        if proposal.tenant_id and not ctx.session_tenant_id:
            return False
        return True

    def _decision(
        self,
        proposal: ActionProposal,
        *,
        kind: ActionDecisionKind,
        reason: str,
        descriptor: ActionDescriptor | None = None,
        descriptor_digest: str | None = None,
        risk_class: RiskClass | None = None,
    ) -> ActionDecision:
        # Risk class always comes from the catalog descriptor when present —
        # grants/context cannot widen or reclassify it.
        bound_risk = risk_class or (
            descriptor.risk_class if descriptor is not None else RiskClass.READ
        )
        if kind not in _NON_EXECUTE_KINDS and kind not in {
            ActionDecisionKind.PERMIT_READ,
            ActionDecisionKind.PERMIT_EXECUTE,
        }:
            kind = ActionDecisionKind.DENY
            reason = "unknown_decision_kind"
        return ActionDecision(
            decision_id=f"dec-{uuid.uuid4().hex[:16]}",
            kind=kind,
            proposal_id=proposal.proposal_id,
            descriptor_id=proposal.descriptor_id,
            descriptor_digest=(
                descriptor_digest
                if descriptor_digest is not None
                else (descriptor.digest if descriptor is not None else "unknown")
            ),
            arguments_digest=proposal.arguments_digest,
            reason=reason,
            policy_revision=self.policy_revision,
            risk_class=bound_risk,
            expires_at_epoch_s=float(self.now()) + float(self.decision_ttl_seconds),
        )


def build_pilot_policy(catalog: ActionCatalog | None = None) -> PilotPolicy:
    """Construct a pilot policy, defaulting to the 211-AI pilot catalog."""

    if catalog is None:
        from .catalog_211ai import build_pilot_catalog

        catalog = build_pilot_catalog()
    return PilotPolicy(catalog=catalog)
