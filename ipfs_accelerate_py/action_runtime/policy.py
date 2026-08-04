"""Fail-closed policy admission for action proposals."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from .catalog import ActionCatalog, ActionDescriptor
from .contracts import (
    ActionDecision,
    ActionDecisionKind,
    ActionProposal,
    RiskClass,
)


class ActionPolicyEngine(Protocol):
    def decide(self, proposal: ActionProposal) -> ActionDecision: ...


@dataclass
class FailClosedPolicy:
    """Default-deny policy that only permits reviewed, in-catalog descriptors.

    Explicit operator/test grants are required for execute. Retrieval confidence
    never increases authority.
    """

    catalog: ActionCatalog
    policy_revision: str = "fail-closed-v1"
    # Explicit grants: proposal_id or descriptor_id -> decision kind
    grants: dict[str, ActionDecisionKind] = field(default_factory=dict)
    # Global allow for read-only descriptors when grant is present for descriptor
    auto_permit_read: bool = False
    now: callable = time.time  # type: ignore[assignment]

    def grant(
        self,
        *,
        proposal_id: str | None = None,
        descriptor_id: str | None = None,
        kind: ActionDecisionKind = ActionDecisionKind.PERMIT_EXECUTE,
    ) -> None:
        if kind not in {
            ActionDecisionKind.PERMIT_READ,
            ActionDecisionKind.PERMIT_EXECUTE,
            ActionDecisionKind.CONFIRM,
            ActionDecisionKind.CLARIFY,
            ActionDecisionKind.HANDOFF,
        }:
            raise ValueError(f"cannot grant decision kind {kind!r}")
        if proposal_id:
            self.grants[f"proposal:{proposal_id}"] = kind
        if descriptor_id:
            self.grants[f"descriptor:{descriptor_id}"] = kind
        if not proposal_id and not descriptor_id:
            raise ValueError("grant requires proposal_id or descriptor_id")

    def decide(self, proposal: ActionProposal) -> ActionDecision:
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

        grant = self.grants.get(f"proposal:{proposal.proposal_id}") or self.grants.get(
            f"descriptor:{proposal.descriptor_id}"
        )

        if grant is None:
            if descriptor.requires_confirmation:
                return self._decision(
                    proposal,
                    kind=ActionDecisionKind.CONFIRM,
                    reason="confirmation_required",
                    descriptor=descriptor,
                )
            if self.auto_permit_read and descriptor.risk_class == RiskClass.READ:
                return self._decision(
                    proposal,
                    kind=ActionDecisionKind.PERMIT_READ,
                    reason="auto_permit_read",
                    descriptor=descriptor,
                )
            return self._decision(
                proposal,
                kind=ActionDecisionKind.DENY,
                reason="no_grant",
                descriptor=descriptor,
            )

        if grant in {ActionDecisionKind.PERMIT_EXECUTE, ActionDecisionKind.PERMIT_READ}:
            if (
                grant == ActionDecisionKind.PERMIT_EXECUTE
                and descriptor.risk_class == RiskClass.ADMIN
            ):
                # Still allow only with explicit grant (already present).
                pass
            return self._decision(
                proposal,
                kind=grant,
                reason="explicit_grant",
                descriptor=descriptor,
            )

        return self._decision(
            proposal,
            kind=grant,
            reason="explicit_non_execute_grant",
            descriptor=descriptor,
        )

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
            risk_class=risk_class or (descriptor.risk_class if descriptor else RiskClass.READ),
            expires_at_epoch_s=float(self.now()) + 300.0,
        )
