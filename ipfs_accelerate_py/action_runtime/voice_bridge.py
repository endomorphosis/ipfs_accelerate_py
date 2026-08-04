"""Bridge voice response-DAG routes to logical action proposals.

This module never executes tools.  It only maps known response-library routes
to catalog descriptor references.  Domain content cannot introduce executable
paths, argv, or credentials.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Mapping

from .catalog import ActionCatalog
from .contracts import ActionProposal

# Route strings from docs/phone_dialog_generation/slotted_response_dag.json
# that are tool-adjacent.  Mapping is deployment-owned, not pack-owned.
DEFAULT_ROUTE_TO_LOGICAL_ACTION: Mapping[str, str] = {
    "app_surface_navigation": "open_app_surface",
    "wallet_document_support": "open_wallet_documents",
    "calendar_event_support": "open_calendar_support",
    "service_interaction_support": "review_service_interaction",
    "provider_contact_support": "provide_provider_contact",
}


DEFAULT_LOGICAL_ACTION_TO_DESCRIPTOR: Mapping[str, str] = {
    "open_app_surface": "voice.cli.open_app_surface.v1",
    "open_wallet_documents": "voice.cli.open_wallet_documents.v1",
    "open_calendar_support": "voice.cli.open_calendar_support.v1",
    "review_service_interaction": "voice.cli.review_service_interaction.v1",
    "provide_provider_contact": "voice.cli.provide_provider_contact.v1",
}


def propose_from_voice_route(
    *,
    route: str,
    transcript: str = "",
    template_id: str | None = None,
    tenant_id: str | None = None,
    session_id: str | None = None,
    channel: str = "voice",
    confidence: float = 0.0,
    evidence: tuple[str, ...] = (),
    route_map: Mapping[str, str] | None = None,
    descriptor_map: Mapping[str, str] | None = None,
) -> ActionProposal | None:
    """Return an authority-free proposal for a known voice route, or None."""

    routes = route_map or DEFAULT_ROUTE_TO_LOGICAL_ACTION
    descriptors = descriptor_map or DEFAULT_LOGICAL_ACTION_TO_DESCRIPTOR
    logical = routes.get(route)
    if not logical:
        return None
    descriptor_id = descriptors.get(logical)
    if not descriptor_id:
        return None
    # Arguments are intentionally empty / non-executable. Surface names may be
    # filled later by a validated slot layer, never free-form shell text.
    return ActionProposal(
        proposal_id=f"prop-{uuid.uuid4().hex[:16]}",
        descriptor_id=descriptor_id,
        logical_action=logical,
        arguments={},
        route=route,
        source="slotted_response_dag_route",
        confidence=max(0.0, min(1.0, float(confidence))),
        tenant_id=tenant_id,
        session_id=session_id,
        channel=channel,
        evidence=evidence,
        metadata={
            "template_id": template_id or "",
            "transcript_sha_prefix": (transcript or "")[:32],
        },
    )


@dataclass
class VoiceActionBridge:
    """Optional catalog-aware helper used by product code and tests."""

    catalog: ActionCatalog
    route_map: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ROUTE_TO_LOGICAL_ACTION)
    )
    descriptor_map: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LOGICAL_ACTION_TO_DESCRIPTOR)
    )

    def propose(
        self,
        *,
        route: str,
        transcript: str = "",
        template_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        channel: str = "voice",
        confidence: float = 0.0,
        evidence: tuple[str, ...] = (),
        require_catalog_entry: bool = True,
    ) -> ActionProposal | None:
        proposal = propose_from_voice_route(
            route=route,
            transcript=transcript,
            template_id=template_id,
            tenant_id=tenant_id,
            session_id=session_id,
            channel=channel,
            confidence=confidence,
            evidence=evidence,
            route_map=self.route_map,
            descriptor_map=self.descriptor_map,
        )
        if proposal is None:
            return None
        if require_catalog_entry and self.catalog.get(proposal.descriptor_id) is None:
            return None
        return proposal
