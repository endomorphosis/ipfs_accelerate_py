"""Bridge voice response-DAG routes to logical action proposals.

Authority-plane proposal factory (VOICE-ACTION-009).  This module never
executes tools.  It maps known slotted-response-DAG routes to catalog
descriptor references, classifies all 12 routes, and fails closed when a
tool-adjacent / proposal-eligible route lacks a catalog entry.

Domain content cannot introduce executable paths, argv, credentials, or
other locator arguments.  Deployments may override ``route_map`` and
``descriptor_map`` (for example to bind the 211-AI pilot catalog) without
widening forbidden argument classes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Mapping

from .catalog import ActionCatalog
from .contracts import ActionProposal

# ---------------------------------------------------------------------------
# Route classification (matches baseline inventory / action-link projection)
# ---------------------------------------------------------------------------

ROUTE_CLASSIFICATION_CONTENT_ONLY: Final = "content-only"
ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE: Final = "proposal-eligible"
ROUTE_CLASSIFICATION_SAFETY_OVERLAY: Final = "safety-overlay"

ROUTE_CLASSIFICATIONS: Final = frozenset(
    {
        ROUTE_CLASSIFICATION_CONTENT_ONLY,
        ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE,
        ROUTE_CLASSIFICATION_SAFETY_OVERLAY,
    }
)

# Explicit content-only sentinel (never a catalog descriptor).
NO_ACTION: Final = "no_action"

# All 12 slotted-DAG routes (stable sorted order for sampling / tests).
SLOTTED_DAG_ROUTES: Final[tuple[str, ...]] = (
    "app_surface_navigation",
    "calendar_event_support",
    "clarifying_prompt",
    "grounded_211_answer",
    "live_agent",
    "provider_contact_support",
    "repeat_or_restate",
    "safety_guardrail_support",
    "service_interaction_support",
    "speech_unclear_clarification",
    "template_guided_fallback",
    "wallet_document_support",
)

EXPECTED_ROUTE_COUNT: Final = 12

# Deployment-owned classification table for every slotted-DAG route.
DEFAULT_ROUTE_CLASSIFICATION: Mapping[str, str] = MappingProxyType(
    {
        "app_surface_navigation": ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE,
        "calendar_event_support": ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE,
        "clarifying_prompt": ROUTE_CLASSIFICATION_CONTENT_ONLY,
        "grounded_211_answer": ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE,
        "live_agent": ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE,
        "provider_contact_support": ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE,
        "repeat_or_restate": ROUTE_CLASSIFICATION_CONTENT_ONLY,
        "safety_guardrail_support": ROUTE_CLASSIFICATION_SAFETY_OVERLAY,
        "service_interaction_support": ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE,
        "speech_unclear_clarification": ROUTE_CLASSIFICATION_CONTENT_ONLY,
        "template_guided_fallback": ROUTE_CLASSIFICATION_CONTENT_ONLY,
        "wallet_document_support": ROUTE_CLASSIFICATION_PROPOSAL_ELIGIBLE,
    }
)

# Historical pilot tool-adjacent routes (CLI probe descriptors).  Catalog
# presence is mandatory for these when require_catalog_entry=true.
TOOL_ADJACENT_ROUTES: Final[frozenset[str]] = frozenset(
    {
        "app_surface_navigation",
        "wallet_document_support",
        "calendar_event_support",
        "service_interaction_support",
        "provider_contact_support",
    }
)

CONTENT_ONLY_ROUTES: Final[frozenset[str]] = frozenset(
    route
    for route, classification in DEFAULT_ROUTE_CLASSIFICATION.items()
    if classification == ROUTE_CLASSIFICATION_CONTENT_ONLY
)

# Route strings that may emit a catalog-bound logical action.  Mapping is
# deployment-owned, not pack-owned.  Content-only routes are intentionally
# absent (bridge returns None → no_action).
DEFAULT_ROUTE_TO_LOGICAL_ACTION: Mapping[str, str] = MappingProxyType(
    {
        # Tool-adjacent pilot set (CLI probe descriptors).
        "app_surface_navigation": "open_app_surface",
        "wallet_document_support": "open_wallet_documents",
        "calendar_event_support": "open_calendar_support",
        "service_interaction_support": "review_service_interaction",
        "provider_contact_support": "provide_provider_contact",
        # Multi-route expansion (proposal-eligible + safety overlay).
        "grounded_211_answer": "open_service_detail",
        "live_agent": "handoff_live_agent",
        "safety_guardrail_support": "escalate_safety",
    }
)

DEFAULT_LOGICAL_ACTION_TO_DESCRIPTOR: Mapping[str, str] = MappingProxyType(
    {
        "open_app_surface": "voice.cli.open_app_surface.v1",
        "open_wallet_documents": "voice.cli.open_wallet_documents.v1",
        "open_calendar_support": "voice.cli.open_calendar_support.v1",
        "review_service_interaction": "voice.cli.review_service_interaction.v1",
        "provide_provider_contact": "voice.cli.provide_provider_contact.v1",
        "open_service_detail": "voice.cli.open_service_detail.v1",
        "handoff_live_agent": "voice.cli.handoff_live_agent.v1",
        "escalate_safety": "voice.cli.escalate_safety.v1",
    }
)

# Align with action_runtime.contracts.ActionProposal banned keys.
_BANNED_ARGUMENT_KEYS: Final = frozenset(
    {
        "command",
        "argv",
        "executable",
        "cwd",
        "env",
        "shell",
        "import",
        "import_path",
        "url",
        "credentials",
        "secret",
        "webhook",
    }
)


def classify_route(
    route: str,
    *,
    classification_map: Mapping[str, str] | None = None,
) -> str | None:
    """Return the classification for a slotted-DAG route, or None if unknown."""

    table = classification_map or DEFAULT_ROUTE_CLASSIFICATION
    return table.get(route)


def is_tool_adjacent(route: str) -> bool:
    """Return True when *route* is in the historical tool-adjacent pilot set."""

    return route in TOOL_ADJACENT_ROUTES


def is_content_only(
    route: str,
    *,
    classification_map: Mapping[str, str] | None = None,
) -> bool:
    """Return True when *route* is classified content-only (speech only)."""

    return classify_route(route, classification_map=classification_map) == (
        ROUTE_CLASSIFICATION_CONTENT_ONLY
    )


def _validate_arguments(arguments: Mapping[str, str] | None) -> dict[str, str]:
    """Reject executable / locator arguments; return a safe string map.

    Proposals never carry shell, process, network, or credential locators.
    Unknown non-banned string slots are allowed for future validated slot
    layers, but the default bridge path always passes an empty map.
    """

    if arguments is None:
        return {}
    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a string-to-string mapping")
    safe: dict[str, str] = {}
    for key, value in arguments.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("arguments must be string-to-string")
        lowered = key.lower()
        if lowered in _BANNED_ARGUMENT_KEYS or lowered.endswith("_path"):
            raise ValueError(f"proposal argument {key!r} is not allowed")
        safe[key] = value
    return safe


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
    arguments: Mapping[str, str] | None = None,
    route_map: Mapping[str, str] | None = None,
    descriptor_map: Mapping[str, str] | None = None,
    classification_map: Mapping[str, str] | None = None,
) -> ActionProposal | None:
    """Return an authority-free proposal for a known voice route, or None.

    Content-only routes and unknown / unmapped routes yield ``None`` (explicit
    no_action at the call site).  Free-text *transcript* never invents
    descriptors.  *arguments* are validated and may not carry executables.
    """

    # Content-only routes never emit side-effect proposals.
    classification = classify_route(route, classification_map=classification_map)
    if classification == ROUTE_CLASSIFICATION_CONTENT_ONLY:
        return None

    routes = route_map if route_map is not None else DEFAULT_ROUTE_TO_LOGICAL_ACTION
    descriptors = (
        descriptor_map
        if descriptor_map is not None
        else DEFAULT_LOGICAL_ACTION_TO_DESCRIPTOR
    )
    logical = routes.get(route)
    if not logical or logical == NO_ACTION:
        return None
    descriptor_id = descriptors.get(logical)
    if not descriptor_id:
        return None

    # Arguments are intentionally empty by default.  Validated slot layers may
    # supply non-locator string pairs later; never free-form shell text.
    safe_arguments = _validate_arguments(arguments)

    return ActionProposal(
        proposal_id=f"prop-{uuid.uuid4().hex[:16]}",
        descriptor_id=descriptor_id,
        logical_action=logical,
        arguments=safe_arguments,
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
            "route_classification": classification or "",
        },
    )


@dataclass
class VoiceActionBridge:
    """Catalog-aware multi-route proposal helper for product code and tests.

    When ``require_catalog_entry`` is true (default), tool-adjacent and other
    mapped routes only emit a proposal if the bound descriptor is present in
    the deployment catalog and its ``logical_action`` matches.  Missing catalog
    entries fail closed to ``None`` (no_action).
    """

    catalog: ActionCatalog
    route_map: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ROUTE_TO_LOGICAL_ACTION)
    )
    descriptor_map: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LOGICAL_ACTION_TO_DESCRIPTOR)
    )
    classification_map: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ROUTE_CLASSIFICATION)
    )

    def classify(self, route: str) -> str | None:
        """Return the classification for *route* under this bridge's table."""

        return classify_route(route, classification_map=self.classification_map)

    def is_tool_adjacent(self, route: str) -> bool:
        return is_tool_adjacent(route)

    def classified_routes(self) -> Mapping[str, str]:
        """Return a snapshot of the classification table (all known routes)."""

        return dict(self.classification_map)

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
        arguments: Mapping[str, str] | None = None,
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
            arguments=arguments,
            route_map=self.route_map,
            descriptor_map=self.descriptor_map,
            classification_map=self.classification_map,
        )
        if proposal is None:
            return None
        if not require_catalog_entry:
            return proposal
        descriptor = self.catalog.get(proposal.descriptor_id)
        if descriptor is None:
            # Fail closed: tool-adjacent and other mapped routes require a
            # reviewed catalog entry when require_catalog_entry=true.
            return None
        if descriptor.logical_action != proposal.logical_action:
            return None
        return proposal
