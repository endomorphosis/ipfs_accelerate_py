"""Client-channel surface exposure classes for authority-plane policy (VAS2-010/011).

Content-free: maps surface_id → exposure class only. Must stay in lockstep with
monorepo exposure matrices under data/voice_app_surface_* /baseline/.
"""

from __future__ import annotations

from typing import Final, Mapping

# Normative classes (fail closed: unknown → never_voice).
SURFACE_EXPOSURE_CLASS: Final[Mapping[str, str]] = {
    "home": "voice_navigable",
    "register": "voice_navigable",
    "check-in": "voice_navigable",
    "calendar": "voice_actionable",
    "messages": "voice_actionable",
    "contacts": "voice_navigable",
    "social-services": "voice_actionable",
    "interactions": "voice_navigable",
    "uploads": "voice_actionable",
    "settings": "voice_navigable",
    "analytics": "voice_read_only",
    "proof-center": "voice_read_only",
    "audit": "never_voice",
    "security": "never_voice",
    "exports": "never_voice",
    "recipient-access": "never_voice",
    "sharing-rules": "never_voice",
    "benefits-protection": "never_voice",
    "shelter": "staff_only",
    "provider-clients": "staff_only",
    "provider-cases": "staff_only",
    "provider-messages": "staff_only",
    "provider-analytics": "staff_only",
    "provider-proofs": "staff_only",
    "provider-operations": "staff_only",
}

CLIENT_VOICE_OPEN_CLASSES: Final[frozenset[str]] = frozenset(
    {"voice_navigable", "voice_actionable"}
)
NEVER_VOICE_CLASSES: Final[frozenset[str]] = frozenset({"never_voice"})
STAFF_ONLY_CLASSES: Final[frozenset[str]] = frozenset({"staff_only"})
STAFF_DENY_CHANNELS: Final[frozenset[str]] = frozenset(
    {"voice", "phone", "telephony", "chat"}
)

# Logical actions that open or target an app surface via arguments.surface_id
# (or open_wallet_documents → uploads).
SURFACE_TARGETING_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "open_app_surface",
        "open_wallet_documents",
    }
)

# Reviewed surface → logical actions for catalog/surface matrix (client voice).
VOICE_CLIENT_SURFACE_ACTIONS: Final[Mapping[str, tuple[str, ...]]] = {
    "home": ("open_app_surface",),
    "register": ("open_app_surface",),
    "check-in": ("open_app_surface",),
    "calendar": (
        "open_app_surface",
        "read_calendar",
        "create_calendar_reminder",
    ),
    "messages": (
        "open_app_surface",
        "read_provider_messages",
        "leave_provider_message",
    ),
    "contacts": ("open_app_surface",),
    "social-services": (
        "open_app_surface",
        "open_service_detail",
        "schedule_service_callback",
    ),
    "interactions": ("open_app_surface",),
    "uploads": ("open_app_surface", "open_wallet_documents"),
    "settings": ("open_app_surface",),
}


def get_surface_exposure_class(surface_id: str) -> str:
    """Return exposure class; unknown surfaces are never_voice."""

    return SURFACE_EXPOSURE_CLASS.get(str(surface_id).strip(), "never_voice")


def resolve_target_surface_id(
    logical_action: str,
    arguments: Mapping[str, str] | None,
) -> str | None:
    """Extract target surface_id from proposal arguments when applicable."""

    args = arguments or {}
    if logical_action == "open_wallet_documents":
        return str(args.get("surface_id") or "uploads").strip() or "uploads"
    if logical_action == "open_app_surface":
        raw = str(args.get("surface_id") or args.get("surface") or "").strip()
        return raw or None
    # Other actions may still name a surface for context.
    raw = str(args.get("surface_id") or "").strip()
    return raw or None


def surface_exposure_deny_reason(
    surface_id: object | None,
    *,
    channel: object | None = "voice",
    role: str = "client",
) -> str | None:
    """Return deny reason code for client channels, or None if allowed to open."""

    if surface_id is None or str(surface_id).strip() == "":
        return "surface_id_required"
    sid = str(surface_id).strip()
    klass = get_surface_exposure_class(sid)
    channel_norm = str(channel or "voice").strip().lower() or "voice"
    role_norm = str(role or "client").strip().lower() or "client"
    if klass in NEVER_VOICE_CLASSES:
        return "surface_never_voice"
    if klass in STAFF_ONLY_CLASSES:
        if role_norm != "staff" and channel_norm in STAFF_DENY_CHANNELS:
            return "surface_staff_only"
    if klass == "voice_read_only" and channel_norm in {
        "voice",
        "phone",
        "telephony",
    }:
        return "surface_voice_read_only"
    if klass not in CLIENT_VOICE_OPEN_CLASSES and channel_norm in STAFF_DENY_CHANNELS:
        # Fail closed for any class that is not explicitly openable.
        if klass not in {"voice_read_only"}:  # already handled
            return "surface_not_voice_openable"
    return None


__all__ = [
    "CLIENT_VOICE_OPEN_CLASSES",
    "NEVER_VOICE_CLASSES",
    "STAFF_DENY_CHANNELS",
    "STAFF_ONLY_CLASSES",
    "SURFACE_EXPOSURE_CLASS",
    "SURFACE_TARGETING_ACTIONS",
    "VOICE_CLIENT_SURFACE_ACTIONS",
    "get_surface_exposure_class",
    "resolve_target_surface_id",
    "surface_exposure_deny_reason",
]
