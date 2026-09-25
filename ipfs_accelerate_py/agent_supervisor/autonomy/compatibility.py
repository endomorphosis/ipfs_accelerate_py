"""SPAR W4 pickle, plugin, and init/API compatibility on the existing board owner.

There is no pickle/repr/str identity fallback. Silent public incompatibility
is a typed terminal. Plugin and initialization-order breaks cannot complete a
task. Missing payload is fail-open. TypeSafe is never this owner.
"""

from __future__ import annotations

from typing import Any, Mapping

PICKLE_FALLBACK_NOT_IDENTITY = "pickle_fallback_not_identity"
PLUGIN_INCOMPATIBLE = "plugin_incompatible"
INIT_ORDER_INCOMPATIBLE = "initialization_order_incompatible"
SILENT_PUBLIC_INCOMPATIBILITY = "silent_public_incompatibility"
COMPATIBILITY_RESPECTED = "compatibility_respected"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record(state: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("compatibility", "binding_compatibility", "identity"):
        nested = state.get(key)
        if isinstance(nested, Mapping):
            return {**state, **nested}
    return state


def claims_compatibility(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    return bool(
        payload.get("compatibility")
        or payload.get("binding_compatibility")
        or payload.get("pickle_fallback") is True
        or payload.get("repr_fallback") is True
        or payload.get("str_fallback") is True
        or payload.get("plugin_incompatible") is True
        or payload.get("initialization_order_cycle") is True
        or payload.get("silent_public_incompatibility") is True
    )


def compatibility_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    claimed = claims_compatibility(payload)
    merged = _record(payload)
    pickle = bool(
        merged.get("pickle_fallback") is True
        or merged.get("repr_fallback") is True
        or merged.get("str_fallback") is True
    )
    plugin = merged.get("plugin_incompatible") is True
    init = merged.get("initialization_order_cycle") is True
    silent = merged.get("silent_public_incompatibility") is True
    reason = ""
    if claimed and pickle:
        reason = PICKLE_FALLBACK_NOT_IDENTITY
    elif claimed and silent:
        reason = SILENT_PUBLIC_INCOMPATIBILITY
    elif claimed and plugin:
        reason = PLUGIN_INCOMPATIBLE
    elif claimed and init:
        reason = INIT_ORDER_INCOMPATIBLE
    elif claimed:
        reason = COMPATIBILITY_RESPECTED
    blocks = bool(
        claimed
        and reason
        in {
            PICKLE_FALLBACK_NOT_IDENTITY,
            SILENT_PUBLIC_INCOMPATIBILITY,
            PLUGIN_INCOMPATIBLE,
            INIT_ORDER_INCOMPATIBLE,
        }
    )
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "respected": reason == COMPATIBILITY_RESPECTED,
        "blocks_completion": blocks,
        "reason_code": reason,
    }


def compatibility_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(compatibility_view(state)["blocks_completion"])


__all__ = [
    "COMPATIBILITY_RESPECTED",
    "INIT_ORDER_INCOMPATIBLE",
    "PICKLE_FALLBACK_NOT_IDENTITY",
    "PLUGIN_INCOMPATIBLE",
    "SILENT_PUBLIC_INCOMPATIBILITY",
    "claims_compatibility",
    "compatibility_blocks_completion",
    "compatibility_view",
]
