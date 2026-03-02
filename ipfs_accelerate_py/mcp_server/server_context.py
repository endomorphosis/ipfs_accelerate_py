"""Unified server context model for canonical bootstrap attachments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnifiedServerContext:
    """Snapshot of canonical unified server bootstrap context."""

    runtime_router: Any
    tool_manager: Any
    services: dict[str, Any]
    preloaded_categories: list[str]
    supported_profiles: list[str]
    bootstrap_enabled: bool = True

    def profile_negotiation(self) -> dict[str, Any]:
        """Return normalized profile negotiation metadata for transport handshakes."""
        return {
            "supports_profile_negotiation": bool(self.bootstrap_enabled),
            "mode": "optional_additive",
            "profiles": list(self.supported_profiles),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a stable, JSON-safe snapshot of context metadata."""
        return {
            "bootstrap_enabled": bool(self.bootstrap_enabled),
            "preloaded_categories": list(self.preloaded_categories),
            "supported_profiles": list(self.supported_profiles),
            "profile_negotiation": self.profile_negotiation(),
            "services": sorted(str(name) for name in self.services.keys()),
            "service_count": len(self.services),
            "runtime_router": type(self.runtime_router).__name__,
            "tool_manager": type(self.tool_manager).__name__,
        }
