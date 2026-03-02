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
