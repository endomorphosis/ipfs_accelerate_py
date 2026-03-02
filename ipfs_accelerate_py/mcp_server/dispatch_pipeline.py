"""Dispatch pipeline helpers for unified MCP server.

This module centralizes parameter coercion and deterministic intent-CID
construction used by ``tools_dispatch``.
"""

from __future__ import annotations

from typing import Any

from .mcplusplus.artifacts import compute_artifact_cid


def normalize_dispatch_parameters(parameters: Any) -> dict[str, Any]:
    """Normalize raw dispatch parameters to a mutable mapping."""
    return dict(parameters) if isinstance(parameters, dict) else {}


def coerce_dispatch_bool(value: Any, *, field_name: str) -> bool:
    """Coerce a dispatch control value into a boolean.

    Accepted representations include native booleans, 0/1 numbers,
    and common truthy/falsy strings.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{field_name} must be boolean-like (0/1)")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{field_name} must be one of true/false/1/0/yes/no/on/off")
    raise ValueError(f"{field_name} must be a boolean, boolean-like number, or boolean-like string")


def coerce_dispatch_list(value: Any, *, field_name: str) -> list[Any]:
    """Coerce a dispatch control value into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise ValueError(f"{field_name} must be a list")


def coerce_dispatch_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    """Coerce a dispatch control value into an object-like mapping."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(f"{field_name} must be an object/dict")


def compute_dispatch_intent_cid(category: str, tool_name: str, parameters: dict[str, Any]) -> str:
    """Build the deterministic intent CID for a dispatch request."""
    return compute_artifact_cid(
        {
            "category": str(category),
            "tool_name": str(tool_name),
            "parameters": dict(parameters),
        }
    )
