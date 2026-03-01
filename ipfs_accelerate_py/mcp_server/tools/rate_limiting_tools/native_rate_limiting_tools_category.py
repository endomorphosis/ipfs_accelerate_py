"""Native rate-limiting-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_rate_limiting_tools_api() -> Dict[str, Any]:
    """Resolve source rate-limiting-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.rate_limiting_tools.rate_limiting_tools import (  # type: ignore
            check_rate_limit as _check_rate_limit,
            configure_rate_limits as _configure_rate_limits,
            manage_rate_limits as _manage_rate_limits,
        )

        return {
            "configure_rate_limits": _configure_rate_limits,
            "check_rate_limit": _check_rate_limit,
            "manage_rate_limits": _manage_rate_limits,
        }
    except Exception:
        logger.warning(
            "Source rate_limiting_tools import unavailable, using fallback rate-limiting-tools functions"
        )

        async def _configure_fallback(
            limits: list[Dict[str, Any]],
            apply_immediately: bool = True,
            backup_current: bool = True,
        ) -> Dict[str, Any]:
            _ = apply_immediately, backup_current
            return {
                "configured_count": len(limits),
                "configured_limits": limits,
                "errors": [],
            }

        async def _check_fallback(
            limit_name: str,
            identifier: str = "default",
            request_metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = request_metadata
            return {
                "allowed": True,
                "limit_name": limit_name,
                "identifier": identifier,
                "remaining_tokens": 1,
            }

        async def _manage_fallback(
            action: str,
            limit_name: Optional[str] = None,
            new_config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = limit_name, new_config
            if action == "list":
                return {"action": "list", "limits": [], "total_count": 0}
            return {"action": action, "status": "success"}

        return {
            "configure_rate_limits": _configure_fallback,
            "check_rate_limit": _check_fallback,
            "manage_rate_limits": _manage_fallback,
        }


_API = _load_rate_limiting_tools_api()


async def configure_rate_limits(
    limits: list[Dict[str, Any]],
    apply_immediately: bool = True,
    backup_current: bool = True,
) -> Dict[str, Any]:
    """Configure rate-limiting rules."""
    result = _API["configure_rate_limits"](
        limits=limits,
        apply_immediately=apply_immediately,
        backup_current=backup_current,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def check_rate_limit(
    limit_name: str,
    identifier: str = "default",
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check whether a request is within a named rate limit."""
    result = _API["check_rate_limit"](
        limit_name=limit_name,
        identifier=identifier,
        request_metadata=request_metadata,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def manage_rate_limits(
    action: str,
    limit_name: Optional[str] = None,
    new_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manage rate-limiting configuration and statistics."""
    result = _API["manage_rate_limits"](
        action=action,
        limit_name=limit_name,
        new_config=new_config,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_rate_limiting_tools_category(manager: Any) -> None:
    """Register native rate-limiting-tools category tools in unified manager."""
    manager.register_tool(
        category="rate_limiting_tools",
        name="configure_rate_limits",
        func=configure_rate_limits,
        description="Configure named rate limit rules.",
        input_schema={
            "type": "object",
            "properties": {
                "limits": {"type": "array", "items": {"type": "object"}},
                "apply_immediately": {"type": "boolean"},
                "backup_current": {"type": "boolean"},
            },
            "required": ["limits"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "rate-limiting-tools"],
    )

    manager.register_tool(
        category="rate_limiting_tools",
        name="check_rate_limit",
        func=check_rate_limit,
        description="Check if an identifier is allowed by a named limit.",
        input_schema={
            "type": "object",
            "properties": {
                "limit_name": {"type": "string"},
                "identifier": {"type": "string"},
                "request_metadata": {"type": ["object", "null"]},
            },
            "required": ["limit_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "rate-limiting-tools"],
    )

    manager.register_tool(
        category="rate_limiting_tools",
        name="manage_rate_limits",
        func=manage_rate_limits,
        description="List and manage rate-limit configurations.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "limit_name": {"type": ["string", "null"]},
                "new_config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "rate-limiting-tools"],
    )
