"""Native rate-limiting-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    return payload


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
            normalized_action = str(action or "").strip().lower()
            if normalized_action == "list":
                return {"action": "list", "limits": [], "total_count": 0}
            if normalized_action in {"enable", "disable", "delete"}:
                if not str(limit_name or "").strip():
                    return {"error": f"limit_name required for {normalized_action} action"}
                return {
                    "action": normalized_action,
                    "limit_name": str(limit_name),
                    "status": "success",
                }
            if normalized_action == "update":
                if not str(limit_name or "").strip() or not isinstance(new_config, dict):
                    return {"error": "limit_name and new_config required for update action"}
                return {
                    "action": "update",
                    "limit_name": str(limit_name),
                    "updated_config": dict(new_config),
                    "status": "success",
                }
            if normalized_action == "stats":
                return {
                    "total_requests": 0,
                    "allowed_requests": 0,
                    "denied_requests": 0,
                    "limit_name": limit_name,
                }
            if normalized_action == "reset":
                return {
                    "status": "success",
                    "action": "reset",
                    "limit_name": limit_name,
                }
            return {"error": f"Unknown action: {action}"}

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
    if not isinstance(limits, list):
        return _error_result(
            "limits must be a list",
            configured_count=0,
            configured_limits=[],
            errors=["limits must be a list"],
        )
    if not isinstance(apply_immediately, bool):
        return _error_result("apply_immediately must be a boolean")
    if not isinstance(backup_current, bool):
        return _error_result("backup_current must be a boolean")

    for index, item in enumerate(limits):
        if not isinstance(item, dict):
            return _error_result(f"limits[{index}] must be an object")

    try:
        result = _API["configure_rate_limits"](
            limits=limits,
            apply_immediately=apply_immediately,
            backup_current=backup_current,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return _error_result(f"configure_rate_limits failed: {exc}")

    payload = dict(result or {})
    if "status" not in payload:
        payload["status"] = "error" if payload.get("errors") else "success"
    return payload


async def check_rate_limit(
    limit_name: str,
    identifier: str = "default",
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check whether a request is within a named rate limit."""
    normalized_limit_name = str(limit_name or "").strip()
    if not normalized_limit_name:
        return _error_result(
            "limit_name is required",
            allowed=False,
            limit_name=limit_name,
            identifier=str(identifier or "default"),
        )

    normalized_identifier = str(identifier or "").strip()
    if not normalized_identifier:
        return _error_result(
            "identifier must be a non-empty string",
            allowed=False,
            limit_name=normalized_limit_name,
            identifier=str(identifier or ""),
        )

    if request_metadata is not None and not isinstance(request_metadata, dict):
        return _error_result(
            "request_metadata must be an object or null",
            allowed=False,
            limit_name=normalized_limit_name,
            identifier=normalized_identifier,
        )

    try:
        result = _API["check_rate_limit"](
            limit_name=normalized_limit_name,
            identifier=normalized_identifier,
            request_metadata=request_metadata,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return _error_result(
            f"check_rate_limit failed: {exc}",
            allowed=False,
            limit_name=normalized_limit_name,
            identifier=normalized_identifier,
        )

    payload = dict(result or {})
    payload.setdefault("status", "success" if payload.get("allowed", True) else "error")
    payload.setdefault("limit_name", normalized_limit_name)
    payload.setdefault("identifier", normalized_identifier)
    return payload


async def manage_rate_limits(
    action: str,
    limit_name: Optional[str] = None,
    new_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manage rate-limiting configuration and statistics."""
    normalized_action = str(action or "").strip().lower()
    if not normalized_action:
        return _error_result(
            "action is required",
            valid_actions=["list", "enable", "disable", "delete", "update", "stats", "reset"],
        )

    valid_actions = {"list", "enable", "disable", "delete", "update", "stats", "reset"}
    if normalized_action not in valid_actions:
        return _error_result(
            f"Unknown action: {action}",
            valid_actions=["list", "enable", "disable", "delete", "update", "stats", "reset"],
        )

    normalized_limit_name = str(limit_name or "").strip() if limit_name is not None else None

    if normalized_action in {"enable", "disable", "delete"} and not normalized_limit_name:
        return _error_result(f"limit_name required for {normalized_action} action")

    if normalized_action == "update" and (not normalized_limit_name or not isinstance(new_config, dict)):
        return _error_result("limit_name and new_config required for update action")

    if normalized_action in {"stats", "reset"} and limit_name is not None and not normalized_limit_name:
        return _error_result(f"limit_name must be a non-empty string when provided for {normalized_action}")

    try:
        result = _API["manage_rate_limits"](
            action=normalized_action,
            limit_name=normalized_limit_name,
            new_config=new_config,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return _error_result(f"manage_rate_limits failed: {exc}", action=normalized_action)

    payload = dict(result or {})
    payload.setdefault("action", normalized_action)
    if "error" in payload:
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    return payload


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
                "apply_immediately": {"type": "boolean", "default": True},
                "backup_current": {"type": "boolean", "default": True},
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
                "limit_name": {"type": "string", "minLength": 1},
                "identifier": {"type": "string", "minLength": 1, "default": "default"},
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
                "action": {
                    "type": "string",
                    "enum": ["list", "enable", "disable", "delete", "update", "stats", "reset"],
                },
                "limit_name": {"type": ["string", "null"], "minLength": 1},
                "new_config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "rate-limiting-tools"],
    )
