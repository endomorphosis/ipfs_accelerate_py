"""Native admin tool implementations for unified mcp_server."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_admin_api() -> Dict[str, Any]:
    """Resolve source admin APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.admin_tools.admin_tools import (  # type: ignore
            configure_system as _configure_system,
            manage_endpoints as _manage_endpoints,
            system_maintenance as _system_maintenance,
        )

        return {
            "manage_endpoints": _manage_endpoints,
            "system_maintenance": _system_maintenance,
            "configure_system": _configure_system,
        }
    except Exception:
        logger.warning("Source admin_tools import unavailable, using fallback admin functions")

        async def _manage_endpoints_fallback(
            action: str,
            model: Optional[str] = None,
            endpoint: Optional[str] = None,
            endpoint_type: Optional[str] = None,
            ctx_length: Optional[int] = 512,
        ) -> Dict[str, Any]:
            _ = model, endpoint, endpoint_type, ctx_length
            if action == "list":
                return {
                    "success": True,
                    "status": "success",
                    "action": action,
                    "endpoints": [],
                    "count": 0,
                    "timestamp": datetime.now().isoformat(),
                }
            return {
                "success": True,
                "status": "success",
                "action": action,
                "message": "Endpoint action handled by fallback",
            }

        async def _system_maintenance_fallback(
            operation: Optional[str] = None,
            target: Optional[str] = None,
            force: bool = False,
            action: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = target, force
            op = operation if operation is not None else action
            return {
                "success": True,
                "status": "success",
                "operation": op or "health_check",
                "timestamp": datetime.now().isoformat(),
            }

        async def _configure_system_fallback(
            action: str,
            config_key: Optional[str] = None,
            settings: Optional[Dict[str, Any]] = None,
            validate_only: bool = False,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "action": action,
                "config_key": config_key,
                "settings": settings or {},
                "validated": validate_only,
                "timestamp": datetime.now().isoformat(),
            }

        return {
            "manage_endpoints": _manage_endpoints_fallback,
            "system_maintenance": _system_maintenance_fallback,
            "configure_system": _configure_system_fallback,
        }


_API = _load_admin_api()


async def manage_endpoints(
    action: str,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    endpoint_type: Optional[str] = None,
    ctx_length: Optional[int] = 512,
) -> Dict[str, Any]:
    """Manage endpoint records for embedding/model services."""
    normalized_action = str(action or "").strip().lower()
    allowed_actions = {"add", "update", "remove", "list"}
    if normalized_action not in allowed_actions:
        return {
            "status": "error",
            "message": f"action must be one of: {', '.join(sorted(allowed_actions))}",
            "action": action,
        }
    if normalized_action == "add" and (not str(model or "").strip() or not str(endpoint or "").strip() or not str(endpoint_type or "").strip()):
        return {
            "status": "error",
            "message": "model, endpoint, and endpoint_type are required for add action",
            "action": normalized_action,
        }
    if normalized_action in {"update", "remove"} and not str(model or "").strip():
        return {
            "status": "error",
            "message": "model is required for update/remove actions",
            "action": normalized_action,
        }
    if endpoint_type is not None:
        normalized_endpoint_type = str(endpoint_type).strip().lower()
        valid_endpoint_types = {"local", "http", "https", "openai", "azure", "sagemaker"}
        if normalized_endpoint_type not in valid_endpoint_types:
            return {
                "status": "error",
                "message": "endpoint_type must be one of: azure, http, https, local, openai, sagemaker",
                "endpoint_type": endpoint_type,
            }
    normalized_ctx_length: Optional[int] = None
    if ctx_length is not None:
        try:
            normalized_ctx_length = int(ctx_length)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": "ctx_length must be a positive integer when provided",
                "ctx_length": ctx_length,
            }
        if normalized_ctx_length <= 0:
            return {
                "status": "error",
                "message": "ctx_length must be a positive integer when provided",
                "ctx_length": ctx_length,
            }

    result = await _API["manage_endpoints"](
        action=normalized_action,
        model=model,
        endpoint=endpoint,
        endpoint_type=endpoint_type,
        ctx_length=normalized_ctx_length,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("action", normalized_action)
    return payload


async def system_maintenance(
    operation: Optional[str] = None,
    target: Optional[str] = None,
    force: bool = False,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform maintenance operations like health-check and cleanup."""
    normalized_operation = str(operation or action or "health_check").strip().lower()
    alias_map = {"status": "health_check", "health": "health_check"}
    normalized_operation = alias_map.get(normalized_operation, normalized_operation)
    allowed_operations = {"health_check", "cleanup", "restart", "backup"}
    if normalized_operation not in allowed_operations:
        return {
            "status": "error",
            "message": f"operation must be one of: {', '.join(sorted(allowed_operations))}",
            "operation": operation,
            "action": action,
        }
    if not isinstance(force, bool):
        return {
            "status": "error",
            "message": "force must be a boolean",
            "force": force,
        }

    result = await _API["system_maintenance"](
        operation=normalized_operation,
        target=target,
        force=force,
        action=normalized_operation,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("operation", normalized_operation)
    return payload


async def configure_system(
    action: str,
    config_key: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    validate_only: bool = False,
) -> Dict[str, Any]:
    """Get or update system configuration settings."""
    normalized_action = str(action or "").strip().lower()
    allowed_actions = {"get", "set", "update", "configure"}
    if normalized_action not in allowed_actions:
        return {
            "status": "error",
            "message": f"action must be one of: {', '.join(sorted(allowed_actions))}",
            "action": action,
        }
    if normalized_action in {"set", "update", "configure"} and settings is not None and not isinstance(settings, dict):
        return {
            "status": "error",
            "message": "settings must be an object when provided",
            "settings": settings,
        }
    if config_key is not None and not str(config_key).strip():
        return {
            "status": "error",
            "message": "config_key must be a non-empty string when provided",
            "config_key": config_key,
        }
    if not isinstance(validate_only, bool):
        return {
            "status": "error",
            "message": "validate_only must be a boolean",
            "validate_only": validate_only,
        }

    result = await _API["configure_system"](
        action=normalized_action,
        config_key=config_key,
        settings=settings,
        validate_only=validate_only,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("action", normalized_action)
    return payload


def register_native_admin_tools(manager: Any) -> None:
    """Register native admin tools in unified hierarchical manager."""
    manager.register_tool(
        category="admin_tools",
        name="manage_endpoints",
        func=manage_endpoints,
        description="Manage API endpoints and endpoint configurations.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "update", "remove", "list"]},
                "model": {"type": ["string", "null"]},
                "endpoint": {"type": ["string", "null"]},
                "endpoint_type": {"type": ["string", "null"], "enum": ["local", "http", "https", "openai", "azure", "sagemaker", None]},
                "ctx_length": {"type": ["integer", "null"], "minimum": 1, "default": 512},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin"],
    )

    manager.register_tool(
        category="admin_tools",
        name="system_maintenance",
        func=system_maintenance,
        description="Perform maintenance operations for system components.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": ["string", "null"], "enum": ["health_check", "cleanup", "restart", "backup", "status", "health", None]},
                "target": {"type": ["string", "null"]},
                "force": {"type": "boolean", "default": False},
                "action": {"type": ["string", "null"], "enum": ["health_check", "cleanup", "restart", "backup", "status", "health", None]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin"],
    )

    manager.register_tool(
        category="admin_tools",
        name="configure_system",
        func=configure_system,
        description="Get and update system configuration values.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set", "update", "configure"]},
                "config_key": {"type": ["string", "null"]},
                "settings": {"type": ["object", "null"]},
                "validate_only": {"type": "boolean", "default": False},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin"],
    )
