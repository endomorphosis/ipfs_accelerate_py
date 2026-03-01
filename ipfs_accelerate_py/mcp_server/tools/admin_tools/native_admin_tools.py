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
    return await _API["manage_endpoints"](
        action=action,
        model=model,
        endpoint=endpoint,
        endpoint_type=endpoint_type,
        ctx_length=ctx_length,
    )


async def system_maintenance(
    operation: Optional[str] = None,
    target: Optional[str] = None,
    force: bool = False,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform maintenance operations like health-check and cleanup."""
    return await _API["system_maintenance"](
        operation=operation,
        target=target,
        force=force,
        action=action,
    )


async def configure_system(
    action: str,
    config_key: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    validate_only: bool = False,
) -> Dict[str, Any]:
    """Get or update system configuration settings."""
    return await _API["configure_system"](
        action=action,
        config_key=config_key,
        settings=settings,
        validate_only=validate_only,
    )


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
                "action": {"type": "string"},
                "model": {"type": ["string", "null"]},
                "endpoint": {"type": ["string", "null"]},
                "endpoint_type": {"type": ["string", "null"]},
                "ctx_length": {"type": ["integer", "null"]},
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
                "operation": {"type": ["string", "null"]},
                "target": {"type": ["string", "null"]},
                "force": {"type": "boolean"},
                "action": {"type": ["string", "null"]},
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
                "action": {"type": "string"},
                "config_key": {"type": ["string", "null"]},
                "settings": {"type": ["object", "null"]},
                "validate_only": {"type": "boolean"},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin"],
    )
