"""Native security tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _load_check_access_permission() -> Any:
    """Resolve source check_access_permission tool with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.security_tools.check_access_permission import (  # type: ignore
            check_access_permission as source_check_access_permission,
        )

        return source_check_access_permission
    except Exception:

        async def _fallback_check_access_permission(
            resource_id: str,
            user_id: Optional[str] = None,
            permission_type: str = "read",
            resource_type: Optional[str] = None,
        ) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "security backend unavailable",
                "allowed": False,
                "user_id": user_id,
                "resource_id": resource_id,
                "permission_type": permission_type,
                "resource_type": resource_type,
            }

        return _fallback_check_access_permission


_CHECK_ACCESS_PERMISSION = _load_check_access_permission()


async def check_access_permission(
    resource_id: str,
    user_id: Optional[str] = None,
    permission_type: str = "read",
    resource_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Check whether a user has access permission to a resource."""
    return await _CHECK_ACCESS_PERMISSION(
        resource_id=resource_id,
        user_id=user_id,
        permission_type=permission_type,
        resource_type=resource_type,
    )


def register_native_security_tools(manager: Any) -> None:
    """Register native security tools in unified hierarchical manager."""
    manager.register_tool(
        category="security_tools",
        name="check_access_permission",
        func=check_access_permission,
        description="Check user permissions for accessing a resource.",
        input_schema={
            "type": "object",
            "properties": {
                "resource_id": {"type": "string"},
                "user_id": {"type": ["string", "null"]},
                "permission_type": {"type": "string"},
                "resource_type": {"type": ["string", "null"]},
            },
            "required": ["resource_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "security"],
    )
