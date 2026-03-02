"""Native security tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, Optional

_VALID_PERMISSION_TYPES = {"read", "write", "delete", "share", "admin", "execute"}


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
    normalized_resource_id = str(resource_id or "").strip()
    normalized_user_id = str(user_id or "").strip()
    normalized_permission_type = str(permission_type or "read").strip().lower() or "read"

    if not normalized_resource_id:
        return {
            "status": "error",
            "error": "resource_id must be provided",
            "allowed": False,
            "user_id": user_id,
            "resource_id": resource_id,
            "permission_type": normalized_permission_type,
            "resource_type": resource_type,
        }
    if not normalized_user_id:
        return {
            "status": "error",
            "error": "user_id must be provided",
            "allowed": False,
            "user_id": user_id,
            "resource_id": normalized_resource_id,
            "permission_type": normalized_permission_type,
            "resource_type": resource_type,
        }
    if normalized_permission_type not in _VALID_PERMISSION_TYPES:
        return {
            "status": "error",
            "error": (
                "permission_type must be one of: "
                + ", ".join(sorted(_VALID_PERMISSION_TYPES))
            ),
            "allowed": False,
            "user_id": normalized_user_id,
            "resource_id": normalized_resource_id,
            "permission_type": normalized_permission_type,
            "resource_type": resource_type,
        }

    return await _CHECK_ACCESS_PERMISSION(
        resource_id=normalized_resource_id,
        user_id=normalized_user_id,
        permission_type=normalized_permission_type,
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
                "user_id": {"type": "string"},
                "permission_type": {
                    "type": "string",
                    "enum": sorted(_VALID_PERMISSION_TYPES),
                    "default": "read",
                },
                "resource_type": {"type": ["string", "null"]},
            },
            "required": ["resource_id", "user_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "security"],
    )
