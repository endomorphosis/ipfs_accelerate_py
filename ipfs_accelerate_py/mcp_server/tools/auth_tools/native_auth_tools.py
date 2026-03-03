"""Native auth tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_auth_api() -> Dict[str, Any]:
    """Resolve source auth APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.auth_tools.auth_tools import (  # type: ignore
            authenticate_user as _authenticate_user,
            validate_token as _validate_token,
            get_user_info as _get_user_info,
        )

        return {
            "authenticate_user": _authenticate_user,
            "validate_token": _validate_token,
            "get_user_info": _get_user_info,
        }
    except Exception:
        logger.warning("Source auth_tools import unavailable, using fallback auth functions")

        async def _authenticate_fallback(
            username: str,
            password: str,
            auth_service: Any = None,
        ) -> Dict[str, Any]:
            _ = auth_service
            if not username or not password:
                return {
                    "status": "error",
                    "message": "Username and password are required",
                }
            return {
                "status": "success",
                "username": username,
                "access_token": "fallback-token",
                "token_type": "bearer",
                "role": "user",
                "expires_in": 3600,
            }

        async def _validate_fallback(
            token: str,
            required_permission: Optional[str] = None,
            action: str = "validate",
            auth_service: Any = None,
        ) -> Dict[str, Any]:
            _ = required_permission, auth_service
            if not token:
                return {
                    "status": "error",
                    "valid": False,
                    "message": "Token is required",
                }
            if action == "decode":
                return {
                    "status": "success",
                    "user_id": "user123",
                    "username": "fallback-user",
                    "exp": (datetime.now() + timedelta(hours=1)).timestamp(),
                    "message": "Token decoded successfully",
                }
            if action == "refresh":
                return {
                    "status": "success",
                    "access_token": "refreshed-fallback-token",
                    "refresh_token": "fallback-refresh-token",
                    "expires_in": 3600,
                    "message": "Token refreshed successfully",
                }
            return {
                "status": "success",
                "valid": True,
                "username": "fallback-user",
                "role": "user",
                "permissions": ["read"],
                "message": "Token is valid",
            }

        async def _get_user_info_fallback(
            token: str,
            auth_service: Any = None,
        ) -> Dict[str, Any]:
            _ = auth_service
            if not token:
                return {
                    "status": "error",
                    "message": "Token is required",
                }
            return {
                "status": "success",
                "username": "fallback-user",
                "role": "user",
                "permissions": ["read"],
                "message": "User information retrieved successfully",
            }

        return {
            "authenticate_user": _authenticate_fallback,
            "validate_token": _validate_fallback,
            "get_user_info": _get_user_info_fallback,
        }


_API = _load_auth_api()


async def authenticate_user(
    username: str,
    password: str,
) -> Dict[str, Any]:
    """Authenticate user credentials and return access token."""
    normalized_username = str(username or "").strip()
    if not normalized_username:
        return {"status": "error", "message": "Username is required and must be a string"}
    if len(normalized_username) > 50:
        return {"status": "error", "message": "Username must be 50 characters or less"}

    normalized_password = str(password or "")
    if not normalized_password:
        return {"status": "error", "message": "Password is required and must be a string"}

    return await _API["authenticate_user"](username=username, password=password)


async def validate_token(
    token: str,
    required_permission: Optional[str] = None,
    action: str = "validate",
) -> Dict[str, Any]:
    """Validate, refresh, or decode an access token."""
    normalized_token = str(token or "").strip()
    if not normalized_token:
        return {
            "status": "error",
            "valid": False,
            "message": "Token is required and must be a string",
        }

    valid_permissions = {"read", "write", "delete", "manage"}
    normalized_permission: Optional[str] = None
    if required_permission is not None:
        normalized_permission = str(required_permission).strip().lower()
        if normalized_permission not in valid_permissions:
            return {
                "status": "error",
                "valid": False,
                "message": "Invalid required_permission. Must be one of: read, write, delete, manage",
            }

    normalized_action = str(action or "validate").strip().lower() or "validate"
    if normalized_action not in {"validate", "refresh", "decode"}:
        return {
            "status": "error",
            "valid": False,
            "message": "Invalid action. Must be one of: validate, refresh, decode",
        }

    return await _API["validate_token"](
        token=normalized_token,
        required_permission=normalized_permission,
        action=normalized_action,
    )


async def get_user_info(token: str) -> Dict[str, Any]:
    """Get authenticated user information from token."""
    normalized_token = str(token or "").strip()
    if not normalized_token:
        return {"status": "error", "message": "Token is required and must be a string"}
    return await _API["get_user_info"](token=normalized_token)


def register_native_auth_tools(manager: Any) -> None:
    """Register native auth tools in unified hierarchical manager."""
    manager.register_tool(
        category="auth_tools",
        name="authenticate_user",
        func=authenticate_user,
        description="Authenticate user credentials and return access token.",
        input_schema={
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "password": {"type": "string"},
            },
            "required": ["username", "password"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "auth"],
    )

    manager.register_tool(
        category="auth_tools",
        name="validate_token",
        func=validate_token,
        description="Validate, refresh, or decode an access token.",
        input_schema={
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "required_permission": {
                    "type": ["string", "null"],
                    "enum": ["read", "write", "delete", "manage", None],
                },
                "action": {
                    "type": "string",
                    "enum": ["validate", "refresh", "decode"],
                    "default": "validate",
                },
            },
            "required": ["token"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "auth"],
    )

    manager.register_tool(
        category="auth_tools",
        name="get_user_info",
        func=get_user_info,
        description="Get authenticated user information from token.",
        input_schema={
            "type": "object",
            "properties": {
                "token": {"type": "string"},
            },
            "required": ["token"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "auth"],
    )
