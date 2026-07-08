"""Native security tool implementations for unified mcp_server."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

_VALID_PERMISSION_TYPES = {"read", "write", "delete", "share", "admin", "execute"}


def _mcp_text_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build MCP text envelope used by legacy JSON-string call paths."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload),
            }
        ]
    }


def _mcp_error_response(message: str, *, error_type: str = "error") -> Dict[str, Any]:
    return _mcp_text_response(
        {
            "status": "error",
            "error": message,
            "error_type": error_type,
        }
    )


def _parse_json_object(request_json: Any) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse JSON-string object payload for source-compatible MCP entrypoints."""
    if not isinstance(request_json, str):
        return None, _mcp_error_response("Input must be a JSON string")

    if not request_json.strip():
        return None, _mcp_error_response("Input JSON is empty", error_type="validation")

    try:
        decoded = json.loads(request_json)
    except json.JSONDecodeError as exc:
        return None, _mcp_error_response(f"Invalid JSON: {exc.msg}", error_type="validation")

    if not isinstance(decoded, dict):
        return None, _mcp_error_response("Input JSON must be an object", error_type="validation")

    return decoded, None


def _load_check_access_permission() -> Any:
    """Resolve source check_access_permission tool with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.security_tools.check_access_permission import (
            # type: ignore
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
    if isinstance(resource_id, str) and user_id is None and (
        not resource_id.strip()
        or resource_id.lstrip().startswith("{")
        or resource_id.lstrip().startswith("[")
        or any(ch.isspace() for ch in resource_id)
    ):
        data, error = _parse_json_object(resource_id)
        if error is not None:
            return error

        for field in ("resource_id", "user_id"):
            if not data.get(field):
                return _mcp_error_response(f"Missing required field: {field}", error_type="validation")

        payload = await check_access_permission(
            resource_id=str(data["resource_id"]),
            user_id=str(data["user_id"]),
            permission_type=str(data.get("permission_type", "read")),
            resource_type=data.get("resource_type"),
        )
        return _mcp_text_response(payload)

    normalized_resource_id = str(resource_id or "").strip()
    normalized_user_id = str(user_id or "").strip()
    normalized_permission_type = str(permission_type or "read").strip().lower() or "read"
    normalized_resource_type = (
        None if resource_type is None else str(resource_type).strip()
    )

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
    if resource_type is not None and not normalized_resource_type:
        return {
            "status": "error",
            "error": "resource_type must be a non-empty string when provided",
            "allowed": False,
            "user_id": normalized_user_id,
            "resource_id": normalized_resource_id,
            "permission_type": normalized_permission_type,
            "resource_type": resource_type,
        }

    try:
        result = await _CHECK_ACCESS_PERMISSION(
            resource_id=normalized_resource_id,
            user_id=normalized_user_id,
            permission_type=normalized_permission_type,
            resource_type=normalized_resource_type,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "allowed": False,
            "user_id": normalized_user_id,
            "resource_id": normalized_resource_id,
            "permission_type": normalized_permission_type,
            "resource_type": normalized_resource_type,
        }

    if isinstance(result, dict):
        normalized = dict(result)
        failed = (
            normalized.get("status") == "error"
            or normalized.get("error")
            or normalized.get("success") is False
        )
        if failed:
            normalized["status"] = "error"
            normalized.setdefault("success", False)
        else:
            normalized.setdefault("status", "success")
        normalized.setdefault("allowed", False)
        normalized.setdefault("user_id", normalized_user_id)
        normalized.setdefault("resource_id", normalized_resource_id)
        normalized.setdefault("permission_type", normalized_permission_type)
        normalized.setdefault("resource_type", normalized_resource_type)
        return normalized

    return {
        "status": "success",
        "allowed": False,
        "result": result,
        "user_id": normalized_user_id,
        "resource_id": normalized_resource_id,
        "permission_type": normalized_permission_type,
        "resource_type": normalized_resource_type,
    }


async def check_access_permissions_batch(
    requests: List[Dict[str, Any]],
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Evaluate multiple access checks with deterministic aggregate envelope."""
    if not isinstance(fail_fast, bool):
        requested = len(requests) if isinstance(requests, list) else 0
        return {
            "status": "error",
            "error": "fail_fast must be a boolean",
            "results": [],
            "processed": 0,
            "requested": requested,
            "all_allowed": False,
            "allowed_count": 0,
            "denied_count": 0,
            "error_count": 0,
            "fail_fast": fail_fast,
        }

    if not isinstance(requests, list) or not requests:
        return {
            "status": "error",
            "error": "requests must be a non-empty array",
            "results": [],
            "processed": 0,
            "requested": len(requests) if isinstance(requests, list) else 0,
            "all_allowed": False,
            "allowed_count": 0,
            "denied_count": 0,
            "error_count": 0,
            "fail_fast": fail_fast,
        }

    results: List[Dict[str, Any]] = []
    allowed_count = 0
    denied_count = 0
    error_count = 0

    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            item_result = {
                "status": "error",
                "error": "request entry must be an object",
                "allowed": False,
                "index": index,
            }
        else:
            item_result = await check_access_permission(
                resource_id=str(request.get("resource_id", "")),
                user_id=request.get("user_id"),
                permission_type=str(request.get("permission_type", "read")),
                resource_type=request.get("resource_type"),
            )
            item_result = dict(item_result)
            item_result.setdefault("index", index)

        if item_result.get("status") == "success" and bool(item_result.get("allowed", False)):
            allowed_count += 1
        elif item_result.get("status") == "error":
            error_count += 1
        else:
            denied_count += 1

        results.append(item_result)

        if fail_fast and item_result.get("status") == "error":
            break

    return {
        "status": "success",
        "results": results,
        "processed": len(results),
        "requested": len(requests),
        "all_allowed": error_count == 0 and denied_count == 0,
        "allowed_count": allowed_count,
        "denied_count": denied_count,
        "error_count": error_count,
        "fail_fast": bool(fail_fast),
    }


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
                "resource_id": {"type": "string", "minLength": 1},
                "user_id": {"type": "string", "minLength": 1},
                "permission_type": {
                    "type": "string",
                    "enum": sorted(_VALID_PERMISSION_TYPES),
                    "default": "read",
                },
                "resource_type": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["resource_id", "user_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "security"],
    )

    manager.register_tool(
        category="security_tools",
        name="check_access_permissions_batch",
        func=check_access_permissions_batch,
        description="Check permissions for multiple resource/user requests.",
        input_schema={
            "type": "object",
            "properties": {
                "requests": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "resource_id": {"type": "string", "minLength": 1},
                            "user_id": {"type": "string", "minLength": 1},
                            "permission_type": {
                                "type": "string",
                                "enum": sorted(_VALID_PERMISSION_TYPES),
                                "default": "read",
                            },
                            "resource_type": {
                                "anyOf": [
                                    {"type": "string", "minLength": 1},
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["resource_id", "user_id"],
                    },
                },
                "fail_fast": {"type": "boolean", "default": False},
            },
            "required": ["requests"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "security"],
    )
