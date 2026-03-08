"""Native session tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_session_manager_class() -> Any:
    """Resolve MockSessionManager from source package with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.sessions.session_engine import MockSessionManager  # type: ignore

        return MockSessionManager
    except Exception:
        return None


class _FallbackSessionManager:
    """Dependency-light in-memory fallback for session lifecycle operations."""

    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def create_session(self, **kwargs: Any) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        timeout_seconds = int(kwargs.get("timeout_seconds", 3600) or 3600)
        session = {
            "session_id": session_id,
            "user_id": kwargs.get("user_id", "default_user"),
            "session_name": kwargs.get("session_name", f"Session-{session_id[:8]}"),
            "status": "active",
            "created_at": now,
            "last_activity": now,
            "expires_at": (datetime.now() + timedelta(seconds=timeout_seconds)).isoformat(),
            "config": kwargs.get("session_config", {}),
            "resources": kwargs.get("resource_allocation", {}),
            "request_count": 0,
        }
        self.sessions[session_id] = session
        return dict(session)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(session_id)
        if not session:
            return None
        session["last_activity"] = datetime.now().isoformat()
        return dict(session)

    async def update_session(self, session_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if session_id not in self.sessions:
            return None
        self.sessions[session_id].update(kwargs)
        self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
        return dict(self.sessions[session_id])

    async def delete_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.pop(session_id, None)
        if session:
            session["status"] = "terminated"
        return session

    async def list_sessions(self, **filters: Any) -> list[Dict[str, Any]]:
        sessions = list(self.sessions.values())
        user_id = filters.get("user_id")
        if user_id is not None:
            sessions = [x for x in sessions if x.get("user_id") == user_id]
        status = filters.get("status")
        if status is not None:
            sessions = [x for x in sessions if x.get("status") == status]
        return [dict(x) for x in sessions]

    async def cleanup_expired_sessions(self, max_age_hours: int = 24) -> list[Dict[str, Any]]:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        expired: list[Dict[str, Any]] = []
        for session_id, session in list(self.sessions.items()):
            created_at = session.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_dt = datetime.fromisoformat(created_at)
                except Exception:
                    created_dt = datetime.now()
                if created_dt < cutoff:
                    deleted = await self.delete_session(session_id)
                    if deleted is not None:
                        expired.append(deleted)
        return expired


_SESSION_MANAGER: Optional[Any] = None


def _get_session_manager() -> Any:
    global _SESSION_MANAGER
    if _SESSION_MANAGER is None:
        manager_class = _load_session_manager_class()
        if manager_class is None:
            logger.warning("Source MockSessionManager import unavailable, using fallback manager")
            _SESSION_MANAGER = _FallbackSessionManager()
        else:
            _SESSION_MANAGER = manager_class()
    return _SESSION_MANAGER


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except Exception:
        return False


def _normalize_datetime_value(value: Any, fallback: datetime) -> tuple[str, datetime]:
    """Return a stable ISO timestamp string and datetime object."""
    if isinstance(value, str) and value:
        try:
            return value, datetime.fromisoformat(value)
        except Exception:
            return value, fallback
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(), value
        except Exception:
            return fallback.isoformat(), fallback
    return fallback.isoformat(), fallback


def _normalize_session_payload(
    session: Optional[Dict[str, Any]],
    *,
    session_id: Optional[str] = None,
    session_name: str = "",
    user_id: str = "",
    session_type: str = "interactive",
    status: str = "active",
    config: Optional[Dict[str, Any]] = None,
    resources: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Apply source-like defaults to sparse session payloads."""
    payload = dict(session or {})
    now = datetime.now()

    created_at_str, created_at_dt = _normalize_datetime_value(payload.get("created_at"), now)
    timeout_seconds = 3600
    raw_timeout = (payload.get("config") or payload.get("configuration") or config or {}).get("timeout_seconds", 3600)
    try:
        timeout_seconds = int(raw_timeout or 3600)
    except Exception:
        timeout_seconds = 3600
    expires_at_str, _ = _normalize_datetime_value(
        payload.get("expires_at"),
        created_at_dt + timedelta(seconds=timeout_seconds),
    )
    last_activity_str, _ = _normalize_datetime_value(payload.get("last_activity"), created_at_dt)

    normalized_config = payload.get("config", payload.get("configuration", config or {}))
    normalized_resources = payload.get("resources", payload.get("resource_limits", resources or {}))

    return {
        **payload,
        "session_id": str(payload.get("session_id") or session_id or ""),
        "session_name": payload.get("session_name", session_name),
        "user_id": payload.get("user_id", user_id),
        "session_type": payload.get("session_type", session_type),
        "status": payload.get("status", status),
        "created_at": created_at_str,
        "expires_at": expires_at_str,
        "last_activity": last_activity_str,
        "config": normalized_config,
        "resources": normalized_resources,
        "metadata": payload.get("metadata", metadata or {}),
        "tags": payload.get("tags", tags or []),
        "request_count": int(payload.get("request_count", 0) or 0),
    }


async def create_session(
    session_name: str,
    user_id: str = "default_user",
    session_type: str = "interactive",
    session_config: Optional[Dict[str, Any]] = None,
    resource_allocation: Optional[Dict[str, Any]] = None,
    resource_limits: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    session_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create and initialize a new session."""
    try:
        if not session_name or not isinstance(session_name, str):
            return {"status": "error", "message": "Session name is required and must be a string"}
        if len(session_name) > 100:
            return {"status": "error", "message": "Session name must be 100 characters or less"}
        if not user_id or not isinstance(user_id, str):
            return {"status": "error", "message": "User ID is required and must be a string"}
        if not isinstance(session_type, str) or not session_type.strip():
            return {"status": "error", "message": "session_type must be a non-empty string"}
        if metadata is not None and not isinstance(metadata, dict):
            return {"status": "error", "message": "metadata must be an object"}
        if tags is not None and (
            not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags)
        ):
            return {"status": "error", "message": "tags must be a list of non-empty strings"}

        config = session_config or {
            "models": ["sentence-transformers/all-MiniLM-L6-v2"],
            "max_requests_per_minute": 100,
            "max_concurrent_requests": 10,
            "timeout_seconds": 3600,
            "auto_cleanup": True,
        }
        resources = resource_limits or resource_allocation or {
            "memory_limit_mb": 2048,
            "cpu_cores": 1.0,
            "gpu_enabled": False,
        }

        manager = session_manager or _get_session_manager()
        session = await manager.create_session(
            session_name=session_name,
            user_id=user_id,
            session_type=session_type,
            session_config=config,
            resource_allocation=resources,
            resource_limits=resources,
            metadata=metadata or {},
            tags=tags or [],
            timeout_seconds=config.get("timeout_seconds", 3600),
        )

        normalized_session = _normalize_session_payload(
            session,
            session_name=session_name,
            user_id=user_id,
            session_type=session_type,
            config=config,
            resources=resources,
            metadata=metadata,
            tags=tags,
        )

        return {
            "status": "success",
            "session_id": normalized_session["session_id"],
            "session_name": normalized_session["session_name"],
            "user_id": normalized_session["user_id"],
            "session_type": normalized_session["session_type"],
            "created_at": normalized_session["created_at"],
            "expires_at": normalized_session["expires_at"],
            "config": normalized_session["config"],
            "resources": normalized_session["resources"],
            "metadata": normalized_session["metadata"],
            "tags": normalized_session["tags"],
            "session": normalized_session,
            "message": f"Session '{session_name}' created successfully",
        }
    except Exception as exc:
        logger.error("Session creation error: %s", exc)
        return {"status": "error", "message": f"Failed to create session: {exc}"}


async def manage_session_state(session_id: str, action: str, **kwargs: Any) -> Dict[str, Any]:
    """Manage session state and lifecycle operations."""
    try:
        if not session_id or not isinstance(session_id, str):
            return {"status": "error", "message": "Session ID is required and must be a string"}

        valid_actions = {"get", "update", "pause", "resume", "extend", "delete"}
        if action not in valid_actions:
            return {
                "status": "error",
                "message": f"Invalid action. Must be one of: {', '.join(sorted(valid_actions))}",
            }

        if not _is_valid_uuid(session_id):
            return {"status": "error", "message": "Invalid session ID format"}

        manager = _get_session_manager()
        if action == "get":
            session = await manager.get_session(session_id)
            if not session:
                return {"status": "error", "message": "Session not found"}
            return {
                "status": "success",
                "session": {
                    "session_id": session.get("session_id", session_id),
                    "session_name": session.get("session_name", ""),
                    "user_id": session.get("user_id", ""),
                    "status": session.get("status", "unknown"),
                    "created_at": session.get("created_at", ""),
                    "expires_at": session.get("expires_at", ""),
                    "last_activity": session.get("last_activity", ""),
                    "request_count": int(session.get("request_count", 0)),
                },
                "message": "Session retrieved successfully",
            }

        if action == "update":
            session = await manager.update_session(session_id, **kwargs)
            if not session:
                return {"status": "error", "message": "Session not found"}
            return {
                "status": "success",
                "session_id": session_id,
                "updated_fields": list(kwargs.keys()),
                "message": "Session updated successfully",
            }

        if action == "pause":
            session = await manager.update_session(session_id, status="paused")
            if not session:
                return {"status": "error", "message": "Session not found"}
            return {
                "status": "success",
                "session_id": session_id,
                "session_status": "paused",
                "message": "Session paused successfully",
            }

        if action == "resume":
            session = await manager.update_session(session_id, status="active")
            if not session:
                return {"status": "error", "message": "Session not found"}
            return {
                "status": "success",
                "session_id": session_id,
                "session_status": "active",
                "message": "Session resumed successfully",
            }

        if action == "extend":
            extend_minutes = kwargs.get("extend_minutes", 60)
            if not isinstance(extend_minutes, int) or extend_minutes <= 0:
                return {"status": "error", "message": "extend_minutes must be a positive integer"}
            session = await manager.get_session(session_id)
            if not session:
                return {"status": "error", "message": "Session not found"}
            current_expires = session.get("expires_at")
            if isinstance(current_expires, str):
                current_expires_dt = datetime.fromisoformat(current_expires)
            else:
                current_expires_dt = current_expires
            new_expires_at = current_expires_dt + timedelta(minutes=extend_minutes)
            await manager.update_session(session_id, expires_at=new_expires_at.isoformat())
            return {
                "status": "success",
                "session_id": session_id,
                "new_expires_at": new_expires_at.isoformat(),
                "extended_by_minutes": extend_minutes,
                "message": f"Session extended by {extend_minutes} minutes",
            }

        if action == "delete":
            deleted = await manager.delete_session(session_id)
            if not deleted:
                return {"status": "error", "message": "Session not found"}
            return {
                "status": "success",
                "session_id": session_id,
                "message": "Session deleted successfully",
            }

        return {"status": "error", "message": "Unsupported action"}
    except Exception as exc:
        logger.error("Session management error: %s", exc)
        return {"status": "error", "message": f"Session management failed: {exc}"}


async def cleanup_sessions(
    cleanup_type: str = "expired",
    user_id: Optional[str] = None,
    session_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Clean up sessions and release resources."""
    try:
        valid_types = {"expired", "all", "by_user"}
        if cleanup_type not in valid_types:
            return {
                "status": "error",
                "message": f"Invalid cleanup_type. Must be one of: {', '.join(sorted(valid_types))}",
            }
        if cleanup_type == "by_user" and not user_id:
            return {"status": "error", "message": "user_id is required for by_user cleanup"}

        manager = session_manager or _get_session_manager()
        if cleanup_type == "expired":
            expired = await manager.cleanup_expired_sessions()
            expired_count = len(expired) if isinstance(expired, list) else int(expired)
            return {
                "status": "success",
                "cleanup_type": "expired",
                "sessions_cleaned": expired_count,
                "message": f"Cleaned up {expired_count} expired sessions",
            }

        if cleanup_type == "all":
            sessions = await manager.list_sessions()
            deleted_count = 0
            for session in sessions:
                if await manager.delete_session(str(session.get("session_id", ""))):
                    deleted_count += 1
            return {
                "status": "success",
                "cleanup_type": "all",
                "sessions_cleaned": deleted_count,
                "message": f"Cleaned up {deleted_count} sessions",
            }

        if cleanup_type == "by_user":
            user_sessions = await manager.list_sessions(user_id=user_id)
            deleted_count = 0
            for session in user_sessions:
                if await manager.delete_session(str(session.get("session_id", ""))):
                    deleted_count += 1
            return {
                "status": "success",
                "cleanup_type": "by_user",
                "user_id": user_id,
                "sessions_cleaned": deleted_count,
                "message": f"Cleaned up {deleted_count} sessions for user {user_id}",
            }

        return {"status": "error", "message": "Unsupported cleanup type"}
    except Exception as exc:
        logger.error("Session cleanup error: %s", exc)
        return {"status": "error", "message": f"Session cleanup failed: {exc}"}


async def manage_session(
    action: str = "get",
    session_id: Optional[str] = None,
    updates: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    cleanup_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Source-compatible enhanced session lifecycle management wrapper."""
    try:
        normalized_action = str(action or "").strip().lower()

        if normalized_action == "get":
            if not session_id:
                return {
                    "status": "error",
                    "error": "session_id is required for get action",
                    "code": "MISSING_SESSION_ID",
                }
            if not _is_valid_uuid(str(session_id)):
                return {
                    "status": "error",
                    "error": "Invalid session ID format",
                    "code": "INVALID_SESSION_ID",
                }
            session = await _get_session_manager().get_session(str(session_id))
            if not session:
                return {
                    "status": "error",
                    "error": "Session not found",
                    "code": "SESSION_NOT_FOUND",
                }
            normalized_session = _normalize_session_payload(
                session,
                session_id=str(session_id),
                status="unknown",
            )
            return {
                "status": "success",
                "session": normalized_session,
                "message": "Session retrieved successfully",
            }

        if normalized_action == "update":
            if not session_id:
                return {
                    "status": "error",
                    "error": "session_id is required for update action",
                    "code": "MISSING_SESSION_ID",
                }
            if not _is_valid_uuid(str(session_id)):
                return {
                    "status": "error",
                    "error": "Invalid session ID format",
                    "code": "INVALID_SESSION_ID",
                }
            session = await _get_session_manager().update_session(str(session_id), **(updates or {}))
            if not session:
                return {
                    "status": "error",
                    "error": "Session not found",
                    "code": "SESSION_NOT_FOUND",
                }
            normalized_session = _normalize_session_payload(
                session,
                session_id=str(session_id),
                status="unknown",
            )
            return {
                "status": "success",
                "session": normalized_session,
                "message": "Session updated successfully",
            }

        if normalized_action == "delete":
            if not session_id:
                return {
                    "status": "error",
                    "error": "session_id is required for delete action",
                    "code": "MISSING_SESSION_ID",
                }
            if not _is_valid_uuid(str(session_id)):
                return {
                    "status": "error",
                    "error": "Invalid session ID format",
                    "code": "INVALID_SESSION_ID",
                }
            deleted = await _get_session_manager().delete_session(str(session_id))
            if not deleted:
                return {
                    "status": "error",
                    "error": "Session not found",
                    "code": "SESSION_NOT_FOUND",
                }
            return {
                "status": "success",
                "session_id": str(session_id),
                "message": "Session deleted successfully",
            }

        if normalized_action == "list":
            sessions = await _get_session_manager().list_sessions(**(filters or {}))
            return {
                "status": "success",
                "sessions": sessions,
                "count": len(sessions),
                "message": f"Found {len(sessions)} sessions",
            }

        if normalized_action == "cleanup":
            if cleanup_options is not None and not isinstance(cleanup_options, dict):
                return {
                    "status": "error",
                    "error": "cleanup_options must be an object when provided",
                    "code": "INVALID_CLEANUP_OPTIONS",
                }

            opts = cleanup_options or {}
            max_age_hours = opts.get("max_age_hours", 24)
            if not isinstance(max_age_hours, int) or max_age_hours <= 0:
                return {
                    "status": "error",
                    "error": "cleanup_options.max_age_hours must be a positive integer",
                    "code": "INVALID_CLEANUP_OPTIONS",
                }

            dry_run = opts.get("dry_run", False)
            if not isinstance(dry_run, bool):
                return {
                    "status": "error",
                    "error": "cleanup_options.dry_run must be a boolean",
                    "code": "INVALID_CLEANUP_OPTIONS",
                }

            if not dry_run:
                expired = await _get_session_manager().cleanup_expired_sessions(max_age_hours=max_age_hours)
            else:
                expired = []
            expired_count = len(expired) if isinstance(expired, list) else int(expired)
            return {
                "status": "success",
                "cleaned_up": expired_count,
                "dry_run": dry_run,
                "cleanup_report": {
                    "max_age_hours": max_age_hours,
                    "dry_run": dry_run,
                    "expired_session_count": expired_count,
                },
                "message": f"Cleaned up {expired_count} expired sessions",
            }

        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "code": "UNKNOWN_ACTION",
            "valid_actions": ["get", "update", "delete", "list", "cleanup"],
        }
    except Exception as exc:
        logger.error("Enhanced session management error: %s", exc)
        return {
            "status": "error",
            "error": "Session management failed",
            "code": "MANAGEMENT_FAILED",
            "message": str(exc),
        }


async def get_session_state(
    session_id: str,
    include_metrics: bool = True,
    include_resources: bool = True,
    include_health: bool = True,
) -> Dict[str, Any]:
    """Source-compatible detailed session state wrapper."""
    try:
        if not _is_valid_uuid(str(session_id or "")):
            return {
                "status": "error",
                "error": "Invalid session ID format",
                "code": "INVALID_SESSION_ID",
            }

        session = await _get_session_manager().get_session(str(session_id))
        if not session:
            return {
                "status": "error",
                "error": "Session not found",
                "code": "SESSION_NOT_FOUND",
            }

        normalized_session = _normalize_session_payload(
            session,
            session_id=str(session_id),
            status="unknown",
        )

        state_data: Dict[str, Any] = {
            "session_id": str(session_id),
            "basic_info": {
                "session_name": normalized_session.get("session_name"),
                "user_id": normalized_session.get("user_id"),
                "session_type": normalized_session.get("session_type"),
                "status": normalized_session.get("status"),
                "created_at": normalized_session.get("created_at"),
                "last_activity": normalized_session.get("last_activity"),
            },
        }

        if include_metrics:
            state_data["metrics"] = {
                "total_requests": normalized_session.get("request_count", 0),
                "successful_requests": normalized_session.get("success_count", 0),
                "failed_requests": normalized_session.get("error_count", 0),
                "average_response_time": normalized_session.get("avg_response_time", 0.0),
                "data_processed_mb": normalized_session.get("data_processed_mb", 0.0),
            }

        if include_resources:
            state_data["resource_usage"] = {
                "memory_mb": normalized_session.get("memory_mb", 0),
                "cpu_percent": normalized_session.get("cpu_percent", 0.0),
                "active_connections": normalized_session.get("active_connections", 0),
                "storage_mb": normalized_session.get("storage_mb", 0),
            }

        if include_health:
            health_status = "healthy"
            health_issues: list[str] = []
            if normalized_session.get("status") != "active":
                health_status = "warning"
                health_issues.append(f"Session status is {normalized_session.get('status')}")
            state_data["health_info"] = {
                "status": health_status,
                "issues": health_issues,
                "last_check": datetime.now().isoformat(),
                "checks_passed": len(health_issues) == 0,
            }

        if "metadata" in normalized_session:
            state_data["metadata"] = normalized_session["metadata"]
        if "tags" in normalized_session:
            state_data["tags"] = normalized_session["tags"]

        return {
            "status": "success",
            "session_state": state_data,
            "message": "Session state retrieved successfully",
        }
    except Exception as exc:
        logger.error("Session state retrieval error: %s", exc)
        return {
            "status": "error",
            "error": "Session state retrieval failed",
            "code": "STATE_RETRIEVAL_FAILED",
            "message": str(exc),
        }


def register_native_session_tools(manager: Any) -> None:
    """Register native session lifecycle tools in unified hierarchical manager."""
    manager.register_tool(
        category="session_tools",
        name="create_session",
        func=create_session,
        description="Create and initialize a session.",
        input_schema={
            "type": "object",
            "properties": {
                "session_name": {"type": "string"},
                "user_id": {"type": "string"},
                "session_type": {"type": "string", "default": "interactive"},
                "session_config": {"type": ["object", "null"]},
                "resource_allocation": {"type": ["object", "null"]},
                "resource_limits": {"type": ["object", "null"]},
                "metadata": {"type": ["object", "null"]},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["session_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "session"],
    )

    manager.register_tool(
        category="session_tools",
        name="manage_session_state",
        func=manage_session_state,
        description="Manage session state and lifecycle operations.",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "action": {"type": "string"},
            },
            "required": ["session_id", "action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "session"],
    )

    manager.register_tool(
        category="session_tools",
        name="cleanup_sessions",
        func=cleanup_sessions,
        description="Clean up sessions and release resources.",
        input_schema={
            "type": "object",
            "properties": {
                "cleanup_type": {"type": "string"},
                "user_id": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "session"],
    )

    manager.register_tool(
        category="session_tools",
        name="manage_session",
        func=manage_session,
        description="Enhanced lifecycle management for sessions (get/update/delete/list/cleanup).",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "default": "get"},
                "session_id": {"type": ["string", "null"]},
                "updates": {"type": ["object", "null"]},
                "filters": {"type": ["object", "null"]},
                "cleanup_options": {
                    "type": ["object", "null"],
                    "properties": {
                        "max_age_hours": {"type": "integer", "minimum": 1, "default": 24},
                        "dry_run": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": True,
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "session"],
    )

    manager.register_tool(
        category="session_tools",
        name="get_session_state",
        func=get_session_state,
        description="Get detailed session state, metrics, resources, and health.",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "include_metrics": {"type": "boolean", "default": True},
                "include_resources": {"type": "boolean", "default": True},
                "include_health": {"type": "boolean", "default": True},
            },
            "required": ["session_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "session"],
    )
