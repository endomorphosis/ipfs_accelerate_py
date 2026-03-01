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


async def create_session(
    session_name: str,
    user_id: str = "default_user",
    session_config: Optional[Dict[str, Any]] = None,
    resource_allocation: Optional[Dict[str, Any]] = None,
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

        config = session_config or {
            "models": ["sentence-transformers/all-MiniLM-L6-v2"],
            "max_requests_per_minute": 100,
            "max_concurrent_requests": 10,
            "timeout_seconds": 3600,
            "auto_cleanup": True,
        }
        resources = resource_allocation or {
            "memory_limit_mb": 2048,
            "cpu_cores": 1.0,
            "gpu_enabled": False,
        }

        manager = session_manager or _get_session_manager()
        session = await manager.create_session(
            session_name=session_name,
            user_id=user_id,
            session_config=config,
            resource_allocation=resources,
            timeout_seconds=config.get("timeout_seconds", 3600),
        )

        created_at = session.get("created_at")
        if isinstance(created_at, str):
            created_at_str = created_at
            try:
                created_at_dt = datetime.fromisoformat(created_at)
            except Exception:
                created_at_dt = datetime.now()
        elif hasattr(created_at, "isoformat"):
            created_at_str = created_at.isoformat()
            created_at_dt = created_at
        else:
            created_at_dt = datetime.now()
            created_at_str = created_at_dt.isoformat()

        expires_at = session.get("expires_at")
        if isinstance(expires_at, str):
            expires_at_str = expires_at
        elif hasattr(expires_at, "isoformat"):
            expires_at_str = expires_at.isoformat()
        else:
            timeout_seconds = int(config.get("timeout_seconds", 3600) or 3600)
            expires_at_str = (created_at_dt + timedelta(seconds=timeout_seconds)).isoformat()

        return {
            "status": "success",
            "session_id": session["session_id"],
            "session_name": session.get("session_name", session_name),
            "user_id": session.get("user_id", user_id),
            "created_at": created_at_str,
            "expires_at": expires_at_str,
            "config": session.get("config", session.get("configuration", config)),
            "resources": session.get("resources", session.get("resource_limits", resources)),
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
                "session_config": {"type": ["object", "null"]},
                "resource_allocation": {"type": ["object", "null"]},
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
