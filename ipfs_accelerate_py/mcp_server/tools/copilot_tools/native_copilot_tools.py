"""Native copilot-tools category implementations for unified mcp_server.

Exposes GitHub Copilot CLI and SDK operations from the legacy
``ipfs_accelerate_py.mcp.tools.copilot_tools`` and
``ipfs_accelerate_py.mcp.tools.copilot_sdk_tools`` modules through the
unified MCP++ tool dispatch surface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_copilot_tools_api() -> Dict[str, Any]:
    """Resolve source copilot-tools APIs with compatibility fallback."""
    result: Dict[str, Any] = {}

    try:
        import ipfs_accelerate_py.mcp.tools.copilot_tools as _ct_mod  # type: ignore

        result["_copilot_mod"] = _ct_mod
    except Exception:
        logger.warning("Source copilot_tools import unavailable")

    try:
        import ipfs_accelerate_py.mcp.tools.copilot_sdk_tools as _sdk_mod  # type: ignore

        result["_sdk_mod"] = _sdk_mod
    except Exception:
        logger.warning("Source copilot_sdk_tools import unavailable")

    return result


_API = _load_copilot_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


def _call_mod(mod_key: str, func_name: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Delegate a call to a module function, returning None if unavailable."""
    mod = _API.get(mod_key)
    if mod is None:
        return None
    fn = getattr(mod, func_name, None)
    if fn is None:
        return None
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Copilot CLI tools
# ---------------------------------------------------------------------------


async def copilot_suggest_command(
    prompt: str,
    shell: str = "bash",
) -> Dict[str, Any]:
    """Get a shell command suggestion from GitHub Copilot CLI."""
    try:
        result = _call_mod("_copilot_mod", "copilot_suggest_command", prompt=prompt, shell=shell)
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"prompt": prompt, "shell": shell, "suggestion": None, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), prompt=prompt)


async def copilot_explain_command(command: str) -> Dict[str, Any]:
    """Get an explanation of a shell command from GitHub Copilot CLI."""
    try:
        result = _call_mod("_copilot_mod", "copilot_explain_command", command=command)
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"command": command, "explanation": None, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), command=command)


async def copilot_suggest_git_command(prompt: str) -> Dict[str, Any]:
    """Get a git command suggestion from GitHub Copilot CLI."""
    try:
        result = _call_mod("_copilot_mod", "copilot_suggest_git_command", prompt=prompt)
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"prompt": prompt, "suggestion": None, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), prompt=prompt)


# ---------------------------------------------------------------------------
# Copilot SDK tools
# ---------------------------------------------------------------------------


async def copilot_sdk_create_session(
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a GitHub Copilot SDK conversation session."""
    try:
        result = _call_mod("_sdk_mod", "copilot_sdk_create_session", model=model, system_prompt=system_prompt)
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"session_id": None, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc))


async def copilot_sdk_send_message(
    session_id: str,
    message: str,
) -> Dict[str, Any]:
    """Send a message to an existing Copilot SDK session."""
    try:
        result = _call_mod("_sdk_mod", "copilot_sdk_send_message", session_id=session_id, message=message)
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"session_id": session_id, "response": None, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), session_id=session_id)


async def copilot_sdk_stream_message(
    session_id: str,
    message: str,
) -> Dict[str, Any]:
    """Stream a message response from a Copilot SDK session."""
    try:
        result = _call_mod("_sdk_mod", "copilot_sdk_stream_message", session_id=session_id, message=message)
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"session_id": session_id, "stream": None, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), session_id=session_id)


async def copilot_sdk_destroy_session(session_id: str) -> Dict[str, Any]:
    """Destroy a Copilot SDK conversation session."""
    try:
        result = _call_mod("_sdk_mod", "copilot_sdk_destroy_session", session_id=session_id)
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"session_id": session_id, "destroyed": False, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), session_id=session_id)


async def copilot_sdk_list_sessions() -> Dict[str, Any]:
    """List active Copilot SDK conversation sessions."""
    try:
        result = _call_mod("_sdk_mod", "copilot_sdk_list_sessions")
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"sessions": [], "count": 0, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc))


async def copilot_sdk_get_tools() -> Dict[str, Any]:
    """Get available tools registered with the Copilot SDK."""
    try:
        result = _call_mod("_sdk_mod", "copilot_sdk_get_tools")
        if result is not None:
            return _normalize_payload(result)
        return _normalize_payload({"tools": [], "count": 0, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc))


def register_native_copilot_tools(manager: Any) -> None:
    """Register native copilot-tools category tools in unified manager."""
    manager.register_tool(
        category="copilot_tools",
        name="copilot_suggest_command",
        func=copilot_suggest_command,
        description="Get a shell command suggestion from GitHub Copilot CLI.",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Natural language description of the task."},
                "shell": {
                    "type": "string",
                    "description": "Target shell (bash, zsh, fish, powershell).",
                    "default": "bash",
                },
            },
            "required": ["prompt"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
    manager.register_tool(
        category="copilot_tools",
        name="copilot_explain_command",
        func=copilot_explain_command,
        description="Get an explanation of a shell command from GitHub Copilot CLI.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to explain."}
            },
            "required": ["command"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
    manager.register_tool(
        category="copilot_tools",
        name="copilot_suggest_git_command",
        func=copilot_suggest_git_command,
        description="Get a git command suggestion from GitHub Copilot CLI.",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Natural language description of the git task."}
            },
            "required": ["prompt"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
    manager.register_tool(
        category="copilot_tools",
        name="copilot_sdk_create_session",
        func=copilot_sdk_create_session,
        description="Create a GitHub Copilot SDK conversation session.",
        input_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Optional Copilot model identifier."},
                "system_prompt": {"type": "string", "description": "Optional system prompt."},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
    manager.register_tool(
        category="copilot_tools",
        name="copilot_sdk_send_message",
        func=copilot_sdk_send_message,
        description="Send a message to an existing Copilot SDK session.",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session identifier."},
                "message": {"type": "string", "description": "Message text to send."},
            },
            "required": ["session_id", "message"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
    manager.register_tool(
        category="copilot_tools",
        name="copilot_sdk_stream_message",
        func=copilot_sdk_stream_message,
        description="Stream a message response from a Copilot SDK session.",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session identifier."},
                "message": {"type": "string", "description": "Message text to stream."},
            },
            "required": ["session_id", "message"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
    manager.register_tool(
        category="copilot_tools",
        name="copilot_sdk_destroy_session",
        func=copilot_sdk_destroy_session,
        description="Destroy a Copilot SDK conversation session.",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session identifier to destroy."}
            },
            "required": ["session_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
    manager.register_tool(
        category="copilot_tools",
        name="copilot_sdk_list_sessions",
        func=copilot_sdk_list_sessions,
        description="List active Copilot SDK conversation sessions.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
    manager.register_tool(
        category="copilot_tools",
        name="copilot_sdk_get_tools",
        func=copilot_sdk_get_tools,
        description="Get the list of tools available in the Copilot SDK.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "copilot-tools"],
    )
