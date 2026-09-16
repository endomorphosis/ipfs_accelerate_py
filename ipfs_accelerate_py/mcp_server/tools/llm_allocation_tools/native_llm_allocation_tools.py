"""MCP tools for the llm_router allocation schema.

Never returns raw API keys, prompts, or credentials.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import anyio


def _ok(**payload: Any) -> Dict[str, Any]:
    result = {"status": "success", "success": True}
    result.update(payload)
    return result


def _err(message: str, **payload: Any) -> Dict[str, Any]:
    result = {"status": "error", "success": False, "error": str(message)[:512]}
    result.update(payload)
    return result


def _strip_secrets(payload: Any) -> Any:
    if isinstance(payload, dict):
        blocked = {
            "secret",
            "api_key",
            "password",
            "token",
            "authorization",
            "credential",
        }
        keep_ids = {"api_key_slot", "api_key_fingerprint"}
        cleaned: Dict[str, Any] = {}
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in keep_ids:
                cleaned[str(key)] = value
                continue
            if lowered in blocked or any(marker in lowered for marker in blocked):
                continue
            cleaned[str(key)] = _strip_secrets(value)
        return cleaned
    if isinstance(payload, list):
        return [_strip_secrets(item) for item in payload[:256]]
    return payload


async def llm_cli_tools_status() -> Dict[str, Any]:
    """Install, auth, and usage status for every CLI coding tool."""
    try:
        from ipfs_accelerate_py.llm_allocation import cli_tools_status

        report = await anyio.to_thread.run_sync(cli_tools_status)
        return _ok(**_strip_secrets(report))
    except Exception as exc:
        return _err(str(exc))


async def llm_choose_cli_route(session_id: str = "") -> Dict[str, Any]:
    """Pick a CLI provider for a new or existing allocation session."""
    try:
        from ipfs_accelerate_py.llm_allocation import choose_cli_route

        route = await anyio.to_thread.run_sync(
            lambda: choose_cli_route(session_id or None)
        )
        return _ok(**_strip_secrets(route))
    except Exception as exc:
        return _err(str(exc))


async def llm_migrate_cli_session(
    session_id: str,
    to_provider: str,
    history: str = "full",
    handoff: str = "",
) -> Dict[str, Any]:
    """Rebind an allocation session to another CLI and port history."""
    if not str(session_id or "").strip() or not str(to_provider or "").strip():
        return _err("session_id and to_provider are required")
    try:
        from ipfs_accelerate_py.llm_allocation import migrate_cli_session

        result = await anyio.to_thread.run_sync(
            lambda: migrate_cli_session(
                session_id,
                to_provider,
                history=history or "full",
                handoff=handoff or "",
            )
        )
        return _ok(**_strip_secrets(result))
    except Exception as exc:
        return _err(str(exc))


async def llm_resolve_session(session_id: str) -> Dict[str, Any]:
    """Resolve an allocation id or native CLI id to current state."""
    if not str(session_id or "").strip():
        return _err("session_id is required")
    try:
        from ipfs_accelerate_py.llm_allocation import resolve_session

        result = await anyio.to_thread.run_sync(lambda: resolve_session(session_id))
        if not result:
            return _err("unknown session", session_id=session_id[:128])
        return _ok(**_strip_secrets(result))
    except Exception as exc:
        return _err(str(exc))


async def llm_get_allocation_session(session_id: str) -> Dict[str, Any]:
    """Return persisted allocation session stats (no prompts or secrets)."""
    if not str(session_id or "").strip():
        return _err("session_id is required")
    try:
        from ipfs_accelerate_py.llm_router import get_allocation_session

        result = await anyio.to_thread.run_sync(
            lambda: get_allocation_session(session_id)
        )
        if not result:
            return _err("unknown session", session_id=session_id[:128])
        return _ok(session=_strip_secrets(result))
    except Exception as exc:
        return _err(str(exc))


async def llm_register_api_key(
    backend: str,
    api_key: str,
    slot_id: str = "",
    label: str = "",
) -> Dict[str, Any]:
    """Store an API key in the secrets manager and index a multiplex slot."""
    if not str(backend or "").strip() or not str(api_key or "").strip():
        return _err("backend and api_key are required")
    try:
        from ipfs_accelerate_py.llm_allocation import register_api_key

        result = await anyio.to_thread.run_sync(
            lambda: register_api_key(
                backend,
                api_key,
                slot_id=slot_id or "",
                label=label or "",
            )
        )
        return _ok(**_strip_secrets(result))
    except Exception as exc:
        return _err(str(exc))


async def llm_list_api_key_slots(backend: str = "") -> Dict[str, Any]:
    """List API key slots and per-key stats. Never includes secrets."""
    try:
        from ipfs_accelerate_py.llm_allocation import list_api_key_slots

        rows = await anyio.to_thread.run_sync(
            lambda: list_api_key_slots(backend or "")
        )
        return _ok(slots=_strip_secrets(rows))
    except Exception as exc:
        return _err(str(exc))


async def llm_bind_session_api_key(
    session_id: str,
    backend: str,
    slot_id: str,
) -> Dict[str, Any]:
    """Pin an allocation session to one API key slot."""
    if not session_id or not backend or not slot_id:
        return _err("session_id, backend, and slot_id are required")
    try:
        from ipfs_accelerate_py.llm_allocation import bind_session_api_key

        result = await anyio.to_thread.run_sync(
            lambda: bind_session_api_key(session_id, backend, slot_id)
        )
        return _ok(**_strip_secrets(result))
    except Exception as exc:
        return _err(str(exc))


async def llm_ensure_cli_tool(
    provider: str,
    auto_install: bool = False,
) -> Dict[str, Any]:
    """Discover or optionally install a CLI coding tool."""
    if not str(provider or "").strip():
        return _err("provider is required")
    try:
        from ipfs_accelerate_py.cli_runtime.installers.catalog import ensure_cli_tool

        result = await anyio.to_thread.run_sync(
            lambda: ensure_cli_tool(provider, auto_install=bool(auto_install))
        )
        return _ok(**_strip_secrets(result.to_dict()))
    except Exception as exc:
        return _err(str(exc))


async def llm_router_generate_text(
    prompt: str,
    provider: str = "",
    model: str = "",
    allocation_session_id: str = "",
    allocation_path: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Generate text through llm_router with allocation session routing."""
    if not isinstance(prompt, str) or not prompt.strip():
        return _err("prompt must be a non-empty string")
    try:
        from ipfs_accelerate_py.llm_router import generate_text, get_allocation_session

        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if provider:
            kwargs["provider"] = provider
        if model:
            kwargs["model_name"] = model
        if allocation_session_id:
            kwargs["allocation_session_id"] = allocation_session_id
        if allocation_path:
            kwargs["allocation_path"] = allocation_path
        text = await anyio.to_thread.run_sync(
            lambda: generate_text(prompt.strip(), **kwargs)
        )
        session = {}
        if allocation_session_id:
            session = await anyio.to_thread.run_sync(
                lambda: get_allocation_session(allocation_session_id)
            )
        return _ok(
            generated_text=str(text),
            provider=provider or "auto",
            session=_strip_secrets(session),
        )
    except Exception as exc:
        return _err(str(exc))


def _uses_hierarchical_register(register: Any) -> bool:
    """True for HierarchicalToolManager; false for StandaloneMCP / FastMCP."""
    try:
        params = inspect.signature(register).parameters
    except (TypeError, ValueError):
        return False
    category = params.get("category")
    func = params.get("func")
    if category is None or func is None:
        return False
    return category.default is inspect.Parameter.empty


def _register_allocation_tool(
    mcp: Any,
    name: str,
    function: Any,
    description: str,
    schema: Dict[str, Any],
) -> None:
    register = getattr(mcp, "register_tool", None)
    if not callable(register):
        raise TypeError("mcp must provide register_tool")
    if _uses_hierarchical_register(register):
        register(
            category="llm_allocation_tools",
            name=name,
            func=function,
            description=description,
            input_schema=schema,
            runtime="fastapi",
            tags=["native", "mcpp", "llm-allocation", "llm-router"],
        )
        return
    register(
        name=name,
        function=function,
        description=description,
        input_schema=schema,
        execution_context="server",
    )


def register_native_llm_allocation_tools(mcp: Any) -> None:
    """Register allocation schema tools on MCP, MCP++, or the hierarchical manager."""
    registrations = (
        (
            "llm_cli_tools_status",
            llm_cli_tools_status,
            "Health, install, auth, and usage stats for every CLI coding tool.",
            {"type": "object", "properties": {}, "required": []},
        ),
        (
            "llm_choose_cli_route",
            llm_choose_cli_route,
            "Choose a CLI provider for a new or existing allocation session.",
            {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": [],
            },
        ),
        (
            "llm_migrate_cli_session",
            llm_migrate_cli_session,
            "Migrate an allocation session to another CLI (full/compact/none history).",
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "to_provider": {"type": "string"},
                    "history": {
                        "type": "string",
                        "enum": ["full", "compact", "none"],
                        "default": "full",
                    },
                    "handoff": {"type": "string"},
                },
                "required": ["session_id", "to_provider"],
            },
        ),
        (
            "llm_resolve_session",
            llm_resolve_session,
            "Resolve an allocation id or native CLI session id to current state.",
            {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        ),
        (
            "llm_get_allocation_session",
            llm_get_allocation_session,
            "Get persisted allocation session cost/token/latency stats.",
            {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        ),
        (
            "llm_register_api_key",
            llm_register_api_key,
            "Store an API key in the secrets manager and index a multiplex slot.",
            {
                "type": "object",
                "properties": {
                    "backend": {"type": "string"},
                    "api_key": {"type": "string"},
                    "slot_id": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["backend", "api_key"],
            },
        ),
        (
            "llm_list_api_key_slots",
            llm_list_api_key_slots,
            "List API key slots and per-key stats (no secrets).",
            {
                "type": "object",
                "properties": {"backend": {"type": "string"}},
                "required": [],
            },
        ),
        (
            "llm_bind_session_api_key",
            llm_bind_session_api_key,
            "Pin an allocation session to one API key slot.",
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "backend": {"type": "string"},
                    "slot_id": {"type": "string"},
                },
                "required": ["session_id", "backend", "slot_id"],
            },
        ),
        (
            "llm_ensure_cli_tool",
            llm_ensure_cli_tool,
            "Discover or optionally install a CLI coding tool.",
            {
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "auto_install": {"type": "boolean", "default": False},
                },
                "required": ["provider"],
            },
        ),
        (
            "llm_router_generate_text",
            llm_router_generate_text,
            "Generate text through llm_router with allocation session routing.",
            {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "minLength": 1},
                    "provider": {"type": "string"},
                    "model": {"type": "string"},
                    "allocation_session_id": {"type": "string"},
                    "allocation_path": {
                        "type": "string",
                        "enum": ["cli", "api", ""],
                    },
                    "max_tokens": {"type": "integer", "default": 512, "minimum": 1},
                    "temperature": {"type": "number", "default": 0.7},
                },
                "required": ["prompt"],
            },
        ),
    )
    for name, function, description, schema in registrations:
        _register_allocation_tool(mcp, name, function, description, schema)


register_llm_allocation_tools = register_native_llm_allocation_tools

__all__ = [
    "llm_bind_session_api_key",
    "llm_choose_cli_route",
    "llm_cli_tools_status",
    "llm_ensure_cli_tool",
    "llm_get_allocation_session",
    "llm_list_api_key_slots",
    "llm_migrate_cli_session",
    "llm_register_api_key",
    "llm_resolve_session",
    "llm_router_generate_text",
    "register_llm_allocation_tools",
    "register_native_llm_allocation_tools",
]
