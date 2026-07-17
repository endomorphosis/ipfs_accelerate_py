"""Native cli-endpoint-tools category implementations for unified mcp_server.

Exposes CLI endpoint adapter operations (Claude Code, OpenAI Codex CLI,
Google Gemini CLI, VSCode Copilot) from the legacy
``ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters`` module through
the unified MCP++ tool dispatch surface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_cli_endpoint_tools_api() -> Dict[str, Any]:
    """Resolve source cli-endpoint-tools APIs with compatibility fallback."""
    try:
        from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore
            register_cli_endpoint as _register_cli_endpoint,
            get_cli_endpoint as _get_cli_endpoint,
            list_cli_endpoints as _list_cli_endpoints,
            execute_cli_inference as _execute_cli_inference,
        )

        return {
            "register_cli_endpoint": _register_cli_endpoint,
            "get_cli_endpoint": _get_cli_endpoint,
            "list_cli_endpoints": _list_cli_endpoints,
            "execute_cli_inference": _execute_cli_inference,
        }
    except Exception:
        logger.warning(
            "Source cli_endpoint_adapters import unavailable, using fallback stubs"
        )
        _registry: Dict[str, Any] = {}

        def _register_fallback(adapter: Any) -> Dict[str, Any]:
            if hasattr(adapter, "endpoint_id"):
                _registry[adapter.endpoint_id] = adapter
                return {"status": "success", "endpoint_id": adapter.endpoint_id}
            return {"status": "error", "error": "Invalid adapter: missing endpoint_id"}

        def _get_fallback(endpoint_id: str) -> Optional[Any]:
            return _registry.get(endpoint_id)

        def _list_fallback() -> List[Dict[str, Any]]:
            return [{"endpoint_id": eid} for eid in _registry]

        def _execute_fallback(
            endpoint_id: str,
            prompt: str,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "endpoint_id": endpoint_id,
                "response": None,
                "backend_available": False,
            }

        return {
            "register_cli_endpoint": _register_fallback,
            "get_cli_endpoint": _get_fallback,
            "list_cli_endpoints": _list_fallback,
            "execute_cli_inference": _execute_fallback,
        }


_API = _load_cli_endpoint_tools_api()


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


async def cli_endpoint_list() -> Dict[str, Any]:
    """List all registered CLI endpoint adapters."""
    try:
        endpoints = _API["list_cli_endpoints"]()
        return _normalize_payload(
            {
                "endpoints": endpoints if isinstance(endpoints, list) else [],
                "count": len(endpoints) if isinstance(endpoints, list) else 0,
            }
        )
    except Exception as exc:
        return _error_result(str(exc))


async def cli_endpoint_get(endpoint_id: str) -> Dict[str, Any]:
    """Get details for a specific CLI endpoint adapter."""
    try:
        endpoint = _API["get_cli_endpoint"](endpoint_id)
        if endpoint is None:
            return _error_result(f"CLI endpoint {endpoint_id!r} not found", endpoint_id=endpoint_id)
        info = endpoint if isinstance(endpoint, dict) else getattr(endpoint, "__dict__", {"endpoint_id": endpoint_id})
        return _normalize_payload({"endpoint": info})
    except Exception as exc:
        return _error_result(str(exc), endpoint_id=endpoint_id)


async def cli_endpoint_execute(
    endpoint_id: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute an inference request through a CLI endpoint adapter."""
    try:
        kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        result = _API["execute_cli_inference"](
            endpoint_id=endpoint_id, prompt=prompt, **kwargs
        )
        return _normalize_payload(result if isinstance(result, dict) else {"response": result, "endpoint_id": endpoint_id})
    except Exception as exc:
        return _error_result(str(exc), endpoint_id=endpoint_id, prompt=prompt)


async def cli_endpoint_register(
    tool: str,
    endpoint_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a new CLI endpoint adapter for a supported CLI tool."""
    try:
        try:
            from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore
                CLIEndpointAdapter,
            )
            adapter_config = config or {}
            adapter_config["tool"] = tool
            if endpoint_id:
                adapter_config["endpoint_id"] = endpoint_id
            adapter = CLIEndpointAdapter(**adapter_config)
            result = _API["register_cli_endpoint"](adapter)
            return _normalize_payload(result if isinstance(result, dict) else {"registered": True})
        except Exception:
            pass
        return _normalize_payload({"tool": tool, "registered": False, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), tool=tool)


def register_native_cli_endpoint_tools(manager: Any) -> None:
    """Register native cli-endpoint-tools category tools in unified manager."""
    manager.register_tool(
        category="cli_endpoint_tools",
        name="cli_endpoint_list",
        func=cli_endpoint_list,
        description="List all registered CLI endpoint adapters.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "cli-endpoint-tools"],
    )
    manager.register_tool(
        category="cli_endpoint_tools",
        name="cli_endpoint_get",
        func=cli_endpoint_get,
        description="Get details for a specific CLI endpoint adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string", "description": "CLI endpoint identifier."}
            },
            "required": ["endpoint_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli-endpoint-tools"],
    )
    manager.register_tool(
        category="cli_endpoint_tools",
        name="cli_endpoint_execute",
        func=cli_endpoint_execute,
        description="Execute an inference request through a CLI endpoint adapter.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string", "description": "CLI endpoint identifier."},
                "prompt": {"type": "string", "description": "Input prompt text."},
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional maximum token count for the response.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Optional sampling temperature.",
                },
            },
            "required": ["endpoint_id", "prompt"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli-endpoint-tools"],
    )
    manager.register_tool(
        category="cli_endpoint_tools",
        name="cli_endpoint_register",
        func=cli_endpoint_register,
        description="Register a new CLI endpoint adapter for a supported CLI AI tool.",
        input_schema={
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "CLI tool name (claude, codex, gemini, copilot).",
                },
                "endpoint_id": {
                    "type": "string",
                    "description": "Optional custom endpoint identifier.",
                },
                "config": {
                    "type": "object",
                    "description": "Optional adapter configuration.",
                },
            },
            "required": ["tool"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli-endpoint-tools"],
    )
