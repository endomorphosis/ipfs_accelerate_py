"""Native cli-endpoint-tools category implementations for unified mcp_server.

Exposes CLI endpoint adapter operations (Claude Code, OpenAI Codex CLI,
Google Gemini CLI, VSCode Copilot, Goose CLI) through the canonical factory at
``ipfs_accelerate_py.cli_runtime.endpoints``. Never instantiates the abstract
``CLIEndpointAdapter`` base class.

Goose agent authority fields are explicitly bounded; unknown authority-bearing
options are rejected fail-closed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bounded authority surface for Goose (and general CLI) execute via MCP.
_MCP_EXECUTE_SAFE_KEYS = frozenset(
    {
        "max_tokens",
        "temperature",
        "model",
        "model_name",
        "goose_provider",
        "provider",
        "output_format",
        "stream",
        "streaming",
        "timeout",
        "task_type",
    }
)
_MCP_EXECUTE_AUTHORITY_KEYS = frozenset(
    {
        "execution_mode",
        "mode",
        "agent",
        "allow_side_effects",
        "enable_agent",
        "package_enable_agent",
        "package_policy",
        "cwd",
        "workspace",
        "path_root",
        "GOOSE_PATH_ROOT",
        "goose_path_root",
        "approval_mode",
        "builtins",
        "extensions",
        "with_builtin",
        "with_extension",
        "allowed_cwd_roots",
        "max_turns",
        "max_tool_repetitions",
        "timeout_seconds",
        "max_output_bytes",
        "session_id",
        "resume_session",
        "agent_policy",
        "side_effecting",
        "with_tools",
    }
)
_MCP_EXECUTE_KNOWN_KEYS = _MCP_EXECUTE_SAFE_KEYS | _MCP_EXECUTE_AUTHORITY_KEYS
_MCP_AUTHORITY_MARKERS = (
    "allow_side",
    "side_effect",
    "path_root",
    "cwd",
    "workspace",
    "approval",
    "builtin",
    "extension",
    "enable_agent",
    "package_policy",
    "agent_policy",
    "execution_mode",
    "max_turn",
    "max_tool",
    "max_output",
    "allowed_cwd",
    "session",
    "resume",
    "GOOSE_PATH",
    "goose_path",
)


def _reject_unknown_authority_options(options: Dict[str, Any]) -> Optional[str]:
    """Return error message if *options* contains unknown authority keys."""
    for key in options:
        k = str(key)
        if k in _MCP_EXECUTE_KNOWN_KEYS:
            continue
        lowered = k.lower()
        if any(marker.lower() in lowered for marker in _MCP_AUTHORITY_MARKERS):
            return f"unknown authority-bearing option rejected: {k}"
    return None


def _load_cli_endpoint_tools_api() -> Dict[str, Any]:
    """Resolve canonical cli-endpoint APIs with a safe fallback."""
    try:
        from ipfs_accelerate_py.cli_runtime.endpoints import (
            execute_cli_inference as _execute_cli_inference,
            get_cli_endpoint as _get_cli_endpoint,
            list_cli_endpoints as _list_cli_endpoints,
            list_cli_endpoint_tools as _list_cli_endpoint_tools,
            register_cli_endpoint as _register_cli_endpoint,
            create_cli_endpoint as _create_cli_endpoint,
            get_default_endpoint_factory as _get_factory,
            error_envelope as _error_envelope,
        )

        return {
            "register_cli_endpoint": _register_cli_endpoint,
            "get_cli_endpoint": _get_cli_endpoint,
            "list_cli_endpoints": _list_cli_endpoints,
            "list_cli_endpoint_tools": _list_cli_endpoint_tools,
            "execute_cli_inference": _execute_cli_inference,
            "create_cli_endpoint": _create_cli_endpoint,
            "get_default_endpoint_factory": _get_factory,
            "error_envelope": _error_envelope,
            "canonical": True,
        }
    except Exception:
        logger.warning(
            "Canonical cli_runtime.endpoints import unavailable; "
            "trying compatibility adapters module"
        )

    try:
        from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (  # type: ignore
            register_cli_endpoint as _register_cli_endpoint,
            get_cli_endpoint as _get_cli_endpoint,
            list_cli_endpoints as _list_cli_endpoints,
            execute_cli_inference as _execute_cli_inference,
            create_cli_endpoint as _create_cli_endpoint,
        )

        return {
            "register_cli_endpoint": _register_cli_endpoint,
            "get_cli_endpoint": _get_cli_endpoint,
            "list_cli_endpoints": _list_cli_endpoints,
            "list_cli_endpoint_tools": lambda: [],
            "execute_cli_inference": _execute_cli_inference,
            "create_cli_endpoint": _create_cli_endpoint,
            "get_default_endpoint_factory": None,
            "error_envelope": None,
            "canonical": False,
        }
    except Exception:
        logger.warning(
            "Source cli_endpoint_adapters import unavailable, using fallback stubs"
        )
        _registry: Dict[str, Any] = {}

        def _register_fallback(adapter: Any = None, **kwargs: Any) -> Dict[str, Any]:
            if adapter is not None and hasattr(adapter, "endpoint_id"):
                _registry[adapter.endpoint_id] = adapter
                return {
                    "status": "success",
                    "endpoint_id": adapter.endpoint_id,
                    "registered": True,
                }
            tool = kwargs.get("tool")
            if tool:
                return {
                    "status": "error",
                    "success": False,
                    "error": f"Unsupported CLI endpoint tool: {tool}",
                    "error_code": "provider_not_found",
                    "registered": False,
                    "tool": tool,
                }
            return {
                "status": "error",
                "success": False,
                "error": "Invalid adapter: missing endpoint_id",
                "registered": False,
            }

        def _get_fallback(endpoint_id: str) -> Optional[Any]:
            return _registry.get(endpoint_id)

        def _list_fallback(**_kwargs: Any) -> List[Dict[str, Any]]:
            return [{"endpoint_id": eid} for eid in _registry]

        def _execute_fallback(
            endpoint_id: str,
            prompt: str,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            return {
                "status": "error",
                "success": False,
                "endpoint_id": endpoint_id,
                "error": "CLI endpoint backend unavailable",
                "error_code": "provider_load_failed",
            }

        def _create_fallback(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("CLI endpoint factory unavailable")

        return {
            "register_cli_endpoint": _register_fallback,
            "get_cli_endpoint": _get_fallback,
            "list_cli_endpoints": _list_fallback,
            "list_cli_endpoint_tools": lambda: [],
            "execute_cli_inference": _execute_fallback,
            "create_cli_endpoint": _create_fallback,
            "get_default_endpoint_factory": None,
            "error_envelope": None,
            "canonical": False,
        }


_API = _load_cli_endpoint_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
            envelope.setdefault("success", False)
        elif "status" not in envelope:
            envelope["status"] = "success"
            envelope.setdefault("success", True)
        # Hard guarantee: never echo prompts through MCP.
        envelope.pop("prompt", None)
        return envelope
    if payload is None:
        return {"status": "success", "success": True}
    return {"status": "success", "success": True, "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent error envelope for wrapper edge failures."""
    # Never include prompt keys.
    context.pop("prompt", None)
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    for key, value in context.items():
        if key == "prompt":
            continue
        envelope[key] = value
    return envelope


async def cli_endpoint_list() -> Dict[str, Any]:
    """List all registered CLI endpoint adapters (no provider probing)."""
    try:
        endpoints = _API["list_cli_endpoints"]()
        return _normalize_payload(
            {
                "endpoints": endpoints if isinstance(endpoints, list) else [],
                "count": len(endpoints) if isinstance(endpoints, list) else 0,
            }
        )
    except Exception as exc:
        return _error_result(f"list failed: {type(exc).__name__}")


async def cli_endpoint_get(endpoint_id: str) -> Dict[str, Any]:
    """Get details for a specific CLI endpoint adapter."""
    try:
        endpoint = _API["get_cli_endpoint"](endpoint_id)
        if endpoint is None:
            return _error_result(
                f"CLI endpoint {endpoint_id!r} not found",
                endpoint_id=endpoint_id,
                error_code="provider_not_found",
            )
        if isinstance(endpoint, dict):
            info = endpoint
        elif hasattr(endpoint, "get_stats"):
            info = dict(endpoint.get_stats())
        else:
            info = {
                "endpoint_id": getattr(endpoint, "endpoint_id", endpoint_id),
                "cli_path": getattr(endpoint, "cli_path", None),
            }
        info.pop("prompt", None)
        return _normalize_payload({"endpoint": info})
    except Exception as exc:
        return _error_result(
            f"get failed: {type(exc).__name__}",
            endpoint_id=endpoint_id,
        )


async def cli_endpoint_execute(
    endpoint_id: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    execution_mode: Optional[str] = None,
    allow_side_effects: Optional[bool] = None,
    enable_agent: Optional[bool] = None,
    package_enable_agent: Optional[bool] = None,
    package_policy: Optional[Dict[str, Any]] = None,
    cwd: Optional[str] = None,
    path_root: Optional[str] = None,
    approval_mode: Optional[str] = None,
    builtins: Optional[List[str]] = None,
    extensions: Optional[List[str]] = None,
    allowed_cwd_roots: Optional[List[str]] = None,
    max_turns: Optional[int] = None,
    max_tool_repetitions: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    max_output_bytes: Optional[int] = None,
    model: Optional[str] = None,
    goose_provider: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_policy: Optional[Dict[str, Any]] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an inference request through a CLI endpoint adapter.

    Goose agent authority fields are bounded. Pass unknown authority-bearing
    keys via *extra_options* and they are rejected fail-closed (never silently
    applied). Default mode is safe chat (same profile as llm_router goose_cli).
    """
    try:
        kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if execution_mode is not None:
            kwargs["execution_mode"] = execution_mode
        if allow_side_effects is not None:
            kwargs["allow_side_effects"] = allow_side_effects
        if enable_agent is not None:
            kwargs["enable_agent"] = enable_agent
        if package_enable_agent is not None:
            kwargs["package_enable_agent"] = package_enable_agent
        if package_policy is not None:
            kwargs["package_policy"] = package_policy
        if cwd is not None:
            kwargs["cwd"] = cwd
        if path_root is not None:
            kwargs["path_root"] = path_root
        if approval_mode is not None:
            kwargs["approval_mode"] = approval_mode
        if builtins is not None:
            kwargs["builtins"] = builtins
        if extensions is not None:
            kwargs["extensions"] = extensions
        if allowed_cwd_roots is not None:
            kwargs["allowed_cwd_roots"] = allowed_cwd_roots
        if max_turns is not None:
            kwargs["max_turns"] = max_turns
        if max_tool_repetitions is not None:
            kwargs["max_tool_repetitions"] = max_tool_repetitions
        if timeout_seconds is not None:
            kwargs["timeout_seconds"] = timeout_seconds
        if max_output_bytes is not None:
            kwargs["max_output_bytes"] = max_output_bytes
        if model is not None:
            kwargs["model"] = model
        if goose_provider is not None:
            kwargs["goose_provider"] = goose_provider
        if session_id is not None:
            kwargs["session_id"] = session_id
        if agent_policy is not None:
            kwargs["agent_policy"] = agent_policy

        if extra_options:
            if not isinstance(extra_options, dict):
                return _error_result(
                    "extra_options must be an object",
                    endpoint_id=endpoint_id,
                    error_code="invalid_contract",
                )
            denied = _reject_unknown_authority_options(extra_options)
            if denied:
                return _error_result(
                    denied,
                    endpoint_id=endpoint_id,
                    error_code="policy_denied",
                    provider="goose_cli",
                    execution_mode=str(execution_mode or "chat"),
                    side_effects_started=False,
                    tool_call_count=0,
                )
            # Only merge known non-authority or known authority keys.
            for key, value in extra_options.items():
                if str(key) in _MCP_EXECUTE_KNOWN_KEYS:
                    kwargs[str(key)] = value

        # Also reject if caller somehow smuggled unknown authority keys into
        # package_policy without enable_agent structure (leave structure free).
        denied_kwargs = _reject_unknown_authority_options(
            {k: v for k, v in kwargs.items() if k not in _MCP_EXECUTE_KNOWN_KEYS}
        )
        if denied_kwargs:
            return _error_result(
                denied_kwargs,
                endpoint_id=endpoint_id,
                error_code="policy_denied",
                side_effects_started=False,
                tool_call_count=0,
            )

        result = _API["execute_cli_inference"](
            endpoint_id=endpoint_id, prompt=prompt, **kwargs
        )
        if isinstance(result, dict):
            return _normalize_payload(result)
        return _normalize_payload(
            {"response": result, "endpoint_id": endpoint_id}
        )
    except Exception as exc:
        # Do not echo the prompt in the error envelope.
        return _error_result(
            f"execute failed: {type(exc).__name__}",
            endpoint_id=endpoint_id,
            error_code="internal",
        )


async def cli_endpoint_register(
    tool: str,
    endpoint_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a new CLI endpoint via the concrete factory.

    Never instantiates the abstract ``CLIEndpointAdapter``. Unsupported tools
    return a typed error envelope (``registered: false`` with ``error_code``)
    rather than silently reporting failure after swallowing an exception.
    """
    try:
        # Prefer the canonical tool-name registration path (creates a concrete
        # adapter via the factory and registers it).
        result = _API["register_cli_endpoint"](
            tool=tool,
            endpoint_id=endpoint_id,
            config=config or {},
        )
        if not isinstance(result, dict):
            return _normalize_payload(
                {
                    "tool": tool,
                    "endpoint_id": endpoint_id,
                    "registered": True,
                }
            )
        envelope = _normalize_payload(result)
        # Ensure unsupported-tool failures are typed, not bare registered:false.
        if envelope.get("status") == "error" or envelope.get("registered") is False:
            envelope.setdefault("registered", False)
            envelope.setdefault("success", False)
            envelope["status"] = "error"
            if not envelope.get("error"):
                envelope["error"] = f"Unsupported CLI endpoint tool: {tool}"
            envelope.setdefault("error_code", "provider_not_found")
            envelope.setdefault("tool", tool)
        else:
            envelope.setdefault("registered", True)
            envelope.setdefault("tool", tool)
        return envelope
    except Exception as exc:
        # Surface a typed error; do not swallow into registered:false alone.
        name = type(exc).__name__
        message = str(exc) if str(exc) else name
        # Strip any accidental prompt content from exception text.
        if "prompt" in message.lower() and len(message) > 200:
            message = name
        return _error_result(
            message,
            tool=tool,
            endpoint_id=endpoint_id,
            registered=False,
            error_code="provider_not_found"
            if "unsupported" in message.lower() or "not found" in message.lower()
            else "internal",
        )


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
                "endpoint_id": {
                    "type": "string",
                    "description": "CLI endpoint identifier.",
                }
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
        description=(
            "Execute an inference request through a CLI endpoint adapter. "
            "Goose defaults to safe chat; agent mode requires explicit "
            "execution_mode, allow_side_effects, package enable policy, "
            "absolute cwd/path_root, approval_mode, allowlists, and finite bounds."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_id": {
                    "type": "string",
                    "description": "CLI endpoint identifier.",
                },
                "prompt": {"type": "string", "description": "Input prompt text."},
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional maximum token count for the response.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Optional sampling temperature.",
                },
                "execution_mode": {
                    "type": "string",
                    "enum": ["chat", "agent"],
                    "description": "Goose execution mode (default chat).",
                },
                "allow_side_effects": {
                    "type": "boolean",
                    "description": "Required true for Goose agent mode.",
                },
                "enable_agent": {
                    "type": "boolean",
                    "description": "Package enable policy gate for agent mode.",
                },
                "package_enable_agent": {
                    "type": "boolean",
                    "description": "Alternate package enable policy flag.",
                },
                "package_policy": {
                    "type": "object",
                    "description": "Package policy object (e.g. enable_agent).",
                },
                "cwd": {
                    "type": "string",
                    "description": "Absolute working directory for agent mode.",
                },
                "path_root": {
                    "type": "string",
                    "description": "Absolute GOOSE_PATH_ROOT for agent mode.",
                },
                "approval_mode": {
                    "type": "string",
                    "description": "Goose approval mode (approve/auto/smart_approve).",
                },
                "builtins": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Builtin allowlist for agent mode.",
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extension allowlist for agent mode.",
                },
                "allowed_cwd_roots": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Absolute roots that must contain cwd.",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Finite max turns for agent (or chat bound).",
                },
                "max_tool_repetitions": {
                    "type": "integer",
                    "description": "Finite max tool repetitions.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Finite wall-clock timeout in seconds.",
                },
                "max_output_bytes": {
                    "type": "integer",
                    "description": "Finite output byte bound for agent mode.",
                },
                "model": {
                    "type": "string",
                    "description": "Model name (Goose --model).",
                },
                "goose_provider": {
                    "type": "string",
                    "description": "Underlying Goose provider name.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session id (agent only).",
                },
                "agent_policy": {
                    "type": "object",
                    "description": "Full GooseAgentPolicy mapping for agent mode.",
                },
                "extra_options": {
                    "type": "object",
                    "description": (
                        "Optional known options only; unknown authority-bearing "
                        "keys are rejected."
                    ),
                },
            },
            "required": ["endpoint_id", "prompt"],
            "additionalProperties": False,
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli-endpoint-tools"],
    )
    manager.register_tool(
        category="cli_endpoint_tools",
        name="cli_endpoint_register",
        func=cli_endpoint_register,
        description=(
            "Register a new CLI endpoint adapter for a supported CLI AI tool "
            "via the concrete factory (never instantiates the abstract base)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": (
                        "CLI tool name (claude, openai/codex, gemini, "
                        "vscode/copilot, goose/goose_cli and aliases)."
                    ),
                },
                "endpoint_id": {
                    "type": "string",
                    "description": "Optional custom endpoint identifier.",
                },
                "config": {
                    "type": "object",
                    "description": (
                        "Optional adapter configuration. Goose agent package "
                        "enable policy may be set via enable_agent."
                    ),
                },
            },
            "required": ["tool"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "cli-endpoint-tools"],
    )
