"""MCP / MCP++ prompt-only entrypoints for agent supervisor.

Tool signatures mirror the Python facade:
  - run/preview: prompt required; advanced params optional overrides
  - steer: prompt required; run_id optional (infers sole run)
  - status/follow: run_id optional (infers sole compatible run)

All tools return the same canonical outcome envelope as the Python facade.
"""

from __future__ import annotations

from typing import Any, Optional

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
    AgentSupervisorFacade,
    get_facade,
)

# Tool metadata for MCP registration
TOOL_NAMES = ("run", "preview", "steer", "status", "follow")

TOOL_SCHEMAS: dict[str, dict] = {
    "run": {
        "name": "agent_supervisor_run",
        "description": "Run agent supervisor from a prompt only. Advanced flags are optional overrides.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Natural-language prompt"},
                "model": {"type": "string", "description": "Optional model override"},
                "timeout": {"type": "number", "description": "Optional timeout override"},
                "backend": {"type": "string", "description": "Optional backend override"},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
    "preview": {
        "name": "agent_supervisor_preview",
        "description": "Preview agent supervisor plan from a prompt only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Natural-language prompt"},
                "model": {"type": "string", "description": "Optional model override"},
                "timeout": {"type": "number", "description": "Optional timeout override"},
                "backend": {"type": "string", "description": "Optional backend override"},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
    "steer": {
        "name": "agent_supervisor_steer",
        "description": "Steer a run with a prompt; run_id optional when sole compatible run exists.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Steer prompt"},
                "run_id": {"type": "string", "description": "Optional run handle"},
                "model": {"type": "string", "description": "Optional model override"},
                "timeout": {"type": "number", "description": "Optional timeout override"},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
    "status": {
        "name": "agent_supervisor_status",
        "description": "Status of a run; infers the sole compatible run when run_id omitted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Optional run handle"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    "follow": {
        "name": "agent_supervisor_follow",
        "description": "Follow a run; infers the sole compatible run when run_id omitted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Optional run handle"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def _facade(backend: Any = None) -> AgentSupervisorFacade:
    return get_facade(backend=backend) if backend is not None else get_facade()


def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# MCP tool handlers (plain functions; MCP server wires these by name)
# ---------------------------------------------------------------------------

def agent_supervisor_run(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    backend: Optional[str] = None,
    **extra: Any,
) -> dict:
    overrides = _strip_none({"model": model, "timeout": timeout, "backend": backend, **extra})
    return _facade().run(prompt, **overrides)


def agent_supervisor_preview(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    backend: Optional[str] = None,
    **extra: Any,
) -> dict:
    overrides = _strip_none({"model": model, "timeout": timeout, "backend": backend, **extra})
    return _facade().preview(prompt, **overrides)


def agent_supervisor_steer(
    prompt: str,
    run_id: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    **extra: Any,
) -> dict:
    overrides = _strip_none({"model": model, "timeout": timeout, **extra})
    return _facade().steer(prompt, run_id=run_id, **overrides)


def agent_supervisor_status(run_id: Optional[str] = None, **extra: Any) -> dict:
    return _facade().status(run_id=run_id, **_strip_none(extra))


def agent_supervisor_follow(run_id: Optional[str] = None, **extra: Any) -> dict:
    return _facade().follow(run_id=run_id, **_strip_none(extra))


HANDLERS = {
    "run": agent_supervisor_run,
    "preview": agent_supervisor_preview,
    "steer": agent_supervisor_steer,
    "status": agent_supervisor_status,
    "follow": agent_supervisor_follow,
    "agent_supervisor_run": agent_supervisor_run,
    "agent_supervisor_preview": agent_supervisor_preview,
    "agent_supervisor_steer": agent_supervisor_steer,
    "agent_supervisor_status": agent_supervisor_status,
    "agent_supervisor_follow": agent_supervisor_follow,
}


def list_tools() -> list[dict]:
    """Return MCP tool descriptors for registration."""
    return [TOOL_SCHEMAS[name] for name in TOOL_NAMES]


def call_tool(name: str, arguments: Optional[dict] = None, **kwargs: Any) -> dict:
    """Dispatch an MCP tool call by name with a prompt-only argument dict."""
    arguments = dict(arguments or {})
    arguments.update(kwargs)
    handler = HANDLERS.get(name)
    if handler is None:
        # Allow short names and full names
        short = name.replace("agent_supervisor_", "")
        handler = HANDLERS.get(short)
    if handler is None:
        from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
            OUTCOME_INVALID,
            _result,
        )

        return _result(OUTCOME_INVALID, error=f"unknown tool: {name}", message=f"unknown tool: {name}")
    return handler(**arguments)


# ---------------------------------------------------------------------------
# MCP++ convenience: same surface, explicit transport tag in result metadata
# ---------------------------------------------------------------------------

def mcplusplus_call(name: str, arguments: Optional[dict] = None, **kwargs: Any) -> dict:
    """MCP++ entry: identical semantics to MCP call_tool with transport marker."""
    result = call_tool(name, arguments, **kwargs)
    # Non-breaking annotation for transport-aware callers
    if isinstance(result, dict):
        data = dict(result.get("data") or {})
        data.setdefault("transport", "mcp++")
        # Keep envelope stable; only add transport if data already present or always
        # Prefer top-level optional key that does not alter outcome/exit_code
        annotated = dict(result)
        annotated["transport"] = "mcp++"
        return annotated
    return result


__all__ = [
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "HANDLERS",
    "list_tools",
    "call_tool",
    "mcplusplus_call",
    "agent_supervisor_run",
    "agent_supervisor_preview",
    "agent_supervisor_steer",
    "agent_supervisor_status",
    "agent_supervisor_follow",
]
