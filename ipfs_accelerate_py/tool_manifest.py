"""Core MCP tool/resource/prompt manifest extraction.

This module intentionally contains no server startup logic. It provides pure
introspection helpers that can be used by:
- the `ipfs-accelerate` CLI,
- the MCP server wrapper layer,
- the libp2p discovery interface.

The returned structures are JSON-friendly (no callables).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str:
    try:
        return "" if value is None else str(value)
    except Exception:
        return ""


def _iter_items(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        # StandaloneMCP uses dict name -> {function, description, input_schema}
        return [dict({"name": k}, **_as_dict(v)) for k, v in value.items()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _extract_schema(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return _as_dict(obj.get("input_schema") or obj.get("schema") or obj.get("inputSchema"))
    for key in ("input_schema", "schema", "inputSchema"):
        try:
            schema = getattr(obj, key)
            if isinstance(schema, dict):
                return schema
        except Exception:
            pass
    return {}


def extract_mcp_manifest(mcp_like: Any, *, include_schemas: bool = True) -> Dict[str, Any]:
    """Extract a JSON-friendly manifest from a FastMCP/StandaloneMCP-like object.

    Accepts either the underlying MCP server object OR a wrapper that has a `.mcp`
    attribute (like `MCPServerWrapper`).
    """

    if mcp_like is None:
        return {"tools": [], "resources": [], "prompts": [], "counts": {"tools": 0, "resources": 0, "prompts": 0}}

    server = getattr(mcp_like, "mcp", None) or mcp_like

    name = _string(getattr(server, "name", ""))
    description = _string(getattr(server, "description", ""))

    tools_raw = getattr(server, "tools", None)
    resources_raw = getattr(server, "resources", None)
    prompts_raw = getattr(server, "prompts", None)

    tools: List[Dict[str, Any]] = []
    for t in _iter_items(tools_raw):
        if isinstance(t, dict):
            tool_name = _string(t.get("name"))
            tool_desc = _string(t.get("description"))
            schema = _extract_schema(t) if include_schemas else {}
        else:
            tool_name = _string(getattr(t, "name", ""))
            tool_desc = _string(getattr(t, "description", ""))
            schema = _extract_schema(t) if include_schemas else {}

        if not tool_name:
            continue

        entry: Dict[str, Any] = {"name": tool_name, "description": tool_desc}
        if include_schemas:
            entry["input_schema"] = schema
        tools.append(entry)

    resources: List[Dict[str, Any]] = []
    for r in _iter_items(resources_raw):
        if isinstance(r, dict):
            path = _string(r.get("path") or r.get("uri") or r.get("name"))
            desc = _string(r.get("description"))
        else:
            path = _string(getattr(r, "path", "") or getattr(r, "uri", "") or getattr(r, "name", ""))
            desc = _string(getattr(r, "description", ""))
        if not path:
            continue
        resources.append({"path": path, "description": desc})

    prompts: List[Dict[str, Any]] = []
    for p in _iter_items(prompts_raw):
        if isinstance(p, dict):
            pname = _string(p.get("name"))
            pdesc = _string(p.get("description"))
            schema = _as_dict(p.get("input_schema") or p.get("schema")) if include_schemas else {}
        else:
            pname = _string(getattr(p, "name", ""))
            pdesc = _string(getattr(p, "description", ""))
            schema = _as_dict(getattr(p, "input_schema", None) or getattr(p, "schema", None)) if include_schemas else {}
        if not pname:
            continue
        entry = {"name": pname, "description": pdesc}
        if include_schemas:
            entry["input_schema"] = schema
        prompts.append(entry)

    tools.sort(key=lambda x: x.get("name", ""))
    resources.sort(key=lambda x: x.get("path", ""))
    prompts.sort(key=lambda x: x.get("name", ""))

    return {
        "server": {"name": name, "description": description},
        "tools": tools,
        "resources": resources,
        "prompts": prompts,
        "counts": {"tools": len(tools), "resources": len(resources), "prompts": len(prompts)},
    }


def extract_tool_names(mcp_like: Any) -> List[str]:
    manifest = extract_mcp_manifest(mcp_like, include_schemas=False)
    return [t["name"] for t in manifest.get("tools", []) if isinstance(t, dict) and t.get("name")]


def jsonable(value: Any) -> Any:
    """Convert common Python objects into JSON-serializable equivalents."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            try:
                key = str(k)
            except Exception:
                continue
            out[key] = jsonable(v)
        return out
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in list(value)]
    try:
        return repr(value)
    except Exception:
        return str(type(value))


def _resolve_tool(server: Any, tool_name: str) -> Tuple[Optional[Any], Optional[Any], str]:
    """Return (tool_obj, function, description)."""

    # StandaloneMCP uses a dict mapping name -> {function, description, input_schema}
    tools_raw = getattr(server, "tools", None)
    if isinstance(tools_raw, dict) and tool_name in tools_raw:
        entry = tools_raw.get(tool_name) or {}
        fn = entry.get("function") if isinstance(entry, dict) else None
        desc = entry.get("description") if isinstance(entry, dict) else ""
        return entry, fn, _string(desc)

    # Mock FastMCP uses a list of ToolDefinition objects with .name/.function
    try:
        for t in _iter_items(tools_raw):
            if isinstance(t, dict):
                if _string(t.get("name")) == tool_name:
                    return t, t.get("function"), _string(t.get("description"))
            else:
                if _string(getattr(t, "name", "")) == tool_name:
                    fn = getattr(t, "function", None)
                    desc = _string(getattr(t, "description", ""))
                    return t, fn, desc
    except Exception:
        pass

    return None, None, ""


def _build_ctx(accelerate_instance: Any = None) -> Any:
    # Mimic the shape used throughout this repo: ctx.state.accelerate
    return SimpleNamespace(state=SimpleNamespace(accelerate=accelerate_instance))


async def invoke_mcp_tool(
    mcp_like: Any,
    *,
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    accelerate_instance: Any = None,
) -> Dict[str, Any]:
    """Invoke a registered MCP tool by name.

    This is core logic: it operates on an MCP-like registry object, but does not
    start servers or depend on transport.

    Returns a JSON-friendly dict: {ok, result?, error?, tool?}.
    """

    import inspect

    server = getattr(mcp_like, "mcp", None) or mcp_like
    args = args if isinstance(args, dict) else {}

    tool_obj, fn, desc = _resolve_tool(server, str(tool_name))
    if fn is None or not callable(fn):
        return {"ok": False, "error": "tool_not_found", "tool": str(tool_name)}

    # Provide ctx/context only if the function asks for it.
    kwargs = dict(args)
    try:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        if "ctx" in sig.parameters and "ctx" not in kwargs:
            kwargs["ctx"] = _build_ctx(accelerate_instance)
        elif "context" in sig.parameters and "context" not in kwargs:
            kwargs["context"] = _build_ctx(accelerate_instance)

        # If tool is strict about kwargs, filter to accepted names.
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        pass

    try:
        if inspect.iscoroutinefunction(fn):
            res = await fn(**kwargs)
        else:
            res = fn(**kwargs)
            if inspect.isawaitable(res):
                res = await res
        return {"ok": True, "tool": str(tool_name), "description": desc, "result": jsonable(res)}
    except TypeError as exc:
        # One more try: call without kwargs filtering.
        try:
            if inspect.iscoroutinefunction(fn):
                res = await fn(**dict(args))
            else:
                res = fn(**dict(args))
                if inspect.isawaitable(res):
                    res = await res
            return {"ok": True, "tool": str(tool_name), "description": desc, "result": jsonable(res)}
        except Exception:
            return {"ok": False, "tool": str(tool_name), "error": f"type_error: {exc}"}
    except Exception as exc:
        return {"ok": False, "tool": str(tool_name), "error": str(exc)}
