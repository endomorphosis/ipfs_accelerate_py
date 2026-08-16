"""Load deterministic surface identity bindings (autonomous repair materializer).

This module is intentionally tiny and import-safe. Bindings are written by
``agent_supervisor.autonomous_repair.materialize`` when
``write_package_bindings=true``. They map operation / IDL / ORB aliases to a
preferred MCP surface path + handler for mediation — not execution grants.

Used by:
* :mod:`package_mcp_interop` — resolve peer/local tool names before tools/call
* :func:`tools_dispatch` — prefer bound handler / category for local dispatch
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


_BINDINGS_PATH = Path(__file__).resolve().parent / "surface_identity_bindings.json"

# path fragment → hierarchical tool category used by tools_dispatch
_PATH_CATEGORY_HINTS: tuple[tuple[str, str], ...] = (
    ("/tools/ipfs/", "ipfs"),
    ("/tools/workflow/", "workflow"),
    ("/tools/provenance", "provenance"),
    ("/tools/search", "search"),
    ("/tools/index_management", "index"),
    ("/tools/embedding", "embedding"),
    ("/tools/backend_management", "backend"),
    ("/tools/ipfs_cluster", "ipfs_cluster"),
    ("/mcp_server/server.py", "meta"),
)


@lru_cache(maxsize=4)
def load_surface_identity_bindings(
    path: str | None = None,
) -> Mapping[str, Any]:
    """Return the package-local binding catalog, or empty bindings."""
    bindings_path = Path(path) if path else _BINDINGS_PATH
    if not bindings_path.is_file():
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/surface-identity-bindings@1",
            "binding_count": 0,
            "bindings": {},
        }
    try:
        doc = json.loads(bindings_path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return {"bindings": {}}
        return doc
    except Exception:  # noqa: BLE001
        return {"bindings": {}}


def clear_surface_identity_bindings_cache() -> None:
    """Drop cached catalog (e.g. after materialize rewrite)."""
    load_surface_identity_bindings.cache_clear()


def resolve_preferred_surface(operation: str) -> dict[str, Any] | None:
    """Look up preferred surface for an operation or known alias."""
    doc = load_surface_identity_bindings()
    bindings = doc.get("bindings") or {}
    if not isinstance(bindings, dict):
        return None
    op = str(operation or "").strip()
    if not op:
        return None
    if op in bindings and isinstance(bindings[op], dict):
        return dict(bindings[op])
    # normalize dotted/kit style
    candidates = {op, op.replace(".", "_"), op.rsplit(".", 1)[-1]}
    for key, rec in bindings.items():
        if not isinstance(rec, dict):
            continue
        if key in candidates:
            return dict(rec)
        aliases = set(str(a) for a in (rec.get("aliases") or []))
        idl = set(str(a) for a in (rec.get("idl_methods") or []))
        if candidates & aliases or candidates & idl:
            return dict(rec)
        handler = str(rec.get("handler") or "")
        if handler and (handler in candidates or handler.rsplit(".", 1)[-1] in candidates):
            return dict(rec)
    return None


def preferred_path_for(operation: str) -> str:
    rec = resolve_preferred_surface(operation)
    return str((rec or {}).get("preferred_path") or "")


def preferred_handler_for(operation: str) -> str:
    rec = resolve_preferred_surface(operation)
    return str((rec or {}).get("handler") or "")


def _category_from_path(path: str) -> str:
    p = str(path or "").replace("\\", "/")
    for frag, cat in _PATH_CATEGORY_HINTS:
        if frag in p:
            return cat
    return ""


def _tool_name_from_binding(rec: Mapping[str, Any], requested: str) -> str:
    """Pick the best local tool_name for hierarchical dispatch."""
    handler = str(rec.get("handler") or "").strip()
    if handler:
        # strip module-qualified prefixes: foo.bar.tools_dispatch → tools_dispatch
        leaf = handler.rsplit(".", 1)[-1]
        if leaf and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", leaf):
            return leaf
    # fall back to binding operation key if simpler than request
    op = str(rec.get("operation") or "").strip()
    if op and "." not in op and op != requested:
        return op
    # kit-style ipfs.add → try snake
    if "." in requested:
        snake = requested.replace(".", "_")
        return snake
    return requested


def resolve_dispatch_target(
    category: str,
    tool_name: str,
) -> dict[str, Any]:
    """Resolve category/tool_name through surface identity bindings.

    Returns a mediation record:
    ``{category, tool_name, binding, resolved, reason}``
    """
    cat = str(category or "").strip()
    tool = str(tool_name or "").strip()
    qualified = f"{cat}.{tool}" if cat and tool else tool
    lookup_keys = [tool, qualified, tool.replace(".", "_"), f"{cat}_{tool}"]

    binding = None
    matched_key = ""
    for key in lookup_keys:
        binding = resolve_preferred_surface(key)
        if binding:
            matched_key = key
            break

    if not binding:
        return {
            "category": cat,
            "tool_name": tool,
            "resolved": False,
            "reason": "no_binding",
            "binding": None,
            "matched_key": "",
            "mediation": "surface_identity_bindings",
        }

    path = str(binding.get("preferred_path") or "")
    inferred_cat = _category_from_path(path)
    # Keep meta/tools_dispatch as-is when binding is the meta tool itself
    new_tool = _tool_name_from_binding(binding, tool)
    new_cat = inferred_cat or cat

    # Special case: tools_dispatch binding points at server.py — stay on meta dispatch
    if binding.get("operation") == "tools_dispatch" or new_tool == "tools_dispatch":
        return {
            "category": cat or "meta",
            "tool_name": tool if tool != "tools_dispatch" else new_tool,
            "resolved": True,
            "reason": "binding_meta_passthrough",
            "binding": binding,
            "matched_key": matched_key,
            "mediation": "surface_identity_bindings",
            "preferred_path": path,
            "handler": binding.get("handler"),
        }

    return {
        "category": new_cat,
        "tool_name": new_tool,
        "resolved": True,
        "reason": "binding_applied",
        "binding": binding,
        "matched_key": matched_key,
        "mediation": "surface_identity_bindings",
        "preferred_path": path,
        "handler": binding.get("handler"),
        "requested_category": cat,
        "requested_tool_name": tool,
    }


def resolve_tool_name_for_interop(tool_name: str) -> dict[str, Any]:
    """Resolve a flat tool name for package_mcp_interop tools/call."""
    tool = str(tool_name or "").strip()
    binding = resolve_preferred_surface(tool)
    if not binding:
        # try leaf / snake forms
        binding = resolve_preferred_surface(tool.replace(".", "_"))
    if not binding:
        return {
            "tool_name": tool,
            "resolved": False,
            "reason": "no_binding",
            "binding": None,
            "mediation": "surface_identity_bindings",
        }
    preferred = _tool_name_from_binding(binding, tool)
    # If binding operation is a clean MCP tool name, prefer it for interop
    op = str(binding.get("operation") or "")
    if op and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", op):
        preferred = op
    elif preferred and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", preferred):
        pass
    else:
        preferred = tool.replace(".", "_")
    return {
        "tool_name": preferred,
        "resolved": True,
        "reason": "binding_applied" if preferred != tool else "binding_identity",
        "binding": binding,
        "requested_tool_name": tool,
        "preferred_path": binding.get("preferred_path"),
        "handler": binding.get("handler"),
        "mediation": "surface_identity_bindings",
    }


__all__ = [
    "clear_surface_identity_bindings_cache",
    "load_surface_identity_bindings",
    "preferred_handler_for",
    "preferred_path_for",
    "resolve_dispatch_target",
    "resolve_preferred_surface",
    "resolve_tool_name_for_interop",
]
