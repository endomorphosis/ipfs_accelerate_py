"""Canonical helpers for discovering and registering P2P-related tool modules."""

from __future__ import annotations

import importlib
from typing import Any

P2P_TOOL_MODULES: tuple[tuple[str, str], ...] = (
    ("ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools", "register_native_p2p_tools"),
    ("ipfs_accelerate_py.mcp_server.tools.p2p_tools.native_p2p_tools", "register_native_p2p_tools_category"),
    (
        "ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools.native_p2p_workflow_tools",
        "register_native_p2p_workflow_tools",
    ),
)


def discover_p2p_tool_modules() -> list[dict[str, str]]:
    """Return discovery records for canonical P2P module registration specs."""
    records: list[dict[str, str]] = []
    for module_path, registrar_name in P2P_TOOL_MODULES:
        try:
            module = importlib.import_module(module_path)
            registrar = getattr(module, registrar_name, None)
            status = "available" if callable(registrar) else "missing_registrar"
            records.append(
                {
                    "module": module_path,
                    "registrar": registrar_name,
                    "status": status,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "module": module_path,
                    "registrar": registrar_name,
                    "status": "import_error",
                    "error": str(exc),
                }
            )
    return records


def register_p2p_category_loaders(manager: Any) -> dict[str, int]:
    """Invoke available P2P registrars against manager and report counts."""
    loaded = 0
    failed = 0

    for record in discover_p2p_tool_modules():
        if record.get("status") != "available":
            failed += 1
            continue

        module = importlib.import_module(record["module"])
        registrar = getattr(module, record["registrar"], None)
        if not callable(registrar):
            failed += 1
            continue

        registrar(manager)
        loaded += 1

    return {"loaded": loaded, "failed": failed, "total": len(P2P_TOOL_MODULES)}


def get_p2p_tool_summary() -> dict[str, Any]:
    """Return canonical summary payload for P2P module discovery."""
    records = discover_p2p_tool_modules()
    available = sum(1 for r in records if r.get("status") == "available")
    return {
        "total": len(records),
        "available": available,
        "records": records,
    }


__all__ = [
    "P2P_TOOL_MODULES",
    "discover_p2p_tool_modules",
    "register_p2p_category_loaders",
    "get_p2p_tool_summary",
]
