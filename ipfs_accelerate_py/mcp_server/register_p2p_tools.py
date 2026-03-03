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


def _resolve_p2p_registrars() -> list[dict[str, Any]]:
    """Resolve P2P registrar modules once for discovery/registration paths."""
    records: list[dict[str, Any]] = []
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
                    "callable": registrar if callable(registrar) else None,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "module": module_path,
                    "registrar": registrar_name,
                    "status": "import_error",
                    "error": str(exc),
                    "callable": None,
                }
            )
    return records


def discover_p2p_tool_modules() -> list[dict[str, str]]:
    """Return discovery records for canonical P2P module registration specs."""
    records = _resolve_p2p_registrars()
    return [
        {
            key: value
            for key, value in record.items()
            if key in {"module", "registrar", "status", "error"} and value is not None
        }
        for record in records
    ]


def register_p2p_category_loaders(manager: Any) -> dict[str, int]:
    """Invoke available P2P registrars against manager and report counts."""
    loaded = 0
    failed = 0

    for record in _resolve_p2p_registrars():
        if record.get("status") != "available":
            failed += 1
            continue

        registrar = record.get("callable")
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
