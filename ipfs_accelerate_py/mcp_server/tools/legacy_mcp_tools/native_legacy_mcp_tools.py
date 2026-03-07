"""Native legacy-mcp-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_legacy_api() -> Dict[str, Any]:
    """Resolve source legacy MCP APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.legacy_mcp_tools import (  # type: ignore
            TEMPORAL_DEONTIC_LOGIC_TOOLS,
        )

        return {"TEMPORAL_DEONTIC_LOGIC_TOOLS": TEMPORAL_DEONTIC_LOGIC_TOOLS}
    except Exception:
        logger.warning("Source legacy_mcp_tools import unavailable, using fallback legacy info")
        return {}


_API = _load_legacy_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        if "status" not in envelope:
            if envelope.get("error") or envelope.get("success") is False:
                envelope["status"] = "error"
            else:
                envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def legacy_tools_inventory() -> Dict[str, Any]:
    """Return inventory metadata for deprecated legacy MCP tools."""
    try:
        tools = _API.get("TEMPORAL_DEONTIC_LOGIC_TOOLS", [])
        envelope = _normalize_payload(
            {
                "deprecated": True,
                "temporal_deontic_tool_count": len(tools) if isinstance(tools, list) else 0,
                "fallback": not bool(_API),
            }
        )
        envelope.setdefault("status", "success")
        envelope.setdefault("deprecated", True)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("temporal_deontic_tool_count", 0)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), deprecated=True)


def register_native_legacy_mcp_tools(manager: Any) -> None:
    """Register native legacy-mcp-tools category tools in unified manager."""
    manager.register_tool(
        category="legacy_mcp_tools",
        name="legacy_tools_inventory",
        func=legacy_tools_inventory,
        description="Inspect deprecated legacy MCP tool inventory metadata.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "legacy-mcp-tools"],
    )
