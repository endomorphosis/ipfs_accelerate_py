"""SAWM-039 MCP adapter for ProgramWorldService."""

from __future__ import annotations

from typing import Any, Mapping

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_service import (
    ProgramWorldService,
    describe_controls,
)

PROGRAM_WORLD_MCP_TOOLS: tuple[str, ...] = (
    "semantic_world_status",
    "semantic_world_resolve",
    "semantic_world_operation",
)


def semantic_world_mcp_tools(
    operation: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> Any:
    if operation is None:
        return PROGRAM_WORLD_MCP_TOOLS
    service = ProgramWorldService()
    if operation in {"status", "semantic_world_status"}:
        status = service.status()
        return {
            "status": status.status,
            "completion_authority": False,
            "subprocess": False,
            "service": describe_controls(),
        }
    if operation in {"resolve", "semantic_world_resolve"}:
        resolved = service.resolve(payload or {})
        return {
            "query": resolved.query,
            "resolved": resolved.resolved,
            "reason_code": resolved.reason_code,
            "completion_authority": False,
            "subprocess": False,
        }
    name = str((payload or {}).get("operation") or operation)
    if name in {"semantic_world_operation", "operation"}:
        name = str((payload or {}).get("name") or "reuse")
        inner = dict(payload or {})
        inner.pop("operation", None)
        inner.pop("name", None)
        result = service.operation(name, inner)
    else:
        result = service.operation(name, payload)
    result = dict(result)
    result["subprocess"] = False
    result["completion_authority"] = False
    return result


def semantic_world_status(arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    del arguments
    return {
        "schema": "sawm/program-world-mcp@1",
        "interface": "SemanticWorldMCP@1",
        "tool": "semantic_world_status",
        "subprocess": False,
        "authoritative": False,
        "completion_authority": False,
        "service": describe_controls(),
    }
