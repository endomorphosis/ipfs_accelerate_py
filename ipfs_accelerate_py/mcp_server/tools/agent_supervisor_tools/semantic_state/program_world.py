"""SAWM-039 MCP adapter for ProgramWorldService."""

from __future__ import annotations

from typing import Any, Mapping

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_service import (
    ProgramWorldService,
    describe_controls,
)

PROGRAM_WORLD_MCP_TOOLS: tuple[str, ...] = ("semantic_world_status",)


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
    return service.operation(operation, payload)


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
