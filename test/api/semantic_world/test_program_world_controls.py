"""SAWM-039 typed controls, CLI, and diagnostics."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.semantic_state.cli import ProgramWorldCLI
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_service import (
    ProgramWorldService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools.semantic_state.program_world import (
    semantic_world_mcp_tools,
)


def test_service_cli_and_mcp_are_proposal_only() -> None:
    service = ProgramWorldService()
    assert service.status().completion_authority is False
    reuse = service.operation("reuse")
    assert reuse["proposal_only"] is True
    assert reuse["admitted"] is False
    assert reuse["completion_authority"] is False
    ranked = service.operation(
        "call-target",
        {"current_symbol": "main", "static_candidates": ("helper",)},
    )
    assert ranked["proposal_only"] is True
    assert ranked["ranked"] == ["helper"]
    assert ProgramWorldCLI().status_json()["completion_authority"] is False
    assert semantic_world_mcp_tools("status")["completion_authority"] is False
    assert "semantic_world_status" in semantic_world_mcp_tools()
