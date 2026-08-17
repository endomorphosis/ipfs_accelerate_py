"""ASE3-011 MCP++ parity smoke for prompt-lifecycle tools."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    PROMPT_LIFECYCLE_TOOLS,
    prompt_lifecycle_discovery_manifest,
    register_prompt_lifecycle_tools,
)


class _Manager:
    def __init__(self) -> None:
        self.names: list[str] = []

    def register_tool(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.names.append(str(kwargs["name"]))


def test_mcplusplus_prompt_tools_register_closed_vocabulary() -> None:
    """MCP++ shares the same closed prompt-lifecycle vocabulary as MCP."""

    manager = _Manager()
    register_prompt_lifecycle_tools(manager)
    assert set(manager.names) == set(PROMPT_LIFECYCLE_TOOLS)
    manifest = prompt_lifecycle_discovery_manifest()
    assert set(manifest["tools"]) == set(manager.names)
    assert manifest["path_authority"] == "server_allowlist_only"
