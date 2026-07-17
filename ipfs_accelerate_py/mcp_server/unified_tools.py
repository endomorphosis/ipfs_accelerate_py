"""Canonical unified MCP tools module for ipfs_accelerate_py.mcp_server.

Provides the same ``register_*_tools`` interface as the deprecated
``ipfs_accelerate_py.mcp.unified_tools`` module but delegates to the
canonical ``mcp_server`` tool categories instead of the legacy ``mcp/tools``
modules.

Drop-in migration:
    Replace::

        from ipfs_accelerate_py.mcp import unified_tools

    With::

        from ipfs_accelerate_py.mcp_server import unified_tools
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_unified_tools(mcp: Any) -> None:
    """Register all canonical unified tools with an MCP registry instance."""
    logger.info("Registering unified MCP tools")

    for _fn in (
        register_github_tools,
        register_docker_tools,
        register_hardware_tools,
        register_runner_tools,
        register_ipfs_files_tools,
        register_network_tools,
    ):
        try:
            _fn(mcp)
        except Exception as exc:
            logger.warning("Failed to register %s tools: %s", _fn.__name__, exc)

    logger.info("All unified tools registered")


def register_github_tools(mcp: Any) -> None:
    """Register GitHub tools with the MCP registry."""
    try:
        from .tools.github_tools.native_github_tools import (
            register_native_github_tools,
        )
        register_native_github_tools(mcp)
    except Exception as exc:
        logger.warning("GitHub tools unavailable: %s", exc)
        # Fallback: register a stub tool so callers can confirm registration ran.
        try:
            async def github_list_repos() -> dict:
                return {"status": "unavailable", "message": str(exc)}
            mcp.register_tool(
                name="github_list_repos",
                function=github_list_repos,
                description="List GitHub repositories (unavailable).",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        except Exception:
            pass


def register_docker_tools(mcp: Any) -> None:
    """Register Docker tools with the MCP registry."""
    try:
        from .tools.docker_tools.native_docker_tools import (
            register_native_docker_tools,
        )
        register_native_docker_tools(mcp)
    except Exception as exc:
        logger.warning("Docker tools unavailable: %s", exc)
        try:
            async def docker_list_containers() -> dict:
                return {"status": "unavailable", "message": str(exc)}
            mcp.register_tool(
                name="docker_list_containers",
                function=docker_list_containers,
                description="List Docker containers (unavailable).",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        except Exception:
            pass


def register_hardware_tools(mcp: Any) -> None:
    """Register hardware detection tools with the MCP registry."""
    try:
        from .tools.hardware_tools.native_hardware_tools import (
            register_native_hardware_tools,
        )
        register_native_hardware_tools(mcp)
    except Exception as exc:
        logger.warning("Hardware tools unavailable: %s", exc)
        try:
            async def hardware_get_info() -> dict:
                return {"status": "unavailable", "message": str(exc)}
            mcp.register_tool(
                name="hardware_get_info",
                function=hardware_get_info,
                description="Get hardware info (unavailable).",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        except Exception:
            pass


def register_runner_tools(mcp: Any) -> None:
    """Register runner/process management tools with the MCP registry."""
    try:
        from .tools.shared_tools.native_shared_tools import (
            register_native_shared_tools,
        )
        register_native_shared_tools(mcp)
    except Exception as exc:
        logger.warning("Runner tools unavailable: %s", exc)
        try:
            async def runner_status() -> dict:
                return {"status": "unavailable", "message": str(exc)}
            mcp.register_tool(
                name="runner_status",
                function=runner_status,
                description="Get runner status (unavailable).",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        except Exception:
            pass


def register_ipfs_files_tools(mcp: Any) -> None:
    """Register IPFS file operation tools with the MCP registry."""
    try:
        from .tools.ipfs.native_ipfs_tools import (
            register_native_ipfs_tools,
        )
        register_native_ipfs_tools(mcp)
    except Exception as exc:
        logger.warning("IPFS file tools unavailable: %s", exc)
        try:
            async def ipfs_add_file(path: str) -> dict:
                return {"status": "unavailable", "message": str(exc)}
            mcp.register_tool(
                name="ipfs_add_file",
                function=ipfs_add_file,
                description="Add a file to IPFS (unavailable).",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        except Exception:
            pass


def register_network_tools(mcp: Any) -> None:
    """Register IPFS network/swarm tools with the MCP registry."""
    try:
        from .tools.ipfs_network_tools.native_ipfs_network_tools import (
            register_native_ipfs_network_tools,
        )
        register_native_ipfs_network_tools(mcp)
    except Exception as exc:
        logger.warning("Network tools unavailable: %s", exc)
        try:
            async def network_get_peers() -> dict:
                return {"status": "unavailable", "message": str(exc)}
            mcp.register_tool(
                name="network_get_peers",
                function=network_get_peers,
                description="Get IPFS network peers (unavailable).",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        except Exception:
            pass


__all__ = [
    "register_unified_tools",
    "register_github_tools",
    "register_docker_tools",
    "register_hardware_tools",
    "register_runner_tools",
    "register_ipfs_files_tools",
    "register_network_tools",
]
