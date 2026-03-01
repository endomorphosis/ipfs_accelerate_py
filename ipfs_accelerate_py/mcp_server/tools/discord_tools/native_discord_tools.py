"""Native discord-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_discord_tools_api() -> Dict[str, Any]:
    """Resolve source discord-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.discord_tools import (  # type: ignore
            discord_list_channels as _discord_list_channels,
            discord_list_guilds as _discord_list_guilds,
        )

        return {
            "discord_list_guilds": _discord_list_guilds,
            "discord_list_channels": _discord_list_channels,
        }
    except Exception:
        logger.warning("Source discord_tools import unavailable, using fallback discord-tools functions")

        async def _list_guilds_fallback(token: Optional[str] = None) -> Dict[str, Any]:
            _ = token
            return {
                "status": "error",
                "error": "Discord wrapper not available",
                "guilds": [],
                "count": 0,
                "tool": "discord_list_guilds",
            }

        async def _list_channels_fallback(
            guild_id: str,
            token: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = token
            if not guild_id:
                return {
                    "status": "error",
                    "error": "guild_id is required",
                    "channels": [],
                    "count": 0,
                    "tool": "discord_list_channels",
                }
            return {
                "status": "error",
                "error": "Discord wrapper not available",
                "guild_id": guild_id,
                "channels": [],
                "count": 0,
                "tool": "discord_list_channels",
            }

        return {
            "discord_list_guilds": _list_guilds_fallback,
            "discord_list_channels": _list_channels_fallback,
        }


_API = _load_discord_tools_api()


async def discord_list_guilds(token: Optional[str] = None) -> Dict[str, Any]:
    """List available Discord guilds for the configured token."""
    result = _API["discord_list_guilds"](token=token)
    if hasattr(result, "__await__"):
        return await result
    return result


async def discord_list_channels(guild_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """List channels in a Discord guild."""
    result = _API["discord_list_channels"](guild_id=guild_id, token=token)
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_discord_tools(manager: Any) -> None:
    """Register native discord-tools category tools in unified manager."""
    manager.register_tool(
        category="discord_tools",
        name="discord_list_guilds",
        func=discord_list_guilds,
        description="List Discord guilds available to the configured token.",
        input_schema={
            "type": "object",
            "properties": {
                "token": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "discord-tools"],
    )

    manager.register_tool(
        category="discord_tools",
        name="discord_list_channels",
        func=discord_list_channels,
        description="List channels for a Discord guild.",
        input_schema={
            "type": "object",
            "properties": {
                "guild_id": {"type": "string"},
                "token": {"type": ["string", "null"]},
            },
            "required": ["guild_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "discord-tools"],
    )
