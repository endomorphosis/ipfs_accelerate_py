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


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend output into deterministic status envelopes."""
    if isinstance(result, dict):
        payload = dict(result)
        if payload.get("error"):
            payload.setdefault("status", "error")
        else:
            payload.setdefault("status", "success")
        return payload
    return {"status": "success", "result": result}


async def discord_list_guilds(token: Optional[str] = None) -> Dict[str, Any]:
    """List available Discord guilds for the configured token."""
    normalized_token = None if token is None else str(token).strip()
    if token is not None and not normalized_token:
        return {
            "status": "error",
            "error": "token must be a non-empty string when provided",
            "token": token,
        }

    try:
        result = _API["discord_list_guilds"](token=normalized_token)
        if hasattr(result, "__await__"):
            payload = _normalize_payload(await result)
        else:
            payload = _normalize_payload(result)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "guilds": [],
            "count": 0,
            "tool": "discord_list_guilds",
        }

    payload.setdefault("tool", "discord_list_guilds")
    payload.setdefault("guilds", [])
    payload.setdefault("count", len(payload.get("guilds", [])) if isinstance(payload.get("guilds", []), list) else 0)
    return payload


async def discord_list_channels(guild_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """List channels in a Discord guild."""
    normalized_guild_id = str(guild_id or "").strip()
    if not normalized_guild_id:
        return {
            "status": "error",
            "error": "guild_id is required",
            "guild_id": guild_id,
            "channels": [],
            "count": 0,
            "tool": "discord_list_channels",
        }

    normalized_token = None if token is None else str(token).strip()
    if token is not None and not normalized_token:
        return {
            "status": "error",
            "error": "token must be a non-empty string when provided",
            "guild_id": normalized_guild_id,
            "channels": [],
            "count": 0,
            "tool": "discord_list_channels",
        }

    try:
        result = _API["discord_list_channels"](guild_id=normalized_guild_id, token=normalized_token)
        if hasattr(result, "__await__"):
            payload = _normalize_payload(await result)
        else:
            payload = _normalize_payload(result)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "guild_id": normalized_guild_id,
            "channels": [],
            "count": 0,
            "tool": "discord_list_channels",
        }

    payload.setdefault("tool", "discord_list_channels")
    payload.setdefault("guild_id", normalized_guild_id)
    payload.setdefault("channels", [])
    payload.setdefault("count", len(payload.get("channels", [])) if isinstance(payload.get("channels", []), list) else 0)
    return payload


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
                "token": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ]
                },
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
                "guild_id": {"type": "string", "minLength": 1},
                "token": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["guild_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "discord-tools"],
    )
