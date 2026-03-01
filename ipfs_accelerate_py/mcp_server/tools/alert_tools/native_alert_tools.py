"""Native alert tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_alert_api() -> Dict[str, Any]:
    """Resolve source alert APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.alert_tools.discord_alert_tools import (  # type: ignore
            evaluate_alert_rules as _evaluate_alert_rules,
            list_alert_rules as _list_alert_rules,
            send_discord_message as _send_discord_message,
        )

        return {
            "send_discord_message": _send_discord_message,
            "evaluate_alert_rules": _evaluate_alert_rules,
            "list_alert_rules": _list_alert_rules,
        }
    except Exception:
        logger.warning("Source alert_tools import unavailable, using fallback alert functions")

        async def _send_fallback(
            text: str,
            role_names: Optional[List[str]] = None,
            channel_id: Optional[str] = None,
            thread_id: Optional[str] = None,
            config_file: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = role_names, channel_id, thread_id, config_file
            return {
                "status": "success",
                "message": "fallback-discord-send",
                "text": text,
            }

        async def _evaluate_fallback(
            event: Dict[str, Any],
            rule_ids: Optional[List[str]] = None,
            config_file: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = rule_ids, config_file
            return {
                "status": "success",
                "triggered_rules": 0,
                "results": [],
                "event": event,
            }

        def _list_fallback(
            enabled_only: bool = False,
            config_file: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = enabled_only, config_file
            return {
                "status": "success",
                "count": 0,
                "rules": [],
            }

        return {
            "send_discord_message": _send_fallback,
            "evaluate_alert_rules": _evaluate_fallback,
            "list_alert_rules": _list_fallback,
        }


_API = _load_alert_api()


async def send_discord_message(
    text: str,
    role_names: Optional[List[str]] = None,
    channel_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    config_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a Discord text notification for alerts."""
    return await _API["send_discord_message"](
        text=text,
        role_names=role_names,
        channel_id=channel_id,
        thread_id=thread_id,
        config_file=config_file,
    )


async def evaluate_alert_rules(
    event: Dict[str, Any],
    rule_ids: Optional[List[str]] = None,
    config_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate alert rules against an event payload."""
    return await _API["evaluate_alert_rules"](
        event=event,
        rule_ids=rule_ids,
        config_file=config_file,
    )


async def list_alert_rules(
    enabled_only: bool = False,
    config_file: Optional[str] = None,
) -> Dict[str, Any]:
    """List configured alert rules."""
    result = _API["list_alert_rules"](
        enabled_only=enabled_only,
        config_file=config_file,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_alert_tools(manager: Any) -> None:
    """Register native alert tools in unified hierarchical manager."""
    manager.register_tool(
        category="alert_tools",
        name="send_discord_message",
        func=send_discord_message,
        description="Send Discord alert notification messages.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "role_names": {"type": ["array", "null"], "items": {"type": "string"}},
                "channel_id": {"type": ["string", "null"]},
                "thread_id": {"type": ["string", "null"]},
                "config_file": {"type": ["string", "null"]},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "alerts"],
    )

    manager.register_tool(
        category="alert_tools",
        name="evaluate_alert_rules",
        func=evaluate_alert_rules,
        description="Evaluate configured alert rules against an event.",
        input_schema={
            "type": "object",
            "properties": {
                "event": {"type": "object"},
                "rule_ids": {"type": ["array", "null"], "items": {"type": "string"}},
                "config_file": {"type": ["string", "null"]},
            },
            "required": ["event"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "alerts"],
    )

    manager.register_tool(
        category="alert_tools",
        name="list_alert_rules",
        func=list_alert_rules,
        description="List available alert rules.",
        input_schema={
            "type": "object",
            "properties": {
                "enabled_only": {"type": "boolean"},
                "config_file": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "alerts"],
    )
