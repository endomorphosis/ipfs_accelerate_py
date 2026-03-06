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
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return {
            "status": "error",
            "message": "text is required",
            "text": text,
        }

    normalized_role_names: Optional[List[str]] = None
    if role_names is not None:
        if not isinstance(role_names, list) or not all(isinstance(item, str) for item in role_names):
            return {
                "status": "error",
                "message": "role_names must be an array of strings when provided",
                "role_names": role_names,
            }
        normalized_role_names = [str(item).strip() for item in role_names]
        if any(not item for item in normalized_role_names):
            return {
                "status": "error",
                "message": "role_names cannot contain empty strings",
                "role_names": role_names,
            }

    normalized_channel_id = str(channel_id).strip() if channel_id is not None else None
    if channel_id is not None and not normalized_channel_id:
        return {
            "status": "error",
            "message": "channel_id must be a non-empty string when provided",
            "channel_id": channel_id,
        }
    normalized_thread_id = str(thread_id).strip() if thread_id is not None else None
    if thread_id is not None and not normalized_thread_id:
        return {
            "status": "error",
            "message": "thread_id must be a non-empty string when provided",
            "thread_id": thread_id,
        }
    normalized_config_file = str(config_file).strip() if config_file is not None else None
    if config_file is not None and not normalized_config_file:
        return {
            "status": "error",
            "message": "config_file must be a non-empty string when provided",
            "config_file": config_file,
        }

    result = await _API["send_discord_message"](
        text=normalized_text,
        role_names=normalized_role_names,
        channel_id=normalized_channel_id,
        thread_id=normalized_thread_id,
        config_file=normalized_config_file,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("text", normalized_text)
    payload.setdefault("role_names", normalized_role_names or [])
    payload.setdefault("channel_id", normalized_channel_id)
    payload.setdefault("thread_id", normalized_thread_id)
    return payload


async def evaluate_alert_rules(
    event: Dict[str, Any],
    rule_ids: Optional[List[str]] = None,
    config_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate alert rules against an event payload."""
    if not isinstance(event, dict):
        return {
            "status": "error",
            "message": "event must be an object",
            "event": event,
        }

    normalized_rule_ids: Optional[List[str]] = None
    if rule_ids is not None:
        if not isinstance(rule_ids, list) or not all(isinstance(item, str) for item in rule_ids):
            return {
                "status": "error",
                "message": "rule_ids must be an array of strings when provided",
                "rule_ids": rule_ids,
            }
        normalized_rule_ids = [str(item).strip() for item in rule_ids]
        if any(not item for item in normalized_rule_ids):
            return {
                "status": "error",
                "message": "rule_ids cannot contain empty strings",
                "rule_ids": rule_ids,
            }

    normalized_config_file = str(config_file).strip() if config_file is not None else None
    if config_file is not None and not normalized_config_file:
        return {
            "status": "error",
            "message": "config_file must be a non-empty string when provided",
            "config_file": config_file,
        }

    result = await _API["evaluate_alert_rules"](
        event=event,
        rule_ids=normalized_rule_ids,
        config_file=normalized_config_file,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("event", event)
    payload.setdefault("rule_ids", normalized_rule_ids or [])
    payload.setdefault("results", [])
    payload.setdefault("triggered_rules", len(payload.get("results") or []))
    return payload


async def list_alert_rules(
    enabled_only: bool = False,
    config_file: Optional[str] = None,
) -> Dict[str, Any]:
    """List configured alert rules."""
    if not isinstance(enabled_only, bool):
        return {
            "status": "error",
            "message": "enabled_only must be a boolean",
            "enabled_only": enabled_only,
        }

    normalized_config_file = str(config_file).strip() if config_file is not None else None
    if config_file is not None and not normalized_config_file:
        return {
            "status": "error",
            "message": "config_file must be a non-empty string when provided",
            "config_file": config_file,
        }

    result = _API["list_alert_rules"](
        enabled_only=enabled_only,
        config_file=normalized_config_file,
    )
    if hasattr(result, "__await__"):
        payload = dict(await result or {})
    else:
        payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("enabled_only", enabled_only)
    payload.setdefault("rules", [])
    payload.setdefault("count", len(payload.get("rules") or []))
    return payload


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
                "enabled_only": {"type": "boolean", "default": False},
                "config_file": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "alerts"],
    )
