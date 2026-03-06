"""Native discord-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_discord_tools_api() -> Dict[str, Any]:
    """Resolve source discord-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.discord_tools import (  # type: ignore
            discord_analyze_export as _discord_analyze_export,
            discord_batch_convert_exports as _discord_batch_convert_exports,
            discord_convert_export as _discord_convert_export,
            discord_export_channel as _discord_export_channel,
            discord_list_channels as _discord_list_channels,
            discord_list_dm_channels as _discord_list_dm_channels,
            discord_list_guilds as _discord_list_guilds,
        )

        return {
            "discord_list_guilds": _discord_list_guilds,
            "discord_list_channels": _discord_list_channels,
            "discord_list_dm_channels": _discord_list_dm_channels,
            "discord_export_channel": _discord_export_channel,
            "discord_analyze_export": _discord_analyze_export,
            "discord_convert_export": _discord_convert_export,
            "discord_batch_convert_exports": _discord_batch_convert_exports,
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

        async def _list_dm_channels_fallback(token: Optional[str] = None) -> Dict[str, Any]:
            _ = token
            return {
                "status": "error",
                "error": "Discord wrapper not available",
                "channels": [],
                "count": 0,
                "tool": "discord_list_dm_channels",
            }

        async def _export_channel_fallback(
            channel_id: str,
            token: Optional[str] = None,
            output_path: Optional[str] = None,
            format: str = "Json",
            after_date: Optional[str] = None,
            before_date: Optional[str] = None,
            filter_text: Optional[str] = None,
            download_media: bool = False,
            reuse_media: bool = False,
            partition_limit: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = token, output_path, format, after_date, before_date, filter_text, download_media, reuse_media, partition_limit
            return {
                "status": "error",
                "error": "Discord wrapper not available",
                "channel_id": channel_id,
                "tool": "discord_export_channel",
            }

        async def _analyze_export_fallback(
            export_path: str,
            analysis_types: Optional[list[str]] = None,
        ) -> Dict[str, Any]:
            return {
                "status": "error",
                "error": "Discord wrapper not available",
                "export_path": export_path,
                "analysis_types": list(analysis_types or []),
                "tool": "discord_analyze_export",
            }

        async def _convert_export_fallback(
            input_path: str,
            output_path: str,
            to_format: str = "jsonl",
            token: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None,
            compression: Optional[str] = None,
            **_: Any,
        ) -> Dict[str, Any]:
            _ = token, context, compression
            return {
                "status": "error",
                "error": "Discord wrapper not available",
                "input_path": input_path,
                "output_path": output_path,
                "to_format": to_format,
                "tool": "discord_convert_export",
            }

        async def _batch_convert_exports_fallback(
            input_dir: str,
            output_dir: str,
            to_format: str = "jsonl",
            file_pattern: str = "*.json",
            token: Optional[str] = None,
            **_: Any,
        ) -> Dict[str, Any]:
            _ = token
            return {
                "status": "error",
                "error": "Discord wrapper not available",
                "input_dir": input_dir,
                "output_dir": output_dir,
                "to_format": to_format,
                "file_pattern": file_pattern,
                "tool": "discord_batch_convert_exports",
            }

        return {
            "discord_list_guilds": _list_guilds_fallback,
            "discord_list_channels": _list_channels_fallback,
            "discord_list_dm_channels": _list_dm_channels_fallback,
            "discord_export_channel": _export_channel_fallback,
            "discord_analyze_export": _analyze_export_fallback,
            "discord_convert_export": _convert_export_fallback,
            "discord_batch_convert_exports": _batch_convert_exports_fallback,
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


async def discord_list_dm_channels(token: Optional[str] = None) -> Dict[str, Any]:
    """List DM channels available to the configured token."""
    normalized_token = None if token is None else str(token).strip()
    if token is not None and not normalized_token:
        return {
            "status": "error",
            "error": "token must be a non-empty string when provided",
            "channels": [],
            "count": 0,
            "tool": "discord_list_dm_channels",
        }

    try:
        result = _API["discord_list_dm_channels"](token=normalized_token)
        if hasattr(result, "__await__"):
            payload = _normalize_payload(await result)
        else:
            payload = _normalize_payload(result)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "channels": [],
            "count": 0,
            "tool": "discord_list_dm_channels",
        }

    payload.setdefault("tool", "discord_list_dm_channels")
    payload.setdefault("channels", [])
    payload.setdefault("count", len(payload.get("channels", [])) if isinstance(payload.get("channels", []), list) else 0)
    return payload


async def discord_export_channel(
    channel_id: str,
    token: Optional[str] = None,
    output_path: Optional[str] = None,
    format: str = "Json",
    after_date: Optional[str] = None,
    before_date: Optional[str] = None,
    filter_text: Optional[str] = None,
    download_media: bool = False,
    reuse_media: bool = False,
    partition_limit: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a Discord channel using the source-compatible wrapper contract."""
    normalized_channel_id = str(channel_id or "").strip()
    if not normalized_channel_id:
        return {"status": "error", "error": "channel_id is required", "channel_id": channel_id, "tool": "discord_export_channel"}
    normalized_token = None if token is None else str(token).strip()
    if token is not None and not normalized_token:
        return {"status": "error", "error": "token must be a non-empty string when provided", "channel_id": normalized_channel_id, "tool": "discord_export_channel"}
    normalized_output_path = None if output_path is None else str(output_path).strip()
    if output_path is not None and not normalized_output_path:
        return {"status": "error", "error": "output_path must be a non-empty string when provided", "channel_id": normalized_channel_id, "tool": "discord_export_channel"}
    normalized_format = str(format or "").strip()
    if not normalized_format:
        return {"status": "error", "error": "format must be a non-empty string", "channel_id": normalized_channel_id, "tool": "discord_export_channel"}
    if not isinstance(download_media, bool):
        return {"status": "error", "error": "download_media must be a boolean", "channel_id": normalized_channel_id, "tool": "discord_export_channel"}
    if not isinstance(reuse_media, bool):
        return {"status": "error", "error": "reuse_media must be a boolean", "channel_id": normalized_channel_id, "tool": "discord_export_channel"}

    try:
        result = _API["discord_export_channel"](
            channel_id=normalized_channel_id,
            token=normalized_token,
            output_path=normalized_output_path,
            format=normalized_format,
            after_date=after_date,
            before_date=before_date,
            filter_text=filter_text,
            download_media=download_media,
            reuse_media=reuse_media,
            partition_limit=partition_limit,
        )
        payload = _normalize_payload(await result if hasattr(result, "__await__") else result)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "channel_id": normalized_channel_id, "tool": "discord_export_channel"}

    payload.setdefault("tool", "discord_export_channel")
    payload.setdefault("channel_id", normalized_channel_id)
    payload.setdefault("format", normalized_format)
    return payload


async def discord_analyze_export(export_path: str, analysis_types: Optional[list[str]] = None) -> Dict[str, Any]:
    """Analyze a previously exported Discord dataset."""
    normalized_export_path = str(export_path or "").strip()
    if not normalized_export_path:
        return {"status": "error", "error": "export_path is required", "export_path": export_path, "tool": "discord_analyze_export"}
    if analysis_types is not None and (
        not isinstance(analysis_types, list) or not all(isinstance(item, str) and item.strip() for item in analysis_types)
    ):
        return {"status": "error", "error": "analysis_types must be an array of non-empty strings when provided", "export_path": normalized_export_path, "tool": "discord_analyze_export"}

    try:
        result = _API["discord_analyze_export"](export_path=normalized_export_path, analysis_types=analysis_types)
        payload = _normalize_payload(await result if hasattr(result, "__await__") else result)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "export_path": normalized_export_path, "tool": "discord_analyze_export"}

    payload.setdefault("tool", "discord_analyze_export")
    payload.setdefault("export_path", normalized_export_path)
    payload.setdefault("analysis_types", list(analysis_types or []))
    return payload


async def discord_convert_export(
    input_path: str,
    output_path: str,
    to_format: str = "jsonl",
    token: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    compression: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a Discord export between supported formats."""
    normalized_input_path = str(input_path or "").strip()
    normalized_output_path = str(output_path or "").strip()
    normalized_to_format = str(to_format or "").strip().lower()
    if not normalized_input_path:
        return {"status": "error", "error": "input_path is required", "tool": "discord_convert_export"}
    if not normalized_output_path:
        return {"status": "error", "error": "output_path is required", "input_path": normalized_input_path, "tool": "discord_convert_export"}
    if normalized_to_format not in {"json", "jsonl", "jsonld", "jsonld-logic", "parquet", "ipld", "car", "csv"}:
        return {"status": "error", "error": "to_format must be one of: car, csv, ipld, json, jsonl, jsonld, jsonld-logic, parquet", "input_path": normalized_input_path, "tool": "discord_convert_export"}
    normalized_token = None if token is None else str(token).strip()
    if token is not None and not normalized_token:
        return {"status": "error", "error": "token must be a non-empty string when provided", "input_path": normalized_input_path, "tool": "discord_convert_export"}
    if context is not None and not isinstance(context, dict):
        return {"status": "error", "error": "context must be an object when provided", "input_path": normalized_input_path, "tool": "discord_convert_export"}
    if compression is not None and not str(compression).strip():
        return {"status": "error", "error": "compression must be a non-empty string when provided", "input_path": normalized_input_path, "tool": "discord_convert_export"}

    try:
        result = _API["discord_convert_export"](
            input_path=normalized_input_path,
            output_path=normalized_output_path,
            to_format=normalized_to_format,
            token=normalized_token,
            context=context,
            compression=compression,
        )
        payload = _normalize_payload(await result if hasattr(result, "__await__") else result)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "input_path": normalized_input_path, "tool": "discord_convert_export"}

    payload.setdefault("tool", "discord_convert_export")
    payload.setdefault("input_path", normalized_input_path)
    payload.setdefault("output_path", normalized_output_path)
    payload.setdefault("to_format", normalized_to_format)
    return payload


async def discord_batch_convert_exports(
    input_dir: str,
    output_dir: str,
    to_format: str = "jsonl",
    file_pattern: str = "*.json",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Batch-convert Discord exports in a directory."""
    normalized_input_dir = str(input_dir or "").strip()
    normalized_output_dir = str(output_dir or "").strip()
    normalized_to_format = str(to_format or "").strip().lower()
    normalized_pattern = str(file_pattern or "").strip()
    if not normalized_input_dir:
        return {"status": "error", "error": "input_dir is required", "tool": "discord_batch_convert_exports"}
    if not normalized_output_dir:
        return {"status": "error", "error": "output_dir is required", "input_dir": normalized_input_dir, "tool": "discord_batch_convert_exports"}
    if normalized_to_format not in {"json", "jsonl", "jsonld", "jsonld-logic", "parquet", "ipld", "car", "csv"}:
        return {"status": "error", "error": "to_format must be one of: car, csv, ipld, json, jsonl, jsonld, jsonld-logic, parquet", "input_dir": normalized_input_dir, "tool": "discord_batch_convert_exports"}
    if not normalized_pattern:
        return {"status": "error", "error": "file_pattern must be a non-empty string", "input_dir": normalized_input_dir, "tool": "discord_batch_convert_exports"}
    normalized_token = None if token is None else str(token).strip()
    if token is not None and not normalized_token:
        return {"status": "error", "error": "token must be a non-empty string when provided", "input_dir": normalized_input_dir, "tool": "discord_batch_convert_exports"}

    try:
        result = _API["discord_batch_convert_exports"](
            input_dir=normalized_input_dir,
            output_dir=normalized_output_dir,
            to_format=normalized_to_format,
            file_pattern=normalized_pattern,
            token=normalized_token,
        )
        payload = _normalize_payload(await result if hasattr(result, "__await__") else result)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "input_dir": normalized_input_dir, "tool": "discord_batch_convert_exports"}

    payload.setdefault("tool", "discord_batch_convert_exports")
    payload.setdefault("input_dir", normalized_input_dir)
    payload.setdefault("output_dir", normalized_output_dir)
    payload.setdefault("to_format", normalized_to_format)
    payload.setdefault("file_pattern", normalized_pattern)
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

    manager.register_tool(
        category="discord_tools",
        name="discord_list_dm_channels",
        func=discord_list_dm_channels,
        description="List Discord DM channels available to the configured token.",
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
        name="discord_export_channel",
        func=discord_export_channel,
        description="Export Discord channel messages to a target format.",
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "minLength": 1},
                "token": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "output_path": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "format": {"type": "string", "minLength": 1, "default": "Json"},
                "after_date": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "before_date": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "filter_text": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "download_media": {"type": "boolean", "default": False},
                "reuse_media": {"type": "boolean", "default": False},
                "partition_limit": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
            },
            "required": ["channel_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "discord-tools"],
    )

    manager.register_tool(
        category="discord_tools",
        name="discord_analyze_export",
        func=discord_analyze_export,
        description="Analyze a Discord export dataset.",
        input_schema={
            "type": "object",
            "properties": {
                "export_path": {"type": "string", "minLength": 1},
                "analysis_types": {"anyOf": [{"type": "array", "items": {"type": "string", "minLength": 1}}, {"type": "null"}]},
            },
            "required": ["export_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "discord-tools"],
    )

    manager.register_tool(
        category="discord_tools",
        name="discord_convert_export",
        func=discord_convert_export,
        description="Convert a Discord export to another format.",
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "minLength": 1},
                "output_path": {"type": "string", "minLength": 1},
                "to_format": {"type": "string", "enum": ["json", "jsonl", "jsonld", "jsonld-logic", "parquet", "ipld", "car", "csv"], "default": "jsonl"},
                "token": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "context": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                "compression": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
            },
            "required": ["input_path", "output_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "discord-tools"],
    )

    manager.register_tool(
        category="discord_tools",
        name="discord_batch_convert_exports",
        func=discord_batch_convert_exports,
        description="Batch-convert Discord exports in a directory.",
        input_schema={
            "type": "object",
            "properties": {
                "input_dir": {"type": "string", "minLength": 1},
                "output_dir": {"type": "string", "minLength": 1},
                "to_format": {"type": "string", "enum": ["json", "jsonl", "jsonld", "jsonld-logic", "parquet", "ipld", "car", "csv"], "default": "jsonl"},
                "file_pattern": {"type": "string", "minLength": 1, "default": "*.json"},
                "token": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
            },
            "required": ["input_dir", "output_dir"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "discord-tools"],
    )
