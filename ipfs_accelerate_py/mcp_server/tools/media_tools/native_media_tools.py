"""Native media-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def _load_media_tools_api() -> Dict[str, Any]:
    """Resolve source media-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.media_tools import (  # type: ignore
            ffmpeg_analyze as _ffmpeg_analyze,
            ytdlp_extract_info as _ytdlp_extract_info,
        )

        return {
            "ffmpeg_analyze": _ffmpeg_analyze,
            "ytdlp_extract_info": _ytdlp_extract_info,
        }
    except Exception:
        logger.warning(
            "Source media_tools import unavailable, using fallback media-tools functions"
        )

        async def _ffmpeg_analyze_fallback(input_file: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "status": "success",
                "input_file": input_file,
                "metadata": {},
                "fallback": True,
            }

        async def _ytdlp_extract_info_fallback(
            url: str,
            flat_playlist: bool = False,
            no_warnings: bool = True,
        ) -> Dict[str, Any]:
            _ = (flat_playlist, no_warnings)
            return {
                "status": "success",
                "url": url,
                "info": {},
                "fallback": True,
            }

        return {
            "ffmpeg_analyze": _ffmpeg_analyze_fallback,
            "ytdlp_extract_info": _ytdlp_extract_info_fallback,
        }


_API = _load_media_tools_api()


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


async def ffmpeg_analyze(input_file: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze media metadata using FFmpeg-backed wrappers."""
    normalized_input: Union[str, Dict[str, Any]]
    if isinstance(input_file, str):
        normalized = input_file.strip()
        if not normalized:
            return {
                "status": "error",
                "error": "input_file must be a non-empty string or object",
            }
        normalized_input = normalized
    elif isinstance(input_file, dict):
        if not input_file:
            return {
                "status": "error",
                "error": "input_file object must not be empty",
            }
        normalized_input = input_file
    else:
        return {
            "status": "error",
            "error": "input_file must be a non-empty string or object",
        }

    try:
        result = _API["ffmpeg_analyze"](input_file=normalized_input)
        resolved = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "input_file": normalized_input,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("input_file", normalized_input)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("metadata", {})
    return payload


async def ytdlp_extract_info(
    url: str,
    flat_playlist: bool = False,
    no_warnings: bool = True,
) -> Dict[str, Any]:
    """Extract media metadata using yt-dlp-backed wrappers."""
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {
            "status": "error",
            "error": "url is required",
        }
    if not (normalized_url.startswith("http://") or normalized_url.startswith("https://")):
        return {
            "status": "error",
            "error": "url must start with http:// or https://",
            "url": normalized_url,
        }
    if not isinstance(flat_playlist, bool):
        return {
            "status": "error",
            "error": "flat_playlist must be a boolean",
            "url": normalized_url,
        }
    if not isinstance(no_warnings, bool):
        return {
            "status": "error",
            "error": "no_warnings must be a boolean",
            "url": normalized_url,
        }

    try:
        result = _API["ytdlp_extract_info"](
            url=normalized_url,
            flat_playlist=flat_playlist,
            no_warnings=no_warnings,
        )
        resolved = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "url": normalized_url,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("url", normalized_url)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("info", {})
    return payload


def register_native_media_tools(manager: Any) -> None:
    """Register native media-tools category tools in unified manager."""
    manager.register_tool(
        category="media_tools",
        name="ffmpeg_analyze",
        func=ffmpeg_analyze,
        description="Analyze media file metadata with FFmpeg wrappers.",
        input_schema={
            "type": "object",
            "properties": {
                "input_file": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "object", "minProperties": 1},
                    ]
                }
            },
            "required": ["input_file"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ytdlp_extract_info",
        func=ytdlp_extract_info,
        description="Extract metadata for media URLs via yt-dlp wrappers.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "minLength": 1},
                "flat_playlist": {"type": "boolean", "default": False},
                "no_warnings": {"type": "boolean", "default": True},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )
