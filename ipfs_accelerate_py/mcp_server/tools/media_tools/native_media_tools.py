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


async def ffmpeg_analyze(input_file: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze media metadata using FFmpeg-backed wrappers."""
    result = _API["ffmpeg_analyze"](input_file=input_file)
    if hasattr(result, "__await__"):
        return await result
    return result


async def ytdlp_extract_info(
    url: str,
    flat_playlist: bool = False,
    no_warnings: bool = True,
) -> Dict[str, Any]:
    """Extract media metadata using yt-dlp-backed wrappers."""
    result = _API["ytdlp_extract_info"](
        url=url,
        flat_playlist=flat_playlist,
        no_warnings=no_warnings,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
                        {"type": "string"},
                        {"type": "object"},
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
                "url": {"type": "string"},
                "flat_playlist": {"type": "boolean"},
                "no_warnings": {"type": "boolean"},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )
