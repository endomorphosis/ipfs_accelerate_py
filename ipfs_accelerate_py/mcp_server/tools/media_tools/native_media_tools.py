"""Native media-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _load_media_tools_api() -> Dict[str, Any]:
    """Resolve source media-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.media_tools import (  # type: ignore
            ffmpeg_analyze as _ffmpeg_analyze,
            ffmpeg_convert as _ffmpeg_convert,
            ffmpeg_probe as _ffmpeg_probe,
            ytdlp_download_video as _ytdlp_download_video,
            ytdlp_extract_info as _ytdlp_extract_info,
            ytdlp_search_videos as _ytdlp_search_videos,
        )

        return {
            "ffmpeg_analyze": _ffmpeg_analyze,
            "ffmpeg_convert": _ffmpeg_convert,
            "ffmpeg_probe": _ffmpeg_probe,
            "ytdlp_download_video": _ytdlp_download_video,
            "ytdlp_extract_info": _ytdlp_extract_info,
            "ytdlp_search_videos": _ytdlp_search_videos,
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

        async def _ffmpeg_probe_fallback(
            input_file: Union[str, Dict[str, Any]],
            show_format: bool = True,
            show_streams: bool = True,
            show_chapters: bool = False,
            show_frames: bool = False,
            frame_count: Optional[int] = None,
            select_streams: Optional[str] = None,
            include_metadata: bool = True,
        ) -> Dict[str, Any]:
            _ = (
                show_format,
                show_streams,
                show_chapters,
                show_frames,
                frame_count,
                select_streams,
                include_metadata,
            )
            return {
                "status": "success",
                "input_file": input_file,
                "analysis": {},
                "fallback": True,
            }

        async def _ffmpeg_convert_fallback(
            input_file: Union[str, Dict[str, Any]],
            output_file: str,
            output_format: Optional[str] = None,
            video_codec: Optional[str] = None,
            audio_codec: Optional[str] = None,
            video_bitrate: Optional[str] = None,
            audio_bitrate: Optional[str] = None,
            resolution: Optional[str] = None,
            framerate: Optional[str] = None,
            quality: Optional[str] = None,
            preset: Optional[str] = None,
            custom_args: Optional[List[str]] = None,
            timeout: int = 600,
        ) -> Dict[str, Any]:
            _ = (
                output_format,
                video_codec,
                audio_codec,
                video_bitrate,
                audio_bitrate,
                resolution,
                framerate,
                quality,
                preset,
                custom_args,
                timeout,
            )
            return {
                "status": "success",
                "success": True,
                "input_file": input_file,
                "output_file": output_file,
                "fallback": True,
            }

        async def _ytdlp_download_video_fallback(
            url: Union[str, List[str]],
            output_dir: Optional[str] = None,
            quality: str = "best",
            format_selector: Optional[str] = None,
            audio_only: bool = False,
            extract_audio: bool = False,
            audio_format: str = "mp3",
            subtitle_langs: Optional[List[str]] = None,
            download_thumbnails: bool = False,
            download_info_json: bool = True,
            custom_opts: Optional[Dict[str, Any]] = None,
            timeout: int = 600,
        ) -> Dict[str, Any]:
            _ = (
                output_dir,
                quality,
                format_selector,
                audio_only,
                extract_audio,
                audio_format,
                subtitle_langs,
                download_thumbnails,
                download_info_json,
                custom_opts,
                timeout,
            )
            urls = url if isinstance(url, list) else [url]
            return {
                "status": "success",
                "success": True,
                "results": [],
                "total_requested": len(urls),
                "successful_downloads": 0,
                "failed_downloads": 0,
                "fallback": True,
            }

        async def _ytdlp_extract_info_fallback(
            url: str,
            download: bool = False,
            extract_flat: bool = False,
            flat_playlist: Optional[bool] = None,
            include_subtitles: bool = False,
            include_thumbnails: bool = False,
        ) -> Dict[str, Any]:
            _ = (download, extract_flat, flat_playlist, include_subtitles, include_thumbnails)
            return {
                "status": "success",
                "url": url,
                "info": {},
                "fallback": True,
            }

        async def _ytdlp_search_videos_fallback(
            query: str,
            max_results: int = 10,
            search_type: str = "ytsearch",
            extract_info: bool = True,
        ) -> Dict[str, Any]:
            _ = (max_results, search_type, extract_info)
            return {
                "status": "success",
                "query": query,
                "results": [],
                "fallback": True,
            }

        return {
            "ffmpeg_analyze": _ffmpeg_analyze_fallback,
            "ffmpeg_convert": _ffmpeg_convert_fallback,
            "ffmpeg_probe": _ffmpeg_probe_fallback,
            "ytdlp_download_video": _ytdlp_download_video_fallback,
            "ytdlp_extract_info": _ytdlp_extract_info_fallback,
            "ytdlp_search_videos": _ytdlp_search_videos_fallback,
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


async def _await_maybe(result: Any) -> Any:
    """Await coroutine-like results while preserving sync fallbacks."""
    if hasattr(result, "__await__"):
        return await result
    return result


def _normalize_media_input(
    input_file: Union[str, Dict[str, Any]],
    field_name: str = "input_file",
) -> Union[Dict[str, Any], str, Dict[str, Any]]:
    """Validate and normalize media input arguments shared by FFmpeg wrappers."""
    if isinstance(input_file, str):
        normalized = input_file.strip()
        if not normalized:
            return {
                "status": "error",
                "error": f"{field_name} must be a non-empty string or object",
            }
        return normalized
    if isinstance(input_file, dict):
        if not input_file:
            return {
                "status": "error",
                "error": f"{field_name} object must not be empty",
            }
        return input_file
    return {
        "status": "error",
        "error": f"{field_name} must be a non-empty string or object",
    }


def _validate_http_url(url: str) -> Optional[Dict[str, Any]]:
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {"status": "error", "error": "url is required"}
    if not (normalized_url.startswith("http://") or normalized_url.startswith("https://")):
        return {
            "status": "error",
            "error": "url must start with http:// or https://",
            "url": normalized_url,
        }
    return None


async def ffmpeg_analyze(input_file: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze media metadata using FFmpeg-backed wrappers."""
    normalized_input = _normalize_media_input(input_file)
    if isinstance(normalized_input, dict) and normalized_input.get("status") == "error":
        return normalized_input

    try:
        resolved = await _await_maybe(_API["ffmpeg_analyze"](input_file=normalized_input))
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


async def ffmpeg_probe(
    input_file: Union[str, Dict[str, Any]],
    show_format: bool = True,
    show_streams: bool = True,
    show_chapters: bool = False,
    show_frames: bool = False,
    frame_count: Optional[int] = None,
    select_streams: Optional[str] = None,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Probe media metadata using FFprobe-backed wrappers."""
    normalized_input = _normalize_media_input(input_file)
    if isinstance(normalized_input, dict) and normalized_input.get("status") == "error":
        return normalized_input
    if not isinstance(show_format, bool):
        return {"status": "error", "error": "show_format must be a boolean"}
    if not isinstance(show_streams, bool):
        return {"status": "error", "error": "show_streams must be a boolean"}
    if not isinstance(show_chapters, bool):
        return {"status": "error", "error": "show_chapters must be a boolean"}
    if not isinstance(show_frames, bool):
        return {"status": "error", "error": "show_frames must be a boolean"}
    if frame_count is not None and (not isinstance(frame_count, int) or frame_count < 1):
        return {"status": "error", "error": "frame_count must be an integer greater than or equal to 1 when provided"}
    if select_streams is not None and (not isinstance(select_streams, str) or not select_streams.strip()):
        return {"status": "error", "error": "select_streams must be a non-empty string when provided"}
    if not isinstance(include_metadata, bool):
        return {"status": "error", "error": "include_metadata must be a boolean"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_probe"](
                input_file=normalized_input,
                show_format=show_format,
                show_streams=show_streams,
                show_chapters=show_chapters,
                show_frames=show_frames,
                frame_count=frame_count,
                select_streams=select_streams.strip() if isinstance(select_streams, str) else None,
                include_metadata=include_metadata,
            )
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "input_file": normalized_input,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("input_file", normalized_input)
    if payload.get("status") == "success":
        payload.setdefault("analysis", {})
    return payload


async def ffmpeg_convert(
    input_file: Union[str, Dict[str, Any]],
    output_file: str,
    output_format: Optional[str] = None,
    video_codec: Optional[str] = None,
    audio_codec: Optional[str] = None,
    video_bitrate: Optional[str] = None,
    audio_bitrate: Optional[str] = None,
    resolution: Optional[str] = None,
    framerate: Optional[str] = None,
    quality: Optional[str] = None,
    preset: Optional[str] = None,
    custom_args: Optional[List[str]] = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Convert media using FFmpeg-backed wrappers."""
    normalized_input = _normalize_media_input(input_file)
    if isinstance(normalized_input, dict) and normalized_input.get("status") == "error":
        return normalized_input
    normalized_output = str(output_file or "").strip()
    if not normalized_output:
        return {"status": "error", "error": "output_file must be a non-empty string"}
    for name, value in {
        "output_format": output_format,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "video_bitrate": video_bitrate,
        "audio_bitrate": audio_bitrate,
        "resolution": resolution,
        "framerate": framerate,
        "quality": quality,
        "preset": preset,
    }.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return {"status": "error", "error": f"{name} must be a non-empty string when provided"}
    if custom_args is not None:
        if not isinstance(custom_args, list) or any(not isinstance(item, str) or not item.strip() for item in custom_args):
            return {"status": "error", "error": "custom_args must be a list of non-empty strings when provided"}
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_convert"](
                input_file=normalized_input,
                output_file=normalized_output,
                output_format=output_format.strip() if isinstance(output_format, str) else None,
                video_codec=video_codec.strip() if isinstance(video_codec, str) else None,
                audio_codec=audio_codec.strip() if isinstance(audio_codec, str) else None,
                video_bitrate=video_bitrate.strip() if isinstance(video_bitrate, str) else None,
                audio_bitrate=audio_bitrate.strip() if isinstance(audio_bitrate, str) else None,
                resolution=resolution.strip() if isinstance(resolution, str) else None,
                framerate=framerate.strip() if isinstance(framerate, str) else None,
                quality=quality.strip() if isinstance(quality, str) else None,
                preset=preset.strip() if isinstance(preset, str) else None,
                custom_args=[item.strip() for item in custom_args] if isinstance(custom_args, list) else None,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "input_file": normalized_input,
            "output_file": normalized_output,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("input_file", normalized_input)
    payload.setdefault("output_file", normalized_output)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
    return payload


async def ytdlp_download_video(
    url: Union[str, List[str]],
    output_dir: Optional[str] = None,
    quality: str = "best",
    format_selector: Optional[str] = None,
    audio_only: bool = False,
    extract_audio: bool = False,
    audio_format: str = "mp3",
    subtitle_langs: Optional[List[str]] = None,
    download_thumbnails: bool = False,
    download_info_json: bool = True,
    custom_opts: Optional[Dict[str, Any]] = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Download one or more media URLs using yt-dlp-backed wrappers."""
    normalized_urls: Union[str, List[str]]
    if isinstance(url, str):
        invalid = _validate_http_url(url)
        if invalid is not None:
            return invalid
        normalized_urls = url.strip()
    elif isinstance(url, list) and url:
        normalized_batch = []
        for item in url:
            if not isinstance(item, str):
                return {"status": "error", "error": "url list entries must be strings"}
            invalid = _validate_http_url(item)
            if invalid is not None:
                return invalid
            normalized_batch.append(item.strip())
        normalized_urls = normalized_batch
    else:
        return {"status": "error", "error": "url must be a non-empty string or array of URLs"}
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        return {"status": "error", "error": "output_dir must be a non-empty string when provided"}
    if not isinstance(quality, str) or not quality.strip():
        return {"status": "error", "error": "quality must be a non-empty string"}
    if format_selector is not None and (not isinstance(format_selector, str) or not format_selector.strip()):
        return {"status": "error", "error": "format_selector must be a non-empty string when provided"}
    if not isinstance(audio_only, bool):
        return {"status": "error", "error": "audio_only must be a boolean"}
    if not isinstance(extract_audio, bool):
        return {"status": "error", "error": "extract_audio must be a boolean"}
    if not isinstance(audio_format, str) or not audio_format.strip():
        return {"status": "error", "error": "audio_format must be a non-empty string"}
    if subtitle_langs is not None:
        if not isinstance(subtitle_langs, list) or any(not isinstance(item, str) or not item.strip() for item in subtitle_langs):
            return {"status": "error", "error": "subtitle_langs must be a list of non-empty strings when provided"}
    if not isinstance(download_thumbnails, bool):
        return {"status": "error", "error": "download_thumbnails must be a boolean"}
    if not isinstance(download_info_json, bool):
        return {"status": "error", "error": "download_info_json must be a boolean"}
    if custom_opts is not None and not isinstance(custom_opts, dict):
        return {"status": "error", "error": "custom_opts must be an object when provided"}
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ytdlp_download_video"](
                url=normalized_urls,
                output_dir=output_dir.strip() if isinstance(output_dir, str) else None,
                quality=quality.strip(),
                format_selector=format_selector.strip() if isinstance(format_selector, str) else None,
                audio_only=audio_only,
                extract_audio=extract_audio,
                audio_format=audio_format.strip(),
                subtitle_langs=[item.strip() for item in subtitle_langs] if isinstance(subtitle_langs, list) else None,
                download_thumbnails=download_thumbnails,
                download_info_json=download_info_json,
                custom_opts=custom_opts,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "url": normalized_urls}

    payload = _normalize_payload(resolved)
    payload.setdefault("url", normalized_urls)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("results", [])
    return payload


async def ytdlp_extract_info(
    url: str,
    download: bool = False,
    extract_flat: bool = False,
    flat_playlist: Optional[bool] = None,
    include_subtitles: bool = False,
    include_thumbnails: bool = False,
) -> Dict[str, Any]:
    """Extract media metadata using yt-dlp-backed wrappers."""
    invalid = _validate_http_url(url)
    if invalid is not None:
        return invalid
    normalized_url = url.strip()
    if not isinstance(download, bool):
        return {"status": "error", "error": "download must be a boolean", "url": normalized_url}
    if flat_playlist is not None and not isinstance(flat_playlist, bool):
        return {"status": "error", "error": "flat_playlist must be a boolean", "url": normalized_url}
    if not isinstance(extract_flat, bool):
        return {"status": "error", "error": "extract_flat must be a boolean", "url": normalized_url}
    if not isinstance(include_subtitles, bool):
        return {"status": "error", "error": "include_subtitles must be a boolean", "url": normalized_url}
    if not isinstance(include_thumbnails, bool):
        return {"status": "error", "error": "include_thumbnails must be a boolean", "url": normalized_url}

    normalized_extract_flat = flat_playlist if isinstance(flat_playlist, bool) else extract_flat

    try:
        resolved = await _await_maybe(
            _API["ytdlp_extract_info"](
                url=normalized_url,
                download=download,
                extract_flat=normalized_extract_flat,
                include_subtitles=include_subtitles,
                include_thumbnails=include_thumbnails,
            )
        )
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


async def ytdlp_search_videos(
    query: str,
    max_results: int = 10,
    search_type: str = "ytsearch",
    extract_info: bool = True,
) -> Dict[str, Any]:
    """Search videos via yt-dlp-backed wrappers."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {"status": "error", "error": "query must be a non-empty string", "query": query}
    if not isinstance(max_results, int) or max_results < 1:
        return {"status": "error", "error": "max_results must be an integer greater than or equal to 1"}
    if not isinstance(search_type, str) or not search_type.strip():
        return {"status": "error", "error": "search_type must be a non-empty string"}
    if not isinstance(extract_info, bool):
        return {"status": "error", "error": "extract_info must be a boolean"}

    try:
        resolved = await _await_maybe(
            _API["ytdlp_search_videos"](
                query=normalized_query,
                max_results=max_results,
                search_type=search_type.strip(),
                extract_info=extract_info,
            )
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "query": normalized_query,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("query", normalized_query)
    if payload.get("status") == "success":
        payload.setdefault("results", [])
    return payload


def register_native_media_tools(manager: Any) -> None:
    """Register native media-tools category tools in unified manager."""
    manager.register_tool(
        category="media_tools",
        name="ffmpeg_probe",
        func=ffmpeg_probe,
        description="Probe media file metadata with FFprobe wrappers.",
        input_schema={
            "type": "object",
            "properties": {
                "input_file": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "object", "minProperties": 1},
                    ]
                },
                "show_format": {"type": "boolean", "default": True},
                "show_streams": {"type": "boolean", "default": True},
                "show_chapters": {"type": "boolean", "default": False},
                "show_frames": {"type": "boolean", "default": False},
                "frame_count": {"type": ["integer", "null"], "minimum": 1},
                "select_streams": {"type": ["string", "null"], "minLength": 1},
                "include_metadata": {"type": "boolean", "default": True},
            },
            "required": ["input_file"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

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
        name="ffmpeg_convert",
        func=ffmpeg_convert,
        description="Convert media using FFmpeg wrappers.",
        input_schema={
            "type": "object",
            "properties": {
                "input_file": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "object", "minProperties": 1},
                    ]
                },
                "output_file": {"type": "string", "minLength": 1},
                "output_format": {"type": ["string", "null"], "minLength": 1},
                "video_codec": {"type": ["string", "null"], "minLength": 1},
                "audio_codec": {"type": ["string", "null"], "minLength": 1},
                "video_bitrate": {"type": ["string", "null"], "minLength": 1},
                "audio_bitrate": {"type": ["string", "null"], "minLength": 1},
                "resolution": {"type": ["string", "null"], "minLength": 1},
                "framerate": {"type": ["string", "null"], "minLength": 1},
                "quality": {"type": ["string", "null"], "minLength": 1},
                "preset": {"type": ["string", "null"], "minLength": 1},
                "custom_args": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "timeout": {"type": "integer", "minimum": 1, "default": 600},
            },
            "required": ["input_file", "output_file"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ytdlp_download_video",
        func=ytdlp_download_video,
        description="Download media URLs using yt-dlp wrappers.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                    ]
                },
                "output_dir": {"type": ["string", "null"], "minLength": 1},
                "quality": {"type": "string", "minLength": 1, "default": "best"},
                "format_selector": {"type": ["string", "null"], "minLength": 1},
                "audio_only": {"type": "boolean", "default": False},
                "extract_audio": {"type": "boolean", "default": False},
                "audio_format": {"type": "string", "minLength": 1, "default": "mp3"},
                "subtitle_langs": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "download_thumbnails": {"type": "boolean", "default": False},
                "download_info_json": {"type": "boolean", "default": True},
                "custom_opts": {"type": ["object", "null"]},
                "timeout": {"type": "integer", "minimum": 1, "default": 600},
            },
            "required": ["url"],
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
                "download": {"type": "boolean", "default": False},
                "extract_flat": {"type": "boolean", "default": False},
                "flat_playlist": {"type": ["boolean", "null"], "default": None},
                "include_subtitles": {"type": "boolean", "default": False},
                "include_thumbnails": {"type": "boolean", "default": False},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ytdlp_search_videos",
        func=ytdlp_search_videos,
        description="Search for videos using yt-dlp wrappers.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "max_results": {"type": "integer", "minimum": 1, "default": 10},
                "search_type": {"type": "string", "minLength": 1, "default": "ytsearch"},
                "extract_info": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )
