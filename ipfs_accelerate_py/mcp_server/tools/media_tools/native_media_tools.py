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
            ffmpeg_apply_filters as _ffmpeg_apply_filters,
            ffmpeg_batch_process as _ffmpeg_batch_process,
            ffmpeg_concat as _ffmpeg_concat,
            ffmpeg_convert as _ffmpeg_convert,
            ffmpeg_cut as _ffmpeg_cut,
            ffmpeg_demux as _ffmpeg_demux,
            ffmpeg_mux as _ffmpeg_mux,
            ffmpeg_probe as _ffmpeg_probe,
            ffmpeg_splice as _ffmpeg_splice,
            ffmpeg_stream_input as _ffmpeg_stream_input,
            ffmpeg_stream_output as _ffmpeg_stream_output,
            ytdlp_batch_download as _ytdlp_batch_download,
            ytdlp_download_playlist as _ytdlp_download_playlist,
            ytdlp_download_video as _ytdlp_download_video,
            ytdlp_extract_info as _ytdlp_extract_info,
            ytdlp_search_videos as _ytdlp_search_videos,
        )

        return {
            "ffmpeg_analyze": _ffmpeg_analyze,
            "ffmpeg_apply_filters": _ffmpeg_apply_filters,
            "ffmpeg_batch_process": _ffmpeg_batch_process,
            "ffmpeg_concat": _ffmpeg_concat,
            "ffmpeg_convert": _ffmpeg_convert,
            "ffmpeg_cut": _ffmpeg_cut,
            "ffmpeg_demux": _ffmpeg_demux,
            "ffmpeg_mux": _ffmpeg_mux,
            "ffmpeg_probe": _ffmpeg_probe,
            "ffmpeg_splice": _ffmpeg_splice,
            "ffmpeg_stream_input": _ffmpeg_stream_input,
            "ffmpeg_stream_output": _ffmpeg_stream_output,
            "ytdlp_batch_download": _ytdlp_batch_download,
            "ytdlp_download_playlist": _ytdlp_download_playlist,
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

        async def _ffmpeg_mux_fallback(
            video_input: Optional[str] = None,
            audio_inputs: Optional[List[str]] = None,
            subtitle_inputs: Optional[List[str]] = None,
            output_file: str = "",
            output_format: Optional[str] = None,
            video_codec: str = "copy",
            audio_codec: str = "copy",
            subtitle_codec: str = "copy",
            map_streams: Optional[List[str]] = None,
            metadata: Optional[Dict[str, str]] = None,
            timeout: int = 300,
        ) -> Dict[str, Any]:
            _ = (
                output_format,
                video_codec,
                audio_codec,
                subtitle_codec,
                map_streams,
                metadata,
                timeout,
            )
            return {
                "status": "success",
                "inputs": {
                    "video": video_input,
                    "audio": audio_inputs or [],
                    "subtitle": subtitle_inputs or [],
                },
                "output_file": output_file,
                "fallback": True,
            }

        async def _ffmpeg_demux_fallback(
            input_file: Union[str, Dict[str, Any]],
            output_dir: str,
            extract_video: bool = True,
            extract_audio: bool = True,
            extract_subtitles: bool = True,
            video_format: str = "mp4",
            audio_format: str = "mp3",
            subtitle_format: str = "srt",
            stream_selection: Optional[Dict[str, List[int]]] = None,
            timeout: int = 300,
        ) -> Dict[str, Any]:
            _ = (
                extract_video,
                extract_audio,
                extract_subtitles,
                video_format,
                audio_format,
                subtitle_format,
                stream_selection,
                timeout,
            )
            return {
                "status": "success",
                "input_file": input_file,
                "output_dir": output_dir,
                "extracted_files": [],
                "fallback": True,
            }

        async def _ffmpeg_stream_input_fallback(
            stream_url: str,
            output_file: str,
            duration: Optional[str] = None,
            video_codec: str = "copy",
            audio_codec: str = "copy",
            format: Optional[str] = None,
            buffer_size: Optional[str] = None,
            timeout: int = 3600,
        ) -> Dict[str, Any]:
            _ = (duration, video_codec, audio_codec, format, buffer_size, timeout)
            return {
                "status": "success",
                "stream_url": stream_url,
                "output_file": output_file,
                "fallback": True,
            }

        async def _ffmpeg_stream_output_fallback(
            input_file: Union[str, Dict[str, Any]],
            stream_url: str,
            video_codec: str = "libx264",
            audio_codec: str = "aac",
            video_bitrate: Optional[str] = None,
            audio_bitrate: Optional[str] = None,
            resolution: Optional[str] = None,
            framerate: Optional[str] = None,
            format: str = "flv",
            preset: str = "fast",
            tune: Optional[str] = None,
            keyframe_interval: Optional[str] = None,
            buffer_size: Optional[str] = None,
            max_muxing_queue_size: str = "1024",
            timeout: int = 3600,
        ) -> Dict[str, Any]:
            _ = (
                video_codec,
                audio_codec,
                video_bitrate,
                audio_bitrate,
                resolution,
                framerate,
                format,
                preset,
                tune,
                keyframe_interval,
                buffer_size,
                max_muxing_queue_size,
                timeout,
            )
            return {
                "status": "success",
                "input_file": input_file,
                "stream_url": stream_url,
                "fallback": True,
            }

        async def _ffmpeg_cut_fallback(
            input_file: Union[str, Dict[str, Any]],
            output_file: str,
            start_time: str,
            end_time: Optional[str] = None,
            duration: Optional[str] = None,
            video_codec: str = "copy",
            audio_codec: str = "copy",
            accurate_seek: bool = True,
            timeout: int = 300,
        ) -> Dict[str, Any]:
            _ = (end_time, duration, video_codec, audio_codec, accurate_seek, timeout)
            return {
                "status": "success",
                "input_file": input_file,
                "output_file": output_file,
                "segment_info": {"start_time": start_time},
                "fallback": True,
            }

        async def _ffmpeg_splice_fallback(
            input_files: List[Union[str, Dict[str, Any]]],
            output_file: str,
            segments: List[Dict[str, Any]],
            video_codec: str = "libx264",
            audio_codec: str = "aac",
            transition_type: str = "cut",
            transition_duration: float = 0.0,
            timeout: int = 600,
        ) -> Dict[str, Any]:
            _ = (video_codec, audio_codec, transition_type, transition_duration, timeout)
            return {
                "status": "success",
                "input_files": input_files,
                "output_file": output_file,
                "segments_processed": len(segments),
                "fallback": True,
            }

        async def _ffmpeg_concat_fallback(
            input_files: List[Union[str, Dict[str, Any]]],
            output_file: str,
            video_codec: str = "copy",
            audio_codec: str = "copy",
            method: str = "filter",
            safe_mode: bool = True,
            timeout: int = 600,
        ) -> Dict[str, Any]:
            _ = (video_codec, audio_codec, method, safe_mode, timeout)
            return {
                "status": "success",
                "input_files": input_files,
                "output_file": output_file,
                "fallback": True,
            }

        async def _ffmpeg_apply_filters_fallback(
            input_file: Union[str, Dict[str, Any]],
            output_file: str,
            video_filters: Optional[List[str]] = None,
            audio_filters: Optional[List[str]] = None,
            filter_complex: Optional[str] = None,
            output_format: Optional[str] = None,
            preserve_metadata: bool = True,
            timeout: int = 600,
        ) -> Dict[str, Any]:
            _ = (
                video_filters,
                audio_filters,
                filter_complex,
                output_format,
                preserve_metadata,
                timeout,
            )
            return {
                "status": "success",
                "input_file": input_file,
                "output_file": output_file,
                "filters_applied": [],
                "fallback": True,
            }

        async def _ffmpeg_batch_process_fallback(
            input_files: Union[List[str], Dict[str, Any]],
            output_directory: str,
            operation: str = "convert",
            operation_params: Optional[Dict[str, Any]] = None,
            max_parallel: int = 2,
            save_progress: bool = True,
            resume_from_checkpoint: bool = True,
            timeout_per_file: int = 600,
        ) -> Dict[str, Any]:
            _ = (output_directory, operation, operation_params, max_parallel, save_progress, resume_from_checkpoint, timeout_per_file)
            total = len(input_files) if isinstance(input_files, list) else 0
            return {
                "status": "success",
                "total_files": total,
                "processed_files": total,
                "failed_files": 0,
                "results": [],
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

        async def _ytdlp_download_playlist_fallback(
            playlist_url: str,
            output_dir: Optional[str] = None,
            quality: str = "best",
            max_downloads: Optional[int] = None,
            start_index: int = 1,
            end_index: Optional[int] = None,
            download_archive: Optional[str] = None,
            custom_opts: Optional[Dict[str, Any]] = None,
            timeout: int = 1200,
        ) -> Dict[str, Any]:
            _ = (output_dir, quality, max_downloads, start_index, end_index, download_archive, custom_opts, timeout)
            return {
                "status": "success",
                "playlist_url": playlist_url,
                "results": [],
                "fallback": True,
            }

        async def _ytdlp_extract_info_fallback(
            url: str,
            download: bool = False,
            extract_flat: bool = False,
            include_subtitles: bool = False,
            include_thumbnails: bool = False,
        ) -> Dict[str, Any]:
            _ = (download, extract_flat, include_subtitles, include_thumbnails)
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

        async def _ytdlp_batch_download_fallback(
            urls: List[str],
            output_dir: Optional[str] = None,
            quality: str = "best",
            concurrent_downloads: int = 3,
            ignore_errors: bool = True,
            custom_opts: Optional[Dict[str, Any]] = None,
            timeout: int = 1800,
        ) -> Dict[str, Any]:
            _ = (output_dir, quality, concurrent_downloads, ignore_errors, custom_opts, timeout)
            return {
                "status": "success",
                "urls": urls,
                "results": [],
                "fallback": True,
            }

        return {
            "ffmpeg_analyze": _ffmpeg_analyze_fallback,
            "ffmpeg_apply_filters": _ffmpeg_apply_filters_fallback,
            "ffmpeg_batch_process": _ffmpeg_batch_process_fallback,
            "ffmpeg_concat": _ffmpeg_concat_fallback,
            "ffmpeg_convert": _ffmpeg_convert_fallback,
            "ffmpeg_cut": _ffmpeg_cut_fallback,
            "ffmpeg_demux": _ffmpeg_demux_fallback,
            "ffmpeg_mux": _ffmpeg_mux_fallback,
            "ffmpeg_probe": _ffmpeg_probe_fallback,
            "ffmpeg_splice": _ffmpeg_splice_fallback,
            "ffmpeg_stream_input": _ffmpeg_stream_input_fallback,
            "ffmpeg_stream_output": _ffmpeg_stream_output_fallback,
            "ytdlp_batch_download": _ytdlp_batch_download_fallback,
            "ytdlp_download_playlist": _ytdlp_download_playlist_fallback,
            "ytdlp_download_video": _ytdlp_download_video_fallback,
            "ytdlp_extract_info": _ytdlp_extract_info_fallback,
            "ytdlp_search_videos": _ytdlp_search_videos_fallback,
        }


_API = _load_media_tools_api()


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend output into deterministic status envelopes."""
    if isinstance(result, dict):
        payload = dict(result)
        failed = payload.get("success") is False or bool(payload.get("error"))
        if failed:
            payload["status"] = "error"
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


def _normalize_non_empty_string(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
) -> Union[str, None, Dict[str, Any]]:
    if value is None:
        if required:
            return {"status": "error", "error": f"{field_name} is required"}
        return None
    if not isinstance(value, str) or not value.strip():
        return {
            "status": "error",
            "error": f"{field_name} must be a non-empty string"
            + ("" if required else " when provided"),
        }
    return value.strip()


def _normalize_string_list(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
) -> Union[List[str], None, Dict[str, Any]]:
    if value is None:
        if required:
            return {"status": "error", "error": f"{field_name} is required"}
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        return {
            "status": "error",
            "error": f"{field_name} must be a list of non-empty strings"
            + ("" if required else " when provided"),
        }
    if required and not value:
        return {"status": "error", "error": f"{field_name} must contain at least one entry"}
    return [item.strip() for item in value]


def _normalize_string_mapping(value: Any, field_name: str) -> Union[Dict[str, str], None, Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        return {"status": "error", "error": f"{field_name} must be a non-empty object when provided"}
    normalized: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip():
            return {
                "status": "error",
                "error": f"{field_name} keys and values must be non-empty strings",
            }
        normalized[key.strip()] = item.strip()
    return normalized


def _normalize_stream_selection(value: Any) -> Union[Dict[str, List[int]], None, Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        return {"status": "error", "error": "stream_selection must be a non-empty object when provided"}
    normalized: Dict[str, List[int]] = {}
    for key, item in value.items():
        if key not in {"video", "audio", "subtitle"}:
            return {"status": "error", "error": "stream_selection keys must be one of video, audio, subtitle"}
        if not isinstance(item, list) or any(not isinstance(index, int) or index < 0 for index in item):
            return {
                "status": "error",
                "error": "stream_selection values must be arrays of non-negative integers",
            }
        normalized[key] = item
    return normalized


def _normalize_media_inputs(
    input_files: Any,
    field_name: str,
) -> Union[List[Union[str, Dict[str, Any]]], Dict[str, Any]]:
    if not isinstance(input_files, list) or not input_files:
        return {"status": "error", "error": f"{field_name} must be a non-empty list"}
    normalized_inputs: List[Union[str, Dict[str, Any]]] = []
    for item in input_files:
        normalized = _normalize_media_input(item, field_name=field_name)
        if isinstance(normalized, dict) and normalized.get("status") == "error":
            return normalized
        normalized_inputs.append(normalized)
    return normalized_inputs


def _normalize_segments(value: Any) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        return {"status": "error", "error": "segments must be a non-empty list of objects"}
    normalized_segments: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or not item:
            return {"status": "error", "error": "segments must be a non-empty list of objects"}
        normalized_segments.append(dict(item))
    return normalized_segments


def _is_error_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") == "error"


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


async def ffmpeg_mux(
    video_input: Optional[str] = None,
    audio_inputs: Optional[List[str]] = None,
    subtitle_inputs: Optional[List[str]] = None,
    output_file: str = "",
    output_format: Optional[str] = None,
    video_codec: str = "copy",
    audio_codec: str = "copy",
    subtitle_codec: str = "copy",
    map_streams: Optional[List[str]] = None,
    metadata: Optional[Dict[str, str]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Mux separate media streams into a single output container."""
    normalized_video: Optional[str] = None
    if video_input is not None:
        normalized_video_value = _normalize_non_empty_string(video_input, "video_input")
        if _is_error_payload(normalized_video_value):
            return normalized_video_value
        normalized_video = normalized_video_value
    normalized_audio = _normalize_string_list(audio_inputs, "audio_inputs")
    if _is_error_payload(normalized_audio):
        return normalized_audio
    normalized_subtitles = _normalize_string_list(subtitle_inputs, "subtitle_inputs")
    if _is_error_payload(normalized_subtitles):
        return normalized_subtitles
    if not normalized_video and not normalized_audio and not normalized_subtitles:
        return {"status": "error", "error": "At least one input stream must be provided"}

    normalized_output = _normalize_non_empty_string(output_file, "output_file", required=True)
    if _is_error_payload(normalized_output):
        return normalized_output
    normalized_output_format = _normalize_non_empty_string(output_format, "output_format")
    if _is_error_payload(normalized_output_format):
        return normalized_output_format
    normalized_video_codec = _normalize_non_empty_string(video_codec, "video_codec", required=True)
    if _is_error_payload(normalized_video_codec):
        return normalized_video_codec
    normalized_audio_codec = _normalize_non_empty_string(audio_codec, "audio_codec", required=True)
    if _is_error_payload(normalized_audio_codec):
        return normalized_audio_codec
    normalized_subtitle_codec = _normalize_non_empty_string(subtitle_codec, "subtitle_codec", required=True)
    if _is_error_payload(normalized_subtitle_codec):
        return normalized_subtitle_codec
    normalized_map_streams = _normalize_string_list(map_streams, "map_streams")
    if _is_error_payload(normalized_map_streams):
        return normalized_map_streams
    normalized_metadata = _normalize_string_mapping(metadata, "metadata")
    if _is_error_payload(normalized_metadata):
        return normalized_metadata
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_mux"](
                video_input=normalized_video,
                audio_inputs=normalized_audio,
                subtitle_inputs=normalized_subtitles,
                output_file=normalized_output,
                output_format=normalized_output_format,
                video_codec=normalized_video_codec,
                audio_codec=normalized_audio_codec,
                subtitle_codec=normalized_subtitle_codec,
                map_streams=normalized_map_streams,
                metadata=normalized_metadata,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "output_file": normalized_output}

    payload = _normalize_payload(resolved)
    payload.setdefault(
        "inputs",
        {"video": normalized_video, "audio": normalized_audio or [], "subtitle": normalized_subtitles or []},
    )
    payload.setdefault("output_file", normalized_output)
    return payload


async def ffmpeg_demux(
    input_file: Union[str, Dict[str, Any]],
    output_dir: str,
    extract_video: bool = True,
    extract_audio: bool = True,
    extract_subtitles: bool = True,
    video_format: str = "mp4",
    audio_format: str = "mp3",
    subtitle_format: str = "srt",
    stream_selection: Optional[Dict[str, List[int]]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Demux media streams into separate output files."""
    normalized_input = _normalize_media_input(input_file)
    if _is_error_payload(normalized_input):
        return normalized_input
    normalized_output_dir = _normalize_non_empty_string(output_dir, "output_dir", required=True)
    if _is_error_payload(normalized_output_dir):
        return normalized_output_dir
    if not isinstance(extract_video, bool):
        return {"status": "error", "error": "extract_video must be a boolean"}
    if not isinstance(extract_audio, bool):
        return {"status": "error", "error": "extract_audio must be a boolean"}
    if not isinstance(extract_subtitles, bool):
        return {"status": "error", "error": "extract_subtitles must be a boolean"}
    normalized_video_format = _normalize_non_empty_string(video_format, "video_format", required=True)
    if _is_error_payload(normalized_video_format):
        return normalized_video_format
    normalized_audio_format = _normalize_non_empty_string(audio_format, "audio_format", required=True)
    if _is_error_payload(normalized_audio_format):
        return normalized_audio_format
    normalized_subtitle_format = _normalize_non_empty_string(subtitle_format, "subtitle_format", required=True)
    if _is_error_payload(normalized_subtitle_format):
        return normalized_subtitle_format
    normalized_stream_selection = _normalize_stream_selection(stream_selection)
    if _is_error_payload(normalized_stream_selection):
        return normalized_stream_selection
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_demux"](
                input_file=normalized_input,
                output_dir=normalized_output_dir,
                extract_video=extract_video,
                extract_audio=extract_audio,
                extract_subtitles=extract_subtitles,
                video_format=normalized_video_format,
                audio_format=normalized_audio_format,
                subtitle_format=normalized_subtitle_format,
                stream_selection=normalized_stream_selection,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "input_file": normalized_input}

    payload = _normalize_payload(resolved)
    payload.setdefault("input_file", normalized_input)
    payload.setdefault("output_dir", normalized_output_dir)
    payload.setdefault("extracted_files", [])
    return payload


async def ffmpeg_stream_input(
    stream_url: str,
    output_file: str,
    duration: Optional[str] = None,
    video_codec: str = "copy",
    audio_codec: str = "copy",
    format: Optional[str] = None,
    buffer_size: Optional[str] = None,
    timeout: int = 3600,
) -> Dict[str, Any]:
    """Capture media from an incoming stream."""
    normalized_stream_url = _normalize_non_empty_string(stream_url, "stream_url", required=True)
    if _is_error_payload(normalized_stream_url):
        return normalized_stream_url
    normalized_output = _normalize_non_empty_string(output_file, "output_file", required=True)
    if _is_error_payload(normalized_output):
        return normalized_output
    normalized_duration = _normalize_non_empty_string(duration, "duration")
    if _is_error_payload(normalized_duration):
        return normalized_duration
    normalized_video_codec = _normalize_non_empty_string(video_codec, "video_codec", required=True)
    if _is_error_payload(normalized_video_codec):
        return normalized_video_codec
    normalized_audio_codec = _normalize_non_empty_string(audio_codec, "audio_codec", required=True)
    if _is_error_payload(normalized_audio_codec):
        return normalized_audio_codec
    normalized_format = _normalize_non_empty_string(format, "format")
    if _is_error_payload(normalized_format):
        return normalized_format
    normalized_buffer_size = _normalize_non_empty_string(buffer_size, "buffer_size")
    if _is_error_payload(normalized_buffer_size):
        return normalized_buffer_size
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_stream_input"](
                stream_url=normalized_stream_url,
                output_file=normalized_output,
                duration=normalized_duration,
                video_codec=normalized_video_codec,
                audio_codec=normalized_audio_codec,
                format=normalized_format,
                buffer_size=normalized_buffer_size,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "stream_url": normalized_stream_url}

    payload = _normalize_payload(resolved)
    payload.setdefault("stream_url", normalized_stream_url)
    payload.setdefault("output_file", normalized_output)
    return payload


async def ffmpeg_stream_output(
    input_file: Union[str, Dict[str, Any]],
    stream_url: str,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    video_bitrate: Optional[str] = None,
    audio_bitrate: Optional[str] = None,
    resolution: Optional[str] = None,
    framerate: Optional[str] = None,
    format: str = "flv",
    preset: str = "fast",
    tune: Optional[str] = None,
    keyframe_interval: Optional[str] = None,
    buffer_size: Optional[str] = None,
    max_muxing_queue_size: str = "1024",
    timeout: int = 3600,
) -> Dict[str, Any]:
    """Publish media from a file to an outgoing stream."""
    normalized_input = _normalize_media_input(input_file)
    if _is_error_payload(normalized_input):
        return normalized_input
    normalized_stream_url = _normalize_non_empty_string(stream_url, "stream_url", required=True)
    if _is_error_payload(normalized_stream_url):
        return normalized_stream_url
    normalized_video_codec = _normalize_non_empty_string(video_codec, "video_codec", required=True)
    if _is_error_payload(normalized_video_codec):
        return normalized_video_codec
    normalized_audio_codec = _normalize_non_empty_string(audio_codec, "audio_codec", required=True)
    if _is_error_payload(normalized_audio_codec):
        return normalized_audio_codec
    normalized_video_bitrate = _normalize_non_empty_string(video_bitrate, "video_bitrate")
    if _is_error_payload(normalized_video_bitrate):
        return normalized_video_bitrate
    normalized_audio_bitrate = _normalize_non_empty_string(audio_bitrate, "audio_bitrate")
    if _is_error_payload(normalized_audio_bitrate):
        return normalized_audio_bitrate
    normalized_resolution = _normalize_non_empty_string(resolution, "resolution")
    if _is_error_payload(normalized_resolution):
        return normalized_resolution
    normalized_framerate = _normalize_non_empty_string(framerate, "framerate")
    if _is_error_payload(normalized_framerate):
        return normalized_framerate
    normalized_format = _normalize_non_empty_string(format, "format", required=True)
    if _is_error_payload(normalized_format):
        return normalized_format
    normalized_preset = _normalize_non_empty_string(preset, "preset", required=True)
    if _is_error_payload(normalized_preset):
        return normalized_preset
    normalized_tune = _normalize_non_empty_string(tune, "tune")
    if _is_error_payload(normalized_tune):
        return normalized_tune
    normalized_keyframe_interval = _normalize_non_empty_string(keyframe_interval, "keyframe_interval")
    if _is_error_payload(normalized_keyframe_interval):
        return normalized_keyframe_interval
    normalized_buffer_size = _normalize_non_empty_string(buffer_size, "buffer_size")
    if _is_error_payload(normalized_buffer_size):
        return normalized_buffer_size
    normalized_max_muxing_queue_size = _normalize_non_empty_string(
        max_muxing_queue_size,
        "max_muxing_queue_size",
        required=True,
    )
    if _is_error_payload(normalized_max_muxing_queue_size):
        return normalized_max_muxing_queue_size
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_stream_output"](
                input_file=normalized_input,
                stream_url=normalized_stream_url,
                video_codec=normalized_video_codec,
                audio_codec=normalized_audio_codec,
                video_bitrate=normalized_video_bitrate,
                audio_bitrate=normalized_audio_bitrate,
                resolution=normalized_resolution,
                framerate=normalized_framerate,
                format=normalized_format,
                preset=normalized_preset,
                tune=normalized_tune,
                keyframe_interval=normalized_keyframe_interval,
                buffer_size=normalized_buffer_size,
                max_muxing_queue_size=normalized_max_muxing_queue_size,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "input_file": normalized_input}

    payload = _normalize_payload(resolved)
    payload.setdefault("input_file", normalized_input)
    payload.setdefault("stream_url", normalized_stream_url)
    return payload


async def ffmpeg_cut(
    input_file: Union[str, Dict[str, Any]],
    output_file: str,
    start_time: str,
    end_time: Optional[str] = None,
    duration: Optional[str] = None,
    video_codec: str = "copy",
    audio_codec: str = "copy",
    accurate_seek: bool = True,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Cut a segment from a media file."""
    normalized_input = _normalize_media_input(input_file)
    if _is_error_payload(normalized_input):
        return normalized_input
    normalized_output = _normalize_non_empty_string(output_file, "output_file", required=True)
    if _is_error_payload(normalized_output):
        return normalized_output
    normalized_start_time = _normalize_non_empty_string(start_time, "start_time", required=True)
    if _is_error_payload(normalized_start_time):
        return normalized_start_time
    normalized_end_time = _normalize_non_empty_string(end_time, "end_time")
    if _is_error_payload(normalized_end_time):
        return normalized_end_time
    normalized_duration = _normalize_non_empty_string(duration, "duration")
    if _is_error_payload(normalized_duration):
        return normalized_duration
    if bool(normalized_end_time) == bool(normalized_duration):
        return {"status": "error", "error": "Specify exactly one of end_time or duration"}
    normalized_video_codec = _normalize_non_empty_string(video_codec, "video_codec", required=True)
    if _is_error_payload(normalized_video_codec):
        return normalized_video_codec
    normalized_audio_codec = _normalize_non_empty_string(audio_codec, "audio_codec", required=True)
    if _is_error_payload(normalized_audio_codec):
        return normalized_audio_codec
    if not isinstance(accurate_seek, bool):
        return {"status": "error", "error": "accurate_seek must be a boolean"}
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_cut"](
                input_file=normalized_input,
                output_file=normalized_output,
                start_time=normalized_start_time,
                end_time=normalized_end_time,
                duration=normalized_duration,
                video_codec=normalized_video_codec,
                audio_codec=normalized_audio_codec,
                accurate_seek=accurate_seek,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "input_file": normalized_input}

    payload = _normalize_payload(resolved)
    payload.setdefault("input_file", normalized_input)
    payload.setdefault("output_file", normalized_output)
    return payload


async def ffmpeg_splice(
    input_files: List[Union[str, Dict[str, Any]]],
    output_file: str,
    segments: List[Dict[str, Any]],
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    transition_type: str = "cut",
    transition_duration: float = 0.0,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Splice multiple media segments into one output file."""
    normalized_inputs = _normalize_media_inputs(input_files, "input_files")
    if _is_error_payload(normalized_inputs):
        return normalized_inputs
    normalized_output = _normalize_non_empty_string(output_file, "output_file", required=True)
    if _is_error_payload(normalized_output):
        return normalized_output
    normalized_segments = _normalize_segments(segments)
    if _is_error_payload(normalized_segments):
        return normalized_segments
    normalized_video_codec = _normalize_non_empty_string(video_codec, "video_codec", required=True)
    if _is_error_payload(normalized_video_codec):
        return normalized_video_codec
    normalized_audio_codec = _normalize_non_empty_string(audio_codec, "audio_codec", required=True)
    if _is_error_payload(normalized_audio_codec):
        return normalized_audio_codec
    normalized_transition_type = _normalize_non_empty_string(transition_type, "transition_type", required=True)
    if _is_error_payload(normalized_transition_type):
        return normalized_transition_type
    if not isinstance(transition_duration, (int, float)) or transition_duration < 0:
        return {"status": "error", "error": "transition_duration must be a non-negative number"}
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_splice"](
                input_files=normalized_inputs,
                output_file=normalized_output,
                segments=normalized_segments,
                video_codec=normalized_video_codec,
                audio_codec=normalized_audio_codec,
                transition_type=normalized_transition_type,
                transition_duration=float(transition_duration),
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "output_file": normalized_output}

    payload = _normalize_payload(resolved)
    payload.setdefault("input_files", normalized_inputs)
    payload.setdefault("output_file", normalized_output)
    return payload


async def ffmpeg_concat(
    input_files: List[Union[str, Dict[str, Any]]],
    output_file: str,
    video_codec: str = "copy",
    audio_codec: str = "copy",
    method: str = "filter",
    safe_mode: bool = True,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Concatenate multiple media files."""
    normalized_inputs = _normalize_media_inputs(input_files, "input_files")
    if _is_error_payload(normalized_inputs):
        return normalized_inputs
    normalized_output = _normalize_non_empty_string(output_file, "output_file", required=True)
    if _is_error_payload(normalized_output):
        return normalized_output
    normalized_video_codec = _normalize_non_empty_string(video_codec, "video_codec", required=True)
    if _is_error_payload(normalized_video_codec):
        return normalized_video_codec
    normalized_audio_codec = _normalize_non_empty_string(audio_codec, "audio_codec", required=True)
    if _is_error_payload(normalized_audio_codec):
        return normalized_audio_codec
    normalized_method = _normalize_non_empty_string(method, "method", required=True)
    if _is_error_payload(normalized_method):
        return normalized_method
    if not isinstance(safe_mode, bool):
        return {"status": "error", "error": "safe_mode must be a boolean"}
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_concat"](
                input_files=normalized_inputs,
                output_file=normalized_output,
                video_codec=normalized_video_codec,
                audio_codec=normalized_audio_codec,
                method=normalized_method,
                safe_mode=safe_mode,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "output_file": normalized_output}

    payload = _normalize_payload(resolved)
    payload.setdefault("input_files", normalized_inputs)
    payload.setdefault("output_file", normalized_output)
    return payload


async def ffmpeg_apply_filters(
    input_file: Union[str, Dict[str, Any]],
    output_file: str,
    video_filters: Optional[List[str]] = None,
    audio_filters: Optional[List[str]] = None,
    filter_complex: Optional[str] = None,
    output_format: Optional[str] = None,
    preserve_metadata: bool = True,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Apply audio and video filters to a media file."""
    normalized_input = _normalize_media_input(input_file)
    if _is_error_payload(normalized_input):
        return normalized_input
    normalized_output = _normalize_non_empty_string(output_file, "output_file", required=True)
    if _is_error_payload(normalized_output):
        return normalized_output
    normalized_video_filters = _normalize_string_list(video_filters, "video_filters")
    if _is_error_payload(normalized_video_filters):
        return normalized_video_filters
    normalized_audio_filters = _normalize_string_list(audio_filters, "audio_filters")
    if _is_error_payload(normalized_audio_filters):
        return normalized_audio_filters
    normalized_filter_complex = _normalize_non_empty_string(filter_complex, "filter_complex")
    if _is_error_payload(normalized_filter_complex):
        return normalized_filter_complex
    normalized_output_format = _normalize_non_empty_string(output_format, "output_format")
    if _is_error_payload(normalized_output_format):
        return normalized_output_format
    if not isinstance(preserve_metadata, bool):
        return {"status": "error", "error": "preserve_metadata must be a boolean"}
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_apply_filters"](
                input_file=normalized_input,
                output_file=normalized_output,
                video_filters=normalized_video_filters,
                audio_filters=normalized_audio_filters,
                filter_complex=normalized_filter_complex,
                output_format=normalized_output_format,
                preserve_metadata=preserve_metadata,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "input_file": normalized_input}

    payload = _normalize_payload(resolved)
    payload.setdefault("input_file", normalized_input)
    payload.setdefault("output_file", normalized_output)
    payload.setdefault("filters_applied", [])
    return payload


async def ffmpeg_batch_process(
    input_files: Union[List[str], Dict[str, Any]],
    output_directory: str,
    operation: str = "convert",
    operation_params: Optional[Dict[str, Any]] = None,
    max_parallel: int = 2,
    save_progress: bool = True,
    resume_from_checkpoint: bool = True,
    timeout_per_file: int = 600,
) -> Dict[str, Any]:
    """Process multiple media files using batch FFmpeg workflows."""
    normalized_input_files: Union[List[str], Dict[str, Any]]
    if isinstance(input_files, list):
        normalized_list = _normalize_string_list(input_files, "input_files", required=True)
        if _is_error_payload(normalized_list):
            return normalized_list
        normalized_input_files = normalized_list
    elif isinstance(input_files, dict) and input_files:
        normalized_input_files = dict(input_files)
    else:
        return {
            "status": "error",
            "error": "input_files must be a non-empty list or dataset object",
        }
    normalized_output_directory = _normalize_non_empty_string(
        output_directory,
        "output_directory",
        required=True,
    )
    if _is_error_payload(normalized_output_directory):
        return normalized_output_directory
    normalized_operation = _normalize_non_empty_string(operation, "operation", required=True)
    if _is_error_payload(normalized_operation):
        return normalized_operation
    if operation_params is not None and not isinstance(operation_params, dict):
        return {"status": "error", "error": "operation_params must be an object when provided"}
    if not isinstance(max_parallel, int) or max_parallel < 1:
        return {"status": "error", "error": "max_parallel must be an integer greater than or equal to 1"}
    if not isinstance(save_progress, bool):
        return {"status": "error", "error": "save_progress must be a boolean"}
    if not isinstance(resume_from_checkpoint, bool):
        return {"status": "error", "error": "resume_from_checkpoint must be a boolean"}
    if not isinstance(timeout_per_file, int) or timeout_per_file < 1:
        return {"status": "error", "error": "timeout_per_file must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ffmpeg_batch_process"](
                input_files=normalized_input_files,
                output_directory=normalized_output_directory,
                operation=normalized_operation,
                operation_params=operation_params,
                max_parallel=max_parallel,
                save_progress=save_progress,
                resume_from_checkpoint=resume_from_checkpoint,
                timeout_per_file=timeout_per_file,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "output_directory": normalized_output_directory}

    payload = _normalize_payload(resolved)
    payload.setdefault("operation", normalized_operation)
    payload.setdefault("results", [])
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


async def ytdlp_download_playlist(
    playlist_url: str,
    output_dir: Optional[str] = None,
    quality: str = "best",
    max_downloads: Optional[int] = None,
    start_index: int = 1,
    end_index: Optional[int] = None,
    download_archive: Optional[str] = None,
    custom_opts: Optional[Dict[str, Any]] = None,
    timeout: int = 1200,
) -> Dict[str, Any]:
    """Download a playlist using yt-dlp-backed wrappers."""
    invalid = _validate_http_url(playlist_url)
    if invalid is not None:
        invalid["error"] = invalid["error"].replace("url", "playlist_url")
        invalid["playlist_url"] = playlist_url.strip()
        return invalid
    normalized_playlist_url = playlist_url.strip()
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        return {"status": "error", "error": "output_dir must be a non-empty string when provided"}
    if not isinstance(quality, str) or not quality.strip():
        return {"status": "error", "error": "quality must be a non-empty string"}
    if max_downloads is not None and (not isinstance(max_downloads, int) or max_downloads < 1):
        return {"status": "error", "error": "max_downloads must be an integer greater than or equal to 1 when provided"}
    if not isinstance(start_index, int) or start_index < 1:
        return {"status": "error", "error": "start_index must be an integer greater than or equal to 1"}
    if end_index is not None and (not isinstance(end_index, int) or end_index < start_index):
        return {"status": "error", "error": "end_index must be an integer greater than or equal to start_index when provided"}
    if download_archive is not None and (not isinstance(download_archive, str) or not download_archive.strip()):
        return {"status": "error", "error": "download_archive must be a non-empty string when provided"}
    if custom_opts is not None and not isinstance(custom_opts, dict):
        return {"status": "error", "error": "custom_opts must be an object when provided"}
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ytdlp_download_playlist"](
                playlist_url=normalized_playlist_url,
                output_dir=output_dir.strip() if isinstance(output_dir, str) else None,
                quality=quality.strip(),
                max_downloads=max_downloads,
                start_index=start_index,
                end_index=end_index,
                download_archive=download_archive.strip() if isinstance(download_archive, str) else None,
                custom_opts=custom_opts,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "playlist_url": normalized_playlist_url}

    payload = _normalize_payload(resolved)
    payload.setdefault("playlist_url", normalized_playlist_url)
    payload.setdefault("results", [])
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


async def ytdlp_batch_download(
    urls: List[str],
    output_dir: Optional[str] = None,
    quality: str = "best",
    concurrent_downloads: int = 3,
    ignore_errors: bool = True,
    custom_opts: Optional[Dict[str, Any]] = None,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Download multiple media URLs concurrently using yt-dlp wrappers."""
    normalized_urls = _normalize_string_list(urls, "urls", required=True)
    if _is_error_payload(normalized_urls):
        return normalized_urls
    for item in normalized_urls:
        invalid = _validate_http_url(item)
        if invalid is not None:
            return {"status": "error", "error": invalid.get("error"), "url": item}
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        return {"status": "error", "error": "output_dir must be a non-empty string when provided"}
    if not isinstance(quality, str) or not quality.strip():
        return {"status": "error", "error": "quality must be a non-empty string"}
    if not isinstance(concurrent_downloads, int) or concurrent_downloads < 1:
        return {"status": "error", "error": "concurrent_downloads must be an integer greater than or equal to 1"}
    if not isinstance(ignore_errors, bool):
        return {"status": "error", "error": "ignore_errors must be a boolean"}
    if custom_opts is not None and not isinstance(custom_opts, dict):
        return {"status": "error", "error": "custom_opts must be an object when provided"}
    if not isinstance(timeout, int) or timeout < 1:
        return {"status": "error", "error": "timeout must be an integer greater than or equal to 1"}

    try:
        resolved = await _await_maybe(
            _API["ytdlp_batch_download"](
                urls=normalized_urls,
                output_dir=output_dir.strip() if isinstance(output_dir, str) else None,
                quality=quality.strip(),
                concurrent_downloads=concurrent_downloads,
                ignore_errors=ignore_errors,
                custom_opts=custom_opts,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc), "urls": normalized_urls}

    payload = _normalize_payload(resolved)
    payload.setdefault("urls", normalized_urls)
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
        name="ffmpeg_mux",
        func=ffmpeg_mux,
        description="Mux separate media streams into a single output container.",
        input_schema={
            "type": "object",
            "properties": {
                "video_input": {"type": ["string", "null"], "minLength": 1},
                "audio_inputs": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "subtitle_inputs": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "output_file": {"type": "string", "minLength": 1},
                "output_format": {"type": ["string", "null"], "minLength": 1},
                "video_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "audio_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "subtitle_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "map_streams": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "metadata": {"type": ["object", "null"], "minProperties": 1},
                "timeout": {"type": "integer", "minimum": 1, "default": 300},
            },
            "required": ["output_file"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ffmpeg_demux",
        func=ffmpeg_demux,
        description="Demux media streams into separate output files.",
        input_schema={
            "type": "object",
            "properties": {
                "input_file": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "object", "minProperties": 1},
                    ]
                },
                "output_dir": {"type": "string", "minLength": 1},
                "extract_video": {"type": "boolean", "default": True},
                "extract_audio": {"type": "boolean", "default": True},
                "extract_subtitles": {"type": "boolean", "default": True},
                "video_format": {"type": "string", "minLength": 1, "default": "mp4"},
                "audio_format": {"type": "string", "minLength": 1, "default": "mp3"},
                "subtitle_format": {"type": "string", "minLength": 1, "default": "srt"},
                "stream_selection": {"type": ["object", "null"], "minProperties": 1},
                "timeout": {"type": "integer", "minimum": 1, "default": 300},
            },
            "required": ["input_file", "output_dir"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ffmpeg_stream_input",
        func=ffmpeg_stream_input,
        description="Capture media from an incoming stream.",
        input_schema={
            "type": "object",
            "properties": {
                "stream_url": {"type": "string", "minLength": 1},
                "output_file": {"type": "string", "minLength": 1},
                "duration": {"type": ["string", "null"], "minLength": 1},
                "video_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "audio_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "format": {"type": ["string", "null"], "minLength": 1},
                "buffer_size": {"type": ["string", "null"], "minLength": 1},
                "timeout": {"type": "integer", "minimum": 1, "default": 3600},
            },
            "required": ["stream_url", "output_file"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ffmpeg_stream_output",
        func=ffmpeg_stream_output,
        description="Publish media from a file to an outgoing stream.",
        input_schema={
            "type": "object",
            "properties": {
                "input_file": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "object", "minProperties": 1},
                    ]
                },
                "stream_url": {"type": "string", "minLength": 1},
                "video_codec": {"type": "string", "minLength": 1, "default": "libx264"},
                "audio_codec": {"type": "string", "minLength": 1, "default": "aac"},
                "video_bitrate": {"type": ["string", "null"], "minLength": 1},
                "audio_bitrate": {"type": ["string", "null"], "minLength": 1},
                "resolution": {"type": ["string", "null"], "minLength": 1},
                "framerate": {"type": ["string", "null"], "minLength": 1},
                "format": {"type": "string", "minLength": 1, "default": "flv"},
                "preset": {"type": "string", "minLength": 1, "default": "fast"},
                "tune": {"type": ["string", "null"], "minLength": 1},
                "keyframe_interval": {"type": ["string", "null"], "minLength": 1},
                "buffer_size": {"type": ["string", "null"], "minLength": 1},
                "max_muxing_queue_size": {"type": "string", "minLength": 1, "default": "1024"},
                "timeout": {"type": "integer", "minimum": 1, "default": 3600},
            },
            "required": ["input_file", "stream_url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ffmpeg_cut",
        func=ffmpeg_cut,
        description="Cut a segment from a media file.",
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
                "start_time": {"type": "string", "minLength": 1},
                "end_time": {"type": ["string", "null"], "minLength": 1},
                "duration": {"type": ["string", "null"], "minLength": 1},
                "video_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "audio_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "accurate_seek": {"type": "boolean", "default": True},
                "timeout": {"type": "integer", "minimum": 1, "default": 300},
            },
            "required": ["input_file", "output_file", "start_time"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ffmpeg_splice",
        func=ffmpeg_splice,
        description="Splice multiple media segments into one output file.",
        input_schema={
            "type": "object",
            "properties": {
                "input_files": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "oneOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "object", "minProperties": 1},
                        ]
                    },
                },
                "output_file": {"type": "string", "minLength": 1},
                "segments": {"type": "array", "minItems": 1, "items": {"type": "object", "minProperties": 1}},
                "video_codec": {"type": "string", "minLength": 1, "default": "libx264"},
                "audio_codec": {"type": "string", "minLength": 1, "default": "aac"},
                "transition_type": {"type": "string", "minLength": 1, "default": "cut"},
                "transition_duration": {"type": "number", "minimum": 0, "default": 0},
                "timeout": {"type": "integer", "minimum": 1, "default": 600},
            },
            "required": ["input_files", "output_file", "segments"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ffmpeg_concat",
        func=ffmpeg_concat,
        description="Concatenate multiple media files.",
        input_schema={
            "type": "object",
            "properties": {
                "input_files": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "oneOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "object", "minProperties": 1},
                        ]
                    },
                },
                "output_file": {"type": "string", "minLength": 1},
                "video_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "audio_codec": {"type": "string", "minLength": 1, "default": "copy"},
                "method": {"type": "string", "minLength": 1, "default": "filter"},
                "safe_mode": {"type": "boolean", "default": True},
                "timeout": {"type": "integer", "minimum": 1, "default": 600},
            },
            "required": ["input_files", "output_file"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ffmpeg_apply_filters",
        func=ffmpeg_apply_filters,
        description="Apply audio and video filters to a media file.",
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
                "video_filters": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "audio_filters": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "filter_complex": {"type": ["string", "null"], "minLength": 1},
                "output_format": {"type": ["string", "null"], "minLength": 1},
                "preserve_metadata": {"type": "boolean", "default": True},
                "timeout": {"type": "integer", "minimum": 1, "default": 600},
            },
            "required": ["input_file", "output_file"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )

    manager.register_tool(
        category="media_tools",
        name="ffmpeg_batch_process",
        func=ffmpeg_batch_process,
        description="Process multiple media files using batch FFmpeg workflows.",
        input_schema={
            "type": "object",
            "properties": {
                "input_files": {
                    "oneOf": [
                        {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                        {"type": "object", "minProperties": 1},
                    ]
                },
                "output_directory": {"type": "string", "minLength": 1},
                "operation": {"type": "string", "minLength": 1, "default": "convert"},
                "operation_params": {"type": ["object", "null"]},
                "max_parallel": {"type": "integer", "minimum": 1, "default": 2},
                "save_progress": {"type": "boolean", "default": True},
                "resume_from_checkpoint": {"type": "boolean", "default": True},
                "timeout_per_file": {"type": "integer", "minimum": 1, "default": 600},
            },
            "required": ["input_files", "output_directory"],
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
        name="ytdlp_download_playlist",
        func=ytdlp_download_playlist,
        description="Download a playlist using yt-dlp wrappers.",
        input_schema={
            "type": "object",
            "properties": {
                "playlist_url": {"type": "string", "minLength": 1},
                "output_dir": {"type": ["string", "null"], "minLength": 1},
                "quality": {"type": "string", "minLength": 1, "default": "best"},
                "max_downloads": {"type": ["integer", "null"], "minimum": 1},
                "start_index": {"type": "integer", "minimum": 1, "default": 1},
                "end_index": {"type": ["integer", "null"], "minimum": 1},
                "download_archive": {"type": ["string", "null"], "minLength": 1},
                "custom_opts": {"type": ["object", "null"]},
                "timeout": {"type": "integer", "minimum": 1, "default": 1200},
            },
            "required": ["playlist_url"],
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

    manager.register_tool(
        category="media_tools",
        name="ytdlp_batch_download",
        func=ytdlp_batch_download,
        description="Download multiple media URLs concurrently using yt-dlp wrappers.",
        input_schema={
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "output_dir": {"type": ["string", "null"], "minLength": 1},
                "quality": {"type": "string", "minLength": 1, "default": "best"},
                "concurrent_downloads": {"type": "integer", "minimum": 1, "default": 3},
                "ignore_errors": {"type": "boolean", "default": True},
                "custom_opts": {"type": ["object", "null"]},
                "timeout": {"type": "integer", "minimum": 1, "default": 1800},
            },
            "required": ["urls"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "media-tools"],
    )
