"""Native file-converter-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_file_converter_tools_api() -> Dict[str, Any]:
    """Resolve source file-converter-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.file_converter_tools import (  # type: ignore
            convert_file_tool as _convert_file_tool,
            download_url_tool as _download_url_tool,
            file_info_tool as _file_info_tool,
        )

        return {
            "convert_file_tool": _convert_file_tool,
            "file_info_tool": _file_info_tool,
            "download_url_tool": _download_url_tool,
        }
    except Exception:
        logger.warning(
            "Source file_converter_tools import unavailable, using fallback file-converter functions"
        )

        async def _convert_fallback(
            input_path: str,
            backend: str = "native",
            extract_archives: bool = False,
            output_format: str = "text",
        ) -> Dict[str, Any]:
            _ = backend, extract_archives, output_format
            return {
                "success": False,
                "status": "error",
                "input_path": input_path,
                "error": "file converter backend unavailable",
            }

        async def _info_fallback(input_path: str) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "input_path": input_path,
                "is_url": input_path.startswith("http://") or input_path.startswith("https://"),
                "mime_type": "application/octet-stream",
                "format": "unknown",
            }

        async def _download_fallback(
            url: str,
            timeout: int = 30,
            max_size_mb: int = 100,
        ) -> Dict[str, Any]:
            _ = timeout, max_size_mb
            return {
                "success": False,
                "status": "error",
                "url": url,
                "error": "download backend unavailable",
            }

        return {
            "convert_file_tool": _convert_fallback,
            "file_info_tool": _info_fallback,
            "download_url_tool": _download_fallback,
        }


_API = _load_file_converter_tools_api()


def _normalize_payload(
    payload: Any,
    *,
    default_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dictionary envelopes."""
    if isinstance(payload, dict):
        merged: Dict[str, Any] = dict(default_fields)
        merged.update(payload)
        merged.setdefault("status", "success")
        return merged
    return {
        **default_fields,
        "status": "success",
        "result": payload,
    }


async def convert_file_tool(
    input_path: str,
    backend: str = "native",
    extract_archives: bool = False,
    output_format: str = "text",
) -> Dict[str, Any]:
    """Convert a file or URL into text/structured output."""
    normalized_input_path = str(input_path or "").strip()
    if not normalized_input_path:
        return {
            "status": "error",
            "error": "input_path is required",
        }

    normalized_backend = str(backend or "").strip()
    if not normalized_backend:
        return {
            "status": "error",
            "error": "backend must be a non-empty string",
        }

    normalized_output_format = str(output_format or "").strip()
    if not normalized_output_format:
        return {
            "status": "error",
            "error": "output_format must be a non-empty string",
        }

    if not isinstance(extract_archives, bool):
        return {
            "status": "error",
            "error": "extract_archives must be a boolean",
        }

    try:
        result = _API["convert_file_tool"](
            input_path=normalized_input_path,
            backend=normalized_backend,
            extract_archives=extract_archives,
            output_format=normalized_output_format,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return {
            "status": "error",
            "error": f"convert_file_tool failed: {exc}",
            "input_path": normalized_input_path,
            "backend": normalized_backend,
            "output_format": normalized_output_format,
        }

    return _normalize_payload(
        result,
        default_fields={
            "tool": "convert_file_tool",
            "input_path": normalized_input_path,
            "backend": normalized_backend,
            "output_format": normalized_output_format,
        },
    )


async def file_info_tool(input_path: str) -> Dict[str, Any]:
    """Get file metadata and inferred type information."""
    normalized_input_path = str(input_path or "").strip()
    if not normalized_input_path:
        return {
            "status": "error",
            "error": "input_path is required",
        }

    try:
        result = _API["file_info_tool"](input_path=normalized_input_path)
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return {
            "status": "error",
            "error": f"file_info_tool failed: {exc}",
            "input_path": normalized_input_path,
        }

    return _normalize_payload(
        result,
        default_fields={
            "tool": "file_info_tool",
            "input_path": normalized_input_path,
        },
    )


async def download_url_tool(
    url: str,
    timeout: int = 30,
    max_size_mb: int = 100,
) -> Dict[str, Any]:
    """Download content from an HTTP/HTTPS URL."""
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return {
            "status": "error",
            "error": "url is required",
        }

    if not isinstance(timeout, int) or timeout < 1:
        return {
            "status": "error",
            "error": "timeout must be an integer >= 1",
        }

    if not isinstance(max_size_mb, int) or max_size_mb < 1:
        return {
            "status": "error",
            "error": "max_size_mb must be an integer >= 1",
        }

    try:
        result = _API["download_url_tool"](
            url=normalized_url,
            timeout=timeout,
            max_size_mb=max_size_mb,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return {
            "status": "error",
            "error": f"download_url_tool failed: {exc}",
            "url": normalized_url,
            "timeout": timeout,
            "max_size_mb": max_size_mb,
        }

    return _normalize_payload(
        result,
        default_fields={
            "tool": "download_url_tool",
            "url": normalized_url,
            "timeout": timeout,
            "max_size_mb": max_size_mb,
        },
    )


def register_native_file_converter_tools(manager: Any) -> None:
    """Register native file-converter-tools in unified manager."""
    manager.register_tool(
        category="file_converter_tools",
        name="convert_file_tool",
        func=convert_file_tool,
        description="Convert a file or URL to text/structured output.",
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "minLength": 1},
                "backend": {"type": "string", "minLength": 1, "default": "native"},
                "extract_archives": {"type": "boolean"},
                "output_format": {"type": "string", "minLength": 1, "default": "text"},
            },
            "required": ["input_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-converter-tools"],
    )

    manager.register_tool(
        category="file_converter_tools",
        name="file_info_tool",
        func=file_info_tool,
        description="Get metadata and inferred info for a file path or URL.",
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "minLength": 1},
            },
            "required": ["input_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-converter-tools"],
    )

    manager.register_tool(
        category="file_converter_tools",
        name="download_url_tool",
        func=download_url_tool,
        description="Download a URL with timeout and size limits.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "minLength": 1},
                "timeout": {"type": "integer", "minimum": 1, "default": 30},
                "max_size_mb": {"type": "integer", "minimum": 1, "default": 100},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-converter-tools"],
    )
