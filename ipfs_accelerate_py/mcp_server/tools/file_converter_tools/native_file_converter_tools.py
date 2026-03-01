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


async def convert_file_tool(
    input_path: str,
    backend: str = "native",
    extract_archives: bool = False,
    output_format: str = "text",
) -> Dict[str, Any]:
    """Convert a file or URL into text/structured output."""
    result = _API["convert_file_tool"](
        input_path=input_path,
        backend=backend,
        extract_archives=extract_archives,
        output_format=output_format,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def file_info_tool(input_path: str) -> Dict[str, Any]:
    """Get file metadata and inferred type information."""
    result = _API["file_info_tool"](input_path=input_path)
    if hasattr(result, "__await__"):
        return await result
    return result


async def download_url_tool(
    url: str,
    timeout: int = 30,
    max_size_mb: int = 100,
) -> Dict[str, Any]:
    """Download content from an HTTP/HTTPS URL."""
    result = _API["download_url_tool"](
        url=url,
        timeout=timeout,
        max_size_mb=max_size_mb,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
                "input_path": {"type": "string"},
                "backend": {"type": "string"},
                "extract_archives": {"type": "boolean"},
                "output_format": {"type": "string"},
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
                "input_path": {"type": "string"},
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
                "url": {"type": "string"},
                "timeout": {"type": "integer"},
                "max_size_mb": {"type": "integer"},
            },
            "required": ["url"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-converter-tools"],
    )
