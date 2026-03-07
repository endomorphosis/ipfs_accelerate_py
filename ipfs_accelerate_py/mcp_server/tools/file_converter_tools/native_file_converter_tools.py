"""Native file-converter-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_file_converter_tools_api() -> Dict[str, Any]:
    """Resolve source file-converter-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.file_converter_tools import (  # type: ignore
            batch_convert_tool as _batch_convert_tool,
            convert_file_tool as _convert_file_tool,
            download_url_tool as _download_url_tool,
            extract_archive_tool as _extract_archive_tool,
            extract_knowledge_graph_tool as _extract_knowledge_graph_tool,
            file_info_tool as _file_info_tool,
            generate_embeddings_tool as _generate_embeddings_tool,
            generate_summary_tool as _generate_summary_tool,
        )

        return {
            "batch_convert_tool": _batch_convert_tool,
            "convert_file_tool": _convert_file_tool,
            "download_url_tool": _download_url_tool,
            "extract_archive_tool": _extract_archive_tool,
            "extract_knowledge_graph_tool": _extract_knowledge_graph_tool,
            "file_info_tool": _file_info_tool,
            "generate_embeddings_tool": _generate_embeddings_tool,
            "generate_summary_tool": _generate_summary_tool,
        }
    except Exception:
        logger.warning(
            "Source file_converter_tools import unavailable, using fallback file-converter functions"
        )

        async def _batch_convert_fallback(
            input_paths: list[str],
            backend: str = "native",
            extract_archives: bool = False,
            max_concurrent: int = 5,
        ) -> Dict[str, Any]:
            _ = backend, extract_archives, max_concurrent
            return {
                "success": False,
                "status": "error",
                "input_paths": input_paths,
                "error": "file converter backend unavailable",
            }

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

        async def _extract_archive_fallback(
            archive_path: str,
            max_depth: int = 3,
            recursive: bool = True,
        ) -> Dict[str, Any]:
            _ = max_depth, recursive
            return {
                "success": False,
                "status": "error",
                "archive_path": archive_path,
                "error": "archive extraction backend unavailable",
            }

        async def _extract_kg_fallback(
            input_path: str,
            enable_ipfs: bool = False,
        ) -> Dict[str, Any]:
            _ = enable_ipfs
            return {
                "success": False,
                "status": "error",
                "input_path": input_path,
                "error": "knowledge graph backend unavailable",
            }

        async def _generate_embeddings_fallback(
            input_path: str,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            vector_store: str = "faiss",
            enable_ipfs: bool = False,
            enable_acceleration: bool = False,
        ) -> Dict[str, Any]:
            _ = embedding_model, vector_store, enable_ipfs, enable_acceleration
            return {
                "success": False,
                "status": "error",
                "input_path": input_path,
                "error": "embedding backend unavailable",
            }

        async def _generate_summary_fallback(
            input_path: str,
            llm_model: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = llm_model
            return {
                "success": False,
                "status": "error",
                "input_path": input_path,
                "error": "summary backend unavailable",
            }

        return {
            "batch_convert_tool": _batch_convert_fallback,
            "convert_file_tool": _convert_fallback,
            "download_url_tool": _download_fallback,
            "extract_archive_tool": _extract_archive_fallback,
            "extract_knowledge_graph_tool": _extract_kg_fallback,
            "file_info_tool": _info_fallback,
            "generate_embeddings_tool": _generate_embeddings_fallback,
            "generate_summary_tool": _generate_summary_fallback,
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
        if merged.get("error") and not merged.get("status"):
            merged["status"] = "error"
        else:
            merged.setdefault("status", "success")
        if merged.get("status") == "success":
            merged.setdefault("success", True)
        return merged
    return {
        **default_fields,
        "status": "success",
        "success": True,
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


async def batch_convert_tool(
    input_paths: list[str],
    backend: str = "native",
    extract_archives: bool = False,
    max_concurrent: int = 5,
) -> Dict[str, Any]:
    """Batch convert multiple files or URLs into text/structured output."""
    if not isinstance(input_paths, list) or not input_paths:
        return {
            "status": "error",
            "error": "input_paths must be a non-empty list of strings",
        }

    normalized_input_paths = [str(path or "").strip() for path in input_paths]
    if any(not path for path in normalized_input_paths):
        return {
            "status": "error",
            "error": "input_paths must be a non-empty list of strings",
        }

    normalized_backend = str(backend or "").strip()
    if not normalized_backend:
        return {
            "status": "error",
            "error": "backend must be a non-empty string",
        }

    if not isinstance(extract_archives, bool):
        return {
            "status": "error",
            "error": "extract_archives must be a boolean",
        }

    if not isinstance(max_concurrent, int) or max_concurrent < 1:
        return {
            "status": "error",
            "error": "max_concurrent must be an integer >= 1",
        }

    try:
        result = _API["batch_convert_tool"](
            input_paths=normalized_input_paths,
            backend=normalized_backend,
            extract_archives=extract_archives,
            max_concurrent=max_concurrent,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return {
            "status": "error",
            "error": f"batch_convert_tool failed: {exc}",
            "input_paths": normalized_input_paths,
            "backend": normalized_backend,
            "max_concurrent": max_concurrent,
        }

    return _normalize_payload(
        result,
        default_fields={
            "tool": "batch_convert_tool",
            "input_paths": normalized_input_paths,
            "backend": normalized_backend,
            "extract_archives": extract_archives,
            "max_concurrent": max_concurrent,
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


async def extract_knowledge_graph_tool(
    input_path: str,
    enable_ipfs: bool = False,
) -> Dict[str, Any]:
    """Extract entities and relationships from a file or URL."""
    normalized_input_path = str(input_path or "").strip()
    if not normalized_input_path:
        return {
            "status": "error",
            "error": "input_path is required",
        }

    if not isinstance(enable_ipfs, bool):
        return {
            "status": "error",
            "error": "enable_ipfs must be a boolean",
        }

    try:
        result = _API["extract_knowledge_graph_tool"](
            input_path=normalized_input_path,
            enable_ipfs=enable_ipfs,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return {
            "status": "error",
            "error": f"extract_knowledge_graph_tool failed: {exc}",
            "input_path": normalized_input_path,
            "enable_ipfs": enable_ipfs,
        }

    return _normalize_payload(
        result,
        default_fields={
            "tool": "extract_knowledge_graph_tool",
            "input_path": normalized_input_path,
            "enable_ipfs": enable_ipfs,
        },
    )


async def generate_summary_tool(
    input_path: str,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a text summary for a file or URL."""
    normalized_input_path = str(input_path or "").strip()
    if not normalized_input_path:
        return {
            "status": "error",
            "error": "input_path is required",
        }

    normalized_llm_model = None
    if llm_model is not None:
        normalized_llm_model = str(llm_model or "").strip()
        if not normalized_llm_model:
            return {
                "status": "error",
                "error": "llm_model must be a non-empty string when provided",
            }

    try:
        result = _API["generate_summary_tool"](
            input_path=normalized_input_path,
            llm_model=normalized_llm_model,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return {
            "status": "error",
            "error": f"generate_summary_tool failed: {exc}",
            "input_path": normalized_input_path,
            "llm_model": normalized_llm_model,
        }

    return _normalize_payload(
        result,
        default_fields={
            "tool": "generate_summary_tool",
            "input_path": normalized_input_path,
            "llm_model": normalized_llm_model,
        },
    )


async def generate_embeddings_tool(
    input_path: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    vector_store: str = "faiss",
    enable_ipfs: bool = False,
    enable_acceleration: bool = False,
) -> Dict[str, Any]:
    """Generate embeddings for a file or URL."""
    normalized_input_path = str(input_path or "").strip()
    if not normalized_input_path:
        return {
            "status": "error",
            "error": "input_path is required",
        }

    normalized_embedding_model = str(embedding_model or "").strip()
    if not normalized_embedding_model:
        return {
            "status": "error",
            "error": "embedding_model must be a non-empty string",
        }

    normalized_vector_store = str(vector_store or "").strip().lower()
    allowed_vector_stores = {"faiss", "qdrant", "elasticsearch"}
    if normalized_vector_store not in allowed_vector_stores:
        return {
            "status": "error",
            "error": "vector_store must be one of: elasticsearch, faiss, qdrant",
        }

    if not isinstance(enable_ipfs, bool):
        return {
            "status": "error",
            "error": "enable_ipfs must be a boolean",
        }

    if not isinstance(enable_acceleration, bool):
        return {
            "status": "error",
            "error": "enable_acceleration must be a boolean",
        }

    try:
        result = _API["generate_embeddings_tool"](
            input_path=normalized_input_path,
            embedding_model=normalized_embedding_model,
            vector_store=normalized_vector_store,
            enable_ipfs=enable_ipfs,
            enable_acceleration=enable_acceleration,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return {
            "status": "error",
            "error": f"generate_embeddings_tool failed: {exc}",
            "input_path": normalized_input_path,
            "embedding_model": normalized_embedding_model,
            "vector_store": normalized_vector_store,
        }

    return _normalize_payload(
        result,
        default_fields={
            "tool": "generate_embeddings_tool",
            "input_path": normalized_input_path,
            "embedding_model": normalized_embedding_model,
            "vector_store": normalized_vector_store,
            "enable_ipfs": enable_ipfs,
            "enable_acceleration": enable_acceleration,
        },
    )


async def extract_archive_tool(
    archive_path: str,
    max_depth: int = 3,
    recursive: bool = True,
) -> Dict[str, Any]:
    """Extract an archive file, optionally recursing into nested archives."""
    normalized_archive_path = str(archive_path or "").strip()
    if not normalized_archive_path:
        return {
            "status": "error",
            "error": "archive_path is required",
        }

    if not isinstance(max_depth, int) or max_depth < 0:
        return {
            "status": "error",
            "error": "max_depth must be an integer >= 0",
        }

    if not isinstance(recursive, bool):
        return {
            "status": "error",
            "error": "recursive must be a boolean",
        }

    try:
        result = _API["extract_archive_tool"](
            archive_path=normalized_archive_path,
            max_depth=max_depth,
            recursive=recursive,
        )
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return {
            "status": "error",
            "error": f"extract_archive_tool failed: {exc}",
            "archive_path": normalized_archive_path,
            "max_depth": max_depth,
            "recursive": recursive,
        }

    return _normalize_payload(
        result,
        default_fields={
            "tool": "extract_archive_tool",
            "archive_path": normalized_archive_path,
            "max_depth": max_depth,
            "recursive": recursive,
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
        name="batch_convert_tool",
        func=batch_convert_tool,
        description="Batch convert multiple files or URLs to text/structured output.",
        input_schema={
            "type": "object",
            "properties": {
                "input_paths": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "backend": {"type": "string", "minLength": 1, "default": "native"},
                "extract_archives": {"type": "boolean"},
                "max_concurrent": {"type": "integer", "minimum": 1, "default": 5},
            },
            "required": ["input_paths"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-converter-tools"],
    )

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
        name="extract_knowledge_graph_tool",
        func=extract_knowledge_graph_tool,
        description="Extract entities and relationships from a file or URL.",
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "minLength": 1},
                "enable_ipfs": {"type": "boolean", "default": False},
            },
            "required": ["input_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-converter-tools"],
    )

    manager.register_tool(
        category="file_converter_tools",
        name="generate_summary_tool",
        func=generate_summary_tool,
        description="Generate a text summary from a file or URL.",
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "minLength": 1},
                "llm_model": {"type": "string", "minLength": 1},
            },
            "required": ["input_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-converter-tools"],
    )

    manager.register_tool(
        category="file_converter_tools",
        name="generate_embeddings_tool",
        func=generate_embeddings_tool,
        description="Generate vector embeddings from a file or URL.",
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "minLength": 1},
                "embedding_model": {
                    "type": "string",
                    "minLength": 1,
                    "default": "sentence-transformers/all-MiniLM-L6-v2",
                },
                "vector_store": {
                    "type": "string",
                    "enum": ["faiss", "qdrant", "elasticsearch"],
                    "default": "faiss",
                },
                "enable_ipfs": {"type": "boolean", "default": False},
                "enable_acceleration": {"type": "boolean", "default": False},
            },
            "required": ["input_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-converter-tools"],
    )

    manager.register_tool(
        category="file_converter_tools",
        name="extract_archive_tool",
        func=extract_archive_tool,
        description="Extract contents from archive files.",
        input_schema={
            "type": "object",
            "properties": {
                "archive_path": {"type": "string", "minLength": 1},
                "max_depth": {"type": "integer", "minimum": 0, "default": 3},
                "recursive": {"type": "boolean", "default": True},
            },
            "required": ["archive_path"],
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
