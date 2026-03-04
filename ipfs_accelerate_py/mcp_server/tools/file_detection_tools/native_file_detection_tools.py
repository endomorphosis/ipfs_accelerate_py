"""Native file-detection tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_file_detection_api() -> Dict[str, Any]:
    """Resolve source file-detection APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.file_detection_tools.analyze_detection_accuracy import (  # type: ignore
            analyze_detection_accuracy as _analyze_detection_accuracy,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.file_detection_tools.batch_detect_file_types import (  # type: ignore
            batch_detect_file_types as _batch_detect_file_types,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.file_detection_tools.detect_file_type import (  # type: ignore
            detect_file_type as _detect_file_type,
        )

        return {
            "detect_file_type": _detect_file_type,
            "batch_detect_file_types": _batch_detect_file_types,
            "analyze_detection_accuracy": _analyze_detection_accuracy,
        }
    except Exception:
        logger.warning("Source file_detection_tools import unavailable, using fallback detection functions")

        def _detect_fallback(
            file_path: str,
            methods: Optional[List[str]] = None,
            strategy: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = methods, strategy
            return {
                "error": "FileTypeDetector not available",
                "file_path": file_path,
                "mime_type": None,
                "extension": None,
                "confidence": 0.0,
                "method": "none",
            }

        def _batch_fallback(
            directory: Optional[str] = None,
            file_paths: Optional[List[str]] = None,
            recursive: bool = False,
            pattern: str = "*",
            methods: Optional[List[str]] = None,
            strategy: Optional[str] = None,
            export_path: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = directory, file_paths, recursive, pattern, methods, strategy, export_path
            return {
                "error": "FileTypeDetector not available",
                "results": {},
                "total_files": 0,
                "successful": 0,
                "failed": 0,
            }

        def _analyze_fallback(
            directory: str,
            recursive: bool = False,
            pattern: str = "*",
        ) -> Dict[str, Any]:
            _ = directory, recursive, pattern
            return {
                "error": "FileTypeDetector not available",
                "total_files": 0,
            }

        return {
            "detect_file_type": _detect_fallback,
            "batch_detect_file_types": _batch_fallback,
            "analyze_detection_accuracy": _analyze_fallback,
        }


_API = _load_file_detection_api()


async def detect_file_type(
    file_path: str,
    methods: Optional[List[str]] = None,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect a single file type via extension/magic/model methods."""
    normalized_file_path = str(file_path or "").strip()
    if not normalized_file_path:
        return {
            "status": "error",
            "message": "file_path is required",
            "file_path": file_path,
        }
    valid_methods = {"extension", "magic", "magika", "all"}
    if methods is not None:
        if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
            return {
                "status": "error",
                "message": "methods must be an array of strings when provided",
                "methods": methods,
            }
        normalized_methods = [str(item).strip().lower() for item in methods]
        if any(not item for item in normalized_methods):
            return {
                "status": "error",
                "message": "methods cannot contain empty strings",
                "methods": methods,
            }
        if any(item not in valid_methods for item in normalized_methods):
            return {
                "status": "error",
                "message": "methods entries must be one of: all, extension, magic, magika",
                "methods": methods,
            }
    else:
        normalized_methods = None

    normalized_strategy: Optional[str] = None
    if strategy is not None:
        normalized_strategy = str(strategy).strip().lower()
        if normalized_strategy not in {"fast", "accurate", "voting", "conservative"}:
            return {
                "status": "error",
                "message": "strategy must be one of: fast, accurate, voting, conservative",
                "strategy": strategy,
            }

    result = _API["detect_file_type"](
        file_path=normalized_file_path,
        methods=normalized_methods,
        strategy=normalized_strategy,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("file_path", normalized_file_path)
    return payload


async def batch_detect_file_types(
    directory: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    recursive: bool = False,
    pattern: str = "*",
    methods: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    export_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect file types for multiple files in a batch run."""
    normalized_directory = str(directory or "").strip() if directory is not None else None
    if normalized_directory == "":
        return {
            "status": "error",
            "message": "directory must be a non-empty string when provided",
            "directory": directory,
        }
    normalized_file_paths: Optional[List[str]] = None
    if file_paths is not None:
        if not isinstance(file_paths, list) or not all(isinstance(item, str) for item in file_paths):
            return {
                "status": "error",
                "message": "file_paths must be an array of strings when provided",
                "file_paths": file_paths,
            }
        normalized_file_paths = [str(item).strip() for item in file_paths]
        if any(not item for item in normalized_file_paths):
            return {
                "status": "error",
                "message": "file_paths cannot contain empty strings",
                "file_paths": file_paths,
            }
    if not normalized_directory and not normalized_file_paths:
        return {
            "status": "error",
            "message": "either directory or file_paths must be provided",
            "directory": directory,
            "file_paths": file_paths,
        }
    if not isinstance(recursive, bool):
        return {
            "status": "error",
            "message": "recursive must be a boolean",
            "recursive": recursive,
        }
    normalized_pattern = str(pattern or "").strip()
    if not normalized_pattern:
        return {
            "status": "error",
            "message": "pattern must be a non-empty string",
            "pattern": pattern,
        }

    valid_methods = {"extension", "magic", "magika", "all"}
    normalized_methods: Optional[List[str]] = None
    if methods is not None:
        if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
            return {
                "status": "error",
                "message": "methods must be an array of strings when provided",
                "methods": methods,
            }
        normalized_methods = [str(item).strip().lower() for item in methods]
        if any(not item for item in normalized_methods):
            return {
                "status": "error",
                "message": "methods cannot contain empty strings",
                "methods": methods,
            }
        if any(item not in valid_methods for item in normalized_methods):
            return {
                "status": "error",
                "message": "methods entries must be one of: all, extension, magic, magika",
                "methods": methods,
            }

    normalized_strategy: Optional[str] = None
    if strategy is not None:
        normalized_strategy = str(strategy).strip().lower()
        if normalized_strategy not in {"fast", "accurate", "voting", "conservative"}:
            return {
                "status": "error",
                "message": "strategy must be one of: fast, accurate, voting, conservative",
                "strategy": strategy,
            }

    normalized_export_path = str(export_path or "").strip() if export_path is not None else None
    if export_path is not None and not normalized_export_path:
        return {
            "status": "error",
            "message": "export_path must be a non-empty string when provided",
            "export_path": export_path,
        }

    result = _API["batch_detect_file_types"](
        directory=normalized_directory,
        file_paths=normalized_file_paths,
        recursive=recursive,
        pattern=normalized_pattern,
        methods=normalized_methods,
        strategy=normalized_strategy,
        export_path=normalized_export_path,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    return payload


async def analyze_detection_accuracy(
    directory: str,
    recursive: bool = False,
    pattern: str = "*",
) -> Dict[str, Any]:
    """Analyze agreement and confidence across detection methods."""
    normalized_directory = str(directory or "").strip()
    if not normalized_directory:
        return {
            "status": "error",
            "message": "directory is required",
            "directory": directory,
        }
    if not isinstance(recursive, bool):
        return {
            "status": "error",
            "message": "recursive must be a boolean",
            "recursive": recursive,
        }
    normalized_pattern = str(pattern or "").strip()
    if not normalized_pattern:
        return {
            "status": "error",
            "message": "pattern must be a non-empty string",
            "pattern": pattern,
        }

    result = _API["analyze_detection_accuracy"](
        directory=normalized_directory,
        recursive=recursive,
        pattern=normalized_pattern,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("directory", normalized_directory)
    return payload


def register_native_file_detection_tools(manager: Any) -> None:
    """Register native file-detection tools in unified hierarchical manager."""
    manager.register_tool(
        category="file_detection_tools",
        name="detect_file_type",
        func=detect_file_type,
        description="Detect file type using configured methods and strategy.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "methods": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["extension", "magic", "magika", "all"]},
                },
                "strategy": {"type": ["string", "null"], "enum": ["fast", "accurate", "voting", "conservative", None]},
            },
            "required": ["file_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-detection"],
    )

    manager.register_tool(
        category="file_detection_tools",
        name="batch_detect_file_types",
        func=batch_detect_file_types,
        description="Detect file types for files from a directory or explicit list.",
        input_schema={
            "type": "object",
            "properties": {
                "directory": {"type": ["string", "null"]},
                "file_paths": {"type": ["array", "null"], "items": {"type": "string"}},
                "recursive": {"type": "boolean", "default": False},
                "pattern": {"type": "string", "default": "*"},
                "methods": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["extension", "magic", "magika", "all"]},
                },
                "strategy": {"type": ["string", "null"], "enum": ["fast", "accurate", "voting", "conservative", None]},
                "export_path": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-detection"],
    )

    manager.register_tool(
        category="file_detection_tools",
        name="analyze_detection_accuracy",
        func=analyze_detection_accuracy,
        description="Analyze method agreement and confidence for file detection.",
        input_schema={
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "recursive": {"type": "boolean", "default": False},
                "pattern": {"type": "string", "default": "*"},
            },
            "required": ["directory"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-detection"],
    )
