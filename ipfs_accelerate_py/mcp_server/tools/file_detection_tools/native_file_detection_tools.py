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
    return _API["detect_file_type"](
        file_path=file_path,
        methods=methods,
        strategy=strategy,
    )


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
    return _API["batch_detect_file_types"](
        directory=directory,
        file_paths=file_paths,
        recursive=recursive,
        pattern=pattern,
        methods=methods,
        strategy=strategy,
        export_path=export_path,
    )


async def analyze_detection_accuracy(
    directory: str,
    recursive: bool = False,
    pattern: str = "*",
) -> Dict[str, Any]:
    """Analyze agreement and confidence across detection methods."""
    return _API["analyze_detection_accuracy"](
        directory=directory,
        recursive=recursive,
        pattern=pattern,
    )


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
                "methods": {"type": ["array", "null"], "items": {"type": "string"}},
                "strategy": {"type": ["string", "null"]},
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
                "recursive": {"type": "boolean"},
                "pattern": {"type": "string"},
                "methods": {"type": ["array", "null"], "items": {"type": "string"}},
                "strategy": {"type": ["string", "null"]},
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
                "recursive": {"type": "boolean"},
                "pattern": {"type": "string"},
            },
            "required": ["directory"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "file-detection"],
    )
