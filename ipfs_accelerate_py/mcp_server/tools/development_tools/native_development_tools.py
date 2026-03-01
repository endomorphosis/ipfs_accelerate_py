"""Native development-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_development_tools_api() -> Dict[str, Any]:
    """Resolve source development-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.development_tools import (  # type: ignore
            codebase_search as _codebase_search,
            vscode_cli_status as _vscode_cli_status,
        )

        return {
            "codebase_search": _codebase_search,
            "vscode_cli_status": _vscode_cli_status,
        }
    except Exception:
        logger.warning(
            "Source development_tools import unavailable, using fallback development-tools functions"
        )

        def _codebase_search_fallback(
            pattern: str,
            path: str = ".",
            case_insensitive: bool = False,
            whole_word: bool = False,
            regex: bool = False,
            extensions: Optional[str] = None,
            exclude: Optional[str] = None,
            max_depth: Optional[int] = None,
            context: int = 0,
            format: str = "text",
            output: Optional[str] = None,
            compact: bool = False,
            group_by_file: bool = False,
            summary: bool = False,
        ) -> Dict[str, Any]:
            _ = (
                path,
                case_insensitive,
                whole_word,
                regex,
                extensions,
                exclude,
                max_depth,
                context,
                format,
                output,
                compact,
                group_by_file,
                summary,
            )
            return {
                "success": True,
                "result": {
                    "matches": [],
                    "summary": {"total_matches": 0, "pattern": pattern},
                },
            }

        def _vscode_cli_status_fallback(install_dir: Optional[str] = None) -> Dict[str, Any]:
            _ = install_dir
            return {
                "success": False,
                "error": "VSCode CLI integration unavailable",
            }

        return {
            "codebase_search": _codebase_search_fallback,
            "vscode_cli_status": _vscode_cli_status_fallback,
        }


_API = _load_development_tools_api()


async def codebase_search(
    pattern: str,
    path: str = ".",
    case_insensitive: bool = False,
    whole_word: bool = False,
    regex: bool = False,
    extensions: Optional[str] = None,
    exclude: Optional[str] = None,
    max_depth: Optional[int] = None,
    context: int = 0,
    format: str = "text",
    output: Optional[str] = None,
    compact: bool = False,
    group_by_file: bool = False,
    summary: bool = False,
) -> Any:
    """Search code in a workspace path with pattern-matching options."""
    result = _API["codebase_search"](
        pattern=pattern,
        path=path,
        case_insensitive=case_insensitive,
        whole_word=whole_word,
        regex=regex,
        extensions=extensions,
        exclude=exclude,
        max_depth=max_depth,
        context=context,
        format=format,
        output=output,
        compact=compact,
        group_by_file=group_by_file,
        summary=summary,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def vscode_cli_status(install_dir: Optional[str] = None) -> Dict[str, Any]:
    """Get VS Code CLI installation and status details."""
    result = _API["vscode_cli_status"](install_dir=install_dir)
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_development_tools(manager: Any) -> None:
    """Register native development-tools category tools in unified manager."""
    manager.register_tool(
        category="development_tools",
        name="codebase_search",
        func=codebase_search,
        description="Search a codebase using text or regex patterns.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "case_insensitive": {"type": "boolean"},
                "whole_word": {"type": "boolean"},
                "regex": {"type": "boolean"},
                "extensions": {"type": ["string", "null"]},
                "exclude": {"type": ["string", "null"]},
                "max_depth": {"type": ["integer", "null"]},
                "context": {"type": "integer"},
                "format": {"type": "string"},
                "output": {"type": ["string", "null"]},
                "compact": {"type": "boolean"},
                "group_by_file": {"type": "boolean"},
                "summary": {"type": "boolean"},
            },
            "required": ["pattern"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "development-tools"],
    )

    manager.register_tool(
        category="development_tools",
        name="vscode_cli_status",
        func=vscode_cli_status,
        description="Return VS Code CLI installation status.",
        input_schema={
            "type": "object",
            "properties": {
                "install_dir": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "development-tools"],
    )
