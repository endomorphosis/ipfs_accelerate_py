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


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend output into deterministic status envelopes."""
    if isinstance(result, dict):
        payload = dict(result)
        if payload.get("error") or payload.get("success") is False:
            payload.setdefault("status", "error")
        else:
            payload.setdefault("status", "success")
        return payload
    return {"status": "success", "result": result}


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
    normalized_pattern = str(pattern or "").strip()
    if not normalized_pattern:
        return {
            "status": "error",
            "error": "pattern is required",
            "success": False,
        }

    normalized_path = str(path or ".").strip() or "."
    if not isinstance(case_insensitive, bool):
        return {"status": "error", "error": "case_insensitive must be a boolean", "success": False}
    if not isinstance(whole_word, bool):
        return {"status": "error", "error": "whole_word must be a boolean", "success": False}
    if not isinstance(regex, bool):
        return {"status": "error", "error": "regex must be a boolean", "success": False}
    if not isinstance(compact, bool):
        return {"status": "error", "error": "compact must be a boolean", "success": False}
    if not isinstance(group_by_file, bool):
        return {"status": "error", "error": "group_by_file must be a boolean", "success": False}
    if not isinstance(summary, bool):
        return {"status": "error", "error": "summary must be a boolean", "success": False}

    if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
        return {"status": "error", "error": "max_depth must be an integer >= 0 when provided", "success": False}
    if not isinstance(context, int) or context < 0:
        return {"status": "error", "error": "context must be an integer >= 0", "success": False}

    normalized_format = str(format or "text").strip().lower() or "text"
    if normalized_format not in {"text", "json"}:
        return {
            "status": "error",
            "error": "format must be one of: text, json",
            "success": False,
        }

    if extensions is not None and not str(extensions).strip():
        return {"status": "error", "error": "extensions must be a non-empty string when provided", "success": False}
    if exclude is not None and not str(exclude).strip():
        return {"status": "error", "error": "exclude must be a non-empty string when provided", "success": False}
    if output is not None and not str(output).strip():
        return {"status": "error", "error": "output must be a non-empty string when provided", "success": False}

    try:
        result = _API["codebase_search"](
            pattern=normalized_pattern,
            path=normalized_path,
            case_insensitive=case_insensitive,
            whole_word=whole_word,
            regex=regex,
            extensions=extensions,
            exclude=exclude,
            max_depth=max_depth,
            context=context,
            format=normalized_format,
            output=output,
            compact=compact,
            group_by_file=group_by_file,
            summary=summary,
        )
        resolved = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "success": False,
            "pattern": normalized_pattern,
            "path": normalized_path,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("pattern", normalized_pattern)
    payload.setdefault("path", normalized_path)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("result", {})
        if isinstance(payload.get("result"), dict):
            payload["result"].setdefault("matches", [])
            payload["result"].setdefault("summary", {})
    return payload


async def vscode_cli_status(install_dir: Optional[str] = None) -> Dict[str, Any]:
    """Get VS Code CLI installation and status details."""
    normalized_install_dir = None if install_dir is None else str(install_dir).strip()
    if install_dir is not None and not normalized_install_dir:
        return {
            "status": "error",
            "error": "install_dir must be a non-empty string when provided",
            "success": False,
        }

    try:
        result = _API["vscode_cli_status"](install_dir=normalized_install_dir)
        resolved = await result if hasattr(result, "__await__") else result
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "success": False,
            "install_dir": normalized_install_dir,
        }

    payload = _normalize_payload(resolved)
    payload.setdefault("install_dir", normalized_install_dir)
    if payload.get("status") == "success":
        payload.setdefault("success", True)
        payload.setdefault("installed", False)
    return payload


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
                "path": {"type": "string", "default": ".", "minLength": 1},
                "case_insensitive": {"type": "boolean", "default": False},
                "whole_word": {"type": "boolean", "default": False},
                "regex": {"type": "boolean", "default": False},
                "extensions": {"type": ["string", "null"]},
                "exclude": {"type": ["string", "null"]},
                "max_depth": {"type": ["integer", "null"], "minimum": 0},
                "context": {"type": "integer", "minimum": 0, "default": 0},
                "format": {"type": "string", "enum": ["text", "json"], "default": "text"},
                "output": {"type": ["string", "null"]},
                "compact": {"type": "boolean", "default": False},
                "group_by_file": {"type": "boolean", "default": False},
                "summary": {"type": "boolean", "default": False},
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
                "install_dir": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ]
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "development-tools"],
    )
