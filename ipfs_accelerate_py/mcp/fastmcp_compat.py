"""FastMCP compatibility helpers.

This module provides a small adapter layer so code written against the older
`StandaloneMCP.register_tool(...)` API can also operate with `fastmcp.FastMCP`.

FastMCP registers tools via the `mcp.tool(...)` decorator and does not expose a
`register_tool` method.
"""

from __future__ import annotations

import logging
import inspect
from typing import Any, Callable, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.fastmcp_compat")


def ensure_register_tool_compat(mcp: Any) -> Any:
    """Ensure `mcp.register_tool(...)` exists.

    If the provided MCP instance already has `register_tool`, this is a no-op.
    If it looks like a FastMCP instance (has `tool`), a `register_tool` method
    is attached that delegates to `mcp.tool(...)`.

    Note: FastMCP does not currently accept an `input_schema` parameter. When
    adapting, schemas are ignored and FastMCP will infer inputs from function
    type hints.

    Returns:
        The same MCP instance, potentially patched.
    """

    if hasattr(mcp, "register_tool"):
        return mcp

    if not hasattr(mcp, "tool"):
        return mcp

    def _register_tool(
        *,
        name: str,
        function: Callable[..., Any],
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> Any:
        if input_schema is not None:
            logger.debug("Ignoring input_schema for FastMCP tool '%s'", name)

        try:
            decorator = mcp.tool(name=name, description=description)
            return decorator(function)
        except Exception as e:
            # Multiple registries may attempt to register overlapping tool names.
            # Prefer to be resilient here; re-raise unknown failures.
            message = str(e).lower()
            if "already" in message or "duplicate" in message or "exists" in message:
                logger.debug("Tool registration skipped for '%s': %s", name, e)
                return None
            raise

    try:
        setattr(mcp, "register_tool", _register_tool)
        logger.info("Attached register_tool compatibility shim to FastMCP")
    except Exception as e:
        logger.warning("Failed to attach register_tool shim: %s", e)

    return mcp


def ensure_register_resource_compat(mcp: Any) -> Any:
    """Ensure `mcp.register_resource(...)` exists.

    StandaloneMCP uses `register_resource(uri=..., function=..., description=...)`.
    FastMCP registers resources via the `mcp.resource(uri, ...)` decorator and
    requires URL-like URIs (e.g. `mcp://server_config`).

    For compatibility, bare URIs are automatically prefixed with `mcp://`.
    """

    if hasattr(mcp, "register_resource"):
        return mcp

    if not hasattr(mcp, "resource"):
        return mcp

    def _normalize_uri(uri: str) -> str:
        if "://" in uri:
            return uri
        return f"mcp://{uri.lstrip('/')}"

    def _register_resource(
        *,
        uri: str,
        function: Callable[..., Any],
        description: Optional[str] = None,
        **_: Any,
    ) -> Any:
        normalized = _normalize_uri(uri)

        # FastMCP requires URI templates with placeholders when the resource
        # function accepts parameters.
        try:
            sig = inspect.signature(inspect.unwrap(function))
            params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        except Exception:
            params = []

        if params and "{" not in normalized:
            for p in params:
                normalized = normalized.rstrip("/") + f"/{{{p.name}}}"
        try:
            decorator = mcp.resource(normalized, description=description)
            return decorator(function)
        except Exception as e:
            message = str(e).lower()
            if "already" in message or "duplicate" in message or "exists" in message:
                logger.debug("Resource registration skipped for '%s': %s", normalized, e)
                return None
            raise

    try:
        setattr(mcp, "register_resource", _register_resource)
        logger.info("Attached register_resource compatibility shim to FastMCP")
    except Exception as e:
        logger.warning("Failed to attach register_resource shim: %s", e)

    return mcp


def ensure_register_prompt_compat(mcp: Any) -> Any:
    """Ensure `mcp.register_prompt(...)` exists.

    StandaloneMCP exposes `register_prompt(name=..., template=..., description=..., input_schema=...)`.
    FastMCP registers prompts via the `mcp.prompt(...)` decorator and infers input
    schemas from function signatures.

    For compatibility, `template` is returned verbatim by a zero-arg function and
    `input_schema` is ignored.
    """

    if hasattr(mcp, "register_prompt"):
        return mcp

    if not hasattr(mcp, "prompt"):
        return mcp

    def _register_prompt(
        *,
        name: str,
        template: str,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> Any:
        if input_schema is not None:
            logger.debug("Ignoring input_schema for FastMCP prompt '%s'", name)

        def _prompt_fn() -> str:
            return template

        try:
            decorator = mcp.prompt(name=name, description=description)
            return decorator(_prompt_fn)
        except Exception as e:
            message = str(e).lower()
            if "already" in message or "duplicate" in message or "exists" in message:
                logger.debug("Prompt registration skipped for '%s': %s", name, e)
                return None
            raise

    try:
        setattr(mcp, "register_prompt", _register_prompt)
        logger.info("Attached register_prompt compatibility shim to FastMCP")
    except Exception as e:
        logger.warning("Failed to attach register_prompt shim: %s", e)

    return mcp
