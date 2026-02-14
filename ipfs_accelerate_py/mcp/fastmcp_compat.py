"""FastMCP compatibility helpers.

This module provides a small adapter layer so code written against the older
`StandaloneMCP.register_tool(...)` API can also operate with `fastmcp.FastMCP`.

FastMCP registers tools via the `mcp.tool(...)` decorator and does not expose a
`register_tool` method.
"""

from __future__ import annotations

import logging
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
