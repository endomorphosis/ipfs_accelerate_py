"""
IPFS Accelerate MCP Tools

.. deprecated::
    This package (``ipfs_accelerate_py.mcp.tools``) is a legacy compatibility
    shim.  All tools have been migrated to the unified canonical runtime at
    ``ipfs_accelerate_py.mcp_server.tools``.  New code should use the
    canonical categories directly (e.g. ``hardware_tools``, ``inference_tools``,
    ``ipfs_network_tools``, ``workflow_management_tools``, ``shared_tools``,
    ``enhanced_inference_tools``, etc.).

    ``register_all_tools`` continues to work for backward compatibility but
    emits a deprecation warning at runtime.
"""

import logging
import warnings
from typing import Any

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.tools")


def register_all_tools(mcp: Any, *, include_p2p_taskqueue_tools: bool = True) -> None:
    """Register all tools with the MCP server.

    .. deprecated::
        All tools have been migrated to ``ipfs_accelerate_py.mcp_server.tools``.
        This function is preserved for backward compatibility only.

    Args:
        mcp: MCP server instance.
    """
    warnings.warn(
        "ipfs_accelerate_py.mcp.tools.register_all_tools is deprecated. "
        "Use the canonical mcp_server tool categories instead "
        "(e.g. register_native_hardware_tools, register_native_inference_tools, etc.).",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.debug("Registering all tools with MCP server (legacy shim)")

    # Provide StandaloneMCP-like register_tool API when running under FastMCP.
    try:
        from ..fastmcp_compat import ensure_register_tool_compat
        ensure_register_tool_compat(mcp)
    except Exception as e:
        logger.debug(f"FastMCP compatibility shim not applied: {e}")

    try:
        # Register unified tools (new architecture) - wraps kit modules
        try:
            from ..unified_tools import register_unified_tools
            register_unified_tools(mcp)
            logger.info("Registered unified tools from kit modules")
        except Exception as e:
            logger.warning(f"Unified tools not registered: {e}")

        # Always register hardware tools (supports both Standalone and FastMCP styles)
        from .hardware import register_hardware_tools
        register_hardware_tools(mcp)

        # Register model tools (search, recommendations, details)
        try:
            from .models import register_model_tools
            register_model_tools(mcp)
            logger.debug("Registered model tools")
        except Exception as e:
            logger.warning(f"Model tools not registered: {e}")

        # If FastMCP-style decorators are available, register decorator-based tool modules
        if hasattr(mcp, "tool"):
            try:
                from .inference import register_tools as register_inference_tools
                register_inference_tools(mcp)
                logger.debug("Registered inference tools")
            except Exception as e:
                logger.warning(f"Inference tools not registered: {e}")

            try:
                from .endpoints import register_tools as register_endpoint_tools
                register_endpoint_tools(mcp)
                logger.debug("Registered endpoint tools")
            except Exception as e:
                logger.warning(f"Endpoint tools not registered: {e}")

            try:
                from .status import register_tools as register_status_tools
                register_status_tools(mcp)
                logger.debug("Registered status tools")
            except Exception as e:
                logger.warning(f"Status tools not registered: {e}")

            try:
                from .manifest import register_tools as register_manifest_tools
                register_manifest_tools(mcp)
                logger.debug("Registered manifest tools")
            except Exception as e:
                logger.warning(f"Manifest tools not registered: {e}")
            
            try:
                from .workflows import register_tools as register_workflow_tools
                register_workflow_tools(mcp)
                logger.debug("Registered workflow tools")
            except Exception as e:
                logger.warning(f"Workflow tools not registered: {e}")
            
            try:
                from .dashboard_data import register_tools as register_dashboard_tools
                register_dashboard_tools(mcp)
                logger.debug("Registered dashboard data tools")
            except Exception as e:
                logger.warning(f"Dashboard data tools not registered: {e}")
            
            try:
                from .github_tools import register_tools as register_github_tools
                register_github_tools(mcp)
                logger.debug("Registered GitHub CLI tools")
            except Exception as e:
                logger.warning(f"GitHub CLI tools not registered: {e}")

            # Register p2p TaskQueue tools
            if include_p2p_taskqueue_tools:
                try:
                    from .p2p_taskqueue import (
                        register_tools as register_p2p_taskqueue_tools,
                    )
                    register_p2p_taskqueue_tools(mcp)
                    logger.debug("Registered p2p TaskQueue tools")
                except Exception as e:
                    logger.warning(f"p2p TaskQueue tools not registered: {e}")

            # Register Docker tools
            try:
                from .docker_tools import register_docker_tools
                register_docker_tools(mcp)
                logger.debug("Registered Docker execution tools")
            except Exception as e:
                logger.warning(f"Docker tools not registered: {e}")
        else:
            logger.warning(
                "FastMCP decorators not available; only hardware and model tools registered in standalone mode"
            )

        logger.debug("All tools registered with MCP server")

    except Exception as e:
        logger.error(f"Error registering tools with MCP server: {e}")
        raise
