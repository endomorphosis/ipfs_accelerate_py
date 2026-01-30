"""
Copilot SDK Tools for MCP Server

This module provides MCP tools for GitHub Copilot SDK integration,
enabling agentic AI features and programmatic Copilot access.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger("ipfs_accelerate_mcp.tools.copilot_sdk")

def _is_pytest() -> bool:
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


# Try imports with fallbacks
try:
    if _is_pytest():
        raise ImportError("Using mock MCP under pytest")
    from fastmcp import FastMCP
except ImportError:
    try:
        from ipfs_accelerate_py.mcp.mock_mcp import FastMCP
    except ImportError:
        from mock_mcp import FastMCP

# Import Copilot SDK operations
try:
    from ...shared import SharedCore, CopilotSDKOperations
    shared_core = SharedCore()
    copilot_sdk_ops = CopilotSDKOperations(shared_core)
    HAVE_COPILOT_SDK = True
except ImportError as e:
    logger.warning(f"Copilot SDK operations not available: {e}")
    HAVE_COPILOT_SDK = False
    copilot_sdk_ops = None


def register_copilot_sdk_tools(mcp: FastMCP) -> None:
    """Register Copilot SDK tools with the MCP server."""
    logger.info("Registering Copilot SDK tools")
    
    if not HAVE_COPILOT_SDK:
        logger.warning("Copilot SDK operations not available, skipping registration")
        return
    
    @mcp.tool()
    def copilot_sdk_create_session(
        model: Optional[str] = None,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new Copilot SDK session
        
        Args:
            model: Model to use (e.g., "gpt-4o", "gpt-5")
            streaming: Whether to enable streaming responses
            
        Returns:
            Session details including session_id
        """
        try:
            result = copilot_sdk_ops.create_session(
                model=model,
                streaming=streaming
            )
            result["tool"] = "copilot_sdk_create_session"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_sdk_create_session: {e}")
            return {
                "error": str(e),
                "tool": "copilot_sdk_create_session",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def copilot_sdk_send_message(
        session_id: str,
        prompt: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Send a message to a Copilot SDK session
        
        Args:
            session_id: Session ID from create_session
            prompt: Message to send
            use_cache: Whether to use cached results
            
        Returns:
            Response messages and metadata
        """
        try:
            result = copilot_sdk_ops.send_message(
                session_id=session_id,
                prompt=prompt,
                use_cache=use_cache
            )
            result["tool"] = "copilot_sdk_send_message"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_sdk_send_message: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "tool": "copilot_sdk_send_message",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def copilot_sdk_stream_message(
        session_id: str,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Stream a message response from a Copilot SDK session
        
        Args:
            session_id: Session ID from create_session
            prompt: Message to send
            
        Returns:
            Streaming response chunks and metadata
        """
        try:
            result = copilot_sdk_ops.stream_message(
                session_id=session_id,
                prompt=prompt
            )
            result["tool"] = "copilot_sdk_stream_message"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_sdk_stream_message: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "tool": "copilot_sdk_stream_message",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def copilot_sdk_destroy_session(session_id: str) -> Dict[str, Any]:
        """
        Destroy a Copilot SDK session
        
        Args:
            session_id: Session ID to destroy
            
        Returns:
            Success status
        """
        try:
            result = copilot_sdk_ops.destroy_session(session_id)
            result["tool"] = "copilot_sdk_destroy_session"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_sdk_destroy_session: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "tool": "copilot_sdk_destroy_session",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def copilot_sdk_list_sessions() -> Dict[str, Any]:
        """
        List all active Copilot SDK sessions
        
        Returns:
            List of active session IDs
        """
        try:
            result = copilot_sdk_ops.list_sessions()
            result["tool"] = "copilot_sdk_list_sessions"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_sdk_list_sessions: {e}")
            return {
                "error": str(e),
                "tool": "copilot_sdk_list_sessions",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def copilot_sdk_get_tools() -> Dict[str, Any]:
        """
        Get all registered Copilot SDK tools
        
        Returns:
            List of registered tool names
        """
        try:
            result = copilot_sdk_ops.get_tools()
            result["tool"] = "copilot_sdk_get_tools"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_sdk_get_tools: {e}")
            return {
                "error": str(e),
                "tool": "copilot_sdk_get_tools",
                "timestamp": time.time()
            }
    
    logger.info("Copilot SDK tools registered successfully")
