"""
Copilot CLI Tools for MCP Server

This module provides MCP tools for GitHub Copilot CLI integration,
enabling AI-assisted development features.
"""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.tools.copilot")

# Try imports with fallbacks
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        from mcp.mock_mcp import FastMCP

# Import Copilot operations
try:
    from ...shared import SharedCore, CopilotOperations
    shared_core = SharedCore()
    copilot_ops = CopilotOperations(shared_core)
    HAVE_COPILOT = True
except ImportError as e:
    logger.warning(f"Copilot operations not available: {e}")
    HAVE_COPILOT = False
    copilot_ops = None


def register_copilot_tools(mcp: FastMCP) -> None:
    """Register Copilot CLI tools with the MCP server."""
    logger.info("Registering Copilot CLI tools")
    
    if not HAVE_COPILOT:
        logger.warning("Copilot operations not available, skipping registration")
        return
    
    @mcp.tool()
    def copilot_suggest_command(
        prompt: str,
        shell: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get command suggestions from GitHub Copilot
        
        Args:
            prompt: Natural language description of desired command
            shell: Shell type (bash, zsh, powershell, etc.)
            
        Returns:
            Suggested command and metadata
        """
        try:
            result = copilot_ops.suggest_command(prompt, shell=shell)
            result["tool"] = "copilot_suggest_command"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_suggest_command: {e}")
            return {
                "error": str(e),
                "prompt": prompt,
                "tool": "copilot_suggest_command",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def copilot_explain_command(command: str) -> Dict[str, Any]:
        """
        Get an explanation for a command from GitHub Copilot
        
        Args:
            command: Command to explain
            
        Returns:
            Command explanation and metadata
        """
        try:
            result = copilot_ops.explain_command(command)
            result["tool"] = "copilot_explain_command"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_explain_command: {e}")
            return {
                "error": str(e),
                "command": command,
                "tool": "copilot_explain_command",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def copilot_suggest_git_command(prompt: str) -> Dict[str, Any]:
        """
        Get Git command suggestions from GitHub Copilot
        
        Args:
            prompt: Natural language description of desired Git operation
            
        Returns:
            Suggested Git command and metadata
        """
        try:
            result = copilot_ops.suggest_git_command(prompt)
            result["tool"] = "copilot_suggest_git_command"
            return result
        except Exception as e:
            logger.error(f"Error in copilot_suggest_git_command: {e}")
            return {
                "error": str(e),
                "prompt": prompt,
                "tool": "copilot_suggest_git_command",
                "timestamp": time.time()
            }
    
    logger.info("Copilot CLI tools registered successfully")
