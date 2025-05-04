"""
Prompt registration and management for MCP server.

This module handles registration and management of prompts for the MCP server.
"""
import logging
from fastmcp import FastMCP
from .templates import register_prompts

logger = logging.getLogger("ipfs_accelerate_mcp.prompts")

def setup_prompts(mcp: FastMCP) -> None:
    """
    Set up prompts for the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    # Register all prompt templates
    register_prompts(mcp)
    
    logger.info("MCP prompts registered successfully")
