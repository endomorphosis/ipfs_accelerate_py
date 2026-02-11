"""
IPFS Accelerate MCP Prompts

This package contains MCP prompt templates for IPFS Accelerate.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger("ipfs_accelerate_mcp.prompts")

# Export the prompt registration functions for easy access
from ipfs_accelerate_py.mcp.prompts.examples import register_prompts as register_example_prompts
from ipfs_accelerate_py.mcp.prompts.distributed_inference import register_prompts as register_distributed_inference_prompts

__all__ = [
    'register_example_prompts',
    'register_distributed_inference_prompts',
]

def register_all_prompts(mcp) -> None:
    """
    Register all prompt modules with the MCP server
    
    Args:
        mcp: The FastMCPServer instance
    """
    logger.info("Registering all prompt modules...")
    
    # Register each prompt module
    register_example_prompts(mcp)
    register_distributed_inference_prompts(mcp)
    
    logger.info("All prompt modules registered successfully")
