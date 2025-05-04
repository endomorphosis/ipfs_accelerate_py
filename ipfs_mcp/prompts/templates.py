"""
IPFS Accelerate MCP Prompts

This module defines prompt templates for interacting with the IPFS Accelerate MCP server.
"""
import logging
from fastmcp import FastMCP

logger = logging.getLogger("ipfs_accelerate_mcp.prompts")

def register_prompts(mcp: FastMCP) -> None:
    """
    Register prompt templates with the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    # System prompt for general interaction
    mcp.prompt(
        name="system",
        template="""You are an AI assistant for IPFS Accelerate, a framework for hardware-accelerated machine learning inference 
with distributed capabilities through IPFS. You can help users with:

1. Running ML models on appropriate hardware
2. Optimizing inference performance
3. Utilizing distributed computation across IPFS networks
4. Managing ML model resources
5. Understanding system hardware capabilities

You have access to system information resources and can run inference on available models.
For technical operations, you can use the provided tools to interact with the IPFS Accelerate system.

When users ask about running models or accessing accelerated infrastructure, guide them through 
the most appropriate approach based on their hardware and requirements.
"""
    )
    
    # Prompt for hardware optimization
    mcp.prompt(
        name="hardware-optimization",
        template="""I'll help you optimize your ML workload for the available hardware.

To provide the best recommendation, I need to gather some information:

1. What type of model are you running? (e.g., text generation, image classification, etc.)
2. What is the approximate size of your model? (parameters or file size)
3. What's your primary goal: speed, efficiency, or balanced approach?

Based on this information, I'll analyze your available hardware using IPFS Accelerate's 
hardware detection capabilities and recommend the best configuration.
"""
    )
    
    # Prompt for distributed inference
    mcp.prompt(
        name="distributed-inference",
        template="""I'll help you set up distributed inference across IPFS networks.

Distributed inference allows you to:
- Scale computation beyond a single machine
- Access specialized hardware remotely
- Balance load across multiple nodes

To get started, I'll need to know:
1. What model are you trying to run?
2. How many IPFS nodes do you have access to?
3. What are your performance requirements?

I can then guide you through configuring the distributed inference pipeline using 
IPFS Accelerate's built-in capabilities.
"""
    )
