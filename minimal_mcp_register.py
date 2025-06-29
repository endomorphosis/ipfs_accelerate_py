#!/usr/bin/env python3
"""
Test MCP Tool Registration

This script tests registering a single tool with the MCP server.
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    try:
        from mcp.server import register_tool, register_resource
        
        # Register hardware resources
        register_resource("system_info", "Information about the system")
        register_resource("accelerator_info", "Information about available hardware accelerators")
        register_resource("ipfs_nodes", "Information about connected IPFS nodes")
        
        # Register a test tool
        def test_ipfs_node_info():
            """Get information about the IPFS node."""
            return {
                "id": "QmNodeMockIPFSAccelerate",
                "version": "0.1.0",
                "protocol_version": "ipfs/0.1.0",
                "agent_version": "ipfs-accelerate/0.1.0",
                "success": True
            }
        
        register_tool("test_ipfs_node_info", "Test getting information about the IPFS node", test_ipfs_node_info, {})
        logger.info("Successfully registered test_ipfs_node_info tool")
        
        # Register model_inference tool
        def model_inference(model_name, input_data, endpoint_type=None):
            """Run inference on a model."""
            return {
                "model": model_name,
                "input": str(input_data)[:100],
                "endpoint_type": endpoint_type,
                "output": "This is a mock response from the model.",
                "is_mock": True
            }
        
        register_tool("model_inference", "Run inference on a model (mock)", model_inference, {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Name of the model"},
                "input_data": {"description": "Input data for inference"},
                "endpoint_type": {"type": "string", "description": "Endpoint type (optional)"}
            },
            "required": ["model_name", "input_data"]
        })
        logger.info("Successfully registered model_inference tool")
        
        # Register list_models tool
        def list_models():
            """Mock list of available models."""
            return {
                "local_models": ["bert-base-uncased", "gpt2", "t5-small"],
                "api_models": ["gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"],
                "libp2p_models": []
            }
        
        register_tool("list_models", "List available models (mock)", list_models, {})
        logger.info("Successfully registered list_models tool")
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
