#!/usr/bin/env python3
"""
Tool registration helper for IPFS Accelerate MCP server.
This script ensures that all tools are properly registered with the MCP server.
"""

import os
import sys
import logging
import inspect
from typing import Callable, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_tools")

# Try to import mcp modules
try:
    import mcp
    from mcp import register_tool, register_resource
    logger.info("Successfully imported MCP modules")
except ImportError as e:
    logger.error(f"Failed to import MCP modules: {e}")
    logger.error("Please install MCP using: pip install model-context-protocol")
    sys.exit(1)

# Define IPFS Accelerate tools that need to be registered
def get_hardware_info() -> dict:
    """Get hardware information about the system"""
    import os
    import platform
    
    hardware_info = {
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "distribution": platform.platform(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count()
        },
        "accelerators": {
            "cpu": {"available": True}
        }
    }
    
    # Check for CUDA availability
    try:
        import torch
        hardware_info["accelerators"]["cuda"] = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            hardware_info["accelerators"]["cuda"]["device_count"] = torch.cuda.device_count()
            hardware_info["accelerators"]["cuda"]["device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        logger.warning("PyTorch not available, cannot check CUDA")
    
    return hardware_info

def register_ipfs_tools() -> bool:
    """Register all IPFS Accelerate tools with MCP."""
    try:
        # Try to import the mock IPFS Accelerate functionality
        try:
            import direct_mcp_server
            mock_accelerate = direct_mcp_server.MockIPFSAccelerate()
            logger.info("Imported MockIPFSAccelerate from direct_mcp_server.py")
        except ImportError:
            # Define a minimal version if the import fails
            class MockIPFSAccelerate:
                """Mock implementation of IPFS Accelerate functionality"""
                def add_file(self, path):
                    import uuid
                    file_hash = f"QmMock{uuid.uuid4().hex[:16]}"
                    return {"cid": file_hash, "path": path, "success": True}
                
                def cat(self, cid):
                    return f"Mock content for {cid}"
                
                def files_write(self, path, content):
                    return {"path": path, "written": True, "success": True}
                
                def files_read(self, path):
                    return f"Mock MFS content for {path}"
                
                def list_models(self):
                    return {"models": {"bert-base-uncased": {"type": "text-embedding"}}, "count": 1}
                
                def create_endpoint(self, model_name, device="cpu", max_batch_size=16):
                    import uuid
                    return {"endpoint_id": f"endpoint-{uuid.uuid4().hex[:8]}", "success": True}
                
                def run_inference(self, endpoint_id, inputs):
                    return {"success": True, "outputs": [f"Result for {input_text}" for input_text in inputs]}
            
            mock_accelerate = MockIPFSAccelerate()
            logger.info("Created minimal MockIPFSAccelerate implementation")

        # Define the functions that use the mock_accelerate object
        def ipfs_add_file(path: str) -> dict:
            """Add a file to IPFS and return its CID"""
            logger.info(f"Adding file to IPFS: {path}")
            result = mock_accelerate.add_file(path)
            return result

        def ipfs_cat(cid: str) -> str:
            """Retrieve content from IPFS by its CID"""
            logger.info(f"Retrieving content from IPFS: {cid}")
            content = mock_accelerate.cat(cid)
            return content

        def ipfs_files_write(path: str, content: str) -> dict:
            """Write content to the IPFS Mutable File System (MFS)"""
            logger.info(f"Writing to IPFS MFS at path: {path}")
            result = mock_accelerate.files_write(path, content)
            return result

        def ipfs_files_read(path: str) -> str:
            """Read content from the IPFS Mutable File System (MFS)"""
            logger.info(f"Reading from IPFS MFS at path: {path}")
            content = mock_accelerate.files_read(path)
            return content

        def health_check() -> dict:
            """Check the health of the IPFS Accelerate MCP server"""
            return {
                "status": "healthy",
                "version": "1.1.0",
                "uptime": 0,  # Not tracking uptime in this implementation
                "ipfs_connected": False  # In mock mode, IPFS is never truly connected
            }

        def list_models() -> dict:
            """List available models for inference"""
            logger.info("Listing models")
            return mock_accelerate.list_models()

        def create_endpoint(model_name: str, device: str = "cpu", max_batch_size: int = 16) -> dict:
            """Create a model endpoint for inference"""
            logger.info(f"Creating endpoint for model: {model_name}")
            return mock_accelerate.create_endpoint(model_name, device, max_batch_size)

        def run_inference(endpoint_id: str, inputs: list) -> dict:
            """Run inference using a model endpoint"""
            logger.info(f"Running inference on endpoint: {endpoint_id}")
            return mock_accelerate.run_inference(endpoint_id, inputs)

        # Define the tools to register
        tools_to_register = [
            {"name": "ipfs_add_file", "description": "Add a file to IPFS and return its CID", "func": ipfs_add_file},
            {"name": "ipfs_cat", "description": "Retrieve content from IPFS by its CID", "func": ipfs_cat},
            {"name": "ipfs_files_write", "description": "Write content to the IPFS Mutable File System (MFS)", "func": ipfs_files_write},
            {"name": "ipfs_files_read", "description": "Read content from the IPFS Mutable File System (MFS)", "func": ipfs_files_read},
            {"name": "health_check", "description": "Check the health of the IPFS Accelerate MCP server", "func": health_check},
            {"name": "get_hardware_info", "description": "Get hardware information about the system", "func": get_hardware_info},
            {"name": "list_models", "description": "List available models for inference", "func": list_models},
            {"name": "create_endpoint", "description": "Create a model endpoint for inference", "func": create_endpoint},
            {"name": "run_inference", "description": "Run inference using a model endpoint", "func": run_inference}
        ]

        # Register all tools
        for tool in tools_to_register:
            try:
                register_tool(tool["name"], tool["description"], tool["func"])
                logger.info(f"Successfully registered tool: {tool['name']}")
            except Exception as e:
                logger.error(f"Failed to register tool {tool['name']}: {e}")
                return False

        # Register a sample resource
        try:
            def get_ipfs_config() -> str:
                """Get IPFS configuration information"""
                return """
                IPFS Configuration:
                - Node ID: 12D3KooWQYAC2J7N8CYoQbAFrwkkBEpvgDZoCEHJGUf4qP89VED2
                - Version: 0.13.1
                - Protocol Version: /ipfs/0.1.0
                """
            
            register_resource("ipfs://config", "IPFS Configuration Information", get_ipfs_config)
            logger.info("Successfully registered resources")
        except Exception as e:
            logger.error(f"Failed to register resource: {e}")
            # Non-fatal error for resources

        return True

    except Exception as e:
        logger.error(f"Error during tool registration: {e}")
        return False

def main():
    """Main function to register tools."""
    logger.info("Registering IPFS Accelerate tools with MCP server...")
    success = register_ipfs_tools()
    
    if success:
        logger.info("All tools registered successfully")
        return 0
    else:
        logger.error("Failed to register some tools")
        return 1

if __name__ == "__main__":
    sys.exit(main())
