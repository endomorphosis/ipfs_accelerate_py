#!/usr/bin/env python3
"""
Standards-compliant MCP server for IPFS Accelerate using the correct MCP package structure.
This implementation follows the proper MCP protocol that Claude expects.
"""

import os
import sys
import uuid
import logging
import inspect
import platform
import traceback

# Import the correct MCP modules based on the actual package structure
import mcp
from mcp import register_tool, register_resource, start_server

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp")

# Import the mock IPFS Accelerate functionality
try:
    import direct_mcp_server
    mock_accelerate = direct_mcp_server.MockIPFSAccelerate()
    logger.info("Imported MockIPFSAccelerate from direct_mcp_server.py")
except ImportError:
    # Define a minimal version if the import fails
    class MockIPFSAccelerate:
        """Mock implementation of IPFS Accelerate functionality"""
        
        def add_file(self, path):
            file_hash = f"QmMock{uuid.uuid4().hex[:16]}"
            return {"cid": file_hash, "path": path, "success": True}
        
        def cat(self, cid):
            return f"Mock content for {cid}"
        
        def files_write(self, path, content):
            return {"path": path, "written": True, "success": True}
        
        def files_read(self, path):
            return f"Mock MFS content for {path}"
            
        def get_hardware_info(self):
            return {"cpu": {"available": True, "cores": 4}}
        
        def list_models(self):
            return {"models": {"bert-base-uncased": {"type": "text-embedding"}}, "count": 1}
        
        def create_endpoint(self, model_name, device="cpu", max_batch_size=16):
            return {"endpoint_id": f"endpoint-{uuid.uuid4().hex[:8]}", "success": True}
        
        def run_inference(self, endpoint_id, inputs):
            return {"success": True, "outputs": [f"Result for {input_text}" for input_text in inputs]}
    
    mock_accelerate = MockIPFSAccelerate()
    logger.info("Created minimal MockIPFSAccelerate implementation")

# Try to import tool_registration module
try:
    import tool_registration
    logger.info("Found tool_registration module")
except ImportError:
    logger.warning("Could not import tool_registration module, will register tools manually")


# Define the IPFS tools using the correct MCP registration format
# Get the actual function signature for register_tool
try:
    sig = inspect.signature(register_tool)
    logger.info(f"register_tool signature: {sig}")
except Exception as e:
    logger.error(f"Failed to get signature: {e}")


# Define all our functions first
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


def get_hardware_info() -> dict:
    """Get hardware information about the system"""
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
    return hardware_info


def get_custom_hardware_info() -> dict:
    """Get hardware information for model acceleration"""
    logger.info("Getting hardware information")
    return mock_accelerate.get_hardware_info()


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


# Now register the tools using the correct approach with all three parameters
try:
    # Register tools with name, description, and function
    register_tool("ipfs_add_file", "Add a file to IPFS and return its CID", ipfs_add_file)
    register_tool("ipfs_cat", "Retrieve content from IPFS by its CID", ipfs_cat)
    register_tool("ipfs_files_write", "Write content to the IPFS Mutable File System (MFS)", ipfs_files_write)
    register_tool("ipfs_files_read", "Read content from the IPFS Mutable File System (MFS)", ipfs_files_read)
    register_tool("health_check", "Check the health of the IPFS Accelerate MCP server", health_check)
    # Register the new standard hardware_info tool that matches the client's expectation
    register_tool("get_hardware_info", "Get hardware information about the system", get_hardware_info)
    # Keep the custom hardware info tool for backward compatibility
    register_tool("custom_hardware_info", "Get custom hardware information for model acceleration", get_custom_hardware_info)
    register_tool("list_models", "List available models for inference", list_models)
    register_tool("create_endpoint", "Create a model endpoint for inference", create_endpoint)
    register_tool("run_inference", "Run inference using a model endpoint", run_inference)
    
    logger.info("Successfully registered tools")
except Exception as e:
    logger.error(f"Error registering tools: {e}")
    sys.exit(1)


# Add a resource to expose IPFS configuration
def get_ipfs_config() -> str:
    """Get IPFS configuration information"""
    return """
    IPFS Configuration:
    - Node ID: 12D3KooWQYAC2J7N8CYoQbAFrwkkBEpvgDZoCEHJGUf4qP89VED2
    - Version: 0.13.1
    - Protocol Version: /ipfs/0.1.0
    """

# Check register_resource signature
try:
    sig_res = inspect.signature(register_resource)
    logger.info(f"register_resource signature: {sig_res}")
    # Register resource with the correct signature based on inspection
    register_resource("ipfs://config", "IPFS Configuration Information", get_ipfs_config)
    logger.info("Successfully registered resources")
except Exception as e:
    logger.error(f"Error registering resources: {e}")
    # Non-fatal error, continue without resources


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the standards-compliant IPFS Accelerate MCP server")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to listen on")
    args = parser.parse_args()
    
    # Set port and host for MCP server
    os.environ["MCP_PORT"] = str(args.port)
    os.environ["MCP_HOST"] = args.host
    
    logger.info(f"Starting standards-compliant IPFS Accelerate MCP server on {args.host}:{args.port}")
    
    # Use uvicorn directly to ensure we bind to the right host/port
    try:
        import uvicorn
        from mcp.server import app
        
        # First check if start_server has the port and host parameters directly
        start_sig = inspect.signature(start_server)
        logger.info(f"start_server signature: {start_sig}")
        
        if 'host' in start_sig.parameters and 'port' in start_sig.parameters:
            # Call with explicit host and port
            logger.info(f"Starting server with explicit host and port: {args.host}:{args.port}")
            if 'name' in start_sig.parameters:
                start_server(
                    host=args.host, 
                    port=args.port, 
                    name="IPFS Accelerate MCP Server", 
                    description="Standards-compliant MCP server for IPFS Accelerate"
                )
            else:
                start_server(host=args.host, port=args.port)
        else:
            # Fall back to uvicorn directly
            logger.info(f"Falling back to direct uvicorn startup on {args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        traceback.print_exc()
        sys.exit(1)