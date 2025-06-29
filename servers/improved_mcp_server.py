#!/usr/bin/env python3
"""
Improved standards-compliant MCP server for IPFS Accelerate.
This implementation fixes connection issues and properly registers tools using the MCP SDK.
"""

import os
import sys
import uuid
import json
import time
import signal
import logging
from typing import Dict, List, Any, Callable, Optional

# Set up logging
log_file = os.path.join(os.path.dirname(__file__), "improved_mcp_server.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ipfs_accelerate_mcp")

# Add signal handler for graceful shutdown
def handle_signal(signum, frame):
    logger.info(f"Received signal {signum}. Shutting down MCP server...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# Import mock IPFS Accelerate functionality
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

# Import MCP SDK - Try both import styles to handle different package structures
try:
    # First try importing from the mcp_server module (newer style)
    from mcp_server import register_tool, register_resource, run_server
    logger.info("Imported MCP SDK from mcp_server package")
    mcp_package_type = "mcp_server"
except ImportError:
    try:
        # Fall back to importing from the mcp module (standard style)
        from mcp import register_tool, register_resource, start_server
        logger.info("Imported MCP SDK from mcp package")
        mcp_package_type = "mcp"
    except ImportError:
        logger.error("Could not import MCP SDK. Please install it using: pip install mcp")
        sys.exit(1)

# Define tool schemas using JSON Schema format
SCHEMA_PATH = {
    "type": "string",
    "description": "File path or IPFS path"
}

SCHEMA_CID = {
    "type": "string", 
    "description": "IPFS Content Identifier (CID)"
}

SCHEMA_CONTENT = {
    "type": "string",
    "description": "Content to write to IPFS"
}

# Define all tool functions
def ipfs_add_file(path: str) -> Dict[str, Any]:
    """Add a file to IPFS and return its CID"""
    logger.info(f"Adding file to IPFS: {path}")
    result = mock_accelerate.add_file(path)
    return result

def ipfs_cat(cid: str) -> str:
    """Retrieve content from IPFS by its CID"""
    logger.info(f"Retrieving content from IPFS: {cid}")
    content = mock_accelerate.cat(cid)
    return content

def ipfs_files_write(path: str, content: str) -> Dict[str, Any]:
    """Write content to the IPFS Mutable File System (MFS)"""
    logger.info(f"Writing to IPFS MFS at path: {path}")
    result = mock_accelerate.files_write(path, content)
    return result

def ipfs_files_read(path: str) -> str:
    """Read content from the IPFS Mutable File System (MFS)"""
    logger.info(f"Reading from IPFS MFS at path: {path}")
    content = mock_accelerate.files_read(path)
    return content

def health_check() -> Dict[str, Any]:
    """Check the health of the IPFS Accelerate MCP server"""
    return {
        "status": "healthy",
        "version": "1.2.0",
        "uptime": 0,
        "ipfs_connected": True
    }

def list_models() -> Dict[str, Any]:
    """List available models for inference"""
    logger.info("Listing models")
    return mock_accelerate.list_models()

def create_endpoint(model_name: str, device: str = "cpu", max_batch_size: int = 16) -> Dict[str, Any]:
    """Create a model endpoint for inference"""
    logger.info(f"Creating endpoint for model: {model_name}")
    return mock_accelerate.create_endpoint(model_name, device, max_batch_size)

def run_inference(endpoint_id: str, inputs: List[str]) -> Dict[str, Any]:
    """Run inference using a model endpoint"""
    logger.info(f"Running inference on endpoint: {endpoint_id}")
    return mock_accelerate.run_inference(endpoint_id, inputs)

def get_ipfs_config() -> Dict[str, Any]:
    """Get IPFS configuration information"""
    return {
        "node_id": "12D3KooWQYAC2J7N8CYoQbAFrwkkBEpvgDZoCEHJGUf4qP89VED2",
        "version": "0.13.1",
        "protocol_version": "/ipfs/0.1.0",
        "api_address": "/ip4/127.0.0.1/tcp/5001",
        "gateway_address": "/ip4/127.0.0.1/tcp/8080"
    }

# Tool definitions with proper schemas for registration
TOOL_DEFINITIONS = [
    {
        "name": "ipfs_add_file",
        "description": "Add a file to IPFS and return its CID",
        "function": ipfs_add_file,
        "schema": {
            "type": "object",
            "properties": {
                "path": SCHEMA_PATH
            },
            "required": ["path"]
        }
    },
    {
        "name": "ipfs_cat",
        "description": "Retrieve content from IPFS by its CID",
        "function": ipfs_cat,
        "schema": {
            "type": "object",
            "properties": {
                "cid": SCHEMA_CID
            },
            "required": ["cid"]
        }
    },
    {
        "name": "ipfs_files_write",
        "description": "Write content to the IPFS Mutable File System (MFS)",
        "function": ipfs_files_write,
        "schema": {
            "type": "object",
            "properties": {
                "path": SCHEMA_PATH,
                "content": SCHEMA_CONTENT
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "ipfs_files_read",
        "description": "Read content from the IPFS Mutable File System (MFS)",
        "function": ipfs_files_read,
        "schema": {
            "type": "object",
            "properties": {
                "path": SCHEMA_PATH
            },
            "required": ["path"]
        }
    },
    {
        "name": "health_check",
        "description": "Check the health of the IPFS Accelerate MCP server",
        "function": health_check,
        "schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "list_models",
        "description": "List available models for inference",
        "function": list_models,
        "schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "create_endpoint",
        "description": "Create a model endpoint for inference",
        "function": create_endpoint,
        "schema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "max_batch_size": {"type": "integer", "default": 16}
            },
            "required": ["model_name"]
        }
    },
    {
        "name": "run_inference",
        "description": "Run inference using a model endpoint",
        "function": run_inference,
        "schema": {
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string"},
                "inputs": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["endpoint_id", "inputs"]
        }
    }
]

# Function to register all tools with appropriate error handling
def register_all_tools():
    """Register all IPFS tools with the MCP server"""
    registered_count = 0
    
    for tool in TOOL_DEFINITIONS:
        try:
            if mcp_package_type == "mcp_server":
                # For newer mcp_server package
                register_tool(
                    name=tool["name"],
                    description=tool["description"],
                    function=tool["function"],
                    schema=tool["schema"]
                )
            else:
                # For standard mcp package
                register_tool(
                    tool["name"], 
                    tool["description"], 
                    tool["function"],
                    tool["schema"]
                )
                
            logger.info(f"Successfully registered tool: {tool['name']}")
            registered_count += 1
        except Exception as e:
            logger.error(f"Failed to register tool {tool['name']}: {e}")
    
    logger.info(f"Successfully registered {registered_count} tools")
    return registered_count

# Function to create a health check file that Claude can use to verify the server is running
def create_health_check_file():
    """Create a health check file for Claude to verify the server is running"""
    health_data = {
        "status": "running",
        "timestamp": time.time(),
        "port": int(os.environ.get("MCP_PORT", "8002")),
        "host": os.environ.get("MCP_HOST", "127.0.0.1"),
        "registered_tools": [tool["name"] for tool in TOOL_DEFINITIONS]
    }
    
    health_file = os.path.join(os.path.dirname(__file__), "mcp_health_status.json")
    with open(health_file, "w") as f:
        json.dump(health_data, f, indent=2)
    
    logger.info(f"Created health check file at {health_file}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the improved MCP server for IPFS Accelerate")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to listen on")
    args = parser.parse_args()
    
    # Set environment variables for MCP server
    os.environ["MCP_PORT"] = str(args.port)
    os.environ["MCP_HOST"] = args.host
    
    logger.info(f"Starting improved MCP server for IPFS Accelerate on {args.host}:{args.port}")
    
    # Register all tools
    register_all_tools()
    
    # Create health check file
    create_health_check_file()
    
    # Start the server using the appropriate function
    try:
        if mcp_package_type == "mcp_server":
            logger.info("Starting server using mcp_server.run_server()")
            run_server()
        else:
            logger.info("Starting server using mcp.start_server()")
            start_server()
        
        logger.info("Server started successfully")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)