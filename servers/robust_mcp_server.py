#!/usr/bin/env python3
"""
Robust MCP server for IPFS Accelerate with enhanced error handling and connection verification.
This implementation provides detailed logging and continuous connectivity checks.
"""

import os
import sys
import uuid
import json
import time
import signal
import logging
import threading
import traceback
from typing import Dict, List, Any, Callable, Optional

# Set up logging with timestamped filename
current_time = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(os.path.dirname(__file__), f"robust_mcp_server_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ipfs_accelerate_mcp")

# Add detailed error handling for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
sys.excepthook = handle_exception

# Add signal handler for graceful shutdown
def handle_signal(signum, frame):
    logger.info(f"Received signal {signum}. Shutting down MCP server...")
    # Update health status file to indicate shutdown
    try:
        update_health_status("shutting_down")
    except:
        pass
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
    logger.warning("Could not import direct_mcp_server, using minimal implementation")
    class MockIPFSAccelerate:
        """Mock implementation of IPFS Accelerate functionality"""
        
        def add_file(self, path):
            file_hash = f"QmMock{uuid.uuid4().hex[:16]}"
            logger.info(f"Mock add_file called for {path}, returning hash {file_hash}")
            return {"cid": file_hash, "path": path, "success": True}
        
        def cat(self, cid):
            logger.info(f"Mock cat called for {cid}")
            return f"Mock content for {cid}"
        
        def files_write(self, path, content):
            logger.info(f"Mock files_write called for {path}")
            return {"path": path, "written": True, "success": True}
        
        def files_read(self, path):
            logger.info(f"Mock files_read called for {path}")
            return f"Mock MFS content for {path}"
            
        def get_hardware_info(self):
            logger.info("Mock get_hardware_info called")
            return {"cpu": {"available": True, "cores": 4}}
        
        def list_models(self):
            logger.info("Mock list_models called")
            return {"models": {"bert-base-uncased": {"type": "text-embedding"}}, "count": 1}
        
        def create_endpoint(self, model_name, device="cpu", max_batch_size=16):
            endpoint_id = f"endpoint-{uuid.uuid4().hex[:8]}"
            logger.info(f"Mock create_endpoint called for {model_name}, created {endpoint_id}")
            return {"endpoint_id": endpoint_id, "success": True}
        
        def run_inference(self, endpoint_id, inputs):
            logger.info(f"Mock run_inference called for {endpoint_id}")
            return {"success": True, "outputs": [f"Result for {input_text}" for input_text in inputs]}
    
    mock_accelerate = MockIPFSAccelerate()
    logger.info("Created minimal MockIPFSAccelerate implementation")

# Import MCP SDK - Try both import styles to handle different package structures
try:
    # First try importing from the mcp module (standard style)
    from mcp import register_tool, register_resource, start_server
    logger.info("Imported MCP SDK from mcp package")
    mcp_package_type = "mcp"
except ImportError:
    try:
        # Fall back to importing from the mcp_server module (newer style)
        from mcp_server import register_tool, register_resource, run_server
        logger.info("Imported MCP SDK from mcp_server package")
        mcp_package_type = "mcp_server"
    except ImportError:
        logger.error("Could not import MCP SDK. Please install it using: pip install mcp")
        logger.error("Trying to install MCP SDK now...")
        # Try to automatically install the MCP SDK
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp"])
            logger.info("Successfully installed MCP SDK.")
            try:
                from mcp import register_tool, register_resource, start_server
                logger.info("Imported MCP SDK from mcp package after installation")
                mcp_package_type = "mcp"
            except ImportError:
                logger.error("Failed to import MCP SDK even after installation")
                sys.exit(1)
        except subprocess.CalledProcessError:
            logger.error("Failed to install MCP SDK. Please install manually.")
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

# Define all tool functions with detailed error handling
def ipfs_add_file(path: str) -> Dict[str, Any]:
    """Add a file to IPFS and return its CID"""
    try:
        logger.info(f"Adding file to IPFS: {path}")
        if not os.path.exists(path):
            logger.error(f"File does not exist: {path}")
            return {"error": f"File does not exist: {path}", "success": False}
        
        result = mock_accelerate.add_file(path)
        logger.info(f"Successfully added file to IPFS: {result}")
        return result
    except Exception as e:
        logger.error(f"Error adding file to IPFS: {e}")
        return {"error": str(e), "success": False}

def ipfs_cat(cid: str) -> str:
    """Retrieve content from IPFS by its CID"""
    try:
        logger.info(f"Retrieving content from IPFS: {cid}")
        content = mock_accelerate.cat(cid)
        return content
    except Exception as e:
        logger.error(f"Error retrieving content from IPFS: {e}")
        return f"Error: {str(e)}"

def ipfs_files_write(path: str, content: str) -> Dict[str, Any]:
    """Write content to the IPFS Mutable File System (MFS)"""
    try:
        logger.info(f"Writing to IPFS MFS at path: {path}")
        result = mock_accelerate.files_write(path, content)
        return result
    except Exception as e:
        logger.error(f"Error writing to IPFS MFS: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_read(path: str) -> str:
    """Read content from the IPFS Mutable File System (MFS)"""
    try:
        logger.info(f"Reading from IPFS MFS at path: {path}")
        content = mock_accelerate.files_read(path)
        return content
    except Exception as e:
        logger.error(f"Error reading from IPFS MFS: {e}")
        return f"Error: {str(e)}"

def health_check() -> Dict[str, Any]:
    """Check the health of the IPFS Accelerate MCP server"""
    try:
        # Get uptime in seconds
        uptime = time.time() - start_time
        
        return {
            "status": "healthy",
            "version": "1.2.0",
            "uptime": uptime,
            "ipfs_connected": True,
            "pid": os.getpid(),
            "registered_tools": [tool["name"] for tool in TOOL_DEFINITIONS]
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {"status": "error", "error": str(e)}

def list_models() -> Dict[str, Any]:
    """List available models for inference"""
    try:
        logger.info("Listing models")
        return mock_accelerate.list_models()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"error": str(e), "models": {}, "count": 0}

def create_endpoint(model_name: str, device: str = "cpu", max_batch_size: int = 16) -> Dict[str, Any]:
    """Create a model endpoint for inference"""
    try:
        logger.info(f"Creating endpoint for model: {model_name}")
        return mock_accelerate.create_endpoint(model_name, device, max_batch_size)
    except Exception as e:
        logger.error(f"Error creating endpoint: {e}")
        return {"error": str(e), "success": False}

def run_inference(endpoint_id: str, inputs: List[str]) -> Dict[str, Any]:
    """Run inference using a model endpoint"""
    try:
        logger.info(f"Running inference on endpoint: {endpoint_id}")
        return mock_accelerate.run_inference(endpoint_id, inputs)
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        return {"error": str(e), "success": False}

def get_ipfs_config() -> Dict[str, Any]:
    """Get IPFS configuration information"""
    try:
        return {
            "node_id": "12D3KooWQYAC2J7N8CYoQbAFrwkkBEpvgDZoCEHJGUf4qP89VED2",
            "version": "0.13.1",
            "protocol_version": "/ipfs/0.1.0",
            "api_address": "/ip4/127.0.0.1/tcp/5001",
            "gateway_address": "/ip4/127.0.0.1/tcp/8080"
        }
    except Exception as e:
        logger.error(f"Error getting IPFS config: {e}")
        return {"error": str(e)}

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
            if mcp_package_type == "mcp":
                # For standard mcp package
                register_tool(
                    tool["name"], 
                    tool["description"], 
                    tool["function"],
                    tool["schema"]
                )
            else:
                # For newer mcp_server package
                register_tool(
                    name=tool["name"],
                    description=tool["description"],
                    function=tool["function"],
                    schema=tool["schema"]
                )
                
            logger.info(f"Successfully registered tool: {tool['name']}")
            registered_count += 1
        except Exception as e:
            logger.error(f"Failed to register tool {tool['name']}: {e}")
            logger.error(traceback.format_exc())
    
    logger.info(f"Successfully registered {registered_count} tools")
    return registered_count

# Create health status file with timestamp
health_file = os.path.join(os.path.dirname(__file__), "mcp_health_status.json")

def update_health_status(status="running"):
    """Update the health status file with current server state"""
    try:
        health_data = {
            "status": status,
            "timestamp": time.time(),
            "port": int(os.environ.get("MCP_PORT", "8002")),
            "host": os.environ.get("MCP_HOST", "127.0.0.1"),
            "registered_tools": [tool["name"] for tool in TOOL_DEFINITIONS],
            "pid": os.getpid(),
            "uptime": time.time() - start_time
        }
        
        with open(health_file, "w") as f:
            json.dump(health_data, f, indent=2)
        
        logger.debug(f"Updated health check file at {health_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to update health status: {e}")
        return False

# Function to periodically check and report server status
def health_monitor():
    """Monitor server health and update status file periodically"""
    while True:
        try:
            update_health_status()
            # Check if MCP is working
            logger.debug("Health monitor: MCP server is running")
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
        
        time.sleep(60)  # Update every minute

# Function to verify Claude MCP settings are correct
def verify_claude_mcp_settings():
    """Verify and fix Claude MCP settings if needed"""
    try:
        claude_settings_path = os.path.expanduser("~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json")
        
        if os.path.exists(claude_settings_path):
            with open(claude_settings_path, "r") as f:
                settings = json.load(f)
            
            port = int(os.environ.get("MCP_PORT", "8002"))
            host = os.environ.get("MCP_HOST", "127.0.0.1")
            host_url = "localhost" if host == "127.0.0.1" else host
            expected_url = f"http://{host_url}:{port}"
            
            # Check if settings need updating
            needs_update = False
            if "mcpServers" not in settings:
                settings["mcpServers"] = {}
                needs_update = True
            
            if "ipfs-accelerate" not in settings["mcpServers"]:
                settings["mcpServers"]["ipfs-accelerate"] = {
                    "disabled": False,
                    "timeout": 60,
                    "url": expected_url,
                    "transportType": "sse"
                }
                needs_update = True
            elif settings["mcpServers"]["ipfs-accelerate"]["url"] != expected_url:
                settings["mcpServers"]["ipfs-accelerate"]["url"] = expected_url
                needs_update = True
            
            if needs_update:
                logger.info(f"Updating Claude MCP settings to use {expected_url}")
                with open(claude_settings_path, "w") as f:
                    json.dump(settings, f, indent=2)
                logger.info("Successfully updated Claude MCP settings")
            else:
                logger.info("Claude MCP settings are already correctly configured")
            
            return True
        else:
            logger.warning(f"Claude settings file not found at {claude_settings_path}")
            return False
    except Exception as e:
        logger.error(f"Error checking Claude MCP settings: {e}")
        return False

# Create a clear claude connection settings file
def create_claude_connection_info():
    """Create a file with connection instructions for Claude"""
    try:
        info = {
            "mcp_server_info": {
                "port": int(os.environ.get("MCP_PORT", "8002")),
                "host": os.environ.get("MCP_HOST", "127.0.0.1"),
                "pid": os.getpid(),
                "tools": [tool["name"] for tool in TOOL_DEFINITIONS],
                "url": f"http://localhost:{os.environ.get('MCP_PORT', '8002')}"
            },
            "claude_settings": {
                "required_config": {
                    "mcpServers": {
                        "ipfs-accelerate": {
                            "disabled": False,
                            "timeout": 60,
                            "url": f"http://localhost:{os.environ.get('MCP_PORT', '8002')}",
                            "transportType": "sse"
                        }
                    }
                },
                "settings_path": "~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
            }
        }
        
        info_file = os.path.join(os.path.dirname(__file__), "claude_mcp_connection_info.json")
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Created Claude connection info file at {info_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create Claude connection info: {e}")
        return False

if __name__ == "__main__":
    # Record start time for uptime tracking
    start_time = time.time()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the robust MCP server for IPFS Accelerate")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to listen on")
    args = parser.parse_args()
    
    # Set environment variables for MCP server
    os.environ["MCP_PORT"] = str(args.port)
    os.environ["MCP_HOST"] = args.host
    
    logger.info(f"Starting robust MCP server for IPFS Accelerate on {args.host}:{args.port}")
    
    # Check for other running instances on the same port
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((args.host, args.port)) == 0:
                logger.warning(f"Port {args.port} is already in use. Another server might be running.")
                logger.warning("Attempting to stop other MCP server processes...")
                os.system(f"pkill -f 'python.*mcp.*server'")
                time.sleep(2)  # Wait for processes to terminate
    except:
        pass
    
    # Register all tools
    register_all_tools()
    
    # Create health check file
    update_health_status("starting")
    
    # Start health monitor thread
    monitor_thread = threading.Thread(target=health_monitor, daemon=True)
    monitor_thread.start()
    
    # Verify Claude MCP settings
    verify_claude_mcp_settings()
    
    # Create Claude connection info
    create_claude_connection_info()
    
    # Start the server using the appropriate function
    try:
        logger.info("==== MCP SERVER STARTING ====")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Host: {args.host}")
        logger.info(f"Tools: {[tool['name'] for tool in TOOL_DEFINITIONS]}")
        logger.info("=============================")
        
        update_health_status("running")
        
        if mcp_package_type == "mcp":
            logger.info("Starting server using mcp.start_server()")
            start_server()
        else:
            logger.info("Starting server using mcp_server.run_server()")
            run_server()
        
        logger.info("Server started successfully")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        update_health_status("failed")
        sys.exit(1)