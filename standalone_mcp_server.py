#!/usr/bin/env python3
"""
Standalone MCP Server with IPFS Tools

This script starts a standalone MCP server that exposes IPFS tools.
It uses a direct implementation approach to avoid dependency issues.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, Any, List, Optional, Callable, Union
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("standalone_mcp")

# Server info
SERVER_INFO = {
    "server_name": "ipfs-accelerate-mcp",
    "description": "IPFS Accelerate MCP Server",
    "version": "0.1.0",
    "mcp_version": "0.1.0"
}

# Tools
TOOLS = {}

# Resources
RESOURCES = {
    "system_info": {
        "description": "Information about the system"
    },
    "accelerator_info": {
        "description": "Information about available hardware accelerators"
    },
    "ipfs_nodes": {
        "description": "Information about connected IPFS nodes"
    },
    "models": {
        "description": "Information about available models"
    }
}

class MockIPFSAccelerate:
    """Mock implementation of the IPFS Accelerate class for testing."""
    
    def __init__(self):
        self.files = {}
        self.endpoints = {
            "local_endpoints": {"gpt2": {}, "bert": {}},
            "api_endpoints": {"openai": {}, "huggingface": {}},
            "libp2p_endpoints": {}
        }
    
    def add_file(self, path: str) -> Dict[str, Any]:
        """Add a file to IPFS."""
        import hashlib
        import random
        
        try:
            if not os.path.exists(path):
                return {"error": "File not found", "success": False}
            
            # Generate a mock CID
            with open(path, 'rb') as f:
                content = f.read()
                hash_obj = hashlib.sha256(content)
                cid = f"QmPy{hash_obj.hexdigest()[:16]}"
                
            # Store the file
            self.files[cid] = {
                "path": path,
                "size": len(content),
                "content": content
            }
            
            return {
                "cid": cid,
                "size": len(content),
                "path": path,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def cat(self, cid: str) -> Dict[str, Any]:
        """Get the content of a file from IPFS."""
        if cid not in self.files:
            return {"error": "CID not found", "success": False}
        
        return {
            "content": self.files[cid]["content"].decode('utf-8', errors='replace'),
            "size": self.files[cid]["size"],
            "success": True
        }
    
    def get(self, cid: str, output_path: str) -> Dict[str, Any]:
        """Get a file from IPFS and save it to a local path."""
        if cid not in self.files:
            return {"error": "CID not found", "success": False}
        
        try:
            with open(output_path, 'wb') as f:
                f.write(self.files[cid]["content"])
            
            return {
                "path": output_path,
                "size": self.files[cid]["size"],
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def process(self, model_name: str, input_data: Any, endpoint_type: Optional[str] = None) -> Dict[str, Any]:
        """Process data using a model."""
        return {
            "model": model_name,
            "input": str(input_data)[:100] if input_data else "",
            "endpoint_type": endpoint_type,
            "output": "This is a mock response from the model.",
            "status": "success"
        }
    
    async def init_endpoints(self, models: List[str]) -> Dict[str, Any]:
        """Initialize endpoints for models."""
        return {
            "initialized": models,
            "status": "success" 
        }
    
    def node_info(self) -> Dict[str, Any]:
        """Get information about the IPFS node."""
        return {
            "id": "QmNodeMockIPFSAccelerate",
            "version": "0.1.0",
            "protocol_version": "ipfs/0.1.0",
            "agent_version": "ipfs-accelerate/0.1.0",
            "success": True
        }

# Register tools
def register_tool(name, description, handler, schema=None):
    """Register a tool with the MCP server."""
    if schema is None:
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    TOOLS[name] = {
        "description": description,
        "schema": schema,
        "handler": handler
    }
    logger.info(f"Registered tool: {name}")

def register_resource(name, description):
    """Register a resource with the MCP server."""
    RESOURCES[name] = {
        "description": description
    }
    logger.info(f"Registered resource: {name}")

def register_hardware_tools():
    """Register hardware-related tools."""
    # Import hardware detection module if available
    try:
        import hardware_detection
    except ImportError:
        hardware_detection = None
    
    def get_hardware_info():
        """Get hardware information about the system."""
        try:
            if hardware_detection and hasattr(hardware_detection, "detect_all_hardware"):
                return hardware_detection.detect_all_hardware()
            elif hardware_detection and hasattr(hardware_detection, "detect_hardware"):
                return hardware_detection.detect_hardware()
            else:
                # Basic hardware info
                import platform
                import psutil
                return {
                    "system": {
                        "os": platform.system(),
                        "os_version": platform.version(),
                        "distribution": platform.platform(),
                        "architecture": platform.machine(),
                        "python_version": platform.python_version(),
                        "processor": platform.processor(),
                        "memory_total": round(psutil.virtual_memory().total / (1024**3), 2),
                        "memory_available": round(psutil.virtual_memory().available / (1024**3), 2),
                        "cpu": {
                            "cores_physical": psutil.cpu_count(logical=False),
                            "cores_logical": psutil.cpu_count(logical=True)
                        }
                    },
                    "accelerators": {
                        "cuda": {
                            "available": False,
                            "version": None,
                            "devices": []
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Error in get_hardware_info: {str(e)}")
            return {"error": str(e)}
    
    register_tool("get_hardware_info", "Get hardware information about the system", get_hardware_info, {})
    logger.info("Registered hardware tools")

def register_ipfs_tools(accel):
    """Register IPFS-related tools."""
    # IPFS Add File
    def ipfs_add_file(path):
        """Add a file to IPFS."""
        try:
            return accel.add_file(path)
        except Exception as e:
            logger.error(f"Error in ipfs_add_file: {str(e)}")
            return {"error": str(e), "success": False}
    
    register_tool("ipfs_add_file", "Add a file to IPFS", ipfs_add_file, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"}
        },
        "required": ["path"]
    })
    
    # IPFS Node Info
    def ipfs_node_info():
        """Get information about the IPFS node."""
        try:
            return accel.node_info()
        except Exception as e:
            logger.error(f"Error in ipfs_node_info: {str(e)}")
            return {"error": str(e), "success": False}
    
    register_tool("ipfs_node_info", "Get information about the IPFS node", ipfs_node_info, {})
    
    # IPFS Cat
    def ipfs_cat(cid):
        """Get the content of a file from IPFS."""
        try:
            return accel.cat(cid)
        except Exception as e:
            logger.error(f"Error in ipfs_cat: {str(e)}")
            return {"error": str(e), "success": False}
    
    register_tool("ipfs_cat", "Get the content of a file from IPFS", ipfs_cat, {
        "type": "object",
        "properties": {
            "cid": {"type": "string", "description": "CID of the file"}
        },
        "required": ["cid"]
    })
    
    logger.info("Registered IPFS tools")

def register_model_tools(accel):
    """Register model-related tools."""
    # Model Inference
    def model_inference(model_name, input_data, endpoint_type=None):
        """Run inference on a model."""
        try:
            return accel.process(model_name, input_data, endpoint_type)
        except Exception as e:
            logger.error(f"Error in model_inference: {str(e)}")
            return {"error": str(e)}
    
    register_tool("model_inference", "Run inference on a model", model_inference, {
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the model"},
            "input_data": {"description": "Input data for the model"},
            "endpoint_type": {"type": "string", "description": "Endpoint type (optional)"}
        },
        "required": ["model_name", "input_data"]
    })
    
    # List Models
    def list_models():
        """List available models."""
        try:
            return {
                "local_models": list(accel.endpoints["local_endpoints"].keys()),
                "api_models": list(accel.endpoints["api_endpoints"].keys()),
                "libp2p_models": list(accel.endpoints["libp2p_endpoints"].keys())
            }
        except Exception as e:
            logger.error(f"Error in list_models: {str(e)}")
            return {"error": str(e)}
    
    register_tool("list_models", "List available models", list_models, {})
    
    logger.info("Registered model tools")

def register_all_tools():
    """Register all tools."""
    logger.info("Registering all tools...")
    
    # Create IPFS Accelerate instance
    accel = MockIPFSAccelerate()
    
    # Register resources
    register_resource("system_info", "Information about the system")
    register_resource("accelerator_info", "Information about available hardware accelerators")
    register_resource("ipfs_nodes", "Information about connected IPFS nodes")
    register_resource("models", "Information about available models")
    
    # Register tools
    register_hardware_tools()
    register_ipfs_tools(accel)
    register_model_tools(accel)
    
    logger.info("All tools registered successfully")

class MCPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the MCP server."""
    
    def _set_headers(self, content_type="application/json"):
        """Set response headers."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests."""
        self._set_headers()
        self.wfile.write(b"{}")
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/status":
            self._set_headers()
            response = {
                "status": "running",
                "server_name": SERVER_INFO["server_name"],
                "version": SERVER_INFO["version"],
                "uptime": time.time() - start_time
            }
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == "/tools":
            self._set_headers()
            response = {"tools": list(TOOLS.keys())}
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == "/mcp/manifest":
            self._set_headers()
            manifest = {
                "name": SERVER_INFO["server_name"],
                "description": SERVER_INFO["description"],
                "version": SERVER_INFO["version"],
                "mcp_version": SERVER_INFO["mcp_version"],
                "tools": {
                    name: {
                        "description": tool["description"],
                        "schema": tool["schema"]
                    }
                    for name, tool in TOOLS.items()
                },
                "resources": RESOURCES
            }
            self.wfile.write(json.dumps(manifest).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(post_data)
        
        if self.path in ["/call", "/call_tool"]:
            tool_name = data.get("tool_name")
            args = data.get("arguments", {})
            
            if not tool_name:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing tool_name"}).encode())
                return
            
            if tool_name not in TOOLS:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Unknown tool: {tool_name}"}).encode())
                return
            
            try:
                result = TOOLS[tool_name]["handler"](**args)
                self._set_headers()
                self.wfile.write(json.dumps({"result": result}).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

def find_available_port(start_port=8002, max_attempts=10):
    """Find an available port to use."""
    for attempt in range(max_attempts):
        port = start_port + attempt
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    return start_port  # Default to the start port if none is available

def start_server(host="0.0.0.0", port=8002):
    """Start the MCP server."""
    # Check if the requested port is available
    if port is None:
        port = find_available_port()
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, MCPHandler)
    logger.info(f"Starting MCP server on {host}:{port}...")
    
    # Start the server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Log server started
    logger.info(f"MCP server running at http://{host}:{port}")
    
    return httpd, server_thread

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start standalone MCP server with IPFS tools")
    parser.add_argument("--port", type=int, default=8002, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Register all tools
    register_all_tools()
    
    # Start the server
    httpd, server_thread = start_server(host=args.host, port=args.port)
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        httpd.shutdown()
        httpd.server_close()

# Track server start time
start_time = time.time()

if __name__ == "__main__":
    main()
