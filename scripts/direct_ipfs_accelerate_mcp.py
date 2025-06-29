#!/usr/bin/env python3
"""
IPFS Accelerate Direct MCP Integration

This script directly creates an MCP server that exposes key functions from the ipfs_accelerate_py package.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_direct_mcp")

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    logger.error("Missing required dependencies. Install with: pip install fastapi uvicorn")
    sys.exit(1)

try:
    # Import ipfs_accelerate_py
    from ipfs_accelerate_py import ipfs_accelerate_py
except ImportError:
    logger.error("Could not import ipfs_accelerate_py. Make sure it's installed.")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title="IPFS Accelerate MCP Server",
    description="Model Context Protocol server for IPFS Accelerate",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    }
}

# Create ipfs_accelerate_py instance
accel = ipfs_accelerate_py()

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

def register_all_tools():
    """Register all tools from ipfs_accelerate_py."""
    
    # Register hardware info tool
    def get_hardware_info():
        """Get hardware information about the system."""
        try:
            import platform
            import psutil
            
            # Try to get hardware info from hardware_detection module if available
            hardware_info = None
            try:
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                import hardware_detection
                if hasattr(hardware_detection, "detect_all_hardware"):
                    hardware_info = hardware_detection.detect_all_hardware()
                elif hasattr(hardware_detection, "detect_hardware"):
                    hardware_info = hardware_detection.detect_hardware()
            except ImportError:
                pass
            
            # If hardware_detection not available or failed, create basic info
            if not hardware_info:
                hardware_info = {
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
            
            return hardware_info
        except Exception as e:
            logger.error(f"Error in get_hardware_info: {str(e)}")
            return {"error": str(e)}
    
    register_tool("get_hardware_info", "Get hardware information about the system", get_hardware_info)
    
    # Register model_inference tool
    def model_inference(model_name, input_data, endpoint_type=None):
        """Run inference on a model."""
        try:
            # Use the process method from ipfs_accelerate_py
            result = accel.process(model_name, input_data, endpoint_type)
            return result
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
    
    # Register list_models tool
    def list_models():
        """List available models."""
        try:
            endpoints = accel.endpoints
            return {
                "local_models": list(endpoints["local_endpoints"].keys()),
                "api_models": list(endpoints["api_endpoints"].keys()),
                "libp2p_models": list(endpoints["libp2p_endpoints"].keys())
            }
        except Exception as e:
            logger.error(f"Error in list_models: {str(e)}")
            return {"error": str(e)}
    
    register_tool("list_models", "List available models", list_models)
    
    # Register init_endpoints tool
    async def init_endpoints(models):
        """Initialize endpoints for models."""
        try:
            result = await accel.init_endpoints(models)
            return result
        except Exception as e:
            logger.error(f"Error in init_endpoints: {str(e)}")
            return {"error": str(e)}
    
    register_tool("init_endpoints", "Initialize endpoints for models", init_endpoints, {
        "type": "object",
        "properties": {
            "models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of model names to initialize"
            }
        },
        "required": ["models"]
    })
    
    logger.info(f"Registered {len(TOOLS)} tools from ipfs_accelerate_py")

# FastAPI endpoints
@app.get("/")
def root():
    """Root endpoint."""
    return SERVER_INFO

@app.get("/health")
def health():
    """Health check endpoint."""
    import datetime
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/mcp/manifest")
def manifest():
    """MCP manifest endpoint."""
    return {
        **SERVER_INFO,
        "tools": {name: {"description": tool["description"], "schema": tool["schema"]} for name, tool in TOOLS.items()},
        "resources": RESOURCES
    }

@app.post("/mcp/tool/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    """Call a tool by name."""
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    
    try:
        # Parse request body
        arguments = await request.json()
        
        # Get the handler
        handler = TOOLS[tool_name]["handler"]
        
        # Call the handler
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**arguments)
        else:
            result = handler(**arguments)
        
        return result
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run IPFS Accelerate MCP Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8002, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Register tools
    register_all_tools()
    
    # Run server
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
