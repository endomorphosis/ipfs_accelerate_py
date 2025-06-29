#!/usr/bin/env python3
"""
Simple MCP server implemented directly with uvicorn

This script serves as a minimal test case for the MCP server using uvicorn directly,
which avoids any issues that might exist in the MCP module's start_server function.
"""

import os
import sys
import logging
import platform
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("minimal_mcp_server")

# Create the FastAPI application
app = FastAPI(title="Minimal MCP Server", version="0.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the MCP tools
tools = {}

# Define some simple tools
def get_hardware_info():
    """Get hardware information about the system"""
    return {
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

def health_check():
    """Check the health of the MCP server"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "uptime": 0
    }

# Register tools
tools["get_hardware_info"] = {
    "name": "get_hardware_info",
    "description": "Get hardware information about the system",
    "function": get_hardware_info
}

tools["health_check"] = {
    "name": "health_check",
    "description": "Check the health of the MCP server",
    "function": health_check
}

# Define the MCP manifest endpoint
@app.get("/mcp/manifest")
async def get_manifest():
    """Return the MCP server manifest"""
    return {
        "server_name": "Minimal MCP Server",
        "version": "0.1.0",
        "mcp_version": "0.1.0",
        "tools": {name: {"description": tool["description"]} for name, tool in tools.items()},
        "resources": {}
    }

# Define the MCP tool endpoint
@app.post("/mcp/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    """Call a specific tool"""
    if tool_name not in tools:
        return JSONResponse(
            status_code=404,
            content={"error": f"Tool {tool_name} not found"}
        )
    
    # Get the tool function
    tool = tools[tool_name]
    
    try:
        # Parse the arguments
        args = await request.json() if request.headers.get("content-length") and int(request.headers.get("content-length")) > 0 else {}
        
        # Call the tool function
        result = tool["function"](**args)
        
        # Return the result
        return result
    except Exception as e:
        logger.exception(f"Error calling tool {tool_name}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a minimal MCP server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    args = parser.parse_args()
    
    logger.info(f"Starting minimal MCP server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
