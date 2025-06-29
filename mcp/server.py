#!/usr/bin/env python
"""
IPFS Accelerate MCP Server

This module provides the server implementation for the MCP protocol.
"""

import os
import sys
import json
import time
import logging
import inspect
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Import FastAPI
try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field, create_model
except ImportError:
    logging.error("FastAPI is required for the MCP server. Install it with 'pip install fastapi uvicorn'")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Server information
SERVER_NAME = "ipfs-accelerate-mcp"
SERVER_DESCRIPTION = "IPFS Accelerate MCP Server"
SERVER_VERSION = "0.1.0"
MCP_VERSION = "0.1.0"

# Define tools and resources
_tools: Dict[str, Dict[str, Any]] = {}
_resources: Dict[str, Dict[str, Any]] = {}

def register_tool(
    name: str,
    description: str,
    function: Callable,
    schema: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register a tool with the MCP server
    
    Args:
        name: Name of the tool
        description: Description of the tool
        function: Function that implements the tool
        schema: JSON schema for the tool's input (optional)
    """
    # Generate schema from function signature if not provided
    if schema is None:
        schema = _generate_schema_from_function(function)
    
    _tools[name] = {
        "name": name,
        "description": description,
        "function": function,
        "schema": schema,
    }
    
    logger.info(f"Registered tool: {name}")

def register_resource(
    name: str,
    description: str,
    getter: Callable[[], Any],
) -> None:
    """
    Register a resource with the MCP server
    
    Args:
        name: Name of the resource
        description: Description of the resource
        getter: Function that returns the resource data
    """
    _resources[name] = {
        "name": name,
        "description": description,
        "getter": getter,
    }
    
    logger.info(f"Registered resource: {name}")

def _generate_schema_from_function(func: Callable) -> Dict[str, Any]:
    """
    Generate a JSON schema from a function signature
    
    Args:
        func: Function to generate schema for
    
    Returns:
        Dict[str, Any]: JSON schema
    """
    signature = inspect.signature(func)
    parameters = {}
    required = []
    
    for name, param in signature.parameters.items():
        # Skip 'self' parameter for methods
        if name == 'self':
            continue
        
        # Get annotation (type hint)
        annotation = param.annotation
        
        # Get default value if available
        has_default = param.default is not inspect.Parameter.empty
        default = param.default if has_default else None
        
        # Add to required list if no default
        if not has_default:
            required.append(name)
        
        # Generate parameter schema based on type
        param_schema = {"type": "object"}
        
        # Try to determine type from annotation
        if annotation is not inspect.Parameter.empty:
            if annotation == str:
                param_schema = {"type": "string"}
            elif annotation == int:
                param_schema = {"type": "integer"}
            elif annotation == float:
                param_schema = {"type": "number"}
            elif annotation == bool:
                param_schema = {"type": "boolean"}
            elif annotation == List[str]:
                param_schema = {"type": "array", "items": {"type": "string"}}
            elif annotation == List[int]:
                param_schema = {"type": "array", "items": {"type": "integer"}}
            elif annotation == Dict[str, Any]:
                param_schema = {"type": "object"}
        
        # Add default value if available
        if has_default and default is not None:
            param_schema["default"] = default
        
        parameters[name] = param_schema
    
    # Create schema
    schema = {
        "type": "object",
        "properties": parameters,
        "required": required,
    }
    
    return schema

# Create FastAPI app
app = FastAPI(
    title=SERVER_NAME,
    description=SERVER_DESCRIPTION,
    version=SERVER_VERSION,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat()
    }


# Standard API endpoints for compatibility
@app.get("/tools")
def get_tools_list():
    '''Return a list of all available tools'''
    return list(_tools.keys())

@app.post("/tools/{tool_name}/invoke")
async def invoke_tool_compat(tool_name: str, request: Request):
    '''Tool invocation endpoint compatible with standard clients'''
    # Reuse the existing tool endpoint logic
    return await call_tool(tool_name, request)

# MCP manifest endpoint
@app.get("/mcp/manifest")
def get_manifest():
    """Get the MCP manifest"""
    # Create tool descriptions
    tools = {}
    for name, tool in _tools.items():
        tools[name] = {
            "description": tool["description"],
            "schema": tool["schema"],
        }
    
    # Create resource descriptions
    resources = {}
    for name, resource in _resources.items():
        resources[name] = {
            "description": resource["description"],
        }
    
    # Create manifest
    manifest = {
        "server_name": SERVER_NAME,
        "description": SERVER_DESCRIPTION,
        "version": SERVER_VERSION,
        "mcp_version": MCP_VERSION,
        "tools": tools,
        "resources": resources,
    }
    
    return manifest

# Generic tool endpoint
@app.post("/mcp/tool/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    """Call a tool"""
    # Check if tool exists
    if tool_name not in _tools:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    
    # Get tool
    tool = _tools[tool_name]
    
    # Parse request body
    body = await request.json()
    
    # Call tool function
    try:
        result = tool["function"](**body)
        return result
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Generic resource endpoint
@app.get("/mcp/resources/{resource_name}")
def access_resource(resource_name: str):
    """Access a resource"""
    # Check if resource exists
    if resource_name not in _resources:
        raise HTTPException(status_code=404, detail=f"Resource not found: {resource_name}")
    
    # Get resource
    resource = _resources[resource_name]
    
    # Call getter function
    try:
        result = resource["getter"]()
        return result
    except Exception as e:
        logger.error(f"Error accessing resource {resource_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Register default tools and resources
def register_defaults():
    """Register default tools and resources"""
    # Register get_hardware_info tool
    try:
        from .tools import get_hardware_info
        
        register_tool(
            name="get_hardware_info",
            description="Get hardware information about the system",
            function=get_hardware_info,
        )
    except ImportError as e:
        logger.warning(f"Could not register get_hardware_info tool: {e}")
    
    # Register system_info resource
    def get_system_info():
        """Get system information"""
        try:
            from .tools.hardware import get_system_info
            return get_system_info()
        except ImportError:
            import platform
            return {
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
            }
    
    register_resource(
        name="system_info",
        description="Information about the system",
        getter=get_system_info,
    )
    
    # Register accelerator_info resource
    def get_accelerator_info():
        """Get accelerator information"""
        try:
            from .tools.hardware import get_hardware_info
            hardware_info = get_hardware_info()
            return hardware_info.get("accelerators", {})
        except ImportError:
            return {
                "cuda": {"available": False},
                "webgpu": {"available": False},
                "webnn": {"available": False},
            }
    
    register_resource(
        name="accelerator_info",
        description="Information about available hardware accelerators",
        getter=get_accelerator_info,
    )

# Register defaults
register_defaults()

# SSE endpoint
@app.get("/sse")
async def sse():
    """Server-Sent Events (SSE) endpoint"""
    
    async def event_generator():
        """Generate SSE events"""
        # Initial connection message
        yield "event: connected\ndata: {\"status\": \"connected\"}\n\n"
        
        # Keep connection alive and send periodic updates
        count = 0
        while True:
            # Send server status every 5 seconds
            await asyncio.sleep(5)
            count += 1
            
            # Create update with timestamp and server info
            timestamp = datetime.datetime.now().isoformat()
            data = {
                "event": "status",
                "timestamp": timestamp,
                "server": SERVER_NAME,
                "version": SERVER_VERSION,
                "count": count,
                "status": "running"
            }
            
            # Format as SSE event
            yield f"event: status\ndata: {json.dumps(data)}\n\n"
    
    # Import asyncio here to avoid issues if not available
    try:
        import asyncio
    except ImportError:
        return Response(
            content="Asyncio is required for SSE support",
            status_code=500,
            media_type="text/plain"
        )
    
    # Return a streaming response with the correct content type for SSE
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

# MCP events endpoint for more structured event streaming
@app.get("/mcp/events")
async def mcp_events():
    """MCP events endpoint for streaming updates"""
    
    async def mcp_event_generator():
        """Generate MCP protocol events"""
        # Initial connection message
        connection_event = {
            "type": "connection",
            "status": "connected",
            "server": SERVER_NAME,
            "timestamp": datetime.datetime.now().isoformat()
        }
        yield f"event: connection\ndata: {json.dumps(connection_event)}\n\n"
        
        # Keep connection alive and send periodic updates
        try:
            import asyncio
            
            # Send hardware info at connection
            try:
                from .tools import get_hardware_info
                hardware_info = get_hardware_info()
                
                hardware_event = {
                    "type": "hardware_info",
                    "data": hardware_info,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                yield f"event: hardware_info\ndata: {json.dumps(hardware_event)}\n\n"
            except Exception as e:
                logger.error(f"Error getting hardware info for SSE: {e}")
            
            # Periodic updates
            count = 0
            while True:
                # Send an update every 10 seconds
                await asyncio.sleep(10)
                count += 1
                
                # Create status update
                status_event = {
                    "type": "status",
                    "server": SERVER_NAME,
                    "version": SERVER_VERSION,
                    "uptime": count * 10,  # seconds
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tools_count": len(_tools),
                    "resources_count": len(_resources)
                }
                
                yield f"event: status\ndata: {json.dumps(status_event)}\n\n"
                
                # Every 30 seconds, send more detailed info
                if count % 3 == 0:
                    try:
                        # Get updated hardware info
                        from .tools import get_hardware_info
                        updated_hardware = get_hardware_info()
                        
                        detail_event = {
                            "type": "detail",
                            "hardware": updated_hardware,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        yield f"event: detail\ndata: {json.dumps(detail_event)}\n\n"
                    except Exception as e:
                        logger.error(f"Error getting detailed info for SSE: {e}")
        
        except Exception as e:
            error_event = {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
    
    # Return a streaming response with the correct content type for SSE
    return StreamingResponse(
        mcp_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

# Add JSON-RPC 2.0 support for MCP protocol
class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request model"""
    jsonrpc: str = Field(default="2.0")
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None

class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response model"""
    jsonrpc: str = Field(default="2.0")
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None

class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error model"""
    code: int
    message: str
    data: Optional[Any] = None

# JSON-RPC error codes
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

def create_jsonrpc_error(code: int, message: str, data: Any = None, request_id: Optional[Union[int, str]] = None) -> JSONRPCResponse:
    """Create a JSON-RPC error response"""
    return JSONRPCResponse(
        error=JSONRPCError(code=code, message=message, data=data).dict(),
        id=request_id
    )

def create_jsonrpc_success(result: Any, request_id: Optional[Union[int, str]] = None) -> JSONRPCResponse:
    """Create a JSON-RPC success response"""
    return JSONRPCResponse(result=result, id=request_id)

# JSON-RPC method handlers
async def handle_get_tools(params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Handle get_tools JSON-RPC method"""
    tools_list = []
    for name, tool in _tools.items():
        tools_list.append({
            "name": name,
            "description": tool["description"],
            "schema": tool["schema"]
        })
    return tools_list

async def handle_tools_list(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handle tools/list JSON-RPC method (alias for get_tools)"""
    return await handle_get_tools(params)

async def handle_ping(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handle ping JSON-RPC method"""
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "server": SERVER_NAME,
        "version": SERVER_VERSION
    }

async def handle_get_server_info(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handle get_server_info JSON-RPC method"""
    return {
        "name": SERVER_NAME,
        "description": SERVER_DESCRIPTION,
        "version": SERVER_VERSION,
        "mcp_version": MCP_VERSION,
        "tools_count": len(_tools),
        "resources_count": len(_resources),
        "timestamp": datetime.datetime.now().isoformat()
    }

async def handle_tool_call(method: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Handle tool calls via JSON-RPC"""
    if method not in _tools:
        raise HTTPException(status_code=404, detail=f"Tool not found: {method}")
    
    tool = _tools[method]
    
    # Use params if provided, otherwise empty dict
    if params is None:
        params = {}
    
    try:
        result = tool["function"](**params)
        return result
    except Exception as e:
        logger.error(f"Error calling tool {method}: {e}")
        raise Exception(f"Tool execution error: {str(e)}")

# JSON-RPC method registry
JSONRPC_METHODS = {
    "get_tools": handle_get_tools,
    "tools/list": handle_tools_list,
    "ping": handle_ping,
    "get_server_info": handle_get_server_info,
}

# Main JSON-RPC endpoint
@app.post("/jsonrpc")
async def jsonrpc_endpoint(request: Request):
    """JSON-RPC 2.0 endpoint for MCP protocol"""
    try:
        # Parse request body
        body = await request.json()
        
        # Validate JSON-RPC request
        if not isinstance(body, dict):
            return create_jsonrpc_error(
                JSONRPC_INVALID_REQUEST,
                "Invalid request format",
                request_id=body.get("id") if isinstance(body, dict) else None
            ).dict()
        
        rpc_request = JSONRPCRequest(**body)
        
        # Check if method exists in registry
        if rpc_request.method in JSONRPC_METHODS:
            # Call registered method
            try:
                result = await JSONRPC_METHODS[rpc_request.method](rpc_request.params)
                return create_jsonrpc_success(result, rpc_request.id).dict()
            except Exception as e:
                logger.error(f"Error executing JSON-RPC method {rpc_request.method}: {e}")
                return create_jsonrpc_error(
                    JSONRPC_INTERNAL_ERROR,
                    str(e),
                    request_id=rpc_request.id
                ).dict()
        
        # Check if method is a tool name
        elif rpc_request.method in _tools:
            try:
                result = await handle_tool_call(rpc_request.method, rpc_request.params)
                return create_jsonrpc_success(result, rpc_request.id).dict()
            except Exception as e:
                logger.error(f"Error executing tool {rpc_request.method}: {e}")
                return create_jsonrpc_error(
                    JSONRPC_INTERNAL_ERROR,
                    str(e),
                    request_id=rpc_request.id
                ).dict()
        
        # Method not found
        else:
            return create_jsonrpc_error(
                JSONRPC_METHOD_NOT_FOUND,
                f"Method not found: {rpc_request.method}",
                request_id=rpc_request.id
            ).dict()
    
    except json.JSONDecodeError:
        return create_jsonrpc_error(
            JSONRPC_PARSE_ERROR,
            "Parse error"
        ).dict()
    
    except Exception as e:
        logger.error(f"Unexpected error in JSON-RPC endpoint: {e}")
        return create_jsonrpc_error(
            JSONRPC_INTERNAL_ERROR,
            "Internal error"
        ).dict()

# Start server function
def start_server(host: str = "0.0.0.0", port: int = 8002, debug: bool = False, log_level: str = "info"):
    """
    Start the MCP server (directly callable for script use)
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Whether to enable debug mode
        log_level: Log level (debug, info, warning, error, critical)
    """
    import uvicorn
    
    # Set log level
    log_level = log_level.lower()
    if log_level not in ["debug", "info", "warning", "error", "critical"]:
        log_level = "info"
    
    # Set debug
    if debug:
        log_level = "debug"
    
    # Start server
    uvicorn.run(
        "mcp.server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=debug,
    )

if __name__ == "__main__":
    # When run directly, start the server
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
        log_level=args.log_level,
    )
