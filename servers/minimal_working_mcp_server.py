#!/usr/bin/env python3
"""
Minimal Working MCP Server for IPFS Accelerate
This server provides basic MCP functionality with IPFS tools using only Flask.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Installing Flask and Flask-CORS...")
    os.system("pip install flask flask-cors")
    from flask import Flask, request, jsonify
    from flask_cors import CORS

import json
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minimal-mcp")

app = Flask(__name__)
CORS(app)

# MCP Tools Registry
MCP_TOOLS = {
    "ipfs_node_info": {
        "description": "Get IPFS node information and status",
        "parameters": {}
    },
    "ipfs_gateway_url": {
        "description": "Get gateway URL for IPFS content",
        "parameters": {
            "cid": {"type": "string", "description": "IPFS Content ID"},
            "ipfs_hash": {"type": "string", "description": "IPFS hash (alias for cid)"}
        }
    },
    "ipfs_get_hardware_info": {
        "description": "Get hardware information for IPFS operations",
        "parameters": {}
    },
    "ipfs_files_write": {
        "description": "Write file to IPFS virtual filesystem",
        "parameters": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "File content"}
        }
    },
    "ipfs_files_read": {
        "description": "Read file from IPFS virtual filesystem",
        "parameters": {
            "path": {"type": "string", "description": "File path"}
        }
    },
    "ipfs_files_ls": {
        "description": "List files in IPFS virtual filesystem",
        "parameters": {
            "path": {"type": "string", "description": "Directory path", "default": "/"}
        }
    },
    "model_inference": {
        "description": "Run model inference on IPFS data",
        "parameters": {
            "model": {"type": "string", "description": "Model name"},
            "input": {"type": "string", "description": "Input data"}
        }
    },
    "list_models": {
        "description": "List available models",
        "parameters": {}
    }
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "server": "IPFS Accelerate MCP",
        "tools_count": len(MCP_TOOLS)
    })

@app.route('/tools', methods=['GET'])
def list_tools():
    """List all available tools"""
    return jsonify(list(MCP_TOOLS.keys()))

@app.route('/tools/<tool_name>', methods=['POST', 'GET'])
def execute_tool(tool_name):
    """Execute a specific tool"""
    if tool_name not in MCP_TOOLS:
        return jsonify({"error": f"Tool '{tool_name}' not found"}), 404
    
    # Get parameters from request
    if request.method == 'POST':
        params = request.get_json() or {}
    else:
        params = request.args.to_dict()
    
    # Execute tool
    try:
        result = _execute_tool_impl(tool_name, params)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return jsonify({"error": str(e)}), 500

def _execute_tool_impl(tool_name, params):
    """Implementation of tool execution"""
    
    if tool_name == "ipfs_node_info":
        return {
            "id": "QmMockNodeID12345",
            "addresses": ["/ip4/127.0.0.1/tcp/4001", "/ip6/::1/tcp/4001"],
            "protocol_version": "ipfs/0.1.0",
            "agent_version": "go-ipfs/0.12.0",
            "public_key": "mock-public-key",
            "status": "online"
        }
    
    elif tool_name == "ipfs_gateway_url":
        cid = params.get('cid') or params.get('ipfs_hash', 'QmMockCID123')
        return {
            "gateway_url": f"https://ipfs.io/ipfs/{cid}",
            "cid": cid,
            "local_gateway": f"http://127.0.0.1:8080/ipfs/{cid}"
        }
    
    elif tool_name == "ipfs_get_hardware_info":
        return {
            "cpu_count": 4,
            "cpu_model": "Mock CPU Model",
            "memory_total": "8.0 GB",
            "memory_available": "6.2 GB",
            "disk_usage": {
                "total": "500 GB",
                "used": "200 GB", 
                "free": "300 GB"
            },
            "network_interfaces": ["eth0", "lo"],
            "platform": "Linux"
        }
    
    elif tool_name == "ipfs_files_write":
        path = params.get('path', '/test.txt')
        content = params.get('content', 'Hello, IPFS!')
        return {
            "success": True,
            "path": path,
            "bytes_written": len(content),
            "hash": f"Qm{hash(content) % 1000000}"
        }
    
    elif tool_name == "ipfs_files_read":
        path = params.get('path', '/test.txt')
        return {
            "success": True,
            "path": path,
            "content": f"Mock content from {path}",
            "size": 100
        }
    
    elif tool_name == "ipfs_files_ls":
        path = params.get('path', '/')
        return {
            "path": path,
            "entries": [
                {"name": "test.txt", "type": "file", "size": 100, "hash": "QmTest123"},
                {"name": "documents", "type": "directory", "size": 0, "hash": "QmDir456"},
                {"name": "images", "type": "directory", "size": 0, "hash": "QmImg789"}
            ]
        }
    
    elif tool_name == "model_inference":
        model = params.get('model', 'gpt-3.5-turbo')
        input_data = params.get('input', 'Hello, world!')
        return {
            "model": model,
            "input": input_data,
            "output": f"Mock inference result for '{input_data}' using {model}",
            "confidence": 0.95,
            "processing_time": "0.123s"
        }
    
    elif tool_name == "list_models":
        return {
            "models": [
                {"name": "gpt-3.5-turbo", "type": "language", "status": "available"},
                {"name": "claude-3", "type": "language", "status": "available"},
                {"name": "llama-2", "type": "language", "status": "available"},
                {"name": "whisper", "type": "audio", "status": "available"}
            ]
        }
    
    else:
        return {
            "result": f"Mock response for {tool_name}",
            "parameters": params,
            "timestamp": "2025-05-25T02:00:00Z"
        }

# MCP Protocol Endpoints
@app.route('/mcp/manifest', methods=['GET'])
def mcp_manifest():
    """MCP manifest endpoint"""
    return jsonify({
        "name": "IPFS Accelerate MCP Server",
        "version": "1.0.0",
        "description": "Model Context Protocol server for IPFS acceleration tools",
        "tools": MCP_TOOLS,
        "capabilities": ["tools", "resources"],
        "protocol_version": "2024-11-05"
    })

@app.route('/status', methods=['GET'])
def status():
    """Server status endpoint"""
    return jsonify({
        "status": "running",
        "server": "IPFS Accelerate MCP",
        "version": "1.0.0",
        "tools_count": len(MCP_TOOLS),
        "uptime": "running",
        "capabilities": ["tools", "resources"]
    })

@app.route('/sse', methods=['GET'])
def sse_endpoint():
    """Server-Sent Events endpoint for real-time updates"""
    def event_stream():
        yield f"data: {json.dumps({'type': 'connected', 'server': 'IPFS Accelerate MCP'})}\n\n"
        yield f"data: {json.dumps({'type': 'tools_ready', 'count': len(MCP_TOOLS)})}\n\n"
    
    return app.response_class(event_stream(), mimetype='text/event-stream')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting IPFS Accelerate MCP Server on {args.host}:{args.port}")
    logger.info(f"Registered {len(MCP_TOOLS)} tools: {list(MCP_TOOLS.keys())}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
