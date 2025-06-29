#!/usr/bin/env python3
"""
Enhanced MCP Server for IPFS Accelerate

This MCP server implements the Model Context Protocol properly to expose 
IPFS Accelerate functionality to Claude and other MCP clients.
"""

import os
import sys
import time
import json
import uuid
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import threading
import queue

from flask import Flask, Response, request, jsonify, stream_with_context
from flask_cors import CORS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_ipfs_mcp")

# Server information
SERVER_INFO = {
    "name": "IPFS Accelerate MCP Server",
    "version": "1.1.0",
    "description": "Enhanced MCP server implementation for IPFS Accelerate Python"
}

# Start time for uptime calculation
startup_time = time.time()

# Import the mock IPFS Accelerate from direct_mcp_server.py
# In a real implementation, you would import the actual IPFS Accelerate package
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

# Create Flask app
app = Flask("enhanced_mcp_server")
CORS(app)

# SSE client management
class ClientManager:
    def __init__(self):
        self.clients = {}  # Maps client_id to client data (including message queue)
        self.lock = threading.Lock()
        self.next_event_id = 0
    
    def register_client(self):
        """Register a new client and return its ID."""
        with self.lock:
            client_id = uuid.uuid4().hex
            self.clients[client_id] = {
                "queue": queue.Queue(),
                "connected_at": time.time(),
                "last_activity": time.time()
            }
            return client_id
    
    def unregister_client(self, client_id):
        """Unregister a client."""
        with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]
                return True
            return False
    
    def send_to_client(self, client_id, event_type, data):
        """Send an event to a specific client."""
        with self.lock:
            if client_id in self.clients:
                event_id = self.next_event_id
                self.next_event_id += 1
                self.clients[client_id]["queue"].put({
                    "id": event_id,
                    "event": event_type,
                    "data": data
                })
                self.clients[client_id]["last_activity"] = time.time()
                return True
            return False
    
    def broadcast(self, event_type, data):
        """Send an event to all clients."""
        with self.lock:
            for client_id in list(self.clients.keys()):
                self.send_to_client(client_id, event_type, data)
    
    def get_next_event(self, client_id, timeout=1.0):
        """Get the next event for a client, or None if no events are available."""
        if client_id not in self.clients:
            return None
        
        try:
            return self.clients[client_id]["queue"].get(timeout=timeout)
        except queue.Empty:
            # Send a heartbeat if queue is empty
            return {
                "id": self.next_event_id,
                "event": "heartbeat",
                "data": {"timestamp": time.time()}
            }

# Create a client manager
client_manager = ClientManager()

# Tool definitions with schema for the manifest
TOOL_SCHEMAS = {
    "ipfs_add_file": {
        "description": "Add a file to IPFS",
        "parameters": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to add"
                }
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "cid": {
                    "type": "string",
                    "description": "IPFS content identifier"
                },
                "path": {
                    "type": "string",
                    "description": "Path of the added file"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                }
            }
        }
    },
    "ipfs_cat": {
        "description": "Retrieve content from IPFS",
        "parameters": {
            "type": "object",
            "required": ["cid"],
            "properties": {
                "cid": {
                    "type": "string",
                    "description": "IPFS content identifier"
                }
            }
        },
        "returns": {
            "type": "string",
            "description": "Content retrieved from IPFS"
        }
    },
    "ipfs_files_write": {
        "description": "Write content to the IPFS MFS",
        "parameters": {
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path in the MFS"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write"
                }
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path in the MFS"
                },
                "written": {
                    "type": "boolean",
                    "description": "Whether the content was written"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                }
            }
        }
    },
    "ipfs_files_read": {
        "description": "Read content from the IPFS MFS",
        "parameters": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path in the MFS"
                }
            }
        },
        "returns": {
            "type": "string",
            "description": "Content read from the MFS"
        }
    },
    "health_check": {
        "description": "Check the health of the MCP server",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Health status"
                },
                "version": {
                    "type": "string",
                    "description": "Server version"
                },
                "uptime": {
                    "type": "number",
                    "description": "Server uptime in seconds"
                }
            }
        }
    },
    "get_hardware_info": {
        "description": "Get hardware information for model acceleration",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": {
            "type": "object",
            "properties": {
                "cpu": {
                    "type": "object",
                    "description": "CPU information"
                },
                "gpu": {
                    "type": "object",
                    "description": "GPU information"
                }
            }
        }
    },
    "list_models": {
        "description": "List available models for inference",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": {
            "type": "object",
            "properties": {
                "models": {
                    "type": "object",
                    "description": "Dictionary of models"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of models"
                }
            }
        }
    },
    "create_endpoint": {
        "description": "Create a model endpoint for inference",
        "parameters": {
            "type": "object",
            "required": ["model_name"],
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of the model"
                },
                "device": {
                    "type": "string",
                    "description": "Device to run on (cpu, cuda, etc.)",
                    "default": "cpu"
                },
                "max_batch_size": {
                    "type": "integer",
                    "description": "Maximum batch size",
                    "default": 16
                }
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "endpoint_id": {
                    "type": "string",
                    "description": "Endpoint ID"
                },
                "model": {
                    "type": "string",
                    "description": "Model name"
                },
                "device": {
                    "type": "string",
                    "description": "Device being used"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                }
            }
        }
    },
    "run_inference": {
        "description": "Run inference using a model endpoint",
        "parameters": {
            "type": "object",
            "required": ["endpoint_id", "inputs"],
            "properties": {
                "endpoint_id": {
                    "type": "string",
                    "description": "Endpoint ID"
                },
                "inputs": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Input strings for inference"
                }
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "outputs": {
                    "type": "array",
                    "description": "Output results"
                }
            }
        }
    }
}

# Define tool handlers
def handle_ipfs_add_file(args):
    """Add a file to IPFS."""
    path = args.get("path")
    if not path:
        return {"error": "Missing required argument: path", "success": False}
    
    return mock_accelerate.add_file(path)

def handle_ipfs_cat(args):
    """Retrieve content from IPFS."""
    cid = args.get("cid")
    if not cid:
        return {"error": "Missing required argument: cid", "success": False}
    
    content = mock_accelerate.cat(cid)
    return {"success": True, "content": content}

def handle_ipfs_files_write(args):
    """Write content to the IPFS MFS."""
    path = args.get("path")
    content = args.get("content")
    
    if not path:
        return {"error": "Missing required argument: path", "success": False}
    if content is None:
        return {"error": "Missing required argument: content", "success": False}
    
    return mock_accelerate.files_write(path, content)

def handle_ipfs_files_read(args):
    """Read content from the IPFS MFS."""
    path = args.get("path")
    if not path:
        return {"error": "Missing required argument: path", "success": False}
    
    content = mock_accelerate.files_read(path)
    return {"success": True, "content": content}

def handle_health_check(args):
    """Check the health of the MCP server."""
    return {
        "status": "healthy",
        "version": SERVER_INFO["version"],
        "uptime": time.time() - startup_time,
        "ipfs_connected": False  # In mock mode, IPFS is never truly connected
    }

def handle_get_hardware_info(args):
    """Get hardware information for model acceleration."""
    return mock_accelerate.get_hardware_info()

def handle_list_models(args):
    """List available models for inference."""
    return mock_accelerate.list_models()

def handle_create_endpoint(args):
    """Create a model endpoint for inference."""
    model_name = args.get("model_name")
    device = args.get("device", "cpu")
    max_batch_size = args.get("max_batch_size", 16)
    
    if not model_name:
        return {"error": "Missing required argument: model_name", "success": False}
    
    return mock_accelerate.create_endpoint(model_name, device, max_batch_size)

def handle_run_inference(args):
    """Run inference using a model endpoint."""
    endpoint_id = args.get("endpoint_id")
    inputs = args.get("inputs")
    
    if not endpoint_id:
        return {"error": "Missing required argument: endpoint_id", "success": False}
    
    if not inputs or not isinstance(inputs, list):
        return {"error": "Missing or invalid argument: inputs must be a list", "success": False}
    
    return mock_accelerate.run_inference(endpoint_id, inputs)

# Tool registry
tools = {
    "ipfs_add_file": handle_ipfs_add_file,
    "ipfs_cat": handle_ipfs_cat,
    "ipfs_files_write": handle_ipfs_files_write,
    "ipfs_files_read": handle_ipfs_files_read,
    "health_check": handle_health_check,
    "get_hardware_info": handle_get_hardware_info,
    "list_models": handle_list_models,
    "create_endpoint": handle_create_endpoint,
    "run_inference": handle_run_inference
}

logger.info(f"Registered {len(tools)} tools")

# API routes
@app.route("/tools", methods=["GET"])
def list_tools():
    """List all available tools."""
    return jsonify({"tools": list(tools.keys())})

@app.route("/mcp/manifest", methods=["GET"])
def get_manifest():
    """Get the MCP manifest, including tool schemas."""
    manifest = {
        "server": {
            "name": SERVER_INFO["name"],
            "version": SERVER_INFO["version"],
            "description": SERVER_INFO["description"]
        },
        "tools": []
    }
    
    for tool_name, schema in TOOL_SCHEMAS.items():
        if tool_name in tools:
            tool_info = {
                "name": tool_name,
                "description": schema["description"],
                "schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Convert parameters format to match Claude's expectations
            if "parameters" in schema:
                if "properties" in schema["parameters"]:
                    tool_info["schema"]["properties"] = schema["parameters"]["properties"]
                if "required" in schema["parameters"]:
                    tool_info["schema"]["required"] = schema["parameters"]["required"]
            
            manifest["tools"].append(tool_info)
    
    return jsonify(manifest)

@app.route("/manifest", methods=["GET"])
def get_standard_manifest():
    """Get the MCP manifest in the standard location that Claude expects."""
    # Delegate to the existing implementation
    return get_manifest()

@app.route("/invoke/<tool_name>", methods=["POST"])
def invoke_tool(tool_name):
    """Standard MCP endpoint to invoke a tool directly."""
    if tool_name not in tools:
        return jsonify({"error": f"Tool not found: {tool_name}"}), 404
    
    try:
        arguments = request.json or {}
        result = tools[tool_name](arguments)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/call_tool", methods=["POST"])
def call_tool():
    """Call a tool with arguments."""
    content = request.json
    tool_name = content.get("tool_name")
    arguments = content.get("arguments", {})
    
    if not tool_name:
        return jsonify({"error": "Missing tool_name"}), 400
    
    if tool_name not in tools:
        return jsonify({"error": f"Tool not found: {tool_name}"}), 404
    
    try:
        result = tools[tool_name](arguments)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mcp/tool/<tool_name>", methods=["POST"])
def call_mcp_tool(tool_name):
    """Claude-specific MCP tool invocation endpoint."""
    if tool_name not in tools:
        return jsonify({"error": f"Tool not found: {tool_name}"}), 404
    
    try:
        arguments = request.json or {}
        result = tools[tool_name](arguments)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error invoking {tool_name}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# SSE endpoint for MCP
@app.route("/sse")
def sse():
    """Server-Sent Events endpoint for MCP with bidirectional support."""
    
    def stream():
        # Register client
        client_id = client_manager.register_client()
        logger.info(f"Client connected: {client_id}")
        
        # Process any query parameters
        init_data = {
            "client_id": client_id,
            "server_info": SERVER_INFO
        }
        
        # Send init event
        client_manager.send_to_client(client_id, "init", init_data)
        
        try:
            while True:
                # Get next event for this client (or heartbeat if none)
                event = client_manager.get_next_event(client_id)
                if event:
                    # Ensure data is properly JSON encoded as a string
                    json_data = json.dumps(event['data'])
                    yield f"id: {event['id']}\nevent: {event['event']}\ndata: {json_data}\n\n"
        except GeneratorExit:
            # Client disconnected
            logger.info(f"Client disconnected: {client_id}")
            client_manager.unregister_client(client_id)
    
    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# Handle MCP requests
@app.route("/sse/request", methods=["POST"])
def sse_request():
    """Handle requests from MCP clients."""
    content = request.json
    client_id = content.get("client_id")
    request_id = content.get("request_id")
    request_type = content.get("type")
    
    if not client_id:
        return jsonify({"error": "Missing client_id"}), 400
    
    if not request_id:
        return jsonify({"error": "Missing request_id"}), 400
    
    # Tool call
    if request_type == "tool_call":
        tool_name = content.get("tool_name")
        arguments = content.get("arguments", {})
        
        if not tool_name:
            return jsonify({"error": "Missing tool_name"}), 400
        
        if tool_name not in tools:
            return jsonify({"error": f"Tool not found: {tool_name}"}), 404
        
        try:
            result = tools[tool_name](arguments)
            # Send result through SSE
            client_manager.send_to_client(client_id, "tool_response", {
                "request_id": request_id,
                "result": result
            })
            return jsonify({"success": True})
        except Exception as e:
            client_manager.send_to_client(client_id, "tool_response", {
                "request_id": request_id,
                "error": str(e)
            })
            return jsonify({"error": str(e)}), 500
    
    # Unknown request type
    return jsonify({"error": f"Unknown request type: {request_type}"}), 400

def main():
    """Main function to start the server."""
    parser = argparse.ArgumentParser(description="Run the Enhanced IPFS Accelerate MCP server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to listen on")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Enhanced MCP server on http://{args.host}:{args.port}")
    logger.info(f"MCP manifest: http://{args.host}:{args.port}/mcp/manifest")
    logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

if __name__ == "__main__":
    main()
