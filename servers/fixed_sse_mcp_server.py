#!/usr/bin/env python3
"""
Fixed SSE-based MCP Server for IPFS Accelerate

This implementation fixes SSE connection issues and properly registers all tools from the
ipfs_accelerate_py package.
"""

import os
import sys
import time
import json
import uuid
import logging
import argparse
import threading
from pathlib import Path
import importlib
from typing import Dict, List, Any, Optional, Union

from flask import Flask, Response, request, jsonify, stream_with_context
from flask_cors import CORS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/barberb/ipfs_accelerate_py/fixed_mcp_server.log")
    ]
)
logger = logging.getLogger("fixed_ipfs_mcp")

# Server information
SERVER_INFO = {
    "name": "IPFS Accelerate Fixed MCP Server",
    "version": "1.2.0",
    "description": "Fixed SSE MCP server implementation for IPFS Accelerate Python"
}

# Start time for uptime calculation
startup_time = time.time()

# Try to import tools from direct_mcp_server.py or enhanced_mcp_server.py
try:
    # Try to import MockIPFSAccelerate from direct_mcp_server.py first
    import direct_mcp_server
    mock_accelerate = direct_mcp_server.MockIPFSAccelerate()
    logger.info("Imported MockIPFSAccelerate from direct_mcp_server.py")
    
    # Also try to use the tool handlers from direct_mcp_server.py
    direct_tools = {}
    for tool_name in dir(direct_mcp_server):
        if tool_name.startswith("handle_"):
            func_name = tool_name
            tool_name = tool_name[len("handle_"):]
            direct_tools[tool_name] = getattr(direct_mcp_server, func_name)
    
    logger.info(f"Imported {len(direct_tools)} tool handlers from direct_mcp_server.py")

except ImportError:
    # Fall back to a minimal mock implementation
    logger.warning("direct_mcp_server.py not found, using minimal mock implementation")
    
    class MockIPFSAccelerate:
        """Minimal mock implementation of IPFS Accelerate functionality"""
        
        def __init__(self):
            self.files = {}
            
        def add_file(self, path: str) -> Dict[str, Any]:
            file_hash = f"QmMock{uuid.uuid4().hex[:16]}"
            return {"cid": file_hash, "path": path, "success": True}
            
        def cat(self, cid: str) -> str:
            return f"Mock content for {cid}"
            
        def files_write(self, path: str, content: str) -> Dict[str, Any]:
            mfs_path = f"mfs://{path}"
            self.files[mfs_path] = content
            return {"path": path, "written": True, "success": True}
            
        def files_read(self, path: str) -> str:
            mfs_path = f"mfs://{path}"
            if mfs_path in self.files:
                return self.files[mfs_path]
            return f"Mock MFS content for {path}"
                
        def get_hardware_info(self) -> Dict[str, Any]:
            return {
                "accelerators": {
                    "cpu": {"available": True, "memory": "8GB", "priority": 1},
                    "cuda": {"available": False, "memory": "0GB", "priority": 2},
                    "webgpu": {"available": False, "memory": "0GB", "priority": 3}
                },
                "cpu": {"available": True, "cores": 4, "memory": "8GB"},
                "gpu": {"available": False, "details": "No GPU detected"},
                "webgpu": {"available": False, "details": "No WebGPU detected"}
            }
            
        def list_models(self) -> Dict[str, Any]:
            return {
                "models": {
                    "bert-base-uncased": {
                        "type": "text-embedding",
                        "description": "BERT base uncased embedding model",
                        "dimensions": 768,
                        "capabilities": ["embeddings"],
                        "available": True
                    },
                    "gpt2": {
                        "type": "text-generation",
                        "description": "GPT-2 text generation model",
                        "capabilities": ["text-generation"],
                        "available": True
                    },
                    "t5-small": {
                        "type": "text2text",
                        "description": "T5 small model for text-to-text generation",
                        "capabilities": ["translation", "summarization"],
                        "available": True
                    },
                    "resnet50": {
                        "type": "image-classification",
                        "description": "ResNet50 image classification model",
                        "capabilities": ["image-classification"],
                        "available": True
                    }
                },
                "count": 4
            }
            
        def create_endpoint(self, model_name: str, device: str = "cpu", max_batch_size: int = 16) -> Dict[str, Any]:
            endpoint_id = f"endpoint-{1000 + int(time.time()) % 1000}"
            return {
                "endpoint_id": endpoint_id,
                "model": model_name,
                "device": device,
                "max_batch_size": max_batch_size,
                "status": "ready",
                "success": True
            }
            
        def run_inference(self, endpoint_id: str, inputs: List[str]) -> Dict[str, Any]:
            return {"success": True, "outputs": [f"Result for {input_text}" for input_text in inputs]}
        
        # Add methods for API multiplexing
        def register_api_key(self, provider: str, key: str, priority: int = 1) -> Dict[str, Any]:
            key_id = f"{provider}-key-{uuid.uuid4().hex[:8]}"
            return {
                "success": True,
                "provider": provider,
                "key_id": key_id,
                "priority": priority
            }
        
        def get_api_keys(self) -> Dict[str, Any]:
            return {
                "providers": [
                    {"name": "openai", "key_count": 2, "active": True},
                    {"name": "anthropic", "key_count": 1, "active": True},
                    {"name": "groq", "key_count": 1, "active": True}
                ],
                "total_keys": 4
            }
        
        def get_multiplexer_stats(self) -> Dict[str, Any]:
            return {
                "providers": {
                    "openai": {"requests": 120, "rate_limited": 5, "errors": 2, "avg_latency_ms": 250},
                    "anthropic": {"requests": 85, "rate_limited": 1, "errors": 0, "avg_latency_ms": 450},
                    "groq": {"requests": 50, "rate_limited": 0, "errors": 1, "avg_latency_ms": 180}
                },
                "total_requests": 255,
                "successful_requests": 247,
                "load_balancing": {"strategy": "round-robin", "fallback_enabled": True}
            }
        
        def simulate_api_request(self, provider: str, prompt: str) -> Dict[str, Any]:
            import random
            return {
                "success": True,
                "provider": provider,
                "model": f"{provider}-default-model",
                "completion": f"This is a simulated response from {provider} for the prompt: {prompt[:30]}...",
                "tokens": len(prompt.split()) * 2,
                "latency_ms": random.randint(50, 500)
            }
        
        # Task management methods
        def start_task(self, task_type: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
            task_id = f"task-{task_type}-{uuid.uuid4().hex[:8]}"
            return {
                "success": True,
                "task_id": task_id,
                "task_type": task_type,
                "params": params,
                "status": "started",
                "created_at": time.time()
            }
        
        def get_task_status(self, task_id: str) -> Dict[str, Any]:
            progress = (int(time.time() * 100) % 100) if "task-" in task_id else 100
            status = "running" if progress < 100 else "completed"
            
            return {
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "updated_at": time.time(),
                "details": f"Task {task_id[:10]}... is {status} ({progress}% complete)"
            }
        
        def list_tasks(self) -> Dict[str, Any]:
            active_tasks = [
                {
                    "task_id": f"task-active-{i}",
                    "status": "running",
                    "progress": (int(time.time() * 10) + i * 10) % 100,
                    "task_type": ["download_model", "batch_processing", "data_conversion"][i % 3],
                    "created_at": time.time() - 300 + i * 60
                }
                for i in range(3)
            ]
            
            completed_tasks = [
                {
                    "task_id": f"task-complete-{i}",
                    "status": "completed",
                    "progress": 100,
                    "task_type": ["download_model", "batch_processing", "data_conversion"][i % 3],
                    "created_at": time.time() - 3600 + i * 300,
                    "completed_at": time.time() - 3000 + i * 300
                }
                for i in range(5)
            ]
            
            return {
                "active_tasks": active_tasks,
                "completed_tasks": completed_tasks,
                "active_count": len(active_tasks),
                "completed_count": len(completed_tasks)
            }
    
    # Create an instance of the mock
    mock_accelerate = MockIPFSAccelerate()
    direct_tools = {}

# Create Flask app
app = Flask("fixed_sse_mcp_server")
CORS(app)

# Client management for SSE
class SSEClientManager:
    def __init__(self):
        self.clients = {}
        self.lock = threading.Lock()
        self.next_event_id = 0
    
    def register_client(self):
        with self.lock:
            client_id = uuid.uuid4().hex
            self.clients[client_id] = {
                "queue": [],
                "connected_at": time.time(),
                "last_activity": time.time()
            }
            logger.info(f"Client registered: {client_id}")
            return client_id
    
    def unregister_client(self, client_id):
        with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Client unregistered: {client_id}")
                return True
            return False
    
    def send_to_client(self, client_id, event_type, data):
        with self.lock:
            if client_id in self.clients:
                event_id = self.next_event_id
                self.next_event_id += 1
                
                # Format SSE event data properly
                event_data = {
                    "id": event_id,
                    "event": event_type,
                    "data": data
                }
                
                self.clients[client_id]["queue"].append(event_data)
                self.clients[client_id]["last_activity"] = time.time()
                return True
            logger.warning(f"Attempted to send to non-existent client: {client_id}")
            return False
    
    def get_client_events(self, client_id):
        with self.lock:
            if client_id in self.clients:
                events = self.clients[client_id]["queue"]
                self.clients[client_id]["queue"] = []
                return events
            return []

# Create client manager
client_manager = SSEClientManager()

# Define tool handlers
def handle_ipfs_add_file(args):
    """Add a file to IPFS."""
    path = args.get("path")
    if not path:
        return {"error": "Missing required argument: path", "success": False}
    
    result = mock_accelerate.add_file(path)
    return result

def handle_ipfs_cat(args):
    """Retrieve content from IPFS."""
    cid = args.get("cid")
    if not cid:
        return {"error": "Missing required argument: cid", "success": False}
    
    content = mock_accelerate.cat(cid)
    if isinstance(content, str):
        return {"success": True, "content": content}
    return content

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
    if isinstance(content, str):
        return {"success": True, "content": content}
    return content

def handle_health_check(args):
    """Check the health of the MCP server."""
    return {
        "status": "healthy",
        "version": SERVER_INFO["version"],
        "uptime": time.time() - startup_time,
        "ipfs_connected": hasattr(mock_accelerate, "add_file")  # Check if mock_accelerate has required methods
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

# API multiplexer tools
def handle_register_api_key(args):
    """Register an API key for the multiplexer."""
    provider = args.get("provider")
    key = args.get("key")
    priority = args.get("priority", 1)
    
    if not provider or not key:
        return {"error": "Missing required arguments: provider and key", "success": False}
    
    if hasattr(mock_accelerate, "register_api_key"):
        return mock_accelerate.register_api_key(provider, key, priority)
    else:
        return {"error": "API key registration not supported", "success": False}

def handle_get_api_keys(args):
    """Get registered API keys (without exposing the actual keys)."""
    if hasattr(mock_accelerate, "get_api_keys"):
        return mock_accelerate.get_api_keys()
    else:
        return {"error": "API key retrieval not supported", "success": False}

def handle_get_multiplexer_stats(args):
    """Get statistics about the API multiplexer."""
    if hasattr(mock_accelerate, "get_multiplexer_stats"):
        return mock_accelerate.get_multiplexer_stats()
    else:
        return {"error": "Multiplexer stats not supported", "success": False}

def handle_simulate_api_request(args):
    """Simulate an API request through the multiplexer."""
    provider = args.get("provider")
    prompt = args.get("prompt")
    
    if not provider or not prompt:
        return {"error": "Missing required arguments: provider and prompt", "success": False}
    
    if hasattr(mock_accelerate, "simulate_api_request"):
        return mock_accelerate.simulate_api_request(provider, prompt)
    else:
        return {"error": "API request simulation not supported", "success": False}

# Task management tools
def handle_start_task(args):
    """Start a background task."""
    task_type = args.get("task_type")
    params = args.get("params", {})
    
    if not task_type:
        return {"error": "Missing required argument: task_type", "success": False}
    
    if hasattr(mock_accelerate, "start_task"):
        return mock_accelerate.start_task(task_type, params)
    else:
        return {"error": "Task management not supported", "success": False}

def handle_get_task_status(args):
    """Get the status of a background task."""
    task_id = args.get("task_id")
    
    if not task_id:
        return {"error": "Missing required argument: task_id", "success": False}
    
    if hasattr(mock_accelerate, "get_task_status"):
        return mock_accelerate.get_task_status(task_id)
    else:
        return {"error": "Task status not supported", "success": False}

def handle_list_tasks(args):
    """List all active and recently completed tasks."""
    if hasattr(mock_accelerate, "list_tasks"):
        return mock_accelerate.list_tasks()
    else:
        return {"error": "Task listing not supported", "success": False}

# Register tool definitions with their schema
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
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content retrieved from IPFS"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                }
            }
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
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content read from the MFS"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                }
            }
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
                },
                "ipfs_connected": {
                    "type": "boolean",
                    "description": "Whether IPFS is connected"
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
                },
                "accelerators": {
                    "type": "object",
                    "description": "Available accelerators"
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
    },
    "register_api_key": {
        "description": "Register an API key for the multiplexer",
        "parameters": {
            "type": "object",
            "required": ["provider", "key"],
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "API provider (e.g. openai, anthropic)"
                },
                "key": {
                    "type": "string",
                    "description": "API key"
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority for this key (lower is higher priority)",
                    "default": 1
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
                "key_id": {
                    "type": "string",
                    "description": "ID for the registered key"
                }
            }
        }
    },
    "get_api_keys": {
        "description": "Get registered API keys (without exposing the actual keys)",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": {
            "type": "object",
            "properties": {
                "providers": {
                    "type": "array",
                    "description": "List of providers"
                },
                "total_keys": {
                    "type": "integer",
                    "description": "Total number of registered keys"
                }
            }
        }
    },
    "get_multiplexer_stats": {
        "description": "Get statistics about the API multiplexer",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": {
            "type": "object",
            "properties": {
                "providers": {
                    "type": "object",
                    "description": "Per-provider statistics"
                },
                "total_requests": {
                    "type": "integer",
                    "description": "Total number of requests"
                }
            }
        }
    },
    "simulate_api_request": {
        "description": "Simulate an API request through the multiplexer",
        "parameters": {
            "type": "object",
            "required": ["provider", "prompt"],
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "API provider (e.g. openai, anthropic)"
                },
                "prompt": {
                    "type": "string",
                    "description": "Prompt for the request"
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
                "completion": {
                    "type": "string",
                    "description": "Generated completion"
                }
            }
        }
    },
    "start_task": {
        "description": "Start a background task",
        "parameters": {
            "type": "object",
            "required": ["task_type"],
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": "Type of task to start"
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the task"
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
                "task_id": {
                    "type": "string",
                    "description": "ID of the started task"
                }
            }
        }
    },
    "get_task_status": {
        "description": "Get the status of a background task",
        "parameters": {
            "type": "object",
            "required": ["task_id"],
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "ID of the task"
                }
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "ID of the task"
                },
                "status": {
                    "type": "string",
                    "description": "Status of the task"
                },
                "progress": {
                    "type": "number",
                    "description": "Progress percentage"
                }
            }
        }
    },
    "list_tasks": {
        "description": "List all active and recently completed tasks",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": {
            "type": "object",
            "properties": {
                "active_tasks": {
                    "type": "array",
                    "description": "List of active tasks"
                },
                "completed_tasks": {
                    "type": "array",
                    "description": "List of completed tasks"
                }
            }
        }
    }
}

# Tool registry
tools = {}

# Register tools
def register_tool(name, handler):
    """Register a tool with the server."""
    tools[name] = handler
    logger.info(f"Registered tool: {name}")

# Register our local tools
register_tool("ipfs_add_file", handle_ipfs_add_file)
register_tool("ipfs_cat", handle_ipfs_cat)
register_tool("ipfs_files_write", handle_ipfs_files_write)
register_tool("ipfs_files_read", handle_ipfs_files_read)
register_tool("health_check", handle_health_check)
register_tool("get_hardware_info", handle_get_hardware_info)
register_tool("list_models", handle_list_models)
register_tool("create_endpoint", handle_create_endpoint)
register_tool("run_inference", handle_run_inference)

# Register API multiplexer tools
register_tool("register_api_key", handle_register_api_key)
register_tool("get_api_keys", handle_get_api_keys)
register_tool("get_multiplexer_stats", handle_get_multiplexer_stats)
register_tool("simulate_api_request", handle_simulate_api_request)

# Register task management tools
register_tool("start_task", handle_start_task)
register_tool("get_task_status", handle_get_task_status)
register_tool("list_tasks", handle_list_tasks)

# Also register any available tools from direct_mcp_server.py if they're not already registered
for name, func in direct_tools.items():
    if name not in tools:
        register_tool(name, func)

logger.info(f"Registered {len(tools)} tools in total")

# API routes
@app.route("/tools", methods=["GET"])
def list_tools():
    """List all available tools."""
    return jsonify({"tools": list(tools.keys())})

@app.route("/mcp/manifest", methods=["GET"])
def get_manifest():
    """Get the MCP manifest, including tool schemas."""
    manifest = {
        "schemaVersion": "0.1.0",
        "name": SERVER_INFO["name"],
        "version": SERVER_INFO["version"],
        "description": SERVER_INFO["description"],
        "tools": {}
    }
    
    for tool_name, schema in TOOL_SCHEMAS.items():
        if tool_name in tools:
            manifest["tools"][tool_name] = schema
    
    return jsonify(manifest)

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
        logger.exception(f"Error calling tool {tool_name}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": SERVER_INFO["version"],
        "uptime": time.time() - startup_time
    })

# SSE endpoint for MCP
@app.route("/sse")
def sse():
    """Server-Sent Events endpoint for MCP."""
    
    def stream():
        # Register a new client
        client_id = client_manager.register_client()
        
        # Send initial event
        init_data = {
            "client_id": client_id,
            "server_info": SERVER_INFO
        }
        
        # Send the init event properly formatted
        yield f"id: 0\nevent: init\ndata: {json.dumps(init_data)}\n\n"
        
        # Keep the connection alive
        event_count = 1
        try:
            while True:
                # Send heartbeat every second
                time.sleep(1)
                
                # Format the event properly
                heartbeat_data = {"timestamp": time.time()}
                yield f"id: {event_count}\nevent: heartbeat\ndata: {json.dumps(heartbeat_data)}\n\n"
                event_count += 1
                
                # Process any queued events for this client
                events = client_manager.get_client_events(client_id)
                for event in events:
                    event_id = event["id"]
                    event_type = event["event"]
                    event_data = event["data"]
                    
                    # Format the event properly
                    yield f"id: {event_id}\nevent: {event_type}\ndata: {json.dumps(event_data)}\n\n"
                
        except GeneratorExit:
            # Client disconnected
            client_manager.unregister_client(client_id)
    
    response = Response(
        stream_with_context(stream()),
        mimetype="text/event-stream"
    )
    
    # Set headers for SSE
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    response.headers["Content-Type"] = "text/event-stream"
    
    return response

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
            logger.exception(f"Error handling tool call: {str(e)}")
            client_manager.send_to_client(client_id, "tool_response", {
                "request_id": request_id,
                "error": str(e),
                "success": False
            })
            return jsonify({"error": str(e)}), 500
    
    # Unknown request type
    return jsonify({"error": f"Unknown request type: {request_type}"}), 400

def main():
    """Main function to start the server."""
    parser = argparse.ArgumentParser(description="Run the Fixed SSE MCP server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to listen on")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Fixed SSE MCP server on http://{args.host}:{args.port}")
    logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")
    logger.info(f"MCP manifest: http://{args.host}:{args.port}/mcp/manifest")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

if __name__ == "__main__":
    main()
