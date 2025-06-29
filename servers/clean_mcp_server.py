#!/usr/bin/env python3
"""
Clean MCP Server Implementation

This server is a clean version of the final MCP server,
incorporating all necessary fixes and integrated tool registration.
It includes automatic tool registration, error recovery, and
proper shutdown handling.
"""

import os
import sys
import json
import logging
import asyncio
import signal
import argparse
import traceback
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
import shutil
import re
import socket
import subprocess
import time

print("Starting Clean MCP Server...")

# --- Handle multihash.FuncReg error ---
try:
    import multihash
    if not hasattr(multihash, 'FuncReg'):
        # Create a mock FuncReg object
        class MockFuncReg:
            @staticmethod
            def register(*args, **kwargs):
                pass
        multihash.FuncReg = MockFuncReg()
except ImportError:
    # Create a mock multihash module with FuncReg
    class MockFuncReg:
        @staticmethod
        def register(*args, **kwargs):
            pass

    class MockMultihash:
        FuncReg = MockFuncReg()

    sys.modules['multihash'] = MockMultihash()

# --- Early Setup: Logging and Path ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clean_mcp_server.log", mode='w'), # Ensure log is overwritten per run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("clean-mcp")

# --- BEGIN MCP SERVER DIAGNOSTIC LOGGING ---
logger.info(f"MCP Server sys.path at start: {sys.path}")
try:
    import transformers
    logger.info(f"MCP Server: Successfully imported 'transformers' module. Version: {getattr(transformers, '__version__', 'unknown')}")
except ImportError as e:
    logger.error(f"MCP Server: FAILED to import 'transformers' module. Error: {e}")
# --- END MCP SERVER DIAGNOSTIC LOGGING ---


# Define the version
__version__ = "1.0.0"

# Global state
PORT = 9998  # Use consistent port for all components
server_initialized = False
initialization_lock = asyncio.Lock()
initialization_event = asyncio.Event()

# Tool registration tracking
registered_tools = {}
registered_tool_categories = set()

# Import availability flags
IPFS_AVAILABLE = False
VFS_AVAILABLE = False
FS_JOURNAL_AVAILABLE = False
IPFS_FS_BRIDGE_AVAILABLE = False
MULTI_BACKEND_FS_AVAILABLE = False

try:
    import uvicorn
    from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import Response as StarletteResponse
    from starlette.background import BackgroundTask as StarletteBackgroundTask
except ImportError as e:
    logger.error(f"Failed to import FastAPI components: {e}")
    logger.info("Please install required dependencies with: pip install fastapi uvicorn[standard]")
    sys.exit(1)

try:
    import jsonrpcserver
    from jsonrpcserver import dispatch, Success, Error
    from jsonrpcserver import method as jsonrpc_method # Use this decorator
except ImportError as e:
    logger.error(f"Failed to import JSON-RPC components: {e}")
    logger.info("Please install required dependencies with: pip install jsonrpcserver")
    sys.exit(1)

# --- Dynamic Imports Check ---
def check_module_availability():
    global IPFS_AVAILABLE, VFS_AVAILABLE, FS_JOURNAL_AVAILABLE
    global IPFS_FS_BRIDGE_AVAILABLE, MULTI_BACKEND_FS_AVAILABLE

    try:
        import unified_ipfs_tools
        IPFS_AVAILABLE = True
        logger.info("✅ IPFS tools module available (unified_ipfs_tools)")
    except ImportError:
        logger.warning("⚠️ unified_ipfs_tools module not available")

    try:
        import mcp_vfs_config
        VFS_AVAILABLE = True
        logger.info("✅ Virtual filesystem module available (mcp_vfs_config)")
    except ImportError:
        logger.warning("⚠️ mcp_vfs_config module not available")

    try:
        import fs_journal_tools
        FS_JOURNAL_AVAILABLE = True
        logger.info("✅ Filesystem journal module available (fs_journal_tools)")
    except ImportError:
        logger.warning("⚠️ fs_journal_tools module not available")

    try:
        import ipfs_mcp_fs_integration
        IPFS_FS_BRIDGE_AVAILABLE = True
        logger.info("✅ IPFS-FS bridge module available (ipfs_mcp_fs_integration)")
    except ImportError:
        logger.warning("⚠️ ipfs_mcp_fs_integration module not available")

    try:
        import multi_backend_fs_integration
        MULTI_BACKEND_FS_AVAILABLE = True
        logger.info("✅ Multi-backend filesystem module available (multi_backend_fs_integration)")
    except ImportError:
        logger.warning("⚠️ multi_backend_fs_integration module not available")

    try:
        import integrate_vfs_to_final_mcp
        logger.info("✅ VFS integration module available (integrate_vfs_to_final_mcp)")
    except ImportError:
        logger.warning("⚠️ integrate_vfs_to_final_mcp module not available")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("MCP Server startup_event triggered in lifespan.")
    check_module_availability()
    if not setup_jsonrpc():
        logger.critical("CRITICAL: Failed to set up JSON-RPC methods. Server may not function correctly.")
    
    # Import IPFS Accelerate module (or mock)
    accel = import_ipfs_accelerate_py()

    # --- Runtime Check and Logging for server instance ---
    logger.info(f"Type of server instance before registration: {type(server)}")
    logger.info(f"Attributes of server instance before registration: {dir(server)}")
    if hasattr(server, 'register_resource'):
        logger.info("server instance HAS 'register_resource' attribute.")
    else:
        logger.error("server instance DOES NOT HAVE 'register_resource' attribute.")
    # --- End Runtime Check and Logging ---

    if not register_all_tools(server, accel): # Pass server instance and accel
        logger.critical("CRITICAL: Failed to register tools. Server will have limited functionality.")
    
    logger.info("MCP Server initialization sequence complete.")
    yield
    # Shutdown code here

app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol Server with IPFS and VFS Integration",
    version=__version__,
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def handle_sigterm(signum, frame):
    logger.info("Received SIGTERM, shutting down")
    sys.exit(0)

def handle_sigint(signum, frame):
    logger.info("Received SIGINT, shutting down")
    sys.exit(0)

class ToolWrapper:
    """Wrapper to hold tool function and metadata."""
    def __init__(self, func: Callable, description: str, parameter_descriptions: Optional[Dict[str, Any]]):
        self.func = func
        self.description = description
        self.parameter_descriptions = parameter_descriptions or {}

    # Make the wrapper callable
    async def __call__(self, *args, **kwargs):
        import asyncio
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

class MCPServer:
    def __init__(self):
        self.tools: Dict[str, ToolWrapper] = {}; # Store ToolWrapper instances
        self.resources = {}
        # tool_descriptions and parameter_descriptions are now stored within ToolWrapper
        # self.tool_descriptions = {}
        # self.parameter_descriptions = {}

    def tool(self, name: str, description: str = "", parameter_descriptions: Optional[Dict[str, str]] = None):
        def decorator(func):
            self.register_tool(name, func, description, parameter_descriptions);
            return func
        return decorator

    def register_tool(self, name: str, func: Callable, description: str = "", parameter_descriptions: Optional[Dict[str, str]] = None):
        """Register a tool with the server"""
        if name in self.tools:
            logger.warning(f"Tool {name} already registered, overwriting")

        # Store the tool wrapped in ToolWrapper
        self.tools[name] = ToolWrapper(func, description, parameter_descriptions)
        logger.debug(f"Registered tool '{name}' with description '{description}')")


    def register_resource(self, name: str, description: str, getter: Callable):
        """Register a resource with the server"""
        if name in self.resources:
            logger.warning(f"Resource {name} already registered, overwriting")

        # Store the resource getter
        self.resources[name] = {"description": description, "getter": getter}

    def generic_handler(self, **kwargs):
        """Generic handler for tools that don't have specific implementations"""
        tool_name = kwargs.pop('tool_name', None)
        if not tool_name:
            return {"error": "No tool_name specified for generic handler"}

        logger.info(f"Generic handler called for tool: {tool_name}")

        try:
            # Check if the tool exists in high-level API
            if hasattr(self.ipfs_api, tool_name):
                method = getattr(self.ipfs_api, tool_name)
                return method(**kwargs)

            # Otherwise, try direct IPFS API
            elif hasattr(self.ipfs_api.ipfs, tool_name):
                method = getattr(self.ipfs_api.ipfs, tool_name)
                return method(**kwargs)

            # Try any registered tools (now stored as ToolWrapper)
            elif tool_name in self.tools and isinstance(self.tools[tool_name], ToolWrapper):
                 # Retrieve the function from the wrapper and call it
                 return self.tools[tool_name].func(**kwargs)

            else:
                return {"error": f"Tool {tool_name} not found in any available APIs"}
        except Exception as e:
            logger.error(f"Error in generic_handler for {tool_name}: {e}")
            return {"error": str(e)}

    # Original method implementations
    # ...existing code...

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any] = None, context: Optional[Dict[str, Any]] = None):
        arguments = arguments or {}; context_obj = SimpleContext(context or {})
        logger.debug(f"Attempting to execute tool: {tool_name} with arguments: {arguments}") # Added logging
        if tool_name not in self.tools or not isinstance(self.tools[tool_name], ToolWrapper):
            logger.error(f"Tool {tool_name} not found or not a valid ToolWrapper instance")
            return {"error": f"Tool {tool_name} not found or invalid"}

        tool_wrapper = self.tools[tool_name]
        func = tool_wrapper.func # Get the actual function from the wrapper

        try:
            import inspect
            # Check the signature of the actual function
            sig = inspect.signature(func)
            logger.debug(f"Calling tool function: {tool_name}") # Added logging
            if "ctx" in sig.parameters or "context" in sig.parameters:
                logger.debug(f"Executing tool {tool_name} with context object.") # Added logging
                result = await func(context_obj, **arguments) if asyncio.iscoroutinefunction(func) else func(context_obj, **arguments)
            else:
                logger.debug(f"Executing tool {tool_name} without context object.") # Added logging
                result = await func(**arguments) if asyncio.iscoroutinefunction(func) else func(**arguments)
            logger.debug(f"Tool function {tool_name} returned.") # Added logging
            logger.debug(f"Tool {tool_name} execution completed. Result type: {type(result)}") # Added logging
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

class SimpleContext:
  def __init__(self, data: Optional[Dict[str, Any]] = None): self.data = data or {}
  async def info(self, message: str): logger.info(message)
  async def error(self, message: str): logger.error(message)
  async def warn(self, message: str): logger.warning(message)
  async def debug(self, message: str): logger.debug(message)

server = MCPServer()

# Import from the ipfs_accelerate_py module
def import_ipfs_accelerate_py():
    """Import the IPFS Accelerate Python module."""
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ipfs_accelerate_py import ipfs_accelerate_py
        logger.info("Imported ipfs_accelerate_py")
        return ipfs_accelerate_py()
    except ImportError:
        logger.warning("Failed to import ipfs_accelerate_py directly")
    
    # Try alternative methods if direct import fails
    try:
        # Try to find the module in the local directory
        module_path = os.path.join(os.path.dirname(__file__), "ipfs_accelerate_py.py")
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location("ipfs_accelerate_py", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info("Imported ipfs_accelerate_py from local file")
            return module.ipfs_accelerate_py()
    except Exception as e:
        logger.error(f"Failed to import ipfs_accelerate_py: {e}")
    
    # If all fails, return a mock implementation
    logger.warning("Using mock implementation for ipfs_accelerate_py")
    return MockIPFSAccelerate()

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
            logger.error(f"Error in ipfs_get: {str(e)}")
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

# --- Tool Registration Functions (Copied from fix_mcp_tool_registration.py) ---

def register_all_tools(server_instance: MCPServer, accel: Any):
  logger.info("START: Registering all available tools with MCP server...")
  successful_tools = []
  try:
    # Define placeholder getter functions for resources
    def get_system_info():
        return {"message": "System info getter not implemented"}

    def get_accelerator_info():
        return {"message": "Accelerator info getter not implemented"}

    def get_ipfs_nodes():
        return {"message": "IPFS nodes getter not implemented"}

    def get_models_info():
        return {"message": "Models info getter not implemented"}

    def get_storage_info_resource():
        return {"message": "Storage info getter not implemented"}

    # Register resources with placeholder getters (Bypassing register_resource due to AttributeError)
    server_instance.resources["system_info"] = {"description": "Information about the system", "getter": get_system_info}
    server_instance.resources["accelerator_info"] = {"description": "Information about available hardware accelerators", "getter": get_accelerator_info}
    server_instance.resources["ipfs_nodes"] = {"description": "Information about connected IPFS nodes", "getter": get_ipfs_nodes}
    server_instance.resources["models"] = {"description": "Information about available models", "getter": get_models_info}
    server_instance.resources["storage"] = {"description": "Information about available storage", "getter": get_storage_info_resource}
    logger.info("Resources registered by directly modifying server_instance.resources")

    # Attempt comprehensive VFS integration (optional, can fail if module not present)
    try:
      import integrate_vfs_to_final_mcp
      logger.info("Attempting comprehensive VFS integration")
      result = integrate_vfs_to_final_mcp.register_all_components(server_instance)
      if isinstance(result, dict):
        for component, success in result.items():
          if success:
            successful_tools.append(component)
            registered_tool_categories.add(component)
            logger.info(f"Successfully integrated component: {component}")
      elif result:
        successful_tools.append("integrated_components")
        registered_tool_categories.add("integrated_components")
        logger.info("Successfully integrated components")
    except ImportError:
      logger.warning("Comprehensive integration (integrate_vfs_to_final_mcp) not available.")
    except Exception as e:
      logger.error(f"Error in comprehensive integration: {e}\n{traceback.format_exc()}")

    # Always attempt individual tool registrations regardless of comprehensive integration success
    logger.info("Attempting individual tool registrations...")
    
    try:
      register_hardware_tools(server_instance, accel)
      successful_tools.append("hardware_tools")
      registered_tool_categories.add("hardware_tools")
      logger.info("Hardware tools registered successfully")
    except Exception as e:
      logger.error(f"Error registering hardware tools: {e}")

    try:
      register_ipfs_tools(server_instance, accel)
      successful_tools.append("ipfs_tools")
      registered_tool_categories.add("ipfs_tools")
      logger.info("IPFS tools registered successfully")
    except Exception as e:
      logger.error(f"Error registering IPFS tools: {e}")

    try:
      register_model_tools(server_instance, accel)
      successful_tools.append("model_tools")
      registered_tool_categories.add("model_tools")
      logger.info("Model tools registered successfully")
    except Exception as e:
      logger.error(f"Error registering model tools: {e}")

    try:
      register_vfs_tools(server_instance, accel)
      successful_tools.append("vfs_tools")
      registered_tool_categories.add("vfs_tools")
      logger.info("VFS tools registered successfully")
    except Exception as e:
      logger.error(f"Error registering VFS tools: {e}")

    try:
      register_storage_tools(server_instance, accel)
      successful_tools.append("storage_tools")
      registered_tool_categories.add("storage_tools")
      logger.info("Storage tools registered successfully")
    except Exception as e:
      logger.error(f"Error registering Storage tools: {e}")

    # Add other individual tool registrations here if needed (e.g., FS_JOURNAL, IPFS_FS_BRIDGE, MULTI_BACKEND)
    # Example:
    # try:
    #   register_fs_journal_tools(server_instance, accel)
    #   successful_tools.append("fs_journal_tools")
    #   registered_tool_categories.add("fs_journal_tools")
    #   logger.info("FS Journal tools registered successfully")
    # except Exception as e:
    #   logger.error(f"Error registering FS Journal tools: {e}")


    logger.info(f"END: Successfully registered tool categories: {', '.join(successful_tools) or 'None'}")
    logger.info(f"Registered tool categories: {registered_tool_categories}")
    logger.info("Tool registration process finished.")
    return True
  except Exception as e:
    logger.error(f"Error during tool registration: {e}\n{traceback.format_exc()}")
    return False

def register_hardware_tools(server_instance: MCPServer, accel: Any):
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
    
    server_instance.register_tool("get_hardware_info", "Get hardware information about the system", get_hardware_info, {})
    logger.info("Registered hardware tools")

def register_ipfs_tools(server_instance: MCPServer, accel: Any):
    """Register IPFS-related tools."""
    # IPFS Add File
    def ipfs_add_file(path):
        """Add a file to IPFS."""
        try:
            return accel.add_file(path)
        except Exception as e:
            logger.error(f"Error in ipfs_add_file: {str(e)}")
            return {"error": str(e), "success": False}
    
    server_instance.register_tool("ipfs_add_file", "Add a file to IPFS", ipfs_add_file, {
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
    
    server_instance.register_tool("ipfs_node_info", "Get information about the IPFS node", ipfs_node_info, {})
    
    # IPFS Cat
    def ipfs_cat(cid):
        """Get the content of a file from IPFS."""
        try:
            return accel.cat(cid)
        except Exception as e:
            logger.error(f"Error in ipfs_cat: {str(e)}")
            return {"error": str(e), "success": False}
    
    server_instance.register_tool("ipfs_cat", "Get the content of a file from IPFS", ipfs_cat, {
        "type": "object",
        "properties": {
            "cid": {"type": "string", "description": "CID of the file"}
        },
        "required": ["cid"]
    })
    
    # IPFS Get
    def ipfs_get(cid, output_path):
        """Get a file from IPFS and save it to a local path."""
        try:
            return accel.get(cid, output_path)
        except Exception as e:
            logger.error(f"Error in ipfs_get: {str(e)}")
            return {"error": str(e), "success": False}
    
    server_instance.register_tool("ipfs_get", "Get a file from IPFS", ipfs_get, {
        "type": "object",
        "properties": {
            "cid": {"type": "string", "description": "CID of the file"},
            "output_path": {"type": "string", "description": "Path to save the file"}
        },
        "required": ["cid", "output_path"]
    })
    
    logger.info("Registered IPFS tools")

def register_model_tools(server_instance: MCPServer, accel: Any):
    """Register model-related tools."""
    # Model Inference
    def model_inference(model_name, input_data, endpoint_type=None):
        """Run inference on a model."""
        try:
            return accel.process(model_name, input_data, endpoint_type)
        except Exception as e:
            logger.error(f"Error in model_inference: {str(e)}")
            return {"error": str(e)}
    
    server_instance.register_tool("model_inference", "Run inference on a model", model_inference, {
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
    
    server_instance.register_tool("list_models", "List available models", list_models, {})
    
    # Initialize Endpoints
    def init_endpoints(models):
        """Initialize endpoints for models."""
        try:
            import asyncio
            result = asyncio.run(accel.init_endpoints(models))
            return result
        except Exception as e:
            logger.error(f"Error in init_endpoints: {str(e)}")
            return {"error": str(e)}
    
    server_instance.register_tool("init_endpoints", "Initialize endpoints for models", init_endpoints, {
        "type": "object",
        "properties": {
            "models": {"type": "array", "items": {"type": "string"}, "description": "List of models to initialize"}
        },
        "required": ["models"]
    })
    
    logger.info("Registered model tools")

def register_vfs_tools(server_instance: MCPServer, accel: Any):
    """Register virtual filesystem tools."""
    # These are basic implementations since we don't have the actual VFS implementation
    
    # VFS List
    def vfs_list(path="/"):
        """List items in the virtual filesystem."""
        try:
            # Mock implementation
            return {
                "path": path,
                "items": ["file1.txt", "file2.txt", "dir1/"],
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in vfs_list: {str(e)}")
            return {"error": str(e), "success": False}
    
    server_instance.register_tool("vfs_list", "List items in the virtual filesystem", vfs_list, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path in the virtual filesystem"}
        }
    })
    
    logger.info("Registered VFS tools")

def register_storage_tools(server_instance: MCPServer, accel: Any):
    """Register storage-related tools."""
    # These are basic implementations since we don't have the actual storage implementation
    
    # Create Storage
    def create_storage(name, size):
        """Create a new storage volume."""
        try:
            # Mock implementation
            return {
                "name": name,
                "size": size,
                "id": f"storage-{name}-{size}",
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in create_storage: {str(e)}")
            return {"error": str(e), "success": False}
    
    server_instance.register_tool("create_storage", "Create a new storage volume", create_storage, {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the storage volume"},
            "size": {"type": "number", "description": "Size of the storage volume in GB"}
        },
        "required": ["name", "size"]
    })
    
    logger.info("Registered storage tools")

# --- End Tool Registration Functions ---


def setup_jsonrpc():
    """Inform that JSON-RPC methods are registered via decorators."""
    logger.info("JSON-RPC methods are registered using @jsonrpc_method decorator.")
    # Perform any other non-method-registration setup for jsonrpcserver if needed
    return True


async def handle_jsonrpc(request: Request):
    """Handle JSON-RPC requests."""
    request_text = ""
    try:
        request_body = await request.body()
        request_text = request_body.decode()
        logger.debug(f"Received JSON-RPC request data: {request_text}")

        # Parse the request to get method and ID
        try:
            parsed_req = json.loads(request_text)
            method = parsed_req.get("method")
            req_id = parsed_req.get("id")
            params = parsed_req.get("params", {})
            logger.debug(f"JSON-RPC method: {method}, id: {req_id}, params: {params}") # Added logging
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in request: {request_text}")
            return JSONResponse(
                content={"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
                status_code=400
            )

        # Special handling for common methods
        if method == "ping":
            # Direct handling for ping to ensure consistent and fast response
            return JSONResponse(
                content={"jsonrpc": "2.0", "result": "pong", "id": req_id},
                status_code=200
            )
        elif method == "get_tools" or method == "list_tools":
        # Direct handling for get_tools and list_tools (alias)
            # Return a flat array of tools for VS Code compatibility
            tool_list = []
            for name, tool in server.tools.items():
                # Check if tool is a dictionary with description and parameters
                if isinstance(tool, dict) and "description" in tool and "parameters" in tool:
                     tool_list.append({"name": name, "description": tool["description"], "parameters": tool["parameters"]})
                # Check if tool is a function and get info from server attributes
                elif callable(tool):
                    description = server.tool_descriptions.get(name, "")
                    parameters = server.parameter_descriptions.get(name, {})
                    tool_list.append({"name": name, "description": description, "parameters": parameters})
                else:
                    logger.warning(f"Tool '{name}' has unexpected format: {type(tool)}")


            return JSONResponse(
                content={"jsonrpc": "2.0", "result": tool_list, "id": req_id},
                status_code=200
            )
        elif method == "get_server_info":
            # Direct handling for get_server_info
            uptime = datetime.now() - server_start_time
            info = {
                "version": __version__,
                "uptime_seconds": uptime.total_seconds(),
                "port": PORT,
                "registered_tools": len(server.tools),
                "registered_tool_categories": list(registered_tool_categories)
            }
            return JSONResponse(
                content={"jsonrpc": "2.0", "result": info, "id": req_id},
                status_code=200
            )
        elif method == "initialize":
            # Special handling for initialize
            global server_initialized
            client_info = params.get("clientInfo", {})
            logger.info(f"Received HTTP initialize request: {client_info}") # Log as HTTP initialize for clarity

            # Mark server as initialized
            server_initialized = True
            initialization_event.set()

            # Return server capabilities
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "result": {
                        "server": "clean-mcp-server", # Updated server name
                        "version": __version__,
                        "supported_models": ["default"],
                        "capabilities": {
                            "streaming": True,
                            "jsonrpc": True,
                            "tools": True,
                            "completion": False,
                            "chat": False
                        }
                    },
                    "id": req_id
                },
                status_code=200
            )
        elif method == "use_tool":
            # Special handling for use_tool since it's asynchronous
            try:
                tool_name = params.get("tool_name")
                arguments = params.get("arguments", {})
                context = params.get("context", {})

                if not tool_name:
                    return JSONResponse(
                        content={"jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params: tool_name is required"}, "id": req_id},
                        status_code=400
                    )

                # Execute the tool
                result = await server.execute_tool(tool_name, arguments, context)
                return JSONResponse(
                    content={"jsonrpc": "2.0", "result": result, "id": req_id},
                    status_code=200
                )
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return JSONResponse(
                    content={"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Tool execution error: {str(e)}"}, "id": req_id},
                    status_code=500
                )
        elif method == "execute":
            # Direct handling for execute
            tool_name = params.get("name")
            tool_params = params.get("parameters", {})

            if not tool_name:
                return JSONResponse(
                    content={"jsonrpc": "2.0", "error": {"code": -32602, "message": "Tool name is required"}, "id": req_id},
                    status_code=400
                )

            try:
                # Execute the tool
                result = await server.execute_tool(tool_name, tool_params)
                return JSONResponse(
                    content={"jsonrpc": "2.0", "result": result, "id": req_id},
                    status_code=200
                )
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}\n{traceback.format_exc()}")
                return JSONResponse(
                    content={"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": req_id},
                    status_code=500
            )

        # For all other methods, route through server.execute_tool
        try:
            tool_name = method
            arguments = params

            # Execute the tool
            result = await server.execute_tool(tool_name, arguments)

            # Wrap the result in a JSON-RPC response format
            return JSONResponse(
                content={"jsonrpc": "2.0", "result": result, "id": req_id},
                status_code=200
            )
        except Exception as e:
            logger.error(f"Error executing tool {method}: {e}\n{traceback.format_exc()}")
            return JSONResponse(
                content={"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Tool execution error: {str(e)}"}, "id": req_id},
                status_code=500
            )

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in handle_jsonrpc: {e}. Request body: '{request_text}'")
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}, status_code=400)
    except Exception as e:
        logger.error(f"Error handling JSON-RPC request in handle_jsonrpc: {e}\n{traceback.format_exc()}")
        req_id = None
        try:
            # Try to parse request_text to get id if it's valid JSON
            if request_text:
                parsed_req = json.loads(request_text)
                req_id = parsed_req.get("id")
        except: # pylint: disable=bare-except
            pass
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Internal server error: {str(e)}"}, "id": req_id}, status_code=500)

@app.get("/")
async def root(): return {"message": "MCP Server", "version": __version__, "tools": len(server.tools)}

@app.get("/health")
async def health():
    uptime = datetime.now() - server_start_time
    return {"status": "ok", "version": __version__, "uptime_seconds": uptime.total_seconds(),
            "tools_count": len(server.tools), "registered_tool_categories": list(registered_tool_categories)}

@app.post("/jsonrpc")
async def jsonrpc_endpoint(request: Request): return await handle_jsonrpc(request)

@app.post("/initialize")
async def initialize_endpoint(request: Request):
    """Initialize endpoint for clients that use HTTP instead of JSON-RPC."""
    global server_initialized

    try:
        # Parse the request body as JSON if it exists
        client_info = {}
        if request.headers.get("content-length") and int(request.headers.get("content-length", "0")) > 0:
            client_info = await request.json()

        logger.info(f"Received HTTP initialize request: {client_info}")

        # Mark server as initialized
        server_initialized = True
        initialization_event.set()

        # Return server capabilities
        return {
            "server": "clean-mcp-server", # Updated server name
            "version": __version__,
            "supported_models": ["default"],
            "capabilities": {
                "streaming": True,
                "jsonrpc": True,
                "tools": True,
                "completion": False,
                "chat": False
            }
        }
    except Exception as e:
        logger.error(f"Error in initialize endpoint: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/tools")
async def list_tools_endpoint():
    # Return a flat array of tools for VS Code compatibility
    return [{"name": n, "description": t["description"], "parameters": t["parameters"]} for n, t in server.tools.items()]

@app.get("/tools-with-metadata")
async def list_tools_with_metadata_endpoint():
    # Return the original format with metadata for test runners
    return {"tools": [{"name": n, "description": t["description"], "parameters": t["parameters"]} for n, t in server.tools.items()],
            "count": len(server.tools), "categories": list(registered_tool_categories)}

@app.post("/tools/{tool_name}")
async def execute_tool_endpoint(tool_name: str, request: Request):
    """Execute a tool directly via HTTP POST."""
    try:
        # Parse the request body as JSON
        body = await request.json()

        # Execute the tool
        result = await server.execute_tool(tool_name, body)

        # Return the result
        return result
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

server_start_time = datetime.now()

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)

    parser = argparse.ArgumentParser(description="Clean MCP Server") # Updated description
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    PORT = args.port

    # Enhanced port check function
    def check_port_availability(host, port, attempt_kill=False):
        """Check if port is available and optionally attempt to kill the process using it"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            s.close()
            logger.info(f"Port {port} on {host} is available")
            return True
        except socket.error as e:
            if e.errno == 98 or e.errno == 10048:  # Address already in use (Linux/Windows)
                logger.warning(f"Port {port} on {host} is already in use")
                
                if attempt_kill:
                    try:
                        # Try to get the PID using the port
                        if os.name == 'posix':  # Linux/Unix
                            try:
                                cmd = f"lsof -t -i:{port}"
                                pid = subprocess.check_output(cmd, shell=True).decode().strip()
                                if pid:
                                    logger.warning(f"Attempting to kill process {pid} using port {port}...")
                                    os.kill(int(pid), signal.SIGTERM)
                                    time.sleep(1)  # Give it time to die
                                    
                                    # Check if killed
                                    try:
                                        os.kill(int(pid), 0)  # Check if process exists
                                        logger.warning(f"Process {pid} still running, forcing kill...")
                                        os.kill(int(pid), signal.SIGKILL)
                                        time.sleep(1)
                                    except OSError:  # Process doesn't exist
                                        logger.info(f"Successfully killed process {pid}")
                                    
                                    # Retry binding
                                    return check_port_availability(host, port, False)
                            except Exception as ke:
                                logger.error(f"Failed to kill process using port {port}: {ke}")
                        else:
                            logger.warning("Port kill not supported on this platform")
                    except Exception as kill_error:
                        logger.error(f"Error attempting to kill process: {kill_error}")
                
                logger.error(f"Port {port} is in use, please use a different port or stop the process using it")
                return False
            else:
                logger.error(f"Socket error: {e}")
                return False
    
    # Check if port is available or try to free it
    if not check_port_availability(args.host, args.port, attempt_kill=True):
        sys.exit(1)

    pid_file = Path("clean_mcp_server.pid") # Updated PID file name
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        logger.info(f"Starting server on {args.host}:{args.port}, debug={args.debug}, PID: {os.getpid()}")

        # Set timeouts to prevent hanging during initialization
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="debug" if args.debug else "info",
            timeout_keep_alive=30,
            timeout_graceful_shutdown=10
        )
    except Exception as e:
        logger.error(f"Server failed to start or run: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if pid_file.exists():
                pid_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error removing PID file: {e}")
        logger.info("Server shutdown complete.")
