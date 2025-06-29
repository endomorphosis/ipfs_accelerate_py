#!/usr/bin/env python3
"""
Simple MCP Tool Registration

This script registers essential tools with the MCP server using direct access
to the server's registration function.
"""

import logging
import platform
import psutil
import os
import sys
import hashlib
import random
from typing import Dict, Any, List, Optional
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import MCP server functions
try:
    # First try to import directly
    from mcp.server import register_tool, register_resource
except ImportError:
    # If that fails, try to find the module
    logger.warning("Could not import directly, trying to find mcp module...")
    import sys
    import importlib.util
    try:
        spec = importlib.util.find_spec("mcp")
        if spec:
            mcp = importlib.util.module_from_spec(spec)
            sys.modules["mcp"] = mcp
            spec.loader.exec_module(mcp)
            from mcp.server import register_tool, register_resource
        else:
            logger.error("Could not find mcp module")
            raise ImportError("Could not find mcp module")
    except Exception as e:
        logger.error(f"Failed to import mcp module: {e}")
        raise

# Mock IPFS class for demo purposes
class MockIPFS:
    """Simple mock IPFS implementation for demonstration."""
    
    def __init__(self):
        self.files = {}
        self.cids = {}
    
    def add_file(self, path):
        """Add a file to IPFS."""
        if not os.path.exists(path):
            return {"error": "File not found", "success": False}
        
        # Generate a mock CID
        with open(path, 'rb') as f:
            content = f.read()
            hash_obj = hashlib.sha256(content)
            cid = f"QmPy{hash_obj.hexdigest()[:16]}"
            
        # Store the file
        self.cids[cid] = {
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
    
    def get_file(self, cid, output_path):
        """Get a file from IPFS."""
        if cid not in self.cids:
            return {"error": "CID not found", "success": False}
        
        # Write content to output path
        with open(output_path, 'wb') as f:
            f.write(self.cids[cid]["content"])
        
        return {
            "path": output_path,
            "size": self.cids[cid]["size"],
            "success": True
        }
    
    def cat_file(self, cid):
        """Get the content of a file from IPFS."""
        if cid not in self.cids:
            return {"error": "CID not found", "success": False}
        
        return {
            "content": self.cids[cid]["content"].decode('utf-8', errors='replace'),
            "size": self.cids[cid]["size"],
            "success": True
        }

def register_resources():
    """Register resources with the MCP server."""
    try:
        # Register hardware resources
        register_resource("system_info", "Information about the system")
        register_resource("accelerator_info", "Information about available hardware accelerators")
        register_resource("ipfs_nodes", "Information about connected IPFS nodes")
        logger.info("Successfully registered resources")
        return True
    except Exception as e:
        logger.error(f"Error registering resources: {e}")
        return False

def register_hardware_tools():
    """Register hardware-related tools."""
    try:
        # Register get_hardware_info tool
        def get_hardware_info():
            """Get hardware information about the system."""
            try:
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
                        },
                        "webgpu": {
                            "available": False,
                            "version": None,
                            "devices": []
                        },
                        "webnn": {
                            "available": False,
                            "version": None,
                            "devices": []
                        }
                    }
                }
                
                # Try to get better hardware info from hardware_detection module
                try:
                    import hardware_detection
                    if hasattr(hardware_detection, "detect_all_hardware"):
                        better_info = hardware_detection.detect_all_hardware()
                        if better_info:
                            hardware_info = better_info
                except ImportError:
                    pass
                
                return hardware_info
            except Exception as e:
                logger.error(f"Error in get_hardware_info: {str(e)}")
                return {"error": str(e)}
        
        # Register the tool
        register_tool("get_hardware_info", "Get hardware information about the system", get_hardware_info, {})
        logger.info("Successfully registered get_hardware_info tool")
        return True
    except Exception as e:
        logger.error(f"Error registering hardware tools: {e}")
        return False

def register_ipfs_tools():
    """Register IPFS-related tools."""
    try:
        # Create IPFS instance
        ipfs = MockIPFS()
        
        # Register ipfs_add_file tool
        def ipfs_add_file(path):
            """Add a file to IPFS."""
            try:
                return ipfs.add_file(path)
            except Exception as e:
                logger.error(f"Error in ipfs_add_file: {e}")
                return {"error": str(e), "success": False}
        
        register_tool("ipfs_add_file", "Add a file to IPFS", ipfs_add_file, {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"]
        })
        logger.info("Successfully registered ipfs_add_file tool")
        
        # Register ipfs_node_info tool
        def ipfs_node_info():
            """Get information about the IPFS node."""
            return {
                "id": "QmNodeMockIPFSAccelerate",
                "version": "0.1.0",
                "protocol_version": "ipfs/0.1.0",
                "agent_version": "ipfs-accelerate/0.1.0",
                "success": True
            }
        
        register_tool("ipfs_node_info", "Get information about the IPFS node", ipfs_node_info, {})
        logger.info("Successfully registered ipfs_node_info tool")
        
        return True
    except Exception as e:
        logger.error(f"Error registering IPFS tools: {e}")
        return False

def register_virtual_filesystem_tools():
    """Register IPFS virtual filesystem tools."""
    try:
        # First try to import directly
        try:
            from ipfs_accelerate_py.mcp.tools.ipfs_vfs import register_with_mcp
            # Pass the register_tool function directly
            register_with_mcp(register_tool)
            logger.info("Successfully registered IPFS virtual filesystem tools")
            return True
        except ImportError:
            logger.warning("Could not import IPFS virtual filesystem module directly, trying alternative...")
            
            # Try alternative import methods
            vfs_module_path = os.path.join(os.path.dirname(__file__), "ipfs_accelerate_py", "mcp", "tools", "ipfs_vfs.py")
            if os.path.exists(vfs_module_path):
                spec = importlib.util.find_spec("ipfs_accelerate_py.mcp.tools.ipfs_vfs")
                if spec:
                    vfs_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(vfs_module)
                    vfs_module.register_with_mcp(register_tool)
                    logger.info("Successfully registered IPFS virtual filesystem tools via spec")
                    return True
            
            logger.warning("IPFS virtual filesystem module not found, virtual filesystem tools not registered")
            return False
    except Exception as e:
        logger.error(f"Error registering IPFS virtual filesystem tools: {e}")
        return False

def register_model_tools():
    """Register model-related tools."""
    try:
        # Register model_inference tool
        def model_inference(model_name, input_data, endpoint_type=None):
            """Run inference on a model."""
            return {
                "model": model_name,
                "input": str(input_data)[:100],
                "endpoint_type": endpoint_type,
                "output": "This is a mock response from the model.",
                "is_mock": True
            }
        
        register_tool("model_inference", "Run inference on a model", model_inference, {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Name of the model"},
                "input_data": {"description": "Input data for the model"},
                "endpoint_type": {"type": "string", "description": "Endpoint type (optional)"}
            },
            "required": ["model_name", "input_data"]
        })
        logger.info("Successfully registered model_inference tool")
        
        # Register list_models tool
        def list_models():
            """List available models."""
            return {
                "models": [
                    {"name": "bert-base-uncased", "type": "embedding"},
                    {"name": "t5-small", "type": "text-generation"},
                    {"name": "gpt2", "type": "text-generation"},
                    {"name": "vit-base", "type": "image-classification"}
                ],
                "count": 4
            }
        
        register_tool("list_models", "List available models", list_models, {})
        logger.info("Successfully registered list_models tool")
        
        return True
    except Exception as e:
        logger.error(f"Error registering model tools: {e}")
        return False

def main():
    """Main entry point for the script."""
    logger.info("Starting MCP tool registration...")
    
    # Register resources
    register_resources()
    
    # Register tools
    success_hw = register_hardware_tools()
    success_ipfs = register_ipfs_tools()
    success_vfs = register_virtual_filesystem_tools()
    success_model = register_model_tools()
    
    if success_hw and success_ipfs and success_vfs and success_model:
        logger.info("Successfully registered all tools")
        return 0
    else:
        logger.warning("Some tools failed to register")
        return 1

if __name__ == "__main__":
    sys.exit(main())
