#!/usr/bin/env python
"""
IPFS Accelerate MCP Tools Registration Script

This script registers all tools from ipfs_accelerate_py with the MCP server.
"""

import os
import sys
import logging
import importlib
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the script"""
    logger.info("Starting MCP tools registration")
    
    # Import the MCP server
    try:
        from mcp.server import register_tool
        logger.info("Successfully imported MCP server")
    except ImportError as e:
        logger.error(f"Error importing MCP server: {e}")
        return 1
    
    # Register hardware tools
    try:
        import platform
        import psutil
        
        # Register get_hardware_info tool
        def get_hardware_info():
            """Get hardware information about the system."""
            try:
                # Try to import hardware_detection module
                hardware_info = {}
                try:
                    import hardware_detection
                    if hasattr(hardware_detection, "detect_all_hardware"):
                        hardware_info = hardware_detection.detect_all_hardware()
                    elif hasattr(hardware_detection, "detect_hardware"):
                        hardware_info = hardware_detection.detect_hardware()
                except ImportError:
                    # Fallback to basic hardware info
                    hardware_info = {
                        "system": {
                            "os": platform.system(),
                            "os_version": platform.version(),
                            "distribution": platform.platform(),
                            "architecture": platform.machine(),
                            "python_version": platform.python_version(),
                            "processor": platform.processor()
                        }
                    }
                    
                    # Add memory info if available
                    try:
                        hardware_info["system"]["memory_total"] = round(psutil.virtual_memory().total / (1024**3), 2)
                        hardware_info["system"]["memory_available"] = round(psutil.virtual_memory().available / (1024**3), 2)
                    except Exception:
                        pass
                    
                    # Add CPU info if available
                    try:
                        hardware_info["system"]["cpu"] = {
                            "cores_physical": psutil.cpu_count(logical=False),
                            "cores_logical": psutil.cpu_count(logical=True)
                        }
                    except Exception:
                        pass
                
                return hardware_info
            except Exception as e:
                logger.error(f"Error in get_hardware_info: {str(e)}")
                return {"error": str(e)}
        
        # Register the tool
        register_tool("get_hardware_info", "Get hardware information about the system", get_hardware_info)
        logger.info("Successfully registered get_hardware_info tool")
        
        # Register test_hardware tool
        def test_hardware():
            """Test hardware capabilities for machine learning."""
            return {
                "cpu": {"available": True, "performance": "baseline"},
                "cuda": {"available": "unknown", "needs_testing": True},
                "rocm": {"available": "unknown", "needs_testing": True},
                "openvino": {"available": "unknown", "needs_testing": True},
                "mps": {"available": "unknown", "needs_testing": True}
            }
        
        register_tool("test_hardware", "Test hardware capabilities", test_hardware)
        logger.info("Successfully registered test_hardware tool")
        
        # Register recommend_hardware tool
        def recommend_hardware(model_name: str):
            """Get hardware recommendations for a model."""
            return {
                "model": model_name,
                "recommendations": [
                    {"hardware": "cuda", "priority": 1, "description": "Best performance with NVIDIA GPUs"},
                    {"hardware": "rocm", "priority": 2, "description": "Good performance with AMD GPUs"},
                    {"hardware": "openvino", "priority": 3, "description": "Good performance with Intel hardware"},
                    {"hardware": "cpu", "priority": 4, "description": "Fallback option for all systems"}
                ]
            }
        
        register_tool("recommend_hardware", "Get hardware recommendations for a model", recommend_hardware, {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Name of the model"}
            },
            "required": ["model_name"]
        })
        logger.info("Successfully registered recommend_hardware tool")
    except Exception as e:
        logger.error(f"Error registering hardware tools: {str(e)}")
    
    # Register IPFS tools
    try:
        class MockIPFS:
            """Simple mock IPFS implementation for demonstration."""
            
            def __init__(self):
                self.files = {}
                self.cids = {}
            
            def add_file(self, path):
                """Add a file to IPFS."""
                import hashlib
                import random
                
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
                "path": {"type": "string", "description": "Path to the file to add"}
            },
            "required": ["path"]
        })
        logger.info("Successfully registered ipfs_add_file tool")
        
        # Register ipfs_get_file tool
        def ipfs_get_file(cid, output_path):
            """Get a file from IPFS."""
            try:
                return ipfs.get_file(cid, output_path)
            except Exception as e:
                logger.error(f"Error in ipfs_get_file: {e}")
                return {"error": str(e), "success": False}
        
        register_tool("ipfs_get_file", "Get a file from IPFS", ipfs_get_file, {
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "CID of the file to get"},
                "output_path": {"type": "string", "description": "Path to save the file to"}
            },
            "required": ["cid", "output_path"]
        })
        logger.info("Successfully registered ipfs_get_file tool")
        
        # Register ipfs_cat_file tool
        def ipfs_cat_file(cid):
            """Get the content of a file from IPFS."""
            try:
                return ipfs.cat_file(cid)
            except Exception as e:
                logger.error(f"Error in ipfs_cat_file: {e}")
                return {"error": str(e), "success": False}
        
        register_tool("ipfs_cat_file", "Get the content of a file from IPFS", ipfs_cat_file, {
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "CID of the file to get"}
            },
            "required": ["cid"]
        })
        logger.info("Successfully registered ipfs_cat_file tool")
        
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
        
        # Register ipfs_gateway_url tool
        def ipfs_gateway_url(cid, gateway="https://ipfs.io"):
            """Get the URL for a CID on the IPFS gateway."""
            # Remove trailing slash from gateway if present
            if gateway.endswith('/'):
                gateway = gateway[:-1]
            
            return {
                "url": f"{gateway}/ipfs/{cid}",
                "gateway": gateway,
                "cid": cid,
                "success": True
            }
        
        register_tool("ipfs_gateway_url", "Get the URL for a CID on the IPFS gateway", ipfs_gateway_url, {
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "CID of the file"},
                "gateway": {"type": "string", "description": "IPFS gateway URL (default: https://ipfs.io)"}
            },
            "required": ["cid"]
        })
        logger.info("Successfully registered ipfs_gateway_url tool")
    except Exception as e:
        logger.error(f"Error registering IPFS tools: {str(e)}")
    
    # Register model tools
    try:
        # Try to import ipfs_accelerate_py
        try:
            from ipfs_accelerate_py import ipfs_accelerate_py
            accel = ipfs_accelerate_py()
            
            # Register model_inference tool
            def model_inference(model_name, input_data, endpoint_type=None):
                """Run inference on a model."""
                try:
                    return accel.process(model_name, input_data, endpoint_type)
                except Exception as e:
                    logger.error(f"Error in model_inference: {e}")
                    return {"error": str(e)}
            
            register_tool("model_inference", "Run inference on a model", model_inference, {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model"},
                    "input_data": {"description": "Input data for inference"},
                    "endpoint_type": {"type": "string", "description": "Endpoint type (optional)"}
                },
                "required": ["model_name", "input_data"]
            })
            logger.info("Successfully registered model_inference tool")
            
            # Register list_models tool
            def list_models():
                """List available models."""
                try:
                    return {
                        "local_models": list(accel.endpoints.get("local_endpoints", {}).keys()),
                        "api_models": list(accel.endpoints.get("api_endpoints", {}).keys()),
                        "libp2p_models": list(accel.endpoints.get("libp2p_endpoints", {}).keys())
                    }
                except Exception as e:
                    logger.error(f"Error in list_models: {e}")
                    return {"error": str(e)}
            
            register_tool("list_models", "List available models", list_models, {})
            logger.info("Successfully registered list_models tool")
        except ImportError:
            logger.warning("ipfs_accelerate_py module not available, using mock implementations")
            
            # Register mock model_inference tool
            def mock_model_inference(model_name, input_data, endpoint_type=None):
                """Mock model inference."""
                return {
                    "model": model_name,
                    "input": str(input_data)[:100],
                    "endpoint_type": endpoint_type,
                    "output": "This is a mock response from the model.",
                    "is_mock": True
                }
            
            register_tool("model_inference", "Run inference on a model (mock)", mock_model_inference, {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the model"},
                    "input_data": {"description": "Input data for inference"},
                    "endpoint_type": {"type": "string", "description": "Endpoint type (optional)"}
                },
                "required": ["model_name", "input_data"]
            })
            logger.info("Successfully registered mock model_inference tool")
            
            # Register mock list_models tool
            def mock_list_models():
                """Mock list of available models."""
                return {
                    "local_models": ["bert-base-uncased", "gpt2", "t5-small"],
                    "api_models": ["gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"],
                    "libp2p_models": []
                }
            
            register_tool("list_models", "List available models (mock)", mock_list_models, {})
            logger.info("Successfully registered mock list_models tool")
    except Exception as e:
        logger.error(f"Error registering model tools: {str(e)}")
    
    logger.info("MCP tools registration complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
