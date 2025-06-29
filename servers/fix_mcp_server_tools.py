#!/usr/bin/env python3
"""
MCP Server Tools Fix Script

This script fixes the MCP server tools by registering all the functionality
from ipfs_accelerate_py properly, ensuring that all features are available
through the MCP server.
"""

import os
import sys
import json
import logging
import asyncio
import importlib
from typing import Dict, Any, List, Optional, Union
import inspect
import traceback
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_server_fix")

def import_ipfs_accelerate():
    """Import the IPFS Accelerate module."""
    try:
        from ipfs_accelerate_py import ipfs_accelerate_py
        logger.info("Successfully imported ipfs_accelerate_py")
        return ipfs_accelerate_py
    except ImportError:
        logger.warning("Could not import ipfs_accelerate_py directly, trying alternative paths")
        
        # Try different paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "ipfs_accelerate_py.py"),
            os.path.join(os.path.dirname(__file__), "ipfs_accelerate_py", "ipfs_accelerate_py.py")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found IPFS Accelerate at {path}")
                spec = importlib.util.spec_from_file_location("ipfs_accelerate_py", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.ipfs_accelerate_py
        
        logger.error("Could not find ipfs_accelerate_py module")
        return None

def create_tool_handler(accelerate_instance, method_name):
    """Create a handler function for an IPFS Accelerate method."""
    method = getattr(accelerate_instance, method_name, None)
    
    if method is None:
        logger.warning(f"Method {method_name} not found on accelerate instance")
        return None
    
    # Check if method is async
    is_async = asyncio.iscoroutinefunction(method)
    
    if is_async:
        def handler(*args, **kwargs):
            try:
                # Create event loop if not running
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run async method
                return loop.run_until_complete(method(*args, **kwargs))
            except Exception as e:
                logger.error(f"Error in async handler for {method_name}: {str(e)}")
                return {"error": str(e), "traceback": traceback.format_exc()}
    else:
        def handler(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in handler for {method_name}: {str(e)}")
                return {"error": str(e), "traceback": traceback.format_exc()}
    
    return handler

def generate_method_schema(method):
    """Generate JSON schema for a method based on its signature."""
    try:
        sig = inspect.signature(method)
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Process parameters
        for name, param in sig.parameters.items():
            # Skip self parameter for instance methods
            if name == "self":
                continue
                
            param_schema = {"type": "string"}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_schema["type"] = "string"
                elif param.annotation == int:
                    param_schema["type"] = "integer"
                elif param.annotation == float:
                    param_schema["type"] = "number"
                elif param.annotation == bool:
                    param_schema["type"] = "boolean"
                elif param.annotation == list or param.annotation == List:
                    param_schema["type"] = "array"
                elif param.annotation == dict or param.annotation == Dict:
                    param_schema["type"] = "object"
            
            # Add description based on parameter name
            param_schema["description"] = f"Parameter: {name}"
            
            # Add to schema
            schema["properties"][name] = param_schema
            
            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                schema["required"].append(name)
        
        return schema
    except Exception as e:
        logger.error(f"Error generating schema: {str(e)}")
        return {}

def get_method_description(method):
    """Get the description for a method from its docstring."""
    docstring = inspect.getdoc(method)
    if docstring:
        lines = docstring.split('\n')
        # Return first line of docstring
        return lines[0].strip()
    else:
        # Generate a description based on method name
        name = method.__name__
        words = name.split('_')
        return ' '.join(word.capitalize() for word in words)

def register_core_tools(server):
    """Register core IPFS Accelerate tools with the MCP server."""
    # Import IPFS Accelerate
    ipfs_accelerate_py = import_ipfs_accelerate()
    if ipfs_accelerate_py is None:
        logger.error("Failed to import ipfs_accelerate_py, cannot register core tools")
        return False
    
    # Create instance
    try:
        accelerate = ipfs_accelerate_py()
        logger.info("Created ipfs_accelerate_py instance")
    except Exception as e:
        logger.error(f"Error creating ipfs_accelerate_py instance: {str(e)}")
        return False
    
    # Core functionality tools
    core_methods = [
        "process",
        "process_async",
        "init_endpoints",
        "query_ipfs",
        "store_to_ipfs",
        "find_providers",
        "connect_to_provider",
        "accelerate_inference"
    ]
    
    # Register each core method
    for method_name in core_methods:
        if hasattr(accelerate, method_name):
            method = getattr(accelerate, method_name)
            handler = create_tool_handler(accelerate, method_name)
            description = get_method_description(method)
            schema = generate_method_schema(method)
            
            try:
                server.register_tool(
                    name=f"ipfs_{method_name}",
                    description=description,
                    handler=handler,
                    schema=schema
                )
                logger.info(f"Successfully registered tool: ipfs_{method_name}")
            except Exception as e:
                logger.error(f"Error registering tool ipfs_{method_name}: {str(e)}")
        else:
            logger.warning(f"Method {method_name} not found on accelerate instance")
    
    # Register additional IPFS specific tools
    try:
        # Add file to IPFS
        def ipfs_add_file(path):
            try:
                if not hasattr(accelerate, "add_file"):
                    return {"error": "add_file method not available", "success": False}
                    
                return accelerate.add_file(path)
            except Exception as e:
                logger.error(f"Error in ipfs_add_file: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="ipfs_add_file",
            description="Add a file to IPFS",
            handler=ipfs_add_file,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to add"}
                },
                "required": ["path"]
            }
        )
        logger.info("Successfully registered ipfs_add_file tool")
        
        # Get file from IPFS
        def ipfs_get_file(cid, output_path):
            try:
                if not hasattr(accelerate, "get_file"):
                    return {"error": "get_file method not available", "success": False}
                    
                return accelerate.get_file(cid, output_path)
            except Exception as e:
                logger.error(f"Error in ipfs_get_file: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="ipfs_get_file",
            description="Get a file from IPFS",
            handler=ipfs_get_file,
            schema={
                "type": "object",
                "properties": {
                    "cid": {"type": "string", "description": "CID of the file to get"},
                    "output_path": {"type": "string", "description": "Path to save the file to"}
                },
                "required": ["cid", "output_path"]
            }
        )
        logger.info("Successfully registered ipfs_get_file tool")
        
        # Cat file from IPFS
        def ipfs_cat(cid):
            try:
                if not hasattr(accelerate, "cat"):
                    return {"error": "cat method not available", "success": False}
                    
                return accelerate.cat(cid)
            except Exception as e:
                logger.error(f"Error in ipfs_cat: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="ipfs_cat",
            description="Get the content of a file from IPFS",
            handler=ipfs_cat,
            schema={
                "type": "object",
                "properties": {
                    "cid": {"type": "string", "description": "CID of the file to get"}
                },
                "required": ["cid"]
            }
        )
        logger.info("Successfully registered ipfs_cat tool")
    except Exception as e:
        logger.error(f"Error registering IPFS file tools: {str(e)}")
    
    return True

def register_endpoint_management_tools(server):
    """Register endpoint management tools with the MCP server."""
    # Import IPFS Accelerate
    ipfs_accelerate_py = import_ipfs_accelerate()
    if ipfs_accelerate_py is None:
        logger.error("Failed to import ipfs_accelerate_py, cannot register endpoint management tools")
        return False
    
    # Create instance
    try:
        accelerate = ipfs_accelerate_py()
        logger.info("Created ipfs_accelerate_py instance for endpoint management")
    except Exception as e:
        logger.error(f"Error creating ipfs_accelerate_py instance: {str(e)}")
        return False
    
    # Register endpoint status tool
    try:
        def endpoint_status():
            try:
                # Get status of all endpoints
                status = {
                    "local_endpoints": {},
                    "api_endpoints": {},
                    "libp2p_endpoints": {}
                }
                
                # Add local endpoints
                for model, endpoints in accelerate.endpoints.get("local_endpoints", {}).items():
                    status["local_endpoints"][model] = endpoints
                
                # Add API endpoints
                for model, endpoints in accelerate.endpoints.get("api_endpoints", {}).items():
                    status["api_endpoints"][model] = endpoints
                
                # Add libp2p endpoints
                for model, endpoints in accelerate.endpoints.get("libp2p_endpoints", {}).items():
                    status["libp2p_endpoints"][model] = endpoints
                
                return status
            except Exception as e:
                logger.error(f"Error in endpoint_status: {str(e)}")
                return {"error": str(e)}
        
        server.register_tool(
            name="endpoint_status",
            description="Get status of all endpoints",
            handler=endpoint_status,
            schema={}
        )
        logger.info("Successfully registered endpoint_status tool")
    except Exception as e:
        logger.error(f"Error registering endpoint_status tool: {str(e)}")
    
    # Register add_endpoint tool
    try:
        def add_endpoint(model, endpoint_type, endpoint_info):
            try:
                # Create event loop if not running
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Add endpoint
                result = loop.run_until_complete(accelerate.add_endpoint(model, endpoint_type, endpoint_info))
                return result
            except Exception as e:
                logger.error(f"Error in add_endpoint: {str(e)}")
                return {"error": str(e)}
        
        server.register_tool(
            name="add_endpoint",
            description="Add an endpoint for a model",
            handler=add_endpoint,
            schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Name of the model"},
                    "endpoint_type": {"type": "string", "description": "Type of endpoint"},
                    "endpoint_info": {"description": "Information about the endpoint"}
                },
                "required": ["model", "endpoint_type", "endpoint_info"]
            }
        )
        logger.info("Successfully registered add_endpoint tool")
    except Exception as e:
        logger.error(f"Error registering add_endpoint tool: {str(e)}")
    
    # Register rm_endpoint tool
    try:
        def rm_endpoint(model, endpoint_type):
            try:
                # Create event loop if not running
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Remove endpoint
                result = loop.run_until_complete(accelerate.rm_endpoint(model, endpoint_type))
                return result
            except Exception as e:
                logger.error(f"Error in rm_endpoint: {str(e)}")
                return {"error": str(e)}
        
        server.register_tool(
            name="rm_endpoint",
            description="Remove an endpoint for a model",
            handler=rm_endpoint,
            schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Name of the model"},
                    "endpoint_type": {"type": "string", "description": "Type of endpoint to remove"}
                },
                "required": ["model", "endpoint_type"]
            }
        )
        logger.info("Successfully registered rm_endpoint tool")
    except Exception as e:
        logger.error(f"Error registering rm_endpoint tool: {str(e)}")
    
    return True

def register_api_multiplexing_tools(server):
    """Register API multiplexing tools with the MCP server."""
    # Import IPFS Accelerate
    ipfs_accelerate_py = import_ipfs_accelerate()
    if ipfs_accelerate_py is None:
        logger.error("Failed to import ipfs_accelerate_py, cannot register API multiplexing tools")
        return False
    
    # Create instance
    try:
        accelerate = ipfs_accelerate_py()
        logger.info("Created ipfs_accelerate_py instance for API multiplexing")
    except Exception as e:
        logger.error(f"Error creating ipfs_accelerate_py instance: {str(e)}")
        return False
    
    # Register API key management
    try:
        def register_api_key(provider, api_key, priority=1):
            try:
                # Store API key in resources or use a specialized method if available
                if hasattr(accelerate, "register_api_key"):
                    return accelerate.register_api_key(provider, api_key, priority)
                
                if not hasattr(accelerate, "resources"):
                    accelerate.resources = {}
                    
                if "api_keys" not in accelerate.resources:
                    accelerate.resources["api_keys"] = {}
                    
                if provider not in accelerate.resources["api_keys"]:
                    accelerate.resources["api_keys"][provider] = []
                
                # Add the key with priority
                key_id = f"{provider}-{time.time()}"
                accelerate.resources["api_keys"][provider].append({
                    "key": api_key,
                    "priority": priority,
                    "key_id": key_id,
                    "registered_at": time.time()
                })
                
                return {
                    "provider": provider,
                    "key_id": key_id,
                    "priority": priority,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Error in register_api_key: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="register_api_key",
            description="Register an API key for a provider",
            handler=register_api_key,
            schema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "API provider name"},
                    "api_key": {"type": "string", "description": "API key"},
                    "priority": {"type": "integer", "description": "Priority level (lower is higher priority)"}
                },
                "required": ["provider", "api_key"]
            }
        )
        logger.info("Successfully registered register_api_key tool")
        
        # Get multiplexer status tool
        def get_multiplexer_status():
            try:
                if hasattr(accelerate, "get_multiplexer_status"):
                    return accelerate.get_multiplexer_status()
                
                # Provide a basic status if method not available
                providers = {}
                if hasattr(accelerate, "resources") and "api_keys" in accelerate.resources:
                    for provider, keys in accelerate.resources["api_keys"].items():
                        providers[provider] = {
                            "key_count": len(keys),
                            "active": True
                        }
                
                return {
                    "providers": providers,
                    "status": "active",
                    "requests_handled": 0,
                    "success_rate": 100
                }
            except Exception as e:
                logger.error(f"Error in get_multiplexer_status: {str(e)}")
                return {"error": str(e)}
        
        server.register_tool(
            name="get_multiplexer_status",
            description="Get status of the API multiplexer",
            handler=get_multiplexer_status,
            schema={}
        )
        logger.info("Successfully registered get_multiplexer_status tool")
        
        # Process API request tool
        def process_api_request(provider, model, prompt, options=None):
            try:
                if hasattr(accelerate, "process_api_request"):
                    # Create event loop if needed
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                    # Run the async method
                    return loop.run_until_complete(accelerate.process_api_request(
                        provider=provider,
                        model=model,
                        prompt=prompt,
                        options=options or {}
                    ))
                
                # Simple fallback if method not available
                return {
                    "provider": provider,
                    "model": model,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "response": f"Mock response for {provider}/{model}: This is a simulated response.",
                    "success": True,
                    "is_fallback": True
                }
            except Exception as e:
                logger.error(f"Error in process_api_request: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="process_api_request",
            description="Process a request through the API multiplexer",
            handler=process_api_request,
            schema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "API provider name"},
                    "model": {"type": "string", "description": "Model name"},
                    "prompt": {"type": "string", "description": "Input prompt/text"},
                    "options": {"type": "object", "description": "Additional options"}
                },
                "required": ["provider", "model", "prompt"]
            }
        )
        logger.info("Successfully registered process_api_request tool")
    except Exception as e:
        logger.error(f"Error registering API multiplexing tools: {str(e)}")
    
    return True

def register_model_server_tools(server):
    """Register model server tools with the MCP server."""
    # Import IPFS Accelerate
    ipfs_accelerate_py = import_ipfs_accelerate()
    if ipfs_accelerate_py is None:
        logger.error("Failed to import ipfs_accelerate_py, cannot register model server tools")
        return False
    
    # Create instance
    try:
        accelerate = ipfs_accelerate_py()
        logger.info("Created ipfs_accelerate_py instance for model serving")
    except Exception as e:
        logger.error(f"Error creating ipfs_accelerate_py instance: {str(e)}")
        return False
    
    # Register model inference tool
    try:
        def model_inference(model, input_data, endpoint_type=None):
            try:
                # Create event loop if needed for async methods
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Use process_async for better control
                if hasattr(accelerate, "process_async"):
                    return loop.run_until_complete(accelerate.process_async(
                        model=model,
                        input_data=input_data,
                        endpoint_type=endpoint_type
                    ))
                # Fallback to sync process method
                elif hasattr(accelerate, "process"):
                    return accelerate.process(
                        model=model,
                        input_data=input_data,
                        endpoint_type=endpoint_type
                    )
                else:
                    return {
                        "error": "No inference method available",
                        "model": model,
                        "success": False
                    }
            except Exception as e:
                logger.error(f"Error in model_inference: {str(e)}")
                return {"error": str(e), "success": False, "traceback": traceback.format_exc()}
        
        server.register_tool(
            name="model_inference",
            description="Run inference on a model",
            handler=model_inference,
            schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name"},
                    "input_data": {"description": "Input data for inference"},
                    "endpoint_type": {"type": "string", "description": "Specific endpoint type to use"}
                },
                "required": ["model", "input_data"]
            }
        )
        logger.info("Successfully registered model_inference tool")
        
        # Register list models tool
        def list_models():
            try:
                # Get models from endpoints
                models = {
                    "local_models": list(accelerate.endpoints.get("local_endpoints", {}).keys()),
                    "api_models": list(accelerate.endpoints.get("api_endpoints", {}).keys()),
                    "libp2p_models": list(accelerate.endpoints.get("libp2p_endpoints", {}).keys())
                }
                
                # Add details about each model if available
                model_details = {}
                
                # Add details from metadata if available
                if hasattr(accelerate, "metadata") and "models" in accelerate.metadata:
                    for model in accelerate.metadata["models"]:
                        if isinstance(model, str):
                            model_details[model] = {"name": model}
                        elif isinstance(model, dict) and "name" in model:
                            model_details[model["name"]] = model
                
                # Return both simple list and detailed info
                return {
                    "models": models,
                    "model_details": model_details,
                    "total_count": len(set().union(*models.values()))
                }
            except Exception as e:
                logger.error(f"Error in list_models: {str(e)}")
                return {"error": str(e)}
        
        server.register_tool(
            name="list_models",
            description="List available models",
            handler=list_models,
            schema={}
        )
        logger.info("Successfully registered list_models tool")
        
        # Register batch processing tool
        def batch_process(model, inputs, endpoint_type=None, batch_size=10):
            try:
                results = []
                
                # Process in batches
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i + batch_size]
                    
                    # Create event loop if needed
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Process each item in the batch
                    batch_results = []
                    for item in batch:
                        try:
                            if hasattr(accelerate, "process_async"):
                                result = loop.run_until_complete(accelerate.process_async(
                                    model=model,
                                    input_data=item,
                                    endpoint_type=endpoint_type
                                ))
                            elif hasattr(accelerate, "process"):
                                result = accelerate.process(model, item, endpoint_type)
                            else:
                                result = {"error": "No processing method available", "success": False}
                            
                            batch_results.append(result)
                        except Exception as e:
                            batch_results.append({"error": str(e), "success": False})
                    
                    # Add batch results
                    results.extend(batch_results)
                
                return {
                    "model": model,
                    "batch_count": len(results),
                    "results": results,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Error in batch_process: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="batch_process",
            description="Process a batch of inputs through a model",
            handler=batch_process,
            schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name"},
                    "inputs": {"type": "array", "description": "List of inputs to process"},
                    "endpoint_type": {"type": "string", "description": "Specific endpoint type to use"},
                    "batch_size": {"type": "integer", "description": "Size of each processing batch"}
                },
                "required": ["model", "inputs"]
            }
        )
        logger.info("Successfully registered batch_process tool")
    except Exception as e:
        logger.error(f"Error registering model server tools: {str(e)}")
    
    return True

def register_distributed_computation_tools(server):
    """Register distributed computation tools with the MCP server."""
    # Import IPFS Accelerate
    ipfs_accelerate_py = import_ipfs_accelerate()
    if ipfs_accelerate_py is None:
        logger.error("Failed to import ipfs_accelerate_py, cannot register distributed computation tools")
        return False
    
    # Create instance
    try:
        accelerate = ipfs_accelerate_py()
        logger.info("Created ipfs_accelerate_py instance for distributed computation")
    except Exception as e:
        logger.error(f"Error creating ipfs_accelerate_py instance: {str(e)}")
        return False
    
    # Register find_providers tool
    try:
        def find_providers(model):
            try:
                # Create event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Find providers for the model
                if hasattr(accelerate, "find_providers"):
                    providers = loop.run_until_complete(accelerate.find_providers(model))
                    return {
                        "model": model,
                        "providers": providers,
                        "count": len(providers),
                        "success": True
                    }
                else:
                    return {
                        "model": model,
                        "providers": [],
                        "count": 0,
                        "success": False,
                        "error": "find_providers method not available"
                    }
            except Exception as e:
                logger.error(f"Error in find_providers: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="find_providers",
            description="Find providers for a model in the IPFS network",
            handler=find_providers,
            schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name to find providers for"}
                },
                "required": ["model"]
            }
        )
        logger.info("Successfully registered find_providers tool")
        
        # Register connect_to_provider tool
        def connect_to_provider(provider_id):
            try:
                # Create event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Connect to the provider
                if hasattr(accelerate, "connect_to_provider"):
                    connected = loop.run_until_complete(accelerate.connect_to_provider(provider_id))
                    return {
                        "provider_id": provider_id,
                        "connected": connected,
                        "success": connected
                    }
                else:
                    return {
                        "provider_id": provider_id,
                        "connected": False,
                        "success": False,
                        "error": "connect_to_provider method not available"
                    }
            except Exception as e:
                logger.error(f"Error in connect_to_provider: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="connect_to_provider",
            description="Connect to a provider in the IPFS network",
            handler=connect_to_provider,
            schema={
                "type": "object",
                "properties": {
                    "provider_id": {"type": "string", "description": "Provider ID to connect to"}
                },
                "required": ["provider_id"]
            }
        )
        logger.info("Successfully registered connect_to_provider tool")
        
        # Register accelerated_inference tool
        def accelerated_inference(model, input_data, use_ipfs=True):
            try:
                # Create event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Use accelerate_inference method if available
                if hasattr(accelerate, "accelerate_inference"):
                    result = loop.run_until_complete(accelerate.accelerate_inference(
                        model=model,
                        input_data=input_data,
                        use_ipfs=use_ipfs
                    ))
                    return result
                # Fallback to process_async
                elif hasattr(accelerate, "process_async"):
                    result = loop.run_until_complete(accelerate.process_async(
                        model=model,
                        input_data=input_data
                    ))
                    return result
                # Final fallback to process
                elif hasattr(accelerate, "process"):
                    return accelerate.process(model, input_data)
                else:
                    return {
                        "error": "No inference method available",
                        "success": False
                    }
            except Exception as e:
                logger.error(f"Error in accelerated_inference: {str(e)}")
                return {"error": str(e), "success": False}
        
        server.register_tool(
            name="accelerated_inference",
            description="Run inference with both local and IPFS network acceleration",
            handler=accelerated_inference,
            schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name"},
                    "input_data": {"description": "Input data for inference"},
                    "use_ipfs": {"type": "boolean", "description": "Whether to use IPFS for acceleration"}
                },
                "required": ["model", "input_data"]
            }
        )
        logger.info("Successfully registered accelerated_inference tool")
    except Exception as e:
        logger.error(f"Error registering distributed computation tools: {str(e)}")
    
    return True

def fix_direct_mcp_server():
    """Fix the Direct MCP Server implementation."""
    # Try to import needed modules
    try:
        import importlib.util
        from flask import Flask
        
        # Define paths
        direct_mcp_server_path = os.path.join(os.path.dirname(__file__), "direct_mcp_server.py")
        
        if not os.path.exists(direct_mcp_server_path):
            logger.error(f"Direct MCP Server file not found at {direct_mcp_server_path}")
            return False
        
        # Load the module
        spec = importlib.util.spec_from_file_location("direct_mcp_server", direct_mcp_server_path)
        direct_mcp_server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(direct_mcp_server)
        
        # Check if tools registry exists
        if not hasattr(direct_mcp_server, "tools"):
            logger.error("tools dictionary not found in direct_mcp_server module")
            return False
        
        # Register all tools with the server
        class ServerProxy:
            @staticmethod
            def register_tool(name, description, handler, schema):
                direct_mcp_server.tools[name] = handler
                logger.info(f"Registered tool {name} with direct_mcp_server")
        
        # Create a proxy server object
        server_proxy = ServerProxy()
        
        # Register tools
        register_core_tools(server_proxy)
        register_endpoint_management_tools(server_proxy)
        register_api_multiplexing_tools(server_proxy)
        register_model_server_tools(server_proxy)
        register_distributed_computation_tools(server_proxy)
        
        logger.info(f"Fixed Direct MCP Server with {len(direct_mcp_server.tools)} tools")
        return True
    except Exception as e:
        logger.error(f"Error fixing Direct MCP Server: {str(e)}")
        return False

def fix_mcp_server():
    """Fix the MCP Server implementation."""
    try:
        # Try to import MCP module
        try:
            sys.path.append(os.path.dirname(__file__))
            from mcp.server import register_tool, get_mcp_server_instance
            logger.info("Successfully imported MCP server")
            
            # Check if we can get server instance
            mcp_server = get_mcp_server_instance()
            if mcp_server is None:
                logger.warning("MCP server instance not available, tools will be registered but not applied to running server")
                
            # Create a proxy server object
            class ServerProxy:
                @staticmethod
                def register_tool(name, description, handler, schema):
                    register_tool(name, description, handler, schema)
                    logger.info(f"Registered tool {name} with MCP server")
            
            # Create a proxy server object
            server_proxy = ServerProxy()
        except ImportError:
            logger.error("Could not import MCP server, creating incomplete fix")
            return False
        
        # Register tools
        register_core_tools(server_proxy)
        register_endpoint_management_tools(server_proxy)
        register_api_multiplexing_tools(server_proxy)
        register_model_server_tools(server_proxy)
        register_distributed_computation_tools(server_proxy)
        
        logger.info("Fixed MCP Server")
        return True
    except Exception as e:
        logger.error(f"Error fixing MCP Server: {str(e)}")
        return False

def main():
    """Main function."""
    logger.info("Starting MCP server fix")
    
    # Fix Direct MCP Server
    direct_success = fix_direct_mcp_server()
    if direct_success:
        logger.info("Successfully fixed direct_mcp_server.py")
    else:
        logger.warning("Failed to fix direct_mcp_server.py")
    
    # Fix MCP Server
    mcp_success = fix_mcp_server()
    if mcp_success:
        logger.info("Successfully fixed MCP server")
    else:
        logger.warning("Failed to fix MCP server")
    
    # Overall result
    if direct_success and mcp_success:
        logger.info("Successfully fixed all MCP servers")
        return 0
    elif direct_success or mcp_success:
        logger.warning("Fixed some MCP servers, but not all")
        return 1
    else:
        logger.error("Failed to fix any MCP servers")
        return 2

if __name__ == "__main__":
    sys.exit(main())
