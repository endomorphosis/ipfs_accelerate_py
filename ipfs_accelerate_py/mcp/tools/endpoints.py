"""
Endpoint Management Tools for IPFS Accelerate MCP Server

This module provides MCP tools for managing model inference endpoints.
"""

import os
import time
import uuid
import logging
import traceback
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("ipfs_accelerate_mcp.tools.endpoints")

# Store the IPFS Accelerate instance for use by the tools
_ipfs_instance = None

# In-memory storage for endpoints (fallback when IPFS Accelerate instance isn't available)
ENDPOINTS = {}

def set_ipfs_instance(ipfs_instance) -> None:
    """
    Set the IPFS Accelerate instance
    
    Args:
        ipfs_instance: IPFS Accelerate instance
    """
    global _ipfs_instance
    _ipfs_instance = ipfs_instance
    logger.info(f"IPFS Accelerate instance set: {ipfs_instance}")

def register_tools(mcp):
    """Register endpoint-related tools with the MCP server"""
    
    @mcp.tool()
    def get_endpoints() -> Dict[str, Any]:
        """
        Get all endpoints
        
        This tool returns all registered endpoints.
        
        Returns:
            Dictionary with all endpoints
        """
        global _ipfs_instance
        
        try:
            # First try to use IPFS Accelerate instance if available
            if _ipfs_instance is not None:
                try:
                    if hasattr(_ipfs_instance, "list_endpoints"):
                        endpoints = _ipfs_instance.list_endpoints()
                        return {
                            "endpoints": endpoints,
                            "count": len(endpoints),
                            "source": "ipfs_accelerate"
                        }
                except Exception as e:
                    logger.warning(f"Failed to get endpoints from IPFS Accelerate: {e}")
            
            # Fallback to in-memory storage
            return {
                "endpoints": list(ENDPOINTS.values()),
                "count": len(ENDPOINTS),
                "source": "in_memory"
            }
        except Exception as e:
            logger.error(f"Error getting endpoints: {e}")
            return {
                "error": f"Error getting endpoints: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    @mcp.tool()
    def add_endpoint(model: str, 
                    device: str = "cpu", 
                    max_batch_size: int = 16,
                    description: str = "") -> Dict[str, Any]:
        """
        Add a new endpoint
        
        This tool adds a new inference endpoint.
        
        Args:
            model: Name or path of the model to use
            device: Device to run inference on (e.g., "cpu", "cuda:0")
            max_batch_size: Maximum batch size for this endpoint
            description: Description of the endpoint
            
        Returns:
            Dictionary with the new endpoint
        """
        try:
            # Get the endpoints config to check limits
            endpoints_config = mcp.access_resource("endpoints_config")
            max_endpoints = endpoints_config.get("max_endpoints", 10)
            
            # Check if we've reached the max endpoints
            if len(ENDPOINTS) >= max_endpoints:
                return {
                    "error": f"Maximum number of endpoints ({max_endpoints}) reached. Remove unused endpoints first."
                }
            
            # Get model info from resources
            model_info = mcp.access_resource("get_model_info", model_name=model)
            if not model_info:
                return {
                    "error": f"Model '{model}' not found. Please check the available models."
                }
            
            # Generate a unique ID for this endpoint
            endpoint_id = str(uuid.uuid4())
            
            # Create the endpoint
            endpoint = {
                "id": endpoint_id,
                "model": model,
                "model_type": model_info.get("type", "unknown"),
                "device": device,
                "max_batch_size": max_batch_size,
                "description": description,
                "created_at": time.time(),
                "status": "active",
                "requests_processed": 0,
                "total_processing_time": 0.0
            }
            
            # Store the endpoint
            ENDPOINTS[endpoint_id] = endpoint
            
            logger.info(f"Added endpoint {endpoint_id} for model {model} on device {device}")
            
            return endpoint
        except Exception as e:
            return {
                "error": f"Error adding endpoint: {str(e)}"
            }
    
    @mcp.tool()
    def remove_endpoint(endpoint_id: str) -> Dict[str, Any]:
        """
        Remove an endpoint
        
        This tool removes an inference endpoint.
        
        Args:
            endpoint_id: ID of the endpoint to remove
            
        Returns:
            Dictionary with the removal status
        """
        try:
            # Check if the endpoint exists
            if endpoint_id not in ENDPOINTS:
                return {
                    "error": f"Endpoint '{endpoint_id}' not found."
                }
            
            # Get the endpoint for return information
            endpoint = ENDPOINTS[endpoint_id]
            
            # Remove the endpoint
            del ENDPOINTS[endpoint_id]
            
            logger.info(f"Removed endpoint {endpoint_id}")
            
            return {
                "status": "success",
                "message": f"Endpoint {endpoint_id} removed successfully.",
                "endpoint": endpoint
            }
        except Exception as e:
            return {
                "error": f"Error removing endpoint: {str(e)}"
            }
    
    @mcp.tool()
    def update_endpoint(endpoint_id: str, 
                       max_batch_size: Optional[int] = None,
                       description: Optional[str] = None,
                       status: Optional[str] = None) -> Dict[str, Any]:
        """
        Update an endpoint
        
        This tool updates an inference endpoint.
        
        Args:
            endpoint_id: ID of the endpoint to update
            max_batch_size: New maximum batch size
            description: New description
            status: New status ("active" or "inactive")
            
        Returns:
            Dictionary with the updated endpoint
        """
        try:
            # Check if the endpoint exists
            if endpoint_id not in ENDPOINTS:
                return {
                    "error": f"Endpoint '{endpoint_id}' not found."
                }
            
            # Get the endpoint
            endpoint = ENDPOINTS[endpoint_id]
            
            # Update the endpoint
            if max_batch_size is not None:
                endpoint["max_batch_size"] = max_batch_size
            
            if description is not None:
                endpoint["description"] = description
            
            if status is not None:
                if status not in ["active", "inactive"]:
                    return {
                        "error": f"Invalid status: {status}. Must be 'active' or 'inactive'."
                    }
                endpoint["status"] = status
            
            # Update last modified time
            endpoint["updated_at"] = time.time()
            
            logger.info(f"Updated endpoint {endpoint_id}")
            
            return endpoint
        except Exception as e:
            return {
                "error": f"Error updating endpoint: {str(e)}"
            }
    
    @mcp.tool()
    def get_endpoint(endpoint_id: str) -> Dict[str, Any]:
        """
        Get an endpoint
        
        This tool gets information about an inference endpoint.
        
        Args:
            endpoint_id: ID of the endpoint to get
            
        Returns:
            Dictionary with the endpoint
        """
        try:
            # Check if the endpoint exists
            if endpoint_id not in ENDPOINTS:
                return {
                    "error": f"Endpoint '{endpoint_id}' not found."
                }
            
            # Return the endpoint
            return ENDPOINTS[endpoint_id]
        except Exception as e:
            return {
                "error": f"Error getting endpoint: {str(e)}"
            }
    
    @mcp.tool()
    def log_request(endpoint_id: str, 
                   model: str,
                   device: str,
                   request_type: str,
                   inputs_processed: int,
                   processing_time: float) -> Dict[str, Any]:
        """
        Log a request to an endpoint
        
        This tool logs a request to an endpoint for monitoring.
        
        Args:
            endpoint_id: ID of the endpoint
            model: Name or path of the model used
            device: Device used for inference
            request_type: Type of request (e.g., "embedding", "generation")
            inputs_processed: Number of inputs processed
            processing_time: Time taken to process the request
            
        Returns:
            Dictionary with the log status
        """
        try:
            # Check if the endpoint exists
            if endpoint_id not in ENDPOINTS:
                return {
                    "error": f"Endpoint '{endpoint_id}' not found."
                }
            
            # Get the endpoint
            endpoint = ENDPOINTS[endpoint_id]
            
            # Update request statistics
            endpoint["requests_processed"] += inputs_processed
            endpoint["total_processing_time"] += processing_time
            
            # Calculate average processing time
            avg_processing_time = endpoint["total_processing_time"] / endpoint["requests_processed"]
            
            logger.info(f"Logged request to endpoint {endpoint_id}: {inputs_processed} inputs processed in {processing_time:.3f}s")
            
            return {
                "status": "success",
                "endpoint_id": endpoint_id,
                "model": model,
                "device": device,
                "request_type": request_type,
                "inputs_processed": inputs_processed,
                "processing_time": processing_time,
                "endpoint_stats": {
                    "requests_processed": endpoint["requests_processed"],
                    "total_processing_time": endpoint["total_processing_time"],
                    "avg_processing_time": avg_processing_time
                }
            }
        except Exception as e:
            return {
                "error": f"Error logging request: {str(e)}"
            }

    # Ensure endpoint control-plane tools run in the server process when using
    # StandaloneMCP (dict-based registry). Do this post-registration to avoid
    # relying on decorator keyword support in other MCP implementations.
    tools_dict = getattr(mcp, "tools", None)
    if isinstance(tools_dict, dict):
        for tool_name in [
            "get_endpoints",
            "add_endpoint",
            "remove_endpoint",
            "update_endpoint",
            "get_endpoint",
            "log_request",
        ]:
            entry = tools_dict.get(tool_name)
            if isinstance(entry, dict):
                entry["execution_context"] = "server"
