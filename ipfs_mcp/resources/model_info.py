"""
Model Information Resources

This module provides MCP resources for model information.
"""
import logging
from typing import Dict, Any, List
from fastmcp import FastMCP

logger = logging.getLogger("ipfs_accelerate_mcp.resources.model_info")

def register_model_resources(mcp: FastMCP) -> None:
    """
    Register model information resources with the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    # Access the ipfs_accelerate_py instance
    accelerate = mcp.state.accelerate
    
    @mcp.resource("models://available")
    def get_available_models() -> List[Dict[str, Any]]:
        """
        Get information about models available in the system.
        
        Returns a list of dictionaries with details about each available model,
        including model type, capabilities, and hardware requirements.
        
        Returns:
            List of dictionaries with model information
        """
        logger.info("MCP resource accessed: models://available")
        
        # In a real implementation, we would query the IPFS Accelerate instance
        # for available models and their capabilities
        
        # Mock data for demonstration
        models = [
            {
                "id": "text-generation-model",
                "name": "Text Generation Model",
                "type": "text-generation",
                "capabilities": ["text", "code", "chat"],
                "hardware_requirements": {
                    "minimum": "cpu",
                    "recommended": "cuda"
                }
            },
            {
                "id": "image-classification-model",
                "name": "Image Classification Model",
                "type": "image-classification",
                "capabilities": ["image"],
                "hardware_requirements": {
                    "minimum": "cpu",
                    "recommended": "cuda"
                }
            },
            {
                "id": "embedding-model",
                "name": "Embedding Model",
                "type": "embedding",
                "capabilities": ["text-embedding"],
                "hardware_requirements": {
                    "minimum": "cpu",
                    "recommended": "cpu"
                }
            }
        ]
        
        return models
    
    @mcp.resource("models://details/{model_id}")
    def get_model_details(model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: The ID of the model to get details for
        
        Returns:
            Dictionary with detailed model information
        """
        logger.info(f"MCP resource accessed: models://details/{model_id}")
        
        # In a real implementation, we would query the IPFS Accelerate instance
        # for detailed information about the specified model
        
        # Mock data based on the model ID
        if model_id == "text-generation-model":
            return {
                "id": model_id,
                "name": "Text Generation Model",
                "type": "text-generation",
                "architecture": "transformer",
                "parameters": "7B",
                "quantization": "int8",
                "capabilities": ["text", "code", "chat"],
                "hardware_requirements": {
                    "minimum": "cpu",
                    "recommended": "cuda",
                    "memory": "16GB"
                },
                "performance_metrics": {
                    "tokens_per_second_cpu": 5,
                    "tokens_per_second_cuda": 50
                }
            }
        elif model_id == "image-classification-model":
            return {
                "id": model_id,
                "name": "Image Classification Model",
                "type": "image-classification",
                "architecture": "vision-transformer",
                "parameters": "1B",
                "capabilities": ["image"],
                "classes": 1000,
                "hardware_requirements": {
                    "minimum": "cpu",
                    "recommended": "cuda",
                    "memory": "8GB"
                },
                "performance_metrics": {
                    "images_per_second_cpu": 1,
                    "images_per_second_cuda": 20
                }
            }
        else:
            # Generic fallback for unknown models
            return {
                "id": model_id,
                "name": f"Unknown Model: {model_id}",
                "type": "unknown",
                "message": "No detailed information available for this model."
            }
