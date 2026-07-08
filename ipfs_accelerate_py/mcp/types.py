"""
Type definitions for the IPFS Accelerate MCP server.

This module provides shared type definitions used across the MCP server components,
including context objects and tool-specific types.
"""

from typing import Any, Dict, List, Optional, Union, cast


class IPFSAccelerateContext:
    """Context object for IPFS Accelerate MCP.
    
    This class stores shared state and resources that are available to all tools
    throughout the lifespan of the server.
    """
    
    def __init__(self):
        """Initialize the IPFS Accelerate context."""
        self.ipfs_client = None
        self.hardware_info = None
        self.accelerated_models = {}
        self.model_metadata = {}
    
    def set_ipfs_client(self, client: Any) -> None:
        """Set the IPFS client.
        
        Args:
            client: The IPFS client instance
        """
        self.ipfs_client = client
    
    def set_hardware_info(self, hardware_info: Dict[str, Any]) -> None:
        """Set hardware information.
        
        Args:
            hardware_info: Hardware information dictionary
        """
        self.hardware_info = hardware_info
    
    def register_accelerated_model(self, original_cid: str, accelerated_cid: str, device: str, metadata: Dict[str, Any]) -> None:
        """Register an accelerated model.
        
        Args:
            original_cid: The original model CID
            accelerated_cid: The accelerated model CID
            device: The device used for acceleration
            metadata: Additional metadata about the model
        """
        if original_cid not in self.accelerated_models:
            self.accelerated_models[original_cid] = {}
        
        self.accelerated_models[original_cid][device] = {
            "accelerated_cid": accelerated_cid,
            "timestamp": metadata.get("timestamp", ""),
            "optimizations": metadata.get("optimizations", []),
            "performance_metrics": metadata.get("performance_metrics", {})
        }
        
        # Store model metadata
        self.model_metadata[original_cid] = metadata
    
    def get_accelerated_model(self, original_cid: str, device: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about an accelerated model.
        
        Args:
            original_cid: The original model CID
            device: The device to get the model for (None for all devices)
            
        Returns:
            Model information or None if not found
        """
        if original_cid not in self.accelerated_models:
            return None
        
        if device is None:
            return self.accelerated_models[original_cid]
        
        if device in self.accelerated_models[original_cid]:
            return self.accelerated_models[original_cid][device]
        
        return None
