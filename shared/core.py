"""
Shared core functionality for IPFS Accelerate CLI and MCP server.

This module provides the main shared core that both CLI and MCP server
can use to ensure consistent behavior and code reuse.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class SharedCore:
    """
    Shared core functionality used by both CLI and MCP server
    
    This class provides a unified interface to IPFS Accelerate functionality
    that can be used consistently across different interfaces.
    """
    
    def __init__(self):
        self.ipfs_instance = None
        self._init_time = time.time()
        logger.info("SharedCore initialized")
        
    def get_ipfs_instance(self):
        """Get or create IPFS Accelerate instance"""
        if self.ipfs_instance is None:
            try:
                from ipfs_accelerate_py import ipfs_accelerate_py
                self.ipfs_instance = ipfs_accelerate_py()
                logger.info("IPFS Accelerate instance initialized")
            except ImportError as e:
                logger.warning(f"IPFS Accelerate core not available: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize IPFS Accelerate: {e}")
        return self.ipfs_instance
    
    def get_status(self) -> Dict[str, Any]:
        """Get general status information"""
        instance = self.get_ipfs_instance()
        
        status = {
            "core_available": instance is not None,
            "uptime": time.time() - self._init_time,
            "timestamp": time.time()
        }
        
        if instance:
            try:
                # Try to get more detailed status from the instance
                if hasattr(instance, 'get_status'):
                    instance_status = instance.get_status()
                    status.update(instance_status)
                else:
                    status["instance_methods"] = [method for method in dir(instance) if not method.startswith('_')]
            except Exception as e:
                logger.warning(f"Error getting instance status: {e}")
                status["instance_error"] = str(e)
        
        return status
    
    def validate_model_id(self, model_id: str) -> bool:
        """Validate model identifier"""
        if not model_id or not isinstance(model_id, str):
            return False
        
        # Basic validation - model ID should not be empty or contain invalid chars
        invalid_chars = ['<', '>', '|', ':', '"', '*', '?', '\n', '\r']
        if any(char in model_id for char in invalid_chars):
            return False
        
        return True
    
    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path"""
        if not file_path or not isinstance(file_path, str):
            return False
        
        import os
        return os.path.exists(file_path)
    
    def safe_call(self, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Safely call a method on the IPFS instance with error handling
        
        Args:
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dictionary with result or error information
        """
        instance = self.get_ipfs_instance()
        
        if not instance:
            return {
                "error": "IPFS Accelerate core not available",
                "method": method_name,
                "fallback": True
            }
        
        if not hasattr(instance, method_name):
            return {
                "error": f"Method '{method_name}' not available",
                "method": method_name,
                "available_methods": [method for method in dir(instance) if not method.startswith('_')]
            }
        
        try:
            method = getattr(instance, method_name)
            result = method(*args, **kwargs)
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"result": result}
            
            result["method"] = method_name
            result["success"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling {method_name}: {e}")
            return {
                "error": str(e),
                "method": method_name,
                "success": False
            }