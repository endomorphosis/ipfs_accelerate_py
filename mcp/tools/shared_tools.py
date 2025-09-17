"""
Shared Tools for IPFS Accelerate MCP Server

This module provides MCP tools that use the shared operations for consistency with CLI.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("ipfs_accelerate_mcp.tools.shared")

# Try imports with fallbacks
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        # Fall back to mock implementation
        from mcp.mock_mcp import FastMCP

# Import shared operations
try:
    from ...shared import SharedCore, InferenceOperations, FileOperations, ModelOperations, NetworkOperations
    shared_core = SharedCore()
    inference_ops = InferenceOperations(shared_core)
    file_ops = FileOperations(shared_core) 
    model_ops = ModelOperations(shared_core)
    network_ops = NetworkOperations(shared_core)
    HAVE_SHARED = True
except ImportError as e:
    logger.warning(f"Shared operations not available: {e}")
    HAVE_SHARED = False
    shared_core = None
    inference_ops = None
    file_ops = None
    model_ops = None
    network_ops = None

def register_shared_tools(mcp: FastMCP) -> None:
    """Register tools that use shared operations with the MCP server."""
    logger.info("Registering shared operation tools")
    
    if not HAVE_SHARED:
        logger.warning("Shared operations not available, skipping registration")
        return
    
    # Inference tools using shared operations
    @mcp.tool()
    def generate_text(
        prompt: str,
        model: str = "gpt2",
        max_length: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text using shared inference operations
        
        Args:
            prompt: Input text prompt
            model: Model to use for generation
            max_length: Maximum length of generated text
            temperature: Temperature for generation
            
        Returns:
            Generated text result
        """
        try:
            result = inference_ops.run_text_generation(
                model=model,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature
            )
            result["tool"] = "generate_text"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in generate_text: {e}")
            return {
                "error": str(e),
                "prompt": prompt,
                "model": model,
                "tool": "generate_text",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def classify_text(
        text: str,
        model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> Dict[str, Any]:
        """
        Classify text using shared inference operations
        
        Args:
            text: Text to classify
            model: Classification model to use
            
        Returns:
            Classification result
        """
        try:
            result = inference_ops.run_text_classification(
                model=model,
                text=text
            )
            result["tool"] = "classify_text"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in classify_text: {e}")
            return {
                "error": str(e),
                "text": text,
                "model": model,
                "tool": "classify_text",
                "timestamp": time.time()
            }
    
    # File tools using shared operations
    @mcp.tool()
    def add_file_to_ipfs(file_path: str) -> Dict[str, Any]:
        """
        Add file to IPFS using shared operations
        
        Args:
            file_path: Path to file to add
            
        Returns:
            File addition result with CID
        """
        try:
            result = file_ops.add_file(file_path)
            result["tool"] = "add_file_to_ipfs"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in add_file_to_ipfs: {e}")
            return {
                "error": str(e),
                "file_path": file_path,
                "tool": "add_file_to_ipfs",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def get_file_from_ipfs(cid: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get file from IPFS using shared operations
        
        Args:
            cid: Content identifier of the file
            output_path: Optional path to save the file
            
        Returns:
            File retrieval result
        """
        try:
            result = file_ops.get_file(cid, output_path)
            result["tool"] = "get_file_from_ipfs"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in get_file_from_ipfs: {e}")
            return {
                "error": str(e),
                "cid": cid,
                "output_path": output_path,
                "tool": "get_file_from_ipfs",
                "timestamp": time.time()
            }
    
    # Model tools using shared operations
    @mcp.tool()
    def list_available_models() -> Dict[str, Any]:
        """
        List available models using shared operations
        
        Returns:
            List of available models
        """
        try:
            result = model_ops.list_models()
            result["tool"] = "list_available_models"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in list_available_models: {e}")
            return {
                "error": str(e),
                "tool": "list_available_models",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def get_model_information(model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_id: ID of the model to get information about
            
        Returns:
            Model information
        """
        try:
            result = model_ops.get_model_info(model_id)
            result["tool"] = "get_model_information"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in get_model_information: {e}")
            return {
                "error": str(e),
                "model_id": model_id,
                "tool": "get_model_information",
                "timestamp": time.time()
            }
    
    # Network tools using shared operations
    @mcp.tool()
    def check_network_status() -> Dict[str, Any]:
        """
        Check IPFS network status using shared operations
        
        Returns:
            Network status information
        """
        try:
            result = network_ops.get_network_status()
            result["tool"] = "check_network_status"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in check_network_status: {e}")
            return {
                "error": str(e),
                "tool": "check_network_status",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def get_connected_peers() -> Dict[str, Any]:
        """
        Get list of connected IPFS peers using shared operations
        
        Returns:
            List of connected peers
        """
        try:
            result = network_ops.get_peers()
            result["tool"] = "get_connected_peers"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in get_connected_peers: {e}")
            return {
                "error": str(e),
                "tool": "get_connected_peers",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def get_system_status() -> Dict[str, Any]:
        """
        Get overall system status using shared operations
        
        Returns:
            System status information
        """
        try:
            result = shared_core.get_status()
            result["tool"] = "get_system_status"
            result["timestamp"] = time.time()
            return result
        except Exception as e:
            logger.error(f"Error in get_system_status: {e}")
            return {
                "error": str(e),
                "tool": "get_system_status",
                "timestamp": time.time()
            }
    
    logger.info("Shared operation tools registered successfully")