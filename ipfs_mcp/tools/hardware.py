"""
Hardware Detection and Management Tools

This module provides MCP tools for detecting and managing available hardware.
"""
import logging
from typing import Dict, Any, List
from fastmcp import FastMCP

logger = logging.getLogger("ipfs_accelerate_mcp.tools.hardware")

def register_hardware_tools(mcp: FastMCP) -> None:
    """
    Register hardware-related tools with the MCP server.
    
    Args:
        mcp: The FastMCP server instance
    """
    # Access the ipfs_accelerate_py instance
    accelerate = mcp.state.accelerate

    @mcp.tool(name="detect_hardware")
    def detect_hardware() -> Dict[str, Any]:
        """
        Detect available hardware accelerators on the system.
        
        Returns a dictionary with information about available hardware accelerators.
        This includes CPU, CUDA (NVIDIA), ROCm (AMD), MPS (Apple), OpenVINO, 
        WebNN, WebGPU, and other supported accelerators.
        
        Returns:
            Dictionary with hardware detection results
        """
        logger.info("MCP tool called: detect_hardware")
        
        # Call the hardware detection function
        hardware_info = accelerate.hardware_detection.detect_all_hardware()
        
        return hardware_info
    
    @mcp.tool(name="get_optimal_hardware")
    def get_optimal_hardware(model_type: str = None) -> Dict[str, Any]:
        """
        Get the optimal hardware accelerator for a given model type.
        
        This function analyzes available hardware and recommends the best
        accelerator based on the model type and hardware capabilities.
        
        Args:
            model_type: Type of model (e.g., "text-generation", "image-classification")
                       If None, returns general hardware recommendation
        
        Returns:
            Dictionary with optimal hardware configuration
        """
        logger.info(f"MCP tool called: get_optimal_hardware with type={model_type}")
        
        # Get hardware detection info
        hardware_info = accelerate.hardware_detection.detect_all_hardware()
        
        # Let's pretend we have a method to determine optimal hardware
        if hasattr(accelerate, "get_optimal_hardware_for_model"):
            return accelerate.get_optimal_hardware_for_model(model_type, hardware_info)
        
        # Simple fallback logic
        optimal = {"device": "cpu", "reason": "Default fallback"}
        
        # Very basic prioritization logic
        if hardware_info.get("cuda", {}).get("available", False):
            optimal = {"device": "cuda", "reason": "CUDA support detected"}
        elif hardware_info.get("rocm", {}).get("available", False):
            optimal = {"device": "rocm", "reason": "ROCm support detected"}
        elif hardware_info.get("mps", {}).get("available", False):
            optimal = {"device": "mps", "reason": "Apple MPS support detected"}
        
        return optimal
