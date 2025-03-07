#!/usr/bin/env python3
"""
Unified WebNN and WebGPU Implementation

This module provides a unified interface to real WebNN and WebGPU implementations,
which now actually use browser-based execution through transformers.js.

Usage:
    # Import the module
    from unified_web_implementation import UnifiedWebImplementation
    
    # Create instance
    impl = UnifiedWebImplementation()
    
    # Get available platforms
    platforms = impl.get_available_platforms()
    
    # Initialize a model on WebGPU
    impl.init_model("bert-base-uncased", platform="webgpu")
    
    # Run inference
    result = impl.run_inference("bert-base-uncased", "This is a test input")
    
    # Clean up
    impl.shutdown()
"""

import os
import sys
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our real implementations
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from real_web_implementation import RealWebImplementation
except ImportError:
    logger.error("Failed to import RealWebImplementation. Make sure real_web_implementation.py exists in the test directory.")
    RealWebImplementation = None

class UnifiedWebImplementation:
    """Unified interface for WebNN and WebGPU implementations."""
    
    def __init__(self, allow_simulation=True):
        """Initialize unified implementation.
        
        Args:
            allow_simulation: If True, continue even if real hardware acceleration isn't available
                              and use simulation mode instead
        """
        self.allow_simulation = allow_simulation
        self.implementations = {}
        self.models = {}
        
        if RealWebImplementation is None:
            raise ImportError("RealWebImplementation is required")
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available platforms.
        
        Returns:
            List of platform names
        """
        # We always list both platforms, but they might be simulation only
        return ["webgpu", "webnn"]
    
    def is_hardware_available(self, platform: str) -> bool:
        """Check if hardware acceleration is available for platform.
        
        Args:
            platform: Platform to check
            
        Returns:
            True if hardware acceleration is available, False otherwise
        """
        if platform not in self.implementations:
            # Start implementation to check availability
            impl = RealWebImplementation(browser_name="chrome", headless=True)
            success = impl.start(platform=platform)
            
            if not success:
                return False
            
            # Check if using simulation
            is_simulation = impl.is_using_simulation()
            impl.stop()
            
            return not is_simulation
        else:
            # Check if existing implementation is using simulation
            return not self.implementations[platform].is_using_simulation()
    
    def init_model(self, model_name: str, model_type: str = "text", platform: str = "webgpu") -> Dict[str, Any]:
        """Initialize model.
        
        Args:
            model_name: Name of the model to initialize
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            platform: Platform to use ('webgpu' or 'webnn')
            
        Returns:
            Dict with initialization result or None if failed
        """
        # Validate platform
        if platform not in ["webgpu", "webnn"]:
            logger.error(f"Invalid platform: {platform}")
            return None
        
        # Create implementation if needed
        if platform not in self.implementations:
            logger.info(f"Creating new {platform} implementation")
            impl = RealWebImplementation(browser_name="chrome", headless=False)
            success = impl.start(platform=platform)
            
            if not success:
                logger.error(f"Failed to start {platform} implementation")
                return None
            
            self.implementations[platform] = impl
        
        # Initialize model
        logger.info(f"Initializing model {model_name} on {platform}")
        result = self.implementations[platform].initialize_model(model_name, model_type=model_type)
        
        if not result:
            logger.error(f"Failed to initialize model {model_name} on {platform}")
            return None
        
        # Store model info
        model_key = f"{platform}:{model_name}"
        self.models[model_key] = {
            "name": model_name,
            "type": model_type,
            "platform": platform,
            "initialized": True,
            "using_simulation": result.get("simulation", True)
        }
        
        return result
    
    def run_inference(self, model_name: str, input_data: Union[str, Dict[str, Any]], 
                     platform: str = None, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run inference with model.
        
        Args:
            model_name: Name of the model to use
            input_data: Input data for inference
            platform: Platform to use (if not specified, uses the platform the model was initialized on)
            options: Additional options for inference
            
        Returns:
            Dict with inference result or None if failed
        """
        # Determine platform
        if platform is None:
            # Check which platforms this model is initialized on
            for model_key, model_info in self.models.items():
                if model_info["name"] == model_name:
                    platform = model_info["platform"]
                    break
            
            if platform is None:
                logger.error(f"Model {model_name} not initialized on any platform")
                return None
        
        # Check if platform implementation exists
        if platform not in self.implementations:
            logger.error(f"Platform implementation not found: {platform}")
            return None
        
        # Run inference
        logger.info(f"Running inference with model {model_name} on {platform}")
        result = self.implementations[platform].run_inference(model_name, input_data, options=options)
        
        return result
    
    def shutdown(self, platform: str = None):
        """Shutdown implementation(s).
        
        Args:
            platform: Platform to shutdown (if None, shutdown all)
        """
        if platform is None:
            # Shutdown all implementations
            for platform_name, impl in self.implementations.items():
                logger.info(f"Shutting down {platform_name} implementation")
                impl.stop()
            
            self.implementations = {}
        elif platform in self.implementations:
            # Shutdown specific implementation
            logger.info(f"Shutting down {platform} implementation")
            self.implementations[platform].stop()
            del self.implementations[platform]
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get status of platforms.
        
        Returns:
            Dict with platform status information
        """
        status = {}
        
        for platform in ["webgpu", "webnn"]:
            if platform in self.implementations:
                # Get status from existing implementation
                impl = self.implementations[platform]
                features = impl.features if hasattr(impl, 'features') else {}
                
                status[platform] = {
                    "available": not impl.is_using_simulation(),
                    "features": features
                }
            else:
                # Check availability without creating a persistent implementation
                hardware_available = self.is_hardware_available(platform)
                
                status[platform] = {
                    "available": hardware_available,
                    "features": {}
                }
        
        return status

def main():
    """Run a simple test of the unified implementation."""
    unified_impl = UnifiedWebImplementation()
    
    try:
        print("\n===== Unified Web Implementation Test =====")
        
        # Get available platforms
        platforms = unified_impl.get_available_platforms()
        print(f"Available platforms: {platforms}")
        
        # Check hardware availability
        for platform in platforms:
            available = unified_impl.is_hardware_available(platform)
            print(f"{platform} hardware acceleration: {'Available' if available else 'Not available'}")
        
        # Initialize model on WebGPU
        print("\nInitializing BERT model on WebGPU...")
        result = unified_impl.init_model("bert-base-uncased", platform="webgpu")
        
        if not result:
            print("Failed to initialize BERT model on WebGPU")
            unified_impl.shutdown()
            return 1
        
        # Run inference
        print("Running inference with BERT model...")
        input_text = "This is a test input for unified web implementation."
        inference_result = unified_impl.run_inference("bert-base-uncased", input_text)
        
        if not inference_result:
            print("Failed to run inference")
            unified_impl.shutdown()
            return 1
        
        # Print result summary
        using_simulation = inference_result.get("is_simulation", True)
        implementation_type = inference_result.get("implementation_type", "UNKNOWN")
        performance = inference_result.get("performance_metrics", {})
        inference_time = performance.get("inference_time_ms", 0)
        
        print(f"\nInference completed using {implementation_type}")
        print(f"Using simulation: {using_simulation}")
        print(f"Inference time: {inference_time:.2f} ms")
        
        # Shutdown
        unified_impl.shutdown()
        print("\nTest completed successfully")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        unified_impl.shutdown()
        return 1

if __name__ == "__main__":
    sys.exit(main())