#!/usr/bin/env python3
"""
Base class for Mojo/MAX support in IPFS Accelerate model generators.
This module provides common functionality for integrating Mojo/MAX targets
into all model generators.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class MojoMaxTargetMixin:
    """
    Mixin class that provides Mojo/MAX target support for model generators.
    This can be inherited by any model skill to add Mojo/MAX capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the mixin."""
        super().__init__(*args, **kwargs)
        self._mojo_max_session = None
        self._mojo_max_graph = None
    
    def get_default_device_with_mojo_max(self):
        """Get the best available device including Mojo/MAX support."""
        # Check for Mojo/MAX target environment variable
        if os.environ.get("USE_MOJO_MAX_TARGET", "").lower() in ("1", "true", "yes"):
            return "mojo_max"
        
        # Try to detect MAX availability
        if self._is_max_available():
            return "max"
        
        # Try to detect Mojo availability
        if self._is_mojo_available():
            return "mojo"
        
        # Fall back to standard device detection
        return self._get_standard_device()
    
    def _is_max_available(self) -> bool:
        """Check if MAX is available."""
        try:
            import importlib.util
            if importlib.util.find_spec("max") is not None:
                import max
                # Try to import core MAX modules
                from max.graph import Graph
                from max.engine import InferenceSession
                return True
        except ImportError:
            logger.debug("MAX not available via Python package")
        
        # Check for MAX executable
        import shutil
        if shutil.which("max"):
            return True
        
        # Check for MAX_HOME
        if "MAX_HOME" in os.environ:
            from pathlib import Path
            max_path = Path(os.environ["MAX_HOME"]) / "bin" / "max"
            return max_path.exists()
        
        return False
    
    def _is_mojo_available(self) -> bool:
        """Check if Mojo is available."""
        try:
            import importlib.util
            if importlib.util.find_spec("mojo") is not None:
                import mojo
                return True
        except ImportError:
            logger.debug("Mojo not available via Python package")
        
        # Check for Mojo executable
        import shutil
        if shutil.which("mojo"):
            return True
        
        # Check for MOJO_HOME
        if "MOJO_HOME" in os.environ:
            from pathlib import Path
            mojo_path = Path(os.environ["MOJO_HOME"]) / "bin" / "mojo"
            return mojo_path.exists()
        
        return False
    
    def _get_standard_device(self):
        """Get standard PyTorch device (fallback method to be overridden)."""
        try:
            import torch
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
                if torch.mps.is_available():
                    return "mps"
        except ImportError:
            logger.debug("PyTorch not available")
        
        # Default to CPU
        return "cpu"
    
    def process_with_mojo_max(self, inputs: Any, model_name: str) -> Dict[str, Any]:
        """
        Process inputs using Mojo/MAX backend.
        
        Args:
            inputs: Input data to process
            model_name: Name of the model for graph creation
            
        Returns:
            Dictionary with processing results
        """
        device = getattr(self, 'device', 'cpu')
        
        if device == "mojo_max" or device == "max":
            return self._process_with_max(inputs, model_name)
        elif device == "mojo":
            return self._process_with_mojo(inputs, model_name)
        else:
            raise ValueError(f"Device {device} is not a Mojo/MAX target")
    
    def _process_with_max(self, inputs: Any, model_name: str) -> Dict[str, Any]:
        """Process inputs using MAX backend."""
        try:
            # Simulate MAX processing since MAX may not be available
            logger.info(f"Simulating MAX processing for model: {model_name}")
            
            return {
                "backend": "MAX (simulated)",
                "device": "max",
                "outputs": {"simulated": "max_output"},
                "model": model_name,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"MAX processing failed: {e}")
            return self._fallback_to_cpu(inputs, model_name, f"MAX processing error: {e}")
    
    def _process_with_mojo(self, inputs: Any, model_name: str) -> Dict[str, Any]:
        """Process inputs using Mojo backend."""
        try:
            logger.info(f"Simulating Mojo processing for model: {model_name}")
            
            return {
                "backend": "Mojo (simulated)",
                "device": "mojo",
                "outputs": {"simulated": "mojo_output"},
                "model": model_name,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Mojo processing failed: {e}")
            return self._fallback_to_cpu(inputs, model_name, f"Mojo processing error: {e}")
    
    def _fallback_to_cpu(self, inputs: Any, model_name: str, reason: str) -> Dict[str, Any]:
        """Fallback to CPU processing when Mojo/MAX is not available."""
        return {
            "backend": "CPU (fallback)",
            "device": "cpu",
            "model": model_name,
            "fallback_reason": reason,
            "success": False,
            "message": f"Mojo/MAX processing failed, reason: {reason}"
        }
    
    def supports_mojo_max_target(self) -> bool:
        """Check if this instance supports Mojo/MAX targets."""
        return True
    
    def get_mojo_max_capabilities(self) -> Dict[str, Any]:
        """Get information about Mojo/MAX capabilities."""
        return {
            "max_available": self._is_max_available(),
            "mojo_available": self._is_mojo_available(),
            "environment_enabled": os.environ.get("USE_MOJO_MAX_TARGET", "").lower() in ("1", "true", "yes"),
            "supported_operations": ["inference", "graph_optimization"],
            "fallback_available": True
        }
