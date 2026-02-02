#!/usr/bin/env python3
"""
Dynamic Precision Manager for HuggingFace Skill Generator

This module provides automatic precision management with dynamic switching
between FP32, FP16, and BF16 based on hardware capabilities and runtime errors.

Features:
- Automatic precision detection per hardware
- Dynamic FP16→FP32 switching on precision errors
- Mixed precision support
- Hardware capability detection
- Graceful degradation
"""

import os
import logging
from typing import Dict, Optional, Any, Callable
from enum import Enum
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecisionType(Enum):
    """Supported precision types."""
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"
    AUTO = "auto"


class PrecisionManager:
    """
    Dynamic precision manager for cross-platform model execution.
    
    Automatically detects hardware capabilities and manages precision
    switching to prevent numerical instabilities.
    """
    
    def __init__(self, hardware_type: str = "cpu", device_id: int = 0):
        """
        Initialize the precision manager.
        
        Args:
            hardware_type: Target hardware (cpu, cuda, rocm, mps, openvino, qnn)
            device_id: GPU device ID
        """
        self.hardware_type = hardware_type.lower()
        self.device_id = device_id
        self.current_precision = PrecisionType.FP32
        self.fallback_precision = PrecisionType.FP32
        self.auto_fallback_enabled = True
        self.precision_errors = []
        
        # Detect hardware capabilities
        self.capabilities = self._detect_capabilities()
        
        # Set default precision based on hardware
        self.current_precision = self._get_default_precision()
        
        logger.info(f"Precision Manager initialized for {hardware_type}")
        logger.info(f"Default precision: {self.current_precision.value}")
        logger.info(f"Hardware capabilities: {self.capabilities}")
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """
        Detect hardware precision capabilities.
        
        Returns:
            Dictionary of supported precision types
        """
        capabilities = {
            "fp32": True,  # Always supported
            "fp16": False,
            "bf16": False,
            "int8": False,
            "mixed_precision": False
        }
        
        try:
            import torch
            
            if self.hardware_type == "cuda":
                if torch.cuda.is_available():
                    # Check compute capability
                    compute_capability = torch.cuda.get_device_capability(self.device_id)
                    major, minor = compute_capability
                    
                    # FP16 supported on compute capability >= 5.3
                    capabilities["fp16"] = (major >= 5 and minor >= 3) or major >= 6
                    
                    # BF16 supported on compute capability >= 8.0 (Ampere+)
                    capabilities["bf16"] = major >= 8
                    
                    # INT8 generally supported on modern CUDA
                    capabilities["int8"] = major >= 6
                    
                    # Mixed precision supported with AMP
                    capabilities["mixed_precision"] = major >= 7
                    
                    logger.info(f"CUDA compute capability: {major}.{minor}")
            
            elif self.hardware_type == "rocm":
                if hasattr(torch, 'hip') and torch.hip.is_available():
                    # ROCm generally supports FP16 and mixed precision
                    capabilities["fp16"] = True
                    capabilities["mixed_precision"] = True
                    # BF16 support depends on ROCm version
                    capabilities["bf16"] = False  # Conservative default
                    capabilities["int8"] = True
            
            elif self.hardware_type == "mps":
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # MPS supports FP16 but with limitations
                    capabilities["fp16"] = True
                    capabilities["mixed_precision"] = False
                    capabilities["bf16"] = False
                    capabilities["int8"] = False
            
            elif self.hardware_type == "cpu":
                # CPU supports all precisions in software
                capabilities["fp16"] = True
                capabilities["bf16"] = True
                capabilities["int8"] = True
                capabilities["mixed_precision"] = False
        
        except ImportError:
            logger.warning("PyTorch not available, using conservative capabilities")
        
        return capabilities
    
    def _get_default_precision(self) -> PrecisionType:
        """
        Get the default precision for current hardware.
        
        Returns:
            Default PrecisionType
        """
        if self.hardware_type == "cuda":
            # Use FP16 if supported (compute capability >= 7)
            if self.capabilities.get("fp16") and self.capabilities.get("mixed_precision"):
                return PrecisionType.FP16
        
        elif self.hardware_type == "rocm":
            # ROCm supports FP16 well
            if self.capabilities.get("fp16"):
                return PrecisionType.FP16
        
        elif self.hardware_type == "mps":
            # MPS can use FP16 but be conservative
            return PrecisionType.FP32
        
        # Default to FP32 for safety
        return PrecisionType.FP32
    
    def get_torch_dtype(self, precision: Optional[PrecisionType] = None):
        """
        Get PyTorch dtype for a precision type.
        
        Args:
            precision: Precision type (uses current if None)
        
        Returns:
            torch.dtype object
        """
        try:
            import torch
            
            if precision is None:
                precision = self.current_precision
            
            dtype_map = {
                PrecisionType.FP32: torch.float32,
                PrecisionType.FP16: torch.float16,
                PrecisionType.BF16: torch.bfloat16,
                PrecisionType.INT8: torch.int8
            }
            
            return dtype_map.get(precision, torch.float32)
        
        except ImportError:
            logger.warning("PyTorch not available")
            return None
    
    def set_precision(self, precision: PrecisionType, force: bool = False) -> bool:
        """
        Set the current precision.
        
        Args:
            precision: Desired precision type
            force: Force setting even if not supported
        
        Returns:
            True if successfully set, False otherwise
        """
        # Check if supported
        if precision == PrecisionType.FP16 and not self.capabilities.get("fp16"):
            if not force:
                logger.warning(f"FP16 not supported on {self.hardware_type}, staying with {self.current_precision.value}")
                return False
        
        if precision == PrecisionType.BF16 and not self.capabilities.get("bf16"):
            if not force:
                logger.warning(f"BF16 not supported on {self.hardware_type}, staying with {self.current_precision.value}")
                return False
        
        self.current_precision = precision
        logger.info(f"Precision set to {precision.value}")
        return True
    
    @contextmanager
    def precision_context(self, precision: PrecisionType):
        """
        Context manager for temporary precision changes.
        
        Args:
            precision: Temporary precision to use
        
        Example:
            with precision_manager.precision_context(PrecisionType.FP16):
                output = model(inputs)
        """
        original_precision = self.current_precision
        self.set_precision(precision)
        
        try:
            yield
        finally:
            self.current_precision = original_precision
    
    def enable_mixed_precision(self) -> bool:
        """
        Enable mixed precision training/inference.
        
        Returns:
            True if enabled, False if not supported
        """
        if not self.capabilities.get("mixed_precision"):
            logger.warning(f"Mixed precision not supported on {self.hardware_type}")
            return False
        
        try:
            import torch
            
            if self.hardware_type == "cuda":
                # Enable TF32 for Ampere+ GPUs
                if self.capabilities.get("bf16"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("TF32 enabled for CUDA")
            
            logger.info("Mixed precision enabled")
            return True
        
        except ImportError:
            return False
    
    def create_autocast_context(self, enabled: bool = True):
        """
        Create an autocast context for mixed precision.
        
        Args:
            enabled: Whether autocast is enabled
        
        Returns:
            torch.autocast context manager or dummy context
        """
        try:
            import torch
            
            if self.hardware_type == "cuda":
                return torch.cuda.amp.autocast(enabled=enabled)
            elif self.hardware_type == "cpu":
                return torch.cpu.amp.autocast(enabled=enabled)
            else:
                # Return a dummy context manager
                from contextlib import nullcontext
                return nullcontext()
        
        except ImportError:
            from contextlib import nullcontext
            return nullcontext()
    
    def handle_precision_error(self, error: Exception, operation_name: str = "unknown") -> bool:
        """
        Handle precision-related errors by falling back to FP32.
        
        Args:
            error: The exception that occurred
            operation_name: Name of the operation that failed
        
        Returns:
            True if fallback was applied, False otherwise
        """
        if not self.auto_fallback_enabled:
            return False
        
        # Check if this is a precision-related error
        error_str = str(error).lower()
        precision_keywords = [
            "inf", "nan", "overflow", "underflow",
            "numerical", "precision", "fp16", "half"
        ]
        
        is_precision_error = any(keyword in error_str for keyword in precision_keywords)
        
        if not is_precision_error:
            return False
        
        # Record the error
        self.precision_errors.append({
            "operation": operation_name,
            "error": str(error),
            "original_precision": self.current_precision.value
        })
        
        # Fall back to FP32
        if self.current_precision != PrecisionType.FP32:
            logger.warning(
                f"Precision error detected in '{operation_name}': {error}"
            )
            logger.warning(
                f"Falling back from {self.current_precision.value} to FP32"
            )
            self.current_precision = PrecisionType.FP32
            return True
        
        return False
    
    @contextmanager
    def safe_precision_context(self, operation_name: str = "operation"):
        """
        Context manager that automatically falls back to FP32 on precision errors.
        
        Args:
            operation_name: Name of the operation for logging
        
        Example:
            with precision_manager.safe_precision_context("inference"):
                try:
                    output = model(inputs)
                except Exception as e:
                    # Will automatically fall back if precision error
                    pass
        """
        original_precision = self.current_precision
        
        try:
            yield
        except Exception as e:
            if self.handle_precision_error(e, operation_name):
                logger.info(f"Retrying {operation_name} with FP32")
                # Caller should retry the operation
            raise
    
    def get_precision_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for current precision settings.
        
        Returns:
            Configuration dictionary
        """
        config = {
            "hardware_type": self.hardware_type,
            "current_precision": self.current_precision.value,
            "torch_dtype": str(self.get_torch_dtype()),
            "capabilities": self.capabilities,
            "auto_fallback_enabled": self.auto_fallback_enabled,
            "precision_errors_count": len(self.precision_errors)
        }
        
        return config
    
    def validate_tensor_precision(self, tensor, expected_precision: Optional[PrecisionType] = None):
        """
        Validate that a tensor has the expected precision.
        
        Args:
            tensor: Tensor to validate
            expected_precision: Expected precision (uses current if None)
        
        Returns:
            True if precision matches, False otherwise
        """
        try:
            import torch
            
            if not isinstance(tensor, torch.Tensor):
                return False
            
            if expected_precision is None:
                expected_precision = self.current_precision
            
            expected_dtype = self.get_torch_dtype(expected_precision)
            
            if tensor.dtype != expected_dtype:
                logger.warning(
                    f"Tensor dtype mismatch: got {tensor.dtype}, expected {expected_dtype}"
                )
                return False
            
            return True
        
        except ImportError:
            return False
    
    def check_for_numerical_issues(self, tensor) -> Dict[str, bool]:
        """
        Check tensor for NaN and Inf values.
        
        Args:
            tensor: Tensor to check
        
        Returns:
            Dictionary with has_nan and has_inf flags
        """
        try:
            import torch
            
            if not isinstance(tensor, torch.Tensor):
                return {"has_nan": False, "has_inf": False, "error": "Not a tensor"}
            
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            
            if has_nan or has_inf:
                logger.warning(
                    f"Numerical issues detected: NaN={has_nan}, Inf={has_inf}"
                )
            
            return {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape)
            }
        
        except ImportError:
            return {"has_nan": False, "has_inf": False, "error": "PyTorch not available"}
    
    def get_precision_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about precision usage and errors.
        
        Returns:
            Dictionary with precision statistics
        """
        return {
            "current_precision": self.current_precision.value,
            "hardware_type": self.hardware_type,
            "capabilities": self.capabilities,
            "total_errors": len(self.precision_errors),
            "errors": self.precision_errors[-10:]  # Last 10 errors
        }
    
    def reset_error_tracking(self):
        """Reset precision error tracking."""
        self.precision_errors.clear()
        logger.info("Precision error tracking reset")


def create_precision_config(hardware_type: str) -> Dict[str, Any]:
    """
    Convenience function to create a precision configuration.
    
    Args:
        hardware_type: Target hardware type
    
    Returns:
        Precision configuration dictionary
    """
    manager = PrecisionManager(hardware_type)
    return manager.get_precision_config()


def safe_model_inference(model, inputs, precision_manager: PrecisionManager):
    """
    Perform model inference with automatic precision fallback.
    
    Args:
        model: The model to run
        inputs: Model inputs
        precision_manager: PrecisionManager instance
    
    Returns:
        Model outputs
    """
    max_retries = 2
    attempt = 0
    
    while attempt < max_retries:
        try:
            with precision_manager.safe_precision_context("inference"):
                # Run inference with current precision
                with precision_manager.create_autocast_context():
                    outputs = model(inputs)
                
                # Check for numerical issues
                if hasattr(outputs, 'logits'):
                    issues = precision_manager.check_for_numerical_issues(outputs.logits)
                    if issues.get("has_nan") or issues.get("has_inf"):
                        raise ValueError("Numerical instability detected in outputs")
                
                return outputs
        
        except Exception as e:
            if precision_manager.handle_precision_error(e, "inference"):
                attempt += 1
                if attempt < max_retries:
                    logger.info(f"Retrying inference with FP32 (attempt {attempt + 1}/{max_retries})")
                    continue
            raise
    
    raise RuntimeError(f"Failed to run inference after {max_retries} attempts")


if __name__ == "__main__":
    # Example usage
    print("=== Precision Manager Examples ===\n")
    
    # CUDA example
    print("CUDA Hardware:")
    cuda_manager = PrecisionManager("cuda")
    print(f"  Default precision: {cuda_manager.current_precision.value}")
    print(f"  Capabilities: {cuda_manager.capabilities}")
    print()
    
    # CPU example
    print("CPU Hardware:")
    cpu_manager = PrecisionManager("cpu")
    print(f"  Default precision: {cpu_manager.current_precision.value}")
    print(f"  Capabilities: {cpu_manager.capabilities}")
    print()
    
    # Precision switching
    print("Precision Switching:")
    success = cuda_manager.set_precision(PrecisionType.FP16)
    print(f"  Switched to FP16: {success}")
    print(f"  Current config: {cuda_manager.get_precision_config()}")
