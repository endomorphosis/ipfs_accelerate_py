#!/usr/bin/env python
"""
Benchmark error handling improvements.

This module provides:
1. Structured error categorization for benchmarks
2. Proper fallback mechanisms for missing hardware
3. Clear delineation of real vs. simulated hardware results

Implementation date: April 9, 2025
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Error categories for clear tracking
ERROR_CATEGORY_HARDWARE_NOT_AVAILABLE = "hardware_not_available"
ERROR_CATEGORY_MODEL_INCOMPATIBLE = "model_incompatible"
ERROR_CATEGORY_MEMORY_ERROR = "memory_error"
ERROR_CATEGORY_IMPORT_ERROR = "import_error"
ERROR_CATEGORY_RUNTIME_ERROR = "runtime_error"
ERROR_CATEGORY_API_ERROR = "api_error"
ERROR_CATEGORY_TIMEOUT = "timeout"
ERROR_CATEGORY_SYSTEM_ERROR = "system_error"
ERROR_CATEGORY_UNKNOWN = "unknown"

class BenchmarkError:
    """Structured error information for benchmarks"""
    
    def __init__(self, 
                 message: str, 
                 category: str = ERROR_CATEGORY_UNKNOWN, 
                 exception: Optional[Exception] = None,
                 hardware_type: Optional[str] = None,
                 model_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize a benchmark error
        
        Args:
            message: Error message
            category: Error category (use constants defined in this module)
            exception: Original exception if available
            hardware_type: Hardware type where the error occurred
            model_name: Model name being tested
            details: Additional error details
        """
        self.message = message
        self.category = category
        self.exception = exception
        self.hardware_type = hardware_type
        self.model_name = model_name
        
        # Initialize details dictionary
        self.details = details or {}
        
        # Add exception details if available
        if exception:
            self.details.update({
                "exception_type": type(exception).__name__,
                "exception_str": str(exception),
                "traceback": traceback.format_exc()
            })
        
        # Add hardware and model info to details if available
        if hardware_type:
            self.details["hardware_type"] = hardware_type
        if model_name:
            self.details["model_name"] = model_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage or JSON serialization"""
        return {
            "message": self.message,
            "category": self.category,
            "hardware_type": self.hardware_type,
            "model_name": self.model_name,
            "details": self.details
        }
    
    def __str__(self) -> str:
        """String representation of the error"""
        parts = [f"[{self.category}] {self.message}"]
        
        if self.hardware_type:
            parts.append(f"Hardware: {self.hardware_type}")
        if self.model_name:
            parts.append(f"Model: {self.model_name}")
        
        if self.exception:
            parts.append(f"Exception: {type(self.exception).__name__}: {str(self.exception)}")
        
        return " | ".join(parts)


class HardwareNotAvailableError(BenchmarkError):
    """Error raised when hardware is not available"""
    
    def __init__(self, 
                 hardware_type: str, 
                 model_name: Optional[str] = None,
                 message: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize a hardware not available error
        
        Args:
            hardware_type: Unavailable hardware type
            model_name: Model being tested (optional)
            message: Custom error message (optional)
            details: Additional error details (optional)
        """
        if not message:
            message = f"Hardware {hardware_type} is not available for testing"
        
        super().__init__(
            message=message,
            category=ERROR_CATEGORY_HARDWARE_NOT_AVAILABLE,
            hardware_type=hardware_type,
            model_name=model_name,
            details=details
        )


class ModelIncompatibleError(BenchmarkError):
    """Error raised when a model is incompatible with hardware"""
    
    def __init__(self, 
                 model_name: str, 
                 hardware_type: str,
                 reason: str,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize a model incompatible error
        
        Args:
            model_name: Name of the incompatible model
            hardware_type: Hardware type being used
            reason: Reason for incompatibility
            details: Additional error details (optional)
        """
        message = f"Model {model_name} is incompatible with {hardware_type}: {reason}"
        
        # Start with provided details or empty dict
        error_details = details or {}
        
        # Add incompatibility reason
        error_details["incompatibility_reason"] = reason
        
        super().__init__(
            message=message,
            category=ERROR_CATEGORY_MODEL_INCOMPATIBLE,
            hardware_type=hardware_type,
            model_name=model_name,
            details=error_details
        )


class MemoryError(BenchmarkError):
    """Error raised when there's insufficient memory"""
    
    def __init__(self, 
                 hardware_type: str, 
                 model_name: str,
                 required_memory: Optional[float] = None,
                 available_memory: Optional[float] = None,
                 exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize a memory error
        
        Args:
            hardware_type: Hardware type being used
            model_name: Name of the model
            required_memory: Required memory in MB (optional)
            available_memory: Available memory in MB (optional)
            exception: Original exception if available
            details: Additional error details (optional)
        """
        if required_memory and available_memory:
            message = f"Insufficient memory for model {model_name} on {hardware_type}: Required {required_memory:.2f}MB, Available {available_memory:.2f}MB"
        else:
            message = f"Memory error when running model {model_name} on {hardware_type}"
        
        # Start with provided details or empty dict
        error_details = details or {}
        
        # Add memory information if available
        if required_memory is not None:
            error_details["required_memory_mb"] = required_memory
        if available_memory is not None:
            error_details["available_memory_mb"] = available_memory
        
        super().__init__(
            message=message,
            category=ERROR_CATEGORY_MEMORY_ERROR,
            exception=exception,
            hardware_type=hardware_type,
            model_name=model_name,
            details=error_details
        )


def categorize_error(exception: Exception, 
                    hardware_type: Optional[str] = None, 
                    model_name: Optional[str] = None) -> BenchmarkError:
    """
    Categorize an exception into a benchmark error
    
    Args:
        exception: Exception to categorize
        hardware_type: Hardware type where the error occurred
        model_name: Model name being tested
    
    Returns:
        BenchmarkError with appropriate category
    """
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__
    
    # Check for out of memory errors
    if "cuda out of memory" in exception_str or "insufficient memory" in exception_str or "out of memory" in exception_str:
        return MemoryError(
            hardware_type=hardware_type,
            model_name=model_name,
            exception=exception
        )
    
    # Check for import errors
    if isinstance(exception, ImportError):
        return BenchmarkError(
            message=f"Failed to import required module: {str(exception)}",
            category=ERROR_CATEGORY_IMPORT_ERROR,
            exception=exception,
            hardware_type=hardware_type,
            model_name=model_name
        )
    
    # Check for runtime errors
    if isinstance(exception, RuntimeError):
        # Check if this is related to hardware not being available
        if "not available" in exception_str or "not found" in exception_str:
            if hardware_type:
                return HardwareNotAvailableError(
                    hardware_type=hardware_type,
                    model_name=model_name,
                    message=str(exception),
                    details={"original_exception": str(exception)}
                )
        
        # Check if this is related to model compatibility
        if "model" in exception_str and ("incompatible" in exception_str or "not supported" in exception_str):
            if model_name and hardware_type:
                return ModelIncompatibleError(
                    model_name=model_name,
                    hardware_type=hardware_type,
                    reason=str(exception)
                )
        
        # General runtime error
        return BenchmarkError(
            message=str(exception),
            category=ERROR_CATEGORY_RUNTIME_ERROR,
            exception=exception,
            hardware_type=hardware_type,
            model_name=model_name
        )
    
    # Check for system errors
    if isinstance(exception, (OSError, SystemError)):
        return BenchmarkError(
            message=str(exception),
            category=ERROR_CATEGORY_SYSTEM_ERROR,
            exception=exception,
            hardware_type=hardware_type,
            model_name=model_name
        )
    
    # Default: unknown error
    return BenchmarkError(
        message=str(exception),
        category=ERROR_CATEGORY_UNKNOWN,
        exception=exception,
        hardware_type=hardware_type,
        model_name=model_name
    )


def handle_hardware_unavailable(hardware_type: str, 
                               model_name: Optional[str] = None,
                               fallback_hardware: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Handle case when hardware is not available with proper fallbacks
    
    Args:
        hardware_type: Unavailable hardware type
        model_name: Model name being tested (optional)
        fallback_hardware: Fallback hardware type to suggest (optional)
    
    Returns:
        Tuple of (success, result_dict)
    """
    error = HardwareNotAvailableError(
        hardware_type=hardware_type,
        model_name=model_name
    )
    
    result = {
        "success": False,
        "hardware_type": hardware_type,
        "model_name": model_name,
        "error_message": str(error),
        "error_category": error.category,
        "error_details": error.details,
        "is_simulated": False
    }
    
    # Add fallback suggestion if provided
    if fallback_hardware:
        result["fallback_suggestion"] = f"Try using {fallback_hardware} instead"
        result["fallback_hardware"] = fallback_hardware
    
    # Log the error
    logger.warning(f"Hardware not available: {error}")
    
    return False, result


def handle_simulated_hardware(hardware_type: str, 
                            model_name: Optional[str] = None,
                            simulation_reason: Optional[str] = None) -> Dict[str, Any]:
    """
    Add simulation flags to a result dictionary
    
    Args:
        hardware_type: Simulated hardware type
        model_name: Model name being tested (optional)
        simulation_reason: Reason for simulation (optional)
    
    Returns:
        Dictionary with simulation flags
    """
    if not simulation_reason:
        simulation_reason = f"Hardware {hardware_type} is being simulated (not physically present)"
    
    # Log clear warning
    logger.warning(f"SIMULATED HARDWARE: {hardware_type} is being simulated for model {model_name}")
    logger.warning(f"Reason: {simulation_reason}")
    logger.warning("Results will NOT reflect real hardware performance!")
    
    # Return dictionary with simulation flags
    return {
        "is_simulated": True,
        "simulation_reason": simulation_reason,
        "hardware_type": hardware_type,
        "model_name": model_name
    }


def handle_benchmark_exception(exception: Exception,
                             hardware_type: Optional[str] = None,
                             model_name: Optional[str] = None,
                             is_simulated: bool = False,
                             simulation_reason: Optional[str] = None) -> Dict[str, Any]:
    """
    Handle exceptions in benchmarks with proper categorization
    
    Args:
        exception: Exception that occurred
        hardware_type: Hardware type where the error occurred
        model_name: Model name being tested
        is_simulated: Whether the hardware is being simulated
        simulation_reason: Reason for simulation if applicable
    
    Returns:
        Dictionary with error information
    """
    # Categorize the error
    error = categorize_error(exception, hardware_type, model_name)
    
    # Create result dictionary
    result = {
        "success": False,
        "error_message": str(error),
        "error_category": error.category,
        "error_details": error.details
    }
    
    # Add hardware and model info if available
    if hardware_type:
        result["hardware_type"] = hardware_type
    if model_name:
        result["model_name"] = model_name
    
    # Add simulation data if applicable
    if is_simulated:
        result["is_simulated"] = True
        result["simulation_reason"] = simulation_reason or "Hardware is being simulated"
    
    # Log the error
    logger.error(f"Benchmark error: {error}")
    
    return result


if __name__ == "__main__":
    # Test error handling functions
    print("Testing benchmark error handling...\n")
    
    # Test handling hardware unavailable
    print("Testing hardware unavailable handling:")
    success, result = handle_hardware_unavailable("cuda", "bert-base-uncased", "cpu")
    print(f"Success: {success}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    print("\nTesting simulated hardware handling:")
    sim_result = handle_simulated_hardware("webgpu", "bert-base-uncased")
    print(f"Result: {json.dumps(sim_result, indent=2)}")
    
    print("\nTesting exception handling:")
    try:
        # Simulate a CUDA out of memory error
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    except Exception as e:
        result = handle_benchmark_exception(e, "cuda", "large-model")
        print(f"Result: {json.dumps(result, indent=2)}")
        
    # Test with simulated error
    try:
        # Simulate a hardware not available error
        raise ImportError("No module named 'webgpu'")
    except Exception as e:
        result = handle_benchmark_exception(e, "webgpu", "bert-base-uncased", is_simulated=True)
        print(f"Result: {json.dumps(result, indent=2)}")