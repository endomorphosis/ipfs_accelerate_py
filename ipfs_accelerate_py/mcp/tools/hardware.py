"""
IPFS Accelerate MCP Hardware Tools

This module provides hardware-related tools for the IPFS Accelerate MCP server.
"""

import os
import sys
import json
import logging
import platform
from typing import Dict, Any, Optional, List, Union

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.tools.hardware")

def register_hardware_tools(mcp: Any) -> None:
    """
    Register hardware tools with the MCP server
    
    This function registers hardware-related tools with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering hardware tools")
    
    try:
        # Register get_hardware_info tool
        mcp.register_tool(
            name="get_hardware_info",
            function=get_hardware_info,
            description="Get information about available hardware accelerators",
            input_schema={
                "type": "object",
                "properties": {
                    "include_detailed": {
                        "type": "boolean",
                        "description": "Include detailed hardware information",
                        "default": False
                    }
                },
                "required": []
            }
        )
        
        # Register test_hardware tool
        mcp.register_tool(
            name="test_hardware",
            function=test_hardware,
            description="Test available hardware accelerators",
            input_schema={
                "type": "object",
                "properties": {
                    "accelerator": {
                        "type": "string",
                        "description": "Hardware accelerator to test (cuda, cpu, webgpu, webnn)",
                        "enum": ["cuda", "cpu", "webgpu", "webnn", "all"],
                        "default": "all"
                    },
                    "test_level": {
                        "type": "string",
                        "description": "Level of testing to perform",
                        "enum": ["basic", "comprehensive"],
                        "default": "basic"
                    }
                },
                "required": []
            }
        )
        
        # Register recommend_hardware tool
        mcp.register_tool(
            name="recommend_hardware",
            function=recommend_hardware,
            description="Get hardware recommendations for a specific model",
            input_schema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the model to get recommendations for",
                    },
                    "task": {
                        "type": "string",
                        "description": "Task to perform with the model",
                        "enum": ["inference", "training", "fine-tuning"],
                        "default": "inference"
                    },
                    "consider_available_only": {
                        "type": "boolean",
                        "description": "Only consider available hardware",
                        "default": True
                    }
                },
                "required": ["model_name"]
            }
        )
        
        logger.debug("Hardware tools registered")
    
    except Exception as e:
        logger.error(f"Error registering hardware tools: {e}")
        raise

def get_hardware_info(include_detailed: bool = False) -> Dict[str, Any]:
    """
    Get information about available hardware accelerators
    
    This function returns information about available hardware accelerators.
    
    Args:
        include_detailed: Include detailed hardware information
        
    Returns:
        Dictionary with hardware information
    """
    logger.debug("Getting hardware information")
    
    try:
        # Try to import ipfs_accelerate_py
        try:
            import ipfs_accelerate_py
            
            # Use ipfs_accelerate_py's hardware detection if available
            if hasattr(ipfs_accelerate_py, "hardware_detection"):
                hardware_info = ipfs_accelerate_py.hardware_detection.get_hardware_info(
                    include_detailed=include_detailed
                )
                
                # Return hardware info if available
                if isinstance(hardware_info, dict):
                    if "cpu" not in hardware_info:
                        accelerators = hardware_info.get("accelerators")
                        if isinstance(accelerators, dict) and "cpu" in accelerators:
                            hardware_info["cpu"] = accelerators["cpu"]
                    logger.debug("Hardware information retrieved from IPFS Accelerate")
                    return hardware_info
        except ImportError:
            pass
        
        # Fallback to basic hardware detection
        hardware_info = get_basic_hardware_info(include_detailed)
        
        logger.debug("Hardware information retrieved")
        
        if isinstance(hardware_info, dict) and "cpu" not in hardware_info:
            accelerators = hardware_info.get("accelerators")
            if isinstance(accelerators, dict) and "cpu" in accelerators:
                hardware_info["cpu"] = accelerators["cpu"]
        return hardware_info
    
    except Exception as e:
        logger.error(f"Error getting hardware information: {e}")
        raise

def get_basic_hardware_info(include_detailed: bool = False) -> Dict[str, Any]:
    """
    Get basic information about available hardware accelerators
    
    This function returns basic information about available hardware accelerators
    when IPFS Accelerate is not available or does not provide this information.
    
    Args:
        include_detailed: Include detailed hardware information
        
    Returns:
        Dictionary with hardware information
    """
    # Basic system information
    system_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor()
    }
    
    # Check for CPU details
    cpu_info = {
        "available": True,
        "name": platform.processor()
    }
    
    # Add CPU core count if psutil is available
    try:
        import psutil
        cpu_info["cores_physical"] = psutil.cpu_count(logical=False)
        cpu_info["cores_logical"] = psutil.cpu_count(logical=True)
        cpu_info["memory_total"] = psutil.virtual_memory().total / (1024 ** 3)  # GB
        cpu_info["memory_available"] = psutil.virtual_memory().available / (1024 ** 3)  # GB
    except ImportError:
        pass
    
    # Check for CUDA availability
    cuda_info = {"available": False}
    try:
        import torch
        cuda_info["available"] = torch.cuda.is_available()
        if cuda_info["available"]:
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["current_device"] = torch.cuda.current_device()
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            cuda_info["cuda_version"] = torch.version.cuda
            
            # Add detailed CUDA information if requested
            if include_detailed and cuda_info["available"]:
                cuda_info["devices"] = []
                for i in range(cuda_info["device_count"]):
                    device_info = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
                    }
                    cuda_info["devices"].append(device_info)
    except ImportError:
        pass
    
    # Check for WebGPU availability (placeholder, actual detection would need browser environment)
    webgpu_info = {"available": False}
    
    # Check for WebNN availability (placeholder, actual detection would need browser environment)
    webnn_info = {"available": False}
    
    # Construct the response
    hardware_info = {
        "system": system_info,
        "accelerators": {
            "cpu": cpu_info,
            "cuda": cuda_info,
            "webgpu": webgpu_info,
            "webnn": webnn_info
        },
        "recommendation": {
            "inference": "cpu" if not cuda_info["available"] else "cuda",
            "training": "cpu" if not cuda_info["available"] else "cuda"
        }
    }
    
    return hardware_info

def test_hardware(accelerator: str = "all", test_level: str = "basic") -> Dict[str, Any]:
    """
    Test available hardware accelerators
    
    This function tests available hardware accelerators to ensure they are working
    correctly.
    
    Args:
        accelerator: Hardware accelerator to test (cuda, cpu, webgpu, webnn, all)
        test_level: Level of testing to perform (basic, comprehensive)
        
    Returns:
        Dictionary with test results
    """
    logger.debug(f"Testing hardware: {accelerator}, level: {test_level}")
    
    try:
        # Try to import ipfs_accelerate_py
        try:
            import ipfs_accelerate_py
            
            # Use ipfs_accelerate_py's hardware testing if available
            if hasattr(ipfs_accelerate_py, "hardware_testing"):
                test_results = ipfs_accelerate_py.hardware_testing.test_hardware(
                    accelerator=accelerator,
                    test_level=test_level
                )
                
                # Return test results if available
                if isinstance(test_results, dict):
                    logger.debug("Hardware test results retrieved from IPFS Accelerate")
                    return test_results
        except ImportError:
            pass
        
        # Fallback to basic hardware testing
        test_results = perform_basic_hardware_test(accelerator, test_level)
        
        logger.debug("Hardware test results retrieved")
        
        return test_results
    
    except Exception as e:
        logger.error(f"Error testing hardware: {e}")
        raise

def perform_basic_hardware_test(accelerator: str = "all", test_level: str = "basic") -> Dict[str, Any]:
    """
    Perform basic hardware tests
    
    This function performs basic tests on hardware accelerators when IPFS Accelerate
    is not available or does not provide this functionality.
    
    Args:
        accelerator: Hardware accelerator to test (cuda, cpu, webgpu, webnn, all)
        test_level: Level of testing to perform (basic, comprehensive)
        
    Returns:
        Dictionary with test results
    """
    results = {
        "test_time": "",
        "test_level": test_level,
        "results": {}
    }
    
    import time
    start_time = time.time()
    
    # Test CPU
    if accelerator in ["cpu", "all"]:
        cpu_result = {"status": "pass", "details": {}}
        
        try:
            # Basic CPU test
            import numpy as np
            size = 1000 if test_level == "basic" else 5000
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            
            cpu_start = time.time()
            c = np.dot(a, b)
            cpu_end = time.time()
            
            cpu_result["details"]["matrix_multiply_time"] = cpu_end - cpu_start
            cpu_result["details"]["matrix_size"] = size
            
        except Exception as e:
            cpu_result["status"] = "fail"
            cpu_result["details"]["error"] = str(e)
        
        results["results"]["cpu"] = cpu_result
    
    # Test CUDA
    if accelerator in ["cuda", "all"]:
        cuda_result = {"status": "unknown", "details": {}}
        
        try:
            import torch
            
            if torch.cuda.is_available():
                # Basic CUDA test
                size = 1000 if test_level == "basic" else 5000
                a = torch.rand(size, size, device="cuda")
                b = torch.rand(size, size, device="cuda")
                
                cuda_start = time.time()
                c = torch.matmul(a, b)
                # Ensure the computation is done
                c = c.cpu()
                cuda_end = time.time()
                
                cuda_result["status"] = "pass"
                cuda_result["details"]["matrix_multiply_time"] = cuda_end - cuda_start
                cuda_result["details"]["matrix_size"] = size
                cuda_result["details"]["device"] = torch.cuda.get_device_name(0)
            else:
                cuda_result["status"] = "unavailable"
                cuda_result["details"]["reason"] = "CUDA not available"
        
        except Exception as e:
            cuda_result["status"] = "fail"
            cuda_result["details"]["error"] = str(e)
        
        results["results"]["cuda"] = cuda_result
    
    # Test WebGPU (placeholder, actual testing would need browser environment)
    if accelerator in ["webgpu", "all"]:
        results["results"]["webgpu"] = {
            "status": "unavailable",
            "details": {
                "reason": "WebGPU can only be tested in a browser environment"
            }
        }
    
    # Test WebNN (placeholder, actual testing would need browser environment)
    if accelerator in ["webnn", "all"]:
        results["results"]["webnn"] = {
            "status": "unavailable",
            "details": {
                "reason": "WebNN can only be tested in a browser environment"
            }
        }
    
    end_time = time.time()
    results["test_time"] = f"{end_time - start_time:.2f} seconds"
    
    return results

def recommend_hardware(
    model_name: str,
    task: str = "inference",
    consider_available_only: bool = True
) -> Dict[str, Any]:
    """
    Get hardware recommendations for a specific model
    
    This function returns hardware recommendations for a specific model based on
    the task to be performed.
    
    Args:
        model_name: Name of the model to get recommendations for
        task: Task to perform with the model (inference, training, fine-tuning)
        consider_available_only: Only consider available hardware
        
    Returns:
        Dictionary with hardware recommendations
    """
    logger.debug(f"Getting hardware recommendations for model: {model_name}, task: {task}")
    
    try:
        # Try to import ipfs_accelerate_py
        try:
            import ipfs_accelerate_py
            
            # Use ipfs_accelerate_py's hardware recommendations if available
            if hasattr(ipfs_accelerate_py, "hardware_recommendations"):
                recommendations = ipfs_accelerate_py.hardware_recommendations.recommend_hardware(
                    model_name=model_name,
                    task=task,
                    consider_available_only=consider_available_only
                )
                
                # Return recommendations if available
                if isinstance(recommendations, dict):
                    logger.debug("Hardware recommendations retrieved from IPFS Accelerate")
                    return recommendations
        except ImportError:
            pass
        
        # Fallback to basic hardware recommendations
        recommendations = get_basic_hardware_recommendations(
            model_name,
            task,
            consider_available_only
        )
        
        logger.debug("Hardware recommendations retrieved")
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error getting hardware recommendations: {e}")
        raise

def get_basic_hardware_recommendations(
    model_name: str,
    task: str = "inference",
    consider_available_only: bool = True
) -> Dict[str, Any]:
    """
    Get basic hardware recommendations for a specific model
    
    This function returns basic hardware recommendations for a specific model when
    IPFS Accelerate is not available or does not provide this information.
    
    Args:
        model_name: Name of the model to get recommendations for
        task: Task to perform with the model (inference, training, fine-tuning)
        consider_available_only: Only consider available hardware
        
    Returns:
        Dictionary with hardware recommendations
    """
    # Get available hardware
    hardware_info = get_hardware_info(include_detailed=False)
    
    # Check if CUDA is available
    cuda_available = hardware_info["accelerators"]["cuda"]["available"]
    
    # Basic model size estimation
    model_sizes = {
        "llama-7b": {"parameters": 7, "size_gb": 14},
        "llama-13b": {"parameters": 13, "size_gb": 26},
        "llama-70b": {"parameters": 70, "size_gb": 140},
        "clip": {"parameters": 0.4, "size_gb": 1},
        "whisper": {"parameters": 1.5, "size_gb": 3},
        "llava": {"parameters": 7, "size_gb": 14},
    }
    
    # Check if model is in known models
    model_info = model_sizes.get(model_name.lower(), {"parameters": 1, "size_gb": 2})
    
    # Determine if model is large
    is_large_model = model_info["parameters"] > 10
    
    # Basic recommendations
    if task == "inference":
        if cuda_available and (not consider_available_only or cuda_available):
            primary_recommendation = "cuda"
            reason = "CUDA provides the best performance for inference"
            settings = {
                "batch_size": 1,
                "dtype": "float16",
                "quantization": "int8" if is_large_model else "none"
            }
        else:
            primary_recommendation = "cpu"
            reason = "CPU is the only available option for inference"
            settings = {
                "batch_size": 1,
                "dtype": "float32",
                "quantization": "int8" if is_large_model else "none"
            }
    elif task in ["training", "fine-tuning"]:
        if cuda_available and (not consider_available_only or cuda_available):
            primary_recommendation = "cuda"
            reason = "CUDA provides the best performance for training"
            settings = {
                "batch_size": 1 if is_large_model else 4,
                "dtype": "bfloat16" if is_large_model else "float32",
                "gradient_accumulation_steps": 8 if is_large_model else 1
            }
        else:
            primary_recommendation = "cpu"
            reason = "CPU is the only available option for training (not recommended for large models)"
            settings = {
                "batch_size": 1,
                "dtype": "float32",
                "gradient_accumulation_steps": 16
            }
    
    # Construct the response
    recommendations = {
        "model": model_name,
        "task": task,
        "recommended_accelerator": primary_recommendation,
        "reason": reason,
        "recommended_settings": settings,
        "alternative_accelerators": [],
        "model_info": model_info,
        "considered_hardware": {
            "cpu": True,
            "cuda": cuda_available or not consider_available_only,
            "webgpu": False,
            "webnn": False
        }
    }
    
    # Add alternative recommendations
    if primary_recommendation == "cuda" and task == "inference":
        recommendations["alternative_accelerators"].append({
            "accelerator": "cpu",
            "reason": "CPU can be used for inference if CUDA is not available",
            "recommended_settings": {
                "batch_size": 1,
                "dtype": "float32",
                "quantization": "int8" if is_large_model else "none"
            }
        })
    
    return recommendations
