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
    
    This function registers hardware tools with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering hardware tools")
    
    try:
        # Register get_hardware_info tool
        mcp.register_tool(
            name="get_hardware_info",
            function=get_hardware_info_tool,
            description="Get information about available hardware accelerators",
            input_schema={
                "type": "object",
                "properties": {
                    "include_detailed": {
                        "type": "boolean",
                        "description": "Include detailed hardware information",
                        "default": False
                    }
                }
            }
        )
        
        # Register test_hardware tool
        mcp.register_tool(
            name="test_hardware",
            function=test_hardware_tool,
            description="Test available hardware accelerators",
            input_schema={
                "type": "object",
                "properties": {
                    "tests": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["cpu", "cuda", "rocm", "webgpu", "webnn"]
                        },
                        "description": "List of hardware tests to run",
                        "default": ["cpu"]
                    },
                    "include_benchmarks": {
                        "type": "boolean",
                        "description": "Include benchmarks in test results",
                        "default": False
                    }
                }
            }
        )
        
        # Register recommend_hardware tool
        mcp.register_tool(
            name="recommend_hardware",
            function=recommend_hardware_tool,
            description="Recommend hardware accelerators for a specific model",
            input_schema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the model",
                        "default": ""
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Type of the model",
                        "default": "llm",
                        "enum": ["llm", "vision", "audio", "multimodal"]
                    },
                    "include_available_only": {
                        "type": "boolean",
                        "description": "Include only available hardware accelerators",
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

def get_hardware_info_tool(include_detailed: bool = False) -> Dict[str, Any]:
    """
    Get information about available hardware accelerators
    
    Args:
        include_detailed: Include detailed hardware information
        
    Returns:
        Dictionary with hardware information
    """
    logger.debug("Getting hardware information")
    
    try:
        # Get hardware information
        hardware_info = get_hardware_info(include_detailed=include_detailed)
        
        logger.debug("Hardware information retrieved")
        
        return hardware_info
    
    except Exception as e:
        logger.error(f"Error getting hardware information: {e}")
        raise

def test_hardware_tool(tests: List[str] = ["cpu"], include_benchmarks: bool = False) -> Dict[str, Any]:
    """
    Test available hardware accelerators
    
    Args:
        tests: List of hardware tests to run
        include_benchmarks: Include benchmarks in test results
        
    Returns:
        Dictionary with test results
    """
    logger.debug(f"Testing hardware: {tests}")
    
    try:
        # Test hardware
        test_results = test_hardware(tests=tests, include_benchmarks=include_benchmarks)
        
        logger.debug("Hardware tests completed")
        
        return test_results
    
    except Exception as e:
        logger.error(f"Error testing hardware: {e}")
        raise

def recommend_hardware_tool(model_name: str, model_type: str = "llm", include_available_only: bool = True) -> Dict[str, Any]:
    """
    Recommend hardware accelerators for a specific model
    
    Args:
        model_name: Name of the model
        model_type: Type of the model
        include_available_only: Include only available hardware accelerators
        
    Returns:
        Dictionary with hardware recommendations
    """
    logger.debug(f"Recommending hardware for model {model_name} (type: {model_type})")
    
    try:
        # Recommend hardware
        recommendations = recommend_hardware(
            model_name=model_name,
            model_type=model_type,
            include_available_only=include_available_only
        )
        
        logger.debug("Hardware recommendations generated")
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error recommending hardware: {e}")
        raise

def get_hardware_info(include_detailed: bool = False) -> Dict[str, Any]:
    """
    Get information about available hardware accelerators
    
    Args:
        include_detailed: Include detailed hardware information
        
    Returns:
        Dictionary with hardware information
    """
    hardware_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "available_accelerators": []
    }
    
    # Check for CUDA
    try:
        import torch
        
        if torch.cuda.is_available():
            cuda_devices = []
            for i in range(torch.cuda.device_count()):
                device_props = {
                    "name": torch.cuda.get_device_name(i),
                    "index": i,
                    "total_memory_mb": round(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)),
                }
                
                if include_detailed:
                    device = torch.cuda.get_device_properties(i)
                    device_props["compute_capability"] = f"{device.major}.{device.minor}"
                    device_props["multi_processor_count"] = device.multi_processor_count
                    device_props["max_threads_per_block"] = device.max_threads_per_block
                    device_props["max_threads_per_multi_processor"] = device.max_threads_per_multi_processor
                
                cuda_devices.append(device_props)
            
            hardware_info["available_accelerators"].append({
                "type": "cuda",
                "version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": cuda_devices
            })
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}")
    
    # Check for CPU
    try:
        import psutil
        
        cpu_info = {
            "type": "cpu",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency_mhz": round(psutil.cpu_freq().max) if psutil.cpu_freq() else None,
            "available_memory_mb": round(psutil.virtual_memory().total / (1024 * 1024))
        }
        
        if include_detailed:
            cpu_info["percent_usage"] = psutil.cpu_percent(interval=0.1)
            cpu_info["memory_percent"] = psutil.virtual_memory().percent
        
        hardware_info["available_accelerators"].append(cpu_info)
    except ImportError:
        # Add basic CPU info
        hardware_info["available_accelerators"].append({
            "type": "cpu",
            "physical_cores": os.cpu_count(),
            "logical_cores": os.cpu_count()
        })
    except Exception as e:
        logger.warning(f"Error checking CPU availability: {e}")
    
    # Additional accelerators can be added here
    
    return hardware_info

def test_hardware(tests: List[str] = ["cpu"], include_benchmarks: bool = False) -> Dict[str, Any]:
    """
    Test available hardware accelerators
    
    Args:
        tests: List of hardware tests to run
        include_benchmarks: Include benchmarks in test results
        
    Returns:
        Dictionary with test results
    """
    test_results = {
        "tests_completed": [],
        "tests_failed": [],
        "results": {}
    }
    
    # Test CPU
    if "cpu" in tests:
        try:
            import time
            import numpy as np
            
            # Simple CPU test
            start_time = time.time()
            size = 2000
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            c = np.dot(a, b)
            elapsed_time = time.time() - start_time
            
            result = {
                "status": "success",
                "description": "Matrix multiplication test completed successfully"
            }
            
            if include_benchmarks:
                result["benchmark"] = {
                    "matrix_size": size,
                    "elapsed_time_s": elapsed_time,
                    "operations": 2 * size**3,
                    "gflops": 2 * size**3 / elapsed_time / 1e9
                }
            
            test_results["results"]["cpu"] = result
            test_results["tests_completed"].append("cpu")
        except Exception as e:
            test_results["results"]["cpu"] = {
                "status": "failed",
                "error": str(e)
            }
            test_results["tests_failed"].append("cpu")
    
    # Test CUDA
    if "cuda" in tests:
        try:
            import torch
            import time
            
            if torch.cuda.is_available():
                # Simple CUDA test
                size = 5000
                a = torch.rand(size, size, device="cuda")
                b = torch.rand(size, size, device="cuda")
                
                # Warm-up
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                
                result = {
                    "status": "success",
                    "device_count": torch.cuda.device_count(),
                    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    "description": "Matrix multiplication test completed successfully"
                }
                
                if include_benchmarks:
                    result["benchmark"] = {
                        "matrix_size": size,
                        "elapsed_time_s": elapsed_time,
                        "operations": 2 * size**3,
                        "gflops": 2 * size**3 / elapsed_time / 1e9
                    }
                
                test_results["results"]["cuda"] = result
                test_results["tests_completed"].append("cuda")
            else:
                test_results["results"]["cuda"] = {
                    "status": "not_available",
                    "description": "CUDA is not available on this system"
                }
                test_results["tests_failed"].append("cuda")
        except Exception as e:
            test_results["results"]["cuda"] = {
                "status": "failed",
                "error": str(e)
            }
            test_results["tests_failed"].append("cuda")
    
    # Other tests can be added here
    
    return test_results

def recommend_hardware(model_name: str, model_type: str = "llm", include_available_only: bool = True) -> Dict[str, Any]:
    """
    Recommend hardware accelerators for a specific model
    
    Args:
        model_name: Name of the model
        model_type: Type of the model
        include_available_only: Include only available hardware accelerators
        
    Returns:
        Dictionary with hardware recommendations
    """
    # Get available hardware
    hardware_info = get_hardware_info()
    available_accelerators = [acc["type"] for acc in hardware_info["available_accelerators"]]
    
    recommendations = {
        "model_name": model_name,
        "model_type": model_type,
        "recommended_hardware": [],
        "recommended_settings": {},
    }
    
    # Default recommendations
    if model_type == "llm":
        recommendations["recommended_hardware"] = ["cuda", "rocm", "cpu"]
        recommendations["recommended_settings"] = {
            "cuda": {
                "description": "Best performance for most LLMs",
                "batch_size": 1,
                "dtype": "float16",
                "device_map": "auto"
            },
            "rocm": {
                "description": "Good performance on AMD GPUs",
                "batch_size": 1,
                "dtype": "float16",
                "device_map": "auto"
            },
            "cpu": {
                "description": "Fallback option, slower but works everywhere",
                "batch_size": 1,
                "dtype": "float32",
                "num_threads": hardware_info["available_accelerators"][0]["logical_cores"] if "cpu" in available_accelerators else 4
            }
        }
    elif model_type == "vision":
        recommendations["recommended_hardware"] = ["cuda", "rocm", "cpu"]
        recommendations["recommended_settings"] = {
            "cuda": {
                "description": "Best performance for vision models",
                "batch_size": 4,
                "dtype": "float16"
            },
            "cpu": {
                "description": "Fallback option for vision models",
                "batch_size": 1,
                "dtype": "float32"
            }
        }
    elif model_type == "audio":
        recommendations["recommended_hardware"] = ["cuda", "cpu"]
        recommendations["recommended_settings"] = {
            "cuda": {
                "description": "Best performance for audio processing",
                "batch_size": 1,
                "dtype": "float16"
            },
            "cpu": {
                "description": "Acceptable for shorter audio clips",
                "batch_size": 1,
                "dtype": "float32"
            }
        }
    elif model_type == "multimodal":
        recommendations["recommended_hardware"] = ["cuda", "cpu"]
        recommendations["recommended_settings"] = {
            "cuda": {
                "description": "Recommended for multimodal models",
                "batch_size": 1,
                "dtype": "float16"
            },
            "cpu": {
                "description": "May be very slow for multimodal models",
                "batch_size": 1,
                "dtype": "float32"
            }
        }
    
    # Filter by available hardware if requested
    if include_available_only:
        for hw_type in list(recommendations["recommended_settings"].keys()):
            if hw_type not in available_accelerators:
                del recommendations["recommended_settings"][hw_type]
        
        recommendations["recommended_hardware"] = [hw for hw in recommendations["recommended_hardware"] if hw in available_accelerators]
    
    # Add model-specific recommendations based on name
    if "llama" in model_name.lower():
        if "cuda" in recommendations["recommended_settings"]:
            recommendations["recommended_settings"]["cuda"]["quantization"] = "4bit"
        if "cpu" in recommendations["recommended_settings"]:
            recommendations["recommended_settings"]["cpu"]["quantization"] = "8bit"
    
    # Get memory requirements based on model name
    model_size_match = None
    import re
    size_match = re.search(r'(\d+)b', model_name.lower())
    if size_match:
        model_size_match = int(size_match.group(1))
    
    if model_size_match:
        recommendations["model_size_billions"] = model_size_match
        
        # Add memory requirements
        memory_required = model_size_match * 4  # Roughly 4GB per billion parameters for fp16
        
        recommendations["memory_required_gb"] = {
            "fp32": model_size_match * 8,
            "fp16": memory_required,
            "int8": model_size_match * 2,
            "int4": model_size_match
        }
        
        # Check if available hardware meets requirements
        for acc in hardware_info["available_accelerators"]:
            if acc["type"] == "cuda" and "devices" in acc:
                for device in acc["devices"]:
                    device_memory_gb = device.get("total_memory_mb", 0) / 1024
                    can_run = {
                        "fp32": device_memory_gb >= recommendations["memory_required_gb"]["fp32"],
                        "fp16": device_memory_gb >= recommendations["memory_required_gb"]["fp16"],
                        "int8": device_memory_gb >= recommendations["memory_required_gb"]["int8"],
                        "int4": device_memory_gb >= recommendations["memory_required_gb"]["int4"]
                    }
                    
                    # Add to recommendations
                    if "cuda" in recommendations["recommended_settings"]:
                        if "devices" not in recommendations["recommended_settings"]["cuda"]:
                            recommendations["recommended_settings"]["cuda"]["devices"] = {}
                        
                        recommendations["recommended_settings"]["cuda"]["devices"][device["name"]] = {
                            "memory_gb": device_memory_gb,
                            "can_run": can_run,
                            "recommended_format": "int4" if can_run["int4"] else (
                                "int8" if can_run["int8"] else (
                                    "fp16" if can_run["fp16"] else "not_recommended"
                                )
                            )
                        }
    
    return recommendations
