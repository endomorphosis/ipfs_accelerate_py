"""
Multi-GPU Utility Functions for model loading and device mapping.

This module provides high-level functions for:
1. Loading models with custom device mapping
2. Configuring tensor parallelism
3. Setting up container-based GPU deployment
4. Creating optimized pipelines for multi-GPU inference
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# Add repository root to path to import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'ipfs_accelerate_py'))

# Import device mapper
from utils.device_mapper import DeviceMapper

# Setup logger
logger = logging.getLogger(__name__)

def load_model_with_device_map(
    model_id: str,
    strategy: str = "auto",
    devices: Optional[List[str]] = None,
    use_auth_token: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
) -> Tuple[Any, Dict[str, str]]:
    """
    Load a Hugging Face model with custom device mapping.
    
    Args:
        model_id: The Hugging Face model ID or local path
        strategy: Device mapping strategy ('auto', 'balanced', 'sequential')
        devices: List of specific devices to use (e.g., ['cuda:0', 'cuda:1'])
        use_auth_token: Hugging Face authentication token
        trust_remote_code: Whether to trust remote code in the model
        **kwargs: Additional arguments for model loading
    
    Returns:
        Tuple of (model, device_map)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

        # Set up device mapper
        mapper = DeviceMapper()
        
        # Create device map
        device_map = mapper.create_device_map(model_id, strategy, devices)
        
        # Print device map information
        logger.info(f"Loading model {model_id} with device map strategy: {strategy}")
        logger.info(f"Device map: {json.dumps(device_map, indent=2)}")
        
        # Check the model type to determine correct loading function
        if "t5" in model_id.lower() or "bart" in model_id.lower() or "mt5" in model_id.lower():
            model_class = AutoModelForSeq2SeqLM
        else:
            model_class = AutoModelForCausalLM
        
        # Load model with device map
        model = model_class.from_pretrained(
            model_id,
            device_map=device_map,
            use_auth_token=use_auth_token,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        return model, device_map
    
    except Exception as e:
        logger.error(f"Failed to load model with device map: {str(e)}")
        raise

def load_model_with_tensor_parallel(
    model_id: str,
    tensor_parallel_size: Optional[int] = None,
    devices: Optional[List[str]] = None,
    **kwargs
) -> Any:
    """
    Load a model with tensor parallelism (for supported backends like VLLM).
    
    Args:
        model_id: The model ID or local path
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        devices: Specific devices to use (e.g., ['cuda:0', 'cuda:1'])
        **kwargs: Additional arguments for model loading
    
    Returns:
        Loaded model with tensor parallelism configured
    """
    try:
        # Import VLLM if available
        try:
            from vllm import LLM
            vllm_available = True
        except ImportError:
            vllm_available = False
            logger.warning("VLLM not available. Falling back to standard PyTorch.")
        
        # Set up device mapper
        mapper = DeviceMapper()
        
        # Get tensor parallel configuration
        config = mapper.get_tensor_parallel_config(model_id, devices)
        
        # Override tensor_parallel_size if provided
        if tensor_parallel_size is not None:
            config["tensor_parallel_size"] = tensor_parallel_size
        
        logger.info(f"Loading model {model_id} with tensor parallel config: {json.dumps(config, indent=2)}")
        
        # Load model with tensor parallelism
        if vllm_available:
            model = LLM(
                model=model_id,
                tensor_parallel_size=config["tensor_parallel_size"],
                gpu_ids=config["gpu_ids"],
                max_parallel_loading_workers=config["max_parallel_loading_workers"],
                **kwargs
            )
        else:
            # Fallback to standard model loading with device map
            model, _ = load_model_with_device_map(
                model_id=model_id,
                strategy="balanced",
                devices=devices,
                **kwargs
            )
            
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model with tensor parallelism: {str(e)}")
        raise

def get_container_gpu_config(devices: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get GPU configuration for container deployment.
    
    Args:
        devices: List of specific devices to use (e.g., ['cuda:0', 'cuda:1'])
    
    Returns:
        Dictionary with container configuration for GPUs
    """
    # Set up device mapper
    mapper = DeviceMapper()
    
    # Get Docker GPU arguments
    gpu_arg, env_vars = mapper.get_docker_gpu_args(devices)
    
    # Create container configuration
    container_config = {
        "gpu_arg": gpu_arg,
        "environment": env_vars,
        "devices": devices or [d for d in mapper.available_devices if d != "cpu"]
    }
    
    return container_config

def create_optimized_pipeline(
    model_id: str,
    strategy: str = "auto",
    devices: Optional[List[str]] = None,
    pipeline_type: str = "text-generation",
    batch_size: int = 1,
    **kwargs
) -> Any:
    """
    Create an optimized pipeline for multi-GPU inference.
    
    Args:
        model_id: The Hugging Face model ID or local path
        strategy: Device mapping strategy ('auto', 'balanced', 'sequential')
        devices: List of specific devices to use (e.g., ['cuda:0', 'cuda:1'])
        pipeline_type: The pipeline type (e.g., 'text-generation', 'summarization')
        batch_size: Batch size for inference
        **kwargs: Additional arguments for pipeline creation
    
    Returns:
        Pipeline object with optimized device configuration
    """
    try:
        from transformers import pipeline
        
        # Set up device mapper
        mapper = DeviceMapper()
        
        # Create device map
        device_map = mapper.create_device_map(model_id, strategy, devices)
        
        # Load the pipeline with device map
        pipe = pipeline(
            pipeline_type,
            model=model_id,
            device_map=device_map,
            batch_size=batch_size,
            **kwargs
        )
        
        return pipe
    
    except Exception as e:
        logger.error(f"Failed to create optimized pipeline: {str(e)}")
        raise

def detect_optimal_device_configuration(model_id: str) -> Dict[str, Any]:
    """
    Detect the optimal device configuration for a model.
    
    Args:
        model_id: The Hugging Face model ID or local path
    
    Returns:
        Dictionary with optimal configuration recommendations
    """
    # Set up device mapper
    mapper = DeviceMapper()
    
    # Get model memory requirements
    memory_req = mapper.estimate_model_memory(model_id)
    
    # Detect available hardware
    hardware = mapper.device_info
    
    # Make recommendations
    recommendations = {
        "model_id": model_id,
        "memory_requirements": memory_req,
        "available_hardware": hardware,
        "recommendations": {}
    }
    
    # Single GPU recommendation
    if hardware["cuda"]["count"] == 1 or hardware["rocm"]["count"] == 1:
        # Check if model fits on single GPU
        device_type = "cuda" if hardware["cuda"]["count"] > 0 else "rocm"
        device_id = 0
        device_mem = None
        
        if device_type == "cuda":
            device_mem = hardware["cuda"]["devices"][0]["memory"]
        else:
            device_mem = hardware["rocm"]["devices"][0]["memory"]
        
        if device_mem >= memory_req["total"]:
            recommendations["recommendations"]["single_gpu"] = {
                "feasible": True,
                "device": f"{device_type}:{device_id}",
                "strategy": "none",
                "reason": f"Model fits on single {device_type.upper()} GPU"
            }
        else:
            recommendations["recommendations"]["single_gpu"] = {
                "feasible": False,
                "reason": f"Model requires {memory_req['total']:.2f}GB but {device_type.upper()} has only {device_mem:.2f}GB"
            }
    
    # Multi-GPU recommendation
    multi_gpu_count = hardware["cuda"]["count"] + hardware["rocm"]["count"]
    if multi_gpu_count > 1:
        # Calculate if model can be distributed
        total_available_mem = 0
        
        for device_type in ["cuda", "rocm"]:
            if hardware[device_type]["count"] > 0:
                for device in hardware[device_type]["devices"]:
                    total_available_mem += device["memory"]
        
        if total_available_mem >= memory_req["total"]:
            # Determine best strategy
            if memory_req["total"] > total_available_mem * 0.7:
                strategy = "balanced"
            else:
                strategy = "auto"
            
            recommendations["recommendations"]["multi_gpu"] = {
                "feasible": True,
                "strategy": strategy,
                "device_count": multi_gpu_count,
                "reason": f"Model can be distributed across {multi_gpu_count} GPUs with {strategy} strategy"
            }
        else:
            recommendations["recommendations"]["multi_gpu"] = {
                "feasible": False,
                "reason": f"Model requires {memory_req['total']:.2f}GB but total available GPU memory is only {total_available_mem:.2f}GB"
            }
    
    # CPU fallback recommendation
    recommendations["recommendations"]["cpu_fallback"] = {
        "feasible": True,
        "device": "cpu",
        "reason": "CPU always works but will be slower"
    }
    
    # Overall recommendation
    if multi_gpu_count > 1 and recommendations["recommendations"]["multi_gpu"]["feasible"]:
        recommendations["recommended_approach"] = "multi_gpu"
    elif (hardware["cuda"]["count"] > 0 or hardware["rocm"]["count"] > 0) and recommendations["recommendations"].get("single_gpu", {}).get("feasible", False):
        recommendations["recommended_approach"] = "single_gpu"
    else:
        recommendations["recommended_approach"] = "cpu_fallback"
    
    return recommendations

# Example usage demonstration
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Multi-GPU Utility Demo")
    parser.add_argument("--model", type=str, default="gpt2", help="Model ID to use")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "balanced", "sequential"], help="Device mapping strategy")
    parser.add_argument("--devices", type=str, nargs="+", help="Specific devices to use (e.g., cuda:0 cuda:1)")
    parser.add_argument("--detect", action="store_true", help="Run device detection only")
    parser.add_argument("--container", action="store_true", help="Show container configuration")
    args = parser.parse_args()
    
    # Run device detection
    mapper = DeviceMapper()
    hardware = mapper.device_info
    print(f"Detected hardware: {json.dumps(hardware, indent=2)}")
    
    # If detection only, exit
    if args.detect:
        sys.exit(0)
    
    # Determine optimal configuration
    if args.container:
        container_config = get_container_gpu_config(args.devices)
        print(f"Container configuration: {json.dumps(container_config, indent=2)}")
        
    # Get optimal configuration recommendation
    recommendations = detect_optimal_device_configuration(args.model)
    print(f"Recommended configuration: {json.dumps(recommendations, indent=2)}")
    
    # Try to load model if we have transformers
    try:
        # Import required libraries
        from transformers import AutoModel
        import torch
        
        # Load model with device map
        print(f"Loading model {args.model} with strategy {args.strategy}...")
        model, device_map = load_model_with_device_map(
            model_id=args.model,
            strategy=args.strategy,
            devices=args.devices
        )
        
        # Print model information
        print(f"Successfully loaded model with device map: {json.dumps(device_map, indent=2)}")
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model size: {total_params:,} parameters")
        
    except ImportError:
        print("Could not import transformers or torch. Model loading demonstration skipped.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")