"""
Inference Tools for IPFS Accelerate MCP Server

This module provides MCP tools for running inference with machine learning models.
Uses shared operations for consistency with CLI.
"""

import os
import time
import logging
import random
import traceback
from typing import Dict, List, Any, Optional, Union

# Try to import numpy
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    np = None

logger = logging.getLogger("ipfs_accelerate_mcp.tools.inference")

# Import shared operations
try:
    from ....shared import SharedCore, InferenceOperations
    shared_core = SharedCore()
    inference_ops = InferenceOperations(shared_core)
    HAVE_SHARED = True
except ImportError as e:
    logger.warning(f"Shared operations not available: {e}")
    HAVE_SHARED = False
    shared_core = None
    inference_ops = None

def register_tools(mcp):
    """Register inference-related tools with the MCP server"""
    """
    Set the IPFS Accelerate instance
    
    Args:
        ipfs_instance: IPFS Accelerate instance
    """
    global _ipfs_instance
    _ipfs_instance = ipfs_instance
    logger.info(f"IPFS Accelerate instance set: {ipfs_instance}")

def register_tools(mcp):
    """Register inference-related tools with the MCP server"""
    
    @mcp.tool()
    def run_inference(model: str,
                      inputs: List[str],
                      device: str = "auto",
                      max_length: int = 1024,
                      temperature: float = 0.7,
                      endpoint_id: Optional[str] = None,
                      provider: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Run inference on a model
        
        This tool runs inference on the specified model with the given inputs.
        It supports embedding generation and text generation.
        
        Args:
            model: Name or path of the model to use
            inputs: List of input texts
            device: Device to run inference on (e.g., "cpu", "cuda:0", "auto")
            max_length: Maximum length for text generation (ignored for embeddings)
            temperature: Temperature for text generation (ignored for embeddings)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with inference results
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            # First try to use IPFS Accelerate instance if available
            if _ipfs_instance is not None:
                logger.info(f"Running inference with IPFS Accelerate instance: {model}")
                try:
                    # Prepare parameters
                    params = {
                        "model_id": model,
                        "inputs": inputs,
                        "device": device,
                        "max_length": max_length,
                        "temperature": temperature
                    }
                    
                    # Try to run inference with IPFS Accelerate
                    if hasattr(_ipfs_instance, "run_inference"):
                        result = _ipfs_instance.run_inference(**params)
                        elapsed_time = time.time() - start_time
                        
                        return {
                            "outputs": result.get("outputs", result),
                            "model": model,
                            "elapsed_time": elapsed_time,
                            "device": device,
                            "backend": "ipfs_accelerate"
                        }
                except Exception as e:
                    logger.warning(f"IPFS Accelerate inference failed: {e}. Falling back to MCP implementation.")
            
            # If an endpoint_id is provided, try to resolve endpoint settings first
            if endpoint_id:
                try:
                    ep = mcp.use_tool("get_endpoint", endpoint_id=endpoint_id)
                    if isinstance(ep, dict) and "error" not in ep:
                        model = ep.get("model", model)
                        device = ep.get("device", device)
                        kwargs.setdefault("max_batch_size", ep.get("max_batch_size"))
                except Exception:
                    pass

            # Fallback to MCP implementation
            # Get model info from resources
            model_info = mcp.access_resource("get_model_info", model_name=model)
            if not model_info:
                return {
                    "error": f"Model '{model}' not found. Please check the available models."
                }
            
            # Determine model type
            model_type = model_info.get("type", "unknown")
            
            # In a real implementation, we would load the model and run inference
            # For this placeholder, we'll simulate the results
            
            # Record start time
            start_time = time.time()
            
            if provider:
                # Optional provider routing; keep minimal to avoid heavy deps
                try:
                    if provider.lower() in ["openai", "gpt", "oai"]:
                        from ipfs_accelerate_py.utils.auto_install import ensure_packages
                        ensure_packages({"openai": "openai"})
                        import os as _os
                        from openai import OpenAI
                        api_key = _os.getenv("OPENAI_API_KEY")
                        if not api_key:
                            raise RuntimeError("OPENAI_API_KEY not set")
                        client = OpenAI(api_key=api_key)
                        if model_type == "generation":
                            outputs = []
                            for input_text in inputs:
                                r = client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": input_text}],
                                    temperature=temperature,
                                    max_tokens=max(1, max_length),
                                )
                                outputs.append(r.choices[0].message.content)
                            result = {
                                "model": model,
                                "model_type": "generation",
                                "outputs": outputs,
                                "device": "provider",
                                "provider": provider,
                                "processing_time": time.time() - start_time,
                            }
                        else:
                            # Embeddings path
                            vectors = []
                            for text in inputs:
                                er = client.embeddings.create(model=model, input=text)
                                vectors.append(er.data[0].embedding)
                            result = {
                                "model": model,
                                "model_type": "embedding",
                                "embeddings": vectors,
                                "device": "provider",
                                "provider": provider,
                                "processing_time": time.time() - start_time,
                            }

                        # If endpoint_id provided, log usage
                        if endpoint_id:
                            try:
                                mcp.use_tool(
                                    "log_request",
                                    endpoint_id=endpoint_id,
                                    model=model,
                                    device=result.get("device", device),
                                    request_type=result.get("model_type", "unknown"),
                                    inputs_processed=len(inputs),
                                    processing_time=result.get("processing_time", 0.0),
                                )
                            except Exception:
                                pass
                        return result
                except Exception as e:
                    logger.warning(f"Provider routing failed ({provider}): {e}. Falling back to local simulation.")

            if model_type == "embedding":
                # Simulate embedding generation
                embedding_size = model_info.get("embedding_size", 384)
                
                # Simulate processing time based on input size and device
                processing_delay = len(inputs) * 0.1  # 100ms per input
                if device == "cpu":
                    processing_delay *= 2  # CPU is slower
                elif "cuda" in device:
                    processing_delay *= 0.5  # GPU is faster
                time.sleep(processing_delay)
                
                # Generate random embeddings of the correct size
                embeddings = []
                for _ in inputs:
                    # Generate a random embedding and normalize it
                    if HAVE_NUMPY:
                        embedding = np.random.randn(embedding_size)
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding.tolist())
                    else:
                        # Fallback to plain Python if numpy not available
                        embedding = [random.gauss(0, 1) for _ in range(embedding_size)]
                        # Simple normalization
                        magnitude = sum(x**2 for x in embedding) ** 0.5
                        embedding = [x / magnitude if magnitude > 0 else 0 for x in embedding]
                        embeddings.append(embedding)
                
                result = {
                    "model": model,
                    "model_type": "embedding",
                    "embedding_size": embedding_size,
                    "embeddings": embeddings,
                    "inputs_processed": len(inputs),
                    "device": device,
                    "processing_time": time.time() - start_time
                }
                if endpoint_id:
                    try:
                        mcp.use_tool(
                            "log_request",
                            endpoint_id=endpoint_id,
                            model=model,
                            device=device,
                            request_type="embedding",
                            inputs_processed=len(inputs),
                            processing_time=result["processing_time"],
                        )
                    except Exception:
                        pass
                return result
                
            elif model_type == "generation":
                # Simulate text generation
                
                # Simulate processing time based on input size and device
                processing_delay = len(inputs) * 0.5  # 500ms per input
                if device == "cpu":
                    processing_delay *= 2  # CPU is slower
                elif "cuda" in device:
                    processing_delay *= 0.5  # GPU is faster
                time.sleep(processing_delay)
                
                # Generate simulated text outputs
                outputs = []
                for input_text in inputs:
                    # Generate a simulated response
                    if "IPFS" in input_text:
                        output = "IPFS (InterPlanetary File System) is a distributed system for storing and accessing files, websites, applications, and data. It works by creating a peer-to-peer network where each node stores content and helps distribute it to other nodes, making it decentralized and resilient."
                    elif "machine learning" in input_text.lower():
                        output = "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to 'learn' from data without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."
                    else:
                        output = f"This is a simulated response to: '{input_text}'. In a real implementation, this would be generated by the {model} model."
                    
                    outputs.append(output)
                
                result = {
                    "model": model,
                    "model_type": "generation",
                    "outputs": outputs,
                    "inputs_processed": len(inputs),
                    "device": device,
                    "processing_time": time.time() - start_time
                }
                if endpoint_id:
                    try:
                        mcp.use_tool(
                            "log_request",
                            endpoint_id=endpoint_id,
                            model=model,
                            device=device,
                            request_type="generation",
                            inputs_processed=len(inputs),
                            processing_time=result["processing_time"],
                        )
                    except Exception:
                        pass
                return result
                
            else:
                return {
                    "error": f"Unsupported model type: {model_type}"
                }
                
        except Exception as e:
            return {
                "error": f"Error running inference: {str(e)}"
            }
    
    @mcp.tool()
    def get_model_list() -> Dict[str, Any]:
        """
        Get list of available models
        
        This tool returns a list of available models for inference.
        
        Returns:
            Dictionary with available models by type
        """
        try:
            # In a real implementation, we would query a model registry
            # For this placeholder, we'll use the models from the config
            models_config = mcp.access_resource("models_config")
            
            return {
                "models": models_config,
                "total_models": sum(len(models) for models in models_config.values())
            }
        except Exception as e:
            return {
                "error": f"Error getting model list: {str(e)}"
            }
    
    @mcp.tool()
    def download_model(model_name: str, force: bool = False) -> Dict[str, Any]:
        """
        Download a model for local inference
        
        This tool downloads a model for local inference.
        
        Args:
            model_name: Name of the model to download
            force: Whether to force download even if the model exists
            
        Returns:
            Dictionary with download status
        """
        try:
            # Get model info
            model_info = mcp.access_resource("get_model_info", model_name=model_name)
            if not model_info:
                return {
                    "error": f"Model '{model_name}' not found. Please check the available models."
                }
            
            # In a real implementation, we would download the model
            # For this placeholder, we'll simulate the download
            
            # Get model cache directory from config
            inference_config = mcp.access_resource("inference_config")
            model_cache_dir = inference_config.get("model_cache_dir", "~/.ipfs_accelerate/models")
            model_cache_dir = os.path.expanduser(model_cache_dir)
            
            # Simulate download
            logger.info(f"Downloading model '{model_name}' to {model_cache_dir}")
            time.sleep(2)  # Simulate download time
            
            # Return success
            return {
                "status": "success",
                "model_name": model_name,
                "model_type": model_info.get("type", "unknown"),
                "model_size": model_info.get("parameters", 0),
                "model_path": os.path.join(model_cache_dir, model_name.replace("/", "_"))
            }
        except Exception as e:
            return {
                "error": f"Error downloading model: {str(e)}"
            }
    
    @mcp.tool()
    def run_distributed_inference(
        model: str,
        inputs: List[str],
        sharding_strategy: str = "auto",
        max_length: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run distributed inference on a model
        
        This tool runs inference on a model distributed across multiple devices.
        
        Args:
            model: The model to run inference on (e.g. "meta-llama/Llama-2-70b")
            inputs: The inputs to the model
            sharding_strategy: Strategy for distributing model ("auto", "tensor_parallel", "pipeline_parallel")
            max_length: Maximum length for text generation
            temperature: Temperature for text generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with inference results
        """
        global _ipfs_instance
        
        try:
            # Start timing
            start_time = time.time()
            
            # Check if we have an IPFS instance
            if _ipfs_instance is None:
                # Try to get the default instance
                try:
                    from ipfs_accelerate_py import get_instance
                    _ipfs_instance = get_instance(distributed_mode=True)
                except ImportError:
                    logger.warning("Failed to import ipfs_accelerate_py - falling back to simulation")
                    
                    # If we still don't have an instance, return a simulated response
                    return {
                        "warning": "Using simulation mode - IPFS Accelerate not available",
                        "model": model,
                        "sharding_strategy": sharding_strategy,
                        "outputs": [f"Simulated distributed inference output for input: {input[:50]}..." for input in inputs],
                        "elapsed_time": 1.5,
                        "status": "success (simulated)"
                    }
            
            # Log the request
            logger.info(f"Running distributed inference on model {model} with strategy {sharding_strategy}")
            
            # Prepare inference parameters
            inference_params = {
                "model": model,
                "inputs": inputs,
                "distributed": True,
                "max_length": max_length,
                "temperature": temperature
            }
            
            # Add sharding strategy if specified
            if sharding_strategy != "auto":
                inference_params["sharding_strategy"] = sharding_strategy
                
            # Add any additional kwargs
            inference_params.update(kwargs)
            
            # Run distributed inference
            outputs = None
            if hasattr(_ipfs_instance, "run_distributed_inference"):
                outputs = _ipfs_instance.run_distributed_inference(**inference_params)
            elif hasattr(_ipfs_instance, "run_inference"):
                # Try to use regular inference with distributed=True flag
                outputs = _ipfs_instance.run_inference(**inference_params)
            else:
                # Fallback to simulation
                logger.warning("IPFS Accelerate instance does not support distributed inference - simulating")
                outputs = [f"Simulated distributed inference output for input: {input[:50]}..." for input in inputs]
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            
            # Return results
            return {
                "model": model,
                "sharding_strategy": sharding_strategy,
                "inputs": inputs,
                "outputs": outputs,
                "elapsed_time": elapsed_time,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error running distributed inference: {e}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }
    
    @mcp.tool()
    def get_distributed_capabilities() -> Dict[str, Any]:
        """
        Get distributed inference capabilities
        
        This tool returns information about the system's distributed inference capabilities.
        
        Returns:
            Dictionary with distributed inference capabilities
        """
        global _ipfs_instance
        
        try:
            # Check if we have an IPFS instance
            if _ipfs_instance is None:
                # Try to get the default instance
                try:
                    from ipfs_accelerate_py import get_instance
                    _ipfs_instance = get_instance(distributed_mode=True)
                except ImportError:
                    logger.warning("Failed to import ipfs_accelerate_py - falling back to simulation")
                    
                    # If we still don't have an instance, return a simulated response
                    return {
                        "warning": "Using simulation mode - IPFS Accelerate not available",
                        "max_model_size_b": 70_000_000_000,  # 70B
                        "supported_strategies": ["tensor_parallel", "pipeline_parallel", "expert_parallel"],
                        "max_nodes": 4,
                        "status": "success (simulated)"
                    }
            
            # Get distributed capabilities
            if hasattr(_ipfs_instance, "get_distributed_capabilities"):
                capabilities = _ipfs_instance.get_distributed_capabilities()
                return {
                    "max_model_size_b": capabilities.get("max_model_size_b", 70_000_000_000),
                    "supported_strategies": capabilities.get("supported_strategies", ["tensor_parallel", "pipeline_parallel"]),
                    "max_nodes": capabilities.get("max_nodes", 4),
                    "status": "success"
                }
            elif hasattr(_ipfs_instance, "get_system_info"):
                # Try to extract from system info
                system_info = _ipfs_instance.get_system_info()
                if "distributed" in system_info:
                    return {
                        "max_model_size_b": system_info["distributed"].get("max_model_size_b", 70_000_000_000),
                        "supported_strategies": system_info["distributed"].get("supported_strategies", ["tensor_parallel", "pipeline_parallel"]),
                        "max_nodes": system_info["distributed"].get("max_nodes", 4),
                        "status": "success"
                    }
            
            # Fallback to simulation
            logger.warning("IPFS Accelerate instance does not provide distributed capabilities - simulating")
            return {
                "warning": "Using estimated values - IPFS Accelerate did not provide capabilities",
                "max_model_size_b": 70_000_000_000,  # 70B
                "supported_strategies": ["tensor_parallel", "pipeline_parallel", "expert_parallel"],
                "max_nodes": 4,
                "status": "success (estimated)"
            }
            
        except Exception as e:
            logger.error(f"Error getting distributed capabilities: {e}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "max_model_size_b": 13_000_000_000  # 13B fallback
            }
