"""
Full MCP Tools Integration for IPFS Accelerate

This module defines comprehensive MCP tools that expose all features from ipfs_accelerate_py,
including model inference, API multiplexing, and task management.
"""

import os
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("ipfs_accelerate_mcp.tools.accelerate")

# Store the IPFS Accelerate instance for use by the tools
_ipfs_instance = None

def set_ipfs_instance(ipfs_instance) -> None:
    """
    Set the IPFS Accelerate instance
    
    Args:
        ipfs_instance: IPFS Accelerate instance
    """
    global _ipfs_instance
    _ipfs_instance = ipfs_instance
    logger.info(f"IPFS Accelerate instance set: {ipfs_instance}")

def register_tools(mcp):
    """Register comprehensive tools with the MCP server"""
    
    # Configuration tools
    @mcp.tool()
    def config_get(section: str, key: str, default_value: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            default_value: Default value if not found
            
        Returns:
            The configuration value
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'config'):
                config = _ipfs_instance.config()
                value = config.get(section, key, default_value)
                return {
                    "section": section,
                    "key": key,
                    "value": value,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "section": section,
                    "key": key,
                    "value": default_value,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in config_get: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def config_set(section: str, key: str, value: Any) -> Dict[str, Any]:
        """
        Set a configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
            
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'config'):
                config = _ipfs_instance.config()
                config.set(section, key, value)
                return {
                    "section": section,
                    "key": key,
                    "value": value,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "section": section,
                    "key": key,
                    "value": value,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in config_set: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
            
    @mcp.tool()
    def config_save() -> Dict[str, Any]:
        """
        Save current configuration to file
        
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'config'):
                config = _ipfs_instance.config()
                config.save()
                return {
                    "success": True,
                    "message": "Configuration saved"
                }
            else:
                # Fall back to mock implementation
                return {
                    "success": True,
                    "message": "Configuration saved (mock)",
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in config_save: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    # Backend management tools
    @mcp.tool()
    def backend_start_container(name: str, image: str, ports: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Start a container with the specified image
        
        Args:
            name: Container name
            image: Docker image name
            ports: Port mappings (host:container)
            
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'backends'):
                backends = _ipfs_instance.backends()
                result = backends.start_container(name, image, ports)
                return {
                    "container_name": name,
                    "image": image,
                    "ports": ports,
                    "result": result,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "container_name": name,
                    "image": image,
                    "ports": ports,
                    "container_id": f"mock-{name}-{int(time.time())}",
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in backend_start_container: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def backend_stop_container(name: str) -> Dict[str, Any]:
        """
        Stop a running container
        
        Args:
            name: Container name
            
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'backends'):
                backends = _ipfs_instance.backends()
                result = backends.stop_container(name)
                return {
                    "container_name": name,
                    "result": result,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "container_name": name,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in backend_stop_container: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def backend_list_containers() -> Dict[str, Any]:
        """
        List all containers
        
        Returns:
            List of containers
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'backends'):
                backends = _ipfs_instance.backends()
                containers = backends.list_containers()
                return {
                    "containers": containers,
                    "count": len(containers),
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "containers": [],
                    "count": 0,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in backend_list_containers: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def backend_docker_tunnel(container_name: str, host_port: int, container_port: int) -> Dict[str, Any]:
        """
        Create a tunnel to a container port
        
        Args:
            container_name: Container name
            host_port: Host port
            container_port: Container port
            
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'backends'):
                backends = _ipfs_instance.backends()
                tunnel = backends.docker_tunnel(container_name, host_port, container_port)
                return {
                    "container_name": container_name,
                    "host_port": host_port,
                    "container_port": container_port,
                    "tunnel": str(tunnel),
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "container_name": container_name,
                    "host_port": host_port,
                    "container_port": container_port,
                    "tunnel": f"mock-tunnel-{container_name}-{host_port}-{container_port}",
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in backend_docker_tunnel: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def backend_list_marketplace_images() -> Dict[str, Any]:
        """
        List available marketplace images
        
        Returns:
            List of marketplace images
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'backends'):
                backends = _ipfs_instance.backends()
                images = backends.list_marketplace_images()
                return {
                    "images": images,
                    "count": len(images),
                    "success": True
                }
            else:
                # Fall back to mock implementation
                mock_images = [
                    {"name": "ipfs/kubo", "description": "IPFS Kubo implementation"},
                    {"name": "ipfs/ipfs-cluster", "description": "IPFS Cluster implementation"},
                    {"name": "textile/powergate", "description": "Powergate implementation"}
                ]
                return {
                    "images": mock_images,
                    "count": len(mock_images),
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in backend_list_marketplace_images: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    # Core IPFS operations
    @mcp.tool()
    def ipfs_add_directory(path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Add a directory to IPFS
        
        Args:
            path: Path to the directory
            recursive: Whether to add recursively
            
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'ipfs_accelerate'):
                result = _ipfs_instance.ipfs_accelerate.add_directory(path, recursive)
                return {
                    "path": path,
                    "recursive": recursive,
                    "result": result,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                mock_cid = f"QmMock{int(time.time())}"
                return {
                    "path": path,
                    "recursive": recursive,
                    "cid": mock_cid,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in ipfs_add_directory: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def ipfs_get_directory(cid: str, output_path: str) -> Dict[str, Any]:
        """
        Get a directory from IPFS
        
        Args:
            cid: Content ID
            output_path: Output directory path
            
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'ipfs_accelerate'):
                _ipfs_instance.ipfs_accelerate.get_directory(cid, output_path)
                return {
                    "cid": cid,
                    "output_path": output_path,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join(output_path, "mock-file.txt"), "w") as f:
                    f.write(f"Mock content for {cid}")
                return {
                    "cid": cid,
                    "output_path": output_path,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in ipfs_get_directory: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def ipfs_cid_exists(cid: str) -> Dict[str, Any]:
        """
        Check if a CID exists in IPFS
        
        Args:
            cid: Content ID
            
        Returns:
            Whether the CID exists
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'ipfs_accelerate'):
                exists = _ipfs_instance.ipfs_accelerate.cid_exists(cid)
                return {
                    "cid": cid,
                    "exists": exists,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "cid": cid,
                    "exists": True,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in ipfs_cid_exists: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def ipfs_get_cid_metadata(cid: str) -> Dict[str, Any]:
        """
        Get metadata for a CID
        
        Args:
            cid: Content ID
            
        Returns:
            CID metadata
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'ipfs_accelerate'):
                metadata = _ipfs_instance.ipfs_accelerate.get_cid_metadata(cid)
                return {
                    "cid": cid,
                    "metadata": metadata,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                mock_metadata = {
                    "size": 1024,
                    "blocks": 1,
                    "created": time.time()
                }
                return {
                    "cid": cid,
                    "metadata": mock_metadata,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in ipfs_get_cid_metadata: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def ipfs_load_checkpoint_and_dispatch(cid: str, 
                                          destination: str = "local",
                                          hardware: str = "cpu",
                                          **kwargs) -> Dict[str, Any]:
        """
        Load a checkpoint from IPFS and dispatch to the specified destination
        
        Args:
            cid: Content ID of the checkpoint
            destination: Destination for the checkpoint (local, remote, etc.)
            hardware: Hardware to use for inference
            **kwargs: Additional arguments
            
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'load_checkpoint_and_dispatch'):
                result = _ipfs_instance.load_checkpoint_and_dispatch(cid, destination, hardware, **kwargs)
                return {
                    "cid": cid,
                    "destination": destination,
                    "hardware": hardware,
                    "result": result,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "cid": cid,
                    "destination": destination,
                    "hardware": hardware,
                    "model_path": f"/tmp/models/{cid}",
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in ipfs_load_checkpoint_and_dispatch: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    # API multiplexing operations - these integrate with the API multiplexing tools we already defined
    
    @mcp.tool()
    def api_multiplexing_roundrobin_request(provider: str, 
                                          prompt: str,
                                          model: Optional[str] = None,
                                          temperature: float = 0.7,
                                          max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Make an API request using round-robin multiplexing
        
        Args:
            provider: API provider name
            prompt: Prompt text
            model: Model name (provider-specific)
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            
        Returns:
            API response
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'api_multiplexing'):
                multiplexer = _ipfs_instance.api_multiplexing.get_multiplexer(provider)
                response = multiplexer.request_roundrobin(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "provider": provider,
                    "model": model,
                    "response": response,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                mock_response = {
                    "text": f"Mock response from {provider} for: {prompt[:30]}...",
                    "tokens": len(prompt.split()) * 2,
                    "model": model or "default-model"
                }
                return {
                    "provider": provider,
                    "model": model,
                    "response": mock_response,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in api_multiplexing_roundrobin_request: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def api_multiplexing_leastloaded_request(provider: str, 
                                           prompt: str,
                                           model: Optional[str] = None,
                                           temperature: float = 0.7,
                                           max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Make an API request using least-loaded multiplexing
        
        Args:
            provider: API provider name
            prompt: Prompt text
            model: Model name (provider-specific)
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            
        Returns:
            API response
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'api_multiplexing'):
                multiplexer = _ipfs_instance.api_multiplexing.get_multiplexer(provider)
                response = multiplexer.request_leastloaded(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "provider": provider,
                    "model": model,
                    "response": response,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                mock_response = {
                    "text": f"Mock response from {provider} for: {prompt[:30]}...",
                    "tokens": len(prompt.split()) * 2,
                    "model": model or "default-model"
                }
                return {
                    "provider": provider,
                    "model": model,
                    "response": mock_response,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in api_multiplexing_leastloaded_request: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    # Quantization tools
    @mcp.tool()
    def quantize_model(model_path: str, 
                      output_path: str,
                      bits: int = 4,
                      method: str = "dynamic", 
                      device: str = "cpu") -> Dict[str, Any]:
        """
        Quantize a model to reduced precision
        
        Args:
            model_path: Path to the model
            output_path: Path to save the quantized model
            bits: Quantization bit width (4, 8, etc.)
            method: Quantization method (dynamic, static, etc.)
            device: Device to use for quantization
            
        Returns:
            Result of the operation
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'quantization'):
                quantizer = _ipfs_instance.quantization.get_quantizer(bits, method)
                result = quantizer.quantize(model_path, output_path, device=device)
                return {
                    "model_path": model_path,
                    "output_path": output_path,
                    "bits": bits,
                    "method": method,
                    "device": device,
                    "result": result,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "model_path": model_path,
                    "output_path": output_path,
                    "bits": bits,
                    "method": method,
                    "device": device,
                    "compression_ratio": 1.0 / (32.0 / bits),
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in quantize_model: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    # Hardware detection and acceleration tools
    @mcp.tool()
    def get_hardware_capabilities() -> Dict[str, Any]:
        """
        Get detailed hardware capabilities of the system
        
        Returns:
            Hardware capabilities
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'hardware'):
                hardware_detector = _ipfs_instance.hardware.get_detector()
                capabilities = hardware_detector.get_capabilities()
                return {
                    "capabilities": capabilities,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                mock_capabilities = {
                    "cpu": {
                        "cores": 4,
                        "threads": 8,
                        "architecture": "x86_64",
                        "simd": ["SSE4.2", "AVX2"]
                    },
                    "gpu": {
                        "available": False,
                        "devices": []
                    },
                    "memory": {
                        "total": 8 * 1024 * 1024 * 1024,  # 8GB in bytes
                        "available": 4 * 1024 * 1024 * 1024  # 4GB in bytes
                    },
                    "disk": {
                        "total": 100 * 1024 * 1024 * 1024,  # 100GB in bytes
                        "available": 50 * 1024 * 1024 * 1024  # 50GB in bytes
                    },
                    "webgpu": {
                        "available": False,
                        "adapter": None
                    },
                    "webnn": {
                        "available": False,
                        "context": None
                    }
                }
                return {
                    "capabilities": mock_capabilities,
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in get_hardware_capabilities: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def get_optimal_hardware_for_model(model_name: str, 
                                      model_type: str = None,
                                      min_memory: int = None) -> Dict[str, Any]:
        """
        Get the optimal hardware for a given model
        
        Args:
            model_name: Name or path of the model
            model_type: Type of the model (transformer, cnn, etc.)
            min_memory: Minimum required memory in MB
            
        Returns:
            Recommended hardware configuration
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'hardware'):
                hardware_selector = _ipfs_instance.hardware.get_selector()
                result = hardware_selector.select_optimal(model_name, model_type, min_memory)
                return {
                    "model_name": model_name,
                    "recommendation": result,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                return {
                    "model_name": model_name,
                    "recommendation": {
                        "device": "cpu",
                        "precision": "fp32",
                        "batch_size": 8,
                        "reason": "No accelerators detected"
                    },
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in get_optimal_hardware_for_model: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    # Load balancing and throughput optimization
    @mcp.tool()
    def create_throughput_optimizer(model_name: str, 
                                   max_batch_size: int = 32,
                                   num_workers: int = None,
                                   strategy: str = "adaptive") -> Dict[str, Any]:
        """
        Create a throughput optimizer for a model
        
        Args:
            model_name: Name or path of the model
            max_batch_size: Maximum batch size
            num_workers: Number of workers (None for auto)
            strategy: Optimization strategy (adaptive, static)
            
        Returns:
            Optimizer configuration
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'throughput'):
                optimizer = _ipfs_instance.throughput.create_optimizer(
                    model_name, 
                    max_batch_size=max_batch_size,
                    num_workers=num_workers,
                    strategy=strategy
                )
                return {
                    "model_name": model_name,
                    "optimizer_id": optimizer.id,
                    "config": optimizer.get_config(),
                    "success": True
                }
            else:
                # Fall back to mock implementation
                optimizer_id = f"optimizer-{model_name}-{int(time.time())}"
                return {
                    "model_name": model_name,
                    "optimizer_id": optimizer_id,
                    "config": {
                        "max_batch_size": max_batch_size,
                        "num_workers": num_workers or 4,
                        "strategy": strategy
                    },
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in create_throughput_optimizer: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    @mcp.tool()
    def throughput_benchmark(model_name: str,
                           batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
                           sequence_lengths: List[int] = [128],
                           warmup_runs: int = 3,
                           benchmark_runs: int = 10,
                           device: str = "cpu") -> Dict[str, Any]:
        """
        Run a throughput benchmark for a model
        
        Args:
            model_name: Name or path of the model
            batch_sizes: Batch sizes to test
            sequence_lengths: Sequence lengths to test
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            device: Device to use
            
        Returns:
            Benchmark results
        """
        global _ipfs_instance
        start_time = time.time()
        
        try:
            if _ipfs_instance and hasattr(_ipfs_instance, 'benchmark'):
                benchmark = _ipfs_instance.benchmark.create_benchmark(
                    model_name,
                    batch_sizes=batch_sizes,
                    sequence_lengths=sequence_lengths,
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                    device=device
                )
                results = benchmark.run()
                return {
                    "model_name": model_name,
                    "results": results,
                    "success": True
                }
            else:
                # Fall back to mock implementation
                mock_results = []
                for batch_size in batch_sizes:
                    for seq_len in sequence_lengths:
                        # Generate realistic latency based on batch size and seq length
                        base_latency = 10.0  # 10ms base latency
                        batch_factor = batch_size / 4.0  # 1x at batch_size=4
                        seq_factor = seq_len / 128.0  # 1x at seq_len=128
                        latency = base_latency * batch_factor * seq_factor
                        throughput = batch_size / (latency / 1000.0)  # items/sec
                        
                        mock_results.append({
                            "batch_size": batch_size,
                            "sequence_length": seq_len,
                            "latency_ms": latency,
                            "throughput": throughput,
                            "device": device
                        })
                
                return {
                    "model_name": model_name,
                    "results": mock_results,
                    "optimal_batch_size": max(batch_sizes) // 2,  # A reasonable guess
                    "success": True,
                    "mock": True
                }
                
        except Exception as e:
            logger.error(f"Error in throughput_benchmark: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}