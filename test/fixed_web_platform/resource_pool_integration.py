#!/usr/bin/env python3
"""
Resource Pool Integration for IPFS Acceleration with WebNN/WebGPU Integration (May 2025)

This module provides integration between the IPFS acceleration framework and
the WebNN/WebGPU resource pool bridge, enabling efficient resource sharing
and optimal utilization of browser-based hardware acceleration.

Key features:
- Seamless integration with IPFS acceleration framework
- Efficient browser resource sharing across test runs
- Automatic hardware selection for optimal model performance
- Connection pooling for browser instances
- Adaptive resource scaling based on workload
- Comprehensive monitoring and metrics collection

Usage:
    from fixed_web_platform.resource_pool_integration import IPFSAccelerateWebIntegration
    
    # Create integration instance
    integration = IPFSAccelerateWebIntegration()
    
    # Get a model with browser acceleration
    model = integration.get_model("bert-base-uncased", platform="webgpu")
    
    # Run inference
    result = model.run_inference(inputs)
    
    # Get performance metrics
    metrics = integration.get_metrics()
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import resource pool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool

# Import resource pool bridge (local import to avoid circular imports)
from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration, EnhancedWebModel

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IPFSAccelerateWebIntegration:
    """
    Integration between IPFS acceleration and WebNN/WebGPU resource pool.
    
    This class provides a unified interface for accessing WebNN and WebGPU
    acceleration through the resource pool, with optimized resource management
    and efficient browser connection sharing.
    """
    
    def __init__(self, max_connections: int = 4, 
                enable_gpu: bool = True, enable_cpu: bool = True,
                browser_preferences: Dict[str, str] = None,
                adaptive_scaling: bool = True):
        """
        Initialize IPFS acceleration integration with web platform.
        
        Args:
            max_connections: Maximum number of browser connections
            enable_gpu: Whether to enable GPU acceleration
            enable_cpu: Whether to enable CPU acceleration
            browser_preferences: Dict mapping model families to preferred browsers
            adaptive_scaling: Whether to enable adaptive resource scaling
        """
        self.resource_pool = get_global_resource_pool()
        self.bridge_integration = self._get_or_create_bridge_integration(
            max_connections=max_connections,
            enable_gpu=enable_gpu,
            enable_cpu=enable_cpu,
            browser_preferences=browser_preferences,
            adaptive_scaling=adaptive_scaling
        )
        self.metrics = {
            "model_load_time": {},
            "inference_time": {},
            "memory_usage": {},
            "throughput": {},
            "latency": {},
            "batch_size": {},
            "platform_distribution": {
                "webgpu": 0,
                "webnn": 0,
                "cpu": 0
            }
        }
        self.loaded_models = {}
        
        logger.info("IPFSAccelerateWebIntegration initialized successfully")
    
    def _get_or_create_bridge_integration(self, max_connections=4, 
                                         enable_gpu=True, enable_cpu=True,
                                         browser_preferences=None,
                                         adaptive_scaling=True) -> ResourcePoolBridgeIntegration:
        """
        Get or create resource pool bridge integration.
        
        Args:
            max_connections: Maximum number of browser connections
            enable_gpu: Whether to enable GPU acceleration
            enable_cpu: Whether to enable CPU acceleration
            browser_preferences: Dict mapping model families to preferred browsers
            adaptive_scaling: Whether to enable adaptive resource scaling
            
        Returns:
            ResourcePoolBridgeIntegration instance
        """
        # Check if integration already exists in resource pool
        integration = self.resource_pool.get_resource("web_platform_integration")
        
        if integration is None:
            # Create new integration
            integration = ResourcePoolBridgeIntegration(
                max_connections=max_connections,
                enable_gpu=enable_gpu,
                enable_cpu=enable_cpu,
                headless=True,  # Always use headless for IPFS acceleration
                browser_preferences=browser_preferences or {
                    'audio': 'firefox',  # Firefox has better compute shader performance for audio
                    'vision': 'chrome',  # Chrome has good WebGPU support for vision models
                    'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
                    'text_generation': 'chrome',
                    'multimodal': 'chrome'
                },
                adaptive_scaling=adaptive_scaling
            )
            
            # Initialize integration
            integration.initialize()
            
            # Store in resource pool for reuse
            self.resource_pool.get_resource(
                "web_platform_integration", 
                constructor=lambda: integration
            )
        
        return integration
    
    def get_model(self, model_name: str, model_type: str = None, 
                 platform: str = "webgpu", batch_size: int = 1,
                 quantization: Dict[str, Any] = None,
                 optimizations: Dict[str, bool] = None) -> EnhancedWebModel:
        """
        Get a model with browser-based acceleration.
        
        Args:
            model_name: Name of the model to load
            model_type: Type of model (text, vision, audio, multimodal)
            platform: Platform to use (webgpu, webnn, or cpu)
            batch_size: Default batch size for model
            quantization: Quantization settings (bits, mixed_precision)
            optimizations: Optional optimizations to use
            
        Returns:
            EnhancedWebModel instance
        """
        # Determine model type if not specified
        if model_type is None:
            model_type = self._infer_model_type(model_name)
        
        # Determine model family for optimal browser selection
        model_family = self._determine_model_family(model_type, model_name)
        
        # Set default optimizations based on model family
        default_optimizations = self._get_default_optimizations(model_family)
        if optimizations:
            default_optimizations.update(optimizations)
        
        # Create model key for caching
        model_key = f"{model_name}:{platform}:{batch_size}"
        if quantization:
            bits = quantization.get("bits", 16)
            mixed = quantization.get("mixed_precision", False)
            model_key += f":{bits}bit{'_mixed' if mixed else ''}"
        
        # Check if model is already loaded
        if model_key in self.loaded_models:
            logger.info(f"Reusing already loaded model: {model_key}")
            return self.loaded_models[model_key]
        
        # Create hardware preferences
        hardware_preferences = {
            'priority_list': [platform, 'cpu'],
            'model_family': model_family,
            'quantization': quantization or {}
        }
        
        # Get model from bridge integration
        start_time = time.time()
        web_model = self.bridge_integration.get_model(
            model_type=model_type,
            model_name=model_name,
            hardware_preferences=hardware_preferences
        )
        load_time = time.time() - start_time
        
        # Store metrics
        self.metrics["model_load_time"][model_key] = load_time
        self.metrics["platform_distribution"][platform] += 1
        
        # Set batch size if specified
        if batch_size > 1 and hasattr(web_model, "set_max_batch_size"):
            web_model.set_max_batch_size(batch_size)
        
        # Cache model
        self.loaded_models[model_key] = web_model
        
        logger.info(f"Model {model_name} loaded with {platform} acceleration in {load_time:.2f}s")
        return web_model
    
    def _infer_model_type(self, model_name: str) -> str:
        """
        Infer model type from model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Inferred model type
        """
        model_name_lower = model_name.lower()
        
        # Text embedding models
        if any(name in model_name_lower for name in ["bert", "roberta", "albert", "distilbert", "mpnet"]):
            return "text_embedding"
        
        # Text generation models
        elif any(name in model_name_lower for name in ["gpt", "t5", "llama", "opt", "bloom", "mistral", "falcon"]):
            return "text_generation"
        
        # Vision models
        elif any(name in model_name_lower for name in ["vit", "resnet", "efficientnet", "beit", "deit", "convnext"]):
            return "vision"
        
        # Audio models
        elif any(name in model_name_lower for name in ["whisper", "wav2vec", "hubert", "mms", "clap"]):
            return "audio"
        
        # Multimodal models
        elif any(name in model_name_lower for name in ["clip", "llava", "blip", "xclip", "flamingo"]):
            return "multimodal"
        
        # Default to text_embedding if unknown
        return "text_embedding"
    
    def _determine_model_family(self, model_type: str, model_name: str) -> str:
        """
        Determine model family for optimal hardware assignment.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            
        Returns:
            Model family
        """
        # Check if model_family_classifier is available for better classification
        try:
            classifier_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "model_family_classifier.py")
            
            if os.path.exists(classifier_path):
                sys.path.append(os.path.dirname(classifier_path))
                from model_family_classifier import classify_model
                
                model_info = classify_model(model_name=model_name)
                family = model_info.get("family")
                
                if family:
                    logger.debug(f"Model {model_name} classified as {family} by model_family_classifier")
                    return family
        except (ImportError, Exception) as e:
            logger.debug(f"Error using model_family_classifier: {e}")
        
        # Map model type to family if classifier not available
        type_to_family = {
            "text_embedding": "text_embedding",
            "text_generation": "text_generation",
            "vision": "vision",
            "audio": "audio",
            "multimodal": "multimodal"
        }
        
        return type_to_family.get(model_type, model_type)
    
    def _get_default_optimizations(self, model_family: str) -> Dict[str, bool]:
        """
        Get default optimizations for a model family.
        
        Args:
            model_family: Model family
            
        Returns:
            Dict of default optimizations
        """
        # Default optimizations for all families
        default = {
            "compute_shaders": False,
            "precompile_shaders": True,  # Always beneficial
            "parallel_loading": False
        }
        
        if model_family == "audio":
            # Enable compute shader optimization for audio models (especially on Firefox)
            default["compute_shaders"] = True
        
        elif model_family == "multimodal":
            # Enable parallel loading for multimodal models
            default["parallel_loading"] = True
        
        return default
    
    def run_inference(self, model, inputs, batch_size=None):
        """
        Run inference with a model.
        
        Args:
            model: Model to use for inference
            inputs: Input data for inference
            batch_size: Optional batch size override
            
        Returns:
            Inference results
        """
        # Track inference time
        start_time = time.time()
        
        # Run inference
        result = model(inputs)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Update metrics
        model_key = model.model_id if hasattr(model, "model_id") else str(id(model))
        if model_key not in self.metrics["inference_time"]:
            self.metrics["inference_time"][model_key] = []
        
        self.metrics["inference_time"][model_key].append(inference_time)
        
        # Calculate running average of metrics
        if len(self.metrics["inference_time"][model_key]) > 10:
            # Only keep the last 10 measurements
            self.metrics["inference_time"][model_key] = self.metrics["inference_time"][model_key][-10:]
        
        # Update performance metrics if available
        if hasattr(model, "get_performance_metrics"):
            perf_metrics = model.get_performance_metrics()
            if "stats" in perf_metrics:
                stats = perf_metrics["stats"]
                self.metrics["throughput"][model_key] = stats.get("throughput", 0)
                self.metrics["latency"][model_key] = stats.get("avg_latency", 0)
                self.metrics["batch_size"][model_key] = stats.get("batch_sizes", {})
            
            if "memory_usage" in perf_metrics:
                self.metrics["memory_usage"][model_key] = perf_metrics["memory_usage"]
        
        return result
    
    def run_parallel_inference(self, model_data_pairs, batch_size=None):
        """
        Run inference on multiple models in parallel.
        
        Args:
            model_data_pairs: List of (model, input_data) pairs
            batch_size: Optional batch size override
            
        Returns:
            List of inference results in the same order
        """
        if not model_data_pairs:
            return []
        
        # Extract EnhancedWebModels for parallel execution
        web_models = []
        other_models = []
        other_inputs = []
        
        for model, inputs in model_data_pairs:
            if isinstance(model, EnhancedWebModel):
                web_models.append((model, inputs))
            else:
                other_models.append(model)
                other_inputs.append(inputs)
        
        results = []
        
        # Process WebModels using concurrent execution
        if web_models:
            # Get first model for execution with others
            first_model, first_input = web_models[0]
            other_web_models = [model for model, _ in web_models[1:]]
            
            # Run concurrent execution
            if batch_size and batch_size > 1:
                # Create batch input
                batch_inputs = [first_input] * batch_size
                batch_results = first_model.run_concurrent(batch_inputs, other_web_models)
                results.extend(batch_results)
            else:
                # Single input with concurrent models
                web_result = first_model.run_concurrent([first_input], other_web_models)
                results.append(web_result[0])
        
        # Process other models sequentially
        for model, inputs in zip(other_models, other_inputs):
            results.append(self.run_inference(model, inputs, batch_size))
        
        return results
    
    def get_metrics(self):
        """
        Get performance metrics for all models.
        
        Returns:
            Dict of performance metrics
        """
        # Get bridge integration metrics
        if self.bridge_integration:
            bridge_stats = self.bridge_integration.get_execution_stats()
            
            # Add bridge stats to metrics
            enhanced_metrics = {
                "models": self.metrics,
                "bridge": bridge_stats,
                "resource_pool": self.resource_pool.get_stats()
            }
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics()
            enhanced_metrics["aggregate"] = aggregate_metrics
            
            return enhanced_metrics
        
        return self.metrics
    
    def _calculate_aggregate_metrics(self):
        """
        Calculate aggregate metrics across all models.
        
        Returns:
            Dict of aggregate metrics
        """
        aggregate = {
            "avg_load_time": 0,
            "avg_inference_time": 0,
            "avg_throughput": 0,
            "avg_latency": 0,
            "total_models": len(self.metrics["model_load_time"]),
            "platform_distribution": self.metrics["platform_distribution"],
            "total_memory_usage": 0
        }
        
        # Calculate average load time
        if self.metrics["model_load_time"]:
            aggregate["avg_load_time"] = sum(self.metrics["model_load_time"].values()) / len(self.metrics["model_load_time"])
        
        # Calculate average inference time
        inference_times = []
        for times in self.metrics["inference_time"].values():
            inference_times.extend(times)
        
        if inference_times:
            aggregate["avg_inference_time"] = sum(inference_times) / len(inference_times)
        
        # Calculate average throughput
        if self.metrics["throughput"]:
            aggregate["avg_throughput"] = sum(self.metrics["throughput"].values()) / len(self.metrics["throughput"])
        
        # Calculate average latency
        if self.metrics["latency"]:
            aggregate["avg_latency"] = sum(self.metrics["latency"].values()) / len(self.metrics["latency"])
        
        # Calculate total memory usage
        for memory_info in self.metrics["memory_usage"].values():
            if isinstance(memory_info, dict) and "reported" in memory_info:
                aggregate["total_memory_usage"] += memory_info["reported"]
        
        return aggregate
    
    def close(self):
        """Clean up resources and close connections."""
        if self.bridge_integration:
            self.bridge_integration.close()
            logger.info("IPFSAccelerateWebIntegration closed successfully")
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.close()

class IPFSWebAccelerator:
    """
    Enhanced IPFS accelerator with WebNN/WebGPU integration.
    
    This class provides a high-level interface for accelerating IPFS models
    using WebNN and WebGPU hardware acceleration in browsers.
    """
    
    def __init__(self, db_path=None, max_connections=4, enable_gpu=True, 
                enable_cpu=True, browser_preferences=None):
        """
        Initialize IPFS Web Accelerator.
        
        Args:
            db_path: Optional path to database for storing results
            max_connections: Maximum number of browser connections
            enable_gpu: Whether to enable GPU acceleration
            enable_cpu: Whether to enable CPU acceleration
            browser_preferences: Dict mapping model families to preferred browsers
        """
        self.integration = IPFSAccelerateWebIntegration(
            max_connections=max_connections,
            enable_gpu=enable_gpu,
            enable_cpu=enable_cpu,
            browser_preferences=browser_preferences
        )
        self.db_path = db_path
        self.model_cache = {}
        
        # Database integration
        self.db_integration = self._setup_db_integration() if db_path else None
        
        logger.info("IPFSWebAccelerator initialized successfully")
    
    def _setup_db_integration(self):
        """
        Set up database integration.
        
        Returns:
            Database integration object or None if not available
        """
        try:
            # Try to import database API
            from benchmark_db_api import DatabaseAPI
            
            # Create DB integration
            db_api = DatabaseAPI(db_path=self.db_path)
            logger.info(f"Database integration initialized with DB: {self.db_path}")
            return db_api
        except ImportError:
            logger.warning("benchmark_db_api module not available. Running without database integration.")
            return None
        except Exception as e:
            logger.error(f"Error setting up database integration: {e}")
            return None
    
    def accelerate_model(self, model_name, model_type=None, platform="webgpu", 
                         quantization=None, optimizations=None):
        """
        Get accelerated model for inference.
        
        Args:
            model_name: Name of the model to accelerate
            model_type: Type of model (inferred if not provided)
            platform: Acceleration platform (webgpu, webnn, cpu)
            quantization: Quantization settings (bits, mixed_precision)
            optimizations: Additional optimizations to enable
            
        Returns:
            Accelerated model ready for inference
        """
        # Create cache key
        cache_key = f"{model_name}:{platform}"
        if quantization:
            bits = quantization.get("bits", 16)
            mixed = quantization.get("mixed_precision", False)
            cache_key += f":{bits}bit{'_mixed' if mixed else ''}"
        
        # Check cache
        if cache_key in self.model_cache:
            logger.debug(f"Using cached accelerated model for {model_name}")
            return self.model_cache[cache_key]
        
        # Get accelerated model from integration
        model = self.integration.get_model(
            model_name=model_name,
            model_type=model_type,
            platform=platform,
            quantization=quantization,
            optimizations=optimizations
        )
        
        # Cache for future use
        self.model_cache[cache_key] = model
        
        return model
    
    def run_inference(self, model_name, inputs, model_type=None, platform="webgpu", 
                     quantization=None, optimizations=None, store_results=True):
        """
        Run inference with accelerated model.
        
        Args:
            model_name: Name of the model to use
            inputs: Input data for inference
            model_type: Type of model (inferred if not provided)
            platform: Acceleration platform (webgpu, webnn, cpu)
            quantization: Quantization settings (bits, mixed_precision)
            optimizations: Additional optimizations to enable
            store_results: Whether to store results in database
            
        Returns:
            Inference results
        """
        # Get accelerated model
        model = self.accelerate_model(
            model_name=model_name,
            model_type=model_type,
            platform=platform,
            quantization=quantization,
            optimizations=optimizations
        )
        
        # Run inference
        start_time = time.time()
        result = self.integration.run_inference(model, inputs)
        inference_time = time.time() - start_time
        
        # Store results in database if enabled
        if store_results and self.db_integration:
            try:
                # Get performance metrics
                metrics = model.get_performance_metrics() if hasattr(model, "get_performance_metrics") else {}
                
                # Prepare metadata
                metadata = {
                    "model_name": model_name,
                    "model_type": model_type or self.integration._infer_model_type(model_name),
                    "platform": platform,
                    "timestamp": time.time(),
                    "inference_time": inference_time,
                    "hardware_type": platform,
                    "quantization": quantization,
                    "optimizations": optimizations,
                    "performance_metrics": metrics
                }
                
                # Store in database
                self.db_integration.store_inference_result(result, metadata)
                logger.debug(f"Stored inference result in database for {model_name}")
            except Exception as e:
                logger.error(f"Error storing results in database: {e}")
        
        return result
    
    def run_batch_inference(self, model_name, batch_inputs, model_type=None, 
                           platform="webgpu", batch_size=None, store_results=True):
        """
        Run batch inference with accelerated model.
        
        Args:
            model_name: Name of the model to use
            batch_inputs: List of input data for batch inference
            model_type: Type of model (inferred if not provided)
            platform: Acceleration platform (webgpu, webnn, cpu)
            batch_size: Batch size for inference (default is auto)
            store_results: Whether to store results in database
            
        Returns:
            List of inference results
        """
        # Get accelerated model
        model = self.accelerate_model(
            model_name=model_name,
            model_type=model_type,
            platform=platform
        )
        
        # Determine batch size if not specified
        if batch_size is None:
            if hasattr(model, "max_batch_size"):
                batch_size = model.max_batch_size
            else:
                batch_size = 1  # Default to 1 if unknown
        
        # Process in batches
        all_results = []
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i+batch_size]
            
            # Run inference with batch
            start_time = time.time()
            if hasattr(model, "run_batch"):
                batch_results = model.run_batch(batch)
            else:
                # Sequential processing if batch not supported
                batch_results = [model(inputs) for inputs in batch]
            
            inference_time = time.time() - start_time
            
            # Store results in database if enabled
            if store_results and self.db_integration:
                try:
                    # Get performance metrics
                    metrics = model.get_performance_metrics() if hasattr(model, "get_performance_metrics") else {}
                    
                    # Prepare metadata
                    metadata = {
                        "model_name": model_name,
                        "model_type": model_type or self.integration._infer_model_type(model_name),
                        "platform": platform,
                        "timestamp": time.time(),
                        "inference_time": inference_time,
                        "batch_size": len(batch),
                        "hardware_type": platform,
                        "performance_metrics": metrics
                    }
                    
                    # Store in database
                    self.db_integration.store_batch_inference_result(batch_results, metadata)
                except Exception as e:
                    logger.error(f"Error storing batch results in database: {e}")
            
            # Add batch results to overall results
            all_results.extend(batch_results)
        
        return all_results
    
    def get_performance_report(self, format="json"):
        """
        Get performance report for all models.
        
        Args:
            format: Output format (json, markdown, html)
            
        Returns:
            Performance report in specified format
        """
        # Get metrics from integration
        metrics = self.integration.get_metrics()
        
        if format == "json":
            return json.dumps(metrics, indent=2)
        
        elif format == "markdown":
            # Generate markdown report
            report = "# WebNN/WebGPU Acceleration Performance Report\n\n"
            
            # Add timestamp
            report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add summary
            if "aggregate" in metrics:
                agg = metrics["aggregate"]
                report += "## Summary\n\n"
                report += f"- Total Models: {agg['total_models']}\n"
                report += f"- Average Load Time: {agg['avg_load_time']:.4f}s\n"
                report += f"- Average Inference Time: {agg['avg_inference_time']:.4f}s\n"
                report += f"- Average Throughput: {agg['avg_throughput']:.2f} items/s\n"
                report += f"- Average Latency: {agg['avg_latency']:.4f}s\n"
                report += f"- Total Memory Usage: {agg['total_memory_usage']/1024/1024:.2f} MB\n\n"
                
                # Add platform distribution
                report += "### Platform Distribution\n\n"
                report += "| Platform | Count |\n"
                report += "|----------|-------|\n"
                for platform, count in agg["platform_distribution"].items():
                    report += f"| {platform} | {count} |\n"
                report += "\n"
            
            # Add model details
            report += "## Model Details\n\n"
            report += "| Model | Platform | Load Time (s) | Avg Inference Time (s) | Throughput (items/s) | Latency (s) |\n"
            report += "|-------|----------|--------------|------------------------|---------------------|------------|\n"
            
            # Process each model
            for model_id, load_time in metrics["models"]["model_load_time"].items():
                # Extract model name
                model_name = model_id.split(':')[0] if ':' in model_id else model_id
                
                # Get platform
                platform = "unknown"
                for p, count in metrics["models"]["platform_distribution"].items():
                    if count > 0 and p in model_id:
                        platform = p
                
                # Get inference time
                inference_times = metrics["models"]["inference_time"].get(model_id, [])
                avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
                
                # Get throughput and latency
                throughput = metrics["models"]["throughput"].get(model_id, 0)
                latency = metrics["models"]["latency"].get(model_id, 0)
                
                # Add row
                report += f"| {model_name} | {platform} | {load_time:.4f} | {avg_inference_time:.4f} | {throughput:.2f} | {latency:.4f} |\n"
            
            return report
            
        elif format == "html":
            # Generate HTML report (simplified version)
            html = """<!DOCTYPE html>
<html>
<head>
    <title>WebNN/WebGPU Acceleration Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>WebNN/WebGPU Acceleration Performance Report</h1>
    <p>Generated: {timestamp}</p>
""".format(timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Add summary
            if "aggregate" in metrics:
                agg = metrics["aggregate"]
                html += """
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Models:</strong> {total_models}</p>
        <p><strong>Average Load Time:</strong> {avg_load_time:.4f}s</p>
        <p><strong>Average Inference Time:</strong> {avg_inference_time:.4f}s</p>
        <p><strong>Average Throughput:</strong> {avg_throughput:.2f} items/s</p>
        <p><strong>Average Latency:</strong> {avg_latency:.4f}s</p>
        <p><strong>Total Memory Usage:</strong> {total_memory_usage:.2f} MB</p>
    </div>
""".format(
    total_models=agg['total_models'],
    avg_load_time=agg['avg_load_time'],
    avg_inference_time=agg['avg_inference_time'],
    avg_throughput=agg['avg_throughput'],
    avg_latency=agg['avg_latency'],
    total_memory_usage=agg['total_memory_usage']/1024/1024
)
                
                # Add platform distribution
                html += """
    <h2>Platform Distribution</h2>
    <table>
        <tr>
            <th>Platform</th>
            <th>Count</th>
        </tr>
"""
                for platform, count in agg["platform_distribution"].items():
                    html += f"        <tr><td>{platform}</td><td>{count}</td></tr>\n"
                
                html += "    </table>\n"
            
            # Add model details
            html += """
    <h2>Model Details</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Platform</th>
            <th>Load Time (s)</th>
            <th>Avg Inference Time (s)</th>
            <th>Throughput (items/s)</th>
            <th>Latency (s)</th>
        </tr>
"""
            
            # Process each model
            for model_id, load_time in metrics["models"]["model_load_time"].items():
                # Extract model name
                model_name = model_id.split(':')[0] if ':' in model_id else model_id
                
                # Get platform
                platform = "unknown"
                for p, count in metrics["models"]["platform_distribution"].items():
                    if count > 0 and p in model_id:
                        platform = p
                
                # Get inference time
                inference_times = metrics["models"]["inference_time"].get(model_id, [])
                avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
                
                # Get throughput and latency
                throughput = metrics["models"]["throughput"].get(model_id, 0)
                latency = metrics["models"]["latency"].get(model_id, 0)
                
                # Add row
                html += f"""        <tr>
            <td>{model_name}</td>
            <td>{platform}</td>
            <td>{load_time:.4f}</td>
            <td>{avg_inference_time:.4f}</td>
            <td>{throughput:.2f}</td>
            <td>{latency:.4f}</td>
        </tr>
"""
            
            html += """    </table>
</body>
</html>"""
            
            return html
        
        else:
            raise ValueError(f"Unsupported format: {format}. Supported formats: json, markdown, html")
    
    def close(self):
        """Clean up resources and close connections."""
        if self.integration:
            self.integration.close()
        
        # Close database connection if open
        if self.db_integration and hasattr(self.db_integration, "close"):
            self.db_integration.close()
        
        logger.info("IPFSWebAccelerator closed and cleaned up")
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.close()


# Helper function to create accelerator
def create_ipfs_web_accelerator(db_path=None, max_connections=4, 
                               enable_gpu=True, enable_cpu=True,
                               browser_preferences=None):
    """
    Create an IPFS Web Accelerator instance.
    
    Args:
        db_path: Optional path to database for storing results
        max_connections: Maximum number of browser connections
        enable_gpu: Whether to enable GPU acceleration
        enable_cpu: Whether to enable CPU acceleration
        browser_preferences: Dict mapping model families to preferred browsers
        
    Returns:
        IPFSWebAccelerator instance
    """
    return IPFSWebAccelerator(
        db_path=db_path,
        max_connections=max_connections,
        enable_gpu=enable_gpu,
        enable_cpu=enable_cpu,
        browser_preferences=browser_preferences
    )


# Automatically integrate with resource pool when imported
def integrate_with_resource_pool():
    """Integrate IPFSAccelerateWebIntegration with resource pool."""
    integration = IPFSAccelerateWebIntegration()
    
    # Store in global resource pool for access
    resource_pool = get_global_resource_pool()
    resource_pool.get_resource("ipfs_web_integration", constructor=lambda: integration)
    
    return integration

# Auto-integration when module is imported
if __name__ != "__main__":
    integrate_with_resource_pool()


# Simple test function for the module
def test_integration():
    """Test IPFS WebNN/WebGPU integration."""
    # Create accelerator
    accelerator = create_ipfs_web_accelerator()
    
    # Test with a simple model
    model_name = "bert-base-uncased"
    model = accelerator.accelerate_model(model_name, platform="webgpu")
    
    # Create sample input
    sample_input = {
        "input_ids": [101, 2023, 2003, 1037, 3231, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1]
    }
    
    # Run inference
    result = accelerator.run_inference(model_name, sample_input)
    
    # Get performance report
    report = accelerator.get_performance_report(format="markdown")
    print(report)
    
    # Close accelerator
    accelerator.close()
    
    return result

if __name__ == "__main__":
    test_integration()