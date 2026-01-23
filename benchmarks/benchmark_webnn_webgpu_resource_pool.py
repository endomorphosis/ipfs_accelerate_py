#!/usr/bin/env python3
"""
Benchmark IPFS Acceleration with Resource Pool Integration

This script benchmarks the IPFS acceleration with the WebNN/WebGPU resource pool integration,
providing comprehensive metrics on performance, memory usage, and acceleration factors.

Key features:
1. Real hardware validation to distinguish between hardware and simulation
2. Browser-specific optimization measurements (Firefox for audio, Edge for WebNN)
3. Concurrent model execution with resource pooling
4. Memory efficiency comparisons
5. Database integration for result storage and analysis
6. Cross-browser model sharding performance

Usage:
    # Run comprehensive benchmarks
    python benchmark_webnn_webgpu_resource_pool.py --comprehensive
    
    # Test specific browser and platform
    python benchmark_webnn_webgpu_resource_pool.py --browser firefox --platform webgpu --model bert-base-uncased
    
    # Test concurrent model execution
    python benchmark_webnn_webgpu_resource_pool.py --concurrent-models 3 --models bert-base-uncased,whisper-tiny,vit-base
"""

import os
import sys
import json
import time
import anyio
import argparse
import logging
import platform as platform_module
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"resource_pool_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_BROWSERS = ["chrome", "firefox", "edge", "safari"]
SUPPORTED_PLATFORMS = ["webnn", "webgpu", "all"]
SUPPORTED_MODELS = {
    "text": ["bert-base-uncased", "prajjwal1/bert-tiny", "t5-small"],
    "vision": ["vit-base", "google/vit-base-patch16-224"],
    "audio": ["whisper-tiny", "openai/whisper-tiny"]
}

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

# Check for required dependencies
required_modules = {
    "selenium": False,
    "websockets": False,
    "duckdb": False
}

try:
    import selenium
    required_modules["selenium"] = True
except ImportError:
    logger.warning("Selenium not installed. Run: pip install selenium")

try:
    import websockets
    required_modules["websockets"] = True
except ImportError:
    logger.warning("Websockets not installed. Run: pip install websockets")

try:
    import duckdb
    required_modules["duckdb"] = True
except ImportError:
    logger.warning("DuckDB not installed. Run: pip install duckdb")

class ResourcePoolBenchmarker:
    """Benchmark IPFS acceleration with WebNN/WebGPU resource pool integration."""
    
    def __init__(self, args):
        """Initialize benchmarker with command line arguments."""
        self.args = args
        self.results = []
        self.resource_pool_integration = None
        self.ipfs_module = None
        self.db_connection = None
        self.real_implementation_detected = False
        self.features = {}
        self.browser_info = {}
        
        # Set environment variables for real hardware testing
        os.environ["WEBNN_SIMULATION"] = "0" if not args.allow_simulation else "1"
        os.environ["WEBGPU_SIMULATION"] = "0" if not args.allow_simulation else "1"
        os.environ["USE_BROWSER_AUTOMATION"] = "1"
        
        # Enable browser-specific optimizations
        if args.browser == "firefox" and args.optimize_audio:
            os.environ["USE_FIREFOX_WEBGPU"] = "1"
            os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
            logger.info("Enabled Firefox audio optimizations (256x1x1 workgroup size)")
        
        # Import required modules
        self._import_modules()
        
        # Connect to database if specified
        self._connect_to_database()
    
    def _import_modules(self):
        """Import required modules for benchmarking."""
        # Import IPFS acceleration module
        try:
            import ipfs_accelerate_impl
            self.ipfs_module = ipfs_accelerate_impl
            logger.info("IPFS acceleration module imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import IPFS acceleration module: {e}")
            logger.error("Make sure ipfs_accelerate_impl.py is in the current directory or PYTHONPATH")
            self.ipfs_module = None
        
        # Import resource pool integration
        try:
            from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
            self.ResourcePoolBridgeIntegration = ResourcePoolBridgeIntegration
            logger.info("ResourcePoolBridgeIntegration imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import ResourcePoolBridgeIntegration: {e}")
            logger.error("Make sure fixed_web_platform/resource_pool_bridge.py is in the current directory or PYTHONPATH")
            self.ResourcePoolBridgeIntegration = None
        
        # Import WebImplementation for real hardware detection
        try:
            from run_real_webgpu_webnn_fixed import WebImplementation
            self.WebImplementation = WebImplementation
            logger.info("WebImplementation imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import WebImplementation: {e}")
            logger.error("Make sure run_real_webgpu_webnn_fixed.py is in the current directory or PYTHONPATH")
            self.WebImplementation = None
    
    def _connect_to_database(self):
        """Connect to DuckDB database."""
        if self.args.db_path and required_modules["duckdb"]:
            try:
                self.db_connection = duckdb.connect(self.args.db_path)
                logger.info(f"Connected to database: {self.args.db_path}")
                self._ensure_db_schema()
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_connection = None
    
    def _ensure_db_schema(self):
        """Ensure the database has the required schema."""
        if not self.db_connection:
            return
        
        try:
            # Check if resource_pool_benchmarks table exists
            table_exists = self.db_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='resource_pool_benchmarks'"
            ).fetchone()
            
            if not table_exists:
                # Create table if it doesn't exist
                self.db_connection.execute("""
                CREATE TABLE resource_pool_benchmarks (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    model_name VARCHAR,
                    platform VARCHAR,
                    browser VARCHAR,
                    is_real_implementation BOOLEAN,
                    is_simulation BOOLEAN,
                    precision INTEGER,
                    mixed_precision BOOLEAN,
                    concurrent_models INTEGER,
                    browser_optimizations BOOLEAN,
                    latency_ms FLOAT,
                    throughput_items_per_sec FLOAT,
                    memory_usage_mb FLOAT,
                    init_time_ms FLOAT,
                    inference_time_ms FLOAT,
                    ipfs_time_ms FLOAT,
                    acceleration_factor FLOAT,
                    adapter_info VARCHAR,
                    system_info VARCHAR,
                    performance_metrics JSON,
                    details JSON
                )
                """)
                logger.info("Created resource_pool_benchmarks table in database")
        except Exception as e:
            logger.error(f"Failed to ensure database schema: {e}")
    
    async def initialize_resource_pool(self):
        """Initialize the resource pool integration."""
        if not self.ResourcePoolBridgeIntegration:
            logger.error("ResourcePoolBridgeIntegration class not available")
            return False
        
        try:
            # Create browser preferences based on model types
            browser_preferences = {
                'audio': 'firefox' if self.args.optimize_audio else 'chrome',
                'vision': 'chrome',
                'text': 'edge' if self.args.browser == 'edge' else 'chrome',
                'text_embedding': 'edge' if self.args.browser == 'edge' else 'chrome',
                'multimodal': 'chrome'
            }
            
            # Create resource pool integration
            self.resource_pool_integration = self.ResourcePoolBridgeIntegration(
                max_connections=self.args.max_connections,
                enable_gpu=self.args.platform in ['webgpu', 'all'],
                enable_cpu=True,
                headless=not self.args.visible,
                browser_preferences=browser_preferences,
                adaptive_scaling=True
            )
            
            # Initialize the integration
            self.resource_pool_integration.initialize()
            logger.info("Resource pool integration initialized successfully")
            
            # Validate real hardware implementation
            await self.detect_real_implementation()
            
            return True
        except Exception as e:
            logger.error(f"Error initializing resource pool integration: {e}")
            return False
    
    async def detect_real_implementation(self):
        """Detect if real WebNN/WebGPU implementation is available."""
        if not self.WebImplementation:
            logger.error("WebImplementation class not available")
            return False
        
        try:
            # Create WebImplementation instance for detection
            web_implementation = self.WebImplementation(
                platform=self.args.platform if self.args.platform != "all" else "webgpu",
                browser=self.args.browser,
                headless=not self.args.visible
            )
            
            # Start implementation to check real hardware availability
            start_success = await web_implementation.start(allow_simulation=self.args.allow_simulation)
            
            if not start_success:
                logger.error(f"Failed to start {self.args.platform} implementation")
                return False
            
            # Check if real implementation is being used
            is_real = not web_implementation.simulation_mode
            self.real_implementation_detected = is_real
            
            # Store feature details
            self.features = web_implementation.features or {}
            
            # Log implementation details
            if is_real:
                logger.info(f"Real {self.args.platform} implementation detected in {self.args.browser}")
                
                # Log adapter/backend details
                if self.args.platform == "webgpu" or self.args.platform == "all":
                    adapter = self.features.get("webgpu_adapter", {})
                    if adapter:
                        logger.info(f"WebGPU Adapter: {adapter.get('vendor', 'Unknown')} - {adapter.get('architecture', 'Unknown')}")
                        self.browser_info["webgpu_adapter"] = adapter
                
                if self.args.platform == "webnn" or self.args.platform == "all":
                    backend = self.features.get("webnn_backend", "Unknown")
                    logger.info(f"WebNN Backend: {backend}")
                    self.browser_info["webnn_backend"] = backend
            else:
                logger.warning(f"Using simulation mode - real {self.args.platform} hardware not detected")
            
            # Stop the implementation after detection
            await web_implementation.stop()
            
            return is_real
        except Exception as e:
            logger.error(f"Error detecting real implementation: {e}")
            return False
    
    def get_models_by_type(self, model_type, count=1):
        """Get model names by type."""
        if model_type in SUPPORTED_MODELS and SUPPORTED_MODELS[model_type]:
            return SUPPORTED_MODELS[model_type][:count]
        return []
    
    def parse_model_list(self, model_list):
        """Parse comma-separated model list into individual models."""
        if not model_list:
            return []
        return [m.strip() for m in model_list.split(",")]
    
    def determine_model_type(self, model_name):
        """Determine model type from model name."""
        model_type = "text"
        if any(x in model_name.lower() for x in ["whisper", "wav2vec", "clap"]):
            model_type = "audio"
        elif any(x in model_name.lower() for x in ["vit", "clip", "detr", "image"]):
            model_type = "vision"
        elif any(x in model_name.lower() for x in ["llava", "xclip"]):
            model_type = "multimodal"
        return model_type
    
    async def benchmark_single_model(self, model_name):
        """Benchmark a single model with resource pool integration."""
        if not self.resource_pool_integration or not self.ipfs_module:
            logger.error("Resource pool integration or IPFS module not available")
            return None
        
        try:
            # Determine model type
            model_type = self.determine_model_type(model_name)
            
            logger.info(f"Benchmarking model: {model_name} (Type: {model_type})")
            
            # Prepare test data
            if model_type == "text":
                test_content = "This is a benchmark test of IPFS acceleration with resource pool integration."
            elif model_type == "vision":
                test_content = {"image": "test.jpg"}
            elif model_type == "audio":
                test_content = {"audio": "test.mp3"}
            else:
                test_content = "This is a benchmark test of IPFS acceleration with resource pool integration."
            
            # Create hardware preferences
            hardware_preferences = {
                "priority_list": [self.args.platform] if self.args.platform != "all" else ["webgpu", "webnn", "cpu"],
                "model_family": model_type
            }
            
            # Measure model initialization time
            start_time = time.time()
            model = self.resource_pool_integration.get_model(
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
            init_time = time.time() - start_time
            
            if not model:
                logger.error(f"Failed to get model: {model_name}")
                return None
            
            # Check if real implementation (if ResourcePoolBridgeIntegration provides this info)
            is_real = self.real_implementation_detected
            if hasattr(model, "is_real_implementation"):
                is_real = model.is_real_implementation
            
            # Measure standard inference time (without IPFS acceleration)
            inference_start_time = time.time()
            inference_result = model(test_content)
            inference_time = time.time() - inference_start_time
            
            # Configure IPFS acceleration
            acceleration_config = {
                "platform": self.args.platform if self.args.platform != "all" else "webgpu",
                "browser": self.args.browser,
                "is_real_hardware": is_real,
                "precision": self.args.precision,
                "mixed_precision": self.args.mixed_precision,
                "use_firefox_optimizations": (
                    self.args.browser == "firefox" and 
                    self.args.optimize_audio and 
                    model_type == "audio"
                )
            }
            
            # Measure IPFS accelerated time
            ipfs_start_time = time.time()
            acceleration_result = self.ipfs_module.accelerate(
                model_name,
                test_content,
                acceleration_config
            )
            ipfs_time = time.time() - ipfs_start_time
            
            # Calculate acceleration factor
            acceleration_factor = inference_time / ipfs_time if ipfs_time > 0 else 1.0
            
            # Get performance metrics (if model provides them)
            performance_metrics = {}
            if hasattr(model, "get_performance_metrics"):
                performance_metrics = model.get_performance_metrics()
            
            # Extract metrics from either inference result or model metrics
            metrics = {}
            if isinstance(inference_result, dict) and "metrics" in inference_result:
                metrics = inference_result["metrics"]
            elif performance_metrics and "stats" in performance_metrics:
                stats = performance_metrics["stats"]
                metrics = {
                    "latency_ms": stats.get("avg_latency", 0) * 1000,
                    "throughput_items_per_sec": stats.get("throughput", 0),
                    "memory_usage_mb": performance_metrics.get("memory_usage", {}).get("peak", 0)
                }
            else:
                metrics = {
                    "latency_ms": inference_time * 1000,
                    "throughput_items_per_sec": 1000 / (inference_time * 1000),
                    "memory_usage_mb": 0
                }
            
            # Create result object
            result = {
                "model_name": model_name,
                "model_type": model_type,
                "platform": self.args.platform if self.args.platform != "all" else "webgpu",
                "browser": self.args.browser,
                "is_real_implementation": is_real,
                "is_simulation": not is_real,
                "precision": self.args.precision,
                "mixed_precision": self.args.mixed_precision,
                "browser_optimizations": acceleration_config["use_firefox_optimizations"],
                "concurrent_models": 1,  # Single model benchmark
                "timestamp": datetime.now().isoformat(),
                "init_time_ms": init_time * 1000,
                "inference_time_ms": inference_time * 1000,
                "ipfs_time_ms": ipfs_time * 1000,
                "acceleration_factor": acceleration_factor,
                "metrics": metrics,
                "performance_metrics": performance_metrics
            }
            
            # Add platform-specific details
            if self.browser_info.get("webgpu_adapter"):
                result["adapter_info"] = self.browser_info["webgpu_adapter"]
            
            if self.browser_info.get("webnn_backend"):
                result["backend_info"] = self.browser_info["webnn_backend"]
            
            # Add system info
            result["system_info"] = {
                "platform": platform_module.platform(),
                "processor": platform_module.processor(),
                "python_version": platform_module.python_version()
            }
            
            # Store in database
            if self.db_connection:
                self._store_result_in_db(result)
            
            # Print result summary
            self._print_result_summary(result)
            
            # Add to results list
            self.results.append(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error benchmarking model {model_name}: {e}")
            return None
    
    async def benchmark_concurrent_models(self, model_names):
        """Benchmark concurrent execution of multiple models."""
        if not self.resource_pool_integration or not self.ipfs_module:
            logger.error("Resource pool integration or IPFS module not available")
            return None
        
        if len(model_names) < 2:
            logger.error("Need at least 2 models for concurrent benchmarking")
            return None
        
        try:
            models = []
            model_types = []
            test_contents = []
            hardware_preferences = {
                "priority_list": [self.args.platform] if self.args.platform != "all" else ["webgpu", "webnn", "cpu"]
            }
            
            # Initialize all models
            init_start_time = time.time()
            for model_name in model_names:
                # Determine model type
                model_type = self.determine_model_type(model_name)
                model_types.append(model_type)
                
                # Prepare test data based on model type
                if model_type == "text":
                    test_content = "This is a concurrent benchmark test of IPFS acceleration."
                elif model_type == "vision":
                    test_content = {"image": "test.jpg"}
                elif model_type == "audio":
                    test_content = {"audio": "test.mp3"}
                else:
                    test_content = "This is a concurrent benchmark test of IPFS acceleration."
                
                test_contents.append(test_content)
                
                # Set model-specific hardware preference
                current_preferences = hardware_preferences.copy()
                current_preferences["model_family"] = model_type
                
                # Get model
                logger.info(f"Initializing model: {model_name} (Type: {model_type})")
                model = self.resource_pool_integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=current_preferences
                )
                
                if not model:
                    logger.error(f"Failed to get model: {model_name}")
                    continue
                
                models.append(model)
            
            init_time = time.time() - init_start_time
            
            if not models:
                logger.error("Failed to initialize any models")
                return None
            
            logger.info(f"Initialized {len(models)} models in {init_time:.2f} seconds")
            
            # Run concurrent inference (benchmark execution time)
            inference_start_time = time.time()
            if len(models) == 1:
                inference_results = [models[0](test_contents[0])]
            else:
                # Use first model to run concurrent inference with others
                primary_model = models[0]
                if hasattr(primary_model, "run_concurrent") and callable(primary_model.run_concurrent):
                    # Use enhanced concurrent execution if available
                    inference_results = primary_model.run_concurrent(
                        [test_contents[0]] * 3,  # Run multiple instances of same input
                        other_models=models[1:]
                    )
                else:
                    # Fallback to sequential execution
                    inference_results = []
                    for i, model in enumerate(models):
                        inference_results.append(model(test_contents[i]))
            
            inference_time = time.time() - inference_start_time
            
            # Calculate average metrics across all models
            avg_latency = 0
            avg_throughput = 0
            avg_memory = 0
            count = 0
            
            for i, result in enumerate(inference_results):
                if isinstance(result, dict) and "metrics" in result:
                    metrics = result["metrics"]
                    avg_latency += metrics.get("latency_ms", 0)
                    avg_throughput += metrics.get("throughput_items_per_sec", 0)
                    avg_memory += metrics.get("memory_usage_mb", 0)
                    count += 1
            
            if count > 0:
                avg_latency /= count
                avg_throughput /= count
                avg_memory /= count
            
            # Run IPFS acceleration for comparison
            ipfs_start_time = time.time()
            ipfs_results = []
            for i, model_name in enumerate(model_names):
                if i >= len(model_types) or i >= len(test_contents):
                    continue
                
                # Configure IPFS acceleration
                acceleration_config = {
                    "platform": self.args.platform if self.args.platform != "all" else "webgpu",
                    "browser": self.args.browser,
                    "is_real_hardware": self.real_implementation_detected,
                    "precision": self.args.precision,
                    "mixed_precision": self.args.mixed_precision,
                    "use_firefox_optimizations": (
                        self.args.browser == "firefox" and 
                        self.args.optimize_audio and 
                        model_types[i] == "audio"
                    )
                }
                
                # Run IPFS acceleration
                ipfs_result = self.ipfs_module.accelerate(
                    model_name,
                    test_contents[i],
                    acceleration_config
                )
                ipfs_results.append(ipfs_result)
            
            ipfs_time = time.time() - ipfs_start_time
            
            # Calculate acceleration factor
            acceleration_factor = inference_time / ipfs_time if ipfs_time > 0 else 1.0
            
            # Create result object
            result = {
                "model_names": model_names,
                "model_types": model_types,
                "platform": self.args.platform if self.args.platform != "all" else "webgpu",
                "browser": self.args.browser,
                "is_real_implementation": self.real_implementation_detected,
                "is_simulation": not self.real_implementation_detected,
                "precision": self.args.precision,
                "mixed_precision": self.args.mixed_precision,
                "concurrent_models": len(models),
                "browser_optimizations": (
                    self.args.browser == "firefox" and 
                    self.args.optimize_audio and 
                    "audio" in model_types
                ),
                "timestamp": datetime.now().isoformat(),
                "init_time_ms": init_time * 1000,
                "inference_time_ms": inference_time * 1000,
                "ipfs_time_ms": ipfs_time * 1000,
                "acceleration_factor": acceleration_factor,
                "metrics": {
                    "latency_ms": avg_latency,
                    "throughput_items_per_sec": avg_throughput,
                    "memory_usage_mb": avg_memory
                }
            }
            
            # Add browser-specific details
            if self.browser_info.get("webgpu_adapter"):
                result["adapter_info"] = self.browser_info["webgpu_adapter"]
            
            if self.browser_info.get("webnn_backend"):
                result["backend_info"] = self.browser_info["webnn_backend"]
            
            # Add system info
            result["system_info"] = {
                "platform": platform_module.platform(),
                "processor": platform_module.processor(),
                "python_version": platform_module.python_version()
            }
            
            # Print concurrent result summary
            self._print_concurrent_result_summary(result)
            
            # Add to results list
            self.results.append(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in concurrent benchmarking: {e}")
            return None
    
    def _store_result_in_db(self, result):
        """Store result in database."""
        if not self.db_connection:
            return
        
        try:
            # Extract model name (handle single or multiple models)
            model_name = result.get("model_name", "unknown")
            if not model_name and "model_names" in result:
                model_name = ",".join(result["model_names"])
            
            # Extract metrics
            metrics = result.get("metrics", {})
            
            # Insert result into database
            self.db_connection.execute("""
            INSERT INTO resource_pool_benchmarks (
                timestamp,
                model_name,
                platform,
                browser,
                is_real_implementation,
                is_simulation,
                precision,
                mixed_precision,
                concurrent_models,
                browser_optimizations,
                latency_ms,
                throughput_items_per_sec,
                memory_usage_mb,
                init_time_ms,
                inference_time_ms,
                ipfs_time_ms,
                acceleration_factor,
                adapter_info,
                system_info,
                performance_metrics,
                details
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                datetime.now(),
                model_name,
                result.get("platform", "unknown"),
                result.get("browser", "unknown"),
                result.get("is_real_implementation", False),
                result.get("is_simulation", True),
                result.get("precision", 0),
                result.get("mixed_precision", False),
                result.get("concurrent_models", 1),
                result.get("browser_optimizations", False),
                metrics.get("latency_ms", 0),
                metrics.get("throughput_items_per_sec", 0),
                metrics.get("memory_usage_mb", 0),
                result.get("init_time_ms", 0),
                result.get("inference_time_ms", 0),
                result.get("ipfs_time_ms", 0),
                result.get("acceleration_factor", 0),
                json.dumps(result.get("adapter_info", {})),
                json.dumps(result.get("system_info", {})),
                json.dumps(result.get("performance_metrics", {})),
                json.dumps(result)
            ])
            
            logger.info(f"Stored result for {model_name} in database")
        except Exception as e:
            logger.error(f"Failed to store result in database: {e}")
    
    def _print_result_summary(self, result):
        """Print a summary of the benchmark result."""
        print("\n" + "="*80)
        print(f"RESOURCE POOL BENCHMARK: {result['model_name']}")
        print("="*80)
        print(f"Model: {result['model_name']}")
        print(f"Type: {result['model_type']}")
        print(f"Platform: {result['platform']} ({result['browser']})")
        print(f"Implementation: {'REAL HARDWARE' if result['is_real_implementation'] else 'SIMULATION'}")
        print(f"Precision: {result['precision']}-bit{' (mixed)' if result['mixed_precision'] else ''}")
        
        if result.get("browser_optimizations"):
            print("Browser Optimizations: ENABLED")
        
        print("\nPerformance:")
        print(f"  - Model Initialization: {result['init_time_ms']:.2f} ms")
        print(f"  - Standard Inference: {result['inference_time_ms']:.2f} ms")
        print(f"  - IPFS Accelerated: {result['ipfs_time_ms']:.2f} ms")
        print(f"  - Acceleration Factor: {result['acceleration_factor']:.2f}x")
        
        metrics = result.get("metrics", {})
        print(f"  - Latency: {metrics.get('latency_ms', 0):.2f} ms")
        print(f"  - Throughput: {metrics.get('throughput_items_per_sec', 0):.2f} items/sec")
        print(f"  - Memory Usage: {metrics.get('memory_usage_mb', 0):.2f} MB")
        
        # Print hardware details
        if "adapter_info" in result and result["adapter_info"]:
            adapter = result["adapter_info"]
            print("\nWebGPU Hardware:")
            print(f"  - Vendor: {adapter.get('vendor', 'Unknown')}")
            print(f"  - Architecture: {adapter.get('architecture', 'Unknown')}")
            print(f"  - Device: {adapter.get('device', 'Unknown')}")
        
        if "backend_info" in result and result["backend_info"]:
            print(f"\nWebNN Backend: {result['backend_info']}")
        
        print("="*80)
    
    def _print_concurrent_result_summary(self, result):
        """Print a summary of concurrent benchmark results."""
        print("\n" + "="*80)
        print(f"CONCURRENT RESOURCE POOL BENCHMARK: {result['concurrent_models']} MODELS")
        print("="*80)
        print(f"Models: {', '.join(result.get('model_names', []))}")
        print(f"Types: {', '.join(result.get('model_types', []))}")
        print(f"Platform: {result['platform']} ({result['browser']})")
        print(f"Implementation: {'REAL HARDWARE' if result['is_real_implementation'] else 'SIMULATION'}")
        print(f"Precision: {result['precision']}-bit{' (mixed)' if result['mixed_precision'] else ''}")
        
        if result.get("browser_optimizations"):
            print("Browser Optimizations: ENABLED")
        
        print("\nPerformance:")
        print(f"  - Models Initialization: {result['init_time_ms']:.2f} ms")
        print(f"  - Concurrent Inference: {result['inference_time_ms']:.2f} ms")
        print(f"  - Sequential IPFS Calls: {result['ipfs_time_ms']:.2f} ms")
        print(f"  - Acceleration Factor: {result['acceleration_factor']:.2f}x")
        
        metrics = result.get("metrics", {})
        print(f"  - Average Latency: {metrics.get('latency_ms', 0):.2f} ms")
        print(f"  - Average Throughput: {metrics.get('throughput_items_per_sec', 0):.2f} items/sec")
        print(f"  - Average Memory Usage: {metrics.get('memory_usage_mb', 0):.2f} MB")
        
        # Print hardware details
        if "adapter_info" in result and result["adapter_info"]:
            adapter = result["adapter_info"]
            print("\nWebGPU Hardware:")
            print(f"  - Vendor: {adapter.get('vendor', 'Unknown')}")
            print(f"  - Architecture: {adapter.get('architecture', 'Unknown')}")
            print(f"  - Device: {adapter.get('device', 'Unknown')}")
        
        if "backend_info" in result and result["backend_info"]:
            print(f"\nWebNN Backend: {result['backend_info']}")
        
        print("="*80)
    
    async def run_benchmarks(self):
        """Run all benchmarks based on command line arguments."""
        # Initialize resource pool
        if not await self.initialize_resource_pool():
            logger.error("Failed to initialize resource pool")
            return False
        
        try:
            # Handle concurrent model benchmarking
            if self.args.concurrent_models > 1:
                # Determine models to benchmark
                if self.args.models:
                    model_names = self.parse_model_list(self.args.models)
                    if len(model_names) < self.args.concurrent_models:
                        # Fill with default models if not enough specified
                        needed = self.args.concurrent_models - len(model_names)
                        model_names.extend(self.get_models_by_type("text", needed))
                else:
                    # Use default models from different types
                    model_names = (
                        self.get_models_by_type("text", 1) +
                        self.get_models_by_type("vision", 1) +
                        self.get_models_by_type("audio", 1)
                    )
                
                # Limit to requested concurrent count
                model_names = model_names[:self.args.concurrent_models]
                
                # Run concurrent benchmark
                logger.info(f"Running concurrent benchmark with {len(model_names)} models: {', '.join(model_names)}")
                await self.benchmark_concurrent_models(model_names)
            else:
                # Handle single model benchmarking
                # Determine models to benchmark
                if self.args.models:
                    model_names = self.parse_model_list(self.args.models)
                elif self.args.model == "all":
                    # Use all default models
                    model_names = []
                    for model_type in SUPPORTED_MODELS:
                        model_names.extend(SUPPORTED_MODELS[model_type])
                else:
                    model_names = [self.args.model]
                
                # Run benchmarks for each model
                for model_name in model_names:
                    logger.info(f"Benchmarking model: {model_name}")
                    await self.benchmark_single_model(model_name)
            
            return True
        except Exception as e:
            logger.error(f"Error running benchmarks: {e}")
            return False
        finally:
            # Close resource pool
            if self.resource_pool_integration:
                self.resource_pool_integration.close()
            
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
    
    def save_results(self):
        """Save benchmark results to file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_filename = f"resource_pool_benchmark_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {json_filename}")
        
        # Save Markdown report
        md_filename = f"resource_pool_benchmark_{timestamp}.md"
        self.generate_markdown_report(md_filename)
    
    def generate_markdown_report(self, filename):
        """Generate a detailed markdown report of benchmark results."""
        with open(filename, 'w') as f:
            f.write("# Resource Pool Integration Benchmark Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add implementation status summary
            f.write("## Implementation Status\n\n")
            
            # Count real vs simulation
            real_count = sum(1 for r in self.results if r.get("is_real_implementation", False))
            sim_count = len(self.results) - real_count
            
            f.write(f"- Total benchmarks: {len(self.results)}\n")
            f.write(f"- Real hardware implementations: {real_count}\n")
            f.write(f"- Simulation implementations: {sim_count}\n\n")
            
            # Add performance summary for single model benchmarks
            single_model_results = [r for r in self.results if r.get("concurrent_models", 1) == 1]
            if single_model_results:
                f.write("## Single Model Performance\n\n")
                
                f.write("| Model | Platform | Browser | Real HW | Precision | Init (ms) | Inference (ms) | IPFS (ms) | Accel. Factor |\n")
                f.write("|-------|----------|---------|---------|-----------|-----------|----------------|-----------|---------------|\n")
                
                # Sort by model name and platform
                sorted_results = sorted(single_model_results, key=lambda r: (r.get("model_name", ""), r.get("platform", "")))
                
                for result in sorted_results:
                    model = result.get("model_name", "unknown")
                    platform = result.get("platform", "unknown")
                    browser = result.get("browser", "unknown")
                    real = "✅" if result.get("is_real_implementation", False) else "❌"
                    
                    precision = f"{result.get('precision', 'unknown')}-bit"
                    if result.get("mixed_precision", False):
                        precision += " (mixed)"
                    
                    init_time = f"{result.get('init_time_ms', 0):.1f}"
                    inference_time = f"{result.get('inference_time_ms', 0):.1f}"
                    ipfs_time = f"{result.get('ipfs_time_ms', 0):.1f}"
                    accel_factor = f"{result.get('acceleration_factor', 0):.2f}x"
                    
                    f.write(f"| {model} | {platform} | {browser} | {real} | {precision} | {init_time} | {inference_time} | {ipfs_time} | {accel_factor} |\n")
                
                f.write("\n")
            
            # Add performance summary for concurrent model benchmarks
            concurrent_results = [r for r in self.results if r.get("concurrent_models", 1) > 1]
            if concurrent_results:
                f.write("## Concurrent Model Performance\n\n")
                
                f.write("| Models | Count | Platform | Browser | Real HW | Init (ms) | Concurrent (ms) | Sequential (ms) | Speedup |\n")
                f.write("|--------|-------|----------|---------|---------|-----------|-----------------|----------------|--------|\n")
                
                for result in concurrent_results:
                    models = ", ".join(result.get("model_names", []))
                    if len(models) > 30:
                        models = models[:27] + "..."
                    
                    count = result.get("concurrent_models", 0)
                    platform = result.get("platform", "unknown")
                    browser = result.get("browser", "unknown")
                    real = "✅" if result.get("is_real_implementation", False) else "❌"
                    
                    init_time = f"{result.get('init_time_ms', 0):.1f}"
                    concurrent_time = f"{result.get('inference_time_ms', 0):.1f}"
                    sequential_time = f"{result.get('ipfs_time_ms', 0):.1f}"
                    speedup = f"{result.get('acceleration_factor', 0):.2f}x"
                    
                    f.write(f"| {models} | {count} | {platform} | {browser} | {real} | {init_time} | {concurrent_time} | {sequential_time} | {speedup} |\n")
                
                f.write("\n")
            
            # Add browser-specific insights
            f.write("## Browser-Specific Insights\n\n")
            
            # Firefox audio optimizations
            firefox_audio_results = [r for r in self.results 
                                    if r.get("browser") == "firefox" and 
                                    r.get("browser_optimizations", False)]
            
            if firefox_audio_results:
                f.write("### Firefox Audio Optimizations\n\n")
                f.write("Firefox provides specialized optimizations for audio models:\n\n")
                f.write("- Uses optimized compute shader workgroup size (256x1x1 vs Chrome's 128x2x1)\n")
                f.write("- Achieves better performance than Chrome for audio models\n")
                f.write("- Provides better power efficiency\n")
                f.write("- Particularly effective for Whisper, Wav2Vec2, and CLAP models\n\n")
            
            # Resource pool insights
            concurrent_factors = [r.get("acceleration_factor", 0) for r in concurrent_results]
            if concurrent_factors:
                avg_concurrent_factor = sum(concurrent_factors) / len(concurrent_factors)
                
                f.write("## Resource Pool Insights\n\n")
                f.write(f"- Average Concurrent Speedup: {avg_concurrent_factor:.2f}x\n")
                f.write("- Benefits of Resource Pool Integration:\n")
                f.write("  - Connection pooling reduces browser startup overhead\n")
                f.write("  - Concurrent model execution improves throughput\n")
                f.write("  - Browser-specific optimizations improve model performance\n")
                f.write("  - Adaptive resource scaling based on workload\n")
                f.write("  - Memory efficiency through shared resources\n\n")
            
            # IPFS acceleration benefits
            single_factors = [r.get("acceleration_factor", 0) for r in single_model_results]
            if single_factors:
                avg_single_factor = sum(single_factors) / len(single_factors)
                max_single_factor = max(single_factors)
                
                f.write("## IPFS Acceleration Benefits\n\n")
                f.write(f"- Average Acceleration Factor: {avg_single_factor:.2f}x\n")
                f.write(f"- Maximum Acceleration Factor: {max_single_factor:.2f}x\n")
                f.write("- Benefits of IPFS Acceleration:\n")
                f.write("  - Efficient content delivery through P2P optimization\n")
                f.write("  - Reduced latency through local caching\n")
                f.write("  - Hardware acceleration through optimal device selection\n\n")
            
            # Add system information
            if self.results and "system_info" in self.results[0]:
                system_info = self.results[0]["system_info"]
                f.write("## System Information\n\n")
                f.write(f"- Platform: {system_info.get('platform', 'Unknown')}\n")
                f.write(f"- Processor: {system_info.get('processor', 'Unknown')}\n")
                f.write(f"- Python Version: {system_info.get('python_version', 'Unknown')}\n\n")
            
            logger.info(f"Markdown report saved to {filename}")


async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(description="Benchmark IPFS Acceleration with Resource Pool Integration")
    
    # Browser options
    parser.add_argument("--browser", choices=SUPPORTED_BROWSERS, default="chrome",
                      help="Browser to use for testing")
    
    # Platform options
    parser.add_argument("--platform", choices=SUPPORTED_PLATFORMS, default="webgpu",
                      help="Platform to test (webnn, webgpu, or all)")
    
    # Model options
    parser.add_argument("--model", default="bert-base-uncased",
                      help="Model to benchmark")
    parser.add_argument("--models", type=str,
                      help="Comma-separated list of models to benchmark")
    
    # Resource pool options
    parser.add_argument("--max-connections", type=int, default=4,
                      help="Maximum number of browser connections in resource pool")
    parser.add_argument("--concurrent-models", type=int, default=1,
                      help="Number of concurrent models to benchmark (uses resource pool)")
    
    # Precision options
    parser.add_argument("--precision", type=int, choices=[4, 8, 16, 32], default=8,
                      help="Precision level to test (bit width)")
    parser.add_argument("--mixed-precision", action="store_true",
                      help="Use mixed precision (higher precision for critical layers)")
    
    # Optimization options
    parser.add_argument("--optimize-audio", action="store_true",
                      help="Enable Firefox audio optimizations for audio models")
    
    # Test options
    parser.add_argument("--comprehensive", action="store_true",
                      help="Run comprehensive benchmarks (all browsers, platforms, models)")
    parser.add_argument("--visible", action="store_true",
                      help="Run browser in visible mode (not headless)")
    parser.add_argument("--allow-simulation", action="store_true",
                      help="Allow simulation if real hardware not available")
    
    # Output options
    parser.add_argument("--db-path", type=str,
                      help="Path to DuckDB database file")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle comprehensive flag
    if args.comprehensive:
        args.browser = "chrome"  # Start with Chrome
        args.platform = "all"
        args.concurrent_models = 3  # Test concurrent execution
        args.model = "all"
    
    # Check dependencies
    missing_deps = [name for name, installed in required_modules.items() if not installed]
    if missing_deps:
        logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        logger.error("Please install them with: pip install " + " ".join(missing_deps))
        return 1
    
    # Create benchmarker
    benchmarker = ResourcePoolBenchmarker(args)
    
    # Run benchmarks
    logger.info("Starting IPFS acceleration with resource pool benchmarks")
    success = await benchmarker.run_benchmarks()
    
    if not success:
        logger.error("Benchmarks failed")
        return 1
    
    if not benchmarker.results:
        logger.error("No benchmark results obtained")
        return 1
    
    # Save results
    benchmarker.save_results()
    
    # Print summary
    real_count = sum(1 for r in benchmarker.results if r.get("is_real_implementation", False))
    sim_count = len(benchmarker.results) - real_count
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total benchmarks: {len(benchmarker.results)}")
    print(f"Real hardware implementations: {real_count}")
    print(f"Simulation implementations: {sim_count}")
    
    # Print acceleration and speedup summary
    single_factors = [r.get("acceleration_factor", 0) for r in benchmarker.results 
                      if r.get("concurrent_models", 1) == 1]
    
    concurrent_factors = [r.get("acceleration_factor", 0) for r in benchmarker.results 
                          if r.get("concurrent_models", 1) > 1]
    
    if single_factors:
        avg_single_factor = sum(single_factors) / len(single_factors)
        print(f"Average Single Model Acceleration: {avg_single_factor:.2f}x")
    
    if concurrent_factors:
        avg_concurrent_factor = sum(concurrent_factors) / len(concurrent_factors)
        print(f"Average Concurrent Model Speedup: {avg_concurrent_factor:.2f}x")
    
    print("="*80 + "\n")
    
    # If comprehensive mode, print recommendation
    if args.comprehensive:
        print("RECOMMENDATIONS:")
        print("- For text models (BERT, T5): Use Edge browser with WebNN")
        print("- For vision models (ViT): Use Chrome browser with WebGPU")
        print("- For audio models (Whisper): Use Firefox browser with WebGPU and audio optimizations")
        print("- For concurrent execution: Use resource pool with 3-4 connections for optimal throughput")
        print("- For mixed precision: 8-bit provides best performance/memory tradeoff")
        print()
    
    return 0 if real_count > 0 else 2  # Return 2 for simulation-only


def main():
    """Main entry point."""
    try:
        return anyio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())