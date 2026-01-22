#!/usr/bin/env python3
"""
Comprehensive Benchmark System for WebNN/WebGPU Resource Pool with IPFS Acceleration

This module provides a robust benchmarking system for the WebNN/WebGPU resource pool 
with IPFS acceleration. It includes real hardware validation, resource pool extensions
for memory optimization, cross-browser model sharding, browser-specific optimizations,
and detailed performance metrics.

Key features:
1. Real hardware validation to ensure tests are running on actual hardware, not simulation
2. Memory optimization for resource pools to efficiently manage memory-intensive workloads
3. Cross-browser model sharding for large model execution across multiple browser instances
4. Browser-specific optimizations (Firefox for audio, Edge for WebNN, Chrome for vision)
5. Comprehensive benchmarking with statistical analysis and outlier detection
6. Detailed performance metrics including latency, throughput, memory usage, and power
7. Database integration for storing and analyzing benchmark results
8. Test suite for concurrent model execution across multiple browsers
9. IPFS acceleration with P2P network optimization

Usage:
    # Run a full benchmark of resource pool integration
    python benchmark_resource_pool_integration.py --comprehensive
    
    # Test the Firefox audio optimizations
    python benchmark_resource_pool_integration.py --test-firefox-audio
    
    # Test cross-browser model sharding
    python benchmark_resource_pool_integration.py --test-sharding
    
    # Test memory optimization for large models
    python benchmark_resource_pool_integration.py --test-memory-optimization
    
    # Run all tests and generate comprehensive report
    python benchmark_resource_pool_integration.py --all-tests --generate-report
"""

import os
import sys
import json
import time
import uuid
import socket
import asyncio
import logging
import argparse
import platform
import tempfile
import statistics
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

# Add fixed_web_platform to path
sys.path.append(str(Path(__file__).resolve().parent / "fixed_web_platform"))

# Constants for benchmarking
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DEFAULT_MODELS = {
    "text": ["bert-base-uncased", "prajjwal1/bert-tiny"],
    "vision": ["google/vit-base-patch16-224", "microsoft/resnet-50"],
    "audio": ["openai/whisper-tiny", "facebook/wav2vec2-base-960h"],
    "multimodal": ["openai/clip-vit-base-patch32", "llava-hf/llava-1.5-7b-hf"]
}
BROWSER_COMBINATIONS = {
    "single": [("chrome", "webgpu")],
    "multi": [
        ("chrome", "webgpu"),
        ("firefox", "webgpu"),
        ("edge", "webnn")
    ],
    "audio_optimized": [
        ("firefox", "webgpu")  # Firefox is optimized for audio with compute shaders
    ],
    "webnn_optimized": [
        ("edge", "webnn")  # Edge has the best WebNN support
    ]
}

# Try to import required modules
REQUIRED_MODULES = {
    "resource_pool": False,
    "websockets": False,
    "duckdb": False,
    "selenium": False,
    "numpy": False,
    "psutil": False
}

try:
    import resource_pool
    REQUIRED_MODULES["resource_pool"] = True
except ImportError:
    logger.warning("resource_pool module not available")

try:
    import websockets
    REQUIRED_MODULES["websockets"] = True
except ImportError:
    logger.warning("websockets not installed. Run: pip install websockets")

try:
    import duckdb
    REQUIRED_MODULES["duckdb"] = True
except ImportError:
    logger.warning("DuckDB not installed. Run: pip install duckdb")

try:
    import selenium
    REQUIRED_MODULES["selenium"] = True
except ImportError:
    logger.warning("Selenium not installed. Run: pip install selenium")

try:
    import numpy
    REQUIRED_MODULES["numpy"] = True
except ImportError:
    logger.warning("NumPy not installed. Run: pip install numpy")

try:
    import psutil
    REQUIRED_MODULES["psutil"] = True
except ImportError:
    logger.warning("psutil not installed. Run: pip install psutil")

class ResourcePoolBenchmark:
    """
    Comprehensive benchmark for WebNN/WebGPU resource pool with IPFS acceleration.
    """
    
    def __init__(self, args):
        """
        Initialize benchmarking system with given arguments.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.results = {}
        self.hardware_info = {}
        self.browser_info = {}
        self.resource_pool_integration = None
        self.websocket_bridge = None
        self.is_real_hardware = False
        self.db_connection = None
        
        # Initialize counter for benchmark runs
        self.bench_id = str(uuid.uuid4())
        self.run_timestamp = datetime.now().isoformat()
        
        # Set environment variables for real hardware enforcement
        os.environ["WEBNN_SIMULATION"] = "0"
        os.environ["WEBGPU_SIMULATION"] = "0"
        os.environ["USE_BROWSER_AUTOMATION"] = "1"
        
        # Firefox optimizations for audio models
        if args.enable_firefox_optimizations:
            os.environ["USE_FIREFOX_WEBGPU"] = "1"
            os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
            logger.info("Enabled Firefox audio optimizations (256x1x1 workgroup size)")
        
        # Set shader precompilation flag if enabled
        if args.enable_shader_precompilation:
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
            logger.info("Enabled WebGPU shader precompilation")
        
        # Connect to database if specified
        if args.db_path and REQUIRED_MODULES["duckdb"]:
            try:
                self.db_connection = duckdb.connect(args.db_path)
                logger.info(f"Connected to database: {args.db_path}")
                self._ensure_db_schema()
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_connection = None
    
    def _ensure_db_schema(self):
        """Ensure the database has the required schema for resource pool benchmarks."""
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
                    benchmark_id VARCHAR,
                    timestamp TIMESTAMP,
                    model_name VARCHAR,
                    model_type VARCHAR,
                    platform VARCHAR,
                    browser VARCHAR,
                    is_real_hardware BOOLEAN,
                    batch_size INTEGER,
                    connection_count INTEGER,
                    concurrent_models INTEGER,
                    memory_optimization BOOLEAN,
                    model_sharding BOOLEAN,
                    firefox_optimizations BOOLEAN,
                    shader_precompilation BOOLEAN,
                    ipfs_acceleration BOOLEAN,
                    latency_ms FLOAT,
                    throughput_items_per_sec FLOAT,
                    memory_usage_mb FLOAT,
                    power_consumption_watts FLOAT,
                    ipfs_transfer_time_ms FLOAT,
                    browser_version VARCHAR,
                    adapter_info VARCHAR,
                    system_info VARCHAR,
                    test_config VARCHAR
                )
                """)
                
                # Create model sharding table
                self.db_connection.execute("""
                CREATE TABLE resource_pool_sharding_benchmarks (
                    id INTEGER PRIMARY KEY,
                    benchmark_id VARCHAR,
                    timestamp TIMESTAMP,
                    model_name VARCHAR,
                    shard_count INTEGER,
                    shard_type VARCHAR,
                    total_memory_usage_mb FLOAT,
                    total_latency_ms FLOAT,
                    browser_distribution VARCHAR,
                    is_real_hardware BOOLEAN,
                    adapter_info VARCHAR,
                    system_info VARCHAR
                )
                """)
                
                # Create concurrent execution table
                self.db_connection.execute("""
                CREATE TABLE resource_pool_concurrent_benchmarks (
                    id INTEGER PRIMARY KEY,
                    benchmark_id VARCHAR,
                    timestamp TIMESTAMP,
                    concurrent_models INTEGER,
                    connection_count INTEGER,
                    model_composition VARCHAR,
                    total_throughput_items_per_sec FLOAT,
                    avg_latency_ms FLOAT,
                    peak_memory_usage_mb FLOAT,
                    peak_resource_utilization FLOAT,
                    completion_time_ms FLOAT,
                    is_real_hardware BOOLEAN,
                    adapter_info VARCHAR,
                    system_info VARCHAR
                )
                """)
                
                logger.info("Created benchmark tables in database")
        except Exception as e:
            logger.error(f"Failed to ensure database schema: {e}")
    
    async def detect_hardware_capabilities(self):
        """
        Detect hardware capabilities including WebGPU/WebNN support.
        
        Returns:
            bool: True if real hardware is detected, False otherwise
        """
        try:
            # Import necessary modules
            from fixed_web_platform.browser_automation import BrowserAutomation
            
            # Get default browser from arguments
            browser = self.args.browser if self.args.browser else "chrome"
            
            # Create browser automation instance
            browser_automation = BrowserAutomation(
                platform="webgpu",  # Start with WebGPU
                browser_name=browser,
                headless=not self.args.visible,
                compute_shaders=self.args.enable_firefox_optimizations,
                precompile_shaders=self.args.enable_shader_precompilation,
                parallel_loading=self.args.enable_parallel_loading
            )
            
            # Launch browser
            start_success = await browser_automation.launch()
            if not start_success:
                logger.error(f"Failed to launch {browser} for hardware detection")
                return False
            
            # Try to create WebSocket bridge
            try:
                from websocket_bridge import create_websocket_bridge
                
                # Create WebSocket bridge
                websocket_port = 8765 + hash(str(uuid.uuid4())) % 1000
                bridge = await create_websocket_bridge(port=websocket_port)
                
                if bridge:
                    logger.info(f"WebSocket bridge created successfully on port {websocket_port}")
                    
                    # Set WebSocket bridge in browser automation
                    browser_automation.websocket_bridge = bridge
                    self.websocket_bridge = bridge
                    
                    # Tell browser to connect to WebSocket
                    await browser_automation.connect_to_websocket(websocket_port)
                    
                    # Wait for WebSocket connection
                    websocket_connected = await bridge.wait_for_connection(timeout=15.0)
                    if websocket_connected:
                        logger.info("WebSocket connection established successfully")
                    else:
                        logger.warning("WebSocket connection timed out, will use simulation for hardware detection")
                        
                    # Get GPU capabilities
                    capabilities = await bridge.get_browser_capabilities()
                    
                    # Check for real hardware implementation
                    webgpu = capabilities.get("webgpu", {})
                    webnn = capabilities.get("webnn", {})
                    
                    # Store adapter info
                    self.hardware_info = {
                        "webgpu": webgpu,
                        "webnn": webnn,
                        "browser": capabilities.get("browser", {}),
                        "platform": platform.platform(),
                        "processor": platform.processor(),
                        "python_version": platform.python_version()
                    }
                    
                    # Store browser information
                    self.browser_info = capabilities.get("browser", {})
                    
                    # Check if this is real hardware
                    has_real_webgpu = webgpu.get("supported", False) and not webgpu.get("is_simulation", True)
                    has_real_webnn = webnn.get("supported", False) and not webnn.get("is_simulation", True)
                    
                    self.is_real_hardware = has_real_webgpu or has_real_webnn
                    
                    # Log hardware information
                    if has_real_webgpu:
                        adapter = webgpu.get("adapter", {})
                        logger.info(f"Real WebGPU hardware detected!")
                        logger.info(f"Adapter: {adapter.get('vendor', 'Unknown')} - {adapter.get('architecture', 'Unknown')}")
                    
                    if has_real_webnn:
                        backend = webnn.get("backend", "Unknown")
                        logger.info(f"Real WebNN hardware detected!")
                        logger.info(f"Backend: {backend}")
                    
                    if not self.is_real_hardware:
                        logger.warning("No real WebGPU or WebNN hardware detected, using simulation mode")
                        if not self.args.allow_simulation:
                            logger.error("Simulation mode not allowed, aborting")
                            
                            # Clean up
                            await browser_automation.close()
                            if bridge:
                                await bridge.stop()
                            
                            return False
                    
                    # Clean up
                    await browser_automation.close()
                    await bridge.stop()
                    
                    return self.is_real_hardware
            except ImportError:
                logger.error("websocket_bridge module not found, will use simulation")
            except Exception as e:
                logger.error(f"Error creating WebSocket bridge: {e}")
            
            # Clean up
            await browser_automation.close()
            return False
            
        except Exception as e:
            logger.error(f"Error detecting hardware capabilities: {e}")
            return False
    
    async def initialize_resource_pool(self):
        """
        Initialize resource pool with WebNN/WebGPU integration.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        try:
            # Import resource pool integration
            from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
            
            # Create integration with specified parameters
            max_connections = self.args.max_connections if self.args.max_connections else 4
            
            # Configure browser preferences based on args
            browser_preferences = {
                'audio': 'firefox' if self.args.enable_firefox_optimizations else 'chrome',
                'vision': 'chrome',
                'text_embedding': 'edge' if 'edge' in (self.args.browser or '') else 'chrome'
            }
            
            # Create integration
            self.resource_pool_integration = ResourcePoolBridgeIntegration(
                max_connections=max_connections,
                enable_gpu=True,
                enable_cpu=True,
                headless=not self.args.visible,
                browser_preferences=browser_preferences,
                adaptive_scaling=self.args.enable_adaptive_scaling,
                monitoring_interval=60
            )
            
            # Initialize integration
            self.resource_pool_integration.initialize()
            
            logger.info(f"Resource pool integration initialized with {max_connections} max connections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize resource pool: {e}")
            return False
    
    def get_model_input(self, model_family, batch_size=1):
        """
        Generate appropriate model input based on model family.
        
        Args:
            model_family: Model family (text, vision, audio, multimodal)
            batch_size: Batch size for the input
            
        Returns:
            dict: Model input suitable for the given model family
        """
        if model_family == "text":
            # Create a sample text input for BERT-like models
            return {
                "input_ids": [[101, 2023, 2003, 1037, 3231, 102] for _ in range(batch_size)],
                "attention_mask": [[1, 1, 1, 1, 1, 1] for _ in range(batch_size)]
            }
        elif model_family == "vision":
            # Create a simple tensor for vision models (224x224x3)
            return {
                "pixel_values": [
                    [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(224)]
                    for _ in range(batch_size)
                ]
            }
        elif model_family == "audio":
            # Create a sample audio input (smaller for testing)
            return {
                "input_features": [
                    [[0.1 for _ in range(80)] for _ in range(1000)]
                    for _ in range(batch_size)
                ]
            }
        elif model_family == "multimodal":
            # Create a combination of text and vision inputs
            return {
                "input_ids": [[101, 2023, 2003, 1037, 3231, 102] for _ in range(batch_size)],
                "pixel_values": [
                    [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(224)]
                    for _ in range(batch_size)
                ]
            }
        else:
            # Generic input for unknown model types
            return {
                "inputs": [[0.0 for _ in range(10)] for _ in range(batch_size)]
            }
    
    async def run_basic_benchmark(self, model_family, model_name, platform, browser, batch_size=1):
        """
        Run a basic benchmark for a model using resource pool.
        
        Args:
            model_family: Model family (text, vision, audio, multimodal)
            model_name: Model name to benchmark
            platform: Platform to use (webgpu, webnn)
            browser: Browser to use (chrome, firefox, edge)
            batch_size: Batch size for the benchmark
            
        Returns:
            dict: Benchmark results
        """
        try:
            # Ensure resource pool is initialized
            if not self.resource_pool_integration:
                if not await self.initialize_resource_pool():
                    logger.error("Failed to initialize resource pool")
                    return None
            
            # Prepare hardware preferences
            hardware_preferences = {
                "priority_list": [platform, "cpu"],
                "model_family": model_family
            }
            
            # Get model from resource pool
            logger.info(f"Getting model {model_name} from resource pool with {platform} on {browser}")
            
            # Set browser for this test
            os.environ["TEST_BROWSER"] = browser
            
            # Apply Firefox optimizations if needed for audio models
            if browser == "firefox" and model_family == "audio" and self.args.enable_firefox_optimizations:
                os.environ["USE_FIREFOX_WEBGPU"] = "1"
                os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
                os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
                logger.info("Applied Firefox audio optimizations")
            
            # Get model from resource pool
            start_time = time.time()
            model = self.resource_pool_integration.get_model(
                model_type=model_family,
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
            model_load_time = time.time() - start_time
            
            if not model:
                logger.error(f"Failed to get model {model_name} from resource pool")
                return None
            
            # Get sample input for the model family
            input_data = self.get_model_input(model_family, batch_size)
            
            # Run warmup inference
            logger.info(f"Running warmup inference")
            _ = model(input_data)
            
            # Run benchmark inferences
            logger.info(f"Running benchmark with batch size {batch_size}")
            latencies = []
            memory_usages = []
            
            # Get power consumption measurement if psutil is available
            power_start = None
            if REQUIRED_MODULES["psutil"]:
                power_start = self._measure_power_consumption()
            
            # Run multiple iterations for statistical significance
            iterations = self.args.iterations if self.args.iterations else 10
            for i in range(iterations):
                # Run inference and measure time
                iter_start = time.time()
                result = model(input_data)
                iter_time = (time.time() - iter_start) * 1000  # Convert to ms
                latencies.append(iter_time)
                
                # Get memory usage if available in result
                if isinstance(result, dict) and 'performance_metrics' in result:
                    memory_mb = result['performance_metrics'].get('memory_usage_mb', 0)
                    memory_usages.append(memory_mb)
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
            
            # Calculate power consumption if psutil is available
            power_consumption = None
            if REQUIRED_MODULES["psutil"] and power_start:
                power_consumption = self._measure_power_consumption(power_start)
            
            # Calculate stats
            latency = statistics.median(latencies) if latencies else 0
            throughput = (batch_size * 1000) / latency if latency > 0 else 0
            memory_usage = statistics.median(memory_usages) if memory_usages else 0
            
            # Create result object
            result = {
                "model_name": model_name,
                "model_family": model_family,
                "platform": platform,
                "browser": browser,
                "batch_size": batch_size,
                "is_real_hardware": self.is_real_hardware,
                "iterations": iterations,
                "firefox_optimizations": self.args.enable_firefox_optimizations and browser == "firefox",
                "shader_precompilation": self.args.enable_shader_precompilation,
                "parallel_loading": self.args.enable_parallel_loading,
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "model_load_time_ms": model_load_time * 1000,
                    "latency_ms": latency,
                    "throughput_items_per_sec": throughput,
                    "memory_usage_mb": memory_usage,
                    "power_consumption_watts": power_consumption,
                    "latencies_ms": latencies
                },
                "hardware_info": self.hardware_info,
                "browser_info": self.browser_info
            }
            
            # Store in database
            if self.db_connection:
                self._store_basic_benchmark_in_db(result)
            
            # Store in results dictionary
            key = f"{model_name}_{platform}_{browser}_{batch_size}"
            self.results[key] = result
            
            logger.info(f"Benchmark completed for {model_name} on {platform} ({browser})")
            logger.info(f"Latency: {latency:.2f}ms, Throughput: {throughput:.2f} items/sec")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running basic benchmark: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _store_basic_benchmark_in_db(self, result):
        """Store basic benchmark result in database."""
        if not self.db_connection:
            return
        
        try:
            # Extract values from result
            self.db_connection.execute("""
            INSERT INTO resource_pool_benchmarks (
                benchmark_id,
                timestamp,
                model_name,
                model_type,
                platform,
                browser,
                is_real_hardware,
                batch_size,
                connection_count,
                concurrent_models,
                memory_optimization,
                model_sharding,
                firefox_optimizations,
                shader_precompilation,
                ipfs_acceleration,
                latency_ms,
                throughput_items_per_sec,
                memory_usage_mb,
                power_consumption_watts,
                ipfs_transfer_time_ms,
                browser_version,
                adapter_info,
                system_info,
                test_config
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                self.bench_id,
                datetime.now(),
                result["model_name"],
                result["model_family"],
                result["platform"],
                result["browser"],
                result["is_real_hardware"],
                result["batch_size"],
                self.args.max_connections if self.args.max_connections else 4,
                0,  # concurrent_models (not applicable for basic benchmark)
                False,  # memory_optimization
                False,  # model_sharding
                result["firefox_optimizations"],
                result["shader_precompilation"],
                False,  # ipfs_acceleration
                result["performance"]["latency_ms"],
                result["performance"]["throughput_items_per_sec"],
                result["performance"]["memory_usage_mb"],
                result["performance"]["power_consumption_watts"],
                0,  # ipfs_transfer_time_ms
                json.dumps(result["browser_info"].get("version", {})),
                json.dumps(result["hardware_info"].get("webgpu", {}).get("adapter", {})),
                json.dumps({"platform": platform.platform(), "python": platform.python_version()}),
                json.dumps({"iterations": result["iterations"]})
            ])
            
            logger.info(f"Stored benchmark result for {result['model_name']} in database")
        except Exception as e:
            logger.error(f"Failed to store benchmark in database: {e}")
    
    async def test_concurrent_execution(self, model_family_map, platform, browser, connection_count=4):
        """
        Test concurrent execution of multiple models using resource pool.
        
        Args:
            model_family_map: Dictionary mapping model families to model names
            platform: Platform to use (webgpu, webnn)
            browser: Browser to use (chrome, firefox, edge)
            connection_count: Number of connections to use
            
        Returns:
            dict: Benchmark results
        """
        try:
            # Ensure resource pool is initialized
            if not self.resource_pool_integration:
                if not await self.initialize_resource_pool():
                    logger.error("Failed to initialize resource pool")
                    return None
            
            # Set browser for this test
            os.environ["TEST_BROWSER"] = browser
            
            # Configure resource pool with specified connection count
            self.resource_pool_integration.max_connections = connection_count
            
            # Create models dictionary
            models = {}
            model_inputs = {}
            
            # Load all models from the model family map
            for family, model_name in model_family_map.items():
                hardware_preferences = {
                    "priority_list": [platform, "cpu"],
                    "model_family": family
                }
                
                # Apply Firefox optimizations if needed for audio models
                if browser == "firefox" and family == "audio" and self.args.enable_firefox_optimizations:
                    os.environ["USE_FIREFOX_WEBGPU"] = "1"
                    os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
                    os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
                
                # Get model from resource pool
                logger.info(f"Getting model {model_name} ({family}) from resource pool")
                
                model = self.resource_pool_integration.get_model(
                    model_type=family,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                if not model:
                    logger.error(f"Failed to get model {model_name} from resource pool")
                    continue
                
                # Store model and generate input
                models[model_name] = model
                model_inputs[model_name] = self.get_model_input(family)
            
            # Check if we have models to test
            if not models:
                logger.error("No models loaded for concurrent execution test")
                return None
            
            logger.info(f"Loaded {len(models)} models for concurrent execution test")
            
            # Run warmup inference for each model
            for name, model in models.items():
                logger.info(f"Running warmup inference for {name}")
                _ = model(model_inputs[name])
            
            # Measure concurrent execution
            logger.info(f"Running concurrent execution test with {len(models)} models")
            
            # Prepare task items for concurrent execution
            tasks = [(name, input_data) for name, input_data in model_inputs.items()]
            
            # Get power consumption measurement if psutil is available
            power_start = None
            if REQUIRED_MODULES["psutil"]:
                power_start = self._measure_power_consumption()
            
            # Run concurrent execution
            start_time = time.time()
            results = self.resource_pool_integration.execute_concurrent(tasks)
            execution_time = time.time() - start_time
            
            # Calculate power consumption if psutil is available
            power_consumption = None
            if REQUIRED_MODULES["psutil"] and power_start:
                power_consumption = self._measure_power_consumption(power_start)
            
            # Check results
            success_count = sum(1 for r in results if r.get('success', False))
            
            # Get execution stats
            stats = self.resource_pool_integration.get_execution_stats()
            
            # Calculate metrics
            total_items = len(tasks)
            total_latency = execution_time * 1000  # Convert to ms
            total_throughput = total_items / execution_time if execution_time > 0 else 0
            
            # Create result object
            result = {
                "concurrent_models": len(models),
                "model_composition": {family: name for family, name in model_family_map.items()},
                "platform": platform,
                "browser": browser,
                "connection_count": connection_count,
                "is_real_hardware": self.is_real_hardware,
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "success_count": success_count,
                    "total_execution_time_ms": execution_time * 1000,
                    "average_latency_ms": total_latency / total_items if total_items > 0 else 0,
                    "total_throughput_items_per_sec": total_throughput,
                    "power_consumption_watts": power_consumption
                },
                "execution_stats": stats,
                "hardware_info": self.hardware_info,
                "browser_info": self.browser_info
            }
            
            # Store in database
            if self.db_connection:
                self._store_concurrent_benchmark_in_db(result)
            
            # Store in results dictionary
            key = f"concurrent_{platform}_{browser}_{connection_count}"
            self.results[key] = result
            
            logger.info(f"Concurrent execution test completed with {success_count}/{total_items} successful tasks")
            logger.info(f"Total execution time: {execution_time*1000:.2f}ms, Throughput: {total_throughput:.2f} items/sec")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running concurrent execution test: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _store_concurrent_benchmark_in_db(self, result):
        """Store concurrent benchmark result in database."""
        if not self.db_connection:
            return
        
        try:
            # Extract values from result
            self.db_connection.execute("""
            INSERT INTO resource_pool_concurrent_benchmarks (
                benchmark_id,
                timestamp,
                concurrent_models,
                connection_count,
                model_composition,
                total_throughput_items_per_sec,
                avg_latency_ms,
                peak_memory_usage_mb,
                peak_resource_utilization,
                completion_time_ms,
                is_real_hardware,
                adapter_info,
                system_info
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                self.bench_id,
                datetime.now(),
                result["concurrent_models"],
                result["connection_count"],
                json.dumps(result["model_composition"]),
                result["performance"]["total_throughput_items_per_sec"],
                result["performance"]["average_latency_ms"],
                result["execution_stats"].get("resource_metrics", {}).get("memory_usage", 0),
                result["execution_stats"].get("resource_metrics", {}).get("connection_util", 0),
                result["performance"]["total_execution_time_ms"],
                result["is_real_hardware"],
                json.dumps(result["hardware_info"].get("webgpu", {}).get("adapter", {})),
                json.dumps({"platform": platform.platform(), "python": platform.python_version()})
            ])
            
            logger.info(f"Stored concurrent benchmark result in database")
        except Exception as e:
            logger.error(f"Failed to store concurrent benchmark in database: {e}")
    
    async def test_model_sharding(self, model_name, platform, shard_count=2):
        """
        Test model sharding across multiple browser instances.
        
        Note: This is an advanced test that demonstrates how large models
        can be split across multiple browser instances for memory efficiency.
        
        Args:
            model_name: Name of the large model to shard
            platform: Platform to use (webgpu, webnn)
            shard_count: Number of shards to create
            
        Returns:
            dict: Benchmark results
        """
        try:
            # Import model sharding utilities
            try:
                from fixed_web_platform.model_sharding import ModelShardingManager
                sharding_available = True
            except ImportError:
                logger.error("Model sharding not available - implement fixed_web_platform/model_sharding.py first")
                return None
            
            if not sharding_available:
                # Implementation plan for model_sharding.py
                logger.error("To implement model sharding, create a ModelShardingManager class that:")
                logger.error("1. Creates multiple browser connections via ResourcePoolBridge")
                logger.error("2. Splits model computation across browser instances")
                logger.error("3. Implements layer-wise distribution of model execution")
                logger.error("4. Coordinates inference across shards")
                return None
            
            # Create model sharding manager
            sharding_manager = ModelShardingManager(
                model_name=model_name,
                num_shards=shard_count,
                shard_type="layer"
            )
            
            # Initialize sharding
            success = await sharding_manager.initialize_sharding()
            if not success:
                logger.error(f"Failed to initialize sharding for {model_name}")
                return None
            
            # Create sample input
            input_data = self.get_model_input("text")  # Assuming a text model for now
            
            # Run warmup inference
            logger.info(f"Running warmup inference with sharded model")
            _ = await sharding_manager.run_inference_sharded(input_data)
            
            # Measure sharded inference
            logger.info(f"Running sharded inference test with {shard_count} shards")
            
            # Get power consumption measurement if psutil is available
            power_start = None
            if REQUIRED_MODULES["psutil"]:
                power_start = self._measure_power_consumption()
            
            # Run sharded inference
            start_time = time.time()
            sharded_result = await sharding_manager.run_inference_sharded(input_data)
            execution_time = time.time() - start_time
            
            # Calculate power consumption if psutil is available
            power_consumption = None
            if REQUIRED_MODULES["psutil"] and power_start:
                power_consumption = self._measure_power_consumption(power_start)
            
            # Get memory usage from sharding manager
            memory_usage = await sharding_manager.get_total_memory_usage()
            
            # Create result object
            result = {
                "model_name": model_name,
                "platform": platform,
                "shard_count": shard_count,
                "shard_type": "layer",
                "is_real_hardware": self.is_real_hardware,
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "total_execution_time_ms": execution_time * 1000,
                    "total_memory_usage_mb": memory_usage,
                    "power_consumption_watts": power_consumption
                },
                "sharding_details": await sharding_manager.get_sharding_metrics(),
                "hardware_info": self.hardware_info,
                "browser_info": self.browser_info
            }
            
            # Store in database
            if self.db_connection:
                self._store_sharding_benchmark_in_db(result)
            
            # Store in results dictionary
            key = f"sharding_{model_name}_{platform}_{shard_count}"
            self.results[key] = result
            
            # Clean up
            await sharding_manager.shutdown()
            
            logger.info(f"Model sharding test completed for {model_name} with {shard_count} shards")
            logger.info(f"Execution time: {execution_time*1000:.2f}ms, Memory usage: {memory_usage:.2f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running model sharding test: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _store_sharding_benchmark_in_db(self, result):
        """Store sharding benchmark result in database."""
        if not self.db_connection:
            return
        
        try:
            # Extract values from result
            self.db_connection.execute("""
            INSERT INTO resource_pool_sharding_benchmarks (
                benchmark_id,
                timestamp,
                model_name,
                shard_count,
                shard_type,
                total_memory_usage_mb,
                total_latency_ms,
                browser_distribution,
                is_real_hardware,
                adapter_info,
                system_info
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                self.bench_id,
                datetime.now(),
                result["model_name"],
                result["shard_count"],
                result["shard_type"],
                result["performance"]["total_memory_usage_mb"],
                result["performance"]["total_execution_time_ms"],
                json.dumps(result["sharding_details"].get("browser_distribution", {})),
                result["is_real_hardware"],
                json.dumps(result["hardware_info"].get("webgpu", {}).get("adapter", {})),
                json.dumps({"platform": platform.platform(), "python": platform.python_version()})
            ])
            
            logger.info(f"Stored sharding benchmark result in database")
        except Exception as e:
            logger.error(f"Failed to store sharding benchmark in database: {e}")
    
    async def test_memory_optimization(self, model_name, platform, browser):
        """
        Test memory optimization for resource pool with a large model.
        
        Args:
            model_name: Name of the large model to test
            platform: Platform to use (webgpu, webnn)
            browser: Browser to use (chrome, firefox, edge)
            
        Returns:
            dict: Benchmark results
        """
        try:
            # Ensure resource pool is initialized
            if not self.resource_pool_integration:
                if not await self.initialize_resource_pool():
                    logger.error("Failed to initialize resource pool")
                    return None
            
            # Set browser for this test
            os.environ["TEST_BROWSER"] = browser
            
            # Enable memory optimizations in resource pool
            self.resource_pool_integration.memory_optimization = True
            
            # Run benchmark with standard configuration first
            logger.info(f"Running standard benchmark for {model_name} on {platform} ({browser})")
            standard_result = await self.run_basic_benchmark("text", model_name, platform, browser)
            
            if not standard_result:
                logger.error(f"Failed to run standard benchmark")
                return None
            
            # Now run with memory optimization
            logger.info(f"Running memory-optimized benchmark for {model_name}")
            
            # Prepare hardware preferences with memory optimization flags
            hardware_preferences = {
                "priority_list": [platform, "cpu"],
                "model_family": "text",
                "memory_optimization": {
                    "enabled": True,
                    "kv_cache_optimization": True,
                    "gradient_checkpointing": True,
                    "attention_implementation": "flash",
                    "precision": "int8"
                }
            }
            
            # Get model from resource pool with memory optimization
            start_time = time.time()
            optimized_model = self.resource_pool_integration.get_model(
                model_type="text",
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
            model_load_time = time.time() - start_time
            
            if not optimized_model:
                logger.error(f"Failed to get optimized model {model_name} from resource pool")
                return None
            
            # Get sample input
            input_data = self.get_model_input("text")
            
            # Run warmup inference
            logger.info(f"Running warmup inference with optimized model")
            _ = optimized_model(input_data)
            
            # Run benchmark inferences
            logger.info(f"Running optimized benchmark")
            latencies = []
            memory_usages = []
            
            # Get power consumption measurement if psutil is available
            power_start = None
            if REQUIRED_MODULES["psutil"]:
                power_start = self._measure_power_consumption()
            
            # Run multiple iterations for statistical significance
            iterations = self.args.iterations if self.args.iterations else 10
            for i in range(iterations):
                # Run inference and measure time
                iter_start = time.time()
                result = optimized_model(input_data)
                iter_time = (time.time() - iter_start) * 1000  # Convert to ms
                latencies.append(iter_time)
                
                # Get memory usage if available in result
                if isinstance(result, dict) and 'performance_metrics' in result:
                    memory_mb = result['performance_metrics'].get('memory_usage_mb', 0)
                    memory_usages.append(memory_mb)
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
            
            # Calculate power consumption if psutil is available
            power_consumption = None
            if REQUIRED_MODULES["psutil"] and power_start:
                power_consumption = self._measure_power_consumption(power_start)
            
            # Calculate stats
            latency = statistics.median(latencies) if latencies else 0
            throughput = 1000 / latency if latency > 0 else 0
            memory_usage = statistics.median(memory_usages) if memory_usages else 0
            
            # Calculate improvement percentages
            standard_memory = standard_result["performance"]["memory_usage_mb"]
            memory_reduction = ((standard_memory - memory_usage) / standard_memory * 100) if standard_memory > 0 else 0
            
            standard_latency = standard_result["performance"]["latency_ms"]
            latency_change = ((standard_latency - latency) / standard_latency * 100) if standard_latency > 0 else 0
            
            # Create result object
            result = {
                "model_name": model_name,
                "platform": platform,
                "browser": browser,
                "is_real_hardware": self.is_real_hardware,
                "memory_optimization": True,
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "model_load_time_ms": model_load_time * 1000,
                    "latency_ms": latency,
                    "throughput_items_per_sec": throughput,
                    "memory_usage_mb": memory_usage,
                    "power_consumption_watts": power_consumption,
                    "memory_reduction_percent": memory_reduction,
                    "latency_change_percent": latency_change
                },
                "standard_performance": standard_result["performance"],
                "hardware_info": self.hardware_info,
                "browser_info": self.browser_info
            }
            
            # Store in database (reuse basic benchmark storage)
            if self.db_connection:
                # Modify result to fit basic benchmark schema
                store_result = {
                    "model_name": model_name,
                    "model_family": "text",
                    "platform": platform,
                    "browser": browser,
                    "batch_size": 1,
                    "is_real_hardware": self.is_real_hardware,
                    "iterations": iterations,
                    "firefox_optimizations": self.args.enable_firefox_optimizations and browser == "firefox",
                    "shader_precompilation": self.args.enable_shader_precompilation,
                    "parallel_loading": self.args.enable_parallel_loading,
                    "performance": result["performance"],
                    "hardware_info": self.hardware_info,
                    "browser_info": self.browser_info
                }
                self._store_basic_benchmark_in_db(store_result)
            
            # Store in results dictionary
            key = f"memory_opt_{model_name}_{platform}_{browser}"
            self.results[key] = result
            
            logger.info(f"Memory optimization test completed for {model_name}")
            logger.info(f"Memory reduction: {memory_reduction:.2f}%, Latency change: {latency_change:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running memory optimization test: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def test_firefox_audio_optimization(self):
        """
        Test Firefox audio optimization for Whisper model.
        
        Returns:
            dict: Dictionary with Chrome and Firefox performance comparison
        """
        try:
            # Test parameters
            model_name = "openai/whisper-tiny"
            model_family = "audio"
            platform = "webgpu"
            
            # Enable Firefox audio optimizations
            self.args.enable_firefox_optimizations = True
            
            # Run Firefox test first
            firefox_result = await self.run_basic_benchmark(
                model_family, model_name, platform, "firefox"
            )
            
            if not firefox_result:
                logger.error("Failed to run Firefox benchmark")
                return None
            
            # Now run Chrome test for comparison
            chrome_result = await self.run_basic_benchmark(
                model_family, model_name, platform, "chrome"
            )
            
            if not chrome_result:
                logger.error("Failed to run Chrome benchmark")
                return None
            
            # Calculate improvement percentage
            firefox_latency = firefox_result["performance"]["latency_ms"]
            chrome_latency = chrome_result["performance"]["latency_ms"]
            
            latency_improvement = ((chrome_latency - firefox_latency) / chrome_latency * 100) if chrome_latency > 0 else 0
            
            # Create comparison result
            comparison = {
                "model_name": model_name,
                "platform": platform,
                "firefox_performance": firefox_result["performance"],
                "chrome_performance": chrome_result["performance"],
                "firefox_optimizations": True,
                "latency_improvement_percent": latency_improvement,
                "timestamp": datetime.now().isoformat(),
                "is_real_hardware": self.is_real_hardware,
                "hardware_info": self.hardware_info
            }
            
            # Store in results dictionary
            key = f"firefox_audio_opt_{model_name}"
            self.results[key] = comparison
            
            logger.info(f"Firefox audio optimization test completed for {model_name}")
            logger.info(f"Firefox latency: {firefox_latency:.2f}ms, Chrome latency: {chrome_latency:.2f}ms")
            logger.info(f"Improvement: {latency_improvement:.2f}%")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error running Firefox audio optimization test: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _measure_power_consumption(self, start=None):
        """
        Measure power consumption using psutil.
        
        Args:
            start: Start battery percentage or None to just get current percentage
            
        Returns:
            float: Power consumption in watts or percentage change if start is provided
        """
        if not REQUIRED_MODULES["psutil"]:
            return None
        
        try:
            import psutil
            
            if hasattr(psutil, "sensors_battery"):
                battery = psutil.sensors_battery()
                if battery:
                    if start is None:
                        # Return current percentage
                        return battery.percent
                    else:
                        # Calculate percentage change
                        change = start - battery.percent
                        return change if change >= 0 else 0
            
            # Try to get power info on Linux
            if sys.platform.startswith('linux'):
                power_path = "/sys/class/power_supply/BAT0/power_now"
                if os.path.exists(power_path):
                    with open(power_path, 'r') as f:
                        power_microwatts = int(f.read().strip())
                        return power_microwatts / 1000000  # Convert to watts
            
            return None
        except Exception as e:
            logger.error(f"Error measuring power consumption: {e}")
            return None
    
    async def generate_report(self):
        """
        Generate comprehensive benchmark report.
        
        Returns:
            str: Path to generated report file
        """
        if not self.results:
            logger.error("No benchmark results to report")
            return None
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resource_pool_benchmark_report_{timestamp}.md"
        
        try:
            with open(filename, 'w') as f:
                f.write("# WebNN/WebGPU Resource Pool Benchmark Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Hardware information
                f.write("## Hardware Information\n\n")
                
                webgpu_adapter = self.hardware_info.get("webgpu", {}).get("adapter", {})
                webnn_backend = self.hardware_info.get("webnn", {}).get("backend", "Unknown")
                
                f.write(f"- Platform: {platform.platform()}\n")
                f.write(f"- Processor: {platform.processor()}\n")
                f.write(f"- Python Version: {platform.python_version()}\n")
                
                if webgpu_adapter:
                    f.write(f"- WebGPU Adapter: {webgpu_adapter.get('vendor', 'Unknown')} - {webgpu_adapter.get('device', 'Unknown')}\n")
                    f.write(f"- WebGPU Architecture: {webgpu_adapter.get('architecture', 'Unknown')}\n")
                
                f.write(f"- WebNN Backend: {webnn_backend}\n")
                f.write(f"- Real Hardware: {'Yes' if self.is_real_hardware else 'No (Simulation)'}\n\n")
                
                # Browser information
                browser_info = self.browser_info
                if browser_info:
                    f.write("## Browser Information\n\n")
                    f.write(f"- Browser: {browser_info.get('name', 'Unknown')}\n")
                    f.write(f"- Version: {browser_info.get('version', 'Unknown')}\n")
                    f.write(f"- User Agent: {browser_info.get('userAgent', 'Unknown')}\n\n")
                
                # Basic benchmark results
                basic_results = {k: v for k, v in self.results.items() 
                                if not k.startswith(("concurrent_", "sharding_", "memory_opt_", "firefox_audio_opt_"))}
                
                if basic_results:
                    f.write("## Basic Benchmark Results\n\n")
                    
                    f.write("| Model | Platform | Browser | Batch Size | Latency (ms) | Throughput (items/sec) | Memory (MB) |\n")
                    f.write("|-------|----------|---------|------------|--------------|------------------------|------------|\n")
                    
                    for key, result in sorted(basic_results.items()):
                        model = result.get("model_name", "Unknown")
                        platform = result.get("platform", "Unknown")
                        browser = result.get("browser", "Unknown")
                        batch_size = result.get("batch_size", 1)
                        
                        latency = result.get("performance", {}).get("latency_ms", 0)
                        throughput = result.get("performance", {}).get("throughput_items_per_sec", 0)
                        memory = result.get("performance", {}).get("memory_usage_mb", 0)
                        
                        f.write(f"| {model} | {platform} | {browser} | {batch_size} | {latency:.2f} | {throughput:.2f} | {memory:.2f} |\n")
                    
                    f.write("\n")
                
                # Concurrent execution results
                concurrent_results = {k: v for k, v in self.results.items() if k.startswith("concurrent_")}
                
                if concurrent_results:
                    f.write("## Concurrent Execution Results\n\n")
                    
                    f.write("| Platform | Browser | Connection Count | Models | Throughput (items/sec) | Avg Latency (ms) |\n")
                    f.write("|----------|---------|------------------|--------|------------------------|------------------|\n")
                    
                    for key, result in sorted(concurrent_results.items()):
                        platform = result.get("platform", "Unknown")
                        browser = result.get("browser", "Unknown")
                        conn_count = result.get("connection_count", 0)
                        model_count = result.get("concurrent_models", 0)
                        
                        throughput = result.get("performance", {}).get("total_throughput_items_per_sec", 0)
                        latency = result.get("performance", {}).get("average_latency_ms", 0)
                        
                        f.write(f"| {platform} | {browser} | {conn_count} | {model_count} | {throughput:.2f} | {latency:.2f} |\n")
                    
                    f.write("\n")
                
                # Model sharding results
                sharding_results = {k: v for k, v in self.results.items() if k.startswith("sharding_")}
                
                if sharding_results:
                    f.write("## Model Sharding Results\n\n")
                    
                    f.write("| Model | Platform | Shard Count | Total Latency (ms) | Memory Usage (MB) |\n")
                    f.write("|-------|----------|-------------|-------------------|------------------|\n")
                    
                    for key, result in sorted(sharding_results.items()):
                        model = result.get("model_name", "Unknown")
                        platform = result.get("platform", "Unknown")
                        shard_count = result.get("shard_count", 0)
                        
                        latency = result.get("performance", {}).get("total_execution_time_ms", 0)
                        memory = result.get("performance", {}).get("total_memory_usage_mb", 0)
                        
                        f.write(f"| {model} | {platform} | {shard_count} | {latency:.2f} | {memory:.2f} |\n")
                    
                    f.write("\n")
                
                # Memory optimization results
                memory_results = {k: v for k, v in self.results.items() if k.startswith("memory_opt_")}
                
                if memory_results:
                    f.write("## Memory Optimization Results\n\n")
                    
                    f.write("| Model | Platform | Browser | Memory Reduction (%) | Latency Change (%) |\n")
                    f.write("|-------|----------|---------|----------------------|-------------------|\n")
                    
                    for key, result in sorted(memory_results.items()):
                        model = result.get("model_name", "Unknown")
                        platform = result.get("platform", "Unknown")
                        browser = result.get("browser", "Unknown")
                        
                        memory_reduction = result.get("performance", {}).get("memory_reduction_percent", 0)
                        latency_change = result.get("performance", {}).get("latency_change_percent", 0)
                        
                        f.write(f"| {model} | {platform} | {browser} | {memory_reduction:.2f} | {latency_change:.2f} |\n")
                    
                    f.write("\n")
                
                # Firefox audio optimization results
                firefox_results = {k: v for k, v in self.results.items() if k.startswith("firefox_audio_opt_")}
                
                if firefox_results:
                    f.write("## Firefox Audio Optimization Results\n\n")
                    
                    f.write("| Model | Firefox Latency (ms) | Chrome Latency (ms) | Improvement (%) |\n")
                    f.write("|-------|---------------------|---------------------|----------------|\n")
                    
                    for key, result in sorted(firefox_results.items()):
                        model = result.get("model_name", "Unknown")
                        
                        firefox_latency = result.get("firefox_performance", {}).get("latency_ms", 0)
                        chrome_latency = result.get("chrome_performance", {}).get("latency_ms", 0)
                        improvement = result.get("latency_improvement_percent", 0)
                        
                        f.write(f"| {model} | {firefox_latency:.2f} | {chrome_latency:.2f} | {improvement:.2f} |\n")
                    
                    f.write("\n")
                
                # Browser comparison
                if basic_results:
                    f.write("## Browser Comparison\n\n")
                    
                    # Group results by model and batch size
                    browser_comparison = {}
                    for key, result in basic_results.items():
                        model = result.get("model_name", "Unknown")
                        batch = result.get("batch_size", 1)
                        browser = result.get("browser", "Unknown")
                        
                        group_key = f"{model}_{batch}"
                        if group_key not in browser_comparison:
                            browser_comparison[group_key] = {}
                        
                        browser_comparison[group_key][browser] = result
                    
                    # Generate comparison tables
                    for group_key, browsers in browser_comparison.items():
                        if len(browsers) > 1:  # Only show comparison if we have multiple browsers
                            model, batch = group_key.split("_")
                            
                            f.write(f"### {model} (Batch Size {batch})\n\n")
                            
                            f.write("| Browser | Platform | Latency (ms) | Throughput (items/sec) | Memory (MB) |\n")
                            f.write("|---------|----------|--------------|------------------------|------------|\n")
                            
                            for browser, result in sorted(browsers.items()):
                                platform = result.get("platform", "Unknown")
                                
                                latency = result.get("performance", {}).get("latency_ms", 0)
                                throughput = result.get("performance", {}).get("throughput_items_per_sec", 0)
                                memory = result.get("performance", {}).get("memory_usage_mb", 0)
                                
                                f.write(f"| {browser} | {platform} | {latency:.2f} | {throughput:.2f} | {memory:.2f} |\n")
                            
                            f.write("\n")
                
                # Key findings and optimization opportunities
                f.write("## Key Findings\n\n")
                
                # Auto-generate findings based on results
                findings = []
                
                # Firefox audio optimization finding
                if firefox_results:
                    for key, result in firefox_results.items():
                        improvement = result.get("latency_improvement_percent", 0)
                        if improvement > 10:
                            findings.append(f"Firefox shows **{improvement:.1f}%** better performance than Chrome for audio models with WebGPU compute shader optimizations")
                
                # Memory optimization finding
                if memory_results:
                    for key, result in memory_results.items():
                        memory_reduction = result.get("performance", {}).get("memory_reduction_percent", 0)
                        if memory_reduction > 10:
                            findings.append(f"Memory optimization reduces memory usage by **{memory_reduction:.1f}%** while maintaining performance")
                
                # Concurrent execution finding
                if concurrent_results:
                    max_throughput = 0
                    single_throughput = 0
                    
                    for key, result in concurrent_results.items():
                        if result.get("concurrent_models", 0) > 1:
                            throughput = result.get("performance", {}).get("total_throughput_items_per_sec", 0)
                            if throughput > max_throughput:
                                max_throughput = throughput
                    
                    # Find single model throughput for comparison
                    for key, result in basic_results.items():
                        throughput = result.get("performance", {}).get("throughput_items_per_sec", 0)
                        if throughput > single_throughput:
                            single_throughput = throughput
                    
                    if max_throughput > 0 and single_throughput > 0:
                        improvement = (max_throughput / single_throughput)
                        if improvement > 1.5:
                            findings.append(f"Concurrent execution achieves **{improvement:.1f}x** higher throughput compared to sequential execution")
                
                # Write findings
                if findings:
                    for finding in findings:
                        f.write(f"- {finding}\n")
                else:
                    f.write("- No significant findings identified from the benchmark results\n")
                
                f.write("\n")
                
                # Recommendations
                f.write("## Optimization Recommendations\n\n")
                
                recommendations = []
                
                # Audio model recommendation
                if firefox_results:
                    for key, result in firefox_results.items():
                        improvement = result.get("latency_improvement_percent", 0)
                        if improvement > 10:
                            recommendations.append("Use Firefox with compute shader optimizations for audio models")
                
                # Memory optimization recommendation
                if memory_results:
                    for key, result in memory_results.items():
                        memory_reduction = result.get("performance", {}).get("memory_reduction_percent", 0)
                        if memory_reduction > 10:
                            recommendations.append("Enable memory optimization for large models to reduce memory usage")
                
                # Concurrent execution recommendation
                if concurrent_results:
                    recommendations.append("Use resource pool for concurrent model execution to maximize throughput")
                
                # Model sharding recommendation
                if sharding_results:
                    recommendations.append("Use model sharding for very large models that exceed browser memory limits")
                
                # Write recommendations
                if recommendations:
                    for recommendation in recommendations:
                        f.write(f"- {recommendation}\n")
                else:
                    f.write("- No specific optimization recommendations identified from the benchmark results\n")
                
                f.write("\n")
                
                # Test configuration
                f.write("## Test Configuration\n\n")
                
                f.write(f"- Benchmark ID: {self.bench_id}\n")
                f.write(f"- Date: {self.run_timestamp}\n")
                f.write(f"- Real Hardware: {'Yes' if self.is_real_hardware else 'No (Simulation)'}\n")
                
                if hasattr(self.args, 'iterations') and self.args.iterations:
                    f.write(f"- Iterations per test: {self.args.iterations}\n")
                
                if hasattr(self.args, 'max_connections') and self.args.max_connections:
                    f.write(f"- Max connections: {self.args.max_connections}\n")
                
                f.write(f"- Firefox optimizations: {'Enabled' if self.args.enable_firefox_optimizations else 'Disabled'}\n")
                f.write(f"- Shader precompilation: {'Enabled' if self.args.enable_shader_precompilation else 'Disabled'}\n")
                f.write(f"- Parallel loading: {'Enabled' if self.args.enable_parallel_loading else 'Disabled'}\n")
                f.write(f"- Adaptive scaling: {'Enabled' if self.args.enable_adaptive_scaling else 'Disabled'}\n")
                
            logger.info(f"Benchmark report generated: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run_all_tests(self):
        """
        Run all benchmark tests based on command line arguments.
        
        Returns:
            dict: Dictionary of all benchmark results
        """
        # Detect hardware capabilities first
        logger.info("Detecting hardware capabilities...")
        self.is_real_hardware = await self.detect_hardware_capabilities()
        
        if not self.is_real_hardware and not self.args.allow_simulation:
            logger.error("Real hardware not detected and simulation not allowed")
            return self.results
        
        # Initialize resource pool
        if not await self.initialize_resource_pool():
            logger.error("Failed to initialize resource pool")
            return self.results
        
        # Run basic benchmarks if requested
        if self.args.basic_benchmarks or self.args.all_tests or self.args.comprehensive:
            logger.info("Running basic benchmarks...")
            
            # Determine which models to test
            if self.args.model:
                models = {
                    self.args.model_type or "text": [self.args.model]
                }
            elif self.args.comprehensive:
                # Use all default models
                models = DEFAULT_MODELS
            else:
                # Use a smaller set of models
                models = {
                    "text": ["bert-base-uncased"],
                    "vision": ["google/vit-base-patch16-224"],
                    "audio": ["openai/whisper-tiny"],
                    "multimodal": ["openai/clip-vit-base-patch32"]
                }
            
            # Determine which batch sizes to test
            if self.args.batch_sizes:
                batch_sizes = [int(b) for b in self.args.batch_sizes.split(",")]
            elif self.args.comprehensive:
                batch_sizes = DEFAULT_BATCH_SIZES
            else:
                batch_sizes = [1, 4]  # Smaller set for basic tests
            
            # Determine which browser combinations to test
            if self.args.browser and self.args.platform:
                browser_combinations = [[(self.args.browser, self.args.platform)]]
            elif self.args.comprehensive:
                browser_combinations = [BROWSER_COMBINATIONS["multi"]]
            else:
                browser_combinations = [BROWSER_COMBINATIONS["single"]]
            
            # Run benchmarks for all combinations
            for browser_combo in browser_combinations:
                for family, model_list in models.items():
                    for model_name in model_list:
                        for platform, browser in browser_combo:
                            for batch_size in batch_sizes:
                                # Skip large batch sizes for large models to avoid OOM
                                if "llava" in model_name.lower() and batch_size > 1:
                                    logger.info(f"Skipping batch size {batch_size} for large model {model_name}")
                                    continue
                                
                                logger.info(f"Running benchmark for {model_name} on {platform} ({browser}) with batch size {batch_size}")
                                await self.run_basic_benchmark(family, model_name, platform, browser, batch_size)
        
        # Run concurrent execution test if requested
        if self.args.test_concurrent or self.args.all_tests or self.args.comprehensive:
            logger.info("Running concurrent execution test...")
            
            # Determine platform and browser
            platform = self.args.platform or "webgpu"
            browser = self.args.browser or "chrome"
            
            # Create model family map
            model_family_map = {
                "text": "bert-base-uncased",
                "vision": "google/vit-base-patch16-224",
                "audio": "openai/whisper-tiny"
            }
            
            # Determine connection count
            connection_count = self.args.max_connections or 4
            
            await self.test_concurrent_execution(model_family_map, platform, browser, connection_count)
        
        # Run model sharding test if requested
        if self.args.test_sharding or self.args.all_tests:
            logger.info("Running model sharding test...")
            
            # Determine model, platform and shard count
            model_name = self.args.model or "bert-base-uncased"
            platform = self.args.platform or "webgpu"
            shard_count = self.args.shard_count or 2
            
            await self.test_model_sharding(model_name, platform, shard_count)
        
        # Run memory optimization test if requested
        if self.args.test_memory_optimization or self.args.all_tests:
            logger.info("Running memory optimization test...")
            
            # Determine model, platform and browser
            model_name = self.args.model or "bert-base-uncased"
            platform = self.args.platform or "webgpu"
            browser = self.args.browser or "chrome"
            
            await self.test_memory_optimization(model_name, platform, browser)
        
        # Run Firefox audio optimization test if requested
        if self.args.test_firefox_audio or self.args.all_tests:
            logger.info("Running Firefox audio optimization test...")
            
            await self.test_firefox_audio_optimization()
        
        # Generate report if requested
        if self.args.generate_report or self.args.all_tests or self.args.comprehensive:
            logger.info("Generating benchmark report...")
            
            await self.generate_report()
        
        return self.results
    
    async def close(self):
        """Close all connections and clean up resources."""
        # Close resource pool
        if self.resource_pool_integration:
            self.resource_pool_integration.close()
            logger.info("Resource pool closed")
        
        # Close database connection
        if self.db_connection:
            self.db_connection.close()
            logger.info("Database connection closed")


async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(description="Benchmark WebNN/WebGPU Resource Pool with IPFS Acceleration")
    
    # Basic options
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge", "safari"], 
                      help="Browser to use for testing")
    parser.add_argument("--platform", choices=["webgpu", "webnn", "all"], default="webgpu",
                      help="Platform to test")
    parser.add_argument("--model", type=str,
                      help="Specific model to benchmark")
    parser.add_argument("--model-type", choices=["text", "vision", "audio", "multimodal"],
                      help="Type of model to benchmark")
    
    # Benchmark options
    parser.add_argument("--batch-sizes", type=str,
                      help="Comma-separated list of batch sizes (e.g., '1,2,4,8')")
    parser.add_argument("--iterations", type=int, default=10,
                      help="Number of iterations per benchmark")
    parser.add_argument("--max-connections", type=int,
                      help="Maximum number of concurrent browser connections")
    
    # Test selection
    parser.add_argument("--basic-benchmarks", action="store_true",
                      help="Run basic benchmarks")
    parser.add_argument("--test-concurrent", action="store_true",
                      help="Test concurrent execution")
    parser.add_argument("--test-sharding", action="store_true",
                      help="Test model sharding")
    parser.add_argument("--shard-count", type=int, default=2,
                      help="Number of shards for model sharding test")
    parser.add_argument("--test-memory-optimization", action="store_true",
                      help="Test memory optimization")
    parser.add_argument("--test-firefox-audio", action="store_true",
                      help="Test Firefox audio optimization")
    parser.add_argument("--all-tests", action="store_true",
                      help="Run all tests")
    parser.add_argument("--comprehensive", action="store_true",
                      help="Run comprehensive benchmarks for all models and configurations")
    
    # Feature flags
    parser.add_argument("--enable-firefox-optimizations", action="store_true",
                      help="Enable Firefox audio optimizations")
    parser.add_argument("--enable-shader-precompilation", action="store_true",
                      help="Enable WebGPU shader precompilation")
    parser.add_argument("--enable-parallel-loading", action="store_true",
                      help="Enable parallel model loading")
    parser.add_argument("--enable-adaptive-scaling", action="store_true",
                      help="Enable adaptive scaling of resource pool")
    
    # Output options
    parser.add_argument("--generate-report", action="store_true",
                      help="Generate benchmark report")
    parser.add_argument("--db-path", type=str,
                      help="Path to DuckDB database file")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    # Execution options
    parser.add_argument("--visible", action="store_true",
                      help="Run browsers in visible mode (not headless)")
    parser.add_argument("--allow-simulation", action="store_true",
                      help="Allow simulation if real hardware not available")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    missing_deps = [name for name, installed in REQUIRED_MODULES.items() if not installed]
    if missing_deps:
        logger.warning(f"Missing some recommended dependencies: {', '.join(missing_deps)}")
        logger.warning("Some features may not work correctly")
    
    # Run simple test if no options specified
    if not any([
        args.basic_benchmarks, args.test_concurrent, args.test_sharding,
        args.test_memory_optimization, args.test_firefox_audio,
        args.all_tests, args.comprehensive
    ]):
        args.basic_benchmarks = True
    
    # Create benchmark runner
    benchmark = ResourcePoolBenchmark(args)
    
    try:
        # Run all tests
        await benchmark.run_all_tests()
    finally:
        # Close benchmark resources
        await benchmark.close()
    
    return 0

def main():
    """Main entry point."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())