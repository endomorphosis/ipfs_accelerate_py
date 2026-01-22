#!/usr/bin/env python3
"""
Test WebNN/WebGPU Resource Pool Integration

This script tests the resource pool integration with WebNN and WebGPU implementations,
including the enhanced connection pooling and parallel model execution capabilities.

Usage:
    python test_web_resource_pool.py --models bert,vit,whisper
    python test_web_resource_pool.py --concurrent-models
    python test_web_resource_pool.py --stress-test
    python test_web_resource_pool.py --test-enhanced  # Test enhanced implementation (May 2025)
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import required modules
try:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    RESOURCE_POOL_AVAILABLE = True
except ImportError as e:
    logger.error(f"ResourcePoolBridge not available: {e}")
    RESOURCE_POOL_AVAILABLE = False
except Exception as e:
    logger.error(f"Error loading ResourcePoolBridge: {e}")
    RESOURCE_POOL_AVAILABLE = False

# Import enhanced implementation (May 2025)
try:
    from fixed_web_platform.resource_pool_integration_enhanced import EnhancedResourcePoolIntegration
    from fixed_web_platform.enhanced_resource_pool_tester import EnhancedWebResourcePoolTester
    ENHANCED_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced Resource Pool Integration not available: {e}")
    ENHANCED_INTEGRATION_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error loading Enhanced Resource Pool Integration: {e}")
    ENHANCED_INTEGRATION_AVAILABLE = False

# Create a mock ResourcePoolBridgeIntegration if the real one is not available
if not RESOURCE_POOL_AVAILABLE:
    logger.warning("Creating mock ResourcePoolBridgeIntegration for testing")
    
# Create a mock EnhancedResourcePoolIntegration if the real one is not available
if not ENHANCED_INTEGRATION_AVAILABLE:
    logger.warning("Creating mock EnhancedResourcePoolIntegration for testing")
    
    class MockEnhancedResourcePoolIntegration:
        def __init__(self, **kwargs):
            self.max_connections = kwargs.get('max_connections', 4)
            self.min_connections = kwargs.get('min_connections', 1)
            self.enable_gpu = kwargs.get('enable_gpu', True)
            self.enable_cpu = kwargs.get('enable_cpu', True)
            self.headless = kwargs.get('headless', True)
            self.adaptive_scaling = kwargs.get('adaptive_scaling', True)
            self.use_connection_pool = kwargs.get('use_connection_pool', True)
            self.db_path = kwargs.get('db_path', None)
            self.initialized = False
            self.models = {}
            self.metrics = {
                "model_load_time": {},
                "inference_time": {},
                "memory_usage": {},
                "throughput": {},
                "latency": {},
                "batch_size": {},
                "platform_distribution": {"webgpu": 0, "webnn": 0, "cpu": 0},
                "browser_distribution": {"chrome": 0, "firefox": 0, "edge": 0, "safari": 0}
            }
            
        async def initialize(self):
            self.initialized = True
            logger.info("Mock EnhancedResourcePoolIntegration initialized")
            return True
            
        async def get_model(self, model_name, model_type=None, platform="webgpu", **kwargs):
            model_id = f"{model_type}:{model_name}"
            
            # Create a mock model
            model = MockModel(model_id, model_type, model_name)
            
            # Update metrics
            self.metrics["model_load_time"][model_name] = 0.1
            self.metrics["platform_distribution"][platform] = self.metrics["platform_distribution"].get(platform, 0) + 1
            self.metrics["browser_distribution"]["chrome"] = self.metrics["browser_distribution"].get("chrome", 0) + 1
            
            return model
        
        async def execute_concurrent(self, models_and_inputs):
            results = []
            for model, inputs in models_and_inputs:
                if hasattr(model, '__call__'):
                    results.append(model(inputs))
                else:
                    results.append({"error": "Invalid model", "success": False})
            return results
        
        def get_metrics(self):
            return self.metrics
            
        async def close(self):
            self.initialized = False
            logger.info("Mock EnhancedResourcePoolIntegration closed")
            
        def store_acceleration_result(self, result):
            logger.info(f"Mock storing result for {result.get('model_name', 'unknown')}")
            return True
    
    EnhancedResourcePoolIntegration = MockEnhancedResourcePoolIntegration
    ENHANCED_INTEGRATION_AVAILABLE = True
    
    class MockModel:
        def __init__(self, model_id, model_type, model_name):
            self.model_id = model_id
            self.model_type = model_type
            self.model_name = model_name
        
        def __call__(self, inputs):
            logger.info(f"Mock inference on {self.model_name}")
            return {
                'success': True,
                'status': 'success',
                'model_id': self.model_id,
                'model_name': self.model_name,
                'is_real_implementation': False,
                'ipfs_accelerated': False,
                'browser': 'mock',
                'platform': 'mock',
                'metrics': {
                    'latency_ms': 100,
                    'throughput_items_per_sec': 10,
                    'memory_usage_mb': 100
                }
            }
    
    class MockResourcePoolBridgeIntegration:
        def __init__(self, **kwargs):
            self.initialized = False
            self.models = {}
            self.db_connection = None
            
        def initialize(self):
            self.initialized = True
            logger.info("Mock ResourcePoolBridgeIntegration initialized")
            
        def get_model(self, model_type, model_name, **kwargs):
            model_id = f"{model_type}:{model_name}"
            model = MockModel(model_id, model_type, model_name)
            self.models[model_id] = model
            return model
            
        def execute_concurrent(self, models_and_inputs):
            results = []
            for model_id, inputs in models_and_inputs:
                if model_id in self.models:
                    model = self.models[model_id]
                else:
                    # Create a model on the fly
                    parts = model_id.split(":")
                    model_type = parts[0] if len(parts) > 1 else "unknown"
                    model_name = parts[-1]
                    model = MockModel(model_id, model_type, model_name)
                
                results.append(model(inputs))
            return results
        
        def get_execution_stats(self):
            return {
                'executed_tasks': len(self.models),
                'current_queue_size': 0,
                'bridge_stats': {
                    'created_connections': 1,
                    'current_connections': 1,
                    'peak_connections': 1,
                    'loaded_models': len(self.models)
                },
                'resource_metrics': {
                    'connection_util': 0.5,
                    'browser_usage': {
                        'chrome': 1,
                        'firefox': 0,
                        'edge': 0,
                        'safari': 0
                    }
                }
            }
            
        def close(self):
            self.initialized = False
            logger.info("Mock ResourcePoolBridgeIntegration closed")
            
        def store_acceleration_result(self, result):
            logger.info(f"Mock storing result for {result.get('model_name', 'unknown')}")
            return True
    
    ResourcePoolBridgeIntegration = MockResourcePoolBridgeIntegration
    RESOURCE_POOL_AVAILABLE = True

def verify_database_schema(db_path):
    """
    Verify and if necessary create the required database schema for test results.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        True if schema is valid, False otherwise
    """
    if not db_path:
        logger.warning("No database path provided, skipping schema verification")
        return False
    
    # Try to import DuckDB
    try:
        import duckdb
    except ImportError:
        logger.error("DuckDB not installed, cannot verify schema")
        return False
    
    # Connect to database
    try:
        conn = duckdb.connect(db_path)
        
        # Check if required tables exist
        table_check = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name IN ('webnn_webgpu_results', 'resource_pool_test_results', 
        'browser_connection_metrics')
        """).fetchone()[0]
        
        # Create tables if they don't exist
        if table_check < 3:
            logger.info("Creating missing tables in database")
            
            # WebNN/WebGPU results table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS webnn_webgpu_results (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                model_name VARCHAR,
                model_type VARCHAR,
                platform VARCHAR,
                browser VARCHAR,
                is_real_implementation BOOLEAN,
                is_simulation BOOLEAN,
                precision INTEGER,
                mixed_precision BOOLEAN,
                ipfs_accelerated BOOLEAN,
                ipfs_cache_hit BOOLEAN,
                compute_shader_optimized BOOLEAN,
                precompile_shaders BOOLEAN,
                parallel_loading BOOLEAN,
                latency_ms FLOAT,
                throughput_items_per_sec FLOAT,
                memory_usage_mb FLOAT,
                energy_efficiency_score FLOAT,
                adapter_info JSON,
                system_info JSON,
                details JSON
            )
            """)
            
            # Resource pool test results table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_pool_test_results (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                total_tests INTEGER,
                successful_tests INTEGER,
                ipfs_accelerated_count INTEGER,
                ipfs_cache_hits INTEGER,
                real_implementations INTEGER,
                test_duration_seconds FLOAT,
                summary JSON,
                detailed_results JSON
            )
            """)
            
            # Browser connection metrics table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS browser_connection_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                browser_name VARCHAR,
                platform VARCHAR,
                connection_id VARCHAR,
                connection_duration_sec FLOAT,
                models_executed INTEGER,
                total_inference_time_sec FLOAT,
                error_count INTEGER,
                connection_success BOOLEAN,
                heartbeat_failures INTEGER,
                browser_version VARCHAR,
                adapter_info JSON,
                backend_info JSON
            )
            """)
            
            # Add indexes for faster querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_model_name ON webnn_webgpu_results(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_browser ON webnn_webgpu_results(browser)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_platform ON webnn_webgpu_results(platform)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_timestamp ON webnn_webgpu_results(timestamp)")
            
            logger.info("Database schema created successfully")
            
        # Validate schema
        try:
            # Validate webnn_webgpu_results table
            conn.execute("SELECT id, timestamp, model_name, browser, ipfs_accelerated FROM webnn_webgpu_results LIMIT 0")
            
            # Validate resource_pool_test_results table
            conn.execute("SELECT id, timestamp, total_tests, ipfs_accelerated_count FROM resource_pool_test_results LIMIT 0")
            
            # Validate browser_connection_metrics table
            conn.execute("SELECT id, timestamp, browser_name, connection_id FROM browser_connection_metrics LIMIT 0")
            
            logger.info("Database schema validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Database schema validation failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return False


class WebResourcePoolTester:
    """Test WebNN/WebGPU Resource Pool Integration"""
    
    def __init__(self, args):
        """Initialize tester with command line arguments"""
        self.args = args
        self.integration = None
        self.results = []
        self.error_retries = args.error_retry if hasattr(args, 'error_retry') else 1
        
        # Configure logging level
        if hasattr(args, 'verbose') and args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Verbose logging enabled")
        
        # Set environment variables for optimizations
        if args.compute_shaders:
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
        
        if args.shader_precompile:
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
        
        if args.parallel_loading:
            os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
        
        # Set precision environment variables
        if hasattr(args, 'precision'):
            os.environ["WEBGPU_PRECISION_BITS"] = str(args.precision)
            logger.info(f"Using {args.precision}-bit precision")
            
        # Verify database schema if requested
        if hasattr(args, 'verify_db_schema') and args.verify_db_schema and hasattr(args, 'db_path') and args.db_path:
            if verify_database_schema(args.db_path):
                logger.info(f"Database schema verified: {args.db_path}")
            else:
                logger.warning(f"Database schema verification failed: {args.db_path}")
                if hasattr(args, 'verbose') and args.verbose:
                    logger.warning("Continuing with tests despite schema verification failure...")
    
    async def initialize(self):
        """Initialize resource pool integration with IPFS acceleration"""
        if not RESOURCE_POOL_AVAILABLE:
            logger.error("Cannot initialize: ResourcePoolBridge not available")
            return False
        
        try:
            # Configure browser preferences with optimization settings
            browser_preferences = {
                'audio': 'firefox',  # Firefox has better compute shader performance for audio
                'vision': 'chrome',  # Chrome has good WebGPU support for vision models
                'text_embedding': 'edge'  # Edge has excellent WebNN support for text embeddings
            }
            
            # Override browser preferences if specific browser is selected
            if hasattr(self.args, 'firefox') and self.args.firefox:
                browser_preferences = {k: 'firefox' for k in browser_preferences}
            elif hasattr(self.args, 'chrome') and self.args.chrome:
                browser_preferences = {k: 'chrome' for k in browser_preferences}
            elif hasattr(self.args, 'edge') and self.args.edge:
                browser_preferences = {k: 'edge' for k in browser_preferences}
            
            # Determine IPFS acceleration setting
            enable_ipfs = not (hasattr(self.args, 'disable_ipfs') and self.args.disable_ipfs)
            
            # Create ResourcePoolBridgeIntegration instance with IPFS acceleration
            self.integration = ResourcePoolBridgeIntegration(
                max_connections=self.args.max_connections,
                enable_gpu=True,
                enable_cpu=True,
                headless=not self.args.visible,
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_ipfs=enable_ipfs,  # Set IPFS acceleration based on command-line flag
                db_path=self.args.db_path if hasattr(self.args, 'db_path') else None,
                enable_heartbeat=True
            )
            
            # Initialize integration
            self.integration.initialize()
            
            # Log initialization status with enabled features
            features = []
            if enable_ipfs:
                features.append("IPFS Acceleration")
            
            if hasattr(self.args, 'compute_shaders') and self.args.compute_shaders:
                features.append("Compute Shaders")
            
            if hasattr(self.args, 'shader_precompile') and self.args.shader_precompile:
                features.append("Shader Precompilation")
            
            if hasattr(self.args, 'parallel_loading') and self.args.parallel_loading:
                features.append("Parallel Loading")
            
            if hasattr(self.args, 'mixed_precision') and self.args.mixed_precision:
                features.append("Mixed Precision")
            
            # Database storage
            if self.args.db_path:
                features.append(f"Database Storage ({self.args.db_path})")
            
            feature_str = ", ".join(features) if features else "No advanced features"
            logger.info(f"ResourcePoolBridgeIntegration initialized successfully with: {feature_str}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_model(self, model_type, model_name, platform='webgpu'):
        """Test a model using the resource pool integration with error handling and retries"""
        if not self.integration:
            logger.error("Cannot test model: integration not initialized")
            return None
        
        # Track retries
        retry_count = 0
        max_retries = self.error_retries
        browser_specific_errors = []
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    logger.warning(f"Retry {retry_count}/{max_retries} for model {model_name}")
                
                logger.info(f"Testing model: {model_name} ({model_type}) on {platform}")
                
                # Configure hardware preferences with IPFS acceleration
                hardware_preferences = {
                    'priority_list': [platform, 'cpu'],
                    'model_family': model_type,
                    'enable_ipfs': not getattr(self.args, 'disable_ipfs', False),
                    'precision': getattr(self.args, 'precision', 16),
                    'mixed_precision': getattr(self.args, 'mixed_precision', False)
                }
                
                # Add browser-specific optimizations
                self._add_browser_optimizations(hardware_preferences, model_type, platform)
                
                if hasattr(self.args, 'verbose') and self.args.verbose:
                    logger.debug(f"Hardware preferences: {hardware_preferences}")
                
                # Get model from resource pool
                start_time_loading = time.time()
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                if not model:
                    if retry_count < max_retries:
                        logger.error(f"Failed to get model: {model_name}, retrying...")
                        retry_count += 1
                        # Wait longer between retries
                        await asyncio.sleep(0.5 * (retry_count + 1))
                        continue
                    else:
                        logger.error(f"Failed to get model after {max_retries} retries: {model_name}")
                        return self._create_error_result(model_name, model_type, platform,
                        "Failed to load model", browser_specific_errors)
                
                loading_time = time.time() - start_time_loading
                logger.info(f"Model loaded in {loading_time:.2f}s: {model_name}")
                
                # Prepare test input based on model type
                try:
                    test_input = self._create_test_input_for_model(model_type)
                except Exception as input_error:
                    logger.error(f"Error creating test input: {input_error}")
                    if retry_count < max_retries:
                        retry_count += 1
                        continue
                    else:
                        return self._create_error_result(model_name, model_type, platform,
                        f"Error creating test input: {input_error}",
                        browser_specific_errors)
                
                # Run inference with timeout protection
                start_time = time.time()
                try:
                    # Set a reasonable timeout based on model type
                    timeout = 60.0  # 1 minute default timeout
                    if model_type == 'audio':
                        timeout = 120.0  # Audio models may take longer
                    
                    # If we're doing mixed precision or ultra-low bit (2/3/4) inference, extend timeout
                    if hardware_preferences['mixed_precision'] or hardware_preferences['precision'] <= 4:
                        timeout *= 2  # Double the timeout
                    
                    # In verbose mode, log the timeout
                    if hasattr(self.args, 'verbose') and self.args.verbose:
                        logger.debug(f"Running inference with timeout: {timeout}s")
                    
                    # Use asyncio.wait_for to add timeout protection
                    try:
                        # Since model() is synchronous, wrap in a thread to make it awaitable
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: model(test_input)),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"Inference timeout after {timeout}s for {model_name}")
                        if retry_count < max_retries:
                            retry_count += 1
                            continue
                        else:
                            return self._create_error_result(model_name, model_type, platform,
                            f"Inference timeout after {timeout}s",
                            browser_specific_errors)
                            
                except Exception as inference_error:
                    logger.error(f"Inference error for {model_name}: {inference_error}")
                    browser_specific_errors.append(str(inference_error))
                    
                    if retry_count < max_retries:
                        retry_count += 1
                        await asyncio.sleep(0.5 * (retry_count + 1))
                        continue
                    else:
                        return self._create_error_result(model_name, model_type, platform,
                        f"Inference error: {inference_error}",
                        browser_specific_errors)
                
                execution_time = time.time() - start_time
                
                # Verify result format
                if not isinstance(result, dict):
                    logger.error(f"Invalid result format from model {model_name}: {type(result)}")
                    if retry_count < max_retries:
                        retry_count += 1
                        continue
                    else:
                        return self._create_error_result(model_name, model_type, platform,
                        "Invalid result format", browser_specific_errors)
                
                # Check for success
                success = result.get('success', result.get('status') == 'success')
                if not success:
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"Model inference failed: {error_msg}")
                    browser_specific_errors.append(error_msg)
                    
                    if retry_count < max_retries:
                        retry_count += 1
                        continue
                    else:
                        return self._create_error_result(model_name, model_type, platform,
                        f"Inference failed: {error_msg}",
                        browser_specific_errors)
                
                # Extract performance metrics
                performance_metrics = result.get('metrics', result.get('performance_metrics', {}))
                
                # Log browser and acceleration information
                browser_name = result.get('browser', 'unknown')
                is_real = result.get('is_real_implementation', False)
                ipfs_accelerated = result.get('ipfs_accelerated', False)
                ipfs_cache_hit = result.get('ipfs_cache_hit', False)
                precision = result.get('precision', hardware_preferences['precision'])
                mixed_precision = result.get('mixed_precision', hardware_preferences['mixed_precision'])
                
                logger.info(f"Test complete - Browser: {browser_name}, Real Implementation: {is_real}, "
                f"IPFS Accelerated: {ipfs_accelerated}, Cache Hit: {ipfs_cache_hit}, "
                f"Precision: {precision}-bit{' mixed' if mixed_precision else ''}")
                
                # Create comprehensive result object
                test_result = {
                    'model_name': model_name,
                    'model_type': model_type,
                    'platform': platform,
                    'loading_time': loading_time,
                    'execution_time': execution_time,
                    'total_time': loading_time + execution_time,
                    'success': success,
                    'is_real_implementation': is_real,
                    'ipfs_accelerated': ipfs_accelerated,
                    'ipfs_cache_hit': ipfs_cache_hit,
                    'browser': browser_name,
                    'precision': precision,
                    'mixed_precision': mixed_precision,
                    'compute_shader_optimized': result.get('compute_shader_optimized', False),
                    'precompile_shaders': result.get('precompile_shaders', False),
                    'parallel_loading': result.get('parallel_loading', False),
                    'retry_count': retry_count,
                    'hardware_preferences': hardware_preferences,
                    'performance_metrics': performance_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Append to results
                self.results.append(test_result)
                
                logger.info(f"Test completed in {execution_time:.2f}s (load: {loading_time:.2f}s): {model_name}")
                
                return test_result
                
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {e}")
                if hasattr(self.args, 'verbose') and self.args.verbose:
                    import traceback
                    traceback.print_exc()
                
                browser_specific_errors.append(str(e))
                
                if retry_count < max_retries:
                    retry_count += 1
                    await asyncio.sleep(0.5 * (retry_count + 1))
                else:
                    return self._create_error_result(model_name, model_type, platform, str(e), browser_specific_errors)
        
        # Should never reach here due to return in the loop
        return self._create_error_result(model_name, model_type, platform, "Unknown error", browser_specific_errors)
    
    def _add_browser_optimizations(self, hardware_preferences, model_type, platform):
        """Add browser-specific optimizations based on model type and platform"""
        # For audio models, use Firefox optimizations
        if model_type == 'audio':
            hardware_preferences['browser'] = 'firefox'
            hardware_preferences['use_firefox_optimizations'] = True
            logger.info("Using Firefox with audio optimizations for audio model")
        
        # For text models, use Edge with WebNN if available
        elif model_type == 'text_embedding' and platform == 'webnn':
            hardware_preferences['browser'] = 'edge'
            logger.info("Using Edge for text embedding model with WebNN")
        
        # For vision models, use Chrome with shader precompilation
        elif model_type == 'vision':
            hardware_preferences['browser'] = 'chrome'
            hardware_preferences['precompile_shaders'] = True
            logger.info("Using Chrome with shader precompilation for vision model")
        
        # Override with command-line browser selection
        if hasattr(self.args, 'firefox') and self.args.firefox:
            hardware_preferences['browser'] = 'firefox'
        elif hasattr(self.args, 'chrome') and self.args.chrome:
            hardware_preferences['browser'] = 'chrome'
        elif hasattr(self.args, 'edge') and self.args.edge:
            hardware_preferences['browser'] = 'edge'
    
    def _create_test_input_for_model(self, model_type):
        """Create appropriate test input based on model type"""
        if model_type == 'text_embedding':
            return {
                'input_ids': [101, 2023, 2003, 1037, 3231, 102],
                'attention_mask': [1, 1, 1, 1, 1, 1]
            }
        elif model_type == 'vision':
            return {'pixel_values': [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]}
        elif model_type == 'audio':
            return {'input_features': [[[0.1 for _ in range(80)] for _ in range(3000)]]}
        else:
            return {'inputs': [0.0 for _ in range(10)]}

    def _create_error_result(self, model_name, model_type, platform, error_message, error_history):
        """Create a result object for errors"""
        error_result = {
            'model_name': model_name,
            'model_type': model_type,
            'platform': platform,
            'success': False,
            'error': error_message,
            'error_history': error_history,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(error_result)
        logger.error(f"Test failed for {model_name}: {error_message}")
        
        return error_result
    
    async def test_concurrent_models(self, models, platform='webgpu'):
        """Test multiple models concurrently with IPFS acceleration"""
        if not self.integration:
            logger.error("Cannot test concurrent models: integration not initialized")
            return []
        
        try:
            logger.info(f"Testing {len(models)} models concurrently on {platform}")
            
            # Create models and inputs
            model_inputs = []
            model_instances = []
            model_configs = []  # Store configs for results processing
            
            for model_info in models:
                model_type, model_name = model_info
                
                # Configure hardware preferences with browser-specific optimizations
                hardware_preferences = {
                    'priority_list': [platform, 'cpu'],
                    'model_family': model_type,
                    'enable_ipfs': True,      # Enable IPFS acceleration for all models
                    'precision': 16,          # Use FP16 precision
                    'mixed_precision': False
                }
                
                # Apply model-specific optimizations
                if model_type == 'audio':
                    # Audio models work best with Firefox and compute shader optimizations
                    hardware_preferences['browser'] = 'firefox'
                    hardware_preferences['use_firefox_optimizations'] = True
                    logger.info(f"Using Firefox with audio optimizations for model: {model_name}")
                elif model_type == 'text_embedding' and platform == 'webnn':
                    # Text models work best with Edge for WebNN
                    hardware_preferences['browser'] = 'edge'
                    logger.info(f"Using Edge for text model with WebNN: {model_name}")
                elif model_type == 'vision':
                    # Vision models work well with Chrome
                    hardware_preferences['browser'] = 'chrome'
                    hardware_preferences['precompile_shaders'] = True
                    logger.info(f"Using Chrome with shader precompilation for vision model: {model_name}")
                
                # Store model config for later
                model_configs.append({
                    'type': model_type,
                    'name': model_name,
                    'platform': platform,
                    'preferences': hardware_preferences.copy()
                })
                
                # Get model from resource pool
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                
                if not model:
                    logger.error(f"Failed to get model: {model_name}")
                    continue
                
                model_instances.append(model)
                
                # Prepare test input based on model type
                if model_type == 'text_embedding':
                    test_input = {
                        'input_ids': [101, 2023, 2003, 1037, 3231, 102],
                        'attention_mask': [1, 1, 1, 1, 1, 1]
                    }
                elif model_type == 'vision':
                    test_input = {'pixel_values': [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]}
                elif model_type == 'audio':
                    test_input = {'input_features': [[[0.1 for _ in range(80)] for _ in range(3000)]]}
                else:
                    test_input = {'inputs': [0.0 for _ in range(10)]}            
                model_inputs.append((model.model_id, test_input))
            
            # Run concurrent execution
            start_time = time.time()
            results = self.integration.execute_concurrent(model_inputs)
            execution_time = time.time() - start_time
            
            # Process results
            concurrent_results = []
            for i, result in enumerate(results):
                if i < len(model_configs):
                    config = model_configs[i]
                    model_type = config['type']
                    model_name = config['name']
                    
                    # Extract performance metrics
                    performance_metrics = {}
                    if isinstance(result, dict):
                        # Extract metrics from result dictionary
                        performance_metrics = result.get('metrics', result.get('performance_metrics', {}))
                    
                    # Create enhanced result object with IPFS and browser information
                    test_result = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'platform': platform,
                        'execution_time': execution_time,
                        'success': result.get('success', result.get('status') == 'success'),
                        'is_real_implementation': result.get('is_real_implementation', False),
                        'ipfs_accelerated': result.get('ipfs_accelerated', False),
                        'ipfs_cache_hit': result.get('ipfs_cache_hit', False),
                        'browser': result.get('browser', config['preferences'].get('browser', 'unknown')),
                        'compute_shader_optimized': result.get('compute_shader_optimized', False),
                        'precompile_shaders': result.get('precompile_shaders', False),
                        'parallel_loading': result.get('parallel_loading', False),
                        'performance_metrics': performance_metrics
                    }
                    
                    # Log browser and acceleration information
                    browser_name = test_result['browser']
                    is_real = test_result['is_real_implementation']
                    ipfs_accelerated = test_result['ipfs_accelerated']
                    ipfs_cache_hit = test_result['ipfs_cache_hit']
                    
                    logger.info(f"Concurrent model {model_name} - Browser: {browser_name}, "
                        f"Real Implementation: {is_real}, IPFS Accelerated: {ipfs_accelerated}, "
                        f"Cache Hit: {ipfs_cache_hit}")
                    
                    concurrent_results.append(test_result)
                    self.results.append(test_result)
            
            # Calculate overall performance metrics
            cache_hits = sum(1 for r in concurrent_results if r.get('ipfs_cache_hit', False))
            ipfs_accelerated = sum(1 for r in concurrent_results if r.get('ipfs_accelerated', False))
            real_impl = sum(1 for r in concurrent_results if r.get('is_real_implementation', False))

            logger.info(f"Concurrent execution summary: {len(models)} models in {execution_time:.2f}s")
            logger.info(f"Performance: {len(models)/execution_time:.2f} models/second")
            logger.info(f"Stats: IPFS Acceleration: {ipfs_accelerated}/{len(models)}, "
                f"Cache Hits: {cache_hits}/{len(models)}, "
                f"Real Implementations: {real_impl}/{len(models)}")
            
            return concurrent_results
        except Exception as e:
            logger.error(f"Error in concurrent model execution: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def run_stress_test(self, duration=60, models=None):
        """Run a stress test on the resource pool with IPFS acceleration and browser-specific optimizations"""
        if not self.integration:
            logger.error("Cannot run stress test: integration not initialized")
            return
        
        if not models:
            # Default models for stress test with appropriate browsers for each model type
            models = [
                ('text_embedding', 'bert-base-uncased'),      # Best with Edge/WebNN
                ('vision', 'google/vit-base-patch16-224'),    # Best with Chrome/WebGPU
                ('audio', 'openai/whisper-tiny')              # Best with Firefox/WebGPU compute shaders
            ]
        
        try:
            logger.info(f"Starting stress test for {duration} seconds with {len(models)} models")
            logger.info("Test includes IPFS acceleration and browser-specific optimizations")
            
            # Tracking metrics
            start_time = time.time()
            end_time = start_time + duration
            
            total_executions = 0
            successful_executions = 0
            ipfs_accelerated_count = 0
            ipfs_cache_hits = 0
            real_implementations = 0
            
            # Performance metrics by model type
            perf_by_model = {}
            perf_by_browser = {
                'firefox': {'count': 0, 'time': 0, 'success': 0},
                'chrome': {'count': 0, 'time': 0, 'success': 0},
                'edge': {'count': 0, 'time': 0, 'success': 0},
                'safari': {'count': 0, 'time': 0, 'success': 0},
                'unknown': {'count': 0, 'time': 0, 'success': 0}
            }
            
            # Run continuous executions until duration expires
            while time.time() < end_time:
                # Execute models in batches
                batch_size = min(len(models), 3)  # Process up to 3 models at once
                for i in range(0, len(models), batch_size):
                    batch = models[i:i+batch_size]
                    
                    # Run models with optimized concurrent execution
                    results = await self.test_concurrent_models(batch)
                    
                    total_executions += len(batch)
                    successful_executions += sum(1 for r in results if r.get('success', False))
                    ipfs_accelerated_count += sum(1 for r in results if r.get('ipfs_accelerated', False))
                    ipfs_cache_hits += sum(1 for r in results if r.get('ipfs_cache_hit', False))
                    real_implementations += sum(1 for r in results if r.get('is_real_implementation', False))
                    
                    # Update per-model and per-browser performance stats
                    for result in results:
                        model_type = result.get('model_type')
                        browser = result.get('browser', 'unknown')
                        execution_time = result.get('execution_time', 0)
                        success = result.get('success', False)
                        
                        # Update model stats
                        if model_type not in perf_by_model:
                            perf_by_model[model_type] = {'count': 0, 'time': 0, 'success': 0}
                        
                        perf_by_model[model_type]['count'] += 1
                        perf_by_model[model_type]['time'] += execution_time
                        perf_by_model[model_type]['success'] += 1 if success else 0
                        
                        # Update browser stats
                        if browser not in perf_by_browser:
                            browser = 'unknown'
                        
                        perf_by_browser[browser]['count'] += 1
                        perf_by_browser[browser]['time'] += execution_time
                        perf_by_browser[browser]['success'] += 1 if success else 0
                    
                    # Brief pause between batches
                    await asyncio.sleep(0.1)
                
                # Get resource pool stats
                stats = self.integration.get_execution_stats()
                
                # Print progress
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                
                # Calculate current throughput
                current_throughput = total_executions / elapsed if elapsed > 0 else 0

                logger.info(f"Progress: {elapsed:.1f}s / {duration}s - "
                    f"Remaining: {remaining:.1f}s - "
                    f"Executions: {total_executions} (Success: {successful_executions})")
                logger.info(f"Current Throughput: {current_throughput:.2f} models/sec, "
                    f"IPFS Accelerated: {ipfs_accelerated_count}/{total_executions} "
                    f"({ipfs_accelerated_count/total_executions*100 if total_executions else 0:.1f}%)")
                
                # Print resource utilization
                if 'resource_metrics' in stats:
                    metrics = stats['resource_metrics']
                    logger.info(f"Resource utilization: "
                        f"Connection: {metrics.get('connection_util', 0):.2f}, "
                        f"Queue: {stats.get('current_queue_size', 0)}")
                
                # Print browser usage
                if 'resource_metrics' in stats and 'browser_usage' in stats['resource_metrics']:
                    browser_usage = stats['resource_metrics']['browser_usage']
                    logger.info("Browser usage: " + 
                        ", ".join([f"{browser}: {count}" for browser, count in browser_usage.items() if count > 0]))
            
            # Final results
            total_time = time.time() - start_time
            
            # Calculate per-model performance
            model_perf = {}
            for model_type, data in perf_by_model.items():
                count = data['count']
                time_total = data['time']
                success_count = data['success']
                
                if count > 0:
                    model_perf[model_type] = {
                        'throughput': count / time_total if time_total > 0 else 0,
                        'success_rate': success_count / count * 100,
                        'count': count
                    }
            
            # Calculate per-browser performance
            browser_perf = {}
            for browser, data in perf_by_browser.items():
                count = data['count']
                time_total = data['time']
                success_count = data['success']
                
                if count > 0:
                    browser_perf[browser] = {
                        'throughput': count / time_total if time_total > 0 else 0,
                        'success_rate': success_count / count * 100,
                        'count': count
                    }
            
            # Print complete results with detailed stats
            logger.info("=" * 80)
            logger.info(f"STRESS TEST COMPLETE - Duration: {total_time:.2f}s")
            logger.info("-" * 80)
            logger.info(f"Total Executions: {total_executions}")
            logger.info(f"Successful Executions: {successful_executions}")
            logger.info(f"Success Rate: {(successful_executions / total_executions * 100) if total_executions else 0:.1f}%")
            logger.info(f"Total Throughput: {total_executions / total_time:.2f} executions/sec")
            logger.info(f"IPFS Acceleration: {ipfs_accelerated_count}/{total_executions} ({ipfs_accelerated_count/total_executions*100 if total_executions else 0:.1f}%)")
            logger.info(f"IPFS Cache Hits: {ipfs_cache_hits}/{ipfs_accelerated_count} ({ipfs_cache_hits/ipfs_accelerated_count*100 if ipfs_accelerated_count else 0:.1f}%)")
            logger.info(f"Real Implementations: {real_implementations}/{total_executions} ({real_implementations/total_executions*100 if total_executions else 0:.1f}%)")
            
            # Per-model performance
            logger.info("-" * 80)
            logger.info("PERFORMANCE BY MODEL TYPE:")
            for model_type, perf in model_perf.items():
                logger.info(f"  {model_type}:")
                logger.info(f"    - Throughput: {perf['throughput']:.2f} models/sec")
                logger.info(f"    - Success Rate: {perf['success_rate']:.1f}%")
                logger.info(f"    - Executions: {perf['count']}")
            
            # Per-browser performance
            logger.info("-" * 80)
            logger.info("PERFORMANCE BY BROWSER:")
            for browser, perf in browser_perf.items():
                if perf['count'] > 0:
                    logger.info(f"  {browser}:")
                    logger.info(f"    - Throughput: {perf['throughput']:.2f} models/sec")
                    logger.info(f"    - Success Rate: {perf['success_rate']:.1f}%")
                    logger.info(f"    - Executions: {perf['count']}")
            
            # Get final stats
            final_stats = self.integration.get_execution_stats()
            
            # Print connection stats
            logger.info("-" * 80)
            logger.info("CONNECTION STATS:")
            if 'bridge_stats' in final_stats:
                bridge_stats = final_stats['bridge_stats']
                logger.info(f"  - Created Connections: {bridge_stats.get('created_connections', 0)}")
                logger.info(f"  - Current Connections: {bridge_stats.get('current_connections', 0)}")
                logger.info(f"  - Peak Connections: {bridge_stats.get('peak_connections', 0)}")
                logger.info(f"  - Loaded Models: {bridge_stats.get('loaded_models', 0)}")
            
            # Include resource metrics
            if 'resource_metrics' in final_stats:
                metrics = final_stats['resource_metrics']
                logger.info("-" * 80)
                logger.info("RESOURCE METRICS:")
                logger.info(f"  - Connection Utilization: {metrics.get('connection_util', 0):.2f}")
                logger.info(f"  - WebGPU Utilization: {metrics.get('webgpu_util', 0):.2f}")
                logger.info(f"  - WebNN Utilization: {metrics.get('webnn_util', 0):.2f}")
                logger.info(f"  - CPU Utilization: {metrics.get('cpu_util', 0):.2f}")
                logger.info(f"  - Memory Usage: {metrics.get('memory_usage', 0):.2f} MB")
            
            # Add optimization stats if available  
            if 'model_execution_times' in final_stats:
                logger.info("-" * 80)
                logger.info("OPTIMIZATION STATS:")
                if 'ipfs_acceleration_count' in final_stats:
                    logger.info(f"  - IPFS Acceleration Count: {final_stats.get('ipfs_acceleration_count', 0)}")
                if 'ipfs_cache_hits' in final_stats:
                    logger.info(f"  - IPFS Cache Hits: {final_stats.get('ipfs_cache_hits', 0)}")
            
            # Save final test results
            self.save_results()
            
            logger.info("=" * 80)
            logger.info("Stress test completed successfully")
            
        except Exception as e:
            logger.error(f"Error in stress test: {e}")
            import traceback
            traceback.print_exc()
    
    async def close(self):
        """Close resource pool integration"""
        if self.integration:
            self.integration.close()
            logger.info("ResourcePoolBridgeIntegration closed")
    
    def save_results(self):
        """Save comprehensive results to file with IPFS acceleration and browser metrics"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"web_resource_pool_test_{timestamp}.json"
        
        # Calculate summary metrics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.get('success', False))
        ipfs_accelerated = sum(1 for r in self.results if r.get('ipfs_accelerated', False))
        ipfs_cache_hits = sum(1 for r in self.results if r.get('ipfs_cache_hit', False))
        real_implementations = sum(1 for r in self.results if r.get('is_real_implementation', False))
        
        # Group by model type
        by_model_type = {}
        for result in self.results:
            model_type = result.get('model_type', 'unknown')
            if model_type not in by_model_type:
                by_model_type[model_type] = []
            by_model_type[model_type].append(result)
        
        # Group by browser
        by_browser = {}
        for result in self.results:
            browser = result.get('browser', 'unknown')
            if browser not in by_browser:
                by_browser[browser] = []
            by_browser[browser].append(result)
        
        # Create comprehensive report
        report = {
            'timestamp': timestamp,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests * 100) if total_tests else 0,
            'ipfs_acceleration': {
                'accelerated_count': ipfs_accelerated,
                'cache_hits': ipfs_cache_hits,
                'acceleration_rate': (ipfs_accelerated / total_tests * 100) if total_tests else 0,
                'cache_hit_rate': (ipfs_cache_hits / ipfs_accelerated * 100) if ipfs_accelerated else 0
            },
            'real_implementations': {
                'count': real_implementations,
                'rate': (real_implementations / total_tests * 100) if total_tests else 0
            },
            'by_model_type': {},
            'by_browser': {},
            'detailed_results': self.results
        }
        
        # Calculate metrics by model type
        for model_type, results in by_model_type.items():
            count = len(results)
            success_count = sum(1 for r in results if r.get('success', False))
            ipfs_count = sum(1 for r in results if r.get('ipfs_accelerated', False))
            cache_hits = sum(1 for r in results if r.get('ipfs_cache_hit', False))
            real_count = sum(1 for r in results if r.get('is_real_implementation', False))
            
            # Calculate average execution times
            exec_times = [r.get('execution_time', 0) for r in results]
            avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
            
            report['by_model_type'][model_type] = {
                'count': count,
                'success_count': success_count,
                'success_rate': (success_count / count * 100) if count else 0,
                'ipfs_accelerated_count': ipfs_count,
                'cache_hits': cache_hits,
                'real_implementations': real_count,
                'average_execution_time': avg_exec_time,
                'acceleration_rate': (ipfs_count / count * 100) if count else 0
            }
        
        # Calculate metrics by browser
        for browser, results in by_browser.items():
            count = len(results)
            success_count = sum(1 for r in results if r.get('success', False))
            ipfs_count = sum(1 for r in results if r.get('ipfs_accelerated', False))
            cache_hits = sum(1 for r in results if r.get('ipfs_cache_hit', False))
            real_count = sum(1 for r in results if r.get('is_real_implementation', False))
            compute_shader_count = sum(1 for r in results if r.get('compute_shader_optimized', False))
            precompile_shader_count = sum(1 for r in results if r.get('precompile_shaders', False))
            
            # Calculate average execution times
            exec_times = [r.get('execution_time', 0) for r in results]
            avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
            
            report['by_browser'][browser] = {
                'count': count,
                'success_count': success_count,
                'success_rate': (success_count / count * 100) if count else 0,
                'ipfs_accelerated_count': ipfs_count,
                'cache_hits': cache_hits,
                'real_implementations': real_count,
                'compute_shader_optimized': compute_shader_count,
                'precompile_shaders': precompile_shader_count,
                'average_execution_time': avg_exec_time
            }
        
        # Check if we should save the report to the database
        if self.args.db_path and hasattr(self.integration, 'db_connection') and self.integration.db_connection:
            try:
                # Store in database
                self.integration.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS resource_pool_test_results (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    total_tests INTEGER,
                    successful_tests INTEGER,
                    ipfs_accelerated_count INTEGER,
                    ipfs_cache_hits INTEGER,
                    real_implementations INTEGER,
                    test_duration_seconds FLOAT,
                    summary JSON,
                    detailed_results JSON
                )
                """)
                
                # Calculate test duration
                if self.results:
                    first_time = min(r.get('execution_time', 0) for r in self.results)
                    last_time = max(r.get('execution_time', 0) for r in self.results)
                    duration = last_time - first_time
                else:
                    duration = 0
                
                # Insert into database
                self.integration.db_connection.execute("""
                INSERT INTO resource_pool_test_results (
                    timestamp,
                    total_tests,
                    successful_tests,
                    ipfs_accelerated_count,
                    ipfs_cache_hits,
                    real_implementations,
                    test_duration_seconds,
                    summary,
                    detailed_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    datetime.now(),
                    total_tests,
                    successful_tests,
                    ipfs_accelerated,
                    ipfs_cache_hits,
                    real_implementations,
                    duration,
                    json.dumps({k: v for k, v in report.items() if k != 'detailed_results'}),
                    json.dumps(self.results)
                ])
                
                logger.info("Results saved to database")
            except Exception as e:
                logger.error(f"Error saving results to database: {e}")
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive results saved to {filename}")

async def main_async():
    """Main async function"""
    parser = argparse.ArgumentParser(description="Test WebNN/WebGPU Resource Pool Integration with IPFS Acceleration")
    
    # Enhanced implementation options (May 2025)
    parser.add_argument("--test-enhanced", action="store_true",
        help="Test enhanced resource pool implementation (May 2025)")
    parser.add_argument("--min-connections", type=int, default=1,
        help="Minimum number of browser connections (for enhanced implementation)")
    parser.add_argument("--adaptive-scaling", action="store_true",
        help="Enable adaptive scaling (for enhanced implementation)")
    
    # Model selection options
    parser.add_argument("--models", type=str, default="bert-base-uncased",
        help="Comma-separated list of models to test")
    
    # Platform options
    parser.add_argument("--platform", type=str, choices=["webnn", "webgpu"], default="webgpu",
        help="Platform to test")
    
    # Test options
    parser.add_argument("--concurrent-models", action="store_true",
        help="Test multiple models concurrently")
    parser.add_argument("--stress-test", action="store_true",
        help="Run a stress test on the resource pool")
    parser.add_argument("--duration", type=int, default=60,
        help="Duration of stress test in seconds")
    
    # Configuration options
    parser.add_argument("--max-connections", type=int, default=4,
        help="Maximum number of browser connections")
    parser.add_argument("--visible", action="store_true",
        help="Run browsers in visible mode (not headless)")
    
    # Optimization options
    parser.add_argument("--compute-shaders", action="store_true",
        help="Enable compute shader optimization for audio models")
    parser.add_argument("--shader-precompile", action="store_true",
        help="Enable shader precompilation for faster startup")
    parser.add_argument("--parallel-loading", action="store_true",
        help="Enable parallel model loading for multimodal models")
    
    # IPFS acceleration options
    parser.add_argument("--disable-ipfs", action="store_true",
        help="Disable IPFS acceleration (enabled by default)")
    
    # Database options
    parser.add_argument("--db-path", type=str, default=os.environ.get("BENCHMARK_DB_PATH"),
        help="Path to DuckDB database for storing test results")
    parser.add_argument("--db-only", action="store_true",
        help="Store results only in database (no JSON files)")
    
    # Browser-specific options
    parser.add_argument("--firefox", action="store_true",
        help="Use Firefox for all tests (best for audio models)")
    parser.add_argument("--chrome", action="store_true",
        help="Use Chrome for all tests (best for vision models)")
    parser.add_argument("--edge", action="store_true",
        help="Use Edge for all tests (best for WebNN)")
    
    # Advanced options
    parser.add_argument("--all-optimizations", action="store_true",
        help="Enable all optimizations (compute shaders, shader precompilation, parallel loading)")
    parser.add_argument("--mixed-precision", action="store_true",
        help="Enable mixed precision inference")
    
    # Precision options
    parser.add_argument("--precision", type=int, choices=[2, 3, 4, 8, 16, 32], default=16,
        help="Precision to use for inference (bits)")
                  
    # Error handling and reporting options
    parser.add_argument("--verbose", action="store_true",
        help="Enable verbose logging")
    parser.add_argument("--error-retry", type=int, default=1,
        help="Number of times to retry on error")
    
    # Database verification
    parser.add_argument("--verify-db-schema", action="store_true",
        help="Verify database schema before running tests")
    
    args = parser.parse_args()
    
    # Handle all optimizations flag
    if args.all_optimizations:
        args.compute_shaders = True
        args.shader_precompile = True
        args.parallel_loading = True
    
    # Set environment variables based on optimization flags
    if args.compute_shaders:
        os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
        logger.info("Enabled compute shader optimization")
    
    if args.shader_precompile:
        os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
        logger.info("Enabled shader precompilation")
    
    if args.parallel_loading:
        os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
        logger.info("Enabled parallel model loading")
    
    if args.mixed_precision:
        os.environ["WEBGPU_MIXED_PRECISION_ENABLED"] = "1"
        logger.info("Enabled mixed precision inference")
    
    # Enable browser-specific environment variables
    if args.firefox:
        os.environ["TEST_BROWSER"] = "firefox"
        os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
        logger.info("Using Firefox for all tests with advanced compute shaders")
    elif args.chrome:
        os.environ["TEST_BROWSER"] = "chrome"
        logger.info("Using Chrome for all tests")
    elif args.edge:
        os.environ["TEST_BROWSER"] = "edge"
        logger.info("Using Edge for all tests")
    
    # Set database path from argument or environment variable
    if args.db_path:
        os.environ["BENCHMARK_DB_PATH"] = args.db_path
        logger.info(f"Using database: {args.db_path}")
        
        # Verify database schema if requested
        if args.verify_db_schema:
            if verify_database_schema(args.db_path):
                logger.info(f"Database schema verified: {args.db_path}")
            else:
                logger.warning(f"Database schema verification failed: {args.db_path}")
    
    # Set precision-related environment variables
    if args.precision:
        os.environ["WEBGPU_PRECISION_BITS"] = str(args.precision)
        logger.info(f"Using {args.precision}-bit precision")
        
    if args.mixed_precision:
        os.environ["WEBGPU_MIXED_PRECISION_ENABLED"] = "1"
        logger.info("Using mixed precision inference")
    
    # Parse models
    if "," in args.models:
        model_names = args.models.split(",")
    else:
        model_names = [args.models]
    
    # Map model names to types
    model_types = []
    for model in model_names:
        if "bert" in model.lower() or "t5" in model.lower():
            model_types.append("text_embedding")
        elif "vit" in model.lower() or "clip" in model.lower():
            model_types.append("vision")
        elif "whisper" in model.lower() or "wav2vec" in model.lower() or "clap" in model.lower():
            model_types.append("audio")
        else:
            model_types.append("text")
    
    # Create model list
    models = list(zip(model_types, model_names))
    
    # Log test configuration
    logger.info(f"Starting WebNN/WebGPU Resource Pool Test with {len(models)} models")
    logger.info(f"Platform: {args.platform}")
    logger.info(f"IPFS Acceleration: {'Disabled' if args.disable_ipfs else 'Enabled'}")
    logger.info(f"Max Connections: {args.max_connections}")
    logger.info(f"Headless Mode: {'Disabled' if args.visible else 'Enabled'}")
    
    # Create appropriate tester based on args
    if args.test_enhanced:
        logger.info("Using Enhanced Resource Pool Integration (May 2025)")
        # Import the tester class if available, otherwise use local implementation
        try:
            from fixed_web_platform.enhanced_resource_pool_tester import EnhancedWebResourcePoolTester
            tester = EnhancedWebResourcePoolTester(args)
        except ImportError:
            logger.info("Using mock EnhancedWebResourcePoolTester")
            from fixed_web_platform.enhanced_resource_pool_tester_mock import EnhancedWebResourcePoolTester
            tester = EnhancedWebResourcePoolTester(args)
    else:
        tester = WebResourcePoolTester(args)

    try:
        # Initialize tester
        if not await tester.initialize():
            logger.error("Failed to initialize tester")
            return 1
        
        if args.stress_test:
            # Run stress test with enhanced metrics
            await tester.run_stress_test(args.duration, models)
        elif args.concurrent_models:
            # Test multiple models concurrently with browser-specific optimizations
            await tester.test_concurrent_models(models, args.platform)
        else:
            # Test each model individually with browser-specific optimizations
            for model_type, model_name in models:
                # For audio models, prefer Firefox with compute shader optimizations
                if model_type == 'audio' and not args.chrome and not args.edge:
                    logger.info(f"Using Firefox for audio model: {model_name}")
                    os.environ["TEST_BROWSER"] = "firefox"
                    os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
                
                # For text embedding with WebNN, prefer Edge
                elif model_type == 'text_embedding' and args.platform == 'webnn' and not args.chrome and not args.firefox:
                    logger.info(f"Using Edge for text model with WebNN: {model_name}")
                    os.environ["TEST_BROWSER"] = "edge"
                
                # For vision models, prefer Chrome
                elif model_type == 'vision' and not args.firefox and not args.edge:
                    logger.info(f"Using Chrome for vision model: {model_name}")
                    os.environ["TEST_BROWSER"] = "chrome"
                
                # Run the test
                await tester.test_model(model_type, model_name, args.platform)
        
        # Save results (only to database if db-only flag is set)
        if not args.db_only:
            tester.save_results()
        
        # Close tester
        await tester.close()
        
        return 0
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure tester is closed
        await tester.close()
        
        return 1

def main():
    """Main entry point"""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())