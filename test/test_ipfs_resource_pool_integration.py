#!/usr/bin/env python3
"""
Test IPFS Acceleration with WebGPU/WebNN Resource Pool Integration (May 2025)

This script tests the enhanced resource pool implementation for WebGPU/WebNN hardware
acceleration with IPFS integration, providing efficient model execution across browsers.

Key features demonstrated:
- Enhanced connection pooling with adaptive scaling
- Browser-specific optimizations (Firefox for audio, Edge for WebNN)
- Hardware-aware load balancing
- Cross-browser resource sharing
- Comprehensive telemetry and database integration
- Distributed inference capability
- Smart fallback with automatic recovery

Usage:
    python test_ipfs_resource_pool_integration.py --model bert-base-uncased --platform webgpu
    python test_ipfs_resource_pool_integration.py --concurrent-models
    python test_ipfs_resource_pool_integration.py --distributed
    python test_ipfs_resource_pool_integration.py --benchmark
    python test_ipfs_resource_pool_integration.py --all-optimizations
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
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Required modules
REQUIRED_MODULES = {
    "resource_pool_integration": False,
    "resource_pool_bridge": False,
    "ipfs_accelerate_impl": False,
    "duckdb": False
}

# Check for new resource_pool_integration
try:
    from fixed_web_platform.resource_pool_integration import IPFSAccelerateWebIntegration
    REQUIRED_MODULES["resource_pool_integration"] = True
    logger.info("IPFSAccelerateWebIntegration available")
except ImportError:
    logger.error("IPFSAccelerateWebIntegration not available. Make sure fixed_web_platform module is properly installed")

# Check for legacy resource_pool_bridge (backward compatibility)
try:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    REQUIRED_MODULES["resource_pool_bridge"] = True
except ImportError:
    logger.warning("ResourcePoolBridgeIntegration not available for backward compatibility")

# Check for ipfs_accelerate_impl
try:
    import ipfs_accelerate_impl
    REQUIRED_MODULES["ipfs_accelerate_impl"] = True
except ImportError:
    logger.warning("IPFS accelerate implementation not available")

# Check for duckdb
try:
    import duckdb
    REQUIRED_MODULES["duckdb"] = True
except ImportError:
    logger.warning("DuckDB not available. Database integration will be disabled"))

class IPFSResourcePoolTester:
    """Test IPFS Acceleration with Enhanced WebGPU/WebNN Resource Pool Integration."""
    
    def __init__(self, args):
        """Initialize tester with command line arguments."""
        self.args = args
        self.results = []
        self.ipfs_module = None
        self.resource_pool_integration = None
        self.legacy_integration = None
        self.db_connection = None
        self.creation_time = time.time()
        self.session_id = str(int(time.time()))
        
        # Set environment variables for optimizations if needed
        if args.optimize_audio or args.all_optimizations:
            os.environ["USE_FIREFOX_WEBGPU"] = "1"
            os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
            logger.info("Enabled Firefox audio optimizations for audio models")
        
        if args.shader_precompile or args.all_optimizations:
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
            logger.info("Enabled shader precompilation")
        
        if args.parallel_loading or args.all_optimizations:
            os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
            logger.info("Enabled parallel model loading")
            
        if args.mixed_precision:
            os.environ["WEBGPU_MIXED_PRECISION_ENABLED"] = "1"
            logger.info("Enabled mixed precision")
            
        if args.precision != 16:
            os.environ["WEBGPU_PRECISION_BITS"] = str(args.precision)
            logger.info(f"Set precision to {args.precision} bits")
        
        # Import IPFS module if needed
        if hasattr(args, 'use_ipfs') and args.use_ipfs and REQUIRED_MODULES["ipfs_accelerate_impl"]:
            self.ipfs_module = ipfs_accelerate_impl
            logger.info("IPFS accelerate module imported successfully")
        
        # Connect to database if specified
        if hasattr(args, 'db_path') and args.db_path and not args.disable_db and REQUIRED_MODULES["duckdb"]:
            try:
                self.db_connection = duckdb.connect(args.db_path)
                logger.info(f"Connected to database: {args.db_path}")
                
                # Initialize database schema if needed
                self._initialize_database_schema()
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_connection = None
                
    def _initialize_database_schema(self):
        """Initialize database schema if needed."""
        if not self.db_connection:
            return
            
        try:
            # Create tables if they don't exist
            # Resource pool test results table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS ipfs_resource_pool_test_results (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                session_id VARCHAR,
                model_name VARCHAR,
                model_type VARCHAR,
                platform VARCHAR,
                browser VARCHAR,
                test_method VARCHAR,
                success BOOLEAN,
                is_real_implementation BOOLEAN,
                is_simulation BOOLEAN,
                execution_time_sec FLOAT,
                precision INTEGER,
                mixed_precision BOOLEAN,
                compute_shader_optimized BOOLEAN,
                precompile_shaders BOOLEAN,
                parallel_loading BOOLEAN,
                memory_usage_mb FLOAT,
                ipfs_cache_hit BOOLEAN,
                ipfs_source VARCHAR,
                p2p_optimized BOOLEAN,
                resource_pool_used BOOLEAN,
                performance_metrics JSON,
                detailed_results JSON
            )
            """)
            
            # Create benchmark results table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS ipfs_resource_pool_benchmark_results (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                session_id VARCHAR,
                benchmark_type VARCHAR,
                total_models INTEGER,
                successful_models INTEGER,
                execution_time_sec FLOAT,
                models_tested JSON,
                success_rate FLOAT,
                real_hardware_rate FLOAT,
                ipfs_cache_hit_rate FLOAT,
                p2p_optimization_rate FLOAT,
                throughput_improvement_factor FLOAT,
                detailed_results JSON
            )
            """)
            
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            
    async def initialize_resource_pool(self):
        """Initialize the resource pool integration with enhanced capabilities."""
        if not REQUIRED_MODULES["resource_pool_integration"]:
            logger.error("Cannot initialize resource pool: IPFSAccelerateWebIntegration not available")
            
            # Try legacy integration if available
            if REQUIRED_MODULES["resource_pool_bridge"]:
                logger.warning("Falling back to legacy ResourcePoolBridgeIntegration")
                return await self._initialize_legacy_resource_pool()
            return False
        
        try:
            # Configure browser preferences for optimal performance
            browser_preferences = {
                'audio': 'firefox',  # Firefox has better compute shader performance for audio
                'vision': 'chrome',  # Chrome has good WebGPU support for vision models
                'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
                'text_generation': 'chrome',
                'multimodal': 'chrome'
            }
            
            # Override browser preferences if specific browser is selected
            if hasattr(self.args, 'browser') and self.args.browser:
                if self.args.browser == 'firefox':
                    browser_preferences = {k: 'firefox' for k in browser_preferences}
                elif self.args.browser == 'chrome':
                    browser_preferences = {k: 'chrome' for k in browser_preferences}
                elif self.args.browser == 'edge':
                    browser_preferences = {k: 'edge' for k in browser_preferences}
            
            # Create IPFSAccelerateWebIntegration instance with enhanced capabilities
            self.resource_pool_integration = IPFSAccelerateWebIntegration(
                max_connections=self.args.max_connections,
                enable_gpu=True,
                enable_cpu=True,
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_telemetry=True,
                db_path=self.args.db_path if hasattr(self.args, 'db_path') and not getattr(self.args, 'disable_db', False) else None,
                smart_fallback=True
            )
            
            logger.info("Enhanced resource pool integration initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize enhanced resource pool integration: {e}")
            import traceback
            traceback.print_exc()
            
            # Try legacy integration if available
            if REQUIRED_MODULES["resource_pool_bridge"]:
                logger.warning("Falling back to legacy ResourcePoolBridgeIntegration")
                return await self._initialize_legacy_resource_pool()
            return False
            
    async def _initialize_legacy_resource_pool(self):
        """Initialize legacy resource pool integration for backward compatibility."""
        if not REQUIRED_MODULES["resource_pool_bridge"]:
            logger.error("Cannot initialize legacy resource pool: ResourcePoolBridgeIntegration not available")
            return False
        
        try:
            # Configure browser preferences for optimal performance
            browser_preferences = {
                'audio': 'firefox',  # Firefox has better compute shader performance for audio
                'vision': 'chrome',  # Chrome has good WebGPU support for vision models
                'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
                'text': 'edge'      # Edge works well for text models
            }
            
            # Create ResourcePoolBridgeIntegration instance
            self.legacy_integration = ResourcePoolBridgeIntegration(
                max_connections=self.args.max_connections,
                enable_gpu=True,
                enable_cpu=True,
                headless=not self.args.visible,
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_ipfs=True,
                db_path=self.args.db_path if hasattr(self.args, 'db_path') and not getattr(self.args, 'disable_db', False) else None
            )
            
            # Initialize integration
            self.legacy_integration.initialize()
            logger.info("Legacy resource pool integration initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize legacy resource pool integration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_model_enhanced(self, model_name, model_type):
        """Test a model using the enhanced IPFSAccelerateWebIntegration."""
        if not self.resource_pool_integration:
            logger.error("Cannot test model: resource pool integration not initialized")
            return None
        
        try:
            logger.info(f"Testing model with enhanced resource pool: {model_name} ({model_type})")
            
            platform = self.args.platform
            
            # Create quantization settings if specified
            quantization = None
            if hasattr(self.args, 'precision') and self.args.precision != 16 or hasattr(self.args, 'mixed_precision') and self.args.mixed_precision:
                quantization = {
                    "bits": self.args.precision if hasattr(self.args, 'precision') else 16,
                    "mixed_precision": self.args.mixed_precision if hasattr(self.args, 'mixed_precision') else False
                }
            
            # Create optimizations dictionary
            optimizations = {}
            if hasattr(self.args, 'optimize_audio') and self.args.optimize_audio or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                optimizations["compute_shaders"] = True
            if hasattr(self.args, 'shader_precompile') and self.args.shader_precompile or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                optimizations["precompile_shaders"] = True
            if hasattr(self.args, 'parallel_loading') and self.args.parallel_loading or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                optimizations["parallel_loading"] = True
            
            # Get model from integration with enhanced features
            start_time = time.time()
            
            model = self.resource_pool_integration.get_model(
                model_name=model_name,
                model_type=model_type,
                platform=platform,
                batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 1,
                quantization=quantization,
                optimizations=optimizations
            )
            
            if not model:
                logger.error(f"Failed to get model: {model_name}")
                return None
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s with enhanced integration")
            
            # Prepare test input based on model type
            test_input = self._create_test_input(model_type)
            
            # Run inference with enhanced integration
            start_time = time.time()
            
            result = self.resource_pool_integration.run_inference(
                model,
                test_input,
                batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 1,
                timeout=self.args.timeout if hasattr(self.args, 'timeout') else 60.0,
                track_metrics=True,
                store_in_db=hasattr(self.args, 'db_path') and self.args.db_path and not getattr(self.args, 'disable_db', False),
                telemetry_data={"test_type": "single_model_enhanced"}
            )
            
            execution_time = time.time() - start_time
            
            # Get model info for detailed metrics
            if hasattr(model, "get_model_info"):
                model_info = model.get_model_info()
            else:
                model_info = {}
            
            # Extract detailed performance metrics
            try:
                performance_metrics = {}
                if hasattr(model, "get_performance_metrics"):
                    metrics = model.get_performance_metrics()
                    if metrics and isinstance(metrics, dict):
                        performance_metrics = metrics
            except Exception as metrics_error:
                logger.warning(f"Error extracting performance metrics: {metrics_error}")
                performance_metrics = {}
            
            # Create result object with enhanced information
            test_result = {
                'model_name': model_name,
                'model_type': model_type,
                'platform': platform,
                'execution_time': execution_time,
                'success': isinstance(result, dict) and result.get('success', False) or True,
                'is_real_implementation': model_info.get('is_real_implementation', False),
                'browser': model_info.get('browser', 'unknown'),
                'compute_shader_optimized': model_info.get('compute_shader_optimized', False),
                'precompile_shaders': model_info.get('precompile_shaders', False),
                'parallel_loading': model_info.get('parallel_loading', False),
                'precision': getattr(self.args, 'precision', 16),
                'mixed_precision': getattr(self.args, 'mixed_precision', False),
                'test_method': "enhanced_resource_pool",
                'performance_metrics': performance_metrics
            }
            
            # Store result
            self.results.append(test_result)
            
            # Store in database if enabled
            if self.db_connection:
                self._store_test_result(test_result)
            
            logger.info(f"Enhanced resource pool test completed in {execution_time:.2f}s: {model_name}")
            return test_result
        except Exception as e:
            logger.error(f"Error testing model with enhanced integration: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _create_test_input(self, model_type):
        """Create a test input based on model type."""
        if model_type == 'text_embedding':
            return {
                'input_ids': [[101, 2023, 2003, 1037, 3231, 102]],
                'attention_mask': [[1, 1, 1, 1, 1, 1]]
            }
        elif model_type == 'vision':
            # Create a simple image input (3x224x224)
            try:
                import numpy as np
                return {'pixel_values': np.zeros((1, 3, 224, 224), dtype=np.float32)}
            except ImportError:
                return {'pixel_values': [[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(3)]}
        elif model_type == 'audio':
            # Create a simple audio input
            try:
                import numpy as np
                return {'input_features': np.zeros((1, 80, 3000), dtype=np.float32)}
            except ImportError:
                return {'input_features': [[[0.1 for _ in range(80)] for _ in range(3000)]]}
        elif model_type == 'multimodal':
            # Create a multimodal input (text + image)
            try:
                import numpy as np
                return {
                    'input_ids': [[101, 2023, 2003, 1037, 3231, 102]],
                    'attention_mask': [[1, 1, 1, 1, 1, 1]],
                    'pixel_values': np.zeros((1, 3, 224, 224), dtype=np.float32)
                }
            except ImportError:
                return {
                    'input_ids': [[101, 2023, 2003, 1037, 3231, 102]],
                    'attention_mask': [[1, 1, 1, 1, 1, 1]],
                    'pixel_values': [[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(3)]
                }
        elif model_type == 'text_generation':
            return {
                'input_ids': [[101, 2023, 2003, 1037, 3231, 102]],
                'attention_mask': [[1, 1, 1, 1, 1, 1]]
            }
        else:
            # Default fallback
            return {'inputs': [0.0 for _ in range(10)]}
            
    def _store_test_result(self, test_result):
        """Store test result in database."""
        if not self.db_connection:
            return
            
        try:
            # Prepare JSON data
            performance_metrics_json = "{}"
            if 'performance_metrics' in test_result and test_result['performance_metrics']:
                try:
                    performance_metrics_json = json.dumps(test_result['performance_metrics'])
                except:
                    pass
                    
            # Store full result as JSON for detailed analysis
            detailed_json = json.dumps(test_result)
            
            # Insert into database
            self.db_connection.execute("""
            INSERT INTO ipfs_resource_pool_test_results (
                timestamp, session_id, model_name, model_type, platform, browser,
                test_method, success, is_real_implementation, execution_time_sec,
                precision, mixed_precision, compute_shader_optimized, precompile_shaders,
                parallel_loading, performance_metrics, detailed_results
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                self.session_id,
                test_result.get('model_name', 'unknown'),
                test_result.get('model_type', 'unknown'),
                test_result.get('platform', 'unknown'),
                test_result.get('browser', 'unknown'),
                test_result.get('test_method', 'unknown'),
                test_result.get('success', False),
                test_result.get('is_real_implementation', False),
                test_result.get('execution_time', 0.0),
                test_result.get('precision', 16),
                test_result.get('mixed_precision', False),
                test_result.get('compute_shader_optimized', False),
                test_result.get('precompile_shaders', False),
                test_result.get('parallel_loading', False),
                performance_metrics_json,
                detailed_json
            ])
            
            logger.info(f"Test result stored in database for {test_result.get('model_name', 'unknown')}")
        except Exception as e:
            logger.error(f"Error storing test result in database: {e}")
            
    async def test_concurrent_models_enhanced(self):
        """Test concurrent execution of multiple models using enhanced integration."""
        if not self.resource_pool_integration:
            logger.error("Cannot test concurrent models: resource pool integration not initialized")
            return []
        
        try:
            # Define models to test
            models = []
            
            if hasattr(self.args, 'models') and self.args.models:
                # Parse models from command line
                for model_spec in self.args.models.split(','):
                    parts = model_spec.split(':')
                    if len(parts) == 2:
                        model_type, model_name = parts
                    else:
                        model_name = parts[0]
                        # Infer model type from name
                        if "bert" in model_name.lower():
                            model_type = "text_embedding"
                        elif "vit" in model_name.lower() or "clip" in model_name.lower():
                            model_type = "vision"
                        elif "whisper" in model_name.lower() or "wav2vec" in model_name.lower():
                            model_type = "audio"
                        elif "llava" in model_name.lower() or "blip" in model_name.lower():
                            model_type = "multimodal"
                        elif "gpt" in model_name.lower() or "llama" in model_name.lower():
                            model_type = "text_generation"
                        else:
                            model_type = "text_embedding"
                    
                    models.append((model_type, model_name))
            else:
                # Use default models
                models = [
                    ("text_embedding", "bert-base-uncased"),
                    ("vision", "google/vit-base-patch16-224"),
                    ("audio", "openai/whisper-tiny")
                ]
            
            logger.info(f"Testing {len(models)} models concurrently with enhanced integration")
            
            # Load models using enhanced integration
            loaded_models = []
            for model_type, model_name in models:
                # Create quantization settings if specified
                quantization = None
                if hasattr(self.args, 'precision') and self.args.precision != 16 or hasattr(self.args, 'mixed_precision') and self.args.mixed_precision:
                    quantization = {
                        "bits": self.args.precision if hasattr(self.args, 'precision') else 16,
                        "mixed_precision": self.args.mixed_precision if hasattr(self.args, 'mixed_precision') else False
                    }
                
                # Create optimizations dictionary
                optimizations = {}
                if hasattr(self.args, 'optimize_audio') and self.args.optimize_audio or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                    if model_type == 'audio':
                        optimizations["compute_shaders"] = True
                if hasattr(self.args, 'shader_precompile') and self.args.shader_precompile or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                    optimizations["precompile_shaders"] = True
                if hasattr(self.args, 'parallel_loading') and self.args.parallel_loading or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                    if model_type == 'multimodal':
                        optimizations["parallel_loading"] = True
                
                # Get model from enhanced integration
                model = self.resource_pool_integration.get_model(
                    model_name=model_name,
                    model_type=model_type,
                    platform=self.args.platform,
                    batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 1,
                    quantization=quantization,
                    optimizations=optimizations
                )
                
                if model:
                    loaded_models.append((model, model_type, model_name))
                else:
                    logger.error(f"Failed to load model: {model_name}")
            
            if not loaded_models:
                logger.error("No models were loaded successfully")
                return []
            
            logger.info(f"Successfully loaded {len(loaded_models)} models")
            
            # Prepare inputs for concurrent execution
            model_data_pairs = []
            for model, model_type, model_name in loaded_models:
                # Create test input based on model type
                test_input = self._create_test_input(model_type)
                model_data_pairs.append((model, test_input))
            
            # Run concurrent inference with enhanced integration
            logger.info(f"Running concurrent inference for {len(model_data_pairs)} models")
            start_time = time.time()
            
            concurrent_results = self.resource_pool_integration.run_parallel_inference(
                model_data_pairs,
                batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 1,
                timeout=self.args.timeout if hasattr(self.args, 'timeout') else 60.0,
                distributed=hasattr(self.args, 'distributed') and self.args.distributed
            )
            
            execution_time = time.time() - start_time
            
            # Process results
            test_results = []
            for i, result in enumerate(concurrent_results):
                if i < len(loaded_models):
                    model, model_type, model_name = loaded_models[i]
                    
                    # Get model info for detailed metrics
                    model_info = {}
                    if hasattr(model, "get_model_info"):
                        model_info = model.get_model_info()
                    
                    # Extract performance metrics
                    performance_metrics = {}
                    if hasattr(model, "get_performance_metrics"):
                        try:
                            metrics = model.get_performance_metrics()
                            if metrics and isinstance(metrics, dict):
                                performance_metrics = metrics
                        except Exception as metrics_error:
                            logger.warning(f"Error extracting performance metrics: {metrics_error}")
                    
                    # Create result object
                    test_result = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'platform': self.args.platform,
                        'execution_time': execution_time,
                        'success': result is not None,
                        'is_real_implementation': model_info.get('is_real_implementation', False),
                        'browser': model_info.get('browser', 'unknown'),
                        'compute_shader_optimized': model_info.get('compute_shader_optimized', False),
                        'precompile_shaders': model_info.get('precompile_shaders', False),
                        'parallel_loading': model_info.get('parallel_loading', False),
                        'test_method': "concurrent_execution_enhanced",
                        'performance_metrics': performance_metrics
                    }
                    
                    test_results.append(test_result)
                    self.results.append(test_result)
                    
                    # Store in database if enabled
                    if self.db_connection:
                        self._store_test_result(test_result)
            
            logger.info(f"Concurrent execution completed in {execution_time:.2f}s")
            
            return test_results
        except Exception as e:
            logger.error(f"Error testing concurrent models with enhanced integration: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    async def run_benchmark_enhanced(self):
        """Run a comprehensive benchmark with the enhanced integration."""
        if not self.resource_pool_integration:
            logger.error("Cannot run benchmark: resource pool integration not initialized")
            return []
        
        try:
            logger.info("Running comprehensive benchmark with enhanced integration")
            
            # Define models to benchmark
            if hasattr(self.args, 'models') and self.args.models:
                # Parse models from command line
                models = []
                for model_spec in self.args.models.split(','):
                    parts = model_spec.split(':')
                    if len(parts) == 2:
                        model_type, model_name = parts
                    else:
                        model_name = parts[0]
                        # Infer model type from name
                        if "bert" in model_name.lower():
                            model_type = "text_embedding"
                        elif "vit" in model_name.lower() or "clip" in model_name.lower():
                            model_type = "vision"
                        elif "whisper" in model_name.lower() or "wav2vec" in model_name.lower():
                            model_type = "audio"
                        elif "llava" in model_name.lower() or "blip" in model_name.lower():
                            model_type = "multimodal"
                        elif "gpt" in model_name.lower() or "llama" in model_name.lower():
                            model_type = "text_generation"
                        else:
                            model_type = "text_embedding"
                    
                    models.append((model_type, model_name))
            else:
                # Use default models
                models = [
                    ("text_embedding", "bert-base-uncased"),
                    ("vision", "google/vit-base-patch16-224"),
                    ("audio", "openai/whisper-tiny")
                ]
            
            # Results for benchmark
            benchmark_results = {
                "single_model": [],
                "concurrent_execution": [],
                "distributed_execution": []
            }
            
            # 1. Test each model individually
            logger.info("Running benchmark with single model execution...")
            for model_type, model_name in models:
                result = await self.test_model_enhanced(model_name, model_type)
                if result:
                    benchmark_results["single_model"].append(result)
                
                # Wait a bit between tests
                await asyncio.sleep(0.5)
            
            # 2. Test concurrent execution
            logger.info("Running benchmark with concurrent execution...")
            # Set flag for concurrent execution
            setattr(self.args, 'concurrent_models', True)
            concurrent_results = await self.test_concurrent_models_enhanced()
            benchmark_results["concurrent_execution"] = concurrent_results
            
            # 3. Test distributed execution if requested
            if hasattr(self.args, 'distributed') and self.args.distributed:
                logger.info("Running benchmark with distributed execution...")
                setattr(self.args, 'distributed', True)
                distributed_results = await self.test_concurrent_models_enhanced()
                benchmark_results["distributed_execution"] = distributed_results
            
            # Calculate benchmark summary
            summary = self._calculate_enhanced_benchmark_summary(benchmark_results)
            
            # Print benchmark summary
            self._print_enhanced_benchmark_summary(summary)
            
            # Store benchmark results in database
            if self.db_connection:
                self._store_benchmark_results(benchmark_results, summary)
            
            # Save benchmark results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ipfs_resource_pool_benchmark_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    "results": benchmark_results,
                    "summary": summary
                }, f, indent=2)
            
            logger.info(f"Enhanced benchmark results saved to {filename}")
            
            return benchmark_results
        except Exception as e:
            logger.error(f"Error running enhanced benchmark: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_enhanced_benchmark_summary(self, benchmark_results):
        """Calculate enhanced summary statistics for benchmark results."""
        summary = {}
        
        # Helper function to calculate average execution time
        def calc_avg_time(results):
            if not results:
                return 0
            return sum(r.get('execution_time', 0) for r in results) / len(results)
        
        # Calculate average execution time for each method
        summary['avg_execution_time'] = {
            'single_model': calc_avg_time(benchmark_results['single_model']),
            'concurrent_execution': calc_avg_time(benchmark_results['concurrent_execution']),
            'distributed_execution': calc_avg_time(benchmark_results['distributed_execution'])
        }
        
        # Calculate success rates
        summary['success_rate'] = {
            'single_model': sum(1 for r in benchmark_results['single_model'] if r.get('success', False)) / 
                                len(benchmark_results['single_model']) if benchmark_results['single_model'] else 0,
            'concurrent_execution': sum(1 for r in benchmark_results['concurrent_execution'] if r.get('success', False)) / 
                                len(benchmark_results['concurrent_execution']) if benchmark_results['concurrent_execution'] else 0,
            'distributed_execution': sum(1 for r in benchmark_results['distributed_execution'] if r.get('success', False)) / 
                                len(benchmark_results['distributed_execution']) if benchmark_results['distributed_execution'] else 0
        }
        
        # Calculate real hardware vs simulation rates
        summary['real_hardware_rate'] = {
            'single_model': sum(1 for r in benchmark_results['single_model'] if r.get('is_real_implementation', False)) / 
                                len(benchmark_results['single_model']) if benchmark_results['single_model'] else 0,
            'concurrent_execution': sum(1 for r in benchmark_results['concurrent_execution'] if r.get('is_real_implementation', False)) / 
                                len(benchmark_results['concurrent_execution']) if benchmark_results['concurrent_execution'] else 0,
            'distributed_execution': sum(1 for r in benchmark_results['distributed_execution'] if r.get('is_real_implementation', False)) / 
                                len(benchmark_results['distributed_execution']) if benchmark_results['distributed_execution'] else 0
        }
        
        # Calculate optimization usage rates
        summary['optimization_usage'] = {
            'compute_shaders': sum(1 for r in benchmark_results['single_model'] if r.get('compute_shader_optimized', False)) / 
                                len(benchmark_results['single_model']) if benchmark_results['single_model'] else 0,
            'precompile_shaders': sum(1 for r in benchmark_results['single_model'] if r.get('precompile_shaders', False)) / 
                                len(benchmark_results['single_model']) if benchmark_results['single_model'] else 0,
            'parallel_loading': sum(1 for r in benchmark_results['single_model'] if r.get('parallel_loading', False)) / 
                                len(benchmark_results['single_model']) if benchmark_results['single_model'] else 0
        }
        
        # Calculate throughput improvement
        if benchmark_results['concurrent_execution'] and benchmark_results['single_model']:
            single_time = calc_avg_time(benchmark_results['single_model'])
            concurrent_time = calc_avg_time(benchmark_results['concurrent_execution'])
            
            if single_time > 0:
                # Calculate improvement factor (higher is better)
                # This is an approximation since concurrent execution returns multiple results in one call
                single_items_per_second = 1 / single_time
                concurrent_items_per_second = len(benchmark_results['concurrent_execution']) / concurrent_time
                improvement_factor = concurrent_items_per_second / single_items_per_second if single_items_per_second > 0 else 0
                
                summary['throughput_improvement_factor'] = improvement_factor
        else:
            summary['throughput_improvement_factor'] = 0
        
        # Calculate distributed execution improvement if available
        if benchmark_results['distributed_execution'] and benchmark_results['concurrent_execution']:
            concurrent_time = calc_avg_time(benchmark_results['concurrent_execution'])
            distributed_time = calc_avg_time(benchmark_results['distributed_execution'])
            
            if concurrent_time > 0:
                # Calculate improvement factor (higher is better)
                distributed_improvement = concurrent_time / distributed_time if distributed_time > 0 else 0
                summary['distributed_improvement_factor'] = distributed_improvement
        else:
            summary['distributed_improvement_factor'] = 0
        
        return summary
    
    def _print_enhanced_benchmark_summary(self, summary):
        """Print an enhanced summary of benchmark results."""
        print("\n" + "="*80)
        print("ENHANCED BENCHMARK SUMMARY")
        print("="*80)
        
        print("\nAverage Execution Time (seconds):")
        print(f"  Single Model:          {summary['avg_execution_time']['single_model']:.3f}")
        print(f"  Concurrent Execution:  {summary['avg_execution_time']['concurrent_execution']:.3f}")
        if 'distributed_execution' in summary['avg_execution_time']:
            print(f"  Distributed Execution: {summary['avg_execution_time']['distributed_execution']:.3f}")
        
        print("\nSuccess Rate:")
        print(f"  Single Model:          {summary['success_rate']['single_model']*100:.1f}%")
        print(f"  Concurrent Execution:  {summary['success_rate']['concurrent_execution']*100:.1f}%")
        if 'distributed_execution' in summary['success_rate']:
            print(f"  Distributed Execution: {summary['success_rate']['distributed_execution']*100:.1f}%")
        
        print("\nReal Hardware Rate:")
        print(f"  Single Model:          {summary['real_hardware_rate']['single_model']*100:.1f}%")
        print(f"  Concurrent Execution:  {summary['real_hardware_rate']['concurrent_execution']*100:.1f}%")
        if 'distributed_execution' in summary['real_hardware_rate']:
            print(f"  Distributed Execution: {summary['real_hardware_rate']['distributed_execution']*100:.1f}%")
        
        print("\nOptimization Usage:")
        print(f"  Compute Shaders:       {summary['optimization_usage']['compute_shaders']*100:.1f}%")
        print(f"  Shader Precompilation: {summary['optimization_usage']['precompile_shaders']*100:.1f}%")
        print(f"  Parallel Loading:      {summary['optimization_usage']['parallel_loading']*100:.1f}%")
        
        print("\nThroughput Improvement:")
        print(f"  Concurrent vs Single:  {summary['throughput_improvement_factor']:.2f}x")
        
        if 'distributed_improvement_factor' in summary:
            print(f"  Distributed vs Concurrent: {summary['distributed_improvement_factor']:.2f}x")
        
        print("="*80)
        
    def _store_benchmark_results(self, benchmark_results, summary):
        """Store benchmark results in database."""
        if not self.db_connection:
            return
            
        try:
            # Prepare data
            timestamp = datetime.now()
            all_models = []
            
            # Collect all tested models
            for test_type, results in benchmark_results.items():
                for result in results:
                    model_name = result.get('model_name', 'unknown')
                    model_type = result.get('model_type', 'unknown')
                    all_models.append(f"{model_type}:{model_name}")
            
            # Make list unique
            unique_models = list(set(all_models))
            
            # Insert benchmark results
            self.db_connection.execute("""
            INSERT INTO ipfs_resource_pool_benchmark_results (
                timestamp, session_id, benchmark_type, total_models, successful_models,
                execution_time_sec, models_tested, success_rate, real_hardware_rate,
                throughput_improvement_factor, detailed_results
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                self.session_id,
                "enhanced_comprehensive",
                len(unique_models),
                sum(1 for r in benchmark_results['single_model'] if r.get('success', False)),
                summary['avg_execution_time']['single_model'],
                json.dumps(unique_models),
                summary['success_rate']['single_model'],
                summary['real_hardware_rate']['single_model'],
                summary['throughput_improvement_factor'],
                json.dumps(benchmark_results)
            ])
            
            logger.info("Benchmark results stored in database")
        except Exception as e:
            logger.error(f"Error storing benchmark results in database: {e}")
            
    async def close(self):
        """Close resources."""
        if self.resource_pool_integration:
            try:
                self.resource_pool_integration.close()
                logger.info("Enhanced resource pool integration closed")
            except Exception as e:
                logger.error(f"Error closing enhanced resource pool integration: {e}")
        
        if self.legacy_integration:
            try:
                self.legacy_integration.close()
                logger.info("Legacy resource pool integration closed")
            except Exception as e:
                logger.error(f"Error closing legacy resource pool integration: {e}")
        
        if self.db_connection:
            try:
                self.db_connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    def save_results(self):
        """Save test results to file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ipfs_resource_pool_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Generate markdown report
        self._generate_markdown_report(f"ipfs_resource_pool_test_{timestamp}.md")
    
    def _generate_markdown_report(self, filename):
        """Generate markdown report from test results."""
        with open(filename, 'w') as f:
            f.write("# IPFS Resource Pool Integration Test Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group results by test method
            methods = {}
            for result in self.results:
                method = result.get('test_method', 'unknown')
                if method not in methods:
                    methods[method] = []
                methods[method].append(result)
            
            # Overall summary
            f.write("## Summary\n\n")
            
            total_tests = len(self.results)
            successful_tests = sum(1 for r in self.results if r.get('success', False))
            
            f.write(f"- Total Tests: {total_tests}\n")
            f.write(f"- Successful Tests: {successful_tests} ({successful_tests/total_tests*100:.1f}%)\n")
            
            # Tests by method
            for method, results in methods.items():
                method_successful = sum(1 for r in results if r.get('success', False))
                f.write(f"- {method.replace('_', ' ').title()}: {len(results)} tests, {method_successful} successful ({method_successful/len(results)*100:.1f}%)\n")
            
            f.write("\n")
            
            # Test results by method
            for method, results in methods.items():
                f.write(f"## {method.replace('_', ' ').title()} Tests\n\n")
                
                f.write("| Model | Type | Platform | Browser | Success | Real HW | Execution Time (s) |\n")
                f.write("|-------|------|----------|---------|---------|---------|--------------------|\n")
                
                for result in sorted(results, key=lambda r: r.get('model_name', '')):
                    model_name = result.get('model_name', 'unknown')
                    model_type = result.get('model_type', 'unknown')
                    platform = result.get('platform', 'unknown')
                    browser = result.get('browser', 'unknown')
                    success = '✅' if result.get('success', False) else '❌'
                    real_hw = '✅' if result.get('is_real_implementation', False) else '❌'
                    execution_time = f"{result.get('execution_time', 0):.2f}"
                    
                    f.write(f"| {model_name} | {model_type} | {platform} | {browser} | {success} | {real_hw} | {execution_time} |\n")
                
                f.write("\n")
            
            # Optimization details
            f.write("## Optimization Details\n\n")
            
            f.write("| Model | Type | Compute Shaders | Shader Precompilation | Parallel Loading | Precision | Mixed Precision |\n")
            f.write("|-------|------|-----------------|------------------------|------------------|-----------|----------------|\n")
            
            for result in sorted(self.results, key=lambda r: r.get('model_name', '')):
                model_name = result.get('model_name', 'unknown')
                model_type = result.get('model_type', 'unknown')
                compute_shaders = '✅' if result.get('compute_shader_optimized', False) else '❌'
                precompile_shaders = '✅' if result.get('precompile_shaders', False) else '❌'
                parallel_loading = '✅' if result.get('parallel_loading', False) else '❌'
                precision = result.get('precision', 16)
                mixed_precision = '✅' if result.get('mixed_precision', False) else '❌'
                
                f.write(f"| {model_name} | {model_type} | {compute_shaders} | {precompile_shaders} | {parallel_loading} | {precision}-bit | {mixed_precision} |\n")
            
            f.write("\n")
            
            logger.info(f"Markdown report saved to {filename}")


async def main_async():
    """Async main function for the test script."""
    parser = argparse.ArgumentParser(description="Test IPFS Acceleration with Enhanced WebGPU/WebNN Resource Pool Integration")
    
    # Model selection options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model to test")
    model_group.add_argument("--models", type=str,
                        help="Comma-separated list of models to test")
    model_group.add_argument("--model-type", type=str, 
                        choices=["text", "text_embedding", "text_generation", "vision", "audio", "multimodal"],
                        default="text_embedding", help="Model type")
    
    # Platform options
    platform_group = parser.add_argument_group("Platform and Browser Options")
    platform_group.add_argument("--platform", type=str, 
                          choices=["webnn", "webgpu", "cpu"], default="webgpu",
                          help="Platform to test")
    platform_group.add_argument("--browser", type=str, 
                          choices=["chrome", "firefox", "edge", "safari"],
                          help="Browser to use")
    platform_group.add_argument("--visible", action="store_true",
                          help="Run browsers in visible mode (not headless)")
    platform_group.add_argument("--max-connections", type=int, default=4,
                          help="Maximum number of browser connections")
    
    # Precision options
    precision_group = parser.add_argument_group("Precision Options")
    precision_group.add_argument("--precision", type=int, 
                          choices=[2, 3, 4, 8, 16, 32], default=16,
                          help="Precision level in bits")
    precision_group.add_argument("--mixed-precision", action="store_true",
                          help="Use mixed precision")
    
    # Optimization options
    opt_group = parser.add_argument_group("Optimization Options")
    opt_group.add_argument("--optimize-audio", action="store_true",
                      help="Enable Firefox audio optimizations")
    opt_group.add_argument("--shader-precompile", action="store_true",
                      help="Enable shader precompilation")
    opt_group.add_argument("--parallel-loading", action="store_true",
                      help="Enable parallel model loading")
    opt_group.add_argument("--all-optimizations", action="store_true",
                      help="Enable all optimizations")
    
    # Test options
    test_group = parser.add_argument_group("Test Options")
    test_group.add_argument("--test-method", type=str, 
                      choices=["enhanced", "legacy", "ipfs", "concurrent", "distributed", "all"],
                      default="enhanced", help="Test method to use")
    test_group.add_argument("--concurrent-models", action="store_true",
                      help="Test multiple models concurrently")
    test_group.add_argument("--distributed", action="store_true",
                      help="Test distributed execution across multiple browsers")
    test_group.add_argument("--benchmark", action="store_true",
                      help="Run comprehensive benchmark comparing all methods")
    test_group.add_argument("--batch-size", type=int, default=1,
                      help="Batch size for inference")
    test_group.add_argument("--timeout", type=float, default=60.0,
                      help="Timeout for operations in seconds")
    
    # Database options
    db_group = parser.add_argument_group("Database Options")
    db_group.add_argument("--db-path", type=str, default="./benchmark_db.duckdb",
                    help="Path to database for storing results")
    db_group.add_argument("--disable-db", action="store_true",
                    help="Disable database storage")
    
    # IPFS options
    ipfs_group = parser.add_argument_group("IPFS Options")
    ipfs_group.add_argument("--use-ipfs", action="store_true",
                      help="Use IPFS acceleration")
    
    # Misc options
    misc_group = parser.add_argument_group("Miscellaneous Options")
    misc_group.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Check required modules
    missing_modules = []
    
    # For enhanced integration
    if args.test_method in ["enhanced", "all"]:
        if not REQUIRED_MODULES["resource_pool_integration"]:
            missing_modules.append("resource_pool_integration")
    
    # For legacy integration
    if args.test_method in ["legacy", "all"]:
        if not REQUIRED_MODULES["resource_pool_bridge"]:
            missing_modules.append("resource_pool_bridge")
    
    # For IPFS integration
    if args.test_method in ["ipfs", "all"] or args.use_ipfs:
        if not REQUIRED_MODULES["ipfs_accelerate_impl"]:
            missing_modules.append("ipfs_accelerate_impl")
    
    # For database
    if args.db_path and not args.disable_db:
        if not REQUIRED_MODULES["duckdb"]:
            missing_modules.append("duckdb")
            logger.warning("DuckDB not available. Database integration will be disabled")
            args.disable_db = True
    
    if missing_modules and args.test_method != "all":
        logger.error(f"Missing required modules for selected test method: {missing_modules}")
        return 1
    
    # Create tester
    tester = IPFSResourcePoolTester(args)
    
    try:
        # Initialize resource pool
        if not await tester.initialize_resource_pool():
            logger.error("Failed to initialize resource pool")
            return 1
        
        # Run tests based on test method
        if args.benchmark:
            # Run enhanced benchmark
            await tester.run_benchmark_enhanced()
        elif args.concurrent_models or args.distributed:
            # Test multiple models concurrently
            await tester.test_concurrent_models_enhanced()
        else:
            # Run tests based on test method
            if args.test_method == "enhanced" or args.test_method == "all":
                await tester.test_model_enhanced(args.model, args.model_type)
            
            if args.test_method == "legacy" or args.test_method == "all":
                # Legacy method would go here
                pass
            
            if args.test_method == "ipfs" or args.test_method == "all":
                # IPFS method would go here
                pass
        
        # Save results
        tester.save_results()
        
        # Close resources
        await tester.close()
        
        return 0
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        
        # Close resources
        await tester.close()
        
        return 1

def main():
    """Main entry point."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())
            db_path=self.args.db_path
            )
            
            # Initialize integration
            self.resource_pool_integration.initialize())))
            logger.info()))"Resource pool integration initialized successfully")
                return True
        except Exception as e:
            logger.error()))f"Failed to initialize resource pool integration: {}}}}}}}}}}}}}}}}}e}")
            import traceback
            traceback.print_exc())))
                return False
    
    async def test_model_direct()))self, model_name, model_type):
        """Test a model using direct ResourcePoolBridge integration."""
        if not self.resource_pool_integration:
            logger.error()))"Cannot test model: resource pool integration not initialized")
        return None
        
        try:
            logger.info()))f"Testing model directly with resource pool: {}}}}}}}}}}}}}}}}}model_name} ())){}}}}}}}}}}}}}}}}}model_type})")
            
            platform = self.args.platform
            
            # Configure hardware preferences with browser optimizations
            hardware_preferences = {}}}}}}}}}}}}}}}}}
            'priority_list': []],,platform, 'cpu'],
            'model_family': model_type,
            'enable_ipfs': True,
            'precision': self.args.precision,
            'mixed_precision': self.args.mixed_precision,
            'browser': self.args.browser
            }
            
            # For audio models, use Firefox optimizations
            if model_type == 'audio' and self.args.optimize_audio:
                hardware_preferences[]],,'browser'] = 'firefox',
                hardware_preferences[]],,'use_firefox_optimizations'] = True,
                logger.info()))"Using Firefox with audio optimizations for audio model")
            
            # Get model from resource pool
                model = self.resource_pool_integration.get_model()))
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
                )
            
            if not model:
                logger.error()))f"Failed to get model: {}}}}}}}}}}}}}}}}}model_name}")
                return None
            
            # Prepare test input based on model type
            if model_type == 'text_embedding':
                test_input = {}}}}}}}}}}}}}}}}}
                'input_ids': []],,101, 2023, 2003, 1037, 3231, 102],
                'attention_mask': []],,1, 1, 1, 1, 1, 1],,
                }
            elif model_type == 'vision':
                test_input = {}}}}}}}}}}}}}}}}}'pixel_values': []],,[]],,[]],,0.5 for _ in range()))3)] for _ in range()))224)] for _ in range()))1)]}:::,,
            elif model_type == 'audio':
                test_input = {}}}}}}}}}}}}}}}}}'input_features': []],,[]],,[]],,0.1 for _ in range()))80)] for _ in range()))3000)]]}:::,,
            else:
                test_input = {}}}}}}}}}}}}}}}}}'inputs': []],,0.0 for _ in range()))10)]}:,,
            # Run inference
                start_time = time.time())))
                result = model()))test_input)
                execution_time = time.time()))) - start_time
            
            # Create result object with enhanced information
                test_result = {}}}}}}}}}}}}}}}}}
                'model_name': model_name,
                'model_type': model_type,
                'platform': platform,
                'execution_time': execution_time,
                'success': result.get()))'success', result.get()))'status') == 'success'),
                'is_real_implementation': result.get()))'is_real_implementation', False),
                'browser': result.get()))'browser', 'unknown'),
                'compute_shader_optimized': result.get()))'compute_shader_optimized', False),
                'precompile_shaders': result.get()))'precompile_shaders', False),
                'parallel_loading': result.get()))'parallel_loading', False),
                'precision': self.args.precision,
                'mixed_precision': self.args.mixed_precision,
                'test_method': "direct_resource_pool"
                }
            
                self.results.append()))test_result)
            
                logger.info()))f"Direct resource pool test completed in {}}}}}}}}}}}}}}}}}execution_time:.2f}s: {}}}}}}}}}}}}}}}}}model_name}")
                return test_result
        except Exception as e:
            logger.error()))f"Error testing model directly: {}}}}}}}}}}}}}}}}}e}")
            import traceback
            traceback.print_exc())))
                return None
    
    async def test_model_ipfs()))self, model_name, model_type):
        """Test a model using IPFS acceleration with resource pool integration."""
        if not self.ipfs_module:
            logger.error()))"Cannot test model: IPFS module not available")
        return None
        
        try:
            logger.info()))f"Testing model with IPFS acceleration: {}}}}}}}}}}}}}}}}}model_name} ())){}}}}}}}}}}}}}}}}}model_type})")
            
            platform = self.args.platform
            
            # Configure acceleration options
            config = {}}}}}}}}}}}}}}}}}
            'platform': platform,
            'hardware': platform,
            'browser': self.args.browser,
            'precision': self.args.precision,
            'mixed_precision': self.args.mixed_precision,
            'use_firefox_optimizations': self.args.optimize_audio,
            'use_resource_pool': True,
            'max_connections': self.args.max_connections,
            'headless': not self.args.visible,
            'adaptive_scaling': True,
            'model_type': model_type,
            'store_results': True,
            'p2p_optimization': True
            }
            
            # Prepare test input based on model type
            if model_type == 'text_embedding':
                test_input = {}}}}}}}}}}}}}}}}}
                'input_ids': []],,101, 2023, 2003, 1037, 3231, 102],
                'attention_mask': []],,1, 1, 1, 1, 1, 1],,
                }
            elif model_type == 'vision':
                test_input = {}}}}}}}}}}}}}}}}}'pixel_values': []],,[]],,[]],,0.5 for _ in range()))3)] for _ in range()))224)] for _ in range()))1)]}:::,,
            elif model_type == 'audio':
                test_input = {}}}}}}}}}}}}}}}}}'input_features': []],,[]],,[]],,0.1 for _ in range()))80)] for _ in range()))3000)]]}:::,,
            else:
                test_input = {}}}}}}}}}}}}}}}}}'inputs': []],,0.0 for _ in range()))10)]}:,,
            # Run IPFS acceleration with resource pool
                start_time = time.time())))
                result = self.ipfs_module.accelerate()))model_name, test_input, config)
                execution_time = time.time()))) - start_time
            
            # Extract performance metrics
                performance_metrics = {}}}}}}}}}}}}}}}}}}
            if isinstance()))result, dict):
                performance_metrics = {}}}}}}}}}}}}}}}}}
                'latency_ms': result.get()))'latency_ms', 0),
                'throughput_items_per_sec': result.get()))'throughput_items_per_sec', 0),
                'memory_usage_mb': result.get()))'memory_usage_mb', 0)
                }
            
            # Create result object with enhanced information
                test_result = {}}}}}}}}}}}}}}}}}
                'model_name': model_name,
                'model_type': model_type,
                'platform': platform,
                'execution_time': execution_time,
                'success': result.get()))'status') == 'success',
                'is_real_hardware': result.get()))'is_real_hardware', False),
                'is_simulation': result.get()))'is_simulation', not result.get()))'is_real_hardware', False)),
                'browser': result.get()))'browser', 'unknown'),
                'precision': result.get()))'precision', self.args.precision),
                'mixed_precision': result.get()))'mixed_precision', self.args.mixed_precision),
                'firefox_optimizations': result.get()))'firefox_optimizations', False),
                'ipfs_cache_hit': result.get()))'ipfs_cache_hit', False),
                'ipfs_source': result.get()))'ipfs_source'),
                'p2p_optimized': result.get()))'p2p_optimized', False),
                'resource_pool_used': result.get()))'resource_pool_used', False),
                'performance_metrics': performance_metrics,
                'test_method': "ipfs_acceleration"
                }
            
                self.results.append()))test_result)
            
                logger.info()))f"IPFS acceleration test completed in {}}}}}}}}}}}}}}}}}execution_time:.2f}s: {}}}}}}}}}}}}}}}}}model_name}")
                return test_result
        except Exception as e:
            logger.error()))f"Error testing model with IPFS acceleration: {}}}}}}}}}}}}}}}}}e}")
            import traceback
            traceback.print_exc())))
                return None
    
    async def test_concurrent_models()))self):
        """Test multiple models concurrently using resource pool integration."""
        if not self.resource_pool_integration:
            logger.error()))"Cannot test concurrent models: resource pool integration not initialized")
        return []],,],
        
        try:
            # Define models to test
            models = []],,],
            
            if self.args.models:
                # Parse models from command line
                for model_spec in self.args.models.split()))','):
                    parts = model_spec.split()))':')
                    if len()))parts) == 2:
                        model_type, model_name = parts
                    else:
                        model_name = parts[]],,0],
                        # Infer model type from name
                        if "bert" in model_name.lower()))):
                            model_type = "text_embedding"
                        elif "vit" in model_name.lower()))) or "clip" in model_name.lower()))):
                            model_type = "vision"
                        elif "whisper" in model_name.lower()))) or "wav2vec" in model_name.lower()))):
                            model_type = "audio"
                        else:
                            model_type = "text"
                    
                            models.append()))()))model_type, model_name))
            else:
                # Use default models
                models = []],,
                ()))"text_embedding", "bert-base-uncased"),
                ()))"vision", "google/vit-base-patch16-224"),
                ()))"audio", "openai/whisper-tiny")
                ]
            
                logger.info()))f"Testing {}}}}}}}}}}}}}}}}}len()))models)} models concurrently")
            
            # Create model configurations
                model_configs = []],,],
            for i, ()))model_type, model_name) in enumerate()))models):
                model_configs.append())){}}}}}}}}}}}}}}}}}
                'model_type': model_type,
                'model_name': model_name,
                'model_id': f"model_{}}}}}}}}}}}}}}}}}i}"
                })
            
            # Get models concurrently
                loaded_models = self.resource_pool_integration.get_models_concurrent()))model_configs)
            
            if not loaded_models:
                logger.error()))"Failed to load any models concurrently")
                return []],,],
            
                logger.info()))f"Successfully loaded {}}}}}}}}}}}}}}}}}len()))loaded_models)} models concurrently")
            
            # Prepare inputs for concurrent execution
                models_and_inputs = []],,],
            for model_id, model in loaded_models.items()))):
                # Get model type from model configuration
                model_type = next()))()))config[]],,'model_type'] for config in model_configs if config[]],,'model_id'] == model_id), "text")
                
                # Prepare test input based on model type:
                if model_type == 'text_embedding':
                    test_input = {}}}}}}}}}}}}}}}}}
                    'input_ids': []],,101, 2023, 2003, 1037, 3231, 102],
                    'attention_mask': []],,1, 1, 1, 1, 1, 1],,
                    }
                elif model_type == 'vision':
                    test_input = {}}}}}}}}}}}}}}}}}'pixel_values': []],,[]],,[]],,0.5 for _ in range()))3)] for _ in range()))224)] for _ in range()))1)]}:::,,
                elif model_type == 'audio':
                    test_input = {}}}}}}}}}}}}}}}}}'input_features': []],,[]],,[]],,0.1 for _ in range()))80)] for _ in range()))3000)]]}:::,,
                else:
                    test_input = {}}}}}}}}}}}}}}}}}'inputs': []],,0.0 for _ in range()))10)]}:,,    
                    models_and_inputs.append()))()))model_id, test_input))
            
            # Run concurrent execution
                    start_time = time.time())))
                    concurrent_results = self.resource_pool_integration.execute_concurrent()))models_and_inputs)
                    execution_time = time.time()))) - start_time
            
            # Process results
                    test_results = []],,],
            for i, result in enumerate()))concurrent_results):
                if i < len()))models):
                    model_type, model_name = models[]],,i]
                    
                    # Extract performance metrics
                    performance_metrics = {}}}}}}}}}}}}}}}}}}
                    if isinstance()))result, dict) and 'performance_metrics' in result:
                        performance_metrics = result[]],,'performance_metrics']
                    
                    # Create result object
                        test_result = {}}}}}}}}}}}}}}}}}
                        'model_name': model_name,
                        'model_type': model_type,
                        'platform': self.args.platform,
                        'execution_time': execution_time,
                        'success': result.get()))'success', False),
                        'is_real_implementation': result.get()))'is_real_implementation', False),
                        'browser': result.get()))'browser', 'unknown'),
                        'performance_metrics': performance_metrics,
                        'test_method': "concurrent_execution"
                        }
                    
                        test_results.append()))test_result)
                        self.results.append()))test_result)
            
                        logger.info()))f"Concurrent execution of {}}}}}}}}}}}}}}}}}len()))models)} models completed in {}}}}}}}}}}}}}}}}}execution_time:.2f}s")
            
                    return test_results
        except Exception as e:
            logger.error()))f"Error testing concurrent models: {}}}}}}}}}}}}}}}}}e}")
            import traceback
            traceback.print_exc())))
                    return []],,],
    
    async def run_benchmark()))self):
        """Run a benchmark comparing direct resource pool, IPFS acceleration, and concurrent execution."""
        if not self.ipfs_module or not REQUIRED_MODULES[]],,"resource_pool_bridge"]:,
        logger.error()))"Cannot run benchmark: required modules not available")
                    return []],,],
        
        try:
            # Initialize resource pool if not already initialized:
            if not self.resource_pool_integration:
                if not await self.initialize_resource_pool()))):
                    logger.error()))"Failed to initialize resource pool for benchmark")
                return []],,],
            
            # Define models to benchmark
            if self.args.models:
                # Parse models from command line
                models = []],,],
                for model_spec in self.args.models.split()))','):
                    parts = model_spec.split()))':')
                    if len()))parts) == 2:
                        model_type, model_name = parts
                    else:
                        model_name = parts[]],,0],
                        # Infer model type from name
                        if "bert" in model_name.lower()))):
                            model_type = "text_embedding"
                        elif "vit" in model_name.lower()))) or "clip" in model_name.lower()))):
                            model_type = "vision"
                        elif "whisper" in model_name.lower()))) or "wav2vec" in model_name.lower()))):
                            model_type = "audio"
                        else:
                            model_type = "text"
                    
                            models.append()))()))model_type, model_name))
            else:
                # Use default models
                models = []],,
                ()))"text_embedding", "bert-base-uncased"),
                ()))"vision", "google/vit-base-patch16-224"),
                ()))"audio", "openai/whisper-tiny")
                ]
            
            # Results for benchmark
                benchmark_results = {}}}}}}}}}}}}}}}}}
                "direct_resource_pool": []],,],,
                "ipfs_acceleration": []],,],,
                "concurrent_execution": []],,],
                }
            
            # 1. Test each model with direct resource pool
                logger.info()))"Running benchmark with direct resource pool...")
            for model_type, model_name in models:
                result = await self.test_model_direct()))model_name, model_type)
                if result:
                    benchmark_results[]],,"direct_resource_pool"].append()))result)
                
                # Wait a bit between tests
                    await asyncio.sleep()))0.5)
            
            # 2. Test each model with IPFS acceleration
                    logger.info()))"Running benchmark with IPFS acceleration...")
            for model_type, model_name in models:
                result = await self.test_model_ipfs()))model_name, model_type)
                if result:
                    benchmark_results[]],,"ipfs_acceleration"].append()))result)
                
                # Wait a bit between tests
                    await asyncio.sleep()))0.5)
            
            # 3. Test all models concurrently
                    logger.info()))"Running benchmark with concurrent execution...")
                    concurrent_results = await self.test_concurrent_models())))
                    benchmark_results[]],,"concurrent_execution"] = concurrent_results
            
            # Calculate benchmark summary
                    summary = self._calculate_benchmark_summary()))benchmark_results)
            
            # Print benchmark summary
                    self._print_benchmark_summary()))summary)
            
            # Save benchmark results
                    timestamp = datetime.now()))).strftime()))"%Y%m%d_%H%M%S")
                    filename = f"ipfs_resource_pool_benchmark_{}}}}}}}}}}}}}}}}}timestamp}.json"
            
            with open()))filename, 'w') as f:
                json.dump())){}}}}}}}}}}}}}}}}}
                "results": benchmark_results,
                "summary": summary
                }, f, indent=2)
            
                logger.info()))f"Benchmark results saved to {}}}}}}}}}}}}}}}}}filename}")
            
                    return benchmark_results
        except Exception as e:
            logger.error()))f"Error running benchmark: {}}}}}}}}}}}}}}}}}e}")
            import traceback
            traceback.print_exc())))
                    return []],,],
    
    def _calculate_benchmark_summary()))self, benchmark_results):
        """Calculate summary statistics for benchmark results."""
        summary = {}}}}}}}}}}}}}}}}}}
        
        # Helper function to calculate average execution time
        def calc_avg_time()))results):
            if not results:
            return 0
        return sum()))r.get()))'execution_time', 0) for r in results) / len()))results)
        
        # Calculate average execution time for each method
        summary[]],,'avg_execution_time'] = {}}}}}}}}}}}}}}}}}
        'direct_resource_pool': calc_avg_time()))benchmark_results[]],,'direct_resource_pool']),
        'ipfs_acceleration': calc_avg_time()))benchmark_results[]],,'ipfs_acceleration']),
        'concurrent_execution': calc_avg_time()))benchmark_results[]],,'concurrent_execution'])
        if benchmark_results[]],,'concurrent_execution'] else 0
        }
        
        # Calculate success rates
        summary[]],,'success_rate'] = {}}}}}}}}}}}}}}}}}:
            'direct_resource_pool': sum()))1 for r in benchmark_results[]],,'direct_resource_pool'] if r.get()))'success', False)): / 
                                    len()))benchmark_results[]],,'direct_resource_pool']) if benchmark_results[]],,'direct_resource_pool'] else 0,:
                                        'ipfs_acceleration': sum()))1 for r in benchmark_results[]],,'ipfs_acceleration'] if r.get()))'success', False)): /
                                len()))benchmark_results[]],,'ipfs_acceleration']) if benchmark_results[]],,'ipfs_acceleration'] else 0,:
                                    'concurrent_execution': sum()))1 for r in benchmark_results[]],,'concurrent_execution'] if r.get()))'success', False)): /
                                    len()))benchmark_results[]],,'concurrent_execution']) if benchmark_results[]],,'concurrent_execution'] else 0
                                    }
        
        # Calculate real hardware vs simulation rates
        summary[]],,'real_hardware_rate'] = {}}}}}}}}}}}}}}}}}:
            'direct_resource_pool': sum()))1 for r in benchmark_results[]],,'direct_resource_pool'] if r.get()))'is_real_implementation', False)) / 
                                    len()))benchmark_results[]],,'direct_resource_pool']) if benchmark_results[]],,'direct_resource_pool'] else 0,:
                                        'ipfs_acceleration': sum()))1 for r in benchmark_results[]],,'ipfs_acceleration'] if r.get()))'is_real_hardware', False)) /
                                len()))benchmark_results[]],,'ipfs_acceleration']) if benchmark_results[]],,'ipfs_acceleration'] else 0,:
                                    'concurrent_execution': sum()))1 for r in benchmark_results[]],,'concurrent_execution'] if r.get()))'is_real_implementation', False)) /
                                    len()))benchmark_results[]],,'concurrent_execution']) if benchmark_results[]],,'concurrent_execution'] else 0
                                    }
        
        # Calculate IPFS-specific metrics:
        if benchmark_results[]],,'ipfs_acceleration']:
            summary[]],,'ipfs_cache_hit_rate'] = sum()))1 for r in benchmark_results[]],,'ipfs_acceleration'] if r.get()))'ipfs_cache_hit', False)) / len()))benchmark_results[]],,'ipfs_acceleration'])
            summary[]],,'p2p_optimization_rate'] = sum()))1 for r in benchmark_results[]],,'ipfs_acceleration'] if r.get()))'p2p_optimized', False)) / len()))benchmark_results[]],,'ipfs_acceleration']):
        else:
            summary[]],,'ipfs_cache_hit_rate'] = 0
            summary[]],,'p2p_optimization_rate'] = 0
        
        # Calculate throughput improvement
        if benchmark_results[]],,'concurrent_execution'] and benchmark_results[]],,'direct_resource_pool']:
            direct_time = calc_avg_time()))benchmark_results[]],,'direct_resource_pool'])
            concurrent_time = calc_avg_time()))benchmark_results[]],,'concurrent_execution'])
            
            if direct_time > 0:
                # Calculate improvement factor ()))higher is better)
                # This is an approximation since concurrent execution returns multiple results in one call
                direct_items_per_second = 1 / direct_time
                concurrent_items_per_second = len()))benchmark_results[]],,'concurrent_execution']) / concurrent_time
                improvement_factor = concurrent_items_per_second / direct_items_per_second if direct_items_per_second > 0 else 0
                
                summary[]],,'throughput_improvement_factor'] = improvement_factor:
        else:
            summary[]],,'throughput_improvement_factor'] = 0
        
                    return summary
    
    def _print_benchmark_summary()))self, summary):
        """Print a summary of benchmark results."""
        print()))"\n" + "="*80)
        print()))"BENCHMARK SUMMARY")
        print()))"="*80)
        
        print()))"\nAverage Execution Time ()))seconds):")
        print()))f"  Direct Resource Pool:  {}}}}}}}}}}}}}}}}}summary[]],,'avg_execution_time'][]],,'direct_resource_pool']:.3f}")
        print()))f"  IPFS Acceleration:     {}}}}}}}}}}}}}}}}}summary[]],,'avg_execution_time'][]],,'ipfs_acceleration']:.3f}")
        print()))f"  Concurrent Execution:  {}}}}}}}}}}}}}}}}}summary[]],,'avg_execution_time'][]],,'concurrent_execution']:.3f}")
        
        print()))"\nSuccess Rate:")
        print()))f"  Direct Resource Pool:  {}}}}}}}}}}}}}}}}}summary[]],,'success_rate'][]],,'direct_resource_pool']*100:.1f}%")
        print()))f"  IPFS Acceleration:     {}}}}}}}}}}}}}}}}}summary[]],,'success_rate'][]],,'ipfs_acceleration']*100:.1f}%")
        print()))f"  Concurrent Execution:  {}}}}}}}}}}}}}}}}}summary[]],,'success_rate'][]],,'concurrent_execution']*100:.1f}%")
        
        print()))"\nReal Hardware Rate:")
        print()))f"  Direct Resource Pool:  {}}}}}}}}}}}}}}}}}summary[]],,'real_hardware_rate'][]],,'direct_resource_pool']*100:.1f}%")
        print()))f"  IPFS Acceleration:     {}}}}}}}}}}}}}}}}}summary[]],,'real_hardware_rate'][]],,'ipfs_acceleration']*100:.1f}%")
        print()))f"  Concurrent Execution:  {}}}}}}}}}}}}}}}}}summary[]],,'real_hardware_rate'][]],,'concurrent_execution']*100:.1f}%")
        
        print()))"\nIPFS-Specific Metrics:")
        print()))f"  Cache Hit Rate:        {}}}}}}}}}}}}}}}}}summary[]],,'ipfs_cache_hit_rate']*100:.1f}%")
        print()))f"  P2P Optimization Rate: {}}}}}}}}}}}}}}}}}summary[]],,'p2p_optimization_rate']*100:.1f}%")
        
        print()))"\nThroughput Improvement:")
        print()))f"  Concurrent vs Direct:  {}}}}}}}}}}}}}}}}}summary[]],,'throughput_improvement_factor']:.2f}x")
        
        print()))"="*80)
    
    async def close()))self):
        """Close resources."""
        if self.resource_pool_integration:
            self.resource_pool_integration.close())))
            logger.info()))"Resource pool integration closed")
        
        if self.db_connection:
            self.db_connection.close())))
            logger.info()))"Database connection closed")
    
    def save_results()))self):
        """Save test results to file."""
        if not self.results:
            logger.warning()))"No results to save")
        return
        
        timestamp = datetime.now()))).strftime()))"%Y%m%d_%H%M%S")
        filename = f"ipfs_resource_pool_test_{}}}}}}}}}}}}}}}}}timestamp}.json"
        
        with open()))filename, 'w') as f:
            json.dump()))self.results, f, indent=2)
        
            logger.info()))f"Results saved to {}}}}}}}}}}}}}}}}}filename}")
        
        # Generate markdown report
            self._generate_markdown_report()))f"ipfs_resource_pool_test_{}}}}}}}}}}}}}}}}}timestamp}.md")
    
    def _generate_markdown_report()))self, filename):
        """Generate markdown report from test results."""
        with open()))filename, 'w') as f:
            f.write()))"# IPFS Resource Pool Integration Test Results\n\n")
            f.write()))f"Generated: {}}}}}}}}}}}}}}}}}datetime.now()))).strftime()))'%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group results by test method
            methods = {}}}}}}}}}}}}}}}}}}
            for result in self.results:
                method = result.get()))'test_method', 'unknown')
                if method not in methods:
                    methods[]],,method] = []],,],
                    methods[]],,method].append()))result)
            
            # Overall summary
                    f.write()))"## Summary\n\n")
            
                    total_tests = len()))self.results)
            successful_tests = sum()))1 for r in self.results if r.get()))'success', False)):
            :
                f.write()))f"- Total Tests: {}}}}}}}}}}}}}}}}}total_tests}\n")
                f.write()))f"- Successful Tests: {}}}}}}}}}}}}}}}}}successful_tests} ())){}}}}}}}}}}}}}}}}}successful_tests/total_tests*100:.1f}%)\n")
            
            # Tests by method
            for method, results in methods.items()))):
                method_successful = sum()))1 for r in results if r.get()))'success', False)):
                    :    f.write()))f"- {}}}}}}}}}}}}}}}}}method.replace()))'_', ' ').title())))}: {}}}}}}}}}}}}}}}}}len()))results)} tests, {}}}}}}}}}}}}}}}}}method_successful} successful ())){}}}}}}}}}}}}}}}}}method_successful/len()))results)*100:.1f}%)\n")
            
                    f.write()))"\n")
            
            # Test results by method
            for method, results in methods.items()))):
                f.write()))f"## {}}}}}}}}}}}}}}}}}method.replace()))'_', ' ').title())))} Tests\n\n")
                
                f.write()))"| Model | Type | Platform | Browser | Success | Real HW | Execution Time ()))s) |\n")
                f.write()))"|-------|------|----------|---------|---------|---------|--------------------|\n")
                
                for result in sorted()))results, key=lambda r: r.get()))'model_name', '')):
                    model_name = result.get()))'model_name', 'unknown')
                    model_type = result.get()))'model_type', 'unknown')
                    platform = result.get()))'platform', 'unknown')
                    browser = result.get()))'browser', 'unknown')
                    success = '✅' if result.get()))'success', False) else '❌'
                    real_hw = '✅' if result.get()))'is_real_implementation', result.get()))'is_real_hardware', False)) else '❌':
                        execution_time = f"{}}}}}}}}}}}}}}}}}result.get()))'execution_time', 0):.2f}"
                    
                        f.write()))f"| {}}}}}}}}}}}}}}}}}model_name} | {}}}}}}}}}}}}}}}}}model_type} | {}}}}}}}}}}}}}}}}}platform} | {}}}}}}}}}}}}}}}}}browser} | {}}}}}}}}}}}}}}}}}success} | {}}}}}}}}}}}}}}}}}real_hw} | {}}}}}}}}}}}}}}}}}execution_time} |\n")
                
                        f.write()))"\n")
            
            # Additional details for IPFS acceleration tests
            if 'ipfs_acceleration' in methods:
                f.write()))"## IPFS Acceleration Details\n\n")
                
                f.write()))"| Model | Cache Hit | P2P Optimized | IPFS Source | Resource Pool Used |\n")
                f.write()))"|-------|-----------|--------------|-------------|-------------------|\n")
                
                for result in sorted()))methods[]],,'ipfs_acceleration'], key=lambda r: r.get()))'model_name', '')):
                    model_name = result.get()))'model_name', 'unknown')
                    cache_hit = '✅' if result.get()))'ipfs_cache_hit', False) else '❌'
                    p2p_optimized = '✅' if result.get()))'p2p_optimized', False) else '❌'
                    ipfs_source = result.get()))'ipfs_source', 'N/A')
                    resource_pool_used = '✅' if result.get()))'resource_pool_used', False) else '❌'
                    
                    f.write()))f"| {}}}}}}}}}}}}}}}}}model_name} | {}}}}}}}}}}}}}}}}}cache_hit} | {}}}}}}}}}}}}}}}}}p2p_optimized} | {}}}}}}}}}}}}}}}}}ipfs_source} | {}}}}}}}}}}}}}}}}}resource_pool_used} |\n")
                
                    f.write()))"\n")
            
                    logger.info()))f"Markdown report saved to {}}}}}}}}}}}}}}}}}filename}")
:
async def main_async()))):
    """Async main function."""
    parser = argparse.ArgumentParser()))description="Test IPFS Acceleration with Resource Pool Integration")
    
    # Model selection options
    parser.add_argument()))"--model", type=str, default="bert-base-uncased",
    help="Model to test")
    parser.add_argument()))"--models", type=str,
    help="Comma-separated list of models to test ()))model_type:model_name format)")
    parser.add_argument()))"--model-type", type=str, choices=[]],,"text", "text_embedding", "vision", "audio", "multimodal"],
    default="text_embedding", help="Model type")
    
    # Platform options
    parser.add_argument()))"--platform", type=str, choices=[]],,"webnn", "webgpu"], default="webgpu",
    help="Platform to test")
    
    # Browser options
    parser.add_argument()))"--browser", type=str, choices=[]],,"chrome", "firefox", "edge", "safari"],
    help="Browser to use")
    parser.add_argument()))"--visible", action="store_true",
    help="Run browsers in visible mode ()))not headless)")
    parser.add_argument()))"--max-connections", type=int, default=4,
    help="Maximum number of browser connections")
    
    # Precision options
    parser.add_argument()))"--precision", type=int, choices=[]],,4, 8, 16, 32], default=16,
    help="Precision level")
    parser.add_argument()))"--mixed-precision", action="store_true",
    help="Use mixed precision")
    
    # Optimization options
    parser.add_argument()))"--optimize-audio", action="store_true",
    help="Enable Firefox audio optimizations")
    parser.add_argument()))"--shader-precompile", action="store_true",
    help="Enable shader precompilation")
    parser.add_argument()))"--parallel-loading", action="store_true",
    help="Enable parallel model loading")
    
    # Test options
    parser.add_argument()))"--test-method", type=str, choices=[]],,"direct", "ipfs", "concurrent", "all"],
    default="all", help="Test method to use")
    parser.add_argument()))"--concurrent-models", action="store_true",
    help="Test multiple models concurrently")
    parser.add_argument()))"--benchmark", action="store_true",
    help="Run benchmark comparing all methods")
    
    # Database options
    parser.add_argument()))"--db-path", type=str,
    help="Path to database")
    
    # Parse arguments
    args = parser.parse_args())))
    
    # Check required modules
    missing_modules = []],,name for name, available in REQUIRED_MODULES.items()))) if not available]:
    if missing_modules:
        logger.error()))f"Missing required modules: {}}}}}}}}}}}}}}}}}missing_modules}")
        return 1
    
    # Create tester
        tester = IPFSResourcePoolTester()))args)
    
    try:
        # Initialize resource pool
        if not args.test_method == "ipfs":
            if not await tester.initialize_resource_pool()))):
                logger.error()))"Failed to initialize resource pool")
            return 1
        
        # Run tests based on test method
        if args.benchmark:
            # Run benchmark comparing all methods
            await tester.run_benchmark())))
        elif args.concurrent_models:
            # Test multiple models concurrently
            await tester.test_concurrent_models())))
        else:
            # Run tests based on test method
            if args.test_method == "direct" or args.test_method == "all":
                await tester.test_model_direct()))args.model, args.model_type)
            
            if args.test_method == "ipfs" or args.test_method == "all":
                await tester.test_model_ipfs()))args.model, args.model_type)
        
        # Save results
                tester.save_results())))
        
        # Close resources
                await tester.close())))
        
                return 0
    except Exception as e:
        logger.error()))f"Error in main: {}}}}}}}}}}}}}}}}}e}")
        import traceback
        traceback.print_exc())))
        
        # Close resources
        await tester.close())))
        
                return 1

def main()))):
    """Main entry point."""
    try:
    return asyncio.run()))main_async()))))
    except KeyboardInterrupt:
        logger.info()))"Interrupted by user")
    return 130

if __name__ == "__main__":
    sys.exit()))main()))))