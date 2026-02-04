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
import anyio
import argparse
import logging
import uuid
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
    from test.tests.web.web_platform.resource_pool_integration import IPFSAccelerateWebIntegration
    REQUIRED_MODULES["resource_pool_integration"] = True
    logger.info("IPFSAccelerateWebIntegration available")
except ImportError:
    logger.error("IPFSAccelerateWebIntegration not available. Make sure fixed_web_platform module is properly installed")

# Check for legacy resource_pool_bridge (backward compatibility)
try:
    from test.tests.web.web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
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
    logger.warning("DuckDB not available. Database integration will be disabled")

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
            
            # Create acceleration results table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS acceleration_results (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                session_id VARCHAR,
                model_name VARCHAR,
                model_type VARCHAR,
                platform VARCHAR,
                browser VARCHAR,
                is_real_hardware BOOLEAN,
                is_simulation BOOLEAN,
                processing_time FLOAT,
                latency_ms FLOAT,
                throughput_items_per_sec FLOAT,
                memory_usage_mb FLOAT,
                details JSON
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
            hardware_preferences = {"priority_list": [platform]}
            hardware_preferences["compute_shaders"] = False
            hardware_preferences["precompile_shaders"] = False
            hardware_preferences["parallel_loading"] = False
            
            if hasattr(self.args, 'optimize_audio') and self.args.optimize_audio or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                optimizations["compute_shaders"] = True
                hardware_preferences["compute_shaders"] = True
            if hasattr(self.args, 'shader_precompile') and self.args.shader_precompile or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                optimizations["precompile_shaders"] = True
                hardware_preferences["precompile_shaders"] = True
            if hasattr(self.args, 'parallel_loading') and self.args.parallel_loading or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                optimizations["parallel_loading"] = True
                hardware_preferences["parallel_loading"] = True
            
            # Get model from integration with enhanced features
            start_time = time.time()
            
            # Ensure hardware_preferences has valid priority_list
            if 'priority_list' not in hardware_preferences:
                hardware_preferences['priority_list'] = [platform]
            
            # Debug final hardware_preferences
            logger.debug(f"Final hardware_preferences for model {model_name}: {hardware_preferences}")
            
            model = self.resource_pool_integration.get_model(
                model_name=model_name,
                model_type=model_type,
                platform=platform,
                batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 1,
                quantization=quantization,
                optimizations=optimizations,
                hardware_preferences=hardware_preferences
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
            
            # Debug model attributes
            if hasattr(model, 'compute_shader_optimized'):
                logger.debug(f"Model {model_name} optimization flags directly from model:")
                logger.debug(f"  compute_shader_optimized: {model.compute_shader_optimized}")
                logger.debug(f"  precompile_shaders: {model.precompile_shaders}")
                logger.debug(f"  parallel_loading: {model.parallel_loading}")
            
            # Extract optimization flags from various sources
            compute_shader_optimized = False
            precompile_shaders = False
            parallel_loading = False
            
            # Try result dict first
            if isinstance(result, dict):
                compute_shader_optimized = result.get('compute_shader_optimized', False)
                precompile_shaders = result.get('precompile_shaders', False)
                parallel_loading = result.get('parallel_loading', False)
            
            # If not found in result, try model attributes
            if not compute_shader_optimized and hasattr(model, 'compute_shader_optimized'):
                compute_shader_optimized = model.compute_shader_optimized
                precompile_shaders = model.precompile_shaders
                parallel_loading = model.parallel_loading
            
            # If still not found, check if optimization flags were set in hardware_preferences
            if not compute_shader_optimized and 'compute_shaders' in hardware_preferences:
                compute_shader_optimized = hardware_preferences['compute_shaders']
                precompile_shaders = hardware_preferences['precompile_shaders']
                parallel_loading = hardware_preferences['parallel_loading']
            
            # Create result object with enhanced information
            test_result = {
                'model_name': model_name,
                'model_type': model_type,
                'platform': platform,
                'execution_time': execution_time,
                'success': isinstance(result, dict) and result.get('success', False) or True,
                'is_real_implementation': model_info.get('is_real_implementation', False),
                'browser': model_info.get('browser', 'unknown'),
                'compute_shader_optimized': compute_shader_optimized,
                'precompile_shaders': precompile_shaders,
                'parallel_loading': parallel_loading,
                'precision': getattr(self.args, 'precision', 16),
                'mixed_precision': getattr(self.args, 'mixed_precision', False),
                'test_method': "enhanced_resource_pool",
                'performance_metrics': performance_metrics
            }
            
            # Debug final flags
            logger.debug(f"Final test result flags for {model_name}: compute_shader_optimized={compute_shader_optimized}, precompile_shaders={precompile_shaders}, parallel_loading={parallel_loading}")
            
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
            
            # Generate a random ID for the record (a trick to avoid using AUTOINCREMENT which is not supported in DuckDB)
            # In a production environment, you would use a more robust ID generation strategy
            import random
            record_id = random.randint(1000000, 9999999)
            
            # Insert into database
            self.db_connection.execute("""
            INSERT INTO ipfs_resource_pool_test_results (
                id, timestamp, session_id, model_name, model_type, platform, browser,
                test_method, success, is_real_implementation, execution_time_sec,
                precision, mixed_precision, compute_shader_optimized, precompile_shaders,
                parallel_loading, performance_metrics, detailed_results
            ) VALUES (
                ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                record_id,
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
            # Initialize hardware_preferences
            hardware_preferences = {}
            
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
                
                # Create optimizations dictionary and add them to hardware_preferences
                optimizations = {}
                # Create or update hardware_preferences
                if 'hardware_preferences' not in locals():
                    hardware_preferences = {}
                
                # Start with all optimizations disabled
                hardware_preferences["compute_shaders"] = False
                hardware_preferences["precompile_shaders"] = False
                hardware_preferences["parallel_loading"] = False
                
                # Debug output
                logger.debug(f"Initial hardware_preferences: {hardware_preferences}")

                if hasattr(self.args, 'optimize_audio') and self.args.optimize_audio or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                    optimizations["compute_shaders"] = True
                    hardware_preferences["compute_shaders"] = True
                if hasattr(self.args, 'shader_precompile') and self.args.shader_precompile or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                    optimizations["precompile_shaders"] = True
                    hardware_preferences["precompile_shaders"] = True
                if hasattr(self.args, 'parallel_loading') and self.args.parallel_loading or hasattr(self.args, 'all_optimizations') and self.args.all_optimizations:
                    optimizations["parallel_loading"] = True
                    hardware_preferences["parallel_loading"] = True
                
                # Debug output after setting optimizations
                logger.debug(f"Optimizations: {optimizations}")
                logger.debug(f"Updated hardware_preferences: {hardware_preferences}")
                
                # Make sure hardware_preferences has priority_list
                if 'priority_list' not in hardware_preferences:
                    hardware_preferences['priority_list'] = [self.args.platform]
                    
                # Pass hardware_preferences to the get_model call 
                model = self.resource_pool_integration.get_model(
                    model_name=model_name,
                    model_type=model_type,
                    platform=self.args.platform,
                    batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 1,
                    quantization=quantization,
                    optimizations=optimizations,
                    hardware_preferences=hardware_preferences
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
                    
                    # Debug model attributes if available
                    if hasattr(model, "compute_shader_optimized"):
                        logger.debug(f"Model {model_name} optimization flags directly from model object:")
                        logger.debug(f"  compute_shader_optimized: {model.compute_shader_optimized}")
                        logger.debug(f"  precompile_shaders: {model.precompile_shaders}")
                        logger.debug(f"  parallel_loading: {model.parallel_loading}")
                    
                    # Extract optimization flags from result
                    compute_shader_optimized = False
                    precompile_shaders = False
                    parallel_loading = False
                    
                    # Try to get from result dict first, then from model_info, then from model attributes
                    if isinstance(result, dict):
                        compute_shader_optimized = result.get('compute_shader_optimized', model_info.get('compute_shader_optimized', getattr(model, 'compute_shader_optimized', False)))
                        precompile_shaders = result.get('precompile_shaders', model_info.get('precompile_shaders', getattr(model, 'precompile_shaders', False)))
                        parallel_loading = result.get('parallel_loading', model_info.get('parallel_loading', getattr(model, 'parallel_loading', False)))
                        
                        logger.debug(f"Result contains optimization flags: {result.get('compute_shader_optimized', None)}, {result.get('precompile_shaders', None)}, {result.get('parallel_loading', None)}")
                    
                    # Create result object
                    test_result = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'platform': self.args.platform,
                        'execution_time': execution_time,
                        'success': result is not None,
                        'is_real_implementation': model_info.get('is_real_implementation', False),
                        'browser': model_info.get('browser', 'unknown'),
                        'compute_shader_optimized': compute_shader_optimized,
                        'precompile_shaders': precompile_shaders,
                        'parallel_loading': parallel_loading,
                        'test_method': "concurrent_execution_enhanced",
                        'performance_metrics': performance_metrics
                    }
                    
                    logger.debug(f"Test result flags for {model_name}: {compute_shader_optimized}, {precompile_shaders}, {parallel_loading}")
                    
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
                await anyio.sleep(0.5)
            
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
            
            # Generate a random ID for the record
            import random
            record_id = random.randint(1000000, 9999999)
            
            # Insert benchmark results
            self.db_connection.execute("""
            INSERT INTO ipfs_resource_pool_benchmark_results (
                id, timestamp, session_id, benchmark_type, total_models, successful_models,
                execution_time_sec, models_tested, success_rate, real_hardware_rate,
                throughput_improvement_factor, detailed_results
            ) VALUES (
                ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                record_id,
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
        return anyio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())