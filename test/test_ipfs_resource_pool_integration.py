#!/usr/bin/env python3
"""
Test IPFS Acceleration with Resource Pool Integration

This script tests the integration between IPFS acceleration and the ResourcePool
for efficient WebNN/WebGPU hardware utilization.

Usage:
    python test_ipfs_resource_pool_integration.py --model bert-base-uncased --platform webgpu
    python test_ipfs_resource_pool_integration.py --concurrent-models
    python test_ipfs_resource_pool_integration.py --benchmark
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

# Required modules
REQUIRED_MODULES = {
    "resource_pool_bridge": False,
    "ipfs_accelerate_impl": False,
    "duckdb": False
}

# Check for resource_pool_bridge
try:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    REQUIRED_MODULES["resource_pool_bridge"] = True
except ImportError:
    logger.error("ResourcePoolBridge not available. Make sure fixed_web_platform module is installed")

# Check for ipfs_accelerate_impl
try:
    import ipfs_accelerate_impl
    REQUIRED_MODULES["ipfs_accelerate_impl"] = True
except ImportError:
    logger.error("IPFS accelerate implementation not available")

# Check for duckdb
try:
    import duckdb
    REQUIRED_MODULES["duckdb"] = True
except ImportError:
    logger.warning("DuckDB not available. Database integration will be disabled")

class IPFSResourcePoolTester:
    """Test IPFS Acceleration with Resource Pool Integration."""
    
    def __init__(self, args):
        """Initialize tester with command line arguments."""
        self.args = args
        self.results = []
        self.ipfs_module = None
        self.resource_pool_integration = None
        self.db_connection = None
        
        # Set environment variables for optimizations if needed
        if args.optimize_audio:
            os.environ["USE_FIREFOX_WEBGPU"] = "1"
            os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
            logger.info("Enabled Firefox audio optimizations for audio models")
        
        if args.shader_precompile:
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
            logger.info("Enabled shader precompilation")
        
        if args.parallel_loading:
            os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
            logger.info("Enabled parallel model loading")
        
        # Import IPFS module
        if REQUIRED_MODULES["ipfs_accelerate_impl"]:
            self.ipfs_module = ipfs_accelerate_impl
            logger.info("IPFS accelerate module imported successfully")
        else:
            logger.error("IPFS accelerate module not available, tests will fail")
            return
        
        # Connect to database if specified
        if args.db_path and REQUIRED_MODULES["duckdb"]:
            try:
                self.db_connection = duckdb.connect(args.db_path)
                logger.info(f"Connected to database: {args.db_path}")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.db_connection = None
    
    async def initialize_resource_pool(self):
        """Initialize the resource pool integration."""
        if not REQUIRED_MODULES["resource_pool_bridge"]:
            logger.error("Cannot initialize resource pool: ResourcePoolBridge not available")
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
            self.resource_pool_integration = ResourcePoolBridgeIntegration(
                max_connections=self.args.max_connections,
                enable_gpu=True,
                enable_cpu=True,
                headless=not self.args.visible,
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_ipfs=True,
                db_path=self.args.db_path
            )
            
            # Initialize integration
            self.resource_pool_integration.initialize()
            logger.info("Resource pool integration initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize resource pool integration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_model_direct(self, model_name, model_type):
        """Test a model using direct ResourcePoolBridge integration."""
        if not self.resource_pool_integration:
            logger.error("Cannot test model: resource pool integration not initialized")
            return None
        
        try:
            logger.info(f"Testing model directly with resource pool: {model_name} ({model_type})")
            
            platform = self.args.platform
            
            # Configure hardware preferences with browser optimizations
            hardware_preferences = {
                'priority_list': [platform, 'cpu'],
                'model_family': model_type,
                'enable_ipfs': True,
                'precision': self.args.precision,
                'mixed_precision': self.args.mixed_precision,
                'browser': self.args.browser
            }
            
            # For audio models, use Firefox optimizations
            if model_type == 'audio' and self.args.optimize_audio:
                hardware_preferences['browser'] = 'firefox'
                hardware_preferences['use_firefox_optimizations'] = True
                logger.info("Using Firefox with audio optimizations for audio model")
            
            # Get model from resource pool
            model = self.resource_pool_integration.get_model(
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
            
            if not model:
                logger.error(f"Failed to get model: {model_name}")
                return None
            
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
            
            # Run inference
            start_time = time.time()
            result = model(test_input)
            execution_time = time.time() - start_time
            
            # Create result object with enhanced information
            test_result = {
                'model_name': model_name,
                'model_type': model_type,
                'platform': platform,
                'execution_time': execution_time,
                'success': result.get('success', result.get('status') == 'success'),
                'is_real_implementation': result.get('is_real_implementation', False),
                'browser': result.get('browser', 'unknown'),
                'compute_shader_optimized': result.get('compute_shader_optimized', False),
                'precompile_shaders': result.get('precompile_shaders', False),
                'parallel_loading': result.get('parallel_loading', False),
                'precision': self.args.precision,
                'mixed_precision': self.args.mixed_precision,
                'test_method': "direct_resource_pool"
            }
            
            self.results.append(test_result)
            
            logger.info(f"Direct resource pool test completed in {execution_time:.2f}s: {model_name}")
            return test_result
        except Exception as e:
            logger.error(f"Error testing model directly: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def test_model_ipfs(self, model_name, model_type):
        """Test a model using IPFS acceleration with resource pool integration."""
        if not self.ipfs_module:
            logger.error("Cannot test model: IPFS module not available")
            return None
        
        try:
            logger.info(f"Testing model with IPFS acceleration: {model_name} ({model_type})")
            
            platform = self.args.platform
            
            # Configure acceleration options
            config = {
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
            
            # Run IPFS acceleration with resource pool
            start_time = time.time()
            result = self.ipfs_module.accelerate(model_name, test_input, config)
            execution_time = time.time() - start_time
            
            # Extract performance metrics
            performance_metrics = {}
            if isinstance(result, dict):
                performance_metrics = {
                    'latency_ms': result.get('latency_ms', 0),
                    'throughput_items_per_sec': result.get('throughput_items_per_sec', 0),
                    'memory_usage_mb': result.get('memory_usage_mb', 0)
                }
            
            # Create result object with enhanced information
            test_result = {
                'model_name': model_name,
                'model_type': model_type,
                'platform': platform,
                'execution_time': execution_time,
                'success': result.get('status') == 'success',
                'is_real_hardware': result.get('is_real_hardware', False),
                'is_simulation': result.get('is_simulation', not result.get('is_real_hardware', False)),
                'browser': result.get('browser', 'unknown'),
                'precision': result.get('precision', self.args.precision),
                'mixed_precision': result.get('mixed_precision', self.args.mixed_precision),
                'firefox_optimizations': result.get('firefox_optimizations', False),
                'ipfs_cache_hit': result.get('ipfs_cache_hit', False),
                'ipfs_source': result.get('ipfs_source'),
                'p2p_optimized': result.get('p2p_optimized', False),
                'resource_pool_used': result.get('resource_pool_used', False),
                'performance_metrics': performance_metrics,
                'test_method': "ipfs_acceleration"
            }
            
            self.results.append(test_result)
            
            logger.info(f"IPFS acceleration test completed in {execution_time:.2f}s: {model_name}")
            return test_result
        except Exception as e:
            logger.error(f"Error testing model with IPFS acceleration: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def test_concurrent_models(self):
        """Test multiple models concurrently using resource pool integration."""
        if not self.resource_pool_integration:
            logger.error("Cannot test concurrent models: resource pool integration not initialized")
            return []
        
        try:
            # Define models to test
            models = []
            
            if self.args.models:
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
                        else:
                            model_type = "text"
                    
                    models.append((model_type, model_name))
            else:
                # Use default models
                models = [
                    ("text_embedding", "bert-base-uncased"),
                    ("vision", "google/vit-base-patch16-224"),
                    ("audio", "openai/whisper-tiny")
                ]
            
            logger.info(f"Testing {len(models)} models concurrently")
            
            # Create model configurations
            model_configs = []
            for i, (model_type, model_name) in enumerate(models):
                model_configs.append({
                    'model_type': model_type,
                    'model_name': model_name,
                    'model_id': f"model_{i}"
                })
            
            # Get models concurrently
            loaded_models = self.resource_pool_integration.get_models_concurrent(model_configs)
            
            if not loaded_models:
                logger.error("Failed to load any models concurrently")
                return []
            
            logger.info(f"Successfully loaded {len(loaded_models)} models concurrently")
            
            # Prepare inputs for concurrent execution
            models_and_inputs = []
            for model_id, model in loaded_models.items():
                # Get model type from model configuration
                model_type = next((config['model_type'] for config in model_configs if config['model_id'] == model_id), "text")
                
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
                
                models_and_inputs.append((model_id, test_input))
            
            # Run concurrent execution
            start_time = time.time()
            concurrent_results = self.resource_pool_integration.execute_concurrent(models_and_inputs)
            execution_time = time.time() - start_time
            
            # Process results
            test_results = []
            for i, result in enumerate(concurrent_results):
                if i < len(models):
                    model_type, model_name = models[i]
                    
                    # Extract performance metrics
                    performance_metrics = {}
                    if isinstance(result, dict) and 'performance_metrics' in result:
                        performance_metrics = result['performance_metrics']
                    
                    # Create result object
                    test_result = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'platform': self.args.platform,
                        'execution_time': execution_time,
                        'success': result.get('success', False),
                        'is_real_implementation': result.get('is_real_implementation', False),
                        'browser': result.get('browser', 'unknown'),
                        'performance_metrics': performance_metrics,
                        'test_method': "concurrent_execution"
                    }
                    
                    test_results.append(test_result)
                    self.results.append(test_result)
            
            logger.info(f"Concurrent execution of {len(models)} models completed in {execution_time:.2f}s")
            
            return test_results
        except Exception as e:
            logger.error(f"Error testing concurrent models: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def run_benchmark(self):
        """Run a benchmark comparing direct resource pool, IPFS acceleration, and concurrent execution."""
        if not self.ipfs_module or not REQUIRED_MODULES["resource_pool_bridge"]:
            logger.error("Cannot run benchmark: required modules not available")
            return []
        
        try:
            # Initialize resource pool if not already initialized
            if not self.resource_pool_integration:
                if not await self.initialize_resource_pool():
                    logger.error("Failed to initialize resource pool for benchmark")
                    return []
            
            # Define models to benchmark
            if self.args.models:
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
                        else:
                            model_type = "text"
                    
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
                "direct_resource_pool": [],
                "ipfs_acceleration": [],
                "concurrent_execution": []
            }
            
            # 1. Test each model with direct resource pool
            logger.info("Running benchmark with direct resource pool...")
            for model_type, model_name in models:
                result = await self.test_model_direct(model_name, model_type)
                if result:
                    benchmark_results["direct_resource_pool"].append(result)
                
                # Wait a bit between tests
                await asyncio.sleep(0.5)
            
            # 2. Test each model with IPFS acceleration
            logger.info("Running benchmark with IPFS acceleration...")
            for model_type, model_name in models:
                result = await self.test_model_ipfs(model_name, model_type)
                if result:
                    benchmark_results["ipfs_acceleration"].append(result)
                
                # Wait a bit between tests
                await asyncio.sleep(0.5)
            
            # 3. Test all models concurrently
            logger.info("Running benchmark with concurrent execution...")
            concurrent_results = await self.test_concurrent_models()
            benchmark_results["concurrent_execution"] = concurrent_results
            
            # Calculate benchmark summary
            summary = self._calculate_benchmark_summary(benchmark_results)
            
            # Print benchmark summary
            self._print_benchmark_summary(summary)
            
            # Save benchmark results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ipfs_resource_pool_benchmark_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    "results": benchmark_results,
                    "summary": summary
                }, f, indent=2)
            
            logger.info(f"Benchmark results saved to {filename}")
            
            return benchmark_results
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_benchmark_summary(self, benchmark_results):
        """Calculate summary statistics for benchmark results."""
        summary = {}
        
        # Helper function to calculate average execution time
        def calc_avg_time(results):
            if not results:
                return 0
            return sum(r.get('execution_time', 0) for r in results) / len(results)
        
        # Calculate average execution time for each method
        summary['avg_execution_time'] = {
            'direct_resource_pool': calc_avg_time(benchmark_results['direct_resource_pool']),
            'ipfs_acceleration': calc_avg_time(benchmark_results['ipfs_acceleration']),
            'concurrent_execution': calc_avg_time(benchmark_results['concurrent_execution']) 
                                    if benchmark_results['concurrent_execution'] else 0
        }
        
        # Calculate success rates
        summary['success_rate'] = {
            'direct_resource_pool': sum(1 for r in benchmark_results['direct_resource_pool'] if r.get('success', False)) / 
                                    len(benchmark_results['direct_resource_pool']) if benchmark_results['direct_resource_pool'] else 0,
            'ipfs_acceleration': sum(1 for r in benchmark_results['ipfs_acceleration'] if r.get('success', False)) / 
                                len(benchmark_results['ipfs_acceleration']) if benchmark_results['ipfs_acceleration'] else 0,
            'concurrent_execution': sum(1 for r in benchmark_results['concurrent_execution'] if r.get('success', False)) / 
                                   len(benchmark_results['concurrent_execution']) if benchmark_results['concurrent_execution'] else 0
        }
        
        # Calculate real hardware vs simulation rates
        summary['real_hardware_rate'] = {
            'direct_resource_pool': sum(1 for r in benchmark_results['direct_resource_pool'] if r.get('is_real_implementation', False)) / 
                                    len(benchmark_results['direct_resource_pool']) if benchmark_results['direct_resource_pool'] else 0,
            'ipfs_acceleration': sum(1 for r in benchmark_results['ipfs_acceleration'] if r.get('is_real_hardware', False)) / 
                                len(benchmark_results['ipfs_acceleration']) if benchmark_results['ipfs_acceleration'] else 0,
            'concurrent_execution': sum(1 for r in benchmark_results['concurrent_execution'] if r.get('is_real_implementation', False)) / 
                                   len(benchmark_results['concurrent_execution']) if benchmark_results['concurrent_execution'] else 0
        }
        
        # Calculate IPFS-specific metrics
        if benchmark_results['ipfs_acceleration']:
            summary['ipfs_cache_hit_rate'] = sum(1 for r in benchmark_results['ipfs_acceleration'] if r.get('ipfs_cache_hit', False)) / len(benchmark_results['ipfs_acceleration'])
            summary['p2p_optimization_rate'] = sum(1 for r in benchmark_results['ipfs_acceleration'] if r.get('p2p_optimized', False)) / len(benchmark_results['ipfs_acceleration'])
        else:
            summary['ipfs_cache_hit_rate'] = 0
            summary['p2p_optimization_rate'] = 0
        
        # Calculate throughput improvement
        if benchmark_results['concurrent_execution'] and benchmark_results['direct_resource_pool']:
            direct_time = calc_avg_time(benchmark_results['direct_resource_pool'])
            concurrent_time = calc_avg_time(benchmark_results['concurrent_execution'])
            
            if direct_time > 0:
                # Calculate improvement factor (higher is better)
                # This is an approximation since concurrent execution returns multiple results in one call
                direct_items_per_second = 1 / direct_time
                concurrent_items_per_second = len(benchmark_results['concurrent_execution']) / concurrent_time
                improvement_factor = concurrent_items_per_second / direct_items_per_second if direct_items_per_second > 0 else 0
                
                summary['throughput_improvement_factor'] = improvement_factor
        else:
            summary['throughput_improvement_factor'] = 0
        
        return summary
    
    def _print_benchmark_summary(self, summary):
        """Print a summary of benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        print("\nAverage Execution Time (seconds):")
        print(f"  Direct Resource Pool:  {summary['avg_execution_time']['direct_resource_pool']:.3f}")
        print(f"  IPFS Acceleration:     {summary['avg_execution_time']['ipfs_acceleration']:.3f}")
        print(f"  Concurrent Execution:  {summary['avg_execution_time']['concurrent_execution']:.3f}")
        
        print("\nSuccess Rate:")
        print(f"  Direct Resource Pool:  {summary['success_rate']['direct_resource_pool']*100:.1f}%")
        print(f"  IPFS Acceleration:     {summary['success_rate']['ipfs_acceleration']*100:.1f}%")
        print(f"  Concurrent Execution:  {summary['success_rate']['concurrent_execution']*100:.1f}%")
        
        print("\nReal Hardware Rate:")
        print(f"  Direct Resource Pool:  {summary['real_hardware_rate']['direct_resource_pool']*100:.1f}%")
        print(f"  IPFS Acceleration:     {summary['real_hardware_rate']['ipfs_acceleration']*100:.1f}%")
        print(f"  Concurrent Execution:  {summary['real_hardware_rate']['concurrent_execution']*100:.1f}%")
        
        print("\nIPFS-Specific Metrics:")
        print(f"  Cache Hit Rate:        {summary['ipfs_cache_hit_rate']*100:.1f}%")
        print(f"  P2P Optimization Rate: {summary['p2p_optimization_rate']*100:.1f}%")
        
        print("\nThroughput Improvement:")
        print(f"  Concurrent vs Direct:  {summary['throughput_improvement_factor']:.2f}x")
        
        print("="*80)
    
    async def close(self):
        """Close resources."""
        if self.resource_pool_integration:
            self.resource_pool_integration.close()
            logger.info("Resource pool integration closed")
        
        if self.db_connection:
            self.db_connection.close()
            logger.info("Database connection closed")
    
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
                    real_hw = '✅' if result.get('is_real_implementation', result.get('is_real_hardware', False)) else '❌'
                    execution_time = f"{result.get('execution_time', 0):.2f}"
                    
                    f.write(f"| {model_name} | {model_type} | {platform} | {browser} | {success} | {real_hw} | {execution_time} |\n")
                
                f.write("\n")
            
            # Additional details for IPFS acceleration tests
            if 'ipfs_acceleration' in methods:
                f.write("## IPFS Acceleration Details\n\n")
                
                f.write("| Model | Cache Hit | P2P Optimized | IPFS Source | Resource Pool Used |\n")
                f.write("|-------|-----------|--------------|-------------|-------------------|\n")
                
                for result in sorted(methods['ipfs_acceleration'], key=lambda r: r.get('model_name', '')):
                    model_name = result.get('model_name', 'unknown')
                    cache_hit = '✅' if result.get('ipfs_cache_hit', False) else '❌'
                    p2p_optimized = '✅' if result.get('p2p_optimized', False) else '❌'
                    ipfs_source = result.get('ipfs_source', 'N/A')
                    resource_pool_used = '✅' if result.get('resource_pool_used', False) else '❌'
                    
                    f.write(f"| {model_name} | {cache_hit} | {p2p_optimized} | {ipfs_source} | {resource_pool_used} |\n")
                
                f.write("\n")
            
            logger.info(f"Markdown report saved to {filename}")

async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(description="Test IPFS Acceleration with Resource Pool Integration")
    
    # Model selection options
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                      help="Model to test")
    parser.add_argument("--models", type=str,
                      help="Comma-separated list of models to test (model_type:model_name format)")
    parser.add_argument("--model-type", type=str, choices=["text", "text_embedding", "vision", "audio", "multimodal"],
                      default="text_embedding", help="Model type")
    
    # Platform options
    parser.add_argument("--platform", type=str, choices=["webnn", "webgpu"], default="webgpu",
                      help="Platform to test")
    
    # Browser options
    parser.add_argument("--browser", type=str, choices=["chrome", "firefox", "edge", "safari"],
                      help="Browser to use")
    parser.add_argument("--visible", action="store_true",
                      help="Run browsers in visible mode (not headless)")
    parser.add_argument("--max-connections", type=int, default=4,
                      help="Maximum number of browser connections")
    
    # Precision options
    parser.add_argument("--precision", type=int, choices=[4, 8, 16, 32], default=16,
                      help="Precision level")
    parser.add_argument("--mixed-precision", action="store_true",
                      help="Use mixed precision")
    
    # Optimization options
    parser.add_argument("--optimize-audio", action="store_true",
                      help="Enable Firefox audio optimizations")
    parser.add_argument("--shader-precompile", action="store_true",
                      help="Enable shader precompilation")
    parser.add_argument("--parallel-loading", action="store_true",
                      help="Enable parallel model loading")
    
    # Test options
    parser.add_argument("--test-method", type=str, choices=["direct", "ipfs", "concurrent", "all"],
                      default="all", help="Test method to use")
    parser.add_argument("--concurrent-models", action="store_true",
                      help="Test multiple models concurrently")
    parser.add_argument("--benchmark", action="store_true",
                      help="Run benchmark comparing all methods")
    
    # Database options
    parser.add_argument("--db-path", type=str,
                      help="Path to database")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check required modules
    missing_modules = [name for name, available in REQUIRED_MODULES.items() if not available]
    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        return 1
    
    # Create tester
    tester = IPFSResourcePoolTester(args)
    
    try:
        # Initialize resource pool
        if not args.test_method == "ipfs":
            if not await tester.initialize_resource_pool():
                logger.error("Failed to initialize resource pool")
                return 1
        
        # Run tests based on test method
        if args.benchmark:
            # Run benchmark comparing all methods
            await tester.run_benchmark()
        elif args.concurrent_models:
            # Test multiple models concurrently
            await tester.test_concurrent_models()
        else:
            # Run tests based on test method
            if args.test_method == "direct" or args.test_method == "all":
                await tester.test_model_direct(args.model, args.model_type)
            
            if args.test_method == "ipfs" or args.test_method == "all":
                await tester.test_model_ipfs(args.model, args.model_type)
        
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