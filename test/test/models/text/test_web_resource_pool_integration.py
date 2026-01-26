#!/usr/bin/env python3
"""
Test WebGPU/WebNN Resource Pool Integration with Fault Tolerance

This script tests the enhanced resource pool integration with enterprise-grade
fault tolerance features and distributed testing framework integration.

Usage:
    python test_web_resource_pool_integration.py --enable-fault-tolerance
    python test_web_resource_pool_integration.py --test-recovery
    python test_web_resource_pool_integration.py --test-model-sharding
    python test_web_resource_pool_integration.py --comprehensive
"""

import os
import sys
import json
import time
import anyio
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

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
    from fixed_web_platform.resource_pool_bridge_integration import ResourcePoolBridgeIntegration
    from fixed_web_platform.fault_tolerant_model_sharding import FaultTolerantModelSharding
    from fixed_web_platform.fault_tolerance_validation import FaultToleranceValidator
    RESOURCE_POOL_AVAILABLE = True
except ImportError as e:
    logger.error(f"ResourcePool components not available: {e}")
    RESOURCE_POOL_AVAILABLE = False

# Try to import distributed testing components
try:
    from distributed_testing.plugins.resource_pool_plugin import ResourcePoolPlugin
    from distributed_testing.worker_registry import WorkerRegistry
    from distributed_testing.circuit_breaker import CircuitBreaker
    DISTRIBUTED_TESTING_AVAILABLE = True
except ImportError:
    logger.warning("Distributed testing framework not available, some tests will be limited")
    DISTRIBUTED_TESTING_AVAILABLE = False

# Define test models and configurations
TEST_MODELS = [
    # Model name, model type, platform, platform_fallback
    ("bert-base-uncased", "text_embedding", "webgpu", "cpu"),
    ("google/vit-base-patch16-224", "vision", "webgpu", "cpu"),
    ("openai/whisper-tiny", "audio", "webgpu", "cpu"),
    ("t5-small", "text", "webgpu", "cpu"),
]

# Define browsers to test
TEST_BROWSERS = ["chrome", "firefox", "edge"]

# Define fault injection scenarios
FAULT_SCENARIOS = [
    "connection_lost",
    "browser_crash",
    "component_timeout",
    "multi_browser_failure",
]

class WebResourcePoolIntegrationTester:
    """Test WebGPU/WebNN Resource Pool Integration with fault tolerance"""
    
    def __init__(self, args):
        """Initialize with command line arguments"""
        self.args = args
        self.results = {}
        self.integration = None
        self.fault_tolerant_integration = None
        self.resource_pool_plugin = None
        
        # Configure logging level
        if hasattr(args, 'verbose') and args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Verbose logging enabled")
        
        # Configure test parameters
        self.fault_tolerance_level = args.fault_tolerance_level
        self.recovery_strategy = args.recovery_strategy
        self.db_path = args.db_path if hasattr(args, 'db_path') else None
        
        # Set environment variables for optimizations
        if args.enable_optimizations:
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
            os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
            logger.info("Enabled browser optimizations")
    
    async def initialize(self) -> bool:
        """Initialize the resource pool integration with fault tolerance"""
        if not RESOURCE_POOL_AVAILABLE:
            logger.error("Cannot initialize: Resource Pool components not available")
            return False
        
        try:
            # Configure browser preferences with optimization settings
            browser_preferences = {
                'audio': 'firefox',  # Firefox has better compute shader performance for audio
                'vision': 'chrome',  # Chrome has good WebGPU support for vision models
                'text_embedding': 'edge'  # Edge has excellent WebNN support for text embeddings
            }
            
            # Override browser preferences if specific browser is selected
            if hasattr(self.args, 'chrome') and self.args.chrome:
                browser_preferences = {k: 'chrome' for k in browser_preferences}
            elif hasattr(self.args, 'firefox') and self.args.firefox:
                browser_preferences = {k: 'firefox' for k in browser_preferences}
            elif hasattr(self.args, 'edge') and self.args.edge:
                browser_preferences = {k: 'edge' for k in browser_preferences}
            
            # Create standard integration first for comparison
            self.integration = ResourcePoolBridgeIntegration(
                max_connections=self.args.max_connections,
                min_connections=1,
                enable_gpu=True,
                enable_cpu=True,
                headless=not self.args.visible,
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_ipfs=not hasattr(self.args, 'disable_ipfs') or not self.args.disable_ipfs,
                db_path=self.db_path,
                enable_heartbeat=True
            )
            
            # Initialize standard integration
            self.integration.initialize()
            logger.info("Standard ResourcePoolBridgeIntegration initialized")
            
            # If fault tolerance is enabled, create enhanced integration with fault tolerance
            if self.args.enable_fault_tolerance:
                # Initialize fault-tolerant integration with distributed testing
                if DISTRIBUTED_TESTING_AVAILABLE:
                    # Create worker registry
                    worker_registry = WorkerRegistry("resource-pool")
                    
                    # Create resource pool plugin
                    self.resource_pool_plugin = ResourcePoolPlugin(
                        integration=self.integration,
                        fault_tolerance_level=self.fault_tolerance_level,
                        recovery_strategy=self.recovery_strategy
                    )
                    
                    # Register plugin with worker registry
                    for i, browser in enumerate(TEST_BROWSERS):
                        worker_id = f"browser-{i}"
                        await worker_registry.register(worker_id, {
                            "type": browser,
                            "capabilities": ["webgpu", "webnn"] if browser == "edge" else ["webgpu"],
                            "status": "ready",
                        })
                    
                    # Initialize plugin
                    await self.resource_pool_plugin.initialize()
                    logger.info("ResourcePoolPlugin initialized with distributed testing framework")
                
                logger.info(f"Enhanced integration initialized with fault tolerance level: {self.fault_tolerance_level}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def close(self):
        """Close resource pool integration"""
        try:
            if self.integration:
                self.integration.close()
                logger.info("Resource pool integration closed")
            
            if self.resource_pool_plugin and hasattr(self.resource_pool_plugin, 'shutdown'):
                await self.resource_pool_plugin.shutdown()
                logger.info("Resource pool plugin shut down")
        except Exception as e:
            logger.error(f"Error closing integration: {e}")
    
    async def test_standard_integration(self):
        """Run standard integration tests for baseline"""
        logger.info("Testing standard integration")
        
        results = {}
        
        for model_name, model_type, platform, fallback in TEST_MODELS:
            try:
                logger.info(f"Testing model: {model_name} ({model_type}) on {platform}")
                
                # Configure hardware preferences
                hardware_preferences = {
                    'priority_list': [platform, fallback],
                    'model_family': model_type,
                    'enable_ipfs': not hasattr(self.args, 'disable_ipfs') or not self.args.disable_ipfs,
                }
                
                # Create browser-specific optimizations
                if model_type == 'audio':
                    hardware_preferences['browser'] = 'firefox'
                    hardware_preferences['use_firefox_optimizations'] = True
                elif model_type == 'text_embedding' and platform == 'webnn':
                    hardware_preferences['browser'] = 'edge'
                elif model_type == 'vision':
                    hardware_preferences['browser'] = 'chrome'
                    hardware_preferences['precompile_shaders'] = True
                
                # Get model from resource pool
                start_time = time.time()
                model = self.integration.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
                load_time = time.time() - start_time
                
                if not model:
                    logger.error(f"Failed to get model: {model_name}")
                    results[model_name] = {
                        "success": False,
                        "error": "Failed to load model"
                    }
                    continue
                
                # Create appropriate test input
                test_input = self._create_test_input(model_type)
                
                # Run inference
                start_time = time.time()
                result = model(test_input)
                inference_time = time.time() - start_time
                
                # Log result
                success = result.get('success', False) or result.get('status') == 'success'
                logger.info(f"Model: {model_name}, Success: {success}, Time: {inference_time:.2f}s")
                
                # Store result
                results[model_name] = {
                    "success": success,
                    "load_time": load_time,
                    "inference_time": inference_time,
                    "platform": platform,
                    "browser": result.get('browser', 'unknown'),
                    "is_real_implementation": result.get('is_real_implementation', False)
                }
                
            except Exception as e:
                logger.error(f"Error testing {model_name}: {e}")
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.results["standard_integration"] = results
        return results
    
    async def test_fault_tolerant_model_sharding(self):
        """Test fault-tolerant model sharding"""
        logger.info("Testing fault-tolerant model sharding")
        
        results = {}
        
        try:
            # Choose a larger model for sharding
            model_name = "llama-7b"  # Simulated large model
            
            # Create fault-tolerant model sharding manager
            model_manager = FaultTolerantModelSharding(
                model_name=model_name,
                browsers=TEST_BROWSERS,
                fault_tolerance_level=self.fault_tolerance_level,
                recovery_strategy=self.recovery_strategy,
                connection_pool=self.integration  # Use integration as connection pool
            )
            
            # Initialize sharding
            init_result = await model_manager.initialize(
                shard_type="optimal",
                enable_state_replication=True
            )
            
            logger.info(f"Model sharding initialization: {init_result}")
            
            if "status" in init_result and init_result["status"] in ["ready", "degraded"]:
                # Run inference
                inference_input = {"input": "This is a test input for sharded model inference."}
                
                start_time = time.time()
                inference_result = await model_manager.run_inference(inference_input)
                inference_time = time.time() - start_time
                
                # Get recovery statistics
                recovery_stats = model_manager.get_recovery_statistics()
                
                # Store results
                results = {
                    "model_name": model_name,
                    "sharding_initialized": True,
                    "initialization_status": init_result["status"],
                    "inference_success": inference_result.get("success", False),
                    "inference_time": inference_time,
                    "browser_count": len(TEST_BROWSERS),
                    "recovery_stats": recovery_stats,
                    "fault_tolerance_level": self.fault_tolerance_level,
                    "recovery_strategy": self.recovery_strategy
                }
                
                logger.info(f"Sharded model inference complete: {results['inference_success']}")
                
                # Shutdown manager
                await model_manager.shutdown()
                logger.info("Model sharding manager shut down")
            else:
                results = {
                    "model_name": model_name,
                    "sharding_initialized": False,
                    "initialization_error": init_result
                }
                
                logger.error(f"Failed to initialize model sharding: {init_result}")
            
        except Exception as e:
            logger.error(f"Error during fault-tolerant model sharding test: {e}")
            import traceback
            traceback.print_exc()
            results = {
                "sharding_initialized": False,
                "error": str(e)
            }
        
        self.results["fault_tolerant_sharding"] = results
        return results
    
    async def test_fault_tolerance_recovery(self):
        """Test fault tolerance recovery capabilities"""
        logger.info("Testing fault tolerance recovery capabilities")
        
        if not self.args.enable_fault_tolerance:
            logger.warning("Fault tolerance not enabled, skipping recovery tests")
            return {"skipped": "Fault tolerance not enabled"}
        
        results = {}
        
        try:
            # Choose a model for testing
            model_name, model_type, platform, fallback = TEST_MODELS[0]
            
            # Create fault-tolerant model sharding manager
            model_manager = FaultTolerantModelSharding(
                model_name=model_name,
                browsers=TEST_BROWSERS,
                fault_tolerance_level=self.fault_tolerance_level,
                recovery_strategy=self.recovery_strategy,
                connection_pool=self.integration  # Use integration as connection pool
            )
            
            # Initialize sharding
            await model_manager.initialize()
            
            # Create validator
            validator = FaultToleranceValidator(
                model_manager=model_manager,
                config={
                    'fault_tolerance_level': self.fault_tolerance_level,
                    'recovery_strategy': self.recovery_strategy,
                    'test_scenarios': FAULT_SCENARIOS[:2]  # Test first two scenarios
                }
            )
            
            # Run validation
            validation_results = await validator.validate_fault_tolerance()
            
            # Analyze results
            analysis = validator.analyze_results(validation_results)
            
            # Store results
            results = {
                "model_name": model_name,
                "validation_status": validation_results.get("validation_status", "unknown"),
                "scenarios_tested": validation_results.get("scenarios_tested", []),
                "overall_metrics": validation_results.get("overall_metrics", {}),
                "analysis": {
                    "strengths": analysis.get("strengths", []),
                    "weaknesses": analysis.get("weaknesses", []),
                    "recommendations": analysis.get("recommendations", [])
                }
            }
            
            # Add details for each scenario
            for scenario in validation_results.get("scenarios_tested", []):
                if scenario in validation_results.get("scenario_results", {}):
                    scenario_result = validation_results["scenario_results"][scenario]
                    results[f"scenario_{scenario}"] = {
                        "success": scenario_result.get("success", False),
                        "recovery_time_ms": scenario_result.get("recovery_time_ms", 0),
                        "metrics": scenario_result.get("metrics", {})
                    }
            
            logger.info(f"Fault tolerance validation complete: {results['validation_status']}")
            
            # Shutdown manager
            await model_manager.shutdown()
            
        except Exception as e:
            logger.error(f"Error during fault tolerance validation: {e}")
            import traceback
            traceback.print_exc()
            results = {
                "validation_status": "error",
                "error": str(e)
            }
        
        self.results["fault_tolerance_validation"] = results
        return results
    
    async def test_distributed_integration(self):
        """Test integration with distributed testing framework"""
        logger.info("Testing integration with distributed testing framework")
        
        if not DISTRIBUTED_TESTING_AVAILABLE:
            logger.warning("Distributed testing framework not available, skipping integration tests")
            return {"skipped": "Distributed testing framework not available"}
        
        if not self.resource_pool_plugin:
            logger.warning("Resource pool plugin not initialized, skipping distributed integration tests")
            return {"skipped": "Resource pool plugin not initialized"}
        
        results = {}
        
        try:
            # Test plugin functionality
            plugin_status = await self.resource_pool_plugin.get_status()
            
            # Run a model through the plugin
            model_name, model_type, platform, fallback = TEST_MODELS[0]
            
            # Run task through plugin
            task_result = await self.resource_pool_plugin.execute_task({
                "action": "run_model",
                "model_name": model_name,
                "model_type": model_type,
                "platform": platform,
                "inputs": self._create_test_input(model_type)
            })
            
            # Get metrics from plugin
            plugin_metrics = await self.resource_pool_plugin.get_metrics()
            
            # Store results
            results = {
                "plugin_status": plugin_status,
                "task_execution": {
                    "success": task_result.get("success", False),
                    "model_name": model_name,
                    "execution_time": task_result.get("execution_time", 0)
                },
                "plugin_metrics": plugin_metrics
            }
            
            logger.info(f"Distributed integration test complete: {results['task_execution']['success']}")
            
        except Exception as e:
            logger.error(f"Error during distributed integration test: {e}")
            import traceback
            traceback.print_exc()
            results = {
                "success": False,
                "error": str(e)
            }
        
        self.results["distributed_integration"] = results
        return results
    
    async def test_concurrent_execution(self):
        """Test concurrent model execution with fault tolerance"""
        logger.info("Testing concurrent model execution with fault tolerance")
        
        results = {}
        
        try:
            # Prepare models for testing
            models_to_test = TEST_MODELS[:min(len(TEST_MODELS), self.args.max_connections)]
            model_configs = []
            
            for model_name, model_type, platform, fallback in models_to_test:
                # Configure hardware preferences
                hardware_preferences = {
                    'priority_list': [platform, fallback],
                    'model_family': model_type,
                    'enable_ipfs': not hasattr(self.args, 'disable_ipfs') or not self.args.disable_ipfs,
                }
                
                # Create browser-specific optimizations
                if model_type == 'audio':
                    hardware_preferences['browser'] = 'firefox'
                    hardware_preferences['use_firefox_optimizations'] = True
                elif model_type == 'text_embedding' and platform == 'webnn':
                    hardware_preferences['browser'] = 'edge'
                elif model_type == 'vision':
                    hardware_preferences['browser'] = 'chrome'
                    hardware_preferences['precompile_shaders'] = True
                
                model_configs.append({
                    "model_name": model_name,
                    "model_type": model_type,
                    "hardware_preferences": hardware_preferences
                })
            
            # Load models
            models = []
            model_inputs = []
            
            for config in model_configs:
                model = self.integration.get_model(
                    model_type=config["model_type"],
                    model_name=config["model_name"],
                    hardware_preferences=config["hardware_preferences"]
                )
                
                if model:
                    models.append(model)
                    model_inputs.append((model.model_id, self._create_test_input(config["model_type"])))
            
            # Run concurrent execution with fault tolerance
            if self.args.enable_fault_tolerance and self.args.inject_fault:
                # Inject fault during execution
                logger.info("Injecting fault during concurrent execution")
                
                # Start a background task to inject fault after a delay
                async def inject_fault():
                    await anyio.sleep(0.5)  # Wait 500ms
                    browser_index = random.randint(0, len(TEST_BROWSERS) - 1)
                    browser = TEST_BROWSERS[browser_index]
                    logger.info(f"Injecting fault in browser: {browser}")
                    
                    # Simulate browser crash via model manager
                    if hasattr(self.integration, "_simulate_browser_crash"):
                        await self.integration._simulate_browser_crash(browser_index)
                    
                    # Alternative: use the plugin if available
                    if self.resource_pool_plugin:
                        await self.resource_pool_plugin.inject_fault({
                            "type": "browser_crash",
                            "browser": browser
                        })
                
                # Start fault injection task
                fault_task = anyio.create_task_group()
                await fault_task.__aenter__()
                fault_task.start_soon(inject_fault)
            
            # Execute models concurrently
            start_time = time.time()
            concurrent_results = self.integration.execute_concurrent(model_inputs)
            execution_time = time.time() - start_time
            
            # Wait for fault injection to complete if active
            if self.args.enable_fault_tolerance and self.args.inject_fault:
                await fault_task.__aexit__(None, None, None)
            
            # Process results
            execution_results = []
            success_count = 0
            
            for i, res in enumerate(concurrent_results):
                success = res.get("success", False) or res.get("status") == "success"
                if success:
                    success_count += 1
                
                execution_results.append({
                    "model_name": model_configs[i]["model_name"] if i < len(model_configs) else "unknown",
                    "success": success,
                    "browser": res.get("browser", "unknown"),
                    "is_real_implementation": res.get("is_real_implementation", False),
                    "recovery": res.get("recovery", {}) if self.args.enable_fault_tolerance else {}
                })
            
            # Store results
            results = {
                "concurrent_execution": True,
                "models_tested": len(model_configs),
                "successful_models": success_count,
                "execution_time": execution_time,
                "fault_injected": self.args.enable_fault_tolerance and self.args.inject_fault,
                "model_results": execution_results,
                "fault_tolerance_enabled": self.args.enable_fault_tolerance,
                "fault_tolerance_level": self.fault_tolerance_level if self.args.enable_fault_tolerance else "none"
            }
            
            logger.info(f"Concurrent execution complete: {success_count}/{len(model_configs)} models successful")
            
        except Exception as e:
            logger.error(f"Error during concurrent execution test: {e}")
            import traceback
            traceback.print_exc()
            results = {
                "concurrent_execution": False,
                "error": str(e)
            }
        
        self.results["concurrent_execution"] = results
        return results
    
    def _create_test_input(self, model_type: str) -> Dict[str, Any]:
        """Create appropriate test input for model type"""
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
            return {'inputs': "This is a test input for the model."}
    
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"web_resource_pool_integration_test_{timestamp}.json"
        
        # Add summary information
        summary = {
            "timestamp": timestamp,
            "fault_tolerance_enabled": self.args.enable_fault_tolerance,
            "fault_tolerance_level": self.fault_tolerance_level if self.args.enable_fault_tolerance else "none",
            "recovery_strategy": self.recovery_strategy if self.args.enable_fault_tolerance else "none",
            "distributed_testing_available": DISTRIBUTED_TESTING_AVAILABLE,
            "tests_run": list(self.results.keys())
        }
        
        # Combine results
        full_results = {
            "summary": summary,
            "tests": self.results
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")

async def main_async():
    """Main async function"""
    parser = argparse.ArgumentParser(description="Test WebGPU/WebNN Resource Pool Integration with Fault Tolerance")
    
    # Test selection options
    parser.add_argument("--test-standard", action="store_true",
        help="Run standard integration test")
    parser.add_argument("--test-sharding", action="store_true",
        help="Test fault-tolerant model sharding")
    parser.add_argument("--test-recovery", action="store_true",
        help="Test fault tolerance recovery capabilities")
    parser.add_argument("--test-distributed", action="store_true",
        help="Test integration with distributed testing framework")
    parser.add_argument("--test-concurrent", action="store_true",
        help="Test concurrent model execution with fault tolerance")
    parser.add_argument("--comprehensive", action="store_true",
        help="Run all tests")
    
    # Fault tolerance options
    parser.add_argument("--enable-fault-tolerance", action="store_true",
        help="Enable fault tolerance features")
    parser.add_argument("--fault-tolerance-level", type=str, 
        choices=["low", "medium", "high", "critical"], default="medium",
        help="Fault tolerance level")
    parser.add_argument("--recovery-strategy", type=str,
        choices=["simple", "progressive", "parallel", "coordinated"], default="progressive",
        help="Recovery strategy")
    parser.add_argument("--inject-fault", action="store_true",
        help="Inject fault during execution")
    
    # Browser selection options
    parser.add_argument("--chrome", action="store_true",
        help="Use Chrome for all tests")
    parser.add_argument("--firefox", action="store_true",
        help="Use Firefox for all tests")
    parser.add_argument("--edge", action="store_true",
        help="Use Edge for all tests")
    
    # Connection options
    parser.add_argument("--max-connections", type=int, default=4,
        help="Maximum number of browser connections")
    parser.add_argument("--visible", action="store_true",
        help="Run browsers in visible mode (not headless)")
    
    # Performance options
    parser.add_argument("--enable-optimizations", action="store_true",
        help="Enable browser optimizations (compute shaders, shader precompilation, parallel loading)")
    parser.add_argument("--disable-ipfs", action="store_true",
        help="Disable IPFS acceleration (enabled by default)")
    
    # Database options
    parser.add_argument("--db-path", type=str,
        help="Path to DuckDB database for storing test results")
    
    # Logging options
    parser.add_argument("--verbose", action="store_true",
        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # If comprehensive flag is set, enable all tests
    if args.comprehensive:
        args.test_standard = True
        args.test_sharding = True
        args.test_recovery = True
        args.test_distributed = True
        args.test_concurrent = True
        args.enable_fault_tolerance = True
        args.enable_optimizations = True
    
    # Default to standard test if no test specified
    if not any([args.test_standard, args.test_sharding, args.test_recovery, 
               args.test_distributed, args.test_concurrent]):
        args.test_standard = True
    
    # Create tester
    tester = WebResourcePoolIntegrationTester(args)
    
    try:
        # Initialize
        if not await tester.initialize():
            logger.error("Failed to initialize tester")
            return 1
        
        # Run selected tests
        if args.test_standard:
            logger.info("=== Running Standard Integration Test ===")
            await tester.test_standard_integration()
        
        if args.test_sharding and args.enable_fault_tolerance:
            logger.info("=== Running Fault-Tolerant Model Sharding Test ===")
            await tester.test_fault_tolerant_model_sharding()
        
        if args.test_recovery and args.enable_fault_tolerance:
            logger.info("=== Running Fault Tolerance Recovery Test ===")
            await tester.test_fault_tolerance_recovery()
        
        if args.test_distributed and args.enable_fault_tolerance:
            logger.info("=== Running Distributed Integration Test ===")
            await tester.test_distributed_integration()
        
        if args.test_concurrent:
            logger.info("=== Running Concurrent Execution Test ===")
            await tester.test_concurrent_execution()
        
        # Save results
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
        return anyio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())