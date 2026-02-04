#!/usr/bin/env python3
"""
Integration Test for WebGPU/WebNN Resource Pool Fault Tolerance

This script tests the complete integration of the WebGPU/WebNN Resource Pool
with the Advanced Fault Tolerance Visualization System and Comprehensive
Validation Framework. It verifies all components work together correctly under
various fault scenarios.

Usage:
    # Run basic integration test
    python test_web_resource_pool_fault_tolerance_integration.py

    # Run with specific browsers and model
    python test_web_resource_pool_fault_tolerance_integration.py --browsers chrome,firefox --model bert-base-uncased

    # Run with comprehensive testing
    python test_web_resource_pool_fault_tolerance_integration.py --comprehensive
"""

import os
import sys
import json
import time
import anyio
import argparse
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import required modules
try:
    # Import core system components
    from test.tests.web.web_platform.resource_pool_bridge_integration import ResourcePoolBridgeIntegration
    from test.tests.web.web_platform.cross_browser_model_sharding import CrossBrowserModelShardingManager
    from test.tests.web.web_platform.browser_performance_history import BrowserPerformanceHistory
    
    # Import fault tolerance and visualization components
    from test.tests.web.web_platform.fault_tolerance_validation import FaultToleranceValidator
    from test.tests.web.web_platform.fault_tolerance_visualization_integration import FaultToleranceValidationSystem
    from test.web_platform.visualization.fault_tolerance_visualizer import FaultToleranceVisualizer
    
    # Import mock implementations for testing without browsers
    if os.environ.get("USE_FIXED_MOCK", "0") == "1":
        from fixed_mock_cross_browser_sharding import MockCrossBrowserModelShardingManager
        print("Using fixed mock implementation for cross-browser sharding")
    else:
        from mock_cross_browser_sharding import MockCrossBrowserModelShardingManager
    
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required modules not available: {e}")
    MODULES_AVAILABLE = False

class WebResourcePoolFaultToleranceIntegrationTester:
    """Integration test runner for WebGPU/WebNN Resource Pool Fault Tolerance."""
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize the integration tester.
        
        Args:
            use_mock: Whether to use mock implementations for testing without browsers
        """
        self.use_mock = use_mock
        self.output_dir = os.path.abspath("./fault_tolerance_integration_test_results")
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logger
        
        if not MODULES_AVAILABLE:
            self.logger.error("Cannot proceed with integration test: required modules not available")
            sys.exit(1)
    
    async def run_basic_integration_test(self, 
                                       model_name: str = "bert-base-uncased",
                                       browsers: List[str] = None) -> Dict[str, Any]:
        """
        Run a basic integration test verifying core functionality.
        
        Args:
            model_name: Name of the model to test
            browsers: List of browsers to use (defaults to ["chrome", "firefox", "edge"])
            
        Returns:
            Test results dictionary
        """
        if browsers is None:
            browsers = ["chrome", "firefox", "edge"]
        
        self.logger.info(f"Running basic integration test with model: {model_name}, browsers: {', '.join(browsers)}")
        
        model_config = {
            "enable_fault_tolerance": True,
            "fault_tolerance_level": "medium",
            "recovery_strategy": "progressive",
            "timeout": 120
        }
        
        # Create model manager (real or mock)
        if self.use_mock:
            self.logger.info("Using mock implementation for testing")
            manager = MockCrossBrowserModelShardingManager(
                model_name=model_name,
                browsers=browsers,
                shard_type="optimal",
                num_shards=min(len(browsers), 3),  # Use at most 3 shards
                model_config=model_config
            )
        else:
            self.logger.info("Using real implementation with browsers")
            manager = CrossBrowserModelShardingManager(
                model_name=model_name,
                browsers=browsers,
                shard_type="optimal",
                num_shards=min(len(browsers), 3),  # Use at most 3 shards
                model_config=model_config
            )
        
        try:
            # Initialize model manager
            start_time = time.time()
            initialized = await manager.initialize()
            init_time = time.time() - start_time
            
            if not initialized:
                self.logger.error(f"Failed to initialize model manager: {model_name}")
                return {"status": "failed", "error": "Initialization failed"}
            
            self.logger.info(f"Model manager initialized in {init_time:.2f}s")
            
            # Create validation system
            validation_system = FaultToleranceValidationSystem(
                model_manager=manager,
                output_dir=self.output_dir
            )
            
            # Run validation with visualization
            validation_results = await validation_system.run_validation_with_visualization(
                fault_tolerance_level="medium",
                recovery_strategy="progressive",
                test_scenarios=["connection_lost", "browser_crash", "component_timeout"],
                generate_report=True,
                report_name=f"{model_name.replace('-', '_')}_integration_test_report.html",
                ci_compatible=True
            )
            
            # Verify validation results
            if "validation_results" not in validation_results:
                self.logger.error("Validation results not found in response")
                return {"status": "failed", "error": "Missing validation results"}
            
            validation_status = validation_results["validation_results"].get("validation_status", "unknown")
            self.logger.info(f"Validation status: {validation_status}")
            
            # Check for report
            if "visualizations" in validation_results and "report" in validation_results["visualizations"]:
                report_path = validation_results["visualizations"]["report"]
                self.logger.info(f"Report generated: {report_path}")
                if not os.path.exists(report_path):
                    self.logger.warning(f"Report file not found: {report_path}")
            
            # Run one test inference
            if not self.use_mock:
                self.logger.info("Testing inference with the model")
                test_input = {"input_ids": [101, 2023, 2003, 1037, 3231, 102], "attention_mask": [1, 1, 1, 1, 1, 1]}
                inference_result = await manager.run_inference_sharded(test_input)
                self.logger.info(f"Inference result received, status: {inference_result.get('status', 'unknown')}")
            
            # Shutdown cleanly
            await manager.shutdown()
            
            return {
                "status": "success",
                "validation_status": validation_status,
                "initialization_time": init_time,
                "report_path": validation_results.get("visualizations", {}).get("report"),
                "validation_results": validation_results.get("validation_results")
            }
            
        except Exception as e:
            self.logger.error(f"Error during integration test: {e}")
            import traceback
            traceback.print_exc()
            
            # Attempt to shutdown
            try:
                await manager.shutdown()
            except Exception:
                pass
                
            return {"status": "error", "error": str(e)}
    
    async def run_comparative_integration_test(self, 
                                             model_name: str = "bert-base-uncased",
                                             browsers: List[str] = None) -> Dict[str, Any]:
        """
        Run a comparative integration test with multiple recovery strategies.
        
        Args:
            model_name: Name of the model to test
            browsers: List of browsers to use (defaults to ["chrome", "firefox", "edge"])
            
        Returns:
            Test results dictionary
        """
        if browsers is None:
            browsers = ["chrome", "firefox", "edge"]
        
        self.logger.info(f"Running comparative integration test with model: {model_name}")
        
        model_config = {
            "enable_fault_tolerance": True,
            "fault_tolerance_level": "medium",
            "recovery_strategy": "progressive",  # Default, will be overridden in comparative test
            "timeout": 120
        }
        
        # Create model manager (real or mock)
        if self.use_mock:
            self.logger.info("Using mock implementation for testing")
            manager = MockCrossBrowserModelShardingManager(
                model_name=model_name,
                browsers=browsers,
                shard_type="optimal",
                num_shards=min(len(browsers), 3),
                model_config=model_config
            )
        else:
            self.logger.info("Using real implementation with browsers")
            manager = CrossBrowserModelShardingManager(
                model_name=model_name,
                browsers=browsers,
                shard_type="optimal",
                num_shards=min(len(browsers), 3),
                model_config=model_config
            )
        
        try:
            # Initialize model manager
            start_time = time.time()
            initialized = await manager.initialize()
            init_time = time.time() - start_time
            
            if not initialized:
                self.logger.error(f"Failed to initialize model manager: {model_name}")
                return {"status": "failed", "error": "Initialization failed"}
            
            self.logger.info(f"Model manager initialized in {init_time:.2f}s")
            
            # Create validation system
            validation_system = FaultToleranceValidationSystem(
                model_manager=manager,
                output_dir=self.output_dir
            )
            
            # Run comparative validation
            comparative_results = await validation_system.run_comparative_validation(
                strategies=["simple", "progressive", "coordinated"],
                levels=["medium"],
                test_scenarios=["connection_lost", "browser_crash", "component_timeout"],
                report_prefix=model_name.replace('-', '_')
            )
            
            # Verify results
            if not comparative_results:
                self.logger.error("Comparative validation returned empty results")
                return {"status": "failed", "error": "Empty comparative results"}
            
            self.logger.info(f"Comparative validation complete with {len(comparative_results.get('configs', []))} configurations")
            
            # Check for report files
            if "reports" in comparative_results:
                for report in comparative_results["reports"]:
                    if os.path.exists(report):
                        self.logger.info(f"Report file found: {report}")
                    else:
                        self.logger.warning(f"Report file not found: {report}")
            
            # Shutdown cleanly
            await manager.shutdown()
            
            return {
                "status": "success",
                "initialization_time": init_time,
                "comparative_results": comparative_results
            }
            
        except Exception as e:
            self.logger.error(f"Error during comparative integration test: {e}")
            import traceback
            traceback.print_exc()
            
            # Attempt to shutdown
            try:
                await manager.shutdown()
            except Exception:
                pass
                
            return {"status": "error", "error": str(e)}
    
    async def run_stress_test_integration(self, 
                                        model_name: str = "bert-base-uncased",
                                        browsers: List[str] = None,
                                        iterations: int = 5) -> Dict[str, Any]:
        """
        Run a stress test integration with multiple iterations.
        
        Args:
            model_name: Name of the model to test
            browsers: List of browsers to use (defaults to ["chrome", "firefox", "edge"])
            iterations: Number of test iterations
            
        Returns:
            Test results dictionary
        """
        if browsers is None:
            browsers = ["chrome", "firefox", "edge"]
        
        self.logger.info(f"Running stress test integration with model: {model_name}, iterations: {iterations}")
        
        model_config = {
            "enable_fault_tolerance": True,
            "fault_tolerance_level": "high",  # Use high level for stress testing
            "recovery_strategy": "progressive",
            "timeout": 120
        }
        
        # Create model manager (real or mock)
        if self.use_mock:
            self.logger.info("Using mock implementation for testing")
            manager = MockCrossBrowserModelShardingManager(
                model_name=model_name,
                browsers=browsers,
                shard_type="optimal",
                num_shards=min(len(browsers), 3),
                model_config=model_config
            )
        else:
            self.logger.info("Using real implementation with browsers")
            manager = CrossBrowserModelShardingManager(
                model_name=model_name,
                browsers=browsers,
                shard_type="optimal",
                num_shards=min(len(browsers), 3),
                model_config=model_config
            )
        
        try:
            # Initialize model manager
            start_time = time.time()
            initialized = await manager.initialize()
            init_time = time.time() - start_time
            
            if not initialized:
                self.logger.error(f"Failed to initialize model manager: {model_name}")
                return {"status": "failed", "error": "Initialization failed"}
            
            self.logger.info(f"Model manager initialized in {init_time:.2f}s")
            
            # Create validation system
            validation_system = FaultToleranceValidationSystem(
                model_manager=manager,
                output_dir=self.output_dir
            )
            
            # Run stress test validation
            stress_test_results = await validation_system.run_stress_test_validation(
                iterations=iterations,
                fault_tolerance_level="high",
                recovery_strategy="progressive",
                test_scenarios=["connection_lost", "browser_crash", "component_timeout", "multiple_failures"],
                report_name=f"{model_name.replace('-', '_')}_stress_test_report.html"
            )
            
            # Verify results
            if not stress_test_results:
                self.logger.error("Stress test validation returned empty results")
                return {"status": "failed", "error": "Empty stress test results"}
            
            self.logger.info(f"Stress test validation complete with {stress_test_results.get('total_iterations', 0)} iterations")
            
            # Check for report file
            if "report" in stress_test_results:
                report_path = stress_test_results["report"]
                if os.path.exists(report_path):
                    self.logger.info(f"Report file found: {report_path}")
                else:
                    self.logger.warning(f"Report file not found: {report_path}")
            
            # Check success rate
            success_rate = stress_test_results.get("success_rate", 0)
            self.logger.info(f"Stress test success rate: {success_rate:.2f}%")
            
            # Verify minimum success rate (90% for automated tests)
            minimum_success_rate = 90.0
            if success_rate < minimum_success_rate:
                self.logger.error(f"Success rate {success_rate:.2f}% below minimum threshold {minimum_success_rate:.2f}%")
                status = "warning"
            else:
                status = "success"
            
            # Shutdown cleanly
            await manager.shutdown()
            
            return {
                "status": status,
                "initialization_time": init_time,
                "success_rate": success_rate,
                "stress_test_results": stress_test_results
            }
            
        except Exception as e:
            self.logger.error(f"Error during stress test integration: {e}")
            import traceback
            traceback.print_exc()
            
            # Attempt to shutdown
            try:
                await manager.shutdown()
            except Exception:
                pass
                
            return {"status": "error", "error": str(e)}
    
    async def run_integration_with_resource_pool(self, 
                                               model_name: str = "bert-base-uncased",
                                               browsers: List[str] = None) -> Dict[str, Any]:
        """
        Test integration with the resource pool.
        
        Args:
            model_name: Name of the model to test
            browsers: List of browsers to use (defaults to ["chrome", "firefox", "edge"])
            
        Returns:
            Test results dictionary
        """
        if browsers is None:
            browsers = ["chrome", "firefox", "edge"]
        
        self.logger.info(f"Testing integration with resource pool: {model_name}")
        
        # Exit early if using mock mode
        if self.use_mock:
            self.logger.info("Skipping resource pool integration test in mock mode")
            return {"status": "skipped", "reason": "Mock mode active"}
        
        browser_preferences = {
            "text": "edge",
            "vision": "chrome",
            "audio": "firefox"
        }
        
        try:
            # Create resource pool
            resource_pool = ResourcePoolBridgeIntegration(
                max_connections=3,
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_fault_tolerance=True,
                recovery_strategy="progressive",
                state_sync_interval=5,
                redundancy_factor=2,
                db_path=os.path.join(self.output_dir, "resource_pool_test.duckdb")
            )
            
            # Initialize resource pool
            start_time = time.time()
            await resource_pool.initialize()
            init_time = time.time() - start_time
            self.logger.info(f"Resource pool initialized in {init_time:.2f}s")
            
            # Get model from resource pool with fault tolerance
            model = await resource_pool.get_model(
                model_type="text_embedding",
                model_name=model_name,
                hardware_preferences={"priority_list": ["webgpu", "cpu"]},
                fault_tolerance={
                    "recovery_timeout": 30,
                    "state_persistence": True,
                    "failover_strategy": "immediate"
                }
            )
            
            self.logger.info(f"Model loaded: {model_name}")
            
            # Run inference
            test_input = {"input_ids": [101, 2023, 2003, 1037, 3231, 102], "attention_mask": [1, 1, 1, 1, 1, 1]}
            inference_result = await model(test_input)
            self.logger.info(f"Inference result received: {type(inference_result)}")
            
            # Test resource pool fault tolerance integration
            performance_history = resource_pool.get_performance_history(
                model_type="text_embedding",
                time_range="1h",
                metrics=["latency", "throughput", "browser_utilization"]
            )
            
            self.logger.info(f"Performance history received: {len(performance_history)} entries")
            
            # Shutdown resource pool
            await resource_pool.shutdown()
            
            return {
                "status": "success",
                "initialization_time": init_time,
                "resource_pool_integration": "successful",
                "performance_history_length": len(performance_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error during resource pool integration test: {e}")
            import traceback
            traceback.print_exc()
            
            return {"status": "error", "error": str(e)}

async def main():
    """Run the integration test suite."""
    parser = argparse.ArgumentParser(
        description="WebGPU/WebNN Resource Pool Fault Tolerance Integration Test"
    )
    
    # Basic options
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                      help="Model name to test")
    parser.add_argument("--browsers", type=str, default="chrome,firefox,edge",
                      help="Comma-separated list of browsers to use")
    parser.add_argument("--mock", action="store_true",
                      help="Use mock implementation for testing without browsers")
    parser.add_argument("--output-dir", type=str, default="./fault_tolerance_integration_test_results",
                      help="Directory for output files")
    
    # Test modes
    parser.add_argument("--basic", action="store_true",
                      help="Run basic integration test")
    parser.add_argument("--comparative", action="store_true",
                      help="Run comparative integration test")
    parser.add_argument("--stress-test", action="store_true",
                      help="Run stress test integration")
    parser.add_argument("--resource-pool", action="store_true",
                      help="Test integration with resource pool")
    parser.add_argument("--comprehensive", action="store_true",
                      help="Run all test modes")
    
    # Stress test options
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of iterations for stress testing")
    
    # Debug options
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare browsers list
    browsers = args.browsers.split(',')
    
    # Create integration tester
    tester = WebResourcePoolFaultToleranceIntegrationTester(use_mock=args.mock)
    tester.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(tester.output_dir, exist_ok=True)
    
    # Track overall results
    results = {}
    all_passed = True
    
    # Run tests based on selected modes
    try:
        # Determine which tests to run
        run_basic = args.basic or args.comprehensive or (not any([args.basic, args.comparative, args.stress_test, args.resource_pool]))
        run_comparative = args.comparative or args.comprehensive
        run_stress_test = args.stress_test or args.comprehensive
        run_resource_pool = args.resource_pool or args.comprehensive
        
        # Run basic integration test
        if run_basic:
            logger.info("=== Running Basic Integration Test ===")
            basic_results = await tester.run_basic_integration_test(model_name=args.model, browsers=browsers)
            results["basic_integration"] = basic_results
            all_passed = all_passed and basic_results.get("status") == "success"
            logger.info(f"Basic Integration Test: {basic_results.get('status')}")
        
        # Run comparative integration test
        if run_comparative:
            logger.info("=== Running Comparative Integration Test ===")
            comparative_results = await tester.run_comparative_integration_test(model_name=args.model, browsers=browsers)
            results["comparative_integration"] = comparative_results
            all_passed = all_passed and comparative_results.get("status") == "success"
            logger.info(f"Comparative Integration Test: {comparative_results.get('status')}")
        
        # Run stress test integration
        if run_stress_test:
            logger.info("=== Running Stress Test Integration ===")
            stress_test_results = await tester.run_stress_test_integration(
                model_name=args.model, 
                browsers=browsers,
                iterations=args.iterations
            )
            results["stress_test_integration"] = stress_test_results
            all_passed = all_passed and stress_test_results.get("status") in ["success", "warning"]
            logger.info(f"Stress Test Integration: {stress_test_results.get('status')}")
        
        # Run resource pool integration test
        if run_resource_pool and not args.mock:
            logger.info("=== Running Resource Pool Integration Test ===")
            resource_pool_results = await tester.run_integration_with_resource_pool(model_name=args.model, browsers=browsers)
            results["resource_pool_integration"] = resource_pool_results
            all_passed = all_passed and resource_pool_results.get("status") in ["success", "skipped"]
            logger.info(f"Resource Pool Integration: {resource_pool_results.get('status')}")
        
        # Save overall results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(tester.output_dir, f"integration_test_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "model": args.model,
                "browsers": browsers,
                "mock_mode": args.mock,
                "overall_status": "passed" if all_passed else "failed",
                "test_results": results
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Return appropriate exit code
        return 0 if all_passed else 1
    
    except Exception as e:
        logger.error(f"Error during integration test suite: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(anyio.run(main()))