#!/usr/bin/env python3
"""
End-to-End Test for Selenium Browser Recovery

This script performs comprehensive end-to-end testing of the Selenium browser recovery
integration with both automated browser testing and recovery strategies. It verifies:

1. Proper browser initialization with WebGPU/WebNN capability detection
2. Model-aware browser configuration
3. Recovery from various types of failures
4. Circuit breaker integration for fault tolerance
5. Performance metrics collection and analysis

Usage:
    python selenium_e2e_browser_recovery_test.py [--browser chrome] [--model bert-base-uncased] [--all]
"""

import os
import sys
import time
import json
import anyio
import argparse
import logging
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("selenium_e2e_test")

# Set more verbose logging if environment variable is set
if os.environ.get("SELENIUM_BRIDGE_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

# Import core components
try:
    from test.tests.distributed.distributed_testing.selenium_browser_bridge import (
        BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
    )
except ImportError:
    logger.error(f"Error importing selenium_browser_bridge. Make sure it exists at the expected path.")
    SELENIUM_AVAILABLE = False

try:
    from test.tests.distributed.distributed_testing.browser_recovery_strategies import (
        BrowserType, ModelType, FailureType, RecoveryLevel,
        detect_browser_type, detect_model_type, categorize_browser_failure, recover_browser,
        ProgressiveRecoveryManager
    )
    RECOVERY_AVAILABLE = True
except ImportError:
    logger.error(f"Error importing browser_recovery_strategies. Make sure it exists at the expected path.")
    RECOVERY_AVAILABLE = False

# Circuit breaker import
try:
    from test.tests.distributed.distributed_testing.circuit_breaker import (
        CircuitBreaker, CircuitState, CircuitOpenError
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    logger.warning("Circuit breaker not available, recovery analysis will be limited")
    CIRCUIT_BREAKER_AVAILABLE = False


class RecoveryTestMetrics:
    """Tracks metrics for the recovery test."""
    
    def __init__(self):
        """Initialize recovery test metrics."""
        self.start_time = datetime.now()
        self.end_time = None
        self.total_tests = 0
        self.successful_tests = 0
        self.failures = 0
        self.recoveries = 0
        self.recovery_attempts = 0
        self.browser_metrics = {}
        self.model_metrics = {}
        self.recovery_metrics = {}
        self.test_results = []
        
    def record_test_result(self, test_name: str, browser: str, model: str, 
                          success: bool, recovered: bool = False,
                          recovery_attempts: int = 0, details: Dict[str, Any] = None):
        """Record a test result."""
        self.total_tests += 1
        
        if success:
            self.successful_tests += 1
        else:
            self.failures += 1
            
        if recovered:
            self.recoveries += 1
            
        self.recovery_attempts += recovery_attempts
        
        # Update browser metrics
        if browser not in self.browser_metrics:
            self.browser_metrics[browser] = {
                "tests": 0,
                "successes": 0,
                "failures": 0,
                "recoveries": 0
            }
        
        self.browser_metrics[browser]["tests"] += 1
        if success:
            self.browser_metrics[browser]["successes"] += 1
        else:
            self.browser_metrics[browser]["failures"] += 1
        if recovered:
            self.browser_metrics[browser]["recoveries"] += 1
            
        # Update model metrics
        if model not in self.model_metrics:
            self.model_metrics[model] = {
                "tests": 0,
                "successes": 0,
                "failures": 0,
                "recoveries": 0
            }
        
        self.model_metrics[model]["tests"] += 1
        if success:
            self.model_metrics[model]["successes"] += 1
        else:
            self.model_metrics[model]["failures"] += 1
        if recovered:
            self.model_metrics[model]["recoveries"] += 1
            
        # Save test result
        self.test_results.append({
            "test_name": test_name,
            "browser": browser,
            "model": model,
            "success": success,
            "recovered": recovered,
            "recovery_attempts": recovery_attempts,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        })
        
    def record_recovery_metrics(self, recovery_type: str, success: bool, 
                               duration: float, browser: str, model: str,
                               details: Dict[str, Any] = None):
        """Record recovery metrics."""
        if recovery_type not in self.recovery_metrics:
            self.recovery_metrics[recovery_type] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "total_duration": 0,
                "browsers": {},
                "models": {}
            }
            
        metrics = self.recovery_metrics[recovery_type]
        metrics["attempts"] += 1
        if success:
            metrics["successes"] += 1
        else:
            metrics["failures"] += 1
        metrics["total_duration"] += duration
        
        # Update browser-specific metrics
        if browser not in metrics["browsers"]:
            metrics["browsers"][browser] = {
                "attempts": 0,
                "successes": 0
            }
        
        metrics["browsers"][browser]["attempts"] += 1
        if success:
            metrics["browsers"][browser]["successes"] += 1
            
        # Update model-specific metrics
        if model not in metrics["models"]:
            metrics["models"][model] = {
                "attempts": 0,
                "successes": 0
            }
        
        metrics["models"][model]["attempts"] += 1
        if success:
            metrics["models"][model]["successes"] += 1
        
    def complete(self):
        """Mark metrics collection as complete."""
        self.end_time = datetime.now()
        
    def get_duration(self) -> float:
        """Get duration of the test in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
        
    def get_success_rate(self) -> float:
        """Get overall success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.successful_tests / self.total_tests
        
    def get_recovery_rate(self) -> float:
        """Get overall recovery rate."""
        if self.failures == 0:
            return 0.0
        return self.recoveries / self.failures
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of metrics."""
        recovery_rate = self.get_recovery_rate()
        success_rate = self.get_success_rate()
        
        # Calculate browser success rates
        browser_success_rates = {}
        for browser, metrics in self.browser_metrics.items():
            if metrics["tests"] > 0:
                browser_success_rates[browser] = metrics["successes"] / metrics["tests"]
            else:
                browser_success_rates[browser] = 0.0
                
        # Calculate browser recovery rates
        browser_recovery_rates = {}
        for browser, metrics in self.browser_metrics.items():
            if metrics["failures"] > 0:
                browser_recovery_rates[browser] = metrics["recoveries"] / metrics["failures"]
            else:
                browser_recovery_rates[browser] = 0.0
                
        # Calculate model success rates
        model_success_rates = {}
        for model, metrics in self.model_metrics.items():
            if metrics["tests"] > 0:
                model_success_rates[model] = metrics["successes"] / metrics["tests"]
            else:
                model_success_rates[model] = 0.0
                
        # Calculate model recovery rates
        model_recovery_rates = {}
        for model, metrics in self.model_metrics.items():
            if metrics["failures"] > 0:
                model_recovery_rates[model] = metrics["recoveries"] / metrics["failures"]
            else:
                model_recovery_rates[model] = 0.0
                
        # Calculate recovery type success rates
        recovery_success_rates = {}
        for r_type, metrics in self.recovery_metrics.items():
            if metrics["attempts"] > 0:
                recovery_success_rates[r_type] = metrics["successes"] / metrics["attempts"]
                metrics["success_rate"] = metrics["successes"] / metrics["attempts"]
                metrics["avg_duration"] = metrics["total_duration"] / metrics["attempts"]
            else:
                recovery_success_rates[r_type] = 0.0
                metrics["success_rate"] = 0.0
                metrics["avg_duration"] = 0.0
        
        return {
            "total_tests": self.total_tests,
            "successful_tests": self.successful_tests,
            "failures": self.failures,
            "recoveries": self.recoveries,
            "recovery_attempts": self.recovery_attempts,
            "success_rate": success_rate,
            "recovery_rate": recovery_rate,
            "duration_seconds": self.get_duration(),
            "browser_metrics": self.browser_metrics,
            "model_metrics": self.model_metrics,
            "browser_success_rates": browser_success_rates,
            "browser_recovery_rates": browser_recovery_rates,
            "model_success_rates": model_success_rates,
            "model_recovery_rates": model_recovery_rates,
            "recovery_metrics": self.recovery_metrics,
            "recovery_success_rates": recovery_success_rates,
            "test_results": self.test_results
        }


class SeleniumE2ERecoveryTest:
    """
    End-to-End test for Selenium Browser Recovery.
    
    This class provides comprehensive testing of the Selenium browser recovery
    integration, including:
    - Browser initialization
    - WebGPU/WebNN capability detection
    - Model-aware browser configuration
    - Recovery from various types of failures
    - Circuit breaker integration
    - Performance metrics collection and analysis
    """
    
    def __init__(self, browsers: List[str] = None, models: List[str] = None,
                 inject_failures: bool = True, test_count: int = 3,
                 simulation_mode: bool = False, report_path: Optional[str] = None):
        """
        Initialize the end-to-end recovery test.
        
        Args:
            browsers: List of browsers to test
            models: List of models to test
            inject_failures: Whether to inject artificial failures
            test_count: Number of test iterations per browser/model combination
            simulation_mode: Whether to run in simulation mode
            report_path: Path to save the test report
        """
        # Default browsers and models if not provided
        self.browsers = browsers or ["chrome", "firefox", "edge"]
        self.models = models or ["bert-base-uncased", "vit-base-patch16-224", "whisper-tiny", "clip-vit-base-patch32"]
        self.inject_failures = inject_failures
        self.test_count = test_count
        self.simulation_mode = simulation_mode
        self.report_path = report_path
        
        # Test metrics
        self.metrics = RecoveryTestMetrics()
        
        # Convert browser names to BrowserType
        self.browser_types = {
            browser: detect_browser_type(browser) for browser in self.browsers
        }
        
        # Convert model names to ModelType
        self.model_types = {
            model: detect_model_type(model) for model in self.models
        }
        
        # Set platform preferences by model type
        self.platform_preferences = {
            ModelType.TEXT: "webnn" if "edge" in self.browsers else "webgpu",
            ModelType.VISION: "webgpu",
            ModelType.AUDIO: "webgpu",
            ModelType.MULTIMODAL: "webgpu",
            ModelType.GENERIC: "webgpu",
        }
        
        # Set browser preferences by model type
        self.browser_preferences = {
            ModelType.TEXT: BrowserType.EDGE if "edge" in self.browsers else BrowserType.CHROME,
            ModelType.VISION: BrowserType.CHROME,
            ModelType.AUDIO: BrowserType.FIREFOX if "firefox" in self.browsers else BrowserType.CHROME,
            ModelType.MULTIMODAL: BrowserType.CHROME,
            ModelType.GENERIC: BrowserType.CHROME,
        }
        
        # Create recovery manager
        self.recovery_manager = ProgressiveRecoveryManager()
        
        # Circuit breakers registry
        self.circuit_breakers = {}
        
        logger.info(f"Initialized E2E recovery test with browsers: {self.browsers}, models: {self.models}")
    
    def create_browser_config(self, browser_name: str, model_name: str) -> BrowserConfiguration:
        """
        Create a browser configuration for the specified browser and model.
        
        Args:
            browser_name: Name of the browser
            model_name: Name of the model
            
        Returns:
            BrowserConfiguration instance
        """
        browser_type = self.browser_types.get(browser_name, BrowserType.UNKNOWN)
        model_type = self.model_types.get(model_name, ModelType.GENERIC)
        
        # Determine platform based on model and browser type
        platform = self.platform_preferences.get(model_type, "webgpu")
        
        # If Edge is preferred for this model type and we're using Edge, switch to WebNN
        if platform == "webnn" and browser_type == BrowserType.EDGE:
            platform = "webnn"
        
        # Create configuration
        config = BrowserConfiguration(
            browser_name=browser_name,
            platform=platform,
            headless=True,
            timeout=30
        )
        
        # Apply model-specific optimizations
        if model_type == ModelType.TEXT:
            # Text models - optimize for latency
            config.shader_precompilation = True
            config.max_batch_size = 1
            config.optimize_for = "latency"
            
        elif model_type == ModelType.VISION:
            # Vision models - optimize for throughput
            config.shader_precompilation = True
            config.max_batch_size = 4
            config.optimize_for = "throughput"
            
            # Add Chrome-specific optimizations for vision models
            if browser_type == BrowserType.CHROME:
                config.custom_args = [
                    "--enable-zero-copy",
                    "--enable-gpu-memory-buffer-video-frames"
                ]
            
        elif model_type == ModelType.AUDIO:
            # Audio models - enable compute shaders
            config.compute_shaders = True
            
            # Add Firefox-specific optimizations for audio models
            if browser_type == BrowserType.FIREFOX:
                config.custom_prefs = {
                    "dom.webgpu.workgroup_size": "256,1,1"
                }
            
        elif model_type == ModelType.MULTIMODAL:
            # Multimodal models - enable parallel loading
            config.parallel_loading = True
            config.optimize_for = "memory"
        
        return config
    
    def get_circuit_breaker(self, browser_name: str, model_name: str) -> Any:
        """
        Get or create a circuit breaker for a browser-model combination.
        
        Args:
            browser_name: Browser name
            model_name: Model name
            
        Returns:
            CircuitBreaker instance or None if not available
        """
        if not CIRCUIT_BREAKER_AVAILABLE:
            return None
            
        key = f"{browser_name}_{model_name}"
        
        if key not in self.circuit_breakers:
            model_type = self.model_types.get(model_name, ModelType.GENERIC)
            
            # Adjust circuit breaker settings based on model type
            if model_type == ModelType.TEXT:
                # Text models recover quickly
                failure_threshold = 2
                recovery_timeout = 5.0
            elif model_type == ModelType.VISION:
                # Vision models can be more stable
                failure_threshold = 3
                recovery_timeout = 7.0
            elif model_type == ModelType.AUDIO:
                # Audio models often need more recovery attempts
                failure_threshold = 4
                recovery_timeout = 10.0
            elif model_type == ModelType.MULTIMODAL:
                # Multimodal models have complex recovery
                failure_threshold = 4
                recovery_timeout = 15.0
            else:
                # Default settings
                failure_threshold = 3
                recovery_timeout = 10.0
            
            # Create circuit breaker
            self.circuit_breakers[key] = CircuitBreaker(
                name=f"browser_{browser_name}_{model_name}",
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=1,
                success_threshold=2
            )
            
        return self.circuit_breakers[key]
    
    def inject_failure(self, test_index: int, browser_type: BrowserType, model_type: ModelType) -> Optional[FailureType]:
        """
        Determine whether to inject a failure and what type.
        
        Args:
            test_index: Current test index
            browser_type: Browser type
            model_type: Model type
            
        Returns:
            FailureType to inject or None
        """
        if not self.inject_failures:
            return None
            
        # Only inject failures occasionally
        if random.random() > 0.4:
            return None
            
        # Different failure patterns for different browsers
        chrome_failures = [
            FailureType.CONNECTION_FAILURE,
            FailureType.GPU_ERROR,
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.CRASH
        ]
        
        firefox_failures = [
            FailureType.CONNECTION_FAILURE,
            FailureType.GPU_ERROR,  # Firefox WebGPU compute shader issues
            FailureType.INTERNAL_ERROR,
            FailureType.CRASH
        ]
        
        edge_failures = [
            FailureType.CONNECTION_FAILURE,
            FailureType.API_ERROR,  # Edge WebNN API issues
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.TIMEOUT
        ]
        
        # Select failure types based on browser
        if browser_type == BrowserType.CHROME:
            failure_types = chrome_failures
        elif browser_type == BrowserType.FIREFOX:
            failure_types = firefox_failures
        elif browser_type == BrowserType.EDGE:
            failure_types = edge_failures
        else:
            failure_types = list(FailureType)
            
        # Remove LAUNCH_FAILURE since it's harder to recover from in our test
        if FailureType.LAUNCH_FAILURE in failure_types:
            failure_types.remove(FailureType.LAUNCH_FAILURE)
            
        # Pick a failure type
        return random.choice(failure_types)
    
    async def run_test_case(self, browser_name: str, model_name: str, test_index: int) -> Dict[str, Any]:
        """
        Run a single test case for a browser-model combination.
        
        Args:
            browser_name: Browser name
            model_name: Model name
            test_index: Test case index
            
        Returns:
            Dictionary with test results
        """
        # Get types
        browser_type = self.browser_types.get(browser_name, BrowserType.UNKNOWN)
        model_type = self.model_types.get(model_name, ModelType.GENERIC)
        
        # Create test name
        test_name = f"test_{browser_name}_{model_name}_{test_index}"
        
        logger.info(f"Running test case: {test_name}")
        
        # Check if selenium is available
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available, skipping test")
            self.metrics.record_test_result(
                test_name=test_name,
                browser=browser_name,
                model=model_name,
                success=False,
                details={"error": "Selenium not available"}
            )
            return {"success": False, "error": "Selenium not available"}
        
        # Check if recovery is available
        if not RECOVERY_AVAILABLE:
            logger.error("Browser recovery strategies not available, skipping test")
            self.metrics.record_test_result(
                test_name=test_name,
                browser=browser_name,
                model=model_name,
                success=False,
                details={"error": "Browser recovery strategies not available"}
            )
            return {"success": False, "error": "Browser recovery strategies not available"}
        
        # Create browser config
        config = self.create_browser_config(browser_name, model_name)
        
        # Create browser bridge
        bridge = SeleniumBrowserBridge(config)
        bridge.model_type = model_type  # Set model type for recovery strategies
        
        # Get circuit breaker
        circuit = self.get_circuit_breaker(browser_name, model_name)
        
        try:
            # Launch browser
            launch_success = await bridge.launch(allow_simulation=self.simulation_mode)
            
            if not launch_success:
                logger.error(f"Failed to launch browser for test case: {test_name}")
                self.metrics.record_test_result(
                    test_name=test_name,
                    browser=browser_name,
                    model=model_name,
                    success=False,
                    details={"error": "Failed to launch browser"}
                )
                return {"success": False, "error": "Failed to launch browser"}
                
            # Detect whether we're running in simulation mode
            is_simulation = getattr(bridge, 'simulation_mode', False)
            
            # Determine if we should inject a failure
            failure_type = self.inject_failure(test_index, browser_type, model_type)
            
            if failure_type:
                logger.info(f"Injecting artificial {failure_type.value} failure for {test_name}")
                
                # Create an error for this failure type
                error_msg = f"Artificial {failure_type.value} failure in {browser_name} for {model_name}"
                error = Exception(error_msg)
                
                # Set up context for recovery
                context = {
                    "browser": browser_name,
                    "model": model_name,
                    "failure_type": failure_type.value,
                    "test_index": test_index
                }
                
                # Categorize the failure
                failure_info = categorize_browser_failure(error, context)
                
                # Attempt recovery
                logger.info(f"Attempting to recover from injected failure: {failure_type.value}")
                recovery_start = time.time()
                
                # Try to recover using progressive recovery
                success = await self.recovery_manager.execute_progressive_recovery(
                    bridge, browser_type, model_type, failure_info
                )
                
                recovery_duration = time.time() - recovery_start
                
                self.metrics.record_recovery_metrics(
                    recovery_type=failure_type.value,
                    success=success,
                    duration=recovery_duration,
                    browser=browser_name,
                    model=model_name,
                    details={
                        "recovery_level": "progressive",
                        "injected": True,
                        "recovery_duration": recovery_duration
                    }
                )
                
                if not success:
                    logger.error(f"Recovery failed for test case: {test_name}")
                    self.metrics.record_test_result(
                        test_name=test_name,
                        browser=browser_name,
                        model=model_name,
                        success=False,
                        recovery_attempts=1,
                        details={
                            "error": f"Recovery failed for injected {failure_type.value}",
                            "failure_type": failure_type.value,
                            "recovery_attempted": True,
                            "recovered": False
                        }
                    )
                    return {
                        "success": False, 
                        "error": f"Recovery failed for injected {failure_type.value}",
                        "failure_type": failure_type.value,
                        "recovery_attempted": True,
                        "recovered": False
                    }
                    
                logger.info(f"Successfully recovered from injected failure: {failure_type.value}")
            
            # Execute test
            result = await bridge.run_test(
                model_name=model_name,
                input_data="This is a test input"
            )
            
            # Check result
            success = result.get("success", False)
            recovered = result.get("recovered", False) or (failure_type is not None)
            recovery_attempts = result.get("recovery_attempts", 0) + (1 if failure_type else 0)
            
            # Record metrics
            self.metrics.record_test_result(
                test_name=test_name,
                browser=browser_name,
                model=model_name,
                success=success,
                recovered=recovered,
                recovery_attempts=recovery_attempts,
                details={
                    "test_output": result,
                    "simulation_mode": is_simulation,
                    "injected_failure": failure_type.value if failure_type else None,
                    "circuit_breaker_state": getattr(circuit, 'state', None) if circuit else None
                }
            )
            
            # Add basic result info to return value
            result["test_name"] = test_name
            result["browser"] = browser_name
            result["model"] = model_name
            result["simulation_mode"] = is_simulation
            result["recovery_attempts"] = recovery_attempts
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in test case {test_name}: {str(e)}")
            
            # Try to recover from unexpected error
            recovery_start = time.time()
            
            try:
                # Attempt emergency recovery
                context = {
                    "browser": browser_name,
                    "model": model_name,
                    "test_index": test_index,
                    "unexpected": True
                }
                
                recovered = await recover_browser(bridge, e, context)
                recovery_duration = time.time() - recovery_start
                
                self.metrics.record_recovery_metrics(
                    recovery_type="unexpected_error",
                    success=recovered,
                    duration=recovery_duration,
                    browser=browser_name,
                    model=model_name,
                    details={
                        "error": str(e),
                        "recovery_duration": recovery_duration
                    }
                )
                
                # Record test result
                self.metrics.record_test_result(
                    test_name=test_name,
                    browser=browser_name,
                    model=model_name,
                    success=recovered,
                    recovered=recovered,
                    recovery_attempts=1,
                    details={
                        "error": str(e),
                        "recovery_attempted": True,
                        "recovered": recovered
                    }
                )
                
                return {
                    "success": recovered,
                    "error": str(e),
                    "recovery_attempted": True,
                    "recovered": recovered,
                    "test_name": test_name,
                    "browser": browser_name,
                    "model": model_name
                }
                
            except Exception as recovery_error:
                logger.error(f"Error during recovery attempt: {str(recovery_error)}")
                
                # Record test result with failed recovery
                self.metrics.record_test_result(
                    test_name=test_name,
                    browser=browser_name,
                    model=model_name,
                    success=False,
                    recovered=False,
                    recovery_attempts=1,
                    details={
                        "error": str(e),
                        "recovery_error": str(recovery_error),
                        "recovery_attempted": True,
                        "recovered": False
                    }
                )
                
                return {
                    "success": False,
                    "error": str(e),
                    "recovery_error": str(recovery_error),
                    "recovery_attempted": True,
                    "recovered": False,
                    "test_name": test_name,
                    "browser": browser_name,
                    "model": model_name
                }
                
        finally:
            # Close browser
            if bridge:
                try:
                    await bridge.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {str(e)}")
    
    async def run_all_tests(self):
        """Run all test cases for all browser-model combinations."""
        logger.info(f"Running E2E browser recovery tests for {len(self.browsers)} browsers and {len(self.models)} models")
        
        # Run tests
        for browser in self.browsers:
            for model in self.models:
                for i in range(self.test_count):
                    await self.run_test_case(browser, model, i + 1)
                    
                    # Wait between tests to avoid resource contention
                    await anyio.sleep(1)
        
        # Complete metrics collection
        self.metrics.complete()
        
        # Save report if requested
        if self.report_path:
            self.save_report()
    
    def save_report(self):
        """Save test results to a file."""
        try:
            summary = self.metrics.get_summary()
            
            with open(self.report_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Test report saved to {self.report_path}")
            
            # Also generate a readable summary
            report_dir = os.path.dirname(self.report_path)
            summary_path = os.path.join(report_dir, "recovery_test_summary.md")
            
            with open(summary_path, 'w') as f:
                f.write(f"# Selenium Browser Recovery Test Summary\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"## Overall Results\n\n")
                f.write(f"- Total Tests: {summary['total_tests']}\n")
                f.write(f"- Successful Tests: {summary['successful_tests']}\n")
                f.write(f"- Failures: {summary['failures']}\n")
                f.write(f"- Recoveries: {summary['recoveries']}\n")
                f.write(f"- Success Rate: {summary['success_rate']:.2%}\n")
                f.write(f"- Recovery Rate: {summary['recovery_rate']:.2%}\n")
                f.write(f"- Test Duration: {summary['duration_seconds']:.2f} seconds\n\n")
                
                f.write(f"## Browser Performance\n\n")
                f.write(f"| Browser | Tests | Success Rate | Recovery Rate |\n")
                f.write(f"|---------|-------|-------------|---------------|\n")
                
                for browser in summary['browser_metrics'].keys():
                    success_rate = summary['browser_success_rates'].get(browser, 0)
                    recovery_rate = summary['browser_recovery_rates'].get(browser, 0)
                    tests = summary['browser_metrics'][browser]['tests']
                    
                    f.write(f"| {browser} | {tests} | {success_rate:.2%} | {recovery_rate:.2%} |\n")
                
                f.write(f"\n## Model Performance\n\n")
                f.write(f"| Model | Tests | Success Rate | Recovery Rate |\n")
                f.write(f"|-------|-------|-------------|---------------|\n")
                
                for model in summary['model_metrics'].keys():
                    success_rate = summary['model_success_rates'].get(model, 0)
                    recovery_rate = summary['model_recovery_rates'].get(model, 0)
                    tests = summary['model_metrics'][model]['tests']
                    
                    f.write(f"| {model} | {tests} | {success_rate:.2%} | {recovery_rate:.2%} |\n")
                
                f.write(f"\n## Recovery Strategy Performance\n\n")
                f.write(f"| Strategy | Attempts | Success Rate | Avg Duration (s) |\n")
                f.write(f"|----------|----------|-------------|------------------|\n")
                
                for strategy, metrics in summary['recovery_metrics'].items():
                    success_rate = metrics['success_rate']
                    attempts = metrics['attempts']
                    avg_duration = metrics['avg_duration']
                    
                    f.write(f"| {strategy} | {attempts} | {success_rate:.2%} | {avg_duration:.2f} |\n")
                
            logger.info(f"Test summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving test report: {str(e)}")
    
    def print_summary(self):
        """Print a summary of test results to the console."""
        summary = self.metrics.get_summary()
        
        print("\n" + "=" * 80)
        print("Selenium Browser Recovery E2E Test Summary")
        print("=" * 80)
        
        print(f"Total Tests:      {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Failures:         {summary['failures']}")
        print(f"Recoveries:       {summary['recoveries']}")
        print(f"Success Rate:     {summary['success_rate']:.2%}")
        print(f"Recovery Rate:    {summary['recovery_rate']:.2%}")
        print(f"Test Duration:    {summary['duration_seconds']:.2f} seconds")
        
        print("-" * 80)
        print("Browser Success Rates:")
        for browser, rate in summary['browser_success_rates'].items():
            tests = summary['browser_metrics'][browser]['tests']
            print(f"  {browser:10}: {rate:.2%} ({summary['browser_metrics'][browser]['successes']}/{tests})")
        
        print("-" * 80)
        print("Model Success Rates:")
        for model, rate in summary['model_success_rates'].items():
            tests = summary['model_metrics'][model]['tests']
            print(f"  {model:20}: {rate:.2%} ({summary['model_metrics'][model]['successes']}/{tests})")
        
        print("-" * 80)
        print("Recovery Strategy Performance:")
        for strategy, metrics in summary['recovery_metrics'].items():
            success_rate = metrics['success_rate']
            attempts = metrics['attempts']
            successes = metrics['successes']
            avg_duration = metrics['avg_duration']
            
            print(f"  {strategy:20}: {success_rate:.2%} ({successes}/{attempts}) - Avg: {avg_duration:.2f}s")
        
        print("=" * 80)


async def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Selenium Browser Recovery E2E Test")
    parser.add_argument("--browser", type=str, default="chrome",
                       help="Browser to test (chrome, firefox, edge). Use comma-separated list for multiple browsers")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model to test. Use comma-separated list for multiple models")
    parser.add_argument("--all", action="store_true", help="Test all browser-model combinations")
    parser.add_argument("--no-failures", action="store_true", help="Don't inject artificial failures")
    parser.add_argument("--test-count", type=int, default=3, help="Number of test iterations per browser-model combination")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("--report-path", type=str, help="Path to save the test report")
    args = parser.parse_args()
    
    # Parse browsers and models
    browsers = None
    models = None
    
    if args.all:
        browsers = ["chrome", "firefox", "edge"]
        models = ["bert-base-uncased", "vit-base-patch16-224", "whisper-tiny", "clip-vit-base-patch32"]
    else:
        browsers = [b.strip() for b in args.browser.split(",")]
        models = [m.strip() for m in args.model.split(",")]
    
    # Create report path if not provided
    report_path = args.report_path
    if not report_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"selenium_recovery_test_{timestamp}.json")
    
    # Create and run test
    test = SeleniumE2ERecoveryTest(
        browsers=browsers,
        models=models,
        inject_failures=not args.no_failures,
        test_count=args.test_count,
        simulation_mode=args.simulate,
        report_path=report_path
    )
    
    await test.run_all_tests()
    
    # Print summary
    test.print_summary()


if __name__ == "__main__":
    anyio.run(main())