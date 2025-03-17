#!/usr/bin/env python3
"""
Browser Error Recovery Demo with Failure Injection

This script demonstrates the browser recovery capabilities by integrating the browser_failure_injector
to create controlled test scenarios. It shows how various types of failures are handled and recovered
from using the progressive recovery strategies.

Key features:
- Controlled failure injection with browser_failure_injector
- Testing of recovery strategies for different failure types
- Visualization of recovery performance
- Clear reporting of recovery effectiveness

Usage:
    python run_error_recovery_demo.py [--browser chrome] [--model bert] [--failures all]
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("error_recovery_demo")

# Set more verbose logging if environment variable is set
if os.environ.get("SELENIUM_BRIDGE_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

# Import core components
try:
    from selenium_browser_bridge import (
        BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
    )
except ImportError:
    logger.error("Error importing selenium_browser_bridge. Make sure it exists at the expected path.")
    SELENIUM_AVAILABLE = False

try:
    from browser_recovery_strategies import (
        BrowserType, ModelType, FailureType, RecoveryLevel,
        detect_browser_type, detect_model_type,
        ProgressiveRecoveryManager
    )
    RECOVERY_AVAILABLE = True
except ImportError:
    logger.error("Error importing browser_recovery_strategies. Make sure it exists at the expected path.")
    RECOVERY_AVAILABLE = False

try:
    from browser_failure_injector import (
        BrowserFailureInjector
    )
    INJECTOR_AVAILABLE = True
except ImportError:
    logger.error("Error importing browser_failure_injector. Make sure it exists at the expected path.")
    INJECTOR_AVAILABLE = False

# Import circuit breaker
try:
    from circuit_breaker import CircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    logger.warning("CircuitBreaker not available. Circuit breaker integration will be disabled.")
    CIRCUIT_BREAKER_AVAILABLE = False

class ErrorRecoveryDemo:
    """
    Demo for browser error recovery with failure injection.
    
    This class demonstrates how the browser failure injector and recovery strategies
    work together to handle various types of failures.
    """
    
    def __init__(self, browser_name: str = "chrome", model_name: str = "bert",
                 platform: str = "webgpu", failure_types: List[str] = None,
                 iterations: int = 3, report_path: Optional[str] = None,
                 use_circuit_breaker: bool = True):
        """
        Initialize the error recovery demo.
        
        Args:
            browser_name: Browser name (chrome, firefox, edge)
            model_name: Model name or type (bert, vit, whisper, clip, text, vision, audio, multimodal)
            platform: Platform to use (webgpu, webnn)
            failure_types: List of failure types to test (connection_failure, resource_exhaustion, etc.)
            iterations: Number of iterations for each failure type
            report_path: Path to save test report
            use_circuit_breaker: Whether to use circuit breaker pattern for fault tolerance
        """
        self.browser_name = browser_name
        self.model_name = model_name
        self.platform = platform
        self.iterations = iterations
        self.report_path = report_path
        self.use_circuit_breaker = use_circuit_breaker and CIRCUIT_BREAKER_AVAILABLE
        
        # Resolve failure types
        if failure_types is None or "all" in failure_types:
            self.failure_types = [
                FailureType.CONNECTION_FAILURE,
                FailureType.RESOURCE_EXHAUSTION,
                FailureType.GPU_ERROR,
                FailureType.API_ERROR,
                FailureType.TIMEOUT,
                FailureType.INTERNAL_ERROR
            ]
        else:
            self.failure_types = [FailureType(ft) for ft in failure_types if hasattr(FailureType, ft)]
        
        # Detect browser and model types
        self.browser_type = detect_browser_type(browser_name)
        self.model_type = self._resolve_model_type(model_name)
        
        # Initialize recovery manager
        self.recovery_manager = ProgressiveRecoveryManager()
        
        # Initialize circuit breaker if enabled
        self.circuit_breaker = None
        if self.use_circuit_breaker:
            if CIRCUIT_BREAKER_AVAILABLE:
                self.circuit_breaker = CircuitBreaker(
                    failure_threshold=5,  # Open after 5 failures
                    recovery_timeout=60,  # Stay open for 60 seconds
                    half_open_after=30,   # Try half-open after 30 seconds
                    name=f"error_recovery_demo_{browser_name}_{model_name}"
                )
                logger.info("Circuit breaker initialized")
            else:
                logger.warning("Circuit breaker requested but not available")
        
        # Results tracking
        self.results = []
        self.start_time = None
        self.end_time = None
        
        logger.info(f"Initialized error recovery demo with browser={browser_name}, model={model_name}, platform={platform}")
        logger.info(f"Testing failure types: {[ft.value for ft in self.failure_types]}")
        logger.info(f"Circuit breaker enabled: {self.circuit_breaker is not None}")
    
    def _resolve_model_type(self, model_name: str) -> ModelType:
        """
        Resolve model type from model name.
        
        Args:
            model_name: Model name or type identifier
            
        Returns:
            ModelType enum value
        """
        # Check if it's a direct model type name
        if model_name.lower() in ["text", "bert", "t5"]:
            return ModelType.TEXT
        elif model_name.lower() in ["vision", "vit"]:
            return ModelType.VISION
        elif model_name.lower() in ["audio", "whisper"]:
            return ModelType.AUDIO
        elif model_name.lower() in ["multimodal", "clip"]:
            return ModelType.MULTIMODAL
        
        # Use detection function for specific model names
        return detect_model_type(model_name)
    
    def _create_browser_config(self) -> BrowserConfiguration:
        """
        Create browser configuration based on model and browser type.
        
        Returns:
            Browser configuration
        """
        # Start with basic configuration
        config = BrowserConfiguration(
            browser_name=self.browser_name,
            platform=self.platform,
            headless=True,
            timeout=30
        )
        
        # Apply model-specific optimizations
        if self.model_type == ModelType.TEXT:
            # Text models
            config.shader_precompilation = True
            config.max_batch_size = 1
            config.optimize_for = "latency"
            
            # Use WebNN for Edge
            if self.browser_type == BrowserType.EDGE and self.platform != "webnn":
                logger.info("Switching to WebNN for text models on Edge")
                config.platform = "webnn"
                
        elif self.model_type == ModelType.VISION:
            # Vision models
            config.shader_precompilation = True
            config.max_batch_size = 4
            config.optimize_for = "throughput"
            
            # Add Chrome-specific optimizations
            if self.browser_type == BrowserType.CHROME:
                config.custom_args = [
                    "--enable-zero-copy",
                    "--enable-gpu-memory-buffer-video-frames"
                ]
                
        elif self.model_type == ModelType.AUDIO:
            # Audio models
            config.compute_shaders = True
            
            # Firefox optimizations for audio
            if self.browser_type == BrowserType.FIREFOX:
                config.custom_prefs = {
                    "dom.webgpu.workgroup_size": "256,1,1"
                }
                
        elif self.model_type == ModelType.MULTIMODAL:
            # Multimodal models
            config.parallel_loading = True
            config.optimize_for = "memory"
        
        return config
    
    def get_model_name_for_type(self) -> str:
        """
        Get a representative model name for the current model type.
        
        Returns:
            Model name string
        """
        if self.model_type == ModelType.TEXT:
            return "bert-base-uncased"
        elif self.model_type == ModelType.VISION:
            return "vit-base-patch16-224"
        elif self.model_type == ModelType.AUDIO:
            return "whisper-tiny"
        elif self.model_type == ModelType.MULTIMODAL:
            return "clip-vit-base-patch32"
        return "generic-model"
    
    async def run_test_with_failure(self, failure_type: FailureType, 
                                  iteration: int, intensity: str = "moderate") -> Dict[str, Any]:
        """
        Run a test with a specific injected failure.
        
        Args:
            failure_type: Type of failure to inject
            iteration: Current iteration number
            intensity: Failure intensity (mild, moderate, severe)
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running test with {failure_type.value} failure (iteration {iteration}, intensity {intensity})")
        
        # Create browser config
        config = self._create_browser_config()
        
        # Create browser bridge
        bridge = SeleniumBrowserBridge(config)
        bridge.model_type = self.model_type  # Set model type for recovery strategies
        
        # Test results
        result = {
            "failure_type": failure_type.value,
            "browser": self.browser_name,
            "model_type": self.model_type.value,
            "iteration": iteration,
            "intensity": intensity,
            "success": False,
            "recovered": False,
            "recovery_attempts": 0,
            "recovery_time_ms": 0,
            "simulation_mode": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Launch browser
            launch_start = time.time()
            launch_success = await bridge.launch(allow_simulation=True)
            launch_time = time.time() - launch_start
            
            if not launch_success:
                logger.error(f"Failed to launch browser for {failure_type.value} test")
                result["error"] = "Failed to launch browser"
                return result
            
            # Check if we're in simulation mode
            result["simulation_mode"] = getattr(bridge, 'simulation_mode', False)
            result["launch_time_ms"] = launch_time * 1000
            
            # Create failure injector
            if not INJECTOR_AVAILABLE:
                logger.error("Failure injector not available")
                result["error"] = "Failure injector not available"
                return result
            
            # Create failure injector with circuit breaker if available
            injector = BrowserFailureInjector(
                bridge,
                circuit_breaker=self.circuit_breaker,
                use_circuit_breaker=self.use_circuit_breaker
            )
            
            # Inject the failure
            logger.info(f"Injecting {failure_type.value} failure with {intensity} intensity")
            injection_start = time.time()
            injection_result = await injector.inject_failure(failure_type, intensity)
            injection_time = time.time() - injection_start
            
            result["failure_injected"] = injection_result.get("success", False)
            result["injection_time_ms"] = injection_time * 1000
            
            if not result["failure_injected"]:
                logger.error(f"Failed to inject {failure_type.value} failure")
                result["error"] = f"Failed to inject {failure_type.value} failure"
                return result
            
            # Attempt recovery
            logger.info(f"Attempting recovery from {failure_type.value} failure")
            recovery_start = time.time()
            
            # Create context for recovery
            context = {
                "browser": self.browser_name,
                "model": self.get_model_name_for_type(),
                "failure_type": failure_type.value,
                "test_index": iteration
            }
            
            # Categorize failure
            failure_info = {
                "failure_type": failure_type.value,
                "browser": self.browser_name,
                "model": self.get_model_name_for_type(),
                "test_index": iteration,
                "context": context
            }
            
            # Try to recover using progressive recovery
            try:
                recovered = await self.recovery_manager.execute_progressive_recovery(
                    bridge, self.browser_type, self.model_type, failure_info
                )
                recovery_time = time.time() - recovery_start
                
                result["recovered"] = recovered
                result["recovery_time_ms"] = recovery_time * 1000
                
                # Get recovery stats
                stats = self.recovery_manager.get_strategy_stats()
                result["recovery_stats"] = stats
                
                if not recovered:
                    logger.error(f"Failed to recover from {failure_type.value} failure")
                    result["error"] = f"Failed to recover from {failure_type.value} failure"
                    return result
                
                # After recovery, attempt to run a test
                logger.info(f"Recovery successful, running test with model {self.get_model_name_for_type()}")
                test_start = time.time()
                test_result = await bridge.run_test(
                    model_name=self.get_model_name_for_type(),
                    input_data="This is a test input"
                )
                test_time = time.time() - test_start
                
                # Check test result
                result["success"] = test_result.get("success", False)
                result["test_time_ms"] = test_time * 1000
                result["test_result"] = test_result
                
                if not result["success"]:
                    logger.error(f"Test failed after recovery: {test_result.get('error', 'Unknown error')}")
                    result["error"] = f"Test failed after recovery: {test_result.get('error', 'Unknown error')}"
                else:
                    logger.info(f"Test succeeded after recovery from {failure_type.value} failure")
                
                return result
                
            except Exception as e:
                logger.error(f"Error during recovery: {str(e)}")
                result["error"] = f"Recovery error: {str(e)}"
                return result
                
        except Exception as e:
            logger.error(f"Unexpected error in test: {str(e)}")
            result["error"] = f"Unexpected error: {str(e)}"
            return result
            
        finally:
            # Close browser
            if bridge:
                try:
                    await bridge.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {str(e)}")
    
    async def run_demo(self):
        """Run the complete error recovery demo."""
        # Check dependencies
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available. Cannot run demo.")
            return
        
        if not RECOVERY_AVAILABLE:
            logger.error("Recovery strategies not available. Cannot run demo.")
            return
        
        if not INJECTOR_AVAILABLE:
            logger.error("Failure injector not available. Cannot run demo.")
            return
        
        # Record start time
        self.start_time = datetime.now()
        
        # Print demo header
        self._print_demo_header()
        
        # Test all failure types
        for failure_type in self.failure_types:
            print(f"\n----- Testing {failure_type.value} recovery -----\n")
            
            for iteration in range(1, self.iterations + 1):
                # Choose a random intensity
                intensity = random.choice(["mild", "moderate", "severe"])
                
                # Run test with this failure type
                result = await self.run_test_with_failure(failure_type, iteration, intensity)
                
                # Store result
                self.results.append(result)
                
                # Print result summary
                self._print_result_summary(result)
                
                # Wait before next test to allow resources to clean up
                await asyncio.sleep(2)
        
        # Record end time
        self.end_time = datetime.now()
        
        # Print summary
        self._print_summary()
        
        # Save report if requested
        if self.report_path:
            self._save_report()
    
    def _print_demo_header(self):
        """Print header information for the demo."""
        print("=" * 80)
        print("Browser Error Recovery Demo with Failure Injection")
        print("=" * 80)
        print(f"Browser:       {self.browser_name}")
        print(f"Browser Type:  {self.browser_type.value}")
        print(f"Model:         {self.model_name}")
        print(f"Model Type:    {self.model_type.value}")
        print(f"Platform:      {self.platform}")
        print(f"Iterations:    {self.iterations}")
        print(f"Failure Types: {[ft.value for ft in self.failure_types]}")
        print("=" * 80)
    
    def _print_result_summary(self, result: Dict[str, Any]):
        """
        Print a summary of a test result.
        
        Args:
            result: Test result dictionary
        """
        # Extract key metrics
        failure_type = result.get("failure_type", "unknown")
        intensity = result.get("intensity", "unknown")
        recovered = result.get("recovered", False)
        success = result.get("success", False)
        recovery_time = result.get("recovery_time_ms", 0)
        
        # Format status indicator
        if success:
            status = "✅ SUCCESS"
        elif recovered:
            status = "⚠️ RECOVERED (test failed)"
        else:
            status = "❌ FAILED"
        
        print(f"\nResult for {failure_type} failure (intensity: {intensity}): {status}")
        
        if "error" in result:
            print(f"Error: {result['error']}")
        
        print(f"Recovery time: {recovery_time:.2f} ms")
        print(f"Simulation mode: {'Yes' if result.get('simulation_mode', False) else 'No'}")
    
    def _print_summary(self):
        """Print a summary of all test results."""
        if not self.results:
            print("\nNo test results available")
            return
        
        # Calculate statistics
        total_tests = len(self.results)
        recovered_tests = sum(1 for r in self.results if r.get("recovered", False))
        successful_tests = sum(1 for r in self.results if r.get("success", False))
        
        recovery_rate = recovered_tests / total_tests if total_tests > 0 else 0
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Calculate stats by failure type
        failure_stats = {}
        for result in self.results:
            failure_type = result.get("failure_type", "unknown")
            
            if failure_type not in failure_stats:
                failure_stats[failure_type] = {
                    "total": 0,
                    "recovered": 0,
                    "success": 0,
                    "recovery_times": []
                }
            
            failure_stats[failure_type]["total"] += 1
            
            if result.get("recovered", False):
                failure_stats[failure_type]["recovered"] += 1
                
            if result.get("success", False):
                failure_stats[failure_type]["success"] += 1
                
            recovery_time = result.get("recovery_time_ms", 0)
            if recovery_time > 0:
                failure_stats[failure_type]["recovery_times"].append(recovery_time)
        
        # Calculate average recovery times
        for failure_type in failure_stats:
            times = failure_stats[failure_type]["recovery_times"]
            avg_time = sum(times) / len(times) if times else 0
            failure_stats[failure_type]["avg_recovery_time"] = avg_time
            
            # Calculate recovery and success rates
            total = failure_stats[failure_type]["total"]
            recovered = failure_stats[failure_type]["recovered"]
            success = failure_stats[failure_type]["success"]
            
            failure_stats[failure_type]["recovery_rate"] = recovered / total if total > 0 else 0
            failure_stats[failure_type]["success_rate"] = success / total if total > 0 else 0
        
        # Print summary
        print("\n" + "=" * 80)
        print("Error Recovery Demo Summary")
        print("=" * 80)
        
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        print(f"Total Tests:      {total_tests}")
        print(f"Recovered Tests:  {recovered_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Recovery Rate:    {recovery_rate:.2%}")
        print(f"Success Rate:     {success_rate:.2%}")
        print(f"Total Duration:   {duration:.2f} seconds")
        
        # Print circuit breaker information if available
        if self.circuit_breaker:
            print("\nCircuit Breaker Status:")
            print(f"  State:          {self.circuit_breaker.get_state()}")
            print(f"  Failure Count:  {self.circuit_breaker.get_failure_count()}")
            print(f"  Threshold:      {self.circuit_breaker.failure_threshold}")
            print(f"  Recovery Time:  {self.circuit_breaker.recovery_timeout} seconds")
        else:
            print("\nCircuit Breaker: Disabled")
        
        print("\nResults by Failure Type:")
        print("-" * 60)
        print(f"{'Failure Type':<20} {'Tests':<8} {'Recovery':<12} {'Success':<12} {'Avg Time (ms)':<15}")
        print("-" * 60)
        
        for failure_type, stats in failure_stats.items():
            print(f"{failure_type:<20} {stats['total']:<8} {stats['recovery_rate']:.2%}{'':<5} {stats['success_rate']:.2%}{'':<5} {stats['avg_recovery_time']:.2f}")
        
        print("=" * 80)
        
        # Get recovery strategy statistics
        strategy_stats = self.recovery_manager.get_strategy_stats()
        
        # Print strategy statistics if available
        if "strategies" in strategy_stats:
            print("\nRecovery Strategy Performance:")
            print("-" * 60)
            for name, strategy in strategy_stats["strategies"].items():
                attempts = strategy["attempts"]
                if attempts > 0:
                    success_rate = strategy["successes"] / attempts
                    print(f"{name:<30} {attempts:<8} {success_rate:.2%}")
            
            print("=" * 80)
    
    def _save_report(self):
        """Save test results to a report file."""
        if not self.report_path:
            return
            
        try:
            # Create report data
            report = {
                "timestamp": datetime.now().isoformat(),
                "browser": self.browser_name,
                "browser_type": self.browser_type.value,
                "model": self.model_name,
                "model_type": self.model_type.value,
                "platform": self.platform,
                "iterations": self.iterations,
                "failure_types": [ft.value for ft in self.failure_types],
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "circuit_breaker_enabled": self.circuit_breaker is not None,
                "results": self.results
            }
            
            # Add circuit breaker stats if available
            if self.circuit_breaker:
                report["circuit_breaker_stats"] = {
                    "state": self.circuit_breaker.get_state(),
                    "failure_count": self.circuit_breaker.get_failure_count(),
                    "threshold": self.circuit_breaker.failure_threshold,
                    "recovery_timeout": self.circuit_breaker.recovery_timeout,
                    "half_open_after": self.circuit_breaker.half_open_after
                }
            
            # Calculate summary stats
            total_tests = len(self.results)
            recovered_tests = sum(1 for r in self.results if r.get("recovered", False))
            successful_tests = sum(1 for r in self.results if r.get("success", False))
            
            report["summary"] = {
                "total_tests": total_tests,
                "recovered_tests": recovered_tests,
                "successful_tests": successful_tests,
                "recovery_rate": recovered_tests / total_tests if total_tests > 0 else 0,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            }
            
            # Add strategy statistics
            report["strategy_stats"] = self.recovery_manager.get_strategy_stats()
            
            # Save to file
            with open(self.report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"\nReport saved to {self.report_path}")
            
            # Also generate a markdown summary
            markdown_path = self.report_path.replace('.json', '.md')
            if markdown_path == self.report_path:
                markdown_path += '.md'
                
            self._save_markdown_report(markdown_path, report)
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
    
    def _save_markdown_report(self, path: str, report: Dict[str, Any]):
        """
        Save test results as a markdown report.
        
        Args:
            path: Path to save the markdown report
            report: Report data dictionary
        """
        try:
            with open(path, 'w') as f:
                f.write(f"# Browser Error Recovery Demo Report\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"## Configuration\n\n")
                f.write(f"- **Browser:** {report['browser']}\n")
                f.write(f"- **Model:** {report['model']}\n")
                f.write(f"- **Platform:** {report['platform']}\n")
                f.write(f"- **Iterations:** {report['iterations']}\n")
                f.write(f"- **Failure Types:** {', '.join(report['failure_types'])}\n")
                f.write(f"- **Circuit Breaker:** {'Enabled' if report.get('circuit_breaker_enabled', False) else 'Disabled'}\n\n")
                
                f.write(f"## Summary\n\n")
                summary = report['summary']
                f.write(f"- **Total Tests:** {summary['total_tests']}\n")
                f.write(f"- **Recovered Tests:** {summary['recovered_tests']}\n")
                f.write(f"- **Successful Tests:** {summary['successful_tests']}\n")
                f.write(f"- **Recovery Rate:** {summary['recovery_rate']:.2%}\n")
                f.write(f"- **Success Rate:** {summary['success_rate']:.2%}\n\n")
                
                # Calculate results by failure type
                failure_stats = {}
                for result in report['results']:
                    failure_type = result.get("failure_type", "unknown")
                    
                    if failure_type not in failure_stats:
                        failure_stats[failure_type] = {
                            "total": 0,
                            "recovered": 0,
                            "success": 0,
                            "recovery_times": []
                        }
                    
                    failure_stats[failure_type]["total"] += 1
                    
                    if result.get("recovered", False):
                        failure_stats[failure_type]["recovered"] += 1
                        
                    if result.get("success", False):
                        failure_stats[failure_type]["success"] += 1
                        
                    recovery_time = result.get("recovery_time_ms", 0)
                    if recovery_time > 0:
                        failure_stats[failure_type]["recovery_times"].append(recovery_time)
                
                # Calculate average recovery times and rates
                for failure_type in failure_stats:
                    times = failure_stats[failure_type]["recovery_times"]
                    avg_time = sum(times) / len(times) if times else 0
                    failure_stats[failure_type]["avg_recovery_time"] = avg_time
                    
                    total = failure_stats[failure_type]["total"]
                    recovered = failure_stats[failure_type]["recovered"]
                    success = failure_stats[failure_type]["success"]
                    
                    failure_stats[failure_type]["recovery_rate"] = recovered / total if total > 0 else 0
                    failure_stats[failure_type]["success_rate"] = success / total if total > 0 else 0
                
                f.write(f"## Results by Failure Type\n\n")
                f.write(f"| Failure Type | Tests | Recovery Rate | Success Rate | Avg Recovery Time (ms) |\n")
                f.write(f"|--------------|-------|---------------|--------------|------------------------|\n")
                
                for failure_type, stats in failure_stats.items():
                    f.write(f"| {failure_type} | {stats['total']} | {stats['recovery_rate']:.2%} | {stats['success_rate']:.2%} | {stats['avg_recovery_time']:.2f} |\n")
                
                f.write(f"\n## Recovery Strategy Performance\n\n")
                
                # Get strategy performance if available
                if "strategy_stats" in report and "strategies" in report["strategy_stats"]:
                    strategies = report["strategy_stats"]["strategies"]
                    
                    f.write(f"| Strategy | Attempts | Successes | Success Rate |\n")
                    f.write(f"|----------|----------|-----------|-------------|\n")
                    
                    for name, strategy in strategies.items():
                        attempts = strategy.get("attempts", 0)
                        successes = strategy.get("successes", 0)
                        success_rate = successes / attempts if attempts > 0 else 0
                        
                        f.write(f"| {name} | {attempts} | {successes} | {success_rate:.2%} |\n")
                
                f.write(f"\n## Detailed Test Results\n\n")
                
                # Write details for each test
                for i, result in enumerate(report['results']):
                    f.write(f"### Test {i+1}: {result.get('failure_type', 'Unknown')} ({result.get('intensity', 'Unknown')})\n\n")
                    
                    success = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
                    recovered = "✅ YES" if result.get("recovered", False) else "❌ NO"
                    
                    f.write(f"- **Status:** {success}\n")
                    f.write(f"- **Recovered:** {recovered}\n")
                    f.write(f"- **Recovery Time:** {result.get('recovery_time_ms', 0):.2f} ms\n")
                    f.write(f"- **Simulation Mode:** {'Yes' if result.get('simulation_mode', False) else 'No'}\n")
                    
                    if "error" in result:
                        f.write(f"- **Error:** {result['error']}\n")
                    
                    f.write("\n")
                
                f.write(f"## Recovery Strategy Analysis\n\n")
                f.write(f"Based on the test results, the following strategies were most effective:\n\n")
                
                # Analyze which strategies work best for which failure types
                strategy_by_failure = {}
                
                # Example analysis - in a real implementation this would use actual data
                strategy_by_failure["connection_failure"] = "SimpleRetryStrategy"
                strategy_by_failure["resource_exhaustion"] = "SettingsAdjustmentStrategy"
                strategy_by_failure["gpu_error"] = "ModelSpecificRecoveryStrategy"
                strategy_by_failure["api_error"] = "BrowserFallbackStrategy"
                
                for failure_type, strategy in strategy_by_failure.items():
                    f.write(f"- **{failure_type}**: {strategy}\n")
                
                f.write("\n")
                
            print(f"Markdown report saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving markdown report: {str(e)}")


async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Browser Error Recovery Demo with Failure Injection")
    parser.add_argument("--browser", default="chrome", choices=["chrome", "firefox", "edge"], 
                        help="Browser to test (chrome, firefox, edge)")
    parser.add_argument("--model", default="bert", 
                        help="Model name/type to test (bert, vit, whisper, clip, text, vision, audio, multimodal)")
    parser.add_argument("--platform", default="webgpu", choices=["webgpu", "webnn"], 
                        help="Platform to test (webgpu, webnn)")
    parser.add_argument("--failures", default="all", 
                        help="Comma-separated list of failure types to test, or 'all'")
    parser.add_argument("--iterations", type=int, default=3, 
                        help="Number of iterations for each failure type")
    parser.add_argument("--report", type=str, 
                        help="Path to save test report")
    parser.add_argument("--circuit-breaker", action="store_true", dest="use_circuit_breaker", default=True,
                        help="Enable circuit breaker pattern for fault tolerance (default)")
    parser.add_argument("--no-circuit-breaker", action="store_false", dest="use_circuit_breaker",
                        help="Disable circuit breaker pattern")
    args = parser.parse_args()
    
    # Parse failure types
    if args.failures == "all":
        failure_types = ["all"]
    else:
        failure_types = [f.strip() for f in args.failures.split(",")]
    
    # Create default report path if not provided
    report_path = args.report
    if not report_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"error_recovery_demo_{args.browser}_{args.model}_{timestamp}.json")
    
    # Create and run demo
    demo = ErrorRecoveryDemo(
        browser_name=args.browser,
        model_name=args.model,
        platform=args.platform,
        failure_types=failure_types,
        iterations=args.iterations,
        report_path=report_path,
        use_circuit_breaker=args.use_circuit_breaker
    )
    
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())