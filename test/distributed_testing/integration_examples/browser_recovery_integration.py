#!/usr/bin/env python3
"""
Browser Recovery Integration Examples

This module demonstrates how to integrate the advanced browser recovery strategies
with the existing BrowserAutomationBridge and circuit breaker pattern.

Usage:
    python -m distributed_testing.integration_examples.browser_recovery_integration --browser firefox --model whisper-tiny
"""

import os
import sys
import anyio
import argparse
import logging
import json
import traceback
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import browser recovery strategies
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from distributed_testing.browser_recovery_strategies import (
    recover_browser, ProgressiveRecoveryManager, 
    BrowserType, ModelType, RecoveryLevel,
    categorize_browser_failure
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("browser_recovery_integration")

# Set more verbose logging if environment variable is set
if os.environ.get("SELENIUM_BRIDGE_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)
    
# Try to import the BrowserAutomationBridge
try:
    from fixed_web_platform.browser_automation import BrowserAutomationBridge
    BRIDGE_AVAILABLE = True
except ImportError:
    logger.warning("BrowserAutomationBridge not available, using mock implementation")
    BRIDGE_AVAILABLE = False
    
    # Mock implementation for demonstration
    class BrowserAutomationBridge:
        """Mock BrowserAutomationBridge for demonstration."""
        
        def __init__(self, platform="webgpu", browser_name="chrome", 
                     headless=True, compute_shaders=False,
                     precompile_shaders=False, parallel_loading=False,
                     model_type="text", test_port=8765):
            self.platform = platform
            self.browser_name = browser_name
            self.headless = headless
            self.compute_shaders = compute_shaders
            self.precompile_shaders = precompile_shaders
            self.parallel_loading = parallel_loading
            self.model_type = model_type
            self.test_port = test_port
            self.initialized = False
            self.simulation_mode = False
            self.browser_args = []
            self.browser_prefs = {}
            self.resource_settings = {}
            self.audio_settings = {}
            self.recovery_attempts = 0
            self.recovery_history = []
            
        async def launch(self, allow_simulation=False):
            """Simulate browser launch."""
            # Simulate launch failures if we're trying to test recovery
            if hasattr(self, 'fail_launch_count') and self.fail_launch_count > 0:
                self.fail_launch_count -= 1
                logger.warning(f"Simulating launch failure ({self.fail_launch_count} more will occur)")
                return False
            
            self.initialized = True
            self.simulation_mode = allow_simulation
            
            if allow_simulation:
                logger.info(f"Launched browser in simulation mode")
            else:
                logger.info(f"Launched browser with platform: {self.platform}")
                
            return True
            
        async def close(self):
            """Simulate browser close."""
            self.initialized = False
            logger.debug(f"Closed browser: {self.browser_name}")
            return True
            
        async def run_test(self, model_name, input_data):
            """Simulate test run."""
            # Include recovery information if we've gone through recovery
            has_recovered = len(self.recovery_history) > 0
            
            return {
                "success": True,
                "model_name": model_name,
                "browser": self.browser_name,
                "platform": self.platform,
                "simulation": self.simulation_mode,
                "recovered": has_recovered,
                "recovery_attempts": self.recovery_attempts
            }
            
        def add_browser_arg(self, arg):
            """Add browser argument."""
            logger.debug(f"Adding browser arg: {arg}")
            self.browser_args.append(arg)
            
        def add_browser_pref(self, pref, value):
            """Add browser preference."""
            logger.debug(f"Adding browser pref: {pref}={value}")
            self.browser_prefs[pref] = value
            
        def set_platform(self, platform):
            """Set platform."""
            logger.debug(f"Setting platform: {platform}")
            self.platform = platform
            
        def set_browser(self, browser):
            """Set browser."""
            logger.debug(f"Setting browser: {browser}")
            self.browser_name = browser
            
        def set_compute_shaders(self, enabled):
            """Set compute shaders flag."""
            logger.debug(f"Setting compute shaders: {enabled}")
            self.compute_shaders = enabled
            
        def set_shader_precompilation(self, enabled):
            """Set shader precompilation flag."""
            logger.debug(f"Setting shader precompilation: {enabled}")
            self.precompile_shaders = enabled
            
        def set_parallel_loading(self, enabled):
            """Set parallel loading flag."""
            logger.debug(f"Setting parallel loading: {enabled}")
            self.parallel_loading = enabled
            
        def set_resource_settings(self, **kwargs):
            """Set resource settings."""
            logger.debug(f"Setting resource settings: {kwargs}")
            self.resource_settings.update(kwargs)
            
        def set_audio_settings(self, **kwargs):
            """Set audio settings."""
            logger.debug(f"Setting audio settings: {kwargs}")
            self.audio_settings.update(kwargs)
            
        def get_browser_args(self):
            """Get browser arguments."""
            return self.browser_args or ["--example-arg"]
        
        def record_recovery_attempt(self, strategy_name, success: bool, context: dict = None):
            """Record a recovery attempt."""
            self.recovery_attempts += 1
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy_name,
                "success": success,
                "browser": self.browser_name,
                "platform": self.platform,
                "context": context or {}
            }
            
            self.recovery_history.append(entry)
            logger.debug(f"Recorded recovery attempt: {strategy_name} (success={success})")
            
            # Simulate setting up launch failures
            if success and 'fail_next_launches' in context:
                self.fail_launch_count = context.get('fail_next_launches', 0)
                
        async def check_browser_responsive(self):
            """Check if browser is responsive."""
            # Usually responsive unless we're testing responsiveness issues
            return not hasattr(self, 'responsive') or self.responsive

# Try to import the CircuitBreaker
try:
    from distributed_testing.circuit_breaker import (
        CircuitBreaker, CircuitState, CircuitOpenError, CircuitBreakerRegistry
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    logger.warning("CircuitBreaker not available, using mock implementation")
    CIRCUIT_BREAKER_AVAILABLE = False
    
    # Mock implementation for demonstration
    class CircuitState:
        """Mock CircuitState for demonstration."""
        CLOSED = "CLOSED"
        OPEN = "OPEN"
        HALF_OPEN = "HALF_OPEN"
    
    class CircuitOpenError(Exception):
        """Mock CircuitOpenError for demonstration."""
        pass
    
    class CircuitBreaker:
        """Mock CircuitBreaker for demonstration."""
        
        def __init__(self, name, failure_threshold=5, recovery_timeout=10.0, 
                     half_open_max_calls=1, success_threshold=2):
            self.name = name
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.recovery_count = 0
            self.consecutive_successes = 0
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.half_open_max_calls = half_open_max_calls
            self.success_threshold = success_threshold
            self.last_failure_time = None
            self.last_success_time = None
            self.last_recovery_time = None
            self.recovery_attempts = []
            self.metrics = {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "recoveries": 0,
                "circuit_open_count": 0,
                "total_downtime_seconds": 0
            }
            
        async def execute(self, action, fallback=None):
            """Execute with circuit breaker protection."""
            self.metrics["executions"] += 1
            
            # Check if circuit should be reset (half-open)
            if self.state == CircuitState.OPEN and self.last_failure_time:
                now = datetime.now()
                time_diff = (now - self.last_failure_time).total_seconds()
                
                # Add to total downtime if we're still open
                if self.state == CircuitState.OPEN:
                    self.metrics["total_downtime_seconds"] += time_diff
                
                if time_diff >= self.recovery_timeout:
                    logger.info(f"Circuit {self.name} is entering half-open state for testing")
                    self.state = CircuitState.HALF_OPEN
                    self.consecutive_successes = 0
            
            # Handle circuit state
            if self.state == CircuitState.OPEN:
                logger.warning(f"Circuit {self.name} is open")
                if fallback:
                    return fallback()
                raise CircuitOpenError(f"Circuit {self.name} is open")
            
            try:
                # Support both async and sync actions
                result = action()
                if inspect.iscoroutine(result):
                    result = await result
                    
                # Record success
                self.success_count += 1
                self.consecutive_successes += 1
                self.last_success_time = datetime.now()
                self.metrics["successes"] += 1
                
                # Check for recovery success (if a result dictionary contains recovered=True)
                if isinstance(result, dict) and result.get("recovered", False):
                    self.recovery_count += 1
                    self.last_recovery_time = datetime.now()
                    self.metrics["recoveries"] += 1
                    
                    # Track recovery details with safe state handling
                    try:
                        state_str = self.state.value if hasattr(self.state, 'value') else str(self.state)
                    except:
                        state_str = str(self.state)
                        
                    recovery_details = {
                        "timestamp": self.last_recovery_time.isoformat(),
                        "success": True,
                        "previous_state": state_str,
                        "recovery_method": result.get("recovery_method", "unknown"),
                        "browser": result.get("browser", "unknown"),
                        "model": result.get("model", "unknown")
                    }
                    self.recovery_attempts.append(recovery_details)
                
                # Close circuit when enough consecutive successes in half-open state
                if self.state == CircuitState.HALF_OPEN and self.consecutive_successes >= self.success_threshold:
                    logger.info(f"Circuit {self.name} is now closed after {self.consecutive_successes} consecutive successes")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    
                return result
            except Exception as e:
                logger.warning(f"Circuit {self.name} recorded a failure: {str(e)}")
                self.failure_count += 1
                self.consecutive_successes = 0
                self.last_failure_time = datetime.now()
                self.metrics["failures"] += 1
                
                # Open circuit if threshold is reached
                if self.failure_count >= self.failure_threshold:
                    logger.warning(f"Circuit {self.name} is now open after {self.failure_count} failures")
                    self.state = CircuitState.OPEN
                    self.metrics["circuit_open_count"] += 1
                
                # Try fallback or re-raise
                if fallback:
                    return fallback()
                raise
                
        def get_metrics(self):
            """Get circuit breaker metrics."""
            # Calculate additional metrics
            success_rate = self.metrics["successes"] / self.metrics["executions"] if self.metrics["executions"] > 0 else 0
            recovery_rate = self.metrics["recoveries"] / self.metrics["failures"] if self.metrics["failures"] > 0 else 0
            avg_downtime = self.metrics["total_downtime_seconds"] / self.metrics["circuit_open_count"] if self.metrics["circuit_open_count"] > 0 else 0
            
            return {
                **self.metrics,
                "success_rate": success_rate,
                "recovery_rate": recovery_rate,
                "avg_downtime_seconds": avg_downtime,
                "current_state": self.state.value,
                "recovery_history": self.recovery_attempts
            }
    
    class CircuitBreakerRegistry:
        """Mock CircuitBreakerRegistry for demonstration."""
        
        def __init__(self):
            self.circuits = {}
            
        def get_or_create(self, name, **kwargs):
            """Get or create a circuit breaker."""
            if name not in self.circuits:
                self.circuits[name] = CircuitBreaker(name)
            return self.circuits[name]


class BrowserRecoveryDemo:
    """
    Demonstration of browser recovery integration.
    
    This class shows how to integrate the browser recovery strategies
    with the BrowserAutomationBridge and circuit breaker pattern.
    """
    
    def __init__(self, browser_name="chrome", model_name="bert-base-uncased",
                 platform="webgpu", inject_failures=True,
                 show_statistics=True):
        """
        Initialize the browser recovery demo.
        
        Args:
            browser_name: Name of the browser to use
            model_name: Name of the model to test
            platform: Platform to use (webgpu or webnn)
            inject_failures: Whether to inject artificial failures
            show_statistics: Whether to show recovery statistics
        """
        self.browser_name = browser_name
        self.model_name = model_name
        self.platform = platform
        self.inject_failures = inject_failures
        self.show_statistics = show_statistics
        
        # Determine model type
        from distributed_testing.browser_recovery_strategies import detect_model_type
        self.model_type = detect_model_type(model_name)
        
        # Set browser options based on model type
        self.compute_shaders = self.model_type == ModelType.AUDIO
        self.precompile_shaders = True
        self.parallel_loading = self.model_type == ModelType.MULTIMODAL
        
        # Create recovery manager
        self.recovery_manager = ProgressiveRecoveryManager()
        
        # Create circuit breaker registry
        self.registry = CircuitBreakerRegistry()
        
        logger.info(f"Initialized browser recovery demo with {browser_name}, {model_name}, {platform}")
        
    async def run_with_fault_tolerance(self):
        """
        Run tests with fault tolerance and recovery.
        
        This demonstrates how to integrate circuit breakers and recovery strategies.
        """
        # Create browser bridge
        bridge = BrowserAutomationBridge(
            platform=self.platform,
            browser_name=self.browser_name,
            compute_shaders=self.compute_shaders,
            precompile_shaders=self.precompile_shaders,
            parallel_loading=self.parallel_loading,
            model_type=self.model_type.value,
            headless=True
        )
        
        # Create circuit breaker
        circuit_name = f"browser_{self.browser_name}_{self.model_name}"
        circuit = self.registry.get_or_create(
            circuit_name,
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_max_calls=1,
            success_threshold=2
        )
        
        # Track metrics
        total_tests = 10
        successes = 0
        failures = 0
        recoveries = 0
        
        # Run multiple tests to demonstrate the system
        for i in range(total_tests):
            logger.info(f"Running test {i+1}/{total_tests}")
            
            try:
                # Run the test with circuit breaker protection
                result = await self._run_protected_test(bridge, circuit, i)
                
                # Process result
                if result.get("success", False):
                    logger.info(f"Test {i+1} succeeded")
                    successes += 1
                else:
                    logger.info(f"Test {i+1} failed: {result.get('error', 'Unknown error')}")
                    failures += 1
                    
                # Add recovery flag if present
                if result.get("recovered", False):
                    recoveries += 1
                    
            except CircuitOpenError:
                logger.warning(f"Test {i+1} skipped: circuit is open")
                failures += 1
                
            except Exception as e:
                logger.error(f"Test {i+1} failed with exception: {str(e)}")
                failures += 1
                
            # Wait between tests
            await anyio.sleep(1)
        
        # Close browser
        try:
            await bridge.close()
        except Exception as e:
            logger.warning(f"Error closing browser: {str(e)}")
        
        # Show summary
        logger.info("=" * 50)
        logger.info("Test Summary:")
        logger.info(f"Total Tests:  {total_tests}")
        logger.info(f"Successes:    {successes}")
        logger.info(f"Failures:     {failures}")
        logger.info(f"Recoveries:   {recoveries}")
        logger.info("=" * 50)
        
        # Show recovery statistics if requested
        if self.show_statistics:
            await self._show_recovery_statistics()
    
    async def _run_protected_test(self, bridge, circuit, test_index):
        """Run a test with circuit breaker protection and recovery."""
        # Define the test operation
        async def run_test():
            # Inject artificial failure if requested
            if self.inject_failures and test_index in [1, 4, 7]:
                failure_type = self._get_failure_type(test_index)
                logger.info(f"Injecting artificial {failure_type.value} failure")
                
                # Create an exception for the failure
                error = self._create_artificial_failure(failure_type)
                
                # Attempt recovery before failing
                logger.warning(f"Test failed with error: {str(error)}")
                logger.info("Attempting to recover from failure...")
                
                # Try to recover from the failure
                context = {
                    "browser": self.browser_name,
                    "model": self.model_name,
                    "platform": self.platform,
                    "test_index": test_index,
                    "failure_type": failure_type.value
                }
                
                # Call recovery function
                recovered = await recover_browser(bridge, error, context)
                
                if recovered:
                    logger.info("Recovery successful! Continuing with test.")
                    if not bridge.initialized:
                        success = await bridge.launch(allow_simulation=True)
                        if not success:
                            raise Exception("Failed to launch browser after recovery")
                    
                    # Run the test with recovered browser
                    result = await bridge.run_test(self.model_name, "This is a test input after recovery")
                    
                    # Add recovery information to the result
                    recovery_method = "unknown"
                    
                    # Determine recovery method based on model type without direct ModelType comparison
                    if hasattr(self, 'model_type'):
                        model_type_str = str(self.model_type).lower()
                        if 'audio' in model_type_str:
                            # Audio models typically use settings_adjustment with Firefox
                            recovery_method = "settings_adjustment_audio"
                        elif 'text' in model_type_str:
                            # Text models typically use WebNN on Edge
                            recovery_method = "settings_adjustment_text"
                        elif 'vision' in model_type_str:
                            # Vision models typically use Chrome with WebGPU
                            recovery_method = "settings_adjustment_vision"
                        elif 'multimodal' in model_type_str:
                            # Multimodal models typically use parallel loading
                            recovery_method = "settings_adjustment_multimodal"
                    
                    if isinstance(result, dict):
                        result["recovered"] = True
                        result["recovery_method"] = recovery_method
                        result["browser"] = self.browser_name
                        result["model"] = self.model_name
                        result["recovery_timestamp"] = datetime.now().isoformat()
                    else:
                        result = {
                            "success": True, 
                            "recovered": True,
                            "recovery_method": recovery_method,
                            "browser": self.browser_name,
                            "model": self.model_name,
                            "recovery_timestamp": datetime.now().isoformat()
                        }
                    return result
                else:
                    # Failed to recover, raise the original error
                    logger.error("Recovery failed, propagating original error")
                    raise error
            
            # Normal execution path (no injected failure)
            # Launch browser if not initialized
            if not bridge.initialized:
                success = await bridge.launch(allow_simulation=False)
                if not success:
                    raise Exception("Failed to launch browser")
            
            # Run the test
            return await bridge.run_test(self.model_name, "This is a test input")
        
        # Define fallback operation
        def fallback():
            return {
                "success": False,
                "error": f"Circuit {circuit.name} is open"
            }
        
        try:
            # Use the circuit breaker to protect the operation
            result = await circuit.execute(run_test, fallback)
            return result
        except Exception as e:
            logger.warning(f"Test operation failed with uncaught exception: {str(e)}")
            
            # Try to recover from the failure
            logger.info("Attempting recovery for uncaught exception...")
            recovered = await recover_browser(bridge, e, {
                "browser": self.browser_name,
                "model": self.model_name,
                "platform": self.platform,
                "test_index": test_index
            })
            
            if recovered:
                logger.info("Recovery successful, retrying operation")
                
                # Retry the operation after successful recovery
                try:
                    if not bridge.initialized:
                        success = await bridge.launch(allow_simulation=True)
                        if not success:
                            return {"success": False, "error": "Failed to launch browser after recovery"}
                    
                    # Run the test again, without injecting failure
                    result = await bridge.run_test(self.model_name, "This is a test input after recovery")
                    
                    # Add detailed recovery information
                    recovery_method = "emergency_recovery"
                    
                    # Determine recovery method based on model type without direct ModelType comparison
                    if hasattr(self, 'model_type'):
                        model_type_str = str(self.model_type).lower()
                        if 'audio' in model_type_str:
                            recovery_method = "emergency_recovery_audio"
                        elif 'text' in model_type_str:
                            recovery_method = "emergency_recovery_text"
                        elif 'vision' in model_type_str:
                            recovery_method = "emergency_recovery_vision"
                        elif 'multimodal' in model_type_str:
                            recovery_method = "emergency_recovery_multimodal"
                    
                    if isinstance(result, dict):
                        result["recovered"] = True
                        result["recovery_method"] = recovery_method
                        result["browser"] = self.browser_name
                        result["model"] = self.model_name
                        result["recovery_timestamp"] = datetime.now().isoformat()
                        result["recovery_path"] = "exception_handler"
                    else:
                        result = {
                            "success": True, 
                            "recovered": True,
                            "recovery_method": recovery_method,
                            "browser": self.browser_name,
                            "model": self.model_name,
                            "recovery_timestamp": datetime.now().isoformat(),
                            "recovery_path": "exception_handler"
                        }
                        
                    return result
                except Exception as retry_error:
                    logger.error(f"Retry after recovery failed: {str(retry_error)}")
                    return {"success": False, "error": str(retry_error), "recovered": True}
            else:
                logger.error("Recovery failed")
                return {"success": False, "error": str(e), "recovered": False}
    
    def _get_failure_type(self, test_index):
        """Get failure type based on test index."""
        from distributed_testing.browser_recovery_strategies import FailureType
        
        # Different failure types for different tests
        failure_types = [
            FailureType.LAUNCH_FAILURE,
            FailureType.CONNECTION_FAILURE,
            FailureType.GPU_ERROR,
            FailureType.API_ERROR,
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.TIMEOUT,
            FailureType.CRASH,
            FailureType.INTERNAL_ERROR
        ]
        
        # Select failure type based on test index (mod length to avoid index error)
        return failure_types[test_index % len(failure_types)]
    
    def _create_artificial_failure(self, failure_type):
        """Create an artificial failure of the specified type."""
        error_messages = {
            BrowserType.CHROME: {
                "LAUNCH_FAILURE": "Failed to launch chrome browser: executable not found",
                "CONNECTION_FAILURE": "Connection refused: Chrome WebDriver connection failed",
                "GPU_ERROR": "WebGPU adapter creation failed: GPU process crashed",
                "API_ERROR": "WebNN API error: Neural network context creation failed",
                "RESOURCE_EXHAUSTION": "Chrome out of memory: Allocation failed",
                "TIMEOUT": "Operation timed out while waiting for browser response",
                "CRASH": "Chrome has crashed unexpectedly",
                "INTERNAL_ERROR": "Internal browser error in chrome://gpu process"
            },
            BrowserType.FIREFOX: {
                "LAUNCH_FAILURE": "Failed to launch firefox: GeckoDriver not found",
                "CONNECTION_FAILURE": "Failed to connect to firefox on localhost:4444",
                "GPU_ERROR": "WebGPU compute shader compilation failed",
                "API_ERROR": "Firefox doesn't support the WebNN API",
                "RESOURCE_EXHAUSTION": "Firefox out of memory: Too many resources allocated",
                "TIMEOUT": "Timeout waiting for WebGPU operations to complete",
                "CRASH": "Firefox process has terminated unexpectedly",
                "INTERNAL_ERROR": "Internal Firefox WebGPU error"
            },
            BrowserType.EDGE: {
                "LAUNCH_FAILURE": "Failed to launch Edge browser: MSEdgeDriver not found",
                "CONNECTION_FAILURE": "Edge WebDriver connection failed: Connection refused",
                "GPU_ERROR": "WebGPU error in Edge: GPU hardware acceleration not available",
                "API_ERROR": "WebNN error in Edge: Neural network backend unavailable",
                "RESOURCE_EXHAUSTION": "Edge memory limit exceeded",
                "TIMEOUT": "Operation timed out waiting for Edge response",
                "CRASH": "Edge browser has crashed",
                "INTERNAL_ERROR": "Internal Edge browser error"
            }
        }
        
        # Get browser type from name
        browser_type = BrowserType.CHROME
        browser_name_lower = self.browser_name.lower()
        if "firefox" in browser_name_lower:
            browser_type = BrowserType.FIREFOX
        elif "edge" in browser_name_lower:
            browser_type = BrowserType.EDGE
        
        # Get error message for this browser and failure type
        error_message = error_messages.get(browser_type, {}).get(
            failure_type.name, f"Artificial {failure_type.value} failure"
        )
        
        # Create an exception with this message
        return Exception(error_message)
    
    async def _show_recovery_statistics(self):
        """Show recovery statistics."""
        # Get strategy statistics
        stats = self.recovery_manager.get_strategy_stats()
        
        # Get recovery history
        history = self.recovery_manager.get_recovery_history()
        
        # Perform performance analysis
        analysis = self.recovery_manager.analyze_performance()
        
        # Show summary
        logger.info("=" * 50)
        logger.info("Recovery System Statistics:")
        logger.info("-" * 50)
        
        # Summary stats
        summary = stats.get("summary", {})
        logger.info("Summary:")
        logger.info(f"Total Strategies: {summary.get('total_strategies', 0)}")
        logger.info(f"Total Attempts:   {summary.get('total_attempts', 0)}")
        logger.info(f"Total Successes:  {summary.get('total_successes', 0)}")
        logger.info(f"Success Rate:     {summary.get('overall_success_rate', 0):.2%}")
        logger.info("-" * 50)
        
        # Browser stats
        logger.info("Browser-Specific Stats:")
        for browser, browser_stats in stats.get("browsers", {}).items():
            success_rate = browser_stats.get("success_rate", 0)
            logger.info(f"{browser}: {success_rate:.2%} success rate ({browser_stats.get('successes', 0)}/{browser_stats.get('attempts', 0)})")
        logger.info("-" * 50)
        
        # Model stats
        logger.info("Model-Specific Stats:")
        for model, model_stats in stats.get("models", {}).items():
            success_rate = model_stats.get("success_rate", 0)
            logger.info(f"{model}: {success_rate:.2%} success rate ({model_stats.get('successes', 0)}/{model_stats.get('attempts', 0)})")
        logger.info("-" * 50)
        
        # Best strategies
        logger.info("Best Strategies by Browser/Model:")
        for browser, model_strategies in analysis.get("best_strategies", {}).items():
            for model, strategy_info in model_strategies.items():
                if strategy_info.get("strategy"):
                    logger.info(f"{browser}/{model}: {strategy_info.get('strategy')} (score: {strategy_info.get('score', 0):.2f})")
        
        # Show circuit breaker metrics if available
        try:
            circuit_name = f"browser_{self.browser_name}_{self.model_name}"
            circuit = self.registry.get_or_create(circuit_name)
            
            if hasattr(circuit, 'get_metrics'):
                metrics = circuit.get_metrics()
                
                logger.info("-" * 50)
                logger.info("Circuit Breaker Metrics:")
                logger.info(f"Name:            {circuit_name}")
                logger.info(f"Current State:   {metrics.get('current_state', 'UNKNOWN')}")
                logger.info(f"Executions:      {metrics.get('executions', 0)}")
                logger.info(f"Successes:       {metrics.get('successes', 0)}")
                logger.info(f"Failures:        {metrics.get('failures', 0)}")
                logger.info(f"Success Rate:    {metrics.get('success_rate', 0):.2%}")
                logger.info(f"Recoveries:      {metrics.get('recoveries', 0)}")
                logger.info(f"Recovery Rate:   {metrics.get('recovery_rate', 0):.2%}")
                
                # If circuit has opened during testing
                if metrics.get('circuit_open_count', 0) > 0:
                    logger.info(f"Circuit Opens:   {metrics.get('circuit_open_count', 0)}")
                    logger.info(f"Avg Downtime:    {metrics.get('avg_downtime_seconds', 0):.2f}s")
                
                # Show recent recovery history if available
                recovery_history = metrics.get('recovery_history', [])
                if recovery_history:
                    logger.info("Recent Recoveries:")
                    for i, recovery in enumerate(recovery_history[-3:]):  # Show last 3 recoveries
                        logger.info(f"  {i+1}. {recovery.get('timestamp', 'Unknown')} - {recovery.get('recovery_method', 'Unknown')} ({recovery.get('browser', 'Unknown')})")
                
        except Exception as e:
            # Don't let metrics display errors break the demo
            logger.debug(f"Error displaying circuit breaker metrics: {str(e)}")
        
        logger.info("=" * 50)


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Browser Recovery Integration Demo")
    parser.add_argument("--browser", type=str, default="chrome", help="Browser to use (chrome, firefox, edge)")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model to test")
    parser.add_argument("--platform", type=str, default="webgpu", choices=["webgpu", "webnn"], help="Platform to use")
    parser.add_argument("--no-failures", action="store_true", help="Don't inject artificial failures")
    parser.add_argument("--no-stats", action="store_true", help="Don't show recovery statistics")
    args = parser.parse_args()
    
    # Create and run demo
    demo = BrowserRecoveryDemo(
        browser_name=args.browser,
        model_name=args.model,
        platform=args.platform,
        inject_failures=not args.no_failures,
        show_statistics=not args.no_stats
    )
    
    await demo.run_with_fault_tolerance()


if __name__ == "__main__":
    anyio.run(main())